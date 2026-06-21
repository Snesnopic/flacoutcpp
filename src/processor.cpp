#include "processor.hpp"
#include "optimizer.hpp"
#include "frame_writer.hpp"
#include "md5.hpp"
#include "FLAC/stream_decoder.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <cstring>

// ============================================================
// Constructor / destructor
// ============================================================

Processor::Processor(const std::string& in, const std::string& out,
                     ProcessorConfig config)
    : m_input(in), m_output(out), m_config(std::move(config)) {}

Processor::~Processor() = default;

// ============================================================
// Raw metadata extraction
// Reads non-STREAMINFO metadata blocks verbatim from the FLAC
// container so we can splice them unchanged into the output.
// ============================================================

bool Processor::read_extra_metadata_blocks(
    std::vector<std::vector<uint8_t>>& out_blocks) const
{
    std::ifstream f(m_input, std::ios::binary);
    if (!f) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "fLaC") return false;

    bool is_last = false;
    while (!is_last && f) {
        uint8_t hdr[4];
        f.read(reinterpret_cast<char*>(hdr), 4);
        if (!f || f.gcount() < 4) break;

        is_last      = (hdr[0] & 0x80u) != 0;
        uint8_t type = hdr[0] & 0x7Fu;
        uint32_t len = ((uint32_t)hdr[1] << 16)
                     | ((uint32_t)hdr[2] <<  8)
                     |  (uint32_t)hdr[3];

        std::vector<uint8_t> data(len);
        f.read(reinterpret_cast<char*>(data.data()), len);

        if (type != 0) { // skip STREAMINFO (we re-generate it)
            // Store: 4-byte header (is_last cleared for now) + payload
            std::vector<uint8_t> block;
            block.reserve(4 + len);
            block.push_back(type & 0x7Fu); // is_last will be set by the caller
            block.push_back(hdr[1]);
            block.push_back(hdr[2]);
            block.push_back(hdr[3]);
            block.insert(block.end(), data.begin(), data.end());
            out_blocks.push_back(std::move(block));
        }
    }
    return true;
}

// ============================================================
// Main pipeline
// ============================================================

bool Processor::process() {
    // --- Step 1: collect raw extra metadata blocks ----
    std::vector<std::vector<uint8_t>> extra_blocks;
    if (m_config.copy_metadata) {
        if (!read_extra_metadata_blocks(extra_blocks))
            std::cerr << "Warning: could not copy metadata from " << m_input << "\n";
    }

    // --- Step 2: decode PCM with libFLAC ----
    FLAC__StreamDecoder* decoder = FLAC__stream_decoder_new();
    if (!decoder) return false;

    FLAC__stream_decoder_set_metadata_respond(decoder, FLAC__METADATA_TYPE_STREAMINFO);
    if (FLAC__stream_decoder_init_file(decoder, m_input.c_str(),
                                       write_callback, metadata_callback,
                                       error_callback, this)
        != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
        FLAC__stream_decoder_delete(decoder);
        return false;
    }

    bool ok = FLAC__stream_decoder_process_until_end_of_stream(decoder);
    FLAC__stream_decoder_delete(decoder);

    if (!ok || m_pcm_data.empty() || m_total_samples == 0) return false;

    std::cout << "Decoded " << m_total_samples << " samples ("
              << m_channels << " ch, " << m_bps << " bps, "
              << m_sample_rate << " Hz)\n";

    // --- Step 2b: compute MD5 over interleaved little-endian PCM ----
    // The FLAC spec mandates MD5 over the raw audio (channel-interleaved,
    // little-endian, ceil(bps/8) bytes per sample).
    const int bytes_per_sample = (m_bps + 7) / 8;
    MD5 md5;
    std::vector<uint8_t> pcm_bytes(m_channels * bytes_per_sample);
    for (uint64_t s = 0; s < m_total_samples; ++s) {
        for (uint32_t c = 0; c < m_channels; ++c) {
            int32_t v = m_pcm_data[c][s];
            for (int b = 0; b < bytes_per_sample; ++b)
                pcm_bytes[c * bytes_per_sample + b] = (uint8_t)(v >> (b * 8));
        }
        md5.update(pcm_bytes.data(), m_channels * bytes_per_sample);
    }
    auto md5_digest = md5.digest();

    // --- Step 3: run optimiser ----
    Optimizer opt(m_channels, m_bps, m_config.windows, m_config.max_threads, m_config.exhaustive);
    std::vector<BlockParams> blocks = opt.find_optimal_block_partitioning(m_pcm_data);

    if (blocks.empty()) {
        std::cerr << "Error: optimizer produced no blocks.\n";
        return false;
    }

    // Compute min/max block size (known before writing)
    uint32_t min_bs = std::numeric_limits<uint32_t>::max(), max_bs = 0;
    for (const auto& b : blocks) {
        min_bs = std::min(min_bs, b.block_size);
        max_bs = std::max(max_bs, b.block_size);
    }

    // --- Step 4: open output file and write header ----
    // We need seekp() later to update STREAMINFO, so use fstream.
    std::fstream out(m_output,
                     std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Error: cannot open output file: " << m_output << "\n";
        return false;
    }

    // 4-byte FLAC magic
    out.write("fLaC", 4);

    // STREAMINFO block (is_last = true only if there are no extra blocks)
    bool si_is_last = extra_blocks.empty();
    auto si_block = FrameWriter::make_streaminfo_block(
        si_is_last,
        min_bs, max_bs,
        0, 0,   // min/max framesize unknown at this point
        m_sample_rate, m_channels, m_bps, m_total_samples);
    out.write(reinterpret_cast<const char*>(si_block.data()),
              (std::streamsize)si_block.size());

    // Extra metadata blocks
    for (size_t i = 0; i < extra_blocks.size(); ++i) {
        bool is_last = (i == extra_blocks.size() - 1);
        auto& blk = extra_blocks[i];
        // Set/clear the is_last bit in the stored header byte
        if (is_last) blk[0] |= 0x80u;
        else         blk[0] &= 0x7Fu;
        out.write(reinterpret_cast<const char*>(blk.data()),
                  (std::streamsize)blk.size());
    }

    // --- Step 5: encode and write frames ----
    FrameWriter fw;
    uint64_t sample_number = 0;
    uint32_t min_frm = std::numeric_limits<uint32_t>::max();
    uint32_t max_frm = 0;

    size_t total_written = 0;
    for (const auto& block : blocks) {
        auto frame_bytes = fw.write_frame(
            block, m_pcm_data, sample_number, m_sample_rate, m_bps);

        out.write(reinterpret_cast<const char*>(frame_bytes.data()),
                  (std::streamsize)frame_bytes.size());

        sample_number += block.block_size;
        min_frm = std::min(min_frm, (uint32_t)frame_bytes.size());
        max_frm = std::max(max_frm, (uint32_t)frame_bytes.size());
        total_written += frame_bytes.size();
    }

    // --- Step 6: seek back and update STREAMINFO with frame sizes + MD5 ----
    // Position: fLaC(4) + STREAMINFO block header(4) = byte 8
    out.seekp(8, std::ios::beg);
    auto si_updated = FrameWriter::make_streaminfo_block(
        si_is_last, min_bs, max_bs,
        min_frm, max_frm,
        m_sample_rate, m_channels, m_bps, m_total_samples,
        md5_digest.data());
    // Write only the 34-byte payload (skip the 4-byte block header)
    out.write(reinterpret_cast<const char*>(si_updated.data() + 4), 34);

    out.flush();
    if (!out) {
        std::cerr << "Error: write failed.\n";
        return false;
    }

    std::cout << "Wrote " << total_written << " bytes of audio data ("
              << blocks.size() << " frames, min=" << min_bs
              << " max=" << max_bs << " samples/frame).\n";
    return true;
}

// ============================================================
// libFLAC decoder callbacks
// ============================================================

FLAC__StreamDecoderWriteStatus Processor::write_callback(
    const FLAC__StreamDecoder*, const FLAC__Frame* frame,
    const FLAC__int32* const buffer[], void* client_data)
{
    auto* self = static_cast<Processor*>(client_data);
    uint32_t nch   = frame->header.channels;
    uint32_t bsize = frame->header.blocksize;

    if (self->m_pcm_data.empty())
        self->m_pcm_data.resize(nch);

    for (uint32_t c = 0; c < nch; ++c)
        self->m_pcm_data[c].insert(
            self->m_pcm_data[c].end(), buffer[c], buffer[c] + bsize);

    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

void Processor::error_callback(
    const FLAC__StreamDecoder*, FLAC__StreamDecoderErrorStatus status, void*)
{
    std::cerr << "Decoder error: " << FLAC__StreamDecoderErrorStatusString[status] << "\n";
}

void Processor::metadata_callback(
    const FLAC__StreamDecoder*, const FLAC__StreamMetadata* metadata, void* client_data)
{
    auto* self = static_cast<Processor*>(client_data);
    if (metadata->type == FLAC__METADATA_TYPE_STREAMINFO) {
        self->m_sample_rate   = metadata->data.stream_info.sample_rate;
        self->m_channels      = metadata->data.stream_info.channels;
        self->m_bps           = metadata->data.stream_info.bits_per_sample;
        self->m_total_samples = metadata->data.stream_info.total_samples;
    }
}
