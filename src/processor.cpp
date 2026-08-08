#include "processor.hpp"
#include "optimizer.hpp"
#include "frame_writer.hpp"
#include "md5.hpp"
#include "FLAC/stream_decoder.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
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

    // For frame reuse we need each input frame's byte range. Decode metadata
    // first so the position query below lands on the first audio frame.
    bool ok = true;
    if (m_config.reuse_frames) {
        ok = FLAC__stream_decoder_process_until_end_of_metadata(decoder);
        if (ok && !FLAC__stream_decoder_get_decode_position(decoder, &m_prev_frame_end))
            m_frame_pos_ok = false;
    }
    ok = ok && FLAC__stream_decoder_process_until_end_of_stream(decoder);
    FLAC__stream_decoder_delete(decoder);

    if (!ok || m_pcm_data.empty() || m_total_samples == 0) return false;

    if (m_config.verbose)
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
    Optimizer opt(m_channels, m_bps, m_sample_rate, m_config.windows, m_config.max_threads,
                  m_config.exhaustive, m_config.verbose, m_config.max_candidates,
                  m_config.adaptive_windows);
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

    // --- Step 4: open a temporary output file and write header ----
    // Written to m_output + ".partial" and renamed into place only after a
    // fully successful write, so a failure or crash partway through never
    // leaves a corrupt/truncated file at m_output (nor clobbers a pre-existing
    // file there before we know we can replace it).
    // We need seekp() later to update STREAMINFO, so use fstream.
    const std::string tmp_output = m_output + ".partial";
    std::fstream out(tmp_output,
                     std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Error: cannot open output file: " << tmp_output << "\n";
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
    uint32_t min_frm = std::numeric_limits<uint32_t>::max();
    uint32_t max_frm = 0;
    uint32_t emit_min_bs = std::numeric_limits<uint32_t>::max();
    uint32_t emit_max_bs = 0;
    size_t   total_written = 0;
    size_t   frames_reused = 0;

    auto emit = [&](const std::vector<uint8_t>& fb, uint32_t bs) {
        out.write(reinterpret_cast<const char*>(fb.data()),
                  (std::streamsize)fb.size());
        min_frm = std::min(min_frm, (uint32_t)fb.size());
        max_frm = std::max(max_frm, (uint32_t)fb.size());
        emit_min_bs = std::min(emit_min_bs, bs);
        emit_max_bs = std::max(emit_max_bs, bs);
        total_written += fb.size();
    };

    // Frame reuse: the input's own frames compete against the re-encoded
    // ones wherever both partitions share a boundary. Validate that the
    // recorded frames tile the stream, then load the input bytes for
    // payload splicing.
    bool reuse = m_config.reuse_frames && m_frame_pos_ok && !m_input_frames.empty();
    std::vector<uint8_t> input_bytes;
    if (reuse) {
        uint64_t expect = 0;
        for (const auto& f : m_input_frames) {
            if (f.first_sample != expect || f.byte_end <= f.byte_start) { reuse = false; break; }
            expect += f.block_size;
        }
        if (expect != m_total_samples) reuse = false;
    }
    if (reuse) {
        std::ifstream inf(m_input, std::ios::binary);
        inf.seekg(0, std::ios::end);
        input_bytes.resize((size_t)inf.tellg());
        inf.seekg(0, std::ios::beg);
        inf.read(reinterpret_cast<char*>(input_bytes.data()),
                 (std::streamsize)input_bytes.size());
        if (!inf || input_bytes.size() < m_input_frames.back().byte_end)
            reuse = false;
    }

    if (!reuse) {
        uint64_t sample_number = 0;
        for (const auto& block : blocks) {
            auto frame_bytes = fw.write_frame(
                block, m_pcm_data, sample_number, m_sample_rate, m_bps);

            // Debug builds: the optimizer's predicted frame size must equal what
            // the writer actually emitted, or the DP is optimizing a cost the
            // stream does not pay. (block.total_bits is exact here — every block
            // in the final sequence came from compute_block.)
            assert(frame_bytes.size() * 8 ==
                   FrameWriter::frame_bits(sample_number, block.block_size,
                                           m_sample_rate, block.total_bits));

            emit(frame_bytes, block.block_size);
            sample_number += block.block_size;
        }
    } else {
        // Two-pointer walk over both partitions. A segment closes whenever
        // they hit a common sample boundary; whichever side spent fewer
        // bytes over the segment is emitted. Ties keep the re-encoded
        // frames, so a second pass over our own output is byte-stable.
        size_t bi = 0, ij = 0;
        uint64_t our_end = 0, in_end = 0;
        std::vector<std::pair<std::vector<uint8_t>, uint32_t>> ours, theirs;
        size_t our_sz = 0, in_sz = 0;
        bool   seg_ok = true; // false if any input frame failed to rewrite

        while (bi < blocks.size() || ij < m_input_frames.size()) {
            if (our_end <= in_end && bi < blocks.size()) {
                const auto& block = blocks[bi++];
                auto fb = fw.write_frame(
                    block, m_pcm_data, our_end, m_sample_rate, m_bps);
                assert(fb.size() * 8 ==
                       FrameWriter::frame_bits(our_end, block.block_size,
                                               m_sample_rate, block.total_bits));
                our_sz += fb.size();
                our_end += block.block_size;
                ours.emplace_back(std::move(fb), block.block_size);
            } else if (ij < m_input_frames.size()) {
                const auto& f = m_input_frames[ij++];
                auto fb = FrameWriter::rewrite_frame(
                    &input_bytes[f.byte_start],
                    (size_t)(f.byte_end - f.byte_start),
                    f.first_sample, f.block_size, m_sample_rate);
                if (fb.empty()) seg_ok = false;
                else in_sz += fb.size();
                in_end += f.block_size;
                theirs.emplace_back(std::move(fb), f.block_size);
            } else {
                break; // partitions disagree on total length; validated above
            }

            if (our_end == in_end && !ours.empty() && !theirs.empty()) {
                const bool take_input = seg_ok && in_sz < our_sz;
                auto& chosen = take_input ? theirs : ours;
                if (take_input) frames_reused += theirs.size();
                for (auto& [fb, bs] : chosen) emit(fb, bs);
                ours.clear(); theirs.clear();
                our_sz = in_sz = 0;
                seg_ok = true;
            }
        }
    }

    // --- Step 6: seek back and update STREAMINFO with frame sizes + MD5 ----
    // Block sizes come from the emitted frames — with reuse they can differ
    // from the DP's blocks.
    // Position: fLaC(4) + STREAMINFO block header(4) = byte 8
    out.seekp(8, std::ios::beg);
    auto si_updated = FrameWriter::make_streaminfo_block(
        si_is_last, emit_min_bs, emit_max_bs,
        min_frm, max_frm,
        m_sample_rate, m_channels, m_bps, m_total_samples,
        md5_digest.data());
    // Write only the 34-byte payload (skip the 4-byte block header)
    out.write(reinterpret_cast<const char*>(si_updated.data() + 4), 34);

    out.flush();
    if (!out) {
        std::cerr << "Error: write failed.\n";
        out.close();
        std::remove(tmp_output.c_str());
        return false;
    }
    out.close();

    // Whole-file guarantee: if the finished file is still larger than the
    // input, ship the input unchanged. Only with copy_metadata — the copy
    // preserves the input's metadata, which -n explicitly asked to drop.
    bool copied_through = false;
    if (m_config.reuse_frames && m_config.copy_metadata) {
        std::ifstream a(tmp_output, std::ios::binary | std::ios::ate);
        std::ifstream b(m_input,    std::ios::binary | std::ios::ate);
        if (a && b && a.tellg() > b.tellg()) {
            b.seekg(0, std::ios::beg);
            std::ofstream repl(tmp_output, std::ios::binary | std::ios::trunc);
            repl << b.rdbuf();
            if (!repl) {
                std::cerr << "Error: copy-through write failed.\n";
                std::remove(tmp_output.c_str());
                return false;
            }
            copied_through = true;
        }
    }

    if (std::rename(tmp_output.c_str(), m_output.c_str()) != 0) {
        std::cerr << "Error: could not finalize output file: " << m_output << "\n";
        std::remove(tmp_output.c_str());
        return false;
    }

    if (m_config.verbose) {
        if (copied_through)
            std::cout << "Output would exceed the input; copied the input through unchanged.\n";
        else
            std::cout << "Wrote " << total_written << " bytes of audio data ("
                      << blocks.size() << " frames, min=" << emit_min_bs
                      << " max=" << emit_max_bs << " samples/frame"
                      << (frames_reused
                          ? ", " + std::to_string(frames_reused) + " input frames reused"
                          : std::string())
                      << ").\n";
    }
    return true;
}

// ============================================================
// libFLAC decoder callbacks
// ============================================================

FLAC__StreamDecoderWriteStatus Processor::write_callback(
    const FLAC__StreamDecoder* decoder, const FLAC__Frame* frame,
    const FLAC__int32* const buffer[], void* client_data)
{
    auto* self = static_cast<Processor*>(client_data);
    uint32_t nch   = frame->header.channels;
    uint32_t bsize = frame->header.blocksize;

    if (self->m_pcm_data.empty())
        self->m_pcm_data.resize(nch);

    // Frame-reuse bookkeeping: the decode position after a frame is that
    // frame's end; its start is the previous frame's end. Sample position
    // comes from the running decode count, so it is right for both fixed-
    // and variable-blocksize inputs.
    if (self->m_config.reuse_frames && self->m_frame_pos_ok) {
        uint64_t end = 0;
        if (FLAC__stream_decoder_get_decode_position(decoder, &end)) {
            self->m_input_frames.push_back(InputFrame{
                (uint64_t)self->m_pcm_data[0].size(), bsize,
                self->m_prev_frame_end, end });
            self->m_prev_frame_end = end;
        } else {
            self->m_frame_pos_ok = false;
            self->m_input_frames.clear();
        }
    }

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
