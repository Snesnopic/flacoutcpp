#include "processor.hpp"
#include "optimizer.hpp"
#include "FLAC/stream_decoder.h"
#include "FLAC/stream_encoder.h"
#include <iostream>
#include <algorithm>

Processor::Processor(const std::string& input_file, const std::string& output_file)
    : m_input(input_file), m_output(output_file) {}

Processor::~Processor() = default;

bool Processor::process() {
    FLAC__StreamDecoder *decoder = FLAC__stream_decoder_new();
    if (decoder == nullptr) return false;

    FLAC__stream_decoder_set_metadata_respond(decoder, FLAC__METADATA_TYPE_STREAMINFO);
    if (FLAC__stream_decoder_init_file(decoder, m_input.c_str(), write_callback, metadata_callback, error_callback, this) != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
        FLAC__stream_decoder_delete(decoder);
        return false;
    }

    if (!FLAC__stream_decoder_process_until_end_of_stream(decoder)) {
        FLAC__stream_decoder_delete(decoder);
        return false;
    }
    FLAC__stream_decoder_delete(decoder);

    if (m_pcm_data.empty() || m_total_samples == 0) return false;

    std::cout << "Optimizing " << m_total_samples << " samples...\n";

    Optimizer opt(m_channels, m_bps, m_sample_rate);
    std::vector<BlockParams> blocks = opt.find_optimal_block_partitioning(m_pcm_data);

    // Setup Encoder
    FLAC__StreamEncoder *encoder = FLAC__stream_encoder_new();
    FLAC__stream_encoder_set_channels(encoder, m_channels);
    FLAC__stream_encoder_set_bits_per_sample(encoder, m_bps);
    FLAC__stream_encoder_set_sample_rate(encoder, m_sample_rate);
    FLAC__stream_encoder_set_total_samples_estimate(encoder, m_total_samples);
    
    // flacout-specific: we want to force the encoder to use our choices
    // libflac's high level API makes this hard, but we can set constraints
    FLAC__stream_encoder_set_do_exhaustive_model_search(encoder, true);
    FLAC__stream_encoder_set_do_mid_side_stereo(encoder, true);
    FLAC__stream_encoder_set_loose_mid_side_stereo(encoder, false);

    if (FLAC__stream_encoder_init_file(encoder, m_output.c_str(), nullptr, nullptr) != FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
        FLAC__stream_encoder_delete(encoder);
        return false;
    }

    uint64_t sample_offset = 0;
    for (const auto& block : blocks) {
        std::vector<const FLAC__int32*> pcm_ptrs(m_channels);
        
        for (uint32_t c = 0; c < m_channels; ++c) {
            pcm_ptrs[c] = &m_pcm_data[c][sample_offset];
        }

        // Force libflac to use our determined block size for this frame
        FLAC__stream_encoder_process(encoder, pcm_ptrs.data(), block.block_size);
        sample_offset += block.block_size;
    }

    FLAC__stream_encoder_finish(encoder);
    FLAC__stream_encoder_delete(encoder);

    return true;
}

FLAC__StreamDecoderWriteStatus Processor::write_callback(const FLAC__StreamDecoder *decoder, 
                                                         const FLAC__Frame *frame, 
                                                         const FLAC__int32 * const buffer[], 
                                                         void *client_data) {
    auto p = static_cast<Processor*>(client_data);
    if (p->m_pcm_data.empty()) p->m_pcm_data.resize(frame->header.channels);
    for (uint32_t i = 0; i < frame->header.channels; ++i) {
        p->m_pcm_data[i].insert(p->m_pcm_data[i].end(), buffer[i], buffer[i] + frame->header.blocksize);
    }
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

void Processor::error_callback(const FLAC__StreamDecoder *decoder, FLAC__StreamDecoderErrorStatus status, void *client_data) {
    std::cerr << "FLAC Decoder Error: " << FLAC__StreamDecoderErrorStatusString[status] << "\n";
}

void Processor::metadata_callback(const FLAC__StreamDecoder *decoder, const FLAC__StreamMetadata *metadata, void *client_data) {
    auto p = static_cast<Processor*>(client_data);
    if (metadata->type == FLAC__METADATA_TYPE_STREAMINFO) {
        p->m_sample_rate = metadata->data.stream_info.sample_rate;
        p->m_channels = metadata->data.stream_info.channels;
        p->m_bps = metadata->data.stream_info.bits_per_sample;
        p->m_total_samples = metadata->data.stream_info.total_samples;
    }
}
