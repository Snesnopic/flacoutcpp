#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include <vector>
#include <cstdint>
#include "FLAC/stream_decoder.h"
#include "optimizer.hpp"   // for WindowType

// Configuration passed to the Processor.
struct ProcessorConfig {
    bool copy_metadata = true;
    // Windows to test during LPC optimisation. Empty = all windows (maximum compression).
    std::vector<WindowType> windows;
    // Maximum number of worker threads (0 = use all logical CPUs).
    int max_threads = 0;
};

class Processor {
public:
    Processor(const std::string& input_file,
              const std::string& output_file,
              ProcessorConfig    config = {});
    ~Processor();

    // Run the full optimise-and-encode pipeline.
    // Returns true on success.
    bool process();

private:
    // --- libFLAC decoder callbacks (PCM extraction) ----
    static FLAC__StreamDecoderWriteStatus write_callback(
        const FLAC__StreamDecoder*, const FLAC__Frame*,
        const FLAC__int32* const[], void*);
    static void error_callback(
        const FLAC__StreamDecoder*, FLAC__StreamDecoderErrorStatus, void*);
    static void metadata_callback(
        const FLAC__StreamDecoder*, const FLAC__StreamMetadata*, void*);

    // Raw byte copy of non-STREAMINFO metadata blocks from input file.
    bool read_extra_metadata_blocks(std::vector<std::vector<uint8_t>>& out_blocks) const;

    // --- Member state ----
    std::string     m_input;
    std::string     m_output;
    ProcessorConfig m_config;

    // Decoded PCM (per-channel, interleaved by channel)
    std::vector<std::vector<int32_t>> m_pcm_data;
    uint32_t m_sample_rate   = 0;
    uint32_t m_channels      = 0;
    uint32_t m_bps           = 0;
    uint64_t m_total_samples = 0;
};

#endif // PROCESSOR_HPP
