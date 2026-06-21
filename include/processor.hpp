/**
 * @file processor.hpp
 * @brief Top-level pipeline coordinator for decoding, optimizing, and encoding.
 */

#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include <vector>
#include <cstdint>
#include "FLAC/stream_decoder.h"
#include "optimizer.hpp"

/**
 * @brief Configuration parameters for the Processor pipeline.
 */
struct ProcessorConfig {
    /**
     * @brief Whether to copy non-audio metadata blocks from the input file.
     * 
     * When true, VORBIS_COMMENT, PICTURE, and other metadata blocks are copied
     * directly into the optimized output FLAC.
     */
    bool copy_metadata = true;

    /**
     * @brief Apodization windows to test during the exhaustive LPC search.
     * 
     * If empty, all 26 supported windows are evaluated to maximize compression.
     */
    std::vector<WindowType> windows;

    /**
     * @brief Maximum number of threads to spawn for dynamic programming evaluation.
     * 
     * A value of 0 indicates that the hardware concurrency limit should be used.
     */
    unsigned max_threads = 0;

    /**
     * @brief If true, performs full exhaustive search over all parameters.
     */
    bool exhaustive = false;
};

/**
 * @brief Main engine that coordinates FLAC decoding, DP optimization, and encoding.
 * 
 * The Processor wraps libFLAC to decode the entire input audio file into memory.
 * It then invokes the Optimizer to find the best variable block-size partition
 * and precise encoding parameters. Finally, it uses FrameWriter to serialize the
 * optimized FLAC bitstream and compute the MD5 checksum.
 */
class Processor {
public:
    /**
     * @brief Construct a new Processor object.
     * 
     * @param input_file Path to the input FLAC file.
     * @param output_file Path to write the optimized output FLAC file.
     * @param config Pipeline configuration options.
     */
    Processor(const std::string& input_file,
              const std::string& output_file,
              ProcessorConfig    config = {});

    /**
     * @brief Destroy the Processor object.
     */
    ~Processor();

    /**
     * @brief Execute the end-to-end optimization pipeline.
     * 
     * Decoding, DP optimization, serialization, and MD5 computation happen here.
     * 
     * @return true if the file was successfully optimized and written.
     * @return false if an error occurred during decoding, optimization, or writing.
     */
    bool process();

private:
    /// @cond INTERNAL

    // --- libFLAC decoder callbacks (PCM extraction) ----
    static FLAC__StreamDecoderWriteStatus write_callback(
        const FLAC__StreamDecoder* decoder, const FLAC__Frame* frame,
        const FLAC__int32* const buffer[], void* client_data);
    static void error_callback(
        const FLAC__StreamDecoder* decoder, FLAC__StreamDecoderErrorStatus status, void* client_data);
    static void metadata_callback(
        const FLAC__StreamDecoder* decoder, const FLAC__StreamMetadata* metadata, void* client_data);

    // Raw byte copy of non-STREAMINFO metadata blocks from input file.
    bool read_extra_metadata_blocks(std::vector<std::vector<uint8_t>>& out_blocks) const;

    // --- Member state ----
    std::string     m_input;
    std::string     m_output;
    ProcessorConfig m_config;

    // Decoded PCM (per-channel, arrays of channel samples)
    std::vector<std::vector<int32_t>> m_pcm_data;
    uint32_t m_sample_rate   = 0;
    uint32_t m_channels      = 0;
    uint32_t m_bps           = 0;
    uint64_t m_total_samples = 0;

    /// @endcond
};

#endif // PROCESSOR_HPP
