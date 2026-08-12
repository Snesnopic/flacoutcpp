/**
 * @file processor.hpp
 * @brief Top-level pipeline coordinator for decoding, optimizing, and encoding.
 */

#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <atomic>
#include <string>
#include <vector>
#include <cstdint>
#include "FLAC/stream_decoder.h"
#include "optimizer.hpp"

namespace flacoutcpp {

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
     * If empty, all 26 standard windows are evaluated to maximize compression
     * (experimental windows must be named explicitly).
     */
    std::vector<WindowType> windows;

    /// DP block-size ladder (`-b`); empty uses the built-in default.
    /// See Config::dp_candidates for the multiple-of-16 constraint.
    std::vector<uint32_t> dp_candidates;

    /**
     * @brief Maximum number of threads to spawn for dynamic programming evaluation.
     * 
     * A value of 0 indicates that the hardware concurrency limit should be used.
     */
    unsigned max_threads = 0;

    /**
     * @brief Exact-search mode: fully encode every block-partitioning choice
     * (and every stereo mode) instead of pricing them by estimate. Orthogonal
     * to `max_candidates`.
     */
    bool exhaustive = false;

    /**
     * @brief Ranked-search budget: (window, order) pairs fully evaluated per
     * subframe, ranked by Levinson-Durbin prediction error. 0 = no limit
     * (exhaustive sweep). Changing it changes the output — it trades
     * compression for speed.
     */
    unsigned max_candidates = 8;

    /**
     * @brief Consecutive non-improving candidates before the ranked scan
     * stops, making `max_candidates` a floor rather than a ceiling.
     * -1 = 2 x `max_candidates`; 0 = plain top-N cut.
     * See flacoutcpp::Config::patience.
     */
    int patience = -1;

    /**
     * @brief LPC precision rungs encoded per candidate; 0 = all 8.
     * See flacoutcpp::Config::precision_rungs.
     */
    unsigned precision_rungs = 0;

    /**
     * @brief Coefficient-lattice refinement sweeps; 0 = off.
     * See flacoutcpp::Config::lattice_sweeps.
     */
    unsigned lattice_sweeps = 0;

    /**
     * @brief Adaptive per-subframe window selection (experimental,
     * estimated-DP modes only). See flacoutcpp::Config::adaptive_windows.
     */
    bool adaptive_windows = false;

    /**
     * @brief Splice input frames that beat the re-encoded ones, and copy the
     * whole input through if the output would still be larger. On by
     * default; see flacoutcpp::Config::reuse_frames.
     */
    bool reuse_frames = true;

    /**
     * @brief Warn on stderr when input frames beat the re-encode. See
     * flacoutcpp::Config::warn_superior.
     */
    bool warn_superior = false;

    /**
     * @brief If false, suppresses progress/statistics stdout output.
     * Errors always go to stderr regardless of this setting.
     */
    bool verbose = true;
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

    /// What a worker records per frame; merged into InputFrame after the join.
    /// Public only because the decode workers keep it in a file-scope
    /// thread_local -- one per worker, since byte offsets chain within a single
    /// decoder and cannot be placed by sample position the way the PCM can.
    struct PendingFrameRec {
        uint64_t first_sample;
        uint32_t block_size;
        uint64_t byte_start;
        uint64_t byte_end;
    };

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

    // --- Parallel decode ----
    /// Decode the whole stream with `nthr` workers into preallocated m_pcm_data,
    /// each seeking to its own block-aligned range, and join them.
    ///
    /// `first` is an already-initialised decoder positioned past the metadata;
    /// worker 0 reuses it, since it starts at sample 0 and needs no seek. The
    /// caller owns and deletes it.
    ///
    /// Collects the frame byte-range map when reuse is enabled, merging the
    /// per-worker records afterwards; sets m_frame_pos_ok false if any worker
    /// could not report positions or the merged map does not tile the stream.
    bool decode_parallel(FLAC__StreamDecoder* first, unsigned nthr,
                         uint64_t range_len);

    /// Threads per parallel decode: half the cores, capped, overridable with
    /// FLACOUT_DECODE_THREADS. 1 disables the parallel path entirely.
    unsigned decode_thread_count(uint64_t total_samples, uint64_t align) const;

    /// Merge and validate what the workers recorded; see decode_parallel().
    void merge_input_frames(std::vector<std::vector<PendingFrameRec>>& per_worker);

    /// True while a parallel decode is in flight: write_callback then places each
    /// frame by its own sample number into a *preallocated* m_pcm_data, clipped to
    /// the writing worker's range, instead of appending. Every sample has exactly
    /// one writer, so the sample data needs no lock; nothing reads it until the
    /// join.
    bool              m_stream_mode = false;
    std::atomic<bool> m_decode_failed{false};

    // Input frame map for frame reuse: sample span and byte range of every
    // frame in the input file, recorded during decode (reuse_frames only).
    struct InputFrame {
        uint64_t first_sample;
        uint32_t block_size;
        uint64_t byte_start;
        uint64_t byte_end;
    };
    std::vector<InputFrame> m_input_frames;
    uint64_t m_prev_frame_end = 0;   // rolling byte offset during decode
    /// STREAMINFO's max blocksize, used to align parallel-decode ranges onto
    /// frame boundaries so a worker's first frame is usually its own.
    uint32_t m_max_blocksize  = 0;
    bool     m_frame_pos_ok   = true; // false if the decoder can't report positions

    // Decoded PCM (per-channel, arrays of channel samples)
    std::vector<std::vector<int32_t>> m_pcm_data;
    uint32_t m_sample_rate   = 0;
    uint32_t m_channels      = 0;
    uint32_t m_bps           = 0;
    uint64_t m_total_samples = 0;

    /// @endcond
};

} // namespace flacoutcpp

#endif // PROCESSOR_HPP
