/**
 * @file frame_writer.hpp
 * @brief Custom FLAC bitstream serializer for fully-optimized parameters.
 */

#ifndef FRAME_WRITER_HPP
#define FRAME_WRITER_HPP

#include "bitwriter.hpp"
#include "optimizer.hpp"
#include <vector>
#include <cstdint>

/**
 * @brief Serializes FLAC audio frames from pre-computed optimal parameters.
 * 
 * This class replaces the libFLAC high-level encoder in the pipeline.
 * It provides full byte-accurate control over the bitstream, taking exact
 * `BlockParams` determined by the DP optimizer and generating valid FLAC frames.
 */
class FrameWriter {
public:
    /**
     * @brief Construct a new FrameWriter.
     */
    FrameWriter() = default;

    /**
     * @brief Serialize one variable-blocksize FLAC audio frame.
     * 
     * Uses absolute sample-number addressing.
     * 
     * @param params        Optimizer output for this block (sizes, modes, LPC coefficients, Rice params).
     * @param pcm_data      Full per-channel PCM for the entire stream.
     * @param sample_number Absolute sample index of the first sample in this block.
     * @param sample_rate   Stream sample rate in Hz.
     * @param bps           Nominal bits-per-sample of the stream.
     * @return              Raw bytes of the complete FLAC frame (header + subframes + CRC-16).
     */
    std::vector<uint8_t> write_frame(
        const BlockParams&                          params,
        const std::vector<std::vector<int32_t>>&   pcm_data,
        uint64_t                                    sample_number,
        uint32_t                                    sample_rate,
        uint32_t                                    bps
    );

    /**
     * @brief Serialize the 38-byte STREAMINFO metadata block.
     * 
     * @param is_last       True if this is the final metadata block before audio frames.
     * @param min_blocksize Minimum block size used in the stream.
     * @param max_blocksize Maximum block size used in the stream.
     * @param min_framesize Minimum frame byte size in the stream.
     * @param max_framesize Maximum frame byte size in the stream.
     * @param sample_rate   Audio sample rate in Hz.
     * @param channels      Number of audio channels.
     * @param bps           Bits per sample.
     * @param total_samples Total absolute number of samples in the stream.
     * @param md5           Pointer to 16 bytes containing the raw audio MD5 signature, or nullptr for all-zeros.
     * @return              Serialized byte vector of the STREAMINFO block.
     */
    static std::vector<uint8_t> make_streaminfo_block(
        bool     is_last,
        uint32_t min_blocksize, uint32_t max_blocksize,
        uint32_t min_framesize, uint32_t max_framesize,
        uint32_t sample_rate,
        uint32_t channels,
        uint32_t bps,
        uint64_t total_samples,
        const uint8_t* md5 = nullptr
    );

private:
    /// @cond INTERNAL

    // Write one subframe (header + payload + residual) into bw.
    static void write_subframe(
        BitWriter&            bw,
        const SubframeParams& sp,
        const int32_t*        samples,
        uint32_t              bsize,
        uint32_t              ch_bps
    );

    // Compute and write the Rice-partitioned residual section.
    static void write_residual(
        BitWriter&            bw,
        const SubframeParams& sp,
        const int32_t*        residuals,
        uint32_t              bsize,
        int                   order
    );

    // Encode block_size → 4-bit code + optional extra bytes.
    static void encode_blocksize(BitWriter& bw, uint32_t bsize);
    // Encode sample_rate → 4-bit code + optional extra bytes.
    static void encode_samplerate(BitWriter& bw, uint32_t sr);
    // Encode bps → 3-bit code (0 = from STREAMINFO, which we never emit here).
    static uint8_t encode_bps(uint32_t bps);

    /// @endcond
};

#endif // FRAME_WRITER_HPP
