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

namespace flacoutcpp {

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
     * @brief Exact encoded size of one frame, in bits: header + payload + footer.
     *
     * Mirrors write_frame — the two must be changed together, or the DP
     * optimizes a different cost than the stream actually pays.
     *
     * @param start_sample  Absolute sample index of the frame's first sample
     *                      (sets the UTF-8 sample-number length).
     * @param block_size    Samples in the frame (sets the blocksize code and
     *                      any trailing blocksize bytes).
     * @param sample_rate   Stream sample rate in Hz (sets any trailing bytes).
     * @param payload_bits  Total subframe bits, exact or estimated. The frame
     *                      header is whole bytes, so the footer's pad to a byte
     *                      boundary depends only on this value — which is why
     *                      it is a parameter rather than a header-only return.
     * @return              Total frame size in bits.
     */
    static uint32_t frame_bits(uint64_t start_sample, uint32_t block_size,
                               uint32_t sample_rate, uint32_t payload_bits);

    /**
     * @brief Re-emit a complete input frame under this stream's conventions.
     *
     * Splices the input frame's subframe payload verbatim (it is byte-aligned
     * after the header and independent of header framing), but rebuilds the
     * header — variable blocking strategy, canonical blocksize/samplerate
     * codes, the given absolute sample number — and recomputes CRC-8/16.
     * The channel-assignment/bps byte is copied from the input header since
     * the payload's interpretation depends on it.
     *
     * @param in           Raw bytes of one complete input frame.
     * @param len          Length of @p in in bytes (header through CRC-16).
     * @param sample_number Absolute first-sample index for the new header.
     * @param block_size   Samples in this frame (drives the blocksize code).
     * @param sample_rate  Stream sample rate (drives the samplerate code).
     * @return Rewritten frame bytes, or an empty vector if the input header
     *         cannot be parsed (caller should fall back to re-encoding).
     */
    static std::vector<uint8_t> rewrite_frame(
        const uint8_t* in, size_t len,
        uint64_t sample_number, uint32_t block_size, uint32_t sample_rate);

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

} // namespace flacoutcpp

#endif // FRAME_WRITER_HPP
