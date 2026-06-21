#ifndef FRAME_WRITER_HPP
#define FRAME_WRITER_HPP

#include "bitwriter.hpp"
#include "optimizer.hpp"
#include <vector>
#include <cstdint>

// Serializes FLAC audio frames from pre-computed BlockParams.
// This replaces the libFLAC high-level encoder in the pipeline,
// giving full control over every parameter in the bitstream.
class FrameWriter {
public:
    FrameWriter() = default;

    // Serialize one FLAC frame (variable-blocksize mode, sample-number addressing).
    // @param params        Optimizer output for this block (sizes, modes, LPC coefficients, Rice params).
    // @param pcm_data      Full per-channel PCM for the entire stream.
    // @param sample_number Absolute sample index of the first sample in this block.
    // @param sample_rate   Stream sample rate in Hz.
    // @param bps           Nominal bits-per-sample of the stream.
    // @returns             Raw bytes of the complete FLAC frame (header + subframes + CRC-16).
    std::vector<uint8_t> write_frame(
        const BlockParams&                          params,
        const std::vector<std::vector<int32_t>>&   pcm_data,
        uint64_t                                    sample_number,
        uint32_t                                    sample_rate,
        uint32_t                                    bps
    );

    // Serialize the 4+34 = 38-byte STREAMINFO metadata block.
    // md5 must point to 16 bytes (the audio MD5 signature), or nullptr for all-zeros.
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
    // Write one subframe (header + payload + residual) into bw.
    // @param bw       BitWriter to append to.
    // @param sp       Pre-computed subframe parameters from the Optimizer.
    // @param samples  Raw PCM samples for this channel/block (already joint-stereo-transformed).
    // @param bsize    Number of samples in this block.
    // @param ch_bps   Effective bits-per-sample for this channel (nominal bps, or bps+1 for side).
    static void write_subframe(
        BitWriter&           bw,
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
};

#endif // FRAME_WRITER_HPP
