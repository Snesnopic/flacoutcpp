#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "optimizer.hpp"   // WindowType

namespace flacoutcpp {

/**
 * @brief Fully GPU-resident FLAC encoder (`-P`).
 *
 * Raw PCM goes into device memory; encoded frame bytes come out. Between those
 * two points the host does no per-candidate, per-subframe or per-frame work:
 * windowing, autocorrelation, Levinson-Durbin, coefficient quantization, the
 * Rice sweep, mode selection, residual generation, bit packing and both CRCs all
 * run as compute dispatches in a single command buffer per chunk.
 *
 * ### This is a different encoder, not a backend for the old one
 *
 * `GpuEvaluator` (`-G`) is bit-exact with the CPU path by contract, which is what
 * lets bench/check.sh gate it. This class is not, and cannot be:
 *
 *  - Everything upstream of the integer coefficients runs in **fp32**. The CPU
 *    path is double throughout, and Metal has no `double` at all, so exactness
 *    across the two was never available on Apple silicon. It does not matter:
 *    the window, the autocorrelation, Levinson and the quantizer are all
 *    heuristics, and a worse heuristic costs compression, never losslessness.
 *  - The search is deliberately shallow and the partition is fixed. No
 *    variable-block DP, no patience, no precision ladder, no frame reuse.
 *
 * What *is* guaranteed is losslessness, and by the same mechanism as the CPU
 * path: once the coefficients are integers, the residual is exact integer
 * arithmetic that the decoder mirrors. Verify with a decode, not with `cmp`.
 *
 * ### MD5 is not here, on purpose
 *
 * STREAMINFO's MD5 chains 64-byte blocks and has no combine operator, so it
 * cannot be parallelized at all (unlike CRC, which is linear over GF(2) and is
 * done on the device). One lane grinding ~500k serial rounds would be the
 * slowest thing in the pipeline by an order of magnitude. The caller computes it
 * on a host thread while the device encodes, which costs nothing in wall clock.
 */
class PureGpuEncoder {
public:
    struct Config {
        /// Fixed frame size. Must be a multiple of 256 and at most 4096 (the
        /// autocorrelation kernel stages the windowed block in shared memory).
        uint32_t block_size = 4096;
        /// Windows evaluated per block. Empty → a built-in shortlist.
        std::vector<WindowType> windows;
        /// LPC coefficient precisions swept per (window, order). 1..4 entries.
        std::vector<int> precisions{15};
        /// Orders fully swept per (block, signal, window), out of 32, chosen by
        /// the Levinson prediction error.
        ///
        /// **Windows are worth far more than orders, so this is 2 and the window
        /// shortlist is 10.** On a real album track `8 windows x 2 orders`
        /// dominates `6 x 6` on both axes, and orders barely register there at
        /// all (`6x2` -> `6x6` is -0.011% for 1.4x). Album corpus, 188 tracks,
        /// 10x2 against the previous 6x3 default: **-0.1372% with all 188 tracks
        /// smaller**, at 1.04x corpus wall clock. The two knobs are not
        /// independent -- 2 orders is only safe because the shortlist widened.
        uint32_t orders = 2;
        /// Cap on the *sweep's* Rice partition-order search (1..8). Ranking only
        /// -- the winner is re-priced with the full search, so this cannot
        /// mis-state what the bitstream pays. The kernel is dominated by
        /// cross-lane partition closes (2^(P+1)-1 of them), which makes this the
        /// largest speed knob it has: 4 measured 3.4x for +0.029%.
        uint32_t partition_cap = 4;
        /// Frames per device chunk. Bounds peak device memory; frames are
        /// independent at a fixed block size, so this costs nothing but memory.
        uint32_t blocks_per_chunk = 256;
        bool verbose = true;
    };

    /// Receives finished frame bytes in stream order. Return false to abort.
    using Sink = std::function<bool(const uint8_t*, size_t)>;

    /// Blocks until at least N samples per channel are readable in the PCM
    /// buffers, returning false if the producer failed or ended short. Lets the
    /// caller decode *while* the device encodes: chunk N is submitted as soon as
    /// its samples exist, rather than after the whole file is decoded. Omit for
    /// a fully-decoded buffer.
    using Wait = std::function<bool(uint64_t)>;

    struct Stats {
        uint64_t frames      = 0;
        uint32_t min_frame   = 0;
        uint32_t max_frame   = 0;
        uint32_t min_block   = 0;
        uint32_t max_block   = 0;
        uint64_t bytes       = 0;
        double   device_secs = 0.0;
    };

    PureGpuEncoder(uint32_t channels, uint32_t bps, uint32_t sample_rate,
                   const Config& cfg);
    ~PureGpuEncoder();

    PureGpuEncoder(const PureGpuEncoder&)            = delete;
    PureGpuEncoder& operator=(const PureGpuEncoder&) = delete;

    /// True when a device came up and encode() can be called.
    bool available() const;
    /// Device description when available(), else why it is not. Always safe.
    const std::string& why() const;

    /**
     * @brief Encode the whole stream.
     *
     * @param pcm   Decoded samples, pcm[channel][sample].
     * @param sink  Called with each chunk's frame bytes, in order.
     * @param out   Frame/size statistics for STREAMINFO.
     * @param wait  Optional producer barrier; see Wait.
     * @return false on a device error, a sink refusal, or a producer failure.
     */
    bool encode(const std::vector<std::vector<int32_t>>& pcm,
                const Sink& sink, Stats* out, const Wait& wait = {});

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief A device context reused across files, for batch encoding.
 *
 * Returns an encoder for this stream shape, building one only when the shape or
 * the configuration differs from the previous call's. Nothing an encoder holds
 * depends on the audio -- the buffers are sized from (channels, bps, block size,
 * windows, chunk length) alone -- so a second file of the same shape can use the
 * first one's device, pipelines and buffers unchanged.
 *
 * It is worth having because the fixed cost is large next to a short track:
 * ~27.6 ms per invocation measured on a 0.02-second file, most of it
 * `vkCreateInstance` and the pipeline cache, which is 5.2 s across a 188-track
 * corpus that takes 21.6 s in total.
 *
 * Never delete the returned pointer; call release_shared_pure_gpu_encoder() when
 * the batch is finished. Not thread-safe: one batch at a time.
 *
 * @return nullptr if a device could not be created; @p why receives the reason.
 */
PureGpuEncoder* shared_pure_gpu_encoder(uint32_t channels, uint32_t bps,
                                        uint32_t sample_rate,
                                        const PureGpuEncoder::Config& cfg,
                                        std::string* why);
/// Drop the cached context. Safe to call when there is none.
void release_shared_pure_gpu_encoder();

} // namespace flacoutcpp
