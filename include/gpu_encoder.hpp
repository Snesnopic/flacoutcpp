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
        /// the Levinson prediction error. This is the dominant speed knob: the
        /// Rice sweep is ~92% of device time and its cost is proportional to the
        /// summed order of the candidates it prices.
        uint32_t orders = 8;
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

} // namespace flacoutcpp
