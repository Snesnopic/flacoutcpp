#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flacoutcpp {

/**
 * @brief Vulkan compute backend for LPC candidate evaluation (`-G`).
 *
 * Evaluates a batch of quantized LPC candidates against one subframe and
 * returns, for each, exactly what Optimizer::calculate_rice_cost would return.
 * Bit-exact by contract, not by approximation: the encoded file is identical
 * to the CPU-only build, so bench/check.sh still gates a GPU build. That is
 * the property that makes this worth having at all — a GPU path that changed
 * the output would cost the project its entire regression net.
 *
 * The cost model is a contract (see CLAUDE.md). shaders/sweep.comp is a second
 * implementation of it; if either side changes, both must.
 *
 * Construction never throws and never aborts: if Vulkan, a suitable device, or
 * a 32-lane subgroup is unavailable, available() returns false and why()
 * explains it, so `-G` can degrade to a diagnostic rather than a failure.
 *
 * evaluate() is thread-safe (internally serialised on one queue).
 */
class GpuEvaluator {
public:
    /// One candidate: order, quantization shift, and up to 32 coefficients.
    struct Candidate {
        int32_t order;
        int32_t shift;
        int32_t qc[32];
    };

    GpuEvaluator();
    ~GpuEvaluator();

    GpuEvaluator(const GpuEvaluator&)            = delete;
    GpuEvaluator& operator=(const GpuEvaluator&) = delete;

    /// True when a device was brought up and evaluate() can be called.
    bool available() const;

    /// Human-readable reason available() is false, or the device description
    /// when it is true. Always safe to call.
    const std::string& why() const;

    /**
     * @brief Price every candidate against one subframe.
     *
     * @param shifted    Subframe samples, already >> wasted bits.
     * @param bsize      Number of samples; must be a multiple of 32.
     * @param cands      Candidates to price.
     * @param out_costs  Resized to cands.size(); receives the Rice cost of
     *                   each, matching calculate_rice_cost exactly.
     * @return false if the batch could not be run, leaving out_costs
     *         untouched; callers must fall back to the CPU path.
     */
    bool evaluate(const int32_t* shifted, uint32_t bsize,
                  const std::vector<Candidate>& cands,
                  std::vector<uint32_t>& out_costs);

    /// Total candidates and wall seconds spent inside evaluate(), for `-G`
    /// reporting. Cheap; reads relaxed counters.
    void stats(uint64_t* candidates, double* seconds) const;

    /**
     * @brief Smallest batch worth dispatching; smaller ones stay on the CPU.
     *
     * Defaults to 0 — dispatch everything — because that measured fastest:
     * on stereo_4s (`-e -c 0 -L 0`, M4 Max) 0 gives 1.35x, 512 and 2048 give
     * 1.26-1.27x, and 8192 gives 0.89x. Declining a batch does not free the
     * GPU for anything else, so the only effect is a worker doing the work
     * itself. The knob exists because the crossover is device-specific and a
     * discrete part with a longer submit path may want it raised.
     */
    void   set_min_batch(size_t n);
    size_t min_batch() const;

    /**
     * @brief Cap on the kernel's partition-order search (1..8; 8 = no cap).
     *
     * The kernel's cost is dominated by partition closes, of which there are
     * 2^(P+1)-1, so lowering P is the cheapest speed available. It cannot make
     * the encoder wrong — the winning candidate is re-priced exactly on the
     * CPU with the full search — but it can change which candidate wins, so
     * anything below 8 is an optimality trade and belongs behind -U.
     */
    void set_partition_cap(int p);
    int  partition_cap() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace flacoutcpp
