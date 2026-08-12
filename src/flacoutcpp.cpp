#include "flacoutcpp.hpp"
#include "processor.hpp"

namespace flacoutcpp {

// Points measured on the master mix's size/time frontier; see apply_effort's
// documentation for the table and the caveats. Kept as a plain array so the
// mapping is one grep away from the measurement that produced it.
namespace {
struct EffortPoint { unsigned candidates; unsigned rungs; bool adaptive; bool exact_dp; };
constexpr EffortPoint EFFORT_LEVELS[] = {
    {  2, 1, true,  false },  // 0
    {  8, 1, true,  false },  // 1
    { 16, 1, true,  false },  // 2
    { 24, 1, true,  false },  // 3
    { 32, 1, true,  false },  // 4
    { 48, 1, true,  false },  // 5
    { 64, 1, true,  false },  // 6
    { 64, 2, true,  false },  // 7
    { 64, 3, true,  false },  // 8
    {  0, 0, true,  false },  // 9 — every candidate, every rung
    // --- exact-DP levels. The dial used to stop at 9 on the grounds that
    // crossing into -e "would jump ~40x", but that figure is bare -e, which
    // implies -c 0 -L 0. Hold the search depth and the jump is far smaller,
    // and the exchange rate is not close: on the 188-track mix level 9 buys
    // -0.051% for 6.4x, while level 10 buys -0.520% for 7.7x. Ten times the
    // compression for 20% more time, which made level 9 the dial's worst
    // point rather than its best.
    //
    //   level 9   -0.051%   6.4x
    //   level 10  -0.520%   7.7x
    //   level 11  -0.572%  12.3x
    //   level 12  -0.610%  23.6x
    //
    // Adaptive windows are off here because exact DP defines its own window
    // set; at a finite -c that is the shortlist (see the gate in
    // Optimizer::Optimizer), which is what these numbers were measured with.
    {  2, 1, false, true  },  // 10
    {  8, 1, false, true  },  // 11
    { 24, 1, false, true  },  // 12
};
constexpr int NUM_EFFORT_LEVELS =
    (int)(sizeof(EFFORT_LEVELS) / sizeof(EFFORT_LEVELS[0]));
} // namespace

bool apply_effort(Config& config, int level) {
    if (level < 0 || level >= NUM_EFFORT_LEVELS) return false;
    config.max_candidates  = EFFORT_LEVELS[level].candidates;
    config.precision_rungs = EFFORT_LEVELS[level].rungs;
    // Every level wants adaptive windows — 16 of the 21 points on the measured
    // 3-D frontier use them, and all of them past the fastest corner. But -e
    // and an explicit window list each define their own set, so the
    // combination is contradictory rather than merely redundant. Reading the
    // config here rather than erroring keeps `-E 5 -e` meaningful: it means
    // level 5's depth under exact DP, which is a sensible thing to ask for.
    // Callers therefore have to set exhaustive/windows *before* this.
    config.adaptive_windows = EFFORT_LEVELS[level].adaptive
                              && !config.exhaustive && config.windows.empty();
    // Levels 10+ select exact DP themselves. They never turn it *off*: -e -E 3
    // still means level 3's depth under exact pricing, as before.
    if (EFFORT_LEVELS[level].exact_dp) config.exhaustive = true;
    return true;
}

bool optimise(const std::string& input_path, const std::string& output_path, const Config& config) {
    ProcessorConfig pc;
    pc.copy_metadata = config.copy_metadata;
    pc.windows       = config.windows;
    pc.max_threads   = config.max_threads;
    pc.exhaustive    = config.exhaustive;
    pc.max_candidates = config.max_candidates;
    pc.adaptive_windows = config.adaptive_windows;
    pc.patience = config.patience;
    pc.precision_rungs = config.precision_rungs;
    pc.lattice_sweeps = config.lattice_sweeps;
    pc.use_gpu        = config.use_gpu;
    pc.gpu_min_batch  = config.gpu_min_batch;
    pc.gpu_partition_cap = config.gpu_partition_cap;
    pc.gpu_slots      = config.gpu_slots;
    pc.gpu_duty       = config.gpu_duty;
    pc.pure_gpu       = config.pure_gpu;
    pc.pg_block_size  = config.pg_block_size;
    pc.pg_precisions  = config.pg_precisions;
    pc.pg_orders      = config.pg_orders;
    pc.pg_blocks_per_chunk = config.pg_blocks_per_chunk;
    pc.dp_candidates = config.dp_candidates;
    pc.reuse_frames  = config.reuse_frames;
    pc.warn_superior = config.warn_superior;
    pc.verbose       = config.verbose;
    Processor proc(input_path, output_path, pc);
    return proc.process();
}

} // namespace flacoutcpp
