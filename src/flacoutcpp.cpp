#include "flacoutcpp.hpp"
#include "processor.hpp"

namespace flacoutcpp {

// Points measured on the master mix's size/time frontier; see apply_effort's
// documentation for the table and the caveats. Kept as a plain array so the
// mapping is one grep away from the measurement that produced it.
namespace {
struct EffortPoint { unsigned candidates; unsigned rungs; bool adaptive; };
constexpr EffortPoint EFFORT_LEVELS[] = {
    {  2, 1, true },  // 0
    {  8, 1, true },  // 1
    { 16, 1, true },  // 2
    { 24, 1, true },  // 3
    { 32, 1, true },  // 4
    { 48, 1, true },  // 5
    { 64, 1, true },  // 6
    { 64, 2, true },  // 7
    { 64, 3, true },  // 8
    {  0, 0, true },  // 9 — every candidate, every rung
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
    pc.dp_candidates = config.dp_candidates;
    pc.reuse_frames  = config.reuse_frames;
    pc.warn_superior = config.warn_superior;
    pc.verbose       = config.verbose;
    Processor proc(input_path, output_path, pc);
    return proc.process();
}

} // namespace flacoutcpp
