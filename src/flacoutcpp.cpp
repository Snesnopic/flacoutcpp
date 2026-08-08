#include "flacoutcpp.hpp"
#include "processor.hpp"

namespace flacoutcpp {

bool optimise(const std::string& input_path, const std::string& output_path, const Config& config) {
    ProcessorConfig pc;
    pc.copy_metadata = config.copy_metadata;
    pc.windows       = config.windows;
    pc.max_threads   = config.max_threads;
    pc.exhaustive    = config.exhaustive;
    pc.max_candidates = config.max_candidates;
    pc.adaptive_windows = config.adaptive_windows;
    pc.patience = config.patience;
    pc.reuse_frames  = config.reuse_frames;
    pc.warn_superior = config.warn_superior;
    pc.verbose       = config.verbose;
    Processor proc(input_path, output_path, pc);
    return proc.process();
}

} // namespace flacoutcpp
