#include "flacoutcpp.hpp"
#include "processor.hpp"

namespace flacoutcpp {

bool optimize(const std::string& input, const std::string& output, const Config& cfg) {
    ProcessorConfig pc;
    pc.copy_metadata = cfg.copy_metadata;
    pc.windows       = cfg.windows;
    pc.max_threads   = cfg.max_threads;
    Processor proc(input, output, pc);
    return proc.process();
}

} // namespace flacoutcpp
