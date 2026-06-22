#include "flacoutcpp.hpp"
#include "processor.hpp"

namespace flacoutcpp {

bool optimise(const std::string& input_path, const std::string& output_path, const Config& config) {
    ProcessorConfig pc;
    pc.copy_metadata = config.copy_metadata;
    pc.windows       = config.windows;
    pc.max_threads   = config.max_threads;
    pc.exhaustive    = config.exhaustive;
    pc.verbose       = config.verbose;
    Processor proc(input_path, output_path, pc);
    return proc.process();
}

} // namespace flacoutcpp
