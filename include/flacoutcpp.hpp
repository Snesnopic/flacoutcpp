#ifndef FLACOUTCPP_HPP
#define FLACOUTCPP_HPP

// flacoutcpp public library API.
// Include this single header to use flacoutcpp as a library.

#include <string>
#include <vector>
#include "optimizer.hpp"   // for WindowType (re-exported below)

namespace flacoutcpp {

// Re-export WindowType so library users don't need to pull in optimizer.hpp.
using Window = WindowType;

// Returns all available window types (maximum compression preset).
inline std::vector<Window> all_windows() { return all_window_types(); }

// Parse a window name string (case-insensitive, separators ignored).
// Returns Window::COUNT if the name is not recognised.
inline Window window_from_string(const std::string& name) {
    return window_from_name(name);
}

// Configuration for the optimize() function.
struct Config {
    bool copy_metadata = true;
    // Windows to test. Empty = all windows (maximum compression).
    std::vector<Window> windows;
    // Max worker threads. 0 = use all logical CPUs.
    int max_threads = 0;
};

// Optimise a FLAC file.
//
// Reads `input`, runs an exhaustive block-partitioning and LPC search, and
// writes the re-encoded stream to `output`.  The output is always lossless:
// decoding it yields bit-identical PCM to the original.
//
// Returns true on success, false on any error (messages printed to stderr).
bool optimize(const std::string& input,
              const std::string& output,
              const Config& config = {});

} // namespace flacoutcpp

#endif // FLACOUTCPP_HPP
