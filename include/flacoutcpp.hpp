/**
 * @file flacoutcpp.hpp
 * @brief Public C++ library API for flacoutcpp.
 *
 * flacoutcpp is a FLAC re-encoder that achieves better compression than
 * `flac --best` by exhaustively searching the LPC parameter space
 * (26 apodization windows × 32 orders × 8 quantization precisions × 4 stereo
 * modes per block) and selecting the globally optimal variable block-size
 * partition via dynamic programming.
 *
 * ### Typical usage
 * @code
 * #include "flacoutcpp.hpp"
 *
 * flacoutcpp::Config cfg;
 * cfg.max_threads = 0;          // use all logical CPUs
 * cfg.copy_metadata = true;     // preserve VORBIS_COMMENT, PICTURE, …
 *
 * bool ok = flacoutcpp::optimise("input.flac", "output.flac", cfg);
 * @endcode
 *
 * The library is header-only at this level; link against `libflacout_lib`.
 */

#ifndef FLACOUTCPP_HPP
#define FLACOUTCPP_HPP

#include <string>
#include <vector>
#include "optimizer.hpp"

/// Top-level namespace for the flacoutcpp library.
namespace flacoutcpp {

/**
 * @brief Configuration for a single optimise-and-encode run.
 *
 * All fields have sensible defaults; zero-initialising the struct is valid.
 */
struct Config {
    /**
     * @brief Copy non-audio metadata blocks to the output file.
     *
     * When @c true (default), VORBIS_COMMENT, PICTURE, PADDING, and other
     * metadata blocks present in the source file are replicated verbatim in
     * the output.  STREAMINFO is always rewritten from scratch.
     */
    bool copy_metadata = true;

    /**
     * @brief Apodization windows to test during LPC optimisation.
     *
     * An empty vector (the default) enables all 26 built-in windows, which
     * yields maximum compression at the cost of higher CPU usage.  Supply a
     * smaller set to trade compression for speed.
     *
     * @see WindowType for the list of available windows.
     */
    std::vector<WindowType> windows;

    /**
     * @brief Maximum number of worker threads.
     *
     * Set to @c 0 (default) to use all logical CPUs reported by the OS.
     * The DP block-evaluation phase is embarrassingly parallel and scales
     * linearly with thread count.
     */
    unsigned max_threads = 0;

    /**
     * @brief If true, performs full exhaustive search over all parameters.
     * 
     * Bypasses all heuristics for window, precision, stereo, and DP pruning.
     * Can be extremely slow.
     */
    bool exhaustive = false;

    /**
     * @brief Print progress and statistics to stdout during the run.
     *
     * Set to @c false to suppress all output — useful when flacoutcpp is
     * embedded as a library and the caller manages its own UI.
     * Errors that cause the function to return @c false are also suppressed;
     * the caller should rely on the return value to detect failures.
     */
    bool verbose = true;
};

/**
 * @brief Re-encode a FLAC file with exhaustive compression optimisation.
 *
 * Decodes @p input_path using libFLAC, runs the variable-block-size DP
 * optimizer, and serializes the result to @p output_path using a custom
 * bit-accurate FLAC frame writer.  The output is a fully valid FLAC stream
 * (verified by `flac --test`) with an MD5 audio signature in STREAMINFO.
 *
 * @param input_path   Path to the source FLAC file.
 * @param output_path  Path for the output FLAC file (created or overwritten).
 * @param config       Optimisation parameters (see Config).
 * @return             @c true on success, @c false on any decode/encode error.
 *
 * @note CPU time is proportional to @c (audio_duration × num_candidates ×
 *       num_windows × max_lpc_order) / @c max_threads.  For reference, a
 *       0.5-second test file takes ~18 minutes on a 4-core Linux runner.
 */
bool optimise(const std::string& input_path,
              const std::string& output_path,
              const Config&      config = {});

} // namespace flacoutcpp

#endif // FLACOUTCPP_HPP
