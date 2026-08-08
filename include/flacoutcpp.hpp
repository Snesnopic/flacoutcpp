/**
 * @file flacoutcpp.hpp
 * @brief Public C++ library API for flacoutcpp.
 *
 * flacoutcpp is a FLAC re-encoder that achieves better compression than
 * `flac --best` by exhaustively searching the LPC parameter space
 * (26 standard apodization windows × 32 orders × 8 quantization precisions ×
 * 4 stereo modes per block) and selecting the globally optimal variable block-size
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
     * An empty vector (the default) enables all 26 standard windows under
     * exhaustive mode, which yields maximum compression at the cost of higher
     * CPU usage.  Supply a smaller set to trade compression for speed.
     * Experimental windows (WindowType values from EXPERIMENTAL_BEGIN on)
     * are used only when named here explicitly.
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
     * @brief Exact-search mode: price every block-partitioning choice exactly.
     *
     * When true, every (position, block size) pair in the partitioning DP is
     * fully encoded rather than estimated from granule autocorrelations, all
     * four stereo modes are fully evaluated per block, and the default window
     * set widens to all 26 standard windows. Can be extremely slow.
     *
     * Orthogonal to @ref max_candidates: this option decides how *blocks* are
     * priced; @c max_candidates decides how deep the per-subframe LPC search
     * goes within whatever blocks get encoded.
     */
    bool exhaustive = false;

    /**
     * @brief Ranked-search budget: (window, order) pairs evaluated per subframe.
     *
     * Levinson-Durbin already computes the prediction error at every order as
     * a by-product; the ranking uses it to estimate each (window, order)
     * pair's cost up front, and only the best @c max_candidates of them are
     * fully evaluated (each across the whole precision ladder). @c 0 means no
     * limit — every pair is fully evaluated, the classic exhaustive sweep.
     *
     * The default of 8 costs ~0.03% size on real music for ~1.75x speed over
     * the unlimited sweep. Known weak spot: near-white high-bps content, where
     * the Levinson errors are almost flat across orders and the ranking is
     * noise — 24-bit whitenoise fixtures grew 2-2.8% at 8 and needed 32 for
     * parity. Real music does not behave that way.
     */
    unsigned max_candidates = 8;

    /**
     * @brief Adaptive per-subframe window selection (experimental).
     *
     * Estimated-DP modes only: instead of the fixed 4-window shortlist, each
     * encoded block picks a 4-window set from its cached granule statistics
     * (stationarity, transient position, spectral tilt) at identical analysis
     * cost. Incompatible with @ref exhaustive and an explicit @ref windows
     * list, both of which define their own window sets.
     */
    bool adaptive_windows = false;

    /**
     * @brief Reuse input frames that beat the new encoding (experimental).
     *
     * The input file arrives already partitioned into frames whose exact
     * compressed sizes are known. With this enabled, wherever the input's
     * frames tile a span of the chosen partition in fewer bytes than the
     * re-encoded frames, the input frames are spliced into the output
     * (payload verbatim, header rewritten to this stream's conventions,
     * CRCs recomputed). If the finished file is still larger than the
     * input, the input is copied through unchanged (only when
     * @ref copy_metadata is true, since copy-through preserves metadata).
     * Together these guarantee re-encoding never grows a file.
     */
    bool reuse_frames = false;

    /**
     * @brief Print progress and statistics to stdout during the run.
     *
     * Set to @c false to suppress progress/statistics output — useful when
     * flacoutcpp is embedded as a library and the caller manages its own UI.
     * Errors are always printed to stderr regardless of this setting, since
     * the boolean return value alone does not say *why* encoding failed.
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
