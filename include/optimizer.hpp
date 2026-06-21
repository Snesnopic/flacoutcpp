/**
 * @file optimizer.hpp
 * @brief Exhaustive FLAC LPC optimizer and variable block-size DP partitioner.
 *
 * The Optimizer class is the computational core of flacoutcpp.  For each
 * candidate block it evaluates every combination of:
 *   - 26 apodization windows (WindowType)
 *   - LPC orders 1–32 (Levinson-Durbin via compute_lpc_all_orders)
 *   - Quantization precisions 8–15 bits
 *   - 4 stereo modes: Independent, Left-Side, Right-Side, Mid-Side
 *
 * It then selects the globally optimal variable block-size partition using
 * exact dynamic programming over candidates {1024, 2048, 4096, 8192, 16384}.
 *
 * ### Parallelism
 * The DP precomputation phase dispatches all @c N×K `compute_block` calls to
 * a flat thread pool (one `std::atomic` fetch_add per item).  The DP itself
 * is O(N×K) and runs sequentially in microseconds.
 */

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
/// @defgroup windows Apodization Windows
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Apodization window applied to samples before autocorrelation.
 *
 * Different windows emphasize different frequency regions of the signal.
 * Testing all windows per block allows the Optimizer to find the LPC
 * coefficients that minimize the Rice-coded residual entropy for that block.
 *
 * The `COUNT` sentinel is used for iteration and range-checking only.
 */
enum class WindowType : uint8_t {
    RECTANGULAR = 0,          ///< No apodization (flat window).
    BARTLETT,                 ///< Triangular window (zero endpoints).
    BARTLETT_HANN,            ///< Bartlett-Hann composite window.
    BLACKMAN,                 ///< Classic 3-term Blackman window.
    BLACKMAN_HARRIS_4TERM_92DB, ///< 4-term Blackman-Harris, −92 dB sidelobes.
    CONNES,                   ///< Connes (cos⁴) window.
    FLATTOP,                  ///< Flat-top window (amplitude accurate).
    GAUSS_025,                ///< Gaussian window, σ = 0.25.
    GAUSS_0125,               ///< Gaussian window, σ = 0.125.
    HAMMING,                  ///< Hamming window.
    HANN,                     ///< Hann (raised cosine) window.
    KAISER_BESSEL,            ///< Kaiser-Bessel window.
    NUTTALL,                  ///< 4-term Nuttall window.
    TRIANGLE,                 ///< Triangular window (non-zero endpoints).
    WELCH,                    ///< Welch (parabolic) window.
    TUKEY_005,                ///< Tukey window, taper fraction 0.05.
    TUKEY_010,                ///< Tukey window, taper fraction 0.10.
    TUKEY_020,                ///< Tukey window, taper fraction 0.20.
    TUKEY_050,                ///< Tukey window, taper fraction 0.50.
    TUKEY_075,                ///< Tukey window, taper fraction 0.75.
    TUKEY_090,                ///< Tukey window, taper fraction 0.90.
    PARTIAL_TUKEY_2_000,      ///< Partial Tukey (2 partitions, offset 0.00).
    PARTIAL_TUKEY_2_033,      ///< Partial Tukey (2 partitions, offset 0.33).
    PARTIAL_TUKEY_2_067,      ///< Partial Tukey (2 partitions, offset 0.67).
    PUNCHOUT_TUKEY_2_033,     ///< Punchout Tukey (2 partitions, offset 0.33).
    PUNCHOUT_TUKEY_2_067,     ///< Punchout Tukey (2 partitions, offset 0.67).
    COUNT                     ///< Sentinel — total number of window types.
};

/**
 * @brief Parse a window type from its name (case-insensitive).
 * @param name  Human-readable window name (e.g. @c "hann", @c "blackman").
 * @return      The corresponding WindowType, or @c WindowType::COUNT if not found.
 */
WindowType window_from_name(const std::string& raw);

/**
 * @brief Return the canonical name of a window type.
 * @param wt  A valid WindowType (not COUNT).
 * @return    Lower-case name string (e.g. @c "hann").
 */
std::string window_to_name(WindowType wt);

/**
 * @brief Return a vector containing all 26 window types (excluding COUNT).
 *
 * Passing this set to the Optimizer enables maximum compression at full
 * CPU cost.
 */
std::vector<WindowType> all_window_types();

/// @}


// ---------------------------------------------------------------------------
/// @defgroup params Optimizer Output Structures
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Encoding parameters for one subframe (one channel of one FLAC frame).
 *
 * Produced by Optimizer::optimize_subframe() and consumed by
 * FrameWriter::write_subframe().  All fields are valid only after
 * optimize_subframe() returns.
 */
struct SubframeParams {
    int     mode;               ///< Subframe type: 0=Constant, 1=Verbatim, 2=Fixed, 3=LPC.
    int     order;              ///< Predictor order (Fixed: 0–4, LPC: 1–32).
    int     lpc_precision;      ///< Bits used to quantize LPC coefficients (8–15).
    int     lpc_shift;          ///< Right-shift applied after dot product during prediction.
    int     wasted_bits;        ///< Number of trailing zero bits common to all samples.
    int     rice_partition_order; ///< log2 of the number of Rice partitions (0–8).
    int     rice_k[256];        ///< Rice parameter k for each partition (k=15 → escape code).
    int32_t q_coeffs[32];       ///< Quantized LPC coefficients (in prediction order).
    uint32_t bits_cost;         ///< Exact total bits for this subframe (header + payload).
};

/**
 * @brief Encoding parameters for one complete FLAC frame (all channels).
 *
 * Produced by Optimizer::compute_block() and consumed by
 * FrameWriter::write_frame().
 */
struct BlockParams {
    uint32_t       block_size;    ///< Number of samples in this frame.
    int            stereo_mode;   ///< Channel coupling: 0=Independent, 8=Left-Side, 9=Right-Side, 10=Mid-Side.
    SubframeParams subframes[2];  ///< Per-channel subframe parameters (index 0 = left/mid, 1 = right/side).
    uint32_t       total_bits;    ///< Sum of subframe bits (header bits excluded).
};

/// @}


// ---------------------------------------------------------------------------
/// @defgroup optimizer Optimizer Class
/// @{
// ---------------------------------------------------------------------------

/**
 * @brief Finds the globally optimal FLAC encoding for a decoded PCM stream.
 *
 * ### Algorithm Overview
 *
 * 1. **Block candidate evaluation** — For every candidate start position and
 *    every block size in `{1024, 2048, 4096, 8192, 16384}`, compute_block()
 *    is called.  Each call evaluates all window/order/precision/stereo-mode
 *    combinations and returns the cheapest SubframeParams.  All calls are
 *    dispatched to a flat thread pool for full CPU utilization.
 *
 * 2. **Dynamic programming** — A shortest-path DP on the precomputed cost
 *    table finds the block-boundary sequence that minimizes total encoded bits.
 *    The DP itself is O(N×K) and runs in microseconds.
 *
 * 3. **Back-trace** — The optimal sequence of BlockParams is returned to the
 *    caller (Processor), which passes it to FrameWriter for serialization.
 */
class Optimizer {
public:
    /**
     * @brief Construct an Optimizer for a specific stream format.
     *
     * @param channels     Number of audio channels (1 or 2).
     * @param bps          Bits per sample (e.g. 16, 24).
     * @param windows      Apodization windows to test.  Empty → all 26 windows.
     * @param max_threads  Worker thread limit.  0 → all logical CPUs.
     */
    Optimizer(uint32_t channels, uint32_t bps,
              std::vector<WindowType> windows = {},
              int max_threads = 0);

    /**
     * @brief Find the optimal variable block-size partition for the stream.
     *
     * This is the main entry point.  It runs the full three-phase pipeline
     * (precompute → DP → back-trace) and returns one BlockParams per frame.
     *
     * @param pcm_data  Decoded PCM samples, indexed as `pcm_data[channel][sample]`.
     * @return          Ordered sequence of BlockParams covering the entire stream.
     */
    std::vector<BlockParams> find_optimal_block_partitioning(
        const std::vector<std::vector<int32_t>>& pcm_data);

    /**
     * @brief Find the cheapest encoding for a single channel block.
     *
     * Tries all windows, orders, precisions, and subframe types (Constant,
     * Verbatim, Fixed, LPC).  Picks the combination with the lowest bit cost.
     *
     * @param samples     Pointer to the first sample of this block.
     * @param block_size  Number of samples.
     * @param bps         Bits per sample for this channel.
     * @param windows     Windows to test.
     * @return            Best SubframeParams found.
     */
    [[nodiscard]] static SubframeParams optimize_subframe(
        const int32_t*              samples,
        uint32_t                    bsize,
        uint32_t                    bps,
        const std::vector<WindowType>& windows);

private:
    /// @cond INTERNAL

    // --- DP fast-path helpers (granule-based autocorrelation cache) ----------
    struct Granule { double autoc[33]; }; ///< Cached autocorrelation for one 1024-sample granule.
    std::vector<std::vector<Granule>> m_granules;
    void precompute_granules(const std::vector<std::vector<int32_t>>& pcm_data);
    [[nodiscard]] uint32_t estimate_lpc_bits_fast(int channel,
                                    uint32_t n_start, uint32_t n_end,
                                    int bps) const;

    // --- Exact subframe cost ------------------------------------------------
    [[nodiscard]] static uint32_t estimate_subframe_cost(
        const int32_t* samples, uint32_t bsize,
        int mode, int order, int precision, int wasted, int bps,
        SubframeParams* out_params = nullptr);

    static uint32_t calculate_rice_cost(
        const int32_t* residuals, uint32_t block_size,
        uint32_t order, SubframeParams* out_params);

    // --- LPC helpers --------------------------------------------------------
    /// Levinson-Durbin recursion for all orders 1..max_order simultaneously.
    static void compute_lpc_all_orders(
        const double* autoc, float out_coeffs[][32], int max_order);

    /// Single-order Levinson-Durbin wrapper (used in granule fast-path).
    static void compute_lpc_coefficients(
        const double* autoc, float* out_coeffs, int order);

    /// Apply apodization window and compute windowed samples as doubles.
    static void apply_window(
        const int32_t* samples, uint32_t N, int wasted_bits,
        WindowType wt, double* out);

    /// Fully-optimised BlockParams for one frame (all stereo modes, all windows).
    [[nodiscard]] BlockParams compute_block(
        const std::vector<std::vector<int32_t>>& pcm_data,
        uint64_t sample_start, uint32_t block_size) const;

    // --- Member state -------------------------------------------------------
    uint32_t              m_channels;
    uint32_t              m_bps;
    std::vector<WindowType> m_windows;
    int                   m_max_threads; ///< 0 = use all logical CPUs.

    /// @endcond
};

/// @}

#endif // OPTIMIZER_HPP
