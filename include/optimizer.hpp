/**
 * @file optimizer.hpp
 * @brief Exhaustive FLAC LPC optimizer and variable block-size DP partitioner.
 *
 * The Optimizer class is the computational core of flacoutcpp.  For each
 * candidate block it evaluates every combination of:
 *   - 26 standard apodization windows (WindowType; plus experimental
 *     windows opt-in via an explicit window list)
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
#include <memory>
#include <string>

#include "gpu.hpp"

namespace flacoutcpp {

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
    // Experimental windows: reachable only via an explicit -w list; excluded
    // from all_window_types() until measurement earns them promotion
    // (WINDOWS_PLAN.md). Keeping them past this line keeps every default
    // window set — and therefore every default-mode bitstream — unchanged.
    LANCZOS,                  ///< Lanczos (sinc) window. Experimental, -w only.
    BOHMAN,                   ///< Bohman window. Experimental, -w only.
    PARZEN,                   ///< Parzen (cubic B-spline) window. Experimental, -w only.
    PLANCKTAPER_010,          ///< Planck-taper window, ε = 0.10. Experimental, -w only.
    PLANCKTAPER_025,          ///< Planck-taper window, ε = 0.25. Experimental, -w only.
    // Tier 2: 3-partition Tukeys, libFLAC-faithful geometry — the exact
    // windows `flac -A partial_tukey(3)` / `punchout_tukey(3)` builds
    // (parts at thirds, 10%/20% overlap, taper p = 0.2).
    PARTIAL_TUKEY_3_1,        ///< partial_tukey(3) part 1 of 3. Experimental, -w only.
    PARTIAL_TUKEY_3_2,        ///< partial_tukey(3) part 2 of 3. Experimental, -w only.
    PARTIAL_TUKEY_3_3,        ///< partial_tukey(3) part 3 of 3. Experimental, -w only.
    PUNCHOUT_TUKEY_3_1,       ///< punchout_tukey(3) hole 1 of 3. Experimental, -w only.
    PUNCHOUT_TUKEY_3_2,       ///< punchout_tukey(3) hole 2 of 3. Experimental, -w only.
    PUNCHOUT_TUKEY_3_3,       ///< punchout_tukey(3) hole 3 of 3. Experimental, -w only.
    // Tier 2: 3-partition Tukeys, house geometry — extends the existing
    // 2-partition style (span/hole + offset, p = 0.5).
    PARTIAL_TUKEY_3H_000,     ///< House partial Tukey, span 1/3 at 0.00. Experimental, -w only.
    PARTIAL_TUKEY_3H_033,     ///< House partial Tukey, span 1/3 at 0.33. Experimental, -w only.
    PARTIAL_TUKEY_3H_067,     ///< House partial Tukey, span 1/3 at 0.67. Experimental, -w only.
    PUNCHOUT_TUKEY_3H_025,    ///< House punchout Tukey, hole 0.25 at 0.25. Experimental, -w only.
    PUNCHOUT_TUKEY_3H_050,    ///< House punchout Tukey, hole 0.25 at 0.50. Experimental, -w only.
    /// Punchout Tukey with the hole at the very start (0.00–0.33) — the
    /// offset the standard _2 pair (0.33, 0.67) leaves uncovered.
    PUNCHOUT_TUKEY_2_000,     ///< Punchout Tukey, hole 0.33 at 0.00. Experimental, -w only.
    // Tier 2: asymmetric windows (absent from libFLAC entirely).
    EXPDECAY_2,               ///< e^(-2·i/(N-1)) decaying exponential. Experimental, -w only.
    EXPDECAY_4,               ///< e^(-4·i/(N-1)) decaying exponential. Experimental, -w only.
    EXPATTACK_2,              ///< Mirrored EXPDECAY_2 (rising). Experimental, -w only.
    EXPATTACK_4,              ///< Mirrored EXPDECAY_4 (rising). Experimental, -w only.
    ATTACKDECAY_005,          ///< Exp rise over first 5%, half-cosine decay. Experimental, -w only.
    ATTACKDECAY_010,          ///< Exp rise over first 10%, half-cosine decay. Experimental, -w only.
    ATTACKDECAY_020,          ///< Exp rise over first 20%, half-cosine decay. Experimental, -w only.
    // Tier 3: DPSS (Slepian) — maximal spectral-energy concentration for a
    // given time-bandwidth product NW. Computed by a deterministic
    // fixed-iteration eigensolve (see compute_dpss).
    DPSS_2,                   ///< DPSS window, NW = 2. Experimental, -w only.
    DPSS_3,                   ///< DPSS window, NW = 3. Experimental, -w only.
    DPSS_4,                   ///< DPSS window, NW = 4. Experimental, -w only.
    // Runtime-loaded windows: shapes read from a knot file by
    // register_custom_window(), reachable only as `-w custom:<file>`. These
    // slots hold no shape until something registers one.
    CUSTOM_0,                 ///< Runtime-loaded window slot 0. -w custom:<file> only.
    CUSTOM_1,                 ///< Runtime-loaded window slot 1. -w custom:<file> only.
    CUSTOM_2,                 ///< Runtime-loaded window slot 2. -w custom:<file> only.
    CUSTOM_3,                 ///< Runtime-loaded window slot 3. -w custom:<file> only.
    COUNT,                    ///< Sentinel — total number of window types.
    EXPERIMENTAL_BEGIN = LANCZOS, ///< First experimental (opt-in) window.
    CUSTOM_BEGIN = CUSTOM_0   ///< First runtime-loaded window slot.
};

/// Number of runtime-loadable window slots (WindowType::CUSTOM_0 onwards).
constexpr int MAX_CUSTOM_WINDOWS =
    (int)WindowType::COUNT - (int)WindowType::CUSTOM_BEGIN;

/**
 * @brief Load a window shape from a knot file into a free custom slot.
 *
 * The file is whitespace-, comma-, or newline-separated decimal values (`#`
 * begins a comment); they are the window's coefficients at @c K evenly spaced
 * positions spanning the whole block, and are linearly interpolated to
 * whatever block size the encoder needs. Two knots therefore describe a ramp,
 * and @c K equal to the block size reproduces the shape exactly. Absolute
 * scale is irrelevant — the LPC search is scale-invariant — so no
 * normalisation is applied; the values are used as written.
 *
 * Registering the same path twice returns the original slot rather than
 * consuming a second one. Coefficient tables for the DP block sizes are built
 * here, so **all registration must happen before encoding starts**; the
 * registry is read lock-free by the worker threads and never mutated
 * afterwards.
 *
 * @param path   File to read.
 * @param error  Set to a human-readable reason when registration fails.
 * @return       The slot's WindowType, or @c WindowType::COUNT on failure.
 */
WindowType register_custom_window(const std::string& path, std::string* error);

/**
 * @brief The window set the estimated-DP modes search by default.
 *
 * Exported so the startup banner can name the real list. It used to print a
 * hardcoded copy, which drifted: the list grew to ten while the banner still
 * said the original four, and CLAUDE.md tells the reader both to trust the
 * banner and never to judge a window against those four.
 */
std::vector<WindowType> default_shortlist();

/**
 * @brief Parse a window type from its name (case-insensitive).
 * @param raw  Human-readable window name (e.g. @c "hann", @c "blackman").
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
 * @brief Return a vector of window types (excluding COUNT).
 *
 * By default returns the 26 standard windows — the set exact-DP mode (-e)
 * sweeps. Pass @p include_experimental to also get the experimental windows,
 * which are otherwise reachable only through an explicit @c -w list.
 */
std::vector<WindowType> all_window_types(bool include_experimental = false);

/**
 * @brief Fill @p out with @p N apodization coefficients for @p wt.
 *
 * Exported for the pure-GPU encoder, which uploads a coefficient table to the
 * device once at startup instead of applying windows on the host. Callers
 * outside the optimizer should have no other reason to want this.
 */
void window_coefficients(WindowType wt, uint32_t N, double* out);

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
    int     rice_method;        ///< Residual coding method: 0 = RICE (4-bit k, k ≤ 14), 1 = RICE2 (5-bit k, k ≤ 30; needed when residuals outgrow k=14, i.e. high-bps content).
    int     rice_k[256];        ///< Rice parameter k per partition. Low byte: k, or the method's escape marker (15/31) with the raw bit-width in the high bits.
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
    int32_t        reuse_index = -1; ///< ≥0: emit input frame #reuse_index verbatim (frame reuse) instead of encoding; other fields besides block_size are unused.
};

/**
 * @brief One input frame offered to the partitioning DP as an exact-cost
 *        alternative edge (frame reuse under exact-DP mode).
 *
 * The caller (Processor) computes @c frame_bytes by actually rewriting the
 * input frame to this stream's header conventions, so the DP compares it
 * against re-encoded candidates on equal, exact terms.
 */
struct ReuseEdge {
    uint64_t start_sample;  ///< Absolute first sample (must lie on the DP grid).
    uint32_t block_size;    ///< Samples covered (must end on the DP grid).
    uint32_t frame_bytes;   ///< Size of the rewritten frame, in bytes.
    uint32_t input_index;   ///< Caller's index for emitting the frame later.
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
     * @param sample_rate  Stream sample rate in Hz (prices frame headers in the DP).
     * @param windows      Apodization windows to test.  Empty → the default
     *                     set (all 26 standard windows under -e).
     * @param max_threads  Worker thread limit.  0 → all logical CPUs.
     * @param max_candidates  Ranked-search budget: the number of
     *        (window, order) pairs fully evaluated per subframe.  0 (default)
     *        evaluates every pair, which is the exhaustive behaviour.
     * @param patience  Consecutive non-improving candidates tolerated before
     *        the ranked scan stops; @c max_candidates becomes a floor rather
     *        than a ceiling.  0 (default) keeps the plain top-N cut.
     * @param precision_rungs  How many of the 8 LPC precisions to encode per
     *        candidate, chosen by the analytic ladder model.  0 (default)
     *        encodes all of them, which is the pre-existing behaviour.
     */
    Optimizer(uint32_t channels, uint32_t bps, uint32_t sample_rate,
              std::vector<WindowType> windows = {},
              unsigned max_threads = 0,
              bool exhaustive = false,
              bool verbose = true,
              unsigned max_candidates = 0,
              bool adaptive_windows = false,
              unsigned patience = 0,
              unsigned precision_rungs = 0,
              std::vector<uint32_t> dp_candidates = {},
              unsigned lattice_sweeps = 0,
              bool     use_gpu = false,
              unsigned gpu_min_batch = 0,
              unsigned gpu_partition_cap = 8,
              unsigned gpu_slots = 3,
              unsigned gpu_duty = 100);

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
     * @brief Offer input frames to the partitioning DP as alternative edges.
     *
     * Exact-DP mode only (the estimated DP would compare exact reuse costs
     * against granule estimates, biasing the partition toward copying);
     * ignored otherwise. Blocks chosen from these edges come back with
     * BlockParams::reuse_index set instead of encoded parameters.
     */
    void set_reuse_edges(std::vector<ReuseEdge> edges) { m_reuse_edges = std::move(edges); }

    /**
     * @brief Find the cheapest encoding for a single channel block.
     *
     * Tries every subframe type (Constant, Verbatim, Fixed, LPC); for LPC,
     * ranks all (window, order) pairs by Levinson-Durbin prediction error and
     * fully evaluates the best @p max_candidates of them across the whole
     * precision ladder. Picks the combination with the lowest bit cost.
     *
     * @param samples        Pointer to the first sample of this block.
     * @param bsize          Number of samples.
     * @param bps            Bits per sample for this channel.
     * @param windows        Windows to test.
     * @param max_candidates Ranked (window, order) pairs to fully evaluate;
     *                       0 means no limit (exhaustive sweep).
     * @param precision_rungs LPC precisions encoded per candidate, picked by
     *                       the analytic ladder model; 0 encodes all 8.
     * @return               Best SubframeParams found.
     */
    [[nodiscard]] static SubframeParams optimize_subframe(
        const int32_t*              samples,
        uint32_t                    bsize,
        uint32_t                    bps,
        const std::vector<WindowType>& windows,
        unsigned                    max_candidates = 0,
        unsigned                    patience = 0,
        unsigned                    precision_rungs = 0,
        unsigned                    lattice_sweeps = 0,
        GpuEvaluator*               gpu = nullptr);

private:
    /// @cond INTERNAL

    // --- DP fast-path helpers (granule-based autocorrelation cache) ----------
    // Lags 0..8 only: estimate_lpc_bits_fast runs Levinson at fixed order 8,
    // which never reads past autoc[8].
    struct Granule { double autoc[9]; }; ///< Cached autocorrelation for one 16-sample granule.
    std::vector<std::vector<Granule>> m_granules;
    void precompute_granules(const std::vector<std::vector<int32_t>>& pcm_data);
    /// @param pcm_data Source samples; the estimate builds real residuals and
    ///        counts their Rice bits rather than modelling them.
    [[nodiscard]] uint32_t estimate_lpc_bits_fast(
                                    const std::vector<std::vector<int32_t>>& pcm_data,
                                    int channel,
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
    ///
    /// @param out_err  Optional; receives the residual energy after predicting
    ///        at each order, with out_err[0] the unpredicted energy. Entries
    ///        the recursion did not reach (it stops early on an unstable
    ///        reflection coefficient) are left negative. Ranked search uses
    ///        out_err[ord]/out_err[0] to estimate an order's cost without
    ///        actually building its residuals.
    static void compute_lpc_all_orders(
        const double* autoc, float out_coeffs[][32], int max_order,
        double* out_err = nullptr);

    /// Single-order Levinson-Durbin wrapper (used in granule fast-path).
    static void compute_lpc_coefficients(
        const double* autoc, float* out_coeffs, int order);

    /// Apply apodization window and compute windowed samples as doubles.
    static void apply_window(
        const int32_t* samples, uint32_t N, int wasted_bits,
        WindowType wt, double* out);

    /// Optimized BlockParams for one frame (can use heuristic or exhaustive).
    [[nodiscard]] BlockParams compute_block(
        const std::vector<std::vector<int32_t>>& pcm_data,
        uint64_t sample_start, uint32_t block_size) const;

    /// Adaptive per-subframe window selection (estimated-DP modes only):
    /// pick a 4-window set for the block spanning [sample_start,
    /// sample_start + block_size) from the cached granule statistics —
    /// stationarity (variance of granule energies), transient position
    /// (energy argmax), and spectral tilt (lag1/lag0). Same set size as the
    /// fixed shortlist, so analysis cost is unchanged; only membership adapts.
    [[nodiscard]] std::vector<WindowType> select_windows(
        uint64_t sample_start, uint32_t block_size) const;

    // --- Member state -------------------------------------------------------
    uint32_t              m_channels;
    uint32_t              m_bps;
    uint32_t              m_sample_rate;
    std::vector<WindowType> m_windows;
    unsigned              m_max_threads;
    bool                  m_exhaustive;
    bool                  m_verbose;
    unsigned              m_max_candidates;
    bool                  m_adaptive;
    unsigned              m_patience;
    unsigned              m_precision_rungs;
    /// Print the GPU's share of the search; no-op without -G.
    void report_gpu(uint64_t total_candidates) const;

    unsigned              m_lattice_sweeps;
    bool                  m_use_gpu = false;
    std::unique_ptr<GpuEvaluator> m_gpu;

    /// DP block-size ladder, ascending. Every entry is a multiple of
    /// m_dp_step (itself a multiple of the 16-sample granule), because the DP
    /// places nodes every m_dp_step samples and each edge spans exactly one
    /// candidate — a size that is not a multiple of the step could never land
    /// on a node. That is also why 65535 is unreachable: it is odd.
    std::vector<uint32_t> m_dp_candidates;
    uint32_t              m_dp_step = 1024;
    std::vector<ReuseEdge> m_reuse_edges;

    /// True when block costs come from real encodes rather than the granule
    /// estimate: exact DP, all four stereo modes, full precision sweep.
    /// Ranked search is cheap enough per block to afford all of that.
    // Exact-DP mode: every (node, candidate) block in the partitioning DP is
    // fully encoded (and all four stereo modes evaluated), rather than priced
    // by the granule estimates. Orthogonal to m_max_candidates, which bounds
    // the per-subframe LPC search depth in every mode.
    [[nodiscard]] bool full_search() const { return m_exhaustive; }

    /// @endcond
};

/// @}

} // namespace flacoutcpp

#endif // OPTIMIZER_HPP
