#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// Apodization window types used for LPC autocorrelation analysis.
// Applying different windows before computing the autocorrelation changes the
// LPC coefficients, which in turn changes the residuals and Rice cost.
// Testing all windows per block finds the globally optimal LPC coefficients.
// ---------------------------------------------------------------------------
enum class WindowType {
    RECTANGULAR = 0,
    BARTLETT,
    BARTLETT_HANN,
    BLACKMAN,
    BLACKMAN_HARRIS_4TERM_92DB,
    CONNES,
    FLATTOP,
    GAUSS_025,
    GAUSS_0125,
    HAMMING,
    HANN,
    KAISER_BESSEL,
    NUTTALL,
    TRIANGLE,
    WELCH,
    TUKEY_005,
    TUKEY_010,
    TUKEY_020,
    TUKEY_050,
    TUKEY_075,
    TUKEY_090,
    PARTIAL_TUKEY_2_000,
    PARTIAL_TUKEY_2_033,
    PARTIAL_TUKEY_2_067,
    PUNCHOUT_TUKEY_2_033,
    PUNCHOUT_TUKEY_2_067,
    COUNT
};

// Parse a window name (case-insensitive). Returns COUNT if unrecognised.
WindowType window_from_name(const std::string& name);
std::string window_to_name(WindowType wt);

// Returns all window types (the full set for maximum compression).
std::vector<WindowType> all_window_types();

// ---------------------------------------------------------------------------
// Subframe and block parameter structures produced by the Optimizer.
// These are used directly by the FrameWriter to serialize the bitstream.
// ---------------------------------------------------------------------------
struct SubframeParams {
    int     mode;               // 0=Constant, 1=Verbatim, 2=Fixed, 3=LPC
    int     order;              // predictor order (Fixed: 0-4, LPC: 1-32)
    int     lpc_precision;      // quantization bits for LPC coefficients (8-15)
    int     lpc_shift;          // right-shift applied to dot product (= precision-1)
    int     wasted_bits;        // trailing zero bits in every sample
    int     rice_partition_order;
    int     rice_k[256];        // rice parameter per partition
    int32_t q_coeffs[32];       // quantized LPC coefficients
    uint32_t bits_cost;         // estimated total bits for this subframe
};

struct BlockParams {
    uint32_t      block_size;
    int           stereo_mode;  // 0=Independent, 8=Left-Side, 9=Right-Side, 10=Mid-Side
    SubframeParams subframes[2];
    uint32_t      total_bits;
};

// ---------------------------------------------------------------------------
// Optimizer: finds the globally optimal block partitioning + encoding params.
// ---------------------------------------------------------------------------
class Optimizer {
public:
    // @param channels     Number of audio channels.
    // @param bps          Bits per sample.
    // @param windows      List of apodization windows to test. Empty = use all.
    // @param max_threads  Maximum parallel threads (0 = use all logical CPUs).
    Optimizer(uint32_t channels, uint32_t bps,
              std::vector<WindowType> windows = {},
              int max_threads = 0);

    // Run the full optimisation pipeline:
    //   1. Precompute granule autocorrelations for the DP.
    //   2. DP to find optimal block boundaries.
    //   3. Exhaustive per-block subframe search (multi-window LPC, all stereo modes).
    std::vector<BlockParams> find_optimal_block_partitioning(
        const std::vector<std::vector<int32_t>>& pcm_data);

    // Public so the FrameWriter can call it for verification / unit tests.
    static SubframeParams optimize_subframe(
        const int32_t*              samples,
        uint32_t                    block_size,
        uint32_t                    bps,
        const std::vector<WindowType>& windows);

private:
    // --- DP fast-path helpers (granule-based approximation) ----------------
    struct Granule { double autoc[33]; };
    std::vector<std::vector<Granule>> m_granules;
    void precompute_granules(const std::vector<std::vector<int32_t>>& pcm_data);
    uint32_t estimate_lpc_bits_fast(int channel,
                                    uint32_t node_start, uint32_t node_end,
                                    int bps) const;

    // --- Subframe cost estimation (exact, used in final pass) --------------
    static uint32_t estimate_subframe_cost(
        const int32_t* samples, uint32_t block_size,
        int mode, int order, int precision, int wasted, int bps,
        SubframeParams* out_params = nullptr);

    static uint32_t calculate_rice_cost(
        const int32_t* residuals, uint32_t block_size,
        uint32_t order, SubframeParams* out_params);

    // --- LPC helpers -------------------------------------------------------
    // Levinson-Durbin for all orders 1..max_order at once.
    // out_coeffs[i][j] = coefficient j+1 for predictor of order i+1.
    static void compute_lpc_all_orders(
        const double* autoc, float out_coeffs[][32], int max_order);

    // Legacy single-order wrapper (used by the DP fast-path).
    static void compute_lpc_coefficients(
        const double* autoc, float* out_coeffs, int order);

    // Apply apodization window to (samples >> wasted_bits), writing doubles.
    static void apply_window(
        const int32_t* samples, uint32_t N, int wasted_bits,
        WindowType wt, double* out);

    // Compute the fully-optimised BlockParams for one block.
    // Tries all stereo modes; picks the cheapest.
    // Used by the DP to get exact (not estimated) bit costs.
    BlockParams compute_block(
        const std::vector<std::vector<int32_t>>& pcm_data,
        uint64_t sample_start, uint32_t block_size) const;

    // --- Internal state ----------------------------------------------------
    uint32_t              m_channels;
    uint32_t              m_bps;
    std::vector<WindowType> m_windows;
    int                   m_max_threads;  // 0 = use all logical CPUs
};



#endif // OPTIMIZER_HPP
