#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <cstdint>

struct SubframeParams {
    int mode;           // 0: Constant, 1: Verbatim, 2: Fixed, 3: LPC
    int order;          // Fixed/LPC order
    int lpc_precision;  // LPC quantization bits
    int lpc_shift;
    int wasted_bits;
    int rice_partition_order;
    int rice_k[256];
    int32_t q_coeffs[32];
    uint32_t bits_cost;
};

struct BlockParams {
    uint32_t block_size;
    int stereo_mode;    // 0: Indep, 8: Left-Side, 9: Right-Side, 10: Mid-Side
    SubframeParams subframes[2];
    uint32_t total_bits;
};

class Optimizer {
public:
    Optimizer(uint32_t channels, uint32_t bps);

    std::vector<BlockParams> find_optimal_block_partitioning(const std::vector<std::vector<int32_t>>& pcm_data);
    static SubframeParams optimize_subframe(const int32_t* samples, uint32_t block_size, uint32_t bps);

private:
    static uint32_t estimate_subframe_cost(const int32_t* samples, uint32_t block_size,
                                   int mode, int order, int precision, int wasted, int bps, SubframeParams* out_params = nullptr);
    
    [[nodiscard]] uint32_t estimate_lpc_bits_fast(int channel, uint32_t node_start, uint32_t node_end, int bps) const;

    static uint32_t calculate_rice_cost(const int32_t* residuals, uint32_t block_size,
                                uint32_t order, SubframeParams* out_params);

    static void compute_lpc_coefficients(const double* autoc, float* out_coeffs, int order);
    
    struct Granule {
        double autoc[33];
    };
    std::vector<std::vector<Granule>> m_granules;
    void precompute_granules(const std::vector<std::vector<int32_t>>& pcm_data);

    uint32_t m_channels;
    uint32_t m_bps;
};

#endif // OPTIMIZER_HPP
