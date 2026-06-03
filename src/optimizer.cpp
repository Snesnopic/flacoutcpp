#include "optimizer.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

Optimizer::Optimizer(const uint32_t channels, const uint32_t bits_per_sample, const uint32_t sample_rate)
    : m_channels(channels), m_bps(bits_per_sample), m_sample_rate(sample_rate) {}

void Optimizer::precompute_granules(const std::vector<std::vector<int32_t>>& pcm_data) {
    size_t num_granules = pcm_data[0].size() / 16;
    m_granules.assign(m_channels, std::vector<Granule>(num_granules));

    for (int c = 0; c < m_channels; ++c) {
        for (size_t g = 0; g < num_granules; ++g) {
            const int32_t* src = &pcm_data[c][g * 16];
            for (int i = 0; i <= 32; ++i) {
                double sum = 0;
                for (int j = 0; j < 16 - i; ++j) sum += (double)src[j] * src[j + i];
                m_granules[c][g].autoc[i] = sum;
            }
        }
    }
}

uint32_t Optimizer::estimate_lpc_bits_fast(int channel, uint32_t node_start, const uint32_t node_end, const int bps) const {
    double autoc[33] = {0};
    for (uint32_t g = node_start; g < node_end; ++g) {
        for (int i = 0; i <= 32; ++i) autoc[i] += m_granules[channel][g].autoc[i];
    }

    float coeffs[32];
    compute_lpc_coefficients(autoc, coeffs, 8);

    double err_sum = autoc[0];
    for (int i = 0; i < 8; ++i) err_sum -= (double)coeffs[i] * autoc[i + 1];
    
    uint32_t block_size = (node_end - node_start) * 16;
    if (err_sum <= 0) return 16 + bps;

    double bits_per_sample = 0.5 * std::log2(2.0 * M_PI * M_E * (err_sum / block_size));
    if (bits_per_sample < 1.0) bits_per_sample = 1.0;
    
    return 16 + static_cast<uint32_t>(block_size * bits_per_sample);
}

void Optimizer::compute_lpc_coefficients(const double* autoc, float* out_coeffs, int order) {
    std::vector<double> error(order + 1);
    std::vector<std::vector<double>> a(order + 1, std::vector<double>(order + 1));

    error[0] = autoc[0];
    if (error[0] <= 0) {
        for(int i=0; i<order; ++i) out_coeffs[i] = 0;
        return;
    }

    for (int i = 1; i <= order; ++i) {
        double lambda = autoc[i];
        for (int j = 1; j < i; ++j) lambda -= a[i - 1][j] * autoc[i - j];
        a[i][i] = lambda / error[i - 1];
        error[i] = error[i - 1] * (1.0 - a[i][i] * a[i][i]);
        if (error[i] <= 0) error[i] = 1e-10;
        for (int j = 1; j < i; ++j) a[i][j] = a[i - 1][j] - a[i][i] * a[i - 1][i - j];
    }
    for (int i = 0; i < order; ++i) out_coeffs[i] = (float)a[order][i + 1];
}

uint32_t Optimizer::calculate_rice_cost(const int32_t* residuals, uint32_t block_size, 
                                       uint32_t order, SubframeParams* out_params) {
    uint32_t best_total_bits = std::numeric_limits<uint32_t>::max();
    int best_partition_order = 0;
    int best_ks[256] = {0};

    for (int p_order = 0; p_order <= 8; ++p_order) {
        uint32_t num_partitions = 1 << p_order;
        if (block_size % num_partitions != 0) continue;
        
        uint32_t p_size = block_size / num_partitions;
        uint32_t total_p_bits = 4 * num_partitions; 
        int current_ks[256];

        for (uint32_t p = 0; p < num_partitions; ++p) {
            uint32_t best_p_k_bits = std::numeric_limits<uint32_t>::max();
            uint32_t start = p * p_size;
            uint32_t end = (p + 1) * p_size;
            int best_k = 0;

            for (int k = 0; k < 15; ++k) {
                uint32_t current_k_bits = 0;
                for (uint32_t i = std::max(start, order); i < end; ++i) {
                    uint32_t uval = (residuals[i] << 1) ^ (residuals[i] >> 31);
                    current_k_bits += (uval >> k) + 1 + k;
                }
                if (current_k_bits < best_p_k_bits) {
                    best_p_k_bits = current_k_bits;
                    best_k = k;
                }
            }
            total_p_bits += best_p_k_bits;
            current_ks[p] = best_k;
        }

        if (total_p_bits < best_total_bits) {
            best_total_bits = total_p_bits;
            best_partition_order = p_order;
            std::memcpy(best_ks, current_ks, num_partitions * sizeof(int));
        }
    }

    if (out_params) {
        out_params->rice_partition_order = best_partition_order;
        std::memcpy(out_params->rice_k, best_ks, (1 << best_partition_order) * sizeof(int));
    }
    return best_total_bits;
}

SubframeParams Optimizer::optimize_subframe(const int32_t* samples, uint32_t block_size, uint32_t bps) {
    SubframeParams best{};
    best.bits_cost = std::numeric_limits<uint32_t>::max();

    int wasted = 0;
    int32_t mask = 0;
    for(uint32_t i=0; i<block_size; ++i) mask |= samples[i];
    if (mask != 0) {
        while ((mask & 1) == 0) { mask >>= 1; wasted++; }
    }

    for (int mode : {0, 1, 2, 3}) {
        if (mode == 2) {
            for (int order = 0; order <= 4; ++order) {
                if (order >= block_size && order > 0) break;
                SubframeParams cur{};
                uint32_t cost = estimate_subframe_cost(samples, block_size, mode, order, 0, wasted, bps, &cur);
                if (cost < best.bits_cost) { best = cur; best.bits_cost = cost; }
            }
        } else if (mode == 3) {
            for (int order = 1; order <= 32; ++order) {
                if (order >= block_size) break;
                for (int prec = 8; prec <= 15; ++prec) {
                    SubframeParams cur{};
                    uint32_t cost = estimate_subframe_cost(samples, block_size, mode, order, prec, wasted, bps, &cur);
                    if (cost < best.bits_cost) { best = cur; best.bits_cost = cost; }
                }
            }
        } else {
            SubframeParams cur{};
            uint32_t cost = estimate_subframe_cost(samples, block_size, mode, 0, 0, wasted, bps, &cur);
            if (cost < best.bits_cost) { best = cur; best.bits_cost = cost; }
        }
    }
    return best;
}

uint32_t Optimizer::estimate_subframe_cost(const int32_t* samples, uint32_t block_size, 
                                          int mode, int order, int precision, int wasted, int bps, SubframeParams* out) {
    if (out) {
        out->mode = mode; out->order = order;
        out->lpc_precision = precision; out->wasted_bits = wasted;
    }

    uint32_t header = 8 + (wasted ? 1 + wasted : 0);
    if (mode == 0) {
        for(uint32_t i=1; i<block_size; ++i) if(samples[i] != samples[0]) return std::numeric_limits<uint32_t>::max();
        return header + (bps - wasted);
    }
    if (mode == 1) return header + (bps - wasted) * block_size;

    std::vector<int32_t> residuals(block_size);
    if (mode == 2) {
        for (uint32_t i = 0; i < block_size; ++i) {
            int32_t s = samples[i] >> wasted;
            if (i < static_cast<uint32_t>(order)) residuals[i] = s;
            else {
                int32_t s1 = samples[i-1] >> wasted;
                int32_t s2 = (i > 1) ? (samples[i-2] >> wasted) : 0;
                int32_t s3 = (i > 2) ? (samples[i-3] >> wasted) : 0;
                int32_t s4 = (i > 3) ? (samples[i-4] >> wasted) : 0;
                switch(order) {
                    case 0: residuals[i] = s; break;
                    case 1: residuals[i] = s - s1; break;
                    case 2: residuals[i] = s - 2*s1 + s2; break;
                    case 3: residuals[i] = s - 3*s1 + 3*s2 - s3; break;
                    case 4: residuals[i] = s - 4*s1 + 6*s2 - 4*s3 + s4; break;
                }
            }
        }
    } else {
        std::vector<double> f_samples(block_size);
        for(uint32_t i=0; i<block_size; ++i) f_samples[i] = (double)(samples[i] >> wasted);
        double autoc[33] = {0};
        for(int i=0; i<=order; ++i) {
            for(uint32_t j=0; j<block_size-i; ++j) autoc[i] += f_samples[j] * f_samples[j+i];
        }
        float lpc_coeffs[32];
        compute_lpc_coefficients(autoc, lpc_coeffs, order);
        int32_t q_coeffs[32];
        for(int i=0; i<order; ++i) q_coeffs[i] = (int32_t)std::round(lpc_coeffs[i] * (1 << (precision-1)));
        if (out) std::memcpy(out->q_coeffs, q_coeffs, order * sizeof(int32_t));

        for (uint32_t i = 0; i < block_size; ++i) {
            int32_t s = samples[i] >> wasted;
            if (i < (uint32_t)order) residuals[i] = s;
            else {
                int64_t prediction = 0;
                for (int j = 0; j < order; ++j) prediction += (int64_t)q_coeffs[j] * (samples[i-1-j] >> wasted);
                residuals[i] = s - (int32_t)(prediction >> (precision-1));
            }
        }
        header += 4 + 5 + (order * precision);
    }

    header += (order * (bps - wasted));
    return header + calculate_rice_cost(residuals.data(), block_size, order, out);
}

std::vector<BlockParams> Optimizer::find_optimal_block_partitioning(const std::vector<std::vector<int32_t>>& pcm_data) {
    precompute_granules(pcm_data);
    size_t num_nodes = pcm_data[0].size() / 16;
    
    std::vector<uint32_t> dp_cost(num_nodes + 1, std::numeric_limits<uint32_t>::max());
    std::vector<int> dp_parent(num_nodes + 1, -1);
    std::vector<int> dp_stereo_mode(num_nodes + 1, 0);

    dp_cost[0] = 0;
    std::cout << "Starting DP search over " << num_nodes << " nodes...\n";

    for (size_t i = 0; i < num_nodes; ++i) {
        if (i % 1000 == 0) std::cout << "\rProgress: " << (i * 100 / num_nodes) << "%" << std::flush;
        size_t max_j = std::min(i + 256, num_nodes);
        for (size_t j = i + 1; j <= max_j; ++j) {
            uint32_t best_total_b_bits = std::numeric_limits<uint32_t>::max();
            int best_sm = 0;

            uint32_t c_indep = 16;
            for (int c = 0; c < m_channels; ++c) c_indep += estimate_lpc_bits_fast(c, i, j, m_bps);
            best_total_b_bits = c_indep;

            if (m_channels == 2) {
                // Approximate stereo decorrelation cost
                uint32_t c_decorr = 16 + (estimate_lpc_bits_fast(0, i, j, m_bps) + estimate_lpc_bits_fast(1, i, j, m_bps + 1)) * 0.9;
                if (c_decorr < best_total_b_bits) {
                    best_total_b_bits = c_decorr;
                    best_sm = 10; // Mid-Side estimate
                }
            }

            if (dp_cost[i] + best_total_b_bits < dp_cost[j]) {
                dp_cost[j] = dp_cost[i] + best_total_b_bits;
                dp_parent[j] = (int)i;
                dp_stereo_mode[j] = best_sm;
            }
        }
    }
    std::cout << "\nDP complete. Recalculating chosen blocks...\n";

    std::vector<BlockParams> result;
    int curr = static_cast<int>(num_nodes);
    while(curr > 0) {
        int parent = dp_parent[curr];
        uint32_t b_size = (curr - parent) * 16;
        uint32_t sample_offset = parent * 16;
        int sm = dp_stereo_mode[curr];

        BlockParams bp{};
        bp.block_size = b_size;
        bp.stereo_mode = sm;
        
        if (sm == 0 || m_channels != 2) {
            for(int c=0; c<m_channels; ++c) bp.subframes[c] = optimize_subframe(&pcm_data[c][sample_offset], b_size, m_bps);
        } else {
            // Actual exhaustive search for best stereo mode during recalculation
            uint32_t best_bits = std::numeric_limits<uint32_t>::max();
            for (int mode : {8, 9, 10}) {
                std::vector<int32_t> ch0(b_size), ch1(b_size);
                for(uint32_t k=0; k<b_size; ++k) {
                    int32_t L = pcm_data[0][sample_offset+k];
                    int32_t R = pcm_data[1][sample_offset+k];
                    if (mode == 10) { ch0[k] = (L + R) >> 1; ch1[k] = L - R; }
                    else if (mode == 8) { ch0[k] = L; ch1[k] = L - R; }
                    else if (mode == 9) { ch0[k] = L - R; ch1[k] = R; }
                }
                SubframeParams s0 = optimize_subframe(ch0.data(), b_size, m_bps);
                SubframeParams s1 = optimize_subframe(ch1.data(), b_size, m_bps + 1);
                if (s0.bits_cost + s1.bits_cost < best_bits) {
                    best_bits = s0.bits_cost + s1.bits_cost;
                    bp.stereo_mode = mode;
                    bp.subframes[0] = s0;
                    bp.subframes[1] = s1;
                }
            }
        }
        result.push_back(bp);
        curr = parent;
    }
    std::reverse(result.begin(), result.end());
    
    size_t total_samples = pcm_data[0].size();
    if (total_samples % 16 != 0) {
        uint32_t rem_size = total_samples % 16;
        uint32_t offset = total_samples - rem_size;
        BlockParams rem_bp;
        rem_bp.block_size = rem_size;
        rem_bp.stereo_mode = 0;
        for(int c=0; c<m_channels; ++c) rem_bp.subframes[c] = optimize_subframe(&pcm_data[c][offset], rem_size, m_bps);
        result.push_back(rem_bp);
    }

    return result;
}
