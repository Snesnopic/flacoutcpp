#include "optimizer.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifdef FLACOUT_INSTRUMENT
#include <cstdio>
struct InstrCounters {
    std::atomic<uint64_t> subframes{0};
    std::atomic<uint64_t> windows_run{0};
    std::atomic<uint64_t> order_iters{0};       // (window,order) pairs entered
    std::atomic<uint64_t> order_pruned_break{0};// orders skipped by hdr_min break
    std::atomic<uint64_t> prec_iters{0};        // (window,order,prec) entered
    std::atomic<uint64_t> prec_pruned_break{0};
    std::atomic<uint64_t> overflow_skips{0};
    std::atomic<uint64_t> residual_calls{0};    // residual loops actually run
    std::atomic<uint64_t> residual_macs{0};     // int64 multiply-accumulates
    std::atomic<uint64_t> rice_calls{0};
    std::atomic<uint64_t> rice_scan_samples{0}; // residuals scanned in sums pass
    std::atomic<uint64_t> rice_k_ops{0};        // (u>>k) accumulations
    std::atomic<uint64_t> rice_fold_ops{0};
    std::atomic<uint64_t> rice_chunk_fast{0};  // 32-bit lane accumulation held
    std::atomic<uint64_t> rice_chunk_slow{0};  // OR proved it could wrap; redone in 64-bit
    std::atomic<uint64_t> autoc_macs{0};
    std::atomic<uint64_t> window_samples{0};
    std::atomic<uint64_t> win_order_hist[33]{};
    std::atomic<uint64_t> best_order_hist[33]{};
    std::atomic<uint64_t> best_prec_hist[16]{};
    std::atomic<uint64_t> best_window_hist[32]{};
    ~InstrCounters() {
        std::fprintf(stderr, "\n===== INSTRUMENTATION =====\n");
        std::fprintf(stderr, "subframes optimized     : %llu\n", (unsigned long long)subframes);
        std::fprintf(stderr, "windows evaluated       : %llu\n", (unsigned long long)windows_run);
        std::fprintf(stderr, "window samples          : %llu\n", (unsigned long long)window_samples);
        std::fprintf(stderr, "autocorrelation MACs    : %llu\n", (unsigned long long)autoc_macs);
        std::fprintf(stderr, "(win,ord) entered       : %llu   pruned-by-break: %llu\n",
                     (unsigned long long)order_iters, (unsigned long long)order_pruned_break);
        std::fprintf(stderr, "(win,ord,prec) entered  : %llu   pruned-by-break: %llu\n",
                     (unsigned long long)prec_iters, (unsigned long long)prec_pruned_break);
        std::fprintf(stderr, "overflow skips          : %llu\n", (unsigned long long)overflow_skips);
        std::fprintf(stderr, "residual loops run      : %llu\n", (unsigned long long)residual_calls);
        std::fprintf(stderr, "residual MACs           : %llu\n", (unsigned long long)residual_macs);
        std::fprintf(stderr, "rice calls              : %llu\n", (unsigned long long)rice_calls);
        std::fprintf(stderr, "rice residuals scanned  : %llu\n", (unsigned long long)rice_scan_samples);
        std::fprintf(stderr, "rice (u>>k) ops         : %llu\n", (unsigned long long)rice_k_ops);
        std::fprintf(stderr, "rice fold ops           : %llu\n", (unsigned long long)rice_fold_ops);
        {
            unsigned long long f = rice_chunk_fast, sl = rice_chunk_slow;
            std::fprintf(stderr, "rice chunks fast/slow   : %llu / %llu (%.4f%% fell back)\n",
                         f, sl, (f + sl) ? 100.0 * (double)sl / (double)(f + sl) : 0.0);
        }
        std::fprintf(stderr, "WINNING order histogram:\n");
        for (int i = 1; i <= 32; ++i)
            if (best_order_hist[i]) std::fprintf(stderr, "   ord %2d: %llu\n", i, (unsigned long long)best_order_hist[i]);
        std::fprintf(stderr, "WINNING precision histogram:\n");
        for (int i = 0; i < 16; ++i)
            if (best_prec_hist[i]) std::fprintf(stderr, "   prec %2d: %llu\n", i, (unsigned long long)best_prec_hist[i]);
        std::fprintf(stderr, "WINNING window histogram:\n");
        for (int i = 0; i < 32; ++i)
            if (best_window_hist[i]) std::fprintf(stderr, "   %-22s: %llu\n",
                window_to_name((WindowType)i).c_str(), (unsigned long long)best_window_hist[i]);
        std::fprintf(stderr, "===========================\n");
    }
};
static InstrCounters g_instr;
#define INSTR(x) do { x; } while (0)
#else
#define INSTR(x) do {} while (0)
#endif

#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif
#ifndef M_E
#define M_E   2.71828182845904523536
#endif

// ============================================================
// WindowType utilities
// ============================================================

std::vector<WindowType> all_window_types() {
    std::vector<WindowType> out;
    for (int i = 0; i < (int)WindowType::COUNT; ++i)
        out.push_back((WindowType)i);
    return out;
}

std::string window_to_name(WindowType wt) {
    switch (wt) {
        case WindowType::RECTANGULAR:              return "rect";
        case WindowType::BARTLETT:                 return "bartlett";
        case WindowType::BARTLETT_HANN:            return "bartletthann";
        case WindowType::BLACKMAN:                 return "blackman";
        case WindowType::BLACKMAN_HARRIS_4TERM_92DB: return "blackmanharris";
        case WindowType::CONNES:                   return "connes";
        case WindowType::FLATTOP:                  return "flattop";
        case WindowType::GAUSS_025:                return "gauss025";
        case WindowType::GAUSS_0125:               return "gauss0125";
        case WindowType::HAMMING:                  return "hamming";
        case WindowType::HANN:                     return "hann";
        case WindowType::KAISER_BESSEL:            return "kaiserbessel";
        case WindowType::NUTTALL:                  return "nuttall";
        case WindowType::TRIANGLE:                 return "triangle";
        case WindowType::WELCH:                    return "welch";
        case WindowType::TUKEY_005:                return "tukey005";
        case WindowType::TUKEY_010:                return "tukey010";
        case WindowType::TUKEY_020:                return "tukey020";
        case WindowType::TUKEY_050:                return "tukey050";
        case WindowType::TUKEY_075:                return "tukey075";
        case WindowType::TUKEY_090:                return "tukey090";
        case WindowType::PARTIAL_TUKEY_2_000:      return "partialtukey2";
        case WindowType::PARTIAL_TUKEY_2_033:      return "partialtukey2_033";
        case WindowType::PARTIAL_TUKEY_2_067:      return "partialtukey2_067";
        case WindowType::PUNCHOUT_TUKEY_2_033:     return "punchouttukey2_033";
        case WindowType::PUNCHOUT_TUKEY_2_067:     return "punchouttukey2_067";
        default:                                   return "unknown";
    }
}

WindowType window_from_name(const std::string& raw) {
    std::string n;
    for (char c : raw) n += (char)std::tolower((unsigned char)c);
    // Remove common separators so "blackman_harris" and "blackmanharris" both work
    std::string clean;
    for (char c : n) if (c != '_' && c != '-' && c != ' ') clean += c;

    if (clean == "rect"       || clean == "rectangular") return WindowType::RECTANGULAR;
    if (clean == "bartlett")                              return WindowType::BARTLETT;
    if (clean == "bartletthann" || clean == "bh")        return WindowType::BARTLETT_HANN;
    if (clean == "blackman")                              return WindowType::BLACKMAN;
    if (clean == "blackmanharris" || clean == "bh4")     return WindowType::BLACKMAN_HARRIS_4TERM_92DB;
    if (clean == "connes")                                return WindowType::CONNES;
    if (clean == "flattop")                               return WindowType::FLATTOP;
    if (clean == "gauss025")                              return WindowType::GAUSS_025;
    if (clean == "gauss0125")                             return WindowType::GAUSS_0125;
    if (clean == "hamming")                               return WindowType::HAMMING;
    if (clean == "hann")                                  return WindowType::HANN;
    if (clean == "kaiserbessel" || clean == "kb")         return WindowType::KAISER_BESSEL;
    if (clean == "nuttall")                               return WindowType::NUTTALL;
    if (clean == "triangle")                              return WindowType::TRIANGLE;
    if (clean == "welch")                                 return WindowType::WELCH;
    if (clean == "tukey005")                              return WindowType::TUKEY_005;
    if (clean == "tukey010")                              return WindowType::TUKEY_010;
    if (clean == "tukey020")                              return WindowType::TUKEY_020;
    if (clean == "tukey050" || clean == "tukey")          return WindowType::TUKEY_050;
    if (clean == "tukey075")                              return WindowType::TUKEY_075;
    if (clean == "tukey090")                              return WindowType::TUKEY_090;
    if (clean == "partialtukey2")                         return WindowType::PARTIAL_TUKEY_2_000;
    if (clean == "partialtukey2033")                      return WindowType::PARTIAL_TUKEY_2_033;
    if (clean == "partialtukey2067")                      return WindowType::PARTIAL_TUKEY_2_067;
    if (clean == "punchouttukey2033")                     return WindowType::PUNCHOUT_TUKEY_2_033;
    if (clean == "punchouttukey2067")                     return WindowType::PUNCHOUT_TUKEY_2_067;
    return WindowType::COUNT; // unrecognised
}

// ============================================================
// Optimizer constructor
// ============================================================

Optimizer::Optimizer(uint32_t channels, uint32_t bps,
                     std::vector<WindowType> windows,
                     unsigned max_threads,
                     bool exhaustive,
                     bool verbose,
                     unsigned max_candidates)
    : m_channels(channels), m_bps(bps), m_max_threads(max_threads),
      m_exhaustive(exhaustive), m_verbose(verbose), m_max_candidates(max_candidates)
{
    if (windows.empty()) {
        // Ranked search wants the widest window set to choose from: it pays per
        // candidate evaluated, not per window offered, so restricting the set
        // only removes options the ranking could have picked.
        if (full_search()) {
            m_windows = all_window_types();
        } else {
            m_windows = {WindowType::TUKEY_050, WindowType::HANN, WindowType::WELCH, WindowType::RECTANGULAR};
        }
    } else {
        m_windows = std::move(windows);
    }
}

// ============================================================
// Window application
// ============================================================
// All window formulas match libFLAC's window.c exactly.

void Optimizer::apply_window(
    const int32_t* samples, uint32_t N, int wasted_bits,
    WindowType wt, double* out)
{
    auto fN = (double)(N - 1);
    auto fNh = fN / 2.0;

    for (uint32_t i = 0; i < N; ++i) {
        double x = (double)(samples[i] >> wasted_bits);
        double w = 1.0;
        double fi = (double)i;

        switch (wt) {
        case WindowType::RECTANGULAR:
            w = 1.0; break;

        case WindowType::BARTLETT:
            if (N & 1) w = (fi <= fNh) ? (2.0*fi/fN) : (2.0 - 2.0*fi/fN);
            else       w = (fi <= (double)(N/2-1)) ? (2.0*fi/fN) : (2.0 - 2.0*fi/fN);
            break;

        case WindowType::BARTLETT_HANN:
            w = 0.62 - 0.48*std::abs(fi/fN - 0.5) - 0.38*std::cos(2.0*M_PI*fi/fN);
            break;

        case WindowType::BLACKMAN:
            w = 0.42 - 0.5*std::cos(2.0*M_PI*fi/fN) + 0.08*std::cos(4.0*M_PI*fi/fN);
            break;

        case WindowType::BLACKMAN_HARRIS_4TERM_92DB:
            w = 0.35875 - 0.48829*std::cos(2.0*M_PI*fi/fN)
                        + 0.14128*std::cos(4.0*M_PI*fi/fN)
                        - 0.01168*std::cos(6.0*M_PI*fi/fN);
            break;

        case WindowType::CONNES: {
            double k = (fi - fNh) / fNh;
            k = 1.0 - k*k;
            w = k*k;
            break;
        }

        case WindowType::FLATTOP:
            w = 0.21557895
              - 0.41663158*std::cos(2.0*M_PI*fi/fN)
              + 0.277263158*std::cos(4.0*M_PI*fi/fN)
              - 0.083578947*std::cos(6.0*M_PI*fi/fN)
              + 0.006947368*std::cos(8.0*M_PI*fi/fN);
            break;

        case WindowType::GAUSS_025: {
            double k = (fi - fNh) / (0.25 * fNh);
            w = std::exp(-0.5*k*k);
            break;
        }
        case WindowType::GAUSS_0125: {
            double k = (fi - fNh) / (0.125 * fNh);
            w = std::exp(-0.5*k*k);
            break;
        }

        case WindowType::HAMMING:
            w = 0.54 - 0.46*std::cos(2.0*M_PI*fi/fN);
            break;

        case WindowType::HANN:
            w = 0.5 - 0.5*std::cos(2.0*M_PI*fi/fN);
            break;

        case WindowType::KAISER_BESSEL:
            w = 0.402 - 0.498*std::cos(2.0*M_PI*fi/fN)
                      + 0.098*std::cos(4.0*M_PI*fi/fN)
                      - 0.001*std::cos(6.0*M_PI*fi/fN);
            break;

        case WindowType::NUTTALL:
            w = 0.3635819 - 0.4891775*std::cos(2.0*M_PI*fi/fN)
                          + 0.1365995*std::cos(4.0*M_PI*fi/fN)
                          - 0.0106411*std::cos(6.0*M_PI*fi/fN);
            break;

        case WindowType::TRIANGLE:
            if (N & 1) {
                w = (fi < (double)((N+1)/2))
                    ? (2.0*(fi+1.0) / ((double)N+1.0))
                    : (2.0*((double)N-fi) / ((double)N+1.0));
            } else {
                w = (fi < (double)(N/2))
                    ? (2.0*(fi+1.0) / ((double)N+1.0))
                    : (2.0*((double)N-fi) / ((double)N+1.0));
            }
            break;

        case WindowType::WELCH: {
            double k = (fi - fNh) / fNh;
            w = 1.0 - k*k;
            break;
        }

        // Tukey variants: taper the first and last p/2 of the window with a Hann ramp
        case WindowType::TUKEY_005: case WindowType::TUKEY_010:
        case WindowType::TUKEY_020: case WindowType::TUKEY_050:
        case WindowType::TUKEY_075: case WindowType::TUKEY_090: {
            double p;
            switch (wt) {
                case WindowType::TUKEY_005: p = 0.05; break;
                case WindowType::TUKEY_010: p = 0.10; break;
                case WindowType::TUKEY_020: p = 0.20; break;
                case WindowType::TUKEY_050: p = 0.50; break;
                case WindowType::TUKEY_075: p = 0.75; break;
                default:                    p = 0.90; break;
            }
            int Np = (int)(p / 2.0 * N) - 1;
            if (Np > 0 && (fi <= Np || fi >= N-Np-1)) {
                double idx = (fi <= Np) ? fi : (double)(N-1) - fi;
                w = 0.5 - 0.5*std::cos(M_PI * idx / Np);
            } else {
                w = 1.0;
            }
            break;
        }

        // Partial Tukey: split into 2 sub-windows, each covering part of the block
        case WindowType::PARTIAL_TUKEY_2_000:
        case WindowType::PARTIAL_TUKEY_2_033:
        case WindowType::PARTIAL_TUKEY_2_067: {
            double start = (wt == WindowType::PARTIAL_TUKEY_2_000) ? 0.0
                         : (wt == WindowType::PARTIAL_TUKEY_2_033) ? 0.33 : 0.67;
            double end   = start + 0.5;
            end   = std::min(end, 1.0);
            int sn = (int)(start * N), en = (int)(end * N);
            int Ng = en - sn;
            double p = 0.5;
            int Np = (int)(p / 2.0 * Ng);
            if ((int)i < sn || (int)i >= en) { w = 0.0; }
            else {
                int li = (int)i - sn;
                if (Np > 0 && (li < Np || li >= Ng-Np)) {
                    double idx = (li < Np) ? (double)li : (double)(Ng-1-li);
                    w = 0.5 - 0.5*std::cos(M_PI * idx / Np);
                } else { w = 1.0; }
            }
            break;
        }

        case WindowType::PUNCHOUT_TUKEY_2_033:
        case WindowType::PUNCHOUT_TUKEY_2_067: {
            // Punchout Tukey: full window with a "hole" punched out
            double start = (wt == WindowType::PUNCHOUT_TUKEY_2_033) ? 0.33 : 0.67;
            double end   = start + 0.33;
            int sn = (int)(start * N), en = (int)(end * N);
            double p = 0.5;
            int Ns = (int)(p / 2.0 * sn);
            int Ne = (int)(p / 2.0 * ((int)N - en));
            if ((int)i < Ns) {
                w = 0.5 - 0.5*std::cos(M_PI*(fi+1.0)/(Ns > 0 ? Ns : 1));
            } else if ((int)i < sn-Ns) {
                w = 1.0;
            } else if ((int)i < sn) {
                w = 0.5 - 0.5*std::cos(M_PI*((double)(sn-(int)i))/(Ns > 0 ? Ns : 1));
            } else if ((int)i < en) {
                w = 0.0;
            } else if ((int)i < en+Ne) {
                w = 0.5 - 0.5*std::cos(M_PI*(fi-(double)en)/(Ne > 0 ? Ne : 1));
            } else if ((int)i < (int)N-Ne) {
                w = 1.0;
            } else {
                w = 0.5 - 0.5*std::cos(M_PI*((double)N-fi)/(Ne > 0 ? Ne : 1));
            }
            break;
        }

        default:
            w = 1.0; break;
        }

        out[i] = x * w;
    }
}

// ============================================================
// Levinson-Durbin — all orders in one pass
// ============================================================

void Optimizer::compute_lpc_all_orders(
    const double* autoc, float out_coeffs[][32], int max_order, double* out_err)
{
    // Negative marks "recursion never got here"; callers must skip those.
    if (out_err) for (int i = 0; i <= max_order; ++i) out_err[i] = -1.0;
    // ld_a[i][j] = coefficient j for predictor of order i (1-indexed in both).
    // We use a stack-allocated array — 33×33 doubles = ~8 KB, safe.
    double ld_a[33][33];
    double ld_e[33];
    std::memset(ld_a, 0, sizeof(ld_a));

    ld_e[0] = autoc[0];
    if (out_err) out_err[0] = ld_e[0];
    if (ld_e[0] <= 0.0) {
        for (int i = 0; i < max_order; ++i)
            for (int j = 0; j < 32; ++j) out_coeffs[i][j] = 0.0f;
        return;
    }

    for (int ord = 1; ord <= max_order; ++ord) {
        double lambda = autoc[ord];
        for (int j = 1; j < ord; ++j)
            lambda -= ld_a[ord-1][j] * autoc[ord-j];

        ld_a[ord][ord] = lambda / ld_e[ord-1];

        // If PARCOR magnitude >= 1 the filter is unstable; zero out remaining orders.
        if (std::abs(ld_a[ord][ord]) >= 1.0) {
            for (int i = ord-1; i < max_order; ++i)
                for (int j = 0; j < 32; ++j) out_coeffs[i][j] = 0.0f;
            return;
        }

        ld_e[ord] = ld_e[ord-1] * (1.0 - ld_a[ord][ord]*ld_a[ord][ord]);
        if (ld_e[ord] <= 0.0) ld_e[ord] = 1e-10;
        if (out_err) out_err[ord] = ld_e[ord];

        for (int j = 1; j < ord; ++j)
            ld_a[ord][j] = ld_a[ord-1][j] - ld_a[ord][ord] * ld_a[ord-1][ord-j];

        // Save coefficients for this order
        for (int j = 0; j < ord; ++j)
            out_coeffs[ord-1][j] = (float)ld_a[ord][j+1];
    }
}

// Legacy single-order wrapper (used by the DP fast path).
void Optimizer::compute_lpc_coefficients(const double* autoc, float* out, int order) {
    float tmp[32][32];
    compute_lpc_all_orders(autoc, tmp, order);
    for (int i = 0; i < order; ++i) out[i] = tmp[order-1][i];
}

// ============================================================
// Rice cost
// ============================================================

// Four-lane unsigned vector for the Rice sum accumulators. GCC/clang lower the
// generic vector extension to whatever the target has (NEON usra, SSE2
// psrld+paddd); MSVC has no equivalent, so it takes the scalar array instead.
// Both spellings compute identical sums — this only changes instruction
// selection, never results.
#if defined(__clang__) || defined(__GNUC__)
#  define FLACOUT_HAVE_VECEXT 1
typedef uint32_t u32x4 __attribute__((vector_size(16)));
#else
#  define FLACOUT_HAVE_VECEXT 0
#endif

// Fold a signed residual into the unsigned value Rice coding actually emits.
// Done through uint32_t because left-shifting a negative int32_t is UB.
static inline uint32_t zigzag(int32_t r) {
    return ((uint32_t)r << 1) ^ (uint32_t)(r >> 31);
}

// Quantized-LPC residuals for one candidate, over already-wasted-bit-shifted
// samples.
//
// The taps are on the outside and the sample index on the inside. The textbook
// order — accumulate one `pred` per sample by looping over j — walks `shifted`
// backwards with a runtime trip count into a serial int64 accumulator, which
// does not vectorize. Swapping the loops makes both streams unit-stride with no
// loop-carried dependency, so each tap becomes a widening multiply-accumulate
// across several samples at once.
//
// Integer addition is associative, so reordering the accumulation is exact:
// every partial product is the same and so is the sum. Tiled so the int64
// partials stay in L1 instead of streaming a bsize-sized array `ord` times.
//
// `pred` is caller-supplied scratch of at least min(tile, bsize) elements; it is
// passed in rather than allocated here because this runs millions of times per
// subframe.
static void compute_lpc_residuals(
    const int32_t* shifted, uint32_t bsize,
    const int32_t* qc, int ord, int shift,
    int64_t* pred, uint32_t tile, int32_t* residuals)
{
    assert(ord >= 1); // tap 0 initialises each tile; there is no zeroing pass
    for (uint32_t t = (uint32_t)ord; t < bsize; t += tile) {
        const uint32_t n = std::min(tile, bsize - t);
        // Tap 0 initialises the tile rather than zeroing it first. Zeroing was
        // its own pass over an 8-byte-per-sample buffer, which for low orders
        // cost as much as the accumulation it was preparing for.
        {
            const int64_t  c   = qc[0];
            const int32_t* src = shifted + t - 1;
            for (uint32_t x = 0; x < n; ++x)
                pred[x] = c * (int64_t)src[x];
        }
        for (int j = 1; j < ord; ++j) {
            const int64_t  c   = qc[j];
            const int32_t* src = shifted + t - 1 - j;
            for (uint32_t x = 0; x < n; ++x)
                pred[x] += c * (int64_t)src[x];
        }
        for (uint32_t x = 0; x < n; ++x)
            residuals[t + x] = shifted[t + x] - (int32_t)(pred[x] >> shift);
    }
    // Warm-up samples are stored verbatim.
    for (int i = 0; i < ord; ++i) residuals[i] = shifted[i];
}

// sum(u_i >> k) is additive over disjoint ranges, so sums are computed once at the finest partition order and folded upward instead of rescanning per order.
uint32_t Optimizer::calculate_rice_cost(
    const int32_t* residuals, uint32_t block_size,
    uint32_t order, SubframeParams* out_params)
{
    static constexpr int NUM_K = 15;
    static constexpr int MAX_PARTS = 256; // 1 << 8

    // Valid partition orders form a contiguous prefix [0, max_p_order]
    // since 2^p | block_size implies 2^(p-1) | block_size. Also cap so the
    // finest partition is never smaller than order: per the FLAC format,
    // only partition 0 may have fewer than p_size residuals (its count is
    // p_size - order); every other partition is always exactly p_size long.
    // A decoder never re-derives "warm-up spilled into partition 1" — that's
    // purely a bitstream desync if we choose a partition order that fine.
    int max_p_order = 0;
    for (int p = 1; p <= 8; ++p) {
        if (block_size % (1u << p) != 0) break;
        if ((block_size >> p) < order) break;
        max_p_order = p;
    }

    uint32_t num_parts = 1u << max_p_order;
    uint32_t p_size    = block_size / num_parts;

    uint64_t sums[MAX_PARTS][NUM_K];
    uint32_t n_res[MAX_PARTS];
    uint32_t max_abs_arr[MAX_PARTS];

    INSTR(g_instr.rice_calls.fetch_add(1, std::memory_order_relaxed));
    INSTR(g_instr.rice_scan_samples.fetch_add(block_size - order, std::memory_order_relaxed));
    INSTR(g_instr.rice_k_ops.fetch_add((uint64_t)(block_size - order) * NUM_K, std::memory_order_relaxed));
    INSTR(g_instr.rice_fold_ops.fetch_add((uint64_t)(num_parts - 1) * NUM_K, std::memory_order_relaxed));

    for (uint32_t p = 0; p < num_parts; ++p) {
        uint32_t start = p * p_size;
        uint32_t end   = start + p_size;
        uint32_t first = std::max(start, order); // skip warm-up in partition 0

        uint64_t* s = sums[p];
        for (int k = 0; k < NUM_K; ++k) s[k] = 0;

        // One pass computing both the magnitude bound and the 15 partial sums.
        //
        // The bound is an OR of every folded residual, not a maximum. Only the
        // position of its highest set bit is ever used — `< (1u << 30)` and the
        // escape-code bit-width search below both depend on nothing else — and
        // OR has the same highest set bit as max, because every u contributes
        // its bits. OR is also what makes the overflow test below cheap: since
        // each u's bits are a subset of the OR, u <= or_all, so `count * or_all`
        // bounds the accumulated sum. Folded u is 2*|r| for r>=0 and 2*|r|-1
        // for r<0, so the |r| the escape path wants is u >> 1.
        //
        // Accumulating in 32-bit lanes lets each (s[k] += u >> k) become a
        // single shift-right-and-accumulate over four residuals at a time,
        // rather than 15 dependent 64-bit adds per residual. 32-bit lanes hold
        // at most 0xFFFFFFFF, so the range is walked in fixed chunks and widened
        // into the 64-bit sums between them. The chunk bound cannot be computed
        // up front without a separate maximum pass, so instead each chunk is
        // accumulated optimistically and re-run in 64-bit on the rare occasion
        // the OR proves it could have wrapped.
        static constexpr uint32_t CHUNK = 1024;
        uint32_t or_all = 0;
        for (uint32_t base = first; base < end; ) {
            const uint32_t stop = std::min(base + CHUNK, end);
            const uint32_t cnt  = stop - base;
            uint32_t or_chunk = 0;
            uint32_t part[NUM_K];
            uint32_t i = base;
#if FLACOUT_HAVE_VECEXT
            u32x4 acc[NUM_K] = {};
            u32x4 orv = {};
            for (; i + 4 <= stop; i += 4) {
                const u32x4 u = { zigzag(residuals[i]),     zigzag(residuals[i + 1]),
                                  zigzag(residuals[i + 2]), zigzag(residuals[i + 3]) };
                orv |= u;
                // Must be fully unrolled: `k` has to be a constant for each
                // shift to fold into the accumulate.
#  if defined(__clang__)
#    pragma unroll
#  elif defined(__GNUC__)
#    pragma GCC unroll 15
#  endif
                for (int k = 0; k < NUM_K; ++k) acc[k] += u >> k;
            }
            uint32_t tail[NUM_K] = {};
            for (; i < stop; ++i) {
                const uint32_t u = zigzag(residuals[i]);
                or_chunk |= u;
                for (int k = 0; k < NUM_K; ++k) tail[k] += u >> k;
            }
            or_chunk |= orv[0] | orv[1] | orv[2] | orv[3];
            for (int k = 0; k < NUM_K; ++k)
                part[k] = acc[k][0] + acc[k][1] + acc[k][2] + acc[k][3] + tail[k];
#else
            for (int k = 0; k < NUM_K; ++k) part[k] = 0;
            for (; i < stop; ++i) {
                const uint32_t u = zigzag(residuals[i]);
                or_chunk |= u;
                for (int k = 0; k < NUM_K; ++k) part[k] += u >> k;
            }
#endif
            // part[] is only trustworthy if no lane could have wrapped. part[0]
            // is the largest of the 15, and is bounded by cnt * or_chunk.
            if ((uint64_t)cnt * or_chunk <= 0xFFFFFFFFull) {
                INSTR(g_instr.rice_chunk_fast.fetch_add(1, std::memory_order_relaxed));
                for (int k = 0; k < NUM_K; ++k) s[k] += part[k];
            } else {
                INSTR(g_instr.rice_chunk_slow.fetch_add(1, std::memory_order_relaxed));
                for (uint32_t j = base; j < stop; ++j) {
                    const uint32_t u = zigzag(residuals[j]);
                    for (int k = 0; k < NUM_K; ++k) s[k] += (u >> k);
                }
            }
            or_all |= or_chunk;
            base = stop;
        }
        const uint32_t mabs = or_all >> 1;

        n_res[p]        = (first < end) ? (end - first) : 0;
        max_abs_arr[p]  = mabs;
    }

    uint64_t best_total = std::numeric_limits<uint64_t>::max();
    int      best_porder = 0;
    // Only tracked when the caller wants parameters back. During the candidate
    // search it does not, and skipping this drops a 1 KB zero-init plus a memcpy
    // per improving partition order from a call made millions of times.
    int      best_ks[MAX_PARTS];
    if (out_params) std::memset(best_ks, 0, sizeof(best_ks));

    uint32_t cur_num_parts = num_parts;
    for (int p_order = max_p_order; p_order >= 0; --p_order) {
        uint64_t total = 4 * cur_num_parts; // 4 bits rice-param per partition (method 0)
        int      ks[MAX_PARTS];

        for (uint32_t p = 0; p < cur_num_parts; ++p) {
            uint32_t   n = n_res[p];
            uint64_t*  s = sums[p];

            uint64_t best_k_bits = std::numeric_limits<uint64_t>::max();
            int      best_k = 0;

            // --- Try Rice parameters k = 0..14 ---
            for (int k = 0; k < NUM_K; ++k) {
                uint64_t bits = (uint64_t)n * (1 + k) + s[k];
                if (bits < best_k_bits) { best_k_bits = bits; best_k = k; }
            }

            // --- Try Rice escape code (k=15): verbatim residuals ---
            // k=15 means: 4-bit marker + 5-bit bps + bps bits per residual.
            // The bps field is 5 bits, so residuals wider than 31 bits cannot be
            // represented by escape at all; skip it so normal Rice (which has no
            // such limit) is chosen instead.
            if (max_abs_arr[p] < (1u << 30)) {
                uint32_t max_abs = max_abs_arr[p];
                int escape_bps = 1;
                while (escape_bps < 31 && (1u << (escape_bps - 1)) <= max_abs) ++escape_bps;

                uint64_t escape_bits = 5ull + (uint64_t)escape_bps * n;
                if (escape_bits < best_k_bits) {
                    best_k_bits = escape_bits;
                    best_k = 15 + (escape_bps << 8); // encode bps in high bits for later
                }
            }

            total  += best_k_bits;
            ks[p]   = best_k;
        }

        // '<=' since we iterate p_order descending: keeps the smallest p_order on a tie, same as before
        if (total <= best_total) {
            best_total  = total;
            best_porder = p_order;
            if (out_params) std::memcpy(best_ks, ks, cur_num_parts * sizeof(int));
        }

        if (p_order == 0) break;

        // Fold pairs of partitions up to the next coarser order.
        uint32_t next_num_parts = cur_num_parts / 2;
        for (uint32_t p = 0; p < next_num_parts; ++p) {
            uint32_t left = 2 * p, right = 2 * p + 1;
            n_res[p]       = n_res[left] + n_res[right];
            max_abs_arr[p] = std::max(max_abs_arr[left], max_abs_arr[right]);
            for (int k = 0; k < NUM_K; ++k)
                sums[p][k] = sums[left][k] + sums[right][k];
        }
        cur_num_parts = next_num_parts;
    }

    if (out_params) {
        out_params->rice_partition_order = best_porder;
        std::memcpy(out_params->rice_k, best_ks, (1u << best_porder) * sizeof(int));
    }
    return best_total > std::numeric_limits<uint32_t>::max()
         ? std::numeric_limits<uint32_t>::max()
         : (uint32_t)best_total;
}


// ============================================================
// Subframe cost (fast path — single window, used by old DP shim)
// ============================================================

uint32_t Optimizer::estimate_subframe_cost(
    const int32_t* samples, uint32_t bsize,
    int mode, int order, int precision, int wasted, int bps,
    SubframeParams* out)
{
    if (out) {
        out->mode = mode; out->order = order;
        out->lpc_precision = precision; out->wasted_bits = wasted;
        out->lpc_shift = (precision > 0) ? precision - 1 : 0;
    }

    uint32_t header = 8u + (wasted ? (uint32_t)(1 + wasted) : 0u);

    if (mode == 0) {
        for (uint32_t i = 1; i < bsize; ++i)
            if (samples[i] != samples[0]) return std::numeric_limits<uint32_t>::max();
        return header + (uint32_t)(bps - wasted);
    }
    if (mode == 1) return header + (uint32_t)(bps - wasted) * bsize;

    std::vector<int32_t> residuals(bsize);

    if (mode == 2) {
        for (uint32_t i = 0; i < bsize; ++i) {
            int32_t s  = samples[i]     >> wasted;
            int32_t s1 = (i>0) ? (samples[i-1] >> wasted) : 0;
            int32_t s2 = (i>1) ? (samples[i-2] >> wasted) : 0;
            int32_t s3 = (i>2) ? (samples[i-3] >> wasted) : 0;
            int32_t s4 = (i>3) ? (samples[i-4] >> wasted) : 0;
            if ((uint32_t)i < (uint32_t)order) { residuals[i] = s; continue; }
            switch (order) {
                case 0: residuals[i]=s; break;
                case 1: residuals[i]=s-s1; break;
                case 2: residuals[i]=s-2*s1+s2; break;
                case 3: residuals[i]=s-3*s1+3*s2-s3; break;
                case 4: residuals[i]=s-4*s1+6*s2-4*s3+s4; break;
            }
        }
    } else {
        // LPC (rectangular window — fast path only)
        std::vector<double> f(bsize);
        for (uint32_t i = 0; i < bsize; ++i) f[i] = (double)(samples[i] >> wasted);
        double autoc[33] = {};
        for (int lag = 0; lag <= order; ++lag)
            for (uint32_t j = 0; j < bsize - (uint32_t)lag; ++j)
                autoc[lag] += f[j] * f[j+lag];

        float lpc[32];
        compute_lpc_coefficients(autoc, lpc, order);

        // Compute shift from max coefficient magnitude (same formula as optimize_subframe)
        double cmax = 1e-10;
        for (int j = 0; j < order; ++j)
            cmax = std::max(cmax, std::abs((double)lpc[j]));
        int log2cmax = (cmax >= 1.0) ? (int)std::floor(std::log2(cmax)) : 0;
        int shift = precision - log2cmax - 2;
        if (shift < 0)  shift = 0;
        if (shift > 14) shift = 14;

        int32_t qc[32];
        int32_t maxq = (1 << (precision-1)) - 1;
        int32_t minq = -(1 << (precision-1));
        for (int j = 0; j < order; ++j) {
            double v = lpc[j] * (double)(1 << shift);
            int32_t qi = (int32_t)std::round(v);
            if (qi > maxq) qi = maxq;
            if (qi < minq) qi = minq;
            qc[j] = qi;
        }
        if (out) {
            out->lpc_shift = shift;
            std::memcpy(out->q_coeffs, qc, order * sizeof(int32_t));
        }

        for (uint32_t i = 0; i < bsize; ++i) {
            int32_t s = samples[i] >> wasted;
            if ((uint32_t)i < (uint32_t)order) { residuals[i] = s; continue; }
            int64_t pred = 0;
            for (int j = 0; j < order; ++j)
                pred += (int64_t)qc[j] * (int64_t)(samples[i-1-j] >> wasted);
            residuals[i] = s - (int32_t)(pred >> shift);
        }
        header += 4u + 5u + (uint32_t)(order * precision);

    }

    header += (uint32_t)order * (uint32_t)(bps - wasted); // warm-up samples
    // +6: residual block header (2-bit coding method + 4-bit partition order),
    // written once per subframe by write_residual but not part of the
    // per-partition cost calculate_rice_cost returns.
    return header + 6u + calculate_rice_cost(residuals.data(), bsize, order, out);
}

// ============================================================
// Exhaustive multi-window subframe optimisation
// ============================================================

SubframeParams Optimizer::optimize_subframe(
    const int32_t* samples, uint32_t bsize, uint32_t bps,
    const std::vector<WindowType>& windows, bool exhaustive,
    unsigned max_candidates)
{
    SubframeParams best{};
    best.bits_cost = std::numeric_limits<uint32_t>::max();
#ifdef FLACOUT_INSTRUMENT
    int instr_best_win = -1;
    struct InstrOnExit {
        const SubframeParams& b; const int& w;
        ~InstrOnExit() {
            if (b.mode == 3) {
                g_instr.best_order_hist[b.order].fetch_add(1, std::memory_order_relaxed);
                g_instr.best_prec_hist[b.lpc_precision].fetch_add(1, std::memory_order_relaxed);
                if (w >= 0) g_instr.best_window_hist[w].fetch_add(1, std::memory_order_relaxed);
            }
        }
    } instr_on_exit{best, instr_best_win};
#endif

    // ---- Wasted-bits detection ----
    int wasted = 0;
    int32_t mask = 0;
    for (uint32_t i = 0; i < bsize; ++i) mask |= samples[i];
    if (mask != 0)
        while ((mask & 1) == 0) { mask >>= 1; ++wasted; }
    const uint32_t eff_bps = bps - (uint32_t)wasted;

    auto try_update = [&](SubframeParams& cur, uint32_t cost) {
        if (cost < best.bits_cost) { best = cur; best.bits_cost = cost; }
    };

    // ---- Mode 0: Constant ----
    {
        SubframeParams cur{};
        uint32_t cost = estimate_subframe_cost(samples, bsize, 0, 0, 0, wasted, (int)bps, &cur);
        try_update(cur, cost);
    }

    // ---- Mode 1: Verbatim ----
    {
        SubframeParams cur{};
        uint32_t cost = estimate_subframe_cost(samples, bsize, 1, 0, 0, wasted, (int)bps, &cur);
        try_update(cur, cost);
    }

    // ---- Mode 2: Fixed (orders 0–4) ----
    for (int ord = 0; ord <= 4; ++ord) {
        if ((uint32_t)ord >= bsize) break;
        SubframeParams cur{};
        uint32_t cost = estimate_subframe_cost(samples, bsize, 2, ord, 0, wasted, (int)bps, &cur);
        try_update(cur, cost);
    }

    // ---- Mode 3: LPC with multi-window exhaustive search ----
    if (bsize >= 2) {
        std::vector<double> windowed(bsize);
        std::vector<int32_t> residuals(bsize);
        float all_lpc[32][32]; // all_lpc[order-1][coeff]

        // Scratch for the tiled prediction accumulator (see the residual loop
        // below). Sized so one tile of int64 partials stays L1-resident.
        static constexpr uint32_t PRED_TILE = 1024;
        std::vector<int64_t> pred(std::min(PRED_TILE, bsize));

        // hoisted: samples[i] >> wasted was re-read for every window/order/precision combo below
        std::vector<int32_t> shifted(bsize);
        for (uint32_t i = 0; i < bsize; ++i) shifted[i] = samples[i] >> wasted;

        // precision set is constant for the whole subframe; build once
        // Ranked search evaluates only a handful of candidates, so it can
        // afford the full precision sweep on each of them.
        std::vector<int> precisions;
        if (exhaustive || max_candidates > 0) { for (int p = 8; p <= 15; ++p) precisions.push_back(p); }
        else                                  { precisions = {12, 15}; }
        const uint32_t min_prec  = (uint32_t)precisions.front();
        const uint32_t hdr_fixed = 8u + (wasted ? (uint32_t)(1 + wasted) : 0u) + 4u + 5u;

        // Winner tracked as a bare description rather than a filled-in
        // SubframeParams; see the cost-only call below. Seeded from the best
        // non-LPC mode so it doubles as the pruning bound, exactly as
        // best.bits_cost did — it starts at the same value and decreases at the
        // same points, so pruning decisions are unchanged.
        uint32_t best_lpc_cost = best.bits_cost;
        int      bl_ord = 0, bl_prec = 0, bl_shift = 0;
        int32_t  bl_qc[32] = {};

        INSTR(g_instr.subframes.fetch_add(1, std::memory_order_relaxed));

        const int max_order = (int)std::min((uint32_t)32, bsize - 1);

        // Evaluate one candidate — a coefficient set at one order — across
        // every precision, updating the winner. Shared by both drivers below so
        // ranked and exhaustive search cost a candidate identically; they differ
        // only in which candidates they hand it.
        auto eval_candidate = [&](const float* lpc, int ord, WindowType wt) {
            (void)wt;
            INSTR(g_instr.order_iters.fetch_add(1, std::memory_order_relaxed));
            INSTR(g_instr.win_order_hist[ord].fetch_add(1, std::memory_order_relaxed));

            // Determine quantization shift from the magnitude of the largest coefficient.
            // This matches the libFLAC approach: shift = precision - floor(log2(max_coeff)) - 2
            // so that quantized(max_coeff) < 2^(precision-1) without clipping.
            double cmax = 1e-10;
            for (int j = 0; j < ord; ++j)
                cmax = std::max(cmax, std::abs((double)lpc[j]));

            // floor(log2(cmax)) gives the position of the highest set bit.
            int log2cmax = 0;
            if (cmax >= 1.0)
                log2cmax = (int)std::floor(std::log2(cmax));
            // else cmax < 1 -> log2cmax <= -1, but we keep it at 0 so shift stays high.

#ifdef FLACOUT_INSTRUMENT
            int prec_idx = 0;
#endif
            for (int prec : precisions) {
                INSTR(++prec_idx);
                // fixed cost is a lower bound on this candidate (rice >= 0);
                // grows with precision, so break once it can't beat best.
                uint32_t hdr = hdr_fixed + (uint32_t)ord * (eff_bps + (uint32_t)prec);
                if (hdr >= best_lpc_cost) {
                    INSTR(g_instr.prec_pruned_break.fetch_add(precisions.size() - prec_idx + 1, std::memory_order_relaxed));
                    break;
                }
                INSTR(g_instr.prec_iters.fetch_add(1, std::memory_order_relaxed));

                // shift = (prec-1) - log2cmax - 1
                // = prec - log2cmax - 2
                // Ensures quantized coefficient = round(coeff * 2^shift) fits in [-(2^(prec-1)), 2^(prec-1)-1].
                int shift = prec - log2cmax - 2;
                if (shift < 0)  shift = 0;
                if (shift > 14) shift = 14; // 5-bit signed: max positive value = 15, but 14 is safe

                int32_t maxq = (1 << (prec-1)) - 1;
                int32_t minq = -(1 << (prec-1));

                // Quantize
                int32_t qc[32];
                bool overflow = false;
                for (int j = 0; j < ord; ++j) {
                    double  v  = (double)lpc[j] * (double)(1 << shift);
                    int32_t qi = (int32_t)std::round(v);
                    if (qi > maxq || qi < minq) { overflow = true; break; }
                    qc[j] = qi;
                }
                if (overflow) { INSTR(g_instr.overflow_skips.fetch_add(1, std::memory_order_relaxed)); continue; }

                INSTR(g_instr.residual_calls.fetch_add(1, std::memory_order_relaxed));
                INSTR(g_instr.residual_macs.fetch_add((uint64_t)(bsize - ord) * ord, std::memory_order_relaxed));
                compute_lpc_residuals(shifted.data(), bsize, qc, ord, shift,
                                      pred.data(), PRED_TILE, residuals.data());

                // Cost only — no out-params. Filling in a SubframeParams
                // here would zero and populate ~1.3 KB (rice_k[256] alone is
                // 1 KB) for every candidate, and all but one of them is
                // discarded. Only the winner's identity is kept; its
                // parameters are reconstructed once, after the search.
                uint32_t rice = calculate_rice_cost(residuals.data(), bsize,
                                                    (uint32_t)ord, nullptr);
                // +6: residual block header (2-bit coding method + 4-bit
                // partition order), see estimate_subframe_cost for detail.
                const uint32_t cost = hdr + 6u + rice;
                if (cost < best_lpc_cost) {
                    best_lpc_cost = cost;
                    bl_ord = ord; bl_prec = prec; bl_shift = shift;
                    std::memcpy(bl_qc, qc, (size_t)ord * sizeof(int32_t));
                    INSTR(instr_best_win = (int)wt);
                }
            }
        };

        // Autocorrelation + Levinson-Durbin for one window. Returns false if the
        // windowed signal has no energy and the window should be skipped.
        auto analyse_window = [&](WindowType wt, double* out_err) -> bool {
            INSTR(g_instr.windows_run.fetch_add(1, std::memory_order_relaxed));
            INSTR(g_instr.window_samples.fetch_add(bsize, std::memory_order_relaxed));
            apply_window(samples, bsize, wasted, wt, windowed.data());

            // Autocorrelation for all lags 0..min(32, bsize-1)
            double autoc[33] = {};
            int max_lag = (int)std::min((uint32_t)32, bsize - 1);
            for (int lag = 0; lag <= max_lag; ++lag) {
                INSTR(g_instr.autoc_macs.fetch_add(bsize - (uint32_t)lag, std::memory_order_relaxed));
                for (uint32_t j = 0; j < bsize - (uint32_t)lag; ++j)
                    autoc[lag] += windowed[j] * windowed[j + lag];
            }
            if (autoc[0] <= 0.0) return false;

            std::memset(all_lpc, 0, sizeof(all_lpc));
            compute_lpc_all_orders(autoc, all_lpc, max_order, out_err);
            return true;
        };

        if (max_candidates == 0) {
            // ---- Exhaustive: every (window, order) pair ----
            for (WindowType wt : windows) {
                if (!analyse_window(wt, nullptr)) continue;
                for (int ord = 1; ord <= max_order; ++ord) {
                    // sound lower bound: fixed cost (header + warm-up + coeffs) alone,
                    // at the cheapest precision, since rice cost >= 0. it grows with
                    // order, so once it can't beat best neither can any higher order.
                    uint32_t hdr_min = hdr_fixed + (uint32_t)ord * (eff_bps + min_prec);
                    if (hdr_min >= best_lpc_cost) {
                        INSTR(g_instr.order_pruned_break.fetch_add(max_order - ord + 1, std::memory_order_relaxed));
                        break;
                    }
                    eval_candidate(all_lpc[ord - 1], ord, wt);
                }
            }
        } else {
            // ---- Ranked: score every (window, order) pair, then fully
            //      evaluate only the most promising `max_candidates` of them.
            //
            // Levinson-Durbin already produces the residual energy at each
            // order as a by-product; the exhaustive path just discards it.
            // err[ord]/err[0] is the fraction of signal energy the predictor
            // fails to remove, which turns into an estimated residual bit cost
            // via the Gaussian entropy 0.5*log2(2*pi*e*var) that
            // estimate_lpc_bits_fast already uses. The constant term and the
            // signal's own variance are identical for every candidate, so they
            // drop out of the comparison and only the log of the ratio remains.
            //
            // Using the *ratio* rather than the raw energy is what makes scores
            // comparable across windows at all: each window scales the signal
            // differently, so raw err[ord] values are in different units, while
            // the fraction removed is dimensionless.
            struct Cand { double score; int wi; int ord; };
            std::vector<Cand> cands;
            cands.reserve(windows.size() * (size_t)max_order);

            // all_lpc for every window, so the winning candidates do not have to
            // re-run apply_window + autocorrelation to get their coefficients
            // back. 4 KB per window.
            constexpr size_t LPC_STRIDE = 32 * 32;
            std::vector<float> lpc_store(windows.size() * LPC_STRIDE, 0.0f);

            for (size_t wi = 0; wi < windows.size(); ++wi) {
                double lderr[33];
                if (!analyse_window(windows[wi], lderr)) continue;
                std::memcpy(&lpc_store[wi * LPC_STRIDE], all_lpc, sizeof(all_lpc));
                if (lderr[0] <= 0.0) continue;
                for (int ord = 1; ord <= max_order; ++ord) {
                    if (lderr[ord] <= 0.0) continue; // recursion stopped short of this order
                    const double ratio = lderr[ord] / lderr[0];
                    if (!(ratio > 0.0)) continue;
                    const double resid_bits = 0.5 * std::log2(ratio) * (double)(bsize - ord);
                    const double coef_bits  = (double)ord * (double)(eff_bps + min_prec);
                    cands.push_back({ resid_bits + coef_bits, (int)wi, ord });
                }
            }

            const size_t keep = std::min((size_t)max_candidates, cands.size());
            std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(),
                              [](const Cand& a, const Cand& b) { return a.score < b.score; });

            // Best-first, so the exact cost of a strong candidate tightens the
            // pruning bound before the weaker ones are tried.
            for (size_t c = 0; c < keep; ++c) {
                const Cand& cd = cands[c];
                uint32_t hdr_min = hdr_fixed + (uint32_t)cd.ord * (eff_bps + min_prec);
                if (hdr_min >= best_lpc_cost) continue;
                eval_candidate(&lpc_store[cd.wi * LPC_STRIDE + (size_t)(cd.ord - 1) * 32],
                               cd.ord, windows[cd.wi]);
            }
        }

        // Materialize the winning LPC candidate, if it beat the non-LPC modes.
        // Re-deriving its residuals costs one more pass out of the millions the
        // search just ran, and in exchange every candidate above skipped the
        // per-candidate parameter bookkeeping.
        if (bl_ord > 0) {
            compute_lpc_residuals(shifted.data(), bsize, bl_qc, bl_ord, bl_shift,
                                  pred.data(), PRED_TILE, residuals.data());
            SubframeParams cur{};
            cur.mode          = 3;
            cur.order         = bl_ord;
            cur.lpc_precision = bl_prec;
            cur.lpc_shift     = bl_shift;
            cur.wasted_bits   = wasted;
            std::memcpy(cur.q_coeffs, bl_qc, (size_t)bl_ord * sizeof(int32_t));
            uint32_t rice = calculate_rice_cost(residuals.data(), bsize,
                                                (uint32_t)bl_ord, &cur);
            const uint32_t hdr = hdr_fixed + (uint32_t)bl_ord * (eff_bps + (uint32_t)bl_prec);
            assert(hdr + 6u + rice == best_lpc_cost);
            try_update(cur, hdr + 6u + rice);
        }
    }

    return best;
}

// ============================================================
// DP fast-path: granule precomputation + cost estimation
// ============================================================

void Optimizer::precompute_granules(
    const std::vector<std::vector<int32_t>>& pcm_data)
{
    size_t num_g = pcm_data[0].size() / 16;
    m_granules.assign(m_channels, std::vector<Granule>(num_g));

    for (uint32_t c = 0; c < m_channels; ++c)
        for (size_t g = 0; g < num_g; ++g) {
            const int32_t* src = &pcm_data[c][g * 16];
            for (int i = 0; i <= 32; ++i) {
                double s = 0;
                for (int j = 0; j < 16 - i; ++j) s += (double)src[j] * src[j+i];
                m_granules[c][g].autoc[i] = s;
            }
        }
}

uint32_t Optimizer::estimate_lpc_bits_fast(
    int channel, uint32_t n_start, uint32_t n_end, int bps) const
{
    double autoc[33] = {};
    for (uint32_t g = n_start; g < n_end; ++g)
        for (int i = 0; i <= 32; ++i)
            autoc[i] += m_granules[channel][g].autoc[i];

    float coeffs[32];
    compute_lpc_coefficients(autoc, coeffs, 8);

    double err = autoc[0];
    for (int i = 0; i < 8; ++i) err -= (double)coeffs[i] * autoc[i+1];

    uint32_t bsize = (n_end - n_start) * 16;
    if (err <= 0) return 16 + (uint32_t)bps;

    double bps_est = 0.5 * std::log2(2.0 * M_PI * M_E * (err / bsize));
    if (bps_est < 1.0) bps_est = 1.0;
    return 16u + (uint32_t)(bsize * bps_est);
}

// ============================================================
// Main entry point
// ============================================================
//
// Strategy:
//   1. Try a small set of candidate block sizes (all powers-of-two and common FLAC sizes).
//   2. For each candidate, estimate the total encoded bits using the fast granule-based LPC model.
//   3. Pick the winning block size.
//   4. Run exhaustive multi-window, multi-mode subframe optimization in parallel on each block.
//
// This avoids the O(n × MAX_GRANULES) DP that becomes prohibitive for long files,
// while still exploring the full block-size search space at a coarse level.

std::vector<BlockParams> Optimizer::find_optimal_block_partitioning(
    const std::vector<std::vector<int32_t>>& pcm_data)
{
    precompute_granules(pcm_data);

    const size_t total_samples  = pcm_data[0].size();

    // -----------------------------------------------------------------
    // Short-stream fast path
    // -----------------------------------------------------------------
    // If the audio is shorter than the smallest candidate block size
    // (1024 samples), skip block-size selection entirely and emit one
    // block covering all samples (FLAC allows block sizes from 1 to 65535).
    if (total_samples < 1024) {
        if (m_verbose)
            std::cout << "Short stream (" << total_samples << " samples): "
                      << "using single block.\n";
        BlockParams bp{};
        bp.block_size = (uint32_t)total_samples;
        if (m_channels == 1) {
            bp.stereo_mode  = 0;
            bp.subframes[0] = optimize_subframe(pcm_data[0].data(),
                                                (uint32_t)total_samples, m_bps, m_windows, m_exhaustive, m_max_candidates);
        } else {
            uint32_t best_bits = std::numeric_limits<uint32_t>::max();
            for (int mode : {0, 8, 9, 10}) {
                std::vector<int32_t> ch0(total_samples), ch1(total_samples);
                for (size_t k = 0; k < total_samples; ++k) {
                    int32_t L = pcm_data[0][k], R = pcm_data[1][k];
                    if (mode == 0)  { ch0[k] = L;        ch1[k] = R; }
                    else if (mode == 8)  { ch0[k] = L;        ch1[k] = L - R; }
                    else if (mode == 9)  { ch0[k] = L - R;    ch1[k] = R; }
                    else             { ch0[k] = (L+R)>>1; ch1[k] = L - R; }
                }
                // mode 9 = right+side: ch0 is side (needs +1 bit), ch1 is right
                uint32_t bps0 = (mode == 9) ? m_bps + 1 : m_bps;
                uint32_t bps1 = (mode == 9) ? m_bps     : (mode == 0 ? m_bps : m_bps + 1);
                SubframeParams s0 = optimize_subframe(ch0.data(), (uint32_t)total_samples, bps0, m_windows, m_exhaustive, m_max_candidates);
                SubframeParams s1 = optimize_subframe(ch1.data(), (uint32_t)total_samples, bps1, m_windows, m_exhaustive, m_max_candidates);
                if (s0.bits_cost + s1.bits_cost < best_bits) {
                    best_bits = s0.bits_cost + s1.bits_cost;
                    bp.stereo_mode  = mode;
                    bp.subframes[0] = s0;
                    bp.subframes[1] = s1;
                }
            }
        }
        bp.total_bits = bp.subframes[0].bits_cost
                      + (m_channels > 1 ? bp.subframes[1].bits_cost : 0);
        return { bp };
    }


    // -----------------------------------------------------------------
    // Variable-block-size DP
    // -----------------------------------------------------------------
    //
    // Candidates: {4096, 8192, 16384} with STEP = GCD = 4096.
    // Node i = position i*STEP in the audio.
    // Edge (i→j) = one FLAC frame covering [i*STEP, j*STEP).
    // Cost = FRAME_OVERHEAD + exact bits from optimize_subframe (no estimates).
    //
    // Phase 1: build all N×K work items, evaluate ALL in parallel with a
    //          flat thread pool.  All compute_block() calls are independent
    //          (different sample offsets, no shared writes), so 100% of
    //          threads stay busy for the full Phase 1 duration.
    // Phase 2: sequential DP over precomputed cost table — O(N×K), instant.
    // Phase 3: back-trace to recover the optimal frame sequence.

    static const uint32_t CANDIDATES[]   = { 1024, 2048, 4096, 8192, 16384 };
    static const size_t   NUM_CANDS      = std::size(CANDIDATES);
    static constexpr uint32_t STEP           = 1024u; // GCD of all candidates
    static constexpr uint32_t FRAME_OVERHEAD = 200u;  // bits per FLAC frame header

    const size_t   num_nodes = total_samples / STEP;
    const uint32_t remainder = (uint32_t)(total_samples % STEP);

    // Phase 1: parallel precomputation -----------------------------------
    struct WorkItem { size_t node; size_t ci; };
    std::vector<WorkItem> work;
    work.reserve(num_nodes * NUM_CANDS);
    for (size_t n = 0; n < num_nodes; ++n)
        for (size_t c = 0; c < NUM_CANDS; ++c)
            if ((uint64_t)n * STEP + CANDIDATES[c] <= total_samples)
                work.push_back({n, c});

    // Longest-processing-time-first. Workers pull from one shared counter, so
    // whatever order this vector is in is the order work is handed out. Built
    // node-major, the 16384-sample blocks — 16x the cost of a 1024 — are spread
    // evenly through the queue, and some land near the end where there is no
    // remaining work to overlap them with. Handing out the expensive ones first
    // leaves only cheap blocks to fill the tail. Purely a scheduling change:
    // cost_table is indexed by (node, candidate), not by completion order.
    std::stable_sort(work.begin(), work.end(),
                     [](const WorkItem& a, const WorkItem& b) {
                         return CANDIDATES[a.ci] > CANDIDATES[b.ci];
                     });

    std::vector<BlockParams> cost_table(num_nodes * NUM_CANDS);

    unsigned nthreads = std::max(1u, static_cast<unsigned>(std::thread::hardware_concurrency()));
    if (m_max_threads > 0) nthreads = std::min(nthreads, (unsigned)m_max_threads);

    if (m_verbose)
        std::cout << "DP: " << num_nodes << " nodes × " << NUM_CANDS
                  << " candidates = " << work.size() << " blocks on "
                  << nthreads << " threads\n";

    {
        std::atomic<size_t> next{0}, done{0};
        std::mutex           cout_mtx;
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
            threads.emplace_back([&]() {
                for (;;) {
                    size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= work.size()) break;
                    auto [node, ci] = work[idx];
                    if (full_search()) {
                        cost_table[node * NUM_CANDS + ci] =
                            compute_block(pcm_data, (uint64_t)node * STEP, CANDIDATES[ci]);
                    } else {
                        uint32_t bits = 0;
                        // granules are 16 samples each; nodes are STEP=1024 samples each
                        static constexpr uint32_t GRANULE_SIZE = 16u;
                        uint32_t g_start = (uint32_t)(node * (STEP / GRANULE_SIZE));
                        uint32_t g_end   = g_start + CANDIDATES[ci] / GRANULE_SIZE;
                        if (m_channels == 1) {
                            bits = estimate_lpc_bits_fast(0, g_start, g_end, m_bps);
                        } else {
                            bits = estimate_lpc_bits_fast(0, g_start, g_end, m_bps) +
                                   estimate_lpc_bits_fast(1, g_start, g_end, m_bps);
                        }
                        BlockParams bp{};
                        bp.block_size = CANDIDATES[ci];
                        bp.total_bits = bits;
                        cost_table[node * NUM_CANDS + ci] = bp;
                    }
                    size_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (m_verbose && (d % 10 == 0 || d == work.size())) {
                        std::lock_guard<std::mutex> lk(cout_mtx);
                        std::cout << "\r  " << d << "/" << work.size()
                                  << " (" << d * 100 / work.size() << "%)" << std::flush;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        if (m_verbose) std::cout << "\n";
    }

    // Remainder block
    BlockParams remainder_bp{};
    if (remainder > 0)
        remainder_bp = compute_block(pcm_data, (uint64_t)num_nodes * STEP, remainder);

    // Phase 2: DP --------------------------------------------------------
    std::vector<uint64_t> dp       (num_nodes + 1, std::numeric_limits<uint64_t>::max());
    std::vector<int>      dp_parent(num_nodes + 1, -1);
    std::vector<int>      dp_cand  (num_nodes + 1, -1);
    dp[0] = 0;

    for (size_t i = 0; i < num_nodes; ++i) {
        if (dp[i] == std::numeric_limits<uint64_t>::max()) continue;
        for (size_t c = 0; c < NUM_CANDS; ++c) {
            size_t j = i + CANDIDATES[c] / STEP;
            if (j > num_nodes) continue;
            if ((uint64_t)i * STEP + CANDIDATES[c] > total_samples) continue;
            uint64_t cost = dp[i] + FRAME_OVERHEAD
                          + cost_table[i * NUM_CANDS + c].total_bits;
            if (cost < dp[j]) {
                dp[j]        = cost;
                dp_parent[j] = (int)i;
                dp_cand  [j] = (int)c;
            }
        }
    }

    // Phase 3: back-trace ------------------------------------------------
    std::vector<std::pair<size_t,size_t>> path;
    for (size_t cur = num_nodes; cur > 0; ) {
        int par = dp_parent[cur], cand = dp_cand[cur];
        if (par < 0) break;
        path.emplace_back((size_t)par, (size_t)cand);
        cur = (size_t)par;
    }
    std::reverse(path.begin(), path.end());

    std::vector<BlockParams> result(path.size() + (remainder > 0 ? 1 : 0));
    if (!full_search()) {
        std::atomic<size_t> next{0}, done{0};
        std::mutex           cout_mtx;
        std::vector<std::thread> threads;
        if (m_verbose) std::cout << "Optimizing " << path.size() << " selected blocks...\n";
        for (int t = 0; t < nthreads; ++t) {
            threads.emplace_back([&]() {
                for (;;) {
                    size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= path.size()) break;
                    auto [node, ci] = path[idx];
                    result[idx] = compute_block(pcm_data, (uint64_t)node * STEP, CANDIDATES[ci]);
                    size_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (m_verbose && (d % 10 == 0 || d == path.size())) {
                        std::lock_guard<std::mutex> lk(cout_mtx);
                        std::cout << "\r  " << d << "/" << path.size() << std::flush;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        if (m_verbose) std::cout << "\n";
    } else {
        for (size_t idx = 0; idx < path.size(); ++idx) {
            auto [node, ci] = path[idx];
            result[idx] = cost_table[node * NUM_CANDS + ci];
        }
    }

    if (remainder > 0)
        result.back() = remainder_bp;

    if (m_verbose) {
        std::map<uint32_t,int> bs_hist;
        for (const auto& bp : result) ++bs_hist[bp.block_size];
        std::cout << "DP done. Distribution:";
        for (const auto& [bs, cnt] : bs_hist)
            std::cout << "  bs=" << bs << "×" << cnt;
        std::cout << "\n";
    }

    return result;
}



// ============================================================
// compute_block: fully-optimised BlockParams for one frame
// ============================================================

BlockParams Optimizer::compute_block(
    const std::vector<std::vector<int32_t>>& pcm_data,
    uint64_t sample_start, uint32_t block_size) const
{
    BlockParams bp{};
    bp.block_size = block_size;

    if (m_channels == 1) {
        bp.stereo_mode  = 0;
        bp.subframes[0] = optimize_subframe(
            &pcm_data[0][sample_start], block_size, m_bps, m_windows, m_exhaustive, m_max_candidates);
    } else {
        uint32_t best_bits = std::numeric_limits<uint32_t>::max();

        // Independent stereo (mode 0)
        std::vector<int> modes_to_test = {0, 8, 9, 10};
        if (!full_search()) {
            // Fast estimation using Fixed Predictor (order 2)
            auto estimate_ch = [&](const int32_t* smp, int bits_per_sample) -> uint32_t {
                int wasted = 0;
                int32_t mask = 0;
                for (uint32_t i = 0; i < block_size; ++i) mask |= smp[i];
                if (mask != 0) while ((mask & 1) == 0) { mask >>= 1; ++wasted; }
                return estimate_subframe_cost(smp, block_size, 2, 2, 0, wasted, bits_per_sample);
            };
            
            uint32_t best_est = std::numeric_limits<uint32_t>::max();
            int best_mode = 0;
            for (int mode : modes_to_test) {
                uint32_t cost = 0;
                if (mode == 0) {
                    cost = estimate_ch(&pcm_data[0][sample_start], m_bps) +
                           estimate_ch(&pcm_data[1][sample_start], m_bps);
                } else {
                    std::vector<int32_t> ch0(block_size), ch1(block_size);
                    for (uint32_t k = 0; k < block_size; ++k) {
                        int32_t L = pcm_data[0][sample_start + k], R = pcm_data[1][sample_start + k];
                        if      (mode == 8)  { ch0[k] = L;        ch1[k] = L - R; }
                        else if (mode == 9)  { ch0[k] = L - R;    ch1[k] = R;     }
                        else                 { ch0[k] = (L+R)>>1; ch1[k] = L - R; }
                    }
                    cost = estimate_ch(ch0.data(), m_bps) + estimate_ch(ch1.data(), m_bps + 1);
                }
                if (cost < best_est) {
                    best_est = cost;
                    best_mode = mode;
                }
            }
            modes_to_test = {best_mode};
        }

        // the 4 stereo modes draw from only 4 distinct channel signals; S=L-R
        // alone appears in modes 8/9/10, so optimize each signal at most once.
        enum { SIG_L = 0, SIG_R, SIG_S, SIG_M };
        bool           have[4] = { false, false, false, false };
        SubframeParams cache[4];

        auto get_sig = [&](int sig) -> const SubframeParams& {
            if (have[sig]) return cache[sig];
            if (sig == SIG_L) {
                cache[sig] = optimize_subframe(&pcm_data[0][sample_start], block_size, m_bps, m_windows, m_exhaustive, m_max_candidates);
            } else if (sig == SIG_R) {
                cache[sig] = optimize_subframe(&pcm_data[1][sample_start], block_size, m_bps, m_windows, m_exhaustive, m_max_candidates);
            } else {
                std::vector<int32_t> ch(block_size);
                uint32_t bps_s;
                if (sig == SIG_S) {
                    for (uint32_t k = 0; k < block_size; ++k)
                        ch[k] = pcm_data[0][sample_start + k] - pcm_data[1][sample_start + k];
                    bps_s = m_bps + 1;
                } else { // SIG_M
                    for (uint32_t k = 0; k < block_size; ++k)
                        ch[k] = (pcm_data[0][sample_start + k] + pcm_data[1][sample_start + k]) >> 1;
                    bps_s = m_bps;
                }
                cache[sig] = optimize_subframe(ch.data(), block_size, bps_s, m_windows, m_exhaustive, m_max_candidates);
            }
            have[sig] = true;
            return cache[sig];
        };

        // (ch0, ch1) signal pair per stereo mode
        auto mode_sigs = [](int mode) -> std::pair<int,int> {
            switch (mode) {
                case 0:  return { SIG_L, SIG_R };
                case 8:  return { SIG_L, SIG_S };
                case 9:  return { SIG_S, SIG_R };
                default: return { SIG_M, SIG_S }; // mode 10
            }
        };

        for (int mode : modes_to_test) {
            auto [sig0, sig1] = mode_sigs(mode);
            const SubframeParams& s0 = get_sig(sig0);
            const SubframeParams& s1 = get_sig(sig1);
            uint32_t cost = s0.bits_cost + s1.bits_cost;
            if (cost < best_bits) {
                best_bits       = cost;
                bp.stereo_mode  = mode;
                bp.subframes[0] = s0;
                bp.subframes[1] = s1;
            }
        }
    }

    bp.total_bits = bp.subframes[0].bits_cost
                  + (m_channels > 1 ? bp.subframes[1].bits_cost : 0);
    return bp;
}
