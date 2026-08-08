#include "optimizer.hpp"
#include "frame_writer.hpp"
#include <algorithm>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif
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
    std::atomic<uint64_t> overflow_skips{0};    // candidates with >=1 clamped tap (kept, not skipped)
    std::atomic<uint64_t> residual_calls{0};    // residual loops actually run
    std::atomic<uint64_t> residual_delta{0};    // of which: precision-ladder delta updates
    std::atomic<uint64_t> residual_macs{0};     // int64 multiply-accumulates (full computes only)
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
    std::atomic<uint64_t> best_window_hist[(size_t)WindowType::COUNT]{};
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
        std::fprintf(stderr, "overflow clamps         : %llu\n", (unsigned long long)overflow_skips);
        std::fprintf(stderr, "residual loops run      : %llu   (delta updates: %llu)\n",
                     (unsigned long long)residual_calls, (unsigned long long)residual_delta);
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
        for (int i = 0; i < (int)WindowType::COUNT; ++i)
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

std::vector<WindowType> all_window_types(bool include_experimental) {
    std::vector<WindowType> out;
    const int end = include_experimental ? (int)WindowType::COUNT
                                         : (int)WindowType::EXPERIMENTAL_BEGIN;
    for (int i = 0; i < end; ++i)
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
        case WindowType::LANCZOS:                  return "lanczos";
        case WindowType::BOHMAN:                   return "bohman";
        case WindowType::PARZEN:                   return "parzen";
        case WindowType::PLANCKTAPER_010:          return "plancktaper010";
        case WindowType::PLANCKTAPER_025:          return "plancktaper025";
        case WindowType::PARTIAL_TUKEY_3_1:        return "partialtukey3_1";
        case WindowType::PARTIAL_TUKEY_3_2:        return "partialtukey3_2";
        case WindowType::PARTIAL_TUKEY_3_3:        return "partialtukey3_3";
        case WindowType::PUNCHOUT_TUKEY_3_1:       return "punchouttukey3_1";
        case WindowType::PUNCHOUT_TUKEY_3_2:       return "punchouttukey3_2";
        case WindowType::PUNCHOUT_TUKEY_3_3:       return "punchouttukey3_3";
        case WindowType::PARTIAL_TUKEY_3H_000:     return "partialtukey3h_000";
        case WindowType::PARTIAL_TUKEY_3H_033:     return "partialtukey3h_033";
        case WindowType::PARTIAL_TUKEY_3H_067:     return "partialtukey3h_067";
        case WindowType::PUNCHOUT_TUKEY_3H_025:    return "punchouttukey3h_025";
        case WindowType::PUNCHOUT_TUKEY_3H_050:    return "punchouttukey3h_050";
        case WindowType::EXPDECAY_2:               return "expdecay2";
        case WindowType::EXPDECAY_4:               return "expdecay4";
        case WindowType::EXPATTACK_2:              return "expattack2";
        case WindowType::EXPATTACK_4:              return "expattack4";
        case WindowType::ATTACKDECAY_005:          return "attackdecay005";
        case WindowType::ATTACKDECAY_010:          return "attackdecay010";
        case WindowType::ATTACKDECAY_020:          return "attackdecay020";
        case WindowType::DPSS_2:                   return "dpss2";
        case WindowType::DPSS_3:                   return "dpss3";
        case WindowType::DPSS_4:                   return "dpss4";
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
    if (clean == "lanczos")                               return WindowType::LANCZOS;
    if (clean == "bohman")                                return WindowType::BOHMAN;
    if (clean == "parzen")                                return WindowType::PARZEN;
    if (clean == "plancktaper010")                        return WindowType::PLANCKTAPER_010;
    if (clean == "plancktaper025")                        return WindowType::PLANCKTAPER_025;
    if (clean == "partialtukey31")                        return WindowType::PARTIAL_TUKEY_3_1;
    if (clean == "partialtukey32")                        return WindowType::PARTIAL_TUKEY_3_2;
    if (clean == "partialtukey33")                        return WindowType::PARTIAL_TUKEY_3_3;
    if (clean == "punchouttukey31")                       return WindowType::PUNCHOUT_TUKEY_3_1;
    if (clean == "punchouttukey32")                       return WindowType::PUNCHOUT_TUKEY_3_2;
    if (clean == "punchouttukey33")                       return WindowType::PUNCHOUT_TUKEY_3_3;
    if (clean == "partialtukey3h000")                     return WindowType::PARTIAL_TUKEY_3H_000;
    if (clean == "partialtukey3h033")                     return WindowType::PARTIAL_TUKEY_3H_033;
    if (clean == "partialtukey3h067")                     return WindowType::PARTIAL_TUKEY_3H_067;
    if (clean == "punchouttukey3h025")                    return WindowType::PUNCHOUT_TUKEY_3H_025;
    if (clean == "punchouttukey3h050")                    return WindowType::PUNCHOUT_TUKEY_3H_050;
    if (clean == "expdecay2")                             return WindowType::EXPDECAY_2;
    if (clean == "expdecay4")                             return WindowType::EXPDECAY_4;
    if (clean == "expattack2")                            return WindowType::EXPATTACK_2;
    if (clean == "expattack4")                            return WindowType::EXPATTACK_4;
    if (clean == "attackdecay005")                        return WindowType::ATTACKDECAY_005;
    if (clean == "attackdecay010")                        return WindowType::ATTACKDECAY_010;
    if (clean == "attackdecay020")                        return WindowType::ATTACKDECAY_020;
    if (clean == "dpss2")                                 return WindowType::DPSS_2;
    if (clean == "dpss3")                                 return WindowType::DPSS_3;
    if (clean == "dpss4")                                 return WindowType::DPSS_4;
    return WindowType::COUNT; // unrecognised
}

// ============================================================
// Optimizer constructor
// ============================================================

Optimizer::Optimizer(uint32_t channels, uint32_t bps, uint32_t sample_rate,
                     std::vector<WindowType> windows,
                     unsigned max_threads,
                     bool exhaustive,
                     bool verbose,
                     unsigned max_candidates,
                     bool adaptive_windows)
    : m_channels(channels), m_bps(bps), m_sample_rate(sample_rate),
      m_max_threads(max_threads),
      m_exhaustive(exhaustive), m_verbose(verbose), m_max_candidates(max_candidates),
      m_adaptive(adaptive_windows)
{
    if (windows.empty()) {
        // Exact-DP mode (-e) affords the widest window set; the ranking pays
        // per candidate evaluated, not per window offered, so offering more
        // windows there only adds options. The heuristic default keeps the
        // short list because its analysis cost (windowing + autocorrelation
        // per window) is paid on every block it touches.
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
// All window formulas up to EXPERIMENTAL_BEGIN match libFLAC's window.c
// exactly; experimental windows past it are our own additions.

// Note for future profilers: a 16-thread `sample` of -c once attributed ~24%
// of runtime to the trig here; a single-threaded profile put it at ~2%. The
// 24% was heterogeneous-core sampling distortion — verify shares
// single-threaded before optimizing this function. The table cache below is
// NOT here to skip the trig (that saved nothing measurable); it exists so
// apply_window stays a tiny multiply loop at its call site, which keeps the
// register allocator honest in the surrounding analyse_window code — the
// wide-band autocorrelation loop spills without it.

// DPSS (Slepian) window: the eigenvector for the largest eigenvalue of the
// symmetric tridiagonal Slepian matrix (diag ((N-1-2i)/2)^2 cos(2piW),
// off-diag i(N-i)/2, W = NW/N). Everything is fixed-count for determinism:
// 100 Sturm-bisection steps pin the eigenvalue (interval shrinks by 2^100,
// far below double resolution), then 4 inverse-iteration solves recover the
// eigenvector — no tolerance loops, straight double arithmetic, so it is on
// the same footing as the closed-form windows under -ffp-contract=off. Cost
// is ~100 O(N) passes at table-build time; the on-the-fly path (non-DP
// sizes, short streams) pays it per block, which is rare and still cheap
// next to encoding the block.
static void compute_dpss(uint32_t N, double NW, double* out)
{
    const double W = NW / (double)N;
    const double c = std::cos(2.0 * M_PI * W);
    std::vector<double> d(N), e(N); // e[i] couples rows i-1 and i; e[0] unused
    for (uint32_t i = 0; i < N; ++i) {
        double k = ((double)(N - 1) - 2.0 * (double)i) / 2.0;
        d[i] = k * k * c;
        e[i] = (double)i * (double)(N - i) / 2.0;
    }

    // Gershgorin interval containing every eigenvalue.
    double lo = d[0], hi = d[0], scale = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        double r = std::abs(e[i]) + (i + 1 < N ? std::abs(e[i + 1]) : 0.0);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
        scale = std::max(scale, std::abs(d[i]) + r);
    }

    // Sturm-sequence bisection down to the largest eigenvalue: count_below(s)
    // is the number of eigenvalues < s via the LDL^T pivot signs.
    auto count_below = [&](double s) {
        uint32_t cnt = 0;
        double q = 1.0;
        for (uint32_t i = 0; i < N; ++i) {
            q = (i == 0) ? d[0] - s : d[i] - s - e[i] * e[i] / q;
            if (q == 0.0) q = -1e-300; // pivot guard; count a zero pivot as below
            if (q < 0.0) ++cnt;
        }
        return cnt;
    };
    for (int it = 0; it < 100; ++it) {
        double mid = 0.5 * (lo + hi);
        if (count_below(mid) == N) hi = mid; else lo = mid;
    }
    const double lam = hi; // λmax to machine precision

    // Inverse iteration: repeatedly solve (T - λI) x = v (Thomas algorithm
    // with a relative pivot floor — the matrix is singular by construction,
    // which is what makes the iteration converge) and renormalize.
    const double piv_floor = 1e-14 * scale;
    std::vector<double> v(N), m(N), z(N);
    for (uint32_t i = 0; i < N; ++i) // Hann start: right symmetry, no zeros inside
        v[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i / (double)(N - 1));
    for (int it = 0; it < 4; ++it) {
        m[0] = d[0] - lam;
        if (std::abs(m[0]) < piv_floor) m[0] = piv_floor;
        z[0] = v[0];
        for (uint32_t i = 1; i < N; ++i) {
            double f = e[i] / m[i - 1];
            m[i] = d[i] - lam - f * e[i];
            if (std::abs(m[i]) < piv_floor) m[i] = piv_floor;
            z[i] = v[i] - f * z[i - 1];
        }
        v[N - 1] = z[N - 1] / m[N - 1];
        for (uint32_t i = N - 1; i-- > 0;)
            v[i] = (z[i] - e[i + 1] * v[i + 1]) / m[i];
        double mx = 0.0;
        for (uint32_t i = 0; i < N; ++i) mx = std::max(mx, std::abs(v[i]));
        for (uint32_t i = 0; i < N; ++i) v[i] /= mx;
    }

    // Peak-normalize to +1 (eigenvector sign is arbitrary; DPSS is unimodal).
    double peak = 0.0;
    for (uint32_t i = 0; i < N; ++i)
        if (std::abs(v[i]) > std::abs(peak)) peak = v[i];
    for (uint32_t i = 0; i < N; ++i) out[i] = v[i] / peak;
}

static void compute_window_coeffs(WindowType wt, uint32_t N, double* out)
{
    auto fN = (double)(N - 1);
    auto fNh = fN / 2.0;

    if (wt == WindowType::DPSS_2 || wt == WindowType::DPSS_3 ||
        wt == WindowType::DPSS_4) {
        compute_dpss(N, (wt == WindowType::DPSS_2) ? 2.0
                      : (wt == WindowType::DPSS_3) ? 3.0 : 4.0, out);
        return;
    }

    for (uint32_t i = 0; i < N; ++i) {
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

        // Partial Tukey: sub-window covering part of the block. The _2 set
        // spans 0.5; the experimental house _3H set spans 0.33 (-w only).
        case WindowType::PARTIAL_TUKEY_2_000:
        case WindowType::PARTIAL_TUKEY_2_033:
        case WindowType::PARTIAL_TUKEY_2_067:
        case WindowType::PARTIAL_TUKEY_3H_000:
        case WindowType::PARTIAL_TUKEY_3H_033:
        case WindowType::PARTIAL_TUKEY_3H_067: {
            double start = (wt == WindowType::PARTIAL_TUKEY_2_000 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_000) ? 0.0
                         : (wt == WindowType::PARTIAL_TUKEY_2_033 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_033) ? 0.33 : 0.67;
            double span  = (wt == WindowType::PARTIAL_TUKEY_3H_000 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_033 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_067) ? 0.33 : 0.5;
            double end   = start + span;
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
        case WindowType::PUNCHOUT_TUKEY_2_067:
        case WindowType::PUNCHOUT_TUKEY_3H_025:
        case WindowType::PUNCHOUT_TUKEY_3H_050: {
            // Punchout Tukey: full window with a "hole" punched out. The _2
            // set punches 0.33; the experimental house _3H set 0.25 (-w only).
            double start = (wt == WindowType::PUNCHOUT_TUKEY_2_033) ? 0.33
                         : (wt == WindowType::PUNCHOUT_TUKEY_2_067) ? 0.67
                         : (wt == WindowType::PUNCHOUT_TUKEY_3H_025) ? 0.25 : 0.50;
            double hole  = (wt == WindowType::PUNCHOUT_TUKEY_3H_025 ||
                            wt == WindowType::PUNCHOUT_TUKEY_3H_050) ? 0.25 : 0.33;
            double end   = start + hole;
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

        case WindowType::LANCZOS: {
            // sinc(2i/(N-1) - 1); not in libFLAC. Experimental (-w only).
            double x = 2.0*fi/fN - 1.0;
            w = (x == 0.0) ? 1.0 : std::sin(M_PI*x) / (M_PI*x);
            break;
        }

        case WindowType::BOHMAN: {
            // (1-|x|)cos(pi|x|) + sin(pi|x|)/pi, x in [-1,1]. Experimental.
            double x = std::abs(2.0*fi/fN - 1.0);
            w = (1.0 - x)*std::cos(M_PI*x) + std::sin(M_PI*x)/M_PI;
            break;
        }

        case WindowType::PARZEN: {
            // Piecewise cubic B-spline (scipy form), x = |i - (N-1)/2| / (N/2).
            // Experimental.
            double x = std::abs(fi - fNh) / ((double)N / 2.0);
            if (x <= 0.5) w = 1.0 - 6.0*x*x*(1.0 - x);
            else { double k = 1.0 - x; w = 2.0*k*k*k; }
            break;
        }

        case WindowType::PLANCKTAPER_010:
        case WindowType::PLANCKTAPER_025: {
            // Flat center, C-infinity edges: w = 1/(e^Z + 1) over the first
            // and last eps*N samples, Z = eps*N*(1/j + 1/(j - eps*N)), with
            // w(0) = w(N-1) = 0. Experimental.
            double eps = (wt == WindowType::PLANCKTAPER_010) ? 0.10 : 0.25;
            double t = eps * (double)N;
            double j = std::min(fi, fN - fi);
            if (j <= 0.0)     w = 0.0;
            else if (j < t) { double Z = t*(1.0/j + 1.0/(j - t));
                              w = 1.0 / (std::exp(Z) + 1.0); }
            else              w = 1.0;
            break;
        }

        // libFLAC-faithful partial_tukey(3): parts at thirds, 10% overlap,
        // taper p = 0.2 — the geometry `flac -A partial_tukey(3)` builds
        // (stream_encoder.c overlap_units math, window.c ramp indexing),
        // re-derived per-index in doubles. Experimental (-w only).
        case WindowType::PARTIAL_TUKEY_3_1:
        case WindowType::PARTIAL_TUKEY_3_2:
        case WindowType::PARTIAL_TUKEY_3_3: {
            double m = (wt == WindowType::PARTIAL_TUKEY_3_1) ? 0.0
                     : (wt == WindowType::PARTIAL_TUKEY_3_2) ? 1.0 : 2.0;
            double u = 1.0/(1.0 - 0.1) - 1.0;              // overlap_units
            int sn = (int)(m / (3.0 + u) * N);
            int en = (int)((m + 1.0 + u) / (3.0 + u) * N);
            int Np = (int)(0.2 / 2.0 * (en - sn));
            int n = (int)i;
            if (n < sn || n >= en)               w = 0.0;
            else if (Np > 0 && n < sn + Np)      w = 0.5 - 0.5*std::cos(M_PI*(double)(n - sn + 1)/Np);
            else if (Np > 0 && n >= en - Np)     w = 0.5 - 0.5*std::cos(M_PI*(double)(en - n)/Np);
            else                                 w = 1.0;
            break;
        }

        // libFLAC-faithful punchout_tukey(3): holes at thirds, 20% overlap,
        // taper p = 0.2. Experimental (-w only).
        case WindowType::PUNCHOUT_TUKEY_3_1:
        case WindowType::PUNCHOUT_TUKEY_3_2:
        case WindowType::PUNCHOUT_TUKEY_3_3: {
            double m = (wt == WindowType::PUNCHOUT_TUKEY_3_1) ? 0.0
                     : (wt == WindowType::PUNCHOUT_TUKEY_3_2) ? 1.0 : 2.0;
            double u = 1.0/(1.0 - 0.2) - 1.0;              // overlap_units
            int sn = (int)(m / (3.0 + u) * N);
            int en = (int)((m + 1.0 + u) / (3.0 + u) * N);
            int Ns = (int)(0.2 / 2.0 * sn);
            int Ne = (int)(0.2 / 2.0 * ((int)N - en));
            int n = (int)i;
            if (n >= sn && n < en)                            w = 0.0;
            else if (Ns > 0 && n < Ns)                        w = 0.5 - 0.5*std::cos(M_PI*(double)(n + 1)/Ns);
            else if (Ns > 0 && n >= sn - Ns && n < sn)        w = 0.5 - 0.5*std::cos(M_PI*(double)(sn - n)/Ns);
            else if (Ne > 0 && n >= en && n < en + Ne)        w = 0.5 - 0.5*std::cos(M_PI*(double)(n - en + 1)/Ne);
            else if (Ne > 0 && n >= (int)N - Ne)              w = 0.5 - 0.5*std::cos(M_PI*(double)((int)N - n)/Ne);
            else                                              w = 1.0;
            break;
        }

        // Asymmetric exponentials: e^(-k·i/(N-1)) and its mirror. Not in
        // libFLAC. Experimental (-w only).
        case WindowType::EXPDECAY_2:  w = std::exp(-2.0*fi/fN);        break;
        case WindowType::EXPDECAY_4:  w = std::exp(-4.0*fi/fN);        break;
        case WindowType::EXPATTACK_2: w = std::exp(-2.0*(fN - fi)/fN); break;
        case WindowType::EXPATTACK_4: w = std::exp(-4.0*(fN - fi)/fN); break;

        // Attack-decay: exponential rise (rate 6, normalized to reach 1) over
        // the first f·N samples, half-cosine decay to 0 after. Experimental
        // (-w only).
        case WindowType::ATTACKDECAY_005:
        case WindowType::ATTACKDECAY_010:
        case WindowType::ATTACKDECAY_020: {
            double f = (wt == WindowType::ATTACKDECAY_005) ? 0.05
                     : (wt == WindowType::ATTACKDECAY_010) ? 0.10 : 0.20;
            double R = f * (double)N;
            if (fi < R) {
                double t = std::min((fi + 1.0) / R, 1.0);
                w = (1.0 - std::exp(-6.0*t)) / (1.0 - std::exp(-6.0));
            } else {
                w = std::cos(0.5*M_PI*(fi - R)/(fN - R));
            }
            break;
        }

        default:
            w = 1.0; break;
        }

        out[i] = w;
    }
}

// The DP's candidate block sizes (shared with find_optimal_block_partitioning).
// Window coefficient tables for these sizes are precomputed once; any other
// size (the remainder block, the short-stream path) computes on the fly.
static const uint32_t DP_CANDIDATES[] = { 1024, 2048, 4096, 8192, 16384 };
static constexpr size_t NUM_DP_CANDIDATES = 5;

static int candidate_slot(uint32_t N)
{
    for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c)
        if (DP_CANDIDATES[c] == N) return (int)c;
    return -1;
}

// All windows (incl. experimental) x 5 candidate sizes, built once on first
// use (~254 KB per window; ~13 MB at 51 windows,
// thread-safe magic static — worker threads block on the first builder and
// read lock-free forever after). The values are computed by the exact code
// that used to run inline per block, so the output is bit-identical.
static const double* window_table(WindowType wt, int slot)
{
    static const std::vector<double> tables = [] {
        size_t total = 0;
        for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) total += DP_CANDIDATES[c];
        std::vector<double> t((size_t)WindowType::COUNT * total);
        size_t off = 0;
        for (int w = 0; w < (int)WindowType::COUNT; ++w)
            for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
                compute_window_coeffs((WindowType)w, DP_CANDIDATES[c], &t[off]);
                off += DP_CANDIDATES[c];
            }
        return t;
    }();

    // offset of (wt, slot): wt strides the whole size set, then sizes before slot
    size_t size_sum = 0, prefix = 0;
    for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) size_sum += DP_CANDIDATES[c];
    for (int c = 0; c < slot; ++c) prefix += DP_CANDIDATES[c];
    return &tables[(size_t)wt * size_sum + prefix];
}

// Sum of squared window coefficients, the normalizer that turns a windowed
// Levinson error into an absolute residual-variance estimate (see the ranked
// scoring in optimize_subframe). Cached for the precomputed table sizes —
// COUNT x 5 doubles, built lazily from the tables themselves — and computed on
// the fly for the rare other sizes (remainder block, short-stream path).
static double window_energy(WindowType wt, uint32_t N)
{
    const int slot = candidate_slot(N);
    if (slot >= 0) {
        static const std::vector<double> energies = [] {
            std::vector<double> e((size_t)WindowType::COUNT * NUM_DP_CANDIDATES);
            for (int w = 0; w < (int)WindowType::COUNT; ++w)
                for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
                    const double* t = window_table((WindowType)w, (int)c);
                    double s = 0.0;
                    for (uint32_t i = 0; i < DP_CANDIDATES[c]; ++i) s += t[i] * t[i];
                    e[(size_t)w * NUM_DP_CANDIDATES + c] = s;
                }
            return e;
        }();
        return energies[(size_t)wt * NUM_DP_CANDIDATES + (size_t)slot];
    }
    std::vector<double> tmp(N);
    compute_window_coeffs(wt, N, tmp.data());
    double s = 0.0;
    for (uint32_t i = 0; i < N; ++i) s += tmp[i] * tmp[i];
    return s;
}

// Fraction of exactly-zero coefficients in a window — the block region the
// window is blind to. Non-zero only for partial/punchout shapes (and the
// single zero endpoint samples of e.g. Planck windows, which round to ~0
// fraction). The ranked scorer uses it to price the blind region at the raw
// signal's variance instead of pretending it is modeled. Cached like
// window_energy for the table sizes.
static double window_zero_frac(WindowType wt, uint32_t N)
{
    const int slot = candidate_slot(N);
    if (slot >= 0) {
        static const std::vector<double> fracs = [] {
            std::vector<double> f((size_t)WindowType::COUNT * NUM_DP_CANDIDATES);
            for (int w = 0; w < (int)WindowType::COUNT; ++w)
                for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
                    const double* t = window_table((WindowType)w, (int)c);
                    uint32_t z = 0;
                    for (uint32_t i = 0; i < DP_CANDIDATES[c]; ++i)
                        if (t[i] == 0.0) ++z;
                    f[(size_t)w * NUM_DP_CANDIDATES + c] =
                        (double)z / (double)DP_CANDIDATES[c];
                }
            return f;
        }();
        return fracs[(size_t)wt * NUM_DP_CANDIDATES + (size_t)slot];
    }
    std::vector<double> tmp(N);
    compute_window_coeffs(wt, N, tmp.data());
    uint32_t z = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (tmp[i] == 0.0) ++z;
    return (double)z / (double)N;
}

void Optimizer::apply_window(
    const int32_t* samples, uint32_t N, int wasted_bits,
    WindowType wt, double* out)
{
    const int slot = candidate_slot(N);
    const double* w;
    std::vector<double> scratch;
    if (slot >= 0) {
        w = window_table(wt, slot);
    } else {
        scratch.resize(N);
        compute_window_coeffs(wt, N, scratch.data());
        w = scratch.data();
    }
    // One multiply per sample, same arithmetic as the fused version, and
    // -ffp-contract=off means nothing can fuse into it: bit-exact.
    for (uint32_t i = 0; i < N; ++i)
        out[i] = (double)(samples[i] >> wasted_bits) * w[i];
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

// Pairwise horizontal add: lane i of the result is the sum of lanes 2i, 2i+1
// of (a ++ b). Both compilers lower this shuffle+add pattern to a single
// pairwise-add instruction where one exists (NEON addp, SSSE3 phaddd).
static inline u32x4 hadd_pairs(u32x4 a, u32x4 b) {
#if defined(__clang__)
    return __builtin_shufflevector(a, b, 0, 2, 4, 6)
         + __builtin_shufflevector(a, b, 1, 3, 5, 7);
#else
    return __builtin_shuffle(a, b, (u32x4){0, 2, 4, 6})
         + __builtin_shuffle(a, b, (u32x4){1, 3, 5, 7});
#endif
}
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
// Register-blocked: a fixed batch of consecutive output samples is held in
// accumulators while the tap loop runs over all `ord` coefficients, so the
// partial sums never reach memory at all.
//
// The obvious alternatives both lose. One `pred` per sample accumulated over j
// walks the history backwards into a serial int64 chain that will not
// vectorize. Hoisting the taps outside and streaming a `pred[]` array
// vectorizes, but then every tap reloads and rewrites that whole array — with
// ord up to 32 the array moves through the load/store units 32 times, and the
// profile showed exactly that: 90 memory ops in the inner loop and not a single
// widening multiply-accumulate.
//
// Blocking the output dimension gets both properties: BLOCK independent
// accumulators give the vectorizer something to work with, and they stay in
// registers across the whole tap loop, so per tap the only memory traffic is
// reading the sample history.
//
// Integer addition is associative, so this is exact — same partial products,
// same sum, and in fact the same order as the original scalar version.
//
// autoc[lag] = sum over j of w[j]*w[j+lag], for lag in [0, max_lag].
//
// Computed a band of lags at a time rather than one lag per pass. The
// straightforward version walks the whole block once per lag — 33 passes over
// `bsize` doubles — and each pass is a reduction, so loads dominate and the
// serial accumulator chain gives the vectorizer nothing to work with. Handling
// LAG_BAND lags together gets LAG_BAND multiply-accumulates out of every loaded
// w[j] and cuts the number of passes by the same factor. At 33 the whole
// max_lag=32 range is one pass; measured in isolation that is 2.0x over the
// original band of 8 (register-pressure sweet spots are not monotonic — 16
// measured no better than 8, so widen all the way or not at all).
//
// This is floating point, so unlike the integer loops the summation order is
// *not* free to change — reassociating would perturb the coefficients and with
// them the output. It is preserved exactly: the parallelism is across lags, each
// keeping its own accumulator summed over ascending j, and each band's ragged
// tail (where the highest lag in the band would run off the end) is finished per
// lag afterwards, which is still ascending j for that lag.
static void autocorrelation(
    const double* w, uint32_t n, int max_lag, double* autoc)
{
    constexpr int LAG_BAND = 33;

    int lag0 = 0;
    for (; lag0 + LAG_BAND <= max_lag + 1; lag0 += LAG_BAND) {
        double a[LAG_BAND] = {};
        // Range of j where every lag in this band is still in bounds.
        const uint32_t jend = n - (uint32_t)(lag0 + LAG_BAND - 1);
        for (uint32_t j = 0; j < jend; ++j) {
            const double x = w[j];
            for (int l = 0; l < LAG_BAND; ++l) a[l] += x * w[j + lag0 + l];
        }
        // Ragged tail: lag0+l runs off the end later for smaller l.
        for (int l = 0; l < LAG_BAND; ++l) {
            const uint32_t lim = n - (uint32_t)(lag0 + l);
            for (uint32_t j = jend; j < lim; ++j) a[l] += w[j] * w[j + lag0 + l];
        }
        for (int l = 0; l < LAG_BAND; ++l) autoc[lag0 + l] = a[l];
    }

    // Leftover lags that do not fill a band.
    for (int lag = lag0; lag <= max_lag; ++lag) {
        double s = 0.0;
        for (uint32_t j = 0; j < n - (uint32_t)lag; ++j) s += w[j] * w[j + lag];
        autoc[lag] = s;
    }
}

static void compute_lpc_residuals(
    const int32_t* shifted, uint32_t bsize,
    const int32_t* qc, int ord, int shift, int32_t* residuals,
    int64_t* pred_out = nullptr)
{
    assert(ord >= 1);

    // 8 int64 accumulators = 4 128-bit registers, which leaves plenty spare for
    // the sample history and coefficients on both NEON (32) and SSE2 (16).
    constexpr uint32_t BLOCK = 16;

    uint32_t i = (uint32_t)ord;
    for (; i + BLOCK <= bsize; i += BLOCK) {
        int64_t acc[BLOCK] = {};
        for (int j = 0; j < ord; ++j) {
            const int64_t  c   = qc[j];
            const int32_t* src = shifted + i - 1 - j;
            // Must be unrolled, or acc[] is indexed dynamically and spills.
#if defined(__clang__)
#  pragma unroll
#elif defined(__GNUC__)
#  pragma GCC unroll 16
#endif
            for (uint32_t k = 0; k < BLOCK; ++k) acc[k] += c * (int64_t)src[k];
        }
        if (pred_out)
            for (uint32_t k = 0; k < BLOCK; ++k) pred_out[i + k] = acc[k];
        for (uint32_t k = 0; k < BLOCK; ++k)
            residuals[i + k] = shifted[i + k] - (int32_t)(acc[k] >> shift);
    }

    // Tail: fewer than BLOCK samples left.
    for (; i < bsize; ++i) {
        int64_t p = 0;
        for (int j = 0; j < ord; ++j)
            p += (int64_t)qc[j] * (int64_t)shifted[i - 1 - j];
        if (pred_out) pred_out[i] = p;
        residuals[i] = shifted[i] - (int32_t)(p >> shift);
    }

    // Warm-up samples are stored verbatim.
    for (int k = 0; k < ord; ++k) residuals[k] = shifted[k];
}

// Precision-ladder delta update. When the ladder steps prec -> prec+1 the
// quantization target doubles exactly (v = lpc[j] * 2^shift, and shift grows
// by one), so the new coefficients are qc'[j] = 2*qc[j] + d[j] where d[j] is
// a small integer near zero (exactly {-1,0,1} under independent rounding;
// error feedback spreads it slightly wider) — many of them zero. The new
// prediction is therefore
//
//     pred'[i] = 2*pred[i] + sum over nonzero d of d[j]*shifted[i-1-j]
//
// which replaces a full ord-tap widening multiply-accumulate with int32
// multiplies over the nonzero taps (twice the SIMD width of the int64 loop):
// every partial sum of the correction is bounded by sum|d| * max|sample|,
// and the caller only takes this path when that is < 2^31. pred itself is
// the exact integer dot product of the new coefficients — no error
// accumulates across steps — so the residuals are bit-identical to a full
// recompute.
static void update_lpc_residuals_delta(
    const int32_t* shifted, uint32_t bsize,
    const int32_t* d, int ord, int shift,
    int64_t* pred, int32_t* residuals)
{
    // Gather the nonzero taps once; the sample loops only touch those.
    int     tap_off [32];
    int32_t tap_sign[32];
    int nnz = 0;
    for (int j = 0; j < ord; ++j)
        if (d[j]) { tap_off[nnz] = j; tap_sign[nnz] = d[j]; ++nnz; }

    // Wider than the int64 loop's 16: int32 accumulators are half the size, so
    // 32 still fits in 8 vector registers, and the extra independent chains
    // cover the multiply-accumulate latency the serial per-lane adds expose.
    constexpr uint32_t BLOCK = 32;

    uint32_t i = (uint32_t)ord;
    for (; i + BLOCK <= bsize; i += BLOCK) {
        int32_t corr[BLOCK] = {};
        for (int t = 0; t < nnz; ++t) {
            const int32_t  c   = tap_sign[t];
            const int32_t* src = shifted + i - 1 - tap_off[t];
#if defined(__clang__)
#  pragma unroll
#elif defined(__GNUC__)
#  pragma GCC unroll 32
#endif
            for (uint32_t k = 0; k < BLOCK; ++k) corr[k] += c * src[k];
        }
        for (uint32_t k = 0; k < BLOCK; ++k) {
            const int64_t p = 2 * pred[i + k] + (int64_t)corr[k];
            pred[i + k] = p;
            residuals[i + k] = shifted[i + k] - (int32_t)(p >> shift);
        }
    }

    for (; i < bsize; ++i) {
        int64_t p = 2 * pred[i];
        for (int t = 0; t < nnz; ++t)
            p += (int64_t)tap_sign[t] * (int64_t)shifted[i - 1 - tap_off[t]];
        pred[i] = p;
        residuals[i] = shifted[i] - (int32_t)(p >> shift);
    }
    // Warm-up residuals are already in place from the full compute.
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
        // The chunked path below accumulates into s[] and needs it zeroed; the
        // single-chunk fast path writes each s[k] exactly once and does not.

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

#if FLACOUT_HAVE_VECEXT
        // Single-chunk fast path. For power-of-two block sizes — every DP
        // candidate — the finest partition is at most 64 residuals, so this is
        // the path every partition of every candidate actually takes; the
        // chunked loop below only ever runs for the odd-sized remainder block.
        // With so few residuals per partition the bookkeeping around the scan
        // costs as much as the scan, so this path strips it down: no zeroed
        // u64 sums to read-modify-write, no part[] staging, and the fifteen
        // per-lane reductions collapse into a pairwise-add tree whose u64
        // widening lands directly in sums[p].
        if (end - first <= CHUNK) {
            const uint32_t cnt = end - first;
            u32x4 acc[16] = {}; // 16th stays zero: pads the reduction tree
            u32x4 orv = {};
            uint32_t i = first;
            for (; i + 4 <= end; i += 4) {
                const u32x4 u = { zigzag(residuals[i]),     zigzag(residuals[i + 1]),
                                  zigzag(residuals[i + 2]), zigzag(residuals[i + 3]) };
                orv |= u;
#  if defined(__clang__)
#    pragma unroll
#  elif defined(__GNUC__)
#    pragma GCC unroll 15
#  endif
                for (int k = 0; k < NUM_K; ++k) acc[k] += u >> k;
            }
            uint32_t tail[16] = {};
            uint32_t or_tail = 0;
            for (; i < end; ++i) {
                const uint32_t u = zigzag(residuals[i]);
                or_tail |= u;
                for (int k = 0; k < NUM_K; ++k) tail[k] += u >> k;
            }
            or_all = or_tail | orv[0] | orv[1] | orv[2] | orv[3];

            if ((uint64_t)cnt * or_all <= 0xFFFFFFFFull) {
                INSTR(g_instr.rice_chunk_fast.fetch_add(1, std::memory_order_relaxed));
                for (int k = 0; k < NUM_K; k += 4) {
                    // r = [sum(acc[k]), sum(acc[k+1]), sum(acc[k+2]), sum(acc[k+3])]
                    const u32x4 r = hadd_pairs(hadd_pairs(acc[k],     acc[k + 1]),
                                               hadd_pairs(acc[k + 2], acc[k + 3]));
                    for (int j = 0; j < 4 && k + j < NUM_K; ++j)
                        s[k + j] = (uint64_t)r[j] + tail[k + j];
                }
            } else {
                INSTR(g_instr.rice_chunk_slow.fetch_add(1, std::memory_order_relaxed));
                for (int k = 0; k < NUM_K; ++k) s[k] = 0;
                for (uint32_t j = first; j < end; ++j) {
                    const uint32_t u = zigzag(residuals[j]);
                    for (int k = 0; k < NUM_K; ++k) s[k] += (u >> k);
                }
            }

            n_res[p]       = cnt;
            max_abs_arr[p] = or_all >> 1;
            continue;
        }
#endif
        for (int k = 0; k < NUM_K; ++k) s[k] = 0;
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
            // bits(k) is exactly convex in k: s[k] = 2*s[k+1] + (count of
            // residuals with bit k set), so the forward difference
            // bits(k+1) - bits(k) = n - s[k+1] - o_k is nondecreasing in k.
            // Scanning ascending, the running best is bits(k-1) until the
            // minimum is passed, so the first k that fails to improve proves
            // every later k is no better — stop there. Strict '<' keeps the
            // smallest k on ties, exactly like the full scan did.
            for (int k = 0; k < NUM_K; ++k) {
                uint64_t bits = (uint64_t)n * (1 + k) + s[k];
                if (bits < best_k_bits) { best_k_bits = bits; best_k = k; }
                else break;
            }

            // --- Try Rice escape code (k=15): verbatim residuals ---
            // k=15 means: 4-bit marker + 5-bit bps + bps bits per residual.
            // The bps field is 5 bits, so residuals wider than 31 bits cannot be
            // represented by escape at all; skip it so normal Rice (which has no
            // such limit) is chosen instead.
            if (max_abs_arr[p] < (1u << 30)) {
                uint32_t max_abs = max_abs_arr[p];
                // Smallest width whose sign bit clears max_abs: 2 + floor(log2)
                // for nonzero values, 1 for zero — same result the old
                // increment loop produced, in one bit-scan.
#if defined(_MSC_VER) && !defined(__clang__)
                unsigned long hi;
                int escape_bps = _BitScanReverse(&hi, max_abs) ? (int)hi + 2 : 1;
#else
                int escape_bps = max_abs ? 33 - __builtin_clz(max_abs) : 1;
#endif

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
// Coefficient quantization
// ============================================================

// Quantize LPC coefficients to `precision` bits (sign included), mirroring
// FLAC__lpc_quantize_coefficients in third_party/libflac/src/libFLAC/lpc.c.
// The two must stay in agreement or we build different predictors than the
// format's reference encoder for the same (window, order, precision).
//
// Error feedback: each tap's rounding shortfall is carried into the next tap,
// so the running sum of quantization errors stays within half an LSB. That
// shapes the coefficient noise high-pass (a (1 - z^-1) character), where audio
// has the least energy, instead of leaving it white — measurably better
// predictors than rounding each tap in isolation.
//
// Out-of-range taps are clamped into [qmin, qmax], not treated as failure: a
// clamped predictor is legal and may still be the best available at this
// order. A negative shift is not representable in the format, so that branch
// scales the coefficients down instead and emits shift = 0.
//
// Returns false only when no usable quantization exists (all-zero
// coefficients, or shift below the 5-bit field's minimum). `*clamped` reports
// whether any tap was clamped, for instrumentation.
// floor(log2(max|lpc[j]|)), or INT32_MIN when the coefficients are all zero.
// Computed via frexp so it may be negative when every coefficient is below 1
// in magnitude — which *raises* the shift and quantizes more finely. Split
// out of quantize_lpc_coeffs because it is precision-invariant: the ladder
// sweeps 8 precisions per candidate and only this part can be hoisted.
static int lpc_log2cmax(const float* lpc, int order)
{
    double cmax = 0.0;
    for (int j = 0; j < order; ++j)
        cmax = std::max(cmax, std::abs((double)lpc[j]));
    if (cmax <= 0.0) return std::numeric_limits<int32_t>::min();
    int e;
    (void)std::frexp(cmax, &e);
    return e - 1;
}

static bool quantize_lpc_coeffs(const float* lpc, int order, int precision,
                                int log2cmax,
                                int32_t* qc, int* out_shift, bool* clamped)
{
    if (log2cmax == std::numeric_limits<int32_t>::min())
        return false; // all-zero coefficients

    const int32_t qmax = (1 << (precision - 1)) - 1;
    const int32_t qmin = -(1 << (precision - 1));

    // 5-bit signed shift field in the subframe header.
    constexpr int max_shiftlimit = 15;
    constexpr int min_shiftlimit = -max_shiftlimit - 1;

    int shift = (precision - 1) - log2cmax - 1;

    if (shift > max_shiftlimit)
        shift = max_shiftlimit;
    else if (shift < min_shiftlimit)
        return false;

    *clamped = false;
    double error = 0.0;
    if (shift >= 0) {
        for (int j = 0; j < order; ++j) {
            error += (double)lpc[j] * (double)(1 << shift);
            int32_t q = (int32_t)std::lround(error);
            if (q > qmax)      { q = qmax; *clamped = true; }
            else if (q < qmin) { q = qmin; *clamped = true; }
            error -= q;
            qc[j] = q;
        }
    } else {
        const int nshift = -shift;
        for (int j = 0; j < order; ++j) {
            error += (double)lpc[j] / (double)(1 << nshift);
            int32_t q = (int32_t)std::lround(error);
            if (q > qmax)      { q = qmax; *clamped = true; }
            else if (q < qmin) { q = qmin; *clamped = true; }
            error -= q;
            qc[j] = q;
        }
        shift = 0;
    }
    *out_shift = shift;
    return true;
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

        int32_t qc[32];
        int  shift   = 0;
        bool clamped = false;
        if (!quantize_lpc_coeffs(lpc, order, precision, lpc_log2cmax(lpc, order),
                                 qc, &shift, &clamped)) {
            // Degenerate coefficients — a zero predictor keeps the old
            // behaviour (residual == signal) without a special-cased return.
            std::memset(qc, 0, (size_t)order * sizeof(int32_t));
            shift = 0;
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
    const std::vector<WindowType>& windows,
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
        std::vector<int64_t> pred(bsize); // unshifted predictions, for the precision-ladder delta
        float all_lpc[32][32]; // all_lpc[order-1][coeff]

        // hoisted: samples[i] >> wasted was re-read for every window/order/precision combo below
        std::vector<int32_t> shifted(bsize);
        for (uint32_t i = 0; i < bsize; ++i) shifted[i] = samples[i] >> wasted;

        // precision set is constant for the whole subframe; build once
        std::vector<int> precisions;
        for (int p = 8; p <= 15; ++p) precisions.push_back(p);
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

            // Precision-invariant: hoisted out of the ladder below.
            const int log2cmax = lpc_log2cmax(lpc, ord);

            // Precision-ladder delta state: prev_qc/prev_shift describe the
            // coefficients pred[] currently holds the dot products of.
            // The int32 correction accumulator needs every partial sum within
            // sum|d| * max|sample| < 2^31, with |sample| <= 2^(eff_bps-1) —
            // checked per step below once sum|d| is known.
            const int64_t max_sum_abs_d =
                (eff_bps <= 31) ? (int64_t)(INT32_MAX >> (eff_bps - 1)) : 0;
            bool       have_pred = false;
            int        prev_shift = 0;
            int32_t    prev_qc[32];

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

                // Quantize with error feedback (mirrors libFLAC; see
                // quantize_lpc_coeffs). Out-of-range taps are clamped, not
                // discarded — a clamped predictor may still win this order.
                int32_t qc[32];
                int     shift   = 0;
                bool    clamped = false;
                if (!quantize_lpc_coeffs(lpc, ord, prec, log2cmax, qc, &shift, &clamped))
                    continue; // degenerate coefficients, no usable quantization
                if (clamped) { INSTR(g_instr.overflow_skips.fetch_add(1, std::memory_order_relaxed)); }

                INSTR(g_instr.residual_calls.fetch_add(1, std::memory_order_relaxed));

                // One ladder step up from the coefficients pred[] was built
                // for: apply the small integer correction d[j] = qc[j] -
                // 2*prev_qc[j] instead of recomputing. With error feedback in
                // the quantizer the per-tap errors can exceed half an LSB, so
                // d[j] lands in a small range around zero rather than strictly
                // {-1,0,1}; the kernel multiplies by d[j] so any magnitude is
                // exact — the only gate is the int32 accumulator bound, which
                // caps sum|d| (a clamped tap can blow d up arbitrarily).
                bool delta_done = false;
                if (have_pred && shift == prev_shift + 1) {
                    int32_t d[32];
                    int64_t sum_abs_d = 0;
                    for (int j = 0; j < ord; ++j) {
                        d[j] = qc[j] - 2 * prev_qc[j];
                        sum_abs_d += std::abs((int64_t)d[j]);
                    }
                    if (sum_abs_d <= max_sum_abs_d) {
                        INSTR(g_instr.residual_delta.fetch_add(1, std::memory_order_relaxed));
                        update_lpc_residuals_delta(shifted.data(), bsize, d, ord,
                                                   shift, pred.data(), residuals.data());
                        delta_done = true;
                    }
                }
                if (!delta_done) {
                    INSTR(g_instr.residual_macs.fetch_add((uint64_t)(bsize - ord) * ord, std::memory_order_relaxed));
                    compute_lpc_residuals(shifted.data(), bsize, qc, ord, shift,
                                          residuals.data(), pred.data());
                }
                std::memcpy(prev_qc, qc, (size_t)ord * sizeof(int32_t));
                prev_shift = shift;
                have_pred  = true;

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
#ifdef FLACOUT_INSTRUMENT
            for (int lag = 0; lag <= max_lag; ++lag)
                g_instr.autoc_macs.fetch_add(bsize - (uint32_t)lag, std::memory_order_relaxed);
#endif
            autocorrelation(windowed.data(), bsize, max_lag, autoc);
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
            // order as a by-product; the exhaustive path just discards it. For
            // a residual that is roughly stationary across the block, that
            // windowed-domain energy is ~ var_e x sum(w^2), so dividing by the
            // window's energy estimates the absolute residual variance, which
            // the Gaussian entropy 0.5*log2(2*pi*e*var) (the same model
            // estimate_lpc_bits_fast uses) turns into estimated residual bits.
            // Absolute variance is what makes scores comparable across windows:
            // each window scales the signal differently, so raw err[ord]
            // values are in different units.
            //
            // Scoring by the err[ord]/err[0] *ratio* instead (energy fraction
            // removed) was tried first and measures ~0.02-0.26% worse across
            // real music and synthetics: it normalizes by the signal power
            // seen *through the window*, which flatters windows aimed at
            // quiet parts of the block. Also tried and rejected: scoring by
            // predicted residual energy on the RAW signal via the quadratic
            // form a'Ra over the unwindowed autocorrelation — the
            // autocorrelation method's zero-extension edge bias exceeds the
            // true residual energy by orders of magnitude on predictable
            // content, which is the very pathology windowing exists to kill.
            //
            // The *peephole* problem needs one more term: partial/punchout
            // windows genuinely predict the slice of block they look at, so
            // their windowed error says nothing about the samples they zero
            // out — pricing those as free floods the top-N with slice-local
            // candidates (measured: all-26 at N=8 lost to tukey050 alone).
            // Fix: charge the zeroed fraction of the block at the raw
            // signal's variance — the model has no information there, so the
            // unpredicted signal is the honest estimate. Dense windows have
            // zero_frac == 0 and score exactly as before.
            struct Cand { double score; int wi; int ord; };
            std::vector<Cand> cands;
            cands.reserve(windows.size() * (size_t)max_order);

            // Entropy terms are clamped at zero: the Gaussian differential
            // entropy goes negative once var < 1/(2πe), but Rice cannot code
            // below ~0 bits/sample, and letting a peephole window's
            // tiny-slice variance contribute large *negative* bits vaulted
            // partial windows over every dense candidate on pure-tone
            // content (+4523 B on the 3-min synthetic tonal fixture).
            double var_raw = 0.0;
            for (uint32_t i = 0; i < bsize; ++i)
                var_raw += (double)shifted[i] * (double)shifted[i];
            var_raw /= (double)bsize;
            const double raw_bits_per_sample = (var_raw > 0.0)
                ? std::max(0.0, 0.5 * std::log2(2.0 * M_PI * M_E * var_raw)) : 0.0;

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
                const double wsq = window_energy(windows[wi], bsize);
                if (!(wsq > 0.0)) continue;
                const double zf = window_zero_frac(windows[wi], bsize);
                for (int ord = 1; ord <= max_order; ++ord) {
                    if (lderr[ord] <= 0.0) continue; // recursion stopped short of this order
                    const double var_e = lderr[ord] / wsq;
                    const double model_bits_per_sample =
                        std::max(0.0, 0.5 * std::log2(2.0 * M_PI * M_E * var_e));
                    const double resid_bits =
                        (model_bits_per_sample * (1.0 - zf)
                         + raw_bits_per_sample * zf) * (double)(bsize - ord);
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
                                  residuals.data());
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
            for (int i = 0; i <= 8; ++i) {
                double s = 0;
                for (int j = 0; j < 16 - i; ++j) s += (double)src[j] * src[j+i];
                m_granules[c][g].autoc[i] = s;
            }
        }
}

uint32_t Optimizer::estimate_lpc_bits_fast(
    int channel, uint32_t n_start, uint32_t n_end, int bps) const
{
    double autoc[9] = {};
    for (uint32_t g = n_start; g < n_end; ++g)
        for (int i = 0; i <= 8; ++i)
            autoc[i] += m_granules[channel][g].autoc[i];

    float coeffs[32];
    compute_lpc_coefficients(autoc, coeffs, 8);

    double err = autoc[0];
    for (int i = 0; i < 8; ++i) err -= (double)coeffs[i] * autoc[i+1];

    uint32_t bsize = (n_end - n_start) * 16;
    // +8: subframe header. Frame-level overhead is priced by the DP itself
    // via FrameWriter::frame_bits, so it must not be baked in here too.
    if (err <= 0) return 8 + (uint32_t)bps;

    double bps_est = 0.5 * std::log2(2.0 * M_PI * M_E * (err / bsize));
    if (bps_est < 1.0) bps_est = 1.0;
    return 8u + (uint32_t)(bsize * bps_est);
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
                                                (uint32_t)total_samples, m_bps, m_windows, m_max_candidates);
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
                SubframeParams s0 = optimize_subframe(ch0.data(), (uint32_t)total_samples, bps0, m_windows, m_max_candidates);
                SubframeParams s1 = optimize_subframe(ch1.data(), (uint32_t)total_samples, bps1, m_windows, m_max_candidates);
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
    // Cost = FrameWriter::frame_bits: the exact encoded frame size, header
    // (including the UTF-8 sample number, which grows with stream position),
    // subframe payload, byte-alignment pad, and CRCs.
    //
    // Phase 1: build all N×K work items, evaluate ALL in parallel with a
    //          flat thread pool.  All compute_block() calls are independent
    //          (different sample offsets, no shared writes), so 100% of
    //          threads stay busy for the full Phase 1 duration.
    // Phase 2: sequential DP over precomputed cost table — O(N×K), instant.
    // Phase 3: back-trace to recover the optimal frame sequence.

    // Shared with the window-table cache, which precomputes coefficients for
    // exactly these sizes (see DP_CANDIDATES at file scope).
    static const auto&    CANDIDATES     = DP_CANDIDATES;
    static const size_t   NUM_CANDS      = std::size(DP_CANDIDATES);
    static constexpr uint32_t STEP = 1024u; // GCD of all candidates

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

    // Reuse edges: input frames as exact-cost alternatives, bucketed by
    // start node. Exact-DP only — both edge types are then exact bit counts
    // (frame_bits for ours, the rewritten frame's size for reuse), so the DP
    // is optimal over the union of both partitions and can mix them. On a
    // tie the re-encoded frame wins (strict <), which keeps a re-encode of
    // our own output byte-stable.
    std::vector<std::vector<size_t>> reuse_at;
    if (full_search() && !m_reuse_edges.empty()) {
        reuse_at.resize(num_nodes);
        for (size_t k = 0; k < m_reuse_edges.size(); ++k) {
            const auto& e = m_reuse_edges[k];
            if (e.start_sample % STEP != 0 || e.block_size % STEP != 0) continue;
            const uint64_t end = e.start_sample + e.block_size;
            if (end > (uint64_t)num_nodes * STEP) continue; // remainder region
            reuse_at[(size_t)(e.start_sample / STEP)].push_back(k);
        }
    }

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
            uint64_t cost = dp[i] + FrameWriter::frame_bits(
                                        (uint64_t)i * STEP, CANDIDATES[c],
                                        m_sample_rate,
                                        cost_table[i * NUM_CANDS + c].total_bits);
            if (cost < dp[j]) {
                dp[j]        = cost;
                dp_parent[j] = (int)i;
                dp_cand  [j] = (int)c;
            }
        }
        if (!reuse_at.empty()) {
            for (size_t k : reuse_at[i]) {
                const auto& e = m_reuse_edges[k];
                size_t j = i + e.block_size / STEP;
                uint64_t cost = dp[i] + (uint64_t)e.frame_bytes * 8u;
                if (cost < dp[j]) {
                    dp[j]        = cost;
                    dp_parent[j] = (int)i;
                    dp_cand  [j] = (int)(NUM_CANDS + k); // reuse edge marker
                }
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
            if (ci >= NUM_CANDS) {
                // Reuse edge: no encoded parameters — the caller emits the
                // rewritten input frame identified by reuse_index.
                const auto& e = m_reuse_edges[ci - NUM_CANDS];
                BlockParams bp{};
                bp.block_size  = e.block_size;
                bp.total_bits  = e.frame_bytes * 8u;
                bp.reuse_index = (int32_t)e.input_index;
                result[idx] = bp;
            } else {
                result[idx] = cost_table[node * NUM_CANDS + ci];
            }
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

// Adaptive window selection: a hand stat→set map over the free granule
// statistics (WINDOWS_PLAN.md "Adaptive ICI window selection"). Every set has
// exactly 4 windows so the analysis cost per encoded block matches the fixed
// shortlist; only membership adapts. Thresholds are first-cut hand picks —
// calibrate against measurement before trusting them.
std::vector<WindowType> Optimizer::select_windows(
    uint64_t sample_start, uint32_t block_size) const
{
    const std::vector<WindowType> def = {WindowType::TUKEY_050, WindowType::HANN,
                                         WindowType::WELCH, WindowType::RECTANGULAR};
    if (m_granules.empty() || m_granules[0].empty()) return def;
    const size_t g0 = (size_t)(sample_start / 16);
    const size_t g1 = std::min((size_t)((sample_start + block_size) / 16),
                               m_granules[0].size());
    if (g1 <= g0 + 8) return def; // too few granules for meaningful stats

    const size_t n = g1 - g0;
    double sumE = 0.0, sumE2 = 0.0, sumL1 = 0.0, maxE = -1.0;
    size_t argmax = g0;
    for (size_t g = g0; g < g1; ++g) {
        double E = 0.0, L1 = 0.0;
        for (uint32_t c = 0; c < m_channels; ++c) {
            E  += m_granules[c][g].autoc[0];
            L1 += m_granules[c][g].autoc[1];
        }
        sumE += E; sumE2 += E * E; sumL1 += L1;
        if (E > maxE) { maxE = E; argmax = g; }
    }
    if (sumE <= 0.0) return def; // digital silence

    const double mean = sumE / (double)n;
    const double var  = std::max(0.0, sumE2 / (double)n - mean * mean);
    const double cv2  = var / (mean * mean); // energy dispersion: high ⇒ transient
    const double tilt = sumL1 / sumE;        // lag1/lag0: ~1 tonal, ~0 noisy
    const double pos  = (double)(argmax - g0) / (double)n;

    // Transient routing gate: 0.5 saturates (0.25 measured identical). Both
    // search modes route: -c 0 prices partial/punchout windows exactly
    // (music_20s −1991 B vs fixed-4, 3-min synthetic percussion −19983 B),
    // and the ranked scorer's zero-fraction term (see optimize_subframe)
    // prices their blind region honestly — before that term existed, routing
    // under -c 8 lost (+2489 B music / +4151 B percussion); with it, it wins
    // (−104 B / −17992 B).
    if (cv2 > 0.5) {
        // Transient content: offer the partial/punchout pair that isolates
        // the energy peak, plus general-purpose tapers. rect stays in the set
        // because the gate has a false-positive mode — amplitude-modulated
        // tonal content (beats) has high energy dispersion too, and dropping
        // rect there cost +4523 B on the 3-min synthetic tonal fixture. A
        // routed block therefore analyses 5 windows instead of 4; routed
        // blocks are a minority, so the extra analysis is marginal.
        if (pos < 0.33)
            return {WindowType::PARTIAL_TUKEY_2_000, WindowType::TUKEY_050,
                    WindowType::HANN, WindowType::WELCH, WindowType::RECTANGULAR};
        if (pos < 0.67)
            return {WindowType::PARTIAL_TUKEY_2_033, WindowType::PUNCHOUT_TUKEY_2_033,
                    WindowType::TUKEY_050, WindowType::HANN, WindowType::RECTANGULAR};
        return {WindowType::PARTIAL_TUKEY_2_067, WindowType::PUNCHOUT_TUKEY_2_067,
                WindowType::TUKEY_050, WindowType::HANN, WindowType::RECTANGULAR};
    }
    if (tilt > 0.95) // stationary tonal: mild tapers, keep the workhorse
        return {WindowType::TUKEY_005, WindowType::TUKEY_010,
                WindowType::TUKEY_050, WindowType::HANN};
    if (tilt < 0.60) // noisy: heavy tapers, keep the workhorse
        return {WindowType::TUKEY_050, WindowType::HANN,
                WindowType::TUKEY_075, WindowType::TUKEY_090};
    return def;
}

BlockParams Optimizer::compute_block(
    const std::vector<std::vector<int32_t>>& pcm_data,
    uint64_t sample_start, uint32_t block_size) const
{
    BlockParams bp{};
    bp.block_size = block_size;

    // Adaptive selection replaces the fixed window list per block. Estimated
    // DP only: under -e the wide set is already offered, so a selector has
    // nothing to add (and the DP's phase-1 blocks would pay it N×K times).
    std::vector<WindowType> selected;
    const std::vector<WindowType>* win_set = &m_windows;
    if (m_adaptive && !full_search()) {
        selected = select_windows(sample_start, block_size);
        win_set = &selected;
    }
    const std::vector<WindowType>& wins = *win_set;

    if (m_channels == 1) {
        bp.stereo_mode  = 0;
        bp.subframes[0] = optimize_subframe(
            &pcm_data[0][sample_start], block_size, m_bps, wins, m_max_candidates);
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
                cache[sig] = optimize_subframe(&pcm_data[0][sample_start], block_size, m_bps, wins, m_max_candidates);
            } else if (sig == SIG_R) {
                cache[sig] = optimize_subframe(&pcm_data[1][sample_start], block_size, m_bps, wins, m_max_candidates);
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
                cache[sig] = optimize_subframe(ch.data(), block_size, bps_s, wins, m_max_candidates);
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
