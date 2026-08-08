#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "flacoutcpp.hpp"

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options] <input.flac> [output.flac]\n"
        << "Options:\n"
        << "  -e, --exhaustive     Exact search: fully encode every block-size and\n"
        << "                       stereo-mode choice instead of estimating them,\n"
        << "                       and offer all windows (extremely slow)\n"
        << "  -c, --candidates N   Fully evaluate only the N most promising\n"
        << "                       (window, order) pairs per subframe, ranked by\n"
        << "                       Levinson-Durbin prediction error. 0 = no limit.\n"
        << "                       Default: 8, or 0 when -e is given without -c.\n"
        << "                       Composes with -e (e.g. -e -c 8). Larger N is\n"
        << "                       slower and compresses better.\n"
        << "  -n, --no-metadata    Do not copy metadata from input to output\n"
        << "  -a, --adaptive-windows  Experimental: pick each block's 4-window set\n"
        << "                       from its signal statistics instead of the fixed\n"
        << "                       shortlist (estimated-DP only; excludes -e/-w)\n"
        << "  -q, --quiet          Suppress all progress output\n"
        << "  -t, --threads N      Limit parallel worker threads (default: all CPUs)\n"
        << "  -w, --windows <list> Comma-separated list of apodization windows to use\n"
        << "                       (default: all 26 with -e, else tukey050,hann,welch,rect)\n"
        << "Available window names:\n"
        << "  rect, bartlett, bartletthann, blackman, blackmanharris, connes, flattop,\n"
        << "  gauss025, gauss0125, hamming, hann, kaiserbessel, nuttall, triangle, welch,\n"
        << "  tukey005, tukey010, tukey020, tukey050, tukey075, tukey090,\n"
        << "  partialtukey2, partialtukey2_033, partialtukey2_067,\n"
        << "  punchouttukey2_033, punchouttukey2_067\n"
        << "Experimental windows (never in a default set; explicit -w only):\n"
        << "  lanczos, bohman, parzen, plancktaper010, plancktaper025,\n"
        << "  partialtukey3_{1,2,3}, punchouttukey3_{1,2,3},\n"
        << "  partialtukey3h_{000,033,067}, punchouttukey3h_{025,050},\n"
        << "  expdecay{2,4}, expattack{2,4}, attackdecay{005,010,020},\n"
        << "  dpss{2,3,4}\n";
}

// Split a comma-separated string into tokens.
static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return EXIT_FAILURE; }

    flacoutcpp::Config cfg;
    std::vector<std::string> positional;
    bool candidates_given = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--version") {
            std::cout << FLACOUTCPP_VERSION << "\n";
            return EXIT_SUCCESS;

        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;

        } else if (arg == "-e" || arg == "--exhaustive") {
            cfg.exhaustive = true;

        } else if (arg == "-c" || arg == "--candidates") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -c requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                // stoul accepts a leading '-' by wrapping; reject it explicitly.
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                cfg.max_candidates = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -c requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            candidates_given = true;

        } else if (arg == "-a" || arg == "--adaptive-windows") {
            cfg.adaptive_windows = true;

        } else if (arg == "-q" || arg == "--quiet") {
            cfg.verbose = false;

        } else if (arg == "-n" || arg == "--no-metadata") {
            cfg.copy_metadata = false;

        } else if (arg == "-w" || arg == "--windows") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -w requires an argument.\n";
                return EXIT_FAILURE;
            }
            ++i;
            for (const auto& name : split_csv(argv[i])) {
                auto wt = window_from_name(name);
                if (wt == WindowType::COUNT) {
                    std::cerr << "Warning: unrecognised window '" << name << "' — skipped.\n";
                } else {
                    cfg.windows.push_back(wt);
                }
            }

        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -t requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                cfg.max_threads = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -t requires a positive integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg.substr(0, 2) == "--" || arg.substr(0, 1) == "-") {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;

        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty()) {
        std::cerr << "Error: no input file specified.\n";
        return EXIT_FAILURE;
    }

    const std::string input  = positional[0];
    const std::string output = (positional.size() >= 2)
                                 ? positional[1]
                                 : input + ".optimized.flac";

    // -e alone means the classic unlimited sweep; -c composes with it to bound
    // the per-subframe search while keeping the exact block-partitioning DP.
    if (!candidates_given && cfg.exhaustive)
        cfg.max_candidates = 0;

    // Adaptive selection chooses the window set itself; -e and -w each define
    // their own set, so the combinations are contradictory rather than merely
    // redundant — reject them.
    if (cfg.adaptive_windows && (cfg.exhaustive || !cfg.windows.empty())) {
        std::cerr << "Error: -a is estimated-DP only and picks its own windows; "
                     "it cannot be combined with -e or -w.\n";
        return EXIT_FAILURE;
    }

    if (cfg.verbose) {
        if (cfg.max_candidates > 0)
            std::cout << "Ranked search: " << cfg.max_candidates
                      << " candidates/subframe\n";
        else
            std::cout << "Ranked search: unlimited (full sweep)\n";
        if (cfg.windows.empty()) {
            if (cfg.exhaustive)
                std::cout << "Windows: all (" << all_window_types().size() << " functions)\n";
            else
                std::cout << "Windows: default short list (tukey050, hann, welch, rect)\n";
        }
        else {
            std::cout << "Windows: ";
            for (size_t i = 0; i < cfg.windows.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << window_to_name(cfg.windows[i]);
            }
            std::cout << "\n";
        }
        std::cout << "Optimising: " << input << " -> " << output << "\n";
        if (!cfg.copy_metadata) std::cout << "Metadata copying disabled.\n";
    }

    if (!flacoutcpp::optimise(input, output, cfg)) {
        std::cerr << "Optimisation failed.\n";
        return EXIT_FAILURE;
    }

    if (cfg.verbose) std::cout << "Done.\n";
    return EXIT_SUCCESS;
}
