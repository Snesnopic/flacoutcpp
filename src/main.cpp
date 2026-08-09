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
        << "                       and offer all windows (extremely slow).\n"
        << "                       Worth far more than any -E level: 0.6% (24-bit)\n"
        << "                       to 2.3% (16-bit) on real music, against ~0.2%\n"
        << "                       across the whole dial. Rarely worth it bare:\n"
        << "                       '-e -L 1' is ~5x faster for 99.5% of the gain,\n"
        << "                       and '-e -E 0' ~80x faster for ~76% of it.\n"
        << "  -c, --candidates N   Fully evaluate only the N most promising\n"
        << "                       (window, order) pairs per subframe, ranked by\n"
        << "                       Levinson-Durbin prediction error. 0 = no limit.\n"
        << "                       Default: 8, or 0 when -e is given without\n"
        << "                       -c/-L/-E. Composes with -e (e.g. -e -c 8).\n"
        << "                       Larger N is slower and compresses better.\n"
        << "  -p, --patience N     Keep scanning past -c N while candidates are\n"
        << "                       still improving; stop after N consecutive that\n"
        << "                       are not. Makes -c a floor, not a ceiling.\n"
        << "                       Default: 2x -c. 0 disables it (plain top-N cut).\n"
        << "  -E, --effort N       Effort 0-9: one dial along the measured\n"
        << "                       size/time frontier, setting -c, -L and -a\n"
        << "                       together (they are not independent —\n"
        << "                       which mix is efficient shifts with the\n"
        << "                       budget). 0 fastest, 9 = every candidate and\n"
        << "                       every rung. Against the -c 8 default on a\n"
        << "                       188-track mix, level 3 is 0.056% smaller and\n"
        << "                       faster; level 9 is 0.113% smaller.\n"
        << "                       An explicit -c/-p/-L/-a wins; the\n"
        << "                       level's -a yields to -e/-w rather than\n"
        << "                       erroring. The dial tunes the search *within*\n"
        << "                       a mode; it is not a substitute for -e.\n"
        << "  -L, --rungs N        Encode only the N most promising of the 8 LPC\n"
        << "                       coefficient precisions per candidate, chosen by\n"
        << "                       an analytic model of the quantization error\n"
        << "                       instead of by encoding all of them.\n"
        << "                       0 = all (default). Against all 8 rungs: 1\n"
        << "                       costs 0.019% for 1.31x, 2 costs 0.009% for\n"
        << "                       1.25x, 3 costs 0.005%.\n"
        << "                       -c and -L are not independent —\n"
        << "                       prefer -E, which pairs them along the measured\n"
        << "                       frontier, unless you know which pair you want.\n"
        << "  -n, --no-metadata    Do not copy metadata from input to output\n"
        << "  -a, --adaptive-windows  Experimental: add windows chosen from each\n"
        << "                       block's signal statistics to the shortlist\n"
        << "                       (estimated-DP only; excludes -e/-w)\n"
        << "  -R, --no-reuse       Disable input-frame reuse. By default, input\n"
        << "                       frames that beat the re-encoded ones are spliced\n"
        << "                       into the output (and the input is copied through\n"
        << "                       if the output would still be larger), so\n"
        << "                       re-encoding never grows a file. -R measures the\n"
        << "                       raw search alone — mainly for testing\n"
        << "  -W, --warn-superior  Warn on stderr when the input's own frames beat\n"
        << "                       the re-encode (i.e. frame reuse fired), naming\n"
        << "                       the input's encoder when its metadata says.\n"
        << "                       Prints even with -q; incompatible with -R\n"
        << "  -q, --quiet          Suppress all progress output\n"
        << "  -t, --threads N      Limit parallel worker threads (default: all CPUs)\n"
        << "  -w, --windows <list> Comma-separated list of apodization windows to use\n"
        << "                       (default: all 26 with -e, else tukey050,hann,welch,\n"
        << "                       rect and the partial/punchout tukey pair at .33/.67)\n"
        << "                       An entry of the form custom:<file> loads a window\n"
        << "                       shape from a knot file (up to 4 per run); see\n"
        << "                       bench/windows/example_taper.txt for the format\n"
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
        << "  punchouttukey2_000,\n"
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
    // Effort is a preset for -c/-L, so an explicit knob must win regardless of
    // the order the two appear in. Record what was named and re-apply it after
    // the level, rather than depending on argv order.
    bool patience_given = false, rungs_given = false;
    bool adaptive_given = false;
    int  effort = -1;
    unsigned given_candidates = 0, given_rungs = 0;
    int      given_patience = -1;

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
                given_candidates = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -c requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            candidates_given = true;

        } else if (arg == "-p" || arg == "--patience") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -p requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                given_patience = static_cast<int>(std::stoul(argv[i]));
                patience_given = true;
            } catch (const std::exception&) {
                std::cerr << "Error: -p requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-E" || arg == "--effort") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -E requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                effort = static_cast<int>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -E requires an effort level 0-9, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-L" || arg == "--rungs") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -L requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                given_rungs = static_cast<unsigned>(std::stoul(argv[i]));
                rungs_given = true;
            } catch (const std::exception&) {
                std::cerr << "Error: -L requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-a" || arg == "--adaptive-windows") {
            cfg.adaptive_windows = true;
            adaptive_given = true;

        } else if (arg == "-R" || arg == "--no-reuse") {
            cfg.reuse_frames = false;

        } else if (arg == "-W" || arg == "--warn-superior") {
            cfg.warn_superior = true;

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
                // A custom: entry names a file, so a bad one is a mistake worth
                // stopping for — silently dropping it would quietly change the
                // window set a measurement run was built around.
                if (name.rfind("custom:", 0) == 0) {
                    std::string err;
                    auto wt = register_custom_window(name.substr(7), &err);
                    if (wt == WindowType::COUNT) {
                        std::cerr << "Error: -w " << name << ": " << err << "\n";
                        return EXIT_FAILURE;
                    }
                    cfg.windows.push_back(wt);
                    continue;
                }
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

    // Effort first, then whatever was named explicitly — so `-E 7 -L 0` means
    // "level 7's depth, but price the whole ladder".
    if (effort >= 0 && !flacoutcpp::apply_effort(cfg, effort)) {
        std::cerr << "Error: -E takes an effort level 0-9, got " << effort << ".\n";
        return EXIT_FAILURE;
    }
    if (candidates_given)  cfg.max_candidates  = given_candidates;
    if (patience_given)    cfg.patience        = given_patience;
    if (rungs_given)       cfg.precision_rungs = given_rungs;
    // An explicit -a still means -a, including the hard error below when it is
    // combined with -e or -w. A level's adaptive setting is a preference, not
    // a request, so apply_effort already dropped it in those cases.
    if (adaptive_given)    cfg.adaptive_windows = true;

    // -e alone means the classic unlimited sweep; any of -c/-L/-E alongside it
    // is a deliberate statement about the search and is left alone, so
    // `-e -E 5` means level 5's depth under exact DP.
    if (cfg.exhaustive && effort < 0 && !candidates_given)
        cfg.max_candidates = 0;

    // -W reads the reuse comparison's results, so it needs reuse enabled.
    if (cfg.warn_superior && !cfg.reuse_frames) {
        std::cerr << "Error: -W detects superior input frames via the reuse "
                     "comparison; it cannot be combined with -R.\n";
        return EXIT_FAILURE;
    }

    // Adaptive selection chooses the window set itself; -e and -w each define
    // their own set, so the combinations are contradictory rather than merely
    // redundant — reject them.
    if (cfg.adaptive_windows && (cfg.exhaustive || !cfg.windows.empty())) {
        std::cerr << "Error: -a is estimated-DP only and picks its own windows; "
                     "it cannot be combined with -e or -w.\n";
        return EXIT_FAILURE;
    }

    // Patience defaults to twice the candidate budget; resolve it here so the
    // rest of the program sees a concrete number.
    if (cfg.patience < 0)
        cfg.patience = static_cast<int>(cfg.max_candidates) * 2;

    if (cfg.verbose) {
        if (cfg.max_candidates > 0)
            std::cout << "Ranked search: " << cfg.max_candidates
                      << " candidates/subframe, patience "
                      << (cfg.patience > 0 ? std::to_string(cfg.patience) : std::string("off"))
                      << "\n";
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
