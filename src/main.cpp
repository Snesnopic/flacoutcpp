#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "flacoutcpp.hpp"

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options] <input.flac> [output.flac]\n"
        << "Options:\n"
        << "  -e, --exhaustive     Perform full exhaustive search (extremely slow)\n"
        << "  -c, --candidates N   Ranked search: fully evaluate only the N most\n"
        << "                       promising (window, order) pairs per subframe,\n"
        << "                       ranked by Levinson-Durbin prediction error.\n"
        << "                       Sits between the default and -e. Larger N is\n"
        << "                       slower and compresses better. Unlike the other\n"
        << "                       options this one changes the output.\n"
        << "  -n, --no-metadata    Do not copy metadata from input to output\n"
        << "  -q, --quiet          Suppress all progress output\n"
        << "  -t, --threads N      Limit parallel worker threads (default: all CPUs)\n"
        << "  -w, --windows <list> Comma-separated list of apodization windows to use\n"
        << "                       (default: all windows — maximum compression)\n"
        << "Available window names:\n"
        << "  rect, bartlett, bartletthann, blackman, blackmanharris, connes, flattop,\n"
        << "  gauss025, gauss0125, hamming, hann, kaiserbessel, nuttall, triangle, welch,\n"
        << "  tukey005, tukey010, tukey020, tukey050, tukey075, tukey090,\n"
        << "  partialtukey2, partialtukey2_033, partialtukey2_067,\n"
        << "  punchouttukey2_033, punchouttukey2_067\n";
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
                cfg.max_candidates = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -c requires a positive integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            if (cfg.max_candidates == 0) {
                std::cerr << "Error: -c requires a value of at least 1.\n";
                return EXIT_FAILURE;
            }

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

    if (cfg.exhaustive && cfg.max_candidates > 0) {
        std::cerr << "Error: -e and -c are mutually exclusive (-e already "
                     "evaluates every candidate).\n";
        return EXIT_FAILURE;
    }

    if (cfg.verbose) {
        if (cfg.max_candidates > 0)
            std::cout << "Ranked search: " << cfg.max_candidates
                      << " candidates/subframe\n";
        if (cfg.windows.empty())
            std::cout << "Windows: all (" << all_window_types().size() << " functions)\n";
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
