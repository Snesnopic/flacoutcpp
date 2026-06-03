#include <iostream>
#include <string>
#include "processor.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: flacoutcpp [options] <input.flac> [output.flac]\n";
        std::cerr << "Options:\n";
        std::cerr << "  -n, --no-metadata    Do not copy metadata from input to output\n";
        return 1;
    }

    bool copy_metadata = true;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--no-metadata") {
            copy_metadata = false;
        } else {
            args.push_back(arg);
        }
    }

    if (args.empty()) {
        std::cerr << "Error: No input file specified.\n";
        return 1;
    }

    const std::string input_file = args[0];
    std::string output_file;
    if (args.size() >= 2) {
        output_file = args[1];
    } else {
        // Default output is input + ".optimized.flac"
        output_file = input_file + ".optimized.flac";
    }

    std::cout << "Optimizing: " << input_file << " -> " << output_file << "\n";
    if (!copy_metadata) std::cout << "Metadata copying is DISABLED.\n";

    Processor processor(input_file, output_file, copy_metadata);
    if (!processor.process()) {
        std::cerr << "Processing failed!\n";
        return 1;
    }

    std::cout << "Optimization complete.\n";
    return 0;
}
