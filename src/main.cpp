#include <iostream>
#include <string>
#include "processor.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: flacoutcpp <input.flac> [output.flac]\n";
        return 1;
    }

    const std::string input_file = argv[1];
    std::string output_file;
    if (argc >= 3) {
        output_file = argv[2];
    } else {
        // Default output is input + ".optimized.flac"
        output_file = input_file + ".optimized.flac";
    }

    std::cout << "Optimizing: " << input_file << " -> " << output_file << "\n";

    Processor processor(input_file, output_file);
    if (!processor.process()) {
        std::cerr << "Processing failed!\n";
        return 1;
    }

    std::cout << "Optimization complete.\n";
    return 0;
}
