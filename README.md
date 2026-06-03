# flacoutcpp

`flacoutcpp` is a modern C++17 recreation of the original `flacout` utility. It focuses on providing extreme lossless FLAC recompression by using advanced dynamic programming techniques to find the absolute optimal block partitioning and LPC coefficients for the FLAC stream.

While the original tool was a closed-source Windows binary, this project reverse-engineered its core compression logic and ported it to a clean, multi-threaded C++ implementation that can run on any modern platform (including macOS ARM64 and Linux). It significantly outperforms the original in speed by leveraging `std::thread` to evaluate block configurations concurrently and includes early-exit heuristics to speed up exhaustive LPC searches without any loss in compression ratio.

By default, `flacoutcpp` acts as a drop-in FLAC optimizer, flawlessly cloning all original IDv3 tags, Vorbis Comments, cover arts, and other metadata into the recompressed output.

## Building from source

The project uses CMake for its build system and includes `libflac` as a submodule. A C++17 compliant compiler is required.

To build the project:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Usage

```bash
Usage: flacoutcpp [options] <input.flac> [output.flac]
Options:
  -n, --no-metadata    Do not copy metadata from input to output
```

If `[output.flac]` is omitted, it will default to `<input.flac>.optimized.flac`.
