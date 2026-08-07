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
  -e, --exhaustive     Perform full exhaustive search (extremely slow)
  -c, --candidates N   Ranked search: evaluate only the N most promising
                       (window, order) pairs per subframe
  -n, --no-metadata    Do not copy metadata from input to output
  -q, --quiet          Suppress all progress output
  -t, --threads N      Limit parallel worker threads (default: all CPUs)
  -w, --windows <list> Comma-separated list of apodization windows to use
```

If `[output.flac]` is omitted, it will default to `<input.flac>.optimized.flac`.

### Choosing a search level

All three levels are lossless — the decoded audio is bit-identical to the input
in every case. They differ only in how hard the encoder works to shrink the file.

The default heuristic search is fast. `-e` searches every apodization window,
LPC order and quantization precision for every block, which compresses better
but costs orders of magnitude more CPU.

`-c N` sits between them. Levinson-Durbin already computes the prediction error
at each order as a by-product of deriving the coefficients; ranked search uses it
to estimate every (window, order) pair's cost up front and fully evaluates only
the best `N`. Everything else matches `-e`: exact DP over block sizes, all four
stereo modes, the full precision sweep, and all 26 windows offered to the ranking.

Measured on a 3-second excerpt of a 24-bit/44.1 kHz track, 16 threads, relative
to `-e`:

| mode | time vs `-e` | size vs `-e` |
|---|---|---|
| default (heuristic) | 717x faster | +1.14% |
| `-c 1` | 32x faster | +0.37% |
| `-c 8` | 25x faster | +0.27% |
| `-c 32` | 15x faster | +0.14% |
| `-e` | — | — |

So `-e` buys about 1.1% over the default, and `-c 8` recovers roughly three
quarters of that for a twenty-fifth of `-e`'s cost. The curve flattens quickly,
and low `N` is dominated by fixed per-block costs rather than by `N` itself,
which is why `-c 1` is not much faster than `-c 8`.

These figures are one excerpt of one track; the trade depends on the material.
