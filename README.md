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
  -e, --exhaustive     Exact search: fully encode every block-size and
                       stereo-mode choice instead of estimating them,
                       and offer all windows (extremely slow)
  -c, --candidates N   Fully evaluate only the N most promising
                       (window, order) pairs per subframe (0 = no limit;
                       default 8, or 0 when -e is given without -c)
  -n, --no-metadata    Do not copy metadata from input to output
  -q, --quiet          Suppress all progress output
  -t, --threads N      Limit parallel worker threads (default: all CPUs)
  -w, --windows <list> Comma-separated list of apodization windows to use
```

If `[output.flac]` is omitted, it will default to `<input.flac>.optimized.flac`.

### Choosing a search level

All three levels are lossless — the decoded audio is bit-identical to the input
in every case. They differ only in how hard the encoder works to shrink the file.

Two independent knobs control the search:

- **`-e`** switches the block-partitioning DP from granule-based *estimates* to
  *exact* costs — every (position, block size) pair and every stereo mode is
  fully encoded before the DP chooses — and widens the window set from 4 to
  all 26. Orders of magnitude more CPU.
- **`-c N`** bounds the per-subframe LPC search. Levinson-Durbin already
  computes the prediction error at each order as a by-product of deriving the
  coefficients; the ranking uses it to estimate every (window, order) pair's
  cost up front and fully evaluates only the best `N`. `0` means no limit.

The flags compose. The default is plain `-c 8` (estimated DP, ranked search);
bare `-e` implies `-c 0` (exact DP, unlimited sweep); `-e -c 8` prices blocks
exactly but keeps the subframe search bounded — what plain `-c 8` meant before
the flags composed.

Measured on a 3-second excerpt of a 24-bit/44.1 kHz track (batched runs,
best-of-3, 16 threads), relative to `-e`:

| mode | time vs `-e` | size vs `-e` |
|---|---|---|
| default (= `-c 8`) | 572x faster | +0.93% |
| `-c 0` | 196x faster | +0.89% |
| `-e -c 1` | 65x faster | +0.37% |
| `-e -c 8` | 42x faster | +0.27% |
| `-e -c 32` | 19x faster | +0.14% |
| `-e` | — | — |

So `-e` buys just under 1% over the default, and `-e -c 8` recovers roughly
three quarters of that at a fortieth of `-e`'s cost. The curve flattens
quickly at low `N`, which is dominated by fixed per-block costs rather than by
`N` itself — `-e -c 1` is not much faster than `-e -c 8`. Note `-c 0` (the
unlimited sweep on the heuristic's 4-window set) barely out-compresses the
default: the ranking finds nearly everything the sweep finds.

These figures are one excerpt of one track; the trade depends on the material.

### Reproducibility

For a given input and options, flacoutcpp produces the same bytes regardless of
optimization level or host CPU tuning. This is not free: the encoder picks LPC
coefficients from a double-precision autocorrelation and quantizes them to 8-15
bits, so a last-bit rounding difference can occasionally tip which candidate
wins. `flacout_lib` is therefore built with floating-point contraction disabled
(`-ffp-contract=off`, or `/fp:precise` on MSVC) — see the comment in
`CMakeLists.txt`. Do not override it if byte-identical output across builds
matters to you.
