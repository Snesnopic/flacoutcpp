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
                       default 24 (effort level 3), or 0 when -e is given
                       without -c/-L/-E)
  -p, --patience N     Keep scanning past -c N while candidates are still
                       improving; stop after N consecutive that are not
                       (default: 2x -c; 0 = plain top-N cut)
  -a, --adaptive-windows  Experimental: pick each block's window set from its
                       signal statistics instead of the fixed shortlist
                       (estimated-DP only; excludes -e/-w)
  -n, --no-metadata    Do not copy metadata from input to output
  -R, --no-reuse       Disable input-frame reuse (see below); mainly for
                       measuring the raw search
  -W, --warn-superior  Warn on stderr when the input's own frames beat the
                       re-encode, naming the input's encoder when its
                       metadata says; incompatible with -R
  -q, --quiet          Suppress all progress output
  -t, --threads N      Limit parallel worker threads (default: all CPUs)
  -w, --windows <list> Comma-separated list of apodization windows to use;
                       an entry custom:<file> loads a shape from a knot file
```

If `[output.flac]` is omitted, it will default to `<input.flac>.optimized.flac`.

### Choosing a search level

All three levels are lossless — the decoded audio is bit-identical to the input
in every case. They differ only in how hard the encoder works to shrink the file.

Three knobs control the search:

- **`-e`** switches the block-partitioning DP from granule-based *estimates* to
  *exact* costs — every (position, block size) pair and every stereo mode is
  fully encoded before the DP chooses — and widens the window set from 4 to
  all 26 standard windows (experimental windows such as `lanczos` stay opt-in
  via `-w`). Orders of magnitude more CPU.
- **`-c N`** bounds the per-subframe LPC search. Levinson-Durbin already
  computes the prediction error at each order as a by-product of deriving the
  coefficients; the ranking uses it to estimate every (window, order) pair's
  cost up front and fully evaluates only the best `N`. `0` means no limit.
- **`-p N`** decides when to stop descending that ranked list. A fixed cut at
  `N` assumes the ranking is right about what lies below it, and it frequently
  is not: measured over every candidate, the winner's rank is heavy-tailed —
  rank 0 wins 56% of subframes on a 24-bit pure sine but the tail reaches rank
  59, and on real music only about half of all winners fall inside rank 7.
  Patience uses the exact costs the scan is already computing as its stopping
  signal instead: keep going while the list is still yielding improvements,
  stop after `N` consecutive candidates that are not. `-c` becomes a floor
  rather than a ceiling, and the extra work lands only on the subframes that
  need it. Defaults to `2x -c`; `-p 0` restores the plain top-N cut.

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
unlimited sweep on the heuristic's shortlist) barely out-compresses the
default: the ranking finds nearly everything the sweep finds.

These figures are one excerpt of one track; the trade depends on the material.

Patience measured separately, on 9 real tracks (200.9 MB) for size and 3 of
them for time (interleaved, best-of-3), against the same default with `-p 0`:

| patience | size | share of `-c 0`'s remaining gain | time |
|---|---|---|---|
| `-p 0` (plain top-N) | — | 0% | 1.00x |
| `-p 8` | −0.015% | 26% | 1.09x |
| `-p 16` (default at `-c 8`) | −0.035% | 59% | 1.34x |
| `-p 32` | −0.051% | 86% | 1.82x |
| `-c 0` | −0.060% | 100% | 3.77x |

The default of `2x -c` was chosen as the smallest patience that leaves no
fixture compressing worse than the pre-ranking `-c 0` default did; `-p 8` still
loses on 24-bit synthetics.
(They also predate the RICE2 residual coding and ranked-scorer improvements,
which shrank 24-bit output by ~2% across all modes — treat the column as a
shape, not gospel.)

### Frame reuse (on by default)

The input file arrives already partitioned into frames whose exact compressed
sizes are known for free, so the encoder lets them compete: wherever the
input's frames cover a span of the chosen partition in fewer bytes than the
re-encoded frames, the input frames are spliced into the output — payload
verbatim, header rewritten to the output stream's conventions, CRCs
recomputed. Under `-e` they additionally enter the block-partitioning DP as
exact-cost edges, so the optimizer can interleave reused frames with
re-encoded ones (including inputs whose frame boundaries don't align with the
DP grid).

Splicing on its own does not quite get there: it takes the cheaper side per
segment, but the input side is priced as *rewritten* frames, and a rewritten
frame carries a variable-blocksize sample number where a fixed-blocksize input
carried a frame number — 1–2 bytes more per frame. So if the finished file
would still be larger than the input, the input ships instead: copied
verbatim, or under `-n` as the input's audio frames beneath a fresh
STREAMINFO-only header, so dropping metadata doesn't also drop the guarantee.

The net guarantee: **re-encoding never grows a file**, at any search level,
and running flacoutcpp over its own output is byte-stable. `-R` turns all of
this off — useful only when measuring what the search itself produces.

`-W` reports the flip side: when reuse fired, some encoder out there beat
this one on part of the file, and the warning names it from the input's
vendor string. Useful for surveying a library for material worth a deeper
look.

### Adaptive window selection (`-a`, experimental)

Estimated-DP modes normally analyse a fixed shortlist. With `-a`,
each block picks its window set from signal statistics the encoder has
already computed (energy dispersion, transient position, spectral tilt):
transient blocks bring in partial/punchout windows aimed at the energy peak,
at unchanged analysis cost. Measured across a 9-album corpus at `-c 0` it
saved 0.03–0.34% per album with zero regressing tracks; at the plain default
it is nearly free but the gains are small. Excludes `-e` (which already
offers every window) and `-w` (which fixes the set by hand).

### Custom window shapes (`-w custom:<file>`)

A `-w` entry of the form `custom:<file>` loads a window shape from a text file
instead of using a compiled-in one, so new shapes can be tried — or searched
over by an external script — without recompiling:

```bash
flacoutcpp -w custom:my_shape.txt,hann input.flac output.flac
```

The file holds the window's coefficients at evenly spaced positions across the
block, one or many per line, separated by whitespace, commas or semicolons,
with `#` starting a comment. They are linearly interpolated to whatever block
size the encoder needs: two values describe a ramp, and as many values as the
block size reproduces a shape exactly. Absolute scale is irrelevant — the LPC
search is scale-invariant — so the values are used exactly as written, with no
normalisation. Up to four may be loaded per run.
[bench/windows/example_taper.txt](bench/windows/example_taper.txt) documents the
format and is a usable starting point.

Custom windows are opt-in like the other experimental windows: they are never
part of any default set, so output is unchanged unless you name one.

### Reproducibility

For a given input and options, flacoutcpp produces the same bytes regardless of
optimization level or host CPU tuning. This is not free: the encoder picks LPC
coefficients from a double-precision autocorrelation and quantizes them to 8-15
bits, so a last-bit rounding difference can occasionally tip which candidate
wins. `flacout_lib` is therefore built with floating-point contraction disabled
(`-ffp-contract=off`, or `/fp:precise` on MSVC) — see the comment in
`CMakeLists.txt`. Do not override it if byte-identical output across builds
matters to you.
