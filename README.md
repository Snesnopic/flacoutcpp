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
                       and offer all windows (extremely slow).
                       Worth far more than any -E level: 0.6% (24-bit)
                       to 2.3% (16-bit) on real music, against ~0.2%
                       across the whole dial. Rarely worth it bare:
                       '-e -L 1' is ~5x faster for 99.5% of the gain,
                       and '-e -E 0' ~80x faster for ~76% of it.
  -c, --candidates N   Fully evaluate only the N most promising
                       (window, order) pairs per subframe, ranked by
                       Levinson-Durbin prediction error. 0 = no limit.
                       Default: 24 (effort level 3), or 0 when -e is
                       given without -c/-L/-E. Composes with -e
                       (e.g. -e -c 8). Larger N is slower and
                       compresses better.
  -p, --patience N     Keep scanning past -c N while candidates are
                       still improving; stop after N consecutive that
                       are not. Makes -c a floor, not a ceiling.
                       Default: 2x -c. 0 disables it (plain top-N cut).
  -E, --effort N       Effort 0-12: one dial along the measured
                       size/time frontier, setting -c, -L, -a and
                       — from level 10 — exact DP, together (they
                       are not independent; which mix is efficient
                       shifts with the budget). 0 fastest, 9 = every
                       candidate and every rung under estimated DP,
                       10-12 = exact DP at increasing depth. Level 3
                       is the default. Against it, on a 188-track
                       mix: 0 is +0.15% at 0.8x the time, 6 is
                       -0.03% at 1.5x, 9 is -0.05% at 5.7x, 10 is
                       -0.52% at 7.7x, 12 is -0.61% at 24x. Level 10
                       is the value corner — ten times level 9's
                       compression for 20% more time — because exact
                       pricing is worth far more than search depth.
                       An explicit -c/-p/-L/-a wins; the level's -a
                       yields to -e/-w rather than erroring.
  -L, --rungs N        Encode only the N most promising of the 8 LPC
                       coefficient precisions per candidate, chosen by
                       an analytic model of the quantization error
                       instead of by encoding all of them.
                       Default: 1 (effort level 3); 0 under a bare
                       -e, which prices the whole ladder. Against
                       all 8 rungs: 1 costs 0.019% for 1.31x, 2
                       costs 0.009% for 1.25x, 3 costs 0.005%.
                       -c and -L are not independent —
                       prefer -E, which pairs them along the measured
                       frontier, unless you know which pair you want.
  -Q, --lattice N      Refine the winning subframe's quantized LPC
                       coefficients by coordinate descent: try each
                       tap at +-1, keep what lowers the exact cost,
                       up to N sweeps (0 = off, the default).
                       Experimental. Never grows a subframe.
  -b, --blocks <list>  Comma-separated block sizes the DP may choose
                       from (default: 1024,2048,4096,8192,16384).
                       Each must be a multiple of 16 in [16, 65520],
                       and every size must be a multiple of the
                       smallest, or the DP cannot reach the stream's
                       end. FLAC's own limits are 16 and 65535, but
                       65535 is odd, so no usable grid reaches it;
                       65520 is the largest attainable size, and
                       needs a smallest size that divides it (e.g.
                       16 or 5040, not 1024). Cost scales with
                       sum(sizes)/gcd(sizes): the default is 31 block
                       -samples of work per input sample, and
                       16,...,32768 is 4095 — about 130x. Best paired
                       with -e, which prices every choice exactly.
  -n, --no-metadata    Do not copy metadata from input to output
  -a, --adaptive-windows  Add windows chosen from each block's signal
                       statistics to the shortlist. On by default;
                       estimated-DP only, so it yields silently to
                       -e/-w and is an error only when named there.
  -A, --no-adaptive-windows  Turn that off.
  -R, --no-reuse       Disable input-frame reuse. By default, input
                       frames that beat the re-encoded ones are spliced
                       into the output (and the input is copied through
                       if the output would still be larger), so
                       re-encoding never grows a file. -R measures the
                       raw search alone — mainly for testing
  -W, --warn-superior  Warn on stderr when the input's own frames beat
                       the re-encode (i.e. frame reuse fired), naming
                       the input's encoder when its metadata says.
                       Prints even with -q; incompatible with -R
  -q, --quiet          Suppress all progress output
  -t, --threads N      Limit parallel worker threads (default: all CPUs)
  -w, --windows <list> Comma-separated list of apodization windows to use
                       (default: all 26 with a bare -e, else
                       tukey005,tukey020,
                       tukey050,hann,welch,rect and the partial/punchout
                       tukey pair at .33/.67)
                       An entry of the form custom:<file> loads a window
                       shape from a knot file (up to 4 per run); see
                       bench/windows/example_taper.txt for the format
Available window names:
  rect, bartlett, bartletthann, blackman, blackmanharris, connes, flattop,
  gauss025, gauss0125, hamming, hann, kaiserbessel, nuttall, triangle, welch,
  tukey005, tukey010, tukey020, tukey050, tukey075, tukey090,
  partialtukey2, partialtukey2_033, partialtukey2_067,
  punchouttukey2_033, punchouttukey2_067
Experimental windows (never in a default set; explicit -w only):
  lanczos, bohman, parzen, plancktaper010, plancktaper025,
  partialtukey3_{1,2,3}, punchouttukey3_{1,2,3},
  partialtukey3h_{000,033,067}, punchouttukey3h_{025,050},
  punchouttukey2_000,
  expdecay{2,4}, expattack{2,4}, attackdecay{005,010,020},
  dpss{2,3,4}
```


If `[output.flac]` is omitted, it will default to `<input.flac>.optimized.flac`.

### Choosing a search level

All three levels are lossless — the decoded audio is bit-identical to the input
in every case. They differ only in how hard the encoder works to shrink the file.

Three knobs control the search:

- **`-e`** switches the block-partitioning DP from granule-based *estimates* to
  *exact* costs — every (position, block size) pair and every stereo mode is
  fully encoded before the DP chooses — and, at an unlimited candidate budget,
  widens the window set from the 10-window shortlist to all 26 standard windows
  (experimental windows such as `lanczos` stay opt-in via `-w`). Reachable from
  the effort dial as `-E 10` and up, which is where most of the available
  compression is.
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

The flags compose. The default is effort level 3 — `-c 24 -L 1 -a`, estimated
DP with a ranked search; bare `-e` implies `-c 0 -L 0` (exact DP, unlimited
sweep, adaptive windows off); `-e -c 8` prices blocks exactly but keeps the
subframe search bounded — what plain `-c 8` meant before the flags composed.
See `-E` and `-L` in the usage block above; `-c 8 -L 0 -A` reproduces the
older default.

The two tables below predate that default (they were measured when it was
plain `-c 8`) and predate the RICE2 residual coding, which shrank 24-bit
output by ~2% across all modes. Read them as the shape of the trade, not as
current absolute numbers.

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

### Adaptive window selection (`-a`, on by default)

Estimated-DP modes normally analyse a fixed shortlist. With `-a`, each block
*adds* to that shortlist from signal statistics the encoder has already
computed (energy dispersion, transient position, spectral tilt): transient
blocks bring in partial/punchout windows aimed at the energy peak, at
unchanged analysis cost. Measured across a 9-album corpus at `-c 0` it saved
0.03–0.34% per album with zero regressing tracks; at the default it is worth
about -0.019% at ~1.03x, which is the cheapest compression on offer here.

Every `-E` level turns it on, and level 3 is the default, so it is on unless
you pass `-A`. It applies to estimated-DP modes only, so it yields silently
to `-e` (which already offers every window) and to `-w` (which fixes the set
by hand) — worth knowing before measuring a shortlist change with `-w`, which
turns it off without saying so.

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
