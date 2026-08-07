# bench/ — profiling and regression tooling

Profiling and regression tooling for the encoder's hot paths.

## Quick start

```sh
bench/make_fixtures.sh                      # needs ffmpeg; writes bench/fixtures/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
bench/check.sh record build                 # snapshot the current bitstream
#   ... make a change ...
cmake --build build -j && bench/check.sh verify build
```

## check.sh — bit-exactness gate

Optimizations to the encoder's hot loops are meant to leave the bitstream
untouched, so the regression test is just `cmp`. Fourteen cases cover exhaustive,
heuristic and ranked (`-c`) mode, mono/stereo, 16- and 24-bit, the
`< 1024`-sample short-stream path, and an explicit `-w` window list. Outputs are
additionally checked with `flac -t`, so a change that corrupts the reference and
the candidate the same way still fails.

`record` refuses to snapshot output that does not decode.

The `rk_*` (ranked) cases are a slightly different kind of test. Ranked search
deliberately produces a different bitstream than `-e` — that is the whole point
of it — so `cmp` there is not asserting "unchanged since forever", only "unchanged
since the reference was recorded". Retuning the ranking is *expected* to move
those files, and the right response is to re-record after checking the size
delta went the way you intended. A reference including them can only be recorded
from the commit that added `-c` or later.

### Why the reference is portable at all

The optimizer derives LPC coefficients from a double-precision autocorrelation
and quantizes them to 8-15 bit integers. A last-bit difference in that sum
usually quantizes to the same integer — but occasionally lands one step over,
which changes the residuals, the cost, and therefore which candidate wins. The
encoded bitstream is that sensitive to floating-point rounding.

That is why `flacout_lib` is built with `-ffp-contract=off` (see the comment in
`CMakeLists.txt`). Without it, whether the compiler fuses `a += x * y` into a
single-rounding FMA leaks into the output: clang at `-O3` and the same clang
with contraction disabled disagreed on 7 of the 14 cases here. With it pinned,
`-O0`, `-O1`, `-O2`, `-O3` and `FLACOUT_NATIVE=ON` all produce identical files.

So if a `verify` fails only on some builds, suspect the FP settings before
suspecting the change. And be aware the guarantee is only as strong as that flag
— a compiler that ignores it, or a different libm for the window functions,
could still diverge.

`cmp` also cannot tell you whether the encoder is still lossless, only whether it
is consistent. For that, decode and compare against the source:

```sh
flac -d -c -s input.flac | md5sum
flac -d -c -s output.flac | md5sum   # must match
```

## compare.sh — A/B timing

```sh
git worktree add /tmp/base <known-good-rev>
cmake -S /tmp/base -B /tmp/base-build -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/base-build -j
bench/compare.sh /tmp/base-build build
```

Runs the two builds alternately and reports best-of-N. **Interleave and take the
best, don't average.** This workload pins every core; on a laptop, clocks sag
over a few minutes of it. Timing all of A and then all of B charges that drift to
whichever ran second — while writing these patches that was worth about 15%,
enough on its own to manufacture or mask a result. An early "10.70 → 8.10s"
measurement here was really 9.12 → 8.10 once interleaved.

## Instrumentation counters

`src/optimizer.cpp` carries a counter block behind `FLACOUT_INSTRUMENT`. It
compiles to nothing when the macro is undefined (verified: byte-identical
output), so it costs the normal build nothing.

```sh
cmake -S . -B build-instr -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DFLACOUT_INSTRUMENT"
cmake --build build-instr -j
build-instr/flacoutcpp -q -e bench/fixtures/stereo_1s.flac /tmp/out.flac
```

A report goes to stderr at exit: search-space sizes, how many candidates each
pruning bound actually eliminates, multiply-accumulate counts for the two hot
loops, and histograms of which order / precision / window ultimately *wins*.

That last group is the one to look at before trying to shrink the search space:
it shows the best window varies per subframe with no clear favourite, so a fixed
`-w` shortlist is the wrong shape for that. Ranking candidates per subframe —
the Levinson-Durbin prediction error is already computed and discarded — is.

Counters are `relaxed` atomics on shared cache lines. They perturb timing
noticeably, so profile with them **off** and count with them **on**; never read
both off the same run.

## Profiling

macOS, whole-process including all worker threads:

```sh
build/flacoutcpp -q -e bench/fixtures/stereo_4s.flac /tmp/o.flac & \
  sample $! 20 -f /tmp/prof.txt; wait
```

Linux:

```sh
perf record -g --call-graph dwarf -- build/flacoutcpp -q -e bench/fixtures/stereo_4s.flac /tmp/o.flac
perf report --sort symbol
```

## About the fixtures

They are synthetic tone-plus-noise, reproducible from `make_fixtures.sh`, which
is what makes a recorded reference portable between machines. They are **not**
representative of music. Their flat noise floor pushes an unusual share of
subframes to LPC order 32, which inflates how valuable high orders look. That is
fine for the bit-exactness gate and fine for relative timing, but any change that
trades compression for speed has to be re-measured on real music before its
constants are chosen.

### Using real music

Drop a track into `fixtures/` and cut excerpts named `music_3s.flac` and
`music_20s.flac`:

```sh
ffmpeg -ss 60 -t 3  -i track.flac -c:a flac bench/fixtures/music_3s.flac
ffmpeg -ss 60 -t 20 -i track.flac -c:a flac bench/fixtures/music_20s.flac
BENCH_SET=music bench/compare.sh <build-a> <build-b>
```

`fixtures/` is gitignored, so nothing you put there gets committed — which is
also why these are not checked in.

It is worth knowing how differently real content behaves. On the synthetic
fixture, LPC order 32 wins 238 of 757 subframes; on real music the distribution
peaks hard at order 6 and 32 still wins 280 of ~2500. Predictions about which
orders matter, made from the synthetic fixture alone, were wrong in both
directions.
