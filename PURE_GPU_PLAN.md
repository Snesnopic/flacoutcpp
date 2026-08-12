# Pure-GPU FLAC encoding: what the transformation actually is

Scope: raw PCM goes into device memory, an encoded FLAC byte stream comes out,
and the host does no per-candidate, per-subframe or per-frame work in between.
**Goal is GPU-residency, not optimality** — a pure-GPU encoder that compresses
slightly worse than `-e` is the target; one that needs a host round trip per
subframe is not, however well it compresses.

Read `GPU_PLAN.md` first for what `-G` already is and where it broke. This
document is about the other 40% of the pipeline.

## The one invariant that matters, and the constraint that dissolves

Everything in the encoder upstream of *"here are the integer LPC coefficients"*
is a **heuristic**. The window, the autocorrelation, Levinson-Durbin, the
coefficient quantization — none of it has to be reproducible, or even
numerically good, for the output to be a valid lossless FLAC. It only has to
produce `int32_t q_coeffs[32]` and a shift.

Everything downstream is **exact integer arithmetic that the decoder mirrors**:

```
residual[i] = s[i] - (int32_t)((sum_j qc[j] * s[i-1-j]) >> shift)
```

Get that right and the file decodes bit-exactly, whatever nonsense chose `qc`.

This is the load-bearing observation for a pure-GPU port, because it kills the
constraint that otherwise ends the project on the author's hardware:

- `apply_window`, `autocorrelation`, `compute_lpc_all_orders` and
  `quantize_lpc_coeffs` are all `double` on the CPU, and the whole project is
  built `-ffp-contract=off` because a last-bit FMA difference flips which
  candidate wins.
- **Metal has no `double` type at all.** On Apple silicon via MoltenVK,
  `shaderFloat64` is not coming. On discrete parts fp64 exists at 1/16–1/64
  of fp32 rate.

Under a bit-exactness requirement that is fatal. Under "GPU-residency, not
optimality" it is a non-issue: run those four stages in **fp32**, accept that a
different candidate sometimes wins, and the output is still lossless. What it
costs is compression, and that cost is separately measurable — the existing
`FLACOUT_DUMP_FP32RANK` result already says it is small in the place it was
measured (the exact winner ranked 0 in the fp32 Rice ordering on 97.5–100% of
58.3M candidates; keeping the top 4 and refining exactly cost **zero bits** on
every fixture including real music).

Corollary: **`bench/check.sh` cannot gate this.** `-G` is gateable today
precisely because it is bit-exact with the CPU. A pure-GPU path is a different
encoder, and its regression net has to be "decodes to the input's md5" plus a
size table, not `cmp`.

## Stage table

Fixed-block configuration (one block size, no variable-block DP) unless noted.
"New" = does not exist in any form today.

| # | stage | GPU shape | status |
|---|---|---|---|
| 0 | de-interleave + sign-extend raw PCM to planar `int32` | 1 lane/sample, pure gather | new, trivial |
| 1 | wasted-bit detect (`ctz` of OR-reduce per subframe) | workgroup reduce | new, trivial |
| 2 | stereo decorrelation → 4 signals (L, R, M, S) | 1 lane/sample × 4 | new, trivial |
| 3 | window × autocorrelation, lags 0..32 | workgroup per (block, signal, window); 33 accumulators, subgroup-reduced | new, easy |
| 4 | Levinson-Durbin, all orders | **one subgroup per solve, lane j holds a[j]** | new, delicate |
| 5 | `lpc_log2cmax` + quantize with error feedback | 1 lane per (order, precision); 32 sequential steps | new, easy but sequential |
| 6 | residual + Rice cost per candidate | `shaders/sweep.comp` | **exists, bit-exact** |
| 7 | argmin over candidates; Constant/Verbatim/Fixed(0..4) alternatives | reduction + 6 cheap kernels | partly exists (CPU) |
| 8 | stereo-mode choice per block (min over 4 pairings) | trivial reduce | new, trivial |
| 9 | materialize the winner's residuals | 1 lane/sample, exact int64 | new, trivial |
| 10 | bit-length prefix scan → absolute bit offset per code | hierarchical scan | **new, the real work** |
| 11 | scatter Rice codes by `atomicOr` | 1 lane/residual | **new, the real work** |
| 12 | frame headers, subframe headers, coefficients | 1 lane/frame | new, fiddly |
| 13 | CRC-8 (header), CRC-16 (frame) | 1 lane/frame, or combine-reduce | new, easy |
| 14 | frame byte-length scan → final file offsets | scan over frames | new, trivial |
| 15 | MD5 of interleaved PCM | **cannot be parallelized** | see below |

Stages 0–2, 7–9, 14 are bandwidth-bound trivia. Stages 3–5 are the fp64
question above, now answered. Stage 6 exists. **Stages 10–13 are the genuinely
new engineering**, and 15 is the one honest exception to purity.

## Why this is worth doing: the host round trip is the binding constraint

Not "the GPU is faster". The current `-G` is at best parity on a discrete part
(`TODO.md`), and the reason is structural rather than a tuning miss:

- `GpuEvaluator::Candidate` is 34 `int32_t` = **136 bytes per candidate**, and
  the host uploads one per candidate priced. A full sweep is 26 windows × 32
  orders × 8 precisions = **6656 candidates per subframe**; the `FP32RANK` run
  logged 58.3M candidates over 8845 subframes of a few seconds of audio, i.e.
  **7.9 GB of candidate upload** for that. The shader's own header estimates
  ~10⁹ candidates for a 3-minute track under `-e`: ~136 GB.
- The host must **quantize every candidate before it can offer it**, and
  `eval_candidate` (quantize + delta residual update) is **33% of `-e`
  runtime**. So the CPU cannot get out of the way even in principle: it is
  doing a third of the work just to feed the device.
- `would_accept()` exists to stop that batch-build cost being wasted, and duty
  cycling makes dispatch nondeterministic (trap 11).

Moving stages 3–5 onto the device replaces 136 bytes/candidate with **~132
bytes per (block, signal, window)** — one autocorrelation — and the candidate
explosion happens entirely in device memory. That is the whole point. The
speed follows from deleting the traffic, not from the arithmetic.

Second-order but real: it also makes the GPU path *simpler*. No slots, no duty
cycle, no `min_batch`, no CPU fallback path to keep in sync with a shader. One
command buffer per chunk, barriers between stages, one readback.

## Stage 4 in detail: Levinson without an 8 KB private array

The CPU version keeps `double ld_a[33][33]` (~8 KB) on the stack. A private
array of that size per GPU lane is not a spill, it is a rout — `sweep.comp`'s
own notes record that a bare `int qc[32]` cost a quarter of the 128-register
budget at SIMD32 and had to be removed.

Only two rows of `ld_a` are ever live (order-1 and order). The decomposition
that fits the hardware is **one subgroup per solve, with lane `j` holding
`a[j]`** — order ≤ 32 and subgroup = 32 is not a coincidence worth wasting:

- `lambda = autoc[ord] - sum_j a[j]*autoc[ord-j]` → one `subgroupAdd`.
- the reversal update `a'[j] = a[j] - k*a[ord-j]` → one
  `subgroupShuffle` (index `ord-j`) plus an FMA.
- ~32 sequential steps of ~3 subgroup ops each: ~100 ops per solve, **zero
  private arrays, zero shared memory**.

Parallelism comes from the batch, not from within a solve: a 3-minute track at
4096-sample blocks gives ~1938 blocks × 4 signals × N windows solves, which is
5k–50k independent subgroups. Occupancy is fine.

The instability early-out (`|k| >= 1` → zero the remaining orders) is
subgroup-uniform, so it stays a clean branch. `out_err[]` per order — the
ranking quantity — falls out for free and can drive a device-side candidate
shortlist, which is how `-c`/`-p` survive the port.

## Stages 10–13 in detail: the bitstream is the interesting problem

This is standard GPU variable-length entropy coding (the shape GPU JPEG and ANS
encoders use), and FLAC's framing is unusually kind to it. Two facts from
`frame_writer.cpp` set the whole design:

1. **`bw.align()` before CRC-16 means every frame is a whole number of bytes.**
   Frames are therefore *independently* placeable: a scan over frame byte
   lengths gives each frame's final file offset, and packing can write straight
   into the final output buffer. One device→host copy produces the file.
2. **Within a frame, subframe boundaries are bit-level** — no padding between
   them. So inside a frame you need genuine bit-granular packing.

The two-pass shape:

**Pass A (lengths).** Each Rice code is `(u >> k) + 1 + k` bits; escape
partitions are fixed `escape_bps` each. `sweep.comp` already computes exactly
these sums per partition (`total0`/`total1`) — the winner's totals are a
by-product, not new work. Hierarchically scan: residual within partition →
partition within subframe → subframe within frame → frame within file. Add
header, coefficient and warm-up widths (closed-form from `SubframeParams`) and
the frame-end pad. Each residual now knows its absolute bit offset; each frame
knows its byte offset.

**Pass B (scatter).** One lane per residual: build the code in a `uint64_t`,
shift into position, `atomicOr` the one or two `uint32_t` words it touches.
Correctness does not depend on ordering — distinct codes occupy **disjoint bit
ranges**, and OR over disjoint ranges is commutative. The output buffer must be
zeroed first, which it is anyway.

The one wart: a long unary run (`u >> k` can be hundreds of bits on a badly
chosen `k`) spans many words. Handle it as a zero-fill plus a terminating `1`
bit — the zeros need no write at all into a zeroed buffer, so a long code is
*one* `atomicOr` for the stop bit and one for the remainder. Long codes get
cheaper, not more expensive.

Headers (stage 12) are ~10 bytes per frame plus ~2–60 bytes of coefficients;
one lane per frame writing bytes serially is correct and invisible in the
profile. `write_utf8` and `blocksize_code` port verbatim.

**CRC (stage 13).** CRC-8 covers the header (≤ 16 bytes, one lane, done).
CRC-16 covers header+payload, so it runs after pass B. One lane per frame with
a slice-by-4 table is a few thousand iterations over a few thousand frames —
adequate. If it ever matters, CRC is linear: `CRC(A||B)` combines from
`CRC(A)`, `CRC(B)` and `|B|`, so it is a subgroup reduce over chunks with a
precomputed shift basis. Do the simple thing first.

## Stage 15: MD5 is the one place purity must give

MD5 chains 64-byte blocks. It is not parallelizable, at all, and a single GPU
lane grinding ~500k serial rounds for a 3-minute track would be the slowest
thing in the pipeline by orders of magnitude.

Three options, and the middle one is the answer:

1. **One GPU lane.** ~500k sequential block compressions for a 3-minute track,
   each a few hundred instructions on a serial dependency chain: **~0.25 s
   estimated**, i.e. ~6x slower than the host pass, while occupying a lane and
   blocking the pipeline instead of overlapping it. Don't.
2. **Keep it on the host, overlapped.** It is a single pass over the input PCM
   at memory bandwidth (~40 ms for a 3-minute 16/44.1 stereo track) and it has
   no dependency on anything the GPU does. Run it on a host thread while the
   device encodes; wall-clock cost is zero. The host still does *no per-frame
   work* — the purity claim survives intact.
3. **Write zeros.** All-zero MD5 in STREAMINFO is legal and means "unknown";
   `flac -t` then reports the signature as unavailable and skips the check.
   Free, and strictly worse for users. Offer it as a flag at most.

Take option 2. State in the docs that MD5 is host-side by design and why.

## Memory: this forces chunking

Whole-stream residency does not scale, and the existing measurements say so
directly (`CLAUDE.md`, Memory footprint: a 74-minute single-track rip projects
to ~5.5 GB under `-e` on the CPU already). For a device buffer set:

| buffer | bytes/sample/channel | 3-min stereo | 74-min stereo |
|---|---|---|---|
| planar PCM | 4 | 63 MB | 1.5 GB |
| 4 decorrelated signals | 8 | 127 MB | 3.1 GB |
| winner residuals | 4 | 63 MB | 1.5 GB |
| packed output | ~2 | ~32 MB | ~0.8 GB |

So: **chunk the stream into ~10–30 s segments** and pipeline them. With a fixed
block size this is exactly free (frames are independent). With variable blocks
it costs one partition decision per chunk boundary, which is negligible if
chunk boundaries are placed on the block grid. Chunking also gets you
compute/transfer overlap for nothing.

## What about the variable-block DP?

It is a shortest path over ~7700 nodes × 5 candidates and runs in
*microseconds* — `O(N×K)` sequential. Two honest choices:

- Read the cost table back (a few tens of KB) and run the DP on the host. Costs
  one small readback per chunk and no per-frame host work. Not a purity
  violation in any sense that matters.
- Or run it as a single-workgroup sequential kernel, 5 lanes wide, ~7700
  launch-latency-bound steps. Doable, pointless, but it makes the "no host in
  the loop" claim literal.

Do the first. Revisit only if a profile says the readback synchronization
stalls the pipeline.

## Cost model: predicted, then measured

**Built and measured (M4 Max, 2026-08-11).** The stage-share prediction held
exactly; the throughput estimate did not.

`FLACOUT_PG_PROFILE=1`, master mix (8.3M samples/channel, stereo 16-bit),
serialised submits so the shares are readable:

| stage | share |
|---|---|
| **sweep (6)** | **92.4%** |
| crc (12) | 2.4% |
| autoc (2) | 1.5% |
| prepare (1) | 1.4% |
| rice (8) | 0.8% |
| pack (11) | 0.3% |
| levinson (3), quant (4), fixed (5), select (7), layout (9), frame (10) | 0.2% each |

So the architectural claim above is confirmed: **every new stage is a rounding
error and the pre-existing sweep kernel is the whole cost.** The bit packer —
the part flagged as the real engineering risk — is 0.3%.

**The ~71 ms estimate was ~14x optimistic, and the reason is worth recording.**
It came from this repo's own "1835 GMAC/s exact-integer" figure
(`FLACOUT_DUMP_FP32RANK`'s commit message). That number is not what this kernel
achieves. Measured on the same device:

| path | rate |
|---|---|
| `-G` (existing kernel, `-e -c 0 -L 0`, music_10s) | 3303.9 GMAC in 25.11 s = **131.6 GMAC/s** |
| `-P` sweep (master mix) | ~70 GMAC in 0.665 s = **105 GMAC/s** |

The two agree to within a factor of 1.25, so the pure-GPU sweep is running at
the kernel's real speed — the 1835 figure must have been a microbenchmark of the
residual loop alone, without the partition search. **Do not use it to size
anything.** ~110-130 GMAC/s is the number.

That reframes the whole exercise: at 32 orders x 4 windows the sweep is a
*deeper* search than `flac -8` performs, and a 16-core NEON CPU with
hand-vectorised residual and Rice loops is genuinely competitive with this
device on it. Speed has to come from pricing fewer candidates, not from the
device being faster per candidate.

### The order shortlist is the lever (`--pg-orders`)

Levinson already produces the per-order prediction error, so ranking orders and
sweeping only the best few costs one extra store and a 32-iteration loop in
`pg_quant` — no extra dispatch, no compaction. Skipped candidates are marked and
the sweep bails before its first subgroup op.

Master mix, against all 32 orders:

| orders | bytes | device s | speedup | size |
|---|---|---|---|---|
| 32 | 16954903 | 0.677 | 1.00x | — |
| 16 | 16955999 | 0.389 | 1.74x | +0.006% |
| **8** | **16959643** | **0.245** | **2.77x** | **+0.028%** |
| 4 | 16969221 | 0.163 | 4.15x | +0.084% |
| 2 | 16984374 | 0.121 | 5.60x | +0.174% |
| 1 | 17005700 | 0.103 | 6.58x | +0.300% |

8 is the default. This is the same shape as the CPU's `-c` frontier and for the
same reason.

### Attacking the sweep: it is not ALU-bound, and two "obvious" fixes prove it

The sweep started at 92.4% of device time, so it is the only stage worth
optimising. Three changes were tried. **The order they are listed in is the
order of increasing payoff, and it is the reverse of the order they look
promising in.**

1. **Skip empty bit-planes** (bit-exact). The kernel ballotted all 32 planes per
   32-sample chunk; the highest plane any lane populates is
   `findMSB(subgroupMax(u))`, and skipping above it is exactly equivalent
   (lane j owns plane j, so a lane above the top keeps `myBallot == 0`, and a
   zero `B_j` contributes nothing to the scan, the `bitCount`, or `Hcur`).
   Real residuals occupy ~10-12 planes, not 32. **Worth 2.7%.**

2. **Narrow the dot-product accumulator to int32** where
   `sum|qc| <= INT32_MAX >> (effBps-1)` — the same bound the CPU uses to pick
   `lpc_residual_blocked<int32_t>`. Metal has no 64-bit integer ALU, so int64 on
   Apple silicon is synthesised from 32-bit ops, and this looked like the big
   one. Verified exact (four fixtures byte-identical). **Worth zero.**

3. **Cap the sweep's partition-order search** (`--pg-pcap`). **Worth 3.4x.**

Two null results in a row are the finding, not a disappointment: **this kernel
is bound by cross-lane operations, not by arithmetic.** At order 8 a 32-sample
chunk costs ~16 arithmetic ops for the residual against ~80 subgroup ops --
~12 plane ballots plus two `closePartition` calls, each a 5-step shuffle scan
with two `subgroupMin`s and its own ballots. The dot product is not the cost.

That is also why the `FP32RANK` route (sweep in fp32, exactly refine the top K)
is much less attractive than it looks: fp32 would speed up the arithmetic the
kernel is not spending its time on, and leave every ballot and shuffle in place.
It was worth reaching for when the sweep was believed to be an int64 MAC loop.

The partition cap works because closes are `2^(P+1)-1` and the kernel is made of
them -- 511 at P=8, 31 at P=4, 7 at P=2. Master mix:

| pcap | bytes | sweep s | speedup | size |
|---|---|---|---|---|
| 8 | 16959643 | 0.1948 | 1.00x | — |
| 6 | 16961874 | 0.0832 | 2.34x | +0.013% |
| **4** | **16964581** | **0.0575** | **3.39x** | **+0.029%** |
| 3 | 16965880 | 0.0522 | 3.73x | +0.037% |
| 2 | 16969536 | 0.0508 | 3.83x | +0.058% |
| 1 | 16978990 | 0.0492 | 3.96x | +0.114% |

Sizes barely move because **the cap is a ranking approximation only** — stage 8
re-prices the winner with the full P=8 search, so the cap can change which
candidate wins but never what the bitstream says a partition costs. Identical
argument to `--gpu-partition-cap`, and here it is worth far more, because
nothing else in this kernel is the bottleneck. Default 4.

After both, the profile has flattened and the sweep is no longer the whole cost
(master mix, serialised submits, so small stages carry ~1.2 ms of submit
overhead each):

| stage | share |
|---|---|
| sweep | 55.0% |
| **crc** | **15.3%** |
| autoc | 8.2% |
| prepare | 7.6% |
| rice | 5.1% |
| everything else | ~1% each |

**CRC-16 is the new second bottleneck**, and it is the naive implementation on
purpose (one lane per frame, byte at a time). The combine trick the stage note
describes — CRC is linear over GF(2), so `CRC(A||B)` folds from its parts — is
now worth building. So is a look at `prepare`, which reads the PCM twice.

### End-to-end, against the CPU

Interleaved best-of-5, quiet machine, `-n` on every arm. Times include decode,
MD5 and file I/O, which are common fixed costs (decode alone is 0.109 s on the
master mix, so the device is the bulk of `-P`).

| fixture | `-P` | CPU default | CPU `-E 0` | size vs CPU default |
|---|---|---|---|---|
| music_10s | 0.075 s | 0.136 s (1.81x) | 0.118 s (1.57x) | +0.40% |
| music_20s | 0.084 s | 0.195 s (2.33x) | 0.163 s (1.95x) | +0.14% |
| MLKDream | 0.367 s | 1.461 s (**3.98x**) | 1.158 s (3.16x) | +1.12% |
| master mix | 0.265 s | 0.981 s (3.70x) | 0.809 s (3.05x) | +0.99% |
| syn3m_mix | 0.272 s | 0.930 s (3.41x) | 0.792 s (2.91x) | **+7.62%** |

Note what the short fixtures are now bounded by: libFLAC's single-threaded
decode is 0.109 s of the master mix's 0.265 s. **The pipeline is approaching
decode-bound**, which is the point at which further kernel work stops showing up
in wall clock and the next move is to overlap decode with the encode (frames are
independent, so chunk N can be encoding while chunk N+1 decodes).

Losslessness verified by decode on all 18 fixtures (`flac -t` ok and decoded
audio MD5 equal to the input's), including 24-bit, mono, and the `<1024`-sample
short-stream path.

### The one real defect left: fp32 makes Levinson declare itself unstable

This is where the remaining size gap lives, and it is not search depth. Held at
an *identical* search configuration (fixed 4096 blocks, same four windows, all
orders — `-R -b 4096 -e -c 0 -L 0 -w tukey050,hann,welch,rect`):

| fixture | CPU, same config | `-P` | gap |
|---|---|---|---|
| music_3s | 482282 | 486300 | +0.83% |
| stereo_4s | 310923 | 349350 | +12.4% |
| s24_2s | 284608 | 348745 | +22.5% |

Diagnosed with `FLACOUT_PG_DEBUG=1`, which dumps the first solve's analysis
chain against a double-precision host reference. The autocorrelation is fine
(relative error ~1e-7 at every lag) and the coefficients are sensible, so the
analysis is not "numerically mushy". The failure is discrete: on `s24_2s`'s side
channel **every candidate of order >= 7 was invalid**, because the fp32
recursion produced `|PARCOR| >= 1` at order 7 and the recursion then abandons
that order and every order above it. The CPU, in double, picks exactly order 7
there. One subframe cost 46649 bits against the CPU's ~23600.

The fix applied is a per-solve retry with a white-noise ridge on lag 0. Applying
a ridge *globally* is a bad trade — it also biases the solves that were fine:

| ridge | s24_2s | stereo_4s | music_3s | music_10s |
|---|---|---|---|---|
| 0 | 348745 | 349350 | 486300 | 1667289 |
| 1e-6 | 338721 | 329645 | **492044** | **1680691** |
| 1e-5 | 351810 | 353673 | 503636 | 1715166 |

So it retries only solves that broke, filling in just the orders the clean pass
never reached. Real music never retries and is **byte-identical** with the retry
on or off; `stereo_4s` gained 5.7% and `s24_2s` 3.0%.

**That is a mitigation, not a cure, and the residual gap is still large on
synthetic content** (`syn3m_mix` +7.6%). The cure is to stop the fp32
autocorrelation being non-PSD in the first place: accumulate it in **emulated
double** (two-float hi/lo with FMA-based two-product), which is ~5-10x the cost
of a stage that is currently **1.5%** of runtime. That is the next thing to
build, and it is cheap precisely because the plan's stage-share prediction was
right.

## Build order

Each step ends somewhere testable. Do not skip to step 4.

1. **Bit-packer first, against CPU-chosen parameters.** Take the existing
   encoder's `BlockParams` output, upload it, and have the GPU do stages 9–14.
   Verify byte-identical against `FrameWriter`. This is the highest-risk
   component and it is testable in complete isolation, with `cmp` as the oracle.
2. **Stages 0–2 + 5** (de-interleave, wasted bits, decorrelation, quantize).
   Verify the quantized coefficients match the CPU's to the last bit — they
   will, if quantize is done in fp32 *and* the CPU comparison arm is also fp32;
   otherwise verify by size delta only.
3. **Stages 3–4** (window/autocorr/Levinson in fp32). Output now diverges from
   the CPU. Switch the oracle from `cmp` to decode-md5 plus a size table over
   the album corpus. Expect a small loss; measure it and write it down.
4. **Fuse into one command buffer per chunk, delete the host round trip.**
   This is where the win is; steps 1–3 will each look like a wash or worse in
   isolation.
5. **Then** consider re-adding search depth (device-side ranking from
   `out_err`, the top-K exact refine that `FP32RANK` measured as free, the
   variable-block DP).

## Traps specific to this work

- **`bench/check.sh` is not the net here** (see above). Build the losslessness +
  size-table harness *before* step 3, not after it diverges.
- **Every timing claim needs trap 9's hygiene** (`pgrep -x flacoutcpp`, steady
  load), and per trap 11 a `-G`-style path needs more reps than a CPU one.
- **A GPU run that fell back is indistinguishable from a GPU run.** `-G`
  already had to learn this — read the counter, not the exit code.
- **fp32 divergence is content-dependent.** `FP32RANK` found 24-bit real music
  the worst case (more sample bits eat more mantissa) and 16-bit real music
  perfect. Judge the fp32 loss on 24-bit content, and per trap 2 never on
  synthetics.
- **`FRAME_OVERHEAD` does not exist any more.** Anything the packer computes
  about frame size must call the same logic `FrameWriter::frame_bits` does, or
  the two drift and the cost model contract breaks silently.