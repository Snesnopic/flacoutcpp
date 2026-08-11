# GPU (`-G`) findings: two defects, measured on real hardware — both fixed

> **Status: fixed and verified.** Both defects below are repaired; `-G` output
> is now byte-identical to the CPU path on every fixture and mode tested, and
> deterministic across runs. The diagnosis is kept in full because the *shape*
> of this bug — a hardware invariant that holds on one vendor and not another,
> plus a sentinel that was arithmetic — is worth not re-deriving. See
> "The fix" at the end.


First test of the Vulkan backend on a discrete GPU (PR #3 was developed and
measured elsewhere). The backend **works** — it builds, enumerates a real
device, dispatches, and never corrupts audio — but two things in the current
`-G` are not what the flag's help text claims.

Summary, in the order they matter:

1. **`-G` output is non-deterministic.** Same input, same flags, different
   bytes each run. Cause is timing-dependent CPU/GPU work routing, isolated
   below.
2. **The GPU path picks far worse candidates** — up to **+4.5%** larger. The
   help text's "Bit-exact with the CPU path, so the output is byte-identical"
   is false. **Root cause found:** the kernel's `0xFFFFFFFF` error sentinel is
   never checked on the host, and `hdr + 6u + cost` wraps it to the *cheapest*
   cost representable, so a poisoned candidate always wins. Details below.
3. **Output is still lossless.** Every variant decodes to the input's exact
   audio md5 and passes `flac -t`, including all divergent runs. This is a
   search-correctness bug, not a data-corruption one — the winner is re-priced
   exactly before encoding, so the bad *cost* never reaches the bitstream.

Not measured: **the >2x speed claim.** See "What is still unmeasured".

## Test environment

| | |
|---|---|
| host | Debian forky/sid, kernel 6.19.11, glibc 2.42, Xeon E5-2698 v4 (40 threads) |
| GPU | Intel Arc A380 (DG2), `i915`, Mesa ANV, Vulkan 1.4.354 |
| device caps | subgroup 32, `shaderInt64` — meets the kernel's requirements |
| binary | cross-built for amd64 in Docker (debian bookworm, gcc 12.2, `libvulkan-dev` 1.3.239, `glslangValidator`), `-DFLACOUT_VULKAN=ON` |
| fixtures | `music_3s`, `music_10s`, `stereo_4s`, `master_250ms_mix`, `MLKDream` |

Staged at `/tmp/flacout-test/` on the test host: binary, `in/`, `out/`,
`remote_test.sh`.

### Getting a device to enumerate at all

Two environment traps cost time here and will cost it again:

- **The user must be in the `render` group.** `/dev/dri/renderD128` is
  `root:render`; membership in `video` is not enough. Without it ANV cannot
  open the node, drops out of enumeration entirely, and the only device left
  is llvmpipe — which `-G` then correctly rejects for its 8-lane subgroups.
  The failure looks exactly like "no driver installed". `sudo usermod -aG
  render $USER`, then a **fresh login**.
- **`TU: error: ... freedreno ... VK_ERROR_INCOMPATIBLE_DRIVER` is noise.**
  That is Turnip probing the same render node and failing. It appears even on
  a healthy system and is not the reason a device is missing.

Cross-build recipe (glibc 2.36 → runs on the host's 2.42, the safe direction;
SPIR-V is embedded in the executable, so no shader file travels with it):

```sh
docker run --rm --platform linux/amd64 \
  -v "$PWD":/src:ro -v /tmp/build:/build -v /tmp/ccache:/ccache \
  <image with cmake g++ libvulkan-dev glslang-tools ccache> bash -c '
    cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DFLACOUT_VULKAN=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    cmake --build /build -j$(nproc)'
```

`find_package(Vulkan)` reports `glslc missing`, which is fine — CMake falls
back to `glslangValidator` as intended.

## Finding 1: `-G` is non-deterministic

Three runs, same binary, same input, same flags:

```
-R -G  music_3s.flac      3e1d6d69...  bf711003...  5956df6a...   (3 distinct)
-R     music_3s.flac      799dcd72...  799dcd72...  799dcd72...   (CPU control)
```

The `-G` banner shows the same search config as the CPU run
(`24 candidates/subframe, patience 48`), so this is not flag composition.

**`--gpu-slots 1` does not fix it** — still three distinct md5s. The slot count
is not the variable.

**Single-threaded is deterministic**, and that is the isolating experiment:

| config | run 1 | run 2 | run 3 | bytes |
|---|---|---|---|---|
| `-R -t 1 -G --gpu-slots 1` | `c032fe00` | `c032fe00` | `c032fe00` | 500009 |
| `-R -t 1` (CPU) | `799dcd72` | `799dcd72` | — | 491936 |

So **the GPU kernel itself is deterministic**. The non-determinism comes from
*which candidates reach it*.

### Mechanism

`GpuEvaluator::evaluate` ([src/gpu.cpp:501-517](src/gpu.cpp#L501-L517)) decides
per batch whether the GPU takes the work:

- return false if `cands.size() < min_batch`,
- otherwise try to claim a slot by CAS on `Slot::busy`; **if no slot is free,
  return false and the caller does it on the CPU.**

That is a perfectly reasonable work-stealing design, and it is exactly what
makes the output vary: with 40 worker threads racing for 3 slots, whether any
given batch is priced by the GPU or by the CPU depends on wall-clock timing.
Confirmed by the GPU's own counter — the same file reports **31616**, **37824**
and **160000** candidates absorbed depending on thread count and timing.

Routing that varies is only *harmless if both paths agree exactly*. They do
not, which is Finding 2. The two defects multiply: fix either one and `-G`
becomes reproducible.

## Finding 2: the GPU's cost model is worse than the CPU's

With one thread and one slot the GPU absorbs all 160000 candidates, and the
result is deterministic and **+8073 bytes = +1.64%** against the CPU
(500009 vs 491936 on `music_3s`).

Multi-threaded, the penalty is a dilution of that number — it tracks how large
a share the GPU happened to win:

| fixture | default (`-R`) | `-R -e -c 8 -L 0` |
|---|---|---|
| master_250ms_mix | +0.217% | +0.023% |
| MLKDream | +0.304% | +0.032% |
| music_10s | +0.205% | +0.016% |
| music_3s | +0.486% | +0.019% |
| stereo_4s | **+1.288%** | +0.014% |

Two caveats on that table. The default-mode column compares **different
searches**: [main.cpp:502](src/main.cpp#L502) silently sets
`precision_rungs = 0` when `-G` is given without `-E`/`-L`, while the CPU arm
keeps effort level 3's `-L 1`. The gap is real anyway — CPU `-L 0` on
`music_3s` is 491916 against the GPU's 494329 — but for an apples-to-apples
comparison pass `-L 0` explicitly on both arms. Second, these were measured
multi-threaded, so each cell also mixes in however large a share the GPU
happened to win.

Held constant at `-t 1 --gpu-slots 1` (deterministic, GPU absorbs everything),
on `stereo_4s`:

| path | CPU | GPU | |
|---|---|---|---|
| `-e` (exhaustive, kernel priced alone) | 312860 | 326862 | **+4.48%** |
| `-c 24 -L 0` (ranked pre-pass) | 314422 | 323291 | **+2.82%** |

The exhaustive path — the simpler of the two, with no ranked replay, no
patience, no `reach` window — is the *worse* of the two. That is what moved
suspicion off the ranked integration and onto the kernel/host boundary.

The penalty is also **one-sided**: GPU output is never smaller, on any fixture,
in any mode, which points at candidates being mis-priced rather than at a
symmetric tie-break difference.

For scale: **the entire estimated-DP effort dial spans ~0.2% end to end.** A
+0.2-1.3% penalty means `-G` currently gives back more compression than every
`-E` level can buy. That has to be fixed before any speed number is worth
having.

### Ruled out: the `-L` ladder mismatch

The obvious hypothesis — the help text says `-G` prices the whole precision
ladder and "does not combine with `-L`", while the default `-E 3` sets
`-L 1`, so CPU and GPU would be searching different rung sets — **does not
explain the gap.** On `music_3s`, CPU-side:

```
-R        (i.e. -L 1)   491936
-R -L 0                 491916    (-20 bytes)
```

The whole ladder setting is worth 20 bytes. The GPU gap is 8073. Whatever the
GPU is doing differently, it is not the ladder.

### Root cause: the kernel's error sentinel wraps into the cheapest cost

**The host never checks the sentinel the shader documents.** The kernel's last
line ([shaders/sweep.comp:244](shaders/sweep.comp#L244)):

```glsl
outCost[cidx] = (gl_SubgroupSize == uint(SG)) ? best_total : 0xFFFFFFFFu;
```

Both host call sites then do, unguarded:

```cpp
const uint32_t cost = hdr + 6u + gcosts[i];   // optimizer.cpp:2786 (exhaustive)
const uint32_t tot  = hdr + 6u + costs[i];    // optimizer.cpp:3046 (ranked)
```

`hdr + 6u + 0xFFFFFFFFu` wraps in `uint32_t` to `hdr + 5`. **The "this candidate
is invalid" sentinel therefore becomes the cheapest cost representable**, so a
poisoned candidate always wins — and the one that wins is whichever has the
smallest header, i.e. the *lowest order at the lowest precision*.

Confirmed on the output. Winning LPC orders, `-e` on `stereo_4s`:

| | CPU | GPU |
|---|---|---|
| top orders | 32 (x13), 28-30 (x7) | **1 (x12)**, 21-24 (x10) |
| subframe types | 26 LPC, 0 FIXED | 21 LPC, **5 FIXED** |

An encoder choosing order 1 on content where order 32 wins is not a rounding
difference; it is the header-minimising choice the wraparound creates. It stays
lossless because the winner is re-priced exactly before encoding — the bad
choice is real, the bad *cost* never reaches the bitstream.

This also explains the otherwise baffling result that
**`--gpu-partition-cap 1`, `4` and `8` produce byte-identical output**
(323291 each): the partition-order search cannot matter when the value being
compared is a sentinel rather than a partition cost.

The one-sidedness, the non-determinism, the losslessness, and the cap
invariance all fall out of this single defect.

### What triggers the sentinel is not yet pinned

The sentinel fires when `gl_SubgroupSize != 32` at runtime. The host checks the
**device property** `subgroupSize` (that is what the banner's "subgroup 32"
reports) — but on Intel ANV the compiler picks SIMD8/16/32 per shader, and the
dispatched width is not pinned to that property unless
`VK_EXT_subgroup_size_control` with an explicit `requiredSubgroupSize` is
requested at pipeline creation. `src/gpu.cpp` never requests it. So the host is
validating a number that does not govern the dispatch.

Against that being the whole story: if *every* invocation were poisoned, every
LPC subframe would be order 1, and 10 of 21 are not. So either the width varies
by dispatch shape, or a second path produces the sentinel. **Dump the raw
`outCost` values for one subframe before concluding** — that single datum
settles it, and nothing else should be changed until it does.

Two fixes are needed regardless of the trigger, and they are independent:

1. **Treat `0xFFFFFFFF` as "no result" on the host**, at both call sites, and
   fall back to `eval_candidate` for that candidate. Saturating or checking
   before the add would also do. As written, the kernel's own error path is
   indistinguishable from a very cheap candidate.
2. **Pin the subgroup size at pipeline creation**, so the guard is a real
   invariant rather than a runtime lottery — or drop the requirement and
   handle any width.

### Ruled out by reading the code

- **fp32 residuals.** [gpu.cpp:551](src/gpu.cpp#L551) builds the push constants
  as `{ncand, bsize, 0, pcap}` — `flags` is hardcoded `0`, and `flags & 1` is
  the only selector for the fp32 path. The exact `int64_t` path
  ([sweep.comp:199-202](shaders/sweep.comp#L199-L202)) always runs. (An earlier
  version of this document named fp32 as the leading suspect. It was wrong.)
- **The kernel's integer arithmetic**, checked line by line against
  `calculate_rice_cost`: suffix-scan weights (`2^(2^st)` = 2, 4, 16, 256,
  65536), k ranges (0-14 method 0, 0-30 method 1), escape width
  (`hi>=1 ? hi+1 : 1` vs `findMSB(mabs)+2`, equivalent since `mabs = or_all>>1`),
  the escape guard (`hi <= 30` vs `mabs < 2^30`), per-partition parameter bits
  (`4<<l` / `5<<l`), warm-up exclusion, and the coarse-first `<=`/`<`
  tie-break. All agree.
- **The CPU's `any_high` gate** ([optimizer.cpp:2011](src/optimizer.cpp#L2011),
  [:2065](src/optimizer.cpp#L2065)), which the shader has no equivalent of, is
  a pure optimisation: with no residual reaching 2^15 the shader's `total1`
  still carries the 5-bit-vs-4-bit parameter penalty and cannot win.
- **Saturating arithmetic.** A lane whose `S_k` saturates costs >= 2^32 while
  the escape bound is <= ~524k, so a saturated lane is never the argmin.
- **The ranked pre-pass's missing dedup.** [optimizer.cpp:3024-3032](src/optimizer.cpp#L3024-L3032)
  omits the duplicate-predictor skip its exhaustive sibling has at
  [:2769-2773](src/optimizer.cpp#L2769-L2773), but a duplicate has identical
  coefficients at a higher precision, so it loses on header alone. Harmless.
- **Delta-chain corruption from mixing paths.** `eval_candidate`'s
  `prev_qc`/`prev_shift`/`have_pred` are declared inside the lambda
  ([optimizer.cpp:2435-2437](src/optimizer.cpp#L2435-L2437)), so they are
  per-call. Interleaving GPU- and CPU-priced candidates cannot corrupt them.
- **The `best_lpc_cost` pruning argument** the batching rests on
  ([optimizer.cpp:2984-2990](src/optimizer.cpp#L2984-L2990)) holds: a precision
  skipped by `hdr >= best_lpc_cost` has `cost >= hdr >= best_lpc_cost` and
  could never have won.

### Two stale comments found while reading

- [optimizer.cpp:2732](src/optimizer.cpp#L2732): "Only the exhaustive path
  offloads" — untrue since the ranked pre-pass at
  [:2999](src/optimizer.cpp#L2999) was added.
- [sweep.comp:161](shaders/sweep.comp#L161): "See `-U`" — no such flag exists
  in `main.cpp`.

## Finding 3: it is lossless

Every GPU output tested — including all three divergent runs and both bit
depths — passes `flac -t` and decodes to the input's exact audio md5:

```
music_3s   input   7b04109852bda76d3c4efe98555c7c44
           det_1   7b04109852bda76d3c4efe98555c7c44   ok
           det_2   7b04109852bda76d3c4efe98555c7c44   ok
           det_3   7b04109852bda76d3c4efe98555c7c44   ok
MLKDream   input   989a58fe428592d149f43925248a87e2
           gpu     989a58fe428592d149f43925248a87e2   ok
```

So the bug is confined to *which* candidate the search picks. Nothing
downstream of the choice is broken.

## What is still unmeasured

**The >2x speed claim in the PR description.** The test host was running
jellyfin (289% CPU), ffmpeg, immich, node and Sonarr, at load average 18-23 on
40 cores. Per trap 9 in CLAUDE.md every timing from that window is junk.
Sizes are deterministic and unaffected, which is why everything above stands.

Re-run the timing half on a quiet machine — and note it cannot be a plain
A/B against the CPU until Finding 2 is fixed, because the arms would be
compressing to different sizes. A speedup measured against a worse search is
not a speedup.

Also untested: any GPU other than this A380, and any driver other than Mesa
ANV. The subgroup-32 requirement means several otherwise-capable devices
(anything reporting 8- or 16-lane subgroups, including llvmpipe) decline the
GPU path entirely and fall back silently to the CPU — correct behaviour, but
it means "it ran" is not evidence the GPU did anything. Read the
`GPU: N candidates in Ts` line before believing a GPU run was a GPU run.

## The fix

Two independent changes, both in `src/gpu.cpp`; the shader is untouched.

**1. The sentinel is no longer arithmetic.** `GpuEvaluator::evaluate` scans the
returned costs and returns `false` if any is `UINT32_MAX`, so the caller prices
that batch on the CPU. One guard covers both call sites, because both already
treat `false` as "the GPU declined" — the exhaustive path leaves `gpu_done`
unset and runs its own sweep, the ranked pre-pass returns before setting
`gpu_ranked`. Rejecting the whole batch rather than individual entries is what
makes the fallback exact rather than merely less wrong. First occurrence warns
on stderr; the count is kept.

**2. The subgroup width is pinned instead of assumed.** Where
`VK_EXT_subgroup_size_control` is available and can deliver 32, the device is
created with `subgroupSizeControl` + `computeFullSubgroups`, and the pipeline
is built with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT`
(`requiredSubgroupSize = 32`) and
`VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT`. The old
`subgroupSize != 32` property check is kept only as the fallback for devices
without the extension. The banner now distinguishes the two:
`(subgroup 32 pinned, shaderInt64)` vs `(subgroup 32 by default, ...)`.

### Verification (Arc A380)

Every combination byte-identical, `-t 1 --gpu-slots 1` and multi-threaded:

| fixture | `-R -c 24 -L 0` | `-R -e -c 8 -L 0` |
|---|---|---|
| master_250ms_mix | 4241313 | 4220145 |
| MLKDream | 27848432 | 27698192 |
| music_10s | 1666958 | 1658133 |
| music_3s | 491916 | 488310 |
| stereo_4s | 314422 | 312996 |

The two paths that isolated the bug now agree exactly: `-e` on `stereo_4s`
326862 → **312860** (= CPU), ranked 323291 → **314422** (= CPU). Determinism:
three multi-threaded `-G` runs give one md5, equal to the CPU's
(`c559fa5512f8`), where they previously gave three. Losslessness re-checked by
decoding (`flac -t` ok, audio md5 matches input) on both bit depths.

The default (non-Vulkan) build is untouched — every change is inside
`#ifdef FLACOUT_HAVE_VULKAN`.

### Cost, and the open performance question

Pinning to SIMD32 **cost throughput on this device**: 1.5e5 → 5.27e4
candidates/s. That is not a regression against anything real — the shader's
width guard is on its *last* line, so the old dispatches ran the entire kernel
and then discarded it. The old figure was the same work with the answer thrown
away.

The likely mechanism is register pressure: SIMD32 doubles per-lane register
demand against SIMD16, and this kernel holds five `NLEV`-deep private arrays
plus `qc[32]`/`fc[32]`, so it plausibly spills to scratch. **Unmeasured** —
`INTEL_DEBUG=cs` prints the chosen width and spill counts.

If that is confirmed, the real Intel performance path is a kernel change, not a
flag: make the bit-plane mapping width-agnostic (two planes per lane at 16,
four at 8) so the driver can pick its preferred width *correctly*, instead of
being forced to one that spills. That is a design change and wants its own
measurement.

Apple is unaffected either way. If MoltenVK does not advertise the extension,
`size_ctl` stays false and the path is exactly as before; if it does, 32 is
pinned, which is what Metal does anyway.

## Suggested order of work

Items 1-3 are **done** (see "The fix"). Remaining:

1. **Measure speed**, on an idle machine, with `-R` — the PR's >2x claim is
   still unverified on any hardware but the author's M4, and the SIMD32 pinning
   makes the Intel number an open question rather than a known one.
2. **Check whether SIMD32 spills on Intel** (`INTEL_DEBUG=cs`), and if so
   consider a width-agnostic plane mapping.
3. Add a `bench/check.sh` case for `-G` if a GPU is available in the
   environment — the byte-identity claim is exactly what that harness exists to
   pin, and it would have caught both defects.
