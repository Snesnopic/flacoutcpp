#!/usr/bin/env bash
# Generate the audio fixtures used by check.sh and compare.sh.
#
# These are synthetic on purpose: they are reproducible from this script alone,
# so a reference recorded on one machine is meaningful on another. They are NOT
# representative of real music — see bench/README.md before drawing conclusions
# about compression ratio from them.
#
# Usage: bench/make_fixtures.sh [outdir]     (default: bench/fixtures)
set -euo pipefail

OUT=${1:-"$(cd "$(dirname "$0")" && pwd)/fixtures"}
mkdir -p "$OUT"

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }

# A tone-plus-noise mix. The noise floor keeps it from being trivially
# compressible, so the optimizer actually explores the search space.
SRC="aevalsrc=0.4*sin(2*PI*440*t)+0.2*sin(2*PI*1319*t)*sin(2*PI*3*t)+0.05*random(0)|0.4*sin(2*PI*443*t)+0.15*sin(2*PI*880*t)+0.05*random(1):s=44100:d=20"

gen() { # outfile, extra ffmpeg args...
  local out=$1; shift
  ffmpeg -y -loglevel error -f lavfi -i "$SRC" "$@" -c:a flac "$OUT/$out"
}

gen stereo_1s.flac  -t 1  -ac 2 -sample_fmt s16
gen stereo_4s.flac  -t 4  -ac 2 -sample_fmt s16
gen mono_2s.flac    -t 2  -ac 1 -sample_fmt s16
gen s24_2s.flac     -t 2  -ac 2 -sample_fmt s32 -bits_per_raw_sample 24
gen short.flac      -t 0.01 -ac 2 -sample_fmt s16   # < 1024 samples: short-stream path

ls -l "$OUT"
