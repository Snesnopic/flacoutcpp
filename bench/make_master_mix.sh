#!/usr/bin/env bash
# Build a single "master mix" fixture: a slice from the center of every track
# under bench/fixtures/albums/, concatenated in path order. One file that
# samples the whole corpus — useful for quick cross-genre A/Bs without
# sweeping 188 tracks.
#
# The slice length is the knob. 1 s is the original and stays the default;
# shorter slices keep the same cross-genre shape at proportionally less
# runtime, which matters because the expensive search modes scale with total
# samples. 0.25 s gives a ~47 s fixture that still touches all 188 tracks.
# Every segment is still taken from the track center, so a shorter mix is a
# subset of the same material rather than different material.
#
# All segments are normalized to 44.1 kHz stereo 16-bit (the corpus is
# entirely 44.1 kHz; mono tracks are upmixed, the few 24-bit tracks are
# dithered down) so the concat is a single uniform stream. This is a
# *statistics* fixture, not a lossless-provenance one.
#
# Usage: bench/make_master_mix.sh [-d SECONDS] [outfile]
#        -d defaults to 1; the output name defaults to
#        bench/fixtures/master_<label>_mix.flac, where <label> is 1s, 250ms, …
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ALBUMS="$HERE/fixtures/albums"

SEG=1
if [ "${1:-}" = "-d" ]; then
  [ $# -ge 2 ] || { echo "error: -d requires a duration in seconds" >&2; exit 2; }
  SEG=$2; shift 2
fi
awk -v s="$SEG" 'BEGIN { exit !(s > 0) }' \
  || { echo "error: -d must be a positive number, got '$SEG'" >&2; exit 2; }

# 1 -> "1s", 0.25 -> "250ms": keeps master_1s_mix.flac's name unchanged while
# giving sub-second slices a name without a dot in it.
LABEL=$(awk -v s="$SEG" 'BEGIN {
  if (s >= 1) { if (s == int(s)) printf "%ds", s; else printf "%gs", s }
  else printf "%gms", s * 1000
}')
OUT=${1:-"$HERE/fixtures/master_${LABEL}_mix.flac"}

command -v ffmpeg  >/dev/null || { echo "error: ffmpeg not found"  >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "error: ffprobe not found" >&2; exit 1; }
[ -d "$ALBUMS" ] || { echo "error: no album collection at $ALBUMS" >&2; exit 1; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

i=0
find "$ALBUMS" -name "*.flac" | sort | while IFS= read -r f; do
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  # Center the slice; clamp to 0 for tracks shorter than the slice itself.
  ss=$(awk -v d="$dur" -v seg="$SEG" 'BEGIN { s = d/2 - seg/2; print (s < 0) ? 0 : s }')
  seg=$(printf '%s/%04d.wav' "$TMP" "$i")
  ffmpeg -y -loglevel error -ss "$ss" -t "$SEG" -i "$f" \
         -ac 2 -ar 44100 -sample_fmt s16 "$seg"
  printf "file '%s'\n" "$seg" >> "$TMP/list.txt"
  i=$((i+1))
done

n=$(wc -l < "$TMP/list.txt" | tr -d ' ')
[ "$n" -gt 0 ] || { echo "error: no tracks found" >&2; exit 1; }

ffmpeg -y -loglevel error -f concat -safe 0 -i "$TMP/list.txt" \
       -ac 2 -ar 44100 -sample_fmt s16 -c:a flac "$OUT"

flac -t -s "$OUT"
samples=$(metaflac --show-total-samples "$OUT")
echo "wrote $OUT ($n segments of ${SEG}s, $samples samples, \
$(awk -v s="$samples" 'BEGIN { printf "%.1f", s / 44100 }')s)"
