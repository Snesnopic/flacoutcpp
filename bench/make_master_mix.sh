#!/usr/bin/env bash
# Build a single "master mix" fixture: 1 second from the center of every
# track under bench/fixtures/albums/, concatenated in path order. One file
# that samples the whole corpus — useful for quick cross-genre A/Bs without
# sweeping 188 tracks.
#
# All segments are normalized to 44.1 kHz stereo 16-bit (the corpus is
# entirely 44.1 kHz; mono tracks are upmixed, the few 24-bit tracks are
# dithered down) so the concat is a single uniform stream. This is a
# *statistics* fixture, not a lossless-provenance one.
#
# Usage: bench/make_master_mix.sh [outfile]
#        (default: bench/fixtures/master_1s_mix.flac)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ALBUMS="$HERE/fixtures/albums"
OUT=${1:-"$HERE/fixtures/master_1s_mix.flac"}

command -v ffmpeg  >/dev/null || { echo "error: ffmpeg not found"  >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "error: ffprobe not found" >&2; exit 1; }
[ -d "$ALBUMS" ] || { echo "error: no album collection at $ALBUMS" >&2; exit 1; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

i=0
find "$ALBUMS" -name "*.flac" | sort | while IFS= read -r f; do
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  # Center-second start; clamp to 0 for tracks shorter than 1 s.
  ss=$(awk -v d="$dur" 'BEGIN { s = d/2 - 0.5; print (s < 0) ? 0 : s }')
  seg=$(printf '%s/%04d.wav' "$TMP" "$i")
  ffmpeg -y -loglevel error -ss "$ss" -t 1 -i "$f" \
         -ac 2 -ar 44100 -sample_fmt s16 "$seg"
  printf "file '%s'\n" "$seg" >> "$TMP/list.txt"
  i=$((i+1))
done

n=$(wc -l < "$TMP/list.txt" | tr -d ' ')
[ "$n" -gt 0 ] || { echo "error: no tracks found" >&2; exit 1; }

ffmpeg -y -loglevel error -f concat -safe 0 -i "$TMP/list.txt" \
       -ac 2 -ar 44100 -sample_fmt s16 -c:a flac "$OUT"

flac -t -s "$OUT"
echo "wrote $OUT ($n segments, $(metaflac --show-total-samples "$OUT") samples)"
