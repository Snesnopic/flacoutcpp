#!/usr/bin/env bash
# Extract the zipped albums in albums/ into bench/fixtures/albums/<album>/.
#
# Unlike make_fixtures.sh these are real music, which is the only thing worth
# judging a compression-vs-speed tradeoff on (see bench/README.md and trap 2 in
# CLAUDE.md). They are big and not reproducible from this repo, so extraction is
# idempotent: an album already fully extracted is skipped, and a partial one is
# finished rather than redone.
#
# Usage: bench/extract_albums.sh [srcdir] [outdir]
#        (defaults: albums/  bench/fixtures/albums/)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-"$ROOT/albums"}
OUT=${2:-"$ROOT/bench/fixtures/albums"}

# macOS ships an Info-ZIP unzip that mangles UTF-8 entry names even when the
# archive sets the UTF-8 flag bit (C418's "Oxygène" comes out as "Oxyg+?ne"),
# and it has no -O CHARSET to override. ditto gets it right, so prefer it.
if command -v ditto >/dev/null; then
  unpack() { ditto -x -k "$1" "$2"; }
elif command -v unzip >/dev/null; then
  unpack() { unzip -q -o "$1" -d "$2"; }
else
  echo "error: neither ditto nor unzip found" >&2; exit 1
fi

[ -d "$SRC" ] || { echo "error: no such directory: $SRC" >&2; exit 1; }

mkdir -p "$OUT"

shopt -s nullglob
zips=("$SRC"/*.zip)
(( ${#zips[@]} )) || { echo "no .zip files in $SRC"; exit 0; }

extracted=0 skipped=0 repaired=0

for zip in "${zips[@]}"; do
  album=$(basename "$zip" .zip)
  dest="$OUT/$album"
  stamp="$dest/.extracted"

  # The stamp records the zip's size+mtime, so replacing a zip re-extracts it.
  sig=$(stat -f '%z %m' "$zip" 2>/dev/null || stat -c '%s %Y' "$zip")

  if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$sig" ]; then
    echo "skip    $album (already extracted)"
    skipped=$((skipped + 1))
    continue
  fi

  # No stamp, or a stale one. Re-extract the whole album rather than trusting
  # whatever a killed run left behind — the last file it wrote is likely
  # truncated, and a truncated FLAC in the bench set is worse than the time.
  # The stamp is only written after a clean unpack, so this stays resumable at
  # album granularity.
  if [ -d "$dest" ]; then
    echo "redo    $album (no valid stamp)"
    repaired=$((repaired + 1))
  else
    echo "extract $album"
    extracted=$((extracted + 1))
  fi

  mkdir -p "$dest"
  unpack "$zip" "$dest"
  printf '%s' "$sig" > "$stamp"
done

echo
echo "$extracted extracted, $repaired resumed, $skipped skipped -> $OUT"
