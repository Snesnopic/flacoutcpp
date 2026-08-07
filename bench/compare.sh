#!/usr/bin/env bash
# Interleaved A/B timing for two builds.
#
#   bench/compare.sh <build-a> <build-b> [reps]
#
# Runs A and B alternately and reports best-of-N for each. Interleaving matters:
# this workload saturates every core, so a laptop's clocks drop measurably over
# a few minutes of it. Timing all of A and then all of B attributes that drift
# to whichever ran second — during this project that was worth ~15%, enough to
# invent or hide a speedup. Best-of-N rather than mean, because noise on a
# shared machine is one-sided.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
A=${1:-}; B=${2:-}; REPS=${3:-3}
FIX=${FIXTURES:-"$HERE/fixtures"}
[ -n "$A" ] && [ -n "$B" ] || { echo "usage: $0 <build-a> <build-b> [reps]" >&2; exit 2; }
for d in "$A" "$B"; do [ -x "$d/flacoutcpp" ] || { echo "error: no executable at $d/flacoutcpp" >&2; exit 2; }; done

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

time_one() { # binary, args...  -> seconds on stdout
  # `time -p` reports on stderr, so only stdout can be discarded here. Take the
  # last "real" line so anything the encoder prints to stderr cannot shadow it.
  local bin=$1; shift
  { /usr/bin/time -p "$bin" "$@" >/dev/null; } 2>&1 | awk '/^real/{t=$2} END{print t+0}'
}

# Default to the ~0.25 s micro fixtures so an A/B is seconds, not minutes.
# BENCH_SET=full for the longer ones — worth doing before believing any result,
# and required for scheduling changes, whose tail effects need enough DP nodes
# to show up at all.
case "${BENCH_SET:-quick}" in
  quick) CASES=("-e micro" "-e micro_s24") ;;
  full)  CASES=("-e stereo_1s" "-e mono_2s" "-e s24_2s" "stereo_4s") ;;
  *)     echo "error: BENCH_SET must be 'quick' or 'full'" >&2; exit 2 ;;
esac

printf '%-16s %10s %10s %8s   %s\n' CASE "$(basename "$A")" "$(basename "$B")" SPEEDUP OUTPUT
for case in "${CASES[@]}"; do
  set -- $case
  if [ "$1" = "-e" ]; then flags=-e; fixture=$2; else flags=""; fixture=$1; fi
  [ -f "$FIX/$fixture.flac" ] || { echo "  skip $fixture (missing fixture)"; continue; }

  ta=999; tb=999
  for _ in $(seq "$REPS"); do
    t=$(time_one "$A/flacoutcpp" -q $flags "$FIX/$fixture.flac" "$TMP/a.flac")
    ta=$(awk -v x="$ta" -v y="$t" 'BEGIN{print (y<x)?y:x}')
    t=$(time_one "$B/flacoutcpp" -q $flags "$FIX/$fixture.flac" "$TMP/b.flac")
    tb=$(awk -v x="$tb" -v y="$t" 'BEGIN{print (y<x)?y:x}')
  done
  cmp -s "$TMP/a.flac" "$TMP/b.flac" && same=identical || same="DIFFERS ($(wc -c <"$TMP/a.flac"|tr -d ' ') -> $(wc -c <"$TMP/b.flac"|tr -d ' '))"
  awk -v n="${flags:+-e }$fixture" -v a="$ta" -v b="$tb" -v s="$same" \
      'BEGIN{printf "%-16s %9.2fs %9.2fs %7.2fx   %s\n", n, a, b, a/b, s}'
done
