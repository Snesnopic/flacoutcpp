#!/usr/bin/env bash
# Bit-exactness harness for optimizations that must not change the bitstream.
#
#   bench/check.sh record <build-dir>    # snapshot outputs as the reference
#   bench/check.sh verify <build-dir>    # diff a new build against that snapshot
#
# `record` against a known-good build, change the encoder, then `verify`. Any
# byte difference is a failure: this gates changes meant to be purely a speedup,
# so `cmp` is the whole test. Outputs are also run through `flac -t`, so a change
# that corrupts both reference and candidate identically still gets caught.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODE=${1:-}
BUILD=${2:-}
FIX=${FIXTURES:-"$HERE/fixtures"}

[ -n "$MODE" ] && [ -n "$BUILD" ] || { echo "usage: $0 {record|verify} <build-dir>" >&2; exit 2; }
BIN="$BUILD/flacoutcpp"
[ -x "$BIN" ] || { echo "error: no executable at $BIN" >&2; exit 2; }
[ -d "$FIX" ] || { echo "error: no fixtures at $FIX — run bench/make_fixtures.sh" >&2; exit 2; }

case "$MODE" in
  record) OUT="$HERE/reference" ;;
  verify) OUT="$HERE/.current" ;;
  *) echo "usage: $0 {record|verify} <build-dir>" >&2; exit 2 ;;
esac
rm -rf "$OUT"; mkdir -p "$OUT"

# Each case pins a distinct path through the encoder: exhaustive vs heuristic,
# stereo vs mono, 16- vs 24-bit, the <1024-sample short-stream path, and an
# explicit -w list (which selects a different window set than either default).
run() { local name=$1; shift
  "$BIN" -q "$@" "$OUT/$name.flac" || { echo "FAIL: encoder returned nonzero for $name" >&2; exit 1; }
}
run ex_stereo -e "$FIX/stereo_1s.flac"
run ex_mono   -e "$FIX/mono_2s.flac"
run ex_24     -e "$FIX/s24_2s.flac"
run ex_short  -e "$FIX/short.flac"
run ex_win    -e -w hann,tukey020,punchouttukey2_067 "$FIX/stereo_1s.flac"
run he_stereo    "$FIX/stereo_4s.flac"
run he_mono      "$FIX/mono_2s.flac"
run he_24        "$FIX/s24_2s.flac"
run he_short     "$FIX/short.flac"
# Ranked exact search (-e -c N; plain -c N before the flags composed). Its
# output is a deliberate compression/speed trade, so it is not comparable to
# bare -e — but it must still be stable and decode losslessly. A reference for
# these can only be recorded from the commit that introduced ranked search or
# later; older builds reject the flag and `record` will fail.
run rk_stereo -e -c 8 "$FIX/stereo_1s.flac"
run rk_mono   -e -c 4 "$FIX/mono_2s.flac"
run rk_24     -e -c 8 "$FIX/s24_2s.flac"
run rk_short  -e -c 8 "$FIX/short.flac"
run rk_win    -e -c 2 -w hann,tukey020 "$FIX/stereo_1s.flac"

fail=0
for f in "$OUT"/*.flac; do
  flac -t -s "$f" 2>/dev/null || { echo "  INVALID  $(basename "$f") — does not decode"; fail=1; }
done

if [ "$MODE" = record ]; then
  [ $fail -eq 0 ] || { echo "refusing to record a reference that does not decode"; exit 1; }
  echo "recorded $(ls "$OUT" | wc -l | tr -d ' ') reference outputs in $OUT"
  exit 0
fi

REF="$HERE/reference"
[ -d "$REF" ] || { echo "error: no reference — run '$0 record <build-dir>' first" >&2; exit 2; }
for f in "$REF"/*.flac; do
  n=$(basename "$f")
  if cmp -s "$f" "$OUT/$n"; then
    echo "  ok        $n"
  else
    echo "  MISMATCH  $n  ($(wc -c <"$f" | tr -d ' ') -> $(wc -c <"$OUT/$n" | tr -d ' ') bytes)"
    fail=1
  fi
done

if [ $fail -eq 0 ]; then echo "ALL BIT-EXACT"; else echo "FAILURES — output changed"; fi
exit $fail
