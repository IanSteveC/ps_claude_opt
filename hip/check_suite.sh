#!/bin/bash
# Validate a suite run dir against (a) the CPU/reference outputs and, if given,
# (b) another run dir for byte-identity.
# Usage: ./check_suite.sh <runs/tag> [runs/othertag]
SUITE="${SUITE:-/home/ian/Asteroids Test cases/gt_2000}"
RUN="$1"; OTHER="$2"
fail=0
for d in "$RUN"/*/; do
  i=$(basename "$d")
  ref="$SUITE/period_search_out_${i}_fma"
  [ -f "$ref" ] || ref="$SUITE/period_search_out_${i}_cuda"
  echo "== $i (vs $(basename "$ref")) =="
  python3 "$(dirname "$0")/compare_out.py" "$d/period_search_out" "$ref" | head -4
  if [ -n "$OTHER" ]; then
    if cmp -s "$d/period_search_out" "$OTHER/$i/period_search_out"; then
      echo "   byte-identical to $OTHER"
    else
      echo "   *** DIFFERS from $OTHER ***"
      fail=1
    fi
  fi
done
exit $fail
