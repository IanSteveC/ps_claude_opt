#!/bin/bash
# Run the 10-input gt_2000 suite against a given binary; one run dir per input.
# Usage: ./run_suite.sh <binary> <tag> [inputs...]
set -e
BIN="$(readlink -f "$1")"; TAG="$2"; shift 2
SUITE="${SUITE:-/home/ian/Asteroids Test cases/gt_2000}"
RUNS="${RUNS:-$(dirname "$BIN")/runs/$TAG}"
INPUTS="${@:-239_1 248_1 251_1 253_1 260_1 273_1 276_1 279_1 319_1 320_1}"
mkdir -p "$RUNS"
for i in $INPUTS; do
  d="$RUNS/$i"; mkdir -p "$d"
  cp "$SUITE/input_$i" "$d/period_search_in"
  ( cd "$d"
    /usr/bin/time -f "%e" -o wall.txt "$BIN" --device 0 > stdout.txt 2> stderr.txt || echo "EXIT $? for $i" >&2
  )
  printf "%-8s %ss\n" "$i" "$(cat "$d/wall.txt")"
done
