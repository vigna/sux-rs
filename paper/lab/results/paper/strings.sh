#!/usr/bin/env bash
# String keys: the 8-bit configurations of STR8 in configs.sh on random
# strings of 10 to 50 bytes (cmpstr), single-threaded, with 9 interleaved
# rounds of 2*10^6 sequential queries, as in run.sh (one construction each).
# Output lines: 1,CSV,<name>,<keys>,<bits/key>,<build ns/key>,<query ns>.
# About ten minutes on a c7i.metal-24xl.
# Usage: strings.sh <cmpstr binary> <output prefix>
# Environment: CPU1, SIZES; see redo.sh. Only sizes up to 10^8 are used (the
# strings take about 70 bytes per key).
. "$(dirname "$0")/configs.sh"
B=${1:-../../target-nexus/release/cmpstr}
O=${2:-strings}
CPU1=${CPU1:-2}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
{
for n in $SIZES; do
  [ $n -gt 100000000 ] && continue
  pin $CPU1 env RAYON_NUM_THREADS=1 $B -n $n -q 2000000 --interleave 9 --order sequential \
    -v $STR8 2>&1 >/dev/null | grep '^CSV' | sed "s/^/1,/"
done
} > $O.csv
echo done >> $O.csv
