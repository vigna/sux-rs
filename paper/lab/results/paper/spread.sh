#!/usr/bin/env bash
# Space and fraction of bumped keys of the configurations of the tables over
# ten key sets of 10^7 keys (cmp --key-seed 1 to 10), single-threaded, one
# construction each: these values do not depend on the hardware, and their
# spread tells which differences in space are significant. About ten
# minutes on a c7i.metal-24xl.
# Usage: spread.sh <cmp binary> <output prefix>
# Environment: CPU1; see redo.sh. SPREAD_N (the number of keys, 10^7 by
# default) and SEEDS (the key seeds, 1 to 10 by default).
. "$(dirname "$0")/configs.sh"
B=${1:-../../target-nexus/release/cmp}
O=${2:-spread}
CPU1=${CPU1:-2}
SPREAD_N=${SPREAD_N:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
for s in $SEEDS; do
  pin $CPU1 env RAYON_NUM_THREADS=1 $B -n $SPREAD_N --key-seed $s -q 1000 --interleave 1 -v $T8,$T10 2>&1 >/dev/null \
    | grep '^CSV' | sed "s/^/$s,/"
done > $O.csv
echo done >> $O.csv
