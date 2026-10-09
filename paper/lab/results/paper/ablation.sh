#!/usr/bin/env bash
# Ablation of patterns: PHast-R with the default expected bucket size and 1,
# 2 or 4 patterns (A8 and A10 in configs.sh), with the same settings as the
# single-threaded runs of run.sh; the configurations of each seed width are
# built and queried by the same process, so their query times are
# interleaved. About twenty minutes on a c7i.metal-24xl.
# Usage: ablation.sh <cmp binary> <output prefix>
# Environment: CPU1, SIZES, BUILDS; see redo.sh. Only sizes up to 10^9 are
# used.
. "$(dirname "$0")/configs.sh"
B=${1:-../../target-nexus/release/cmp}
O=${2:-ablation}
CPU1=${CPU1:-2}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
{
for n in $SIZES; do
  [ $n -gt 1000000000 ] && continue
  for v in $A8 $A10; do
    pin $CPU1 env RAYON_NUM_THREADS=1 $B -n $n -q 2000000 --interleave 9 --order sequential \
      --builds $(builds $n) -v $v 2>&1 >/dev/null | grep '^CSV' | sed "s/^/1,/"
  done
done
} > $O.csv
echo done >> $O.csv
