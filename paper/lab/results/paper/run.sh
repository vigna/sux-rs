#!/usr/bin/env bash
# Experiments of the paper: PHast, PHast+ (with and without wrapping) and
# PHast-R on u64 keys with GxHash.
#
# Single-threaded runs, pinned to a core: for each number of keys and each
# seed width, one process builds the sweeps of the expected bucket size of
# configs.sh (which contain the configurations of the tables), and times
# queries in 9 interleaved rounds of 2*10^6 sequential queries (medians,
# with minimum and maximum). Then the configurations of the tables are built
# with several threads (construction time only). Construction times are
# medians of BUILDS constructions (by default 3 up to 10^8 keys and 1 above).
# Above 10^9 keys only the configurations of the tables are built.
#
# On a c7i.metal-24xl this takes about four hours and a half, two of them
# spent building PHast with 10-bit seeds on 10^9 keys; the peak memory usage
# on 10^9 keys is about 30 GB.
# Usage: run.sh <cmp binary> <output prefix>
# Environment: CPU1 (the core for single-threaded runs), CPUS (the cores
# for multithreaded runs, as a taskset list), THREADS (their number), SIZES
# (the numbers of keys), BUILDS; see redo.sh.
. "$(dirname "$0")/configs.sh"
B=${1:-../../target-nexus/release/cmp}
O=${2:-run}
CPU1=${CPU1:-2}
CPUS=${CPUS:-0-7}
THREADS=${THREADS:-8}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
{
for n in $SIZES; do
  if [ $n -le 1000000000 ]; then V="$G8 $G10"; else V="$T8 $T10"; fi
  for v in $V; do
    pin $CPU1 env RAYON_NUM_THREADS=1 $B -n $n -q 2000000 --interleave 9 --order sequential \
      --builds $(builds $n) -v $v 2>&1 >/dev/null | grep '^CSV' | sed "s/^/1,/"
  done
done
for n in $SIZES; do
  pin $CPUS env RAYON_NUM_THREADS=$THREADS $B -n $n -t $THREADS -q 1000 --interleave 1 --order sequential \
    --builds $(builds $n) -v $T8,$T10 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$THREADS,/"
done
} > $O.csv
echo done >> $O.csv
