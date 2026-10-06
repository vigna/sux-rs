#!/usr/bin/env bash
# Analysis of large key sets (Section 4.1 of the paper): query time split
# between first-level and bumped keys, thread scaling, and peak memory.
# Usage: large.sh <directory of the binaries> <output directory>
# Environment: CPU1, CORES (a list of distinct cores), THREADS, SIZES; see
# redo.sh. With 10^9 keys the peak memory usage is about 60 GB.
B=${1:-../../target-nexus/release}
O=${2:-large}
CPU1=${CPU1:-2}
CORES=${CORES:-0 1 2 3 4 5 6 7}
THREADS=${THREADS:-8}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
LARGE=$(echo $SIZES | tr ' ' '\n' | grep -v '^10000000$' | tr '\n' ' ')
# The first t cores, as a taskset list
cpus() { echo $CORES | tr ' ' '\n' | head -n $1 | tr '\n' ',' | sed 's/,$//'; }
mkdir -p $O

# Query time of first-level and bumped keys
for n in $SIZES; do
  RAYON_NUM_THREADS=1 taskset -c $CPU1 $B/qsplit -n $n -q 4000000 -r 9 \
    -v ref:w3:8:5.0,ref:phast:8:4.5,8:10:4.5,8:10:4.75 | sed "s/^/$n /"
done > $O/qsplit.txt

# Thread scaling of the construction: powers of two up to the number of
# cores, and then all hardware threads
scale() {
  RAYON_NUM_THREADS=$1 taskset -c $2 $B/btime -n $n -r 2 -v 8:10:4.5 | sed "s/^/$n $1 PHast-R /"
  RAYON_NUM_THREADS=$1 taskset -c $2 $B/cmp -n $n -t $1 -q 1000 --interleave 1 -v ref:w3:8:5.0 2>&1 >/dev/null \
    | grep '^CSV' | sed "s/^/$n $1 /"
}
for n in $LARGE; do
  for t in 1 2 4 8 16 32 64; do
    [ $t -gt $THREADS ] && break
    scale $t $(cpus $t)
  done
  [ $(nproc) -gt $THREADS ] && scale $(nproc) 0-$(($(nproc) - 1))
done > $O/scaling.txt

# Peak memory of a multithreaded construction (it includes 8 bytes per key
# for the keys)
CPUS=$(cpus $THREADS)
for n in $LARGE; do
  RAYON_NUM_THREADS=$THREADS taskset -c $CPUS /usr/bin/time -v $B/btime -n $n -r 1 -v 8:10:4.5 2>&1 \
    | grep -E "bits|Maximum resident" | sed "s/^/$n PHast-R /"
  for v in ref:w3:8:5.0 ref:plus:8:5.25; do
    RAYON_NUM_THREADS=$THREADS taskset -c $CPUS /usr/bin/time -v $B/cmp -n $n -t $THREADS -q 1000 --interleave 1 -v $v 2>&1 \
      | grep -E "^CSV|Maximum resident" | sed "s/^/$n $v /"
  done
done > $O/memory.txt
