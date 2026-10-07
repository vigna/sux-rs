#!/usr/bin/env bash
# Analysis of large key sets (Section 4.1 of the paper): query time split
# between first-level and bumped keys, thread scaling of the construction,
# and peak memory.
# Usage: large.sh <directory of the binaries> <output directory>
# Environment: CPU1, CORES (a list of distinct cores), THREADS, SIZES,
# BUILDS; see redo.sh. On a c7i.metal-24xl this takes about an hour, most of
# it spent building PHast; with 10^9 keys the peak memory usage is about
# 60 GB, with 10^10 keys about 250 GB.
. "$(dirname "$0")/configs.sh"
B=${1:-../../target-nexus/release}
O=${2:-large}
CPU1=${CPU1:-2}
CORES=${CORES:-0 1 2 3 4 5 6 7}
THREADS=${THREADS:-8}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
LARGE=$(echo $SIZES | tr ' ' '\n' | grep -v '^10000000$' | tr '\n' ' ')
# The first t cores, as a taskset list
cpus() { echo $CORES | tr ' ' '\n' | head -n $1 | tr '\n' ',' | sed 's/,$//'; }
# Runs a command and prints its peak memory usage as GNU time does (on
# macOS, converting the output of BSD time)
peak() {
  if /usr/bin/time -v true >/dev/null 2>&1; then
    /usr/bin/time -v "$@" 2>&1
  else
    /usr/bin/time -l "$@" 2>&1 | awk '/maximum resident set size/ { printf "\tMaximum resident set size (kbytes): %d\n", $1 / 1024; next } { print }'
  fi
}
mkdir -p $O

# Query time of first-level and bumped keys
for n in $SIZES; do
  pin $CPU1 env RAYON_NUM_THREADS=1 $B/qsplit -n $n -q 4000000 -r 9 --order sequential \
    -v ref:w3:8:5.0,ref:phast:8:4.5,8:10:4.5,8:10:4.75 | sed "s/^/$n /"
done > $O/qsplit.txt

# Thread scaling of the construction: powers of two up to the number of
# cores, and then all hardware threads
S8=ref:plus:8:5.25,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:4.5
scale() {
  pin $2 env RAYON_NUM_THREADS=$1 $B/cmp -n $n -t $1 -q 1000 --interleave 1 --builds $(builds $n) -v $S8 2>&1 >/dev/null \
    | grep '^CSV' | sed "s/^/$n $1 /"
}
NPROC=$(nthreads)
for n in $LARGE; do
  for t in 1 2 4 8 16 32 64; do
    [ $t -gt $THREADS ] && break
    scale $t $(cpus $t)
  done
  [ $NPROC -gt $THREADS ] && scale $NPROC 0-$(($NPROC - 1))
done > $O/scaling.txt

# Peak memory of a multithreaded construction (it includes 8 bytes per key
# for the keys); PHast-R is built by btime, as cmp checks that the function
# is a bijection using n more bytes
CPUS=$(cpus $THREADS)
for n in $LARGE; do
  peak $(pinning $CPUS) env RAYON_NUM_THREADS=$THREADS $B/btime -n $n -r 1 -v 8:10:4.5 \
    | grep -E "bits|Maximum resident" | sed "s/^/$n PHast-R /"
  for v in ref:w3:8:5.0 ref:plus:8:5.25 ref:phast:8:4.5; do
    peak $(pinning $CPUS) env RAYON_NUM_THREADS=$THREADS $B/cmp -n $n -t $THREADS -q 1000 --interleave 1 -v $v \
      | grep -E "^CSV|Maximum resident" | sed "s/^/$n $v /"
  done
done > $O/memory.txt
