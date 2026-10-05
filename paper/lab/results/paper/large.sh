#!/usr/bin/env bash
# Analysis of large key sets (Section 4.1 of the paper): query time split
# between first-level and bumped keys, thread scaling, peak memory, and the
# effect of transparent huge pages (through the glibc tunable
# glibc.malloc.hugetlb, which makes malloc ask for them).
# Usage: large.sh <directory of the binaries> <output directory>
# With 10^9 keys the peak memory usage is about 60 GB.
B=${1:-../../target-nexus/release}
O=${2:-large}
mkdir -p $O

# Query time of first-level and bumped keys
for n in 10000000 100000000 1000000000; do
  RAYON_NUM_THREADS=1 taskset -c 2 $B/qsplit -n $n -q 4000000 -r 9 \
    -v ref:w3:8:5.0,ref:phast:8:4.5,8:10:4.5,8:10:4.75 | sed "s/^/$n /"
done > $O/qsplit.txt

# Thread scaling of the construction (CPUs 0-7 are distinct cores)
for n in 100000000 1000000000; do
  for t in 1 2 4 8 16; do
    if [ $t = 16 ]; then cpus=0-15; else cpus=0-$((t - 1)); fi
    RAYON_NUM_THREADS=$t taskset -c $cpus $B/btime -n $n -r 2 -v 8:10:4.5 | sed "s/^/$n $t PHast-R /"
    RAYON_NUM_THREADS=$t taskset -c $cpus $B/cmp -n $n -t $t -q 1000 --interleave 1 -v ref:w3:8:5.0 2>&1 >/dev/null \
      | grep '^CSV' | sed "s/^/$n $t /"
  done
done > $O/scaling.txt

# Peak memory of a construction with 8 threads (it includes 8 bytes per key
# for the keys)
for n in 100000000 1000000000; do
  RAYON_NUM_THREADS=8 taskset -c 0-7 /usr/bin/time -v $B/btime -n $n -r 1 -v 8:10:4.5 2>&1 \
    | grep -E "bits|Maximum resident" | sed "s/^/$n PHast-R /"
  for v in ref:w3:8:5.0 ref:plus:8:5.25; do
    RAYON_NUM_THREADS=8 taskset -c 0-7 /usr/bin/time -v $B/cmp -n $n -t 8 -q 1000 --interleave 1 -v $v 2>&1 \
      | grep -E "^CSV|Maximum resident" | sed "s/^/$n $v /"
  done
done > $O/memory.txt

# Transparent huge pages
for n in 100000000 1000000000; do
  GLIBC_TUNABLES=glibc.malloc.hugetlb=1 RAYON_NUM_THREADS=1 taskset -c 2 $B/cmp -n $n -q 2000000 --interleave 9 \
    -v ref:plus:8:5.25,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:4.5,r:8:10:4.75 2>&1 >/dev/null | grep '^CSV' | sed "s/^/1,/"
  GLIBC_TUNABLES=glibc.malloc.hugetlb=1 RAYON_NUM_THREADS=8 taskset -c 0-7 $B/cmp -n $n -t 8 -q 1000 --interleave 1 \
    -v ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:4.5,r:8:10:4.75 2>&1 >/dev/null | grep '^CSV' | sed "s/^/8,/"
done > $O/hugepages.csv
echo done >> $O/hugepages.csv
