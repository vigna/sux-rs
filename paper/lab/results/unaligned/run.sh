#!/usr/bin/env bash
# Seeds of 10 and 12 bits in a BitFieldVec: aligned reads (bfv), unaligned
# reads via TryIntoUnaligned (bfvu), and the previous private 32-bit unaligned
# read ("old"); u64 keys, GxHash, single thread, pinned, with reference rows
S=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad
R=ref:plus:8:5.25,ref:w3:10:6.0
for n in 1000000 10000000 100000000; do
  K="0 1 2"; [ $n -ge 100000000 ] && K="0"
  for ks in $K; do
    RAYON_NUM_THREADS=1 taskset -c 2 $S/cmp-unal -n $n -q 2000000 --interleave 11 --key-seed $ks \
      -v $R,r:10:11:1:6.0:2:bfv,r:10:11:1:6.0:2:bfvu,r:12:12:1:7.0:2:bfv,r:12:12:1:7.0:2:bfvu 2>&1 >/dev/null | grep '^CSV' | sed "s/^/new,$ks,/"
    RAYON_NUM_THREADS=1 taskset -c 2 $S/cmp-inl -n $n -q 2000000 --interleave 11 --key-seed $ks \
      -v $R,r:10:11:1:6.0:2:bfv,r:12:12:1:7.0:2:bfv 2>&1 >/dev/null | grep '^CSV' | sed "s/^/old,$ks,/"
  done
done
