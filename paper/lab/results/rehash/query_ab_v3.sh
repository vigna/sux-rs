#!/usr/bin/env bash
# Query A/B (u64 keys, GxHash): base (next_level mixing, d2a9cb35) vs V3
# (keys rehashed at each level, offsets o ^ h); both processes include the
# reference rows; single thread, pinned
S=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad
V=ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:1:4.75,r:8:10:2:4.75
for n in 100000 1000000 10000000 100000000; do
  K="0 1 2"; [ $n -ge 100000000 ] && K="0"
  for ks in $K; do
    for b in base v3; do
      if [ $b = base ]; then B=$S/base64/cmp; else B=$S/cmp-v3; fi
      RAYON_NUM_THREADS=1 taskset -c 2 $B -n $n -q 2000000 --interleave 11 --key-seed $ks -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$b,$ks,/"
    done
  done
done
