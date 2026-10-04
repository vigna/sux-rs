#!/usr/bin/env bash
# Query A/B (u64 keys, GxHash): base (next_level mixing) vs rehash; both
# processes include the reference rows; single thread, pinned
V=ref:plus:8:5.25,ref:w3:8:5.0,r:8:10:1:4.75,r:8:10:2:4.75
for n in 100000 1000000 10000000 100000000; do
  K="0 1 2"; [ $n -ge 100000000 ] && K="0"
  for ks in $K; do
    for b in base rehash; do
      if [ $b = base ]; then B=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad/base64/cmp; else B=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad/cmp-rehash; fi
      RAYON_NUM_THREADS=1 taskset -c 2 $B -n $n -q 2000000 --interleave 11 --key-seed $ks -v $V 2>&1 >/dev/null | grep '^CSV' | sed "s/^/$b,$ks,/"
    done
  done
done
