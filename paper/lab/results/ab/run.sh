#!/usr/bin/env bash
# A/B: original PHast-R (base worktree) vs new, GxHash, pinned to core 2
V1=ref:plus:8:5.25,ref:w3:8:5.0,ref:phast:8:4.5,ref:plus:10:5.15,ref:w3:10:6.0,r:8:9:0:5.0,r:8:9:1:5.0,r:8:9:2:5.0,r:9:10:1:5.75,r:10:11:0:6.0,r:10:11:1:6.0,r:10:11:2:6.25,r:11:12:1:6.75
V2=ref:plus:8:5.25,ref:w3:8:5.0,ref:w3:10:6.0,r:8:9:1:5.0,r:8:9:2:5.0,r:10:11:1:6.0,r:10:11:2:6.25
for round in 1 2; do
  for w in base new; do
    if [ $w = base ]; then B=$HOME/git/sux-rs-phast-base/paper/lab/target-dev/release/cmp; else B=$HOME/git/sux-rs-phast/paper/lab/target-dev/release/cmp; fi
    echo "### $w round $round 1e7"
    RAYON_NUM_THREADS=1 taskset -c 2 $B -n 10000000 -q 10000000 -r 3 -v $V1 2>/dev/null
  done
done
for w in base new; do
  if [ $w = base ]; then B=$HOME/git/sux-rs-phast-base/paper/lab/target-dev/release/cmp; else B=$HOME/git/sux-rs-phast/paper/lab/target-dev/release/cmp; fi
  echo "### $w 1e8"
  RAYON_NUM_THREADS=1 taskset -c 2 $B -n 100000000 -q 10000000 -r 3 -v $V2 2>/dev/null
done
