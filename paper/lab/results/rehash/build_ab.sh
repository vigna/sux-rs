#!/usr/bin/env bash
# Construction A/B: base (next_level mixing, d2a9cb35) vs rehash of the keys
for round in 1 2 3; do
  for n in 1000000 10000000 100000000; do
    for th in 1 8; do
      if [ $th = 1 ]; then T="taskset -c 2"; else T="taskset -c 0-7"; fi
      for b in base rehash; do
        if [ $b = base ]; then B=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad/base64/btime; else B=/tmp/claude-703800006/-home-vigna-git-sux-rs/3af12a88-5bb5-4698-abe2-5d4c74c9a13b/scratchpad/btime-rehash; fi
        RAYON_NUM_THREADS=$th $T $B -n $n -r 3 -v 8:10:1:4.75,8:10:2:4.75 | sed "s/^/$round $n $th $b /"
      done
    done
  done
done
