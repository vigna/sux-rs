#!/usr/bin/env bash
# Space of reference PHast (SeedOnly) for a grid of S and lambda, 1e6 keys
B=target-dev/release/cmp
for S in 6 7 8 9 10 11 12 13 14; do
  v=""
  for r in 1.50 1.58 1.66 1.74 1.82 1.90 2.00; do
    l=$(python3 -c "print(round($S/$r,2))")
    v="$v,ref:phast:$S:$l"
  done
  taskset -c 4-7 $B -n 1000000 -q 100000 -r 1 -t 4 -v ${v#,} 2>/dev/null
done
