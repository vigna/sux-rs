#!/usr/bin/env python3
"""For each n and configuration, the median over key seeds of the query
time, raw and relative to `ref plus` in the same process."""
import sys, collections, statistics
proc = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    p = line.strip().split(',')
    if len(p) < 8: continue
    b, ks, _, name, n, bits, build, q = p[:8]
    proc[(b, ks, int(n))][name] = (float(bits), float(build), float(q))
out = collections.defaultdict(list)
for (b, ks, n), d in proc.items():
    ref = d['ref plus S=8 l=5.25'][2]
    for name, (bits, build, q) in d.items():
        if name.startswith('ref plus'): continue
        label = name if name.startswith('ref') else f'{b:3} {name[8:]}'
        out[(n, label)].append((q, q - ref, bits, build))
for (n, label) in sorted(out):
    v = out[(n, label)]
    print(f'{n:>10} {label:44} q {statistics.median(x[0] for x in v):6.1f}  vs plus {statistics.median(x[1] for x in v):+6.2f}  '
          f'{statistics.mean(x[2] for x in v):.4f} b/k  build {statistics.median(x[3] for x in v):6.1f}')
