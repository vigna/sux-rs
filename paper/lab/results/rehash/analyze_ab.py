#!/usr/bin/env python3
"""Summarizes a query A/B CSV: for each n and PHast-R variant, the median over
key seeds of the query time minus that of `ref plus` in the same process, for
each binary, and the difference between binaries."""
import sys, collections, statistics
rows = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    p = line.strip().split(',')
    if len(p) < 8: continue
    b, ks, _, name, n, bits, build, q = p[:8]
    rows[(int(n), ks, b)][name] = (float(bits), float(build), float(q))
bins = sorted({k[2] for k in rows})
names = sorted({nm for d in rows.values() for nm in d if nm.startswith('PHast-R')})
for n in sorted({k[0] for k in rows}):
    for nm in names:
        out = []
        for b in bins:
            d = [rows[k][nm][2] - rows[k]['ref plus S=8 l=5.25'][2] for k in rows if k[0] == n and k[2] == b and nm in rows[k]]
            raw = [rows[k][nm][2] for k in rows if k[0] == n and k[2] == b and nm in rows[k]]
            bits = [rows[k][nm][0] for k in rows if k[0] == n and k[2] == b and nm in rows[k]]
            build = [rows[k][nm][1] for k in rows if k[0] == n and k[2] == b and nm in rows[k]]
            out.append((b, statistics.median(d), statistics.median(raw), statistics.mean(bits), statistics.median(build)))
        s = '  '.join(f'{b}: q {r:5.1f} (vs plus {d:+5.2f}) {bi:.4f} b/k build {bu:6.1f}' for b, d, r, bi, bu in out)
        print(f'{n:>10} {nm[8:30]:22} {s}  Δ {out[1][1]-out[0][1]:+.2f} ns')
