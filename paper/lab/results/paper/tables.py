#!/usr/bin/env python3
"""Formats the rows of the tables of the paper from the CSV written by
run.sh: tables.py <csv>."""
import sys, collections, re

rows = collections.OrderedDict()
for line in open(sys.argv[1]):
    p = line.strip().split(',')
    if len(p) < 7:
        continue
    threads, _, name, n, bits, build, query = p[:7]
    rows.setdefault((int(n), name), {})[threads] = (float(bits), float(build), float(query))

def label(name):
    m = re.match(r'ref (\w+) S=(\d+) l=([\d.]+)', name)
    if m:
        c, s, l = m.groups()
        method = {'plus': r'\PHastP', 'phast': r'\PHast'}.get(c) or rf'\PHastP wrap $\delta={c[1]}$'
        return method, rf'$S={s}$, $\lambda={l}$'
    m = re.match(r'PHast-R S=(\d+) L=(\d+) R=(\d+) l=([\d.]+)', name)
    s, l, r, lam = m.groups()
    return r'\PHastR', rf'$S={s}$, $L={l}$, $\lambda={lam}$'

for n in sorted({k[0] for k in rows}):
    print(f'% n = {n}')
    for (m, name), v in rows.items():
        if m != n:
            continue
        method, conf = label(name)
        bits, build, query = v['1']
        if n == 10**7:
            print(rf'{method} & {conf} & {bits:.3f} & {build:.0f} & {query:.1f} \\')
        else:
            b8 = f"{v['8'][1]:.1f}" if '8' in v else '--'
            print(rf'{method} & {conf} & {bits:.3f} & {build:.0f} & {b8} & {query:.1f} \\')
