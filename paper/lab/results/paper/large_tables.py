#!/usr/bin/env python3
"""Formats the rows of the tables of Section 4.1 of the paper from the
output of large.sh and from the CSV written by run.sh:
large_tables.py <directory written by large.sh> <csv written by run.sh>."""
import sys, re, collections

d, csv = sys.argv[1], sys.argv[2]

def label(name):
    name = name.replace(' u8', '')
    m = re.match(r'ref[: ](\w+)(?:[: ]|\s+S=)(\d+)(?:[: ]|\s+l=)([\d.]+)', name)
    if m:
        c, s, l = m.groups()
        method = {'plus': r'\PHastP', 'phast': r'\PHast'}.get(c) or rf'\PHastP wrap $\delta={c[1]}$'
        return rf'{method}, $\lambda={float(l):g}$'
    m = re.match(r'PHast-R S=(\d+) L=(\d+) R=(\d+) l=([\d.]+)', name)
    if m:
        return rf'\PHastR, $\lambda={float(m.group(4)):g}$'
    m = re.match(r'(\d+):(\d+):([\d.]+)', name)
    return rf'\PHastR, $\lambda={float(m.group(3)):g}$'

# Query time of the keys of the first level and of all keys
split = collections.OrderedDict()
for line in open(f'{d}/qsplit.txt'):
    m = re.match(r'(\d+) (\S+)(?: u8)?\s+all\s+([\d.]+) ns\s+fast\s+([\d.]+) ns\s+slow\s+([\d.]+) ns\s+bumped ([\d.]+)%', line)
    n, name, all_, fast, slow, beta = m.groups()
    split.setdefault(name, {})[int(n)] = (float(fast), float(all_), float(beta))
# Query time of all keys with huge pages (single-threaded runs)
huge = {}
for line in open(f'{d}/hugepages.csv'):
    p = line.strip().split(',')
    if len(p) >= 7 and p[0] == '1':
        huge[(label(p[2]), int(p[3]))] = float(p[6])
print('% TABLE4: first level / all keys at 10^7, 10^8, 10^9; all keys with huge pages at 10^8, 10^9; extra time per bumped key at 10^9')
for name, v in split.items():
    cells = []
    for n in (10**7, 10**8, 10**9):
        fast, all_, beta = v[n]
        cells += [f'{fast:.1f}', f'{all_:.1f}']
    for n in (10**8, 10**9):
        cells.append(f'{huge[(label(name), n)]:.1f}' if (label(name), n) in huge else '--')
    print(rf'{label(name)} & {v[10**9][2]:.1f} & ' + ' & '.join(cells) + r' \\')
print('% extra time of a bumped key in a mixed workload (ns), by n')
for name, v in split.items():
    print('% ', label(name), ' '.join(f'{(v[n][1] - v[n][0]) / (v[n][2] / 100):.0f}' for n in (10**7, 10**8, 10**9)))

# Construction: threads and memory
scaling = collections.OrderedDict()
for line in open(f'{d}/scaling.txt'):
    f = line.split()
    n, t = int(f[0]), int(f[1])
    if f[2] == 'PHast-R':
        name, build = label(f[3]), float(re.search(r'build min\s+([\d.]+)', line).group(1))
    else:
        p = line.split(',')
        name, build = label(p[1]), float(p[4])
    scaling.setdefault(name, {})[(n, t)] = build
memory = {}
lines = open(f'{d}/memory.txt').read().splitlines()
cur = None
for line in lines:
    f = line.split()
    n = int(f[0])
    if 'Maximum resident' in line:
        name = cur if f[1] == 'PHast-R' else label(f[1])
        # Kilobytes, minus eight bytes per key for the keys
        memory[(name, n)] = (int(f[-1]) * 1024 - 8 * n) / n
    elif f[1] == 'PHast-R':
        cur = label(f[2])
n = max(k[0] for v in scaling.values() for k in v)
threads = sorted({k[1] for v in scaling.values() for k in v if k[0] == n})
# The speedup refers to eight threads (the number of cores), if present
ref = 8 if 8 in threads else threads[-1]
print(f'% TABLE5: build ns/key at {n} with {threads} threads; speedup with {ref} threads; bytes/key')
for name, v in scaling.items():
    cells = [f'{v[(n, t)]:.1f}' for t in threads]
    print(rf'{name} & ' + ' & '.join(cells) + rf' & {v[(n, 1)] / v[(n, ref)]:.1f} & {memory[(name, n)]:.1f} \\')
print('% build ns/key by (n, threads):', {k: sorted(v.items()) for k, v in scaling.items()})
print('% bytes/key:', memory)
print('% huge pages:', huge)
# Construction with huge pages
for line in open(f'{d}/hugepages.csv'):
    p = line.strip().split(',')
    if len(p) >= 7:
        print('% huge', p[0], 'threads', p[3], label(p[2]), 'build', p[5])
