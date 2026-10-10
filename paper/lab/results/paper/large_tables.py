#!/usr/bin/env python3
"""Formats the rows of the tables of Section 4.1 of the paper from the
output of large.sh: the query time of first-level keys and of all keys, and
the thread scaling and peak memory of the construction. Sizes that are
missing are skipped.
Usage: large_tables.py <directory written by large.sh>"""
import sys, re, collections

d = sys.argv[1]

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
sizes = sorted({n for v in split.values() for n in v})
print(f'% TABLE4: first level / all keys at {sizes}; bumped keys at {sizes[-1] if sizes else None}')
for name, v in split.items():
    cells = []
    for n in sizes:
        fast, all_, beta = v.get(n, (None, None, None))
        cells += ['--', '--'] if fast is None else [f'{fast:.1f}', f'{all_:.1f}']
    beta = v[max(v)][2]
    print(rf'{label(name)} & {beta:.1f} & ' + ' & '.join(cells) + r' \\')
print('% extra time of a bumped key in a mixed workload (ns), by n')
for name, v in split.items():
    print('% ', label(name), ' '.join(f'{n}: {(v[n][1] - v[n][0]) / (v[n][2] / 100):.0f}' for n in sizes if n in v))

# Construction: threads (one CSV line of cmp per structure, number of keys
# and number of threads) and memory
scaling = collections.OrderedDict()
for line in open(f'{d}/scaling.txt'):
    n, t, csv = line.split(maxsplit=2)
    p = csv.strip().split(',')
    scaling.setdefault(label(p[1]), {})[(int(n), int(t))] = float(p[4])
memory = {}
cur = None
for line in open(f'{d}/memory.txt'):
    f = line.split()
    n = int(f[0])
    if 'Maximum resident' in line:
        name = cur if f[1] == 'PHast-R' else label(f[1])
        # Kilobytes, minus eight bytes per key for the keys
        memory[(name, n)] = (int(f[-1]) * 1024 - 8 * n) / n
    elif f[1] == 'PHast-R':
        cur = label(f[2])
if scaling:
    n = max(k[0] for v in scaling.values() for k in v)
    threads = sorted({k[1] for v in scaling.values() for k in v if k[0] == n})
    # The speedup refers to the largest power of two, which is the largest
    # number of threads running on distinct cores (see large.sh)
    ref = max(t for t in threads if t & (t - 1) == 0)
    print(f'% TABLE5: build ns/key at {n} with {threads} threads; speedup with {ref} threads; bytes/key')
    for name, v in scaling.items():
        cells = [f'{v[(n, t)]:.1f}' if (n, t) in v else '--' for t in threads]
        speedup = f'{v[(n, 1)] / v[(n, ref)]:.1f}' if (n, 1) in v and (n, ref) in v else '--'
        mem = f'{memory[(name, n)]:.1f}' if (name, n) in memory else '--'
        print(rf'{name} & ' + ' & '.join(cells) + rf' & {speedup} & {mem} \\')
    print('% build ns/key by (n, threads):', {k: sorted(v.items()) for k, v in scaling.items()})
print('% bytes/key:', memory)
