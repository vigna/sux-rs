#!/usr/bin/env python3
"""Summarizes results/sizes/raw.csv: mean over key sets of bits/key,
build ns/key and query ns, for each size and structure."""
import csv, collections, statistics, sys
d = collections.defaultdict(list)
for r in csv.reader(open(sys.argv[1] if len(sys.argv) > 1 else 'results/sizes/raw.csv')):
    ks, _, name, n, bits, build, q = r
    name = (name.replace('ref ', '').replace(' S=8', '').replace('PHast-R L=1024 R=4 ', 'R ')
            .replace('PHast-R S=8 L=1024 R=4 ', 'R ').replace(' u8', ''))
    d[(int(n), name)].append((float(bits), float(build), float(q)))
names = []
for k in d:
    if k[1] not in names:
        names.append(k[1])
sizes = sorted({k[0] for k in d})
for title, idx, fmt in (("bits/key", 0, "{:8.3f}"), ("build ns/key", 1, "{:8.0f}"), ("query ns", 2, "{:8.1f}")):
    print(f"\n{title} (mean over key sets)")
    print(f"{'n':>8} " + " ".join(f"{nm[:14]:>14}" for nm in names))
    for n in sizes:
        print(f"{n:>8} " + " ".join(f"{fmt.format(statistics.mean(x[idx] for x in d[(n, nm)])):>14}" if (n, nm) in d else f"{'-':>14}" for nm in names))
