#!/usr/bin/env python3
"""Space/time trade-off plots from the CSV written by pareto.sh: two panels
sharing the space axis, query time and construction time, one line per
structure and seed width along its sweep of expected bucket sizes; the
dashed staircase is the Pareto front of each panel, and the configurations
that are Pareto-optimal in all three dimensions are printed on standard
output. Usage: plot_pareto.py <csv> <number of keys> <output file>."""
import sys, re, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# (method, seed bits) -> list of (lambda, bits/key, build ns/key, query ns)
series = collections.defaultdict(list)
for line in open(csv):
    p = line.strip().split(',')
    if len(p) < 7 or int(p[3]) != n:
        continue
    name, bits, build, query = p[2], float(p[4]), float(p[5]), float(p[6])
    m = re.match(r'ref (\w+) S=(\d+) l=([\d.]+)', name)
    if m:
        c, s, l = m.groups()
        method = {'plus': 'PHast+', 'phast': 'PHast'}.get(c) or f'PHast+ wrap $\\delta={c[1]}$'
    else:
        m = re.match(r'PHast-R S=(\d+) L=\d+ R=\d+ l=([\d.]+)', name)
        s, l = m.groups()
        method = 'PHast-R'
    series[(method, int(s))].append((float(l), bits, build, query))

style = {'PHast+': ('tab:gray', 's'), 'PHast+ wrap $\\delta=3$': ('tab:blue', 'o'),
         'PHast': ('tab:orange', '^'), 'PHast-R': ('tab:red', 'D')}

def front(points):
    """The staircase of the points not dominated in (x, y), both minimized."""
    best, res = float('inf'), []
    for x, y in sorted(points):
        if y < best:
            res.append((x, y))
            best = y
    return res

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharex=True)
for ax, idx, label in ((axes[0], 3, 'query time (ns)'), (axes[1], 2, 'construction time (ns/key)')):
    pts = []
    for (method, s), v in sorted(series.items()):
        v.sort()
        color, marker = style[method]
        ax.plot([x[1] for x in v], [x[idx] for x in v], marker=marker, color=color, ms=4,
                lw=0.8, mfc=color if s == 8 else 'white', label=f'{method}, $S={s}$')
        pts += [(x[1], x[idx]) for x in v]
    f = front(pts)
    ax.step([x for x, _ in f] + [max(p[0] for p in pts)], [y for _, y in f] + [f[-1][1]],
            where='post', color='black', lw=0.6, ls='--', zorder=0)
    ax.set_xlabel('bits/key')
    ax.set_ylabel(label)
    ax.grid(True, lw=0.3, alpha=0.5)
axes[1].set_yscale('log')
axes[0].legend(loc='best', frameon=False)
fig.tight_layout(w_pad=1.5)
fig.savefig(out)

# Configurations that no other configuration beats in space, query time, and
# construction time at once
all_pts = [(method, s, l, b, bu, q) for (method, s), v in series.items() for l, b, bu, q in v]
for m, s, l, b, bu, q in sorted(all_pts, key=lambda x: x[3]):
    if not any((b2 <= b and bu2 <= bu and q2 <= q) and (b2, bu2, q2) != (b, bu, q)
               for _, _, _, b2, bu2, q2 in all_pts):
        print(f'{m} S={s} lambda={l:g}: {b:.3f} bits/key, build {bu:.1f} ns/key, query {q:.1f} ns')
