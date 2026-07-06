#!/usr/bin/env python3
"""Summarize comp_vfunc_stress CSVs.

Acceptance criterion: a cell (shard_edge, dist, n) is OK if ≥ 2/5
trials succeeded with `build_ok` (matching VBuilder's retry budget).
"""

import csv
import sys
from collections import defaultdict

PASS_THRESHOLD = 2  # out of 5 trials


def summarize(path):
    cells = defaultdict(lambda: {"ok": 0, "bad": 0})
    meta = None
    for row in csv.DictReader(open(path)):
        if meta is None:
            meta = (row["shard_edge"], row["distribution"])
        key = int(row["n"])
        if row["outcome"] == "build_ok":
            cells[key]["ok"] += 1
        else:
            cells[key]["bad"] += 1

    if not cells:
        print(f"{path}: empty")
        return

    shard, dist = meta
    total_ok = sum(c["ok"] for c in cells.values())
    total_bad = sum(c["bad"] for c in cells.values())
    weak = sum(1 for c in cells.values() if c["ok"] < PASS_THRESHOLD)

    status = "OK" if weak == 0 else f"WEAK={weak}"
    print(
        f"{shard:22s} {dist:10s}  trials_ok={total_ok:4d}/{total_ok + total_bad:4d}  "
        f"n_cells={len(cells):4d}  [{status}]"
    )

    if weak > 0:
        for n in sorted(cells):
            c = cells[n]
            if c["ok"] < PASS_THRESHOLD:
                print(f"    n={n:10d}: ok={c['ok']} bad={c['bad']}")


if __name__ == "__main__":
    paths = sorted(sys.argv[1:])
    for p in paths:
        summarize(p)
