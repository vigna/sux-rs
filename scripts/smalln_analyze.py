#!/usr/bin/env python3
"""Analyze the Phase 3 small-n exhaustive solvability sweep.

Acceptance criterion: each (n) cell is OK if ≥ 2 / 5 trials succeeded
with `build_ok`. Reports per-file:
  - total n values swept
  - pass count   (≥ 2 of 5 trials ok)
  - warn count   (exactly 1 of 5 ok)
  - fail count   (0 of 5 ok)
  - list of the first few failing n values
"""

import csv
import sys
from collections import defaultdict

PASS_THRESHOLD = 2  # out of 5 trials


def analyze(path):
    by_n = defaultdict(lambda: {"ok": 0, "bad": 0})
    meta = None
    for row in csv.DictReader(open(path)):
        if meta is None:
            meta = (row["shard_edge"], row["distribution"])
        n = int(row["n"])
        if row["outcome"] == "build_ok":
            by_n[n]["ok"] += 1
        else:
            by_n[n]["bad"] += 1

    if not by_n:
        print(f"{path}: empty")
        return

    total = len(by_n)
    pass_ = sum(1 for r in by_n.values() if r["ok"] >= PASS_THRESHOLD)
    warn = sum(1 for r in by_n.values() if r["ok"] == 1)
    fail = sum(1 for r in by_n.values() if r["ok"] == 0)

    shard, dist = meta
    tag = "OK" if fail == 0 and warn <= max(5, total // 200) else "ISSUES"
    line = (
        f"{shard:20s} {dist:10s}  n={total:5d}  "
        f"pass={pass_:5d}  warn={warn:3d}  fail={fail:3d}  [{tag}]"
    )
    print(line)

    if fail > 0 or warn > 0:
        worst = sorted(
            (n for n, r in by_n.items() if r["ok"] < PASS_THRESHOLD),
            key=lambda n: (by_n[n]["ok"], n),
        )
        print(f"  First {min(len(worst), 10)} weak/failing n values:")
        for n in worst[:10]:
            r = by_n[n]
            print(f"    n={n:10d}: ok={r['ok']} bad={r['bad']}")


if __name__ == "__main__":
    paths = sorted(sys.argv[1:])
    for p in paths:
        analyze(p)
