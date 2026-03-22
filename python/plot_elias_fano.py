#!/usr/bin/env python3
"""Parse criterion benchmark output for Elias-Fano and generate an
interactive HTML visualization.

Usage:
    cargo bench --bench bench_elias_fano 2>&1 | python3 python/plot_elias_fano.py > out.html
    # or from a file:
    python3 python/plot_elias_fano.py bench_output.txt > out.html

All dimensions (operation, size, l, backend) are multi-selectable.
Colors encode l (hue) and backend (shade), with opacity for size.
"""

import argparse
import re
import sys
import json

# ── Parse ────────────────────────────────────────────────────────────

UNIT_TO_NS = {"ps": 0.001, "ns": 1.0, "\u00b5s": 1000.0, "us": 1000.0, "ms": 1_000_000.0}

SINGLE_RE = re.compile(
    r"^(\S+)\s+time:\s+\["
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms)\s+"
    r"([0-9.]+) (ps|ns|\xb5s|us|ms)\s+"
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms)"
    r"\]",
)

TIME_RE = re.compile(
    r"time:\s+\["
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms)\s+"
    r"([0-9.]+) (ps|ns|\xb5s|us|ms)\s+"
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms)"
    r"\]",
)

OP_LABELS = {
    "ef_get_unchecked": "get (unchecked)",
    "ef_get": "get",
    "ef_succ_unchecked": "succ (unchecked)",
    "ef_succ": "succ",
    "ef_pred_unchecked": "pred (unchecked)",
    "ef_pred": "pred",
}

OP_ORDER = list(OP_LABELS.keys())


def parse(lines):
    """Return list of dicts with keys: group, op, backend, size, l, ns_per_op."""
    raw = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        m = SINGLE_RE.match(line)
        if m:
            raw[m.group(1)] = float(m.group(2)) * UNIT_TO_NS[m.group(3)]
            i += 1
            continue

        if (
            "/" in line
            and not line.startswith("Benchmarking")
            and i + 1 < len(lines)
        ):
            tm = TIME_RE.search(lines[i + 1])
            if tm:
                raw[line] = float(tm.group(1)) * UNIT_TO_NS[tm.group(2)]
                i += 2
                continue

        i += 1

    entries = []
    for name, ns_per_op in raw.items():
        parts = name.split("/")
        if len(parts) != 4:
            continue
        group, backend, size, l_str = parts
        if not l_str.startswith("l="):
            continue
        entries.append(
            {
                "group": group,
                "op": OP_LABELS.get(group, group),
                "backend": backend,
                "size": size,
                "l": int(l_str[2:]),
                "ns_per_op": round(ns_per_op, 2),
            }
        )

    return entries


# ── HTML generation ──────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Elias\u2013Fano benchmarks</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1120px;
    margin: 2em auto;
    background: #fafafa;
  }}
  h1 {{ text-align: center; }}
  .row {{
    display: flex;
    justify-content: center;
    gap: 1.2em;
    margin: 0.6em 0;
    flex-wrap: wrap;
    align-items: center;
  }}
  .row label {{
    font-weight: 600;
    font-size: 0.9em;
    color: #555;
    min-width: 3.5em;
    text-align: right;
  }}
  .toggle-group {{ display: flex; gap: 0; }}
  .toggle-group button {{
    padding: 0.35em 0.9em;
    border: 2px solid #aaa;
    background: white;
    cursor: pointer;
    font-size: 0.9em;
    color: #666;
    transition: all 0.15s;
  }}
  .toggle-group button:first-child {{ border-radius: 5px 0 0 5px; }}
  .toggle-group button:last-child  {{ border-radius: 0 5px 5px 0; }}
  .toggle-group button:not(:first-child) {{ border-left: none; }}
  .toggle-group button.on {{
    background: #4a7cff;
    color: white;
    border-color: #4a7cff;
    font-weight: 600;
  }}
  .toggle-group button.on + button:not(.on) {{ border-left-color: #4a7cff; }}
  canvas {{ background: white; border-radius: 8px; margin-top: 0.8em; }}
  .hint {{
    text-align: center;
    font-size: 0.82em;
    color: #999;
    margin-top: 0.3em;
  }}
</style>
</head>
<body>
<h1>Elias\u2013Fano benchmarks</h1>
<p class="hint">Toggle any combination \u2014 each bar is a (size, l, backend) triple.</p>

<div class="row">
  <label>Ops</label>
  <div class="toggle-group" id="tg-ops"></div>
</div>
<div class="row">
  <label>Size</label>
  <div class="toggle-group" id="tg-size"></div>

  <label>l</label>
  <div class="toggle-group" id="tg-l"></div>

  <label>Backend</label>
  <div class="toggle-group" id="tg-backend"></div>
</div>

<canvas id="chart"></canvas>

<script>
const DATA = {data_json};
const OP_ORDER = {op_order_json};

// ── Discover dimensions from data ──

function unique(arr) {{ return [...new Set(arr)]; }}

const SIZES    = unique(DATA.map(d => d.size));
const LS       = unique(DATA.map(d => d.l)).sort((a, b) => a - b);
const BACKENDS = ['aligned', 'unaligned'];
const seenOps  = unique(DATA.map(d => d.group));
const OPS      = OP_ORDER.filter(g => seenOps.includes(g))
                          .concat(seenOps.filter(g => !OP_ORDER.includes(g)));
const OP_LABEL = Object.fromEntries(DATA.map(d => [d.group, d.op]));

// ── Color palette ──
// Hue encodes l; shade encodes backend (dark = aligned, light = unaligned).
// Opacity distinguishes sizes when both are shown.
// Uses Tableau-inspired paired colors.

const HUE_PAIRS = [
  ['#4e79a7', '#a0cbe8'],   // blue
  ['#f28e2b', '#ffbe7d'],   // orange
  ['#59a14f', '#8cd17d'],   // green
  ['#e15759', '#ff9d9a'],   // red
  ['#76b7b2', '#9dd0cb'],   // teal
  ['#edc949', '#f5e08a'],   // yellow
  ['#af7aa1', '#d4a6c8'],   // purple
  ['#9c755f', '#c9a88e'],   // brown
];

const lHue = {{}};
LS.forEach((l, i) => {{ lHue[l] = HUE_PAIRS[i % HUE_PAIRS.length]; }});

function barColor(l, backend, size) {{
  const pair = lHue[l] || ['#bab0ab', '#d4ccc8'];
  const hex  = backend === 'aligned' ? pair[0] : pair[1];
  // Add alpha for multi-size distinction (1G = semi-transparent).
  if (SIZES.length > 1 && size !== SIZES[0]) {{
    return hex + 'a0';
  }}
  return hex;
}}

// ── Toggle buttons (multi-select) ──

let chart = null;

function makeToggles(id, items, labelFn, defaults) {{
  const c = document.getElementById(id);
  items.forEach(item => {{
    const b = document.createElement('button');
    b.textContent = labelFn(item);
    b.dataset.value = String(item);
    if (defaults.includes(item)) b.classList.add('on');
    b.onclick = () => {{ b.classList.toggle('on'); render(); }};
    c.appendChild(b);
  }});
}}

function selected(id) {{
  return [...document.querySelectorAll('#' + id + ' button.on')]
    .map(b => b.dataset.value);
}}

// Default: first size, all l, both backends, all ops.
makeToggles('tg-ops',     OPS,      g => OP_LABEL[g] || g, OPS);
makeToggles('tg-size',    SIZES,    s => s,                 [SIZES[0]]);
makeToggles('tg-l',       LS,       l => 'l=' + l,         LS);
makeToggles('tg-backend', BACKENDS, b => b,                 BACKENDS);

// ── Build chart data ──

function buildChart() {{
  const ops  = selected('tg-ops');
  const szs  = selected('tg-size');
  const ls   = selected('tg-l').map(Number);
  const bks  = selected('tg-backend');

  const labels   = ops.map(g => OP_LABEL[g] || g);
  const datasets = [];

  for (const size of szs) {{
    for (const l of ls) {{
      for (const bk of bks) {{
        const data = ops.map(g => {{
          const d = DATA.find(e =>
            e.group === g && e.size === size && e.l === l && e.backend === bk);
          return d ? d.ns_per_op : null;
        }});

        // Smart label: omit dimensions with only one selection.
        const parts = [];
        if (szs.length > 1) parts.push(size);
        if (ls.length > 1)  parts.push('l=' + l);
        if (bks.length > 1) parts.push(bk);
        // If everything is single-select, show at least backend.
        if (parts.length === 0) parts.push(bk);

        datasets.push({{
          label: parts.join(' '),
          data,
          backgroundColor: barColor(l, bk, size),
        }});
      }}
    }}
  }}

  return {{ labels, datasets }};
}}

// ── Render ──

function render() {{
  const cfg = buildChart();
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chart'), {{
    type: 'bar',
    data: {{ labels: cfg.labels, datasets: cfg.datasets }},
    options: {{
      responsive: true,
      skipNull: true,
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const v = ctx.parsed.y;
              if (v == null) return null;
              return ctx.dataset.label + ': ' + v.toFixed(1) + ' ns/op';
            }},
          }},
        }},
        legend: {{ position: 'top' }},
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: 'Time (ns / op)' }},
        }},
        x: {{
          ticks: {{ autoSkip: false }},
        }},
      }},
    }},
  }});
}}

render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Parse criterion benchmark output for Elias-Fano "
        "and generate an interactive HTML visualization.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="benchmark output file (default: stdin)",
    )
    args = parser.parse_args()

    lines = args.input.readlines()

    entries = parse(lines)
    if not entries:
        print("No benchmark results found in input.", file=sys.stderr)
        sys.exit(1)

    html = HTML_TEMPLATE.format(
        data_json=json.dumps(entries, indent=2),
        op_order_json=json.dumps(OP_ORDER),
    )

    sys.stdout.write(html)
    print(f"{len(entries)} benchmarks.", file=sys.stderr)


if __name__ == "__main__":
    main()
