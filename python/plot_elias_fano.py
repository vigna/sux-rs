#!/usr/bin/env python3
"""Parse criterion benchmark output for Elias-Fano and generate an
interactive HTML visualization.

Reads from stdin (pipe) or from a file given as argument.

  cargo bench --bench bench_elias_fano | python3 python/plot_elias_fano.py > out.html

Or from a file:

  python3 python/plot_elias_fano.py bench_output.txt > out.html
"""

import argparse
import json
import re
import sys

# ── Parse ────────────────────────────────────────────────────────────

UNIT_TO_NS = {
    "ps": 0.001,
    "ns": 1.0,
    "\u00b5s": 1000.0,
    "us": 1000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}

SINGLE_RE = re.compile(
    r"^(\S+)\s+time:\s+\["
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms|s)\s+"
    r"([0-9.]+) (ps|ns|\xb5s|us|ms|s)\s+"
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms|s)"
    r"\]",
)

TIME_RE = re.compile(
    r"time:\s+\["
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms|s)\s+"
    r"([0-9.]+) (ps|ns|\xb5s|us|ms|s)\s+"
    r"[0-9.]+ (?:ps|ns|\xb5s|us|ms|s)"
    r"\]",
)

OP_LABELS = {
    "ef_get_unchecked": "get (unchecked)",
    "ef_get": "get",
    "ef_succ_unchecked": "succ (unchecked)",
    "ef_succ": "succ",
    "ef_pred_unchecked": "pred (unchecked)",
    "ef_pred": "pred",
    "ef_rank_unchecked": "rank (unchecked)",
    "ef_rank": "rank",
    "ef_build_seq": "build (seq)",
    "ef_build_conc": "build (conc)",
}

OP_ORDER = list(OP_LABELS.keys())

QUERY_GROUPS = [k for k in OP_LABELS if not k.startswith("ef_build")]
BUILD_GROUPS = [k for k in OP_LABELS if k.startswith("ef_build")]


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

        if "/" in line and not line.startswith("Benchmarking") and i + 1 < len(lines):
            tm = TIME_RE.search(lines[i + 1])
            if tm:
                raw[line] = float(tm.group(1)) * UNIT_TO_NS[tm.group(2)]
                i += 2
                continue

        i += 1

    entries = []
    for name, ns_per_op in raw.items():
        parts = name.split("/")

        # Query benchmarks: group/backend/size/l=N
        if len(parts) == 4 and parts[3].startswith("l="):
            group, backend, size, l_str = parts
            entries.append(
                {
                    "group": group,
                    "op": OP_LABELS.get(group, group),
                    "backend": backend,
                    "size": size,
                    "l": int(l_str[2:]),
                    "threads": 0,
                    "ns_per_op": round(ns_per_op, 2),
                }
            )
        # Build sequential: group/push/size/l=N
        elif len(parts) == 4 and parts[1] == "push" and parts[3].startswith("l="):
            group, _, size, l_str = parts
            entries.append(
                {
                    "group": group,
                    "op": OP_LABELS.get(group, group),
                    "backend": "push",
                    "size": size,
                    "l": int(l_str[2:]),
                    "threads": 1,
                    "ns_per_op": round(ns_per_op, 2),
                }
            )
        # Build concurrent: group/set/size/l=N/t=T
        elif len(parts) == 5 and parts[1] == "set" and parts[3].startswith("l=") and parts[4].startswith("t="):
            group, _, size, l_str, t_str = parts
            entries.append(
                {
                    "group": group,
                    "op": OP_LABELS.get(group, group),
                    "backend": f"set/t={t_str[2:]}",
                    "size": size,
                    "l": int(l_str[2:]),
                    "threads": int(t_str[2:]),
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
  #tg-view button.on {{
    background: #2d5aa0;
    border-color: #2d5aa0;
    font-size: 1em;
  }}
  #tg-view button {{
    font-size: 1em;
    padding: 0.4em 1.4em;
  }}
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

<div class="row">
  <div class="toggle-group" id="tg-view"></div>
</div>

<div id="view-query">
  <p class="hint">Toggle any combination \u2014 each bar is a (size, l, backend) triple.</p>
  <div class="row">
    <label>Ops</label>
    <div class="toggle-group" id="tg-q-ops"></div>
  </div>
  <div class="row">
    <label>Size</label>
    <div class="toggle-group" id="tg-q-size"></div>
    <label>l</label>
    <div class="toggle-group" id="tg-q-l"></div>
    <label>Backend</label>
    <div class="toggle-group" id="tg-q-backend"></div>
  </div>
</div>

<div id="view-build" style="display:none">
  <p class="hint">Toggle any combination \u2014 each bar is a (size, l, method) triple.</p>
  <div class="row">
    <label>Ops</label>
    <div class="toggle-group" id="tg-b-ops"></div>
  </div>
  <div class="row">
    <label>Size</label>
    <div class="toggle-group" id="tg-b-size"></div>
    <label>l</label>
    <div class="toggle-group" id="tg-b-l"></div>
    <label>Method</label>
    <div class="toggle-group" id="tg-b-method"></div>
  </div>
</div>

<canvas id="chart"></canvas>

<script>
const DATA = {data_json};
const OP_ORDER = {op_order_json};

const QUERY_GROUPS = {query_groups_json};
const BUILD_GROUPS = {build_groups_json};

function unique(arr) {{ return [...new Set(arr)]; }}

const OP_LABEL = Object.fromEntries(DATA.map(d => [d.group, d.op]));

const queryData = DATA.filter(d => QUERY_GROUPS.includes(d.group));
const buildData = DATA.filter(d => BUILD_GROUPS.includes(d.group));

const Q_OPS      = OP_ORDER.filter(g => queryData.some(d => d.group === g));
const Q_SIZES    = unique(queryData.map(d => d.size));
const Q_LS       = unique(queryData.map(d => d.l)).sort((a, b) => a - b);
const Q_BACKENDS = unique(queryData.map(d => d.backend));

const B_OPS     = OP_ORDER.filter(g => buildData.some(d => d.group === g));
const B_SIZES   = unique(buildData.map(d => d.size));
const B_LS      = unique(buildData.map(d => d.l)).sort((a, b) => a - b);
const B_METHODS = unique(buildData.map(d => d.backend));

// ── Color palette ──

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

function makeColorMap(ls, backends) {{
  const lHue = {{}};
  ls.forEach((l, i) => {{ lHue[l] = HUE_PAIRS[i % HUE_PAIRS.length]; }});
  const bkList = [...backends];
  return function(l, backend, size, sizes) {{
    const pair = lHue[l] || ['#bab0ab', '#d4ccc8'];
    const bkIdx = bkList.indexOf(backend);
    const hex = pair[bkIdx % pair.length];
    if (sizes.length > 1 && size !== sizes[0]) {{
      return hex + 'a0';
    }}
    return hex;
  }};
}}

const qColor = makeColorMap(Q_LS, Q_BACKENDS);
const bColor = makeColorMap(B_LS, B_METHODS);

// ── Toggle buttons ──

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

// ── View toggle (exclusive) ──

let currentView = 'query';

function setupViewToggle() {{
  const c = document.getElementById('tg-view');
  ['Queries', 'Build'].forEach(label => {{
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.value = label.toLowerCase().replace('queries', 'query');
    if (b.dataset.value === currentView) b.classList.add('on');
    b.onclick = () => {{
      currentView = b.dataset.value;
      c.querySelectorAll('button').forEach(x => x.classList.remove('on'));
      b.classList.add('on');
      document.getElementById('view-query').style.display =
        currentView === 'query' ? '' : 'none';
      document.getElementById('view-build').style.display =
        currentView === 'build' ? '' : 'none';
      render();
    }};
    c.appendChild(b);
  }});
}}

setupViewToggle();

// ── Populate toggles ──

makeToggles('tg-q-ops',     Q_OPS,      g => OP_LABEL[g] || g, Q_OPS);
makeToggles('tg-q-size',    Q_SIZES,    s => s,                 Q_SIZES.length ? [Q_SIZES[0]] : []);
makeToggles('tg-q-l',       Q_LS,       l => 'l=' + l,         Q_LS);
makeToggles('tg-q-backend', Q_BACKENDS, b => b,                 Q_BACKENDS);

makeToggles('tg-b-ops',     B_OPS,      g => OP_LABEL[g] || g, B_OPS);
makeToggles('tg-b-size',    B_SIZES,    s => s,                 B_SIZES.length ? [B_SIZES[0]] : []);
makeToggles('tg-b-l',       B_LS,       l => 'l=' + l,         B_LS);
makeToggles('tg-b-method',  B_METHODS,  m => m,                 B_METHODS);

// ── Chart data ──

function buildChartData() {{
  if (currentView === 'query') {{
    const ops = selected('tg-q-ops');
    const szs = selected('tg-q-size');
    const ls  = selected('tg-q-l').map(Number);
    const bks = selected('tg-q-backend');

    const labels   = ops.map(g => OP_LABEL[g] || g);
    const datasets = [];

    for (const size of szs) {{
      for (const l of ls) {{
        for (const bk of bks) {{
          const data = ops.map(g => {{
            const d = queryData.find(e =>
              e.group === g && e.size === size && e.l === l && e.backend === bk);
            return d ? d.ns_per_op : null;
          }});
          const parts = [];
          if (szs.length > 1) parts.push(size);
          if (ls.length > 1)  parts.push('l=' + l);
          if (bks.length > 1) parts.push(bk);
          if (parts.length === 0) parts.push(bk);
          datasets.push({{
            label: parts.join(' '),
            data,
            backgroundColor: qColor(l, bk, size, szs),
            skipNull: true,
          }});
        }}
      }}
    }}
    return {{ labels, datasets }};

  }} else {{
    const ops = selected('tg-b-ops');
    const szs = selected('tg-b-size');
    const ls  = selected('tg-b-l').map(Number);
    const mts = selected('tg-b-method');

    const labels   = ops.map(g => OP_LABEL[g] || g);
    const datasets = [];

    for (const size of szs) {{
      for (const l of ls) {{
        for (const mt of mts) {{
          const data = ops.map(g => {{
            const d = buildData.find(e =>
              e.group === g && e.size === size && e.l === l && e.backend === mt);
            return d ? d.ns_per_op : null;
          }});
          const parts = [];
          if (szs.length > 1) parts.push(size);
          if (ls.length > 1)  parts.push('l=' + l);
          if (mts.length > 1) parts.push(mt);
          if (parts.length === 0) parts.push(mt);
          datasets.push({{
            label: parts.join(' '),
            data,
            backgroundColor: bColor(l, mt, size, szs),
            skipNull: true,
          }});
        }}
      }}
    }}
    return {{ labels, datasets }};
  }}
}}

// ── Auto-scale time unit ──

function bestUnit(datasets) {{
  let maxVal = 0;
  for (const ds of datasets) {{
    for (const v of ds.data) {{
      if (v != null && v > maxVal) maxVal = v;
    }}
  }}
  if (maxVal >= 1e9) return {{ divisor: 1e9, label: 's / op',  suffix: ' s'  }};
  if (maxVal >= 1e6) return {{ divisor: 1e6, label: 'ms / op', suffix: ' ms' }};
  if (maxVal >= 1e3) return {{ divisor: 1e3, label: '\u00b5s / op', suffix: ' \u00b5s' }};
  return {{ divisor: 1, label: 'ns / op', suffix: ' ns' }};
}}

// ── Render ──

function render() {{
  const cfg = buildChartData();
  const unit = bestUnit(cfg.datasets);
  for (const ds of cfg.datasets) {{
    ds.data = ds.data.map(v => v == null ? null : v / unit.divisor);
  }}
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chart'), {{
    type: 'bar',
    data: {{ labels: cfg.labels, datasets: cfg.datasets }},
    options: {{
      responsive: true,
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const v = ctx.parsed.y;
              if (v == null) return null;
              return ctx.dataset.label + ': ' + v.toFixed(2) + unit.suffix;
            }},
          }},
        }},
        legend: {{ position: 'top' }},
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: 'Time (' + unit.label + ')' }},
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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        query_groups_json=json.dumps(QUERY_GROUPS),
        build_groups_json=json.dumps(BUILD_GROUPS),
    )

    sys.stdout.write(html)
    print(f"{len(entries)} benchmarks.", file=sys.stderr)


if __name__ == "__main__":
    main()
