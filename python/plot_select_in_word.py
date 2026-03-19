#!/usr/bin/env python3
"""Parse criterion benchmark output for select-in-word and generate an
interactive HTML visualization.

Usage:
    cargo bench --bench bench_select_in_word 2>&1 | python3 python/plot_select_in_word.py
    # or from a file:
    python3 python/plot_select_in_word.py bench_output.txt

The script reads from a file argument or stdin, extracts the estimate
(middle value) from each ``time: [low estimate high]`` line, and writes
``select_in_word.html`` in the current directory.
"""

import re
import sys
import json

# ── Parse ────────────────────────────────────────────────────────────

UNIT_TO_NS = {"ps": 0.001, "ns": 1.0, "µs": 1000.0, "us": 1000.0, "ms": 1_000_000.0}

TIME_RE = re.compile(
    r"^(\S+)\s+time:\s+\["
    r"[0-9.]+ (?:ps|ns|µs|us|ms)\s+"
    r"([0-9.]+) (ps|ns|µs|us|ms)\s+"
    r"[0-9.]+ (?:ps|ns|µs|us|ms)"
    r"\]",
)


def parse(lines):
    """Return list of (technique, word_type, time_ns) tuples."""
    results = {}
    for line in lines:
        m = TIME_RE.match(line.strip())
        if not m:
            continue
        name, value, unit = m.group(1), float(m.group(2)), m.group(3)
        time_ns = value * UNIT_TO_NS[unit]

        # Split e.g. "popcount2_u128" → ("popcount2", "u128")
        idx = name.rfind("_u")
        if idx == -1:
            continue
        technique = name[:idx]
        word = name[idx + 1 :]

        # Keep only the last result for each benchmark
        results[(technique, word)] = time_ns

    return [(t, w, v) for (t, w), v in results.items()]


# ── HTML generation ──────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>select-in-word benchmarks</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 960px;
    margin: 2em auto;
    background: #fafafa;
  }}
  h1 {{ text-align: center; }}
  .controls {{
    text-align: center;
    margin: 1.5em 0;
  }}
  .controls button {{
    padding: 0.5em 1.5em;
    margin: 0 0.3em;
    border: 2px solid #555;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 1em;
  }}
  .controls button.active {{
    background: #4a7cff;
    color: white;
    border-color: #4a7cff;
  }}
  canvas {{ background: white; border-radius: 8px; }}
</style>
</head>
<body>
<h1>select-in-word benchmarks</h1>
<div class="controls">
  <button id="btn-word" class="active" onclick="showView('word')">By word type</button>
  <button id="btn-tech" onclick="showView('tech')">By technique</button>
</div>
<canvas id="chart"></canvas>
<script>
const DATA = {data_json};

const WORD_ORDER = ['u8', 'u16', 'u32', 'u64', 'u128'];
const COLORS = [
  '#4a7cff', '#ff6b6b', '#51cf66', '#fcc419', '#cc5de8',
  '#20c997', '#ff922b', '#868e96', '#e64980', '#15aabf',
];

let chart = null;
let currentView = 'word';

function getWords() {{
  return [...new Set(DATA.map(d => d.word))].sort(
    (a, b) => WORD_ORDER.indexOf(a) - WORD_ORDER.indexOf(b)
  );
}}

function getTechniques() {{
  // Preserve insertion order (benchmark order)
  const seen = new Set();
  const result = [];
  for (const d of DATA) {{
    if (!seen.has(d.technique)) {{
      seen.add(d.technique);
      result.push(d.technique);
    }}
  }}
  return result;
}}

function buildByWord() {{
  const words = getWords();
  const techniques = getTechniques();
  const datasets = techniques.map((tech, i) => ({{
    label: tech,
    data: words.map(w => {{
      const entry = DATA.find(d => d.technique === tech && d.word === w);
      return entry ? entry.time_ns : null;
    }}),
    backgroundColor: COLORS[i % COLORS.length],
  }}));
  return {{ labels: words, datasets }};
}}

function buildByTech() {{
  const words = getWords();
  const techniques = getTechniques();
  const datasets = words.map((w, i) => ({{
    label: w,
    data: techniques.map(tech => {{
      const entry = DATA.find(d => d.technique === tech && d.word === w);
      return entry ? entry.time_ns : null;
    }}),
    backgroundColor: COLORS[i % COLORS.length],
  }}));
  return {{ labels: techniques, datasets }};
}}

function showView(view) {{
  currentView = view;
  document.getElementById('btn-word').classList.toggle('active', view === 'word');
  document.getElementById('btn-tech').classList.toggle('active', view === 'tech');
  render();
}}

function render() {{
  const cfg = currentView === 'word' ? buildByWord() : buildByTech();
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chart'), {{
    type: 'bar',
    data: cfg,
    options: {{
      responsive: true,
      skipNull: true,
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const v = ctx.parsed.y;
              if (v == null) return null;
              return ctx.dataset.label + ': ' + (v < 1 ? v.toFixed(3) + ' ns' : v.toFixed(2) + ' ns');
            }}
          }}
        }},
        legend: {{ position: 'top' }},
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: 'Time (ns)' }},
        }},
        x: {{
          title: {{ display: true, text: currentView === 'word' ? 'Word type' : 'Technique' }},
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
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    entries = parse(lines)
    if not entries:
        print("No benchmark results found in input.", file=sys.stderr)
        sys.exit(1)

    data = [
        {"technique": t, "word": w, "time_ns": round(v, 4)}
        for t, w, v in entries
    ]

    html = HTML_TEMPLATE.format(data_json=json.dumps(data, indent=2))

    out = "select_in_word.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out} with {len(data)} benchmarks.", file=sys.stderr)


if __name__ == "__main__":
    main()
