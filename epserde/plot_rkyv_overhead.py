#!/usr/bin/env python3
"""Paper figure: the query cost of a zero-copy rkyv archive relative to an
ε-serde image of the same structure.

Both are memory-mapped images of the identical unaligned structure, so the
comparison isolates the cost of the format itself rather than of serializing
at all.

USAGE.  It draws from a `samples.json` produced by `extract_samples.py`:

    ./plot_rkyv_overhead.py samples.json -o rkyv_overhead

Options:

    --width 5.478             full acmsmall \\textwidth (default 3.5)
    --color                   colour instead of grayscale
    --op rank --op succ       a subset of the operations
    --baseline-arm unaligned  measure against memory instead of ε-serde
    --font "Some Family"      a text face other than the paper's

TYPOGRAPHY.  The target document is ``epserde.tex``, which is
``\\documentclass[acmsmall,review,anonymous]{acmart}``: Linux Libertine text,
Inconsolata typewriter, newtxmath math, a 10pt base size and a 5.478in text
width.  This script therefore sets Linux Libertine for both text and math and
defaults to 8pt, which is ``\\footnotesize`` in a 10pt document.  Include the
result at natural size -- ``\\includegraphics{rkyv_overhead}`` with no scaling
-- or the type will no longer be 8pt.

Glyphs are embedded as TrueType (``pdf.fonttype = 42``) rather than Type 3,
which some publishers reject.

Linux Libertine must be installed (Fedora: ``linux-libertine-fonts``; Debian:
``fonts-linuxlibertine``; MacPorts: ``texlive-fonts-extra``).  If matplotlib
still cannot see it, clear the font cache: ``rm ~/.cache/matplotlib/fontlist-*.json``.
"""

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OPS = ["get", "succ", "pred", "rank"]
ARMS = ["aligned", "unaligned", "eps", "rkyv"]

#: How each representation is named in prose.
ARM_PROSE = {"aligned": "aligned", "unaligned": "unaligned",
             "eps": "$\\varepsilon$-serde", "rkyv": "rkyv"}

#: Text face of the target document, most-preferred first.
SERIF = ["Linux Libertine O", "Linux Libertine", "Libertinus Serif"]

#: Typewriter face of the target document. The operation names are Rust
#: methods, which the paper sets in \texttt; if it is installed we match it,
#: otherwise the tick labels stay in the text face.
MONO = ["Inconsolata", "Inconsolata LGC"]

#: acmsmall \textwidth: 6.75in paper less 46pt inner and 46pt outer margins.
ACMSMALL_TEXTWIDTH = 6.75 - 92 / 72.27

#: criterion parameter labels, e.g. `1M_l=8` or `1G_l=8_t=4`.
PARAM_RE = re.compile(r"^(\d+[KMG]?)_l=(\d+)(?:_t=(\d+))?$")


# ---------------------------------------------------------------- input
def load(source):
    """Per-benchmark means from a `samples.json` of raw timings.

    criterion's mean point estimate is just the mean of the per-iteration
    times, so averaging the raw timings reproduces it exactly.
    """
    path = Path(source)
    if not path.is_file():
        sys.exit(f"error: no such file: {path}\n"
                 f"  Produce one with:  extract_samples.py target/criterion -o {path}")
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        sys.exit(f"error: {path} is not a samples file (expected an object keyed "
                 f"`group/arm/param`).\n"
                 f"  Produce one with:  extract_samples.py target/criterion -o {path}")
    rows = []
    for key, v in obj.items():
        group, arm, param = key.split("/")
        m = PARAM_RE.match(param)
        if not m:
            continue
        per = [t / i for t, i in zip(v["times"], v["iters"])]
        rows.append({
            "group": group, "arm": arm, "n": m.group(1), "l": int(m.group(2)),
            "t": int(m.group(3)) if m.group(3) else None, "mean": st.mean(per),
        })
    if not rows:
        sys.exit(f"error: {path} holds no recognisable benchmarks")
    return rows


def index_of(rows):
    return {(r["group"], r["arm"], r["n"], r["l"]): r["mean"]
            for r in rows if r["t"] is None}


def n_value(label):
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return (int(label[:-1]) * units[label[-1].upper()]
            if label and label[-1].upper() in units else int(label))


def scale_label(label):
    v = n_value(label)
    return f"$n = 2^{{{v.bit_length() - 1}}}$" if v and not (v & (v - 1)) else f"$n = {label}$"


def overhead(index, op, n, l, subject, baseline):
    """Percentage by which `subject` is slower than `baseline`."""
    g = f"ef_{op}_unchecked"
    try:
        return (index[(g, subject, n, l)] / index[(g, baseline, n, l)] - 1) * 100
    except KeyError as e:
        sys.exit(f"error: the dataset has no measurement for {e.args[0]}")


# ---------------------------------------------------------------- style
def pick_serif(override):
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ([override] if override else SERIF):
        if name in available:
            return name
    sys.exit(
        f"error: none of {[override] if override else SERIF} is available to "
        f"matplotlib, so the figure would not match the paper.\n"
        f"  Fedora: sudo dnf install linux-libertine-fonts\n"
        f"  Debian: sudo apt install fonts-linuxlibertine\n"
        f"  then:   rm ~/.cache/matplotlib/fontlist-*.json\n"
        f"  or pass --font to name a face you do have."
    )


def pick_mono():
    available = {f.name for f in fm.fontManager.ttflist}
    return next((n for n in MONO if n in available), None)


def style(serif, font_size):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif],
        # newtxmath is not available to matplotlib; setting the text face for
        # math too keeps the few mathematical labels consistent with the body.
        "mathtext.fontset": "custom",
        "mathtext.rm": serif,
        "mathtext.it": f"{serif}:italic",
        "mathtext.bf": f"{serif}:bold",
        # Unused here, but an unset cal face makes mathtext warn on every run.
        "mathtext.cal": f"{serif}:italic",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "font.size": font_size,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.handlelength": 1.2,
        "legend.handleheight": 0.7, "legend.columnspacing": 1.0,
        "legend.borderpad": 0.0, "legend.borderaxespad": 0.3,
    })


# ---------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=str(here / "samples.json"),
                    help="a samples.json of raw timings (default: samples.json here)")
    ap.add_argument("-o", "--out", default=str(here / "rkyv_overhead"),
                    help="output path without extension")
    ap.add_argument("--op", action="append", choices=OPS, dest="ops",
                    help="restrict to these operations (repeatable)")
    ap.add_argument("--subject", choices=ARMS, default="rkyv",
                    help="the representation whose cost is plotted")
    ap.add_argument("--baseline-arm", choices=ARMS, default="eps", dest="ref",
                    help="the representation it is measured against")
    ap.add_argument("--width", type=float, default=3.5,
                    help=f"inches; acmsmall \\textwidth is {ACMSMALL_TEXTWIDTH:.3f}")
    ap.add_argument("--height", type=float, default=2.2, help="inches")
    ap.add_argument("--font-size", type=float, default=8.0,
                    help="points; 8 is \\footnotesize in a 10pt document")
    ap.add_argument("--font", help="text face to use instead of the paper's")
    ap.add_argument("--color", action="store_true")
    args = ap.parse_args()

    rows = load(args.source)
    index = index_of(rows)
    query = [r for r in rows if not r["group"].startswith("ef_build")]
    ops = args.ops or [o for o in OPS if any(r["group"] == f"ef_{o}_unchecked" for r in query)]
    ls = sorted({r["l"] for r in query})
    scales = sorted({r["n"] for r in query}, key=n_value)
    if not ops or not scales:
        sys.exit("error: the dataset contains no unchecked query measurements")

    serif = pick_serif(args.font)
    mono = pick_mono()
    style(serif, args.font_size)

    fills = ["#9ec5f4", "#2a78d6"] if args.color else ["#d4d4d4", "#585858"]
    if len(scales) > len(fills):  # more sizes than the two-tone scheme covers
        fills = [str(v) for v in np.linspace(0.85, 0.3, len(scales))]

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    x = np.arange(len(ops))
    bw = 0.68 / len(scales)
    top = 0.0

    for k, n in enumerate(scales):
        means, lo, hi = [], [], []
        for op in ops:
            vs = [overhead(index, op, n, l, args.subject, args.ref) for l in ls]
            m = st.mean(vs)
            means.append(m)
            lo.append(m - min(vs))
            hi.append(max(vs) - m)
        pos = x + (k - (len(scales) - 1) / 2) * bw
        ax.bar(pos, means, bw, label=scale_label(n), color=fills[k],
               edgecolor="black", linewidth=0.5, zorder=3)
        ax.errorbar(pos, means, yerr=[lo, hi], fmt="none", ecolor="black",
                    elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=4)
        for p, m, h in zip(pos, means, hi):
            ax.annotate(f"{m:.1f}", (p, m + h), textcoords="offset points",
                        xytext=(0, 2), ha="center",
                        fontsize=args.font_size - 1.5, zorder=5)
        top = max(top, max(m + h for m, h in zip(means, hi)))

    ax.set_xticks(x)
    ax.set_xticklabels(ops, **({"fontfamily": mono} if mono else {}))
    ax.set_ylabel(f"{ARM_PROSE[args.subject]} overhead over {ARM_PROSE[args.ref]} (%)")
    ax.set_ylim(0, top * 1.20)
    ax.set_xlim(-0.55, len(ops) - 0.45)
    ax.yaxis.grid(True, color="#d8d8d8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.0, pad=2)
    ax.legend(ncol=len(scales), loc="upper left", fontsize=args.font_size - 0.5)

    fig.tight_layout(pad=0.2)
    pdf, png = f"{args.out}.pdf", f"{args.out}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)

    print(f"read   {args.source} ({len(rows)} measurements)")
    print(f"plot   {args.subject} against {args.ref}; "
          f"{len(ops)} ops x {len(scales)} sizes, averaged over l in {ls}")
    print(f"font   {serif} at {args.font_size}pt"
          + (f"; operation names in {mono}" if mono else
             "; operation names in the text face (Inconsolata not installed)"))
    print(f"size   {args.width:.3f} x {args.height:.3f} in")
    print(f"wrote  {pdf}\nwrote  {png}")


if __name__ == "__main__":
    main()
