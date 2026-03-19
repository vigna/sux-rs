#!/usr/bin/env python3
"""Plot benchmark results for the sux rank/select structures.

This script reads criterion's JSON output (stored under
``target/criterion/``) and produces comparison plots for the rank and
select structures benchmarked by ``benches/sux/``.

Prerequisites
-------------
- Python 3.8+
- matplotlib, pandas, numpy (``pip install matplotlib pandas numpy``)

Usage
-----
First, run the benchmarks.  The ``sux`` benchmark binary has its own CLI
(see ``cargo bench --bench sux -- --help``); typical invocations::

    # Benchmark Rank9 at density 0.5
    cargo bench --bench sux -- Rank9 -d 0.5

    # Compare several select structures
    cargo bench --bench sux -- SelectSmall SelectAdapt0 -d 0.5 \\
        -l 100000,1000000,10000000

Then generate plots::

    python3 python/plot_benches.py \\
        --benches-path ./target/criterion/ \\
        --plot-dir plots

    # Include Pareto-front plots (time vs. memory overhead):
    python3 python/plot_benches.py \\
        --benches-path ./target/criterion/ \\
        --plot-dir plots \\
        --pareto

Criterion directory layout
--------------------------
The ``sux`` benchmark uses criterion *benchmark groups*, so results are
stored in a two-level directory structure::

    target/criterion/
    └── <StructureName>/          # e.g. "Rank9", "SelectAdapt0"
        ├── <size>_<density>/     # e.g. "1000000_0.5"
        │   └── new/
        │       └── estimates.json    # ← this script reads mean.point_estimate
        ├── mem_cost.json             # written by the benchmark (for --pareto)
        └── ...

Each parameter directory name encodes ``size_density``.

What it produces
----------------
- ``plots/plot.svg`` — line plots of time (ns/op) vs. bit-vector size,
  one subplot per density.  Each structure is a separate line/color.
  Structures are added incrementally: each benchmark group you run is
  overlaid on the previous ones.
- ``plots/csv/<StructureName>.csv`` — raw data for each structure.
- ``plots/pareto_<density>.svg`` (with ``--pareto``) — Pareto front of
  time vs. memory overhead at a given density, one line per size.

"""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import math
import pandas as pd
import numpy as np
import argparse

# colors = plt.cm.tab20(np.linspace(0, 1, 30))
colors = ["#8FBC8F", "#4682B4", "#DDA0DD", "#CD5C5C", "#F4A460",
          "#6B8E23", "#B0C4DE", "#DA70D6", "#D2B48C", "#87CEFA",
          "#AFEEEE", "#E6E6FA", "#FFA07A", "#20B2AA", "#778899",
          "#BDB76B", "#FFDEAD", "#BC8F8F", "#6495ED", "#F0E68C"]
markers = ['+', 'o', '^', 'x', 'v', '*', '>', '<', '3', 'X',
           'd', 'p', 's', 'd', 'H', '1', '2', 'D', '4', '8', 'P', 'h', '8', 'v', '^', '<']


def load_criterion_benches(base_path, load_mem_cost=False):
    benches_list = []

    for dir in sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]):
        parts = dir.split("_")
        # Parameter directories are named "<size>_<density>".
        if len(parts) < 2:
            continue
        try:
            size = int(parts[0], 10)
            dense = float(parts[1])
        except (ValueError, IndexError):
            continue

        path = os.path.join(base_path, dir, "new/estimates.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            estimates = json.load(f)
        benches_list.append({
            "size": size,
            "dense": dense,
            "time": estimates["mean"]["point_estimate"],
        })

    benches_df = pd.DataFrame(benches_list)
    if benches_df.empty:
        return benches_df

    if load_mem_cost:
        mem_json = os.path.join(base_path, "mem_cost.json")
        if os.path.exists(mem_json):
            with open(mem_json, "r") as f:
                mem_data = json.load(f)
            mem_cost_df = pd.DataFrame(mem_data)
            benches_df = pd.merge(benches_df, mem_cost_df,
                                  how="left", on=["size", "dense"])

    benches_df = benches_df.sort_values(by="size", ignore_index=True)
    return benches_df


def compare_benches(benches, plots_dir, nonuniform):
    num_densities = len(benches[0][0]["dense"].unique())
    fig, ax = plt.subplots(1, num_densities, constrained_layout=True,
                           sharex=True, sharey=True, squeeze=False)
    fig.set_size_inches(10, 5)
    fig.text(0.5, -0.02, 'size [num of bits]', ha='center', va='center')
    fig.text(-0.01, 0.5, 'time [ns/op]', ha='center',
             va='center', rotation='vertical')

    for i, (bench, bench_name) in enumerate(benches):
        for d, (name, group) in enumerate(bench.groupby("dense")):
            ax[0, d].plot(group["size"], group["time"], label=bench_name.removesuffix("_nonuniform"),
                          color=colors[i], marker=markers[i], markersize=3, linewidth=1.0)
            if nonuniform:
                ax[0, 0].set_title(f"non-uniform density = {round((0.9 * 0.01)*100, 2)}% | {round((0.9 * 0.99)*100, 2)}%")
            else:
                ax[0, d].set_title(f"density={float(name)*100}%")
            ax[0, d].grid(True)
            ax[0, d].set_xscale("log")

    times = np.sort(np.concatenate(
        list(map(lambda x: x[0]["time"].unique(), benches)), axis=0))
    ticks = np.linspace(0, times[-1]*1.05, num=8)
    ticks = list(map(lambda x: math.ceil(x), ticks))
    ax[0, 0].set_yticks(ticks)
    ax[0, 0].set_yticklabels(ticks)
    ax[0, 0].yaxis.set_minor_locator(plt.NullLocator())

    h1, _ = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles=h1, loc='upper center', bbox_to_anchor=(
        0.5, -0.04), fancybox=True, shadow=True, ncol=3)


    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, "plot.svg"),
                format="svg", bbox_inches="tight")
    plt.close(fig)

    # save pandas dataframes to csv
    csv_dir = os.path.join(plots_dir, "csv")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    for i, (bench, bench_name) in enumerate(benches):
        bench.sort_values(["dense", "size"]).to_csv(
            os.path.join(csv_dir, "{}.csv".format(bench_name)), index=False)

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
            np.any(costs[i+1:] > c, axis=1))
    return is_efficient


def draw_pareto_front(benches, plots_dir, op_type, density=0.5):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(10, 6)
    ax.set_ylabel("memory cost [%]")
    ax.set_xlabel(f"time [ns/{op_type}]")

    bench_per_len = []
    lens = benches[0][0]["size"].unique()
    for l in lens:
        bench_per_len.append([])
        for bench, _ in benches:
            b = bench[bench["dense"] == density]
            b = b[b["size"] == l]
            bench_per_len[-1].append(np.ndarray.flatten(
                b[["time", "mem_cost"]].values))

    for i, bench in enumerate(bench_per_len):
        bench = np.array(bench)
        pareto = bench[is_pareto_efficient(bench)]
        pareto = pareto[np.argsort(pareto[:, 0])]
        ax.plot(pareto[:, 0], pareto[:, 1], label=f"size={lens[i]}",
                color=colors[i], linewidth=1.0)
        for j, p in enumerate(bench):
            # if p in pareto:
            plt.scatter(p[0], p[1], color=colors[i],
                        marker=markers[j], s=20)
    ax.grid(True)
    artists = []

    for i, l in enumerate(lens):
        artists.append(mpatches.Patch(
            color=colors[i], label="size={}".format(l, 1)))

    for i, bench in enumerate(benches):
        artists.append(
            Line2D([0], [0], color='black', marker=markers[i], markersize=5, label=bench[1]))

    ax.legend(handles=artists, loc='best', fancybox=True, shadow=False, ncol=1)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, "pareto.svg"), format="svg", dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot benchmark results.')

    group1 = parser.add_argument_group()

    group2 = parser.add_argument_group()
    group2.add_argument('--benches-path', type=str,
                        help='Path to the benches directory')
    group2.add_argument('--plot-dir', type=str, help='Directory containing the plot(s)')
    parser.add_argument("--pareto",
                        action="store_true", help="Draw pareto front")

    args = parser.parse_args()
    if not args.benches_path or not args.plot_dir:
        parser.print_help()
        exit(1)
    benches_path = args.benches_path
    plot_dir = args.plot_dir
    if not os.path.exists(benches_path):
        print("The benches directory does not exist.")
        exit(1)
    bench_dirs = sorted([d for d in os.listdir(
        benches_path) if os.path.isdir(os.path.join(benches_path, d))])
    if len(bench_dirs) == 0:
        print("The benches directory is empty.")
        exit(1)
    benches = []
    for bench_dir in bench_dirs:
        benches.append(
            (load_criterion_benches(os.path.join(benches_path, bench_dir), load_mem_cost=args.pareto), bench_dir))
        compare_benches(benches, plot_dir, bench_dir.endswith("nonuniform"))
    if args.pareto:
        densities = benches[0][0]["dense"].unique()
        for d in densities:
            draw_pareto_front(benches, f"pareto_{d}", d)
