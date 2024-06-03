import matplotlib.pyplot as plt
import os
import json
import math
import pandas as pd
import numpy as np
import argparse
import os

colors = ['b', 'g', 'r', 'c', 'm', 'purple',
          'gold', 'teal', 'orange', 'brown', 'pink']
markers = np.array(["v", "o", "+", "*", "^", "s", "D", "x"])


def load_csv_benches(path):
    df = pd.read_csv(path, header=None, names=[
                     "size", "dense", "time"])
    return df


def load_criterion_benches(base_path):
    benches_list = []

    for dir in sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]):
        run_name = dir.split("_")
        data = {}

        data["size"] = int(run_name[0], 10)
        data["dense"] = float(run_name[1])
        data["rep"] = int(run_name[2], 10)

        path = os.path.join(base_path, dir, "new/estimates.json")
        with open(path, "r") as f:
            estimates = json.load(f)
            data["time"] = estimates["mean"]["point_estimate"]
        benches_list.append(data)

    benches_df = pd.DataFrame(benches_list)

    benches_df = benches_df.groupby(
        ["size", "dense"], as_index=False)["time"].mean()

    benches_df = benches_df.sort_values(by="size", ignore_index=True)
    return benches_df


def compare_benches(benches, compare_name, op_type):
    num_densities = len(benches[0][0]["dense"].unique())
    fig, ax = plt.subplots(1, num_densities, constrained_layout=True,
                           sharex=True, sharey=True, squeeze=False)
    fig.set_size_inches(10, 6)
    fig.text(0.5, -0.02, 'size [num of bits]', ha='center', va='center')
    fig.text(-0.01, 0.5, f'time [ns/{op_type}]', ha='center',
             va='center', rotation='vertical')

    for i, (bench, bench_name) in enumerate(benches):
        for d, (name, group) in enumerate(bench.groupby("dense")):
            ax[0, d].plot(group["size"], group["time"], label=bench_name,
                          color=colors[i], marker=markers[i], markersize=3, linewidth=1.0)
            ax[0, d].set_title(f"density={float(name)*100}%")
            ax[0, d].grid(True)
            ax[0, d].set_xscale("log")
            ax[0, d].set_yscale("log")

    times = np.sort(np.concatenate(
        list(map(lambda x: x[0]["time"].unique(), benches)), axis=0))
    ticks = np.logspace(np.log10(times[0]), np.log10(times[-1]), num=8)
    ticks = list(map(lambda x: math.ceil(x), ticks))
    ax[0, 0].set_yticks(ticks)
    ax[0, 0].set_yticklabels(ticks)
    ax[0, 0].yaxis.set_minor_locator(plt.NullLocator())

    h1, _ = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles=h1, loc='upper center', bbox_to_anchor=(
        0.5, -0.04), fancybox=True, shadow=True, ncol=3)

    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    plots_dir = os.path.join(scripts_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, "{}.svg".format(compare_name)),
                format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_cpp_vs_rust():
    cpp_dir = "../bash-scripts/runs/c++_vs_rust/cpp"
    rust_dir = "../bash-scripts/runs/c++_vs_rust/rust"

    cpp_rank_csvs = [f for f in os.listdir(os.path.join(cpp_dir, "rank")) if f.endswith(
        ".csv")]
    cpp_select_csvs = [f for f in os.listdir(os.path.join(cpp_dir, "select")) if f.endswith(
        ".csv")]
    cpp_select_non_uniform_csvs = [f for f in os.listdir(os.path.join(cpp_dir, "select_non_uniform")) if f.endswith(
        ".csv")]

    rust_rank_csvs = [f for f in os.listdir(os.path.join(rust_dir, "rank")) if f.endswith(
        ".csv")]
    rust_select_csvs = [f for f in os.listdir(os.path.join(rust_dir, "select")) if f.endswith(
        ".csv")]
    rust_select_non_uniform_csvs = [f for f in os.listdir(os.path.join(rust_dir, "select_non_uniform")) if f.endswith(
        ".csv")]

    for cpp, rust in zip(cpp_rank_csvs, rust_rank_csvs):
        assert cpp == rust
        compare_benches(
            [(load_csv_benches(os.path.join(cpp_dir, "rank", cpp)), f"{cpp[:-4]}_cpp"),
             (load_csv_benches(os.path.join(rust_dir, "rank", rust)),
              f"{rust[:-4]}_rust")
             ], f"{cpp[:-4]}_cpp_vs_rust", "rank")

    for cpp, rust in zip(cpp_select_csvs, rust_select_csvs):
        assert cpp == rust
        compare_benches(
            [(load_csv_benches(os.path.join(cpp_dir, "select", cpp)), f"{cpp[:-4]}_cpp"),
             (load_csv_benches(os.path.join(rust_dir, "select", rust)),
              f"{rust[:-4]}_rust")
             ], f"{cpp[:-4]}_cpp_vs_rust", "select")

    for cpp, rust in zip(cpp_select_non_uniform_csvs, rust_select_non_uniform_csvs):
        assert cpp == rust
        compare_benches(
            [(load_csv_benches(os.path.join(cpp_dir, "select_non_uniform", cpp)), f"{cpp[:-4]}_cpp"),
             (load_csv_benches(os.path.join(rust_dir, "select_non_uniform", rust)),
              f"{rust[:-4]}_rust")
             ], f"{cpp[:-4]}_cpp_vs_rust", "select")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot benchmark results.')

    group1 = parser.add_argument_group()
    group1.add_argument('--cpp-vs-rust', action='store_true',
                        help='Compare C++ vs Rust benchmarks')

    group2 = parser.add_argument_group()
    group2.add_argument('--op_type', choices=[
        'rank', 'select'], help='Operation type')
    group2.add_argument('--benches_path', type=str,
                        help='Path to the benches directory')
    group2.add_argument('--plot_name', type=str, help='Name of the plot')

    args = parser.parse_args()
    if args.cpp_vs_rust:
        plot_cpp_vs_rust()
    else:
        if not args.op_type or not args.benches_path or not args.plot_name:
            parser.print_help()
            exit(1)
        op_type = args.op_type
        benches_path = args.benches_path
        plot_name = args.plot_name
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
                (load_criterion_benches(os.path.join(benches_path, bench_dir)), bench_dir))
        compare_benches(benches, plot_name, op_type)
