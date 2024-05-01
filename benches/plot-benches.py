#!/usr/bin/env python3

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import math
import pandas as pd
import numpy as np

colors = ['b', 'g', 'r', 'c', 'm']
markers = ["v", "o", "+", "*", "^", "s", "D", "x"]
num_of_densities = 3


def load_benches(base_path):
    benches_list = []

    for dir in os.listdir(base_path):
        if not os.path.isdir(base_path + dir):
            continue

        run_name = dir.split("_")
        data = {}

        data["size"] = int(run_name[0], 10)
        data["dense"] = float(run_name[1])
        data["rep"] = int(run_name[2], 10)

        path = base_path + dir + "/new/estimates.json"
        with open(path, "r") as f:
            estimates = json.load(f)
            data["mean"] = estimates["mean"]["point_estimate"]

        benches_list.append(data)

    benches_df = pd.DataFrame(benches_list)
    benches_df = benches_df.groupby(
        ["size", "dense"], as_index=False)["mean"].mean()
    benches_df = benches_df.sort_values(by="size", ignore_index=True)
    return benches_df


def compare_benches(benches, compare_name):
    fig, ax = plt.subplots(1, len(
        benches), constrained_layout=True, sharex=True, sharey=True, squeeze=False)
    fig.set_size_inches(10, 6)
    fig.text(0.5, -0.02, 'size [num of bits]', ha='center', va='center')
    fig.text(-0.01, 0.5, 'time [ns]', ha='center',
             va='center', rotation='vertical')

    for i, (bench, bench_name) in enumerate(benches):
        for d, (name, group) in enumerate(bench.groupby("dense")):
            ax[0, i].plot(group["size"], group["mean"], label=f"density={float(name)*100}%",
                          color=colors[d], marker="o", markersize=3, linewidth=1.0)
        ax[0, i].set_title(bench_name)
        ax[0, i].grid(True)
        ax[0, i].set_xscale("log")
        ax[0, i].set_yscale("log")

    means = np.sort(np.concatenate(
        list(map(lambda x: x[0]["mean"].unique(), benches)), axis=0))
    ticks = np.logspace(np.log10(means[0]), np.log10(means[-1]), num=6)
    ticks = list(map(lambda x: math.ceil(x), ticks))
    ax[0, 0].set_yticks(ticks)
    ax[0, 0].set_yticklabels(ticks)

    h1, l1 = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles=h1, loc='upper center', bbox_to_anchor=(
        0.5, -0.04), fancybox=True, shadow=True, ncol=5)

    plt.savefig("./plots/{}.svg".format(compare_name),
                format="svg", bbox_inches="tight")
    plt.close(fig)


def compare_benches_same_plot(bench1, bench2, compare_name):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(10, 6)
    fig.text(0.5, -0.02, 'size [num of bits]', ha='center', va='center')
    fig.text(-0.01, 0.5, 'time [ns]', ha='center',
             va='center', rotation='vertical')

    for d, (name, group) in enumerate(bench1[0].groupby("dense")):
        ax.plot(group["size"], group["mean"], label=f"density={float(name)*100}%",
                color=colors[d], marker=markers[0], markersize=3, linewidth=1.0, linestyle="-")

    for d, (name, group) in enumerate(bench2[0].groupby("dense")):
        ax.plot(group["size"], group["mean"], label=f"density={float(name)*100}%",
                color=colors[d], marker=markers[1], markersize=3, linewidth=1.0, linestyle="--")

    means = []
    means.append(bench1[0]["mean"].unique())
    means.append(bench2[0]["mean"].unique())

    means = np.unique(np.concatenate(means), axis=0)
    ticks = np.logspace(np.log10(means[0]), np.log10(means[-1]), num=6)
    ticks = list(map(lambda x: round(x, 1), ticks))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.grid(True)
    ax.yaxis.set_minor_locator(plt.NullLocator())

    density25_patch = mpatches.Patch(color='blue', label='density=25%')
    density50_patch = mpatches.Patch(color='green', label='density=50%')
    density75_patch = mpatches.Patch(color='red', label='density=75%')

    marker1 = Line2D([0], [0], color='black',
                     linewidth=1.0, linestyle="-", marker=markers[0], markersize=4, label=bench1[1])
    marker2 = Line2D([0], [0], color='black',
                     linewidth=1.0, linestyle="--", marker=markers[1], markersize=4, label=bench2[1])

    handles = [density25_patch, density50_patch,
               density75_patch, marker1, marker2]

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(
        0.5, -0.04), fancybox=True, shadow=True, ncol=3)

    plt.draw_all()
    plt.savefig("./plots/{}.svg".format(compare_name),
                format="svg", bbox_inches="tight")
    plt.close(fig)


def compare_select():
    simple_benches = load_benches("../target/criterion/simple_select/")
    rank9sel_benches = load_benches("../target/criterion/rank9sel/")

    simple_non_uniform_benches = load_benches(
        "../target/criterion/simple_select_non_uniform/")
    rank9sel_non_uniform_benches = load_benches(
        "../target/criterion/rank9sel_non_uniform/")

    compare_benches([(simple_benches, "simple_select"),
                    (rank9sel_benches, "rank9sel"),
                    (simple_non_uniform_benches, "simple_select_non_uniform"),
                    (rank9sel_non_uniform_benches, "rank9sel_non_uniform")], "compare_select")


def compare_rank():
    rank9 = load_benches("../target/criterion/rank9/")
    rank10 = load_benches("../target/criterion/rank10/")
    rank11 = load_benches("../target/criterion/rank11/")
    rank12 = load_benches("../target/criterion/rank12/")
    rank16 = load_benches("../target/criterion/rank16/")

    compare_benches([(rank9, "rank9"), (rank10, "rank10"), (rank11, "rank11"), (rank12, "rank12"),
                    (rank16, "rank16")], "compare_rank")


if __name__ == "__main__":
    compare_select()
    compare_rank()
