#!/usr/bin/env python3

import subprocess
from tqdm import tqdm

benches = [
    "rank9",
    "rank11",
    "simple_select",
    "rank9sel",
    "simple_select_non_uniform",
    "rank9sel_non_uniform",
    "rng",
    "rng_non_uniform",
]

if __name__ == "__main__":
    for bench in tqdm(benches):
        subprocess.run(
            "cargo bench --bench my_benchmarks -- {} --noplot --quiet --exact --nocapture".format(bench), shell=True)
