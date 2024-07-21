#!/usr/bin/env bash

rm -rf ../target/criterion/
cargo bench --bench my_benchmarks -- select
./move_benches.sh ../target/criterion/ ./runs/select/
python3 ../python-scripts/plot-benches.py select ./runs/select/ select_benches

rm -rf ../target/criterion/
cargo bench --bench my_benchmarks -- select_non_uniform
./move_benches.sh ../target/criterion/ ./runs/select_non_uniform/
python3 ../python-scripts/plot-benches.py select ./runs/select_non_uniform/ select_non_unif_benches