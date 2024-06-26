#!/usr/bin/env bash

## Rust
rm -rf ../target/bench_like_cpp/
cargo run --release --bin bench_like_cpp -- select;
./move_benches.sh ../target/bench_like_cpp/ ./runs/c++_vs_rust/rust/select/

rm -rf ../target/bench_like_cpp/
cargo run --release --bin bench_like_cpp -- select_non_uniform
./move_benches.sh ../target/bench_like_cpp/ ./runs/c++_vs_rust/rust/select_non_uniform/

rm -rf ../target/bench_like_cpp/
cargo run --release --bin bench_like_cpp -- rank
./move_benches.sh ../target/bench_like_cpp/ ./runs/c++_vs_rust/rust/rank/


## C++
cd ../../sux
make ranksel

rm -rf ./bench-results/
python3 ./bench-scripts.py select
../sux-rs/scripts/move_benches.sh ./bench-results/ ../sux-rs/scripts/runs/c++_vs_rust/cpp/select/

python3 ./bench-scripts.py select_non_uniform
../sux-rs/scripts/move_benches.sh ./bench-results/ ../sux-rs/scripts/runs/c++_vs_rust/cpp/select_non_uniform/

rm -rf ./bench-results/
python3 ./bench-scripts.py rank
../sux-rs/scripts/move_benches.sh ./bench-results/ ../sux-rs/scripts/runs/c++_vs_rust/cpp/rank/

# Plot the benchmarks
cd ../sux-rs/python-scripts
python3 plot-benches.py --cpp-vs-rust