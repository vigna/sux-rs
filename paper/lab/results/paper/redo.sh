#!/usr/bin/env bash
# Redoes all the experiments of the paper on this machine: builds the lab
# (in a target directory of its own, as the code is optimized for the local
# CPU), detects cores and memory, and writes the results to
# results/paper/<hostname>/ (run.csv, large/, env.txt). The reference
# implementation must be in ../../../bsuccinct-rs (see lab/Cargo.toml).
# Usage: redo.sh [max keys]      (default: 10^9 if there are 64 GB free)
# It takes a few hours: run it under nohup or in a terminal multiplexer.
set -e
cd "$(dirname "$0")/../.."
HOST=$(hostname -s)
O=$PWD/results/paper/$HOST
mkdir -p $O
export CARGO_TARGET_DIR=target-$HOST
cargo build --release --bin cmp --bin qsplit --bin btime
B=$PWD/$CARGO_TARGET_DIR/release

# Distinct cores (the first hardware thread of each), and the memory
CORES=$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | tr '\n' ' ')
NCORES=$(echo $CORES | wc -w)
THREADS=$NCORES; [ $THREADS -gt 8 ] && THREADS=8
CPU1=$(echo $CORES | awk '{print ($3 != "") ? $3 : $1}')
FREE=$(free -g | awk '/^Mem:/ {print $7}')
MAX=${1:-$([ $FREE -ge 64 ] && echo 1000000000 || echo 100000000)}
SIZES=$(for n in 10000000 100000000 1000000000; do if [ $n -le $MAX ]; then echo -n "$n "; fi; done)
{
  echo "date: $(date)"; echo "host: $HOST"; echo "cores: $CORES"; echo "threads: $THREADS"; echo "free memory (GB): $FREE"
  echo "sizes: $SIZES"; rustc -V; ldd --version | head -1
  lscpu
} > $O/env.txt
cd results/paper
export CPU1 CORES THREADS SIZES
CPUS=$(echo $CORES | tr ' ' '\n' | head -n $THREADS | tr '\n' ',' | sed 's/,$//')
export CPUS
bash run.sh $B/cmp $O/run
bash large.sh $B $O/large
echo "done: $O"
