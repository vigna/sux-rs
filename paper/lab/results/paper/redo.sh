#!/usr/bin/env bash
# Redoes all the experiments of the paper on this machine: builds the lab
# (in a target directory of its own, as the code is optimized for the local
# CPU), detects cores and memory, and writes the results to
# results/paper/<hostname>/ (env.txt, run.csv, large/, spread.csv). The
# reference implementation must be in ../../../bsuccinct-rs (see
# lab/Cargo.toml).
# Usage: redo.sh [max keys]      (default: 10^9 if there are 64 GB free)
# It takes about six hours on a c7i.metal-24xl (run.sh four hours and a
# half, large.sh one hour, spread.sh ten minutes): run it under nohup or in a
# terminal multiplexer. Experiments on 10^10 keys must be requested
# explicitly (redo.sh 10000000000): they need 250 GB of free memory and
# about twenty more hours, most of them spent building the PHast
# structures. Setting BUILDS overrides the number of constructions of each
# configuration (see configs.sh).
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
if [ $MAX -ge 10000000000 ] && [ $FREE -lt 250 ]; then echo "10^10 keys need 250 GB of free memory ($FREE GB free)"; exit 1; fi
SIZES=$(for n in 10000000 100000000 1000000000 10000000000; do if [ $n -le $MAX ]; then echo -n "$n "; fi; done)
{
  echo "date: $(date)"; echo "host: $HOST"; echo "cores: $CORES"; echo "threads: $THREADS"; echo "free memory (GB): $FREE"
  echo "sizes: $SIZES"; rustc -V; ldd --version | head -1
  lscpu
} > $O/env.txt
cd results/paper
export CPU1 CORES THREADS SIZES BUILDS
CPUS=$(echo $CORES | tr ' ' '\n' | head -n $THREADS | tr '\n' ',' | sed 's/,$//')
export CPUS
bash run.sh $B/cmp $O/run
bash large.sh $B $O/large
bash spread.sh $B/cmp $O/spread
echo "done: $O"
