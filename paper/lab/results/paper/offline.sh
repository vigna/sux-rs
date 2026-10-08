#!/usr/bin/env bash
# Offline construction (Section 4 of the paper): time and peak memory of
# PHast-R built in memory and offline, with one and with THREADS threads, on
# each key set of SIZES above 10^7, and offline only on 10^10 keys if TMPDIR
# has enough free space (the offline construction writes 16 bytes per key in
# TMPDIR). Memory is reported as the peak of the memory allocated by the
# process (which, in memory, includes the eight bytes per key of the keys)
# and as the peak resident set size reported by GNU time (BSD time on
# macOS). Finally, it checks on the largest key set of SIZES that the two
# constructions yield the same structure.
# Output lines: <keys> <threads> <online|offline> <bits/key> <ns/key>
# <allocated MiB> <resident KiB>
# Usage: offline.sh <directory of the binaries> <output file>
# Environment: CPU1, CPUS, THREADS, SIZES; see redo.sh.
. "$(dirname "$0")/configs.sh"
B=${1:-../../target/release}
O=${2:-offline.txt}
CPU1=${CPU1:-2}
CPUS=${CPUS:-0-7}
THREADS=${THREADS:-8}
SIZES=${SIZES:-"10000000 100000000 1000000000"}
LARGE=$(echo $SIZES | tr ' ' '\n' | grep -v '^10000000$' | tr '\n' ' ')
# Free space in TMPDIR, in GB
free_disk() { df -Pk "${TMPDIR:-/tmp}" | awk 'NR == 2 { print int($4 / 1048576) }'; }
# Runs a command and prints its peak memory usage as GNU time does (on
# macOS, converting the output of BSD time)
peak() {
  if /usr/bin/time -v true >/dev/null 2>&1; then
    /usr/bin/time -v "$@" 2>&1
  else
    /usr/bin/time -l "$@" 2>&1 | awk '/maximum resident set size/ { printf "\tMaximum resident set size (kbytes): %d\n", $1 / 1024; next } { print }'
  fi
}
# Runs a construction: run <keys> <threads> <mode>
run() {
  local cores=$CPU1
  [ $2 -gt 1 ] && cores=$CPUS
  peak $(pinning $cores) env RAYON_NUM_THREADS=$2 $B/offline -n $1 -m $3 |
    awk -v n=$1 -v t=$2 -v m=$3 '
      /bits\/key/ { for (i = 1; i <= NF; i++) { if ($i == "bits/key") bits = $(i - 1); if ($i == "ns/key") ns = $(i - 1) } }
      /peak allocated memory/ { alloc = $4 }
      /Maximum resident set size/ { rss = $NF }
      END { print n, t, m, bits, ns, alloc, rss }'
}
{
  for n in $LARGE; do
    for t in 1 $THREADS; do
      run $n $t online
      run $n $t offline
    done
  done
  # Offline on 10^10 keys (160 GB on disk), unless already done
  if ! echo $LARGE | grep -qw 10000000000 && [ $(free_disk) -ge 170 ]; then
    for t in 1 $THREADS; do run 10000000000 $t offline; done
  fi
  n=$(echo $LARGE | tr ' ' '\n' | tail -1)
  pin $CPUS env RAYON_NUM_THREADS=$THREADS $B/offline -n $n -m both | grep -q '^identical$' &&
    echo "$n identical" || echo "$n DIFFERENT"
} > $O
