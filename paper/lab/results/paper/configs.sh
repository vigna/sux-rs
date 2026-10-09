# Configurations and helpers shared by run.sh, large.sh and spread.sh
# (sourced, not run).
#
# Configurations are those of cmp: ref:<chooser>:<S>:<lambda> for the
# reference implementation, r:<S>:<log2 L>:<lambda>[:<log2 R>:<storage>] for
# PHast-R.

# A sweep of the expected bucket size: sweep <prefix> <suffix> <lambda>...
sweep() {
  local p=$1 s=$2
  shift 2
  for l in "$@"; do printf '%s:%s%s,' "$p" "$l" "$s"; done
}

# The configurations of the tables: for each structure, the expected bucket
# size that minimizes space (PHast-R: the default size and the next two)
T8=ref:plus:8:5.25,ref:w1:8:5.25,ref:w2:8:5.0,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:4.25,r:8:10:4.5,r:8:10:4.75
T10=ref:plus:10:5.15,ref:w1:10:6.2,ref:w2:10:5.9,ref:w3:10:6.0,ref:phast:10:6.05,r:10:11:5.75:2:bfvu,r:10:11:6.0:2:bfvu,r:10:11:6.25:2:bfvu

# The sweeps of the trade-off figure, which contain the configurations of
# the tables
G8=$(sweep ref:plus:8 '' 5.0 5.25 5.5)$(sweep ref:w1:8 '' 5.0 5.25 5.5)$(sweep ref:w2:8 '' 4.75 5.0 5.25)
G8=$G8$(sweep ref:w3:8 '' 4.5 4.75 5.0 5.25 5.5)$(sweep ref:phast:8 '' 4.25 4.5 4.75)
G8=$G8$(sweep r:8:10 '' 4.0 4.25 4.5 4.75 5.0)
G8=${G8%,}
G10=$(sweep ref:plus:10 '' 4.9 5.15 5.4)$(sweep ref:w1:10 '' 5.95 6.2 6.45)$(sweep ref:w2:10 '' 5.65 5.9 6.15)
G10=$G10$(sweep ref:w3:10 '' 5.5 5.75 6.0 6.25 6.5)$(sweep ref:phast:10 '' 5.8 6.05 6.3)
G10=$G10$(sweep r:10:11 :2:bfvu 5.5 5.75 6.0 6.25 6.5)
G10=${G10%,}

# The ablation of patterns: PHast-R with the default expected bucket size and
# 1, 2 or 4 patterns
A8=r:8:10:4.25:0,r:8:10:4.25:1,r:8:10:4.25:2
A10=r:10:11:5.75:0:bfvu,r:10:11:5.75:1:bfvu,r:10:11:5.75:2:bfvu

# The configurations for string keys (cmpstr supports only byte seeds and the
# plus, w3 and phast choosers)
STR8=ref:plus:8:5.25,ref:w3:8:5.0,ref:phast:8:4.5,r:8:10:4.25,r:8:10:4.5,r:8:10:4.75

# The prefix pinning a command to the given cores (a taskset list), empty if
# taskset is not available (e.g., on macOS, for smoke tests): $(pinning
# <cores>) <command>...
pinning() { command -v taskset >/dev/null && echo taskset -c $1; }

# Runs a command pinned to the given cores: pin <cores> <command>...
pin() {
  local c=$1
  shift
  $(pinning $c) "$@"
}

# The number of constructions of each configuration on n keys: BUILDS if
# set, otherwise 3 up to 10^8 keys and 1 above (where a construction of PHast
# takes from ten minutes to half an hour): builds <n>
builds() {
  if [ -n "$BUILDS" ]; then echo $BUILDS; elif [ $1 -le 100000000 ]; then echo 3; else echo 1; fi
}

# The number of hardware threads
nthreads() { nproc 2>/dev/null || sysctl -n hw.ncpu; }
