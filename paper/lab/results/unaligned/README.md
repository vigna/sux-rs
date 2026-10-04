# Unaligned reads of seeds stored in a BitFieldVec

`run.sh` compares, for 10- and 12-bit seeds, aligned reads (`bfv`), unaligned
reads after `try_into_unaligned` (`bfvu`), and the previous private 32-bit
unaligned read (`old`, binary built before the change); u64 keys, GxHash,
single thread, pinned, 11 interleaved rounds, reference rows in each process.
`analyze.py run.csv` reports medians over key seeds, raw and relative to
`ref plus` in the same process.
