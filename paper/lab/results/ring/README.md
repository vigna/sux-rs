# Ring patterns

`run.sh` compares PHast+ (plain and with wrapping, reference implementation)
with PHast-R with wrapping and repair (`w3`) and with ring patterns (`g4`,
see the section "Second brainstorm" of `paper/NOTES.md`): u64 keys, GxHash;
single thread pinned with 9 interleaved rounds of queries at 10⁶, 10⁷ and 10⁸
keys, then construction with 8 threads. Results in `run.txt`.
