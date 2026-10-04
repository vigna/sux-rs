# Fast/slow query split for PHast-R and the reference

`qsplit2.rs` needs a `ph` exposing `Function2::is_bumped` (`ph-is_bumped.patch`,
against commit `7d18454` of `vigna/bsuccinct-rs`). To run it, copy the lab,
point `ph` in `Cargo.toml` to a patched checkout (`path = ...`), add
`sux010 = { package = "sux", version = "=0.10.3" }` only if you also want
`efcmp`, and copy `qsplit2.rs` into `src/bin/`:

    RAYON_NUM_THREADS=1 taskset -c 2 target/release/qsplit2 -n 100000000 -r 11 -v 5.0:1,4.5:1

`--only <i> --kind <0|1|2>` runs a single structure on all/fast/slow keys
(for `perf stat`).
