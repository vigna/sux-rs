# Elias–Fano representation benchmark

`rkyv_overhead.pdf`: the query cost of a zero-copy rkyv archive relative to an
ε-serde image of the same Elias–Fano structure.

Three independent steps. Each reads its input and writes its output; nothing
is shared but the files between them.

## 1. Measure

From the repository root:

```sh
taskset -c 2 cargo bench --features rkyv,epserde,mmap --bench bench_elias_fano -- \
    --save-baseline full --noplot
```

Writes `target/criterion`, baseline `full`. Takes hours;
`EF_BENCH_CONFIGS=1M/8` in front of the command measures one configuration
instead, in minutes. Stabilize the machine's clocks first — on an unprepared
machine these measurements drifted by up to 18%.

## 2. Extract

```sh
epserde/extract_samples.py target/criterion -o epserde/samples.json
```

Pulls every raw timing out of the criterion tree into one file.
`target/criterion` is not checked in and does not survive `cargo clean`;
`samples.json` does.

## 3. Draw

```sh
epserde/plot_rkyv_overhead.py epserde/samples.json -o epserde/rkyv_overhead
```

Writes `rkyv_overhead.pdf` and `.png`. Run it as often as you like — it only
reads.

`--help` lists the options: `--width`, `--height`, `--bar-width`,
`--font-size`, `--color`, `--op`, and `--subject`/`--baseline-arm` to compare a
different pair of representations.

Drawing needs Linux Libertine or Libertinus Serif, the paper's text face,
either installed (Fedora: `linux-libertine-fonts`, Debian:
`fonts-linuxlibertine`) or from TeX Live, where the script finds it with
`kpsewhich`. Only the machine that draws needs it, not the one that measures.
If it is missing the script says so and names the package rather than silently
substituting a face.

### Other hardware

The same three steps, with different names in steps 2 and 3 so nothing is
overwritten:

```sh
epserde/extract_samples.py target/criterion -o epserde/ryzen.json
epserde/plot_rkyv_overhead.py epserde/ryzen.json -o epserde/ryzen_overhead
```

## Including it in the paper

The figure targets `epserde.tex`
(`\documentclass[acmsmall,review,anonymous]{acmart}`): Linux Libertine at 8 pt,
which is `\footnotesize` in that 10 pt document. Include it at natural size —
scaling changes the type size and breaks the match.

```latex
\begin{figure}
  \centering
  \includegraphics{rkyv_overhead}
  \caption{Cost of querying a zero-copy rkyv archive relative to an
    \eserde{} image of the identical Elias--Fano structure, for $2^{20}$ and
    $2^{30}$ elements. Both are memory-mapped images of the same unaligned
    structure, so the comparison isolates the format. Each bar averages the
    four lower-bit widths $\ell \in \{2,4,8,16\}$; whiskers span their
    minimum and maximum.}
  \label{fig:rkyv-overhead}
\end{figure}
```

The default size, 2.638 × 1.8 in, is half of `acmsmall`'s 5.478 in
`\textwidth` less a 0.2 in gap, so that it sits on a line next to the
companion figure of `epserde-rs/rkyv-bench`; `--width 5.478` gives a
full-width figure. Operation names are set in the text face because Inconsolata, the
document's typewriter face, is not installed here; install it and redraw and
they will be set in it.

`pdffonts` warns "Mismatch between font type and embedded font file". That is
poppler being fussy about matplotlib's OpenType/CFF wrapper; ghostscript
validates it and both render correctly. Leave it unless a checker objects.

---

## The stored run — 11 September 2026

`samples.json` here is from this machine:

| | |
|---|---|
| Source | commit `f6496216`, `--features rkyv,epserde,mmap` |
| Started | 2026-09-11 22:27:13 |
| Finished | 2026-09-12 05:27:24 (7 h 00 m) |
| Host | 12th Gen Intel i7-12700KF, 62 GB RAM, Fedora 43 |
| Pinning | every benchmark `taskset -c 2`, clocks stabilized beforehand |
| Harness | criterion 0.7, default 100 samples after a 3 s warm-up |

288 benchmarks, 28 800 timings: 8 configurations (n ∈ {2²⁰, 2³⁰} ×
l ∈ {2, 4, 8, 16}) × 8 query benchmarks (get/succ/pred/rank, checked and
unchecked) × 4 representations, plus sequential and concurrent construction.
The 95% confidence interval was a median of 0.83% of the mean and never
exceeded 3.21%.

`samples.json` maps `group/arm/param` to the `times` and `iters` criterion
recorded; dividing elementwise gives nanoseconds per operation. Averaging those
reproduces criterion's own mean estimate to within 1e-13%.

### Results

- **ε-serde costs nothing.** A mapped ε-serde image queries at the speed of the
  in-memory structure it was written from: −0.23% at 2²⁰ and +1.17% at 2³⁰, at
  or below the measurement floor.
- **rkyv does not.** Against an ε-serde image of the identical structure — both
  mapped, so the comparison isolates the format — the rkyv archive is
  **+10.4% at 2²⁰ and +17.1% at 2³⁰**, +13.8% overall, worst case +23.9%.
- **Unaligned is never slower than aligned**, and is faster where the lower
  bits dominate: −3.0% on get, −1.7% on rank. Both images hold the unaligned
  structure, so it is their correct term of comparison.
- **Scaling.** 2²⁰ → 2³⁰ multiplies query time by 3.43× (rank) to 4.19× (succ)
  — the cache hierarchy, not the algorithm.

**Caveat on the concurrent build.** Everything in this run was pinned to one
core, `ef_build_conc` included, so its threads all shared that core. The
`t = 4, 8, 16` measurements do not show parallel scaling: at 2³⁰ with l = 8 the
three land within 0.03% of one another, all ≈3.1× the sequential build. That
flatness is the pinning. Measuring the concurrent builder needs an unpinned run.
