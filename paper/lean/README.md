# Lean formalization of `phast.tex`

Machine-checked verification of Proposition 1 of *PHast-R: Faster
Minimal Perfect Hashing with Rings of Layouts* (`../phast.tex`). Lean 4
(v4.35.0-rc2) + Mathlib; `lake build` checks everything. No
`sorry`/`admit`/axioms beyond `propext`, `Classical.choice`, `Quot.sound`.

## Model

- `PHastR/Bump.lean` — `n` keys (`Fin n`) and the `n` slots `[0 . . n)`; key
  `x` can be placed only in its window `[σ x . . σ x + W)`. A `Placement` puts
  a subset of the keys into distinct slots of their windows (`placed`, `pos`,
  injectivity on `placed`); the other keys are bumped (`bumped = n - |placed|`).
  `X σ t = t - |{x | σ x < t}|`, and the range `D` of `X` on `[0 . . n]` is
  written with `Finset.sup'`/`Finset.inf'` over `range (n + 1)`.

## Results

References are to sections and statements of `phast.tex` rather than to line
numbers, which change with every edit.

| Paper claim | Status |
|---|---|
| Section 2, first paragraph: with `m = n`, holes are exactly the bumped keys | ✔ `holes_eq_bumped` |
| Proposition 1 (Section 6): every placement bumps at least `D - W + 1` keys | ✔ `prop1`, with the hypothesis `W ≥ 1` (stated in the paper since the formalization showed it missing); the hypothesis is necessary (`prop1_needs_W_pos`: for `n = 1`, `σ = 1`, and `W = 0` we have `D = 1`, but only one key is bumped) |
| Proof of Proposition 1: free slots in `[t + W - 1 . . t')` | ✔ `deficit` (needs only `t' ≤ n`) |
| Proof of Proposition 1: bumped keys among those with `σ ∈ [t . . t')` | ✔ `excess` (no hypotheses) |
| Proof of Proposition 1: the bound `\|X t' - X t\| - W + 1` for all `t ≤ t'` | ✔ `bumped_ge`, slightly stronger: `X a - X b - W + 1` for all `a ≤ n` and all `b` |
| Section 6, after the proof: `D = n·V_n`, `E[D] = √(πn/2)(1 + o(1))` | not formalized (needs Donsker's theorem and the distribution of the Brownian-excursion maximum) |
