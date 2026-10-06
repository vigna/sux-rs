# Lean formalization of `phast.tex`

Machine-checked verification of Proposition 1 of *PHast-R: Bit–Parallel
Perfect Hashing with Rings of Patterns* (`../phast.tex`). Lean 4
(v4.35.0-rc2) + Mathlib; `lake build` checks everything. No
`sorry`/`admit`/axioms beyond `propext`, `Classical.choice`, `Quot.sound`.

## Model

- `PHastR/Bump.lean` — `n` keys (`Fin n`) and the `n` slots `[0 . . n)`; key
  `x` can be placed only in its window `[σ x . . σ x + W)`. A `Placement` puts
  a subset of the keys into distinct slots of their windows (`placed`, `pos`,
  injectivity on `placed`); the other keys are bumped (`bumped = n - |placed|`).
  `X σ t = t - |{x | σ x < t}|`, and the range `R` of `X` on `[0 . . n]` is
  written with `Finset.sup'`/`Finset.inf'` over `range (n + 1)`.

## Results (line numbers refer to `phast.tex`)

| Paper claim | Status |
|---|---|
| Section 2 (130): with `m = n`, holes are exactly the bumped keys | ✔ `holes_eq_bumped` |
| Prop. 1 (760): every placement leaves at least `R - W + 1` keys unplaced | ✔ `prop1`, with the hypothesis `W ≥ 1` (stated in the paper since the formalization showed it missing); the hypothesis is necessary (`prop1_needs_W_pos`: for `n = 1`, `σ = 1`, and `W = 0` we have `R = 1`, but only one key is bumped) |
| Proof (771–781): empty slots in `[s + W - 1 . . t)` | ✔ `deficit` (needs only `t ≤ n`) |
| Proof (771–781): bumped keys among those with `σ ∈ [s . . t)` | ✔ `excess` (no hypotheses) |
| Proof (771–781): the bound `\|X t - X s\| - W + 1` for all `s ≤ t` | ✔ `bumped_ge`, slightly stronger: `X a - X b - W + 1` for all `a ≤ n` and all `b` |
| Asymptotics (782–801): `R = n·V_n`, `E[R] = √(πn/2)(1 + o(1))` | not formalized (needs Donsker's theorem and the distribution of the Brownian-excursion maximum) |
