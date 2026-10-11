# Lean formalization of `phast.tex`

Machine-checked verification of the combinatorial results of Section 7 of
*PHast-R: Faster Minimal Perfect Hashing with Rotated Layouts*
(`../phast.tex`). Lean 4 (v4.35.0-rc2) + Mathlib; `lake build` checks
everything. No `sorry`/`admit`/axioms beyond `propext`, `Classical.choice`,
`Quot.sound`.

## Model

- `PHastR/Bump.lean` — `n` keys (`Fin n`) and the `n` slots `[0 . . n)`; key
  `x` can be placed only in its reach `[σ x . . σ x + W)`. A `Placement` puts
  a subset of the keys into distinct slots of their reaches (`placed`, `pos`,
  injectivity on `placed`); the other keys are bumped (`bumped = n - |placed|`).
  `X σ t = t - |{x | σ x < t}|`, and the range `D` of `X` on `[0 . . n]` is
  written with `Finset.sup'`/`Finset.inf'` over `range (n + 1)`.
- `PHastR/Disjoint.lean` — the free slots of `[t . . t')` (`freeIn`) and the
  bumped keys whose reach starts in `[t . . t')` (`bumpedIn`); a family of
  disjoint intervals `[a i . . c i)`, `i ∈ s` (with `c i ≤ n` when free slots
  are counted); `x⁺` is `max x 0`.
- `PHastR/Greedy.lean` — the greedy placement is characterized by a local
  property at each slot `s` (`IsGreedy`): if a key that is not placed before
  `s` can be placed in `s` (`Avail`), then `s` is occupied, and its occupant
  has the smallest `σ` among such keys. This covers every tie-breaking rule.

## Results

References are to sections and statements of `phast.tex` rather than to line
numbers, which change with every edit.

| Paper claim | Status |
|---|---|
| Section 3, first paragraph: with `m = n`, holes are exactly the bumped keys | ✔ `holes_eq_bumped` |
| Proposition 1 (Section 7): every placement bumps at least `D - W + 1` keys | ✔ `prop1`, with the hypothesis `W ≥ 1` (stated in the paper since the formalization showed it missing); the hypothesis is necessary (`prop1_needs_W_pos`: for `n = 1`, `σ = 1`, and `W = 0` we have `D = 1`, but only one key is bumped) |
| Proof of Proposition 1: free slots in `[t + W - 1 . . t')` | ✔ `deficit` (needs only `t' ≤ n`) |
| Proof of Proposition 1: bumped keys among those with `σ ∈ [t . . t')` | ✔ `excess` (no hypotheses) |
| Proof of Proposition 1: the bound `\|X t' - X t\| - W + 1` for all `t ≤ t'` | ✔ `bumped_ge`, slightly stronger: `X a - X b - W + 1` for all `a ≤ n` and all `b` |
| Section 7, after the proof: `D = n·V_n - 1`, `E[D] = √(πn/2)(1 + o(1))` | not formalized (needs Donsker's theorem and the distribution of the Brownian-excursion maximum) |
| Section 7: if `X` increases by `Δ` on `[t . . t')`, at least `Δ - W + 1` slots in `[t . . t')` are free; if it decreases by `Δ`, at least `Δ - W + 1` keys whose reach starts in `[t . . t')` are bumped | ✔ `rising_in` and `falling_in` (from `deficit_in` and `excess_in`, the localized versions of `deficit` and `excess`), with `W ≥ 1` |
| Section 7: for disjoint intervals, the bumped keys are at least the sum of the bounds of the intervals on which `X` decreases, and the free slots at least the sum of the bounds of the intervals on which `X` increases | ✔ `falling_bound` and `rising_bound` (the latter needs the intervals in `[0 . . n)`), with `W ≥ 1` |
| Section 7: for disjoint intervals, at least `½ Σ (\|X t'_i - X t_i\| - W + 1)⁺` keys are bumped | ✔ `disjoint_bound` (stated as `Σ ≤ 2·bumped`), with `W ≥ 1`; `blocks_bound` is the special case of the `⌊n/ℓ⌋` consecutive intervals of `ℓ` slots |
| Section 7, figure `fig:intervals`: every placement of the example bumps at least `7` keys | an instance of `falling_bound` with the two intervals of the figure (the values of `X` of the example are not checked in Lean) |
| Section 7: the expected number of bumped keys is at least about `0.1·n/W` | not formalized (normal approximation of the increments of `X`) |
| Proposition 2: the greedy placement exists and bumps the minimum number of keys | ✔ `prop2`, from `greedy_exists` (the greedy algorithm, by induction on the slots) and `greedy_optimal` (every placement with the greedy property is optimal, by an exchange argument); no hypothesis on `W` |
| Section 7, after Proposition 2: the minimum is about `n/(2W)` | not formalized (a heuristic estimate) |
