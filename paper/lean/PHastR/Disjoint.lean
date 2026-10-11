import PHastR.Bump

/-!
# The bounds of disjoint intervals add up (Section 7 of `phast.tex`)

Proposition 1 uses a single interval of slots, but the argument of its proof
applies to every interval `[t . . t')`: if `X` increases by `Δ` on it, at
least `Δ - W + 1` of its slots are free (`rising_in`), and if `X` decreases
by `Δ`, at least `Δ - W + 1` keys whose reach starts in it are bumped
(`falling_in`). Free slots in disjoint intervals are distinct, and so are
bumped keys whose reaches start in disjoint intervals; as free slots and
bumped keys are equally many (`holes_eq_bumped`), for disjoint intervals
`[a i . . c i)`, `i ∈ s`, with `c i ≤ n`, every placement bumps at least
`Σ (X (c i) - X (a i) - W + 1)⁺` keys (`rising_bound`), at least
`Σ (X (a i) - X (c i) - W + 1)⁺` keys (`falling_bound`), and thus at least
`½ Σ (|X (c i) - X (a i)| - W + 1)⁺` keys (`disjoint_bound`). The last bound
holds in particular for the consecutive blocks of `ℓ` slots (`blocks_bound`),
which is the form used in the paper.
-/

open Finset

namespace PHastR

variable {n W : ℕ} {σ : Fin n → ℕ}

/-- The slots of `[t . . t')` that a placement leaves free. -/
def Placement.freeIn (P : Placement n W σ) (t t' : ℕ) : Finset ℕ :=
  Ico t t' \ P.placed.image P.pos

/-- The bumped keys whose reach starts in `[t . . t')`. -/
def Placement.bumpedIn (P : Placement n W σ) (t t' : ℕ) : Finset (Fin n) :=
  (univ \ P.placed).filter fun x => t ≤ σ x ∧ σ x < t'

/-- Localized `deficit`: only keys whose reach starts in `[t . . t')` can be
placed in the slots in `[t + W - 1 . . t')`, so the others are free slots of
`[t . . t')`. -/
lemma deficit_in (P : Placement n W σ) (hW : 0 < W) (t t' : ℕ) :
    t' - (t + W - 1) ≤ (P.freeIn t t').card + N σ t t' := by
  set used := P.placed.image P.pos
  set J := Ico (t + W - 1) t'
  have h1 : J ∩ used ⊆
      (P.placed.filter fun x => t ≤ σ x ∧ σ x < t').image P.pos := by
    intro q hq
    obtain ⟨hqJ, hqo⟩ := mem_inter.1 hq
    obtain ⟨hq1, hq2⟩ := mem_Ico.1 hqJ
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hqo
    have hw := P.reach x hx
    exact mem_image.2 ⟨x, mem_filter.2 ⟨hx, by omega, by omega⟩, rfl⟩
  have h2 : (J ∩ used).card ≤ N σ t t' :=
    calc (J ∩ used).card
        ≤ ((P.placed.filter fun x => t ≤ σ x ∧ σ x < t').image P.pos).card :=
          card_le_card h1
      _ ≤ (P.placed.filter fun x => t ≤ σ x ∧ σ x < t').card := card_image_le
      _ ≤ N σ t t' := card_le_card (filter_subset_filter _ (subset_univ _))
  have h3 : J \ used ⊆ P.freeIn t t' := by
    unfold Placement.freeIn
    exact sdiff_subset_sdiff (fun q hq => mem_Ico.2 (by rw [mem_Ico] at hq; omega))
      (subset_refl _)
  have h5 := card_sdiff_add_card_inter J used
  have h6 := card_le_card h3
  rw [Nat.card_Ico] at h5
  omega

/-- Localized `excess`: keys whose reach starts in `[t . . t')` can be placed
only in the slots in `[t . . t' + W - 1)`, so the excess is made of bumped
keys whose reach starts in `[t . . t')`. -/
lemma excess_in (P : Placement n W σ) (t t' : ℕ) :
    N σ t t' ≤ (P.bumpedIn t t').card + (t' + W - 1 - t) := by
  set K := P.placed.filter fun x => t ≤ σ x ∧ σ x < t'
  have hK : K.card ≤ t' + W - 1 - t := by
    have himg : K.image P.pos ⊆ Ico t (t' + W - 1) := by
      intro q hq
      obtain ⟨x, hx, rfl⟩ := mem_image.1 hq
      obtain ⟨hxp, ht, ht'⟩ := mem_filter.1 hx
      have hw := P.reach x hxp
      exact mem_Ico.2 ⟨by omega, by omega⟩
    have hinj : Set.InjOn P.pos K :=
      P.inj.mono (fun x hx => (mem_filter.1 (mem_coe.1 hx)).1)
    calc K.card = (K.image P.pos).card := (card_image_of_injOn hinj).symm
      _ ≤ (Ico t (t' + W - 1)).card := card_le_card himg
      _ = t' + W - 1 - t := Nat.card_Ico _ _
  have hsplit : N σ t t' ≤ K.card + (P.bumpedIn t t').card := by
    unfold N Placement.bumpedIn
    calc (univ.filter fun x => t ≤ σ x ∧ σ x < t').card
        ≤ (K ∪ ((univ \ P.placed).filter fun x => t ≤ σ x ∧ σ x < t')).card := by
          apply card_le_card
          intro x hx
          have hx' := (mem_filter.1 hx).2
          by_cases h : x ∈ P.placed
          · exact mem_union_left _ (mem_filter.2 ⟨h, hx'⟩)
          · exact mem_union_right _ (mem_filter.2 ⟨mem_sdiff.2 ⟨mem_univ _, h⟩, hx'⟩)
      _ ≤ _ := card_union_le _ _
  omega

/-- If `X` increases by `Δ` on `[t . . t')`, at least `Δ - W + 1` slots of
`[t . . t')` are free. -/
lemma rising_in (P : Placement n W σ) (hW : 0 < W) {t t' : ℕ} (h : t ≤ t') :
    X σ t' - X σ t - ((W : ℤ) - 1) ≤ ((P.freeIn t t').card : ℤ) := by
  have h1 := deficit_in P hW t t'
  have h2 := X_sub σ h
  omega

/-- If `X` decreases by `Δ` on `[t . . t')`, at least `Δ - W + 1` keys whose
reach starts in `[t . . t')` are bumped. -/
lemma falling_in (P : Placement n W σ) (hW : 0 < W) {t t' : ℕ} (h : t ≤ t') :
    X σ t - X σ t' - ((W : ℤ) - 1) ≤ ((P.bumpedIn t t').card : ℤ) := by
  have h1 := excess_in P t t'
  have h2 := X_sub σ h
  omega

/-- The bumped keys, as a set. -/
lemma card_bumped_keys (P : Placement n W σ) : (univ \ P.placed).card = P.bumped := by
  unfold Placement.bumped
  rw [card_sdiff_of_subset (subset_univ _), card_univ, Fintype.card_fin]

/-- The free slots of disjoint intervals in `[0 . . n)` are distinct free
slots, which are as many as the bumped keys. -/
lemma sum_freeIn_le {ι : Type*} (P : Placement n W σ) (s : Finset ι) (a c : ι → ℕ)
    (hcn : ∀ i ∈ s, c i ≤ n) (hdisj : (s : Set ι).PairwiseDisjoint fun i => Ico (a i) (c i)) :
    ∑ i ∈ s, (P.freeIn (a i) (c i)).card ≤ P.bumped := by
  rw [← card_biUnion]
  · calc (s.biUnion fun i => P.freeIn (a i) (c i)).card
        ≤ (range n \ P.placed.image P.pos).card := by
          apply card_le_card
          intro q hq
          obtain ⟨i, hi, hq⟩ := mem_biUnion.1 hq
          unfold Placement.freeIn at hq
          rw [mem_sdiff, mem_Ico] at hq
          exact mem_sdiff.2 ⟨mem_range.2 (by have := hcn i hi; omega), hq.2⟩
      _ = P.bumped := holes_eq_bumped P
  · intro i hi j hj hij
    have h := hdisj hi hj hij
    simp only [Function.onFun] at h ⊢
    unfold Placement.freeIn
    exact h.mono sdiff_subset sdiff_subset

/-- Bumped keys whose reaches start in disjoint intervals are distinct. -/
lemma sum_bumpedIn_le {ι : Type*} (P : Placement n W σ) (s : Finset ι) (a c : ι → ℕ)
    (hdisj : (s : Set ι).PairwiseDisjoint fun i => Ico (a i) (c i)) :
    ∑ i ∈ s, (P.bumpedIn (a i) (c i)).card ≤ P.bumped := by
  rw [← card_biUnion]
  · calc (s.biUnion fun i => P.bumpedIn (a i) (c i)).card
        ≤ (univ \ P.placed).card := by
          apply card_le_card
          intro x hx
          obtain ⟨i, _, hx⟩ := mem_biUnion.1 hx
          exact (mem_filter.1 hx).1
      _ = P.bumped := card_bumped_keys P
  · intro i hi j hj hij
    have h := hdisj hi hj hij
    simp only [Function.onFun] at h ⊢
    rw [Finset.disjoint_left]
    intro x hxi hxj
    unfold Placement.bumpedIn at hxi hxj
    exact Finset.disjoint_left.1 h (mem_Ico.2 (mem_filter.1 hxi).2)
      (mem_Ico.2 (mem_filter.1 hxj).2)

/-- **Increasing intervals.** If the intervals `[a i . . c i)`, for `i ∈ s`,
are disjoint and contained in `[0 . . n)`, every placement leaves free, and
thus bumps, at least `Σ (X (c i) - X (a i) - W + 1)⁺` keys. -/
theorem rising_bound {ι : Type*} (P : Placement n W σ) (hW : 0 < W) (s : Finset ι)
    (a c : ι → ℕ) (hac : ∀ i ∈ s, a i ≤ c i) (hcn : ∀ i ∈ s, c i ≤ n)
    (hdisj : (s : Set ι).PairwiseDisjoint fun i => Ico (a i) (c i)) :
    ∑ i ∈ s, max (X σ (c i) - X σ (a i) - ((W : ℤ) - 1)) 0 ≤ (P.bumped : ℤ) :=
  calc ∑ i ∈ s, max (X σ (c i) - X σ (a i) - ((W : ℤ) - 1)) 0
      ≤ ∑ i ∈ s, ((P.freeIn (a i) (c i)).card : ℤ) :=
        sum_le_sum fun i hi => max_le (rising_in P hW (hac i hi)) (Int.natCast_nonneg _)
    _ = ((∑ i ∈ s, (P.freeIn (a i) (c i)).card : ℕ) : ℤ) := (Nat.cast_sum _ _).symm
    _ ≤ (P.bumped : ℤ) := Nat.cast_le.2 (sum_freeIn_le P s a c hcn hdisj)

/-- **Decreasing intervals.** If the intervals `[a i . . c i)`, for `i ∈ s`,
are disjoint, every placement bumps at least `Σ (X (a i) - X (c i) - W + 1)⁺`
keys. -/
theorem falling_bound {ι : Type*} (P : Placement n W σ) (hW : 0 < W) (s : Finset ι)
    (a c : ι → ℕ) (hac : ∀ i ∈ s, a i ≤ c i)
    (hdisj : (s : Set ι).PairwiseDisjoint fun i => Ico (a i) (c i)) :
    ∑ i ∈ s, max (X σ (a i) - X σ (c i) - ((W : ℤ) - 1)) 0 ≤ (P.bumped : ℤ) :=
  calc ∑ i ∈ s, max (X σ (a i) - X σ (c i) - ((W : ℤ) - 1)) 0
      ≤ ∑ i ∈ s, ((P.bumpedIn (a i) (c i)).card : ℤ) :=
        sum_le_sum fun i hi => max_le (falling_in P hW (hac i hi)) (Int.natCast_nonneg _)
    _ = ((∑ i ∈ s, (P.bumpedIn (a i) (c i)).card : ℕ) : ℤ) := (Nat.cast_sum _ _).symm
    _ ≤ (P.bumped : ℤ) := Nat.cast_le.2 (sum_bumpedIn_le P s a c hdisj)

/-- **Disjoint intervals.** If the intervals `[a i . . c i)`, for `i ∈ s`, are
disjoint and contained in `[0 . . n)`, every placement bumps at least
`½ Σ (|X (c i) - X (a i)| - W + 1)⁺` keys. -/
theorem disjoint_bound {ι : Type*} (P : Placement n W σ) (hW : 0 < W) (s : Finset ι)
    (a c : ι → ℕ) (hac : ∀ i ∈ s, a i ≤ c i) (hcn : ∀ i ∈ s, c i ≤ n)
    (hdisj : (s : Set ι).PairwiseDisjoint fun i => Ico (a i) (c i)) :
    ∑ i ∈ s, max (|X σ (c i) - X σ (a i)| - ((W : ℤ) - 1)) 0 ≤ 2 * (P.bumped : ℤ) := by
  have h1 := rising_bound P hW s a c hac hcn hdisj
  have h2 := falling_bound P hW s a c hac hdisj
  calc ∑ i ∈ s, max (|X σ (c i) - X σ (a i)| - ((W : ℤ) - 1)) 0
      ≤ ∑ i ∈ s, (max (X σ (c i) - X σ (a i) - ((W : ℤ) - 1)) 0 +
          max (X σ (a i) - X σ (c i) - ((W : ℤ) - 1)) 0) := by
        apply sum_le_sum
        intro i _
        have hA1 := le_max_left (X σ (c i) - X σ (a i) - ((W : ℤ) - 1)) 0
        have hA2 := le_max_right (X σ (c i) - X σ (a i) - ((W : ℤ) - 1)) 0
        have hB1 := le_max_left (X σ (a i) - X σ (c i) - ((W : ℤ) - 1)) 0
        have hB2 := le_max_right (X σ (a i) - X σ (c i) - ((W : ℤ) - 1)) 0
        refine max_le ?_ (by linarith)
        rcases abs_cases (X σ (c i) - X σ (a i)) with ⟨h, _⟩ | ⟨h, _⟩ <;> rw [h] <;> linarith
    _ = _ := sum_add_distrib
    _ ≤ (P.bumped : ℤ) + P.bumped := add_le_add h1 h2
    _ = 2 * (P.bumped : ℤ) := by ring

/-- **Blocks.** The bound applied to the `⌊n / ℓ⌋` consecutive blocks of `ℓ`
slots, as in the paper. -/
theorem blocks_bound (P : Placement n W σ) (hW : 0 < W) (ℓ : ℕ) :
    ∑ k ∈ range (n / ℓ), max (|X σ ((k + 1) * ℓ) - X σ (k * ℓ)| - ((W : ℤ) - 1)) 0
      ≤ 2 * (P.bumped : ℤ) := by
  refine disjoint_bound P hW (range (n / ℓ)) (fun k => k * ℓ) (fun k => (k + 1) * ℓ)
    (fun k _ => Nat.mul_le_mul_right ℓ (Nat.le_succ k)) (fun k hk => ?_) ?_
  · calc (k + 1) * ℓ ≤ n / ℓ * ℓ := Nat.mul_le_mul_right ℓ (mem_range.1 hk)
      _ ≤ n := Nat.div_mul_le_self n ℓ
  · intro i _ j _ hij
    simp only [Function.onFun]
    rw [Finset.disjoint_left]
    intro q hqi hqj
    rw [mem_Ico] at hqi hqj
    rcases Nat.lt_or_gt_of_ne hij with h | h
    · exact absurd (lt_of_lt_of_le hqi.2 ((Nat.mul_le_mul_right ℓ h).trans hqj.1)) (lt_irrefl q)
    · exact absurd (lt_of_lt_of_le hqj.2 ((Nat.mul_le_mul_right ℓ h).trans hqi.1)) (lt_irrefl q)

end PHastR
