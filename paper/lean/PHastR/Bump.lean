import Mathlib

/-!
# A lower bound on bumping (Proposition 1 of `phast.tex`)

We consider `n` keys (the elements of `Fin n`) and the `n` slots `0`, …,
`n - 1`. Each key `x` can be placed only in the slots of its *window*
`[σ x . . σ x + W)`. A *placement* puts a subset of the keys into distinct
slots of their windows; the other keys are *bumped*.

Let `X t = t - |{x | σ x < t}|` for `0 ≤ t ≤ n`, and let `D` be the range of
`X`. Proposition 1 states that, if `W ≥ 1`, every placement bumps at least
`D - W + 1` keys.

The proof follows the paper. For `t ≤ t'`, only keys whose window starts in
`[t . . t')` can be placed in the slots in `[t + W - 1 . . t')`, so at least
`X t' - X t - W + 1` of those slots are free (`deficit`); symmetrically, these
keys can be placed only in the slots in `[t . . t' + W - 1)`, so at least
`X t - X t' - W + 1` of them are bumped (`excess`). As there are as many keys
as slots, free slots and bumped keys are equally many.
-/

open Finset

namespace PHastR

/-- A placement of the keys `Fin n` into the slots `[0 . . n)`, in which key
`x` can be placed only in the slots `[σ x . . σ x + W)`. -/
structure Placement (n W : ℕ) (σ : Fin n → ℕ) where
  /-- The keys that are placed; the other ones are bumped. -/
  placed : Finset (Fin n)
  /-- The slot of each placed key. -/
  pos : Fin n → ℕ
  /-- Placed keys are placed in distinct slots. -/
  inj : Set.InjOn pos placed
  /-- Slots are in `[0 . . n)`. -/
  lt_n : ∀ x ∈ placed, pos x < n
  /-- A placed key lies in its window. -/
  window : ∀ x ∈ placed, σ x ≤ pos x ∧ pos x < σ x + W

variable {n W : ℕ} {σ : Fin n → ℕ}

/-- The number of bumped keys. -/
def Placement.bumped (P : Placement n W σ) : ℕ :=
  n - P.placed.card

/-- `X t = t - |{x | σ x < t}|`. -/
def X (σ : Fin n → ℕ) (t : ℕ) : ℤ :=
  (t : ℤ) - ((univ.filter fun x => σ x < t).card : ℤ)

/-- The number of keys whose window starts in `[t . . t')`. -/
def N (σ : Fin n → ℕ) (t t' : ℕ) : ℕ :=
  (univ.filter fun x => t ≤ σ x ∧ σ x < t').card

lemma card_lt_eq (σ : Fin n → ℕ) {t t' : ℕ} (h : t ≤ t') :
    (univ.filter fun x => σ x < t').card =
      (univ.filter fun x => σ x < t).card + N σ t t' := by
  unfold N
  rw [← card_union_of_disjoint]
  · congr 1
    ext x
    simp only [mem_filter, mem_univ, true_and, mem_union]
    omega
  · rw [disjoint_filter]
    intro x _ h1 h2
    omega

/-- For `t ≤ t'`, `X t' - X t` is the length of `[t . . t')` minus the number
of keys whose window starts in `[t . . t')`. -/
lemma X_sub (σ : Fin n → ℕ) {t t' : ℕ} (h : t ≤ t') :
    X σ t' - X σ t = (t' : ℤ) - t - N σ t t' := by
  unfold X
  rw [card_lt_eq σ h]
  push_cast
  ring

/-- Only keys whose window starts in `[t . . t')` can be placed in the slots
in `[t + W - 1 . . t')`, and every free slot corresponds to a bumped key. -/
lemma deficit (P : Placement n W σ) {t t' : ℕ} (htn : t' ≤ n) :
    t' - (t + W - 1) ≤ P.bumped + N σ t t' := by
  set used := P.placed.image P.pos
  have hcard : used.card = P.placed.card := card_image_of_injOn P.inj
  have hsub : used ⊆ range n := by
    intro q hq
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hq
    exact mem_range.2 (P.lt_n x hx)
  set J := Ico (t + W - 1) t'
  -- The used slots of J are used by keys whose window starts in [t . . t')
  have h1 : J ∩ used ⊆
      (P.placed.filter fun x => t ≤ σ x ∧ σ x < t').image P.pos := by
    intro q hq
    obtain ⟨hqJ, hqo⟩ := mem_inter.1 hq
    obtain ⟨hq1, hq2⟩ := mem_Ico.1 hqJ
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hqo
    have hw := P.window x hx
    exact mem_image.2 ⟨x, mem_filter.2 ⟨hx, by omega, by omega⟩, rfl⟩
  have h2 : (J ∩ used).card ≤ N σ t t' :=
    calc (J ∩ used).card
        ≤ ((P.placed.filter fun x => t ≤ σ x ∧ σ x < t').image P.pos).card :=
          card_le_card h1
      _ ≤ (P.placed.filter fun x => t ≤ σ x ∧ σ x < t').card := card_image_le
      _ ≤ N σ t t' := card_le_card (filter_subset_filter _ (subset_univ _))
  -- The free slots of J are free slots of [0 . . n)
  have h3 : J \ used ⊆ range n \ used :=
    sdiff_subset_sdiff (fun q hq => mem_range.2 (by rw [mem_Ico] at hq; omega))
      (subset_refl _)
  have h4 : (range n \ used).card = n - P.placed.card := by
    rw [card_sdiff_of_subset hsub, card_range, hcard]
  have h5 := card_sdiff_add_card_inter J used
  have h6 := card_le_card h3
  rw [Nat.card_Ico] at h5
  unfold Placement.bumped
  omega

/-- Keys whose window starts in `[t . . t')` can be placed only in the slots
in `[t . . t' + W - 1)`, so the excess must be bumped. -/
lemma excess (P : Placement n W σ) (t t' : ℕ) :
    N σ t t' ≤ P.bumped + (t' + W - 1 - t) := by
  set K := P.placed.filter fun x => t ≤ σ x ∧ σ x < t'
  have hK : K.card ≤ t' + W - 1 - t := by
    have himg : K.image P.pos ⊆ Ico t (t' + W - 1) := by
      intro q hq
      obtain ⟨x, hx, rfl⟩ := mem_image.1 hq
      obtain ⟨hxp, ht, ht'⟩ := mem_filter.1 hx
      have hw := P.window x hxp
      exact mem_Ico.2 ⟨by omega, by omega⟩
    have hinj : Set.InjOn P.pos K :=
      P.inj.mono (fun x hx => (mem_filter.1 (mem_coe.1 hx)).1)
    calc K.card = (K.image P.pos).card := (card_image_of_injOn hinj).symm
      _ ≤ (Ico t (t' + W - 1)).card := card_le_card himg
      _ = t' + W - 1 - t := Nat.card_Ico _ _
  -- Each key whose window starts in [t . . t') is either in K or bumped
  have hsplit : N σ t t' ≤ K.card + (univ \ P.placed).card := by
    unfold N
    calc (univ.filter fun x => t ≤ σ x ∧ σ x < t').card
        ≤ (K ∪ (univ \ P.placed)).card := by
          apply card_le_card
          intro x hx
          by_cases h : x ∈ P.placed
          · exact mem_union_left _ (mem_filter.2 ⟨h, (mem_filter.1 hx).2⟩)
          · exact mem_union_right _ (mem_sdiff.2 ⟨mem_univ _, h⟩)
      _ ≤ K.card + (univ \ P.placed).card := card_union_le _ _
  have hb : (univ \ P.placed).card = n - P.placed.card := by
    rw [card_sdiff_of_subset (subset_univ _), card_univ, Fintype.card_fin]
  unfold Placement.bumped
  omega

/-- The core of Proposition 1: for all `a ≤ n` and all `b`, every placement
bumps at least `X a - X b - W + 1` keys. -/
theorem bumped_ge (P : Placement n W σ) (hW : 0 < W) {a : ℕ} (b : ℕ)
    (ha : a ≤ n) :
    X σ a - X σ b - ((W : ℤ) - 1) ≤ P.bumped := by
  rcases le_total b a with h | h
  · have h1 := deficit P (t := b) ha
    have h2 := X_sub σ h
    omega
  · have h1 := excess P a b
    have h2 := X_sub σ h
    omega

/-- **Proposition 1.** Every placement bumps at least `D - W + 1` keys, where
`D` is the range of `X` on `[0 . . n]`. -/
theorem prop1 (P : Placement n W σ) (hW : 0 < W) :
    ((range (n + 1)).sup' nonempty_range_add_one (X σ)
        - (range (n + 1)).inf' nonempty_range_add_one (X σ))
      - ((W : ℤ) - 1) ≤ P.bumped := by
  obtain ⟨a, ha, hmax⟩ := exists_mem_eq_sup' nonempty_range_add_one (X σ)
  obtain ⟨b, -, hmin⟩ := exists_mem_eq_inf' nonempty_range_add_one (X σ)
  rw [hmax, hmin]
  exact bumped_ge P hW b (Nat.lt_succ_iff.1 (mem_range.1 ha))

/-- Free slots and bumped keys are equally many, so the bound of
Proposition 1 applies to holes, too. -/
theorem holes_eq_bumped (P : Placement n W σ) :
    (range n \ P.placed.image P.pos).card = P.bumped := by
  have hsub : P.placed.image P.pos ⊆ range n := by
    intro q hq
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hq
    exact mem_range.2 (P.lt_n x hx)
  rw [card_sdiff_of_subset hsub, card_range, card_image_of_injOn P.inj]
  rfl

end PHastR

namespace PHastR

/-- One key whose window starts at `1`. -/
def σ₁ : Fin 1 → ℕ := fun _ => 1

/-- The hypothesis `0 < W` of `prop1` is necessary: with one key whose window
starts at `1` and `W = 0`, the range of `X` is `1`, so the bound would be `2`,
but a placement bumps a single key. -/
theorem prop1_needs_W_pos :
    ∃ P : Placement 1 0 σ₁,
      ¬ (((range 2).sup' nonempty_range_add_one (X σ₁)
          - (range 2).inf' nonempty_range_add_one (X σ₁))
        - ((0 : ℤ) - 1) ≤ P.bumped) := by
  refine ⟨⟨∅, fun _ => 0, by simp, by simp, by simp⟩, ?_⟩
  decide

end PHastR
