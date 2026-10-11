import PHastR.Bump

/-!
# The greedy placement is optimal (Proposition 2 of `phast.tex`)

The greedy placement scans the slots in increasing order, and places in each
slot the unplaced key with the smallest `σ` among those whose reach contains
the slot, if any. We characterize it by a local property (`IsGreedy`): if
some key that is not placed in an earlier slot can be placed in slot `s`,
then slot `s` is occupied, and its occupant has the smallest `σ` among such
keys. We prove that such a placement exists (`greedy_exists`) and that it
bumps the minimum number of keys (`greedy_optimal`), so the minimum is the
number of keys bumped by the greedy placement (`prop2`).

The optimality proof is an exchange argument: by induction on `t`, there is a
placement agreeing with the greedy one on the slots in `[0 . . t)` and
placing at least as many keys as any given placement. To extend the agreement
to slot `t`, we move into slot `t` the key `g` placed there by the greedy
placement, possibly exchanging it with the key `h` occupying the slot; since
`σ g ≤ σ h` and all reaches have the same length `W`, `h` can take the
former slot of `g`.
-/

open Finset

namespace PHastR

variable {n W : ℕ} {σ : Fin n → ℕ}

/-- Key `x` is placed in slot `s`. -/
def Placement.At (P : Placement n W σ) (x : Fin n) (s : ℕ) : Prop :=
  x ∈ P.placed ∧ P.pos x = s

/-- Key `x` is not placed in a slot before `s`, and its reach contains `s`. -/
def Placement.Avail (P : Placement n W σ) (s : ℕ) (x : Fin n) : Prop :=
  (x ∉ P.placed ∨ s ≤ P.pos x) ∧ σ x ≤ s ∧ s < σ x + W

/-- The greedy property at slot `s`. -/
def Placement.GreedyAt (P : Placement n W σ) (s : ℕ) : Prop :=
  (∀ x, P.Avail s x → ∃ y, P.At y s) ∧ ∀ y x, P.At y s → P.Avail s x → σ y ≤ σ x

/-- The greedy property at every slot. -/
def Placement.IsGreedy (P : Placement n W σ) : Prop :=
  ∀ s < n, P.GreedyAt s

/-! ### Elementary modifications of placements -/

/-- The placement with no placed key. -/
def Placement.empty : Placement n W σ where
  placed := ∅
  pos := fun _ => 0
  inj := by simp
  lt_n := by simp
  reach := by simp

/-- Remove key `h` from a placement. -/
def Placement.erase (P : Placement n W σ) (h : Fin n) : Placement n W σ where
  placed := P.placed.erase h
  pos := P.pos
  inj := P.inj.mono (fun _ hx => mem_of_mem_erase (mem_coe.1 hx))
  lt_n := fun x hx => P.lt_n x (mem_of_mem_erase hx)
  reach := fun x hx => P.reach x (mem_of_mem_erase hx)

/-- Put key `g` (placed or not) in slot `s`, which must not be occupied by
another key. -/
def Placement.put (P : Placement n W σ) (g : Fin n) (s : ℕ) (hs : s < n)
    (hw : σ g ≤ s ∧ s < σ g + W) (hfree : ∀ x ∈ P.placed, x ≠ g → P.pos x ≠ s) :
    Placement n W σ where
  placed := insert g P.placed
  pos := Function.update P.pos g s
  inj := by
    intro x hx y hy hxy
    simp only [coe_insert, Set.mem_insert_iff, mem_coe] at hx hy
    by_cases hxg : x = g <;> by_cases hyg : y = g
    · rw [hxg, hyg]
    · subst hxg
      rw [Function.update_self, Function.update_of_ne hyg] at hxy
      exact absurd hxy.symm (hfree y (hy.resolve_left hyg) hyg)
    · subst hyg
      rw [Function.update_self, Function.update_of_ne hxg] at hxy
      exact absurd hxy (hfree x (hx.resolve_left hxg) hxg)
    · rw [Function.update_of_ne hxg, Function.update_of_ne hyg] at hxy
      exact P.inj (hx.resolve_left hxg) (hy.resolve_left hyg) hxy
  lt_n := by
    intro x hx
    by_cases hxg : x = g
    · subst hxg; rw [Function.update_self]; exact hs
    · rw [Function.update_of_ne hxg]
      exact P.lt_n x ((mem_insert.1 hx).resolve_left hxg)
  reach := by
    intro x hx
    by_cases hxg : x = g
    · subst hxg; rw [Function.update_self]; exact hw
    · rw [Function.update_of_ne hxg]
      exact P.reach x ((mem_insert.1 hx).resolve_left hxg)

lemma Placement.erase_at (P : Placement n W σ) (h x : Fin n) (s : ℕ) :
    (P.erase h).At x s ↔ x ≠ h ∧ P.At x s := by
  simp only [Placement.At, Placement.erase, mem_erase]
  tauto

lemma Placement.put_at (P : Placement n W σ) (g : Fin n) (s : ℕ) (hs hw hfree)
    (x : Fin n) (r : ℕ) :
    (P.put g s hs hw hfree).At x r ↔ (x = g ∧ s = r) ∨ (x ≠ g ∧ P.At x r) := by
  simp only [Placement.At, Placement.put, mem_insert]
  by_cases hxg : x = g
  · subst hxg; simp
  · simp [hxg]

lemma Placement.card_erase (P : Placement n W σ) {h : Fin n} (hh : h ∈ P.placed) :
    (P.erase h).placed.card + 1 = P.placed.card :=
  card_erase_add_one hh

lemma Placement.card_put_ge (P : Placement n W σ) (g : Fin n) (s : ℕ) (hs hw hfree) :
    P.placed.card ≤ (P.put g s hs hw hfree).placed.card :=
  card_le_card (subset_insert _ _)

lemma Placement.card_put_not_mem (P : Placement n W σ) {g : Fin n} (hg : g ∉ P.placed)
    (s : ℕ) (hs hw hfree) : (P.put g s hs hw hfree).placed.card = P.placed.card + 1 :=
  card_insert_of_notMem hg

/-- At most one key is placed in a slot. -/
lemma Placement.at_unique (P : Placement n W σ) {x y : Fin n} {s : ℕ}
    (hx : P.At x s) (hy : P.At y s) : x = y :=
  P.inj hx.1 hy.1 (hx.2.trans hy.2.symm)

/-! ### Existence -/

/-- A placement in slots before `t` that has the greedy property at all of
them. -/
def Placement.GreedyUpTo (P : Placement n W σ) (t : ℕ) : Prop :=
  (∀ x ∈ P.placed, P.pos x < t) ∧ ∀ s < t, P.GreedyAt s

/-- Placing a key in a later slot does not change availability. -/
lemma avail_put_iff (P : Placement n W σ) {g : Fin n} (hg : g ∉ P.placed) {t : ℕ}
    (hs hw hfree) {s : ℕ} (hst : s ≤ t) (x : Fin n) :
    (P.put g t hs hw hfree).Avail s x ↔ P.Avail s x := by
  unfold Placement.Avail Placement.put
  simp only [mem_insert]
  by_cases hxg : x = g
  · subst hxg; simp [hg, hst]
  · simp [hxg]

lemma greedy_up_to (t : ℕ) (ht : t ≤ n) : ∃ G : Placement n W σ, G.GreedyUpTo t := by
  classical
  induction t with
  | zero => exact ⟨Placement.empty, by simp [Placement.empty], by simp⟩
  | succ t ih =>
    obtain ⟨G, hpos, hgr⟩ := ih (by omega)
    have htn : t < n := by omega
    -- The candidates for slot t
    set C := univ.filter fun x => G.Avail t x
    by_cases hC : C.Nonempty
    · obtain ⟨g, hgC, hmin⟩ := exists_min_image C σ hC
      obtain ⟨hga, hgw⟩ := (mem_filter.1 hgC).2
      have hg : g ∉ G.placed := by
        rcases hga with h | h
        · exact h
        · intro hm; have := hpos g hm; omega
      have hfree : ∀ x ∈ G.placed, x ≠ g → G.pos x ≠ t := fun x hx _ =>
        (hpos x hx).ne
      refine ⟨G.put g t htn hgw hfree, ?_, ?_⟩
      · intro x hx
        simp only [Placement.put, mem_insert] at hx ⊢
        by_cases hxg : x = g
        · subst hxg; rw [Function.update_self]; omega
        · rw [Function.update_of_ne hxg]
          have := hpos x (hx.resolve_left hxg); omega
      · intro s hs
        have hav := avail_put_iff G hg htn hgw hfree (show s ≤ t by omega)
        rcases Nat.lt_succ_iff_lt_or_eq.1 hs with hs | rfl
        · -- Slots before t: occupants and availability are unchanged
          have hat : ∀ y, (G.put g t htn hgw hfree).At y s ↔ G.At y s := by
            intro y
            rw [Placement.put_at]
            constructor
            · rintro (⟨_, h⟩ | ⟨_, h⟩)
              · omega
              · exact h
            · intro h
              right
              refine ⟨?_, h⟩
              rintro rfl
              exact hg h.1
          obtain ⟨ha, hb⟩ := hgr s hs
          refine ⟨fun x hx => ?_, fun y x hy hx => ?_⟩
          · obtain ⟨y, hy⟩ := ha x ((hav x).1 hx)
            exact ⟨y, (hat y).2 hy⟩
          · exact hb y x ((hat y).1 hy) ((hav x).1 hx)
        · -- Slot t: occupied by g, which has the smallest σ
          refine ⟨fun _ _ => ⟨g, ?_⟩, fun y x hy hx => ?_⟩
          · rw [Placement.put_at]; left; exact ⟨rfl, rfl⟩
          · rw [Placement.put_at] at hy
            rcases hy with ⟨rfl, _⟩ | ⟨_, hy⟩
            · exact hmin x (mem_filter.2 ⟨mem_univ _, (hav x).1 hx⟩)
            · have := hpos y hy.1; have := hy.2; omega
    · -- No candidate: slot t stays free
      refine ⟨G, fun x hx => by have := hpos x hx; omega, fun s hs => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hs with hs | rfl
      · exact hgr s hs
      · refine ⟨fun x hx => ?_, fun y x hy _ => ?_⟩
        · exact absurd ⟨x, mem_filter.2 ⟨mem_univ _, hx⟩⟩ hC
        · have := hpos y hy.1; have := hy.2; omega

/-- A greedy placement exists. -/
theorem greedy_exists : ∃ G : Placement n W σ, G.IsGreedy := by
  obtain ⟨G, _, hG⟩ := greedy_up_to (W := W) (σ := σ) n le_rfl
  exact ⟨G, hG⟩

/-! ### Optimality -/

/-- Two placements agree on the slots before `t`. -/
def Agree (Q G : Placement n W σ) (t : ℕ) : Prop :=
  ∀ x s, s < t → (Q.At x s ↔ G.At x s)

/-- A key that is not placed by `Q` before `t` is available at `t` in `G`, if
its reach contains `t`. -/
lemma avail_of_agree {Q G : Placement n W σ} {t : ℕ} (hA : Agree Q G t) {x : Fin n}
    (hx : x ∉ Q.placed ∨ t ≤ Q.pos x) (hw : σ x ≤ t ∧ t < σ x + W) : G.Avail t x := by
  refine ⟨?_, hw⟩
  by_contra hc
  push Not at hc
  obtain ⟨hm, hlt⟩ := hc
  have := (hA x (G.pos x) hlt).2 ⟨hm, rfl⟩
  rcases hx with hx | hx
  · exact hx this.1
  · rw [this.2] at hx; omega

/-- A key placed by `G` in slot `t` is not placed by `Q` before `t`. -/
lemma not_before_of_agree {Q G : Placement n W σ} {t : ℕ} (hA : Agree Q G t)
    {x : Fin n} (hx : G.At x t) : x ∉ Q.placed ∨ t ≤ Q.pos x := by
  by_contra hc
  push Not at hc
  obtain ⟨hm, hlt⟩ := hc
  have := ((hA x (Q.pos x) hlt).1 ⟨hm, rfl⟩).2
  rw [hx.2] at this
  omega

/-- The exchange step. -/
lemma agree_step {G : Placement n W σ} (hG : G.IsGreedy) {t : ℕ} (htn : t < n)
    {Q : Placement n W σ} (hA : Agree Q G t) :
    ∃ Q' : Placement n W σ, Q.placed.card ≤ Q'.placed.card ∧ Agree Q' G (t + 1) := by
  obtain ⟨ha, hb⟩ := hG t htn
  -- Q is unchanged on the slots before t, so it suffices to fix slot t
  suffices h : ∃ Q' : Placement n W σ, Q.placed.card ≤ Q'.placed.card ∧
      (∀ x s, s < t → (Q'.At x s ↔ Q.At x s)) ∧ ∀ x, (Q'.At x t ↔ G.At x t) by
    obtain ⟨Q', hc, hpre, ht⟩ := h
    refine ⟨Q', hc, fun x s hs => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hs with hs | rfl
    · exact (hpre x s hs).trans (hA x s hs)
    · exact ht x
  by_cases hocc : ∃ g, G.At g t
  · obtain ⟨g, hg⟩ := hocc
    have hgw := G.reach g hg.1
    rw [hg.2] at hgw
    have hgQ := not_before_of_agree hA hg
    -- The slots before t are unchanged if we only move keys placed by Q at t
    -- or later into slots at t or later
    by_cases hQt : ∃ h, Q.At h t
    · obtain ⟨h, hh⟩ := hQt
      by_cases hhg : h = g
      · -- Q already places g in slot t
        subst hhg
        refine ⟨Q, le_rfl, fun _ _ _ => Iff.rfl, fun x => ⟨fun hx => ?_, fun hx => ?_⟩⟩
        · rw [Q.at_unique hx hh]; exact hg
        · rw [G.at_unique hx hg]; exact hh
      · -- Q places another key h in slot t: σ g ≤ σ h
        have hhw := Q.reach h hh.1
        rw [hh.2] at hhw
        have hσ : σ g ≤ σ h :=
          hb g h hg (avail_of_agree hA (Or.inr (by rw [hh.2])) hhw)
        set E := Q.erase h
        have hfreeE : ∀ x ∈ E.placed, x ≠ g → E.pos x ≠ t := by
          intro x hx _ hxt
          have hxh := ne_of_mem_erase hx
          exact hxh (Q.at_unique ⟨mem_of_mem_erase hx, hxt⟩ hh)
        set F := E.put g t htn hgw hfreeE
        by_cases hgm : g ∈ Q.placed
        · -- Exchange g and h
          have hu : t < Q.pos g := by
            rcases hgQ with h' | h'
            · exact absurd hgm h'
            · rcases Nat.lt_or_eq_of_le h' with h' | h'
              · exact h'
              · exact absurd (Q.at_unique ⟨hgm, h'.symm⟩ hh) (Ne.symm hhg)
          have hun := Q.lt_n g hgm
          have hgu := (Q.reach g hgm).2
          have hhw' : σ h ≤ Q.pos g ∧ Q.pos g < σ h + W := ⟨by omega, by omega⟩
          have hfreeF : ∀ x ∈ F.placed, x ≠ h → F.pos x ≠ Q.pos g := by
            intro x hx hxh hxu
            have hxF : F.At x (Q.pos g) := ⟨hx, hxu⟩
            rw [Placement.put_at, Placement.erase_at] at hxF
            rcases hxF with ⟨_, h'⟩ | ⟨hxg, _, hxQ⟩
            · omega
            · exact hxg (Q.at_unique hxQ ⟨hgm, rfl⟩)
          refine ⟨F.put h (Q.pos g) hun hhw' hfreeF, ?_, ?_, ?_⟩
          · have h1 : E.placed.card + 1 = Q.placed.card := Q.card_erase hh.1
            have h2 : E.placed.card ≤ F.placed.card := E.card_put_ge g t htn hgw hfreeE
            have h3 := F.card_put_not_mem (g := h) (by
              simp only [F, E, Placement.put, Placement.erase, mem_insert, mem_erase]
              tauto) (Q.pos g) hun hhw' hfreeF
            omega
          · intro x s hs
            rw [Placement.put_at, Placement.put_at, Placement.erase_at]
            constructor
            · rintro (⟨_, h'⟩ | ⟨_, ⟨_, h'⟩ | ⟨_, _, h'⟩⟩)
              · omega
              · omega
              · exact h'
            · intro hx
              have hxg : x ≠ g := by
                rintro rfl
                rcases hgQ with h' | h'
                · exact h' hx.1
                · rw [hx.2] at h'; omega
              have hxh : x ≠ h := by rintro rfl; have := hx.2; have := hh.2; omega
              exact Or.inr ⟨hxh, Or.inr ⟨hxg, hxh, hx⟩⟩
          · intro x
            rw [Placement.put_at, Placement.put_at, Placement.erase_at]
            constructor
            · rintro (⟨_, h'⟩ | ⟨_, ⟨rfl, _⟩ | ⟨_, _, h'⟩⟩)
              · omega
              · exact hg
              · exact absurd (Q.at_unique h' hh) ‹x ≠ h›
            · intro hx
              rw [G.at_unique hx hg]
              exact Or.inr ⟨hhg ∘ Eq.symm, Or.inl ⟨rfl, rfl⟩⟩
        · -- Replace h with g
          refine ⟨F, ?_, ?_, ?_⟩
          · have h1 : E.placed.card + 1 = Q.placed.card := Q.card_erase hh.1
            have h2 : F.placed.card = E.placed.card + 1 := E.card_put_not_mem (g := g)
              (fun hm => hgm (mem_of_mem_erase hm)) t htn hgw hfreeE
            omega
          · intro x s hs
            rw [Placement.put_at, Placement.erase_at]
            constructor
            · rintro (⟨_, h'⟩ | ⟨_, _, h'⟩)
              · omega
              · exact h'
            · intro hx
              have hxg : x ≠ g := by rintro rfl; exact hgm hx.1
              have hxh : x ≠ h := by rintro rfl; have := hx.2; have := hh.2; omega
              exact Or.inr ⟨hxg, hxh, hx⟩
          · intro x
            rw [Placement.put_at, Placement.erase_at]
            constructor
            · rintro (⟨rfl, _⟩ | ⟨_, hxh, h'⟩)
              · exact hg
              · exact absurd (Q.at_unique h' hh) hxh
            · intro hx
              rw [G.at_unique hx hg]
              exact Or.inl ⟨rfl, rfl⟩
    · -- Slot t is free in Q: put g there
      push Not at hQt
      have hfree : ∀ x ∈ Q.placed, x ≠ g → Q.pos x ≠ t := fun x hx _ hxt =>
        hQt x ⟨hx, hxt⟩
      refine ⟨Q.put g t htn hgw hfree, Q.card_put_ge g t htn hgw hfree, ?_, ?_⟩
      · intro x s hs
        rw [Placement.put_at]
        constructor
        · rintro (⟨_, h'⟩ | ⟨_, h'⟩)
          · omega
          · exact h'
        · intro hx
          refine Or.inr ⟨?_, hx⟩
          rintro rfl
          rcases hgQ with h' | h'
          · exact h' hx.1
          · rw [hx.2] at h'; omega
      · intro x
        rw [Placement.put_at]
        constructor
        · rintro (⟨rfl, _⟩ | ⟨_, h'⟩)
          · exact hg
          · exact absurd h' (hQt x)
        · intro hx
          rw [G.at_unique hx hg]
          exact Or.inl ⟨rfl, rfl⟩
  · -- Slot t is free in G, so no key is available at t, and slot t is free in Q
    push Not at hocc
    refine ⟨Q, le_rfl, fun _ _ _ => Iff.rfl, fun x => ⟨fun hx => ?_, fun hx => ?_⟩⟩
    · have hw := Q.reach x hx.1
      rw [hx.2] at hw
      obtain ⟨y, hy⟩ := ha x (avail_of_agree hA (Or.inr (by rw [hx.2])) hw)
      exact absurd hy (hocc y)
    · exact absurd hx (hocc x)

/-- **The greedy placement is optimal.** -/
theorem greedy_optimal {G : Placement n W σ} (hG : G.IsGreedy) (P : Placement n W σ) :
    G.bumped ≤ P.bumped := by
  have key : ∀ t ≤ n, ∃ Q : Placement n W σ, P.placed.card ≤ Q.placed.card ∧ Agree Q G t := by
    intro t ht
    induction t with
    | zero => exact ⟨P, le_rfl, fun _ _ h => absurd h (Nat.not_lt_zero _)⟩
    | succ t ih =>
      obtain ⟨Q, hc, hA⟩ := ih (by omega)
      obtain ⟨Q', hc', hA'⟩ := agree_step hG (by omega) hA
      exact ⟨Q', hc.trans hc', hA'⟩
  obtain ⟨Q, hc, hA⟩ := key n le_rfl
  -- Agreement on all slots means the same placed keys
  have hsub : Q.placed ⊆ G.placed := fun x hx =>
    ((hA x (Q.pos x) (Q.lt_n x hx)).1 ⟨hx, rfl⟩).1
  have hle := card_le_card hsub
  unfold Placement.bumped
  omega

/-- **Proposition 2.** The minimum number of bumped keys is attained by a
greedy placement. -/
theorem prop2 : ∃ G : Placement n W σ, G.IsGreedy ∧ ∀ P : Placement n W σ,
    G.bumped ≤ P.bumped := by
  obtain ⟨G, hG⟩ := greedy_exists (n := n) (W := W) (σ := σ)
  exact ⟨G, hG, greedy_optimal hG⟩

end PHastR
