import Init

/- Controlled scalar discrete memory over Int, all times k; no model claim about ViT. -/
namespace ControlledMemory

def memory (e : Int) (q : Nat → Int) : Nat → Int
  | 0 => 0
  | k + 1 => e * memory e q k + q k

def propagated (e h0 : Int) : Nat → Int
  | 0 => h0
  | k + 1 => e * propagated e h0 k

theorem memory_add (e : Int) (q r : Nat → Int) :
    ∀ k, memory e (fun j => q j + r j) k = memory e q k + memory e r k := by
  intro k
  induction k with
  | zero => simp [memory]
  | succ k ih =>
    simp only [memory, ih, Int.mul_add]
    simp [Int.add_assoc, Int.add_comm, Int.add_left_comm]


theorem memory_prefix (e : Int) (q r : Nat → Int) :
    ∀ k, (∀ j, j < k → q j = r j) → memory e q k = memory e r k := by
  intro k
  induction k with
  | zero => intro same; rfl
  | succ k ih =>
    intro same
    rw [memory, memory, ih (fun j hj => same j (Nat.lt_succ_of_lt hj)),
      same k (Nat.lt_succ_self k)]

theorem hidden_representation (e h0 : Int) (q h : Nat → Int)
    (initial : h 0 = h0)
    (step : ∀ k, h (k + 1) = e * h k + q k) :
    ∀ k, h k = propagated e h0 k + memory e q k := by
  intro k
  induction k with
  | zero => simp [memory, propagated, initial]
  | succ k ih =>
    rw [step, ih, memory, propagated]
    simp [Int.mul_add, Int.add_assoc]

theorem propagated_power (e h0 : Int) :
    ∀ k, propagated e h0 k = e ^ k * h0 := by
  intro k
  induction k with
  | zero => simp [propagated, Int.pow_zero]
  | succ k ih =>
    rw [propagated, ih, Int.pow_succ]
    rw [← Int.mul_assoc, Int.mul_comm e (e ^ k)]


theorem controlled_elimination (a b c e d f h0 : Int)
    (x h u : Nat → Int)
    (initial : h 0 = h0)
    (hiddenStep : ∀ k, h (k + 1) = e * h k + (c * x k + f * u k))
    (observedStep : ∀ k, x (k + 1) = a * x k + b * h k + d * u k)
    (k : Nat) :
    x (k + 1) = a * x k + d * u k + b * (e ^ k * h0) +
      b * memory e (fun j => c * x j) k +
      b * memory e (fun j => f * u j) k := by
  rw [observedStep, hidden_representation e h0 _ h initial hiddenStep k,
      propagated_power, memory_add]
  simp [Int.mul_add, Int.add_assoc, Int.add_comm, Int.add_left_comm]


theorem input_memory_causal (b e f : Int) (u v : Nat → Int) (k : Nat)
    (same : ∀ j, j < k → u j = v j) :
    b * memory e (fun j => f * u j) k = b * memory e (fun j => f * v j) k := by
  rw [memory_prefix e (fun j => f * u j) (fun j => f * v j) k
    (fun j hj => congrArg (fun z : Int => f * z) (same j hj))]

theorem kernel_three (b e g : Int) (v : Nat → Int) :
    b * memory e (fun j => g * v j) 3 =
      b * e * e * g * v 0 + b * e * g * v 1 + b * g * v 2 := by
  simp [memory, Int.mul_add, Int.mul_assoc, Int.add_assoc]

end ControlledMemory

#print axioms ControlledMemory.memory_add
#print axioms ControlledMemory.memory_prefix
#print axioms ControlledMemory.hidden_representation
#print axioms ControlledMemory.propagated_power
#print axioms ControlledMemory.controlled_elimination
#print axioms ControlledMemory.input_memory_causal
#print axioms ControlledMemory.kernel_three
