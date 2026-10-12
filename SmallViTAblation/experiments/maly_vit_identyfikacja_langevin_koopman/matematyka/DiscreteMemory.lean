import Init

/- Exact integer scalar recurrence. Matrix version checked separately with SymPy. -/
namespace DiscreteMemory

def memory (e : Int) (q : Nat → Int) : Nat → Int
  | 0 => 0
  | k + 1 => e * memory e q k + q k

def propagated (e h0 : Int) : Nat → Int
  | 0 => h0
  | k + 1 => e * propagated e h0 k

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


theorem observed_elimination (a b c e d f h0 : Int)
    (x h u : Nat → Int)
    (initial : h 0 = h0)
    (hiddenStep : ∀ k, h (k + 1) = e * h k + (c * x k + f * u k))
    (observedStep : ∀ k, x (k + 1) = a * x k + b * h k + d * u k)
    (k : Nat) :
    x (k + 1) = a * x k + b * (e ^ k * h0) +
      b * memory e (fun j => c * x j + f * u j) k + d * u k := by
  rw [observedStep, hidden_representation e h0 _ h initial hiddenStep k,
      propagated_power]
  simp [Int.mul_add, Int.add_assoc]


theorem memory_three (e : Int) (q : Nat → Int) :
    memory e q 3 = e * e * q 0 + e * q 1 + q 2 := by
  simp [memory, Int.mul_add, Int.mul_assoc, Int.add_assoc]

end DiscreteMemory

#print axioms DiscreteMemory.hidden_representation
#print axioms DiscreteMemory.propagated_power
#print axioms DiscreteMemory.observed_elimination
#print axioms DiscreteMemory.memory_three
