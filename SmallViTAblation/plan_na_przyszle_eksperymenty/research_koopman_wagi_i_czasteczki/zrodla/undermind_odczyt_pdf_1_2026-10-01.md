[Dog20]:
### 1. State Modeled
* **Variables modeled:** The state consists strictly of the network’s trainable weights and biases $\mathbf{w} \equiv [w_1, \dots, w_N]^\top \in \mathbb{R}^N$ (Sec. 2.2, p. 3, Eq. 5). The observable evaluated is the identity observable on the parameters, $I(\mathbf{w}) = \mathbf{w}$ (Sec. 2.3, p. 3, Eqs. 6–8).
* **Optimizer state & updates:** Optimizer internal memory (e.g., first/second gradient moments in Adam, accumulated squared gradients in Adagrad/Adadelta) is **not** included in the state vector (Sec. 2.2–2.3, p. 3; Sec. 3, pp. 4–5).
* **Activations/Representations vs. Weights:** The paper strictly performs **weight trajectory forecasting** (predicting the forward evolution of weight and bias parameters). It does **not** forecast fixed-image 768D token representations, transformer states, or layer activations.

---

### 2. Time Axis, Snapshot Spacing, and Data Matrix Construction
* **Time parameter:** Discrete algorithmic time $t \in \mathbb{Z}$, representing completed training iterations/steps (Sec. 2.2, p. 3, Eq. 5).
* **Snapshot spacing:** Snapshots are recorded at every consecutive training step ($\Delta t = 1$) between indices $t_1$ and $t_2$ (Sec. 2.1, p. 3; Sec. 3, Algorithm 1, p. 5).
  * *DE Solver:* Recorded every iteration from $t_1 = 35{,}000$ to $t_2 = 45{,}000$ ($k = 10{,}000$ consecutive snapshots) (Sec. S9, p. 15, Tables S4–S6).
  * *MNIST Classifier:* Recorded every iteration from the beginning of Epoch 3 ($t_1$) to the end of Epoch 5 ($t_2$) with 938 iterations per epoch ($k = 2{,}814$ consecutive steps) (Sec. S10, p. 16, Table S7).

---

### 3. Finite Koopman Identification, Observables, and Forecast Scheme
* **Identification method:** Finite section approximation via the Moore-Penrose pseudoinverse (Sec. 2.1, p. 3, Eq. 4):
  $$\tilde{U} = F' F^+ = F' (F^\dagger F)^{-1} F^\dagger$$
  where $F = [\mathbf{g}(t_1) \dots \mathbf{g}(t_2 - 1)]$ and $F' = [\mathbf{g}(t_1 + 1) \dots \mathbf{g}(t_2)]$.
* **Observables & Partitioning:** Rather than fitting a single global operator over all $N$ weights ($O(n^6)$ per iteration complexity), the weights are partitioned into local patches (Sec. 3, pp. 4–5; Sec. S7, p. 12):
  * *Node Koopman Operators:* Models incoming weights and bias $\mathbf{w}_j^l$ to node $j$ of layer $l$ as an autonomous sub-system $\mathbf{w}_j^l(t+1) = \tilde{U}_j^l \mathbf{w}_j^l(t)$ (Sec. 3, p. 5, Algorithm 1).
  * *Quasi-Node Koopman Operators:* Subdivides large incoming node vectors into $q$-sized slices (used for MNIST layer 1 with $q = 157$) (Sec. 3, p. 5; Sec. 5, p. 7; Sec. S7, p. 12; Table S7, p. 18).
  * *Single Weight / Layer Koopmans:* Evaluated in complexity analysis and toy settings (Sec. 3, pp. 4–5; Sec. S7, p. 12; Sec. S8, pp. 13–14).
* **Forecast protocol:** **Pure open-loop multi-step forecasting**, without alternating retraining phases. After computing $\tilde{U}$ once at $t_2$, weights are autonomously evolved forward for $T$ consecutive steps via matrix multiplications $\mathbf{w}(t_2 + \tau) = \tilde{U}^\tau \mathbf{w}(t_2)$ (Sec. 3, Algorithm 1, p. 5; Sec. 4.1, p. 6; Sec. 6, p. 9).

The schematic and weight groupings are illustrated below:

Architecture-dependent partitioning strategies (Single Weight, Node, Layer, and Network) used to decompose the full Koopman operator into tractable sub-operators.
[REGION: page=4, bbox=634,169,910,830]

---

### 4. Treatment of Optimizer Components, Minibatches, and Normalization
* **Stochastic Minibatches:**
  * In the DE solver, batch stochasticity was removed: trained with a single fixed full batch (Sec. 4, p. 5).
  * In MNIST, stochastic minibatches of size 64 were used (Table S7, p. 18). The operator was fit directly on the resulting noisy weight trajectories without stochastic modeling extensions (Sec. 5, p. 8; Sec. S10, p. 16).
* **Momentum & Adaptive Optimizer States:** Ignored. Neither momentum buffers nor adaptive learning rate histories are tracked or passed to the linear operator (Sec. 2.2–2.3, p. 3; Sec. 3, p. 5).
* **Learning Rate Scheduling:**
  * DE solver: Explicit step-dependent schedule $\eta(t) = 8/(1000 + t)$ (Sec. 4, p. 5; Tables S4–S6, pp. 15–16).
  * MNIST: Constant learning rate $\eta = 1.0$ (Table S7, p. 18).
* **Batch Normalization:** Not evaluated or mentioned; networks were standard unnormalized MLPs (Sec. 4, p. 5; Sec. 5, p. 7).

---

### 5. Evaluated Networks, Datasets, Optimizers, and Horizons

Summary of performance benchmarks and speedups across evaluated setups:

Numerical benchmark results for the NN DE solver (Table 1, showing success rate, equivalent iteration ratio, and speedup for Adam, Adagrad, and Adadelta) and MNIST (Table 2).
[REGION: page=7, bbox=606,285,712,713]
[REGION: page=8, bbox=710,285,798,713]

1. **Perceptron (Toy Problem):**
   * *Architecture:* 4 inputs, 2 outputs, single-layer step activation (Sec. S8.2, p. 13, Table S3).
   * *Dataset:* Synthetic 4-bit vectors performing logical OR (Sec. S8.2, p. 13).
   * *Optimizer & Horizon:* Perceptron learning rule ($\eta = 0.005$); Koopman applied at step 100 up to step 1,000 (Sec. S8.2, pp. 13–14).
2. **Hamiltonian Differential Equation (DE) Solver:**
   * *Architecture:* Fully connected MLP, 1:10:10:2 (1 input, two hidden layers of 10 units each, 2 outputs; 152 parameters), sigmoid activations (Sec. 4, p. 5; Sec. S9, p. 15; Table 1, p. 7).
   * *Dataset/Objective:* Unsupervised Hamiltonian integration loss (Sec. 4, p. 5).
   * *Optimizers:* Adam, Adagrad, Adadelta (Sec. 4, p. 5; Tables S4–S6, pp. 15–16).
   * *Horizon:* Data collected over $t \in [35{,}000, 45{,}000]$ ($k = 10{,}000$ steps); extrapolated forward for $T = 2{,}500$ steps (Sec. S9, pp. 15–16; Fig. 2b,d, p. 6).
3. **MNIST Digit Classifier:**
   * *Architecture:* Fully connected MLP, 784:20:20:20:10 (3 hidden layers of 20 units; 16,550 parameters), ReLU activations (Sec. 5, p. 7; Table 2, p. 8; Table S7, p. 18).
   * *Dataset:* MNIST (batch size 64, test batch 1,000) (Sec. 5, p. 7; Table S7, p. 18).
   * *Optimizer:* Stochastic Adadelta ($\eta = 1.0$, $\rho = 0.999$, $\gamma = 0.7$) (Sec. 5, p. 7; Table S7, p. 18).
   * *Horizon:* Data collected from Epoch 3 to Epoch 5 ($t_1$ to $t_2$, $k = 2{,}814$ steps); forecast horizon $T = 2$ epochs ($1{,}876$ steps) (Sec. S10, pp. 16, 18).

---

### 6. Success and Convergence Criteria
* **Equivalent Iterations ($T^{\mathrm{eq}}$):** The number of conventional optimizer iterations required to achieve the identical loss obtained after $T$ Koopman training iterations (Sec. 4.1, p. 6).
  * *Criterion for Success:* Defined as $T^{\mathrm{eq}}/T > 0$ (confirming the forecast lowered network loss below the loss at $t_2$) (Sec. 4.1, p. 6).
  * *Interpolated Epoch Formulation (MNIST):* For epoch-based evaluation, $T^{\mathrm{eq}} = Q + R$, where $Q = \max_i \{i \mid \ell_i > \ell_T^{\mathrm{KO}}\}$ and $R = (\ell_Q - \ell_T^{\mathrm{KO}})/(\ell_Q - \ell_{Q+1})$ (Sec. S10, p. 17, Eq. 12).
* **Observed Success Rates:** 100% for Adadelta on DE solver ($T^{\mathrm{eq}}/T = 1.00$), 92% for Adagrad ($T^{\mathrm{eq}}/T = 1.00$), 84% for Adam ($T^{\mathrm{eq}}/T = 0.85$) (Table 1, p. 7), and 100% for MNIST ($T^{\mathrm{eq}}/T = 0.97$) (Table 2, p. 8).

---

### 7. Speedup Claims, Operator Fitting Costs, and Hardware Setup
* **Reported speedups:** $98\times$ (Adam), $132\times$ (Adagrad), $141\times$ (Adadelta) on the DE solver (Table 1, p. 7), and $15\times$ on MNIST (Table 2, p. 8).
* **Does speedup include fitting costs?** **No.** The empirical speedup compares the forward stepping execution time of $T$ Koopman steps against $T^{\mathrm{eq}}$ standard backpropagation steps (Sec. 4.3, p. 7; Sec. S10, p. 17, Eq. 13). Operator construction complexity ($O(\max\{kn^4, n^5\})$ for Node KOs) is analyzed separately as an upfront offline cost (Sec. S7, p. 12).
* **Hardware comparison:** Both algorithms ran serially on identical hardware: a Dell workstation with a 3.4 GHz Intel Core i7-6700 CPU and 32 GB RAM (Sec. S9, p. 15). Backpropagation baseline used PyTorch CPU, while Koopman matrix operations were executed in MATLAB 2018b without parallelization (Sec. 4.3, p. 7; Sec. S9, p. 15).

---

### 8. Explicit Scope and Limitations
* **Architectural scope:** Limited exclusively to small, feedforward, fully connected networks (MLPs); convolutional layers, recurrent units, attention/transformers, or normalization layers were not evaluated (Sec. 1, p. 2; Sec. 6, p. 8; Sec. S10, p. 16).
* **Basin of attraction requirement:** The method relies on the Hartman-Grobman theorem and assumes weights have already settled into a hyperbolic fixed-point basin / local minimum before fitting ($t_2 \gg 0$) (Sec. 3, p. 4; Sec. S7, p. 13).
* **Heuristic hyperparameters:** Window start $t_1$ and transition point $t_2$ were selected heuristically via fixed constants rather than detected adaptively (Sec. 6, p. 9).
* **Weight correlation failure mode:** Partitioned operators degrade if weights update in an uncorrelated manner (e.g., when scaling input units in perceptrons with sparse activations) (Sec. S8.3, pp. 14–15).
* **Finite horizon:** Open-loop extrapolation is only valid across a restricted horizon $T$ (e.g., 2,500 iterations or 2 epochs) before predictions drift (Sec. 1, p. 1; Sec. 4.1–4.2, pp. 6–7).

---

[Tan20b]:
### 1. State Modeled
* **State variables:** The modeled state consists strictly of flattened layer-wise weight matrices $W^{\ell,m} \in \mathbb{R}^{n \times m}$ for layer $\ell$ across $m$ optimization steps (Section 3, pp. 2–3; Algorithm 1, p. 5). Biases are not explicitly distinguished or detailed.
* **Excluded quantities:** Optimizer internal memory (e.g., Adam first and second moments), intermediate updates, layer activations, and network outputs are **not** modeled or tracked by DMD.
* **Forecast type:** The paper demonstrates **weight trajectory forecasting** across optimization steps. It does **not** forecast fixed-image 768D token representations.

---

### 2. Time Axis and Snapshot Spacing
* **Time axis & spacing:** Optimization steps (backpropagation iterations/epochs). Snapshots are collected sequentially at unit spacing ($\Delta t = 1$ optimization step) for $m$ consecutive steps before each DMD update (Section 3, pp. 2–4; Algorithm 1, p. 5).

---

### 3. Finite Koopman / DMD Identification, Observables, and Forecasting Scheme
* **Observables:** Direct identity observables of layer weights $w^{\ell}$ (Section 3, p. 3).
* **Identification:** Standard exact/projected DMD applied to lagged matrices $W^{\ell,m-}$ (first $m-1$ steps) and forwarded matrices $W^{\ell,m+}$ (last $m-1$ steps) (Section 3, pp. 3–4, Eqs. 1–3):
  * Economy SVD via snapshot Gramian: $W^{\ell,m-\top} W^{\ell,m-} = V_r \Sigma_r^2 V_r^\top$, reconstructing $U_r = W^{\ell,m-} V_r \Sigma_r^{-1}$ (p. 3).
  * Reduced Koopman matrix: $A_r^{\ell,m} = U_r^{\ell,m\top} W^{\ell,m+} V_r^{\ell,m} (\Sigma_r^{\ell,m})^{-1}$ (p. 4, Eq. 3).
  * Eigendecomposition: $A_r^{\ell,m} Y_r^{\ell,m} = \Lambda_r^{\ell,m} Y_r^{\ell,m}$ (p. 4, Eq. 4).
  * Mode computation: $\Phi_r^{\ell,m} = U_r^{\ell,m+} Y^{\ell,m}$ (p. 4).
* **Forecasting rule:** Weights at step $s > m$ are extrapolated via:
  $$w^{\ell,s} = \Phi_r^{\ell,m} (\Lambda_r^{\ell,m})^{s-m} b_r^{\ell,m}, \quad b_r^{\ell,m} = (\Phi_r^{\ell,m})^\top w^{\ell,m} \quad \text{(Section 3, p. 4, Eq. 5)}$$
* **Alternating scheme:** Optimization alternates between $m$ standard backpropagation steps and a single DMD leap of $s$ steps, after which weights are overwritten and backpropagation resumes (Algorithm 1, p. 5).

---

### 4. Treatment of Minibatches, Optimizer Memory, Schedulers, and Normalization
* **Stochastic minibatches & noise:** Minibatch noise is filtered strictly by retaining only the leading $r$ singular modes based on a relative singular value cutoff ($10^{-10}$) (Section 3, p. 4; Section 4, p. 6).
* **Momentum / adaptive optimizer state:** Unaddressed. The paper does not describe updating, projecting, or resetting Adam momentum and variance vectors after DMD weight assignments.
* **Learning-rate scheduling:** Not utilized; the text notes that DMD steps implicitly use an effective learning rate of 1.0 (Section 4, p. 7).
* **Batch normalization:** Not present or discussed.

---

### 5. Benchmark Setup, Horizons, and Success Criteria
* **Dataset:** 2D advection-diffusion-reaction PDE system predicting atmospheric dispersion of pollutant $c_3$ from reactants $c_1, c_2$ governed by 6 uncertain parameters ($K_{12}, K_3, D, U_0, u_h, u_v$) (Section 4 & Appendix 1, pp. 5–6, 10–13). 1,000 Latin hypercube simulation samples (800 train, 200 test); target output is concentration at 2,670 spatial coordinates (Section 4, pp. 5–6).
* **Network & optimizer:** Feed-forward DNN with 3 hidden layers (40, 200, 1,000 units), soft-sign activations, Xavier initialization, $\sim 2.9 \times 10^6$ parameters, trained with Adam on MSE loss for 3,000 epochs (Section 2, p. 2; Section 4, pp. 5–6).
* **Horizon exploration:** Sensitivity studied for $m \in [2, 20]$ snapshots and extrapolation steps $s \in [5, 100]$ (Section 4, pp. 6–7). Optimal selection: $m = 14, s = 55$.

Heatmaps showing the mean relative error as a function of the number of snapshot steps $m$ and extrapolation horizon $s$ demonstrate error minima around $m=14$ to $20$ and $s \approx 55$:
[REGION: page=7, bbox=176,183,299,822]

* **Success criteria:** Mean squared error reduction over epochs and mean relative error per DMD application ($\text{MSE}_{\text{after DMD}} / \text{MSE}_{\text{before DMD}}$) (Section 4, pp. 6–7).

Training and test mean squared error curves across 3,000 epochs show that DMD acceleration lowers the loss by approximately two orders of magnitude compared to baseline Adam:
[REGION: page=8, bbox=104,178,284,792]

---

### 6. Wall-Clock Speedup, Computational Overhead, and Hardware
* **Speedup claim:** There is **no wall-clock speedup**. The authors explicitly state:
  > "the wall time necessary to train the DNN with DMD iterations was 1.41 times larger" (Section 4, p. 8)
* **Operator fitting & data transfer overhead:** Although operation count predicts a $1.07\times$ computational overhead ($t < 3m^2 + r^2$), the actual wall-clock training was 41% slower due to extracting/assigning weights in TensorFlow and executing DMD on host CPU rather than GPU (Section 4, p. 8).
* **Error gain:** While wall time increased, final MSE was roughly 100 times lower after 3,000 epochs (Section 4, p. 7; Section 5, p. 9).

---

### 7. Limitations and Explicit Scope
* **Nonlinear trajectory drift:** Large extrapolation steps ($s$) degrade accuracy because weight trajectories depart from linear Koopman approximations (Section 4, p. 7).
* **Diminishing returns:** Late in training, large DMD jumps become less effective when MSE is already small, indicating a need for annealing (Section 4, pp. 7–8).
* **Damping of stochasticity:** Truncating modes filters out noise, which could harm escape from saddle points or local minima in other architectures (Section 4, p. 8).
* **Scope:** Demonstrated solely on a single physical regression problem on a fully connected MLP architecture (Section 4, pp. 5–8).

---

### References
* **[1]** M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, et al. Tensorflow: A system for large-scale machine learning. In *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, pages 265–283, 2016.

---

[Lin21b]:
### 1. How Data-Driven Mori–Zwanzig Extends Finite Koopman Modeling Under Partial Observations

In the Koopman framework, nonlinear dynamics are mapped to an infinite-dimensional linear system governing observables in a Hilbert space $\mathcal{H}$ (Sec. 2, p. 4; Sec. 3.2, p. 8). Under partial observations, one chooses a finite set of $M$ observables $\mathcal{M} = \{g_i\}_{i=1}^M$ spanning a subspace $\mathcal{H}_g \subset \mathcal{H}$. 

- **Finite Koopman modeling (e.g., EDMD):** Seeks an $M \times M$ linear operator $\mathbf{K}_\delta^{\text{Koop}}$ minimizing one-step mean squared error (Sec. 4, Eqs. 45–46, pp. 18–19). As the sample size goes to infinity, this yields:
  $$\mathbf{K}_\delta^{\text{Koop}} = \mathbf{C}(\delta)\mathbf{C}^{-1}(0)$$
  where $\mathbf{C}(t) := \langle \mathbf{g}(t), \mathbf{g}^T(0) \rangle$ (Eq. 47, p. 19). Multi-step predictions are obtained by recursively powering this matrix: $\mathbf{g}(n\delta) \approx (\mathbf{K}_\Delta^{\text{Koop}})^{n/k} \mathbf{g}(0)$ (Sec. 6.2, p. 28). When the system is not closed in $\mathcal{H}_g$, this assumes a purely memoryless (Markovian) propagation within $\mathcal{H}_g$ and completely ignores the unresolved subspace $\mathcal{H}_{\bar{g}}$.
- **Data-driven Mori–Zwanzig (MZ) extension:** 
  1. **Subspace decomposition:** MZ decomposes the full Hilbert space into $\mathcal{H}_g = \text{span}(\mathcal{M})$ and its orthogonal complement $\mathcal{H}_{\bar{g}}$ using an inner product (Sec. 3.2, p. 9).
  2. **Operator generalization:** The lowest-order MZ discrete operator $\mathbf{\Omega}_\Delta^{(0)}$ is mathematically identical to the EDMD Koopman operator $\mathbf{C}(\Delta)\mathbf{C}^{-1}(0)$ (Sec. 4, p. 19–20; Sec. 5, p. 21). However, the paper proves that $\mathbf{\Omega}_\Delta^{(0)}$ is *not* simply $\exp(\Delta \mathbf{M})$ (where $\mathbf{M}$ is the continuous-time Markov operator), but inherently includes integrated memory effects over $[0, \Delta)$ (Lemma 1, Appendix B.1, p. 20, 26, 38–39). Thus, repeatedly powering the single-step operator accumulates error.
  3. **Inclusion of non-Markovian memory:** Rather than stopping at the lowest-order Markov term, the data-driven MZ framework computes higher-order discrete memory kernels $\mathbf{\Omega}_\Delta^{(\ell)}$ ($\ell \ge 1$) from two-time correlation matrices (Sec. 5, Eq. 52, p. 22). This captures the "echo" of resolved variables propagating into unresolved degrees of freedom and returning at a later time (Sec. 3.2, p. 11).
  4. **Utilizing past trajectory history:** By convolving past observable states with the memory kernel, the discrete Generalized Langevin Equation (GLE) incorporates partial information about the orthogonal state $\mathbf{g}_\perp$, yielding significantly reduced prediction errors compared to memoryless Koopman operators (Sec. 6.2, Fig. 7, p. 28–29).

Below is the geometric interpretation showing how Mori's projection maps full Hilbert space observables onto $\mathcal{H}_g$:

Figure 1 illustrates the decomposition of observable dynamics $\mathbf{g}(t)$ into a parallel component $\mathbf{g}_\parallel(t) \in \mathcal{H}_g$ and an orthogonal component $\mathbf{g}_\perp(t) \in \mathcal{H}_{\bar{g}}$ via Mori's linear projection operator $\mathcal{P}$.
[REGION: page=12, bbox=88,185,255,815]

---

### 2. Projection Assumptions

The derivation and algorithms rely on the following explicit assumptions:
1. **Hilbert Space and Measure:** Observables $g$ belong to $\mathcal{H} = L^2(\mathbb{R}^N, \mu)$, where $\mu$ is an invariant or stationary measure (Sec. 3.1, p. 6; Sec. 3.2, p. 8).
2. **Inner Product Definition:** The inner product is defined via continuous long-time trajectory averaging (Sec. 3.4, Eq. 27, p. 14):
   $$\langle f, g \rangle := \lim_{T \to \infty} \frac{1}{T} \int_0^T (f \circ \mathbf{\Phi}(s))(g \circ \mathbf{\Phi}(s))\, ds$$
   Trajectories must be bounded, and the dynamical system is assumed ergodic so that temporal averages converge to state-space expectations and boundary terms vanish under integration by parts (Sec. 3.6, p. 16).
3. **Linear Independence:** The finite set of observables $\mathcal{M} = \{g_i\}_{i=1}^M$ is assumed linearly independent, ensuring that the autocorrelation matrix $\mathbf{C}(0) = \langle \mathbf{g}, \mathbf{g}^T \rangle$ is full-rank and invertible (Sec. 3.4, p. 13; Sec. 4, p. 19).
4. **Mori's Linear Projection:** The projection operator $\mathcal{P}: \mathcal{H} \to \mathcal{H}_g$ is linear (Sec. 3.1, Eq. 11, p. 7; Sec. 3.4, Eq. 26, p. 13):
   $$\mathcal{P}f = \langle f, \mathbf{g}^T \rangle \cdot \langle \mathbf{g}, \mathbf{g}^T \rangle^{-1} \cdot \mathbf{g}$$
   This contrasts with Zwanzig's nonlinear conditional-expectation projector (Eq. 12, p. 7).
5. **Discrete vs. Continuous Inner Product Consistency:** For the discrete-time formulation derived via time discretization, the discrete correlation matrix $\mathbf{C}_\Delta(k\Delta)$ must match the continuous-time correlation $\mathbf{C}(k\Delta)$ (Appendix B.3, Eqs. 87–88, p. 43).

---

### 3. Exact Discrete GLE and Operator Formulas

All discrete-time formulas below are exact for any arbitrary sampling interval $\Delta > 0$:

- **Discrete-Time Generalized Langevin Equation (GLE):**
  *(Theorem 2, Eq. 41, p. 17; Eq. 74, p. 40)*
  $$\mathbf{g}((k+1)\Delta) = \sum_{\ell=0}^k \mathbf{\Omega}_\Delta^{(\ell)} \cdot \mathbf{g}((k-\ell)\Delta) + \mathbf{W}_k$$
- **Correlation Matrix Evolution:**
  *(Corollary 2, Eq. 42, p. 17; Eq. 72, p. 40)*
  $$\mathbf{C}((k+1)\Delta) = \sum_{\ell=0}^k \mathbf{\Omega}_\Delta^{(\ell)} \cdot \mathbf{C}((k-\ell)\Delta)$$
- **Projected Image Evolution:**
  *(Corollary 3, Eq. 43, p. 17; Eq. 73, p. 40)*
  $$\mathcal{P}\mathbf{g}((k+1)\Delta) = \sum_{\ell=0}^k \mathbf{\Omega}_\Delta^{(\ell)} \cdot \mathcal{P}\mathbf{g}((k-\ell)\Delta)$$
- **Recursive Operator Formulas:**
  *(Sec. 5, Eq. 52, p. 22; Appendix B.1, Eq. 67, p. 39; Algorithm 2, p. 49)*
  - Lowest-order (Markov / Koopman) operator:
    $$\mathbf{\Omega}_\Delta^{(0)} = \mathbf{C}(\Delta) \cdot \mathbf{C}^{-1}(0)$$
  - Higher-order memory operators ($k \ge 1$):
    $$\mathbf{\Omega}_\Delta^{(k)} = \left[ \mathbf{C}((k+1)\Delta) - \sum_{\ell=0}^{k-1} \mathbf{\Omega}_\Delta^{(\ell)} \cdot \mathbf{C}((k-\ell)\Delta) \right] \cdot \mathbf{C}^{-1}(0)$$
- **Discrete Orthogonal Dynamics (Noise):**
  *(Appendix B.1, Eq. 77, p. 41; Algorithm 2, p. 49)*
  $$\mathbf{W}_k = \mathbf{g}((k+1)\Delta) - \sum_{\ell=0}^k \mathbf{\Omega}_\Delta^{(\ell)} \cdot \mathbf{g}((k-\ell)\Delta)$$
- **Discrete Generalized Fluctuation–Dissipation (GFD) Relation:**
  *(Appendix B.4, Eq. 89, p. 44)*
  $$\mathbf{\Omega}_\Delta^{(k)} = - \langle \mathbf{W}_k, \mathbf{W}_0^T \rangle \cdot \mathbf{C}^{-1}(-\Delta)$$
  where $\mathbf{C}(-\Delta) = \mathbf{C}^T(\Delta)$.

---

### 4. Meaning of Orthogonal Residual vs. Future Random Inputs

- **Deterministic Orthogonal Residual:** In a deterministic continuous system, the orthogonal dynamics $\mathbf{F}(t) = \mathcal{L}_{\mathcal{M}\bar{\mathcal{M}}} e^{t\mathcal{L}_{\bar{\mathcal{M}}\bar{\mathcal{M}}}} \mathbf{g}_{\bar{\mathcal{M}}}(0)$ (Eq. 33, p. 15) and its discrete equivalent $\mathbf{W}_k$ (Eq. 77, p. 41) are **entirely deterministic**. They represent the influence of the unresolved initial conditions $\mathbf{g}_{\bar{\mathcal{M}}}(0)$ propagating through the orthogonal Liouville generator $\mathcal{Q}\mathcal{L}$. As the authors state, it is termed "noise" solely because:
  > *"its resemblance of a Langevin noise in a Langevin equation"* (Sec. 3.2, p. 11).
- **Practical Handling During Prediction:** Because the unresolved coordinates are not measured, future values of $\mathbf{F}(t \ge m\delta)$ and $\mathbf{W}_k$ cannot be calculated. For making out-of-sample future predictions, $\mathbf{W}_k$ is **set to zero** (Sec. 6.2, p. 28).
- **Distinction from Future Random Inputs:** The model does not add artificial stochastic Wiener or Gaussian noise inputs. Instead, the past trajectory $\mathbf{g}(k\delta)$ contains latent traces of the past orthogonal component $\mathbf{g}_\perp$. The memory convolution propagates these historical orthogonal contributions into the future, improving predictions over pure Koopman projections without needing noise injection (Sec. 6.2, p. 28–29).

---

### 5. Numerical Systems Tested

1. **Lorenz '96 System:** (Sec. 6.1, Eq. 53, p. 22–24)
   $$\frac{d}{dt}\phi_i(t) = (\phi_{i+1} - \phi_{i-2})\phi_{i-1} - \phi_i + F, \quad i = 1, \dots, N$$
   Tested with $N = 20$ physical variables, forcing parameter $F = 8$ (chaotic, largest Lyapunov exponent $\approx 1.608$, Lyapunov time $\approx 0.622$, Kaplan–Yorke dimension $\approx 13.37$).
   - **Observables:** 5 observables were chosen: four state variables $g_1 = \phi_1, g_2 = \phi_4, g_3 = \phi_8, g_4 = \phi_{14}$ plus a constant $g_0 = 1$ to account for non-zero means (p. 23).
2. **1D Harmonic Oscillator:** (Appendix B.3, p. 43)
   $$\dot{x} = p, \quad \dot{p} = -x$$
   Examined as an analytical counterexample to highlight differences between continuous-time and discrete-time inner product measures.

Figure 7 below displays the resulting prediction errors on the Lorenz '96 system:

Figure 7 compares prediction errors across different prediction horizons and memory lengths, demonstrating that incorporating memory lowers error compared to memoryless Koopman ($\text{Memory}=0$), but error plateaus at a finite memory timescale.
[REGION: page=29, bbox=90,140,240,860]

---

### 6. Limits for AdamW Training Plus BatchNorm

- **Fact:** The paper **does not use, discuss, or evaluate AdamW training or Batch Normalization (BatchNorm)** anywhere in the text. 
- The proposed algorithms (Algorithms 1 and 2, pp. 48–49) are **purely linear-algebraic and non-iterative**, relying exclusively on:
  1. Direct empirical estimation of correlation matrices $\mathbf{C}(k\Delta)$ from trajectories via time averages.
  2. Matrix inversion of $\mathbf{C}(0) \in \mathbb{R}^{M \times M}$.
  3. Direct recursive linear substitution for $\mathbf{\Omega}_\Delta^{(k)}$ and $\mathbf{W}_k$.
- The only mention of neural networks is in the discussion (Sec. 7, p. 32), referencing external work by Yeung et al. [50] as a potential future direction to optimize observable dictionaries.

---

### 7. Can Finite History Guarantee Full State Reconstruction?

- **Fact (from the paper):** **No.** A finite history does **not** guarantee full state reconstruction:
  1. In Section 7 (p. 31), the authors discuss time-delay embedding (Takens' theorem) and explicitly note that:
     > *"with a finite ($h < \infty$) past history, the Mori–Zwanzig memory kernel would emerge"* (Sec. 7, p. 31)
     if the past $h$ steps do not linearly span an invariant subspace.
  2. In Section 6.2 (p. 29) and Figure 7, as past history length increases, the prediction error decreases but **levels off at a finite timescale**, leaving a substantial residual error.
  3. Because the future orthogonal dynamics $\mathbf{F}(t \ge m\delta)$ remain unmeasured and are set to zero, exact reconstruction of the full phase space from finite observables and finite history is not possible without a complete basis (Sec. 3.2, p. 11; Sec. 6.2, p. 28–30).
- **Extrapolation / Context:** From dynamical systems theory, Takens' theorem guarantees that a finite delay embedding of a generic single observable can reconstruct the differential topology of an attractor. However, in the linear Mori–Zwanzig and Koopman projection context with a fixed, small observable dictionary, finite linear memory only recovers the projection of the orthogonal dynamics onto the linear history, not the exact unobserved microstate.

---

### References
* [25] E. N. Lorenz, *Predictability: A problem partly solved*, in Proc. Seminar on predictability, vol. 1, 1996.
* [29] H. Mori, *Transport, collective motion, and Brownian motion*, Progress of Theoretical Physics, 33 (1965), pp. 423–455.
* [47] M. O. Williams, I. G. Kevrekidis, and C. W. Rowley, *A data–driven approximation of the koopman operator: Extending dynamic mode decomposition*, Journal of Nonlinear Science, 25 (2015), pp. 1307–1346.
* [50] E. Yeung, S. Kundu, and N. Hodas, *Learning deep neural network representations for koopman operators of nonlinear dynamical systems*, in 2019 American Control Conference (ACC), 2019, pp. 4832–4839.
* [51] R. Zwanzig, *Nonlinear generalized Langevin equations*, Journal of Statistical Physics, 9 (1973), pp. 215–220.

---

[Iac22b]:
### 1. Page, Equation, and Theorem Showing State- and Input-Dependency in Discrete Time

* **Theorem and Equations:** **Theorem 2** and **Equations (39a)–(39b)** (p. 5), along with the factorization in **Equations (45)–(47)** and **Equation (51)** (p. 6).
* **Why it occurs:** In discrete time, the absence of a chain rule for the shift operator requires using the Fundamental Theorem of Calculus (FTC) along the line segment between $f(x_k)$ and $f(x_k) + g(x_k, u_k)$. Integrating the observable Jacobian along this segment yields:
  $$\mathcal{B}(x_k, u_k) = \left( \int_{0}^{1} \frac{\partial \Phi}{\partial x}\big(f(x_k) + \lambda g(x_k, u_k)\big) \, d\lambda \right) g(x_k, u_k) \quad \text{(Eq. 39b, p. 5)}$$
  Even for input-linear systems $x_{k+1} = f(x_k) + b u_k$ (Eq. 50, p. 6), the resulting matrix is:
  $$B(x_k, u_k) = \left( \int_{0}^{1} \frac{\partial \Phi}{\partial x}\big(f(x_k) + \lambda b u_k\big) \, d\lambda \right) b \quad \text{(Eq. 51, p. 6)}$$
  The authors emphasize on page 6 that the input matrix function $B$ inherently maintains a dependency on both state and input.

---

### 2. Does $A\Phi(x) + Bu$ Follow Automatically for Arbitrary Controls?

**No.** 
* A strictly linear time-invariant (LTI) form with constant $B$ does not hold generally. 
* As proven in Sections 3.2–3.3 (pp. 5–6) and demonstrated in Section 4 and Section 5.3 (pp. 6–10, Fig. 3–5), forcing a constant $\hat{B}$ constitutes an approximation that can incur substantial modeling error (Theorem 3, p. 6). 
* The exact finite-dimensional lifting is not LTI, but rather a Linear Parameter-Varying (LPV) representation:
  $$z_{k+1} = A z_k + B_z(p_k) u_k, \quad p_k = \mu(z_k, u_k) \quad \text{(Eq. 47, p. 6)}$$

---

### 3. Exact Assumptions and Scope

From Section 3.1, Theorem 2 (p. 5), and Lemma 1 (p. 4):
1. **State-Space Form:** Discrete-time system $x_{k+1} = f_d(x_k, u_k) = f(x_k) + g(x_k, u_k)$ with $g(x_k, 0) = 0$ (Eq. 34–35).
2. **Domain Properties:** State domain $\mathbb{X} \subseteq \mathbb{R}^{n_x}$ is compact, forward invariant under $f(\cdot)$, and convex (Theorem 2, p. 5).
3. **Input Domain:** $U \subseteq \mathbb{R}^{n_u}$ is a convex set containing the origin (Lemma 1, p. 4).
4. **Smoothness:** Observables $\Phi: \mathbb{X} \to \mathbb{R}^{n_f}$ are continuously differentiable ($C^1$).
5. **Subspace Invariance:** The autonomous dynamics admit a finite-dimensional Koopman invariant subspace: $\Phi \circ f \in \operatorname{span}\{\Phi\}$ (Eq. 32, p. 5).

---

### 4. Implications for Modeling Nonlinear Minibatch Inputs to AdamW

*(Note: The paper does not study AdamW or optimization algorithms.)*

Applying these theoretical results to an optimization setting such as AdamW yields several direct implications:
* **Discrete-Time Step Formulation:** An optimizer step $x_{k+1} = f_d(x_k, u_k)$ treats the optimizer state (weights, first/second moments) as $x_k$, while minibatch data/gradients act as the external input $u_k$.
* **Nonlinear Input Coupling:** In AdamW, the minibatch gradient enters nonlinearly via moment updates, denominator rescaling ($\sqrt{v_k} + \epsilon$), and weight decay.
* **Failure of LTI Embedding:** Because the DT Koopman input matrix integrates $\frac{\partial \Phi}{\partial x}(f(x_k) + \lambda g(x_k, u_k))$ over $\lambda \in [0, 1]$ (Eq. 39b), any nonlinear observable $\Phi$ combined with input coupling will force the lifted input operator to be both state- and input-dependent.
* **Necessity of LPV Modeling:** Attempting to model AdamW's trajectory under minibatches via a standard LTI Koopman model ($A z_k + B u_k$) will incur persistent approximation bias (Theorem 3). To represent the dynamics faithfully without severe state error, an LPV framework with scheduling variables $p_k = \mu(z_k, u_k)$ is required.

---

### Reference
* **[Iac22b]**: L. C. Iacob, R. Tóth, and M. Schoukens, "Koopman Form of Nonlinear Systems with Inputs," preprint submitted to *Automatica*, arXiv:2207.12132v2, 2023.

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
