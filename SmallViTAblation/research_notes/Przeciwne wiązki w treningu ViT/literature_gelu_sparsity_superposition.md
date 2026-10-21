# Literature check: GELU negative region, activation sparsity in training, negative biases, superposition, and the "opposite beams" mechanism (ViT, 2026-10-21)

Scope: whether published work supports our mechanism. Our claim: opposite per-token contributions of two MLP neurons come from nearly parallel output weights combined with opposite GELU signs. Neurons drift toward negative effective bias and spend most tokens in GELU's negative tail, and the opposition is a by-product, not a cancellation circuit.

Sources and tools: Undermind workspace 40294fa6-c9a3-4469-9a61-d3eddeaee893. Two new deep searches were run on 2026-10-20/21:
- "GELU negative region usage and activation sparsity emergence during transformer and ViT training" (27 papers).
- "Antipodal feature pairs negative biases and neurons writing opposite directions in superposition" (48 papers).

We also reused the existing search "How individual MLP neuron contributions add up and cancel on a single token…" (97 papers). Full texts were read with Undermind read_pdfs for 27 papers, marked [FT] below. Scite was used for metadata, DOI checks and full-text excerpts, marked [Scite-excerpt]. Papers already covered in `SmallViTAblation/docs/2026-10-20_literatura_przeciwstawne_neurony/PRACE_PELNE_TEKSTY.md` (Mar25l, Ahm26c, Pat25d, Sam26b, Che23an, …) are only cross-referenced here, not re-summarised.

Formatting note: the cite keys (Li22ap, Elh22, …) are Undermind workspace keys.

---

## KQ1. Activation sparsity in transformers/ViTs: how it develops during training and across layers, and whether anyone measured use of GELU's negative tail

### Takeaway
The canonical sparsity result (Lazy Neuron) is about ReLU exact zeros. For ViT it was obtained by replacing GELU with ReLU. It shows sparsity appearing early in training, with middle layers the sparsest and the first and last layers denser. It does not show "deeper = sparser". GELU-specific evidence exists, but it is about negative pre-activations rather than zeros:
- BERT pre-activation means turn negative early in training (Li22ap App. B).
- The fraction of GELU pre-activations below zero is 64–89% across architectures (Shv26).
- The resting pre-activation of trained GELU gates is about −1.1 to −1.6 in Phi-2 (Tuo26). This is close to our −1.1 and −1.5.

### Cited Findings
- **Li22ap [FT]**: Zonglin Li, Chong You, Srinadh Bhojanapalli, et al. (2022/2023), "The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers", arXiv:2210.06313. The venue is ICLR 2023; the arXiv v2 PDF does not print it.
  - Sparsity is defined as the % of nonzero post-activation MLP entries. ViT-B/16 on ImageNet-21k reaches 6.3% nonzero; T5-Base reaches 3.0% (abstract) and a 2.7% average over layers (min 1.1%, max 12.0%; §1.1). — [arXiv](https://arxiv.org/abs/2210.06313)
  - ViT's GELU was replaced by ReLU so that exact zeros can be counted. Top-1 on ImageNet-21k is 47.78% with GELU vs 47.58% with ReLU (footnote 3). No threshold-based GELU sparsity is reported (App. F). — [arXiv](https://arxiv.org/abs/2210.06313)
  - Training dynamics (T5, Fig. 1): the nonzero fraction falls quickly early in training, from about 50% at initialization to low single digits, then levels off. It does not fall monotonically in every layer. — [arXiv](https://arxiv.org/abs/2210.06313)
  - Depth (§2.1, Fig. 2): "the first few and last few layers tend to be denser than intermediate layers" for both T5 and ViT. This contradicts a simple "deeper layers are sparser" claim. Larger and deeper models are sparser overall (abstract). — [arXiv](https://arxiv.org/abs/2210.06313)
  - GELU and negativity (App. B.1, Fig. B.1–B.2; FAQ App. F): BERT pre-activations have mean 0 at initialization, and the mean "quickly becomes negative" during training. GELU and sigmoid give pre-activation histograms similar to ReLU; tanh gives no sparsity. The FAQ says pre-activations are "strongly biased towards negativity" even with GELU. No bias values are reported. — [arXiv](https://arxiv.org/abs/2210.06313)
  - Proposed cause (Thm 3.1): at random initialization, the expected gradient of the loss with respect to a positive activation is positive (MSE or cross-entropy), so gradient descent shrinks positive activations. The authors say this holds only for the first steps. Sparsity also appears with random labels, random images and "infinite data" (§3). — [arXiv](https://arxiv.org/abs/2210.06313)
- **Wil24b [FT]**: Cody Wild, Jesper Anderson (2024), "Uncovering Layer-Dependent Activation Sparsity Patterns in ReLU Transformers", arXiv:2407.07848 (preprint).
  - Setup: 6-layer decoder, ReLU, 32,768 hidden units per layer, C4. Per-token use falls early in training, starting in the last layer and moving down almost in order, so deeper layers sparsify earlier. It stabilises by about 70k steps.
  - Per-batch use at convergence goes from 13.3% (layer 0) to 95.6% (layer 5). Units turned off per layer: 83.8/15.6/22.5/20.2/10.6/3.8%.
  - Units inactive at 50k steps can be masked with no accuracy change, whereas masking the same number of random units collapses performance.
  - Biases and pre-activations are not measured. — [arXiv](https://arxiv.org/abs/2407.07848)
- **Shv26 [FT]**: Egor Shvetsov, Aleksandr Serkov, Viacheslav Shokorov, Dmitry Redko, Vladislav Goloshchapov, Evgeny Burnaev (2026), "Bug or Feature: Weight Drift, Activation Sparsity and Spikes", arXiv:2605.17659v2 (preprint).
  - "Negative weight drift": layer weight means move downward early in training. The proof is for ReLU at initialization (Thm 1.2–1.3, same expected-gradient argument as Li22ap). The drift also appears on random data, so the authors call it an optimisation effect, not a data effect.
  - Fraction of negative pre-activations, Table 1:

    | Model | ReLU | GELU | SiLU |
    |---|---|---|---|
    | MLP | 70.8% | 63.6% | 61.0% |
    | MaxViT-Tiny | 79.8% | 70.5% | 70.2% |
    | GPT-nano | 89.0% | 89.3% | — |

  - GELU/SiLU show almost no hard sparsity, because drift gives small negative outputs rather than zeros.
  - Mean-centring normalisation can suppress the drift. Weight decay is not isolated as a cause.
  - The MLP experiments have no bias terms, so this drift is in weights, not biases. — [arXiv](https://arxiv.org/abs/2605.17659)
- **Tuo26 [FT]**: Olli Tuomi (2026), "Transformer MLP Gate Thresholds Are Couplings to a Carried Reference Direction", arXiv:2609.31956 (preprint).
  - Exact decomposition of the mean ("resting") gate pre-activation: ⟨w_n·x_ln + b_n⟩ = w_n·b_L + b_n, where b_L is the corpus mean of the LayerNorm-ed MLP input.
  - Phi-2 (GELU) medians, Table 3: resting pre-activation −1.568 (L6), −1.250 (L10), −1.115 (L14), −1.149 (L22). The explicit bias b_n is only about −0.02. The coupling term dominates in ≥99.9% of gates (|w·b_L|/|b_n| ≈ 48–56× in median magnitude).
  - The same holds in GPT-2-medium and Pythia-1.4B (coupling 15–18× and 38–43× the bias), but not in OPT.
  - Causal test (Phi-2, L10): removing the reference direction raises the fraction of activations >0.5 from 0.0105 by +0.086, about 9× the baseline.
  - Training (Pythia-410M): the coupling builds from about 0.16 to 0.38 (steps 512–1000), reaches 0.84–0.95 by step 4000, peaks at step 8000 and then partly relaxes. The duty cycle (fraction of positive activations) falls from about 0.51 to 0.16.
  - Neurons that share an input direction but differ in threshold are not analysed. — [arXiv](https://arxiv.org/abs/2609.31956)
- **Gur24 [FT]**: Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, Dimitris Bertsimas (2024), "Universal Neurons in GPT2 Language Models", TMLR 2024, arXiv:2401.12181.
  - Universal neurons (excess correlation >0.5 across 5 seeds: 4.16% of GPT-2 small, 1.23% of medium) "typically have" large negative input bias, a large negative mean pre-activation and low activation frequency, defined as the fraction of activations >0. These are reported as within-layer percentiles, not absolute values (§4.2, Fig. 3).
  - Activations are GELU (`gelu_new`); negative outputs are not clipped. — [arXiv](https://arxiv.org/abs/2401.12181)
- **Pen23b [FT]**: Ze Peng, Lei Qi, Yinghuan Shi, Yang Gao (2023), arXiv:2309.03004. Explains sparsity through flat minima, implicit adversarial robustness and gradient sparsity, not through negative bias.
  - Vanilla ViT-B/16 on ImageNet-1K has nonzero fractions of 0.104 (train) and 0.087 (test), with ReLU.
  - Direct activation sparsity dominates in shallow ViT layers; gradient sparsity dominates in deep ViT layers.
  - GELU is not tested experimentally ("sparsity is hard to define with GELU"). — [arXiv](https://arxiv.org/abs/2309.03004)
- **Pen26i [FT]**: Ze Peng, Jian Zhang, Lei Qi, Yang Gao, Yinghuan Shi (2026), "Towards the Connection between Activation Sparsity and Flat Minima", arXiv:2605.25612 (preprint). Uses ViT-Base/16 checkpoints on CIFAR-10 and ImageNet-1K. For ImageNet-1K, sparsity gets worse later in training, which the authors hypothesise is due to shrinking LayerNorm affine parameters. — [arXiv](https://arxiv.org/abs/2605.25612)
- **Luo24d [FT]**: Yuqi Luo, Chenyang Song, Xu Han, et al. (2024), "Sparsing Law: Towards Large Language Models with Greater Activation Sparsity", ICML 2025 (PMLR 267), arXiv:2411.02335.
  - Sparsity is defined by output contribution (CETT), because thresholding activation values "is unsuitable for activations with meaningful negative outputs, such as SiLU".
  - With more training data, ReLU sparsity rises while SiLU sparsity falls. The fitted limiting activation ratio is about 6–8% for ReLU and 38–41% for SiLU (Table 2). — [arXiv](https://arxiv.org/abs/2411.02335)
- **Mir23 [FT]**: Iman Mirzadeh, Keivan Alizadeh, Sachin Mehta, Carlo C. Del Mundo, Oncel Tuzel, Golnoosh Samei, Mohammad Rastegari, Mehrdad Farajtabar (2023), "ReLU Strikes Back", ICLR 2024, arXiv:2310.04564.
  - Sparsity of the down-projection input: 1% in Falcon-7B (GELU), 0% in Llama-7B (SiLU), 96–97% in OPT (ReLU).
  - The paper does not report the fraction of negative pre-activations (Fig. 5 shows mass on both sides of zero).
  - Relufication plus fine-tuning recovers most accuracy (Falcon 66.8 → 65.2; Llama 68.4 → 67.1, average zero-shot). — [arXiv](https://arxiv.org/abs/2310.04564)
- **Sza25b [FT]**: Filip Szatkowski, Patryk Będkowski, Alessio Devoto, et al. (2025), "Universal Properties of Activation Sparsity in Modern LLMs", ICLR 2026, arXiv:2509.00454. Uses top-p (L1-mass) induced sparsity, which ignores sign. There is no monotonic depth law and no breakdown by sign. — [arXiv](https://arxiv.org/abs/2509.00454)
- **Voi23 [FT]**: Elena Voita, Javier Ferrando, Christoforos Nalmpantis (2023), "Neurons in Large Language Models: Dead, N-gram, Positional", arXiv:2309.04827.
  - OPT models use ReLU. Dead neurons are concentrated in the first half of the network, reaching more than 70% in some layers of OPT-66B. The second half is fully "alive".
  - GELU is not discussed. — [arXiv](https://arxiv.org/abs/2309.04827)

### Inferences
- Our block-6 numbers (91% of tokens in the negative regime; effective biases −1.5 and −1.1) are typical of published GELU transformers.
  - GPT-nano with GELU has 89.3% negative pre-activations (Shv26).
  - Phi-2 has median resting pre-activations of −1.1 to −1.6 (Tuo26).
  - So this is not an anomaly of our model, and we can cite it as a known phenomenon.
- Our rise–peak–decline trajectory (pairs peak at epochs 80–110, then fall by epoch 200) has partial precedents:
  - Tuo26: the negative operating point builds, peaks at step 8000 and partly relaxes.
  - Pen26i: ViT ImageNet sparsity worsens late in training.
  - Li22ap: sparsity appears early and then plateaus.
  - None of these track neuron pairs.
- Do not write "sparsity is higher in deeper layers (Li et al. 2023)". Li22ap says middle layers are the sparsest. Wil24b (ReLU) says deeper layers sparsify earlier, but final per-batch use is highest in the last layer.
- GELU minimum, computed here (not from a paper): x·Φ(x) reaches its minimum ≈ −0.170 at x ≈ −0.752.
  - At our effective biases: GELU(−1.5) ≈ −0.100 and GELU(−1.1) ≈ −0.149. These are 60–90% of the deepest possible negative output.
  - The derivative is negative there (GELU′(−1.5) ≈ −0.13; GELU′(−1.1) ≈ −0.10). So in this regime the output becomes more negative as the pre-activation rises, until about −0.75. Both neurons write roughly constant small vectors along −W2 on most tokens.

### Gaps
- None of the papers measures the fraction of GELU pre-activations below zero per layer across training checkpoints in a ViT. Shv26 gives one end-of-training number per model; Li22ap gives BERT histograms. This is likely a gap that our measurements fill.
- No paper reports the magnitude of negative GELU outputs (e.g., mean |GELU(z)| for z<0) over training.
- Lazy Neuron's per-layer ViT values exist only as plots (log-scale); no exact per-layer numbers can be quoted.

---

## KQ2. Do GELU/SiLU negative outputs carry information or get used functionally (vs ReLU)?

### Takeaway
Yes, at least in some neurons. Kong et al. (ICLR 2026) clamp negative pre-activations to zero in the top-1% "Wasserstein" neurons per layer. Perplexity roughly doubles, and grammar accuracy drops sharply. Flipping the sign of negative outputs is catastrophic. Gated-neuron work states explicitly that a negative activation writes along −w_out, and that these negative-gate cases matter. On the other hand, ReLU-replacement studies show models can recover after fine-tuning. So the negative tail is used, but it can be re-learned away.

### Cited Findings
- **Kon25e [FT]**: Linghao Kong, Angelina Ning, Micah Adler, Nir Shavit (2025/2026), "Negative Pre-activations Differentiate Syntax", ICLR 2026, arXiv:2509.24198.
  - Models: Pythia 70M–12B (GELU) and Llama-3.1-8B / Mistral-7B / Qwen3-8B (SiLU gate).
  - Intervention: a′ = max(a, 0) before the nonlinearity, applied to the 1% of neurons per layer with the highest Wasserstein distance (WD).
  - Perplexity at the 5% endpoints: 6.50→17.94 (Llama), 5.44→13.88 (Mistral), 7.40→13.78 (Qwen).
  - At 1% per layer: BLiMP 81.2→77.5 and TSE 84.4→55.2 (Llama). Random and perplexity-matched controls show much smaller drops.
  - Flipping the sign of negative post-activations in the top-2% WD neurons raises perplexity from 6.50 to 2,277 (App. A.4).
  - In Pythia, high-WD neurons emerge within about 25k steps; negative–negative differentiated pairs become common early in training (Fig. 7).
  - The paper does not report the overall fraction of negative pre-activations. — [arXiv](https://arxiv.org/abs/2509.24198)
- **Ger25b [FT]**: Sebastian Gerstner, Hinrich Schütze (2025), "Understanding Gated Neurons in Transformers from Their Input-Output Functionality", arXiv:2505.17936.
  - 12 GLU LLMs (SwiGLU; Gemma uses GeGLU). Neurons are classified by cos(w_in, w_out) into enrichment, depletion, conditional, proportional-change and orthogonal-output classes (0.5 threshold).
  - Early-to-middle layers are dominated by conditional enrichment; late layers lean toward depletion. In Llama-3.2-3B, input manipulators are 25% of all neurons and up to 50% in layers 7–11.
  - §3.2: a negative activation scalar (weakly negative Swish gate) means "the contribution is in the −w_out direction". Weak negative gate activation can switch enrichment into depletion behaviour (§4.3).
  - Neuron pairs are not analysed. — [arXiv](https://arxiv.org/abs/2505.17936)
- **Ger26c [FT]**: Sebastian Gerstner, Hilal AlQuabeh, Kentaro Inui, Hinrich Schütze (2026), "Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence", arXiv:2609.18612.
  - Weakening neurons have cos(w_in, w_out) < −0.5. They concentrate in the last layers (e.g., OLMo-7B: 50 in layer 30 and 138 in layer 31, of 526 total).
  - "If x_post < 0, the neuron writes in the direction −w_out". The negative-gate quadrants "contribute substantially" to ablation effects (Fig. 6).
  - Mean-ablating 526 weakening neurons has the largest effects on entropy and loss among the classes tested.
  - No pair analysis; one entropy effect needed a subset of 8 neurons. — [arXiv](https://arxiv.org/abs/2609.18612)
- **Hen16 [FT]**: Dan Hendrycks, Kevin Gimpel (2016), "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415. Defines GELU(x) = xΦ(x) and the tanh approximation. Calls it non-monotonic, with negative outputs for x<0. No minimum value is stated. The only benefit claimed is that curvature and non-monotonicity may help approximate complicated functions. — [arXiv](https://arxiv.org/abs/1606.08415)
- **Mir23 [FT]**: ReLU replacement plus fine-tuning recovers most performance (see KQ1). The authors do not isolate the role of negative outputs. — [arXiv](https://arxiv.org/abs/2310.04564)
- **Luo24d [FT]**: SiLU's negative outputs are "meaningful", so value thresholding would harm performance (§2.2). — [arXiv](https://arxiv.org/abs/2411.02335)
- **Shv26 [FT]**: with GELU, negative drift produces small negative outputs rather than zeros, and hard sparsity stays near zero. — [arXiv](https://arxiv.org/abs/2605.17659)
- **Sam26b [Scite-excerpt]**: Akash Samanta, Manish Pratap Singh, Debasis Chaudhuri (2026), "Hidden not Deleted: How Networks Suppress Entangled Features", arXiv:2609.27593. In their GELU variant the output bias stays near zero, "in contrast to the clearly negative bias that characterizes both ReLU basins". The authors hypothesise that GELU's soft negative region may provide enough suppression without a strongly negative bias (App. B5). — [arXiv](https://arxiv.org/abs/2609.27593)

### Inferences
- Reviewers can cite Kon25e against "the negative regime is functionally irrelevant". We claim only that the opposition is a by-product; we do not claim the negative tail is useless. These are different claims and should be kept apart in the text.
- Our pair is the GELU (non-gated) analogue of Ger25b/Ger26c's sign rule: direction = sign(GELU(z))·W2[:,c]. We can cite them for the principle. Their results are on GLU LLMs, not ViTs.
- On 91% of tokens the negative contributions are nearly constant (see KQ1 numbers). Functionally they may act like a bias in the residual stream, much like the "reference" coupling in Tuo26. Zero-ablation would remove this offset; mean-ablation would not.

### Gaps
- No ViT or detection paper tests clamping GELU negative outputs. The closest are on LLMs (Kon25e) and on GLU gates (Ger25b/26c).
- "Dead neurons in GELU networks" is barely defined in the literature. Voi23's dead neurons are ReLU; Gur24 uses an activation >0 frequency. No GELU-specific "near-dead" criterion was found.

---

## KQ3. Negative biases, interference filtering and antipodal pairs in superposition: how do they relate to our opposite contributions?

### Takeaway
In Toy Models of Superposition, an "antipodal pair" means two features embedded as opposite directions (W_i = −W_j) in the bottleneck. It is not two MLP neurons with opposite output weights. The negative bias there filters small positive interference through ReLU. Anti-correlated features prefer antipodal placement; correlated features prefer orthogonal placement. Our pair is a different object: same input direction, near-parallel output weights, offset thresholds. The superposition literature supports the role of negative bias as a threshold, but it does not predict or describe our sign-crossing pair.

### Cited Findings
- **Elh22 [FT]**: Nelson Elhage, Tristan Hume, Catherine Olsson, …, Chris Olah (2022), "Toy Models of Superposition", Transformer Circuits Thread (14 Sep 2022). Scite/Undermind DOI 10.48550/arXiv.2209.10652; the paper text does not list an arXiv ID.
  - Antipodal pairs ("digons"): W_i = −W_j share one dimension. They appear as features become sparse. For uniform importance there is a "sticky" 1/2-dimension-per-feature plateau. In the n=2, m=1 phase diagram the choice (ignore / dedicated / antipodal) depends jointly on sparsity and relative importance.
  - Negative bias (Experiment Setup; Mathematical Understanding), for the model x′ = ReLU(WᵀWx + b): the bias lets the model "discard small amounts of noise". In the authors' words, "ReLU makes negative interference free in the 1-sparse case" and "using a negative bias can convert small positive interferences into essentially being negative interferences".
  - Correlated vs anti-correlated features: correlated features are ideally orthogonal; failing that, close together; failing that, they collapse to their principal component. Anti-correlated features "prefer to have them interfere, especially with negative interference", "ideally being antipodal".
  - Computation in superposition: abs(x) = ReLU(x) + ReLU(−x), using two neurons with opposite input weights.
  - Training (Superposition and Learning Dynamics): discrete "energy level" jumps line up with loss drops; geometries transform in stages (e.g., into an antiprism/octahedron with antipodal pairs).
  - The antipodal structure is on the embedding (W column) side, not on MLP output weights. — [Transformer Circuits](https://transformer-circuits.pub/2022/toy_model/index.html); [arXiv DOI](https://doi.org/10.48550/arXiv.2209.10652)
- **Sch22h [FT]**: Adam Scherlis, Kshitij Sachan, Adam S. Jermyn, Joe Benton, Buck Shlegeris (2022), "Polysemanticity and Capacity in Neural Networks", arXiv:2210.01892.
  - Capacity: C_i = (W_i·W_i)² / Σ_j (W_i·W_j)², with Σ C_i ≤ D. In the quadratic model, features are ignored, partially represented or fully represented depending on importance thresholds and kurtosis (k>3). Efficient embeddings are block-semi-orthogonal.
  - Anthropic's toy model's c≈1/2 regions are identified with the "antipode polytope" (Fig. 7; ReLU and GeLU are also tested).
  - The optimal bias b = −Σ D_ii; its sign depends on the parameters, and no general claim about negative bias is made. — [arXiv](https://arxiv.org/abs/2210.01892)
- **Jer22 [FT]**: Adam S. Jermyn, Nicholas Schiefer, Evan Hubinger (2022), "Engineering Monosemanticity in Toy Models", arXiv:2211.09169.
  - Monosemantic neurons have moderate negative biases; polysemantic neurons have small positive biases (§4.1.1, Fig. 5).
  - Initialising biases at a mean of −1 (with bias decay 0.03, switched off for the last 50% of training) gives about one monosemantic neuron per feature, plus exactly 65 positive-bias polysemantic neurons (d = 64).
  - Proposed mechanism: a negative bias "denoise[s] interference from unrelated features" by suppressing low-amplitude apparent features.
  - No antipodal neuron pairs are reported. — [arXiv](https://arxiv.org/abs/2211.09169)
- **Pri26b [FT]**: Lucas Prieto, Edward Stevinson, Melih Barsbey, Tolga Birdal, Pedro A. M. Mediano (2026), "From Data Statistics to Feature Geometry: How Correlations Shape Superposition", ICLR 2026, arXiv:2603.09972.
  - With correlated features, interference can be constructive; ReLU plus negative bias suppress false positives.
  - Antipodal pairs appear in a ReLU autoencoder for anti-correlated cyclic data at larger latent sizes (m = 6–10), with the negative interference zeroed by ReLU.
  - Weight decay makes correlation-shaped arrangements more common. — [arXiv](https://arxiv.org/abs/2603.09972)
- **Lec23 [FT]**: Victor Lecomte, Kushal Thaman, Rylan Schaeffer, Naomi Bashkansky, Trevor Chow, Sanmi Koyejo (2023), "What Causes Polysemanticity? …Incidental Causes", arXiv:2312.03096.
  - Polysemanticity can arise from random initialisation collisions reinforced by L1 or non-Gaussian noise (winner-take-all).
  - "Benign" collisions have opposite-signed weights on the shared hidden coordinate.
  - The model has no biases. — [arXiv](https://arxiv.org/abs/2312.03096)
- **Shi26o [Scite-excerpt]**: Dai Shi, Xiaoyu Li, Andi Han, et al. (2026), "Feature Superposition in Neural Networks: From Theory to Practice" (survey), arXiv:2609.06862. On the trade-off: "A negative bias can also remove small positive cross-talk, although the same bias shrinks active signals." — [arXiv](https://arxiv.org/abs/2609.06862)
- **Sam26b** (full text in prior notes; GELU excerpt via Scite): toy "mirror" vs "shadow" (negative-bias) suppression of one feature of an antipodal pair. The GELU variant does not use a strongly negative bias (KQ2). — [arXiv](https://arxiv.org/abs/2609.27593)

### Inferences
- Map from toy models to our case:
  - Elh22's negative bias corresponds to our negative effective bias. In both, the neuron is "off" unless its input is clearly present.
  - With GELU, "off" means a small negative output (about −0.10 to −0.15 × W2), not zero. So with GELU the filtering is leaky.
  - Elh22's antipodal pairs do not correspond to our pairs. Ours are not opposite embedding directions and not opposite output weights.
  - The closest toy construct is Elh22's ReLU(x) + ReLU(−x) pair. Its input weights are opposite; ours are the same.
- Our pair is two thresholds on the same input direction writing in the same output direction. Read this way, it is a two-knot basis expansion of a single ridge function g(w·x)·Ŵ2:
  - Below both thresholds, both outputs are small and negative.
  - Between the thresholds, one is positive and one negative: this is our 6% of tokens.
  - Above both thresholds, both are positive and add up.
  - The "opposition" is just the window between the two thresholds. This is an interpretation, not a published result.
- A useful algebraic identity for reviewers (computed here): GELU(z) − GELU(−z) = z. A pair with opposite input weights and opposite output weights would therefore implement a linear map, the GELU analogue of ReLU(x) − ReLU(−x) = x. Our pairs have neither opposite input weights nor opposite output weights, so this "linear pass-through" reading does not apply.

### Gaps
- Bricken et al. 2023 "Towards Monosemanticity" was not found as a record in Undermind or Scite; only citing mentions appear. Cunningham et al. 2023 (arXiv:2309.08600) was found but not read. Neither is used for factual claims here.
- No toy-model paper studies GELU-specific superposition where the bias filters interference through a negative non-zero tail. The closest is Sam26b's GELU remark.

---

## KQ4. Residual-stream geometry: neurons or heads writing opposite directions (erasure, suppression, antipodal neurons), and near-duplicate or redundant neurons

### Takeaway
Opposite writing is well documented, but it comes from anti-parallel weights or from distinct components:
- erasure heads (projection ratio −0.907);
- token-detector neurons that suppress their own trigger tokens (>80%);
- an antipodal entropy/anti-entropy neuron pair (output cosine −0.886);
- depletion/weakening neurons, where cos(w_in, w_out) < 0.

Near-duplicate neurons are explained by condensation, embedding and splitting, and redundancy. None of these papers reports pairs with near-parallel output weights whose opposition comes from activation sign.

### Cited Findings
- **Gur24 [FT]** (TMLR 2024, arXiv:2401.12181):
  - Prediction, suppression and partition neurons are defined by the moments of W_U·w_out (kurtosis, skew). Prediction neurons increase after mid-depth; suppression neurons appear in the last layers (GPT-2 medium mean prediction-neuron counts in layers 20–23: 130.8, 142.2, 207.2, 339.2).
  - §5.2 gives an antipodal pair: entropy neuron L23.945 and anti-entropy neuron L22.2882, whose output weights have cosine −0.886.
  - §6.1: high-frequency neurons (mostly in the first quarter of layers) have w_in and w_out nearly opposite (cosine up to about ±0.8).
  - Near-duplicate universal neurons exist (e.g., 105 BOS neurons), but "few literal duplicates" are found by weight similarity (Fig. 10). An ensemble hypothesis is proposed. — [arXiv](https://arxiv.org/abs/2401.12181)
- **Voi23 [FT]**: token-detector neurons are triggered by 1–5 tokens; OPT-66B has 5,351 of them. For more than 80% of them, the vocabulary projection of the update points against the triggering tokens, which the authors read as explicit removal of current-token information. No neuron-pair or output-geometry analysis. — [arXiv](https://arxiv.org/abs/2309.04827)
- **Dao23 [FT]**: Jett Janiak, Can Rager, James Dao, Yeu-Tong Lau (2023/2024), "An Adversarial Example for Direct Logit Attribution: Memory Management in GELU-4L", BlackboxNLP 2024, DOI 10.18653/v1/2024.blackboxnlp-1.15, arXiv:2310.07325.
  - The erasers are attention heads L2H2–L2H7 (not MLP neurons), which write against head L0H2. Their aggregate median projection ratio is −0.907.
  - Patching the V-composition path leaves 0.91 vs 0.12 of the projection, so about 85% of the erasure depends on that path.
  - The DLA correlation between writer and erasers is r = −0.702 (slope −0.613). — [DOI](https://doi.org/10.18653/v1/2024.blackboxnlp-1.15)
- **Ger25b / Ger26c [FT]**: depletion and weakening neurons write against what they read (cos(w_in, w_out) < −0.5), mostly in late layers. This is a single-neuron property, not a pair property. — [arXiv 2505.17936](https://arxiv.org/abs/2505.17936); [arXiv 2609.18612](https://arxiv.org/abs/2609.18612)
- **Zho21h [FT]**: Hanxu Zhou, Qixuan Zhou, Tao Luo, Yaoyu Zhang, Zhi-Qin John Xu (2022), "Towards Understanding the Condensation of Neural Networks at Initial Training", NeurIPS 2022, arXiv:2105.11686 (Undermind DOI 10.52202/068431-0159).
  - Input weights (bias included) condense onto a few orientations. For activations of multiplicity p (GELU and Swish have p = 1), there are at most 2p oriented directions.
  - Condensed neurons act as one effective neuron whose output weight is the sum of theirs.
  - This holds for small initialisation and the initial phase only (until loss reaches 70% of its starting value).
  - No rule on output-weight signs. — [arXiv](https://arxiv.org/abs/2105.11686)
- **Zha21ar [FT]**: Yaoyu Zhang, Zhongwang Zhang, Tao Luo, Zhi-Qin John Xu (2021), "Embedding Principle of Loss Landscape of Deep Neural Networks", NeurIPS 2021, arXiv:2105.14573.
  - Splitting a neuron means copying its input weights and bias and dividing its output weight a into αa and (1−α)a, for any real α. This preserves the function and gives critical points.
  - Embedding K extra neurons gives a K-dimensional affine set of critical points.
  - The construction covers identical biases only; same input weights with different biases are not covered. — [arXiv](https://arxiv.org/abs/2105.14573)
- **Fuk19 [FT]**: Kenji Fukumizu, Shoichiro Yamaguchi, Yoh-ichi Mototake, Mirai Tanaka (2019), "Semi-flat minima and saddle points by embedding neural networks to overparameterization", arXiv:1906.04868.
  - Unit replication: v_j = λ_j ζ with Σλ_j = 1.
  - For scalar output, with G positive definite: all λ_j > 0 (same-sign split) gives a local minimum, and any λ_j < 0 gives a saddle.
  - With G negative definite the result reverses.
  - At zero training error the replicas form semi-flat minima. — [arXiv](https://arxiv.org/abs/1906.04868)
- **Mar25l** (full text in prior notes; verified again today [FT]):
  - "Channels" are defined by output weights of opposite sign (a_i·a_j < 0) that diverge while the input vectors (biases included) converge.
  - Exact same-sign duplicates are a different structure: equal-loss lines of critical points or plateau-saddles. Adding biases "sharply reduces their prevalence" (App. A.1).
  - Regularisation can stop travel along a channel.
  - Activations used: softplus, erf, sigmoid and ReLU; GELU is not mentioned. — [arXiv](https://arxiv.org/abs/2506.14951)
- **Cas19e [FT]**: Stephen Casper, Xavier Boix, Vanessa D'Amario, Ling Guo, Martin Schrimpf, Kasper Vinken, Gabriel Kreiman (2021), "Frivolous Units: Wider Networks Are Not Really That Wide", AAAI 2021, 35(8): 6921–6929, DOI 10.1609/aaai.v35i8.16853 (verified via Scite), arXiv:1912.04783.
  - Prunable units vs redundant units (activations are a linear combination of others; PCA at 0.05 unexplained variance).
  - Redundancy grows faster than width (gain g ≈ 1.8–2.6). Weights are not compared.
  - The DOI the PDF reader extracted (…v35i11.17280) was wrong; the earlier notes have the correct one. — [DOI](https://doi.org/10.1609/aaai.v35i8.16853)
- **Doi21 [FT]**: Diego Doimo, Aldo Glielmo, Sebastian Goldt, Alessandro Laio, "Redundant representations help generalization in wide neural networks", J. Stat. Mech. (2023) 114011, DOI 10.1088/1742-5468/aceb4f; NeurIPS 2022 version.
  - Last-layer "clones" are random chunks that reproduce the full representation (R² ≈ 1, residuals close to uncorrelated noise). Critical chunk size is about 50 / 100 / 200 neurons for pMNIST / CIFAR-10 / CIFAR-100.
  - Clones appear only with weight decay and after zero training error (around epochs 160–180 in WRN28-8 on CIFAR-10).
  - Weight similarity is not tested. — [DOI](https://doi.org/10.1088/1742-5468/aceb4f)
- **Nan25b [FT]**: Advey Nandan, Cheng-Ting Chou, Amrit Kurakula, Cole Blondin, Kevin Zhu, Vasu Sharma, Sean O'Brien (2025), "Universal Neurons in GPT-2: Emergence, Persistence, and Functional Impact", NeurIPS 2025 MechInterp Workshop, arXiv:2508.00903.
  - Fewer than 5% of neurons are universal at 80k steps, and the share grows during training. Persistence between adjacent checkpoints is about 80% or more in layers 0, 1, 10 and 11.
  - Ablating universal neurons is about as harmful as ablating 5× as many random neurons, at the 0.5 threshold.
  - Biases are not tracked. — [arXiv](https://arxiv.org/abs/2508.00903)

### Inferences
- Our M2 finding (opposite pairs never come from anti-parallel output weights) is a substantive contrast with the main published examples of opposition: Gur24's antipodal neurons (cosine −0.886), Mar25l's opposite-sign channels, and Ger26c's weakening neurons. In our ViT the opposition comes from activation sign. This difference should be stated explicitly; it is likely new at the pair level.
- Our pair (input cosine 0.96, output angle 7°, biases −1.5 vs −1.1) sits between two things:
  - An exact split neuron (Zha21ar/Fuk19), which would have identical biases.
  - A genuine two-threshold unit, whose different biases give a richer 1-D function.
  - Fuk19's result that a same-sign split is a semi-flat minimum, not a saddle, under positive-definite G is consistent with such pairs persisting for some time. It does not by itself explain why they decline by epoch 200. Weight decay pulling redundant copies together or toward zero is one candidate (our inference; compare Doi21 on the role of weight decay).
- Condensation theory (Zho21h) covers only the small-initialisation, initial-training regime. Our pairs form at epochs 10–40. Cite it as related, not as the mechanism.

### Gaps
- No paper reports pairs of MLP neurons with near-parallel output weights and offset thresholds, in any transformer.
- Copy-suppression heads (McDougall et al., Mcd24: BlackboxNLP 2024, DOI 10.18653/v1/2024.blackboxnlp-1.22) were found by the deep search but not read. They are listed here only as a candidate citation.
- Xu25ad (condensation overview, arXiv:2504.09484) was not read.

---

## KQ5. Is our interpretation consistent, and what alternative explanations will reviewers raise?

### Takeaway
Each component of the mechanism has independent published support: negative drift of pre-activations and negative resting operating points; GELU's negative outputs; the sign rule for the direction of a contribution; near-duplicate and condensed neurons. The specific conjunction we report has no precedent in the papers retrieved: near-parallel output weights plus offset negative thresholds produce opposite per-token writes on a small token fraction, with a small share of MLP output. The main risks are that the negative tail can be functionally important (Kon25e), that ablation results depend on the ablation baseline, and that "opposite angle" is almost automatic once output weights are parallel.

### Cited Findings
Per-claim verdicts. Evidence is from the papers cited in KQ1–KQ4.

- **M1** (most opposed pair in block 6: near-identical inputs, parallel outputs, 91% negative regime, opposition on 6% of tokens, κ ≈ 0.19 when both are active, 0.3–0.6% of MLP output)
  - **Verdict: consistent with the literature in every component; the configuration as a whole appears unreported.** Support:
  - Negative regime on most tokens: GELU negative pre-activations of 70.5% in MaxViT-Tiny and 89.3% in GPT-nano. — [Shv26](https://arxiv.org/abs/2605.17659)
  - Pre-activation mean turns negative in training, even with GELU. — [Li22ap](https://arxiv.org/abs/2210.06313)
  - Resting pre-activation of about −1.1 to −1.6 in Phi-2. — [Tuo26](https://arxiv.org/abs/2609.31956)
  - Negative input bias in universal neurons. — [Gur24](https://arxiv.org/abs/2401.12181)
  - Sign rule (negative activation writes −w_out). — [Ger25b](https://arxiv.org/abs/2505.17936); [Ger26c](https://arxiv.org/abs/2609.18612)
  - GELU outputs are negative for x<0. — [Hen16](https://arxiv.org/abs/1606.08415)
  - Same-input neurons behave as one effective neuron with summed output weights. — [Zho21h](https://arxiv.org/abs/2105.11686); [Zha21ar](https://arxiv.org/abs/2105.14573)
- **M2** (across blocks, opposite pairs at ≥150° never come from anti-parallel output weights; counts rise at epochs 10–40, peak at 80–110 and decline by 200)
  - **Verdict: the sign-not-weights finding contrasts with the published examples, which are anti-parallel. The time course is partly anticipated: negative drift and sparsity appear early, and Tuo26 shows a peak followed by partial relaxation. Pair-count dynamics have not been reported.** Contrasting examples:
  - Antipodal output weights, cosine −0.886. — [Gur24](https://arxiv.org/abs/2401.12181)
  - Opposite-sign channels. — [Mar25l](https://arxiv.org/abs/2506.14951)
  - Antipodal pairs, which are in embedding space. — [Elh22](https://transformer-circuits.pub/2022/toy_model/index.html)
  - Time-course precedents:
  - Peak of the operating-point build at step 8000, then partial relaxation. — [Tuo26](https://arxiv.org/abs/2609.31956)
  - Sparsity appears quickly early in training. — [Li22ap](https://arxiv.org/abs/2210.06313)
  - Late worsening of ViT sparsity on ImageNet. — [Pen26i](https://arxiv.org/abs/2605.25612)
- **M3** (removing the pair barely changes detection)
  - **Verdict: consistent with redundancy and prunability results, but contested by targeted negative-region ablations; the ablation baseline must be controlled.** Supporting:
  - Redundant and prunable units proliferate with width. — [Cas19e](https://doi.org/10.1609/aaai.v35i8.16853)
  - Masking inactive units does not change accuracy. — [Wil24b](https://arxiv.org/abs/2407.07848)
  - Clones carry the same information. — [Doi21](https://doi.org/10.1088/1742-5468/aceb4f)
  - Contrasting:
  - Clamping negative pre-activations in 1% of neurons doubles perplexity. — [Kon25e](https://arxiv.org/abs/2509.24198)
  - Weakening neurons have outsize influence. — [Ger26c](https://arxiv.org/abs/2609.18612)
  - Universal-neuron ablation is as harmful as 5× as many random neurons. — [Nan25b](https://arxiv.org/abs/2508.00903)
  - Baseline dependence: zero vs mean ablation, counterweights. — Ahm26c (prior notes); [Sam26b](https://arxiv.org/abs/2609.27593)

Bibliography, with what each paper measured and how it bears on our claims:

| Key | Citation | Measured quantity (exact) | Relevance |
|---|---|---|---|
| Li22ap | Li et al., ICLR 2023, arXiv:2210.06313 | % nonzero after ReLU; ViT-B/16 6.3%, T5-Base 3.0%; early emergence; middle layers sparsest; BERT pre-activation mean turns negative | M1/M2 background; correct the depth claim |
| Shv26 | Shvetsov et al. 2026, arXiv:2605.17659 | negative pre-activation fraction: GELU 63.6% (MLP), 70.5% (MaxViT-T), 89.3% (GPT-nano) | M1 (91% is typical) |
| Tuo26 | Tuomi 2026, arXiv:2609.31956 | resting pre-activation (median) −1.115 to −1.568 in Phi-2; duty 0.51→0.16 in Pythia-410M; peak at step 8000 | M1 (effective-bias magnitude), M2 (time course), definition of effective bias |
| Kon25e | Kong et al., ICLR 2026, arXiv:2509.24198 | clamping negatives in top-1% WD neurons per layer: perplexity about ×2 and TSE −29 pp (Llama); sign flip: perplexity 2,277 | Main reviewer risk for M3 |
| Ger25b / Ger26c | Gerstner et al. 2025 / 2026 | cos(w_in, w_out) classes; negative activation writes −w_out; weakening neurons concentrated in late layers | Sign rule (M1) |
| Gur24 | Gurnee et al., TMLR 2024 | negative input bias of universal neurons; antipodal pair with cosine −0.886; few literal duplicates | M1 (bias), M2 contrast |
| Elh22 | Elhage et al. 2022, TC Thread | antipodal pairs in embedding; negative bias filters interference; anti-correlated features prefer antipodal placement | Conceptual contrast (KQ3) |
| Jer22 | Jermyn et al. 2022 | negative bias ↔ monosemantic neurons; −1 bias initialisation gives about 1 neuron per feature | Interpretation of negative bias |
| Sch22h | Scherlis et al. 2022 | capacity C_i; antipode polytope at c≈1/2 | Background |
| Pri26b | Prieto et al., ICLR 2026 | correlations make interference constructive; ReLU plus negative bias suppress false positives; weight decay favours correlation-shaped geometry | Background, weight-decay angle |
| Dao23 | Janiak et al., BlackboxNLP 2024 | erasure heads: median projection ratio −0.907 | Purposeful erasure (alternative explanation) |
| Voi23 | Voita et al. 2023 | >80% of token detectors suppress their own triggers; dead neurons >70% in some layers (ReLU) | Purposeful suppression (alternative explanation) |
| Zho21h / Zha21ar / Fuk19 | NeurIPS 2022 / NeurIPS 2021 / arXiv 2019 | condensation; neuron splitting gives critical-point sets; same-sign split can be a semi-flat minimum | Origin of near-duplicate pairs |
| Mar25l | "Flat Channels to Infinity in Neural Loss Landscapes" (2025), arXiv:2506.14951; author list not retrieved in this session | channels: a_i·a_j<0, converging inputs, GLU limit; same-sign duplicates are plateau-saddles, rarer with biases | Reviewers will ask whether our pairs are channels; they are not |
| Cas19e / Doi21 | AAAI 2021 / J. Stat. Mech. 2023 | redundancy gain g ≈ 1.8–2.6; clones need weight decay and zero training error | M3 support |
| Luo24d / Mir23 / Sza25b | ICML 2025 / ICLR 2024 / ICLR 2026 | SiLU's meaningful negative outputs; GELU sparsity 1% (Falcon); top-p sparsity | GELU-sparsity measurement caveats |
| Pen23b / Pen26i | arXiv 2023 / 2026 | flat-minima explanation; ViT-B/16 nonzero 0.104/0.087; late worsening on ImageNet | M2 time course |
| Nan25b | NeurIPS 2025 workshop | universal neurons emerge during training (<5% at 80k steps) | M3 contrast |

### Inferences

**How to phrase this in the article (suggested wording):**
- "The per-token contribution of hidden neuron c is GELU(z_c)·W2[:,c]. Its direction is therefore fixed by W2[:,c] up to the sign of GELU(z_c). This sign rule has been noted for gated MLPs (Gerstner & Schütze 2025; Gerstner et al. 2026), and GELU produces negative outputs for z<0 (Hendrycks & Gimpel 2016)."
- "Consistent with reports that transformer MLP pre-activations drift negative during training (Li et al. 2023, App. B; Shvetsov et al. 2026), and that trained GELU gates rest at a negative operating point of about −1.1 to −1.6 (Tuomi 2026), neurons in our block 6 are in GELU's negative regime on 91% of tokens."
- "Unlike the antipodal configurations described in superposition toy models (Elhage et al. 2022) and the antipodal output weights reported in GPT-2 (Gurnee et al. 2024), our most opposed pairs have nearly parallel output weights (7°). The 173° angle between their contributions arises only on the ~6% of tokens where one neuron's pre-activation has crossed zero and the other's has not."
- "We therefore interpret the opposition as a by-product of two neurons reading the same direction with offset thresholds. On these tokens the pair approximates a single ridge function, as with condensed or split neurons (Zhou et al. 2022; Zhang et al. 2021). We do not interpret it as a cancellation circuit of the erasure or suppression type (Janiak et al. 2024; Voita et al. 2023)."
- Do not claim the negative tail is functionally irrelevant. Write instead: "the opposition contributes 0.3–0.6% of the block's MLP output and its removal barely changes detection". Cite Kong et al. 2026 as evidence that negative pre-activations can matter in other models.

**Reviewer risks and alternative explanations, with suggested controls:**
1. *The angle is trivial.* With parallel W2 columns, the angle between the two contributions can only be about 7° or about 173°. A reviewer will say the "opposition" just restates sign(GELU(z1)) ≠ sign(GELU(z2)).
   - Answer: we report it as a decomposition (weight angle vs activation sign), which is the point of M2.
   - Add a null model: for random pairs with matched output-weight angle and matched pre-activation distributions, the expected rate of opposite-sign tokens.
2. *Definition of "effective bias".* Tuo26 shows the negative resting point is carried mostly by w·E[LN(x)], not by b (48–56× larger).
   - State whether our "effective bias" means b, or b + w·E[x_ln], or the mean pre-activation.
   - Report both terms.
3. *The negative tail may be functional* (Kon25e; Ger26c's negative-gate quadrants).
   - Test by clamping z<0 to 0 for the pair, for all neurons in block 6, and for matched random neurons. Report IoU and score separately.
   - Also test the sign flip, to separate "the constant offset matters" from "the sign carries information".
4. *Ablation baseline.* On 91% of tokens both neurons output roughly −0.10 to −0.15 × ‖W2‖ (computed), which is nearly constant. Zero-ablation therefore injects a constant residual shift; mean-ablation does not.
   - Report zero, mean and "dose" variants (Ahm26c, prior notes).
5. *Split neuron / embedding redundancy* (Zha21ar, Fuk19, Zho21h).
   - Merging test: replace the pair with one neuron using the averaged input weight, the summed output weight and either bias. If detection is unchanged, the pair is redundant. If a two-threshold fit is needed, the bias offset matters.
6. *Mar25l channels.* A reviewer may assume the pair is a channel. It cannot be: channels need a_i·a_j < 0 and diverging output norms, while our outputs are parallel.
   - Report the output-norm trajectory and the sign of W2·W2 for each pair.
7. *Weight decay (0.05) and AdamW.* Doi21 clones need weight decay; Mar25l says regularisation halts channels; Pri26b says weight decay changes the geometry. The decline of pairs after epoch 110 could be a weight-decay effect.
   - Control: a run with no or low weight decay, plus at least a second seed (one seed is a standard reviewer objection).
8. *Purposeful cancellation* (erasure, suppression, hydra).
   - Show that the opposite-sign tokens are not selectively those where the pair's sum would harm the loss, e.g. by correlating sign-crossing tokens with gradient alignment or with detection error.
   - κ ≈ 0.19 only on co-active tokens is weak evidence on its own.
9. *Depth claims.* Do not state that deeper layers are sparser in general (Li22ap: middle layers sparsest, last layers denser). Report our per-block negative-regime fractions instead.
10. *GELU non-monotonicity.* For z < −0.752 the output becomes more negative as z rises (GELU′(−1.5) ≈ −0.13, computed). A reviewer might argue that neurons resting at z ≈ −1.1 to −1.5 carry a graded, inverted signal in the tail, which is Kon25e's "negative pre-activations differentiate" claim. Check whether the variance of tail outputs correlates with the targets.

### Gaps
- No retrieved paper tracks pairs of MLP neurons (in any model) across training checkpoints, or measures κ-type per-token cancellation between two neurons. The Oct-14 deep search and today's two searches agree on this.
- Not found: a ViT or detection study of the GELU negative tail; a measurement of GELU's negative-output magnitude during training; a pair-level analysis with parallel output weights.
- Abstract-level or excerpt-only knowledge, not used for quantitative claims: Shi26o (survey, Scite excerpts only), Mcd24 (copy suppression; not read), Cun23 (SAE; not read), Xu25ad (not read), and Bricken et al. 2023 (not retrieved).
