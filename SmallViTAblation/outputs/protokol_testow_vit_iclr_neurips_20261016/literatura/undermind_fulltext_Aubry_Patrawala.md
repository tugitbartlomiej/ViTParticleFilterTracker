# Notatki czytania pełnych tekstów przez Undermind

Status: ekstrakcja narzędziowa; interpretację i adaptację do SmallViT ocenia protokół główny. Nie są to wyniki naszych eksperymentów.

[Aub24]:
## 1. Exact coupling metric

The paper measures coupling between **residual-branch Jacobians**, not between the full residual-block Jacobians.

For layer \(l\), input token \(t_1\), and output token \(t_2\), it defines
\[
J^l_{t_1t_2}
=\frac{\partial f^l(X^{l-1})_{t_2}}{\partial x^{l-1}_{t_1}},
\tag{8}
\]
where \(f^l\) is the block’s transformation **without the skip connection**. The paper takes a truncated SVD, \(J=USV^\top\), and compares two Jacobians using the top-\(K\) singular-vector bases. In its notation,
\[
A^{ll'}_{t_1t_2t'_1t'_2,K}
=U_{l',K}^{\top}J^l_{t_1t_2}V_{l',K}.
\]
The mis-coupling and coupling scores are
\[
m_K(J^l_{t_1t_2},J^{l'}_{t'_1t'_2})
=\frac{\left\|A^{ll'}_{t_1t_2t'_1t'_2,K}-S_{l,K}\right\|_F}
{\|s_{l,K}\|_p},
\qquad
c_K=1-m_K.
\tag{9}
\]
Here \(S_{l,K}\) contains the top \(K\) singular values of the first Jacobian, and \(s_{l,K}\) is their vector. The usual choice is \(p=1\), i.e. normalization by the sum of those singular values. Thus this is **not simply a cosine similarity**: it scores whether the other Jacobian’s singular-vector bases make the first Jacobian approximately diagonal with its corresponding singular values. The paper also gives a variant \(B=V_{l',K}^{\top}J^lU_{l',K}\) for examining alignment involving left and right singular-vector bases (§3.1).

For its LLM benchmark plot, the paper specifies \(K=\frac{1}{10}d_{\text{model}}\) (Fig. 1); it also shows robustness to using 1%, 3%, and 5% of the hidden dimension (Fig. 8). The main text does not clearly specify the exact \(K\) used for the ViT plot, so that is a replication detail to verify rather than assume.

**Figure 2** diagrams the Jacobian construction and the depth-wise, self-token, and context-token comparisons; it is the paper’s compact visual guide to what “coupling” means here.  
[REGION: page=4, bbox=86,154,457,847]

## 2. Token-diagonal and cross-token Jacobians; the residual skip

The subscripts \(t_1,t_2\) refer to **tokens**, not feature coordinates. Causality makes \(J^l_{t_1t_2}=0\) when \(t_1>t_2\), so the paper considers \(t_1\leq t_2\) (§3.1).

- **Token-diagonal/self Jacobians:** \(J^l_{tt}\) measure a token’s effect on its own output representation. Depth-wise coupling fixes \(t\) and compares \(J^l_{tt}\) across layers. Self-coupling compares these self-Jacobians across token positions.
- **Cross-token/context Jacobians:** \(J^l_{t_1t_2}\) for \(t_1<t_2\) measure the effect of an earlier context token on a later output token. The paper examines context coupling with the input token fixed while varying the output token, or with the output fixed while varying the input token (§3.1; Figs. 15–16).

A visible diagonal in the paper’s coupling heatmaps is a diagonal in the **comparison matrix \(A\)**—evidence of alignment between singular-vector bases. It should not be confused with the token-diagonal blocks \(J_{tt}\) or with the ordinary diagonal of a feature-space Jacobian.

For a residual block \(X\mapsto X+f(X)\), the full block Jacobian is \(I+J\). The measured coupling score is based on \(J\), the residual branch **excluding** the skip. The skip reappears in the paper’s dynamical interpretation: \(x^{l+1}=(I+J_l)x^l\) (§6.1; also Appendix A.5, Eqs. 14–19). The paper does **not** report the same coupling analysis computed on the full \(I+J\) matrices. That distinction matters if a replication is intended to claim coupling of complete residual-block Jacobians.

## 3. What the ViT and training experiments do

### Vision Transformer experiment

The core ViT study trains three DEiT-style ViTs on CIFAR-10. The architecture is held fixed at embedding dimension 192, depth 24, and 3 attention heads; stochastic-depth rates are \(0.025, 0.05, 0.075\). Appendix A.7 reports weight decay \(0.05\), Adam with learning rate \(5\times10^{-3}\), cosine scheduling, 600 epochs, 10 warm-up epochs, mixup with \(\alpha=0.8\), and batch size 128. Figure 6 reports:

- coupling increases over checkpoints 50 through 550;
- test accuracy and coupling, evaluated at checkpoints, have a positive association, reported as \(R^2=0.93\);
- importantly, coupling is measured on **training data**, while the reported accuracy is **test accuracy** (§5.4; Fig. 6).

This is evidence that coupling tracks training and is associated with test accuracy in that setup—not evidence that increasing coupling itself causes better test performance.

There is also an auxiliary stochastic-depth/weight-decay plot (Fig. 7), with coupling plotted against stochastic depth for eight weight-decay values. Appendix A.7, however, describes the principal three-ViT experiment with weight decay fixed at \(0.05\). The paper does not fully explain how the wider Fig. 7 sweep relates to the core setup; a replication should record which setting it is matching.

**Figure 6** shows the ViT coupling–test-accuracy scatter and coupling trajectories across training checkpoints; it is the central graphic for the paper’s ViT claims.  
[REGION: page=8, bbox=622,155,938,847]

**Figure 7** shows coupling versus stochastic-depth rate for multiple weight-decay values. It is relevant if reproducing or extending the hyperparameter sweep, but its relationship to Appendix A.7’s fixed-weight-decay core experiment is not fully documented.  
[REGION: page=22, bbox=90,155,406,845]

### LLM training and benchmark comparisons

The paper also tracks Pythia 6.9B and 12B across checkpoints: coupling is low initially and rises during training, with increasingly strong local/adjacent-layer interactions in the depth-wise visualizations (Fig. 1; §5.3). This is a longitudinal observation of training, not a controlled intervention on coupling.

Across LLMs, the authors report a positive correlation between average depth-wise coupling and Open LLM benchmark score, with \(R^2=0.8\) and \(p=9.99\times10^{-10}\) in Fig. 1 (§5.5). They compare that relationship with score associations for parameter count, number of layers, and embedding dimension (Fig. 9), and state coupling is more strongly correlated. Figure 9’s plotted \(R^2\) values are approximately 0.20 for layer count, 0.63 for embedding dimension, and 0.58 for parameter count. This is a comparison of cross-model associations, not a controlled causal analysis or a demonstration that coupling predicts performance independently of all confounders.

**Figure 1** combines the LLM coupling–benchmark-score association with Pythia checkpoint trajectories and layer-adjacency visualizations. The benchmark plot reports \(R^2=0.8\); the checkpoint plots show coupling increasing with training.  
[REGION: page=2, bbox=92,155,645,846]

**Figure 9** compares benchmark score against model depth, embedding dimension, and parameter count. It supports the paper’s stated comparison that coupling has a stronger score association than these plotted hyperparameters, but does not itself establish causality or remove confounding.  
[REGION: page=23, bbox=384,155,611,850]

## 4. What is causal, and what is correlational?

**What the paper directly demonstrates:**

- Trained versus randomly initialized models show different coupling patterns; the authors observe coupling in trained models and little/no corresponding coupling at initialization (§§5.1–5.3; Figs. 3–4, 13–16).
- In the tracked Pythia checkpoints, coupling increases as training proceeds (§5.3; Fig. 1b).
- In the ViTs, coupling increases across training checkpoints, and coupling is positively associated with test accuracy across the observed checkpoints/configurations (§5.4; Fig. 6).
- The paper reports cross-model coupling–benchmark correlations and compares them with correlations for several model hyperparameters (§5.5; Figs. 1, 8–9).

**What it does not establish causally:**

- It does not directly manipulate coupling while holding architecture, data, optimization, and other relevant factors fixed.
- The ViT variations manipulate **stochastic-depth rate** (and, in the auxiliary plot, weight decay), not coupling itself. Their effects cannot isolate coupling as the cause of any accuracy change.
- A checkpoint trend shows coupling emerging during training, but training changes many properties simultaneously; it does not identify coupling as the mechanism causing improved generalization.
- The cross-LLM association is observational and potentially confounded by model family, data, training recipe, and other differences. The comparison with depth, dimension, and parameter count is not a causal adjustment for these factors.
- The authors’ proposed link from coupling to generalization and their dynamical explanation are interpretations/hypotheses (§§5.5–6.1), not a coupling-specific intervention result.

## 5. What is already covered for your six-layer, 192-dimensional residual ViT detector?

**Covered or close to covered:** a 192-dimensional ViT, residual-branch Jacobians, coupling tracked through training, variation of stochastic depth, and the association of coupling with held-out accuracy are all within the paper’s general experimental scope. The exact CIFAR-10 setup is nevertheless a **24-layer classifier**, not a six-layer detector.

**Not covered by the paper:** a six-layer architecture; detection rather than classification; detection-specific generalization measures; an intervention that directly strengthens or weakens coupling; coupling computed on the full residual Jacobian \(I+J\); or a causal demonstration that coupling mediates the effect of training choices on held-out performance. The paper also does not provide a direct comparison of several alternative coupling metrics as baselines. Its primary comparisons are trained versus untrained models, different token/layer Jacobian pairings, and coupling versus selected model hyperparameters or performance measures.

For your intervention study, the key distinction to preserve is therefore: reproducing the paper’s measurement and showing that it tracks training or held-out detector performance would extend its **descriptive/correlational** evidence; a claim that coupling causes improved detection generalization requires a direct coupling-targeted intervention and controls not present in this paper.

**Relevant limitations for replication:** the ViT results use a single CIFAR-10 classification setting and only three stochastic-depth rates in the main description; the paper’s ViT caption does not make the \(K\) choice explicit; its coupling is measured on training data in the ViT experiment; and the coupling score is for \(J\), not \(I+J\). The reported evidence therefore does not settle how the metric behaves in a shallow detector, on detection data, or under a targeted coupling manipulation.

**Paper:** Murdock Aubry, Haoming Meng, Anton Sugolov, and Vardan Papyan, “Transformer Block Coupling and its Correlation with Generalization in LLMs,” published as a conference paper at ICLR 2025.

---

[Pat25d]:
## What the paper actually tests

**TLCM’s observational signature** is a negative cosine similarity between adjacent layers’ marginal contributions at the *same token and forward pass*. The paper defines \(c_{i,t}\) as the layer’s attention contribution plus its MLP contribution, and computes \(M[i,j]\) by averaging \(\operatorname{cos}(c_{i,t},c_{j,t})\) across tokens (Sec. 3; Sec. 4.1). For Llama 3.1 8B, adjacent-layer cosine similarities average about \(-0.2\); the authors operationally define a TLCM activation as cosine similarity \(\leq -0.1\), at a valley in a bimodal distribution. That threshold is a paper-specific convention, not a universal criterion.

[REGION: page=3, bbox=34,155,315,845]

Figure 1 plots average layer-contribution cosine similarity matrices. The adjacent-layer blue bands appear in Llama 3.1 8B and Qwen 2.5 0.5B, but not in Phi 4 or GPT-2 XL; it illustrates the cross-depth *observational* signature and its absence in some models.

[REGION: page=5, bbox=75,155,359,847]

Figure 2 shows the distribution of adjacent-layer cosine similarities, with the paper’s \(-0.1\) activation cutoff, and reports that average TLCM activation counts rise with token position. The distribution motivates the paper’s cutoff, while the position trend is observational evidence related to contextual dependence.

### Causal scaling intervention

In Sec. 5.1, for a token \(t\), let \(a_t=c_{i,t}\) and let \(b_t\) be layer \(i+1\)’s contribution. The authors intervene on the input to layer \(i+1\), replacing \(x_{i,t}=x_{i-1,t}+a_t\) with
\[
x_{i-1,t}+a_t+\alpha a_t=x_{i-1,t}+(1+\alpha)a_t,
\]
for \(\alpha\in\{-1,-0.9,\ldots,0.9,1\}\). They measure \(\operatorname{cos}(a_t,b'_t(\alpha a_t))\), where \(b'_t\) is the following layer’s contribution under that intervention. As \(\alpha\) increases, the following layer’s contribution becomes more opposed to \(a_t\), approximately linearly over the tested range. At \(\alpha=1\), the prior contribution is doubled in the residual stream; at \(\alpha=-1\), it is canceled from the next layer’s input, and the next layer can instead align with it (a compensatory effect). The authors caution that extreme interventions can create counterfactual states and that this compensation may reflect self-repair mechanisms beyond TLCM.

[REGION: page=7, bbox=73,155,389,847]

Figure 3b plots the intervention result for five sampled tokens at layer 8: cosine similarity between the preceding contribution and the following layer’s contribution shifts with \(\alpha\), and negative \(\alpha\) shows compensatory alignment. This is the paper’s direct test that changing the preceding contribution changes the next layer’s response.

### Jacobian subspaces

In Sec. 5.2, the authors locally linearize the following layer’s contribution as \(b'_t(\Delta)\approx b_t+J\Delta\), then use the symmetrized Jacobian \(\bar J=(J+J^\top)/2\). Negative eigenvalues indicate locally corrected directions; positive ones indicate reinforced directions; near-zero ones indicate directions with little response. They project \(a_t\) into the eigenbasis and examine eigenvalues for its 50 largest squared projection components, which together account for 15% or more of its variance. Across 50 TLCM Jacobian–contribution pairs and 50 non-TLCM pairs, TLCM pairs show modes near \(-1\) and \(+0.5\); non-TLCM pairs have a strong mode near zero. The authors also report that roughly 3,000 of the 4,096 full Jacobian eigenvalues are negative. The \(-1\) mode means a unit input perturbation in that direction receives a unit-sized opposing response locally—not, by itself, a semantic identification of a feature.

[REGION: page=9, bbox=38,154,325,847]

Figure 4 shows the full Jacobian eigenvalue histogram and the projection-weighted eigenvalue histograms for TLCM and non-TLCM cases. The contrast supports selective correction/reinforcement rather than uniform attenuation, while the non-TLCM comparison is an important control.

### Training emergence and what it does—and does not—establish

Sec. 4.2 examines checkpoints from OLMo 7B, OLMo 2 1B, and OLMo 2 7B, computing the layer-similarity matrix on fixed text at each checkpoint. The OLMo 7B pattern is absent at step 33,000 (reported as 5% of training), begins to appear around step 135,500 (21%, approximately 0.5T tokens), and strengthens in later checkpoints. The authors interpret this as evidence that TLCM is learned during pretraining rather than inherent at initialization or introduced only by post-training.

[REGION: page=29, bbox=71,155,302,851]

Figure A9 shows OLMo 7B layer-similarity matrices at four checkpoints, from step 33,000 through 651,581. Adjacent-layer negative structure emerges and strengthens across those checkpoints.

This checkpoint comparison is **not** a lagged-correlation test, and it is not the causal intervention. It shows that a same-forward-pass cross-depth pattern changes as the model trains. It does not show that a change in one layer at one checkpoint causes a later layer’s change at a subsequent checkpoint. Conversely, the scaling intervention tests immediate dependence between adjacent layers within a forward pass, not training-time causality.

## Adapting these tests to a six-block Vision Transformer

The following are adaptations, not experiments reported in the paper.

1. **Measure the observational signature first.** For each image and token/patch position, record each block’s actual residual contribution and compute adjacent-block cosines for the five transitions. Match the same image and position across the two contributions. Report distributions and blockwise summaries, not just a pooled mean. Treat \(-0.1\) only as a candidate descriptive cutoff from this paper; estimate the ViT’s own distribution rather than assuming the same threshold or that TLCM exists.

2. **Run the paired scaling intervention.** For block \(i\), hold the image, positional information, and incoming state fixed, and intervene at block \(i+1\)’s input by adding \(\alpha a_{i,t}\), using the paper’s \([-1,1]\) grid as a directly comparable starting protocol. Measure both the cosine used in the paper and the **paired change in the next block’s contribution**, \(b'(x+\alpha a)-b'(x)\), projected onto \(a\). The latter helps distinguish a changed response from a negative cosine that was already present in the baseline. In a ViT, perturbing one patch can affect other patches through attention; specify whether the intervention is on one token, the class token, or a patch set, and measure the response at the intervened position as well as any intended downstream positions.

3. **Adapt the Jacobian analysis carefully.** Compute the derivative of the next block’s contribution with respect to its input using autodiff, symmetrize it, and compare projection-weighted eigenvalues for active and non-active adjacent transitions. A token-local \(d_{\text{model}}\times d_{\text{model}}\) Jacobian captures only perturbations at one token. Because ViT attention couples tokens, a more complete analysis should either include the relevant sequence/patch dimensions in the Jacobian or explicitly label the token-local result as a restricted sub-Jacobian. Compare with non-TLCM transitions within the same model and examples.

4. **Use checkpoints only for the training-emergence question.** If checkpoints are available, evaluate the same fixed image set and preprocessing at each checkpoint, and report how the adjacent-block metric changes. That is a development-over-training analysis, not proof of causal cross-depth correction. To make a causal claim at a checkpoint, still intervene on block \(i+1\)’s input and measure its response.

## Controls and limitations

**Controls actually in the paper:** the non-TLCM Jacobian comparison; two independence-based cosine null models (Sec. 4.1 and Appendix C); and comparisons across models, including families where the signature is absent. The training experiment uses fixed text across checkpoints. The paper does **not** report a full set of sham or matched-direction controls for the scaling intervention.

**Useful additional controls for a ViT:**

- Use within-image, within-checkpoint paired runs: clean input versus the intervention, with all other activations and inputs held fixed. Do not infer causality by subtracting activations from different images, positions, or checkpoints.
- Compare \(a_i\) perturbations with norm-matched random directions, orthogonal directions, and directions from non-adjacent blocks. These help test whether the response is specific to the preceding block’s contribution rather than a generic response to perturbation magnitude.
- Include a zero-intervention/sham run and, where relevant, a matched-norm perturbation that preserves the input norm. This helps separate directional correction from norm or scale effects.
- Obtain contributions from the model’s actual residual branches/hooks, respecting the architecture’s normalization and sublayer ordering. Avoid treating a difference between unrelated residual states as a layer contribution. For Jacobians, differentiate the defined block-contribution function directly rather than estimating it by subtracting outputs from mismatched runs.
- Report the perturbation-response curve as well as the raw adjacent-layer cosine. A negative cosine alone is correlational; a response that changes systematically under a controlled input intervention is the relevant causal evidence.

The paper’s main limitations apply to the adaptation too: the Jacobian is a local linear approximation (Appendix D.0.1); \(\alpha=-1\) and large perturbations can be off-distribution; negative directions are not shown to correspond to semantically identified features; and the proposed “propose-and-reject” account is explicitly a conceptual framework that the authors do **not** directly test (Sec. 5.3). It also remains unknown whether TLCM appears in other architectures, so a six-block ViT should be tested rather than presumed to exhibit it.

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
