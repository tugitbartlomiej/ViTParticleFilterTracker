[Liu26p]:
## Provenance

The supplied PDF identifies itself as **“Finite-Horizon Input–Output Dynamics of Minibatch Perturbations in AdamW,” by Kang Liu and Suyan Li**. Its first page gives **arXiv:2608.19762v1**, dated **20 August 2026**. The PDF includes the main text, appendices A–E, and references (30 pages as supplied). It does not identify a journal or conference publication. These are details printed in the PDF; I cannot independently authenticate the document’s provenance. The key `[Liu26p]` is not itself given as a citation key in the paper.

## State, paired trajectories, and response

The optimizer state is joint: immediately **before** minibatch \(s\), it is
\[
x_s=[\theta_s,m_{s-1},v_{s-1}]^\top,
\]
where \(\theta_s\) are model parameters and \(m,v\) are AdamW’s first- and second-moment states. The moments are stored uncorrected; bias correction enters the parameter update. (Section 3.2, PDF p. 3.)

The control and shock trajectories start from the same state. At update \(t\), their gradients differ by a perturbation:
\[
g_t^{\rm shock}=g_t^{\rm control}+\xi_t.
\]
From \(t+1\) onward, they share the same minibatches, stochastic realizations, learning-rate schedule, and optimizer configuration. Their state and probe-loss differences are defined as shock minus control, so the response is signed: a positive \(\delta F\) means the shock raises the probe loss. (Section 3.2, PDF p. 3.)

The shock’s **linearized write-in** to the joint state is
\[
B_t\xi_t,\qquad
B_t=
\begin{bmatrix}
\Gamma_t\\
(1-\beta_1)I\\
2(1-\beta_2)\operatorname{Diag}(g_t^{\rm control})
\end{bmatrix}.
\]
The exact second-moment write-in also has a quadratic term in \(\xi_t\), which is part of the remainder. (Eq. 1, Section 3.3, PDF p. 4.) Subsequent joint parameter–moment dynamics are linearized by transition Jacobians \(A_s\) (Eq. 2, Section 3.4, PDF p. 4). Theorem 3.1 gives the signed, horizon-\(h\) tangent response:
\[
\delta F_{t+h}
=\alpha\,c_{t+h}^{\top}\Phi_{t+h,t+1}B_t\xi_t+o(\alpha),
\]
for a scaled perturbation \(\alpha\xi_t\). Here \(\Phi\) is the ordered product of future transition Jacobians and \(c_{t+h}\) reads out the probe-loss sensitivity. Horizon \(h=1\) is the first post-shock state. (Theorem 3.1, PDF pp. 4–5.) The paper also defines exact paired-trajectory loss differences; the displayed formula is their first-order approximation.

## Assumptions and finite-scale qualifications

The directional-response theorem assumes a fixed finite horizon, differentiable AdamW transitions along the control trajectory, a control trajectory in the interior of the chosen smooth regions, differentiable probe objectives, and positive bias-corrected second-moment coordinates where the relevant derivatives are taken. For piecewise-smooth networks, the theorem is local to the control activation regions. The stronger \(O(\alpha^2)\) finite-horizon accuracy claim additionally requires locally Lipschitz derivatives, bounded finite-horizon propagators and probe gradients, and a quadratic bound on activation-switching defects; smooth networks have no switching defect. (Theorem 3.1, PDF pp. 4–5; Theorem 3.4 and Corollary 3.5, PDF pp. 6–7.)

Empirically, the tangent is most accurate in a local perturbation range, with accuracy worsening at larger scales; the CNN–ReLU results also show later-horizon degradation associated with activation-pattern changes. The Pythia experiment is explicitly a **single-seed, descriptive** scaling analysis across nine fixed model–dataset conditions, not a set of independent replicated model-size experiments. The paper’s stated limitation is that exact ISO is pathwise, while present-time approximations can degrade with nonlinear dynamics, activation switching, or future dynamical drift. (Sections 4 and 5, PDF pp. 7–9; Appendix E.2.2–E.2.3, PDF pp. 24–27.)

## Is the future schedule alone enough?

**No.** Sharing a future minibatch sequence isolates the perturbation’s effect, but it does not make the future updates identical. At each future minibatch, the gradient is evaluated at the trajectory’s current parameters; since the parameters and moment states have diverged, the two trajectories generally have different gradients and updates. The paper states explicitly that future gradients “generally differ once their parameters diverge” (Section 3.4, PDF p. 4). Thus a schedule alone is not enough to obtain the response: the evolving parameter–moment states, their transition dynamics, and the probe readout matter. The pathwise ISO operator depends on the realized future sequence and the dynamics along the resulting control trajectory.

The repeated-future experiment reinforces that distinction: delayed candidate-specific structure persists after matching the immediate response, but its recovery from present-time scores is regime-dependent. For example, after one-step matching, the present-frozen ISO correlation is \(-0.066\) in the standard quadratic setting, versus \(0.941\) for MLP–GELU and \(0.755\) for CNN–ReLU. (Section 4.2 and Table 4, PDF pp. 8–9; Appendix E.3, PDF pp. 28–30.)

Figure 1 summarizes the mechanism: shock write-in to parameters and both AdamW moments, propagation through future updates, and delayed signed loss readout.  
[REGION: page=2, bbox=67,111,411,860]

Table 4 compares repeated-future structure and present-time score correlations. In the one-step-matched cases, the reported prospective-structure ratios remain \(0.730\) (quadratic), \(0.921\) (MLP–GELU), and \(0.700\) (CNN–ReLU), while present-time ISO performance varies by system.  
[REGION: page=9, bbox=75,69,265,805]

---

[Den26b]:
## What the paper claims

The paper is a **preprint** titled *How Faithful Is Trajectory-Based Data Attribution? Error Sources, Remedies, and Practical Guidelines*, by Junwei Deng, Pingbang Hu, Suliang Jin, Hao Lu, Jiachen T. Wang, Shichang Zhang, and Jiaqi W. Ma. The supplied PDF identifies itself as **arXiv:2605.18814v1**, dated May 12, 2026, and gives May 20, 2026 as the manuscript date.

It separates attribution error into:

- **Config-level error:** the attribution calculation does not match the training procedure. The paper focuses on the mismatch between SGD-based attribution formulas and models actually trained with AdamW.
- **Algorithm-level error:** the first-order Taylor approximation used to make trajectory attribution tractable.
- **System-level error:** effects such as floating-point precision and GPU nondeterminism, which the authors regard as largely irreducible by attribution-algorithm design.

The error taxonomy is in Section 2.2; Eq. (3) in Section 4.1 decomposes the SGD-based error into the optimizer-mismatch share, the remaining update-estimation error after optimizer alignment, and a residual.

### AdamW-influence and the Taylor error

In Section 3.1, **AdamW-influence** is derived by injecting a perturbation at the gradient level and propagating it through the parameters and AdamW’s first- and second-moment states. This is meant to address the fact that AdamW’s adaptive, coordinate-wise update is not the SGD update assumed by existing SGD-influence calculations. The recurrence is given in Appendix B, especially Eqs. (7)–(12); the authors claim the same asymptotic trajectory-length dependence, with a constant-factor cost overhead.

Against trajectory-specific leave-one-out (TSLOO) retraining ground truth, Table 1 reports higher Spearman correlation for AdamW-influence than SGD-influence in the four tested AdamW settings: MNIST+MLP, MNIST+CNN, WikiText+GPT-2, and Alpaca+Llama 3.2-1B. Reported relative gains range from 10% to 500%. In Section 4, the authors attribute most of the remaining algorithm-level error to first-order **update-estimation error**. They identify learning rate and remaining trajectory length as governing factors: larger learning rates worsen error, while shorter horizons reduce accumulated error (Section 4.2). Their per-sample error proxy in Eq. (4) is tested on MNIST MLP/CNN settings, with reported Spearman correlations of 0.83–0.89 against retraining-based error norms (Section 4.3).

These results support the paper’s claims **within the tested settings**; they do not establish that the proxy or error pattern generalizes to all architectures or training regimes. The paper itself notes that its LLM experiments reach only Llama 3.2-1B and that its empirical evaluation is supervised classification and language modeling (Section 7, Limitations).

## Online look-ahead versus retrospective “oracle”

The paper unifies online and offline selection by a horizon \(K\) (Section 5.1):

- **Online selection** scores candidates during the training run using an estimated effect \(K\) steps ahead, then selects the top-\(B\) candidates for the current update. \(K=0\) recovers the immediate-effect boundary case; finite positive \(K\) is a short-horizon look-ahead.
- **Offline selection** first completes a reference training trajectory, uses that trajectory to score data retrospectively, then retrains from scratch on a selected subset. The paper treats it as an approximation to the full-trajectory horizon, not as an exact counterfactual oracle.

Thus, online is prospective in *when* selection occurs, but the score is still a **first-order estimate**. It is not a direct observation of each candidate’s true future effect. Conversely, offline has hindsight about a completed reference trajectory, but it is not itself the paper’s ground-truth oracle. The ground truth used for attribution-fidelity experiments is **TSLOO**, computed by retraining with a sample excluded at its injection step (Section 2.1).

Section 5.2 reports that online AdamW-influence beats online SGD-influence in all four selection settings and matches or exceeds offline AdamW-influence in the tested cases. The authors do not claim offline is universally inferior. They also find that the best \(K\) grows as the learning rate falls in the MNIST MLP sweep: \(K=5,10,25\) at learning rates \(10^{-2},10^{-3},10^{-4}\), respectively (Figure 6).

## Falsification controls: what is shown and what would make the claims vulnerable

**Controls actually reported:**

1. **Optimizer-matching crossover.** Tables 3 and 4 compare the methods under AdamW and SGD training, respectively. AdamW-influence wins under AdamW; the symmetric SGD-trained check finds SGD-influence dominates under SGD. This is an important control against the simpler explanation that AdamW-influence is intrinsically better regardless of the optimizer.
2. **Direct retraining comparisons.** The fidelity and error analyses compare estimates against TSLOO or leave-one-out retraining-derived quantities. Section 4.2 also varies learning rate and evaluates along the trajectory.
3. **Horizon/learning-rate sweep.** Figure 6 varies \(K\) across learning rates, rather than relying only on a single chosen horizon.
4. **Validation and test reporting for selection.** Appendix D.3 says the online-method choices use validation examples, with final results reported on the test set.

**Useful stronger falsification checks, not established by the reported results:**

- For the optimizer claim, run a fully matched optimizer-by-attribution crossover across the same models, seeds, batches, and schedules—not just the settings shown—and test whether the matched method reliably wins against TSLOO.
- For the Taylor-error claim, compare predicted candidate rankings with actual \(K\)-step inclusion/exclusion retraining from identical checkpoints, controlling future batches and randomness. This directly tests prospective look-ahead accuracy; downstream test performance alone does not show that the estimated per-candidate future effects were accurate.
- For selection, compare against random and other non-attribution selection baselines under equal data, compute, and selection budgets; tune \(K\) only on validation and report untouched test results. The paper’s highlighted comparisons are offline AdamW-influence and online SGD-/AdamW-influence.
- Repeat the proxy and horizon results on larger models and different regimes. The paper explicitly identifies broader scale and task coverage as limitations.

## Bibliography and availability

The supplied reference list contains the relevant cited works, including Hara et al. (2019) for SGD-based data cleansing/influence, Loshchilov and Hutter (2017) for AdamW, Pruthi et al. (2020) for TracIn, Wang et al. (2024) for GREATS, and Wang et al. (2025b) for *Capturing the Temporal Dependence of Training Data Influence*. That verifies that these works are listed in this PDF’s bibliography; **I cannot independently verify the references’ external publication records or the preprint’s live availability from the supplied PDF alone**.

The PDF provides experimental details and reports roughly 400 A100 GPU-hours (Appendix D.4), but I do not see a code or data release statement in the supplied document. Accordingly, I can confirm only that the provided document labels itself an arXiv preprint with the identifier above—not that a public code/data artifact or an externally accessible version is available.

**Relevant figure/table locations:**

Figure 2 decomposes error over training steps and learning rates: the optimizer-mismatch share (green), Taylor/update-estimation error (blue), and residual (gray). It illustrates that the update-estimation component dominates much of the remaining error after optimizer alignment.  
[REGION: page=5, bbox=617,96,939,906]

Figure 3 plots parameter error norm and within-step Spearman correlation across learning rates and trajectory positions for MLP and CNN settings. It supports the reported trends that smaller learning rates and shorter remaining horizons improve the estimates.  
[REGION: page=6, bbox=326,96,604,908]

Figure 5 compares offline AdamW-influence with online SGD- and AdamW-influence on MNIST and Llama fine-tuning tasks. It is the central displayed evidence for the online-versus-offline selection claim.  
[REGION: page=8, bbox=73,97,362,906]

Figure 6 reports the MNIST MLP horizon sweep. The best displayed \(K\) increases from 5 to 10 to 25 as the learning rate falls from \(10^{-2}\) to \(10^{-4}\).  
[REGION: page=8, bbox=527,517,849,907]

Table 4 is the SGD-training crossover control: it reports higher fidelity for SGD-influence than AdamW-influence in the displayed MNIST settings trained with SGD.  
[REGION: page=19, bbox=528,96,783,827]

**Relevant bibliography entries as printed in the paper:**

- Satoshi Hara, Atsushi Nitanda, and Takanori Maehara. “Data cleansing for models trained with SGD.” *Advances in Neural Information Processing Systems*, 32, 2019.
- Ilya Loshchilov and Frank Hutter. “Decoupled weight decay regularization.” arXiv preprint arXiv:1711.05101, 2017.
- Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. “Estimating training data influence by tracing gradient descent.” *Advances in Neural Information Processing Systems*, 33:19920–19930, 2020.
- Jiachen T. Wang, Tong Wu, Dawn Song, Prateek Mittal, and Ruoxi Jia. “Greats: Online selection of high-quality data for llm training in every iteration.” *Advances in Neural Information Processing Systems*, 37:131197–131223, 2024.
- Jiachen T. Wang, Dawn Song, James Zou, Prateek Mittal, and Ruoxi Jia. “Capturing the temporal dependence of training data influence.” In *The Thirteenth International Conference on Learning Representations*, 2025b.

---

[Mis26b]:
## What is proved—and under what assumptions

The paper defines prompt reachability for a **fixed model** as whether a naturally produced activation has a preimage among discrete token sequences. With finite vocabulary \(V\) and context length \(K\), the prompt set \(S=V^{\le K}\) is finite (and in any case countable). Thus, for a fixed model, the naturally reachable activations form a countable set in the real-valued activation space. “Non-surjective” here means **no exact prompt preimage for the steered activation**, not merely that it is unlikely under ordinary prompts. See §3 and the steering rule in Definition 4.1, Eq. (1).

The probability claims are **genericity results using real analyticity and measure zero**, not empirical probability estimates over prompts:

- **Theorem 4.2 (random steering):** For fixed prompt pair and inspected token positions, the paper assumes model parameters \(\Theta\) and steering vector \(v\) are drawn from distributions with nonzero densities on their respective real-valued spaces. The collision function \(g(\Theta,v)\), defined from the squared activation difference, is real-analytic and not identically zero. The zero set of such a function has measure zero, so the probability of that particular collision is zero. The paper sets \(\lambda=1\) without loss of generality; this argument concerns nonzero steering, not the no-steering case. To extend from a fixed candidate prompt to *any* discrete candidate prompt, one uses that the candidate set is countable and takes a countable union of measure-zero collision events. The analyticity condition is supported by Theorem 3.1, which assumes real-analytic Transformer operations, including the MLP activation (for example, GELU or tanh).

- **Theorem 4.4 (difference-of-means steering):** Here \(v=v(\Theta,D)\) is extracted from a **fixed** contrast dataset \(D=D^+\sqcup D^-\), so it too is analytic in \(\Theta\). The main-text theorem requires \(|D^+|,|D^-|\ge 2\). Its witness construction in Appendix A additionally assumes **non-uniform lengths** in the contrast sets (A.1) and that there is no token position at which all prompts in either set share the same token (A.2). Under the paper’s density assumption on \(\Theta\), analyticity plus a witness where the collision function is nonzero again gives a measure-zero collision set.

- **Theorem 4.5 (adversarial steering):** This is a more limited claim: even if a vector is chosen to force a collision at one position between distinct prompts, the steered and natural activations **almost surely diverge at the next position**, under the paper’s parameter-distribution and analytic-function argument. It is not a claim that an adversarial vector cannot force a one-position collision.

The authors say the injectivity result they rely on is preserved after finitely many gradient-descent training steps (§3, citing Nikolaou et al.). But the formal probability arguments above explicitly use parameter distributions with nonzero densities. So the statement about trained LLMs relies on the paper’s stated extension from injectivity; it should not be read as an unconditional measure-theoretic theorem for every fixed trained model or every training procedure.

A further scope point: the theorem concerns **exact activation equality**. It does not say that no prompt can elicit similar output text or similar behavior. Indeed, §6 gives a black-box “Here” attack that can induce harmful responses without reproducing the steered internal trajectory.

## Why this does not automatically transfer to continuous RGB vision

The crucial discrete-input step does not carry over to an idealized continuous RGB input domain such as \([0,1]^{H\times W\times 3}\): that domain is **uncountable**. For a fixed vision model, its naturally reachable activations therefore need not be a countable set, so the paper’s “countable prompts leave holes” argument—and the countable-union extension over candidate inputs—does not apply.

Uncountability does **not** prove that every activation is reachable. It means only that non-surjectivity must be established by a different argument. The image-to-activation map could have a lower-dimensional image, but it could also have an image with nonempty volume, depending on input dimension, model map, and the activation being considered. The paper supplies no theorem for continuous images. Its analyticity assumptions would also need to be checked for the vision architecture and preprocessing; they cannot simply be presumed.

## What analogy does survive

The useful conceptual analogy is to define a **natural-input-to-activation map** for the vision model, then ask whether a steered activation lies in its image—possibly with constraints such as natural-image distributions, valid pixel ranges, or allowed image perturbations. White-box steering and black-box image-based reachability should still be evaluated separately. But the answer for continuous RGB inputs requires new assumptions and analysis; it does not follow from this paper’s discrete-prompt theorem.

Figure 1 illustrates the paper’s core picture: discrete prompts map to a sparse set of activation states, while steering can move states off that set; it also distinguishes random/DOM steering from a forced one-step collision that diverges next.  
[REGION: page=2, bbox=77,446,428,848]

---

[Koh19c]:
## What the paper establishes

**Group effects can be ranked well, but are not necessarily estimated precisely.** Influence functions use a first-order approximation to the effect of removing training examples (Section 2.1, Eq. 4). In the constructed-group experiments, predicted and actual effects were strongly rank-correlated (Spearman \(\rho=0.89\)–\(0.99\)), despite sometimes large absolute and relative errors (Section 3). Influence usually underestimated the actual effect; the stated exception is test-loss effects when the actual effect is negative. For natural groups, the reported correlations were \(\rho=1\) for CDR labeling functions and \(0.77\)–\(0.86\) for MultiNLI crowdworkers (Section 5).

**There is no general accuracy guarantee for large groups.** Removing a coherent group can be a large change, outside the small-perturbation setting that motivates the first-order approximation. The paper’s one-step Newton approximation is empirically closer to the actual effect (Figure 2); Proposition 1 (Section 4.1) bounds its error under smoothness assumptions. But the paper’s theoretical correlation/underestimation results are restricted: they obtain cone-type constraints for self-loss (Proposition 3, Section 4.3) and for test prediction when removing copies of a single point (Proposition 4, Section 4.4). In general test-prediction settings, those properties need not hold; the paper constructs counterexamples (Section 4.4, Figure 3).

**Additivity applies to influence in specific cases—not to actual group effects generally.** For test prediction or test loss, Eq. 4 makes influence additive across subsets: the influence of a union is the sum of its constituent influences. That is a property of the approximation at the original fitted model. Retraining after removing a group changes the model, so actual group effects need not equal sums of single-point actual effects. Moreover, influence on **self-loss** is itself not additive: its evaluation function depends on which subset was removed (Section 2.1).

**Training-point weighting is part of the setup.** The main formulation represents removal with binary weights \(w_i\in\{0,1\}\) (Eqs. 1–3). The CDR application uses fractional weights: if an example receives labels from \(k\) labeling functions, the authors include \(k\) copies, each with weight \(1/k\), and account for those weights in training, influence, and actual-effect calculations (Appendix B.1). Thus, the reported CDR effects are for weighted contributions, not simply unweighted removal of distinct examples.

**Nonconvex models are not evaluated.** The theoretical setup assumes a convex, twice-differentiable loss in the parameters (Section 2); the experiments use logistic regression or softmax regression. Although the MultiNLI task is language inference, their influence experiment uses logistic regression on a continuous-bag-of-words representation (Appendix B.2). The Discussion explicitly leaves open whether the observed trends hold for nonconvex models such as neural networks. The paper therefore does not establish these accuracy results for neural-network training.

## Separate inference: minibatch interactions, AdamW, and batch normalization

The paper studies removing **groups of training examples from a fitted objective**; it does not study minibatch composition, batch order, or optimizer-state effects. Its additive-influence result should not be read as saying that examples’ effects during minibatch training are independent.

In particular, AdamW’s training trajectory and optimizer state, and batch normalization’s dependence on batch statistics, can introduce interactions that are not represented by the paper’s pointwise-loss/Hessian calculation. With batch normalization, a point can affect the batch statistics used for other examples; removing it can therefore change more than its own loss contribution. These are implications of those training mechanisms, **not results demonstrated in this paper**. Applying the paper’s conclusions to such settings would require separately accounting for the training procedure and its batch-dependent behavior.

Figure 1 compares influence with actual group-removal effects across constructed groups and datasets; the key visual point is strong association alongside noticeable deviations from the slope-one reference.  
[REGION: page=5, bbox=71,154,511,848]

Figure 3 gives synthetic counterexamples for test prediction: influence and the Newton approximation can disagree in sign, and even in the single-point-copy setting the scaling between them can vary across groups.  
[REGION: page=7, bbox=70,160,252,863]

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
