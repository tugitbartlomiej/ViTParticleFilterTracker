# Projecting high-dimensional vectors (velocities, forces, per-component contributions) and vector fields into 2D/3D: practice, pitfalls, remedies

Scope note for the report writer: our system is a ViT with residual dimension d = 192, 600 tokens ("colloids") per frame, 768 MLP neurons ("laser beams") each contributing a force u_c in R^192, net force f = sum_c u_c, a local field with a 192x192 Jacobian J defining "traps", and one fixed linear 3D view P3 (top-3 PCA of fc2 columns). Measured: P3 keeps a median 47% of net-force length and 18.8% of beam energy; orthogonal beam pairs look "opposite" in 3D at 13.5-15.5%; vortex/saddle labels depend on the slice; the slice zero of the field is not a 192D equilibrium.

Tooling note: the Scite MCP tool returned "monthly MCP usage limit (250 calls)" on the first call, so no Scite citation statements were used. Sources come from Undermind (deep search "Projecting high-dimensional velocity vectors and vector fields onto 2D 3D embeddings artifacts", 83 relevant papers, workspace link: https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Projecting%20high-dimensional%20velocity%20vectors%20and%20vector%20fields%20onto%202D%203D%20embeddings%20artifacts), Undermind full-text PDF reads, and direct fetches of publisher, arXiv, Europe PMC and Transformer Circuits pages.

## Q1. RNA velocity as the closest analogue: how much of the 2D arrow field is an artifact of the embedding and the kNN smoothing, and what are the remedies?

### Takeaway
In RNA velocity, the standard 2D arrows are not projections of the high-dimensional velocity. They are neighbour-weighted averages of observed displacements in the embedding (a cosine/correlation kernel over a kNN graph), so they interpolate among sampled states, mostly discard magnitude, and inherit the embedding's distortions. Quantitative critiques show low speed fidelity (PCC 0.112 for the kernel vs 0.896 for direct PCA projection), more than 50% of cell-level arrows strongly wrong under realistic noise, and only 2-9.5% of displayed cell arrows passing a faithfulness test. The robust remedies are to compute in the high-dimensional or PCA space and display afterwards (exact linear projection, Dynamo, ddHodge), or to test each displayed arrow (velotest). Our P3 is linear, so the kernel artifacts do not apply. The norm-loss and view-dependence lessons do.

### Cited Findings
**Original methods (what is computed where)**
- La Manno et al. 2018 define RNA velocity as "a high-dimensional vector that predicts the future state of individual cells on a timescale of hours" (estimated per gene from unspliced/spliced counts). Citation: La Manno G, Soldatov R, Zeisel A, et al., Kharchenko PV. RNA velocity of single cells. Nature 560:494-498 (2018). — [doi:10.1038/s41586-018-0414-6](https://doi.org/10.1038/s41586-018-0414-6)
- velocyto projects velocities with a transition matrix built from Pearson correlations between each cell's velocity vector and its displacements to neighbouring cells, then takes a vector sum over neighbours. The method ignores vector length, so vectors with the same direction but different magnitudes can look the same. — [Marot-Lassauzaie et al. 2022, PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010031)
- scVelo's `velocity_embedding` "Projects the single cell velocities into any embedding". The embedded velocity is the expected displacement: normalized embedding differences to other cells, weighted by the transition matrix minus a uniform 1/n baseline. `direct_pca_projection` instead projects "directly into PCA space, thus skipping the velocity graph", and `autoscale` rescales arrows by a scalar for display. (So displayed arrow length is rescaled by design.) — [scVelo docs](https://scvelo.readthedocs.io/en/stable/scvelo.tl.velocity_embedding.html). Underlying paper: Bergen V, Lange M, Peidli S, Wolf FA, Theis FJ. Generalizing RNA velocity to transient cell states through dynamical modeling. Nat Biotechnol 38:1408-1414 (2020). — [doi:10.1038/s41587-020-0591-3](https://doi.org/10.1038/s41587-020-0591-3)
- Both velocyto and scVelo compute velocity in gene space, compute the kNN graph and transition probabilities in a reduced space, and compute the arrows in the 2D embedding. Gorin et al. distinguish two conversion routes: a deterministic difference of embedded points (Eq. 3) and a nonlinear softmax kernel average over k embedding neighbours (Eq. 4). Packages "almost exclusively" use the nonlinear route, which they say "has no physical interpretation" and has no formal validation against a linear or simulated ground truth. — [Gorin, Fang, Chari & Pachter 2022, PLoS Comput Biol 18(9):e1010492](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010492)
  - Correction to the assignment: the authors of "RNA velocity unraveled" are Gorin G, Fang M, Chari T, Pachter L, not "Gorin, Vastola & Pachter". DOI 10.1371/journal.pcbi.1010492.

**Critiques with quantitative results**
- Gorin et al. 2022:
  - "There is no consensus on the appropriate choice of embedding, number of neighbors, or measure of concordance". velocyto uses k from 5 to 300 and scVelo uses k = 30.
  - Moving from ambient space to 2D gives average Jaccard distances of neighbourhoods (k = 150) of about 70-75%. They conclude that cell embedding "substantially distorts precisely the local structure relevant to velocity embedding".
  - Angle deviations of kernel-embedded arrows from a linear PCA baseline were "all severely biased".
  - A sign-only (Boolean) velocity performed "nearly identical to the built-in methods", so velocity magnitudes have little effect on the displayed field.
  - A model with no latent dynamics can still produce apparent signal in the embedding (S1 Text, Fig F).
  - "Using visualizations for discovery thus risks confirmation bias."
  — [Gorin et al. 2022](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010492)
- Zheng et al. 2023, low-noise simulation with true velocities:
  - Kernel (transition-probability) and direct PCA projection agree on direction (median cosine similarity 0.984).
  - They disagree on speed. The correlation between high-dimensional speed and displayed speed is PCC = 0.896 for PCA projection vs PCC = 0.112 for transition probabilities (RMSE 1.613).
  - The authors attribute this to the kernel not using velocity magnitude.
  — [Zheng, Stein-O'Brien, Boukas, Goff & Hansen 2023, Genome Biol 24:246, doi:10.1186/s13059-023-03065-x](https://doi.org/10.1186/s13059-023-03065-x) (full text read via Undermind)
- Zheng et al. 2023, effects of noise and the neighbour graph:
  - At noise level 5 with the learned kNN graph, the kernel field "looks smooth" but more than 50% of cell-level vectors are strongly dissimilar to the true mapped vectors. Using the true graph substantially improves this.
  - At noise level 3, gene-level velocity PCC rises from 0.455 (learned graph) to 0.823 (true graph), and speed PCC from 0.636 to 0.860.
  - The kNN graph matters most in the pre-processing smoothing step.
  - Removing beta cells (53 pre-endocrine cells tested) redirected arrows toward alpha cells with the gene-level velocities unchanged (Watson test P < 0.001). The arrow must point into the convex hull of neighbours-of-neighbours, so the kernel "interpolates" within the embedding.
  - Using RNA velocity to validate an embedding is circular.
  — [Zheng et al. 2023](https://doi.org/10.1186/s13059-023-03065-x)
- Marot-Lassauzaie et al. 2022:
  - t-SNE maps dense regions of gene space onto larger areas, so arrows there look longer than they are.
  - Distance-preserving embeddings (PCA, diffusion maps with a constant kernel width) represent arrow length more faithfully.
  - PCA is the only approach in which arrows truly represent the high-dimensional vectors.
  - The neighbourhood sum is not an orthonormal basis, so the projected direction depends on neighbourhood shape.
  - Remedies: a Nystrom projection of velocity end points (κ-velo); computing the embedding on the same gene set as the velocities; and eco-velo, which builds arrows directly in the embedding from mutual nearest neighbours.
  — [Marot-Lassauzaie V, Bouman BJ, Donaghy FD, Demerdash Y, Essers MAG, Haghverdi L. PLoS Comput Biol 18(9):e1010031 (2022)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010031)
- Bergen et al. 2021 (review):
  - "Arbitrary directions are projected onto the UMAP representation" of PBMCs, even though the velocities came from only three genes (NKG7, IGHM, GNLY) with no transient states.
  - Simulations of mature cell types produce "projected arrows ... that are not seen in the ground-truth vector field".
  - "Incorrect directions can not only result from erroneous velocity estimates, but also result from biases in the way velocities are projected."
  - Uncertainty in the estimates is "not being propagated into the projection".
  — [Bergen V, Soldatov RA, Kharchenko PV, Theis FJ. RNA velocity—current challenges and future perspectives. Mol Syst Biol 17:e10282 (2021), doi:10.15252/msb.202110282, Europe PMC full text PMC8388041](https://europepmc.org/article/PMC/PMC8388041)
- velotest (Bischoff et al. 2025, preprint) is a per-cell test. It asks whether the high-dimensional preimage of the displayed 2D direction aligns with nearby cells better than alternative 2D directions at least γ degrees away (mean high-dimensional cosine to neighbours inside a 2β cone, Bonferroni-corrected).
  - Fractions of cells whose displayed velocity is significantly better than random: scVelo pancreas 9.5% (stochastic) and 8.1% (dynamical); dentate gyrus 6.0%; COVID 2.0%; bone marrow 2.2%; gastrulation erythroid 3.9%; VeloViz 5.0%; Nyström 3.2%; SIRV spatial 6.7% and 3.1%.
  - A non-significant result means "not shown faithful", not "proven wrong".
  — [Bischoff S, Policar PG, Mukherjee S, Macke J, Claassen M, Schröder C. bioRxiv (2025) doi:10.1101/2025.10.26.683064](https://doi.org/10.1101/2025.10.26.683064)

**Remedies: compute in high-dimensional space, or build velocity-consistent embeddings**
- Dynamo (Qiu et al. 2022) learns an analytic continuous vector field with sparseVFC, a vector-valued RKHS kernel expansion with EM-based outlier rejection. Its STAR Methods give analytic formulas:
  - Jacobian; divergence ∇·f = Tr J; acceleration a = Jv; curvature κ = (Jv(v·v) − v(v·Jv))/‖v‖^4.
  - "The curl is only computable in 2D or 3D."
  - Fixed points satisfy f(x) = 0, solved with fsolve from Latin-hypercube starts, and are classified by Jacobian eigenvalues (all negative = attractor, all positive = repulsor, mixed = saddle).
  - A PCA-space Jacobian is mapped to genes as G = Q J Q^T with loadings Q (d x k). The text says this "is only an approximation of the true gene-specific Jacobian, as only k < d principal components are used".
  - The analytic formulae give "nearly 1000-fold speedup" over numerical approaches.
  — [Qiu X, Zhang Y, Martin-Rufino JD, et al., Weissman JS. Mapping transcriptomic vector fields of single cells. Cell 185(4):690-711.e45 (2022), doi:10.1016/j.cell.2021.12.045, PMC9332140](https://pmc.ncbi.nlm.nih.gov/articles/PMC9332140/)
- Dynamo's spaces, and a correction to a common belief. In the hematopoiesis analysis, the fields were reconstructed in top-30-PC space and in 2D UMAP space, and gene-regulation Jacobians came from the 30-PC field. Fixed points were searched for in the UMAP-space field: "As the dimension increases, the confidence of fixed point identification deteriorates, so we used the vector field constructed from the UMAP space to search for fixed points". So Dynamo does not compute everything in the full space before display. Its fixed points and their topology came from a 2D field. — [Qiu et al. 2022, STAR Methods, read via Undermind PDF](https://doi.org/10.1016/j.cell.2021.12.045)
- ddHodge (Maehara & Ohkawa, Nat Commun 2025) computes gradient, curl and divergence on local PCA tangent patches (e.g., r = 4, k = 12) of the data manifold in the original space. UMAP is used only for display.
  - The authors argue that UMAP equalizes density, so "geometric features such as length and volume are lost", and that velocity embeddings can show "nonexistent closed trajectories at the data boundaries".
  - They state that "the full ambient-space Jacobian cannot be identified by any method—including ddHodge or Dynamo—unless additional assumptions" are made about unobserved directions.
  - Simulated MSEs against Dynamo: divergence 0.0027 vs 0.1142, curl 0.0027 vs 0.0742, Jacobian 0.0051 vs 0.1050.
  - Their curl is approximate and recovers rotation only within the PC1-PC2 plane.
  — [doi:10.1038/s41467-025-67782-6](https://doi.org/10.1038/s41467-025-67782-6)
- Contrasting claim: Su, Tong & Wei 2024 run a Hodge decomposition (discrete exterior calculus) of the velocity field directly in UMAP/t-SNE space. They claim the components reveal cell cycle and bifurcation "regardless of the choice of the low-dimensional representation". This is a consistency-across-embeddings argument, not a fidelity-to-ambient-space test. — [J Chem Inf Model 64:3558-3568 (2024), doi:10.1021/acs.jcim.4c00132](https://doi.org/10.1021/acs.jcim.4c00132)
- Graph-Dynamo constrains velocities to the tangent space of the data manifold. "Unlike a popular cosine correlation kernel ... this tangent space projection (TSP) preserves the magnitude information of a vector when one transforms between different representations." — [Zhang Y, Qiu X, Ni K, Weissman JS, Bahar I, Xing J. bioRxiv 2023, doi:10.1101/2023.09.24.559170](https://doi.org/10.1101/2023.09.24.559170)
- Velocity-informed embeddings:
  - VeloViz builds 2D/3D embeddings from predicted future states. — [Atta L, Sahoo A, Fan J. Bioinformatics 38(2):391-396 (2022), doi:10.1093/bioinformatics/btab653](https://doi.org/10.1093/bioinformatics/btab653)
  - FlowArtist jointly learns an embedding and a vector field. — [MacDonald K, et al., Krishnaswamy S. IEEE MLSP 2023, doi:10.1109/MLSP55844.2023.10285942](https://doi.org/10.1109/MLSP55844.2023.10285942)
  - FlowMap constrains velocity to the reconstructed manifold geometry (preprint). — [Hu J-Y, et al., Ma R. bioRxiv 2026, doi:10.64898/2026.09.21.753192](https://doi.org/10.64898/2026.09.21.753192)
  - VeloAE learns a low-dimensional velocity representation. — [Qiao C, Huang Y. PNAS 2021 (bioRxiv doi:10.1101/2021.03.19.436127)](https://doi.org/10.1101/2021.03.19.436127)
- CellRank avoids reading arrows: it builds a Markov chain from velocity plus similarity and "accounts for uncertainty in velocity vectors" to compute fate probabilities. — [Lange M, Bergen V, Klein M, et al., Theis FJ. Nat Methods 19(2):159-170 (2022), doi:10.1038/s41592-021-01346-6](https://doi.org/10.1038/s41592-021-01346-6)

### Inferences
- The failure modes that dominate RNA velocity are the kNN kernel, the nonlinear embedding and pre-smoothing. None of them is present in our pipeline, because P3 is a fixed linear map applied directly to u_c and f. Zheng et al. found that direct PCA projection keeps speed (PCC 0.896) where the kernel does not (0.112). This supports keeping a linear view rather than moving to UMAP/t-SNE for the force display.
- The lessons that do transfer are these:
  - Arrow magnitude and direction in a low-dimensional view need explicit validation against the full-space quantity. This is velotest's logic, and its 2-10% pass rates show how optimistic visual coherence can be.
  - Smooth-looking fields can hide more than 50% wrong local arrows.
  - Fixed points found in a reduced field are a different object from full-space fixed points; even Dynamo falls back to UMAP for them.
- A velotest-style check maps onto our system. For each token, compare the displayed 3D direction of f with alternative 3D directions using the 192D cosine to the true f, or to the displacement of the token to the next layer. This would give a per-token "faithful / not shown faithful" flag.

### Gaps
- The full text of La Manno et al. 2018 (exact kernel formula, σ, and their own caveats) was not read directly in this session. The kernel description above comes from Marot-Lassauzaie et al., Gorin et al. and the scVelo docs.
- Gorin et al. S1 Text figures (the no-dynamics control and angle-deviation magnitudes) were not read, because the fetched text was truncated at 100k characters.
- No peer-reviewed head-to-head benchmark of PCA vs UMAP vs t-SNE faithfulness for velocity arrows was found. velotest does not report one.
- Scite citation statements (supporting/contrasting citations of these critiques) could not be retrieved because of the usage limit.

## Q2. Exact pushforward: linear projections vs nonlinear embeddings (Jacobians, parametric embeddings, autoencoders)

### Takeaway
For a linear projection P, the displayed vector P·v is the exact pushforward, and sums are preserved (P·Σu_c = ΣP·u_c). What it hides is the out-of-view component (I−P)v, and with it the fraction of norm retained. For a nonlinear embedding φ, the exact local pushforward is J_φ(x)·v. That is available only when φ is an explicit differentiable map (parametric t-SNE/UMAP, autoencoders) or through implicit differentiation of the embedding objective. Neighbour-kernel arrows are neither. In all cases, PCA views chosen by variance can produce structure (for example Lissajous curves) that is a property of the projection, not of the dynamics.

### Cited Findings
- PCA is "the only approach in which arrows truly represent the high-dimensional vectors", though it may not capture complex branching. — [Marot-Lassauzaie et al. 2022](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010031)
- The linear PCA baseline is "only consistently used in Revelio". scVelo implements a PCA projection but disclaims it as unrepresentative of high-dimensional dynamics. — [Gorin et al. 2022](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010492)
- Dynamo's linear "pushforward/pullback" of a Jacobian is G = Q J Q^T. It is explicitly "only an approximation" when k < d PCs are used. — [Qiu et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9332140/)
- Implicit multidimensional projection (Bian et al. 2020, IEEE TVCG) transforms local subspace basis vectors through non-parametric projections by "analytical differentiation of multidimensional projections formulated as implicit functions", and displays the results as glyphs. — [doi:10.1109/TVCG.2020.3030368](https://doi.org/10.1109/TVCG.2020.3030368)
- Parametric UMAP keeps UMAP's graph-construction step and replaces the layout optimization with a neural network that maps data to the embedding. This gives an explicit differentiable φ, so vectors could be pushed forward through autodiff Jacobians. — [Sainburg, McInnes & Gentner, arXiv:2009.12981](https://arxiv.org/abs/2009.12981v3); implementation: [umap-learn docs](https://umap-learn.readthedocs.io/en/latest/parametric_umap.html)
- Zhang & Mueller (2026, arXiv) show that projection glyphs, map-continuity scores and transport analyses all derive from one geometric object of a differentiable embedding.
  - First order recovers projection glyphs; second-order curvature quantifies how far the linear approximation can be trusted.
  - A path-integral view detects path dependence, which "no amount of local measurement" reproduces.
  — [arXiv:2608.06809](https://arxiv.org/abs/2608.06809)
- Liu, Ma & Zhong use a leave-one-out map ("LOO-map") to show that t-SNE/UMAP maps have two kinds of discontinuity.
  - Overconfidence-inducing: mixtures are drawn as well-separated clusters.
  - Fracture-inducing: "Small spurious and artificial clusters form in the embedding space, even for non-clustered data".
  - Perturbation scores detect misplaced out-of-distribution points with AUROC about 0.70-0.80.
  - So the local Jacobian of t-SNE/UMAP is ill-defined or unstable at many points.
  — [Nat Commun (2025), doi:10.1038/s41467-025-60434-9](https://doi.org/10.1038/s41467-025-60434-9)
- PCA of high-dimensional random walks (Antognini & Sohl-Dickstein, NeurIPS 2018):
  - Explained variance of component k is ρ_k = 6/(π²k²): "Roughly 60% of the variance is explained by the first component, ∼80% by the first two".
  - The projection onto any pair of PCs is a Lissajous curve.
  - Neural-network training trajectories in PCA views resemble a random walk with drift.
  - The authors call low-dimensional embedding of such spaces "a difficult, perhaps even futile, task".
  — [arXiv:1806.08805](https://arxiv.org/abs/1806.08805)
- Phase Space Projection (Bartolovic, Gross & Günther 2020) optimizes a linear-operator-like projection by energy minimization to "preserve properties of the flow in the high-dimensional space". Their "3D projection maintains a number of useful flow properties, such as critical points and flow maps", and they compare it against PCA, t-SNE and UMAP. — [Comput Graph Forum 39(3) (2020), doi:10.1111/cgf.13978](https://doi.org/10.1111/cgf.13978)
- Hypertrix adapts Tissot's indicatrix (distortion ellipses) to show distortion when "data is projected from arbitrarily high dimensions onto a 2D plane". — [Raval, Viégas & Wattenberg, IEEE VIS 2024, doi:10.1109/VIS55277.2024.00073](https://doi.org/10.1109/VIS55277.2024.00073)
- Aupetit 2007 is the earlier visualization of distortions in continuous projections. — [Neurocomputing 70:1304-1330, doi:10.1016/j.neucom.2006.11.018](https://doi.org/10.1016/j.neucom.2006.11.018)

### Inferences
Synthesis table: where each technique computes, and what it preserves (each row is supported by the cited findings above).

| Technique | Computed in | Preserved | Lost / risk |
|---|---|---|---|
| Linear projection P·v (PCA, our P3) | exact, full → view | additivity (P·Σu = ΣP·u), linear ratios along a direction | out-of-view component (I−P)v; retained fraction invisible unless encoded |
| velocyto/scVelo kernel arrows | velocity in gene space; arrows in embedding via kNN cosine kernel | rough direction toward sampled neighbours | magnitude (PCC 0.112), extrapolation, additivity (kernel is nonlinear) |
| Nystrom / κ-velo, eco-velo | embedding space using sampled points | positivity, a weighted average of real states | cannot extrapolate beyond sampled region |
| Parametric embedding + J_φ·v | full → view by autodiff | exact local pushforward and linearity at a point | sums across points; valid only where φ is smooth (cf. LOO-map discontinuities) |
| Bian 2020 implicit differentiation | full → view | local subspace transform of a non-parametric projection | global consistency |
| Dynamo | field in PCA (30 PCs) or UMAP; Jacobian in PCA, mapped by QJQ^T | analytic J, div, acceleration, curvature | curl only in 2D/3D; fixed points from a UMAP field; G approximate for k < d |
| ddHodge | tangent patches of manifold in original space | divergence, gradient/curl components on the manifold | full ambient Jacobian unidentifiable; curl only in PC1-PC2 |
| Phase Space Projection | optimized projection operator | critical points, flow maps (by construction) | other geometry traded off |

- Our P3 is exact and additive, so the drawn beam arrows sum exactly to the drawn net force. This is the strongest property of the current design and should be kept.
- The measured "median 47% of net-force length, 18.8% of beam energy" needs a chance baseline. For an isotropic random vector in R^192 projected onto a fixed 3D subspace, E[‖Pv‖²/‖v‖²] = 3/192 ≈ 1.6%, a typical length ratio of about √(3/192) ≈ 12.5%. This is my elementary derivation (by symmetry each coordinate contributes 1/d), not a cited result. P3 therefore captures about 12x the chance energy for beams and well above chance for the net force. Still, more than half of the net-force length and about 80% of beam energy lie outside the view.
- P3 = PCA of fc2 columns weights all 768 beams equally, regardless of their activations on a given token. Marks & Tegmark's observation (Q5) that top PCs can be dominated by a nuisance feature suggests the same risk here. An activation-weighted PCA, or per-token adaptive views (e.g., span of f and the two largest out-of-view residual directions), would likely retain more of the displayed quantity. This is a hypothesis to test.
- The "orthogonal pairs look opposite" rate of 13.5-15.5% is what isotropy predicts. Two orthogonal 192D vectors project to nearly independent isotropic 3D vectors. The cosine between two independent isotropic 3D directions is uniform on [−1, 1] (Archimedes' hat-box property), so P(cos < −c) = (1 − c)/2, which is 15% for c = 0.7 and 13.5% for c = 0.73. This is my derivation. It should be checked against the threshold actually used in our measurement. The rate is therefore a pure projection artifact, and "opposition" should be computed as the 192D cosine and encoded explicitly (Q4).

### Gaps
- No source was found that gives an established, cited formula for the distribution of the retained norm under PCA projections of structured (non-isotropic) vector sets. Dasgupta & Gupta's Johnson-Lindenstrauss lemma (Random Struct Algorithms 22:60-65, 2003) covers random projections, but its PDF could not be fetched (HTTP 403), so the Beta(k/2, (d−k)/2) statement is not verified here.
- The publication venue of Parametric UMAP (Neural Computation 2021) was not confirmed in this session. Only arXiv versions were verified.
- Not found: any paper that explicitly visualizes per-component contribution vectors with exact linear pushforward and an encoded retained-norm channel, which is our exact use case.

## Q3. Visualizing vector fields and their topology: critical points, slices/projections vs the full field, projection-invariant descriptors, and fields in more than 3 dimensions

### Takeaway
Classical topology (Helman & Hesselink; Perry & Chong; Chong, Perry & Cantwell) classifies critical points by the eigenvalues of the Jacobian, or in 3D by its invariants P, Q, R, and connects them by separatrices. The classification is valid for whatever field it is applied to. Applied to a 3D slice or compression of a 192D field, it describes the slice field, not the 192D dynamics. Linear algebra gives one-directional guarantees. If the symmetric part of the full Jacobian is negative definite (a contracting trap), every compression is also contracting. A saddle or vortex seen in a slice implies nothing comparable about the full field. Zeros of the projected field are not full-space equilibria. Full-space fixed and slow points should be found by optimization in 192D, then projected. Visualization of vector-field topology beyond 3D exists only for 4D and is rare.

### Cited Findings
- Helman & Hesselink introduced vector field topology to visualization. The pipeline locates critical points, classifies them by Jacobian eigenvalues (attracting, repelling, saddle-like) and traces separatrices to form a topological skeleton. Only first-order critical points are detected with linear interpolation. Citations: Representation and display of vector field topology in fluid flow data sets, Computer 22(8):27-36 (1989), [doi:10.1109/2.35197](https://doi.org/10.1109/2.35197); Visualizing vector field topology in fluid flows, IEEE CG&A 11(3):36-46 (1991), [doi:10.1109/38.79452](https://doi.org/10.1109/38.79452). Method summary from secondary sources: [arXiv:1503.07137](https://arxiv.org/pdf/1503.07137), [ar5iv 1004.4485](https://ar5iv.arxiv.org/html/1004.4485)
- Chong, Perry & Cantwell 1990 relate and classify solution trajectories of three coupled linear ODEs "using three matrix invariants". The results "are valid for any smooth three-dimensional vector field". — [Phys Fluids A 2:765-777 (1990), doi:10.1063/1.857730](https://doi.org/10.1063/1.857730)
- Perry & Chong 1987 review critical-point concepts for eddying motions and flow patterns. — [Annu Rev Fluid Mech 19:125-155, doi:10.1146/annurev.fl.19.010187.001013](https://doi.org/10.1146/annurev.fl.19.010187.001013)
- Classic vortex criteria (Q, Δ, λ2, λcr/λci) use the rate-of-rotation tensor, "which is known to vary with respect to a reference frame". Martins et al. construct objective versions. So "vortex vs saddle" labels depend on the frame even in 3D. — [Martins et al., Comptes Rendus Mécanique 344:52-59 (2015), doi:10.1016/j.crme.2015.08.002](https://doi.org/10.1016/j.crme.2015.08.002)
- Dynamo: "The curl is only computable in 2D or 3D". Divergence = Tr J is defined in any dimension. Fixed-point identification "deteriorates" with dimension, and nullclines are limited to 2D. — [Qiu et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9332140/)
- ddHodge's curl recovers rotation only within the PC1-PC2 plane, an explicit plane-dependence of rotational descriptors. — [Maehara & Ohkawa 2025](https://doi.org/10.1038/s41467-025-67782-6)
- Numerical range facts: W(A) "contains all the eigenvalues of A", and "the restricted numerical range is contained in the full numerical range of A" (compression by a projector). — [Wikipedia: Numerical range](https://en.wikipedia.org/wiki/Numerical_range)
- Logarithmic norm facts:
  - Euclidean logarithmic norm: M₂[A] = α[(A + A*)/2] (the largest eigenvalue of the symmetric part).
  - The spectral abscissa satisfies α[A] ≤ M[A].
  - "The continuous system is stable with a contractive flow if M₂[A] ≤ 0". For nonlinear ẋ = f(x), M₂[f] ≤ 0 implies a contractive flow.
  — [Wikipedia: Logarithmic norm](https://en.wikipedia.org/wiki/Logarithmic_norm)
- Fixed points in high-dimensional neural dynamics:
  - Sussillo & Barak 2013 is the seminal work on finding fixed and slow points in the full state space of RNNs and linearizing around them. — [Neural Comput 25:626-649, doi:10.1162/NECO_a_00409](https://doi.org/10.1162/NECO_a_00409)
  - FixedPointFinder is a toolbox "for identifying and characterizing fixed points in recurrent neural networks". — [Golub & Sussillo, JOSS 3:1003 (2018), doi:10.21105/joss.01003](https://joss.theoj.org/papers/10.21105/joss.01003)
- Vector field topology beyond 3D:
  - Hofmann, Rieck & Sadlo 2018 classify and display critical points of 4D vector fields and extract their invariant manifolds. They use "a 4D camera that provides concise representation by exploiting projection degeneracies, and a 4D clipping approach that avoids self-intersection in the 3D projection". — [Comput Graph Forum 37(3) (2018), doi:10.1111/cgf.13421](https://doi.org/10.1111/cgf.13421)
  - Earlier: Dickinson 1991, interactive analysis of 4D vector-field topology. — [IBM J Res Dev 35:59-66, doi:10.1147/rd.351.0059](https://doi.org/10.1147/rd.351.0059)
- Peng & Laramee's "Higher Dimensional Vector Field Visualization: A Survey" (2009) covers only 2.5D and 3D spatial domains: "only 2.5D and 3D are taken into account". In this survey, 4D means 3D plus time. — [TPCG 2009, doi:10.2312/LocalChapterEvents/TPCG/TPCG09/149-163](https://doi.org/10.2312/LocalChapterEvents/TPCG/TPCG09/149-163)
- 3D vector-field topology tool: Globus, Levit & Lasinski 1991. — [Proc. Visualization '91, doi:10.1109/VISUAL.1991.175773](https://doi.org/10.1109/VISUAL.1991.175773)
- Phase portraits of high-dimensional dynamical systems: Phase Space Projection optimizes a 3D projection that preserves critical points and flow maps, and outperforms PCA, t-SNE and UMAP views of the same trajectories. — [Bartolovic et al. 2020](https://doi.org/10.1111/cgf.13978)

### Inferences
(Linear-algebra consequences for our 192D "traps". These are my derivations from the numerical-range and log-norm facts above, not claims from a cited paper.)
- Let the displayed slice Jacobian be the compression J₃ = P J Pᵀ (P with orthonormal rows spanning P3). Its eigenvalues lie in W(J₃) ⊆ W(J). The real part of W(J) is bounded by [λmin, λmax] of S = (J + Jᵀ)/2. So:
  - If λmax(S) < 0 (a strictly contracting trap in 192D), every 3D slice shows a sink (all eigenvalues with negative real part). Full-space traps of this kind are slice-robust.
  - A slice saddle, source or vortex does not imply a 192D saddle, source or vortex. Equally, a 192D saddle can appear as a sink in a slice that misses its unstable direction. This explains our slice-dependent labels.
  - A slice vortex (complex pair in J₃) requires a nonzero compressed antisymmetric part P Ω Pᵀ, where Ω = (J − Jᵀ)/2. If J is symmetric (gradient-like), no slice can show a vortex. Rotational content should therefore be measured in 192D, e.g., ‖Ω‖_F/‖S‖_F or the imaginary parts of the eigenvalues of the full J.
- Projection-invariant, or at least basis-invariant, descriptors to compute in 192D and display as scalar colours or badges:
  - Tr J (divergence) and the eigenvalue spectrum of J (spectral abscissa; number of eigenvalues with positive real part, i.e., the saddle index).
  - λmax(S) (logarithmic norm, a contraction certificate) and ‖S‖, ‖Ω‖.
  - The generalized Q = ½(‖Ω‖² − ‖S‖²).
  - These are invariant under orthogonal change of the 192D basis, which matters because the residual stream has no privileged basis (Q5). They are not invariant under projection.
- Slice zeros. P f(x) = 0 on a 3D slice imposes 3 of 192 conditions. A slice zero is a 192D equilibrium only if the remaining 189 components also vanish, which generically does not happen. This is why our slice zero is not a 192D equilibrium. Remedy, in the spirit of Sussillo & Barak / FixedPointFinder:
  - Minimize q(x) = ½‖f(x)‖² in 192D from many starts (e.g., token positions and slice zeros).
  - Report true fixed points and slow points, with q and the eigenvalues of J.
  - Project them into P3 for display, labelling slice zeros as "pseudo-equilibria" with the residual ‖(I−P)f‖ shown. (The q-function formulation is from my background knowledge of Sussillo & Barak; their full text was not read here.)
- Chong-Perry-Cantwell invariants are mathematically valid on the 3x3 J₃, since they hold for any smooth 3D field. Reporting them is legitimate only if labelled "slice-field topology". They should be paired with the full-space descriptors above.

### Gaps
- No paper was found that formally relates the topology (critical-point types, separatrices) of a linearly compressed or sliced field to the topology of the parent field in dimension greater than 3. The interlacing and numerical-range argument above is my own synthesis.
- The full texts of Helman & Hesselink 1989/1991 and of Sussillo & Barak 2013 were not accessible. Their content above relies on abstracts, DOIs and secondary summaries.
- Laramee et al.'s topology-based state-of-the-art report (2007) and McLoughlin et al. 2010, "Over two decades of integration-based, geometric flow visualization" (Comput Graph Forum 29(6)), were not retrieved, so their DOIs and content are not verified here. The dense/texture STAR is verified: Laramee RS, Hauser H, Doleisch H, Vrolijk B, Post FH, Weiskopf D, Comput Graph Forum 23(2) 2004, [doi:10.1111/j.1467-8659.2004.00753.x](https://doi.org/10.1111/j.1467-8659.2004.00753.x).
- Streamlines on manifolds and Perrault-Joncas & Meilă 2014 (estimating vector fields on manifolds; [arXiv](https://arxiv.org/abs/1406.0013), listed by Undermind) were not read.

## Q4. Glyph and uncertainty design for projected vectors (encoding retained norm, out-of-view component, uncertainty, depth)

### Takeaway
The vector-visualization literature offers mature glyph designs for magnitude and direction uncertainty in 2D and 3D, and perceptual and illustrative techniques for depth and occlusion in 3D. Distortion glyphs exist for nonlinear embeddings. No source was found that proposes a standard glyph for "fraction of norm retained under projection". Encoding it is a design extrapolation from uncertainty and focus+context practice.

### Cited Findings
- Wittenbrink, Pang & Lodha 1996 developed uncertainty glyphs that show "uncertainty in direction and magnitude, as well as the mean direction and length". They coined "verity visualization" for visualizations that "incorporate uncertainty in an unambiguous manner", and evaluated the glyphs with subjective expert tests and objective information-density tests. — [IEEE TVCG 2(3):266-279, doi:10.1109/2945.537309](https://doi.org/10.1109/2945.537309)
- 3D uncertainty glyphs:
  - The "disk-tailed arrow" maps statistical trends and uncertainties of 3D vectors to visual cues. — [Lee & Park, EuroVis Short 2018, doi:10.2312/eurovisshort.20181069](https://doi.org/10.2312/eurovisshort.20181069)
  - Vector uncertainty glyphs are "often limited to 2D"; Ouermi et al. present 3D vector uncertainty glyphs and a tool built around them. — [Ouermi et al., IEEE UncertaintyVis 2024, doi:10.1109/UncertaintyVisualization63963.2024.00014](https://doi.org/10.1109/UncertaintyVisualization63963.2024.00014)
- User study (Laidlaw et al. 2005, six 2D methods). Users performed better with methods that "1) showed the sign of vectors within the vector field, 2) visually represented integral curves, and 3) visually represented the locations of critical points". Expert and non-expert performance did not differ significantly. — [IEEE TVCG 11(1):59-70, doi:10.1109/TVCG.2005.4](https://doi.org/10.1109/TVCG.2005.4)
- Illustrative flow visualization (Brambilla et al. 2012, Eurographics STAR):
  - Illuminated streamlines (Phong-lit lines) improve depth perception, and depth-dependent halos convey depth in dense line data.
  - "Cut-aways and clip planes effectively improve visibility and reduce occlusions".
  - Interactive navigation is "one of the most effective ways to perceive the location and the shape of an object".
  - Importance can be mapped to opacity and saturation, with focus in warm opaque colours and context transparent and desaturated.
  - "There is no approach that is universally better".
  — [doi:10.2312/conf/EG2012/stars/075-094](https://doi.org/10.2312/conf/EG2012/stars/075-094) (full text read via Undermind)
- Direct glyph methods "generally suffer from visual complexity and occlusion" in 3D. — [Peng & Laramee 2009](https://doi.org/10.2312/LocalChapterEvents/TPCG/TPCG09/149-163)
- Distortion encodings for projections:
  - Hypertrix: distortion ellipses for any-dimensional-to-2D maps. — [Raval et al. 2024](https://doi.org/10.1109/VIS55277.2024.00073)
  - Bian et al. 2020: glyphs for transformed local subspaces. — [doi:10.1109/TVCG.2020.3030368](https://doi.org/10.1109/TVCG.2020.3030368)
  - LOO-map point-wise perturbation and singularity scores, used to flag unreliable embedded points. — [Liu et al.](https://doi.org/10.1038/s41467-025-60434-9)
  - velotest: per-cell significance overlays showing which displayed velocities are supported. — [Bischoff et al. 2025](https://doi.org/10.1101/2025.10.26.683064)
- RNA velocity work recognizes that uncertainty is not propagated into the projection, which motivates showing it. — [Bergen et al. 2021](https://europepmc.org/article/PMC/PMC8388041)

### Inferences
(Design recommendations for our app, extrapolated from the cited practice.)
- Per arrow (beam u_c and net f), compute and encode in 192D:
  - The retained fraction ρ = ‖P u‖/‖u‖. Map it to opacity or saturation (focus+context style), so arrows that are mostly out of view fade.
  - The out-of-view magnitude ‖(I−P)u‖. Show it as a "disk tail" or ring at the arrow base (analogous to Lee & Park's disk-tailed arrow), or as a short orthogonal stub, or as a faint "ghost" arrow of full length ‖u‖ along the projected direction.
  - A global per-token badge: ρ_f for the net force and the energy fraction Σ‖P u_c‖²/Σ‖u_c‖².
- Pairwise "opposition" between beams should be coloured by the 192D cosine, not the 3D angle. That removes the chance-level (about 15%) false "opposite" impressions derived in Q2.
- Field glyphs:
  - Show critical-point markers (Laidlaw et al.) with a two-level label: slice-field type and full-space type.
  - Use contrasting styles where they disagree (e.g., hollow vs filled marker).
  - Show the 192D contraction certificate λmax(S) as colour.
- Depth: use lit or haloed arrows, clip planes and interactive rotation (Brambilla et al.). Avoid encoding magnitude by length alone in 3D, where foreshortening confounds it. Add a redundant channel (width or colour). This is a general perception inference, not a cited quantitative result.

### Gaps
- No source quantifies how viewers misjudge projected 3D arrow lengths or angles under foreshortening. Forsberg et al. 2009 ("Comparing 3D vector field visualization methods: a user study", IEEE TVCG) is relevant but was not retrieved.
- No source found for a standard encoding of "fraction of norm in view" for linear projections. This appears to be an open design niche.
- McLoughlin et al. 2010 and the Laramee et al. topology STAR (2007) were not retrieved (see Q3 Gaps).

## Q5. ML interpretability analogues: how per-neuron / per-head contributions to the residual stream are visualized, and whether projection artifacts are acknowledged

### Takeaway
Mechanistic interpretability avoids generic 2D/3D projections of residual-stream contribution vectors. It projects contributions onto task-defined directions (unembedding / vocabulary, logit-difference directions), where linear decomposition is exact and additive. When geometry is shown, it uses basis-invariant summaries (W^T W Gram matrices, feature dimensionality) or natively 2D toy models. PCA plots are used as one line of evidence and corroborated with probes and causal interventions. Explicit warnings concern the lack of a privileged basis, superposition and representational drift across layers, rather than a specific "arrow projection" artifact.

### Cited Findings
- Elhage et al. 2021 (A Mathematical Framework for Transformer Circuits):
  - The residual stream's "linear, additive structure" means "the residual stream doesn't have a 'privileged basis'; we could rotate it by rotating all the matrices interacting with it, without changing model behavior".
  - Components "communicate with each other by reading and writing to different subspaces of the residual stream". Rather than analyze residual vectors directly, "it can be helpful to decompose the residual stream into all these different communication channels".
  - The residual stream "may be hundreds of dimensions" in small models. MLP activations are "bottleneck activations" communicating "in superposition", which they "expect ... to be unusually challenging to interpret".
  — [Transformer Circuits Thread 2021](https://transformer-circuits.pub/2021/framework/index.html)
- Elhage et al. 2022 (Toy Models of Superposition):
  - "The main question is how to visualize the results. The simplest way is to visualize W^TW (a features by features matrix) and b".
  - They define feature dimensionality by "projecting all other features onto the direction vector of W_i" (D_i = ‖W_i‖²/Σ_j(Ŵ_i·W_j)²).
  - Arrow plots of feature directions are used where the hidden space is natively 2D (e.g., 5 features in 2D, or 2 features in 1D). In 1D, "antipodal" pairs are mapped to opposite directions.
  - Geometry in higher dimensions is described via polytopes and tegum products, not projections. Example: "we observe 3D Thomson problem solutions, despite the fact that we're actually studying a higher dimensional version of the problem".
  — [Transformer Circuits Thread 2022](https://transformer-circuits.pub/2022/toy_model/index.html); [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
- Geva et al. 2022 view "the output from each FFN layer as an additive update" to the vocabulary distribution. Each update decomposes into "sub-updates corresponding to single FFN parameter vectors", i.e., second-layer value vectors scaled by activations, which are interpreted by projection to vocabulary space. — [EMNLP 2022, arXiv:2203.14680](https://arxiv.org/abs/2203.14680)
- Tuned lens (Belrose et al. 2023):
  - The logit lens, which decodes intermediate residuals with the final unembedding, is "unreliable", failing on BLOOM and GPT-Neo.
  - "Even when the logit lens appears to work, its outputs are hard to interpret due to representational drift: features may be represented differently at different layers".
  - The fix is a learned affine probe per layer.
  — [arXiv:2303.08112](https://arxiv.org/abs/2303.08112) (HTML full text grep)
- Marks & Tegmark (COLM 2024):
  - True and false statements separate "in the top two principal components" of LLaMA-2 residual activations, using datasets curated "to have little variation with respect to non-truth features".
  - In some datasets "the top principal components are dominated by features representing the sizes of numbers".
  - They use three lines of evidence: visualization, probe transfer, and causal interventions.
  — [arXiv:2310.06824](https://arxiv.org/abs/2310.06824) (HTML full text grep)
- Hypertrix comes from interpretability-adjacent visualization researchers (Wattenberg, Viégas) and explicitly targets distortion in high-dimensional-to-2D maps. — [Raval et al. 2024](https://doi.org/10.1109/VIS55277.2024.00073)

### Inferences
- Our 768 beams are exactly Geva et al.'s FFN sub-updates (activation × fc2 column) in a 192D residual stream. The interpretability practice suggests two things:
  - Project contributions onto meaningful 1D directions, where the decomposition is exact and lossless along that axis: the token's net force f̂, the displacement to the next layer, a class or logit direction, or a trap's unstable eigenvector.
  - Use 3D views only as navigational context.
- Because the residual stream has no privileged basis (Elhage et al. 2021), any fixed "top-3 PCA of fc2 columns" is one arbitrary choice among rotations. Basis-invariant summaries are the defensible quantitative layer under the 3D picture: Gram matrices of the u_c, 192D cosines, and J's spectrum.
- Representational drift (Belrose et al.) implies that one fixed P3 across all layers may be misaligned for some layers. Per-layer views, or reporting ρ per layer, would expose this.
- The Marks & Tegmark "number size dominates top PCs" example is a concrete precedent: a variance-selected plane can be captured by a nuisance direction. This is relevant because P3 is chosen by variance of fc2 columns, not by the forces actually exerted on tokens.

### Gaps
- nostalgebraist 2020 ("interpreting GPT: the logit lens", LessWrong) and the direct-logit-attribution literature were not fetched in this session. Their claims are summarized only through Belrose et al.
- No interpretability paper was found that draws per-neuron contribution arrows in a 3D PCA of the residual stream and quantifies the retained-norm fraction. This appears to be unexplored.
- Not verified: whether Toy Models contains an explicit sentence warning against projecting features with dimension above 3 into 2D. Its practice (W^T W, feature dimensionality) implies avoidance, but no direct quote was found.

## Q6. Physics analogues: drawing many force contributions (force chains, contact-force networks, many-body decompositions)

### Takeaway
Granular physics measures individual contact forces (photoelasticity) and displays force networks in the native 2D geometry. Relevance is shown by partitioning forces (strong vs weak networks by a mean-force threshold) rather than by projecting. The threshold-based "strong network" is itself debated. This suggests showing only the beams that carry most of the net force, with thresholds stated, rather than all 768.

### Cited Findings
- Majmudar & Behringer 2005 measured contact forces between photoelastic disks under isotropic compression and shear, inferring normal and tangential contact forces from polarized-light patterns. Behringer described it as the first study to determine forces at the contacts of large numbers of particles (about 2,500 disks, per secondary coverage). Citation: Majmudar TS, Behringer RP. Contact force measurements and stress-induced anisotropy in granular materials. Nature 435:1079-1082 (2005). — [doi:10.1038/nature03805](https://doi.org/10.1038/nature03805); description: [Soft Bites summary](https://softbites.org/2018/07/19/seeing-inside-sand-visualizing-force-chains-with-photoelastic-disks/?amp)
- Radjai et al. 1998 decompose the contact network "unambiguously into two subnetworks with complementary mechanical properties". Contacts with above-mean normal forces form a "strong" network carrying most of the load, and the "weak" network stabilizes it. — [HAL record, Phys Rev Lett 80:61 (1998)](https://hal.archives-ouvertes.fr/hal-01407369)
- Later sheared granular work: the shear stress is almost totally sustained by the strong network. — [arXiv:physics/0702055](https://arxiv.org/pdf/physics/0702055); [arXiv:0801.4722](https://arxiv.org/pdf/0801.4722)
- Caveat: Peters et al. showed that only about half of the "strong" contacts belong to force-transmission chains, so the mean-force criterion "is not universally accepted". — [Comptes Rendus Physique review](https://comptes-rendus.academie-sciences.fr/physique/item/10.1016/j.crhy.2015.01.003.pdf)

### Inferences
- The granular strong/weak decomposition carries over to our beams. Rank the u_c by their 192D projection onto f̂ (signed contribution to the net force) and draw the strong set: beams that together account for, say, 80% of ‖f‖. Draw the weak or cancelling remainder aggregated, e.g., as a single residual arrow Σ_weak u_c. The display then stays exact (sums preserved) and becomes legible. As the Peters caveat shows, any threshold should be reported, because strong-set membership depends on it.
- In granular media forces are drawn in the native 2D or 3D space, so no projection loss occurs. Our case adds projection loss, so the strong-set selection should be done in 192D, before projecting.

### Gaps
- The display conventions of Majmudar & Behringer (e.g., line width proportional to force) were not verified from the primary source in this session.
- Many-body force decomposition displays in molecular dynamics, such as Force Distribution Analysis (Stacklies, Seifert & Gräter, BMC Bioinformatics 2011), were not retrieved or verified.
- Peters et al. 2005 (Phys Rev E 72:041307, force-chain characterization) was cited only via a secondary source.
