# Nonlinear dimensionality-reduction embeddings (2D/3D): what they preserve, how they distort, and how projection fidelity is measured and visualized

Research notes for the ViT "colloids and lasers" app (600 tokens x 192-D residual stream, 768 MLP force contributions, 192x192 Jacobian traps, 26 checkpoints, current view = fixed top-3 PCA of fc2 columns).

Provenance conventions used below:
- **[full text / page]** = claim checked against the fetched paper or page in this session.
- **[abstract]** = claim checked only against the abstract or metadata (Undermind records).
- **[not re-verified]** = standard, well-known result from the cited seminal paper. I did not fetch it in this session. The citation and DOI are given so the report writer can check them.
- Scite was **unavailable**: every Scite call returned "monthly MCP usage limit (250 calls) reached; resets 2026-11-13". There are no Scite citation statements here. Evidence comes from Undermind (2 deep searches + paper search), WebFetch and WebSearch.
- Undermind deep searches (workspace "ViT: detekcja końcówki narzędzia w chirurgii zaćmy"): "Nonlinear embeddings t-SNE UMAP distortions, vector fields and temporal consistency" (195 relevant papers) and "Quality measures and distortion visualization for dimensionality reduction projections" (227 relevant papers). Links: https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Nonlinear%20embeddings%20t-SNE%20UMAP%20distortions%2C%20vector%20fields%20and%20temporal%20consistency and https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Quality%20measures%20and%20distortion%20visualization%20for%20dimensionality%20reduction%20projections

---

## Q1. Nonlinear embedding methods: what each preserves (local vs global), and their documented distortions

### Takeaway
Neighbour-embedding methods (t-SNE, UMAP, and the related PaCMAP/TriMap) reliably preserve one thing: the kNN graph at a scale set by perplexity or n_neighbors. Cluster sizes, inter-cluster distances, densities and global arrangement are not reliably preserved, and the global arrangement depends mostly on initialization. Spectral and MDS-type methods target distances or geodesic/diffusion distances instead. **No** nonlinear method preserves vector sums. None preserves vector-field topology unless it was built for that purpose.

### Cited Findings

**Critiques of t-SNE/UMAP (established, heavily cited)**
- Wattenberg, Viégas & Johnson (2016), Distill, "How to Use t-SNE Effectively" **[full text]**. Findings:
  - Perplexity changes the picture a lot. In a two-cluster example, perplexity 2 showed local noise and perplexity 100 merged the clusters.
  - Plots taken at 10–120 steps showed "pinched" shapes that were not yet converged. Their examples used 5,000 iterations.
  - **Cluster sizes are not meaningful.** A Gaussian cluster with 10x the spread of another looked about the same size, because t-SNE expands dense regions and contracts sparse ones.
  - **Distances between clusters can mislead.** With 200 points per cluster, no perplexity they tried recovered the global geometry.
  - Pure noise (500 points, 100-D Gaussian) produced spurious clumps at perplexity 2.
  - For trefoil knots at perplexity 2, 3 of 5 runs produced artificial breaks.
  - Topology sometimes needs several perplexities to read. — [Distill 2016](https://doi.org/10.23915/distill.00002)
- Kobak & Berens (2019), "The art of using t-SNE for single-cell transcriptomics", *Nat Commun* 10:5416 **[full text]**:
  - t-SNE "excels at revealing local structure" but often fails to preserve global geometry. Relative cluster positions are nearly arbitrary and depend mainly on the random initialization.
  - Recipe: PCA initialization (PC1–2 divided by std(PC1), times 1e-4); multi-scale perplexity (30 combined with n/100); learning rate n/12; exaggeration of about 4 for very large n.
  - Metrics: KNN preservation (k=10), KNC (k-nearest class means preserved) and CPD (Spearman correlation of pairwise distances, over 1,000 random points = 499,500 pairs).
  - Synthetic-data result: t-SNE vs PCA scored KNN 0.13 vs 0.00 but KNC 0.23 vs 1.00 and CPD 0.51 vs 0.85. The trade-off is local vs global. — [Kobak & Berens 2019](https://doi.org/10.1038/s41467-019-13056-x)
- Kobak & Linderman (2021), *Nat Biotechnol* 39:156–157 **[abstract/first page]**. Becht et al. had claimed UMAP preserves global structure better than t-SNE. Kobak & Linderman show this is "entirely" due to initialization: UMAP used Laplacian-eigenmap init, t-SNE used random init. Toy example: a noisy circle with n=7,000 points. Random init gave a knot for t-SNE and a torn, tangled web for UMAP. With informative init, both are comparable. They argue informative initialization should be the default. — [Kobak & Linderman 2021](https://doi.org/10.1038/s41587-020-00809-z)
- Böhm, Berens & Kobak (2022), "Attraction-Repulsion Spectrum in Neighbor Embeddings", *JMLR* 23(95):1–32 **[abstract page]**:
  - Varying the exaggeration (the attraction/repulsion balance) in t-SNE gives a continuum of embeddings. More attraction represents continuous manifolds better; more repulsion represents discrete clusters better and gives higher kNN recall.
  - UMAP ≈ t-SNE with increased attraction, because negative sampling lowers effective repulsion. ForceAtlas2 has even more attraction. Laplacian eigenmaps sit at the extreme. — [arXiv 2007.08902 / JMLR](https://arxiv.org/abs/2007.08902)
- Damrich, Böhm, Hamprecht & Kobak (2023, ICLR), "From t-SNE to UMAP with contrastive learning" **[abstract]**. Characterizes mathematically the distortion that negative sampling introduces: UMAP gives "more compact embeddings with tighter clusters". It also interpolates between t-SNE and UMAP, which trades discrete/local structure against continuous/global structure. — [arXiv 2206.01816](https://arxiv.org/abs/2206.01816)
- Damrich & Hamprecht (2021, NeurIPS), "On UMAP's true loss function". The Undermind deep-search summary cites it for the point that negative sampling changes the effective objective **[metadata only]**. — [NeurIPS 2021 / arXiv 2103.14608](https://arxiv.org/abs/2103.14608)
- Chari & Pachter (2023), "The specious art of single-cell genomics", *PLoS Comput Biol* 19(8):e1011288; correction 2025, 21(11):e1013655 **[full text]**:
  - **Local:** average Jaccard distance between 30-NN sets in 2D and in the ambient space stayed above 0.7 for t-SNE/UMAP. Against the PCA space it was above 0.8. Distortion grew with dataset size.
  - **Global:** correlations of cell-type neighbour rankings were ≤0.4, at least 33% lower than in higher-dimensional PCA spaces.
  - **Distances:** the max/min distance ratio grew 4–200x in 2D. By Johnson–Lindenstrauss, keeping pairwise distances within 20% for 10,000 points needs at least 1,842 dimensions.
  - Neighbour and perplexity settings changed RNA-velocity arrows and density contours. Some populations appeared or disappeared.
  - A "Picasso" autoencoder forced the data into an elephant shape and matched the distance-preservation scores of t-SNE/UMAP.
  - Recommendation: treat 2D embeddings as "cartoon diagrams", not exploratory tools. — [Chari & Pachter 2023](https://doi.org/10.1371/journal.pcbi.1011288)
- **Contradicted in part by** Lause, Berens & Kobak (2024), *PLoS Comput Biol* 20(10):e1012403 **[full text]**. They argue Chari & Pachter used only distance-correlation metrics. On neighbourhood and class metrics:
  - kNN accuracy: t-SNE/UMAP >90%, PCA and the elephant <62%.
  - kNN recall: t-SNE/UMAP >15%, PCA and elephant <5%; below 40% for every method.
  - Silhouette is >0.3 higher and AMI >0.25 higher for t-SNE/UMAP.
  - Both sides agree that 2D embeddings do not preserve distances and must not be used for quantitative downstream analysis. — [Lause et al. 2024](https://doi.org/10.1371/journal.pcbi.1012403)
- Bergam, Snoeck & Verma (2025, arXiv preprint, not peer-reviewed), "t-SNE Exaggerates Clusters, Provably" **[abstract]**. They prove that neither the strength of input clustering nor the extremity of outliers can be reliably inferred from t-SNE output. — [arXiv 2510.07746](https://doi.org/10.48550/arXiv.2510.07746)
- Liu, Ma & Zhong (2025), *Nat Commun*, "Assessing and improving reliability of neighbor embedding methods: a map-continuity perspective" **[abstract]**. A leave-one-out "LOO-map" extends t-SNE/UMAP to the whole input space and exposes two map discontinuities: one exaggerates cluster separation, the other creates spurious local structure. They propose point-wise diagnostic scores. — [Liu et al. 2025](https://doi.org/10.1038/s41467-025-60434-9)
- Narayan, Berger & Cho (2021), *Nat Biotechnol* 39:765–774 **[abstract]**. t-SNE/UMAP ignore local density, so dense subsets get more visual space than their variability warrants. den-SNE and densMAP add density preservation. — [Narayan et al. 2021](https://doi.org/10.1038/s41587-020-00801-7)
- Jeon et al. (2025, arXiv preprint), "Stop Misusing t-SNE and UMAP for Visual Analytics" **[abstract]**. A review of 136 papers shows widespread use of t-SNE/UMAP to read inter-cluster relationships, even though these projections "often do not faithfully reflect the original distances between clusters". They attribute the misuse to limited DR literacy. — [arXiv 2506.08725](https://doi.org/10.48550/arXiv.2506.08725)
- Xia, Lee & Li (2024), *Nat Commun*, scDEED **[abstract]**. Per-point reliability score from the agreement between pre-embedding and 2D neighbours. It flags "dubious" embeddings and tunes perplexity / n_neighbors. — [Xia et al. 2024](https://doi.org/10.1038/s41467-024-45891-y)

**Newer methods claiming better global structure**
- Wang, Huang, Rudin & Shaposhnik (2021), PaCMAP, *JMLR* 22(201):1–73 **[abstract page]**. t-SNE, UMAP and TriMap preserve local or global structure but not both. The authors derive design principles for DR losses and propose PaCMAP to preserve both. The specific principles (neighbour, mid-near and further pairs; dynamic weights) were **not** verified on the fetched page. — [PaCMAP JMLR](https://jmlr.org/papers/v22/20-1061.html)
- Huang, Wang, Rudin & Browne (2022), *Commun Biol* **[abstract]**. Systematic evaluation of PCA, t-SNE, art-SNE, UMAP, PaCMAP, TriMap, ForceAtlas2 and PHATE on local structure, global structure, sensitivity to parameters, sensitivity to preprocessing, and speed. It states these methods "do not preserve important aspects of high-dimensional structure and are sensitive to arbitrary user choices". — [Huang et al. 2022](https://doi.org/10.1038/s42003-022-03628-x)
- de Bodt, …, Kobak (2025, arXiv review, ~20 authors), "Low-dimensional embeddings of high-dimensional data" **[full text, partial]**:
  - "Distances in PCA and MDS are meaningful; in neighbour embeddings they are not interpretable."
  - Across 4 benchmark datasets, t-SNE had the highest local quality (10-NN preserved). PCA and MDS had the highest global quality (correlation of pairwise distances). On organoid data, PHATE and Laplacian eigenmaps also beat t-SNE/UMAP on global quality.
  - Horseshoe shapes from spectral methods and MDS may be artefacts.
  - Best practices include: "do not run downstream analysis on 2D embeddings"; remove axis ticks where distances are meaningless; report explained variance or preserved neighbours; provide alternative embeddings. — [arXiv 2508.15929](https://doi.org/10.48550/arXiv.2508.15929)

**Seminal method papers (what each optimizes) [not re-verified]**
- Classical (Torgerson) MDS = eigendecomposition of double-centred squared distances. On Euclidean input it is equivalent to PCA. Metric/non-metric MDS minimizes Kruskal stress; non-metric MDS fits only the rank order of dissimilarities — [Kruskal 1964, Psychometrika 29:1–27](https://doi.org/10.1007/BF02289565)
- Sammon mapping: stress weighted by 1/d_ij, so small distances are emphasized relative to metric MDS — [Sammon 1969, IEEE Trans Comput C-18:401–409](https://doi.org/10.1109/T-C.1969.222678)
- Isomap: classical MDS on kNN-graph geodesic distances. It preserves geodesic distances on a manifold that is isometric to a convex subset of R^d, and is sensitive to graph short-circuits — [Tenenbaum, de Silva & Langford 2000, Science 290:2319](https://doi.org/10.1126/science.290.5500.2319)
- LLE: preserves local linear reconstruction weights, i.e. local affine structure within neighbourhoods, with no global metric — [Roweis & Saul 2000, Science 290:2323](https://doi.org/10.1126/science.290.5500.2323)
- Laplacian eigenmaps: locality-preserving spectral embedding (neighbours map close together) — [Belkin & Niyogi 2003, Neural Comput 15:1373](https://doi.org/10.1162/089976603321780317)
- Diffusion maps: Euclidean distance in diffusion coordinates approximates diffusion distance at time t (exactly, if all eigenvectors are kept) — [Coifman & Lafon 2006, ACHA 21:5–30](https://doi.org/10.1016/j.acha.2006.04.006)
- t-SNE — [van der Maaten & Hinton 2008, JMLR 9:2579](https://jmlr.org/papers/v9/vandermaaten08a.html); UMAP — [McInnes, Healy & Melville 2018, arXiv 1802.03426](https://arxiv.org/abs/1802.03426); PHATE (potential of heat-diffusion, designed for continual progressions and trajectories) — [Moon et al. 2019, Nat Biotechnol 37:1482](https://doi.org/10.1038/s41587-019-0336-3); TriMap (triplet constraints for global structure) — [Amid & Warmuth 2019, arXiv 1910.00204](https://arxiv.org/abs/1910.00204)
- Textbook reference for the preservation taxonomy (distance preservation vs topology/neighbourhood preservation) — [Lee & Verleysen 2007, *Nonlinear Dimensionality Reduction*, Springer](https://doi.org/10.1007/978-0-387-39351-3)

### Inferences
The method/property table below is a synthesis. Cells rest on the findings above plus elementary linear algebra. Columns: (a) linear structure / vector sums; (b) distances; (c) neighbourhoods; (d) dynamics / field topology.

| Method | (a) linear structure / vector sums | (b) distances | (c) neighbourhoods | (d) dynamics / field topology |
|---|---|---|---|---|
| PCA / any fixed linear projection P (e.g. current fc2-PCA basis) | **Yes, exactly**: P(Σf_k)=ΣPf_k, parallelism and ratios along lines kept | Only within the retained subspace. Orthogonal projection never stretches distances (‖P(x−y)‖≤‖x−y‖), so the failure mode is *false neighbours/overlap*, not tears | Partially (low kNN recall when variance is spread: Kobak & Berens KNN 0.00 synthetic) | Force arrows map exactly (Pf). Fixed-point type is **not** preserved unless span(Pᵀ) is J-invariant. The 3D field is not a function of the 3D position |
| Classical MDS | = PCA on Euclidean data | Global (large distances dominate) | Weak | No |
| Metric / non-metric MDS, Sammon | No | Metric MDS: distances; non-metric: rank order; Sammon: favours small distances | Moderate | No |
| Isomap | No | Geodesic distances (if the manifold assumptions hold) | Moderate | No |
| LLE | Only *locally* (affine weights) | No | Yes, locally | No |
| Laplacian eigenmaps / diffusion maps | No | Diffusion distance (DM, many eigenvectors); not Euclidean | Yes (locality) | No; DM relates to diffusion operators, not the given force field |
| t-SNE | No | **No**; cluster sizes and gaps meaningless | **Yes** (best kNN recall) | No |
| UMAP | No | **No** | Yes (slightly lower recall, tighter clusters) | No |
| PaCMAP / TriMap | No | Coarse global arrangement better (claimed); not metric | Yes | No |
| PHATE / M-PHATE | No | Potential distance; aimed at trajectories | Yes | Captures progression, not the force field |
| Parametric t-SNE/UMAP, autoencoders | No globally; Jacobian gives local linearization | Depends on loss | Depends on loss | Vectors can be pushed forward with J_φ (Q2); topology not guaranteed |

- For this system, the 3.15-D participation ratio at the end of training suggests a linear 3D view may capture most token variance late in training, but much less at epoch 0 (PR 10.6). The fidelity of the fixed fc2-PCA view is likely to change strongly across the 26 checkpoints. It should be measured per checkpoint (Q4), not assumed.
- The tool vs background split is a 2-cluster structure. t-SNE/UMAP would probably make it look sharper (repulsion, cluster exaggeration). They cannot be trusted for the *gap size* or the *relative spreads* of the two groups (Wattenberg; Bergam; Narayan).

### Gaps
- PaCMAP's design principles and its quantitative global-structure scores were not verified (only the abstract page was fetched).
- I found no study that evaluates any of these methods on transformer token clouds over training, beyond the NN-training visualizations listed in Q3.
- Seminal-method characterizations are standard results. They were not re-fetched in this session.

---

## Q2. Can nonlinear embeddings carry vectors (forces/velocities)? Pushforwards, Jacobians, out-of-sample extensions, and what is lost

### Takeaway
A vector f at x has a well-defined image J_φ(x)·f only if there is a fixed, differentiable map φ. Parametric UMAP/t-SNE and autoencoders have one. Spectral methods have one through the Nyström/eigenfunction extension. Plain t-SNE/UMAP produce only point coordinates, so arrows must be derived: by implicit differentiation, by leave-one-out maps, or (as in RNA velocity) by neighbour-transition interpolation. Interpolation is known to reflect kNN-graph smoothing rather than true displacement, and it gets speed especially wrong.

### Cited Findings
- **RNA velocity as a cautionary case.** Zheng, Stein-O'Brien, Boukas, Goff & Hansen (2023), *Genome Biol* **[abstract]**:
  - The transition-probability method that maps velocities onto an embedding "is effectively interpolating in the embedding space".
  - Results depend strongly on kNN-graph smoothing, which gives large direction and speed errors when the kNN graph misrepresents the data. Speed is poorly estimated except at very low noise.
  - Using velocity to check the correctness of an embedding "is circular". — [Zheng et al. 2023](https://doi.org/10.1186/s13059-023-03065-x)
- Chari & Pachter (2023): changing the number of neighbours changed the RNA-velocity arrows drawn on UMAP **[full text]** — [Chari & Pachter 2023](https://doi.org/10.1371/journal.pcbi.1011288)
- Bischoff, Poličar, …, Schröder (2025, bioRxiv preprint), "velotest" **[abstract]**. A cell-level statistical test of whether a 2D velocity embedding is faithful to its high-D counterpart. It finds "strong differences in embedding quality between methods, datasets and regions". — [Bischoff et al. 2025](https://doi.org/10.1101/2025.10.26.683064)
- Further critical reviews of velocity projection (listed by the deep search, not read): [Bergen et al. 2021, Mol Syst Biol](https://doi.org/10.15252/msb.202110282); [Gorin, Fang & Pachter 2022, "RNA velocity unraveled", PLoS Comput Biol](https://doi.org/10.1371/journal.pcbi.1010492); [Marot-Lassauzaie et al. 2022, "Towards reliable quantification of cell state velocities", PLoS Comput Biol](https://doi.org/10.1371/journal.pcbi.1010031) **[metadata only; the last DOI was not verified]**.
- **Implicit-function Jacobians of nonparametric embeddings.**
  - Zabel, Hennig & Nieselt (2026), *Front Bioinform* **[abstract]**: combine the implicit function theorem with autodiff to get the Jacobian of t-SNE output with respect to its input. They use it for feature-sensitivity heatmaps and for propagating input uncertainty into per-point positional uncertainty — [Zabel et al. 2026](https://doi.org/10.3389/fbinf.2025.1719516)
  - Bian et al. (2021), *IEEE TVCG* 27(2) **[abstract]**: "implicit multidimensional projection of local subspaces". They analytically differentiate projections written as implicit functions and transform local basis vectors (ellipse glyphs) into the 2D view. This is a vector-transformation (pushforward) method for projections — [Bian et al. 2021](https://doi.org/10.1109/TVCG.2020.3030368)
  - Faust, Glickenstein & Scheidegger (2019), DimReader, *IEEE TVCG* 25(1) **[abstract]**: infinitesimal input perturbations, i.e. directional derivatives via forward-mode autodiff, yield "axis lines" for t-SNE/LLE. A perturbation direction in input space is a vector, so this is a pushforward of chosen directions — [Faust et al. 2019](https://doi.org/10.1109/tvcg.2018.2865194)
  - Liu, Ma & Zhong (2025): the LOO-map defines an embedding map over the whole input space. It reveals discontinuities, and where a map is discontinuous a Jacobian pushforward is meaningless **[abstract]** — [Liu et al. 2025](https://doi.org/10.1038/s41467-025-60434-9)
- **Parametric maps / out-of-sample.**
  - Parametric UMAP learns a neural map with "comparable" quality to non-parametric UMAP and gives fast embedding of new data **[abstract]** — [Sainburg, McInnes & Gentner 2021, Neural Comput 33(11):2881](https://doi.org/10.1162/neco_a_01434)
  - Parametric t-SNE **[not re-verified]** — [van der Maaten 2009, AISTATS, PMLR 5:384](https://proceedings.mlr.press/v5/maaten09a.html)
  - Spectral methods (LLE, Isomap, MDS, Laplacian eigenmaps, spectral clustering) are eigenfunctions of a data-dependent kernel, so a Nyström-type formula extends them to new points. Out-of-sample error is comparable to the variability from the choice of training sample **[search snippet + abstract]** — [Bengio et al. 2004, NIPS 16](https://papers.nips.cc/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html)
  - t-SNE has no mapping function. Kobak & Berens place new points at the median of their 10 nearest reference points, with mean displacement 3.2 ± 2.4 in cross-validation **[full text]** — [Kobak & Berens 2019](https://doi.org/10.1038/s41467-019-13056-x)
  - Parametric projections can be locally unstable. Gaussian perturbations around anchor points produce displacements and neighbourhood deformations "invisible to reconstruction error or neighborhood-preservation metrics". Jacobian regularization is studied as a remedy **[abstract; arXiv preprint]** — [Dennig & Keim 2026](https://doi.org/10.48550/arXiv.2604.21617)
- **Projections designed for flows.** Bartolovic, Gross & Günther (2020), *Comput Graph Forum* **[abstract]**: a DR built for high-dimensional dynamical systems. Energy minimization preserves flow properties. The 3D projection "maintains a number of useful flow properties, such as critical points and flow maps". It was compared against PCA, t-SNE and UMAP. **This is the closest literature analogue to projecting "traps" faithfully.** — [Bartolovic et al. 2020](https://doi.org/10.1111/cgf.13978)
- Cantareira, Etemad & Paulovich (2020), *Information* 11(9):426 **[abstract]**: turn the changes between projections of successive NN layers into trajectories and vector fields to show the "flow of information". The vector field is built *in the projection*, not pushed forward from high-D. — [Cantareira et al. 2020](https://doi.org/10.3390/info11090426)
- Undermind deep-search synthesis (abstract-based, not a primary source): "A vector f at x maps to Jφ(x)f only when a fixed differentiable map φ exists; a nonparametric embedding is not automatically such a map… keep force calculations and geometric claims in the original 192-D space (or a fixed linear projection); use nonlinear maps as exploratory views."

### Inferences
- **What is lost with a nonlinear φ (mathematical points, not from a single paper):**
  1. Additivity across tokens. J_φ(x)·Σ_k f_k = Σ_k J_φ(x)·f_k holds at *one* point, because the linearization is linear. But J_φ varies from token to token, so arrows of different tokens are drawn in different local metrics. Comparing arrow lengths or directions across the map is invalid wherever J_φ varies. With t-SNE that is everywhere: it expands dense regions and contracts sparse ones (Wattenberg).
  2. Finite displacements. φ(x+Δ) − φ(x) ≠ J_φ(x)Δ for large Δ, i.e. large inter-checkpoint movements.
  3. Field topology. Stability type at a fixed point survives a diffeomorphism (conjugacy). A map from 192-D to 3-D is never a diffeomorphism, so attracting/repelling/saddle/vortex labels computed from the 192×192 Jacobian cannot be read off any 3D view. This holds even for a linear P, where P J Pᵀ has the eigen-structure of J only when the subspace is J-invariant.
- **Under the current fixed linear projection P:** force arrows Pf_k are exact projections and add up correctly (ΣPf_k = P·F). The two honest per-token fidelity numbers are the captured force fraction ‖Pf‖/‖f‖ and the captured position variance. For trap faithfulness, a natural diagnostic is the invariance residual ‖(I−PᵀP) J Pᵀ‖ relative to ‖J Pᵀ‖. Comparing the eigenvalues of P J Pᵀ with the leading eigenvalues of J is another check. These are my proposals; I found no paper using them on transformer Jacobians.
- If a nonlinear view is wanted *with* arrows, the defensible route is: one **parametric** map trained once on all 26 checkpoints, with arrows drawn as J_φ(x)f; local stretch shown with ellipse glyphs (Bian 2021, hypertrix in Q5); arrow faithfulness checked with a velotest-style test. Neighbour-interpolated arrows à la RNA velocity should be avoided.

### Gaps
- I found no peer-reviewed paper that defines and validates a Jacobian-pushforward of *force* fields through t-SNE/UMAP for physical interpretation. The nearest are Bian 2021 (subspace vectors), DimReader (perturbation axes), Zabel 2026 (sensitivity) and Bartolovic 2020 (flow-preserving projection).
- Exact formulas and accuracy numbers for Bian 2021 and Bartolovic 2020 were not read (abstracts only).

---

## Q3. Time-dependent embeddings: keeping views consistent across snapshots (training checkpoints)

### Takeaway
Embedding each checkpoint independently with t-SNE/UMAP gives arbitrary jumps between frames. Dynamic/joint methods (dynamic t-SNE, Joint t-SNE, Aligned UMAP, M-PHATE, TimeVis) trade per-frame fidelity for temporal smoothness. In the only broad benchmark (Vernier et al. 2020), **global PCA** (one basis for all timesteps) and **VAEs** gave the best overall balance between spatial quality and stability. Per-timeframe t-SNE was the least stable of all methods. This supports a fixed linear basis for the app.

### Cited Findings
- Rauber, Falcão & Telea (2016), "Visualizing Time-Dependent Data Using Dynamic t-SNE", EuroVis short paper **[abstract + secondary descriptions]**. Independent per-step t-SNE "may introduce unnecessary variability". Dynamic t-SNE optimizes all time steps jointly, giving each point one copy per step. A quadratic penalty with weight λ ≥ 0 on the displacement between consecutive copies sets a "controllable trade-off between temporal coherence and projection reliability" (λ→0: per-frame quality; large λ: smoothness). — [Rauber et al. 2016](https://doi.org/10.2312/eurovisshort.20161164); λ description per [Vernier et al. 2020 (arXiv)](https://arxiv.org/pdf/2002.07481)
- Vernier, Garcia, da Silva, Comba & Telea (2020), *Comput Graph Forum* 39(3):241–252 **[full text via ar5iv]**:
  - **Methods (11 variants):** global and per-timeframe t-SNE, UMAP and PCA; dt-SNE; AE, VAE, CAE, CVAE. 9 are in the main evaluation, across 10 datasets.
  - **Temporal metrics** (the authors say none existed before):
    - per-sample change in data δ = ‖xᵗ − xᵗ⁺¹‖ and change in projection δ̄ = ‖P(xᵗ) − P(xᵗ⁺¹)‖;
    - temporal normalized stress Σ(δ−δ̄)²/Σδ²;
    - temporal Shepard correlations (Pearson, Spearman and Kendall of δ vs δ̄).
  - **Findings (qualitative; no aggregate numbers in the text):**
    - VAE and global PCA score best across spatial quality and stability.
    - Per-timeframe t-SNE has strong neighbourhood metrics but the poorest stability "regardless of dataset".
    - Per-timeframe PCA preserves distances well but is less stable.
    - dt-SNE is spatially about as good as global t-SNE but slightly less stable, and is hyperparameter-sensitive.
    - Visual quality and stability generally compete. — [Vernier et al. 2020](https://doi.org/10.1111/cgf.13977); [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2002.07481)
- Vernier, Comba & Telea (2021), "Guided Stable Dynamic Projections", *CGF* 40(3) **[abstract]**. PCD-tSNE and LD-tSNE steer t-SNE with global guides, avoiding "unstable movement that does not encode data dynamics". PCD-tSNE balances stability, neighbourhood preservation and distance preservation. Compared with 11 other techniques on the 2020 benchmark. — [Vernier et al. 2021](https://doi.org/10.1111/cgf.14291)
- Wang, Chen, Jo & Wang (2022), Joint t-SNE, *IEEE TVCG* **[abstract]**. Uses graphlet-based edge similarity between adjacent frames plus "vector constraints" that preserve displacement vectors between frames. Outperforms Dynamic t-SNE in local coherence error, KL and neighbourhood preservation. Demonstrated on **activations of different layers of a neural network**. — [Wang et al. 2022](https://doi.org/10.1109/TVCG.2021.3114765)
- AlignedUMAP (umap-learn docs) **[page]**:
  - Optimizes several UMAP embeddings jointly, with `relations` dictionaries between consecutive slices.
  - `alignment_regularisation` weighs alignment against per-slice quality; `alignment_window_size` sets how many neighbouring slices are aligned.
  - The docs claim this works better than Procrustes-aligning independent embeddings or initializing from the previous one. They promise only "general alignment" and "rough positions". Online `update` is "less good" because only backward relations exist.
  - — [AlignedUMAP docs](https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html)
- Gigante, Charles, Krishnaswamy & Mishne (2019, arXiv; NeurIPS 2019), M-PHATE **[abstract]**. Multislice PHATE is "the first method designed explicitly to visualize how a neural network's hidden representations evolve throughout training". It captures dynamics and community structure better than Isomap and t-SNE. Use cases: catastrophic forgetting, and heterogeneity of hidden units vs generalization. — [arXiv 1908.02831](https://arxiv.org/abs/1908.02831). Extension to RNN training: [Xie et al. 2024, Multiway Multislice PHATE, arXiv](https://arxiv.org/abs/2406.01969) **[metadata only; arXiv ID not verified]**
- Yang, Lin, Liu & Dong (2022), TimeVis, *IJCAI* **[abstract]**. "Time-travelling visualization" of classifier training. One spatio-temporal complex combines within-epoch neighbours and the same sample across epochs. A single visualization model projects and inverse-projects across epochs; reported 15x faster than DeepVisualInsight and more faithful. — [Yang et al. 2022](https://doi.org/10.24963/ijcai.2022/558)
- Poličar & Zupan (2025), *Machine Learning* **[abstract]**. Adds two temporal loss terms to t-SNE/UMAP-type methods; this improves temporal coherence "while preserving the fidelity". — [Poličar & Zupan](https://doi.org/10.1007/s10994-025-06734-z)
- Rauber, Fadel, Falcão & Telea (2017), "Visualizing the hidden activity of artificial neural networks", *IEEE TVCG* 23(1):101–110 **[not re-verified]**. Uses t-SNE projections of activations, including across training epochs (dynamic t-SNE). — [Rauber et al. 2017](https://doi.org/10.1109/TVCG.2016.2598838)
- Cantareira & Paulovich (2020), "A Generic Model for Projection Alignment Applied to Neural Network Visualization", EuroVA **[metadata only]** — [EG Digital Library](https://doi.org/10.2312/eurova.20201088) (DOI not verified)

### Inferences
- With 26 checkpoints and the same 600 tokens matched across frames, dynamic t-SNE, Joint t-SNE and AlignedUMAP are all feasible, because correspondences are known. Their alignment term is a regularizer, though. It suppresses real motion in proportion to λ, so apparent stillness in a smoothed view is not evidence of stillness in 192-D. Vernier's temporal Shepard correlation (δ vs δ̄) is the right check.
- The current fixed linear basis is a "G-PCA" analogue, except that the basis comes from fc2 columns rather than the token cloud. It has perfect temporal coordinate stability: the frame of reference never moves. Its δ̄ is an exact linear contraction of δ. The cost is that, at epochs where token variance lies outside the fc2-PCA span, spatial fidelity drops.

### Gaps
- I did not retrieve the exact dt-SNE equation (normalization of the λ-term) or any numeric dt-SNE vs AlignedUMAP comparison.
- I found no benchmark of AlignedUMAP against dt-SNE / Joint t-SNE on NN-checkpoint data.
- The M-PHATE NeurIPS proceedings version was not checked.

---

## Q4. Quantitative quality measures for any projection, and which to report

### Takeaway
Fidelity is multi-objective: neighbourhood ranks, pairwise distances, cluster/class relations and temporal stability are different targets, and the measures for them are correlated and gameable. The literature recommends a **panel** of measures across scales:
- trustworthiness/continuity, or the co-ranking R_NX(K) curve with its log-K AUC;
- kNN recall;
- Shepard diagram plus Spearman correlation of distances;
- a scale-invariant stress;
- class-based measures (neighbourhood hit, kNN accuracy, Steadiness/Cohesiveness);
- for linear maps, explained variance.

Metrics should be treated as necessary but not sufficient.

### Cited Findings

**Definitions and frameworks**
- Trustworthiness and continuity were introduced by Venna & Kaski (2001, ICANN) — [Venna & Kaski 2001](https://doi.org/10.1007/3-540-44668-0_68) — and used further in Venna & Kaski (2006), "Local multidimensional scaling", *Neural Networks* 19:889 **[not re-verified]** — [Venna & Kaski 2006](https://doi.org/10.1016/j.neunet.2006.05.014).
  - Formula as given in Vernier et al. 2020 **[full text]**: T(k) = 1 − 2/(Nk(2N−3k−1)) Σ_i Σ_{j∈U_k(i)} (r(i,j) − k). Here U_k(i) is the set of points in the low-D k-neighbourhood of i but not in its high-D one ("intrusions"), and r is the high-D rank.
  - Continuity is the mirror image: points lost from the neighbourhood ("extrusions"), ranked in low-D.
  - Terminology varies: "false neighbours" vs "missing neighbours". Trustworthiness penalizes intrusions (false neighbours); continuity penalizes extrusions (missing neighbours) — [Vernier et al. 2020](https://ar5iv.labs.arxiv.org/html/2002.07481)
- Information-retrieval view: precision and recall of neighbour retrieval from the display. NeRV optimizes a user-weighted trade-off (cost of missing similar points vs retrieving dissimilar ones) **[abstract]** — [Venna, Peltonen, Nybo, Aidos & Kaski 2010, JMLR 11:451](https://jmlr.org/papers/v11/venna10a.html)
- Co-ranking matrix, Lee & Verleysen 2009 (rank-based criteria), *Neurocomputing* 72:1431 **[not re-verified]**:
  - Q_kl counts pairs with high-D rank k and low-D rank l.
  - Q_NX(K) = (1/KN) Σ_{k,l≤K} Q_kl is the average fraction of shared K-neighbours.
  - B_NX(K) shows whether intrusions or extrusions dominate.
  - LCMC(K) = Q_NX(K) − K/(N−1).
  - Rescaled R_NX(K) = ((N−1)Q_NX(K) − K)/(N−1−K): 0 = random embedding, 1 = perfect.
  - The scalar summary is AUC of R_NX over log K, which weights small K more (Lee, Peluffo-Ordóñez & Verleysen 2015).
  - — [Lee & Verleysen 2009](https://doi.org/10.1016/j.neucom.2008.12.017); [Lee, Peluffo-Ordóñez & Verleysen 2015, Neurocomputing 169:246](https://doi.org/10.1016/j.neucom.2014.12.095); K-ary neighbourhood review [Lee & Verleysen 2008](https://www.semanticscholar.org/paper/443da7f542d8641da1d1ba920da6691c0ed24197); scale-independent criteria [Lee & Verleysen 2010, Pattern Recognit Lett 31:2248](https://doi.org/10.1016/j.patrec.2010.04.013)
- Lueks, Mokbel, Biehl & Hammer (2011) **[abstract]**: the single K in co-ranking measures conflates two notions (the rank-error tolerance and the neighbourhood size that matters). They propose two separate parameters and per-point colouring by local contribution to quality — [arXiv 1110.3917](https://arxiv.org/abs/1110.3917); journal version [Mokbel et al. 2013, Neurocomputing 112:109](https://doi.org/10.1016/j.neucom.2012.11.046)
- Implementations: R packages dimRed + coRanking (co-ranking framework, 15 DR methods) — [Kraemer, Reichstein & Mahecha 2018, R Journal](https://doi.org/10.32614/RJ-2018-039). ZADU, a Python library of distortion measures with point-wise contributions, plus ZADUVis — [Jeon et al. 2023, IEEE VIS](https://doi.org/10.1109/VIS54172.2023.00048). pyDRMetrics — [Zhang, Shang & Zhang 2021, Heliyon](https://doi.org/10.1016/j.heliyon.2021.e06199) **[metadata only; DOI not verified]**

**Benchmarks and surveys: what they report**
- Espadoto, Martins, Kerren, Hirata & Telea (2021), *IEEE TVCG* 27(3):2153–2173 **[abstract]**: a quantitative survey that characterizes the data space, the techniques and projection quality "by several quantitative metrics". Methodology, datasets, implementations and results are open. Per Vernier et al. 2020 (who "follow EMK*19" with changes), the metric set includes:
  - trustworthiness, continuity, neighbourhood hit (class labels in the low-D kNN), normalized stress (on standardized distances);
  - Shepard-diagram correlation (Pearson in Espadoto; Vernier added Spearman and Kendall) **[full text of Vernier]**.
  - Vernier evaluated T/C/NP at 20 neighbourhood sizes from 1% to 20% of N, and NH at 0.25%–5% of N.
  - — [Espadoto et al. 2021](https://doi.org/10.1109/TVCG.2019.2944182); [Vernier et al. 2020](https://ar5iv.labs.arxiv.org/html/2002.07481)
- Nonato & Aupetit (2019), *IEEE TVCG* 25(8):2650–2673 **[abstract]**: a survey linking projection techniques to *distortion types*, quantitative distortion measures, analytic tasks and "layout enrichment" to debunk distortions. Gives guidelines for choosing a projection per task — [Nonato & Aupetit 2019](https://doi.org/10.1109/TVCG.2018.2846735)
- Bertini, Tatu & Keim (2011), *IEEE TVCG* **[metadata]**: overview and systematization of quality metrics in high-dimensional visualization — [Bertini et al. 2011](https://doi.org/10.1109/TVCG.2011.229) (DOI not verified)
- Kobak & Berens 2019: KNN (k=10), KNC and CPD (Spearman correlation of distances) **[full text]** — [link](https://doi.org/10.1038/s41467-019-13056-x). de Bodt et al. 2025 use 10-NN preserved (local) and correlation of pairwise distances (global). They note that absolute metric values are hard to interpret because the best achievable value is unknown **[full text, partial]** — [arXiv 2508.15929](https://doi.org/10.48550/arXiv.2508.15929)
- Lause et al. 2024: kNN recall, kNN accuracy, silhouette, AMI (see Q1 for numbers). Chari & Pachter 2023: Jaccard distance of 30-NN sets, correlation of rankings and of distances — [Lause 2024](https://doi.org/10.1371/journal.pcbi.1012403); [Chari 2023](https://doi.org/10.1371/journal.pcbi.1011288)
- Tian, Zhai, van Steenpaal, Yu, Dimara, Espadoto & Telea (2021), *Information* 12(6):239 **[abstract]**: quantitative and user comparison of 2D vs 3D projections. "In general, 3D projections bring only limited added-value atop of … 2D counterparts", though some 3D techniques "show more structure". — [Tian et al. 2021](https://doi.org/10.3390/info12060239)

**Pitfalls of the measures themselves**
- Smelser, Miller & Kobourov (2024, BELIV) and Smelser, Gunaratne, Miller & Kobourov (2025/26, *IEEE TVCG*) **[abstract]**: "normalized" stress is sensitive to uniform scaling of the projection, and so is t-SNE's KL divergence. Rescaling alone changes values and can change algorithm rankings. They propose scale-invariant versions (minimizing over a scale factor) — [arXiv 2408.07724](https://arxiv.org/abs/2408.07724); [TVCG](https://doi.org/10.1109/TVCG.2026.3657654); related [Ahmed et al. 2024, "Size should not matter", arXiv](https://arxiv.org/abs/2408.04688) **[metadata only; arXiv ID not verified]**
- Machado, Behrisch & Telea (2025), *CGF*, "Necessary but not Sufficient" **[abstract]**: they generate projections that score as high as or higher than well-known techniques on projection quality metrics (PQMs) but show "different, often confusing, visual patterns". So PQMs are exclusion criteria: a low score implies a bad projection, but a high score does not imply a good one. Not all metrics can be fooled equally — [Machado et al. 2025](https://doi.org/10.1111/cgf.70101)
- Jeon et al. (2022), *IEEE TVCG*, Steadiness and Cohesiveness **[abstract]**: trustworthiness and continuity "fail to measure inter-cluster reliability". Steadiness asks whether projected clusters are clusters in the original space; Cohesiveness asks the converse. Both are point-wise and give a "reliability map" — [Jeon et al. 2022](https://doi.org/10.1109/TVCG.2021.3114833)
- Thrun, Märte & Stier (2023), *MAKE* 5:1076 **[abstract]**: 19 unsupervised quality measures, grouped with graph theory, are "biased towards assumed underlying structures" and fail on simple benchmark structures. They propose the Gabriel Classification Error — [Thrun et al. 2023](https://doi.org/10.3390/make5030056)
- Bae, Jeon & Seo (2025, IEEE VIS) **[abstract]**: picking correlated metrics biases evaluations. Cluster metrics by their *empirical* correlation and report one per cluster — [Bae et al. 2025](https://doi.org/10.1109/VIS60296.2025.00014)
- Kazempour et al. (2024, ICDMW), "Do good scores imply good embeddings?": argues for inspecting results with multiple criteria **[title/metadata only]**.

### Inferences
**Recommended reporting panel for the app.** At 600 tokens, a 600×600 distance and rank matrix per checkpoint is trivial, so exact computation for all 26 checkpoints is cheap.
1. **Linear-specific:** fraction of token-cloud variance captured by the fixed fc2-PCA basis at each checkpoint, compared with the cloud's own top-3 PCA (the upper bound for any 3D linear view); fraction of net-force norm captured, Σ‖PF_i‖² / Σ‖F_i‖².
2. **Neighbourhood:** T(k), C(k) and the R_NX(K) curve with log-K AUC. k ≈ 10 matches Kobak/de Bodt; a log-spaced K range covers multiple scales.
3. **Distances:** Shepard diagram plus Spearman correlation of all pairwise distances (CPD), and scale-invariant stress (Smelser).
4. **Groups:** neighbourhood hit / kNN accuracy for the tool vs background labels; Steadiness/Cohesiveness if inter-group distance is being interpreted.
5. **Temporal:** Vernier temporal Shepard correlation and temporal stress of per-token displacement δ vs δ̄ between consecutive checkpoints.
6. **Vectors:** per-token cosine and norm ratio between the projected force and the high-D force. Under a linear P this is the in-subspace fraction (Q2).

Report the panel as curves over epoch, not single numbers. The participation-ratio drop (10.6 → 3.15) predicts that the linear metrics will improve with training. Whether they do is an empirical check.

### Gaps
- Espadoto et al.'s exact metric list, neighbourhood size (I recall K=7 but did not verify) and benchmark counts (number of techniques and datasets) could not be confirmed: the PDF host refused the connection and the abstract gives no numbers.
- The exact co-ranking formulas above are given as standard definitions; the Lee & Verleysen papers were not re-fetched.
- No peer-reviewed metric for the fidelity of projected *Jacobian/trap classifications* was found.

---

## Q5. Visualizing per-point distortion on the map

### Takeaway
There is a mature toolkit for drawing fidelity onto the view itself:
- colour points or Voronoi cells by local compression/stretching or by local stress;
- colour points by their contribution to trustworthiness/continuity or R_NX;
- overlay high-D kNN edges that are missing or false in the view;
- colour all points by their true high-D distance to the hovered point;
- draw Tissot-like ellipses (Jacobian/indicatrix glyphs);
- draw reliability maps from cluster-level metrics.

### Cited Findings
- Aupetit (2007), "Visualizing distortions and recovering topology in continuous projection techniques", *Neurocomputing* 70:1304 **[abstract unavailable; content per deep-search summary]**: per-point distortion indices for *compression* and *stretching* of distances, displayed by colouring Voronoi cells of the projected points, plus methods to recover topology (e.g. which projected neighbours are false) — [Aupetit 2007](https://doi.org/10.1016/j.neucom.2006.11.018); French extended version [Aupetit & Gaillard 2008, RIA](https://doi.org/10.3166/ria.22.443-472) (DOI not verified); position paper [Lespinats & Aupetit 2010, ESANN, "Mapping without visualizing local default is nonsense"](https://www.esann.org/sites/default/files/proceedings/legacy/es2010-45.pdf) **[metadata only; URL not verified]**
- CheckViz: Lespinats & Aupetit (2011), *Comput Graph Forum* 30(1):113–125 **[not re-verified]**. A two-dimensional perceptual colour scale encodes, per point, false neighbourhoods (compression) and tears (stretching) at the same time — [Lespinats & Aupetit 2011](https://doi.org/10.1111/j.1467-8659.2010.01835.x)
- Seifert, Sabol & Kienreich (2010), "Stress Maps" **[abstract]**: local stress shown as a heat map on the projection; a global stress value "does not provide insight into localised phenomena" — [Seifert et al. 2010](https://doi.org/10.2312/PE/EuroVAST/EuroVAST10/013-018)
- Martins, Coimbra, Minghim & Telea (2014), "Visual analysis of dimensionality reduction quality for parameterized projections", *Comput Graph* 41:26–42 **[abstract unavailable]**: per-point error views (aggregate distance error, false neighbours and missing neighbours) for comparing parameter settings — [Martins et al. 2014](https://doi.org/10.1016/j.cag.2014.01.006). Martins, Minghim & Telea (2015) **[abstract]**: metrics and views that explain neighbourhood-preservation errors at several scales — [Martins et al. 2015](https://doi.org/10.2312/cgvc.20151234)
- Colange, Vuillon, Lespinats & Dutykh (2019, IEEE VIS), MING **[abstract]**: kNN graphs from data space and embedding are drawn on the map. Shared edges are reliable; unshared ones show false or missing neighbourhoods. The pairs precision/recall and trustworthiness/continuity are interpreted this way — [Colange et al. 2019](https://doi.org/10.1109/VISUAL.2019.8933568); journal follow-up [Kim et al. 2021, Inf Vis](https://doi.org/10.1177/14738716211004019) **[metadata only; DOI not verified]**
- Lueks et al. (2011): colour every point by its local contribution to co-ranking quality **[abstract]** — [arXiv 1110.3917](https://arxiv.org/abs/1110.3917)
- Jeon et al. (2022): point-wise Steadiness/Cohesiveness gives a "reliability map" that helps "prevent misinterpretation while performing inter-cluster tasks" **[abstract]** — [Jeon et al. 2022](https://doi.org/10.1109/TVCG.2021.3114833). ZADUVis draws distortion maps from ZADU's point-wise contributions **[abstract]** — [Jeon et al. 2023](https://doi.org/10.1109/VIS54172.2023.00048)
- Raval, Viégas & Wattenberg (2024, IEEE VIS), "Hypertrix" **[abstract]**: a Tissot-indicatrix analogue that draws ellipses showing how a high-D → 2D projection distorts local distances — [Raval et al. 2024](https://doi.org/10.1109/VIS55277.2024.00073)
- Sankaran, Zhang, Chenab & Meilă (2026), *Brief Bioinform*, "distortions" package **[abstract]**: interactive displays of local metric distortion in UMAP/t-SNE that flag fragmented neighbourhoods and support hyperparameter tuning and method selection — [Sankaran et al. 2026](https://doi.org/10.1093/bib/bbag136)
- Ovchinnikova & Anders (2019), Sleepwalk, *Genome Res* **[abstract]**: hovering over a point colours all others by their *original-space* distance to it, which reveals distortions directly — [Ovchinnikova & Anders 2019 (bioRxiv DOI)](https://doi.org/10.1101/603589)
- Heiter et al. (2024, arXiv), TRACE **[abstract]**: point-wise local and global quality, high-D neighbours highlighted for any selected group, and high-D distances displayed — [arXiv 2406.12953](https://doi.org/10.48550/arXiv.2406.12953)
- Chatzimparmpas, Martins & Kerren (2020), t-viSNE, *IEEE TVCG* **[metadata]**: interactive assessment of t-SNE projections — [t-viSNE](https://doi.org/10.1109/TVCG.2020.2986996) (DOI not verified). Stahnke et al. (2016), "Probing Projections", *IEEE TVCG* **[metadata]**: interaction techniques for errors of DR — [Stahnke et al. 2016](https://doi.org/10.1109/TVCG.2015.2467717)
- Xia et al. 2024 (scDEED) and Liu et al. 2025 (LOO-map) give point-wise reliability scores that can be used directly as colour overlays **[abstract]** — [scDEED](https://doi.org/10.1038/s41467-024-45891-y); [LOO-map](https://doi.org/10.1038/s41467-025-60434-9)
- Jeon et al. (2025, CHI) survey of 133 papers on reliability of DR visual analytics: research is "skewed toward developing new DR techniques rather than their interpretation or evaluation" **[abstract]** — [Jeon et al. 2025](https://doi.org/10.1145/3706598.3713551)

### Inferences
Direct transfer to the 3D app:
- per-token colour = local trustworthiness/continuity contribution, or compression vs stretching (CheckViz-style bivariate colour);
- toggleable MING-style edges: high-D kNN edges missing in 3D in one colour, 3D-only neighbours in another;
- Sleepwalk hover: colour every token by its true 192-D distance to the hovered token;
- per-token "captured fraction" (‖Px−μ‖²/‖x−μ‖² for position, ‖Pf‖/‖f‖ for force) as point size or opacity;
- for traps, an indicator of how much of the trap's Jacobian action leaves the 3D subspace (Q2 invariance residual).

In 3D, occlusion and viewpoint add a visibility dimension that is separate from numerical fidelity. Tian et al. 2021 found that 3D adds only limited quantitative value over 2D.

### Gaps
- The full texts of Aupetit 2007, CheckViz 2011 and Martins 2014 were not accessible in this session, so the descriptions rest on well-known content and the deep-search summary.
- I found no study of distortion overlays in *3D* or animated views specifically. All cited tools are 2D.

---

## Q6. Practical consensus: when to avoid nonlinear embeddings, and implications for the ViT force-field app

### Takeaway
There is broad agreement, including between the two sides of the Chari–Pachter / Lause–Kobak debate, that 2D/3D t-SNE/UMAP views do not preserve distances. They should not be used for quantitative downstream analysis, inter-cluster distances, cluster sizes or densities, and they are exploratory only. Where geometry, distances, vector arithmetic or dynamics matter, as with forces summed over 768 neurons and Jacobian traps, the measurements should stay in 192-D. The display should be a fixed linear projection whose fidelity is reported. Nonlinear views can be an auxiliary, clearly labelled neighbourhood view.

### Cited Findings
- Both sides of the debate agree that 2D embeddings do not preserve distances and should not be used for quantitative downstream analysis **[full text]** — [Chari & Pachter 2023](https://doi.org/10.1371/journal.pcbi.1011288); [Lause et al. 2024](https://doi.org/10.1371/journal.pcbi.1012403)
- De Bodt et al. 2025 best practices **[full text, partial]**: "Do not run downstream analysis on 2D embeddings"; "in neighbour embeddings, between-cluster distances are not indicative"; distances in PCA/MDS are meaningful; remove ticks where distances are meaningless; report explained variance or preserved neighbours; show alternative embeddings; validate observations independently — [arXiv 2508.15929](https://doi.org/10.48550/arXiv.2508.15929)
- Kobak & Berens 2019 / Kobak & Linderman 2021: if t-SNE/UMAP are used at all, use informative (PCA/spectral) initialization, because global arrangement otherwise reflects random init **[full text / abstract]** — [KB 2019](https://doi.org/10.1038/s41467-019-13056-x); [KL 2021](https://doi.org/10.1038/s41587-020-00809-z)
- Velocity arrows on nonlinear embeddings are interpolation artefacts of the kNN graph, and using them to validate the embedding is circular **[abstract]** — [Zheng et al. 2023](https://doi.org/10.1186/s13059-023-03065-x)
- When dynamics must be preserved, use projections designed for it, e.g. energy-based preservation of critical points and flow maps **[abstract]** — [Bartolovic et al. 2020](https://doi.org/10.1111/cgf.13978)
- In the time-dependent benchmark, global PCA had the best balance of spatial quality and stability, and per-frame t-SNE was the least stable **[full text]** — [Vernier et al. 2020](https://ar5iv.labs.arxiv.org/html/2002.07481)
- Quality metrics are exclusion criteria, not proofs of faithfulness **[abstract]** — [Machado et al. 2025](https://doi.org/10.1111/cgf.70101)
- 3D adds limited value over 2D on average **[abstract]** — [Tian et al. 2021](https://doi.org/10.3390/info12060239)

### Inferences
**Applicability to this system.**
- **Keep the linear view as the primary "physics" view.** Only a linear map keeps the sum of 768 neuron forces equal to the projected net force, and keeps arrows comparable across tokens and checkpoints. It is also perfectly temporally stable.
- Consider replacing or augmenting the fc2-column PCA basis with a basis fitted to the *token cloud pooled over all checkpoints*. This would be a "global PCA" in Vernier's sense. Report captured variance and force per epoch for both bases.
- **Nonlinear views (t-SNE/UMAP/PaCMAP, aligned across checkpoints) are useful only for neighbourhood questions.** Examples: "which tokens are each other's nearest neighbours", "are tool tokens a contiguous group". They should come with R_NX/T&C and neighbourhood-hit numbers and a distortion overlay. Arrows and trap glyphs should not be drawn on them, unless a single parametric map is used and the arrows are J_φf with ellipse glyphs and a faithfulness test.
- Trap classification (attracting/repelling/saddle/vortex) must be computed from the 192×192 Jacobian and displayed as a label or glyph. It must not be inferred from the look of the projected field. Even the linear view can misrepresent it when the projected subspace is not J-invariant.
- These recommendations are synthesized from the cited findings. No paper studies this exact transformer setting.

### Gaps
- No direct empirical comparison exists of linear vs nonlinear 3D views for interpreting force fields or Jacobian fixed points in neural-network representation spaces. Bartolovic 2020 is the nearest, and it uses physical dynamical systems.
- Scite citation-statement checks (supporting or contrasting citations for the Chari–Pachter vs Lause–Kobak dispute and for Espadoto's metric choices) could not be run because the Scite monthly limit was exhausted.
