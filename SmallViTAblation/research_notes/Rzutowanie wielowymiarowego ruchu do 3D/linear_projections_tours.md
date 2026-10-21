# Linear and interactive projections of high-dimensional data (points AND vectors) into 2D/3D: methods, guarantees, tours, view design, temporal alignment

Scope note for the report writer. These notes cover linear and interactive projections: what each one optimizes, what it preserves, where it breaks, and whether it can carry vectors as well as points. They are checked against our system: a 192-D residual stream, 600 tokens ("colloids"), 768 MLP "beams" u_c = GELU(z_c)·v_c, and one fixed 3-D subspace P3 (top-3 PCs of the normalized fc2 columns), compared across 26 checkpoints.

Sources: Undermind deep search (176 ranked papers; link: https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Linear%20projections%20tours%20projection%20pursuit%20for%20high%20dimensional%20data), full-text reads of the key PDFs, Scite (only 3 queries before the plan quota ran out), and web search.

Labels used below: **[Theorem]** = proven mathematical result; **[Empirical]** = measured in a study; **[Claim]** = the authors' opinion or recommendation; **[Computed]** = arithmetic I did with a cited formula, applied to our numbers.

---

## 1. PCA and variants (weighted, sparse, probabilistic), LDA / supervised / targeted projections, CCA/PLS, ICA: what each optimizes and preserves, and when it is the right choice

### Takeaway
Every method here is an orthonormal (or at least linear) map, so all of them preserve vector sums and affine structure exactly. They differ only in which subspace they pick, i.e. in the objective each one optimizes. PCA keeps the most variance of whatever matrix you feed it. LDA keeps the most class separation. CCA and PLS keep the most correlation or covariance between two sets of variables. ICA and projection pursuit keep the most non-Gaussianity. For our app, the main point is that P3 is the PCA of beam *directions*, so it is optimal for none of the quantities actually drawn: token positions, net forces, or the tool–background axis. Each drawn quantity has its own optimal subspace.

### Cited Findings
- **Unifying view** [Claim/Theorem]: Cunningham & Ghahramani survey linear dimensionality reduction as optimization over matrix manifolds of orthonormal projections. This covers PCA, factor analysis, Fisher LDA, CCA and slow feature analysis. They argue the optimization view exposes the "suboptimality of certain eigenvector solutions" (for example, the classical eigenvector shortcut is not the optimal answer for multi-dimensional LDA). — [Cunningham & Ghahramani 2015, JMLR 16:2859–2900](https://jmlr.org/papers/v16/cunningham15a.html)
- **PCA objective** [Claim]: "Principal component analysis chooses a projection to maximize the variance. Fisher's linear discriminant chooses a projection that maximizes the relative separation between group means. Projection pursuit generalizes these ideas into a common method, where an arbitrary function on projections is optimized." The scatterplot matrix is also described as a projection method (all 2-D coordinate marginals). — [Cook, Buja, Lee & Wickham 2008, Handbook of Data Visualization, pp. 295–314](https://doi.org/10.1007/978-3-540-33037-0_13)
- **Covariance vs. "interesting" structure** [Claim]: Before projection pursuit the data are usually *sphered* (replaced by standardized PCs), "the major reason ... is that covariance structure is not interesting to find using PP. Covariance can be adequately captured by principal component analysis." — [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13). Friedman puts it the same way: linear effects are "directly captured by the covariance structure", so exploratory PP should focus on nonlinear effects such as clustering. — [Friedman 1987, JASA 82:249–266](https://doi.org/10.1080/01621459.1987.10478427)
- **PCA failure mode, ranking** [Empirical/Claim]: In the Distill grand-tour article, "the first two principal components of softmax activations are not substantially better than the third". The top-variance pair is not automatically the most informative view. — [Li, Zhao & Scheidegger 2020, Distill](https://doi.org/10.23915/distill.00025)
- **PCA failure mode, trajectories** [Theorem/Empirical]: For long high-dimensional random walks, "most of the variance is in the first few PCA components". Projecting the walk onto any subspace of PCs gives Lissajous curves, and training trajectories of a linear CIFAR-10 model and ResNet-50 on ImageNet have PCA variance spectra resembling a random walk with drift. — [Antognini & Sohl-Dickstein 2018, NeurIPS](https://arxiv.org/abs/1806.08805)
- **Probabilistic PCA** [Theorem]: PCA restated as the maximum-likelihood solution of a latent Gaussian model with isotropic noise. The ML subspace is the principal subspace, and the model also estimates the noise variance. — [Tipping & Bishop 1999, JRSS-B 61:611–622](https://doi.org/10.1111/1467-9868.00196)
- **Sparse PCA** [Method]: Lasso/elastic-net penalties give loadings with few non-zero coordinates. — [Zou, Hastie & Tibshirani 2006, JCGS 15:265–286](https://doi.org/10.1198/106186006X113430). Related theorem: when p/n → γ > 0, *sparse* 1-D projections of structureless Gaussian data are asymptotically all Gaussian (Theorem 6), whereas dense optimized projections can show spurious structure. — [Bickel, Kur & Nadler 2018, PNAS 115:9151–9156](https://doi.org/10.1073/pnas.1801177115)
- **LDA** [Method]: Fisher 1936 is the classical source for the class-separation axis. — [Fisher 1936, Annals of Eugenics 7:179–188](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x). The projection-pursuit version (the "LDA index", based on the within/between-group scatter ratio) and its penalized version for large-p/small-n data (the "PDA index") are in [Lee, Cook, Klinke & Lumley 2005, JCGS 14:831–846](https://doi.org/10.1198/106186005X77702) and [Lee & Cook 2010, Statistics & Computing 20:381–392](https://doi.org/10.1007/s11222-009-9131-1).
- **Targeted / contrastive / demixed projections** [Method]:
  - Contrastive PCA finds "low-dimensional structures enriched in one dataset compared to another" (a target vs. a background dataset). — [Abid et al. 2018, Nat. Commun. 9](https://doi.org/10.1038/s41467-018-04608-8)
  - Demixed PCA splits population activity into components that "each relate to only a single aspect of the task". — [Kobak et al. 2016, eLife 5:e10989](https://doi.org/10.7554/eLife.10989)
  - Targeted projection pursuit (the user specifies a target view and the projection is fitted to it). — [Faith 2007, IV'07, pp. 286–292](https://doi.org/10.1109/IV.2007.107)
- **Rotation-targeted projection (jPCA)** [Method + critique]: jPCA first reduces to 6 PCs, then fits a skew-symmetric linear dynamics matrix M_skew and projects onto the planes of its eigenvectors, which shows rotational dynamics. — [Churchland et al. 2012, Nature 487:51–56](https://doi.org/10.1038/nature11129). The method is described in a re-analysis arguing that rotations can come from mere temporal shifts of peak firing across neurons. — [Lebedev et al. 2019, PMC6908571](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6908571/)
- **CCA** [Method]: Hotelling's canonical correlations give paired directions in two variable sets that maximize correlation. — [Hotelling 1936, Biometrika 28:321–377](https://doi.org/10.1093/biomet/28.3-4.321)
- **ICA** [Method]: finds maximally non-Gaussian, statistically independent directions. — [Hyvärinen & Oja 2000, Neural Networks 13:411–430](https://doi.org/10.1016/S0893-6080(00)00026-5). Caveat [Theorem]: "with a limited amount of high-dimensional data, the results of projection pursuit and related ICA methods should be interpreted with great care ... even with structureless data, one may find projections with highly non-Gaussian components." — [Bickel et al. 2018](https://doi.org/10.1073/pnas.1801177115)
- **3-D PP** [Method/Empirical]: The Jones–Sibson moment index has been extended to 3-D. On a tetrahedral test set it produced better clusters than PCA. — [Nason 1995, JRSS-C 44:411–430](https://doi.org/10.2307/2986135)

### Inferences
- **The right subspace depends on the quantity drawn.**
  - Expected captured squared length of a vector-valued quantity Q (token offsets, net forces, single beams) is tr(Pᵀ E[QQᵀ] P). By the Ky Fan maximum principle this is maximized by the top-3 eigenvectors of E[QQᵀ] (standard linear algebra, consistent with the Cunningham & Ghahramani framing).
  - Token-cloud PCA maximizes token spread, which is easy to capture: participation ratio (PR) ≈ 3.15 at late epochs. "Force PCA" (eigenvectors of Σ_F = E[F Fᵀ] over tokens) maximizes the drawn net-force length. Activation-weighted beam PCA, with weights E[GELU(z_c)²]·‖v_c‖², maximizes the captured beam energy actually delivered.
  - P3 uses *unweighted, normalized* fc2 columns, so it optimizes the geometry of the beam dictionary rather than any of these energies. That is consistent with the measured 18.8% beam energy and the median 47% of net-force length.
- **Tool vs. background.** The literature's pattern is "1 supervised axis + PCA of the residual", e.g. LDA/contrastive or class-mean-difference axes followed by PCA in the orthogonal complement (dPCA and Mante-style targeted dimensionality reduction belong to this family). With 2 classes LDA gives exactly 1 axis, so the remaining 2 display axes should come from PCA or force-PCA of the residual after removing that axis. This guarantees 100% of the separation vector is in view; currently only ~16% is.
- **Displacement–force coupling.** CCA, PLS, or reduced-rank regression between token offset X and net force F would find the directions where displacement and force co-vary most, i.e. the dominant input–output modes of the local linear field F ≈ J x. This fits "trap" visualizations better than plain PCA of either X or F. (Inference; no source applying CCA to force fields was found.)
- **ICA / PP on our data need null controls.** γ = p/n ≈ 192/600 = 0.32 for tokens and 192/768 = 0.25 for beams. Bickel et al.'s Theorem 3 then allows optimized projections of structureless data to look like non-Gaussian mixtures. Run the same index on shuffled or Gaussianized data, or restrict to sparse or PCA-reduced inputs.

### Gaps
- I found no peer-reviewed work that selects a linear 3-D view *jointly* for a point cloud and a vector field defined on it, e.g. a weighted trace objective over token variance + force energy + class separation. Such an objective is a straightforward Stiefel-manifold problem in the Cunningham–Ghahramani framework, but it is not documented as a visualization method.
- Weighted PCA for vector dictionaries (activation-weighted) has no specific visualization reference. The choice of weights is our design decision.

---

## 2. Random projections and Johnson–Lindenstrauss for 3 output dims; Diaconis & Freedman (1984): most low-dimensional projections look Gaussian

### Takeaway
At k = 3, JL guarantees nothing about distances: the required dimension is of order ε⁻² log n, so distortion is of order 1. A fixed unit vector keeps on average only k/p = 3/192 ≈ 1.6% of its squared length (≈12.5% of its length) in a random 3-D subspace. The Diaconis–Freedman effect says that, for data that are genuinely high-dimensional (spread over many directions), almost every low-dimensional projection is a scale mixture of Gaussians, so random views of the 37-effective-dimension beam cloud will mostly look like a Gaussian blob. The effect does not apply to a cloud whose variance sits in ~3 directions (tokens late in training).

### Cited Findings
- **JL optimality** [Theorem]: For a suitable n-point set, any embedding (even nonlinear) preserving all distances within (1 ± ε) needs dimension Ω(ε⁻² log n). This matches the JL upper bound and holds for ε larger than ≈ 1/min{n,d}^0.4999. — [Larsen & Nelson 2017, FOCS](https://doi.org/10.1109/FOCS.2017.64) (arXiv 1609.02094). An earlier result showed the same for linear maps. — [Larsen & Nelson, arXiv 1411.2404](https://ar5iv.arxiv.org/html/1411.2404)
- **Explicit JL constant** [Theorem]: A sufficient dimension is k ≥ 24 ln n / (3ε² − 2ε³) (as restated by later work), proved with elementary moment-generating-function bounds. — [Dasgupta & Gupta 2003, Random Struct. Alg. 22:60–65](https://doi.org/10.1002/rsa.10073)
- **Best linear embedding in very low k** [Theorem/Method]: There are algorithms whose k-D orthonormal embeddings come within δ of the *optimal* achievable distortion ε_opt(k). This is reported as beating PCA and random projections in experiments. Note that ε_opt(k) can be large: the guarantee is relative to the optimum, not absolute. — [Grant, Hegde & Indyk 2013, IEEE GlobalSIP, pp. 973–976](https://doi.org/10.1109/GlobalSIP.2013.6737055)
- **Diaconis–Freedman** [Theorem]: "Theorems are given to show that under suitable conditions, most projections are approximately Gaussian." (564 citing publications per Scite.) — [Diaconis & Freedman 1984, Ann. Statist. 12:793–815](https://doi.org/10.1214/aos/1176346703)
- **Necessary and sufficient conditions** [Theorem]: For fixed d with q → ∞, most d-dimensional projections converge to a Gaussian *scale mixture* ∫N(0, vI_d)R(dv) **iff** ‖X‖²/q converges weakly to some R and ⟨X, X′⟩/q → 0 in probability (X, X′ independent). The Diaconis–Freedman conditions are thus necessary as well as sufficient. — [Dümbgen & Del Conte-Zerial 2013, IMS Collections 9:91–104](https://doi.org/10.1214/12-IMSCOLL908)
- **Scale-mixture form and dependence on d/D** [Theorem]: almost all projections of X ∈ R^D into R^d look like a mixture of N(0, σ²I_d) weighted by P(‖X‖² = σ²D). How strong the effect is depends on d/D and on an "eccentricity" coefficient of X's distribution. — [Dasgupta, Hsu & Verma 2006, arXiv 1206.6813](https://arxiv.org/abs/1206.6813)
- **Quantitative versions** [Theorem]:
  - E d_BL(X_θ, σZ) ≤ C^k B / d^{2/(9k+4)} + (σ√k(A+1) + σk)/(d−1). Here B is the largest directional second moment, so the bound is weak for anisotropic data. The bound also concentrates exponentially in d, giving Gaussianity for k up to c·log d. — [Meckes 2012, J. Theor. Probab. 25:333–352](https://doi.org/10.1007/s10959-010-0299-2)
  - The 1-D case gives a lower bound on the "waiting time" to find a non-Gaussian projection when directions are tried uniformly at random. — [Meckes 2009, Electron. Commun. Probab. 14:176–185](https://doi.org/10.1214/ECP.v14-1457)
  - The phenomenon persists for k < 2 log d / log log d, and this is best possible. — [Meckes 2012, measure-theoretic Dvoretzky theorem](https://doi.org/10.1007/978-3-642-29849-3_18)
  - Conditional CLTs for i.i.d. Gaussian projection matrices, with bounds in quadratic Wasserstein distance and relative entropy. — [Reeves 2017, ISIT](https://doi.org/10.1109/ISIT.2017.8007089)
- **Visual consequence** [Claim]: "projected points are approximately Gaussian in most views, and increasingly concentrating near the center as dimensionality increases." — [Lee, Cook, da Silva, Laa, Spyrison, Wang & Zhang 2022, WIREs Comput. Stat. 14:e1573](https://doi.org/10.1002/wics.1573) (arXiv 2104.08016)

### Inferences
- **[Computed] JL is vacuous at k = 3.** With n = 600 tokens (ln n = 6.40), Dasgupta–Gupta needs 3ε² − 2ε³ ≥ 24·6.40/3 ≈ 51. But 3ε² − 2ε³ ≤ 1 on (0,1], so no ε < 1 satisfies it; even ε = 1 needs k ≥ 154. By Larsen–Nelson, no 3-D map (linear or not) can guarantee small distortion for all pairs of a general 600-point set.
- **[Computed] Norm retention under a random 3-D subspace.** E‖Pu‖² = k/p = 3/192 = 1.56% (the k/d expectation that Dasgupta–Gupta's proof starts from), so typical length is ≈ √(3/192) ≈ 12.5%. Against this baseline:
  - P3's 18.8% of beam energy is ~12× chance.
  - The median 47% of net-force length is ~3.8× the 12.5% chance level.
  - The 16% of the tool–background vector is near chance if it measures *length* (chance ≈ 12.5%). It is ~10× chance if it measures *energy*. Check which definition the app uses.
- **[Computed] Orthogonal beams that "look opposite" are at chance level.** For p ≫ k, the projections of two orthonormal 192-D vectors onto a random 3-D subspace are approximately independent isotropic Gaussian 3-vectors. In 3-D, the cosine between two independent isotropic directions is uniform on [−1, 1] (Archimedes' hat-box theorem), so P(cos < −t) = (1 − t)/2. That gives 15% for t = 0.70 and 13.5% for t = 0.73. The measured 13.5–15.5% rate of orthogonal beam pairs looking opposite is therefore exactly what a direction-agnostic 3-D projection produces. 3-D angles between projected beams carry no information about 192-D orthogonality unless both beams lie mostly inside P3.
- **[Computed] The Meckes threshold k < 2 log d / log log d at d = 192 is ≈ 6.3,** so 3-D views are in the regime where Gaussian-looking projections are typical *if* the data are isotropic enough (finite-sample constants unknown).
  - The Dümbgen conditions (⟨X,X′⟩/q → 0) and Meckes' B term mean the effect is governed by **effective** dimension, not nominal p.
  - **Beams** (PR ≈ 37): random 3-D views will be close to Gaussian blobs, so random or grand-tour views of the beam dictionary will rarely look structured.
  - **Tokens** (PR 10.6 → 3.15): late-epoch token clouds are not in the Diaconis–Freedman regime. Random views show a shrunken linear image of the ~3-D structure, but most random views miss its main axes (energy fraction ≈ 3/192 per direction).

### Gaps
- I found no explicit finite-sample statement of the form "for p = 192, k = 3, n = 600, the expected distance to Gaussian is ≤ X". The Meckes bounds contain unspecified absolute constants C.
- I did not locate the full text of Diaconis & Freedman (1984). The conditions above come from Dümbgen & Del Conte-Zerial, who prove they are necessary and sufficient.

---

## 3. Projection pursuit (Friedman & Tukey 1974; Huber 1985; holes, central mass, LDA/PDA and newer indices): how it finds "interesting" views

### Takeaway
Projection pursuit optimizes an index over orthonormal projection matrices A to find views that depart from Gaussianity (holes, central mass, multimodality) or that separate classes (LDA/PDA). The guided tour animates the search. Optimizers should find many local maxima, not only the global one. Sphering first removes the covariance structure that PCA already shows. In high dimension with finite n, apparent structure may be spurious unless sparsity or dimension reduction is imposed.

### Cited Findings
- **Origins** [Method]:
  - Friedman & Tukey introduced the PP algorithm for exploratory analysis. — [Friedman & Tukey 1974, IEEE Trans. Computers C-23:881–890](https://doi.org/10.1109/T-C.1974.224051)
  - Huber's review framed PP as a general framework. — [Huber 1985, Ann. Statist. 13:435–475](https://doi.org/10.1214/aos/1176349519)
  - Jones & Sibson's moment index. — [Jones & Sibson 1987, JRSS-A 150:1–36](https://doi.org/10.2307/2981662)
  - Friedman: "The goal of exploratory projection pursuit is to use the data to find low- (one-, two-, or three-) dimensional projections that provide the most revealing views of the full-dimensional data." — [Friedman 1987](https://doi.org/10.1080/01621459.1987.10478427)
- **Holes and central-mass indices (sphered data, Y = XA, n × d)** [Definition]:
  - I_holes(A) = [1 − (1/n)Σ exp(−½ yᵢᵀyᵢ)] / [1 − exp(−d/2)]
  - I_CM(A) = [(1/n)Σ exp(−½ yᵢᵀyᵢ) − exp(−d/2)] / [1 − exp(−d/2)]
  - Holes finds gaps between clusters, central mass reveals outliers, and LDA finds class-separating projections. — [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13), from the orthonormal-function-expansion family in [Cook, Buja & Cabrera 1993, JCGS 2:225–250](https://doi.org/10.1080/10618600.1993.10474610)
- **Legendre/Hermite indices** [Theorem]: these are weighted L² distances between the projected density and the standard normal. The weight function controls which departures from normality the index is sensitive to. — [Cook, Buja & Cabrera 1993](https://doi.org/10.1080/10618600.1993.10474610)
- **Guided tour** [Method]: "the next target basis is selected by optimizing a function defining interesting projections". The purpose of the optimization "is to find all of the interesting projections, not only to find one global maximum, because often the local maxima can reveal unexpectedly interesting data structure". — [Cook, Buja, Cabrera & Hurley 1995, JCGS 4:155–172](https://doi.org/10.1080/10618600.1995.10474674); [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13)
- **Optimization difficulty: "squintability"** [Empirical]: Some indices have optima detectable only from close range (small "squint angle"). Higher squintability improved optimization success, while smoothness had no significant effect (tested for d = 4–12). The Jellyfish Search Optimizer is now in tourr. — [Zhang, Cook, Langrené & Leung, JCGS (2024 online; DOI as indexed)](https://doi.org/10.1080/10618600.2026.2614076)
- **Section pursuit** [Method]: an index on *slices*, comparing the distribution inside vs. outside a section, to find holes (low density) or grains (high density) near the centre that projections obscure. — [Laa, Cook, Buja & Valencia 2022, JCGS 31:739–752](https://doi.org/10.1080/10618600.2022.2035230)
- **Spurious structure** [Theorem] (structureless N(0, I_p) data, p/n → γ):
  - (i) γ = ∞: projections exist whose empirical CDF approximates **any** distribution; this extends to 2-D.
  - (ii) γ > 1: target G is attainable if μ₂(G) < γ − 1 and impossible if μ₂(G) > (1 + √γ)².
  - (iii) 0 < γ ≤ 1: mixtures (γ/L)G + (1 − γ/L)N(0,1) are attainable. Targets with ‖G − Φ‖_∞ > C√(γ log(1/γ)) are not.
  - (iv) γ = 0: all projections are asymptotically Gaussian.
  - Practical conclusion: "unless sparsity is enforced, and regardless of the chosen projection index, projection pursuit may detect an apparent structure that has no statistical significance." — [Bickel, Kur & Nadler 2018](https://doi.org/10.1073/pnas.1801177115)
- **Grassmannian Atlas** [Claim]: ranking candidate projections by one quality score selects "multiple highly ranked but redundant (similar) projections ... lower ranked ones are discarded even though they may contain complementary information". The authors instead analyse the topology of the quality measure over a sample of the Grassmannian. — [Liu, Bremer, Thiagarajan, Wang, Summa & Pascucci 2016, Computer Graphics Forum (EuroVis)](https://doi.org/10.1111/cgf.12876)

### Inferences
- **[Computed] Spurious-structure bound for our data.** For tokens, γ = 0.32, so Bickel et al.'s exclusion threshold is C·√(0.32·ln 3.125) ≈ 0.60·C, with C an unknown universal constant. The asymptotics allow a non-Gaussian component in optimized 1-D views of pure noise. Any PP- or ICA-chosen axis for the token cloud or beam set needs a permutation/Gaussian null before being read as "a trap axis" or "a beam family".
- **Index choice should match the question.**
  - Hollowness or rings in the token cloud: holes index or section pursuit.
  - Tool vs. background tokens: LDA/PDA index.
  - Beam families: a central-mass or clustering index on the beam direction cloud, after PCA reduction to the ~37 effective dims, which also mitigates the γ problem.

### Gaps
- No PP index found that scores a view by how well it represents a *vector field*, e.g. retained force norm or the sign structure of the Jacobian. The nearest is the flow-preserving projection of Bartolovic et al. 2020 (see section 6).
- Friedman & Tukey's original index formula (trimmed spread × local density) is not quoted from the source here. Its abstract was unavailable in the tools used.

---

## 4. Tours: grand, guided, manual/radial, slice, burning sage; tourr; projecting vs. slicing; how long a grand tour must run

### Takeaway
A tour is an animated, geodesically interpolated sequence of d-dimensional projections. The grand tour picks random targets and becomes dense in the Grassmannian, but no practical stopping rule exists. Asimov tabulated only up to 20 dimensions, and the number of samples for fixed density grows exponentially with dimension; Gr(3,192) has dimension 567. Guided tours steer by a PP index. Manual/radial tours rotate one variable in or out, and a user study found them clearly more accurate for variable-importance judgements than PCA. Slice tours and sage displays address the "piling at the centre" artefact. Slice thickness must grow toward the data radius as dimension rises, so slicing is only practical in a low effective dimension.

### Cited Findings
- **Grand tour** [Method/Claim]: "orthogonal projections onto a sequence of two-dimensional subspaces ... chosen so that it is dense in the set of all two-dimensional subspaces"; Asimov provides "tabulations ... of the minimum length of a grand tour sequence necessary to achieve various degrees of denseness in dimensions up to 20". — [Asimov 1985, SIAM J. Sci. Stat. Comput. 6:128–143](https://doi.org/10.1137/0906011)
- **Random targets** [Method]: the default target is a random frame (standardized Gaussian vectors, orthonormalized). "A grand tour may be considered to be an interpolated random walk over the space of all planes." It "shows all projections ... This requires time and we may spend a lot of time wandering around boring places and miss the highlights." — [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13)
- **Interpolation** [Method]: geodesic interpolation between planes "conserves the orthonormality of the projection vectors" and eliminates within-plane spin. — [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13); [Buja, Cook, Asimov & Hurley 2005, Handbook of Statistics 24:391–413](https://doi.org/10.1016/S0169-7161(04)24014-7); [Asimov & Buja 1994, SPIE](https://www.nas.nasa.gov/assets/nas/pdf/techreports/1994/rnr-94-004.pdf). Frame-to-frame (Givens-rotation) interpolation can reach an exact target *frame*, not just a plane, which matters when an index or display is not rotation-invariant. — [Batsaikhan, Cook & Laa, woylier, arXiv 2311.08181 / R Journal](https://arxiv.org/abs/2311.08181)
- **Coverage cost** [Empirical]: Liu et al. note that the Grassmannian Gr(r,n) has dimension r(n − r). Keeping a fixed nearest-neighbour density (d_ann ≈ 0.3) "requires the sample count to grow exponentially with dimension". A good Grassmannian sample "does not necessarily guarantee that a given metric defined on the Grassmannian is well sampled". Their examples use 1,500–10,000 samples. — [Liu et al. 2016](https://doi.org/10.1111/cgf.12876)
- **Equi-spaced targets** [Method]: Grassmannian packings choose N maximally separated subspaces, and a shortest (Hamiltonian/TSP) circuit through them gives a "planned" tour. The authors note the application to Asimov's grand tour. — [Conway, Hardin & Sloane 1996, Experimental Mathematics (arXiv math/0208004)](https://arxiv.org/abs/math/0208004); described in [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13). The little tour (McDonald 1982) interpolates between the frames of a scatterplot matrix.
- **Tour family and software** [Method]:
  - tourr implements grand, guided, little, local, frozen, section (slice) and sage tours, with 1-D/2-D/3-D/d-D displays and saved paths. — [Wickham, Cook, Hofmann & Buja 2011, JSS 40(2):1–18](https://doi.org/10.18637/jss.v040.i02); [Lee et al. 2022 review](https://doi.org/10.1002/wics.1573)
  - detourr (Three.js/WebGL) gives smooth tours for large data, **extends the sage radial transform to 3-D**, and simplifies the slice tour. — [Hart & Wang 2023, R Journal 15(2):307–329](https://doi.org/10.32614/RJ-2023-052)
  - langevitour produces tours by Langevin dynamics, with a guide and manual variable positioning. — [Harrison 2022, bioRxiv](https://doi.org/10.1101/2022.08.24.505207)
- **Manual tour** [Method]: the user controls one variable's projection coefficient and rotates it fully in or out, which tests how sensitive the visible structure is to that variable. — [Cook & Buja 1997, JCGS 6:464–480](https://doi.org/10.1080/10618600.1997.10474754); spinifex in R: [Spyrison & Cook 2020, R Journal 12(1)](https://doi.org/10.32614/RJ-2020-027). Newer manual controls also let the user shift a *slice centre*. — [Laa, Aumann, Cook & Valencia 2023, JCGS](https://doi.org/10.1080/10618600.2023.2206459)
- **Radial tour user study** [Empirical]: 108 crowdsourced participants, 648 trials, task = identify variables contributing to two-cluster separation in 4-D or 6-D data. The radial tour gave +0.14 accuracy over PCA (SE 0.04, t = 3.77, p < .001). The grand tour gave +0.06 (n.s., p = .104). The advantage was largely lost when the separation signal was mixed 33/66 across variables (interaction −0.12, p = .033). The radial tour was the most preferred method. — [Spyrison, Cook & Marriott 2023, arXiv 2301.00077](https://arxiv.org/abs/2301.00077)
- **Piling / burning sage** [Theorem + method]:
  - For a uniform p-ball of radius R, the fraction of points whose 2-D projected radius is ≤ r is v₂ = 1 − (1 − (r/R)²)^{p/2}. At r/R = 0.5 this is 0.763 for p = 10 and 0.9999994 for p = 100, while the p-D volume fraction is (r/R)^p.
  - The sage display remaps radius r′ = R·√v₂(r), with tuning p_eff = γp and trimming R.
  - "The burning sage transformation is global and does not magnify local structure like t-SNE does" and "with more center magnification, the non-linear transformation can introduce distortions". — [Laa, Cook & Lee 2022, JCGS 31:40–49](https://doi.org/10.1080/10618600.2021.1963264)
- **Slice tour** [Method]:
  - Orthogonal distance of xᵢ to the projection plane: ṽᵢ = ‖xᵢ − Σⱼ(xᵢ·aⱼ)aⱼ‖. Points with ṽᵢ < h are highlighted.
  - For a uniform p-ball, the slice volume fraction is V_rel ≈ ½(h/R)^{p−2}, so h = ε^{1/(p−2)} with default ε = 0.1. This gives h = 0.1 in 3-D and 0.46 in 5-D.
  - An anchor c shifts the slice off-centre.
  - Slices reveal hollowness, concavity and nonlinear class boundaries hidden in projections.
  - Source: [Laa, Cook & Valencia 2020, JCGS 29:681–687](https://doi.org/10.1080/10618600.2020.1777140)
- **Projection vs. section** [Theorem/Claim]: "Projections can easily display aspects of structure that are only of low dimension, and sections ... can easily display structure of only low codimension (and hence often high dimension). However, compositions of sections and projections, here called prosections, can display aspects of structure of any intermediate dimension." Brushing approximates sections. — [Furnas & Buja 1994, JCGS 3:323–353](https://doi.org/10.1080/10618600.1994.10474649)
- **Tours for neural networks** [Claim/Empirical]: linear tours keep "data-visual correspondence", so changes in data and picture "match in magnitude". Views can be aligned across fully connected layers via the SVD of the layer. The approach is "only practical for a small dimensionality (e.g. up to 1000)", and with hundreds of dimensions there are "too many axis handles to naturally interact with". — [Li, Zhao & Scheidegger 2020, Distill](https://doi.org/10.23915/distill.00025)
- **Motion helps** [Empirical, n = 10]: in Subspace Voyager (3-D subspaces + trackball), no participant found a hidden "stick" in a scatterplot matrix (SPLOM), but all did with the trackball (mean 48 s). All participants found a third cluster only while *transitioning* between two saved views, which they had judged to contain two clusters. — [Wang & Mueller 2018, IEEE TVCG 24(3):1204–1222](https://doi.org/10.1109/TVCG.2017.2672987)

### Inferences
- **[Computed] Grand-tour coverage at our dimension.** Gr(3,192) has dimension 3·189 = 567, and Gr(2,192) has 380. Covering numbers of a manifold of dimension D scale like (c/ε)^D (standard metric-entropy reasoning; consistent with the exponential growth Liu et al. measured). A grand tour in the raw 192-D space will never come close to "all" 3-D views in usable time. Practical recommendation: tour inside a pre-reduced subspace, e.g. the top 6–10 PCs of the relevant quantity (Gr(3,10) has dimension 21), or use guided, planned or packing-based target sets.
- **[Computed] Piling at p = 192** (2-D formula for a uniform ball): v₂(r/R = 0.2) = 1 − 0.96^96 ≈ 0.98 and v₂(r/R = 0.1) ≈ 0.62. 98% of points land within 20% of the radius, which motivates the 3-D sage transform in detourr for the beam cloud.
- **[Computed] Slice thickness at p = 192** (Laa et al.'s 2-D-slice rule): h = 0.1^{1/190} ≈ 0.988R. A slice holding 10% of uniformly spread volume is nearly the full ball, so slices are informative only if the data are effectively low-dimensional *and* the slice subspace is aligned with the data. The app's "3-D-slice fields" therefore need a slice subspace aligned with the token cloud (token PCs, PR ≈ 3), not with the beam PCs (P3). They should also report each token's orthogonal distance ṽᵢ, so the viewer knows which tokens actually lie in the slice.
- For "which beams matter for this trap", the radial tour is the best-evidenced interactive control: rotate one beam direction v_c (or one beam cluster's mean direction) into or out of the 3-D view.

### Gaps
- I could not access Asimov's (1985) actual coverage tables (no PDF), so there are no numbers for dimensions ≤ 20.
- I found no published grand-tour run-length rule for p > 20, and no user study of tours at p ≈ 200. Spyrison's study used p = 4–6, and Distill notes interaction breaks down with hundreds of axes.
- The 3-D generalization of the sage radial CDF is implemented in detourr, but I did not verify its formula.

---

## 5. Small multiples and designed sets of views: scatterplot matrices, generalized/designed view sets, combinatorial designs (Steiner triple systems, Fano plane)

### Takeaway
The literature designs view sets with graph theory (Hamiltonian decompositions and Eulerian tours covering all variable pairs in about p/2 parallel-coordinate orderings), Grassmannian packings, diversity-aware optimal sets (≤ n/2 2-D projections), and topological atlases of the Grassmannian. I found **no** visualization paper that uses Steiner triple systems or the Fano plane to cover all axis pairs with 3-D views. The idea is combinatorially sound but appears unexplored. Applied to 192 raw axes it would need 6,144 views, so it only makes sense on a small set of meaningful axes (7 → 7 views, 9 → 12, 13 → 26, 37 → 222).

### Cited Findings
- **SPLOMs and brushing** [Method]: the scatterplot matrix is the set of all 2-D coordinate marginals. — [Becker & Cleveland 1987, Technometrics 29:127–142](https://doi.org/10.1080/00401706.1987.10488204); [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13). For selecting among many views:
  - Ranking by features. — [Seo & Shneiderman 2005, Information Visualization 4:96–113](https://doi.org/10.1057/palgrave.ivs.9500091)
  - Graph-theoretic scagnostics. — [Wilkinson, Anand & Grossman 2005, IEEE InfoVis](https://doi.org/10.1109/INFVIS.2005.1532142)
- **Pair coverage by graph decompositions** [Theorem/Method]: About p/2 axis permutations make every variable adjacent to every other in some permutation of a parallel-coordinate plot (Wegman's zig-zag, related to the Lucas–Walecki construction). — [Wegman 1990, JASA 85:664–675](https://doi.org/10.1080/01621459.1990.10474926), as summarized by [GenStat KB](https://genstat.kb.vsni.co.uk/knowledge-base/dparalle/) and [Oldford slides](https://www.math.uwaterloo.ca/~rwoldfor/talks/Bremen/dviz.pdf). Further detail:
  - For K_n with n = 2m+1 there are m Hamiltonian cycles; with n = 2m there are m Hamiltonian paths. — [Oldford slides](https://www.math.uwaterloo.ca/~rwoldfor/talks/Bremen/dviz.pdf)
  - Eulerian tours and Hamiltonian decompositions are used to order components of a display; PairViz implements them, including "a road map for dynamic scatterplots". — [Hurley & Oldford 2010, JCGS 19:861–886](https://doi.org/10.1198/jcgs.2010.09136); [Hurley & Oldford 2011, Comput. Stat. 26:613–633](https://doi.org/10.1007/s00180-011-0229-5)
- **Optimal sets of projections** [Method/Claim]: a data-driven way to find *minimal* sets of projections that uniquely show data patterns, using a dissimilarity that ignores affine transformations so patterns are not repeated. It gives "complete data tours of at most n/2 projections" (n = data dimension). — [Lehmann & Theisel 2016, IEEE TVCG 22:609–618](https://doi.org/10.1109/TVCG.2015.2467132) (abstract only)
- **Axis-aligned decomposition** [Empirical]: "in practice the information of k linear projections is often jointly encoded in ∼k axis-aligned plots". — [Thiagarajan, Liu, Ramamurthy & Bremer, Computer Graphics Forum (online 2017)](https://doi.org/10.1111/cgf.13416)
- **Grassmannian Atlas** [Method]: explores locally optimal, *structurally distinct* projections instead of the top-k ranked ones. — [Liu et al. 2016](https://doi.org/10.1111/cgf.12876)
- **Subspace Voyager** [Method]: decomposes the space into a "continuum of generalized 3D subspaces" (random or ACO-optimized projections, or per-cluster top-3 PCs). It notes the number of subspaces "suffers from combinatorial explosion" and gives no recommended count. — [Wang & Mueller 2018](https://doi.org/10.1109/TVCG.2017.2672987)
- **Packings for tour targets** [Method]: [Conway, Hardin & Sloane 1996](https://arxiv.org/abs/math/0208004)
- **Combinatorics** [Theorem]: a Steiner triple system STS(v) (each pair in exactly one triple) exists iff v ≡ 1 or 3 (mod 6). The minimum covering number C(v,3,2) equals the Schönheim bound for all v > 3 (Fort & Hedlund 1958). — [ETSU design-theory notes (Lindner & Rodger)](https://faculty.etsu.edu/gardnerr/Design-Theory/notes-Design-Theory-LR2/Design-Theory-LR2-4-1.pdf); [Yugoslav J. Oper. Res. paper citing Fort–Hedlund](https://yujor.fon.bg.ac.rs/index.php/yujor/article/download/28/22)
- **Closest visualization prior art** [Patent]: a patent describes "a plurality of 3D scatter plots representing the combination of the three state members taken three at a time" (all triples). It does not mention Steiner or covering designs. — [USPTO patent 6750864](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6750864)

### Inferences
- **[Computed] View counts for covering all pairs of v axes with 3-D views** (each view covers 3 pairs):
  - v = 7: Fano plane, 7 views.
  - v = 9: 12 views.
  - v = 13: 26 views.
  - v = 15: 35 views.
  - v = 37 (≈ beam effective dimension, 37 ≡ 1 mod 6): STS(37) gives 37·36/6 = **222** views.
  - Raw v = 192 (≡ 0 mod 6, no STS): Schönheim bound ⌈(192/3)·⌈191/2⌉⌉ = **6,144** views, against 18,336 pairs in a 2-D SPLOM.
- **[Computed] Lighter alternatives to pair covering:**
  - A partition into 64 mutually orthogonal 3-D blocks shows every direction's energy exactly once (no pair coverage across blocks). This is the 3-D analogue of Lehmann–Theisel's ≤ n/2 2-D projections.
  - A Hamiltonian-style ordering of the 37 beam-PCs gives about 19 overlapping triples in which consecutive axes share a view.
- **Axes worth covering.** Covering pairs of *raw residual coordinates* is probably meaningless, because the residual basis is not obviously privileged (an inference; not tested here). The axes worth pairing are meaningful directions: token PCs, force PCs, the tool–background axis, beam-cluster centroids, and Jacobian eigen- or Schur-vectors at traps. With a small curated set (7 or 9 axes), a Fano or STS(9) design (7 or 12 linked 3-D views) is a clean, defensible way to guarantee every pair of key directions appears together in one view. This would be a methodological novelty worth stating as such.

### Gaps
- I found no peer-reviewed use of Steiner triple systems, Fano planes, or covering designs to choose 3-D data views. This was searched across Undermind (176 papers) and the web.
- No user study compares designed 3-D view sets with tours or with PCA for vector-field reading.

---

## 6. Linear projections and vector sums; biplots (Gabriel 1971; Gower & Hand); loss of norm; displaying forces and fields

### Takeaway
Any linear map P satisfies P(Σ_c u_c) = Σ_c P u_c. The projected net force is therefore exactly the sum of projected beam contributions, and contributions that cancel in 192-D also cancel in 3-D. The converse fails: lengths, angles and apparent cancellations in 3-D are not those of 192-D, because the orthogonal components are invisible. Biplots are the classical theory for drawing points and variable vectors together; their accuracy is set by the rank of the approximation. A projected vector field is not the vector field of a 3-D system. Its zeros and its 3 × 3 Jacobian compression P J Pᵀ show the true trap type only when the view subspace is (approximately) invariant under J. Field-preserving projections exist but are nonlinear optimizations.

### Cited Findings
- **Biplot theory** [Theorem]: "Any matrix of rank two can be displayed as a biplot which consists of a vector for each row and a vector for each column, chosen so that any element of the matrix is exactly the inner product of the vectors corresponding to its row and to its column. If a matrix is of higher rank, one may display it approximately by a biplot of a matrix of rank two." In PCA the biplot shows inter-unit distances, clustering, and variable variances and correlations. — [Gabriel 1971, Biometrika 58:453–467](https://doi.org/10.1093/biomet/58.3.453)
- **3-D biplots** [Method]: Gower's 3-D biplot for bilinear models places row-effect points and column-effect points in orthogonal planes, with "implications for interactive computer graphics". — [Gower 1990, Biometrika 77:773–785](https://doi.org/10.1093/biomet/77.4.773). Biplots "can just as well be displayed in three-dimensions, with more accurate data representation", and rest on the SVD as a two-sided regression. — [Greenacre 2012, "Biplots: the joy of singular value", WIREs Comput. Stat.](https://www.semanticscholar.org/paper/9ce06241d274a3e37bae6aab32f031e20eaed6b5). The monograph treatment is [Gower, Lubbe & le Roux 2011, *Understanding Biplots*, Wiley](https://doi.org/10.1002/9780470973196).
- **Variable vectors in tours** [Method]: tours and manual tours draw the projected unit axis of each variable as a handle. Dragging handles redefines the projection (re-orthonormalized by Gram–Schmidt). — [Li et al. 2020 Distill](https://doi.org/10.23915/distill.00025); [Cook & Buja 1997](https://doi.org/10.1080/10618600.1997.10474754); [Wang & Mueller 2018](https://doi.org/10.1109/TVCG.2017.2672987)
- **Linear sums in NN visualization** [Claim]: because "projections are linear and the coefficients of vectors in the classification layer sum to one", a 50/50 prediction lands midway between two class handles. Linearity makes compositions predictable. — [Li et al. 2020](https://doi.org/10.23915/distill.00025)
- **Norm loss under random projection** [Theorem]: the expected squared length of a fixed vector projected onto a random k-subspace of R^d is (k/d) of the original, with exponential concentration (the basis of JL). — [Dasgupta & Gupta 2003](https://doi.org/10.1002/rsa.10073)
- **Flow-preserving projection** [Method]: an energy-minimizing projection operator for high-dimensional dynamical systems "maintains a number of useful flow properties, such as critical points and flow maps", is matched to the geometry of the high-dimensional input, and is compared with PCA, t-SNE and UMAP on second-order systems (three-body problem, damped double pendulum). — [Bartolovic, Gross & Günther 2020, CGF 39(3):253–264](https://doi.org/10.1111/cgf.13978) (abstract only; I could not confirm whether the operator is linear)
- **Linearization of high-dimensional dynamics around fixed points** [Method]: fixed and slow points of RNNs are found numerically, and the dynamics are linearized at them, with the low-dimensional structure shown in PC coordinates. — [Sussillo & Barak 2013, Neural Computation 25:626–649](https://doi.org/10.1162/NECO_a_00409). jPCA chooses projection planes from a fitted skew-symmetric dynamics matrix, i.e. planes chosen for their dynamics rather than variance. — [Churchland et al. 2012](https://doi.org/10.1038/nature11129)
- **Compression eigenvalues** [Theorem]: if P has orthonormal columns and PᵀAPx = λx with ‖x‖ = 1, then (Px)*A(Px) = λ. Every eigenvalue of the compression PᵀAP therefore lies in the numerical range W(A). — [search-assistant derivation; see compression literature, e.g. Adam & Maroulas 2010, Operators and Matrices](https://oam.ele-math.com/04-07/Limited-approximation-of-numerical-range-of-normal-matrix). Caveat: finer containment properties do not hold for all compressions. — [Gau & Wu 2013](https://ir.lib.nycu.edu.tw:443/handle/11536/22307)

### Inferences
- **Vector sums: safe.** Σ_c P u_c = P F exactly, so stacking projected beam arrows head-to-tail always ends at the projected net force. The measured "median 47% of net-force length" is the norm loss of F itself (‖PF‖/‖F‖), not an error of summation.
- **Norms and angles: not safe.**
  - ‖Pu‖ = ‖u‖·cos(angle between u and the subspace), so arrow length mixes magnitude with alignment to the view. Show each arrow's *retained fraction* ‖Pu‖/‖u‖, e.g. as opacity or a ring, alongside it. This is the vector analogue of the per-variable "quality of representation" used in PCA biplots.
  - Apparent 3-D cancellation (Pu ≈ −Pv) does not mean u ≈ −v (see the chance computation in section 2).
- **Biplot reading for beams.**
  - The pre-activation z_c = w_cᵀx + b is an inner product, so a biplot of tokens (rows) and fc1 rows w_c (columns) in the same 3-D subspace shows z_c exactly only for the in-subspace parts. The error is w_c⊥ᵀ x⊥.
  - Output directions v_c (fc2 columns) are the "column markers" of a second biplot.
  - A view chosen for tokens (token PCs) and one chosen for beams (P3) cannot both be biplot-accurate unless the subspaces coincide. Hence the case for linked views (section 5) or a joint objective (section 1).
- **Field topology in 3-D slices.**
  - The slice field g(y) = P F(x₀ + Pᵀy) has Jacobian P J Pᵀ, the 3 × 3 compression of the 192 × 192 J.
  - Its eigenvalues lie in W(J) but are generally *not* eigenvalues of J. A 3-D "attractor", "saddle" or "vortex" can be an artefact of the compression, especially for non-normal J.
  - Zeros of P F form a set of codimension 3 in 192-D (a 189-D set), so "fixed points" seen in a projected field need not be fixed points of F.
  - Trap type is preserved exactly when the view subspace is J-invariant, e.g. spanned by 3 real eigenvectors, or by the 2-D real invariant plane of a complex eigenpair plus one more Schur vector. Then the compression's eigenvalues are a subset of J's (standard linear-algebra fact). Recommendation: at each detected trap, offer a *local* 3-D view from the real Schur or eigen-basis of J, in the spirit of Sussillo–Barak linearization and jPCA rotation planes, and report the residual ‖(I − PPᵀ)F‖ next to every projected field.
- **Nonlinear transforms.** Sage (radial) and other nonlinear display transforms break vector sums, so do not apply them to arrows. Use them only for point density.

### Gaps
- Gower, Lubbe & le Roux's per-axis and per-sample "predictivity/adequacy" measures (fraction of each variable's variation reproduced by the display) are not quoted here from full text. Verify them before citing.
- No paper was found on biplot-style display of a *dictionary* of 768 vectors weighted by data-dependent activations (our beams).
- I could not read the Bartolovic et al. full text (HTTP 403), so the energy terms and whether the projection is linear are unconfirmed.

---

## 7. Time series of projections: keeping camera/axes continuous across checkpoints (Procrustes, aligned PCA, principal angles)

### Takeaway
There are three options.
1. Keep **one fixed projection** for all checkpoints (what P3 does). This is perfectly continuous, and the tour literature recommends it for comparing epochs, but it is optimal for no single epoch.
2. Recompute a projection per checkpoint and **align** it to the previous one with orthogonal or generalized Procrustes. This removes sign flips and rotations of PCA bases.
3. Find a **common basis** for all snapshots (common principal components) and quantify drift with principal angles.

Smooth geodesic interpolation between consecutive subspaces turns a per-epoch view sequence into a "tour through training".

### Cited Findings
- **Fixed linear views across epochs** [Claim]: linear projections keep data and visual changes matched in magnitude. Nonlinear embeddings with "non-convex objectives also tend to be sensitive to initial conditions", which makes epoch-to-epoch comparison unreliable. — [Li et al. 2020 Distill](https://doi.org/10.23915/distill.00025)
- **Incremental PCA + Procrustes** [Method]: incremental PCA can produce "an arbitrary rotation and/or flipping of data points at subsequent time steps". "To address both the flipping and arbitrary rotation, we apply the Procrustes transformation" (translation, uniform scaling, and an orthogonal rotation/reflection from an SVD, computed on points shared between steps). It "reduces unnecessary changes across the time points" without changing relative distances, and is combined with staged animation to preserve the mental map. — [Fujiwara, Chou, Shilpika, Xu, Ren & Ma, IEEE TVCG (online 2019, VIS 2019 issue)](https://doi.org/10.1109/TVCG.2019.2934433) (full text via ar5iv)
- **Procrustes foundations** [Theorem/Method]:
  - Orthogonal Procrustes solution. — [Schönemann 1966, Psychometrika 31:1–10](https://doi.org/10.1007/BF02289451)
  - Generalized Procrustes analysis for aligning many configurations to a consensus (here, 26 checkpoints). — [Gower 1975, Psychometrika 40:33–51](https://doi.org/10.1007/BF02291478)
- **Principal angles** [Theorem/Method]: numerically stable computation of the angles between subspaces from the SVD of AᵀB, with θᵢ = arccos σᵢ. — [Björck & Golub 1973, Math. Comp. 27:579–594](https://doi.org/10.1090/S0025-5718-1973-0348991-3). The geodesic distance on the Grassmannian is (Σθᵢ²)^{1/2}. — [Liu et al. 2016](https://doi.org/10.1111/cgf.12876)
- **Common principal components** [Method]: one orthogonal basis shared by k covariance matrices (groups or snapshots), with group-specific variances. — [Flury 1984, JASA 79:892–898](https://doi.org/10.1080/01621459.1984.10477108)
- **Smooth transitions** [Method]: geodesic interpolation of planes ([Buja et al. 2005](https://doi.org/10.1016/S0169-7161(04)24014-7)), Givens-based frame-to-frame interpolation to land on an exact target frame ([Batsaikhan et al.](https://arxiv.org/abs/2311.08181)), and SVD-based transitions between saved 3-D subspaces ([Wang & Mueller 2018](https://doi.org/10.1109/TVCG.2017.2672987)). Distill aligns views across *layers* using the SVD of each linear layer. — [Li et al. 2020](https://doi.org/10.23915/distill.00025)
- **Nonlinear analogue** [Method]: dynamic t-SNE adds a temporal-coherence term across snapshots. — [Rauber, Falcão & Telea 2016, EuroVis short papers](https://doi.org/10.2312/eurovisshort.20161164)
- **Caution for trajectory-PCA axes** [Theorem/Empirical]: PCA of a parameter or representation trajectory over training yields Lissajous-like curves even for random walks. — [Antognini & Sohl-Dickstein 2018](https://arxiv.org/abs/1806.08805)

### Inferences
- **Recommended protocol for 26 checkpoints:**
  - (i) Compute per-epoch optimal subspaces for each quantity: token-PCA S_tok(e), force-PCA S_F(e), beam-weighted PCA S_beam(e).
  - (ii) Plot the principal angles θ₁..θ₃ between consecutive epochs and against P3. This directly shows *when* P3 stops being representative; the token PR falling 10.6 → 3.15 suggests the token subspace stabilizes late.
  - (iii) For display, either keep one consensus basis (generalized Procrustes or common PCA over epochs) or use per-epoch bases chained by orthogonal Procrustes (rotation/reflection only, no scaling, so lengths stay comparable), with geodesic interpolation between epochs.
  - (iv) Show the captured-energy fraction per epoch so the viewer can tell rotation of the content from rotation of the camera.
- Do not choose display axes by PCA of the *trajectory across epochs* without a random-walk null (Antognini & Sohl-Dickstein).

### Gaps
- I found no visualization paper that tracks a *vector field* (rather than points) through training checkpoints with aligned linear projections. Fujiwara et al. and Rauber et al. handle points only.
- I did not retrieve details of Flury's estimation algorithm or tests (only the citation).

---

## 8. Local / adaptive projections: local PCA, per-point explanatory projections, focus+context

### Takeaway
Local projections give each region, cluster, or point its own linear frame: local PCA, per-cluster 3-D subspaces, a per-observation explanation basis plus a radial tour, or off-centre slices. They are linear, so they preserve sums within one view, but they give up a common coordinate system. Focus+context in the tour literature means mostly global display transforms (sage) or slicing around an anchor, rather than fisheye lenses. For traps, the natural local frame is the invariant subspace of the local Jacobian.

### Cited Findings
- **Local PCA** [Method]: dimension reduction by partitioning the space and fitting PCA per region (mixture of local linear models). — [Kambhatla & Leen 1997, Neural Computation 9:1493–1516](https://doi.org/10.1162/neco.1997.9.7.1493)
- **Local affine (orthogonal) per-point maps** [Method]: LAMP builds an orthogonal affine map for each point from weighted control points. — [Joia, Coimbra, Cuminato, Paulovich & Nonato 2011, IEEE TVCG 17:2563–2571](https://doi.org/10.1109/TVCG.2011.220)
- **Per-cluster 3-D subspaces** [Method]: each cluster's top-3 PCs define its 3-D subspace, with transitions between adjacent subspaces ("cluster chasing" with Gram–Schmidt re-orthonormalization). — [Wang & Mueller 2018](https://doi.org/10.1109/TVCG.2017.2672987)
- **Per-observation explanatory projections** [Method]: the normalized local variable importance of a chosen observation (e.g. SHAP) is used as a 1-D projection basis. A radial tour then varies it to test where the explanation holds and to diagnose misclassifications, outliers and clusters (R package cheem). — [Spyrison, Cook & Biecek 2025, Comput. Stat. 40:1071–1095](https://doi.org/10.1007/s00180-023-01453-2)
- **Subset-driven / locally enhanced projections** [Method]: point-wise quality metrics (outliers, clusters, trends) accumulated over a user-selected subset recommend the 2-D linear projection where that subset's pattern is most salient. — [Xia et al. 2018, J. Vis. Lang. Comput., pp. 52–60](https://doi.org/10.1016/j.jvlc.2018.08.003). See also [Lai, Zhao & Yuan 2018, J. Vis. Lang. Comput., pp. 144–156](https://www.semanticscholar.org/search?q=Exploring%20high-dimensional%20data%20through%20locally%20enhanced%20projections) (abstract not retrieved).
- **Focus+context in tours** [Method]:
  - Sage radial transform: global, user-tunable centre magnification. — [Laa, Cook & Lee 2022](https://doi.org/10.1080/10618600.2021.1963264)
  - Anchored slices and interactive slice-centre controls. — [Laa et al. 2020](https://doi.org/10.1080/10618600.2020.1777140); [Laa et al. 2023](https://doi.org/10.1080/10618600.2023.2206459)
  - Local tour that "rocks back and forth from a particular plane to randomly chosen targets in a small neighborhood". — [Lee et al. 2022 review](https://doi.org/10.1002/wics.1573)
- **Explaining nonlinear projections with derivatives** [Method]: DimReader perturbs inputs and propagates tangent vectors through a projection to draw "axis lines". This is a way of drawing *derivative/vector* information through a projection. — [Faust, Glickenstein & Scheidegger 2019, IEEE TVCG](https://doi.org/10.1109/TVCG.2018.2865194)

### Inferences
- **Trap-centred local view.** For a selected trap or token, build the 3-D frame from:
  - (a) the real Schur or eigen-basis of the local 192 × 192 Jacobian, which preserves trap type (section 6), or
  - (b) local force-PCA over the token's neighbourhood, or
  - (c) the top-3 beams by |GELU(z_c)|·‖v_c‖ at that token, orthonormalized: a per-point "explanatory projection" analogous to cheem's local-explanation basis.
- Keep the global view and add a radial-tour slider that rotates the local frame toward or away from the global P3 (the focus+context transition). Report captured-energy fractions in both frames.
- Local frames preserve sums only within one frame. Arrows from different local frames must not be visually added.

### Gaps
- I found no paper that uses Jacobian invariant subspaces as the per-point projection frame in an interactive multi-dimensional-data visualization tool. Sussillo & Barak, and jPCA, do this offline in neuroscience. This looks like an opening for our app.
- Lai, Zhao & Yuan (2018) was not read beyond its title.

---

## 9. Cross-cutting: what each method preserves — (a) linear structure / vector sums, (b) distances, (c) neighbourhoods, (d) dynamics / field topology — and how many views

### Takeaway
All linear orthogonal projections preserve (a) exactly, (b) only as a contraction, and (c) only in one direction: true neighbours stay close, but false neighbours appear. They preserve (d) only on invariant subspaces. Choosing the method means choosing which loss to minimize, not escaping loss. The literature does not give a universal number of views. Concrete anchors from the literature are ≤ n/2 projections for a "complete" set, about k axis-aligned plots for k linear ones, about p/2 Hamiltonian orderings for all pairs, and exponentially many samples to cover the Grassmannian.

### Cited Findings
- **Linear projections are global linear maps** [Definition]: Y = XA with orthonormal A. [Cook et al. 2008](https://doi.org/10.1007/978-3-540-33037-0_13); [Lee et al. 2022](https://doi.org/10.1002/wics.1573). Points at very different perpendicular distances "can overlap in the same 2D view", and "a lot is easily hidden, especially in the case of concave structures". — [Laa et al. 2020 slice tour](https://doi.org/10.1080/10618600.2020.1777140)
- **No 3-D distance guarantee** [Theorem]: [Larsen & Nelson 2017](https://doi.org/10.1109/FOCS.2017.64)
- **Field-preserving alternative** [Method]: [Bartolovic et al. 2020](https://doi.org/10.1111/cgf.13978)
- **View-count anchors** [Claim/Empirical]:
  - ≤ n/2 projections. — [Lehmann & Theisel 2016](https://doi.org/10.1109/TVCG.2015.2467132)
  - ∼k axis-aligned plots for k linear projections. — [Thiagarajan et al. 2018](https://doi.org/10.1111/cgf.13416)
  - ≈ p/2 orderings for all pairs. — [Wegman 1990](https://doi.org/10.1080/01621459.1990.10474926); [Hurley & Oldford 2010](https://doi.org/10.1198/jcgs.2010.09136)
  - Exponential Grassmannian sampling. — [Liu et al. 2016](https://doi.org/10.1111/cgf.12876)
  - Grand-tour density tables only up to p = 20. — [Asimov 1985](https://doi.org/10.1137/0906011)

### Inferences
**Preservation matrix.** "Yes" means exact preservation. "Contracts" means distances can only shrink. "False nbrs" means non-neighbours can overlap in the view.

| Method | (a) vector sums / linear | (b) distances | (c) neighbourhoods | (d) dynamics / field topology | Vectors OK? |
|---|---|---|---|---|---|
| PCA (points) | Yes | Contracts; max. mean squared spread | False nbrs | No (only if subspace is invariant) | Yes, but optimal for point spread, not vectors |
| Weighted / force-PCA | Yes | Contracts | False nbrs | No (only if invariant) | **Best for drawn-vector energy** |
| Sparse PCA | Yes | Contracts (worse than PCA) | False nbrs | No | Yes; interpretable loadings; robust to spurious PP |
| PPCA | Yes (ML subspace = PCA) | Contracts | False nbrs | No | Yes; adds noise model and k selection |
| LDA / PDA index / class-mean axis + residual PCA | Yes | Contracts (maximizes between/within ratio) | False nbrs | No | Yes; guarantees the separation vector is visible |
| Contrastive PCA / dPCA | Yes | Contracts | False nbrs | No | Yes; target vs. background |
| CCA / PLS / reduced-rank regression | Yes (two linear maps) | Not preserved; variance can be rescaled | — | Partially: captures dominant input–output modes of a linear map | Yes, for displacement–force coupling |
| ICA / projection pursuit | Yes | Contracts | False nbrs | No | Yes, but spurious non-Gaussianity at γ = 0.25–0.32 |
| jPCA / Jacobian Schur/eigen-basis / DMD-like | Yes | Contracts | False nbrs | **Yes on an exactly invariant subspace; approximately otherwise** | Best for trap/vortex type |
| Random projection (JL) | Yes | **No guarantee at k = 3** | False nbrs | No | Arrows keep ≈ 12.5% of length on average |
| Grand / guided / manual / radial tour | Yes (each frame) | Contracts (each frame) | False nbrs (motion helps) | No | Yes; arrows can be drawn per frame |
| Slice tour | Yes (in-slice) | In-slice only | **Removes false nbrs** within thickness h | Local section; still the compression P J Pᵀ | Yes; h ≈ R at 192-D unless effective dim is low |
| Sage display | **No** (nonlinear radial) | No | Angular order kept | No | **Do not use for arrows** |
| Local PCA / LAMP / per-cluster 3-D | Yes within one frame only | Locally | Locally | Locally, if frame = local invariant subspace | Within-frame only |
| Flow-preserving optimized projection (Bartolovic) | Unconfirmed | Optimized trade-off | — | **Critical points and flow maps by design** | Designed for trajectories |

**How many views for our system.** These are suggestions combining the literature anchors with our numbers.
- **Fixed views (always shown):**
  1. A *token* view: token-PCA, ~3 dims late in training (PR 3.15).
  2. A *force* view: force-PCA or a class-axis + residual-force-PCA hybrid.
  3. Optionally a *beam* view: activation-weighted beam PCA (≈ P3, improved).
  Linking these three views covers the three different optimal subspaces.
- **Exploration of the beam space:** a guided or radial tour within the top ~6–10 beam PCs (Gr(3,10) has dimension 21, versus 567 raw), or a Fano / STS(9) set of 7 / 12 linked 3-D views over a curated 7- or 9-direction basis.
- **Trap inspection:** one local view per trap from its Jacobian invariant subspace.
- **Every view:** show retained-energy fractions, so a 3-D picture is never read as the whole 192-D story.

### Gaps
- I found no empirical study of how many 3-D views users need to read vector fields from ≈ 200-D data. All view-count anchors above concern points (clusters, outliers) in 2-D views.
- The Lehmann–Theisel ≤ n/2 bound was taken from the abstract only, so its exact definition of "complete data tour" is unverified.
