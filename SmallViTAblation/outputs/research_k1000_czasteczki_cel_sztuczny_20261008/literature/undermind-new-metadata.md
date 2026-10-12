[Gra25c] Addressing divergent representations from causal interventions on neural networks (2025)
ArXiv
Date: 2025-11-06
By Satchel Grant, S. Han, Alexa R. Tartaglini, Christopher Potts - 5 citations - 5.4 cit./yr
DOI: 10.48550/arXiv.2511.04638
PDF ✓

A common approach to mechanistic interpretability is to causally manipulate model representations via targeted interventions in order to understand what those representations encode. Here we ask whether such interventions create out-of-distribution (divergent) representations, and whether this raises concerns about how faithful their resulting explanations are to the target model in its natural state. First, we demonstrate theoretically and empirically that common causal intervention techniques often do shift internal representations away from the natural distribution of the target model. Then, we provide a theoretical analysis of two cases of such divergences:"harmless"divergences that occur in the behavioral null-space of the layer(s) of interest, and"pernicious"divergences that activate hidden network pathways and cause dormant behavioral changes. Finally, in an effort to mitigate the pernicious cases, we apply and modify the Counterfactual Latent (CL) loss from Grant (2025) allowing representations from causal interventions to remain closer to the natural distribution, reducing the likelihood of harmful divergences while preserving the interpretive power of the interventions. Together, these results highlight a path towards more reliable interpretability methods.

[Gei23c] Don't trust your eyes: on the (un)reliability of feature visualizations (2023)
International Conference on Machine Learning, pp. 15294-15330
Date: 2023-06-07
By Robert Geirhos, R. Zimmermann, Blair Bilodeau, Wieland Brendel, Been Kim - 44 citations - 13 cit./yr
DOI: 10.48550/arXiv.2306.04719
PDF ✓

How do neural networks extract patterns from pixels? Feature visualizations attempt to answer this important question by visualizing highly activating patterns through optimization. Today, visualization methods form the foundation of our knowledge about the internal workings of neural networks, as a type of mechanistic interpretability. Here we ask: How reliable are feature visualizations? We start our investigation by developing network circuits that trick feature visualizations into showing arbitrary patterns that are completely disconnected from normal network behavior on natural input. We then provide evidence for a similar phenomenon occurring in standard, unmanipulated networks: feature visualizations are processed very differently from standard input, casting doubt on their ability to"explain"how neural networks process natural images. This can be used as a sanity check for feature visualizations. We underpin our empirical findings by theory proving that the set of functions that can be reliably understood by feature visualization is extremely small and does not include general black-box neural networks. Therefore, a promising way forward could be the development of networks that enforce certain structures in order to ensure more reliable feature visualizations.

[Hao20] LabelEnc: A New Intermediate Supervision Method for Object Detection (2020)
European Conference on Computer Vision, pp. 529-545
Date: 2020-07-07
By Miao Hao, Yitao Liu, X. Zhang, Jian Sun - 29 citations - 4.6 cit./yr
DOI: 10.1007/978-3-030-58595-2_32
PDF ✓

In this paper we propose a new intermediate supervision method, named LabelEnc, to boost the training of object detection systems. The key idea is to introduce a novel label encoding function, mapping the ground-truth labels into latent embedding, acting as an auxiliary intermediate supervision to the detection backbone during training. Our approach mainly involves a two-step training procedure. First, we optimize the label encoding function via an AutoEncoder defined in the label space, approximating the "desired" intermediate representations for the target object detector. Second, taking advantage of the learned label encoding function, we introduce a new auxiliary loss attached to the detection backbones, thus benefiting the performance of the derived detector. Experiments show our method improves a variety of detection systems by around 2% on COCO dataset, no matter one-stage or two-stage frameworks. Moreover, the auxiliary structures only exist during training, i.e. it is completely cost-free in inference time. Code is available at: this https URL

[Mis26b] Steered LLM Activations are Non-Surjective (2026)
ArXiv
Date: 2026-04-10
By Aayush Mishra, Daniel Khashabi, Anqi Liu - 5 citations - 10 cit./yr
DOI: 10.48550/arXiv.2604.09839
PDF ✓

Activation steering is a popular white-box control technique that modifies model activations to elicit an abstract change in its behavior. It has also become a standard tool in interpretability (e.g., probing truthfulness, or translating activations into human-readable explanations) and safety research (e.g., jailbreakability). However, it is unclear whether steered behavior is realizable by any textual prompt. In this work, we cast this question as a surjectivity problem: for a fixed model, does every steered activation admit a preimage under the model's natural forward pass? Under practical assumptions, we prove that activation steering pushes the residual stream off the manifold of states reachable from discrete prompts. Almost surely, no prompt can reproduce the same internal behavior induced by steering. We also illustrate this finding empirically across three widely used LLMs. Our results establish a formal separation between white-box steerability and black-box prompting. We therefore caution against interpreting the ease and success of activation steering as evidence of prompt-based interpretability or vulnerability, and argue for evaluation protocols that explicitly decouple white-box and black-box interventions.

[Mak23] Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching (2023)
ArXiv
Date: 2023-11-28
By Aleksandar Makelov, Georg Lange, Atticus Geiger, Neel Nanda - 78 citations - 27 cit./yr
DOI: 10.48550/arXiv.2311.17030
PDF ✓

Mechanistic interpretability aims to understand model behaviors in terms of specific, interpretable features, often hypothesized to manifest as low-dimensional subspaces of activations. Specifically, recent studies have explored subspace interventions (such as activation patching) as a way to simultaneously manipulate model behavior and attribute the features behind it to given subspaces. In this work, we demonstrate that these two aims diverge, potentially leading to an illusory sense of interpretability. Counterintuitively, even if a subspace intervention makes the model's output behave as if the value of a feature was changed, this effect may be achieved by activating a dormant parallel pathway leveraging another subspace that is causally disconnected from model outputs. We demonstrate this phenomenon in a distilled mathematical example, in two real-world domains (the indirect object identification task and factual recall), and present evidence for its prevalence in practice. In the context of factual recall, we further show a link to rank-1 fact editing, providing a mechanistic explanation for previous work observing an inconsistency between fact editing performance and fact localization. However, this does not imply that activation patching of subspaces is intrinsically unfit for interpretability. To contextualize our findings, we also show what a success case looks like in a task (indirect object identification) where prior manual circuit analysis informs an understanding of the location of a feature. We explore the additional evidence needed to argue that a patched subspace is faithful.

[Cao21b] Inverting and Understanding Object Detectors (2021)
ArXiv
Date: 2021-06-26
By Ang Cao, Justin Johnson - 3 citations - 0.6 cit./yr
No DOI found. Link: https://arxiv.org/abs/2106.13933
PDF ✓

As a core problem in computer vision, the performance of object detection has improved drastically in the past few years. Despite their impressive performance, object detectors suffer from a lack of interpretability. Visualization techniques have been developed and widely applied to introspect the decisions made by other kinds of deep learning models; however, visualizing object detectors has been underexplored. In this paper, we propose using inversion as a primary tool to understand modern object detectors and develop an optimization-based approach to layout inversion, allowing us to generate synthetic images recognized by trained detectors as containing a desired configuration of objects. We reveal intriguing properties of detectors by applying our layout inversion technique to a variety of modern object detectors, and further investigate them via validation experiments: they rely on qualitatively different features for classification and regression; they learn canonical motifs of commonly co-occurring objects; they use diff erent visual cues to recognize objects of varying sizes. We hope our insights can help practitioners improve object detectors.

[Bae22b] If Influence Functions are the Answer, Then What is the Question? (2022)
ArXiv
Date: 2022-09-12
By Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, R. Grosse - 182 citations - 45 cit./yr
DOI: 10.48550/arXiv.2209.05364
PDF ✓

Influence functions efficiently estimate the effect of removing a single training data point on a model's learned parameters. While influence estimates align well with leave-one-out retraining for linear models, recent works have shown this alignment is often poor in neural networks. In this work, we investigate the specific factors that cause this discrepancy by decomposing it into five separate terms. We study the contributions of each term on a variety of architectures and datasets and how they vary with factors such as network width and training time. While practical influence function estimates may be a poor match to leave-one-out retraining for nonlinear networks, we show they are often a good approximation to a different object we term the proximal Bregman response function (PBRF). Since the PBRF can still be used to answer many of the questions motivating influence functions, such as identifying influential or mislabeled examples, our results suggest that current algorithms for influence function estimation give more informative results than previous error analyses would suggest.

[Gen24] The Unmet Promise of Synthetic Training Images: Using Retrieved Real Images Performs Better (2024)
ArXiv
Date: 2024-06-07
By Scott Geng, Cheng-Yu Hsieh, V. Ramanujan, Matthew Wallingford, Chun-liang Li, Pang Wei Koh, Ranjay Krishna - 27 citations - 12 cit./yr
DOI: 10.48550/arXiv.2406.05184
PDF ✓

Generative text-to-image models enable us to synthesize unlimited amounts of images in a controllable manner, spurring many recent efforts to train vision models with synthetic data. However, every synthetic image ultimately originates from the upstream data used to train the generator. Does the intermediate generator provide additional information over directly training on relevant parts of the upstream data? Grounding this question in the setting of image classification, we compare finetuning on task-relevant, targeted synthetic data generated by Stable Diffusion -- a generative model trained on the LAION-2B dataset -- against finetuning on targeted real images retrieved directly from LAION-2B. We show that while synthetic data can benefit some downstream tasks, it is universally matched or outperformed by real data from the simple retrieval baseline. Our analysis suggests that this underperformance is partially due to generator artifacts and inaccurate task-relevant visual details in the synthetic images. Overall, we argue that targeted retrieval is a critical baseline to consider when training with synthetic data -- a baseline that current methods do not yet surpass. We release code, data, and models at https://github.com/scottgeng00/unmet-promise.

[Liu26p] Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW (2026)
Unknown Journal
Date: 2026-08-20
By Kangning Liu, Suyan Li
No DOI found. Link: https://arxiv.org/abs/2608.19762
PDF ✓

A minibatch can influence training beyond the update at which it is observed because AdamW stores past gradient information in its optimizer states. We study this delayed effect through paired trajectories that differ only in one gradient update and share the same subsequent training sequence. We formulate AdamW as a finite-horizon input--state--output (ISO) system whose state contains the model parameters and first- and second-moment estimates. Linearizing the joint dynamics yields a signed response operator that maps a localized gradient perturbation to its future loss effects, revealing how optimizer memory shapes their magnitude, timing, and sign. We further derive an exact multistep error decomposition and establish first-order finite-horizon accuracy under local smoothness and controlled activation switching. Experiments validate the response mechanism and optimizer-state effects, while repeated-future analyses reveal substantial prospective structure in delayed influence that can be partially recovered from ISO approximations. Code is available at https://github.com/Kanyooo/Loss_ISO.

[Den26b] How Faithful Is Trajectory-Based Data Attribution? Error Sources, Remedies, and Practical Guidelines (2026)
ArXiv
Date: 2026-05-12
By Junwei Deng, Pingbang Hu, Suliang Jin, Haosen Lu, Jia-Chen T. Wang, Shichang Zhang, Jiaqi W. Ma - 2 citations - 4.0 cit./yr
DOI: 10.48550/arXiv.2605.18814
PDF ✓

Trajectory-based data attribution methods estimate the influence of training samples on model predictions by unrolling the training trajectory. They are widely used in applications such as data selection, data valuation, and model diagnosis, but there is a lack of comprehensive error analysis of these methods, raising concerns about method faithfulness and hindering reliable deployment. In this work, we provide the first systematic analysis of error sources in trajectory-based data attribution, together with concrete remedies to mitigate them and practical guidelines for downstream use. We organize the total error into three categories, config-level, algorithm-level, and system-level. We make three contributions. First, we identify optimizer mismatch as the dominant config-level error: existing methods derive their attribution under the assumption of SGD, even for models trained with the modern de facto optimizer AdamW. We propose AdamW-influence to fully account for AdamW's optimization dynamics, yielding improvements from 10% to over 300% in Spearman correlation between estimated and ground-truth influence across four settings spanning MLP, CNN, GPT-2, and Llama 3.2-1B. Second, we isolate the remaining algorithm-level error arising from the first-order Taylor approximation, identify the learning rate and trajectory length as factors governing the error magnitude, and derive a closed-form error proxy that can be evaluated along the original trajectory without retraining. Third, we translate these insights into practical guidelines for data selection by unifying offline and online strategies under a K-step look-ahead framework. Under this framework, online selection with a short horizon often matches or exceeds offline, and the optimal horizon can be tuned jointly with the learning rate. Together, these results turn the framework into an actionable selection recipe for practitioners.

[Guo26] Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions (2026)
Unknown Journal
Date: 2026-08-25
By Jin-Hui Guo
No DOI found. Link: https://arxiv.org/abs/2608.24593
PDF ✓

Adaptive optimizers retain gradient history in moment variables, allowing a local change in loss weighting to alter later updates. We examine whether this delayed transport is large enough to change prospective short-horizon decisions. On committed future-minibatch sequences, we differentiate eight-step AdamW trajectories through the complete model--optimizer state and select exposure-matched Math--Code loss schedules before independent evaluation. Across 12 unused 0.3M Transformer histories, full transport lowers token-disjoint loss relative to an optimizer-aware immediate derivative in 10/12 histories (mean benefit $4.71\times10^{-4}$; exact one-sided sign test, $p=0.0193$). The two controllers act equally often but select different schedules in 60/96 windows. Crossed checkpoint--future-path tests attribute this reordering to the interaction between optimizer state and near-future data, while an independent Ising--CNN experiment shows that deleting moment-state transport destroys accurate response prediction. Full-transport scores also concentrate exact-rollout winners in larger candidate libraries, focusing finite-amplitude evaluation on a shortlist. On these committed short paths, optimizer memory and near-future data order are therefore actionable components of the training state, providing a mechanism-based criterion for when finite-horizon rather than one-step intervention is required.

PDF symbols - ✓ PDF available; X No PDF available; … PDF download attempt in progress

When using cite keys in your reply to the user, format as [[cite_key]](url), where url is https://doi.org/<DOI> for the DOI shown above (or the exact Link shown for papers without one), separating multiple with commas. In workspace files, keep bare [cite_key] markers with no URLs.
