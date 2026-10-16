/Falsifiable causal tests for task aligned geometry in trained vision transformers (status: completed, created: Oct 15 23:18, updated: Oct 15 23:20)
  Research goal:
  ```
  I need to design experiments that distinguish a real learned computational mechanism from a descriptive visualization of Vision Transformer activations. The model is a six block pre LayerNorm vision transformer with a convolutional stem, 600 spatial tokens, 192 dimensional residual states, 768 MLP hidden units, and a dense object localization head. We observe mixed symmetric and antisymmetric components of local MLP Jacobians, cancellation of signed neuron contributions, and changes across training checkpoints. I want to establish whether learned directional geometry selectively transmits object localization information while suppressing irrelevant image changes. The key problem is constructing causal interventions and null models that isolate orientation from update magnitude, respect normalization symmetries, avoid off distribution activation edits, separate fixed candidate box regression from discontinuous detection selection, and support claims across independent training runs and images. Find methodological precedents and counterexamples that determine which experiments could genuinely falsify this mechanism, including whether comparable effects already exist at initialization or are generic consequences of local linearity.
  ```
  Summary of results (based on available abstracts and metadata, not full texts):
  ```
  The decisive test is whether rotating a fixed-norm perturbation toward the proposed task-aligned Jacobian geometry selectively improves localization beyond initialization and matched orientation nulls—not whether the Jacobian or neuron map looks structured [Mak23, Mel25c, Lea20e].
  
  At each checkpoint and pre-LayerNorm MLP site, compare object-changing and nuisance-changing image pairs, then test actual image-induced activation differences and equal-realized-norm rotations of their directions. Preserve the Jacobian’s singular spectrum and symmetric/antisymmetric energy in orientation-shuffled controls, and fit label-shuffled axes; repeat the full protocol at initialization. Define edits in the LayerNorm output’s reachable tangent space and calibrate the realized displacement, since normalization changes which directions are admissible [Rie24, Gup24b]. Arbitrary activation edits can recruit dormant pathways and yield the right output through the wrong computation [Gra25c, Mak23].
  
  Evaluate continuous box regression for a fixed candidate and object assignment separately from candidate ranking or detection selection; classification and regression can rely on different cues [Cao21b]. Treat signed neuron cancellation as a hypothesis for joint intervention, not evidence of causal contribution [Shi24p]. Replicate paired effects across held-out images and independent training runs. Generic local linearity and perturbation geometry can occur before learning, while short-range linear fit need not predict downstream behavior [Cow24, Pet21, Asw26]; an initialization-level effect or failure of orientation-matched controls would therefore falsify task-specific learned geometry.
  ```
  Relevant Papers (showing 1-16 of 136):
    [Mak23] r=1.850 - Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching (2023)
    ArXiv
    By Aleksandar Makelov, ..., and Neel Nanda - 86 citations - 30 cit./yr
    PDF ✓
    [Fer26d] r=1.800 - Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology (2026)
    ArXiv
    By Jesseba Fernando, Grigori Guitchounts - 6 citations - 12 cit./yr
    PDF ✓
    [Gra25c] r=1.650 - Addressing divergent representations from causal interventions on neural networks (2025)
    ArXiv
    By Satchel Grant, ..., and Christopher Potts - 7 citations - 7.5 cit./yr
    PDF ✓
    [Mel25c] r=1.570 - The Dead Salmons of AI Interpretability (2025)
    ArXiv
    By Maxime Méloux, ..., and Maxime Peyrard - 11 citations - 13 cit./yr
    PDF ✓
    [Shi24p] r=1.560 - Hypothesis Testing the Circuit Hypothesis in LLMs (2024)
    ArXiv
    By Claudia Shi, ..., and David M. Blei - 36 citations - 18 cit./yr
    PDF ✓
    [Zha23aj] r=1.530 - Towards Best Practices of Activation Patching in Language Models: Metrics and Methods (2023)
    ArXiv
    By Fred Zhang, Neel Nanda - 386 citations - 127 cit./yr
    PDF ✓
    [Kvi22] r=1.520 - Internal representations of vision models through the lens of frames on data manifolds (2022)
    NeurReps, pp. 75-115
    By Henry Kvinge, ..., and T. Emerson - 4 citations - 1.0 cit./yr
    PDF ✓
    [Hua25i] r=1.500 - Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs (2025)
    ArXiv
    By Lianghuan Huang, Yingshan Chang - 2 citations - 2.0 cit./yr
    PDF ✓
    [Hei24] r=1.480 - How to use and interpret activation patching (2024)
    ArXiv
    By S. Heimersheim, Neel Nanda - 217 citations - 88 cit./yr
    PDF ✓
    [Zad26b] r=1.480 - What Would Falsify It? A Variable Specific Evidence Standard for Mechanistic Claims About Self Explanation (2026)
    Unknown Journal
    By Arshia Eftekhari Zadeh
    PDF ✓
    [Liu26be] r=1.470 - Right Answer, Wrong Mechanism: Detecting Pernicious Divergence in Causal Interventions (2026)
    Unknown Journal
    By Bei-Ming Liu, Min-Jie Chen
    PDF ✓
    [Mud26] r=1.450 - Feed-Forward Steering in Transformer Residual Dynamics (2026)
    Unknown Journal
    By Timur Mudarisov, M. Burtsev, Radu State - 1 citations - 2.0 cit./yr
    PDF ✓
    [Vai26] r=1.390 - The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching (2026)
    ArXiv
    By Sankaran Vaidyanathan, ..., and David Jensen - 10 citations - 20 cit./yr
    PDF ✓
    [Mil24c] r=1.390 - Transformer Circuit Faithfulness Metrics are not Robust (2024)
    ArXiv
    By Joseph Miller, Bilal Chughtai, William Saunders - 21 citations - 9.3 cit./yr
    PDF ✓
    [Mel25b] r=1.380 - Mechanistic Interpretability as Statistical Estimation: A Variance Analysis of EAP-IG (2025)
    ArXiv
    By Maxime Méloux, François Portet, Maxime Peyrard - 15 citations - 14 cit./yr
    PDF ✓
    [Pet21] r=1.370 - Relative stability toward diffeomorphisms indicates performance in deep nets (2021)
    Journal of Statistical Mechanics: Theory and Experiment
    By Leonardo Petrini, ..., and M. Wyart - 16 citations - 2.9 cit./yr
    PDF ✓

Scores (r): r=1 is the 'precisely relevant' threshold — a paper that just barely meets the search goal. Everything else is relative importance: r=2 is 2× as relevant, r=0.5 is half as relevant.

PDF symbols - ✓ PDF available; X No PDF available; … PDF download attempt in progress

Use detail_level='full' for paper abstracts and expanded details.

When presenting papers in your reply, fetch their DOIs with get_paper_info (show_doi=true) and present cite keys as [[cite_key]](https://doi.org/<DOI>). Use the Link returned by get_paper_info for papers without a DOI.

When mentioning this deep search to the user, include its web-app link: https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=<its URL-encoded path>
