# Literatura z Undermind (2026-10-23), workspace „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”

Trzy wyszukiwania (klucze cytowań Undermind w nawiasach):

1. `/Attention as barycentric affine map on the token cloud and training-time change of the depth profile` (94 prace)
2. `/Sequences of saddle maps that trap on average alternating gradient focusing rotating saddle Paul trap and layered network analogues` (210)
3. `/Transformer blocks as operator splitting of attention and MLP substeps and the non commutativity of the two sublayers` (85)

## Uwaga jako mapa barycentryczna, profil głębokości, uczenie
- Geshkovski i in. 2023, The emergence of clusters in self-attention dynamics [Ges23] — 10.52202/075280-2493
- Castin, Ablin, Carrillo, Peyré 2025, A Unified Perspective on the Dynamics of Deep Transformers [Cas25] — 10.48550/arXiv.2501.18322
- Rigollet 2025, The Mean-Field Dynamics of Transformers [Rig25] — 10.48550/arXiv.2512.01868
- Karagodin i in. 2025, Normalization in Attention Dynamics [Kar25] — 10.48550/arXiv.2510.22026
- Álvarez-López, Geshkovski, Ruiz-Balet 2026, Perceptrons and localization of attention's mean-field landscape [Alv26] — 10.48550/arXiv.2601.21366
- Isobe, Inoue, Imaizumi 2026, Training-Induced Escape from Token Clustering [Iso26] — 10.48550/arXiv.2605.07772 (czas = głębokość; brak profilu po checkpointach)
- Tian i in. 2023, JoMA: joint dynamics of MLP and attention [Tia23] — 10.48550/arXiv.2310.00535 (uwaga najpierw rzadka, potem gęsta)
- Adler, Byers, Crovella 2026, Attention Mean Fields Predict Average Representation Dynamics [Adl26] — arXiv:2609.16382 (wcześnie model = jego pole średnie)
- Zhai i in. 2023, Stabilizing Transformer Training by Preventing Attention Entropy Collapse [Zha23au] — 10.48550/arXiv.2303.06296
- Dong, Cordonnier, Loukas 2021, Pure Attention Loses Rank Doubly Exponentially with Depth [Don21d] — arXiv:2103.03404
- Dovonon, Bronstein, Kusner 2024, Setting the Record Straight on Transformer Oversmoothing [Dov24] — 10.48550/arXiv.2401.04301
- Wang i in. 2022, Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain Analysis [Wan22as] — 10.48550/arXiv.2203.05962
- Mudarisov, Burtsev, State 2026, Feed-Forward Steering in Transformer Residual Dynamics [Mud26] — arXiv:2608.02071 (składowa styczna/radialna FFN, defekt komutatora)
- Maranzatto, Akkoç, Ulukus 2026, The Role of Feed-Forward Layers in Transformer Dynamics [Mar26m] — arXiv:2609.36230 (FFN jako wspólne pole sterujące, może zdominować uwagę)
- Varam i in. 2025, Class Separation Dynamics in Vision Transformers [Var25] — 10.1109/ACCESS.2025.3602733
- Järve i in. 2026, Feature Evolution and Migration during Vision Transformer Training [Jar26] — 10.1145/3799682.3841045
- Odonnat, Bouaziz, Cabannes 2024, A Mechanistic Study of Transformers Training Dynamics [Odo24] — arXiv:2410.24050
- Sander i in. 2021, Sinkformers [San21e] — arXiv:2110.11773

## Sekwencje siodeł, ogniskowanie naprzemienne, siodło obrotowe, pułapki
- Tandeitnik i in. 2024, All-optical saddle trap [Tan24n] — 10.1103/physrevapplied.22.044073 (warunek |Ω| > ω_y bez tłumienia; Ω_c/2π ≈ 280 kHz)
- Courant, Snyder 1958, Theory of the Alternating-Gradient Synchrotron [Cou58] — 10.1016/0003-4916(58)90012-5
- Howard, MacKay 1987, Linear stability of symplectic maps [How87] — 10.1063/1.527544
- Qin 2018, Stability of linear Hamiltonian systems with periodic coefficients [Qin18c] — 10.1063/1.5067391
- Thompson, Harmon, Ball 2002, The rotating-saddle trap [Tho02c] — 10.1139/P02-110
- Kirillov, Levi 2015, Rotating saddle trap as Foucault's pendulum [Kir15b] — 10.1119/1.4933206
- Carey i in. 2025, The asymmetric rotating saddle potential as a mechanical analog to the RF Paul trap [Car25j] — 10.1119/5.0274638
- Paul 1990, Electromagnetic Traps for Charged and Neutral Particles (Nobel) [Pau90] — 10.1002/ANIE.199007391
- Nazerian, Sorrentino, Aminzare 2024, Reactivity, Contraction and Finite-Time Lyapunov Exponents [Naz24] — 10.48550/arXiv.2410.23435
- Ng, Lin, Chan 2009, Theory of optical trapping by an optical vortex beam [Ng09] — 10.1103/PhysRevLett.104.103601
- Kiselev, Plutenko 2016, Optical trapping by Laguerre-Gaussian beams [Kis16b] — 10.1103/PHYSREVA.94.013804
- Fernando, Guitchounts 2026, Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology [Fer26d] — 10.48550/arXiv.2605.14258 (jakobiany warstw, 98 % par zespolonych, wąskie gardło iloczynu)
- Haas i in. 2026, Why Deep Jacobian Spectra Separate [Haa26b] — 10.48550/arXiv.2602.12384
- Aubry i in. 2024, Transformer Block Coupling and its Correlation with Generalization [Aub24] — arXiv:2407.07810
- Patrawala i in. 2026, LLM Layers Immediately Correct Each Other [Pat25d] — 10.52202/085713-4643 (cos sąsiednich warstw −0,15…−0,3; powstaje w uczeniu)

## Rozszczepienie operatorowe uwaga / MLP
- Lu i in. 2019, Transformer From a Multi-Particle Dynamic System Point of View [Lu19e] — arXiv:1906.02762 (Lie–Trotter, Strang/Macaron)
- Zhong i in. 2022, A Neural ODE Interpretation of Transformer Layers [Zho22i] — 10.48550/arXiv.2212.06011
- Press, Smith, Levy 2019, Improving Transformer Models by Reordering their Sublayers [Pre19d] — 10.18653/v1/2020.acl-main.270
- Kobayashi i in. 2023, Analyzing Feed-Forward Blocks in Transformers through the Lens of Attention Maps [Kob23] — arXiv:2302.00456
