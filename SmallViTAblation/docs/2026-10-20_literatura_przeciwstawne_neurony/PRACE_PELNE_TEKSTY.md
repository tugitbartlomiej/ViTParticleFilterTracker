# Prace: pełne teksty, metadane i ocena obserwacji (2026-10-20)

Zakres: 11 prac czytanych w pełnym tekście (sekcje 1–11) oraz 15 prac znanych tylko z abstraktu lub metadanych (tabela na końcu). Liczby podane poniżej pochodzą z pełnych tekstów lub abstraktów wymienionych w notatce źródłowej, bez dodatkowych wartości. Notatka źródłowa: `../../plan_na_przyszle_eksperymenty/2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md`.

## 1. Mar25l: Flat Channels to Infinity in Neural Loss Landscapes (2025)

- **Klucz:** Mar25l
- **DOI/link:** 10.48550/arXiv.2506.14951
- **Co zmierzono (pełny tekst):** kanał to co najmniej dwa neurony ukryte, których wagi wyjściowe rozbiegają się z przeciwnymi znakami (a_i·a_j < 0), a wektory wag wejściowych zbiegają się. Duże wkłady prawie się znoszą, a suma dąży do c·σ(w·x) + a(Δ·x)·σ'(w·x), czyli bramkowanej jednostki liniowej (GLU; dla softplus bramka to sigmoid). Zjawisko dotyczy późnego uczenia: po fazie przejściowej aktualizacje stają się równoległe do kierunku linii siodłowej, normy wyjść rosną bardzo wolno, strata spada bardzo wolno, a frakcja aktualizacji w podprzestrzeni kanału pod koniec treningu jest skupiona blisko 1.
- **Kryteria klasyfikacji (App. B.3):** norma parametrów > 1e3; odległość cosinusowa wektorów wejściowych pary < 1e-3; > 90% aktualizacji w podprzestrzeni kanału (frakcja ‖Δθ_proj‖/‖Δθ‖).
- **Eksperymenty:** płytkie MLP o szerokości 2–16 i wymiarze wejścia 2–16 oraz głębsze (2–3 warstwy po 4 neurony); 4420 sieci, 50 inicjalizacji na konfigurację.
- **Kluczowe liczby:** częstość kanałów ok. 24% (Rosenbrock), 20/30/61/64% (GP, s = 0,1/0,5/2/10), częściej dla „szorstkich” celów. Strata i błąd przybliżenia skalują się jak ε², ostrość jak 1/ε².
- **Uwagi:** w pracy występuje niespójność dotycząca softplus i softplus+erf.
- **Co bierzemy do naszych eksperymentów:** test kanału na naszych checkpointach (`PLAN_TESTU_KANALU_MAR25L.md`), przy czym dosłowne kryterium normy prawdopodobnie nie zadziała przy weight decay, więc potrzebne są miary stopniowane. Hipoteza mechanistyczna: późne pary to kanały/GLU powstające z symetrii gradientu.

## 2. Che23an: Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition (2023)

- **Klucz:** Che23an
- **DOI/link:** 10.48550/arXiv.2310.06301
- **Co zmierzono (pełny tekst):** geometria kolumn macierzy wag w toy modelu superpozycji TMS (r = 2, c = 6) śledzona jako k-gony. Przy SGD plateau straty pojawia się przy konfiguracjach krytycznych, a nagłe spadki towarzyszą zmianie geometrii (np. 4++- → 4+ → 5 → 5+).
- **Kluczowe liczby:** 400 trajektorii SGD. Lokalny współczynnik uczenia (LLC), estymowany SGLD, rośnie przy spadkach.
- **Ograniczenia:** autorzy ostrzegają o niestabilności LLC (nawet ujemne wartości) w punktach silnie osobliwych. Antypodalnych par osobno nie liczą.
- **Co bierzemy do naszych eksperymentów:** estymacja LLC na checkpointach wokół 10000–16000 kroków (plan, pkt 7), z uwzględnieniem ryzyka niestabilności.

## 3. Ahm26c: Every Ablation Is a Dose: Counterweights and the Semblance of Self-Repair (2026)

- **Klucz:** Ahm26c
- **DOI/link:** https://arxiv.org/abs/2610.02173
- **Co zmierzono (pełny tekst):** counterweight to kierunek, którego wkład E_r(λ) = own_r + γ_r·λ ma nachylenie γ_r < 0 (λ = +1 clean, λ = 0 neutralnie, λ = −1 zamiana na kontrfakt). Zero-, mean- i resample-ablation trafiają w różne „dawki” zależnie od pary.
- **Kluczowe liczby:** 68/81 kierunków ma niezerowe nachylenie (52 counterweights, 16 relays). W GPT-2 IOI 7/10 głów (wszystkie counterweights; przykład L10.h7, γ = −0,886). Statystyka z wag z_r koreluje z |γ_r| (ρ = 0,73); przewiduje wielkość, nie znak.
- **Wniosek autorów:** pozorne self-repair to zwykle uwolnienie stałej opozycji.
- **Zalecenia autorów:** kalibrować ablacje jako dawki, |λ| ≤ 1, mierzyć wewnątrz zaburzonego reżimu.
- **Ograniczenia:** brak dynamiki uczenia (porównanie tylko base vs instruct).
- **Co bierzemy do naszych eksperymentów:** dawka zamiast zera (plan, pkt 2): skalowanie z_A współczynnikiem λ ∈ [−1, 1], gdzie λ = 0 oznacza średnią, a nie zero.

## 4. Ran23c: On the special role of class-selective neurons in early training (2023)

- **Klucz:** Ran23c
- **DOI/link:** 10.48550/arXiv.2305.17409
- **Co zmierzono (pełny tekst):** ResNet-50 na ImageNet. Indeks selektywności kanałów śledzony od inicjalizacji do epoki 90. Ablacje narastające (od najbardziej selektywnych vs losowe, AUC krzywych) wykonywane na checkpointach. Regularyzator tłumiący selektywność (α = −20) włączany od epoki 0 albo od epok 1–5.
- **Kluczowy wynik:** tylko tłumienie od epoki 0 szkodzi uczeniu, czyli krytyczne okno jest w pierwszej epoce.
- **Co bierzemy do naszych eksperymentów:** metodę. Regularyzator własności geometrycznej włączany w różnych momentach (plan, pkt 3): kara za cos < −0,5 od kroku 0, 300 albo 10000.

## 5. Hut25: Detection Transformers Under the Knife (2025)

- **Klucz:** Hut25
- **DOI/link:** 10.48550/arXiv.2507.21723
- **Co zmierzono (pełny tekst):** losowe zerowanie parametrów (100 konfiguracji na poziom; 5/15/30/50% oraz blokowo 30%) w DETR, DDETR i DINO. Osobno mierzone mgIoU (lokalizacja) i F1 (klasyfikacja po dopasowaniu węgierskim).
- **Kluczowe liczby:** decoder MHCA w DETR: −26 pp mgIoU przy 50% zerowania, przy stabilnym F1. DINO MHCA: regresja spada, klasyfikacja rośnie.
- **Ograniczenia:** brak FFN i brak metryki pewności/kalibracji.
- **Co bierzemy do naszych eksperymentów:** raportowanie mgIoU/IoU i score osobno oraz ablacje „ostrożnie” (plan, pkt 8).

## 6. Pat25d: LLM Layers Immediately Correct Each Other (2026)

- **Klucz:** Pat25d
- **DOI/link:** 10.52202/085713-4643
- **Co zmierzono (pełny tekst):** cosinus wkładów sąsiednich warstw do tego samego tokenu wynosi średnio ok. −0,2 (do −0,4) w 5 z 7 rodzin modeli. Test przyczynowy: dodanie α·c_i do wejścia warstwy i+1; opozycja rośnie liniowo z α.
- **Dynamika (OLMo 7B):** brak wzoru przy kroku 33 000 (ok. 5% treningu); pojawia się ok. kroku 135 500 (ok. 21%, ok. 0,5 bln tokenów) i narasta. Podobnie w OLMo 2 1B/7B.
- **Ograniczenia:** brak zestawienia z stratą.
- **Co bierzemy do naszych eksperymentów:** tanią kontrolę przeciwstawności sąsiednich bloków, czyli cos wkładów bloków do tokenu po checkpointach (plan, pkt 6).

## 7. Gon26h: Conditional Co-Ablation (2026)

- **Klucz:** Gon26h
- **DOI/link:** 10.48550/arXiv.2607.01940
- **Co zmierzono (pełny tekst):** interakcja I_uv = δz_uv − δz_u − δz_v oraz S_uv = E‖Ĩ_uv‖² (Fisher). Skan comp_u(S) = E(δz_u | S) − E(δz_u) znajduje neurony lub głowy zapasowe po wyłączeniu zbioru S.
- **Kluczowe liczby:** AUC 0,91 vs 0,82 (AtP* GradDrop); 4 seedy promptów; testy permutacyjne i DeLong.
- **Ograniczenia:** brak checkpointów, więc brak dynamiki uczenia.
- **Co bierzemy do naszych eksperymentów:** I_uv odpowiada naszemu q w innej postaci; skan comp_u po wyłączeniu par funkcjonalnych (plan, pkt 5).

## 8. Sam26b: Hidden not Deleted: How Networks Suppress Entangled Features (2026)

- **Klucz:** Sam26b
- **DOI/link:** https://arxiv.org/abs/2609.27593
- **Co zmierzono (pełny tekst):** model zabawkowy. Cecha tłumiona przez obrót odczytu (rozwiązanie „mirror”, mediana cos −0,86) albo przez ujemny bias („shadow”); 15 wartości splątania × 100 seedów.
- **Kluczowe liczby:** przy ρ = −1 jest 75% mirror, przy ρ = 0 87% shadow. Cechę da się przywrócić łatką (0 → 0,38 i 0 → 0,13).
- **Ograniczenia:** toy model, nie ViT.
- **Co bierzemy do naszych eksperymentów:** argument, że wyłączenie czegoś z wyjścia nie oznacza usunięcia cechy. Stąd wniosek, by mierzyć również wkład do odczytu, a nie tylko zmianę wyjścia (interpretacja własna).

## 9. Zhu26l: Task success in trained spiking neural network models coincides with emergence of cross-stimulus-modulated inhibition (2026)

- **Klucz:** Zhu26l
- **DOI/link:** 10.1007/s00422-025-01030-4 (Biol. Cybern. 2026)
- **Co zmierzono (pełny tekst):** hamowanie między jednostkami o przeciwnej modulacji rośnie w trakcie uczenia: stosunek across/within 1,06 ± 0,15 → 2,0 ± 0,7; pobudzenie 1,00 → 0,6.
- **Kluczowe liczby:** bez prawa Dale’a motyw znika (0,914/0,842), a zadanie wychodzi gorzej. n = 28 (w wariancie rate-only n = 38, bez motywu). Śledzone po epokach.
- **Ograniczenia:** brak analizy, czy motyw poprzedza wzrost jakości, czy po nim następuje.
- **Co bierzemy do naszych eksperymentów:** wzorzec push-pull oraz pytanie o kolejność zdarzeń (przyczyna czy skutek) w naszej serii czasowej.

## 10. Urb26b: A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning (2026)

- **Klucz:** Urb26b
- **DOI/link:** 10.48550/arXiv.2601.06851
- **Co zmierzono (pełny tekst):** ΦID (Syn→Syn, Red→Red) dla par głów; ranking synergia minus redundancja. W Pythii-1B rdzeń synergiczny w środkowych warstwach wyłania się w trakcie uczenia.
- **Kluczowe wyniki:** wyłączanie wg rankingu daje większą dywergencję KL niż losowe porządki (5 losowych). Zaburzenie rdzenia synergicznego najbardziej obniża wynik w benchmarku MATH, zaburzenie rdzenia redundantnego najmniej.
- **Co bierzemy do naszych eksperymentów:** ranking wg synergii minus redundancji jako porządek ablacji porównywany z porządkiem losowym (ablacje narastające, jak w Mor18c).

## 11. Saf26: Task Relevance Is Not Local Replaceability (2026)

- **Klucz:** Saf26
- **DOI/link:** 10.48550/arXiv.2605.07086
- **Co zmierzono (pełny tekst):** CIFAR-100 (ResNet-18, VGG-16, MobileNetV2). Korelacja przechwytywania wejścia z istotnością zadania wynosi 0,71/0,72 na starcie, a po ok. 10 epokach 0,39/0,05. Lezja kanału plus zastąpienie go liniową rekonstrukcją z 8 najlepszych kanałów tej samej warstwy.
- **Kluczowe liczby:** zastąpienie pomaga w 84,3% lezji i odzyskuje medianę 68,7% szkody. 45 komórek (model/seed/warstwa), 2720 kanałów.
- **Co bierzemy do naszych eksperymentów:** test zastąpienia przez rówieśników (plan, pkt 4): po wyłączeniu neuronu A rekonstruować jego aktywację liniowo z top-k neuronów tej samej warstwy i sprawdzić, czy kompensuje konkretnego partnera B, czy cały zbiór rówieśników.

## Pozostałe prace (tylko abstrakt lub metadane)

Dla tych prac notatka nie zawiera danych z pełnego tekstu, więc nie podano tu wyników.

| Klucz | Tytuł | DOI/link | Uwagi z notatki |
|---|---|---|---|
| Kwo05b | Anti-pairing in learning of a neural network with redundant hidden units (J. Phys. A, 2005) | 10.1088/0305-4470/38/25/003 | „Anti-pairing” jednostek ukrytych w sieci z nadmiarowymi neuronami (mechanika statystyczna uczenia on-line). Abstrakt niedostępny w Undermind; PDF otwarty na IOP, do przeczytania. |
| Guo17e | Stability analysis of opposite singularity in multilayer perceptrons (Neurocomputing) | 10.1016/j.neucom.2017.12.018 | Płatna pozycja (ok. 42 USD). Nie pobrana. |
| Min23b | Early Neuron Alignment in Two-layer ReLU Networks | 10.48550/arXiv.2307.12851 | Neurony wyrównują się do danych dodatnich lub ujemnych, zależnie od znaku wagi wyjściowej. |
| Ma20c | Quenching-Activation Behavior | arXiv 2006.14450 | Brak opisu w notatce. |
| Cas19e | Frivolous Units | 10.1609/aaai.v35i8.16853 | Najbliższa praca dla obserwacji 5. |
| Yu22f | The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another | 10.48550/arXiv.2203.04466 | Pruning wag, które się znoszą; obserwacja 5. |
| Lan22 | Clustering units in neural networks: upstream vs downstream information | 10.48550/arXiv.2203.11815 | Obserwacja 3. |
| Mor18c | On the importance of single directions for generalization | arXiv 1803.06959 | Ablacje narastające, 10 losowych porządków. |
| Fak21 | Systematic perturbation of an ANN (MSA) | 10.1371/journal.pcbi.1010250 | Metoda MSA. |
| Elh22 | Toy Models of Superposition | 10.48550/arXiv.2209.10652 | Pochodzenie obrazu antypodalnych par w superpozycji. |
| Mcg23 | Hydra effect | 10.48550/arXiv.2307.15771 | Obserwacja 6. |
| Rus24b | Self-repair | 10.48550/arXiv.2402.15390 | Obserwacja 6. |
| Xu25ad | Przegląd kondensacji | 10.48550/arXiv.2504.09484 | Obserwacja 1 (kondensacja). |
| Wu19d | Rethinking Classification and Localization for Object Detection (CVPR 2020) | 10.1109/CVPR42600.2020.01020 | Obserwacja 4: osobne reprezentacje klasyfikacji i regresji ramki. |
| Zha23bf | Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection (ICCV 2023) | 10.1109/ICCV51070.2023.00607 | Obserwacja 4. |
| Fak24 | Downstream network transformations dissociate neural activity from causal functional contributions (Sci. Rep. 2024) | 10.1038/s41598-024-52423-7 | Obserwacja 3: aktywność nie przewiduje wkładu przyczynowego. |
| Zho21h | Towards Understanding the Condensation of Neural Networks at Initial Training (2021) | 10.52202/068431-0159 | Obserwacja 1: kondensacja (grupy neuronów o podobnych kierunkach). |
| Kei04c | Fair Attribution of Functional Contribution in Artificial and Biological Networks (Neural Comput. 2004) | 10.1162/0899766041336387 | Metoda MSA (wartości Shapleya dla lezji). |
| Poc24 | Investigating Neuron Ablation in Attention Heads: The Case for Peak Activation Centering (2024) | DOI nie pobrany | Różne rodzaje ablacji dają różne efekty. |
| Mil24c | Transformer Circuit Faithfulness Metrics are not Robust (2024) | DOI nie pobrany | Wrażliwość metryk na sposób ablacji. |
| Li24bi | Optimal ablation for interpretability (2024) | DOI nie pobrany | Wybór wartości ablacji. |

Dane bibliograficzne Wu19d, Zha23bf, Fak24, Zho21h i Kei04c pochodzą z Undermind (get_paper_info). Dla Poc24, Mil24c i Li24bi znane są tylko tytuły z listy deep searcha.

## Wnioski z przeglądu

- Nasz wynik „przeciwstawność wag rośnie głównie PO głównym skoku jakości i nie koreluje z przyrostami IoU po kontroli trendu” nie ma precedensu (ocena Undermind: prawdopodobnie nowe).
- Mar25l daje gotową hipotezę mechanistyczną: późne pary to kanały/GLU powstające z symetrii gradientu, a nie „przeszkadzające sobie lasery”.
- Ahm26c i Sam26b wskazują, że zerowanie nie jest tym samym co usunięcie cechy (dawki; cecha ukryta, a nie usunięta). Hut25 pokazuje, że mgIoU i F1 trzeba raportować osobno.

## Ocena naszych sześciu obserwacji (deep search „Prior art check”, 62 prace)

| Nr | Obserwacja | Ocena | Najbliższe prace |
|---|---|---|---|
| 1 | Wzrost liczby neuronów z antyrównoległym partnerem 0 → 424/768 i wymiana partnerów pod koniec | częściowo antycypowane | Mar25l, Kwo05b; kondensacja: Xu25ad, Zho21h (raczej grupy podobne niż przeciwne) |
| 2 | Przeciwstawność wag rośnie po głównym zysku jakości; brak korelacji z przyrostami IoU po kontroli trendu | prawdopodobnie nowe | brak; metodycznie Ran23c, Saf26 (korelacja zmian metryk) |
| 3 | Geometria wag to nie funkcjonalne znoszenie | częściowo antycypowane | Lan22 (upstream vs downstream), Fak24, Ahm26c (z_r przewiduje wielkość, nie znak) |
| 4 | Przeciwne skutki na krawędzie przy zgodnym wspieraniu logitu | antycypowane na poziomie zadania, nowe na poziomie par | Wu19d, Zha23bf, Hut25 (DETR: MHCA psuje mgIoU, F1 stabilne) |
| 5 | Usuwanie par nie poprawia detektora | częściowo antycypowane | Cas19e (frivolous units), Yu22f (pruning wag, które się znoszą), Mor18c |
| 6 | Opozycja rośnie, a koaktywacja spada (86 % → 18 %) | częściowo antycypowane | Mar25l, Pat25d, Mcg23, Rus24b, Ahm26c |

Uwaga: Fak24 i Zho21h znane są z tabeli deep searcha i abstraktów; pełnych tekstów nie czytano.
