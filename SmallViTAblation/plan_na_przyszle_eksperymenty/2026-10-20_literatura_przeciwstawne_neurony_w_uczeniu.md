# Literatura: przeciwstawne neurony w trakcie uczenia — podstawy pod kolejne eksperymenty

Data: 2026-10-20. Źródła: Undermind (workspace „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”) i Scite.
Pełne teksty czytane dla: Mar25l, Che23an, Ahm26c, Ran23c, Hut25, Pat25d, Gon26h, Sam26b, Zhu26l, Urb26b, Saf26.
Pozostałe prace znane tylko z abstraktu/metadanych.

Deep searche (nowe, 2026-10-20):
- Emergence of opposing antagonistic neurons during neural network training tracked across checkpoints (69 prac)
- Measuring synergy redundancy and compensation between neurons through training with interventions (132 prace)
- Prior art check for specific measured findings on opposing MLP neuron pairs in a ViT detector across training (62 prace; cel zawierał nasze liczby z kampanii pruning/wkład względny/sfera)

Wcześniejsze, nadal aktualne: „Opposing neuron contributions and localization confidence dissociation…” (19.10),
„How individual MLP neuron contributions add up and cancel…” (14.10), „Causal emergence of coordinated neuron force fields…” (15.10).

## Odpowiedź wprost

Nikt nie zmierzył tego, co my: neuronów ostatniego MLP ViT, których przeciwstawność śledzona jest przez checkpointy
i seedy razem z interwencjami A/B/AB i rozdzieleniem IoU od score. Są jednak cztery bliskie precedensy:

1. **Pary neuronów z przeciwnymi wagami wyjściowymi powstające w uczeniu — teoria i małe MLP.**
   Mar25l (Flat Channels to Infinity, 2025): gradient prowadzi pary neuronów, których wagi WEJŚCIOWE się zbiegają
   (cos → 1), a wyjściowe rozbiegają do ±∞ z przeciwnymi znakami; ich duże wkłady prawie się znoszą, a suma daje
   skończoną funkcję typu gated linear unit: c·σ(w·x) + a(Δ·x)·σ'(w·x). To zjawisko PÓŹNEGO uczenia (frakcja
   aktualizacji w podprzestrzeni kanału bliska 1 pod koniec), z bardzo wolnym spadkiem straty. 4420 sieci, 50 inicjalizacji
   na konfigurację; kanały w 20–64 % biegów, częściej dla „szorstkich” celów. Kryteria: norma parametrów > 1e3,
   cos-odległość wag wejściowych < 1e-3, > 90 % aktualizacji w podprzestrzeni kanału.
   Kwo05b (Anti-pairing…, J. Phys. A 2005): „anti-pairing” jednostek ukrytych w sieci z nadmiarowymi neuronami
   (mechanika statystyczna uczenia on-line). Abstrakt niedostępny w Undermind; PDF otwarty na IOP — do przeczytania.
   Guo17e (opposite singularity w MLP, Neurocomputing) — płatny.
2. **Opozycja/korekta powstająca stopniowo w pretreningu transformerów.**
   Pat25d (LLM Layers Immediately Correct Each Other, 2026): cos wkładów sąsiednich warstw do tego samego tokenu
   ≈ −0,2 (do −0,4); w OLMo brak wzoru przy 5 % treningu, pojawia się ok. 21 % i narasta. Test przyczynowy przez
   dodanie α·c_i do wejścia warstwy i+1 (opozycja rośnie liniowo z α). Brak zestawienia ze stratą.
3. **Wzorzec push-pull wyłaniający się w uczeniu sieci rekurencyjnych/spikujących.**
   Zhu26l (Biol. Cybern. 2026): hamowanie między jednostkami o przeciwnej modulacji rośnie (stosunek across/within
   1,06 → 2,0); znika bez prawa Dale’a i wtedy zadanie wychodzi gorzej. Śledzone po epokach, ale BEZ analizy, czy
   poprzedza czy następuje po wzroście jakości.
4. **Przejścia geometrii wag w toy modelach superpozycji.**
   Che23an (2023): SGD przechodzi przez konfiguracje k-gonów (z wierzchołkami antypodalnymi), plateau straty → nagły
   spadek; lokalny współczynnik uczenia (LLC) rośnie przy spadkach. Antypodalnych par osobno nie liczą.
   Elh22 — pochodzenie obrazu antypodalnych par w superpozycji.

Wniosek: nasz wynik „przeciwstawność wag rośnie głównie PO głównym skoku jakości i nie koreluje z przyrostami IoU po
kontroli trendu” nie ma precedensu (ocena Undermind: likely novel). Mar25l daje gotową hipotezę mechanistyczną:
późne pary to kanały/GLU powstające z symetrii gradientu, a nie „przeszkadzające sobie lasery”.

## Ocena naszych sześciu obserwacji (deep search z liczbami)

| Obserwacja | Ocena | Najbliższe prace |
|---|---|---|
| 1. Wzrost liczby neuronów z antyrównoległym partnerem 0 → 424/768, wymiana partnerów pod koniec | częściowo antycypowane | Mar25l, Kwo05b; kondensacja Xu25ad, Zho21h (raczej grupy podobne niż przeciwne) |
| 2. Opozycja po głównym zysku jakości, brak korelacji po kontroli trendu | prawdopodobnie nowe | brak; metodycznie Ran23c, Saf26 (korelacja zmian metryk) |
| 3. Geometria wag ≠ funkcjonalne znoszenie | częściowo antycypowane | Lan22 (upstream vs downstream), Fak24, Ahm26c (z_r przewiduje wielkość, nie znak) |
| 4. Przeciwne skutki na krawędzie, zgodne wspieranie logitu | antycypowane na poziomie zadania, nowe na poziomie par | Wu19d, Zha23bf, Hut25 (DETR: MHCA psuje mgIoU, F1 stabilne) |
| 5. Usuwanie par nie poprawia detektora | częściowo antycypowane | Cas19e (frivolous units), Yu22f (pruning wag, które się znoszą), Mor18c |
| 6. Opozycja rośnie, a koaktywacja spada (86 % → 18 %) | częściowo antycypowane | Mar25l, Pat25d, Mcg23, Rus24b, Ahm26c |

## Co z tych prac warto przenieść do naszych eksperymentów

1. **Test kanału Mar25l na istniejących 288 checkpointach (bez treningu, tanie).** Dla par z cos(W2) < −0,5 liczyć
   po checkpointach: cos wag wejściowych W1 (czy → 1), normy |a_i|, |a_j| (czy rosną), frakcję aktualizacji w
   podprzestrzeni pary ‖Δθ_proj‖/‖Δθ‖, oraz czy suma pary ≈ σ(w·x) + (Δ·x)σ'(w·x). Jeśli tak — nasze pary to GLU
   z kanałów, co tłumaczy późne pojawienie się i brak związku z IoU. Uwaga: AdamW/weight decay może hamować
   rozbieganie norm — sprawdzić.
2. **Dawka zamiast zera (Ahm26c).** Skalować z_A współczynnikiem λ ∈ [−1, 1] (λ = 0 to średnia/neutralna, nie zero)
   i mierzyć nachylenie wkładu B oraz zmian krawędzi i logitu. W ostatnim MLP B nie może się „naprawić” po A
   (ta sama warstwa, dalej tylko norm + głowa), więc każda kompensacja jest stałą opozycją (counterweight), nie
   self-repair — to warto podkreślić w artykule jako czystą sytuację pomiarową.
3. **Interwencja w czasie uczenia (Ran23c) — jedyny test przyczynowy roli w uczeniu.** Regularyzator karzący
   antyrównoległe kolumny W2 (np. kara za cos < −0,5) włączany od kroku 0, od 300, od 10000; porównać krzywe IoU/score
   i liczbę par. Wymaga nowych biegów na Edenie (kilka seedów wystarczy na pilot).
4. **Zastąpienie przez rówieśników (Saf26).** Po wyłączeniu A rekonstruować jego aktywację liniowo z top-k neuronów
   tej samej warstwy; sprawdzić, czy kompensuje konkretny partner B, czy cały zbiór rówieśników.
5. **Interakcja i zapasowe neurony (Gon26h, Fak21, Kei04c).** I_uv = δz_uv − δz_u − δz_v (to nasze q w innej postaci)
   i skan comp_u(S) po wyłączeniu zbioru S: czy po zgaszeniu par funkcjonalnych rośnie rola innych neuronów.
6. **Kontrole zerowe.** Liczba antyrównoległych kolumn w sieci losowej o tym samym spektrum norm / po losowej rotacji;
   wpływ weight decay; porównanie przeciwstawności sąsiednich bloków à la Pat25d (cos wkładów bloków do tokenu
   po checkpointach) — tanie.
7. **Czy 10000–16000 to przejście fazowe?** Estymacja LLC (Che23an; biblioteka devinterp) na checkpointach wokół
   tego przedziału, obok krzywych straty. Uwaga autorów: LLC bywa niestabilne przy silnie osobliwych punktach.
8. **Ablacje „ostrożnie” (Hut25, Poc24, Mil24c, Li24bi).** Raportować mgIoU/IoU i score osobno; porównać zero-,
   mean- i resample-ablation (różne „dawki”).

## Statystyka (do protokołu)
- Bootstrap na poziomie seeda z zachowaniem zależności checkpointów w seedzie; obrazy nie są niezależne między
  seedami (już robimy).
- Selekcja na VAL, ocena na nowym, nieoglądanym TEST (obecny TEST był oglądany — kolejna kampania powinna to naprawić).
- Korelacje geometrii z jakością: pierwsze różnice, zaplanowane lagi, korekta na wielokrotność; nie wybierać najlepszego lagu.

## Prace kluczowe (DOI)
- Mar25l — Flat Channels to Infinity in Neural Loss Landscapes — 10.48550/arXiv.2506.14951
- Kwo05b — Anti-pairing in learning of a neural network with redundant hidden units — 10.1088/0305-4470/38/25/003
- Pat25d — LLM Layers Immediately Correct Each Other — 10.52202/085713-4643
- Zhu26l — Task success in trained SNN … cross-stimulus-modulated inhibition — 10.1007/s00422-025-01030-4
- Che23an — Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition — 10.48550/arXiv.2310.06301
- Ahm26c — Every Ablation Is a Dose: Counterweights and the Semblance of Self-Repair — arXiv 2610.02173
- Ran23c — On the special role of class-selective neurons in early training — 10.48550/arXiv.2305.17409
- Saf26 — Task Relevance Is Not Local Replaceability — 10.48550/arXiv.2605.07086
- Gon26h — Conditional Co-Ablation — 10.48550/arXiv.2607.01940
- Hut25 — Detection Transformers Under the Knife — 10.48550/arXiv.2507.21723
- Lan22 — Clustering units in neural networks: upstream vs downstream information — 10.48550/arXiv.2203.11815
- Yu22f — The Combinatorial Brain Surgeon: Pruning Weights That Cancel One Another — 10.48550/arXiv.2203.04466
- Cas19e — Frivolous Units — 10.1609/aaai.v35i8.16853
- Mor18c — On the importance of single directions for generalization — arXiv 1803.06959
- Fak21 — Systematic perturbation of an ANN (MSA) — 10.1371/journal.pcbi.1010250
- Urb26b — A Brain-like Synergistic Core in LLMs — 10.48550/arXiv.2601.06851
- Sam26b — Hidden not Deleted: How Networks Suppress Entangled Features — arXiv 2609.27593
- Min23b — Early Neuron Alignment in Two-layer ReLU Networks — 10.48550/arXiv.2307.12851
- Elh22 — Toy Models of Superposition — 10.48550/arXiv.2209.10652
- Mcg23 / Rus24b — Hydra effect / self-repair — 10.48550/arXiv.2307.15771, 10.48550/arXiv.2402.15390
