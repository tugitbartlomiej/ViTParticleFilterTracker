# Literatura i uzasadnienie testów

Stan weryfikacji: 16 października 2026. To ukierunkowany przegląd do protokołu mechanistycznego, nie deklaracja przeglądu systematycznego całego pola. Numery R w tym pliku odnoszą się do katalogu TESTY; lokalne numery R w audytach pomocniczych mają własne bibliografie.

## Jak wykonano przegląd

Zastosowano skille **paper-search** i **scientific-problem-selection**. Wyszukiwanie łączyło CLI paper-search, Undermind, uzupełniające wyszukiwania agentów przez SciSpace/Scite i sprawdzenie źródeł pierwotnych. Wnioski metodologiczne poniżej wynikają z dokumentów autorów i wydawców, nie z samego opisu wygenerowanego przez wyszukiwarkę.

- Zapytanie CLI: „vision transformer representation dynamics mechanistic interpretability causal interventions”, arXiv/Semantic Scholar/Crossref, do 5 trafień na źródło. Zapisano 10 wyników arXiv/Crossref. Semantic Scholar zwrócił 429; nie przypisujemy mu wyników.
- Przejrzano istniejące przeglądy w projekcie Undermind i wykonano nowe deep search: **Falsifiable causal tests for task aligned geometry in trained vision transformers**. Otrzymano ranking 136 prac; przejrzano podsumowanie i metadane czołowych wyników, a nie pełne teksty wszystkich 136.
- Bezpośrednio pobrano przez Undermind teksty Aubry i Patrawala. Definicję Aubry dodatkowo sprawdzono w arXiv v4 i oficjalnym PDF. Szczegółowe odczyty geometrii, patchingu i statystyki opisują audyty.
- Zapytania obejmowały Jacobiany i coupling bloków, lokalną geometrię residualu, attention jako dynamikę, activation/path patching, self-repair, subspace illusion, ViT attribution, wariancję seedów i bootstrap skrzyżowany.
- Nie znaleziono podstaw do twierdzenia, że samo wykrycie antysymetrii Jacobianu, par przeciwnych neuronów albo patchingu w ViT jest nowością. Potencjalny wkład to **przewidziany i potwierdzony interwencją mechanizm lokalizacji**, jego rozwój oraz zakres transferu.

[Przegląd Undermind](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Falsifiable%20causal%20tests%20for%20task%20aligned%20geometry%20in%20trained%20vision%20transformers).

Zapis wyszukiwań: [CLI](../../outputs/protokol_testow_vit_iclr_neurips_20261016/literatura/paper_search_mechanism.json), [błędy źródła CLI](../../outputs/protokol_testow_vit_iclr_neurips_20261016/literatura/paper_search_mechanism.stderr.txt), [deep search](../../outputs/protokol_testow_vit_iclr_neurips_20261016/literatura/undermind_deep_search.md), [odczyty Aubry/Patrawala](../../outputs/protokol_testow_vit_iclr_neurips_20261016/literatura/undermind_fulltext_Aubry_Patrawala.md), [wyszukiwania przyczynowości](../../outputs/protokol_testow_vit_iclr_neurips_20261016/causal_zrodla.json).

## Najbliżsi poprzednicy i obowiązek porównania

| Źródło | Co już badano | Co musimy wykazać ponad to |
|---|---|---|
| R03 Aubry | Coupling Jacobianów, rozwój w treningu, związek z generalizacją; również ViT | Przyczynową rolę określonej orientacji dla bbox, kontrolowaną względem coupling i czułości |
| R01 Tomihari/Karakida | Dynamikę recurrent attention poprzez Jacobiany, bez konieczności jednego potencjału energii | Reakcję rzeczywistego forward sześciu różnych bloków attention+MLP i głowy detektora |
| R04 Patrawala | Natychmiastową adaptacyjną korektę kolejnych warstw | Czy taka korekta jest w naszym ViT, czego dotyczy i jaki ma skutek dla lokalizacji |
| R10 CAAP, R13 Prisma | Patching oraz narzędzia mechanistycznej analizy modeli wizyjnych | Precyzyjną tezę i test jej obalenia; interfejs sam nie dowodzi nowego mechanizmu |
| R06/R07/R16/R17 | Ryzyka interpretacji patchingu i miar faithfulness | Dowód wykraczający poza skuteczne sterowanie wyjściem i jeden rodzaj ablacji |

## Rejestr źródeł

<a id="r01"></a>
### R01 — Jacobiany i dynamika attention

**Tomihari, A.; Karakida, R. — Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians. NeurIPS 2025.**
[Pełny tekst arXiv v4](https://arxiv.org/html/2505.19458v4). DOI proceedings: 10.52202/085713-4675.

Weryfikacja: tekst, sekcje 3–7 i ograniczenia, metadane publikacji. Uzasadnia analizę lokalnej pochodnej i rozdzielenie składników dynamiki. Badany recurrent attention nie jest naszym pełnym ViT; nie przenosimy jego równowag ani asymptotyki na jednorazowy forward sześciu bloków. Testy T02–T06, T09, T18, T26.

<a id="r02"></a>
### R02 — Czułość attention

**Castin, V.; Ablin, P.; Peyré, G. — How Smooth Is Attention? ICML 2024.**
[Oficjalny artykuł PMLR](https://proceedings.mlr.press/v235/castin24a.html).

Weryfikacja: metadane i abstrakt pierwotny; szczegółowych stałych twierdzeń nie wykorzystujemy. Podstawa uwzględnienia lokalnej czułości attention jako konkurencyjnego wyjaśnienia. T02, T04, T07, T24, T26. Sama norma pochodnej nie dowodzi znaczenia dla lokalizacji.

<a id="r03"></a>
### R03 — Coupling Jacobianów, także ViT

**Aubry, M.; Meng, H.; Sugolov, A.; Papyan, V. — Transformer Block Coupling and its Correlation with Generalization in LLMs. ICLR 2025.**
[Oficjalny PDF](https://openreview.net/pdf?id=kvLenbZZgg), [arXiv v4](https://arxiv.org/html/2407.07810v4), [strona autora](https://sugolov.github.io/coupling/).

Weryfikacja: pełny tekst przez Undermind, metryka eq. 9 w v4, część eksperymentalna ViT. Tytuł dotyczy LLM, ale praca obejmuje ViT — nie wolno pominąć tego poprzednika. Zamrażamy **v4**, bo starsza wersja używa innej definicji miernika.

Baseline dotyczy Jacobianu aktualizacji **całego bloku**:
`J_block=(I+J_M)(I+J_A)−I`, z pochodnymi w rzeczywistych stanach forward. Dla kolejnych Jacobianów J1/J2 wykorzystuje bazę singular vectors J2 i porównuje U2,Kᵀ J1 V2,K z Σ1,K; normalizacja również musi odpowiadać wybranej wersji. Stosowanie do samych podkroków lub ilorazu LN oznaczamy jako nasze rozszerzenie. Dokładna recepta: G28 w audycie geometrii. T06–T08, T18, T20, T22, T26.

<a id="r04"></a>
### R04 — Korekta następnego bloku

**Patrawala, A.; Feng, J.; Jones, E.; Steinhardt, J. — LLM Layers Immediately Correct Each Other. NeurIPS 2025.**
[arXiv](https://arxiv.org/abs/2609.07876), [kod autorów](https://github.com/arjunpat/transformer-correction). DOI: 10.52202/085713-4643.

Weryfikacja: pełny tekst przez Undermind i metadane; wersja arXiv pochodzi z 2026 mimo publikacji NeurIPS 2025. Interwencja w rzeczywisty update i odpowiedź następnego bloku jest mocniejszym testem niż sam cosinus dwóch zaobserwowanych przyrostów. T14, T20, T26. Nie mylimy korekty w głębokości z opóźnieniem między checkpointami.

<a id="r05"></a>
### R05 — Ablacje detektorów transformerowych

**Tercan; Meisen — Detection Transformers Under the Knife: A Neuroscience-Inspired Approach to Ablations. 2025.**
[Preprint](https://arxiv.org/abs/2507.21723).

Weryfikacja: metadane i abstrakt pierwotny. To punkt porównania zakresu ablacji w detektorach, a nie gotowy przepis dla naszego dense head. T12–T13, T16, T23. Samo wykonanie ablacji detektora nie jest nową klasą metody.

<a id="r06"></a>
### R06 — Iluzja subspace patching

**Makelov, A.; Lange, G.; Nanda, N. — Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching. 2023.**
[Pełny tekst v2](https://arxiv.org/html/2311.17030v2).

Weryfikacja: odczyt tekstu i kontrprzykładu. Skuteczna zmiana wyjścia po patchingu podprzestrzeni nie wystarcza do wskazania normalnie używanej informacji. T09–T11, T15, T17; wymagane naturalne konteksty, kontrola normy i test konkurencyjnych wyjaśnień.

<a id="r07"></a>
### R07 — Wybór metryk i korupcji patchingu

**Zhang, F.; Nanda, N. — Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. ICLR 2024.**
[Oficjalna publikacja](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2309.16042).

Weryfikacja: oficjalny abstrakt i opis metod. To uzasadnienie zamrożenia korupcji, metryki i lokalizacji interwencji. T09, T11, T13, T15–T16, T24. Dla detekcji trzeba osobno zdefiniować krawędzie, ranking i pełne IoU.

<a id="r08"></a>
### R08 — Self-repair

**McGrath, T.; Rahtz, M.; Kramar, J.; Mikulik, V.; Legg, S. — The Hydra Effect: Emergent Self-repair in Language Model Computations. 2023.**
[Preprint](https://arxiv.org/abs/2307.15771).

Weryfikacja: metadane i abstrakt pierwotny. Brak spadku po ablacji nie implikuje zbędności, bo późniejsze elementy mogą kompensować zmianę. T13–T15. Nie zakładamy, że efekt z LM automatycznie występuje w ViT.

<a id="r09"></a>
### R09 — Kontrole randomizacji

**Adebayo i in. — Sanity Checks for Saliency Maps. NeurIPS 2018.**
[Oficjalna publikacja](https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).

Weryfikacja: publikacja i opis testów randomizacji modelu/danych. Przenosimy zasadę sprawdzania zależności miernika od wyuczonych parametrów, nie twierdzenie że saliency jest tożsame z naszym polem. T12, T17, T20.

<a id="r10"></a>
### R10 — Patching w vision ma już poprzedników

**Izadi i in. — Causal Attribution via Activation Patching. 2026.**
[Preprint](https://arxiv.org/abs/2603.13652).

Weryfikacja: abstrakt pierwotny i fragmenty tekstu przez Scite. CAAP przenosi reprezentacje ViT do neutralnego kontekstu i uwzględnia sąsiedztwa patchy. T10, T15, T23, T26. Nie przedstawiamy tego jako zaakceptowanej pracy konferencyjnej bez potwierdzenia venue.

<a id="r11"></a>
### R11 — Ruch z aktualizacji parametrów

**Pruthi, G.; Liu, F.; Sundararajan, M.; Kale, S. — Estimating Training Data Influence by Tracing Gradient Descent. NeurIPS 2020.**
[arXiv](https://arxiv.org/abs/2002.08484), [oficjalny PDF](https://papers.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf).

Weryfikacja: źródło pierwotne i opis TracIn. Uzasadnia oddzielne analizowanie zmian treningowych, ale prostego iloczynu gradientów SGD nie przedstawiamy jako dokładnej dynamiki AdamW, BN ani rzadkich checkpointów. T19 wymaga rzeczywistego Δθ, buforów i kontroli reszty.

<a id="r12"></a>
### R12 — Podobieństwo reprezentacji jako baseline

**Kornblith, S.; Norouzi, M.; Lee, H.; Hinton, G. — Similarity of Neural Network Representations Revisited. ICML 2019.**
[Oficjalna publikacja PMLR](https://proceedings.mlr.press/v97/kornblith19a.html).

Weryfikacja: metadane, abstrakt i definicja zakresu porównania CKA. T04, T08, T26. Podobieństwo reprezentacji nie identyfikuje przyczynowo wykorzystywanych kierunków.

<a id="r13"></a>
### R13 — Narzędzia interpretowalności vision

**Joseph i in. — Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video. 2025.**
[Pełny tekst v3](https://arxiv.org/html/2504.19475v3).

Weryfikacja: HTML, zakres narzędzia i obsługiwanych analiz. T01, T22, T26. Aktywacje, hooki i wizualizacja są infrastrukturą; publikacyjny wkład trzeba określić przez pytanie i wynik, nie sam interfejs.

<a id="r14"></a>
### R14 — Wariancja między treningami

**Bouthillier i in. — Accounting for Variance in Machine Learning Benchmarks. MLSys 2021.**
[Oficjalna publikacja](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html), [arXiv](https://arxiv.org/abs/2103.03098).

Weryfikacja: źródła pierwotne i metodologia uwzględniania źródeł wariancji. T00, T21, T25, T27–T29. Wiele tokenów jednego modelu nie zastępuje niezależnych treningów.

<a id="r15"></a>
### R15 — Oficjalne zasady raportowania

[NeurIPS Paper Checklist](https://nips.cc/public/guides/PaperChecklist).
[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Weryfikacja: strony oficjalne, 16 października 2026. Wymagają lub zachęcają, zależnie od pozycji, do jawnego opisania założeń, niepewności, odtwarzalności i zasobów. **Nie narzucają naszego katalogu 30 testów ani liczby 3/5/8 seedów.** Termin pełnych zgłoszeń ICLR 2027, 25 września 2026, już minął; plan służy standardowi naukowemu, nie obiecuje zgłoszenia do zamkniętej edycji. Terminy kolejnej docelowej edycji trzeba sprawdzić oddzielnie.

<a id="r16"></a>
### R16 — Co znaczy wynik patchingu

**Heimersheim, S.; Nanda, N. — How to use and interpret activation patching. 2024.**
[Pełny tekst](https://arxiv.org/html/2404.15255v1).

Weryfikacja: noising/denoising, discovery/confirmatory, AND/OR i path patching. Uzasadnia rozdzielenie konieczności, warunkowej wystarczalności i redundancji. T11, T13, T15.

<a id="r17"></a>
### R17 — Wrażliwość miar faithfulness

**Miller, J.; Chughtai, B.; Saunders, W. — Transformer Circuit Faithfulness Metrics Are Not Robust. COLM 2024.**
[Oficjalny PDF](https://openreview.net/pdf?id=zSf8PJyQb2), [HTML](https://arxiv.org/html/2407.08734v1).

Weryfikacja: tekst, wybory granularity/component/value/positions/direction/set. T11, T15–T17, T29. Każdy wynik musi przechowywać dokładne ustawienia interwencji.

<a id="r18"></a>
### R18 — Bootstrap dla czynników skrzyżowanych

**Owen, A.; Eckles, D. — Bootstrapping data arrays of arbitrary order. Annals of Applied Statistics, 2012.**
[arXiv](https://arxiv.org/abs/1106.2125). DOI: 10.1214/12-AOAS547.

Weryfikacja: źródło pierwotne, założenia i wagi poziomów czynników. T21/T27: seedy i wspólne filmy są czynnikami skrzyżowanymi. Przy trzech seedach wymagamy analizy pokrycia/przeciwwskazań; cytat nie gwarantuje poprawności małopróbkowego CI.

<a id="r19"></a>
### R19 — Normalizacja jako możliwa część naprawy

**Rushing, C.; Nanda, N. — Explorations of Self-Repair in Language Models. ICML 2024.**
[Źródło pierwotne](https://arxiv.org/abs/2402.15390).

Weryfikacja: metadane i abstrakt, w tym rola LayerNorm. T14/T15: mierzyć adaptacyjną odpowiedź i osobno zmiany skali/normalizacji.

<a id="r20"></a>
### R20 — Odtworzenie funkcji obwodu

**Hanna, M.; Pezzelle, S.; Belinkov, Y. — Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms. 2024.**
[Pełny tekst](https://arxiv.org/html/2403.17806v1).

Weryfikacja: definicje faithfulness i testy funkcji obwodu. T11/T17: podobieństwo list neuronów nie zastępuje odtworzenia zachowania i transferu efektu.

<a id="r21"></a>
### R21 — Pole niegradientowe w sieci jest znaną możliwością

**Im, D. J.; Belghazi, M.; Memisevic, R. — Conservativeness of Untied Auto-Encoders. AAAI 2016.**
[Oficjalny artykuł](https://ojs.aaai.org/index.php/AAAI/article/view/10268). DOI: 10.1609/aaai.v30i1.10268.

Weryfikacja: metadane i opis analizy konserwatywności. T03–T04: sama niesymetria pochodnej modelu nie jest nowością; nie przenosimy dokładnych twierdzeń dla autokodera na ViT bez dowodu.

<a id="r22"></a>
### R22 — LayerNorm

**Ba, J. L.; Kiros, J. R.; Hinton, G. E. — Layer Normalization. 2016.**
[Źródło pierwotne](https://arxiv.org/abs/1607.06450).
Uzupełnienie: [Xiong i in., On Layer Normalization in the Transformer Architecture, ICML 2020](https://proceedings.mlr.press/v119/xiong20b.html).

Weryfikacja: definicja i umiejscowienie normalizacji; dokładny Jacobian w protokole wyprowadzono dla naszego eps i affine. T01/T02/T04. Nie usuwać automatycznie kierunku radialnego przy niezerowym eps.

<a id="r23"></a>
### R23 — Test jawnej hipotezy mechanistycznej

**Chan i in. — Causal Scrubbing: a method for rigorously testing interpretability hypotheses. 2022.**
[Raport autorów](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), [metadane autora](https://chanlawrence.me/publication/chan-2022-causal/).

Weryfikacja: opis warunkowego resamplingu dopuszczonego przez hipotezę. To raport techniczny, nie recenzowana publikacja konferencyjna. T11/T15: nie nazywać dowolnego shuffle aktywacji „causal scrubbing”.

## Granice wnioskowania z literatury

Literatura określa poprzedników i sensowne kontrole; nie potwierdza naszych hipotez H1–H5. Nowość należy ocenić ponownie po wybraniu głównego wyniku, przed pisaniem twierdzenia „pierwsi”. Nie interpretujemy liczby cytowań ani samej obecności w NeurIPS/ICLR jako dowodu poprawności metody. W pracy należy cytować zarówno zgodne wyniki, jak i kontrprzykłady oraz ograniczenia.

