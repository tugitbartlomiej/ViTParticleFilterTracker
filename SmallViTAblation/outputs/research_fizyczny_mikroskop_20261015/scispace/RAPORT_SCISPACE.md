# SciSpace: mikroskop dynamiki SmallViT — przegląd i ocena nowości

Data: 15.10.2026. Zakres tego podraportu: reprezentacja transformerów jako układów dynamicznych, rozkład MLP, znoszenie wkładów, trajektorie reprezentacji, geometria i projekt pomiarów. Nie uruchomiono eksperymentów i nie zmieniono plików źródłowych.

## Wynik główny

Najmocniejsze uzasadnienie artykułu to **ilościowy mikroskop dwóch osi czasu w rzeczywistym, wytrenowanym ViT**, z dokładnym rozkładem algebraicznym, pełnowymiarową kontrolą projekcji i przewidywaniem skutków interwencji. Samo traktowanie tokenów jako cząstek, warstw jako kroków dynamiki albo neuronów MLP jako wkładów kierunkowych jest już znane. Szczególnie bliscy poprzednicy to Lu (2019/2020), Geva (2021, 2022), Geshkovski (2023/2025), Rushing i Nanda (2024) oraz Tomihari i Karakida (2025).

W samym korpusie 12 zapytań SciSpace nie ustalono pracy, która łączyłaby **dokładne wkłady wszystkich neuronów GELU, lokalne pola i odpowiedź na perturbacje oraz rozwój tych wielkości przez setki checkpointów małego detektora ViT**. Jednak równoległe wyszukiwanie Undermind znalazło trzy bardzo bliskie preprinty z 2026 roku (niżej), dlatego nie wolno traktować tego braku w SciSpace jako potwierdzenia luki. Ocena pierwszeństwa musi uwzględnić ich pełnotekstową analizę z raportu głównego.

## Wykonane wyszukiwanie i jakość dowodów

- 12 różnych pytań pełnymi zdaniami w `SciSpace.search_papers`.
- 120 zwróconych rekordów, 113 różnych identyfikatorów SciSpace. Ta liczba nie oznacza 113 różnych prac: duplikaty preprint/publikacja występują też pod różnymi identyfikatorami.
- 4 wywołania `add_column`, łącznie 8 ekstrakcji dla wybranych prac.
- Dodatkowa weryfikacja w źródłach pierwotnych: ACL Anthology, arXiv, PMLR, NeurIPS, repozytoria autorów/uczelni.
- Pliki `query_01.json`–`query_12.json` zachowują wyniki narzędzia; `column_1.json`–`column_4.json` jego ekstrakcje; `search_manifest.json` wszystkie pytania i metadane.
- `web_verify_1.json`–`web_verify_7.json` zachowują późniejsze weryfikacje stron i pełnych tekstów. Pierwszy, wstępny odczyt stron ACL nie został osobno zapisany; jego URL-e są w raporcie.
- **Abstrakt przeczytany** oznacza ocenę na podstawie opisu autorów. **Fragmenty pełnego tekstu** oznaczają przeczytane wskazane części metod/wyników, nie audyt całej pracy ani odtworzenie eksperymentów.
- SciSpace często oznacza arXiv jako „Journal Article”, nawet gdy brak potwierdzonej publikacji. Takie etykiety nie są dowodem recenzji.
- `add_column` zwróciło teksty bez cytowanych fragmentów (`context_map` puste). Co najmniej jedna odpowiedź o „braku limitations” została obalona pełnym tekstem: Activation Transport Operators ma sekcję Limitations. Dlatego ekstrakcje traktowano jako pomoc wyszukiwania, a nie źródło faktów.
- W wynikach znaleziono starsze tytuły tych samych identyfikatorów arXiv; w bibliografii należy używać wersji faktycznie cytowanej.
- To szerokie wyszukiwanie eksploracyjne, nie przegląd systematyczny z kontrolą kompletności i podwójnym screeningiem.

Pytania dotyczyły kolejno: (1) dynamiki cząstek i splittingu, (2) MLP jako key–value, (3) trajektorii stałych obrazów przez uczenie, (4) cancellation/GELU, (5) geometrii/CKA/gauge, (6) Jacobianów i stabilności, (7) Geva2022, (8) Kornblith2019, (9) LayerNorm, (10) NTK i wspólnych aktualizacji, (11) OU/szkła, (12) mechanistycznej ablacji i self-repair. Pełne brzmienie każdego pytania jest w manifeście.

## Najbliższe źródła i co rzeczywiście wspierają

| Praca i status | Sprawdzenie | Znaczenie dla SmallViT i granica wniosku |
|---|---|---|
| Lu i in., **Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View**, arXiv 2019; ICLR2020 Workshop, niepotwierdzona główna konferencja. [arXiv](https://arxiv.org/abs/1906.02762), [strona autora](https://web.stanford.edu/~yplu/pub.html), DOI 10.48550/arXiv.1906.02762 | Abstrakt oraz metadane autora | Istniejący most do równań różniczkowych, oddziaływania cząstek, Lie–Trotter i Euler; inspiruje Macaron. Czas oznacza głębokość. Nie dowodzi termodynamiki trajektorii tokenu podczas AdamW. |
| Geva, Schuster, Berant, Levy, **Transformer Feed-Forward Layers Are Key-Value Memories**, EMNLP2021. [ACL](https://aclanthology.org/2021.emnlp-main.446/), DOI 10.18653/v1/2021.emnlp-main.446 | Abstrakt i metadane wydawcy | Bardzo bliski poprzednik pary „celownik–wiązka”: pierwszy wektor rozpoznaje wzorzec, drugi wnosi kierunek do reprezentacji/predykcji. Interpretowalność wzorców pokazana w modelach językowych; nie gwarantuje jednoznacznego znaczenia każdemu neuronowi detektora. |
| Geva, Caciularu, Wang, Goldberg, **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space**, EMNLP2022. [ACL i PDF](https://aclanthology.org/2022.emnlp-main.3/), DOI 10.18653/v1/2022.emnlp-main.3 | Fragmenty pełnego PDF: §5.2 i A.3 | Rozkłada aktualizację MLP na wkłady pojedynczych wektorów; bada silne dodatnie i ujemne zmiany oraz klastry wektorów. A.3 jawnie uwzględnia ujemne GELU i ranking przez moduł aktywacji razy normę wektora. To bezpośredni prior dla „jasności ze znakiem” i sumowania wiązek. |
| Geshkovski, Letrouit, Polyanskiy, Rigollet, **A mathematical perspective on Transformers**, arXiv2023; **Bulletin AMS 62 (2025), 427–479**. [arXiv](https://arxiv.org/abs/2312.10794), [opublikowany PDF](https://par.nsf.gov/servlets/purl/10633837), DOI 10.1090/bull/1863 | Fragmenty pełnego tekstu arXiv v2: §1–3; zweryfikowane nowsze metadane publikacji | Formalny język cząstek, miar i klasteryzacji. Analiza dotyczy uproszczonej dynamiki uwagi z normalizacją; pełne MLP pozostawiono poza ówczesną analizą. Hipotez sferycznych, granicy ciągłej i głębokiego limitu nie wolno bez sprawdzenia przenosić na 6 pre-LN bloków. |
| Tomihari, Karakida, **Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians**, **NeurIPS2025**. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cca569b38ec3612904e625dad7a743dc-Abstract-Conference.html), [pełny tekst](https://arxiv.org/html/2505.19458v4), DOI 10.52202/085713-4675 | Pełny tekst: wstęp, definicje, §4–5, fragment dyskusji | Najbliższy prior dla dynamiki bez potencjału: Jacobiany, złożone wartości własne, normalizacja, wykładniki Lyapunova. Badany przypadek jest rekurencyjny; 12 różnych podkroków SmallViT to inny obiekt niż iterowanie jednej mapy. Nowością nie może być samo użycie Jacobianu. |
| Rushing, Nanda, **Explorations of Self-Repair in Language Models**, **ICML2024, PMLR235:42836–42855**. [pełny tekst](https://arxiv.org/html/2402.15390v2), DOI preprintu 10.48550/arXiv.2402.15390 | Pełny tekst: §2–5 i ograniczenia | MLP może częściowo negować wcześniejsze wkłady; ablacja zmienia dalsze wkłady i skale LayerNorm. „Mały efekt ablacji” nie oznacza automatycznie „mały wkład”. Erasure wymaga związku przyczynowego; sam przeciwny znak lub korelacja nie wystarcza. |
| Gurnee i in., **Universal Neurons in GPT2 Language Models**, **TMLR, czerwiec2024**. [recenzowana wersja](https://openreview.net/forum?id=ZeI104QZ8I), [arXiv](https://arxiv.org/abs/2401.12181) | Abstrakt oraz nagłówek publikacji w PDF | Bada powtarzalność neuronów między seedami oraz role prediction/suppression. To właściwy wcześniejszy punkt odniesienia dla neuronów tłumiących, odrębny od nowszych podobnie nazwanych preprintów zwróconych przez SciSpace. |
| Kornblith i in., **Similarity of Neural Network Representations Revisited**, **ICML2019**. [PMLR](https://proceedings.mlr.press/v97/kornblith19a.html), [PDF](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf) | PDF §2.1–2.3 i metadane | CKA umożliwia porównania niezmiennicze na ortogonalne transformacje i globalną skalę; pełna niezmienniczość na odwracalne mapy liniowe bywa zbyt silna. CKA kontroluje podobieństwo geometrii próbek, ale nie identyfikuje przyczyny ani konkretnej trajektorii neuronu. |
| Raghu i in., **Do Vision Transformers See Like Convolutional Neural Networks?**, **NeurIPS2021**. [Proceedings](https://papers.nips.cc/paper/2021/hash/652cf38361a209088302ba2b8b7f51e0-Abstract.html) | Abstrakt i rekord proceedings | Ważny punkt odniesienia dla analiz reprezentacji ViT, propagacji cech przez residual i lokalizacji. Głównie struktura reprezentacji przez warstwy; nie dokładne ścieżki wszystkich neuronów przez 806 checkpointów. |
| Sharon, Dar, **How Do the Architecture and Optimizer Affect Representation Learning? On the Training Dynamics of Representations in Deep Neural Networks**, arXiv2405.17377, wersja2 z 2025. [arXiv](https://arxiv.org/abs/2405.17377) | Aktualny abstrakt | CKA oraz sondy liniowe w trakcie uczenia; zachowanie zależy od architektury i optymalizatora. SciSpace zwróciło stary tytuł „How Does Perfect Fitting Affect…”. To ważna kontrola dla twierdzenia, że sama analiza ewolucji reprezentacji jest nowa. Brak tu potwierdzonego statusu recenzowanej publikacji. |
| Varam i in., **Class Separation Dynamics in Vision Transformers: An Empirical Study**, **IEEE Access2025**, DOI 10.1109/ACCESS.2025.3602733. [DOI](https://doi.org/10.1109/ACCESS.2025.3602733) | Abstrakt SciSpace; potwierdzenie DOI/statusu w manuskrypcie autorów, bez pełnego audytu wydawcy | Bada separację klas przez głębokość i epoki. Bliski prior dla dwóch osi czasu w ViT, ale obserwablą jest geometria klas, nie pola MLP. Szczegóły replikacji i jakość praw empirycznych wymagają osobnego pełnego odczytu. |
| Riechers, **Geometry and Dynamics of LayerNorm**, arXiv2405.04134. [arXiv](https://arxiv.org/abs/2405.04134) | Abstrakt | LayerNorm to projekcja, skalowanie i przekształcenie afiniczne, zależne od wyróżnionej bazy. Preprint/technical note; pomaga uniknąć utożsamienia całego residual streamu ze sferą. |
| Szablewski, Masiak, **Activation Transport Operators**, arXiv2508.17540v2. [pełny tekst](https://arxiv.org/html/2508.17540v2) | Abstrakt oraz fragment metody, spis sekcji | Liniowe mapy transportu reprezentacji między warstwami, sprawdzane w przestrzeni cech SAE. Potencjalny baseline transportu, nie odpowiednik pól pojedynczego MLP. Potwierdzono preprint; szablon workshop w PDF sam nie dowodzi przyjęcia. |
| Mandt, Hoffman, Blei, **Stochastic Gradient Descent as Approximate Bayesian Inference**, **JMLR18(134), 2017**. [JMLR](https://jmlr.org/beta/papers/v18/17-214.html), [PDF](https://jmlr.csail.mit.edu/papers/volume18/17-214/17-214.pdf) | Abstrakt i metadane | Uzasadnia lokalne przybliżenia stochastycznej dynamiki parametrów dla stałego SGD; opisuje też momentum. Nie ustanawia automatycznie OU dla nieliniowej obserwabli tokenu podczas AdamW z clippingiem i zmienną głową. |
| Baity-Jesi i in., **Comparing Dynamics: Deep Neural Networks versus Glassy Systems**, **ICML2018**, później **JSTAT2019,124013**. [ICML PDF](https://proceedings.mlr.press/v80/baity-jesi18a/baity-jesi18a.pdf), [repozytorium autorów](https://iris.uniroma1.it/handle/11573/1472319), DOI10.1088/1742-5468/ab3281 | Abstrakt obu publikowanych wersji i metadane | Badanie dynamiki wag narzędziami fizyki szkła; wskazuje też różnice zależne od nadparametryzowania. Nie jest dowodem szklistego stanu tokenów małego ViT. |
| **Glassy dynamics in deep neural networks: A structural comparison**, arXiv2405.13098v4; starsze tytuły „How glassy…” i „Do deep neural networks behave…”. [aktualny rekord](https://arxiv.org/abs/2405.13098), [czytana wersja3](https://arxiv.org/html/2405.13098v3) | Fragment pełnego tekstu v3, §III i metadane v4 | Przykład ważnej kontroli: niektóre podobieństwa potęgowe wynikają z regularyzacji, a nie szkła. Odczyt starszej wersji nie uprawnia do przypisania wszystkich detali najnowszej bez kontroli zmian. Status publikacji niepotwierdzony. |

## Dokładna matematyka i granice analogii

### Uzupełnienie najbliższego prior art z równoległego Undermind

Główny agent przekazał poniższe prace po weryfikacji oficjalnych stron arXiv. Ich pełnych tekstów nie czytałem w tym podzadaniu, aby nie powielać jego lektury. Nie są to pozycje znalezione w powyższych 12 zapytaniach SciSpace, a ich szczegółową ocenę należy zaczerpnąć z raportu głównego.

| Praca | Identyfikator i status | Konsekwencja dla niniejszego raportu |
|---|---|---|
| Mudarisov, Burtsev, State, **Feed-Forward Steering in Transformer Residual Dynamics** | [arXiv2608.02071](https://arxiv.org/abs/2608.02071), preprint2026; istnienie/metadane zweryfikowane przez głównego agenta | Potencjalnie bezpośrednio przecina proponowane pozycjonowanie pól MLP. Twierdzenie o nowej roli FFN jako sterowania musi zostać porównane szczegółowo. |
| Fernando, Guitchounts, **Dynamics of Transformer Residual Stream Coupling Spectral Geometry to Network Topology** | [arXiv2605.14258](https://arxiv.org/abs/2605.14258), preprint2026; istnienie/metadane zweryfikowane przez głównego agenta | Potencjalnie bliski prior dla analizy spektralnej i sprzężeń residual streamu. H3 wymaga rozróżnienia wobec tej pracy. |
| Halperin, **Observable Matrix Dynamics** | [arXiv2606.29679](https://arxiv.org/abs/2606.29679), preprint2026; istnienie/metadane zweryfikowane przez głównego agenta | Potencjalnie istotne ramy dynamiki obserwabli. H2 trzeba pozycjonować po pełnym porównaniu definicji stanu i czasu. |

Do czasu zakończenia tej analizy tytuł i teza „two-time-scale mechanistic microscope” są propozycją organizacji wyników, a nie stwierdzeniem nowości metodologicznej.

Poniższe punkty są analizą matematyczną konkretnej konstrukcji z plików lokalnych, nie wynikiem wykonanych tu eksperymentów.

### 1. Stan, odczyt i obie osie czasu

Dla tego SmallViT stan tokenu w residual streamie ma **192 współrzędne**. Wektor **768 aktywacji GELU** jest kodem aktywacji MLP, nie tym samym stanem. 600 tokenów to fragmenty jednego obrazu, nie paczki treningowe.

W ustalonym checkpoincie można dokładnie zapisać krok MLP:

```
u_c(x) = GELU(w_c^T LN(x) + b_c) v_c
F(x) = sum_c u_c(x) + b2
x_next = x + F(x)
```

Jest to dyskretna mapa rezydualna. Interpretacja jako krok Eulera jest matematycznie naturalna; interpretacja jako realizacja danego ciągłego procesu fizycznego potrzebuje dodatkowej definicji czasu i limitu. Wagi różnią się między blokami. Punkt zerowy jednego pola nie musi być celem kolejnego bloku.

Przy uczeniu stałego obrazu zapis ma inną postać:

```
z_j(t) = f_j(image; theta_t, running_statistics_t)
Delta z_j ≈ J_theta,j Delta theta + J_statistics,j Delta statistics
```

AdamW pamięta momenty; łodyga ma BatchNorm i jego statystyki. Sprowadzenie całego stanu do samego tokenu lub samych wag może ukrywać pamięć i wymuszać pozornie niemarkowowską dynamikę. Wspólne aktualizacje powodują korelacje wielu tokenów nawet bez bezpośredniej siły między nimi w osi uczenia.

### 2. „Jasność” nie jest natężeniem światła

GELU ma znak. Każdy składnik jest wektorem o amplitudzie ze znakiem; „jasność” jest konwencją wizualizacji. Nie ma jednostek siły, energii, masy, tarcia lub temperatury wynikających z samej architektury. „Reflektor” jest obrazem pary wektorów, nie położonym w przestrzeni emitującym laserem. Znane prace key–value uzasadniają rozkład funkcjonalny, nie prawa optyki.

Sam wzrost liczby par o ujemnym cosinusie nie wykazuje aktywnego hamowania: dwa wektory mogą nie być współaktywne. Potrzebny jest rozkład ich rzeczywistych wkładów na danych i interwencja.

Przydatne, rozdzielone wskaźniki:

```
cancellation(x) = 1 - ||sum_c u_c(x)|| / sum_c ||u_c(x)||
signed_share_c = <u_c, sum_d u_d> / ||sum_d u_d||^2
```

Pierwszy mierzy znoszenie w całej przestrzeni i należy do [0,1] przy niezerowym mianowniku. Drugi sumuje się do 1, ale może być ujemny lub przekraczać 1; nie jest udziałem probabilistycznym. Bias należy analizować osobno albo jawnie dodać jako dodatkowy składnik.

### 3. Jacobian: pole, mapa i pełny układ to różne obiekty

```
J_F(x) = W2 diag(GELU'(W1 LN(x)+b1)) W1 J_LN(x)
J_map(x) = I + J_F(x)
```

Ślad J_F jest dywergencją lokalnego pola. Zmianę małej objętości przez skończony podkrok mierzy log|det(I+J_F)|. Ujemny ślad nie wystarcza do stabilności dyskretnego kroku. Symetryczna część J_F opisuje lokalne rozciąganie/ściskanie w metryce euklidesowej; antysymetryczna lokalną rotację. Nie jest to automatycznie globalny rozkład Helmholtza na potencjał i wir.

Uwaga zależy od wszystkich tokenów. Jacobian 192×192 przy zamrożeniu pozostałych tokenów jest blokiem warunkowym. Pełny stan ma 600×192=115200 współrzędnych. Stabilności całego sprzężonego układu nie ustala się z samego bloku diagonalnego; praktycznie można badać ją JVP/VJP i perturbacjami bez budowania całej macierzy.

### 4. LayerNorm i projekcje

Dla eps>0 normowany odczyt ma długość zależną od wariancji:

```
||center(x)/sqrt(var(x)+eps)||^2 = d*var(x)/(var(x)+eps)
```

Sfera o promieniu sqrt(d) jest granicznym przybliżeniem przy pomijalnym eps, przed gamma i beta. Residual stream pre-LN nie jest ograniczony do tej sfery. Z samego ujemnego cosinusa celownik–wiązka nie wynika dokładny ślad: liczy się pochodna GELU oraz pełny Jacobian LayerNorm z projekcją i skalą.

Dla stałej ortogonalnej projekcji P lokalne pole przekroju ma Jacobian P J P^T. Jego ślad, punkt zerowy i wartości własne nie reprezentują automatycznie pełnego J. Projekcja może ukryć pchnięcie: P F(x)=0 nie oznacza F(x)=0. Jeżeli baza PCA jest osobno dopasowana w każdym checkpoincie, ruch projekcji dodatkowo miesza się z ruchem reprezentacji. Potrzebne są jedna ustalona baza do filmu oraz pomiary w 192D.

CKA i Procrustes są kontrolami porównania reprezentacji. Nie każdy obrót przestrzeni jest rzeczywistą symetrią całej sieci z LayerNorm, stałym kodowaniem pozycji i nieliniowościami. Należy odróżnić zmianę współrzędnych obserwatora od reparametryzacji zachowującej funkcję modelu.

## Trzy hipotezy nadające się do falsyfikacji

### H1. Uczenie tworzy funkcjonalne zespoły znoszących się wkładów MLP

**Przewidywanie:** wzrost ujemnych relacji między wektorami wartości podczas uczenia wystąpi razem ze wzrostem współaktywowanego znoszenia na obrazach, a naruszenie tych relacji zmieni wybrane odpowiedzi bardziej niż kontrola dobrana według normy, aktywności i warstwy.

**Test:** śledzić pełne 192D wkłady, mierzyć cancellation, współaktywacje i zależność od tokenów obiektu/tła; porównać seed, zdjęcie, model. Wykonać sparowane interwencje na neuronach lub grupach, resample/mean ablation i pomiar bezpośredniej zmiany residual oraz końcowej głowy.

**Falsyfikacja:** efekt znika po kontroli anisotropii/aktywnych norm albo pary nie współdziałają funkcjonalnie; podobny efekt dają losowe grupy. Sam histogram cosinusów nie potwierdza H1.

**Różnica względem prior:** Geva2022 opisuje signed sub-updates, Rushing2024 erasure. Potencjalny wkład to geneza i przewidywalna funkcja tej organizacji przez uczenie rzeczywistego ViT.

### H2. Pozorne uwięzienie tokenu w osi uczenia jest przewidywalne z aktualizacji wspólnego stanu modelu

**Przewidywanie:** model zmian reprezentacji uwzględniający aktualizacje bloków, głowy, BatchNorm i stan optymalizatora przewidzi kolejne przyrosty lepiej niż autonomiczny OU dopasowany tylko do trajektorii tokenu; znacząca część korelacji między tokenami będzie wspólnym napędem.

**Test:** walidacja na przyszłych checkpointach, bez wygładzania z użyciem przyszłości; porównanie AR/OU, modelu z dryfem, modelu projekcji aktualizacji wag oraz wariantów z pamięcią; oddzielne okna przed/po zmianie rozdzielczości zapisu. Do identyfikacji szybkiego czasu relaksacji potrzebny zapis znacznie gęstszy niż ten czas.

**Falsyfikacja:** po odjęciu przewidywanego napędu pozostają stabilne, odtwarzalne prawa relaksacji i odpowiedzi interwencyjnej, których modele wspólnego stanu nie przewidują.

**Różnica względem prior:** Mandt dotyczy dynamiki parametrów w określonym przybliżeniu; tutaj testowana jest nieliniowa obserwabla z realnego AdamW.

### H3. Pełnowymiarowa odpowiedź różniczkowa przewiduje znaczenie perturbacji lepiej niż obraz 3D

**Przewidywanie:** singular values skończonej mapy i kierunki czytane przez głowę lepiej przewidzą rzeczywistą zmianę logitu/ramki niż etykieta „studnia/siodło” z rzutu 3D.

**Test:** małe perturbacje o dobranej normie w kierunkach głowy, najsilniejszego wzrostu oraz kontrolnych; wyznaczyć zakres liniowości i sprawdzać predykcję na nowych obrazach/checkpointach. Porównać mapę pojedynczego MLP, całego bloku i całego przejścia.

**Falsyfikacja:** mierniki pełnej mapy nie poprawiają przewidywania ponad proste normy/aktywizacje albo pozorna przewaga wynika z wyboru tokenu 226 i jednej klatki.

**Różnica względem prior:** Tomihari2025 bada rekurencyjną uwagę i stabilność; tutaj celem jest mechanistycznie sprawdzony pomiar zmieniającej się sieci detekcyjnej i zgodność obserwacji z różnych przestrzeni.

## Najważniejsze korekty przed składaniem artykułu

1. Lokalny opis z 15.10 mówi o dodatniej dywergencji MLP i wzroście objętości w pomiarach 3D; Z6 z 14.10 podaje ujemny ślad oraz kontrakcję w 192D. To może być zgodne przy różnych projekcjach i definicjach, ale musi zostać jawnie wyjaśnione wspólnym zestawem definicji, jednostek i map. Nie wolno łączyć liczb jak jednego pomiaru.
2. Wnioskowanie o czasie powrotu 8–50 aktualizacji z checkpointów co100 ma problem identyfikowalności. Nie wykazuje zaobserwowanej trajektorii w tej skali.
3. Średnia symetryczna z przyszłymi checkpointami nie jest przyczynowym centrum pułapki. OU dopasowany do reszt po takim odjęciu wymaga porównania z artefaktem filtracji.
4. Przy głowie 192→5 rząd5 daje lokalnie, w regularnym punkcie, zbiór poziomic o wymiarze187. Nie jest automatycznym twierdzeniem o całej globalnej dolinie ani o minimum straty.
5. Jedna klatka i jeden token mogą być studium przypadku. Twierdzenia ogólne o narzędziach/tle lub transformerach wymagają replikacji wielu klatek, tokenów, seedów i przynajmniej jednego innego ustawienia architektury/danych.
6. Uzasadnienie nowości powinno wymieniać najbliższe negatywne przypadki: istniejące signed-GELU attribution, attention-as-particles, Jacobian dynamics i train-time representation tracking. Wtedy wkład „microscope” będzie konkretny i obronny.

## Proponowane pozycjonowanie

Robocza teza: **A two-time-scale mechanistic microscope for a trained vision transformer**. Przedmiotem pracy byłby mierzalny związek między składnikami obliczeń, zmianą geometrii przez uczenie i odpowiedzią na interwencje. „Optical tweezers” może być ilustracją/słownikiem interfejsu; główne wnioski powinny być sformułowane w języku pól rezydualnych, przepływu reprezentacji, odpowiedzi i wspólnego napędu parametrów.

Sam atlas czy efektowny film daje słabszy artykuł. Jedna dobrze potwierdzona hipoteza z powyższych, wsparta otwartym narzędziem i kontrolą pełnego wymiaru, jest mocniejsza od wielu analogii bez rozstrzygającego testu.

