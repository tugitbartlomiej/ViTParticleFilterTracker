# 37. Plan eksperymentu: Koopman i Langevin w diagnostyce reprezentacji obrazów i filmów

**Data:** 2026-09-21. **Status:** plan badania, przed implementacją poprawek i przed nowymi pomiarami.
**Cel:** ustalić, co dynamika reprezentacji wnosi do diagnostyki detektora. Dobór już oznaczonych klatek do treningu zwykłego RF-DETR opisuje teraz plan 39; budżet dotyczy danych końcowego treningu, nie nowych anotacji.
**Projekt:** `SmallViTAblation`, CATARACT-LMM, zamrożone reprezentacje obrazu i RF-DETR Nano.

Plan rozwija [dokument 36](36_KOOPMAN_LANGEVIN_TOZSAMOSC_NARZEDZIA_I_DOBOR_KLATEK.md) i koryguje jego nadmierne interpretacje. Dokument 36 zachowuje rolę zapisu historycznego. Ten plan określa, co dopiero należy wykonać, jak to mierzyć i jakie wnioski wolno wyciągnąć. Nie zakłada dodatniego wyniku. Zapisanie planu nie oznacza uruchomienia obliczeń.

**Aktualizacja po rozmowie z autorem:** [39 — selekcja klatek RF-DETR i rozwój JEPA](39_PLAN_SELEKCJI_KLATEK_RFDETR_I_ROZWOJU_JEPA.md) jest głównym protokołem selekcji. Używa train GT, strat detektora i ROI narzędzia, porównuje je z dynamiką oraz zachowuje standardową inferencję. Ten dokument pozostaje zapleczem diagnostycznym; rozbudowana diagnostyka B/C nie jest warunkiem logicznym dodatniego wyniku selekcji. Dokument 38 zawiera późniejsze wyniki eksploracyjne z cech RF-DETR.

## 1. Pytania badawcze i zakres

Badanie ma trzy rozdzielone cele:

1. **Reprezentacja:** czy cechy obrazu zachowują informację o stanie narzędzia, a historia poprawia opis jego zmian?
2. **Diagnostyka detektora:** czy sygnał z historii i reszt pozwala rozpoznawać błędy lokalizacji ponad confidence, ruch i jakość obrazu?
3. **Dobór danych:** czy dodanie takiego sygnału do pokrycia reprezentacji poprawia detektor przy tej samej liczbie oznaczonych klatek i tej samej recepturze treningu?

Wynik dodatni w punkcie 1 nie dowodzi punktu 2 ani 3. Możliwy pełnoprawny rezultat to dobra diagnostyka bez poprawy selekcji. Analiza nie ma dowodzić, że latent jest fizyczną cząstką ani osiągać arbitralnego celu „95 procent”.

**Główna hipoteza H1:** historia reprezentacji tego samego narzędzia poprawia predykcję jego stanu na innych operacjach ponad odpowiednio elastyczny model bieżącego stanu.

**H2:** sygnał dynamiki poprawia wykrywanie błędów detektora po uwzględnieniu prostych sygnałów porównawczych.

**H3, doprecyzowana w planie 39:** sygnał dynamiki poprawia selekcję ponad stratę detektora i różnorodność przy równym budżecie końcowego treningu; osobne kontrasty badają historię oraz zwykły model autoregresyjny.

**H4, osobny wariant obrazowy:** kontrolowane przekształcenia tego samego obrazu ujawniają, czy reprezentacja zachowuje semantykę i poprawnie koduje zmianę geometrii. Ta hipoteza nie wymaga naturalnej sekwencji wideo.

## 2. Co uzasadnia literatura

Przeprowadzono wyszukiwanie wskazanym przez autora CLI `paper-search` oraz weryfikację stron źródeł pierwotnych. Rejestr zapytań, surowe odpowiedzi i granice odczytu: [evidence_latent_cv_2026-09-21](evidence_latent_cv_2026-09-21/README.md). Jest to przegląd celowany, a nie systematyczny ani pełnotekstowy przegląd całej dziedziny.

| ID | Publikacja, autorzy i rok | Co wnosi do planu | Granica przeniesienia |
|---|---|---|---|
| L1 | Comas, Ghimire, Li, Sznaier, Camps, 2021, *Self-Supervised Decomposition, Disentanglement and Prediction of Video Sequences while Interpreting Dynamics: A Koopman Perspective*. [arXiv:2110.00547](https://arxiv.org/abs/2110.00547) | Reprezentacja obiektu, rozdzielenie wyglądu i dynamiki oraz predykcja w osadzeniu Koopmana. | Praca opisuje testy syntetyczne. Nie potwierdza poprawy detekcji końcówki ani wartości anotacji. |
| L2 | Comas, Fernandez Lopez, Ghimire, Li, Sznaier, Camps, 2023, *Learning Object-Centric Dynamic Modes from Video and Emerging Properties*, L4DC. [PMLR 211](https://proceedings.mlr.press/v211/comas23a.html) | Bezpośredni precedens modelowania dynamiki obiektów z wideo, analizy modów oraz prognozy trajektorii. | Nie przenosimy wyników na chirurgię bez pomiaru. To odrębny rekord bibliograficzny od L1. |
| L3 | Lenc, Vedaldi, 2015, *Understanding Image Representations by Measuring Their Equivariance and Equivalence*, CVPR. [arXiv:1411.5908](https://arxiv.org/abs/1411.5908) | Kontrolowana zmiana obrazu i pomiar odpowiadającej jej zmiany reprezentacji; podstawa wariantu H4. | Detekcja wymaga odpowiedniej zmiany lokalizacji, a nie niezmienności wszystkich współrzędnych. |
| L4 | Lin, Tian, Anghel, Livescu, 2021, *Data-driven learning for the Mori-Zwanzig formalism: a generalization of the Koopman learning framework*. [arXiv:2101.05873](https://arxiv.org/abs/2101.05873) | Związek ograniczonej obserwacji z pamięcią oraz estymacja operatorów Markowa i pamięci z danych. | Przykład numeryczny dotyczy Lorenz-96. Nie jest walidacją GLE dla obrazów ani gwarancją odzyskania fizycznego szumu. |
| L5 | Wu, Marquez-Neila, Zheng, Rafii-Tari, Sznitman, 2024, *Correlation-Aware Active Learning for Surgery Video Segmentation*, WACV; preprint 2023. [COWAL](https://arxiv.org/abs/2311.08811) | Kontrola korelacji i redundancji klatek, różnorodność reprezentacji oraz niepewność modelu. | Segmentacja to inne zadanie niż detekcja końcówki. Wersję porównawczą dla ramek nazwać adaptacją, nie reprodukcją COWAL. |
| L6 | Shi, Jin, Dou, Heng, 2020, *LRTD: Long-Range Temporal Dependency based Active Learning for Surgical Workflow Recognition*. [arXiv:2004.09845](https://arxiv.org/abs/2004.09845) | Uzasadnia sprawdzanie zależności czasowych przy aktywnym doborze danych; słabsza zależność może wskazywać wartościowy klip. | Rozpoznawanie faz Cholec80 nie dowodzi, że dowolna duża reszta latentu jest użyteczna dla detekcji. |
| L7 | Brunton, Brunton, Proctor, Kutz, 2016, *Koopman Invariant Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems for Control*. [DOI](https://doi.org/10.1371/journal.pone.0150171) | Wyjaśnia rolę obserwabli i ograniczenia skończonej reprezentacji liniowej. | Skończona regresja EDMD jest przybliżeniem; nie gwarantuje globalnego liniowego domknięcia rzeczywistego układu. |

Wniosek projektowy z tych prac: analizować obiekt i jawnie określone przejścia, porównać pamięć z elastycznością modelu bez pamięci, a wartość klatki sprawdzać przez zadanie końcowe. Nie ma podstaw do automatycznego odrzucania klatek o dużej reszcie.

## 3. Punkt wyjścia: co już zmierzono

Poniższe wartości są historycznymi wynikami z dokumentu 36 i lokalnych JSON-ów, a nie wynikami nowego protokołu.

| Obserwacja | Wynik historyczny | Co wolno powiedzieć |
|---|---|---|
| DMD na cechach całej klatki, 8 filmów | Trafienie szczytu mapy w ramkę: 0,353; różnica klatek: 0,398. | Ta implementacja DMD nie pokazała przewagi lokalizacyjnej nad prostą kontrolą. |
| Tory GT, krok 5 klatek, 167 grup | Redukcja błędu względem braku ruchu: dryf 25,95; pamięć 28,12; korektor reszt 27,78 procent. | W tym pomiarze historia dawała mały zysk. Wynik wymaga powtórzenia po izolacji foldów. |
| Latent całej klatki | Dokument 36 raportuje 15,7 procent redukcji błędu. | Może dotyczyć sceny lub oświetlenia, niekoniecznie narzędzia. |
| Duże reszty na 8 filmach | Związek z szybkim ruchem oraz wejściem/wyjściem narzędzia. | Duża reszta nie jest synonimem bezużytecznej lub błędnej etykiety. |
| Detektor po nowej selekcji | Brak wyniku przedstawionego w dokumencie 36. | Nie stwierdzono przewagi nowego selektora. |

Źródła liczb: [mapy ruchu](../../results/koopman_langevin_smoke/koopman_tool_report.json), [tory, stride 5](../../results/koopman_langevin_smoke/tracks/track_dynamics_stride5.json), [dokument 36](36_KOOPMAN_LANGEVIN_TOZSAMOSC_NARZEDZIA_I_DOBOR_KLATEK.md).

Nie interpretować niewyjaśnionej części błędu jako oszacowania nieusuwalnego szumu. Nie interpretować wolnych modów jako dowodu tożsamości narzędzia. Obecna część trackingowa obejmuje cystotom i sama nie wystarcza do badania rozróżniania klas narzędzi.

## 4. Obiekt, czas i modele matematyczne

### 4.1. Co jest obserwowane

Zamrożony encoder `f` przekształca obraz `I_t` w reprezentację. Porównujemy:

- **R0, geometria:** środek ramki podzielony przez rozmiar obrazu, logarytm względnej szerokości i wysokości oraz widoczność. Końcówkę mierzymy osobno tam, gdzie istnieje właściwa anotacja.
- **R1, scena:** globalny latent całej klatki. Jest kontrolą wpływu oka, tła i oświetlenia.
- **R2, ROI-GT:** latent obszaru narzędzia wyznaczonego przez GT, razem z geometrią. To wariant korzystający z anotacji; jest legalnym głównym wejściem nadzorowanego selektora planu 39, ale nie selektorem bez etykiet.
- **R3, ROI-detektor:** latent obszaru wyznaczonego przez detektor, razem z przewidywaną geometrią, widocznością i informacją o jakości przypisania. To wariant użytkowy.

ROI po zmianie rozmiaru może usunąć informację o translacji i skali, dlatego geometria jest osobnym kanałem. Nie porównujemy jakości różnych reprezentacji wyłącznie przez ich własne MSE. Wspólnym celem zadaniowym jest geometria narzędzia i błąd detekcji.

Dla diagnostycznego wariantu R1: DINOv2 ViT-S/14 z identycznymi wagami i preprocessingiem we wszystkich ramionach. Receptura Static36 wskazuje commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; przed użyciem porównać go z manifestem cache. Domyślne `:main` w `tokens.py` zastąpić przypiętą wersją. Użyć `eval()` i wyłączyć losowe augmentacje podczas ekstrakcji.

Dla ROI podstawą jest pooling tokenów pokrywających ramkę; kontrola wpływu rozdzielczości to osobne kodowanie cropu 224 x 224. Nie mieszać tych dwóch sposobów w jednym kontrastowym porównaniu. Wariant DINO pozostaje kontrolą reprezentacji. Po pilocie opisanym w dokumencie 38 główny selektor z planu 39 używa cech obszaru narzędzia z RF-DETR; checkpoint, warstwa i dostęp do GT muszą być jawne. Nie wymaga to ukończenia wszystkich analiz DINO.

### 4.2. Tożsamość i widoczność

Każdy odcinek ma `track_id`. W R2 wynika on z anotacji. W R3 przypisanie jest przyczynowe, na podstawie zgodności klasy, geometrii i wyglądu z poprzednimi klatkami; parametry ustala development. Numer query DETR nie jest trwałą tożsamością.

Przy braku lub niejednoznaczności przypisania odcinek geometryczny kończy się. Nie interpolować przez brak widoczności jako przez rzeczywistą obserwację. Osobny kanał zdarzeń zachowuje wejścia, wyjścia, utratę detekcji i zasłonięcia. Warunek kompletnego toru nie może usunąć tych przypadków z oceny zadania.

### 4.3. Co oznacza krok

Podstawą są znaczniki PTS w sekundach. Projektowany krok główny wynosi **0,20 s**, a dopuszczalny błąd doboru obserwacji to **10 procent kroku**; to parametr projektu, nie stała z literatury. Bez interpolacji obrazów. Poza tolerancją odcinek należy przerwać. Zachować rzeczywiste odstępy w danych.

Jeśli cache zawiera tylko `frame_index`, najpierw ustalić FPS i mapowanie do PTS z metadanych źródła. Do tego momentu wynik ma jednostkę „krok próbkowania”, bez deklaracji fizycznego czasu. Historyczny stride 5 nie oznacza automatycznie 0,20 s dla każdego filmu.

Horyzonty: 1, 2 i 4 kroki. Osobno ocenić prognozę z obserwowaną historią oraz swobodny rollout, w którym przewidziany stan zasila kolejne kroki. Nie używać prawdziwych przyszłych ROI w rolloutach. Początkowe badanie H1 dotyczy głównie prognozy jednego kroku.

### 4.4. Koopman i model inspirowany GLE

Dla wybranych obserwabli `g`:

```text
g(z[t+1]) ≈ A g(z[t])

z[t+1] = F(z[t]) + sum_j K[j] z[t-j] + C(e[t], e[t-1], ...) + e[t+1]
```

Pierwsza postać jest skończonym przybliżeniem operatora Koopmana. Druga jest modelem z pamięcią inspirowanym GLE. W tym badaniu nie odzyskujemy automatycznie dokładnego jądra Mori–Zwanziga, fizycznego tarcia ani termicznego szumu. `C` jest opcjonalnym korektorem historycznych reszt, a nie zmierzoną siłą losową.

Ruch ręki chirurga, kontakt z tkanką i zmiana oświetlenia mogą być nieobserwowanymi wymuszeniami. Brak wejść znanych z góry nie oznacza, że film przedstawia autonomiczny układ bez wymuszeń. Bieżące obserwowalne informacje o ruchu kamery i jakości obrazu można dodać jako jawnie opisane wejścia; nie nazywać ich znanymi przyszłymi sterowaniami.

Rozkład błędu dla `m(H)=E[Y|H]` obowiązuje w wartości oczekiwanej:

```text
E[||Y-h(H)||² | H] = E[||Y-m(H)||² | H] + ||m(H)-h(H)||².
```

Dla pojedynczej klatki występuje człon krzyżowy. Ani różnica dwóch zrealizowanych strat, ani rozbieżność predykcji nie jest sama w sobie dowodem usuwalności błędu lub wartości etykiety dla detektora.

## 5. Dane, podziały i kontrakty

### 5.1. Oddzielenie zbiorów

Punktem odniesienia jest kampania [Static36](35_KAMPANIA_STATIC36_EDEN.md): **117/25/25 grup operacji**. Nie zastępować jej starszym podziałem 35/8/7 innego zbioru z `protocol.yaml`. Grupy operacji nie są potwierdzonymi niezależnymi pacjentami; do czasu identyfikacji pacjentów używać tego ostrożniejszego opisu.

- Pula train: uczenie modeli dynamiki, dobór hiperparametrów wewnątrz grup i ranking do selekcji.
- Validation/development: wybór jednej końcowej receptury oraz eksploracyjna ocena związku z zadaniem.
- Test: końcowe porównanie po zamrożeniu receptury, o ile audyt potwierdzi jego dotychczasowy zakres wykorzystania.

**Audyt dostępu jest konieczny:** historyczny `read_tracks()` czyta wszystkie archiwa, a wyniki obejmują 167 grup. Sprawdzić pokrycie tych grup z planowanym testem. Nie nazywać testu „nigdy nieoglądanym”, jeżeli jego tory lub agregaty posłużyły już do analizy. Można kontynuować badanie eksploracyjne z ujawnieniem tego faktu; mocne potwierdzenie wymaga nowego odłożonego materiału, którego nie używano do decyzji.

Pula Static36 jest ograniczona do klatek z dostępną ramką cystotomu. Wniosek o całym filmie, wejściach/wyjściach i FP wymaga osobnego manifestu gęstych sekwencji zawierających również negatywy i stany widoczności. Nie dopisywać takich klatek do jednego selektora i nie porównywać go ze starym baseline na innej puli.

**Osobny podział do B/C:** z 117 grup train wylosować raz 20 grup na uczenie detektora obserwacyjnego (seed 42), a pozostałe 97 przeznaczyć na diagnostykę. Zamrożony detektor nigdy nie widzi etykiet tych 97 grup. Wszystkie R0–R3 porównujemy na tych samych 97 grupach, z CV opisanym poniżej. Dzięki temu transformacje cech w CV nie ukrywają przecieku przez wcześniej nauczony detektor. Ten podział i checkpoint służą wyłącznie niezależnej diagnostyce B/C. W selekcji D z planu 39 dostępne są etykiety wszystkich 117 grup train; wspólne proxy ma oddzielny koszt, a końcowe modele zaczynają od COCO. Wyników B/C nie używać do strojenia na końcowym teście.

### 5.2. Minimalny rekord danych

| Pole | Znaczenie |
|---|---|
| `frame_id`, `video_id`, `group_id`, `split` | Tożsamość źródła i podział. Brak mapowania grupy jest błędem, nie powodem do zastąpienia jej nazwą filmu. |
| `timestamp_s`, `frame_index`, `source_fps` | Czas źródłowy; PTS ma pierwszeństwo nad przybliżeniem z FPS. |
| `track_id`, `segment_id`, `visibility`, `association_valid` | Ciągłość obiektu i odcinka. |
| `representation`, `encoder_hash`, `preprocessing_hash` | Źródło cech i wersja ich wyliczenia. |
| `observation_time_s`, `target_time_s`, `horizon_steps` | Kiedy była dostępna informacja i czego dotyczy wynik. |
| `outer_fold`, `fit_group_ids_hash`, `fit_transform_hash` | Pochodzenie transformacji i dopasowania. |
| `score_mode` | `forecast` albo `retrospective_offline`; rozróżnienie obowiązkowe. |

GT oceny val/test przechowywać poza wejściami selektora. GT train jest jawnym, wersjonowanym wejściem nadzorowanego wariantu z planu 39. Wynik na klatkę zapisuje prognozę, obserwację, resztę i nazwy modelu/targetu; brak widoczności nie jest zerową ramką. Hashować split, cache, checkpoint, konfigurację i listy wybranych klatek.

## 6. Etap A: poprawność metody przed pomiarami

Poniżej są zadania do wykonania, nie poprawki już wdrożone. Linie odnoszą się do stanu kodu odczytanego 2026-09-21; przy implementacji kierować się również nazwą funkcji.

| Zadanie | Istniejące miejsce | Wymagana zmiana i warunek odbioru |
|---|---|---|
| A1. Transformacje w foldach | `pipeline/koopman_langevin/latent_langevin.py:81,87`; `track_dynamics.py:116,130` | Dopasować skalery stanu, przyrostu i ewentualne PCA wyłącznie na danych uczących konkretnego folda. To samo w inner CV. Podmiana outer-test nie zmienia transformacji, alpha ani wag. Stała losowa projekcja niezależna od danych może być wspólna. |
| A2. Zagnieżdżone reszty | `track_dynamics.py:147–162` | W każdym outer-train wytworzyć reszty inner-OOF; tylko na nich uczyć korektor `C`. Obecny globalny OOF może przenieść informację outer-test do treningu korektora. Podmiana targetów outer-test nie zmienia treningowych reszt ani wag korektora. |
| A3. Jawny dostęp do targetu | `latent_langevin.py:106,147` | Rozdzielić score retrospektywny od dostępnego przed targetem. Przyczynowy score używa samej rozbieżności prognoz; wariant normalizowany może użyć wariancji przewidzianej z historii przez kalibrator uczony na inner-OOF. Nie używa błędu bieżącego targetu w mianowniku. |
| A4. Porównywalne modele | `latent_langevin.py:scores` | Przy diagnostycznym porównaniu `h_known` i `h_pool` oba wykluczają ten sam outer-test. Pusty zbiór znanych przejść w outer-train uruchamia jawny fallback persistence. Transdukcyjne wykorzystanie całej puli train raportować osobno. |
| A5. Reszta i język wniosków | `track_dynamics.py:167–178`, docstringi obu modułów | ACF dla lagów 1–10 osobno dla współrzędnych i grup; korelacje krzyżowe, kwadraty reszt i test przewidywania nieliniowego. Usunąć utożsamianie pojedynczego kontrastu strat z błędem nieusuwalnym. |
| A6. Widmo poza próbką | `track_dynamics.py:koopman_spectrum` | Fit na train, defekt funkcji własnej na outer-test. Osobno raportować mody stałe/trivialne i niestabilne. Nie obcinać modułu lambda do wartości poniżej 1 przy obliczaniu czasu zaniku. Jednostką jest rzeczywiste delta-t. |
| A7. Czas i tożsamość | `track_dynamics.py:windows`, `select.py:observation_rows`, `tokens.py` | Walidacja PTS, odcinków, pełnego mapowania grup i masek widoczności. Zmiana query lub brak obiektu nie może tworzyć sztucznego ruchu jednego obiektu. |
| A8. Offline/online | `koopman_tool.py:saliency_maps` | Obecne pełne okno DMD oznaczyć jako retrospektywne. Wariant online ma okno kończące się w chwili decyzji. |

**Testy mechanizmu:** stały stan, liniowy proces Markowa, nieliniowy proces Markowa, proces AR(2) z ukrytą pamięcią, niezależny szum, skok widoczności, zmiana przypisania i drganie anotacji. W nieliniowym procesie Markowa metoda ma odróżnić zbyt słaby model liniowy od rzeczywistej korzyści z historii. Dla szumu nie zakładać dodatniej umiejętności poza przedziałem losowych fluktuacji.

**Testy izolacji:** po zamrożeniu historii, modeli i kalibratora zmiana następnej klatki nie zmienia `forecast_score`; może zmienić `retrospective_score`. Zmiana danych grup testowych nie zmienia parametrów uczonych. Dotyczy to również korektora reszt. Podmiana danych po chwili `t` nie zmienia prognozy wystawionej w `t`.

**Brama A:** przed wynikami empirycznymi muszą przejść testy izolacji, czasu, przypisania obiektu i zgodności etykiet ze splitem. Istniejące testy projektu nie zastępują tych kontraktów.

## 7. Etap B: diagnostyka reprezentacji i roli historii

### 7.1. Macierz modeli

Każdy model przewiduje ten sam target, na tych samych poprawnych oknach. Główne porównanie pamięci to **M4 kontra M2**. Różnica M4–M1 miesza korzyść z pamięci i nieliniowości, więc nie wystarcza.

| Model | Informacja wejściowa | Rola |
|---|---|---|
| M0 | Bieżący stan | Persistence: następny stan równy obecnemu. |
| MV | Dwa ostatnie stany | Stała prędkość; to już model z historią. |
| M1 | Tylko bieżący stan | Afiniczna regresja ridge. |
| M2 | Tylko bieżący stan | Nieliniowa regresja z ustalonymi cechami RFF i ridge. |
| M3 | Bieżący stan i opóźnienia | Liniowy model autoregresyjny, kontrola prostej pamięci. |
| M4 | Te same nieliniowe cechy bieżącego stanu co M2 oraz opóźnienia | Główny model z pamięcią inspirowany GLE. |
| M5 | M4 i przeszłe reszty | Korektor reszt, tylko po A2; etap dodatkowy, jeśli reszty dają sygnał. |

Prędkość i przyspieszenie wymagają wcześniejszych stanów. Model `D(z,v,a)` nie jest kontrolą „jedna klatka”. W M1/M2 nie przemycać tych wielkości ani stanu rekurencyjnego z przeszłości.

Pilotaż: R0 i R1 oraz M0/MV/M1/M2/M3/M4. Następnie porównać R2 i R3, zachowując tę samą listę okien do sparowanych porównań; osobno raportować przypadki odrzucone przez brak ROI. Duża strata pokrycia w R3 jest wynikiem, nie powodem do przemilczenia trudnych klatek.

### 7.2. Dopasowanie

- Outer CV: 5 foldów na 97 grupach diagnostycznych wewnątrz train. Inner CV: 3 foldy po grupach. Wszystkie reprezentacje i modele korzystają z tego samego podziału. Detektor R3 jest zamrożony po uczeniu na rozłącznych 20 grupach, więc jego cechy nie wymagają globalnego banku proxy OOF.
- Ridge alpha: `0.01, 0.1, 1, 10, 100`. Długość historii: `1, 2, 4, 8` wcześniejszych kroków. Selekcja wyłącznie w inner CV.
- Podstawowy wymiar projekcji latentu: 24; kontrola 48 dopiero po podstawowym pomiarze. R0 zachowuje interpretowalne współrzędne.
- RFF: 48 cech, seed 42; standaryzacja i szerokość funkcji ustalane na inner-train. Kontrola elastyczności M2: 192 RFF z tym samym budżetem strojenia co odpowiadający jej model z historią.
- Wybór konfiguracji: średni błąd makro po grupach na wewnętrznej walidacji; remis do 1 procent błędu rozstrzyga mniejszy model/krótsza historia.
- Fit ograniczyć do 40 000 okien na fold, z równym limitem na grupę i deterministycznym losowaniem. Te same okna i wagi w modelach porównawczych; logować niewykorzystane okna.
- Rozdzielić target własnego latentu od wspólnego targetu geometrii. Wynik latentny jest wtórny wobec zachowania informacji zadaniowej.

### 7.3. Miary

```text
skill = 100 * (1 - SSE_model / SSE_persistence)
delta_history = skill_M4 - skill_M2
```

Raportować mikro po wszystkich oknach oraz makro po operacjach. Dla niemal nieruchomej grupy, gdy średnia kwadratowa kroku w jednostkach skali train jest poniżej `1e-8`, skill jest niezdefiniowany: podać MSE/MAE i liczność zamiast dzielić przez prawie zero. Podać także wynik bez 1 procenta największych kroków, wyłącznie jako analizę wrażliwości.

Miary wspólnego zadania: błąd środka ramki, szerokości i wysokości w jednostkach obrazu, IoU prognozowanej ramki, a dla prawdziwego predyktora końcówki błąd końcówki. **Środek ramki nie jest końcówką.** Dodatkowo zachowanie rozróżnialności: sonda geometrii na zamrożonych cechach oraz widoczność. Mała zmienność latentu nie jest sukcesem, jeśli znika informacja zadaniowa.

Kontrole przyczyn: jasność, ostrość, ruch kamery, wielkość narzędzia i dostępność ROI. Powtórzyć analizę na odcinkach o stałej widoczności oraz na zdarzeniach osobno. Permutacja kolejności wewnątrz ciągłego filmu jest kontrolą istnienia zależności czasowej, a nie konkurencyjną metodą prognozowania ani pełnym dowodem pamięci.

**Widmo Koopmana:** raportować błąd `psi(z[t+1]) - lambda*psi(z[t])` na innych grupach, rząd macierzy, regularizację, moduł i fazę lambda. Sprawdzać stabilność podprzestrzeni w resamplingu grup, nie identyczność znaków i kolejności wektorów. Mod bliski 1 może oznaczać stałe tło albo średnią operacji. Związek z tożsamością wymaga niezależnego celu rozpoznawania obiektów i danych wielonarzędziowych.

**Brama B:** przejść do mocnych wniosków o historii, jeżeli M4 przewyższa M2 poza próbką także po kontroli elastyczności M2. Jeżeli nie, zapisać brak wykazanej korzyści i nie rozbudowywać jądra pamięci tylko dlatego, że nazywa się Langevinem. Analiza błędów i wariant H4 nadal mogą być użyteczne.

## 8. Etap C: czy dynamika rozpoznaje błędy detektora?

### 8.1. Predykcje i target błędu

Użyć standardowych wyjść RF-DETR Nano i istniejącego ewaluatora. Żadnego pojedynczego „wyróżnionego query”. Checkpoint, próg operacyjny i reguły matchingu muszą być wspólne. Miar AP nie liczyć po odfiltrowaniu wyników progiem operacyjnym.

Na klatkach z GT: `bad_frame=1`, gdy co najmniej jeden wymagany obiekt nie ma poprawnego dopasowania tej samej klasy z IoU >= 0.5 albo pozostała nieprzypisana detekcja powyżej zamrożonego progu. Rozbić wynik na FN, FP i złe położenie. Jeśli oceniana pula ma wyłącznie obecny cystotom, nie wyciągać wniosku o FP na klatkach bez narzędzia.

Detektor obserwacyjny jest trenowany raz na rozłącznych 20 grupach opisanych w sekcji 5.1, seed 42, receptura Nano z kampanii Static36, checkpoint `last_epoch`. Zamrozić jego liczbę kroków odpowiednio do osobnej puli B/C; nie utożsamiać go z proxy selekcji planu 39. Ten sam zamrożony model generuje predykcje 97 grup diagnostycznych oraz val/test; nie zastępować go przy ocenie modelem nauczonym na wszystkich 117 grupach. DINO pozostaje zamrożony. W zewnętrznych foldach 97 grup model dynamiki i klasyfikator błędu są dopasowywane tylko na outer-train. Do uczenia klasyfikatora użyć reszt modeli dynamiki wygenerowanych inner-OOF w obrębie tego outer-train. Dopiero potem ocenić cały układ na outer-test. Wcześniejsze checkpointy mogą służyć do eksploracji, ale bez potwierdzonej rozłączności ich treningu nie zastępują detektora obserwacyjnego.

### 8.2. Rozdzielenie informacji

- **Prognoza:** `h(H[t-1])` przewiduje stan klatki `t`; nie ma dostępu do jej obrazu.
- **Diagnostyka bieżąca:** po obejrzeniu klatki `t` porównuje prognozę z jej reprezentacją i detekcjami. Jest to legalna obserwacja bieżącego błędu, ale nie prognoza przed nadejściem klatki.
- **Opis offline:** może używać całego okna filmu treningowego; dostęp oznaczyć i nie porównywać bezpośrednio z modelem online.

Kandydatów z dużą rozbieżnością poddać kontroli obrazów: prawdziwy gwałtowny ruch, brak detekcji, odblask, zasłonięcie, zmiana przypisania. Taki podział opisuje przyczynę alarmu, nie dowodzi jej na podstawie samego score.

### 8.3. Kontrasty i ocena

Porównać ten sam klasyfikator błędu (regresja logistyczna z regularyzacją wybieraną na train):

1. Confidence detektora, brak detekcji, rozmiar ramki, ruch, jasność, ostrość i informacja o ciągłości obiektu.
2. Dokładnie te same cechy kontrolne plus reszta M2.
3. Dokładnie te same cechy kontrolne plus reszta M4, w identycznym formacie i wymiarze jak dla M2.

Główne porównanie C3–C2 rozstrzyga wkład historii, a C3–C1 wartość całej diagnostyki. Wszystkie normalizacje i wybór cech powstają w train. Ocenić AUPRC oraz precision i recall przy budżecie alarmów obejmującym 10 procent klatek. Pokazać częstość błędu, ponieważ AUPRC zależy od niej. Dodatkowo wykres kalibracji i wyniki per operacja/zdarzenie.

Rozbieżność między dwoma predyktorami można zbadać jako dodatkowy kontrast, ale nie dodawać jej wyłącznie do C3 i nadal nazywać różnicy czystym efektem historii. Przy braku poprawnej reszty stosować wspólną maskę dostępności i te same cechy kontrolne w C1–C3; raport obejmuje także te klatki.

**Brama C:** interpretacja score jako diagnostyki nowych błędów wymaga, by C3 poprawiało odpowiedni baseline na innych grupach. Sama korelacja reszty z prędkością nie spełnia tego warunku. Wartość selekcji jest osobnym pytaniem: wolno ją zbadać w etapie D również przy ujemnym wyniku C, ale nie uzasadniać jej wtedy potwierdzoną diagnostyką błędów.

## 9. Etap D: nadzorowany dobór już oznaczonych klatek

**Pełny aktualny protokół:** [plan 39](39_PLAN_SELEKCJI_KLATEK_RFDETR_I_ROZWOJU_JEPA.md), sekcje 3–8. Poprzedni projekt L0 + nowe anotacje i ramiona S0–S5 został zastąpiony po doprecyzowaniu celu przez autora.

- Dostępne są etykiety całego train; główny baseline to loss detektora + różnorodność, nie wyłącznie niepewność.
- Wspólne proxy i cechy ROI, w tym GT-ROI, służą tylko przygotowaniu selekcji. Końcowe modele uczą się od tego samego COCO.
- Pilot: 1500 unikalnych źródeł, U/H/V/K/KM/AR, 3 seedy treningu, razem 18 końcowych treningów oraz koszt wspólnego proxy.
- Główne porównanie KM–H bada użyteczność selekcji. KM–K bada historię, a KM–AR i kontrola V ograniczają przypisywanie zysku formalizmowi zamiast zwykłemu ruchowi.
- RF-DETR podczas inferencji pozostaje standardowym modelem jednoklatkowym. Nie dołączamy pamięci ani filtra.
- Czas przygotowania, proxy i odczytu pełnej puli jest raportowany; budżet wybranych klatek nie jest budżetem wszystkich poznanych etykiet.
- Pozytywne B/C wspierają interpretację, lecz same nie dowodzą wartości selekcji. D można badać po krytycznych poprawkach A bez ukończenia pełnej analizy widma i sond.

Kryteria ilościowe, fallbacki, reguły dopasowania, zakres użycia etykiet i liczby rozszerzeń utrzymujemy w jednym miejscu — w planie 39. Nie używać historycznej średniej z dokumentu 36 jako progu zastępującego nowe sparowane ramię kontrolne.

## 10. Etap E: obrazy statyczne i kontrolowane przekształcenia

To osobny eksperyment H4, możliwy bez treningu nowych detektorów. Zamiast naturalnego czasu używa parametru transformacji. Nie mieszać jego wniosków o odporności z wnioskami o pamięci wideo.

1. Z train wybrać po 10 obrazów z każdej z 30 grup, równomiernie po czasie, seed 42. Podział źródeł: 20 grup fit, 5 development, 5 ocena; wszystkie przekształcenia jednego źródła zostają razem. To kontrola wewnątrz train, bez dostępu do końcowego testu.
2. Dla każdego obrazu wykonać oddzielnie przesunięcia `-8, -4, 0, 4, 8` procent szerokości/wysokości, obroty `-10, -5, 0, 5, 10` stopni, skale `0.9, 1.0, 1.1` oraz mnożniki jasności `0.8, 1.0, 1.2`. Zapisać maski obszarów poza obrazem. Nie stosować wszystkich zmian naraz w podstawowym kontrastowym pomiarze.
3. Wyliczyć zamrożone reprezentacje i odpowiednio przekształcić GT. Dla map tokenów porównać obszary po wyrównaniu geometrii; dla ROI zachować oddzielną informację o globalnym położeniu.
4. Dopasować na fit prostą mapę `z(Tx) ≈ A_T z(x)+b_T` i porównać z identycznością oraz nieliniową mapą o kontrolowanej złożoności. Ocenić na innych źródłach. Test złożenia map afinicznych używa `A_T2 A_T1 z + A_T2 b_T1 + b_T2`, a nie samego iloczynu macierzy. Porównanie z bezpośrednim przekształceniem przeprowadzić tylko dla kombinacji pozostających w zakresie poprawnej geometrii.
5. Ocenić błąd reprezentacji oraz zgodność ramek po odwrotnym przekształceniu. Dla geometrycznych zmian oczekiwać ekwiwariancji lokalizacji. Dla oświetlenia sprawdzać stabilność zadania, nie wymuszać zerowej zmiany całego latentu.

Jeżeli pojawi się zasłonięcie lub obcięcie obiektu przez granicę obrazu, oznaczyć je jako utratę obserwacji, a nie jako zwykły test zachowania tożsamości. Kolejność dowolnych niezależnych fotografii, indeks patcha, warstwy ani epoki nie jest w tym badaniu zegarem fizycznym.

## 11. Statystyka i kryteria rozstrzygnięcia

Jednostką resamplingu jest cała grupa operacji, a po potwierdzeniu mapowania — cały pacjent. Bootstrap klatek zawyżałby efektywną liczbę obserwacji. Użyć **2000 sparowanych replik**, seed `20260921`, przedziały 95 procent. Dla AP losować grupy i ponownie liczyć AP z ich predykcji; nie zastępować tego średnią AP klatek.

Raportować osobno zmienność między grupami, seedami treningu i seedami selekcji. Przedział po grupach przy stałym seedzie nie obejmuje całej niepewności uczenia. W potwierdzeniu zastosować sparowany bootstrap hierarchiczny po seedach selekcji, seedach treningu i grupach, zachowując odpowiadające sobie wyniki metod. Przy trzech seedach ujawnić ograniczoną precyzję oszacowania tej składowej.

Poniższe minimalne efekty są **proponowanymi progami praktycznymi** do zamrożenia przed uruchomieniem, nie wynikami ani prawami z literatury:

| Cel | Kontrast główny | Próg praktyczny |
|---|---|---|
| H1, historia | Makro skill M4–M2 na wspólnym celu geometrii | +1 punkt procentowy |
| H2, diagnoza błędów | AUPRC C3–C2 | +0.02 bezwzględnie |
| H3, selekcja | KM–H; następnie atrybucja KM–K i KM–AR według planu 39 | +0.004 bezwzględnie, czyli +0.4 punktu AP; szczegóły w planie 39 |

Efekt dodatni: punktowe oszacowanie przekracza próg praktyczny, a przedział różnicy jest powyżej zera. Próg 0.004 nawiązuje do dokumentu 36; warunek „każdy seed ponad średnią baseline” nie zastępuje przedziału dla sparowanych różnic. Dla H3 obowiązuje uporządkowanie kontrastów z sekcji 7 planu 39: główny KM–H i osobne porównania atrybucji. Nie przenosić starej macierzy S0–S5 do nowego badania.

Wynik nierozstrzygający: przedział obejmuje zero i efekty większe od progu praktycznego. Brak istotności nie jest dowodem równoważności. Praktyczną równoważność można stwierdzić dopiero, gdy cały odpowiedni przedział mieści się w paśmie `[-próg, +próg]`. H4 jest w pierwszej rundzie analizą opisową bez deklarowanego dodatniego efektu.

## 12. Jakie wnioski zapisać po eksperymencie

| Zaobserwowany wynik | Dozwolony wniosek | Czego nie twierdzić |
|---|---|---|
| M4 wygrywa z M2 także przy kontroli pojemności | Historia dostarcza użytecznej informacji dla badanej reprezentacji i horyzontu. | „Udowodniono fizyczne równanie Langevina”. |
| M2 dogania M4 po zwiększeniu pojemności | Wcześniejszy zysk mógł wynikać z niedostatecznej nieliniowości modelu bieżącego stanu. | „W danych nie ma żadnej pamięci”. |
| Reszta nadal przewidywalna poza próbką | Pozostała niewykorzystana informacja w badanej klasie obserwacji. | „Cała pozostała reszta to szum”. |
| Nie wykryto przewidywalności reszty | W badanych modelach, danych i testach nie znaleziono dalszego zysku. | „Wyznaczono nieusuwalny błąd Bayesowski”. |
| Wolne mody stabilne na nowych grupach | Istnieją trwałe wielkości w tej reprezentacji i skali czasu. | „Znaleziono semantyczną tożsamość narzędzia” bez niezależnego testu. |
| R2 pomaga, R3 nie | Informacja obiektu jest użyteczna, ale lokalizacja/przypisanie ogranicza wariant użytkowy. | „Latent obrazu nie ma dynamiki”. |
| Reszta wskazuje szybki ruch, ale nie błąd detektora | Score mierzy przede wszystkim zmianę sceny. | „Warto kupować/odrzucać te etykiety” bez eksperymentu D. |
| H1/H2 dodatnie, H3 bez praktycznej poprawy | Metoda jest użyteczną diagnostyką, lecz nie wykazano wartości selekcji w tym budżecie. | „Eksperyment się nie udał” tylko dlatego, że AP nie wzrosło. |
| KM wygrywa z H, K i AR według planu 39 | Wsparcie dodatkowej wartości selekcji i historii przy mocnych kontrolach zadaniowych. | Uogólnienie na inne operacje, architektury i budżety bez transferu. |
| H4 pokazuje stabilną semantykę i poprawną geometrię | Reprezentacja zachowuje badane własności przy kontrolowanych zmianach. | „Odporność na wszystkie rzeczywiste warunki chirurgiczne”. |

## 13. Mapa implementacji i kolejność wykonania

Wszystkie ścieżki poniżej są względem `SmallViTAblation/`. Pliki oznaczone jako nowe są planowanymi artefaktami, a nie już dostępnymi poleceniami.

| Kolejność | Pliki / wynik | Co wykonać |
|---|---|---|
| 1 | Nowe `experiments/latent_cv_diagnostics/README.md`, `protocol.lock.json`, `audit_data.json` | Zapisać specyfikację tego dokumentu, hashe, mapowanie grup, zakres wcześniejszego dostępu do testu i jednostkę czasu. |
| 2 | Modyfikacja `pipeline/koopman_langevin/latent_langevin.py`, `track_dynamics.py`, `tokens.py`, `select.py` | Wdrożyć A1–A8, wykorzystując istniejące pomocnicze funkcje `pipeline/latent_dynamics/gle.py`. Nie przenosić opisowych modów reszt do walidacji jako bazy dopasowanej na teście. |
| 3 | Nowe `pipeline/koopman_langevin/tests/test_cv_protocol.py` | Testy perturbacji outer-test, przyszłego targetu, stackingu, nieregularnego czasu, braku obiektu i rozdzielenia forecast/offline; kontrakty opisane w etapie A. |
| 4 | Nowe `experiments/latent_cv_diagnostics/build_observations.py` | Jeden manifest cech, geometrii, czasu, masek i grup dla R0–R3; osobny plik GT do oceny. |
| 5 | Nowe `experiments/latent_cv_diagnostics/run_diagnostics.py` | Wspólne outer/inner foldy, macierz M0–M5, predykcje OOF, porównanie targetów i raport B. |
| 6 | Nowe `experiments/latent_cv_diagnostics/analyse_detector_errors.py` | Dopasowanie identycznych modeli C1–C3 oraz raport PR, alarmów i przyczyn błędu. |
| 7 | `pipeline/active_learning/acquisition.py`, `pipeline/koopman_langevin/select.py`; nowe `experiments/latent_cv_diagnostics/build_selection.py` | Wspólne proxy, train GT, strata per obraz, cechy ROI, kwoty i ramiona U/H/V/K/KM/AR według planu 39; manifesty zgodne z trenerem. Budżet dotyczy treningu końcowego, koszt proxy osobno. |
| 8 | `pipeline/train/run_with_validation.py`, `tools/build_active_manifest.py`, `pipeline/eval/group_bootstrap.py` | Generacja ograniczonej macierzy, weryfikacja dry-run, trening i sparowana ocena. Rozszerzyć bootstrap o powtórzenia selekcji/treningu, jeśli obecna wersja obsługuje tylko grupy. |
| 9 | Nowe `experiments/latent_cv_diagnostics/run_equivariance.py` | Niezależny wariant obrazowy E, z grupowaniem przekształceń według źródła. |
| 10 | Nowe `results/latent_cv_diagnostics/report.md` i artefakty poniżej | Raport obejmujący dodatnie, negatywne i nierozstrzygające kontrasty oraz odchylenia od planu. |

Po implementacji testów z punktu 3, uruchomić je z korzenia repozytorium:

```powershell
py -3.11 -m pytest SmallViTAblation/pipeline/koopman_langevin/tests/test_cv_protocol.py -q
py -3.11 -m pytest SmallViTAblation/pipeline/koopman_langevin/tests SmallViTAblation/pipeline/tests/test_active_curation.py SmallViTAblation/pipeline/eval/tests/test_evaluation_contract.py -q
```

Pierwsze polecenie wskazuje planowany nowy test. Nie jest poleceniem dostępnym przed implementacją. Nowe runner'y powinny przyjmować wyłącznie zamrożony manifest eksperymentu i katalog wyników, a nie niejawnie wybierać najnowszy checkpoint lub dowolny cache.

## 14. Koszt, artefakty i warunek zakończenia

**Najpierw CPU i cache:** audyt danych, testy syntetyczne, izolacja foldów, geometria R0 i istniejące cechy R1. Ekstrakcję ROI mierzyć na 8 grupach train; raportować czas/klatkę, pamięć i rozmiar cache, zanim uruchomi się pełną pulę. Nie wpisywać oszacowania jako pomiaru.

**Koszt GPU zależy od etapu:** niezależny detektor obserwacyjny na 20 grupach jest opcją dla pełnej diagnostyki B/C, nie obowiązkowym poprzednikiem selekcji. Pilot z planu 39 obejmuje 18 końcowych modeli i jedno wspólne proxy, o ile istniejący checkpoint nie spełnia ustalonej receptury. Koszt kolejnych pakietów i zasady ponownego użycia biegów podano w planie 39. Rzeczywiste czasy starej kampanii są dostępne w rejestrze wyników; nie są pomiarem kosztu nowego pipeline.

Historyczny zapis o wolnym miejscu na Edenie nie jest aktualnym pomiarem. Przed nowymi obliczeniami sprawdzić dostępne zasoby, rozmiary checkpointów i sposób archiwizacji. Plan nie zleca usuwania dotychczasowych wyników.

Wymagane wyniki w `results/latent_cv_diagnostics/`:

- `protocol.lock.json`, `data_audit.json`, `environment.json`: hashe, wersje, dane i odchylenia od planu;
- `predictions_oof.parquet`, `scores.parquet`: modele, targety, czasy, splity i wyniki na klatkę;
- `diagnostics.csv`, `detector_errors.csv`, `selection_results.csv`, `equivariance.csv`: kompletne kontrasty z licznościami;
- `figures/`: prognoza i reszta na osi czasu, efekt historii per operacja, stabilność modów, PR błędów detektora, AP względem budżetu, kontrolowane przekształcenia;
- `failure_cases/`: przykłady wysokich i niskich score, dobierane według zapisanej reguły z obu grup wyników, z GT tylko w warstwie oceny;
- `report.md`: odpowiedzi H1–H4 w formacie „pomiar, niepewność, dozwolony wniosek, ograniczenie”, także przy wyniku zerowym.

**Pierwszy kamień milowy diagnostyki:** poprawny protokół A i porównanie modeli na wspólnych oknach. **Priorytet autora:** pierwszy kamień milowy selekcji określa plan 39 — poprawne manifesty i kontrolowany pilot wpływu doboru klatek na RF-DETR. Poprawa predykcji latentu nie jest ani wystarczającym, ani koniecznym dowodem poprawy selekcji.
