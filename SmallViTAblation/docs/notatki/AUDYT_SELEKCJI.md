# Audyt selekcji danych — SmallViTAblation i AdvancedDatasetSelection

> Archiwum audytu sprzed refaktoru kodu. Aktualne poprawki i testy: [21_STAN_REFAKTORU.md](../21_STAN_REFAKTORU.md). Numery linii i wyniki reprodukcji odnoszą się do zapisanej wersji źródeł.

Data: 2026-09-11. Docelowa architektura użytkownika: RT-DETR. Obecny kod proxy: RF-DETR-Small.

**Wniosek:** nowy pipeline nie jest wiernym przeniesieniem oryginału. Ma potwierdzone błędy blokujące ekstrakcję DINO oraz błędy zmieniające ważenie cech, ocenę negatywów, k-center, obsługę cache i materializację. Wyników nie należy obecnie interpretować jako izolowanego porównania metod selekcji lub zmiany RF-DETR na RT-DETR.

**Korekta identyfikacji produkcyjnego podzbioru:** przedmiotem statystyk 71,3% augmentacji jest `AdvancedDatasetSelection/output/selected_dataset1/merged_20k_annotations_fixed.json`. Ten plik zawiera 20 000 obrazów, 15 565 źródeł i 14 259 augmentowanych obrazów, czyli 71,295%. Nie wolno podstawiać w to miejsce raportu `selected_dataset_weighted_features`, który opisuje inny zestaw. Skrypt dowodowy potwierdził równość multizbiorów nazw plików pomiędzy produkcyjnym COCO i `selected_dataset1/selection_reasons.json`.

## 1. Zakres, ograniczenia i pochodzenie dowodów

- Odczytano obowiązujący repozytoryjny `AGENTS.md`; nie znaleziono podrzędnych instrukcji AGENTS w audytowanych modułach ani instrukcji dla katalogu artefaktów w sprawdzonym łańcuchu rodziców.
- Zakres: `SmallViTAblation/pipeline/features`, `SmallViTAblation/pipeline/selection`, oryginalne `main_selection_pipeline.py`, `config.yaml`, `selection_methods`, `feature_extractors`; pomocniczo kontrakt manifestu i handler COCO, aby prześledzić eksport.
- Nie badano `pipeline/eval` ani `pipeline/train`. `features/train_el2n_proxy.py` analizowano wyłącznie jako część przygotowania cechy selekcyjnej. Nie szukano literatury, nie uruchamiano treningów, nie wykonywano inferencji modeli, nie pobierano wag ani nie odczytywano checkpointów modeli.
- Repozytorium pozostawało tylko do odczytu. Utworzono wyłącznie ten raport oraz dwa nowe artefakty w `evidence/`. Skrypt uruchomiono z `-B`; definicje funkcji i klas wyodrębnia przez AST, bez uruchamiania importów i kodu inicjalizacyjnego audytowanych modułów.
- **S** oznacza potwierdzenie statyczne; **T** — reprodukcję na syntetycznych danych lub atrapach; **D** — pomiar istniejącego artefaktu; **H** — hipotezę wymagającą rzeczywistych danych albo konkretnej wersji biblioteki; **U** — dane wyliczone przez głównego agenta z lokalnych manifestów i przekazane do tej części audytu, potwierdzone odczytem wspólnego artefaktu bez ponownego odtwarzania jego procedury.
- P1: blokada wykonania albo istotne naruszenie metodologii/integralności danych. P2: błąd warunkowy, ograniczenie lub zmiana metody wymagająca jawnego rozstrzygnięcia. Priorytet nie oznacza, że wykazano wystąpienie błędu w historycznym treningu.

Dowody wykonawcze: [reproduce_selection_bugs.py](../pochodne/evidence/reproduce_selection_bugs.py), [selection_repro_results.json](../pochodne/evidence/selection_repro_results.json). Uruchomienie zakończyło się kodem 0: **13 sprawdzeń, 0 nieoczekiwanych błędów**; 12 sprawdzeń odtwarza problemy lub rozbieżności, jedno potwierdza działający warunek train-only. Hashe odczytanych źródeł nie zmieniły się podczas wykonania. Nie oznacza to, że błędy zostały naprawione.

## 2. Produkcyjny selected_dataset1 — statystyki i zgodność z tekstem

Źródło podstawowe: [merged_20k_annotations_fixed.json](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/output/selected_dataset1/merged_20k_annotations_fixed.json).

| Pomiar | Wynik | Pochodzenie |
|---|---:|---|
| Obrazy | 20 000 | D, ponownie policzone |
| Anotacje | 20 000 | D, ponownie policzone |
| Źródła po usunięciu końcowego `_aug_N` z nazwy bez rozszerzenia | 15 565 | D, ponownie policzone |
| Obrazy z końcowym `_aug_N` | 14 259 | D, ponownie policzone |
| Udział augmentacji | 71,295%, po zaokrągleniu 71,3% | D; zgodność z wartością w tex odczytaną przez głównego agenta |
| Dodatkowe obrazy ponad jeden na źródło | 4 435 | 20 000 − 15 565 |
| Membership bieżącego video split: train | 15 648 | U |
| Membership bieżącego video split: val | 2 322 | U |
| Membership bieżącego video split: test | 1 982 | U |
| Membership unknown / zewnętrzne obrazy | 48 | U |
| Nazwy z listy wycieków obecne w COCO | 48 | U + D: sprawdzono przecięcie nazw z listą zawartą we wspólnym artefakcie |

Membership i lista nazw pochodzą z pomiaru głównego agenta zapisanego w [production_pool_audit.json](../pochodne/evidence/production_pool_audit.json), wykonanego skryptem [collect_evidence.py](../pochodne/evidence/collect_evidence.py). Skrypt tej części audytu sprawdził wskazanie na właściwy COCO, sumę membership 20 000 oraz obecność wszystkich 48 przekazanych nazw. Nie odtwarzał mapowania do splitu ani nie czytał kodu ewaluacji. Zatem 2 322 + 1 982 = **4 304 obrazy należą według tego mapowania do bieżących val/test**. Jest to niezgodne z zasadą użycia tego podzbioru jako train-only względem bieżącego splitu. Nie stanowi samodzielnego dowodu, jaki split miał historyczny trening konkretnego checkpointu. Historyczne określenie `User-provided counts` w pliku JSON tego podaudytu oznacza dane przekazane w jego zleceniu przez głównego agenta, a nie niezweryfikowaną deklarację użytkownika.

Nie ma sprzeczności między 71,3% augmentacji a 4 435 dodatkowymi kopiami źródeł. Pierwsza wartość liczy nazwy wariantów augmentowanych; druga — nadmiar obrazów ponad jedno wystąpienie każdego źródła. Wariant augmentowany może być jedynym reprezentantem swojego źródła.

### Osobne raporty selection_reasons — nie mieszać podzbiorów

| Pomiar | selected_dataset1 | selected_dataset_weighted_features |
|---|---:|---:|
| Pula raportowana | 90 389 | 90 141 |
| Wybrane obrazy | 20 000 | 20 000 |
| Unikalne nazwy plików | 20 000 | 20 000 |
| Unikalne źródła według sufiksu nazwy | 15 565 | 15 310 |
| Źródła z więcej niż jednym reprezentantem | 4 006 | 4 227 |
| Obrazy augmentowane | 14 259 | 15 070 |
| Klastry jednoelementowe | 2 662 | 2 577 |
| Ścieżki zawierające katalog `valid` | 48 | 0 |
| Rekordy z zapisanymi szczegółami `detr_q81` | 0 | 0 |

Źródła: [selected_dataset1/selection_reasons.json:2](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/output/selected_dataset1/selection_reasons.json:2), [selected_dataset_weighted_features/selection_reasons.json:2](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/output/selected_dataset_weighted_features/selection_reasons.json:2).

W starszym raporcie 19 952 ścieżki pochodzą z katalogu `DETR_augmented_dataset_20250218/images`, a 48 z zewnętrznego `cataract_surgery_Instruments_detection.v1i.coco/valid`. W ważonym raporcie wszystkie 20 000 ścieżek pochodzą z pierwszego katalogu. Brak katalogu `valid` w ważonym raporcie **nie dowodzi train-only względem obecnego video splitu**. Nie wyznaczano dla niego membership. Nazw typu `test01` nie traktowano jako samodzielnej informacji o splicie.

Powtórzenia źródeł są pomiarem nazw, nie testem duplikatów pikselowych. Brak szczegółów Q81 uniemożliwia odtworzenie z tych raportów confidence i matching. Odczyt cache EL2N nie przywraca `detection_info`, choć generator raportu próbuje je wykorzystać (`cluster_selector.py:260–269, 924–987`).

## 3. Trzy różne pipeline’y

| Element | Oryginał `cluster` — domyślny | Oryginał `kcenter` — starszy | SmallViTAblation |
|---|---|---|---|
| Pula | Rekurencyjnie znalezione obrazy / cache | Jak obok | Manifest `split == train` |
| Fourier | 9 cech w klasteryzacji | Prefiltr podobieństwa | 1 cecha w klasteryzacji |
| SAM | 1 score FastSAM lub proxy | 1 score proxy | 9 cech FastSAM |
| DINO | PCA, potem z-score | Normalizacja L2 wierszy | Z-score, potem PCA |
| Wagi | `sqrt(w/d)` | Osobne skalowanie DINO/SAM | `w` na każdy wymiar |
| EL2N | Cecha Query 81 confidence/entropy | Końcowy ranking; Query 81 lub proxy cech | Cecha oparta na confidence po progu IoU |
| Dobór | Klaster → centroid/max_el2n/medoid | Prefiltr → k-center do większego budżetu → najtrudniejsze | KMeans lub k-center z ponownym wyborem reprezentantów |

Oryginał dobiera backend klasteryzacji automatycznie: KMeans, MiniBatchKMeans lub FAISS; SmallViTAblation używa KMeans z `n_init=10`. Identyczne ziarno nie czyni tych implementacji równoważnymi. Oryginalny `cluster` nie ma prefiltra Fourier, więc jego brak w nowym klastrowym pipeline nie jest sam w sobie regresją. Źródła: [main_selection_pipeline.py:323](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/main_selection_pipeline.py:323), [cluster_selector.py:583](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/cluster_selector.py:583), [combined_selector.py:386](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/combined_selector.py:386).

## 4. Ustalenia szczegółowe

### F01 — P1 — Trzy przeszkody produkcyjnej ekstrakcji DINO [S,T]

1. `dino_model`, `dino_image_size`, `fastsam_weights` i sąsiednie parametry są zagnieżdżone pod `analysis`, podczas gdy ekstraktor wymaga ich pod `features`. Test aktualnego YAML daje `KeyError: 'dino_model'`. Budowa wszystkich ekstraktorów poprzedza obsługę wybranych grup, więc dotyczy to również produkcyjnego żądania innej grupy.
2. Po poprawieniu samego zagnieżdżenia produkcyjny `DinoExtractor` nadal nie ma ani `dim`, ani `DIM`, wymaganego w głównej pętli. Test daje `AttributeError`.
3. Usuwanie cyfr z prefiksu `dinov3` daje `dinov`, którego nie ma w słowniku repozytoriów; kod wybiera domyślne repo DINOv2. Test wykonał oryginalne instrukcje przypisania wyodrębnione przez AST, bez `torch.hub`.

Dowody: `production_config_wrong_nesting`, `production_dino_missing_dimension`, `dinov3_routed_to_dinov2`. Lokalizacje: [protocol.yaml:187](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/configs/protocol.yaml:187), [build_cache.py:269](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/build_cache.py:269), [build_cache.py:336](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/build_cache.py:336), [build_cache.py:115](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/build_cache.py:115).

### F02 — P1 — Wagi i PCA zmieniają udziały grup [S,T]

Nowy kod skaluje każdy wymiar przez `w`, więc wkład grupy do kwadratu odległości zależy od `w² × sumy wariancji`. Oryginał wykonuje PCA na centrowanym DINO, standaryzuje komponenty i skaluje grupę przez `sqrt(w/d)`. Nowy kod standaryzuje pierwotne współrzędne przed PCA i nie standaryzuje komponentów po PCA. Zmienia zarówno kierunki główne, jak względną skalę komponentów.

W syntetycznym przykładzie o jednostkowych wariancjach i wymiarach D/E/S/F = 32/1/9/1 nowy kod daje udziały **89,243 / 2,049 / 8,196 / 0,512 procent**, zamiast 35/30/20/15. Test uruchomił rzeczywiste `select_one()` z atrapą odczytu cache i przechwyceniem macierzy przekazanej do klasteryzacji. Oddzielny test oryginalnego `combine_features()` potwierdził sumy wariancji bloków 0,35/0,30/0,20/0,15. Nie jest to pomiar rzeczywistego cache produkcyjnego. Po PCA rzeczywiste wkłady zależą od widma wariancji danych.

Dowody: `group_weighting_and_fake_cache_acceptance`, `pca_order_and_unstandardized_components`. Lokalizacje: [select_subset.py:64](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/selection/select_subset.py:64), [select_subset.py:226](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/selection/select_subset.py:226), [cluster_selector.py:324](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/cluster_selector.py:324).

### F03 — P1 — Nowy score błędnie traktuje negatywy i pomija FP [S,T]

Wynik inicjalizowany jest na 1.0; bez GT pozostaje taki niezależnie od predykcji. Na pozytywie wybiera się najwyższą confidence spośród detekcji o IoU co najmniej 0,5. Klasa nie jest sprawdzana. Dodatkowe lub niedopasowane detekcje nie wpływają na wynik. Bbox jest traktowany jako pojedynczy obiekt; nie ma query matching jeden-do-jednego.

Test z atrapą detektora: poprawny pusty negatyw → **1.0**; negatyw z false positive → **1.0**; pozytyw z przestrzennie dopasowaną predykcją klasy 99 i confidence 0,9 → **0.1**. Test potwierdza logikę score, a nie interfejs prawdziwego RF-DETR. W modelu faktycznie jednoklasowym brak kontroli klasy może się nie ujawnić, ale brak obsługi negatywów i FP pozostaje.

Matematyka wymaga rozróżnienia: dla wektora binarnego `[p, 1-p]` i dodatniego celu `[1,0]`, norma L2 wynosi `sqrt(2) * (1-p)`. Przy celu negatywnym wynosi `sqrt(2) * p`. Brak stałego `sqrt(2)` dla pozytywów sam nie zmienia późniejszej selekcji po normalizacji. Komentarz `sqrt((1-p)^2 + p^2)` w wierszu 89 nie opisuje tej normy poprawnie. Istotne błędy to definicja celu, negatywy, progi i pominięte predykcje. Min-max przed kolejnym z-score jest dla niestałej cechy algebraicznie zbędny; test potwierdza tę równoważność.

Dowód: `el2n_negatives_false_positives_class_matching`. Lokalizacja: [el2n_from_proxy.py:61](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/el2n_from_proxy.py:61), zwłaszcza wiersze 72–91.

### F04 — P1 — Oryginalny aktywny EL2N jest heurystyką Query 81 [S]

Scorer używa wyłącznie Query 81 i wylicza `0.7 * difficulty + 0.3 * normalized_entropy`. Difficulty to `1-confidence` po przekroczeniu progu, inaczej 1.0. Nie otrzymuje GT, nie sprawdza klasy docelowej obrazu ani błędu lokalizacji względem anotacji. Pewny false positive na negatywie może być uznany za łatwy obraz. Tego wyniku nie należy nazywać bez zastrzeżeń normą błędu względem etykiety.

Konfiguracja wskazuje checkpoint epoki 170; domyślny pipeline nie tworzy tutaj wczesnego proxy. Czy użyty checkpoint miał dostęp do danych obecnego val/test — nie ustalono. YAML wskazuje confidence 0,5, lecz konstruktor otrzymuje tylko checkpoint i device: działa domyślne **0,3**. Poprawna klasyfikacyjna norma istnieje w odrębnym `EL2NScorer.compute_el2n_scores()`, lecz nie jest aktywną ścieżką domyślnego selektora. Pomocnicze `compute_el2n_for_detection()` pobiera target, ale go nie wykorzystuje w obliczeniu score mimo zapowiedzi użycia IoU.

Lokalizacje: [detr_el2n_scorer.py:144](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/detr_el2n_scorer.py:144), wiersze 360–395 i 470–500; [cluster_selector.py:85](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/cluster_selector.py:85); [config.yaml:16](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/config.yaml:16); [el2n_scorer.py:239](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/el2n_scorer.py:239).

### F05 — P1 — Oryginał nie egzekwuje train-only; cache może zastąpić pulę [S,T,D]

`os.walk()` zbiera wszystkie obrazy z poddrzew. Przy odczycie cache Fourier selektor przyjmuje `valid_paths` bez porównania z żądanym `image_paths`. Test żądał `requested/train/image.jpg`, a selektor użył `cached/test/image.jpg`. DINO, SAM i EL2N zapisane są jako tablice bez identyfikatorów wierszy. Fourier zbiera wyniki przez `as_completed()`, zatem ponowne liczenie może zmienić kolejność; połączenie nowego Fourier z dawnym cache innych grup może przypisać cechy niewłaściwym obrazom.

To potwierdzony mechanizm umożliwiający naruszenie puli. W produkcyjnym selected_dataset1 osobno potwierdzono obecność 48 zewnętrznych nazw oraz membership bieżącego splitu opisany w sekcji 2; nie ustalono, czy przyczyną tego konkretnego składu był cache, czy wejściowa pula.

Dowód: `original_cache_replaces_requested_pool`. Lokalizacje: [main_selection_pipeline.py:272](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/main_selection_pipeline.py:272), [cluster_selector.py:203](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/cluster_selector.py:203), [fourier_analyzer.py:164](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/feature_extractors/fourier_analyzer.py:164).

### F06 — P1 — Nowy cache sprawdza obecność ID, ale nie pochodzenie [S,T]

`gather_rows()` mapuje po `frame_id` i odrzuca brakujące ID, co poprawia oryginał. Nie sprawdza jednak unikalności ID, fingerprintu danych, wag, konfiguracji ani znaczenia cech. Duplikat ID przyjmuje ostatni wiersz: test dał `[99,99]`. `select_one()` ignoruje `_meta`; test potwierdził akceptację metadanych `fake=True`.

Partial checkpoint sprawdza wyłącznie wymiar. Możliwe jest np. wznowienie dziewięciowymiarowego SAM fake jako prawdziwego SAM, mieszanie konfiguracji lub cech obrazów o zmienionej treści pod tym samym ID. To potwierdzony brak kontroli, ale wystąpienia takiego mieszania w rzeczywistym nowym cache nie wykazano. Oryginalny checkpoint postępu DETR także nie zawiera fingerprintu checkpointu modelu i konfiguracji.

Metadane selekcji dodatkowo zapisują żądany `pca_dim`, choć `pca_dino()` może go zmniejszyć do rozmiaru danych; faktyczny wymiar z `used_weights` nie trafia do wyniku. Dla k-center raportowany representative może różnić się od wymuszonego `centroid`. Końcowy cache cech nie jest wykorzystywany jako ukończony checkpoint przy ponownym `build_cache`: wznawianie dotyczy pliku partial.

Dowody: `duplicate_cache_ids_last_wins`, `group_weighting_and_fake_cache_acceptance`. Lokalizacje: [common.py:91](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/common.py:91), wiersze 126–134 i 165–177; [select_subset.py:226](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/selection/select_subset.py:226), wiersze 247 i 260–268; [detr_el2n_scorer.py:205](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/detr_el2n_scorer.py:205).

### F07 — P1 — Nowy k-center zastępuje wybrane centra [S,T]

Po wyborze centrów zachłannych funkcja zwraca etykiety klastrów. Kolejny etap wybiera punkt najbliższy średniej klastra, zamiast zachować centrum wybrane przez farthest-point. Na punktach `[0,10,17,27,32,38,39,40,56]` centra `[27,56,0]` zamieniono na `[32,56,0]`, a promień pokrycia wzrósł **z 13 do 15**. Nie jest to ten sam algorytm k-center. Przy identycznych cechach `argmax(dist)` może wielokrotnie wskazywać ten sam indeks; późniejsze uzupełnienie budżetu odbywa się po rosnącym `frame_id`.

Przy dużym budżecie istnieje również ryzyko pamięci: tensor różnic w gałęzi chunk ma rozmiar `4096 × budget × D`. Przy budżecie 20 000 i 43 cechach sam taki tensor float32 to około **13,12 GiB**, bez buforów pośrednich normy. Nie uruchamiano tej alokacji.

Dowód: `kcenter_centers_replaced_and_radius_worsened`. Lokalizacje: [select_subset.py:113](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/selection/select_subset.py:113), wiersze 130–138 i 241–248.

### F08 — P2 — Zmienione grupy Fourier/SAM i niepoprawna interpretacja coverage [S,T]

Fourier: 9 wymiarów zastąpiono pojedynczym high-band ratio. SAM: jeden score zastąpiono dziewięcioma cechami. Zmieniono też obliczanie krawędzi i ograniczanie składników score. Są to zmiany badanego sygnału, niezależne od zmiany wag.

W obu wersjach coverage masek to `min(sum(area), area_image) / area_image`, a nie pole sumy logicznej masek. Dwie identyczne maski połowy obrazu dają **1,0 zamiast 0,5**. Jeśli intencją jest suma powierzchni segmentów z nasyceniem, nazwa coverage wymaga doprecyzowania; jako fizyczny udział pokrytego obrazu wynik jest błędny.

Nowy ekstraktor dzieli powierzchnie masek przez rozmiar oryginalnego obrazu, bez wymuszenia natywnej siatki masek. Używa `YOLO(...)`, nie ustawia `retina_masks`, `imgsz` ani nie przeskalowuje masek przed statystykami. Lokalny Ultralytics 8.3.236 domyślnie ma `retina_masks=False`; segmentacyjny predictor tworzy wtedy maski w siatce wejścia modelu. Oryginalny FastSAM miał `retina_masks=True`, `imgsz=1024` i filtr małych masek. Test zależności od podanego rozmiaru dał coverage 0,25 po zmianie `orig_shape` z 10×10 na 20×20 przy tych samych maskach. Wpływu na rzeczywiste obrazy nie mierzono.

Dowód: `sam_overlap_and_coordinate_grid`. Lokalizacje: [build_cache.py:182](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/build_cache.py:182), wiersze 214–265; [fastsam_extractor.py:33](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/feature_extractors/fastsam_extractor.py:33), wiersz 284; [lokalny predictor:99](/F:/Python/Lib/site-packages/ultralytics/models/yolo/segment/predict.py:99).

### F09 — P1 — Nowa materializacja nie sprawdza unikalności i spójności bbox [S,T]

Powtórzone `frame_ids` dają różne COCO IDs wskazujące jeden obraz. `has_instrument=True` bez `bbox` staje się po cichu obrazem bez anotacji. Test dał dwa wpisy jednego pliku i zero anotacji. Brakuje kontroli geometrii bbox, unikalności źródłowych ID, zgodności liczby obrazów z budżetem oraz wieloobiektowego formatu anotacji. Sama deklaracja `bbox` jako pojedynczego pola ogranicza zarówno scoring, jak COCO do jednego obiektu na klatkę.

`--absolute-paths` rozwiązuje ścieżkę względem SmallViTAblation, ignorując `images_root` stosowany przez ekstrakcję. Przy domyślnym `images_root='.'` nie powoduje różnicy, ale alternatywny root może wskazać inne lub nieistniejące pliki. Walidacja COCO nie wykrywa tych klas błędów. Potwierdzono natomiast poprawne odrzucenie wskazanego `split=test`; poprawnie opisane negatywy pozostają w `images` bez anotacji.

Dowody: `coco_duplicate_paths_and_silent_missing_bbox`, `coco_train_only_guard`. Lokalizacje: [materialize.py:26](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/selection/materialize.py:26), wiersze 74–96. Kontrakt pojedynczego bbox: [agents/README.md:64](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/agents/README.md:64).

### F10 — P1 — Oryginalny eksport może zgubić anotacje i nadpisać obrazy [S]

Kopiowanie pod samym basename powoduje kolizję nazw z różnych katalogów. Eksport wyszukuje anotacje w `self.coco_handler.images`, chociaż nowe źródło ładowane jest przez odrębny handler, a `merge_datasets()` zwraca osobny słownik bez aktualizacji głównego handlera. Dotyczy to szczególnie przypadku tylko nowego źródła i łączenia kilku źródeł. Zmienna `annotations` pełni przy eksporcie praktycznie rolę warunku obecności, nie stanowi rzeczywistego źródła filtrowanych rekordów.

Jest to błąd pokazanego eksportera. Nie dowodzi błędu w ręcznie lub osobno naprawionym `merged_20k_annotations_fixed.json`, dla którego stwierdzono 20 000 obrazów i 20 000 anotacji.

Lokalizacje: [main_selection_pipeline.py:257](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/main_selection_pipeline.py:257), wiersze 265–267 i 400–423; [coco_handler.py:141](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/utils/coco_handler.py:141).

### F11 — P2 — B-UNIFORM i B-RANDSRC wymagają doprecyzowania [S,T,D]

B-UNIFORM po wyczerpaniu małego pacjenta wypełnia brak pierwszymi niewybranymi klatkami w kolejności puli. Nie rozdziela nadwyżki równomiernie między pozostałych pacjentów. Dla pojemności `[1,10,10]` i budżetu 12 test dał **[1,7,4]**.

B-RANDSRC rozpoznaje źródło wyłącznie przez końcowe `_aug_N` w `frame_id`. Działa dla takiej konwencji identyfikatorów; nie rozpoznaje np. ID z rozszerzeniem `.jpg` ani dowolnych innych nazw augmentacji. Ostatnia grupa jest ucinana deterministycznie po nazwie wariantu. Nie stanowi zatem zawsze wyboru kompletnych grup źródłowych, mimo skróconego opisu. Bez augmentacji konceptualnie sprowadza się do losowania źródeł będących pojedynczymi klatkami.

Selekcja klastrowa i B-RANDAUG nie narzucają limitu wariantów jednego źródła. Budżet 20 000 obrazów nie oznacza 20 000 niezależnych źródeł — produkcyjny pomiar daje 15 565. Samo dopuszczenie wielu augmentacji nie jest błędem, jeśli jest zamierzoną jednostką budżetu; musi jednak pozostać jawne w porównaniu z baseline źródłowym.

Dowód: `uniform_leftover_bias`; pomiary w sekcji 2. Lokalizacje: [baselines.py:33](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/selection/baselines.py:33), wiersze 73–86 i 97–119.

### F12 — P1 — Starszy oryginalny kcenter ma błędy interfejsów [S]

Przy skonfigurowanym DETR wywoływana jest nieistniejąca metoda `DETR_EL2N_Scorer.get_score_statistics()`. Dalej kombinacja cech oczekuje kluczy `dino_diversity` i `sam_complexity`, choć bieżący YAML dostarcza `dino` i `sam`. Są to niezależne przeszkody dla `--method kcenter`, nie dla domyślnego `cluster`. Oryginalne wywołanie k-center nie przekazuje także ziarna do losowania pierwszego centrum.

Lokalizacje: [combined_selector.py:189](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/combined_selector.py:189), wiersze 281–282 i 315–317; [k_center_greedy.py:104](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/k_center_greedy.py:104).

## 5. Proxy, coverage i checkpointy — fakty oraz hipotezy

**Train-only nowego przepływu [S,T]:** `train_pool()` filtruje `split == 'train'`. Proxy losuje z tej puli, ekstrakcja i scoring obejmują train, a materializacja odrzuca inne splity. Z-score i PCA obliczane są na przekazanej puli train. To poprawny kierunek w porównaniu z rekurencyjną pulą oryginału. Zgodność samych etykiet split z pacjentami i źródłami nie jest przez te funkcje sprawdzana.

**Coverage proxy [S,H]:** proxy losuje maksymalnie 2 500 obrazów bez stratyfikacji po pacjencie, źródle, negatywach i pochodzeniu anotacji. Nie ma gwarancji pokrycia małych grup. Część ocenianej puli była użyta do uczenia proxy, a część nie; przy augmentacjach możliwe są również pokrewne warianty. Wpływ tej asymetrii na ranking jest hipotezą, nie wykazanym pomiarem. Lokalizacja: [train_el2n_proxy.py:146](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/train_el2n_proxy.py:146).

**Coverage klastrów [S,D]:** jeden reprezentant każdego niepustego klastra daje pokrycie etykiet klastrów, nie gwarancję pokrycia pacjentów, narzędzi, klatek źródłowych czy ogona rozkładu. Oryginał pomija puste klastry i może oddać mniej obrazów niż target. Nowy kod uzupełnia do budżetu po `frame_id`, co może wprowadzić bias kolejności. Oba odczytane historyczne raporty faktycznie zawierają 20 000 reprezentantów 20 000 klastrów; to nie rozstrzyga pokrycia innych jednostek danych. Lokalizacje: [cluster_selector.py:717](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/selection_methods/cluster_selector.py:717), wiersz 773; [select_subset.py:185](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/selection/select_subset.py:185).

**Wybór checkpointu proxy [S,H]:** kod szuka `best_checkpoint.pth`, a awaryjnie bierze leksykograficznie ostatni `.pth`. Nie egzekwuje konkretnej epoki, hasha ani semantyki checkpointu. Nie przekazuje jawnego output directory do `det.train()`, a wag szuka w katalogu danych. To potwierdzone założenia kodu, ale ich zgodności z konkretną wersją rfdetr nie zweryfikowano. Sprawdzony Python 3.11 nie ma pakietu `rfdetr`. Nie instalowano go. Lokalizacja: [train_el2n_proxy.py:83](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/train_el2n_proxy.py:83), wiersze 96–104.

**Adapter RF-DETR [H]:** pozostają nieweryfikowane: argument `pretrained`, poziom `det.model` dla `load_state_dict/eval/to`, format iteratora `predict()`, rekonstrukcja głowicy klas, rozdzielczość i format klucza wag. `strict=False` bez kontroli missing/unexpected keys nie dokumentuje poprawności załadowania wag; nie oznacza jednak, że dowolny mismatch wymiarów będzie zaakceptowany. Test atrapy detektora nie potwierdza kompatybilności API. Lokalizacja: [el2n_from_proxy.py:63](/F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/pipeline/features/el2n_from_proxy.py:63).

**Geometria i wiele obiektów [S,H]:** kontrakt opisuje bbox w pikselach obrazu po normalizacji, a exporter zapisuje `native_width/native_height`. Bez rzeczywistego manifestu i obrazu nie wiadomo, czy to wspólna siatka współrzędnych. Interfejs przechowuje jeden bbox; nie ustalono liczby pomijanych dodatkowych instancji w nowym zbiorze. Nie wolno wyprowadzać jej wyłącznie z opisowych oczekiwanych liczebności w konfiguracji.

**Brak nowych danych [D]:** w czasie audytu nie istniały domyślne `SmallViTAblation/pool_manifest.jsonl`, `queue.jsonl` ani `feature_cache/manifest.json`. Nie wykonano więc pomiarów rzeczywistej nowej selekcji, skutków cache, pokrycia proxy ani rozłączności źródeł. Nie przypisano hipotetycznych skutków do niewykonanego nowego biegu.

## 6. Minimalne warunki porównania dla RT-DETR

1. Ustalić, czy celem jest odtworzenie oryginalnego `cluster`, starszego `kcenter`, czy nowa metodologia. Zachować i opisać uzgodnione definicje D/F/S/E, kolejność PCA i normalizacji oraz znaczenie wag.
2. Jawnie zdefiniować score z surowych wyjść głowicy właściwego detektora: kodowanie klas, query matching, nieprzypisane queries, negatywy, pominięte GT, FP i agregację. Oddzielić score klasyfikacyjny od ewentualnego błędu lokalizacji. Nie przenosić Query 81.
3. Wiązać proxy i cache z architekturą, wagami, epoką, preprocessingiem, hashem manifestu i anotacji. Odrzucać fake w produkcji. Sama zmiana `model.name` albo `run.arch` nie przełączy obecnych wywołań `RFDETRSmall` na RT-DETR.
4. Zdefiniować jednostkę budżetu i pokrycia: obraz po augmentacji versus źródło; rozłączność splitów sprawdzać na właściwej jednostce. Dla produkcyjnego selected_dataset1 uwzględnić ustalone membership bieżącego splitu oraz 48 nazw z listy wycieków.
5. Przed uruchomieniem produkcji sprawdzić adapter detektora na kontrolowanych przykładach GT/negatywów i potwierdzić dokładny checkpoint. Zapewnić integralność COCO, ścieżek, wielu obiektów oraz zgodności wymiarów obrazów i bbox.

To zakres zalecanych poprawek wynikający z audytu; żadnej z nich tutaj nie wdrażano.

## 7. Reprodukcja

Pierwsze wykonanie, już zrealizowane:

```powershell
py -3.11 -B 'F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/evidence/reproduce_selection_bugs.py'
```

Ponowne wykonanie musi używać nowej nazwy wyniku, ponieważ skrypt odmawia nadpisywania artefaktów:

```powershell
py -3.11 -B 'F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/evidence/reproduce_selection_bugs.py' `
  --out 'F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/evidence/selection_repro_results_repeat.json'
```

Skrypt wymaga wyłącznie numpy, PyYAML i scikit-learn. Odczytuje istniejący kod oraz wskazane JSON-y, tworzy jeden nowy plik wynikowy poza repozytorium i nie wykonuje modeli. Wynik zawiera wersje pakietów, hashe odczytanych źródeł, dane wejściowe kontrprzykładów, oczekiwane symptomy i ograniczenia. Nie jest testem end-to-end produkcyjnego pipeline’u. Po naprawie kodu część sprawdzeń ma przestać odtwarzać stare błędy i zgłosić `unexpected_failure`; jest to skrypt dowodowy stanu z dnia audytu, nie gotowa docelowa suita regresyjna.
