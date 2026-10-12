# PLAN: rzeczywisty replay 900 000 cząsteczek

## Cel i kryterium obalenia

Eksperyment odtwarza dokładnie 500 aktualizacji starego per-image oracle dla exact1500 i zapisuje rzeczywiste stany co 5 aktualizacji. Wynik jest odrzucony, jeżeli stan naturalny, krok 100 albo krok 500 przekroczy tolerancję `2e-6` względem starego artefaktu, zmieni się dyskretna telemetria, model przestanie być zamrożony albo zabraknie choć jednego obrazu, kroku lub tokenu.

## Wejścia i niezmienniki

- checkpoint K1000: SHA-256 `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`;
- exact1500: SHA-256 `33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7`;
- stary indeks oracle: SHA-256 `e990f9032efc8b8e13727752d099488688d0f643e1b49e6d0ac601eb3caee3fb`;
- stała baza PCA3: SHA-256 bazy `8d3cb28ad2b1581efeca89fab659b9d6fa518bffe4d452b64c68ea7dcd1b9cca`;
- FP32, TF32 wyłączone, model w `eval`, wszystkie wagi i bufory zamrożone;
- ten sam Adam, schedule, GELU floor, przełączenie fit/minimum-norm i pełny horyzont 500 co w starym oracle;
- obrazy są czytane bezpośrednio z istniejącego TAR do pamięci. Eksperyment nie ekstrahuje ani nie zapisuje obrazów.

## Dane wyjściowe

Każdy z 1500 plików zawiera `projected_positions(101,600,3)`, `movement_norm(101,600)`, kroki 0..500 co 5 oraz pełną telemetrię 501 stanów. Projekcja PCA3 służy wyłącznie wizualizacji.

Globalne deskryptory wszystkich 900 000 cząsteczek są liczone przed PCA, w pełnych 768 wymiarach. Dla każdego interwału pięciu rzeczywistych aktualizacji zapisujemy normę przyrostu, iloczyn skalarny i cosinus względem poprzedniego przyrostu, normy gradientu oraz normy m1/m2 Adama na początku przejścia, średnią i końcową normę gradientu, fazę oraz role tokenu. Gradient i momenty z początku są znane przy starcie przejścia. Średnia i końcowa norma gradientu oraz faktyczny przyrost są diagnostyką teacher-forced/outcome i nie mogą być podawane jako przyszła siła do prognozy tego samego przejścia. Są to deskryptory skalarne; nie udają pełnej trajektorii ani kierunku sterowania 768D.

Jedna zamrożona analityczna PCA32 powstaje przed replay z historycznych stanów 0, 100 i 500. Fit obejmuje wyłącznie obrazy 40 filmów train z istniejącego kontraktu panelu oraz po jednym deterministycznie wybranym tokenie z każdej komórki przestrzennej 4x4. Validation i test służą wyłącznie do pomiaru zachowanej energii stanu i ruchu. Każdy obraz zapisuje `analysis_positions_pca32(101,600,32)` oraz pełnowymiarowe normy residuum stanu i pięciokrokowego przyrostu. PCA32 daje wspólne kierunki całej populacji, ale bez wektorowych g/m1/m2 całej populacji nie zamienia analizy w pełny kontrolowany DMDc; ten pozostaje dostępny dla panelu 64x16.

Panel 64 obrazów po 16 tokenów zapisuje pełne 768D w każdym kroku 0..500 oraz gradient, pierwszą i drugą chwilę Adama przed każdą aktualizacją. Panel zawsze obejmuje target, naturalny top1 i konkurentów. Split 40/12/12 jest rozłączny po filmach. To zakres analizy wstępnej, nie dowód o całej przestrzeni kierunków 900 000 cząsteczek.

## Budżet i wykonanie

- PCA3 i norma netto: 1 454 400 000 bajtów;
- globalne deskryptory 768D-derived: 2 880 000 000 bajtów;
- panel pełny 768D: 6 294 601 728 bajtów;
- analityczna PCA32 pełnej populacji: 11 635 200 000 bajtów;
- normy residuum PCA32 w 768D: 723 600 000 bajtów;
- razem surowe tablice replay: 22 987 801 728 bajtów, bez zakładania kompresji;
- końcowe 101 buforów wizualizacji: 1 454 400 000 bajtów; łącznie około 24 442 201 728 bajtów przed małymi metadanymi;
- próg preflight: co najmniej 35 GB wolnego miejsca;
- smoke: obrazy 91, 684 i 0;
- produkcja: osiem niezależnych shardów GPU, bez DDP i bez współdzielenia korekt;
- merge publikuje manifesty dopiero po walidacji wszystkich 1500 obrazów i 64 paneli.

`VISUALIZATION_MANIFEST.json` jest wejściem dla eksportera należącego do main. Na jawne polecenie użytkownika ten niezmodyfikowany exporter jest wywoływany w tym samym zadaniu merge, a jego `results/visualization_dense/manifest.json` stanowi końcową bramkę. `TEMPORAL_PANEL_MANIFEST.json`, `GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json` i `ANALYTICAL_PCA32_MANIFEST.json` są wejściami dla analizy Zeno. Eksperyment nie uruchamia PyDMD/deeptime i nie modyfikuje viewer'a.
