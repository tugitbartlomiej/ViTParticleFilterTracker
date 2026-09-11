# Proponowany protokół selekcji, treningu i oceny RT-DETR

> Archiwum audytu sprzed refaktoru kodu. Aktualne poprawki i testy: [21_STAN_REFAKTORU.md](../21_STAN_REFAKTORU.md). Numery linii i wyniki reprodukcji odnoszą się do zapisanej wersji źródeł.

Data: 2026-09-11. **Projekt eksperymentu, jeszcze niewdrożony.** Wartości budżetów, liczby powtórzeń i etapów poniżej są propozycją dla tego projektu, nie obowiązującym standardem z jednej publikacji. Lista źródeł i zakres ich lektury: [ZRODLA.md](ZRODLA.md).

## 1. Najpierw zdefiniować, co sprawdzamy

Główne pytanie: czy przy tej samej liczbie **unikalnych klatek źródłowych**, tej samej inicjalizacji i tym samym treningu metoda selekcji poprawia detekcję końcówki narzędzia na nowych operacjach?

To jest selekcja z już opisanej puli, czyli supervised subset/coreset selection. Nie należy nazywać jej oszczędnością etykietowania, jeśli GT wszystkich kandydatów było używane do selekcji. Eksperyment active learning wymaga osobnego protokołu: GT niewybranych kandydatów pozostaje niewidoczne. Rozróżnienie to występuje w [OFDS](https://arxiv.org/abs/2412.10032).

Drugie pytanie, badane osobno: czy dobry podzbiór przenosi się na inny detektor? W tym eksperymencie przekazujemy **ten sam manifest wybranych obrazów**. Ponowne obliczenie trudności inną architekturą i wybranie nowego podzbioru bada inny problem.

## 2. Dane i granice dostępu

1. Utworzyć kanoniczny manifest oryginalnych obrazów: `image_id`, `source_frame_id`, `patient_id` gdy znany, `procedure_id`, `video_id`, `timestamp`, `path`, `image_sha256`, `annotations[]`, `split`. Lista adnotacji może mieć zero, jeden lub kilka obiektów.
2. Najpierw podzielić po pacjencie; gdy znana jest tylko operacja, podzielić po operacji i tak to nazwać. Kilka klipów tej samej operacji musi należeć do jednej grupy. Brak mapowania pacjentów nie daje prawa do deklarowania niezależności pacjentów.
3. Zamrozić train/val/test przed ekstrakcją cech, PCA, klastrowaniem i uczeniem proxy. Te operacje widzą wyłącznie train. Zachować identyczny split we wszystkich porównywanych wariantach.
4. Selekcję przeprowadzić **przed augmentacją**. Po wyborze identyczny, jawny schedule augmentacji w treningu wszystkich wariantów. Budżet nie rośnie od dodania `_aug_1`.
5. Val/test mają zawierać oryginalne klatki, w tym klatki bez końcówki. Nie dobierać ich pod predykcje modelu. Zdefiniować wspólny harmonogram próbkowania wideo, aby jeden długi film nie stanowił prawie całego testu.
6. Sprawdzić wspólne operacje, `source_frame_id`, dokładne hashe oraz potencjalne duplikaty percepcyjne. Sam brak wspólnych nazw jest niewystarczający.
7. GT sprawdzić wizualnie na próbie warstwowej: małe końcówki, rozmycie, odbłyski, zasłonięcia, brzegi kadru, więcej niż jeden obiekt oraz tło. Brak etykiety widocznej końcówki nie jest prawidłowym przykładem negatywnym.

W lokalnej kopii `fullpool_videosplit/annotations_train.json` policzono 71 316 obrazów i **17 832 nieaugmentowane nazwy / identyfikatory po usunięciu suffixu augmentacji**. Zapis: [train_pool_counts.json](../pochodne/evidence/train_pool_counts.json). Zatem 20 000 unikalnych klatek źródłowych nie jest wykonalnym budżetem dla tej konkretnej puli. Pochodzenie i kompletność jej plików należy jeszcze sprawdzić. Nie należy szacować rozmiaru train jako 70% klatek tylko dlatego, że zawiera 70% filmów.

Jeśli rozszerzamy dane o Cataract-LMM, zbiory tracking i segmentation nie mogą być uznane za niezależne tylko dlatego, że są w różnych folderach. Brak znanego wspólnego identyfikatora operacji nie dowodzi braku nakładania. Dotyczy to również użycia jednego z nich jako external test.

## 3. Wybór modelu i kontrakt implementacji

**RT-DETR, RT-DETRv2 i RF-DETR nie są zamiennymi nazwami.** Obecny `SmallViTAblation` zawiera adapter RF-DETR. Dla RT-DETR proponuję zacząć od oficjalnego RT-DETRv2 z ResNet-18 jako taniej sondy, a najlepszy podzbiór potwierdzić na ResNet-50. Jeśli potrzebny jest dokładnie RT-DETR v1, trzeba jawnie wybrać jego implementację i checkpoint, zachowując resztę protokołu. To propozycja kosztowa, nie dowód, że mniejsza sonda zachowa ranking większej.

Przypiąć commit oficjalnego repozytorium, wersje bibliotek oraz hash wag COCO. W konfiguracji podać architekturę, backbone, rozdzielczość, liczbę zapytań, głowę jedno-klasową, EMA i precision. Nie wznawiać historycznego ep170: badanie selekcji ma zaczynać każdy wariant od tego samego niezależnego punktu startowego. Proxy również nie może dziedziczyć ekspozycji na test.

Adapter ma przyjmować obraz i zwracać `boxes_xyxy_original_pixels`, `scores`, `class_ids`, `image_id`. Sprawdzić transformację resize/padding i jej odwrócenie dla niesymetrycznego obrazu. Oficjalny postprocessor RT-DETRv2 korzysta z konfiguracji aktywacji i top-k; w trybie focal stosuje sigmoid, nie softmax po jednej klasie. Nie używać stałego Query 81. [Kod postprocessora](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py).

**Pułapka klas w RT-DETRv2:** `num_classes=1` wymaga wewnętrznej etykiety klasy 0. Jeżeli pliki COCO przechowują `category_id=1`, potrzebne jest jawne mapowanie `1 -> 0` w treningu i odwrotne w eksporcie. Samo `remap_mscoco_category=False` w oficjalnym loaderze pozostawia surowe identyfikatory. Alternatywnie można konsekwentnie użyć id 0 w całym osobnym eksporcie COCO. Nie naprawiać tego przez uznanie każdej przewidzianej klasy za tooltip. [Loader](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/data/dataset/coco_dataset.py).

## 4. Selekcja: prosty wariant referencyjny i rozszerzenie z trudnością

### Wariant referencyjny: cechy obiektu i pokrycie

Wyliczyć zamrożone cechy DINO dla obszaru GT z ustalonym marginesem kontekstu oraz opcjonalnie cechy całego obrazu. W tej domenie bardzo mały obiekt może ginąć w globalnym CLS. Crop lub pooling tokenów w obszarze obiektu jest hipotezą uzasadnioną metodami obiektowymi, którą trzeba porównać z CLS na własnych danych. Nie oznacza to, że globalny kontekst jest bezwartościowy. Dla kilku obiektów zachować reprezentacje wszystkich i wybierać całe obrazy. Dla tła użyć osobnej grupy oraz cech globalnych. Punkty odniesienia: [CSOD](https://arxiv.org/abs/2404.09161), [OFDS](https://arxiv.org/abs/2412.10032).

Normalizację i PCA dopasować tylko na train; zamrozić i zapisać ich parametry. Przy reprodukcji artykułu zachować kolejność `PCA -> z-score -> sqrt(w_g/d_g) -> concatenate`, wymiary i definicje cech. Mnożenie bloku przez samo `w_g` zmienia metrykę klastrowania, bo wkład do kwadratu odległości rośnie jak `d_g*w_g^2` dla niezależnych standaryzowanych wymiarów. Po PCA bez ponownej standaryzacji trzeba uwzględnić rzeczywiste wariancje składowych.

K-means z najbliższym centroidowi reprezentantem jest poprawnym baseline'em prototypów. Nie ma zakazu `k=budget` ani gwarancji równych liczebności klastrów. Liczy się porównanie jakości i kosztu. Selekcja po centroidach z dołączonym skalarem trudności **nie oznacza automatycznie wyboru najtrudniejszych próbek**.

### Rozszerzenie: modelowo zależna trudność detekcji

Najbardziej przejrzysta pierwsza wersja to krótki trening proxy RT-DETRv2 na **całej puli train**. Ustalić kilka wczesnych checkpointów na etapie pilotażu i zapisać osobno ziarna proxy oraz finalnego treningu. Następnie policzyć wynik każdej oryginalnej klatki w trybie ewaluacji bez losowych augmentacji.

Proponowany sygnał: **per-image detection loss / detection difficulty**, oparty na głównym wyjściu dekodera, z dopasowaniem wszystkich GT do zapytań przez natywny matcher. Zachować składowe klasyfikacji, L1 i GIoU oraz ich jednostki/normalizację. Klasyfikacja tła nie może zostać pominięta. Obliczenie loss dla pojedynczej klatki albo jawnych niere­dukowanych składowych musi dawać wynik niezależny od obrazów współdzielących batch. Nie wystarczy podzielić batch loss przez batch size. [Matcher](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/rtdetrv2_pytorch/src/zoo/rtdetr/matcher.py).

Nie filtrować zapytań progiem 0,3 ani IoU 0,5 przed tym treningowym dopasowaniem. Matcher treningowy przyporządkowuje również słabą lokalizację, żeby jej błąd pozostał mierzalny. Jego koszt i późniejszy loss to różne wielkości. Obsłużyć pusty GT: brak predykcji obiektu powinien oznaczać mały błąd tła, a silne odpowiedzi na tło duży. Obsłużyć też więcej GT niż dostępnych zapytań jako jawny przypadek niewykonalny.

Można testować odrębne składowe `1-p_target`, `1-IoU` lub zmienność po checkpointach, ale nazwać je dokładnie. Oryginalny EL2N to `E ||p-y||_2`, opracowany dla klasyfikacji. Dla dwuklasowego softmax `[p,1-p]` i celu `[1,0]` norma wynosi `sqrt(2)*(1-p)`; dla pojedynczej odpowiedzi sigmoid z celem 1 norma skalarna to `1-p`. Zbieżność rang w tym szczególnym przypadku nie czyni progowania, spatial matchingu i agregacji detekcyjnej oryginalnym EL2N. [Paul et al., definicja 2.3](https://arxiv.org/abs/2107.07075).

**Ocena train przez proxy uczony na train nie jest sama w sobie wyciekiem testowym.** Jest standardowym składnikiem wielu metod data pruning. Problem obecnej propozycji z 2500 obrazami to mieszanie ekspozycji: część rankowanych obrazów proxy widział, innych nie. Możliwe rozwiązania: jednakowo krótki trening na całym train; cross-fitting po operacjach; albo jawna adaptacyjna metoda uwzględniająca kept/pruned. OOF nie jest warunkiem definicji EL2N. Usunięcie z rankingu całych 2500 obrazów też zmienia pulę i wymaga identycznego ograniczenia baseline'ów.

Na początek porównać centroidy z wyborem według trudności wewnątrz tych samych klastrów. Ograniczyć bliskie czasowo duplikaty, zachowując rzadkie zdarzenia i tło; minimalny odstęp ustalać na train/val i raportować jako ograniczenie, nie jako arbitralne usuwanie każdej podobnej klatki. Wariant z budżetem per klaster lub adaptacją feedbacku można dodać później. [COWAL](https://openaccess.thecvf.com/content/WACV2024/html/Wu_Correlation-Aware_Active_Learning_for_Surgery_Video_Segmentation_WACV_2024_paper.html), [AdaDeDup](https://arxiv.org/abs/2507.00049) są inspiracją, nie walidacją tej konkretnej hybrydy.

## 5. Macierz doświadczeń i uczciwy koszt

Proponowany pierwszy etap: budżet 2500 lub 5000 unikalnych klatek, 3 sparowane powtórzenia każdego wariantu, potem co najmniej 5 dla głównych porównań, jeśli koszt pozwala. Przy małych różnicach liczba powtórzeń może wymagać zwiększenia. Nie istnieje reguła, że 5 seedów automatycznie daje wiarygodny przedział.

| Wariant | Co rozstrzyga |
|---|---|
| Random źródłowych klatek bez zwracania | Główne odniesienie przy tym samym budżecie |
| Równomiernie w czasie w obrębie operacji | Czy metoda robi więcej niż redukcja sąsiednich klatek |
| Random z ograniczeniem sąsiedztwa czasowego | Czy przewaga wynika głównie z unikania sąsiadów |
| Globalne DINO + centroid | Wartość prostych cech całej sceny |
| Obiektowe DINO + centroid / CSOD | Wartość reprezentacji obiektu |
| Sam sygnał trudności | Wkład modelu proxy |
| Obiektowe cechy + trudność + ograniczenie czasowe | Proponowana metoda |
| Wszystkie oryginalne train | Punkt odniesienia pełnych danych; nie gwarantowany sufit |

Porównanie z oryginalnym DESF musi zachować jego cechy i skalowanie. Zmiany DINOv2/v3, Fourier 9/1 oraz SAM 1/9 są osobnymi osiami eksperymentu. Przeprowadzenie 15 kombinacji zmienionych cech nie jest reprodukcją starej ablacji.

W obrębie tego samego budżetu i architektury ustalić identyczne: początkowe wagi, aktualizacje optymalizatora, efektywny batch, LR i schedule, augmentation, resolution, EMA, precision i regułę checkpointu. Pełny pipeline jest normalnym wierszem tej samej macierzy. Model finalny nie startuje od proxy po zobaczeniu całej puli.

Dla różnych budżetów rozdzielić dwa pytania: (a) ten sam limit aktualizacji/GPU-czasu, (b) ten sam trening na epokę lub trening do uprzednio ustalonej zbieżności. Przy równej liczbie epok mniejszy zbiór otrzymuje mniej aktualizacji. Raportować obie osie: **AP względem unikalnych klatek** i **AP względem całkowitego kosztu**. Koszt obejmuje ekstrakcję, proxy, wybór i trening, z zaznaczeniem amortyzacji wspólnego cache. [Selection via Proxy](https://arxiv.org/abs/1906.11829), [DeepCore](https://arxiv.org/abs/2204.08499).

Porównanie różnych architektur nie wymaga tych samych wartości LR. Wymaga równego budżetu strojenia i jasno określonego kryterium porównania. Nie nazywać dwóch odmiennych recept „iso-resource” bez pomiaru wykorzystanych zasobów.

## 6. Jak policzyć metryki

| Cel | Metryka i sposób liczenia |
|---|---|
| Podstawowa jakość detekcji | COCO AP@[.50:.95], AP50, AP75 oraz AR@100, jeden evaluator dla wszystkich modeli |
| Małe końcówki | AP_small według definicji COCO, z liczbą GT w tej grupie; dodatkowe przedziały względnego rozmiaru bbox jako analiza domenowa |
| Punkt pracy | TP, FP, FN, precision, recall, F1 przy progu wybranym na val i zamrożonym przed testem |
| Fałszywe alarmy | FP/image oraz recall przy ustalonym limicie FP/image; opcjonalnie krzywa FROC |
| Precyzja lokalizacji | Błąd środka bbox w pikselach obrazu oryginalnego i względem przekątnej obrazu, median/p95; jawnie tylko dla dopasowanych oraz z podanym odsetkiem niewykrytych |
| Rzeczywisty punkt końcówki | Wymaga anotacji punktowych; środek ramki nie jest automatycznie końcówką. Jednostki mm wymagają kalibracji |
| Stabilność między operacjami | Wyniki per operacja, rozkład i sparowana różnica; globalne COCO AP raportowane osobno |
| Kalibracja | Reliability diagram / jawnie zdefiniowany błąd kalibracji detekcji po dopasowaniu, rozpatrywany oddzielnie od AP |
| Działanie w czasie | Latency batch=1, mediana/p95, hardware/runtime/precision/resolution; pełny czas i czas samego forward oddzielnie |

**Kontrakt COCO:** ocena obejmuje wszystkie `image_id` z zamrożonego manifestu testu. Nie wolno budować tej listy z predykcji. GT i predykcje używają wspólnych id obrazów i klas, boxy mają poprawne jednostki. Obsłużyć brak predykcji, brak GT, duplikaty, kilka obiektów, crowd/ignore. Niezdefiniowane AP dla pustej kategorii zwracać jako NA, nie po cichu zero. Standard COCO używa progów IoU .50:.05:.95, 101 punktów recall i `maxDets=[1,10,100]`; liczba top-k detektora jest osobnym parametrem. [Kod COCO](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py).

Dla AP zachować standardowe top-k detektora bez progu operacyjnego. Jeśli oszczędności wymagają cutoff 0,001, sprawdzić na val, czy obniżenie go do zera nie zmienia metryk i liczby obrazów objętych oceną. 0,001 nie jest uniwersalną definicją pełnej krzywej. Próg 0,3 należy wyłącznie do osobno opisanej oceny operacyjnej. Przy identycznej kolejności i zbiorze predykcji ściśle monotoniczna transformacja confidence nie zmienia AP. Dlatego niskie AP samo w sobie nie dowodzi złej kalibracji.

Maksimum F1 wybierać z kandydatów na val; przy próbie obejmującej bardzo małe confidence siatka 0,01...0,99 jest niewystarczająca. Brak zapisanego progu dla testu ma kończyć obliczenie błędem. Można zamiast max-F1 ustalić próg realizujący docelowe recall lub FP/image, ale kryterium musi być określone przed testem.

Próg lokalizacji IoU 0,5 nie jest sam w sobie potwierdzeniem przydatności klinicznej. Małe ramki silnie reagują na przesunięcie kilku pikseli. Wybór dodatkowych metryk powinien odpowiadać wymaganej dokładności anotacji i zadania. [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z), [Metric pitfalls](https://doi.org/10.1038/s41592-023-02150-0).

Pomiar szybkości wymaga rozgrzewki modelu do stabilnego czasu, wyłączenia gradientów i trybu `eval`. Na CUDA należy użyć zdarzeń GPU lub synchronizacji przed i po mierzonym odcinku; sam zegar CPU bez synchronizacji może mierzyć wyłącznie wysłanie pracy do GPU. Podawać osobno batch=1 latency oraz throughput dla większego batcha. Pełny czas obejmuje ustalony zakres dekodowania obrazu, preprocess, transfer, forward i postprocess; dla YOLO również NMS. Dysk i dekodowanie wideo można mierzyć osobno, ale granice pomiaru muszą być jawne i wspólne. Czas ładowania wag/kompilacji jest osobnym kosztem startowym. Porównywać na tym samym urządzeniu i runtime; nie zestawiać własnego PyTorch FP32 z reklamowanym TensorRT FP16 jako równoważnych FPS. [RT-DETR, sekcja 3.2](https://arxiv.org/abs/2304.08069).

## 7. Niepewność, checkpointy i test zewnętrzny

Zapisać osobno `selection_seed`, `proxy_seed`, `train_seed` i `split_id`. Raportować surowe wyniki oraz mean/SD między treningami; nie mylić SD z przedziałem ufności. W głównym porównaniu parować warianty po splitach i seedach. [Bouthillier et al.](https://arxiv.org/abs/2103.03098).

Proponowany CI dla wyniku **ustalonych modeli**: sparowany bootstrap po operacjach/pacjentach. Losujemy te same operacje ze zwracaniem dla obu metod, zachowujemy ich klatki i ponownie liczymy globalne AP oraz różnicę AP. Powtórzone grupy muszą dostać nowe `image_id` i `annotation_id`; inaczej evaluator może usunąć ich krotność. Nie bootstrapować samych skorelowanych klatek tak, jakby pochodziły od niezależnych pacjentów. Seed variability i test-cohort uncertainty raportować oddzielnie; wspólny hierarchiczny CI wymaga jasnej definicji estymandu. Przy tylko siedmiu operacjach testowych przedziały mogą być niestabilne; warto pokazać wyniki każdej operacji i czułość leave-one-operation-out.

Przy większym porównaniu 15 wariantów unikać wybierania zwycięzcy na teście. Wybrać finalną receptę na val albo w wewnętrznym group CV, a następnie potwierdzić na odłożonym teście. Testowanie wielu hipotez i wybór najlepszej wymagają uwzględnienia wielokrotnych porównań. [Cawley i Talbot](https://www.jmlr.org/papers/v11/cawley10a.html).

`last_epoch` to rzeczywiście ostatni checkpoint według metadanych epoki/global_step; `best_val` to najlepszy według uprzednio określonego AP na val. Nazwa pliku i kolejność alfabetyczna nie wystarczają. EMA i non-EMA nie są zamiennymi checkpointami. Artefakty: `run_id/checkpoint_sha256/dataset_sha256/{predictions,metrics,operating_point}.json`.

Wyniki obu reguł można opublikować, jeżeli reguły wybrano przed testem. Nie wolno wybierać potem lepszej kolumny. Limit „jeden checkpoint na run” w obecnym configu wymaga uzgodnienia z deklaracją raportowania dwóch reguł; nie jest sam w sobie wymogiem naukowym. Również powtórzenie ewaluacji po naprawie błędu jest dopuszczalne, jeśli zostanie jawnie opisane, bez strojenia na nowym wyniku.

Historycznie oglądany benchmark zewnętrzny może pozostać zbiorem porównawczym, ale nie odzyskuje statusu nigdy niewidzianego po utworzeniu ledgeru. Dla mocnego potwierdzenia zebrać nową, niezależną kohortę; albo uczciwie ograniczyć deklarację. Wariant CLEAN usuwający 48 nazw nie gwarantuje niezależności pozostałych klatek tej samej operacji.

## 8. Minimalne warunki przed uruchomieniem kosztownej macierzy

- Testy syntetyczne evaluatora: perfect, empty predictions, empty GT, missed entire image, FP-only image, duplicate detection, wrong class, several overlapping GT, crowd/ignore, invalid image id, xyxy/xywh i odwrócenie resize. Oczekiwane wartości porównać z referencyjnym COCOeval.
- Testy selekcji: dokładny budżet unikalnych źródeł, wyłącznie train, niezależność score od batch size, zgodność wag/PCA, klatka bez GT, zmiana kolejności inputu, powtarzalność seed, odrzucenie starego/fake cache.
- Mały trening kontrolny na kilkunastu obrazach: sprawdzić, czy model potrafi się na nich przeuczyć. To diagnostyka loadera/loss/adaptera, nie wynik generalizacji.
- Jeden pełny pilotaż: random kontra prosty selector, wspólny train/val, poprawnie załadowane i zapisane wagi, potwierdzone num_classes, augmentacje, efektywny batch oraz liczba aktualizacji. Bez dostępu do finalnego testu podczas kalibracji.
- Dopiero po tych kontrolach: cała macierz, potwierdzenie na większym modelu, zamrożone testy i tabela końcowa.

Nie uruchomiono tych treningów w ramach audytu. Testy błędów wykonane teraz znajdują się w `evidence/`; ich przejście oznacza reprodukcję problemu, nie naprawę pipeline'u.
