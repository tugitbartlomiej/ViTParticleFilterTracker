# Kuracja wideo do RT-DETR: audyt literatury i plan ponownej publikacji

Data: 2026-09-11. **Status: ocena i projekt nowych eksperymentów, bez nowych wyników treningu.** Dokument uzupełnia [protokół](01_PROTOCOL.md), [mapę recenzji](04_REVIEWER_MAP.md) i [stan refaktoru](21_STAN_REFAKTORU.md). Nie zastępuje jeszcze konfiguracji wykonawczej. Wskazane niżej parametry eksperymentów są propozycją do zamrożenia po pilotażu na danych rozwojowych.

## 1. Ocena, od której warto zacząć

**Problem ma wartość publikacyjną. Obecny materiał nie dowodzi jednak, że pełna kombinacja DESF jest przyczyną deklarowanej poprawy.** Ponowne zgłoszenie wymaga nowych kontrolowanych treningów, a nie tylko rozszerzenia bibliografii. Podział po klatkach, odmienne harmonogramy full-pipeline i ablacji, dodatkowy fine-tuning oraz wybór odczytu Query 81 utrudniają oddzielenie selekcji od pozostałych zmian. Te problemy są zapisane w TEX i wskazane przez recenzentów.

Sam zestaw DINO + PCA + klastrowanie nie stanowi nowej metody. Istnieją również prace łączące różnorodność z trudnością, selekcję obiektową oraz klastrową selekcję klatek chirurgii zaćmy. **Nie znaleziono podstaw do deklaracji „pierwszy taki pipeline” ani do gwarantowania przyjęcia artykułu.** Najbardziej obronny kierunek to rzetelne badanie, kiedy kuracja pomaga w lokalizacji małych końcówek, co dokładnie daje każda składowa oraz czy zysk utrzymuje się po zmianie pacjentów, budżetu i detektora.

Możliwy wkład nowej wersji: reprodukowalny benchmark kuracji wideo okulistycznego, kontrolowana analiza bloków DESF, porównanie z selektorami obiektowymi i czasowymi, poprawnie rozliczony koszt oraz ocena błędów lokalizacji i ciągłości detekcji. Jeżeli prosty baseline okaże się równie dobry, należy uprościć metodę i opublikować ten wynik w odpowiednio zawężonym zakresie. Dodatni wynik dla pełnego DESF nie jest założeniem badania.

## 2. Co rzeczywiście wczytano i jak oceniano źródła

W `Ocr/raw` zinwentaryzowano **123 pliki Markdown, około 10,13 mln znaków**. Przejrzano tytuły wszystkich plików; dla materiałów istotnych czytano abstrakty i wybrane części metod, eksperymentów, dyskusji lub ograniczeń. **Nie jest to deklaracja przeczytania wszystkich 123 prac w całości.** Zakres dla każdego pliku i hash zawartości podaje [rejestr lektury](pochodne/audyt_ocr_2026-09-11/rejestr_lektury.json). Duplikaty arXiv/CVF oraz wersje preprint/proceedings nie są niezależnymi dowodami. Publikacje o mykotoksynach, aflatoksynach czy kontekstualności poznawczej nie uzasadniają tego pipeline'u; prace o meta-learningu i continual learningu są odległym tłem.

Wykorzystano wyjście istniejącego MinerU. **Nie wykonano OCR.** Sprawdzono bieżący kod `features/selection`, stan wcześniej audytowanego evaluatora, dokumentację, wskazane fragmenty `DETR_IEEE_FINAL.tex` oraz istniejące ekstrakcje tekstu obu recenzji. Główny audyt kodu jest statyczny; dodatkowo odtworzono zmianę centrów k-center na pięciu punktach [diagnostyka](pochodne/audyt_ocr_2026-09-11/kcenter_diagnostyka.json). Nie uruchamiano ekstraktorów produkcyjnych ani nowych treningów. Historyczny test na fake features nie potwierdza produkcyjnej metody.

Uruchomiono i odebrano zakończony [przegląd Undermind](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Kuracja%20wideo%20chirurgicznego%20walidacja%20przyczyn%20efektu%20i%20nowosc%20publikacyjna): 190 wyników, obejrzane pierwsze 50 w rankingu. Dodatkowo Undermind przeczytał pełne PDF-y CSOD, AdaDeDup i COWAL, odpowiadając na pytania o etykiety, reprezentacje, koszty i podziały. To pomocnicza analiza, nie dowód poprawności każdego zdania wygenerowanego przez usługę. Sprawdzono wybrane wnioski w lokalnych tekstach i źródłach pierwotnych. Na przykład brak danej cechy w AdaDeDup **nie dowodzi nowości względem całej literatury**. Surowe odpowiedzi zachowano w kolekcji, w `Analiza_ponownej_publikacji/`.

## 3. Najbliższe odpowiedniki i konsekwencje dla nowości

| Źródło i zakres lektury | Co już istnieje | Znaczenie dla tej pracy |
|---|---|---|
| Yang i in., *Frame Selection Methods to Streamline Surgical Video Annotation for Tool Detection Tasks*, CCECE 2024; tekst/metoda w PDF autora [S1] | Selekcja klatek zaćmy na embeddingach ViT/MAE, PCA i affinity propagation; porównanie z selekcją czasową, ocena kosztu korekt anotacji | Bardzo bliski poprzednik zastosowania. Porównać założenia, reprezentację, budżet i rodzaj celu: anotacja całych narzędzi a lokalizacja końcówek. Nie przypisywać sobie pierwszego użycia klastrowania do tego rodzaju filmów. |
| Wu i in., COWAL, WACV 2024; lokalny tekst i pełna lektura przez Undermind [S2] | Iteracyjne AL segmentacji: klastry uwzględniające już wybrane klatki, następnie wysoka niepewność wewnątrz klastra | Dowód znaczenia redundancji i porównania temporal coverage. Adaptacja do detekcji wymaga innego score; nie nazywać jej reprodukcją oryginalnej segmentacji. Autorzy uczą embedding także na obrazach validation; tego ustawienia nie kopiować do projektowanego tutaj protokołu train-only. |
| Lee i in., CSOD, CVPRW 2024; lokalny tekst i pełna lektura przez Undermind [S3] | Reprezentacje ROI z GT, agregacja w obrębie obrazu i klasy, reprezentatywność/różnorodność, transfer na inne detektory | Obowiązkowy mocny kandydat dla kuracji już anotowanej puli. Dla jednej klasy nadal ma sens; trzeba jawnie obsłużyć klatki bez obiektu. Sam globalny CLS nie zastępuje obiektowych reprezentacji. |
| Popp i in., OFDS, preprint 2024; lokalne części metody, eksperymentów i ograniczeń [S4] | Jednorazowy wybór przed anotacją, cechy obiektowe i pokrycie klas z foundation models | Właściwy konkurent, jeśli teza dotyczy oszczędności anotacji. Nie korzysta z pełnego GT jak CSOD. W badanym ustawieniu utrzymuje jednakową liczbę kroków treningu podzbiorów. |
| Kang i in., AdaDeDup, preprint 2025; lokalny tekst i pełna lektura przez Undermind [S5] | Deduplikacja w klastrach plus feedback z loss proxy i adaptacyjny budżet klastra | Połączenie geometrii i trudności w detekcji już istnieje. Badane treningi utrzymują liczbę epok, więc kroki maleją z pulą. Raportować także koszt proxy i selekcji. |
| Yagi, *Extending Dataset Pruning to Object Detection: A Variance-based Approach*, preprint 2025 [S6] | Przypisanie score obiektom, IoU/confidence, wariancja i agregacja do obrazu | Punkt odniesienia dla detekcyjnego difficulty. Sama klasyfikacyjna norma EL2N nie rozwiązuje problemu lokalizacji i wielu obiektów. |
| Paul i in., EL2N, 2021; abstrakt, eksperymenty i dyskusja [S7] | Wczesne score uśredniane po inicjalizacjach, wpływ trudności i odstających przykładów | Uzasadnienie proxy, nie gwarancja transferu do końcówek. Najtrudniejsze klatki mogą być błędnie anotowane. Nie kopiować nazwy EL2N do innego wzoru. |
| D² Pruning oraz Coverage-centric Coreset Selection [S8–S9] | Równoważenie trudności i pokrycia; problemy z wyborem wyłącznie według wysokiego score | „Diversity + difficulty” nie jest samodzielną nowością. Pełny DESF musi wygrać z prostszymi połączeniami przy tym samym K. |
| Gómez-Zamanillo i in., 2025; lokalny materiał [S10] | DINOv2 + k-means dla selekcji danych rolniczych | Prosty DINO-only clustering jest pełnoprawnym baseline'em, nie pustą kontrolą. Inna dziedzina ogranicza bezpośredni transfer wyników. |
| StepAL oraz *Parsimonious Dataset Construction…* [S11–S12] | Dobór całych filmów do rozpoznawania etapów albo klatek do segmentacji | Uzasadniają pytanie o jednostkę selekcji. Nie są gotowymi detektorami tooltip ani bezpośrednio porównywalnymi wynikami AP. |
| *Temporal Coherence for Active Learning in Videos* i *Efficient video annotation with visual interpolation and frame selection guidance* [S13–S14] | Temporalne sygnały FP/FN oraz wspólny problem doboru klatek i propagacji bboxów | Sama propagacja lub „temporal novelty” też ma poprzedników. Potrzebna kontrola błędów i rzeczywistego czasu pracy anotatora. |

Lokalny FAST dotyczy między innymi dopasowania rozkładów przez funkcje charakterystyczne, a nie po prostu ostrości obrazu z FFT. Swift Sampling dotyczy doboru klatek m.in. do rozumienia długich filmów. Podobne słowa „frequency” i „temporal” nie dowodzą równoważności z blokiem Fourier ani skuteczności w treningu RT-DETR. Materiały te są kierunkami poszerzenia przeglądu, bez przesądzania o nowości i bez przenoszenia ich wyników liczbowych.

## 4. Ocena bieżącego pipeline'u: co jest poprawne, co blokuje badanie

Uzasadniony szkielet to: jawna pula kandydatów → cechy train → selekcja reprezentatywnych źródeł → identyczne przetwarzanie treningowe → ocena na odłożonych grupach. Trzeba jednak rozdzielić **historyczny selektor z artykułu**, **jego wierną kontrolę na nowym poprawnym splicie** i **nową metodę ulepszoną dla RT-DETR**. Wynik ulepszonej wersji nie wyjaśnia automatycznie mechanizmu wersji historycznej.

| Priorytet | Ustalenie i dowód lokalny | Działanie przed macierzą |
|---|---|---|
| P0 | TEX opisuje losowy split po klatkach; dotychczasowy checkpoint nie staje się niezależny przez podpięcie nowego splitu | Wszystkie podstawowe porównania trenować od wspólnych wag ogólnych, po nowym podziale grupowym. Zweryfikować pochodzenie każdego checkpointu proxy i target. |
| P0 | `selection/materialize.py:build_coco` oraz `features/el2n_from_proxy.py:score_real` czytają pojedyncze `bbox` | Kontrakt listy obiektów, klas, ignore i widoczności; testy 0/1/2 obiekty i pełność anotacji. Jedna klasa nie oznacza jednego obiektu. |
| P0 | `features/build_cache.py`: Fourier ma 1 wymiar, FastSAM 9; TEX deklaruje Fourier9/SAM1 | Wersjonować F9/S1 jako kontrolę starej metody; F1/S9 oznaczyć nową wersją. Nie mieszać ich w jednej macierzy pod nazwą tego samego DESF. |
| P0 | W `DinoExtractor` usuwanie cyfr zmienia `dinov2`/`dinov3` w `dinov`, którego nie ma w słowniku repozytoriów; uruchamia się fallback DINOv2. Brak też deklaracji `dim`/`DIM`, choć `main` jej wymaga | Poprawić wybór repo/modelu i jawny wymiar wyjścia; pin commita i wag, rzeczywisty smoke test. FakeDino nie sprawdza tego kontraktu. |
| P0 | `cluster_kcenter_greedy` zwraca etykiety; `select_one` potem wybiera centroid-nearest z tych grup | Zwracać rzeczywiste greedy centra. Obecnego wariantu nie raportować jako wiernego k-center. Sprawdzić powtórzone punkty, remisy i rozmiar wyniku. |
| P0 | Aktualny YAML i trening dotyczą RF-DETR, nie RT-DETR; dokumenty nie są w pełni wdrożone | Oddzielny adapter RT-DETRv2, bez zmiany samej nazwy. Zablokować biegi przy sprzeczności konfiguracji/metadanych. |
| P1 | Difficulty: próg inferencji, dopasowanie po IoU, `1-confidence`, negatywy domyślnie 1; proxy zna tylko 2500 klatek | Zdefiniować detekcyjny score, pusty GT, brak dopasowania, FP, agregację wielu obiektów i ekspozycję proxy. Nie przedstawiać obecnego kodu jako standardowego EL2N. |
| P1 | Standaryzacja przed PCA i mnożenie bloków przez wagi nie oznacza zadanych udziałów bloków w odległości | Zapisać kolejność transformacji, parametry PCA/scalera i rzeczywisty udział bloków; osobno porównać normalizację energii bloków. |
| P1 | FastSAM `coverage` sumuje pola masek, obcina do pola obrazu; mianownik może pochodzić z innej rozdzielczości | Coverage liczyć przez union masek na spójnej siatce. Zdefiniować granice i skalę; składowe FastSAM walidować osobno. |
| P1 | Fourier sumuje `log1p(abs(FFT))`; to nie jest energia `abs(FFT)^2` | Nazwać miernik zgodnie z implementacją lub zmienić wzór i wersję. Ostrość całej sceny nie dowodzi widoczności końcówki. |
| P1 | Puste klastry uzupełniane pierwszymi `frame_id` | Logować ich udział; przetestować seeded random uzupełnianie z ograniczeniami pokrycia. Obecne uzupełnianie może preferować początek uporządkowanych filmów. |
| P1 | Wznowienie cech oparte na `frame_id` nie wystarcza przy zmianie obrazu/modelu/preprocessingu | Fingerprint obrazu, źródła, ekstraktora, wag, konfiguracji i schematu. Odrzucać niezgodny cache zamiast łączyć wersje. |
| P1 | Evaluator CPU poprawiony, lecz ma opisane ograniczenia [21](21_STAN_REFAKTORU.md) | Przed badaniem rozstrzygnąć AP bez GT jako NA, checkpoint↔próg, crowd/ignore i dokładność siatki progów; porównać rzeczywiste predykcje z referencyjnym evaluatorem. |

Tabela opisuje kod z tej kopii roboczej, łącznie z plikami nieśledzonymi w Git. Nie oznacza, że wszystkie wymienione problemy wystąpiły w historycznych treningach. Nie należy także traktować poprawionej ewaluacji CPU jako dowodu poprawności treningu na GPU.

### Dlaczego nominalne wagi nie są udziałami cech

Dla sklejania bloków `z_g` przez `w_g` kwadrat odległości wynosi `d²(i,j)=Σ_g w_g² ||z_g(i)-z_g(j)||²`. Wymiar i wariancja bloku mają znaczenie. Przy niezależnych, standaryzowanych współrzędnych oczekiwana skala wkładu jest proporcjonalna do `w_g² d_g`; po PCA zależy od zachowanej wariancji. To prosta własność algebraiczna, nie wynik treningu. Waga 0,35 nie znaczy 35% wpływu DINO.

Propozycja osobnej poprawki: normalizować każdy blok przez pierwiastek jego empirycznej średniej kwadratowej normy na train, a następnie mnożyć przez `sqrt(alpha_g)`. Wtedy `alpha_g` określa nominalny udział energii bloku w metryce; nadal nie mierzy przyczynowego wpływu na AP. Bloki stałe wyłączyć jawnie. W kontrolnej macierzy nie zmieniać równocześnie tej normalizacji i zawartości F/S.

## 5. Docelowy przepływ danych wideo

1. **Pochodzenie i niezależność.** Manifest: `patient/group_id`, `surgery_id`, `video_id`, hash pliku źródłowego, czas PTS, indeks klatki, źródło/licencja, rozdzielczość, FPS, `source_frame_id`. Przy braku wiarygodnego patient ID stosować konserwatywne grupy, opisać ograniczenie i nie nazywać grup pacjentami. Powiązane wycinki tej samej operacji pozostają razem.
2. **Zamrożenie podziału.** Split grupowy przed selekcją, fittingiem PCA, proxy i augmentacją. Ekstrakcja identycznym, z góry ustalonym ffmpeg nie uczy parametrów i może poprzedzać split, jeśli pochodzenie jest zachowane; żadne adaptacyjne decyzje o jakości/odrzucaniu nie mogą korzystać z testu. Istniejący 35/8/7 jest kandydatem, nie dowodem odpowiedniej mocy.
3. **Pula źródeł.** K oznacza unikalne klatki źródłowe. Deduplikacja ścisła według pochodzenia/hash jest wspólna wszystkim metodom. Usuwanie semantycznej redundancji pozostaje badanym selektorem, nie ukrytym filtrem tylko jednej metody. Nie wycinać wcześniej całej puli do 1 FPS, jeśli chcemy później wykazać wartość selekcji czasowej na gęstym wideo.
4. **Geometria i jakość.** Zachować źródła; crop pola widzenia, resize/letterbox, padding i inverse transform bbox mają jawny zapis. Uszkodzony plik można wykluczyć wspólną regułą; blur, glare i zasłonięcie są często prawdziwymi warunkami zadania, nie automatycznym „śmieciem”. Odrzucenie końcówki poza kadrem i oznaczenie niewidocznego obiektu muszą odpowiadać instrukcji anotacji.
5. **Reference standard.** Rozróżnić całą końcówkę, punkt końca i całe narzędzie. Maska/box całego instrumentu w zbiorze zewnętrznym nie jest etykietą tooltip. Brak anotacji nie oznacza pewnego tła; zapewnić kompletność na ocenianych klatkach. Walidację/test kontrolować ręcznie według jednej instrukcji i z rozstrzyganiem niezgodności, niezależnie od predykcji ocenianego modelu.
6. **Cechy i selekcja wyłącznie z train.** Fit scaler/PCA oraz wszelkie adaptacje encoderów na train; dla wewnętrznego CV przeliczyć fit i selekcję w każdym foldzie. Gotowe ogólne wagi mogą być wspólne, ale ich pochodzenie musi być zapisane. Ten sam ustalony selektor generuje te same manifesty dla badania transferu między targetami.
7. **Augmentacja po wyborze źródeł.** Jedna recepta i prawidłowe transformacje wszystkich bboxów wewnątrz architektury. Nie zwiększać K przez kopie augmentowane. Val/test bez losowej augmentacji; TTA wyłącznie jako osobny wcześniej opisany wariant.
8. **Test z realistycznym czasem.** Pełne odłożone filmy lub z góry wybrane gęste odcinki z niezależnych operacji. Testu nie kuratorować proponowanym selektorem. Rzadszy, stały grid do COCO AP można raportować jako dodatkowy protokół; do luk i migotania potrzeba gęstych sekwencji. Zadanie offline może korzystać z przyszłych klatek przy selekcji train; twierdzenie o selekcji online wymaga osobnego wariantu przyczynowego bez dostępu do przyszłości.

Szczególnie ważne: tracking i segmentation w Cataract-LMM nie są automatycznie niezależnymi pacjentami. Dopóki nie ma mapowania, nie stosować jednej części jako train, a drugiej jako „niezależnego external test”. Benchmark oglądany wielokrotnie pozostaje benchmarkiem rozwojowym/historycznym; nowy ledger nie przywraca mu statusu nietkniętego testu.

## 6. Jak ulepszyć metodę bez rozdmuchiwania zakresu

**Najpierw wierna kontrola DESF, potem trzy rozdzielone ulepszenia.** Nie dodawać od razu kolejnych foundation models, drzew, SHAP i Koopmana. Każda taka składowa zwiększa koszt, liczbę decyzji i ciężar dowodowy.

| Ulepszenie | Konkretna wersja do testu | Kontrast rozstrzygający |
|---|---|---|
| Reprezentacja obiektu i kontekstu | Globalny embedding plus ROI końcówki/otoczenia; dla anotowanej train-pool ROI z GT, wszystkie obiekty | Global-only vs ROI-only vs global+ROI; ten sam encoder i budżet. Kontrola z CSOD. GT-ROI nie uzasadnia oszczędności anotacji. |
| Redundancja czasowa i pokrycie | Rozdzielić liczbę klastrów `C` od liczby klatek `K`; budżety na klastry/filmy, kontrolowany odstęp czasowy z wyjątkami dla wejścia/wyjścia i szybkich zmian | Uniform temporal i patient-stratified random vs ten sam selektor bez ograniczeń vs z ograniczeniami. Ustalić obsługę niedoboru kandydatów i zachować dokładnie K. |
| Trudność powiązana z detekcją | Score oparty na natywnym loss lub ustalonej adaptacji VPS; zachować osobno błąd klasy/obiektu, lokalizacji i FP na tle | Centroid vs umiarkowana trudność wewnątrz klastra vs najwyższa trudność; raportować jakość etykiet i negatywy. Nie zakładać, że hardest oznacza najbardziej wartościowy. |

Przykładowa nowa miara difficulty dla RT-DETR może agregować klasyfikacyjny loss dopasowanych queries oraz L1/GIoU dla GT, z oddzielnym składnikiem tła. Trzeba ustalić normalizację po obiektach/queries, przypadek pustego GT, wybór warstwy decoder i obsługę auxiliary/denoising losses. Loss liczony w batchu może mieć normalizację wspólną: **sprawdzić, że score pojedynczego obrazu nie zależy przypadkowo od liczby GT w innych obrazach batcha**. Nie wolno po prostu uznać gotowego sumarycznego `loss` za porównywalny score każdej klatki. To propozycja nowej adaptacji, nie gotowa implementacja oryginalnego EL2N.

Proxy referencyjnie może uczyć się krótko na całej train-pool przy wyrównanej ekspozycji. Scoring tych samych przykładów nie jest sam w sobie wyciekiem. Wariant cross-fit po grupach jest kontrolą kosztowniejszą i bada inną wielkość. Jeśli proxy widzi tylko podzbiór, zapisać `seen_by_proxy` i zmierzyć wpływ ekspozycji. Uśrednianie score kilku proxy może poprawiać stabilność, ale jego koszt jest częścią selekcji.

W przypadku celu „mniej ręcznej anotacji” potrzebna jest **osobna gałąź przed anotacją**: propozycje ROI z zamrożonego modelu albo metoda w rodzaju OFDS, wspólny seed labeled set i niewykorzystywanie ukrytych etykiet niewybranych obrazów. Kuracja już opisanej puli z GT i proxy nie dowodzi oszczędności jej wcześniejszego opisania.

## 7. Hipotezy i zasady falsyfikacji

| Hipoteza | Pierwotny kontrast | Co osłabia lub odrzuca tezę |
|---|---|---|
| H1: DESF poprawia jakość wyboru przy ustalonym K | Różnica COCO AP@[.50:.95] DESF minus random-source na odłożonych grupach, przy tej samej liczbie kroków i recepturze | Efekt niepewny albo równy prostemu temporal/embedding baseline. Sam dodatni wynik względem jednego losowania nie wystarcza. |
| H2: bloki wnoszą uzupełniającą informację | Zaplanowane kontrasty pełnej macierzy, szczególnie DESF vs D, DE, DSF i DEF | Pełna metoda nie przewyższa prostszego wariantu; wniosek powinien wtedy dotyczyć tego prostszego wariantu. |
| H3: kuracja oszczędza całkowity koszt przy akceptowalnej jakości | Koszt selekcji + koszt targetu vs full-pool, z ustalonym wcześniej marginesem tolerowanej utraty AP | Koszt proxy/cech przewyższa oszczędność albo przedział różnicy AP nie mieści się w kryterium non-inferiority. Brak istotnej różnicy nie dowodzi równoważności. |
| H4: wybrany podzbiór jest użyteczny poza jednym targetem | Te same manifesty kuracji na RT-DETRv2 i drugim detektorze; niezależny zbiór/operacje | Zysk tylko dla modelu, który wyprodukował difficulty, lub tylko na historycznie oglądanym external. |

Primary: H1 na jednym z góry wybranym budżecie, np. 25% dostępnych train-source. H2–H4 wymagają oddzielnych kontrastów i raportowania niepewności. Margines np. **1 punkt AP** może być roboczym założeniem planistycznym dla H3, lecz nie jest normą kliniczną i wymaga uzasadnienia przed testem. Wynik „hipotezy nie potwierdzono” jest dopuszczalny. Nie planować z góry wartości p ani liczby razy poprawy.

## 8. Ablacje i kolejność treningów

### Etap A: poprawność i pilotaż, bez zaglądania do testu

Sprawdzić kontrakty P0, rzeczywisty forward/backward RT-DETR, mapowanie klas i geometrii oraz mały overfit test na train. Wyświetlić przykłady 0/1/2 obiekty z predykcjami po odwróceniu transformacji. Zamrozić wersję referencyjnego evaluatora. Następnie random-source, DESF i full-pool, po 3 sparowane powtórzenia na RT-DETR, oceniane na danych rozwojowych: **9 treningów pilotażowych**. To sprawdzenie wykonalności i wariancji, nie „naukowy dowód z trzech seedów”. Jeśli recepta się zmieni, starych biegów nie zaliczać do finalnej macierzy.

Pilotaż służy do ustalenia LR, batcha, liczby kroków i rozdzielczości oraz oszacowania wariancji. Przy podejrzanym załamaniu random sprawdzić loader, bboxy, loss i checkpoint przed wyjaśnianiem wyniku „synergią”. Kryteria awarii: brak poprawnych etykiet, NaN/Inf, niezgodny manifest, przerwany harmonogram; niskie AP samo nie wystarcza do usunięcia poprawnie wykonanego biegu.

### Etap B: pełna analiza czterech bloków przy jednym K

Zamrozić cztery definicje: D — konkretny encoder i PCA, E — konkretny difficulty, S — ustalony score FastSAM, F — ustalony deskryptor Fourier. Ta macierz dotyczy jednej wersji metody. **Nie zmieniać między wierszami F9 na F1, S1 na S9 ani wag dostrojonych osobno na każdym teście.**

| Poziom | Warianty |
|---|---|
| Pojedyncze | D, E, S, F |
| Pary | DE, DS, DF, ES, EF, SF |
| Trójki | DES, DEF, DSF, ESF |
| Pełny | DESF |
| Kontrole | Random-source, uniform temporal, prawdziwy k-center na D, CSOD, AdaDeDup, full-pool |

D-only k-means jest już jednym z 15 wierszy. Łącznie powyżej **21 warunków × 2 architektury × 3 powtórzenia = 126 treningów**. To przykładowa pierwsza macierz, bez proxy i tuningów, nie ustalona moc statystyczna. Full-pool ma N zamiast K, więc jest kontekstem, nie równobudżetowym konkurentem. Jeśli po pilotażu potrzebne jest 5 powtórzeń, odpowiednio rośnie koszt. Kosztowne baseline'y trzeba implementować wiernie lub jawnie nazwać adaptacją.

Uniform temporal: K indeksów rozłożonych po czasie w każdym train-video, z budżetem filmu proporcjonalnym do liczby kandydatów i kontrolowanym offsetem; bez użycia wyniku detektora. Osobna kontrola wyrównująca udział filmów/pacjentów pozwala sprawdzić, czy korzyść pochodzi tylko z ich reprezentacji. Jeżeli nowa metoda narzuca kwoty grupowe lub negatywów, baseline'y porównawcze powinny mieć także wariant z tymi samymi kwotami.

Dla wiernej odpowiedzi na A7 potrzebna jest macierz także dla **YOLOv8**. Proponowana macierz RT-DETRv2 + współczesny YOLO zmienia zakres pracy i nie zamyka dosłownie tego punktu. Możliwości: dodać kontrolną macierz YOLOv8 (15 bloków + random + full, 3 powtórzenia = 51 dodatkowych treningów) albo wyraźnie przenieść główną tezę na nowe modele i uzasadnić odejście od dawnego porównania. Nie udawać, że trzy wiersze transferu są pełną ablacją YOLOv8.

Nie traktować tej macierzy jako automatycznego pełnego eksperymentu `2^4`: wariant bez cech nie definiuje tego samego selektora, a random jest zewnętrzną kontrolą. Da się badać wcześniej określone interakcje warunkowe, np. `I(F,E | D,S) = AP_DESF − AP_DSF − AP_DES + AP_DS`. Pozostałe parametry i relacje wag muszą być stałe. To interakcja operacji dodania bloków w tej metryce, nie dowód biologicznej przyczynowości ani uniwersalnej synergii.

### Etap C: krzywe budżetu i całkowitego kosztu

Budżety: `K=floor(p*N_train_source)`, np. p=10%, 25%, 50%, 75%, a 100% jako full-pool. Przy wcześniej stwierdzonym N=17 832 daje to 1783, 4458, 8916, 13 374 i 17 832. **N należy ponownie policzyć po zamrożeniu manifestu**; 20 000 unikalnych źródeł z tej puli jest niemożliwe. K=5000 może zostać dodatkowym punktem dla ciągłości z dawnym planem.

Krzywe dla random, temporal, D-only, DESF oraz mocnego konkurenta wybranego na development. Zachować też pierwotnie zdefiniowany DESF, jeśli development wybierze jego uproszczenie. Pokazać AP vs K i AP vs całkowity koszt; przy podsumowaniu polem pod krzywą podać oś, zakres i interpolację. 100% to ta sama pula dla wszystkich metod i nie wymaga osobnego liczenia cech/selekcji dla każdej.

Dwa oddzielne reżimy:

- **Stałe kroki:** dla wszystkich K ten sam limit optimizer updates, effective batch, harmonogram LR wyrażony krokami, checkpoint selection i augmentacje. To izoluje jakość podzbioru przy przybliżonej równości treningowego compute; mniejsza pula jest częściej powtarzana. Nie dowodzi oszczędności czasu treningu.
- **Stałe epoki:** pełna liczba przejść przez każde K, harmonogram jako udział całego biegu. Mniejsza pula oznacza mniej kroków. To bada praktyczną receptę oszczędzania; różnica jakości zawiera także wpływ długości optymalizacji. Osobno można pokazać time-to-target-AP na val, z wcześniej ustalonym celem i jawnym cenzurowaniem biegów, które go nie osiągnęły.

`C_total = C_decode + C_features + C_proxy_train + C_proxy_score + C_selection + C_target_train`. Rozdzielić CPU/GPU, wall-clock, model GPU, liczbę urządzeń, storage oraz rzeczywisty narzut. Przy wielokrotnym użyciu podzbioru pokazać zarówno koszt pierwszego użycia, jak i amortyzowany. Próg opłacalności dla R treningów: `R*(C_full_train − C_subset_train) > C_selection_overhead`, o ile różnica po prawej i po lewej jest mierzona na porównywalnym sprzęcie. Koszt strojenia badawczego raportować oddzielnie i nie ukrywać go jako „jednorazowego darmowego preprocessing”.

### Etap D: wrażliwość, propagacja i transfer

Wrażliwość na development: PCA 10/16/20/32/64, normalizacja bloków, wagi, centroid/medoid/random-in-cluster, składowe FastSAM, proxy exposure i epoka scoringu. Screening tanimi statystykami służy ograniczeniu siatki; końcowe wnioski o użyteczności wymagają downstream AP. Jeden seed może służyć screeningowi, lecz nie wystarcza do deklaracji stabilności. Mierzyć overlap wybranych źródeł, rozkład rozmiarów i liczby obiektów, negatywy, pokrycie grup, rozmycie i odstępy czasowe. Korelacja rankingów lub retained variance PCA nie zastępuje wyniku targetu.

Propagację anotacji zbadać osobno: na wspólnej puli źródeł z ręcznie sprawdzonym odniesieniem, `2 selektory (random/DESF) × 2 wersje etykiet (referencyjne/propagowane)`. Najpierw zamrozić dwa zestawy wybranych klatek, potem zmieniać etykiety używane do treningu, żeby izolować ich wpływ. Jeżeli zmiana etykiet wpływa także na E/ROI i wybór klatek, nazwać to osobnym testem całego procesu. Zachować stałe K, grupy, model i kroki. Raportować brakujące/driftujące bboxy, straty po okluzji, liczbę poprawek oraz zmierzony czas anotacji, nie tylko liczbę kliknięć. Ocena testowa używa wyłącznie niezależnego reference standard.

Transfer: wykorzystać te same wybrane źródła i ten sam cache selektora na obu targetach. Osobno zbadać zmianę architektury proxy przy stałym target i targetu przy stałym proxy. RF-DETR-Small w 512 px nie jest automatycznie wierną sondą RT-DETR w 640/wyższej rozdzielczości. Zgodność rang na kilku wariantach jest słabym testem; potrzebne są również różnice AP i koszty. Jeżeli zmiana rozdzielczości zmienia kolejność, raportować interakcję zamiast zakładać transfer.

## 9. Konkretna recepta treningowa RT-DETR

Rekomendowany punkt startowy to **RT-DETRv2-S, oficjalny kod PyTorch**, zamrożony commit i konkretne ogólne wagi COCO [S15]. To świadomy wybór reprodukowalnego baseline'u, nie deklaracja najnowszej architektury. Oficjalne repo wskazuje także RT-DETRv4; nie trzeba testować każdej wersji. RF-DETR jest odrębną biblioteką i pozostaje osobnym targetem.

1. Przygotować COCO z wszystkimi GT obiektami, również `images` bez obiektów. Sprawdzić identyfikator klasy i opcję remap kategorii; nie zakładać mapowania 80 klas COCO dla danych jednoklasowych. Sprawdzić, że head odpowiada liczbie klas i poprawnie załadowano pozostałe wagi.
2. Start każdego wiersza z tej samej ogólnej inicjalizacji. Proxy nie jest checkpointem inicjalizującym target. Dotyczy to także full-pool. Jeżeli badamy fine-tuning, każdy selektor, także random, dostaje taki sam model wstępny i taki sam dodatkowy budżet; jego wcześniejszy trening wchodzi do kosztu.
3. Rozdzielczość początkowa 640 jako ustawienie referencyjne. Na train/development porównać ją z większą, dopuszczaną przez wybraną implementację, np. 768. Rozkład pixel-area końcówek i AP75 pomagają wybrać rozdzielczość. Nie podejmować tej decyzji na external. Recepta po wyborze jest identyczna w obrębie architektury.
4. Skorzystać z oficjalnej recepty jako punktu startowego, kalibrując LR i effective batch na development. Rejestrować AdamW/param groups, LR backbone/head, warmup, scheduler, clipping, mixed precision, gradient accumulation, EMA, dynamic augmentation i liczbę queries. Nie narzucać LR z RF-DETR-Small na podstawie samej nazwy „DETR”.
5. Budżet kroków i harmonogram zmian augmentacji zapisać w manifestach. Ta sama liczba epok przy zmiennym K oznacza inny czas treningu. W głównej macierzy bez zmiany limitu na podstawie testu. Best-val-AP i final checkpoint są oddzielnymi kolumnami; nie mieszać EMA/bare-model między wierszami.
6. Inferencja standardowym postprocessorem i konfiguracją top-k wybranej implementacji. Liczba queries nie wynika z liczby klas. Bez Query 81 i bez dodatkowego NMS w bazowym RT-DETR. Hungarian dotyczy treningowego przypisania; COCO AP ma własny ranking i matching.
7. Dla drugiego targetu można wybrać współczesny YOLO dostępny w oficjalnej bibliotece. Na dzień audytu dokumentacja pokazuje YOLO26 [S16]. Zamrozić konkretną skalę, wersję i wariant postprocessingu; **nie przenosić założenia „każdy YOLO wymaga NMS” z YOLOv8 na wszystkie nowe modele**. NMS/Soft-NMS jest osobną analizą tam, gdzie dany model rzeczywiście go stosuje.

Każdy bieg: `run_id`, commit kodu, hash konfiguracji i manifestu źródeł, lista train/val/test groups, checkpoint inicjalizacji i jego hash, wersje bibliotek/CUDA, trzy rodzaje seedów, effective batch, liczba kroków, harmonogram augmentacji, czas, rozdzielczość, checkpoint końcowy/wybrany/EMA, zapis wszystkich predykcji oraz wersja evaluatora.

## 10. Metryki, niepewność i uczciwa interpretacja

### Detekcja, lokalizacja i czas

- Główna metryka: **COCO AP@[.50:.95]**, wspólnie z AP50, AP75 i AR100. Wszystkie klatki ustalonego testu, także bez detekcji. Podawać `maxDets`, cutoff eksportu i wersję pycocotools. Dla jednej klasy „mAP po klasach” jest po prostu AP tej klasy.
- P/R/F1 przy progu wybranym na val; dokładnie ten próg zastosować na internal i external. AP z możliwie pełnego rankingu; na val sprawdzić stabilność AP przy obniżaniu cutoff, zamiast zakładać, że 0,001 zawsze wystarcza. Nie wybierać testowego maksimum F1.
- AP pooled po wszystkich obrazach oraz osobny wynik per-patient/per-surgery i średnia makro po tych grupach. Makro-AP nie jest oficjalnym pooled COCO AP; obie agregacje odpowiadają na inne pytania. Jeśli brak GT, AP jest NA, a FP na tle nadal należy raportować.
- Podgrupy wcześniej zdefiniowane: małe końcówki, okluzja, glare, blur, obecność wielu obiektów, wejście/wyjście z kadru i etap zabiegu, jeżeli anotowany. Podawać liczebności. COCO small opiera się na polu <32² w koordynatach ocenianego GT, nie na warunku obu boków <32. Dodatkowo opisać faktyczny rozmiar obiektu na wejściu sieci.
- Lokalizacja: błąd punktu końca tylko przy niezależnych anotacjach punktowych; środek bbox nie jest automatycznie końcówką. Raportować błędy trafionych obiektów razem z miss rate, żeby nie ukrywać niewykryć. Milimetry wymagają kalibracji obrazu i uzasadnienia geometrii.
- Czas na gęstych testowych odcinkach: długości nieprzerwanych luk detekcji widocznego obiektu, np. mediana/95. percentyl w sekundach; błędne alarmy na czas; odsetek poprawnych detekcji przy wejściu obiektu. Dla FP odróżnić liczbę boxów od liczby epizodów, bo zliczanie na każdej klatce zależy od FPS. Jitter liczyć jako błąd ruchu predykcji względem ruchu referencji na kolejnych dopasowanych klatkach, razem z coverage. Nie karać prawdziwego ruchu narzędzia za samą zmianę położenia.
- HOTA/IDF1 wymagają zadania tracking i tożsamości obiektów; nie dopisywać ich do detekcji klatek bez takich danych. Pomiar latency/FPS: jeden ustalony GPU/runtime, batch=1 dla scenariusza real-time, warmup, synchronizacja, p50/p95 i rozdzielenie model-only od end-to-end.

Dobór metryk opiera się na Metrics Reloaded i pracy o pułapkach walidacji, a powyższe definicje czasowe są propozycją dla tego zadania [S17–S18]. Nowszy preprint Reinke i in. (v2, 21.04.2026) bezpośrednio opisuje ryzyko pomijania hierarchii i czasu w chirurgicznym AI [S19]. Nie oznacza to, że samo dodanie „temporal consistency” dowodzi użyteczności klinicznej. Praca Ozbulak i in. wskazuje również na wpływ gęstości próbkowania na ocenę SAM2; jest ostrzeżeniem metodologicznym, a nie wynikiem dla RT-DETR [S20].

### Powtórzenia i analiza statystyczna

Oddzielić `selection_seed`, `clustering_seed/proxy_seed` i `train_seed`. Dla porównań metod używać bloków ze wspólną inicjalizacją targetu, tym samym splitem i budżetem. Jeśli wszystkie losowości zmieniamy razem, SD oznacza zmienność całej procedury, nie wyłącznie treningu. Dla kluczowego random/DESF można skrzyżować 3 niezależne selekcje z 3 seedami treningu: 9 wyników na metodę, ale nie 9 niezależnych pacjentów ani 9 niezależnych losowań podzbioru. Liczbę powtórzeń ustalić na podstawie zmienności z pilotażu i najmniejszego interesującego efektu [S21–S22].

Primary contrast: sparowana różnica AP metod. **Nie wykonywać t-testu po tysiącach klatek.** Proponowany warunkowy CI po testowych grupach: losować z powtórzeniami pacjentów/operacje, wziąć wszystkie ich ustalone klatki i przeliczyć AP dla obu metod na identycznym resamplingu. Dla ponownie wylosowanego pacjenta nadać nowe identyfikatory obrazów/GT i skopiować także predykcje; sama lista powtórzonych ID zostałaby zdeduplikowana przez COCO. Średnią sparowaną różnicę po ustalonych biegach liczyć wewnątrz każdej repliki. Tak otrzymany CI opisuje niepewność testowej populacji warunkowo względem tych treningów; SD biegów pokazać oddzielnie. Łączny przedział wymaga zaprojektowanego resamplingu grup i bloków treningowo-selekcyjnych, bez traktowania ich jako niezależnych klatek.

Przy siedmiu testowych operacjach bootstrap będzie niestabilny i nie obejmuje niezaobserwowanej różnorodności ośrodków. Pokazać wyniki każdej operacji i analizę leave-one-group-out jako wrażliwość, nie jako nowe niezależne powtórzenia. Jeśli dostępne są dodatkowe niezależne operacje, zwiększenie ich liczby może być bardziej wartościowe niż dokładanie seedów. Alternatywą przy małej próbie jest zaplanowany zagnieżdżony group CV, w którym także dobór selektora i preprocessing są ponawiane wewnątrz foldów; nie używać testowych foldów do ciągłego projektowania nowej metody.

Zarejestrować kontrast primary, rodzinę porównań wtórnych i ewentualną korektę Holm. Podawać efekt w punktach AP i CI; wartość p jest uzupełnieniem. Dla H3 kryterium non-inferiority to np. dolna granica wcześniej ustalonego przedziału dla `AP_subset−AP_full` powyżej `−delta` plus rzeczywista redukcja kosztu. Nie uznawać `p>0,05` za dowód równoważności. Kalibrację score badać osobno: reliability diagram / detekcyjne ECE z definicją TP i binów, kalibrator fit na danych rozwojowych; dobra kalibracja nie zastępuje recall.

## 11. Jak zamknąć uwagi recenzentów

Oznaczenia: **A = Comments_to_the_Author.pdf**, **B = Reviewer_Comments_to_Authors.pdf**. Nie jest to oficjalna numeracja recenzentów. Pełna mapa 30 uwag pozostaje w [dokumencie źródłowym](notatki/MAPA_UWAG_RECENZENTOW.md).

| Uwagi | Nowy dowód do artykułu |
|---|---|
| A2/A8/A15, B1/B10 | Diagram pochodzenia source→selected→augmented, liczby grup i anotacji, grupowy split, historia dostępu do external i nowych checkpointów. |
| A1/A5/A13/A14, B3/B4 | Macierz 15 kombinacji, jawne definicje bloków, składowe FastSAM, normalizacja i czułość PCA/wag; alternatywne selektory. |
| A6/A9, B2 | Niezależne losowania source, powtórzenia treningu, jednostkowe wyniki, SD i poprawny CI grupowy. |
| A3/A7/A10, B7/B8 | RT-DETRv2 + nowy YOLO, kontrola full/random/curated dla każdego, odpowiedź co stało się z żądaną macierzą YOLOv8; koszty i rozdzielczości. |
| A4/A17, B5/B6/B9 | Standardowy postprocessing, wspólne AP50/AP75/AP50:95, progi z val, kalibracja, NMS tam gdzie stosowany, realistyczna lokalizacja i sekwencje błędów. |
| A11/A12/A16, B-drobne | Przykłady wybierane jawną regułą, zgodny schemat i nazwy wierszy, legenda znanych składowych i rzeczywistego wkładu, ograniczenie deklaracji klinicznych. |

Nie kopiować z `RESPONSE_TO_REVIEWERS_DETAILED.md` niepotwierdzonych średnich, SD, p<0,001, analiz wrażliwości ani historii „12 uszkodzonych klatek”. Dopiero istniejący log/manifest/wynik umożliwia zapis „wykonaliśmy”. Nowe wyniki mogą całkowicie zmienić abstrakt, tytuł i kierunek wniosków. W szczególności wcześniejsze „7× gain attributable to curation” należy zastąpić pomiarem z kontrolowanego kontrastu, a historyczne porównanie opisać jako różnicę całych recept.

## 12. Książki i literatura metodologiczna: do czego ich używać

| Książka / źródło | Przydatny zakres | Zastosowanie tutaj i granica lektury |
|---|---|---|
| James, Witten, Hastie, Tibshirani, Taylor, *An Introduction to Statistical Learning with Applications in Python* (2023) [B1] | Resampling, dobór modeli i multiple testing | Rozdzielenie strojenia i testowania; praktyczna baza do planu. Sprawdzono oficjalną stronę i źródło pobrania; nie deklarujemy lektury całej książki w tej sesji. |
| Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, wyd. 2 [B2] | Rozdział 7: Model Assessment and Selection, cross-validation i bootstrap | Ocena całej procedury, nie tylko wybranego zwycięskiego modelu. Oficjalna strona i link do wydania po korektach; przykładowe resampling i.i.d. trzeba dostosować do hierarchii wideo. |
| Montgomery, *Design and Analysis of Experiments*, wyd. 10 [B3] | Rozdz. 4 bloki, 5–7 efekty/interakcje, 13–14 czynniki losowe i układy zagnieżdżone | Uzasadnienie blokowania seedów i kontrastów macierzy; potwierdzono spis treści wydawcy, bez pełnej lektury płatnych rozdziałów. Macierz 15 kombinacji z random nie jest automatycznie regularnym układem `2^4`. |
| Efron i Tibshirani, *An Introduction to the Bootstrap* [B4] | Błędy standardowe, przedziały, bardziej złożone struktury danych | Podstawa metody resampling; sprawdzono metadane wydawcy, nie cały tekst. Konkretny paired cluster bootstrap opisany wyżej jest projektem dla tego badania, nie gotową procedurą COCO z książki. |
| CLAIM 2024 [S23] | Przejrzystość danych, reference standard, trenowania i oceny | Checklist raportowania, nie punktowy certyfikat jakości ani zamiennik badań. Każde „tak” musi mieć stronę/artefakt, „nie dotyczy” uzasadnienie. |

Najpierw wykonać bramy P0 i pilotaż; następnie zamrozić prerejestrację/plan analizy oraz wersję danych; później macierz i zaplanowane kontrole; na końcu jednokrotna finalna ocena zamrożonych modeli oraz przepisanie artykułu. Rzetelny wynik nie musi być spektakularny: stabilny zysk nad silnymi kontrolami lub dobrze udokumentowana możliwość uproszczenia metody będzie wiarygodniejsza niż duży iloraz wobec pojedynczego niemal zerowego baseline'u.

## 13. Źródła i linki

Linki PDF/DOI i strony książek zapisano również przez istniejący zbieracz w kolekcji `linki/audyt-publikacyjny-kuracji-wideo-rt-detr-i-projekt-eksperymentow-2026-09-11.txt`: 25 nowych adresów, 15 pominiętych jako już zapisane. To nie oznacza 25 nowych publikacji ani 25 pobranych PDF-ów. Status wydawniczy podano ostrożnie; data rekordu Undermind może oznaczać preprint albo aktualizację, nie rok konferencji.

- **S1:** Yang i in., 2024, [PDF autora](https://people.scs.carleton.ca/~matthewholden/assets/Yang2024b.pdf), [DOI](https://doi.org/10.1109/CCECE59415.2024.10667104).
- **S2:** Wu i in., COWAL, WACV 2024, [DOI](https://doi.org/10.1109/WACV57701.2024.00201), [preprint](https://arxiv.org/abs/2311.08811).
- **S3:** Lee i in., CSOD, CVPRW 2024, [DOI](https://doi.org/10.1109/CVPRW63382.2024.00764), [preprint](https://arxiv.org/abs/2404.09161).
- **S4:** Popp i in., *Single-Pass Object-Focused Data Selection*, [preprint](https://arxiv.org/abs/2412.10032).
- **S5:** Kang i in., AdaDeDup, [preprint](https://arxiv.org/abs/2507.00049).
- **S6:** Yagi, VPS/detection pruning, [preprint](https://arxiv.org/abs/2505.17245).
- **S7:** Paul i in., *Deep Learning on a Data Diet*, [preprint](https://arxiv.org/abs/2107.07075).
- **S8:** Maharana i in., D² Pruning, [preprint](https://arxiv.org/abs/2310.07931).
- **S9:** Zheng i in., Coverage-centric Coreset Selection, [preprint](https://arxiv.org/abs/2210.15809).
- **S10:** Gómez-Zamanillo i in., DINOv2/k-means, [artykuł](https://doi.org/10.3389/fpls.2025.1546756).
- **S11:** Shah i in., StepAL, [preprint](https://arxiv.org/abs/2507.22059).
- **S12:** Zhou i in., *Parsimonious Dataset Construction…*, [preprint](https://arxiv.org/abs/2504.12573).
- **S13:** *Temporal Coherence for Active Learning in Videos*, [preprint](https://arxiv.org/abs/1908.11757).
- **S14:** *Efficient video annotation with visual interpolation and frame selection guidance*, [preprint](https://arxiv.org/abs/2012.12554).
- **S15:** RT-DETR [oficjalny kod](https://github.com/lyuwenyu/RT-DETR), [artykuł RT-DETR](https://arxiv.org/abs/2304.08069), [RT-DETRv2](https://arxiv.org/abs/2407.17140).
- **S16:** [Oficjalny katalog modeli Ultralytics](https://docs.ultralytics.com/models/).
- **S17:** *Metrics Reloaded*, [tekst lokalnie dostępnej wersji/preprint](https://arxiv.org/abs/2206.01653).
- **S18:** *Understanding metric-related pitfalls in image analysis validation*, [preprint](https://arxiv.org/abs/2302.01790).
- **S19:** Reinke i in., *Current validation practice undermines surgical AI development*, [v2, 21.04.2026](https://arxiv.org/html/2511.03769v2), [kod autorów](https://github.com/IMSY-DKFZ/metrics-reloaded-surgery-pitfalls).
- **S20:** Ozbulak i in., *Revisiting the Evaluation Bias Introduced by Frame Sampling Strategies in Surgical Video Segmentation Using SAM2*, [DOI](https://doi.org/10.1007/978-3-032-05870-6_20); lektura metadanych/abstraktu Undermind, bez pełnego tekstu.
- **S21:** Bouthillier i in., *Accounting for Variance in Machine Learning Benchmarks*, [preprint](https://arxiv.org/abs/2103.03098).
- **S22:** Cawley i Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, [JMLR](https://www.jmlr.org/papers/v11/cawley10a.html).
- **S23:** Tejani i in., *CLAIM: 2024 Update*, [publikacja](https://doi.org/10.1148/ryai.240300).
- **S24:** Varoquaux i Cheplygina, *Machine learning for medical imaging: methodological failures and recommendations for the future*, [wersja czasopismowa 2022](https://www.nature.com/articles/s41746-022-00592-y). Lokalny OCR wcześniejszej wersji ma inny tytuł; cytować wersję finalną.
- **S25:** Fox i Schoeffmann, *The Impact of Dataset Splits on Classification Performance in Medical Videos*, [DOI](https://doi.org/10.1145/3512527.3531424); metadane/abstrakt Undermind, bez pełnego tekstu.
- **S26:** Coleman i in., *Selection via Proxy*, [preprint](https://arxiv.org/abs/1906.11829); lokalna lektura, główne eksperymenty klasyfikacyjne.
- **S27:** Okanovic i in., *Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning*, [preprint](https://arxiv.org/abs/2305.18424); uzasadnienie dodatkowego baseline'u czasowego, jeśli praca utrzymuje tezę o przyspieszeniu.
- **S28:** Ayed i Hayou, *Data pruning and neural scaling laws: fundamental limitations of score-based algorithms*, [preprint](https://arxiv.org/abs/2302.06960).
- **B1:** [ISLP — strona autorów](https://www.statlearning.com/), [oficjalny odsyłacz pobrania](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html).
- **B2:** [ESL — strona autorów](https://hastie.su.domains/ElemStatLearn/), [oficjalne pobieranie](https://hastie.su.domains/ElemStatLearn/download.html).
- **B3:** [Montgomery — spis treści wydawcy](https://bcs.wiley.com/he-bcs/Books?action=contents&bcsId=11447&itemId=1119492440).
- **B4:** [Efron i Tibshirani — wydawca](https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317).

**Rezultat tej sesji:** przegląd i plan; skuteczność pipeline'u pozostaje do zbadania. Następna decyzja implementacyjna to usunięcie P0, wierne wersjonowanie selektora i adapter RT-DETRv2, zanim ruszą drogie cechy i treningi.
