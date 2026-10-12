# Adaptacyjny dobór klatek według brakujących cech

**Stan: kod przygotowany do testów na Edenie. Testów modelu ani treningów nie uruchomiono. Nie ma jeszcze wyniku potwierdzającego skuteczność.**

Eksperyment dotyczy jednego narzędzia, cystotomu. Po każdej epoce sprawdza, w jakich mierzalnych warunkach model nadal się myli, znajduje w puli obrazy o takich cechach i wybiera z nich następny podzbiór według przewidywanej poprawy straty. Trening zachowuje wagi, optymalizator AdamW i harmonogram. Nie rozróżniamy tu typów narzędzi ani egzemplarzy.

## Co dokładnie realizuje kod

1. `przygotuj.py` dzieli operacje z oryginalnego `train` na FIT i zbiór sterujący D. Całe operacje D są wyłączone z FIT, również klatki D niewylosowane do diagnozy. Oficjalne `val` służy ocenie; `test` nie jest używany przez model ani selektor.
2. Pierwsza epoka jest wspólna dla ramion danego seedu: wagi COCO i ten sam losowany podzbiór, zawierający stałe klatki wsparcia sond. Nie korzystamy ze starego L0, które mogło widzieć operacje D.
3. Po walidacji `trenuj.py` mierzy rzeczywistą stratę detektora na D. Rejestruje również błąd w stałych przedziałach cech. Ten raport powstaje dla każdego ramienia, także losowego i ostatniej epoki.
4. Ramiona cechowe odczytują aktualne tokeny. Sondy liniowe uczone wyłącznie na FIT próbują z przestrzennego opisu tokenów odtworzyć mierzalne cechy. Duży błąd sondy to dodatkowa przesłanka, że reprezentacja słabo zachowuje daną cechę. Sam nie dowodzi, że dodanie takich zdjęć pomoże.
5. `diagnoza.py` tworzy potrzeby, np. przedział małego rozmiaru narzędzia połączony z niską jasnością. Potrzeba musi mieć podwyższoną stratę zadania, odpowiednią liczbę klatek i niezależnych operacji. Progi cech pochodzą z FIT. Braki pomiarów pozostają brakami.
6. Koopman–GLE może zwiększyć priorytet takiej potrzeby. Nie tworzy potrzeby na podstawie samego błędu prognozy. Jego wpływ, podobnie jak wpływ sondy, jest ograniczony do dodatkowych 25 procent w formule priorytetu.
7. `selekcja.py` wyszukuje klatki FIT spełniające opis potrzeb. Część kandydatów jest losowa. Gdy nie ma wiarygodnej potrzeby, następuje jawnie zapisany powrót do losowania.
8. `glister.py` wybiera spośród kandydatów obrazy o korzystnym kierunku gradientu względem D. Co kilka wyborów odświeża gradient D po wirtualnym kroku. Wirtualne wagi są zawsze cofane. Prawdziwy optymalizator nie wykonuje tych kroków.
9. Zmieniamy indeksy samplera, a kolejna epoka kontynuuje ten sam trening. Audyt porównuje faktyczne identyfikatory obrazów odczytane przez loader z listą zaplanowaną przez sampler.

**Odpowiedź eksperymentu ma więc postać „model ma większy błąd dla takich cech; te konkretne klatki je spełniają; taki podzbiór rokuje poprawę straty”.** Nie jest to automatyczny odczyt dowolnego pojęcia z pojedynczego neuronu. Przykład powyżej jest ilustracją formatu, nie wynikiem pomiaru.

## Jakie cechy umiemy nazwać

| Cecha | Źródło i znaczenie |
|---|---|
| `bbox_area_fraction` | Udział prostokąta narzędzia w powierzchni obrazu. |
| `bbox_aspect_ratio` | Proporcja boków bbox. Nie jest pomiarem kąta narzędzia. |
| `bbox_center_x`, `bbox_center_y` | Znormalizowane położenie środka bbox. |
| `bbox_border_distance` | Odległość najbliższej krawędzi bbox od brzegu obrazu. |
| `roi_brightness` | Średnia jasność obszaru bbox. Obejmuje również tło w prostokącie. |
| `roi_blur` | Odwrotna miara energii Laplasjanu. Większa wartość oznacza mniej szczegółów; to przybliżenie, nie pewna etykieta rozmycia. |
| `roi_glare_fraction` | Udział bardzo jasnych pikseli w bbox. To przybliżenie odblasku. |
| `orientation_deg` | Opcjonalna jawna adnotacja kąta. |
| `occlusion_fraction` | Opcjonalna jawna adnotacja zasłonięcia od 0 do 1. |
| `tip_visible` | Opcjonalna jawna adnotacja widoczności końcówki: 0 lub 1. |

Bez trzech ostatnich adnotacji system nie będzie twierdził, że zdiagnozował zasłonięcie albo obrót. Nieznane cechy nie są automatycznie odkrywane semantycznie w tej wersji. Kąt ma na razie zwykłe przedziały liczbowe; do analizy przejścia przez 0/360 stopni potrzebna będzie reprezentacja kołowa.

Opcjonalny plik JSONL, przekazywany przez `FEATURE_ANNOTATIONS` na Edenie:

```json
{"frame_id":"identyfikator_z_manifestu", "orientation_deg":35.0, "occlusion_fraction":0.4, "tip_visible":0}
```

Pula FIT i D jest już oznakowana. Jest to dobór podzbioru do treningu z dostępnych etykiet. Bez etykiet puli nie można wykonać obecnej oceny gradientów GLISTER. Pilot wymaga dokładnie jednego bbox narzędzia na klatkę FIT/D; nie obsługuje jeszcze selekcji negatywnych klatek bez narzędzia ani wielu narzędzi.

## Koopman i rozszerzone GLE

`dynamika.py` podłącza hook do istniejącego modelu. Jeden token oznacza jeden patch ostatniej mapy ViT przed projektorem. Przy rozdzielczości 384 jest to siatka 24 × 24. Stała losowa projekcja redukuje kanały do 24, zachowując wszystkie patche. Trening i diagnoza używają tego samego skalowania obrazu do kwadratu, bez losowych augmentacji w tym pilocie.

GLE analizuje czas filmu przy zamrożonych wagach danej epoki. Nie dopasowuje równania do kilku checkpointów. Patche tworzą nieruchomą siatkę obrazu: indeks patcha nie śledzi fizycznego punktu przemieszczającego się narzędzia. Opis przestrzenny do sond cech (średnia i momenty x/y w bbox) jest osobnym modułem; nie zastępuje cząstek GLE.

Zamrożony port równania jest w `vendor/`, z hashami źródeł w `PROVENANCE.json`:

\[
\widehat z_{t+1,p}=z_{t,p}+D_{t,p}+JP_{t,p}+W_{t,p}+Z_{t,p}+N_{t,p}.
\]

- **D:** uczony dryf zależny od tokenu, prędkości i przyspieszenia.
- **JP:** pamięć z przyczynową uwagą, zanikiem i obrotem par kanałów.
- **W:** wymuszenie z historii sześciu metadanych: pewności detektora, czterech parametrów bbox jego predykcji i jasności obrazu.
- **Z:** powrót do ostatniej kotwicy przy spadku pewności słabej sondy narzędzie/tło. Nie oznacza potwierdzonego fizycznego zniknięcia narzędzia.
- **N:** deterministyczny filtr wcześniejszych reszt, z zanikami/oscylacjami i odczepami 1 oraz 7. Nie jest niezależnym losowaniem szumu.

`ScaledTokenLangevin` dodaje lokalną skalę ruchu patcha i uczone potęgi tej skali dla D, JP i W, zgodnie z adaptacją `tft_langevin_optymalizacja_95`. Kod struktury pamięci i zdarzeń pochodzi z wcześniejszego portu eksperymentów Grant. Okno ma osiem wcześniejszych obserwacji co pięć klatek. Człony nie widzą metadanych ani tokenu przyszłego celu. Warunkowy skok wymagający wiedzy o przyszłym zdarzeniu jest wyłączony. To adaptacja struktury równań TS do obrazu, nie identyczny model jednostek czasu i wymuszeń.

Koopman jest dopasowaniem liniowej ewolucji w rozszerzonych obserwablach tokenu (współrzędne i stałe losowe cechy). Prognoza końcowa to średnia prognoz Koopmana i GLE. W tej wersji nie przypisujemy znaczeń semantycznych poszczególnym wektorom własnym.

Słaba sonda narzędzie/tło jest uczona z patchy wewnątrz i poza bbox. Sygnał pomocniczy jest równy:

\[
g_i=\max\{\operatorname{BCE}(h(z_i),y_i^{patch})-
\operatorname{BCE}(h(\widehat z_i),y_i^{patch}),0\}.
\]

Oznacza to, że prognozowany token jest łatwiejszy do rozpoznania przez słabą sondę niż token obserwowany. **Nie jest to zmierzona korzyść z douczenia detektora.** Osobno zapisujemy kontrolę utrzymania poprzedniego tokenu. Dopiero GLISTER przybliża korzyść zadaniową, a niezależna walidacja po prawdziwym treningu ją sprawdza.

Normalizacje, odczyt patchy, Koopman i GLE są dopasowane z wyłączeniem całych grup D. W FIT stosujemy dodatkowo podział grup na foldy. Sam detektor jest bieżącym modelem uczonym na FIT, więc jego reprezentacje FIT nie są reprezentacjami modelu trenowanego bez danego foldu. Audyt jawnie zapisuje to ograniczenie.

## GLISTER i kontrola eksperymentu

Przybliżenie korzyści dla przykładu to zgodność gradientu jego straty z gradientem D. Gradienty obejmują końcową warstwę klasyfikacji i ostatnią warstwę regresji bbox. Odświeżenie D używa wirtualnego kroku SGD po średnim gradiencie już wybranego podzbioru; prawdziwy trening używa AdamW. Diagnoza pracuje w trybie eval z jedną grupą DETR. To jawna adaptacja GLISTER, bez automatycznego przenoszenia gwarancji oryginalnej pracy.

| Ramię | Sposób wyboru |
|---|---|
| `random` | Losowanie z równoważeniem operacji. |
| `glister` | Losowana po operacjach lista kandydatów, następnie przybliżenie GLISTER. |
| `features_glister` | Potrzeby z błędu zadania i sond cech, wyszukiwanie, GLISTER. |
| `features_kg_glister` | Jak wyżej, dodatkowo priorytet z Koopmana–GLE. |
| `features_placebo_glister` | Jak pełna metoda, lecz sygnał GLE jest tasowany w obrębie operacji i przedziału błędu odczytu patchy. Brak historii nie uczestniczy w tasowaniu. |

Domyślny plan: 5 ramion × 3 seedy, 10 epok, 1500 unikalnych wybranych klatek na epokę. Batch 16 wymaga dopełnienia do 1504 ekspozycji, czyli 94 aktualizacji na epokę i 940 na bieg. Dopełnienia są jawnie zapisane. Pierwsza epoka jest wspólna, potem następuje dziewięć zmian podzbioru. 3000 kandydatów obejmuje 20 procent eksploracji w ramionach cechowych. Stałe wsparcie sond liczy 384 klatki. D: 12 wyłączonych operacji, z których 192 klatki służą diagnozie, a 36 gradientowi referencyjnemu.

Ustawienia są w `config.json`. Kontrola losowa jest sensownym losowaniem po operacjach, więc porównanie nie opiera się na słabym braniu kolejnych klatek jednego filmu. Samo wystąpienie GLE w kodzie nie oznacza jego skuteczności. Główne porównania to pełna metoda względem wariantu bez dynamiki, samego GLISTER i placebo.

Raportujemy AP na niezależnym `val` w kolejnych epokach, liczbę aktualizacji, unikalnych obrazów, ekspozycji i rzeczywisty czas. Czas selekcji może zniwelować korzyść z mniejszej liczby obrazów. `job_timing.json` obejmuje także rozpakowanie danych i przygotowanie cech, a `metric_elapsed_seconds` zapisuje czas osiągnięcia wyniku danej epoki przed wyborem następnego podzbioru. Czekanie w kolejce Slurm nie jest kosztem algorytmu. Przed kampanią trzeba ustalić wspólny próg AP do porównania czasu jego osiągnięcia; plik wynikowy zachowuje pełne krzywe i nie wybiera korzystnego progu po wynikach.

Poprawa D jest obserwacją sterującą, nie niezależnym dowodem generalizacji. Przy trzech seedach wynik traktujemy jako pilot. Obecny zestaw rozdziela wpływ całego dodatku Koopman–GLE; do osobnego dowodu wkładu Koopmana oraz każdego członu równania potrzebne będą dalsze ablacje.

## Artefakty

- Protokół: manifesty FIT/D/val/kontekstu, cechy, COCO, role, hashe wszystkich plików. Archiwa źródłowe mają osobne SHA256, obejmujące rzeczywiste bajty obrazów.
- Runda: wagi `weights.pth`, pełny archiwalny `training_state.ckpt`, faktycznie trenowane identyfikatory, wynik walidacji i raport warunków D.
- Selekcja cechowa: `needs.json`, `candidates.jsonl`, `chosen_reasons.jsonl`, `next_frame_ids.json`, sondy i ich błędy. `feature` oraz przedział w potrzebie określają, czego szuka selektor; `need_ids` wiąże zdjęcie z potrzebą.
- Dynamika: checkpointy foldów, `audit.json`, składowe błędu i sygnał pomocniczy. Duży tymczasowy plik tokenów jest domyślnie usuwany po diagnozie. Pozostają projekcja, metadane i dopasowane modele; `keep_tokens=true` zachowuje również tokeny.
- `completed.json` pojawia się wyłącznie po zakończonym treningu. Nie ma automatycznego resume częściowo wykonanego biegu; pełne checkpointy zachowujemy do audytu i późniejszego jawnego wznowienia. Częściowych katalogów nigdy nie nadpisujemy.

## Testy i uruchomienie na Edenie

Użytkownik zlecił przygotowanie kodu i testów, bez lokalnych smoke testów. Lokalne sprawdzenie tej dostawy ogranicza się do parsowania składni, odczytu kodu i przeglądu. Testy wymagające uruchomienia są odroczone. Kontraktowe testy pokrywają m.in. rozłączność grup i ID, brak przyszłości w GLE, zachowanie patchy, brakujące cechy, budżety, zmianę kierunku gradientu i cofanie wirtualnych wag/RNG.

`run_tests.slurm` sprawdza środowisko i wagi COCO, uruchamia testy jednostkowe, a następnie rzeczywisty test dwóch epok RF-DETR dla losowania i pełnej metody. Mały protokół integracyjny nie służy ocenie AP. Test sprawdza ciągłość liczników i stanu AdamW, zmianę wag, użycie wyłącznie FIT oraz zapis wyników GLE i kolejnego podzbioru.

Środowisko: `rfdetr_nano_clean_1_10_1`, RF-DETR 1.10.1, jedna widoczna karta GPU. Domyślne ścieżki istniejących archiwów i wag są zapisane w `submit_eden.sh`. Oficjalne wagi COCO zweryfikowane odczytem na Edenie mają SHA256 `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`.

Po umieszczeniu kodu na Edenie, z katalogu głównego tej kopii:

```bash
export PROJECT_ROOT=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_cech_adaptacyjny/source
export RESULTS_ROOT=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_cech_adaptacyjny/results
bash "$PROJECT_ROOT/SmallViTAblation/experiments/dobor_cech_adaptacyjny/submit_eden.sh" tests
```

Dopiero po zaliczonych testach, wskazując rzeczywisty numer zadania:

```bash
export TEST_PASS_FILE="$RESULTS_ROOT/tests_NUMER_JOBU/integration/INTEGRATION_PASSED.json"
bash "$PROJECT_ROOT/SmallViTAblation/experiments/dobor_cech_adaptacyjny/submit_eden.sh" pilot
```

Drugie polecenie świadomie zgłasza całą kampanię 15 zadań. Marker testów jest związany z hashami kodu, konfiguracji i archiwów; po ich zmianie testy trzeba wykonać ponownie. Skrypty nie instalują pakietów i nie pobierają wag podczas treningu. `pytest==8.4.2` jest przygotowany w osobnym katalogu `test_dependencies` obok `source`; środowisko RF-DETR pozostaje bez zmian. `TEST_DEPS` trafia do ścieżki importu tylko na czas testów jednostkowych; integracja i trening korzystają ze zwykłego środowiska RF-DETR. Dane są rozpakowywane na dysk lokalny węzła i usuwane dopiero po zakończeniu zadania; wyniki pozostają w `RESULTS_ROOT`.

Jeżeli zadanie testowe jest jeszcze w kolejce, można zgłosić kampanię warunkowo:

```bash
bash "$PROJECT_ROOT/SmallViTAblation/experiments/dobor_cech_adaptacyjny/submit_eden.sh" pilot-after-tests NUMER_JOBU_TESTOW
```

Slurm uruchomi te zadania dopiero po zakończeniu testów kodem zero (`afterok`). Każdy bieg dodatkowo sprawdzi marker integracji oraz zgodność kodu i danych. Nieudane lub anulowane testy uniemożliwiają uruchomienie zależnej kampanii (`kill-on-invalid-dep`).

Zebranie zakończonych wyników:

```bash
cd "$PROJECT_ROOT"
"$PY" -B -m SmallViTAblation.experiments.dobor_cech_adaptacyjny.raport "$RESULTS_ROOT" > comparison.json
```

`PY` można ustawić na `/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python`. Raport odrzuci mieszanie różnych podziałów, cech, wag startowych, kodu i archiwów.

## Podstawa metodologiczna

- [GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning](https://ojs.aaai.org/index.php/AAAI/article/view/16988): dobór danych przez przewidywaną poprawę straty zbioru sterującego.
- [CORDS — implementacje selekcji danych](https://github.com/decile-team/cords): punkt odniesienia do dalszego porównania z pełną implementacją metod gradientowych.
- [TCAV](https://proceedings.mlr.press/v80/kim18d.html): koncepty wymagają powiązania reprezentacji z mierzalnymi przykładami; obecne sondy są prostszym testem przewidywalności cech, nie implementacją TCAV.
- [Domino](https://arxiv.org/abs/2203.14960): diagnoza grup przypadków o podwyższonym błędzie; obecna implementacja używa jawnych cech i przedziałów, nie algorytmu Domino.

Połączenie tych idei z własnym Koopmanem–GLE jest hipotezą tego eksperymentu. Kod przygotowuje jej sprawdzenie, nie zastępuje dowodu empirycznego.
