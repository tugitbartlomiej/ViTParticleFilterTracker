# 01 — Specyfikacja protokołu (dokument nadrzędny)

Wersja: v2.1, projekt po audycie i decyzji o małych modelach (2026-09-11). Każda zmiana wymaga wpisu w historii.
`configs/protocol.yaml` opisuje obecną implementację i nie jest jeszcze zgodny ze
wszystkimi regułami v2. Status wdrożenia: [21_STAN_REFAKTORU.md](21_STAN_REFAKTORU.md).

---

## P0. Zasada

Jeden bieg określa model/recepta, K, budżet, wariant selekcji i ziarna. **W obrębie kontrastu wariantów wszystkie pozostałe ustawienia są stałe.** Zmiana architektury, K albo budżetu oznacza osobny blok porównań, nie podmianę jednego ramienia.
Pełny pipeline jest **jednym z wariantów w tej samej macierzy**, nie osobnym
biegiem produkcyjnym. Pozwala to kontrolować wpływ selekcji, jeśli zgodne są także
inicjalizacja, dane, ewaluacja i rzeczywisty budżet; samo zapisanie planu nie zamyka uwagi.

---

## P1. Podział danych — po pacjencie, nie po klatce

**Istnieje już podział po pacjentach** i jest zapisany:
`Eden/Checkpoints/Ablations/ablation/video_split.json`

- ziarno 42, **50 nagrań = 50 pacjentów**: 35 train / 8 val / 7 test,
- **jedno nagranie odpowiada jednemu pacjentowi** (potwierdzone przez autora
  2026-09-08), więc podział po filmie **jest** podziałem po pacjencie,
- kopie augmentowane trafiają do tego samego koszyka co ich klatka źródłowa
  (prefiks nazwy pliku `test01_frame_000123.jpg` → stem `test01`),
- generator: `AdvancedDatasetSelection/ablation/generate_video_split.py`,
- konsument dla pełnej puli: `Eden/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/prepare_fullpool_videosplit.py`.

Ten manifest może być wspólnym podziałem nowych biegów. Nie dowodzi, że
historyczne checkpointy korzystały z niego od początku treningu. W tabeli podziału
podajemy **liczbę pacjentów obok liczby klatek** — to pierwsza liczba, na którą
patrzy recenzent.

Ten sam plik splitu obsługuje wszystkie warianty, wszystkie ziarna
i obie architektury. **Nigdy nie generujemy splitu per wariant.**

## P2. Pula selekcji — tylko klatki treningowe

Poprzednio pipeline selekcji jechał po **całej puli 91 tys.**, łącznie z klatkami
z filmów walidacyjnych i testowych. To jest wyciek: wybór zbioru treningowego
podglądał dane, na których potem mierzymy wynik.

Nowy protokół:

- selekcja widzi **wyłącznie klatki z 35 filmów treningowych**,
- cechy (Fourier, DINO, SAM, EL2N) liczone i cache'owane **tylko na tej puli**,
- klatki z filmów val i test nie wchodzą do żadnego kroku selekcji ani do PCA,
  ani do K-Means.

## P3. Sygnał trudności — train-only i jawna definicja

Stary sygnał z checkpointu ep170 wymaga audytu danych i inicjalizacji tego
checkpointu. Ocenianie własnych próbek treningowych nie jest samo w sobie
wyciekiem: tak działa wczesny scoring w oryginalnym EL2N. Trening proxy na
podzbiorze może natomiast mieszać trudność z nierówną ekspozycją danych.

Nowa reguła:

1. trenujemy **model wstępny (proxy) tylko na klatkach treningowych**, przez
   **10 epok**, z tym samym ziarnem co bieg docelowy,
2. zapisujemy definicję błędu: EL2N to estymacja oczekiwanej normy `||p-y||₂`;
   obecny thresholded `1-confidence` dopasowanej detekcji nie odtwarza ogólnie
   tej definicji. Dla RT-DETR alternatywą do zbadania jest natywny loss detekcji
   z surowych wyjść i matchingu treningowego, wyraźnie nazwany innym sygnałem,
3. proxy jest **tej samej architektury** co model docelowy (RF-DETR-S), więc nie
   ma przeskoku między rodzinami,
4. wynik zapisujemy jako `el2n_scores_<seed>.pkl` — **EL2N zależy od ziarna**,
   więc dokumentujemy osobno seedy proxy, selekcji i treningu docelowego.
   Dziesięć epok jest punktem startowym pilotażu, nie definicją EL2N.

Dodatkowo (tanie, a domyka zarzut o błędne koło): policzyć EL2N także z modelu
**innej rodziny** i pokazać korelację rang. Plan YOLO-EL2N z 25 kwietnia
(`AdvancedDatasetSelection/ablation/yolo/PLAN.md`) wchodzi tu jako podpunkt.

## P4. Budżet unikalnych źródeł

Każdy porównywany selector dostaje **dokładnie tyle samo unikalnych klatek
źródłowych z train**. Augmentacja następuje po wyborze, identycznie dla wariantów.
Walidacja i test zawierają oryginały, w tym negatywy. Rozmiar budżetu jest
osobnym wymiarem eksperymentu (sweep), nie wolną zmienną wewnątrz porównania.

| Poziom | Budżet | Po co |
|---|---|---|
| A (macierz) | **5 000** | pełna macierz + ziarna, tanio |
| B (potwierdzenie) | **K ≤ N_train**, po zamrożeniu manifestu | większy wspólny budżet źródeł |
| sweep | np. 2 500 / 5 000 / 10 000 / N_train | krzywa jakości i kosztu; punkty muszą być wykonalne |

W lokalnej anotacji train jest **17 832 nazw źródłowych**, nie 70% wszystkich
klatek wyliczone z udziału filmów. Faktycznie dostępne, poprawne pliki mogą
utworzyć mniejszą pulę. 20 000 różnych źródeł jest tu niewykonalne. Wariant
5000 źródeł × 4 kopie realizuje budżet 20 000 obrazów, ale nie żądany baseline
20 000 źródeł. Historyczną selekcję po augmentacji raportować osobno.

## P5. Trening — identyczny dla wszystkich

Aktualna kolejność: szybki development na YOLOv8n, potem potwierdzenie i transfer według [dokumentu 24](Analiza_ponownej_publikacji/24_SZYBKIE_ABLACJE_I_OPTYMALIZACJA_EDEN.md). Dotychczasowy start od pełnej macierzy RF-DETR-Small zastąpiono poniższą specyfikacją planu; adaptery i YAML wymagają wdrożenia.

| Parametr | Wartość | Uwaga |
|---|---|---|
| model | YOLOv8n w pierwszym etapie | druga rodzina: RT-DETRv2-S; RF-DETR-Nano opcjonalnie dla ViT |
| rozdzielczość | 640 jako punkt startowy YOLO/RT-DETR, dalej kalibracja | jedna w danym bloku; mierzyć geometrię końcówek po transformacji, nie utożsamiać COCO small z oboma bokami <32 px |
| budżet | **stałe rzeczywiste optimizer updates w torze jakości**, bez indywidualnego early stopping | limity z kalibracji; równy czas/równe epoki to osobny tor praktycznej efektywności |
| optymalizator, LR, warmup, weight decay | ta sama recepta między selektorami danego modelu | harmonogram względem aktualizacji; jawna kontrola ustawień automatycznych |
| batch, gradient accumulation | stały fizyczny i efektywny batch w danym bloku | mierzyć faktyczną akumulację i world size |
| augmentacja w treningu | **identyczna między selektorami po wyborze źródeł** | między architekturami jawne recepty; pula augmentowana to osobna kontrola |
| precyzja | wspierany i zweryfikowany AMP frameworka | bf16/fp16 według adaptera i testu, nie założenia o samym GPU |
| inicjalizacja | te same ogólne wagi COCO dla danej architektury | osobny nowy bieg; nie checkpoint proxy ani historyczny ep170 |

**Brak early stopping to decyzja świadoma.** Wcześniej warianty kończyły trening
w różnych epokach, bo każdy miał własne kryterium na własnym zbiorze
walidacyjnym. To znaczy, że każdy dostał inny budżet obliczeń — czyli porównanie
mierzyło dwie rzeczy naraz.

## P6. Wybór checkpointu — jedna reguła, dwie kolumny

Przed testem deklarujemy **obie** reguły, jeśli obie mają wejść do raportu:

- **(a) koniec ustalonego budżetu** — reguła bez żadnego wyboru, najbardziej odporna
  na zarzut,
- **(b) najlepsza na walidacji same-dist**, gdzie „najlepsza" to jedno,
  zapisane kryterium (mAP@[.5:.95] na `val`), identyczne dla wszystkich.

Nigdy nie mieszamy kolumn w jednej tabeli i nigdy nie wybieramy per wariant tego,
co lepiej wygląda. Predykcje, progi i metryki każdego checkpointu mają osobny
katalog. Obecny limit jednego checkpointu w ledgerze nie realizuje automatycznie
planu dwóch reguł — wymaga osobnego ujednolicenia kontraktu przed produkcją.

## P7. Inferencja — standardowa

Standardowe dekodowanie i postprocessing danej wersji detektora, bez wyboru
jednego slotu. Hungarian jest przypisaniem targetów w treningu; COCOeval ma
własne dopasowanie według score/IoU. Historyczne wyniki z jednym zapytaniem
zostają wyłącznie jako osobna analiza „co się dzieje, gdy jedno zapytanie
przejmuje klasę" — ciekawa obserwacja, nie metoda.

## P8. Metryki

- **mAP@0.5** i **mAP@[.5:.95]** przez `pycocotools`, licząc **pełną krzywą PR**
  (próg ufności do generowania predykcji: 0,001, nie 0,3).
  Poprzedni protokół obcinał krzywą progiem 0,3, co zaniża mAP i miesza dwie
  rzeczy: jakość modelu i kalibrację ufności.
- **Operacyjne TP/FP/FN, precision, recall, F1** przy progu ufności **wybranym
  na zbiorze walidacyjnym**, a nie ustawionym arbitralnie. Do artykułu idzie też
  krzywa F1 w funkcji progu — recenzent 1 poprosił o to wprost (zarzut 6).
- Wszystkie zadeklarowane obrazy wchodzą do oceny, także bez predykcji i bez
  GT. AP50/AP75/AR100 są uzupełnieniem głównego AP@[.5:.95]. Próg 0,001 jest
  praktycznym cutoffem do sprawdzenia na val, a nie matematycznie pełną krzywą.
- **Kalibracja pod przesunięciem dziedziny**: czy próg wybrany na walidacji
  własnej przenosi się na zbiór zewnętrzny. To jest osobny rysunek.

## P9. Ewaluacja i dyscyplina „zero-look"

Dwa zbiory oceny:

1. **same-dist test** — 7 filmów testowych, nigdy nie widzianych,
2. **external** — 2 333 obrazy z benchmarku zewnętrznego, wariant FULL
   i CLEAN (minus 48 obrazów z wyciekiem).

Reguła: **wybór epoki i progu odbywa się wyłącznie na walidacji**. Zbiór
zewnętrzny liczymy **raz**, na końcu, dla checkpointów już wybranych.

Wprowadzamy `external_access_ledger.jsonl` — plik, do którego dopisuje się
każdy dostęp do zbioru zewnętrznego: data, kto, który checkpoint, po co.
Recenzent z `Reviewer_Comments_to_Authors.pdf`, punkt 10, wskazał, że benchmark
zewnętrzny był oglądany wielokrotnie w trakcie rozwoju. Ledger pozwala napisać
w artykule prawdę zamiast deklaracji. Nie cofa historycznych analiz ani wpływu
testu na wybór metody. Obecny benchmark trzeba opisać zgodnie z chronologią;
potwierdzenie nowej metody powinno korzystać z odłożonych niezależnych operacji.

## P10. Ziarna i raportowanie

| Grupa | Liczba ziaren |
|---|---|
| macierz 15 wariantów (poziom A) | 3 |
| konfiguracje do głównej tabeli (poziom B) | 5 |
| baseline losowy | 10 niezależnych losowań |
| analizy czułości | 1 |

Liczby powtórzeń w tabeli są propozycją budżetu, nie progiem poprawności
statystycznej. Raportujemy **średnią ± SD między treningami** i surowe wyniki.
CI generalizacji szacujemy osobno przez resampling pacjentów/operacji, nie
niezależnych klatek; przy siedmiu operacjach pokazujemy także każdą z osobna.
Bootstrap pięciu seedów nie jest bootstrapem pacjentów. Jednoseedowa analiza
czułości ma status eksploracyjny i wymaga powtórzeń dla głównych wniosków.

Ziarno steruje: inicjalizacją głowy, kolejnością danych, dropoutem, K-Means
(inicjalizacja centroidów) **oraz** modelem proxy liczącym EL2N. Czyli jedno
ziarno = jeden pełny, niezależny przebieg całego łańcucha.

## P11. Rejestrowanie biegu

Każdy bieg zapisuje `run_manifest.json` z: hashem commita, hashem
`protocol.yaml`, identyfikatorem wariantu, ziarnem, listą wybranych plików
(hash SHA-256 listy, nie sama lista), wersjami bibliotek, czasem i sprzętem.
Bez tego nie da się po pół roku odtworzyć, co było czym — a poprzednio dokładnie
to się stało z etykietą wariantu V4.

---

## Historia zmian

- **v2.1, 2026-09-11** — na życzenie autora development zaczyna się od małych detektorów. Dokumenty 23–24 dodają integrację z grantem, bramki pilota, kontrolowane kroki optymalizacji i transfer zamrożonych subsetów. Zaktualizowano P0/P5/P6; historyczne macierze pozostają osobnymi planami. YAML i trening nie zostały jeszcze zmigrowane.
- **v2, 2026-09-11** — rozdzielono plan od wdrożenia, źródła od augmentacji,
  scoring train-only od wycieku, SD seedów od CI pacjentów i nowe oceny od
  historii testu. YAML pozostawiono bez nadpisywania wcześniejszych zmian użytkownika.
- **v1, 2026-09-08** — pierwsza wersja, napisana po odrzuceniu artykułu.
