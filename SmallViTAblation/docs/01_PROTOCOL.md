# 01 — Specyfikacja protokołu (dokument nadrzędny)

Wersja: v1 (2026-09-08). Każda zmiana tutaj wymaga wpisu w sekcji „Historia zmian"
na końcu pliku. Maszynowy odpowiednik: `configs/protocol.yaml`.

---

## P0. Zasada

Jeden bieg = (wariant selekcji) × (ziarno losowe). **Wszystko inne jest stałe.**
Pełny pipeline jest **jednym z wariantów w tej samej macierzy**, nie osobnym
biegiem produkcyjnym. To jest jedyna zmiana, która sama w sobie unieważnia
zarzut 4 recenzenta 1.

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

Recenzent 1 zarzut 1 żądał „video/procedure or **patient** level". Ten wymóg jest
spełniony i wchodzi do artykułu jednym zdaniem: *patient-level split, one
recording per patient, 50 patients partitioned 35/8/7*. W tabeli podziału
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

## P3. Sygnał trudności (EL2N) bez błędnego koła

Obecny EL2N pochodzi z DETR po 170 epokach trenowanego na pełnej puli — czyli
model, który widział też klatki testowe, ocenia trudność danych treningowych,
a potem te dane trenują model, którym się chwalimy. Recenzenci tego nie
zauważyli wprost, ale zarzut o błędne koło (zarzut 5 recenzenta 1, „DETR-derived
difficulty → DETR wygrywa") jest tuż obok.

Nowa reguła:

1. trenujemy **model wstępny (proxy) tylko na klatkach treningowych**, przez
   **10 epok**, z tym samym ziarnem co bieg docelowy,
2. z niego liczymy EL2N zgodnie z definicją oryginalną (wcześnie w treningu),
3. proxy jest **tej samej architektury** co model docelowy (RF-DETR-S), więc nie
   ma przeskoku między rodzinami,
4. wynik zapisujemy jako `el2n_scores_<seed>.pkl` — **EL2N zależy od ziarna**,
   więc każdy seed ma swój; to zwiększa koszt, ale bez tego cała macierz mierzy
   jeden przypadkowy przebieg.

Dodatkowo (tanie, a domyka zarzut o błędne koło): policzyć EL2N także z modelu
**innej rodziny** i pokazać korelację rang. Plan YOLO-EL2N z 25 kwietnia
(`AdvancedDatasetSelection/ablation/yolo/PLAN.md`) wchodzi tu jako podpunkt.

## P4. Budżet obrazów

Każdy wariant dostaje **dokładnie tyle samo obrazów**. Rozmiar budżetu jest
osobnym wymiarem eksperymentu (sweep), nie wolną zmienną wewnątrz porównania.

| Poziom | Budżet | Po co |
|---|---|---|
| A (macierz) | **5 000** | pełna macierz + ziarna, tanio |
| B (potwierdzenie) | **20 000** | zgodność z dotychczasową narracją artykułu |
| sweep | 2 500 / 5 000 / 10 000 / 20 000 | krzywa „ile danych vs jakość" |

**Uwaga licząca się dla baseline'u „losowo z klatek źródłowych"**: klatek
źródłowych jest 22 837 w całej puli, a filmy treningowe to 70% filmów, więc
w treningu jest ich rzędu 16 tys. Przy budżecie 20 000 tego baseline'u **nie da
się zrobić bez powtórzeń**. Liczbę trzeba policzyć na danych (`05_OPEN_QUESTIONS`,
punkt O2) i albo obniżyć budżet poziomu B, albo zdefiniować ten baseline jako
„wszystkie dostępne klatki źródłowe treningowe + augmentacja do budżetu".

## P5. Trening — identyczny dla wszystkich

| Parametr | Wartość | Uwaga |
|---|---|---|
| model | RF-DETR-Small | backbone DINOv2 ViT, Apache 2.0 |
| rozdzielczość | **512 albo 704 — decyduje kalibracja** | przy 512 aż 56,9% ramek narzędzia spada poniżej progu 32×32, przy 704 tylko 24,6%; patrz `13_STANDARD_DANYCH.md` |
| epoki | **stała liczba, bez early stopping** | wartość z kalibracji, plan: 60 |
| optymalizator, LR, warmup, weight decay | jedna wartość dla wszystkich | z kalibracji |
| batch, gradient accumulation | jedna wartość | efektywny batch stały |
| augmentacja w treningu | **wyłączona albo identyczna** | pula już jest augmentowana |
| precyzja | AMP bf16 | |
| inicjalizacja | te same wagi startowe COCO | ziarno steruje tylko kolejnością i dropoutem |

**Brak early stopping to decyzja świadoma.** Wcześniej warianty kończyły trening
w różnych epokach, bo każdy miał własne kryterium na własnym zbiorze
walidacyjnym. To znaczy, że każdy dostał inny budżet obliczeń — czyli porównanie
mierzyło dwie rzeczy naraz.

## P6. Wybór checkpointu — jedna reguła, dwie kolumny

Zawsze raportujemy **obie**:

- **(a) ostatnia epoka** — reguła bez żadnego wyboru, najbardziej odporna
  na zarzut,
- **(b) najlepsza na walidacji same-dist**, gdzie „najlepsza" to jedno,
  zapisane kryterium (mAP@[.5:.95] na `val`), identyczne dla wszystkich.

Nigdy nie mieszamy kolumn w jednej tabeli i nigdy nie wybieramy per wariant tego,
co lepiej wygląda.

## P7. Inferencja — standardowa

Standardowe dekodowanie detektora, top-k po wszystkich zapytaniach. **Zero
[Query 81] (kju osiemdziesiąt jeden).** Historyczne wyniki z jednym zapytaniem
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
Recenzent 1 zarzut 10 dotyczył dokładnie tego, że wcześniej benchmark
zewnętrzny był oglądany wielokrotnie w trakcie rozwoju. Ledger pozwala napisać
w artykule prawdę zamiast deklaracji.

## P10. Ziarna i raportowanie

| Grupa | Liczba ziaren |
|---|---|
| macierz 15 wariantów (poziom A) | 3 |
| konfiguracje do głównej tabeli (poziom B) | 5 |
| baseline losowy | 10 niezależnych losowań |
| analizy czułości | 1 |

Raportujemy **średnią ± odchylenie standardowe**. Przy trzech ziarnach nie
robimy testów istotności udających więcej, niż mamy — podajemy surowe wartości
per ziarno w załączniku i przedział bootstrapowy tylko tam, gdzie ziaren jest
pięć albo więcej.

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

- **v1, 2026-09-08** — pierwsza wersja, napisana po odrzuceniu artykułu.
