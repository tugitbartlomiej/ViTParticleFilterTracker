# Model dyfuzyjny wycinków narzędzia i generowanie prowadzone małym ViT (FreeDoM), 2026-09-29

## Prośba użytkownika

„Opracuj, jak wytrenować taki model na Edenie, potem go pobrać i generować na moim komputerze obrazy, ewentualnie na
Edenie podczas testowania. Wygeneruj kody i zatrzymaj się, jak skończysz i będziesz chciał to wysłać na Eden.”
Dodatkowo: uczyć na klatkach filmów („wtedy model nauczy się dobrze generować obrazy”) i nie opierać się na końcówce
narzędzia, tylko na etykiecie narzędzia i jego obrysie.

Cel: zamiennik FastGAN-a, który rysuje różne ułożenia narzędzia. Model dyfuzyjny rysuje wycinek od czystego szumu,
a mały ViT w trakcie rysowania popycha obraz w stronę zadanej siły tożsamości (FreeDoM, Yu i in. 2023). Każdy start
z innego szumu daje inne położenie i kąt narzędzia. Wzorem jest Frisch i in. 2023 (własny model dyfuzyjny na
CATARACTS; klinicyści mylili jego obrazy z prawdziwymi w 61 % par). Przegląd: `../fastgan_odwrotnosc_mlp/NOTATKA_LITERATURA.md`.

## Stan 2026-09-29 17:45: uczenie na Edenie trwa (zgoda użytkownika: „tak zrób i potem rozpocznij trening na Edenie”)

- Dane (wariant zatwierdzony): klatki już leżące na Edenie w `~/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/
  context_train.tar` (57 073 klatki, co piąta klatka 118 filmów treningowych, bez adnotacji; tylko do odczytu) +
  obrysy cystotomu z lokalnych archiwów (`kod/przygotuj_obrysy.py` → `dane/obrysy/obrysy.npz`, 56 005 klatek, 23 MB).
  Pliki JPEG na Edenie są bajt w bajt takie same jak w lokalnych archiwach (SHA256 trzech klatek). Wycinanie w trakcie
  uczenia: klatka zmniejszona do 480×320 (jak w małym ViT; część filmów ma 720×576, np. `TR_0027`), maska z obrysu,
  kwadrat 176 px w losowym miejscu wokół obrysu, odbicie poziome.
- Próba 1818351 padła na klatce 720×576 (kod zakładał 720×480); poprawione. Próba 1818352: 500 kroków, strata
  0,97 → 0,05, 0,385 s/krok na H100 PCIe, 41 GB pamięci karty; podgląd danych poprawny (`wyniki/eden_proba/`).
- Uczenie: 1818354 (100 000 kroków, batch 64), kolejne fragmenty 1818355 i 1818356 z `--dependency=afterany`.
  Przy 0,385 s/krok całość ~10,7 h, czyli dwa 6-godzinne fragmenty; trzeci zakończy się od razu, jeśli niepotrzebny.
- Próbna wysyłka filmów (prośba użytkownika): 3 archiwa (1,38 GB) w 322 s, czyli ~4,3 MB/s, sumy zgodne;
  leżą w `~/smallvit_runs/2026-09-29_dyfuzja_narzedzie/filmy_zip/`. Wszystkie 118 filmów treningowych to 28,4 GB
  (~1 h 50 min wysyłki); zmieszczą się (Eden: 121,8 z 200 GB, 168 473 z 200 000 plików), ale nie są potrzebne do
  obecnego uczenia.

## Dane (`kod/przygotuj_filmy.py` → `dane/filmy3/`)

- Źródło: całe filmy z archiwów Cataract-LMM `3_Object_Tracking/annotations/<film>.zip` (lokalnie, 42 GB). Każdy film
  ma wszystkie klatki i obrys cystotomu na każdej klatce. Na Edenie tych filmów nie ma: jest tylko co piąta klatka
  (22 680 klatek w `tracking_static36`) i bez obrysów, dlatego okna wycinamy lokalnie i wysyłamy jeden plik.
- Tylko 118 filmów ze splitu train (`pool_manifest.jsonl`); 25 filmów walidacji i filmy spoza manifestu pominięte
  (zbiory filmów rozłączne, sprawdzane w kodzie). Co trzecia klatka z obrysem, bo sąsiednie są prawie identyczne.
- Okno: klatka 720×480 zmniejszona do 480×320 (dokładnie skala, w której uczy się mały ViT i w której sondy mają
  176 px), okno 240 px ze środkiem w środku prostokąta opisanego na obrysie narzędzia, dosunięte do brzegu klatki.
- 4 kanały: RGB i obrys narzędzia. Obrys jest potrzebny, bo siłę tożsamości liczymy na tokenach narzędzia, a przy
  obrazie z generatora tokeny wskazuje wygenerowany obrys (tak samo jak kanał maski FastGAN-a).
- Zapis w jednym pliku (na Edenie zajętych jest 159 873 z 200 000 plików): `okna.bin` (JPEG jakości 95 + PNG obrysu
  każdego okna), `okna_indeks.npy`, `okna.json`, `MANIFEST.json`, `SHA256SUMS`, `PODGLAD.jpg`.
- Wynik: 83 091 okien ze 118 filmów (313 359 klatek), 1,59 GB w `okna.bin`; w filmach `TR_0019` (307) i `TR_0057` (183)
  część adnotacji nie ma obrazu w archiwum; te klatki pominięto.
- Wcześniejsza wersja z samych klatek puli (10 421 okien, środek w końcówce, `kod/przygotuj_dane.py`,
  `dane/okna240.npy`, 2,4 GB) zostaje lokalnie i nie jest używana. W `dane/filmy/` leży niedokończony `okna.bin`
  (233 MB) z przerwanej próby (brak kilku obrazów w archiwum `TR_0019`); do usunięcia po zgodzie użytkownika.

## Model i uczenie (`kod/model_dyfuzji.py`, `kod/trenuj.py`)

- UNet 22,3 mln parametrów (poziomy 176, 88, 44, 22 px; kanały 64, 128, 192, 256; uwaga na 22 px), przewiduje szum
  (DDPM), harmonogram kosinusowy T = 1000, próbkowanie DDIM. Bez warunku: model rysuje zawsze wycinek z narzędziem.
- Każdy przykład: losowy wycinek 176×176 z okna 240, środek obrysu w wycinku z marginesem 16 px, losowe odbicie
  poziome (jak w uczeniu małego ViT). Batch 64, AdamW lr 2·10⁻⁴ z rozgrzewką 1000 kroków, bf16, obcięcie normy
  gradientu 1, średnia wykładnicza wag 0,9995.
- Zapis: `ostatni.pt` co 1000 kroków (wznowienie), `ckpt_XXXXXX.pt` (wagi EMA) co 10 000, `podglad_XXXXXX.png`
  (16 próbek) co 5000, `LOG.jsonl` (strata także w 4 przedziałach t, s/krok, pamięć), `STATUS.json`, `model_ema.pt`.
- Przerwanie i wznowienie: SLURM przy końcu czasu tworzy `STOP`, Python zapisuje `ostatni.pt`; kolejne zgłoszenie
  z tymi samymi argumentami wznawia.
- Budżet: najpierw próba 500 kroków (pomiar s/krok na H100), potem decyzja o liczbie kroków. Robocza propozycja:
  100 000 kroków × 64 = 6,4 mln przykładów (~77 przejść przez 83 091 okien), w 6-godzinnych fragmentach.

## Generowanie (`kod/generuj.py`)

Od czystego szumu, DDIM 50 kroków. Na każdym kroku: przewidziany czysty obraz → mały ViT (wagi z pętli, np.
`wspolny/k02.pt`) → siła tożsamości na tokenach narzędzia wskazanych przez wygenerowany obrys → strata
((s − s_cel)/skala)² → gradient przez UNet i ViT → poprawka przewidzianego szumu. Cel: `--cel` albo `--kotwica`
+ `--zmiana` (siła sondy-kotwicy + zmiana przewidziana przez Koopmana lub Langevina). Wektor tożsamości z
`maly_vit_tozsamosc_tla/wyniki/WEKTORY.npz` (`s47_nowy` itd.) albo `KIERUNKI_CZASTEK.npz`.

Różnice względem pracy FreeDoM (moje decyzje, do zatwierdzenia):
1. Poprawka w przestrzeni szumu z normowaniem gradientu do normy przewidzianego szumu (jak Shirahama 2025), a nie
   krok ρ_t w przestrzeni obrazu; siła poprawki `--waga` (domyślnie 0,3, do dobrania po uczeniu).
2. Podróż w czasie tylko w fazie t/T ∈ [0,5; 0,8], 3 powtórzenia (zakres jak w FreeDoM, liczba powtórzeń moja).
3. Siła liczona na 4 zaburzonych kopiach obrazu (jasność, kontrast, szum, rozmycie; jak `sila._sila_odporna`),
   żeby prowadzenie nie szło w niewidoczny wzorek pikseli; `--odporna 0` wyłącza.

## Sprawdzone lokalnie (RTX 3070, 8 GB)

- Dane: 2 filmy → 2547 okien w 13 s; okna i obrysy na podglądzie poprawne.
- Uczenie: 100 kroków z wznowieniem od kroku 60, batch 8: 0,27 s/krok, 5,9 GB; dane dekodowane w procesach
  (4 ms na partię, więc czas zajmuje sama sieć). Na 8 GB mieści się batch 8, więc pełne uczenie tylko na Edenie.
- Generowanie z nieuczonym modelem (sprawdzenie przepływu gradientu), cel 6,59: bez prowadzenia siła odporna 3,83,
  waga 0,3 → 6,27, waga 1,0 → 7,24. Około 1-2 s na obraz przy 20 krokach.

## Kroki na Edenie (po zgodzie)

1. `bash eden/wyslij.sh` — sprawdza sumy, kwotę `lfs quota -p 1299` przed i po, wysyła kod, dane, `WEKTORY.npz`,
   poprawia końce linii; nie zgłasza zadań. Katalog: `~/smallvit_runs/2026-09-29_dyfuzja_narzedzie`.
2. `sbatch trenuj_eden.slurm 500 64 proba` — próba (osobny katalog `wyniki/proba`), pomiar czasu kroku.
3. `sbatch trenuj_eden.slurm <KROKI> 64` i kolejne fragmenty z `--dependency=afterany:<JOB>`.
4. Test na Edenie: `sbatch generuj_eden.slurm 47 2.0 16 0.3` (ziarno 47, zmiana siły +2, 16 obrazów, waga 0,3).
5. `bash eden/pobierz.sh trening` — `model_ema.pt`, `ostatni.pt`, podglądy, log, sumy SHA256; generowanie lokalne:
   `python kod/generuj.py --dyfuzja wyniki/eden_trening/model_ema.pt ...` (RTX 3070 wystarcza do generowania).

Stan Edenu 2026-09-29 16:5x: 118,6 z 200 GB, 159 873 z 200 000 plików. Ten eksperyment doda ~1,6 GB danych,
~0,4 GB na `ostatni.pt` i ~0,09 GB na każdy `ckpt`, kilkanaście plików.

## Czego ten etap nie dowodzi

Ani dobra jakość obrazów, ani trafienie w siłę tożsamości nie oznaczają szybszego uczenia detektora. Wersja 3 pętli
pokazała, że zysk daje samo wyszukiwanie podobnych klatek, a prognoza nie dodaje nic ponad obraz bez przesunięcia.
Użycie tego generatora w pętli to osobne ramię i osobna kampania (plan i zgoda).

## Po uczeniu (osobne kroki)

- Ocena jakości: podglądy, odległość próbek do najbliższych okien uczących (czy model nie kopiuje klatek).
- Dobór `--waga` i liczby powtórzeń: trafienie w siłę zwykłą i odporną, wygląd obrazu.
- Różnorodność: czy starty z różnych szumów dają różne położenia narzędzia.
