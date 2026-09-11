# 03 — Koszt i optymalizacja pod ~133 biegi

Status 2026-09-11: historyczny kosztorys v1. Liczby 133/144/170 w notatkach
dotyczą różnych planów. Aktualną liczbę liczyć z zamrożonego manifestu biegów;
zmiana jednostki budżetu na źródła wymaga przeliczenia. Parametry klastra są
historycznym inwentarzem, nie bieżącym odczytem dostępności.

---

## 1. Sprzęt

| Zasób | Co jest | Ograniczenia |
|---|---|---|
| lokalnie | RTX 3070, 8 GB, torch 2.6.0+cu124, transformers 5.1.0, Python 3.11 | tylko smoke testy i ewaluacja |
| Eden `hopper` | H100 × 8, **MaxTime 5 dni** | zalecana partycja dla długich biegów |
| Eden `short` / `long` | A100 × 8 na węzeł (dgx-1..4) | `short` 1 dzień, `long` 5 dni |
| Eden `hopper-2` | H200 × 4, **MaxTime 24 h** | za krótkie dla łańcuchów, dobre dla krótkich biegów |

Szczegóły i wzorce SLURM: `Eden/EDEN_CONSTRAINTS.md`. Partycja `debug` jest
**zablokowana przez administratora** — nie używać.

## 2. Kalibracja zamiast zgadywania

Nie wpisujemy tu wymyślonych czasów. **Pierwszy krok całego planu to jeden bieg
kalibracyjny**, który mierzy:

- czas na epokę przy 5 000 obrazów, RF-DETR-Small, 512×512, na jednym H100,
- zajętość VRAM przy kilku rozmiarach batcha,
- ile epok potrzeba, żeby krzywa walidacyjna się wypłaszczyła (to ustala stałą
  liczbę epok dla wszystkich biegów),
- przepustowość wczytywania danych — czy wąskim gardłem jest dysk czy GPU.

Dopiero po nim wypełniamy tabelę poniżej i decydujemy o poziomie B.

| Wielkość | Zmierzona | Data pomiaru |
|---|---|---|
| s/epokę, 5k obrazów, 1× H100 | — | — |
| VRAM przy batch 16 | — | — |
| epok do wypłaszczenia | — | — |
| pełny bieg poziomu A (min) | — | — |

Wzór na całość, do wypełnienia po kalibracji:

```
GPU_czas = cechy + proxy + strojenie + treningi + ewaluacja
wall_time ≈ GPU_czas / efektywnie_zajete_GPU + transfery + kolejka
```

**Sanity check przed startem produkcji:** jeżeli jeden bieg poziomu A wychodzi
dłużej niż około godziny na H100, to plan nie mieści się w rozsądnym czasie
i trzeba zejść na RF-DETR-Nano przy 384×384 albo obciąć budżet do 2 500.

## 3. Optymalizacje, które trzeba zrobić RAZ, a oszczędzają na każdym biegu

### 3.1 Cache cech — jeden na całą macierz

Cechy DINO, Fourier i SAM **nie zależą od wariantu ani od ziarna**. Liczymy je
raz na pulę treningową i trzymamy w `feature_cache/`. Wariant to tylko inne wagi
przy tych samych wektorach.

Wyjątek: **EL2N zależy od ziarna** (bo zależy od modelu proxy) — osiem plików,
nie jeden. To jedyny sygnał liczony wielokrotnie.

Wzorzec cache'u już istnieje:
`Eden/Checkpoints/Ablations/ablation/feature_cache/` z plikami `dino_features.pkl`
(90141×1024), `el2n_scores.pkl`, `sam_scores.pkl`, `fourier_features.pkl`
w spójnej kolejności. Nowy cache ma tę samą strukturę, ale liczony **tylko na
klatkach treningowych**.

### 3.2 Zbiory jako manifesty, nie kopie plików

Sto trzydzieści biegów × 5 000 obrazów to potencjalnie setki gigabajtów
duplikatów. Zamiast tego każdy wariant zapisuje **plik JSON z listą nazw**
plus wygenerowany `annotations_train.json`. Obrazy leżą raz, w jednym katalogu.

Na węźle: rozpakować obrazy **raz** do `/raid` (14 TB, lokalny, szybki) i puścić
z tego wszystkie biegi jednego zadania SLURM. Kopiowanie danych per bieg to
najczęstsza strata czasu na klastrze.

### 3.3 Wstępne przeskalowanie obrazów

Opcjonalny cache obrazów tworzymy osobno dla wybranego preprocessingu i
rozdzielczości, zachowując oryginały. Nie zakładamy z góry 512×512 ani zysku
rzędu dni. Trzeba zmierzyć dekodowanie, transfer i resize; gotowy JPEG nadal
wymaga dekodowania. Każda zmiana geometrii musi mieć zgodne anotacje i inverse
transform predykcji. Letterbox nie zastępuje automatycznie preprocessingu modelu.

### 3.4 Upakowanie biegów na węźle

Węzeł `hopper` ma 8 GPU. Mały ViT przy 512×512 zajmuje ułamek H100, więc:

- **jedno zadanie SLURM = jeden węzeł = 8 biegów równolegle**, po jednym na GPU,
- kolejka biegów w pliku, workery biorą następny po skończeniu poprzedniego,
- przy dwóch biegach na GPU (jeśli VRAM pozwoli) — szesnaście naraz.

To zamienia „133 biegi po kolei" w kilkanaście zadań po kilka godzin.

### 3.5 Wspólny punkt startowy

Wszystkie biegi startują z **tych samych wag COCO**. Pobrać raz, trzymać lokalnie
na węźle. Ziarno steruje inicjalizacją głowy i kolejnością danych, nie
backbonem — inaczej porównanie miałoby dodatkowe źródło rozrzutu.

### 3.6 Predykcje zapisywane raz

Ewaluacja zapisuje predykcje w formacie COCO. Każda kolejna metryka (mAP przy
innym progu, F1 w funkcji progu, kalibracja) liczy się **z zapisanego pliku**,
bez powtórnej inferencji. Ten wzorzec już działa w
`AdvancedDatasetSelection/ablation/detr/local_eval_2026-06-11/` i zostaje.

Uwaga na rozmiar: przy standardowej inferencji i progu 0,001 pliki predykcji
robią się duże (poprzednio 34 MB na wariant). Trzymać je poza gitem — reguła
`*results/` w `.gitignore` już to załatwia; do repozytorium wchodzą tylko
`metrics.json` i raporty.

## 4. Kolejność prac

| # | Krok | GPU | Blokuje |
|---|---|---|---|
| 0 | sprawdzić podział pacjent vs film (O1) | nie | wszystko |
| 1 | policzyć dostępne klatki źródłowe w treningu (O2) | nie | budżet poziomu B |
| 2 | zbudować pulę treningową i przeskalować obrazy | nie | 3, 5 |
| 3 | cache cech DINO / Fourier / SAM na puli treningowej | tak, raz | 6 |
| 4 | **bieg kalibracyjny** RF-DETR-Small | tak | cała reszta |
| 5 | modele proxy EL2N, 3 ziarna × 10 epok | tak | 6 |
| 6 | wygenerować 15 wariantów × 3 ziarna + baseline'y | nie | 7 |
| 7 | poziom A — około 98 biegów | tak | 8 |
| 8 | agregacja, wybór konfiguracji do poziomu B | nie | 9 |
| 9 | poziom B + przeniesienie architektur | tak | 10 |
| 10 | jedno przejście po zbiorze zewnętrznym, wpis do ledgera | tak | tabele do artykułu |

Kroki 0 i 1 są darmowe i rozstrzygają kształt reszty. Robimy je pierwsze.
