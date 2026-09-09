# 02 — Macierz eksperymentu

Maszynowy odpowiednik: `configs/variants.yaml`. Rozwinięcie do listy biegów:
`tools/build_run_manifest.py`.

---

## 1. Cztery grupy cech

| Kod | Grupa | Waga bazowa | Wymiar w przestrzeni złączonej |
|---|---|---|---|
| `D` | DINOv3 (semantyka) | 0,35 | 32 po PCA |
| `E` | EL2N (trudność) | 0,30 | 1 |
| `S` | FastSAM (złożoność sceny) | 0,20 | 9 |
| `F` | Fourier (jakość/ostrość) | 0,15 | 1 |

**Reguła wyłączenia grupy:** waga → 0, pozostałe **przeskalowane
proporcjonalnie** do sumy 1. Jedna reguła dla wszystkich piętnastu kombinacji.

Przykład dla `DES` (bez Fouriera): 0,35 / 0,30 / 0,20 podzielone przez 0,85 daje
0,412 / 0,353 / 0,235.

> To jest zmiana wobec poprzedniej ablacji, gdzie V2 dostał ręcznie 0,45 / 0,25 /
> 0,30, a V3 0,45 / 0,20 / 0,35. Te liczby nie wynikały z żadnej reguły, więc
> „bez Fouriera" i „bez EL2N" różniły się od pełnego pipeline'u także
> proporcjami pozostałych cech. Dwie zmiany naraz.

## 2. Pełna macierz — 15 niepustych kombinacji

| ID | Grupy | Nazwa opisowa |
|---|---|---|
| `M-DESF` | D+E+S+F | pełny pipeline (dawne V1) |
| `M-DES` | D+E+S | bez Fouriera (dawne V2) |
| `M-DEF` | D+E+F | bez SAM |
| `M-DSF` | D+S+F | bez EL2N (dawne V3) |
| `M-ESF` | E+S+F | bez DINO |
| `M-DE` | D+E | semantyka + trudność |
| `M-DS` | D+S | semantyka + złożoność |
| `M-DF` | D+F | semantyka + jakość |
| `M-ES` | E+S | trudność + złożoność |
| `M-EF` | E+F | trudność + jakość |
| `M-SF` | S+F | złożoność + jakość |
| `M-D` | D | tylko semantyka (bliskie dawnemu V6) |
| `M-E` | E | tylko trudność (bliskie dawnemu V7) |
| `M-S` | S | tylko złożoność |
| `M-F` | F | tylko jakość (bliskie dawnemu V4) |

Recenzent 2, zarzut 5, poprosił dokładnie o to: „systematic ablation matrix
covering all relevant combinations of the four feature groups", z pojedynczymi
cechami, parami i trójkami. Piętnaście pozycji to komplet, więc nie ma miejsca
na zarzut o wybiórczość.

## 3. Baseline'y

| ID | Definicja | Po co |
|---|---|---|
| `B-RANDAUG` | losowe N z puli augmentowanej treningowej | odniesienie, którego pipeline musi bić |
| `B-RANDSRC` | losowe z **klatek źródłowych** treningowych, potem augmentacja do N | **żądanie recenzenta 2, zarzut 6** — odpowiada, czy pipeline bije zwykłe zmniejszenie zbioru |
| `B-UNIFORM` | równa liczba klatek z każdego filmu treningowego | tani baseline pokrycia; mocniejszy niż losowy |
| `B-FULLPOOL` | wszystkie klatki treningowe, bez selekcji | sufit przy pełnym budżecie; inny budżet, raportowany osobno |

`B-RANDAUG` idzie na **10 losowaniach** — to on ma największą wariancję, więc
bez tego nie da się powiedzieć, czy przewaga pipeline'u jest realna.
`B-RANDSRC` ma ograniczenie liczbowe opisane w `01_PROTOCOL.md`, punkt P4.

## 4. Analiza czułości (zarzut 3 recenzenta 1)

Wszystko przy budżecie poziomu A, jedno ziarno, wariant odniesienia `M-DESF`.

| ID | Co zmieniamy | Wartości |
|---|---|---|
| `S-W±` | wagi grup | każda grupa ±0,10 przy proporcjonalnym przeskalowaniu reszty (8 biegów) |
| `S-PCA` | wymiar PCA dla DINO | 8, 16, 32, 64 |
| `S-REP` | wybór reprezentanta klastra | centroid / losowy z klastra / najbliższy medoid |
| `S-ALG` | algorytm pokrycia | K-Means / K-Center Greedy |
| `S-DIFF` | współczynniki w mierze trudności | 0,7·(1−s) + 0,3·H zamienione na 0,5/0,5 i 0,9/0,1 |

Recenzent napisał wprost, że wagi „chosen experimentally by the first author" to
za mało. Ta tabela zamienia zdanie usprawiedliwiające w wynik.

## 5. Sweep rozmiaru zbioru

Budżety 2 500 / 5 000 / 10 000 / 20 000 dla `M-DESF`, `B-RANDAUG` i `B-RANDSRC`,
po 3 ziarna. To odpowiada na uwagę drobną recenzenta 1 („evaluating multiple
subset sizes") i daje najładniejszy rysunek do artykułu: dwie krzywe, które się
rozjeżdżają albo nie.

## 6. Poziom B — potwierdzenie i przeniesienie na inne architektury

Wchodzi **tylko to, co poziom A wskaże jako istotne**, plus stałe punkty
odniesienia:

- konfiguracje: `M-DESF`, dwie najlepsze niepełne z poziomu A, `B-RANDAUG`,
  `B-RANDSRC` — 5 pozycji,
- budżet 20 000, RF-DETR-Small, **5 ziaren** → 25 biegów,
- przeniesienie architektur: te same 5 konfiguracji × 1 ziarno × {D-FINE-S,
  aktualna YOLO} → 10 biegów, przy **iso-resource**: ta sama rozdzielczość
  wejścia, ten sam budżet epok, ten sam zbiór.

Iso-resource jest tu obowiązkowe, bo zarzut 8 recenzenta 1 dotyczył właśnie
tego, że DETR chodził na 800×1333, a YOLOv8 na 640×640, więc porównanie mierzyło
dwie recepty treningowe, a nie dwie architektury.

## 6b. Walidacja sondy: 512 kontra 704

Macierz jedzie na RF-DETR-Small przy 512 pikselach, bo to tanie. Ale przy tej
rozdzielczości **56,9% ramek narzędzia spada poniżej progu małego obiektu**,
a przy 704 tylko 24,6% (`13_STANDARD_DANYCH.md`). Trzeba sprawdzić, czy tania
sonda szereguje warianty tak samo jak model o większym wejściu.

| ID | Konfiguracje | Rozdzielczości | Ziarna | Biegi |
|---|---|---|---|---|
| `P-RES` | `M-DESF`, dwie okrojone, `B-RANDAUG`, `B-RANDSRC` | 512 i 704 | 3 | 30 |

Miara: korelacja rang między wynikami przy 512 i 704. Zgodność uzasadnia sondę
i sama w sobie jest wynikiem do opisania — to empiryczne potwierdzenie
założenia Selection via Proxy w naszym zadaniu. Niezgodność oznacza, że cała
macierz musi pójść na 704, za mniej więcej dwukrotność kosztu.

## 7. Analizy bez GPU

Tanie, a domykają uwagi drobne:

- **skład augmentacji w wybranym zbiorze** — czy pipeline preferuje określone
  typy przekształceń (uwaga drobna recenzenta 1); czysta analiza nazw plików,
- **pokrycie rozkładu zewnętrznego** — osadzenia DINO benchmarku zewnętrznego
  i odległość wybranego zbioru do nich; to jest właściwa miara różnorodności,
  której wcześniej brakowało,
- **korelacja rang EL2N z DETR vs EL2N z YOLO** — odpowiedź na zarzut o błędne
  koło,
- **rekoncyliacja liczby 91 336** — recenzent 2 policzył 22 837 × 4 = 91 348
  i wskazał niezgodność; trzeba ją wyjaśnić albo poprawić.

## 8. Rachunek biegów

| Grupa | Biegi |
|---|---|
| macierz 15 × 3 ziarna | 45 |
| `B-RANDAUG` × 10 | 10 |
| `B-RANDSRC` × 3 | 3 |
| `B-UNIFORM` × 3 | 3 |
| `B-FULLPOOL` × 1 | 1 (drogi — pełna pula) |
| czułość | ~18 |
| sweep budżetu (bez pokrycia z macierzą) | ~18 |
| **poziom A razem** | **~98** |
| poziom B, RF-DETR-S, 20k | 25 |
| przeniesienie architektur | 10 |
| **razem** | **~133** |

Do tego **modele proxy dla EL2N**: jeden na ziarno, nie na wariant — czyli
3 (poziom A) + 5 (poziom B) = 8 krótkich biegów po 10 epok.
