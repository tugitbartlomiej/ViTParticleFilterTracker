# Eden: analiza przed sprzątaniem (2026-09-28)

Polecenie użytkownika: „możesz usunąć wszystko z Edena, zostawiając tylko zbiory uczące. Przed usunięciem zrób
analizę i powiedz, co można usunąć”. **Nic jeszcze nie usunięto.** Pomiar: `lfs quota -p 1299`, `du`, pełna lista
plików (`find -printf "%s %p"`, 122 tys. plików), porównanie z dyskiem lokalnym po nazwie, rozmiarze i tokenach
ścieżki (ziarno, ramię, obrót, bieg), tylko pliki ≥ 1 MB.

Stan: **192,4 GB z 200 GB**, 146 993 plików z 200 000. Zadania: brak działających; `1810406`
(`koopman_fastgan_v1`, równoległa sesja) wisi od 2026-09-24 w stanie PENDING/held
(`user_env_retrieval_failed_requeued_held`) i wskazuje na `smallvit_runs/2026-09-24_koopman_fastgan_petla`.
Zadanie `1817712_2` (Large, s43, `bez`, k06) zakończone COMPLETED 2026-09-28 02:08 — jego wynik nie jest jeszcze
pobrany.

## A. Zbiory uczące — ZOSTAJĄ (~12 GB)

| Ścieżka na Edenie (`~` = /mnt/evafs/faculty/home/bpiotrowski) | GB | Uwagi |
|---|---|---|
| `~/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar` | 2,0 | Cataract-LMM tracking (22 680 obrazów, pula, val); z tego archiwum startują zadania pętli; kopia obrazów jest też lokalnie |
| `~/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36/` | 1,95 | ta sama zawartość rozpakowana (22,7 tys. plików) — **duplikat archiwum, można usunąć** |
| `~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation/data/tracking_calibration.tar` | 1,1 | zbiór kalibracyjny Nano |
| `~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation/data/tracking_calibration/` | 1,08 | rozpakowany duplikat (12,5 tys. plików) — **można usunąć** |
| `~/smallvit_runs/2026-09-14_tracking_nano/index/pool_manifest.jsonl` | 0,08 | manifest puli (drugi identyczny w `2026-09-14_nano_smoke`) |
| `~/smallvit_runs/2026-09-24_koopman_gesta_historia/probe/` + `results/run_*/datasets/` | 0,05 | sondy i manifesty podzbiorów używane przez pętle (`$ROOT/probe/manifest.json`) — przenieść do katalogu danych |
| `~/grant_tft/data/` (favorita_pelny, favorita_promocja) | 4,5 | dane Favority projektu grantowego; kopia lokalna jest |
| `~/grant_tft/droga95/outputs/tft_deep_koopman_mz/dane/` | 3,2 | dane pochodne (wejścia eksperymentów grantowych), **tylko na Edenie** |
| `~/merged_20k_annotations.json`, `~/merged_20k_annotations_fixed.json` | 0,01 | adnotacje |

## B. Można usunąć bez straty (kopia lokalna albo odtwarzalne) — ~29 GB

| Ścieżka | GB | Dlaczego bez straty |
|---|---|---|
| `~/smallvit_runs/2026-09-24_koopman_gesta_historia` (poza `probe/`) | 22,5 | 100 % dużych plików jest lokalnie (`experiments/koopman_gesta_historia`, folder równoległej sesji) |
| `~/smallvit_runs/2026-09-22_dobor_cech_adaptacyjny` | 1,2 | 100 % lokalnie |
| `~/.vscode-server` | 4,1 | serwer VS Code Remote; odtwarza się sam przy połączeniu (19 tys. plików) |
| `~/.roboflow` | 0,35 | jeden plik, kopia lokalna |
| `~/smallvit_runs/2026-09-15_tracking_static36` poza `data/` (cache DINOv2, kolejka) | 0,2 | pamięć podręczna i logi |
| rozpakowane duplikaty zbiorów z sekcji A | 3,0 | archiwa `.tar` zostają |

## C. Wyniki pośrednie tylko na Edenie — usunięcie = utrata (~122 GB wyników ViT)

Wnioski z tych biegów są zapisane lokalnie (WYNIKI.md, ANALIZA.json w eksperymentach). Na Edenie zostały głównie
wagi pośrednie, punkty kontrolne `.ckpt` i zrzuty latentu.

| Ścieżka | GB | Tylko na Edenie | Co to jest |
|---|---|---|---|
| `2026-09-27_petla_large` | 35,0 | 18,6 GB | wagi ramion `losowe`/`bez` Large; **wynik s43 bez k06 niepobrany** |
| `2026-09-21_koopman_langevin` | 26,1 | 13,9 GB | zrzuty cząstek `CZASTKI_*` |
| `2026-09-24_koopman_fastgan_petla` | 23,1 | 18,7 GB | latenty biegów próbnych (`smoke_*/continuous`); tu wskazuje wiszące zadanie 1810406 |
| `2026-09-22_dobor_klatek_stan_sieci` | 11,7 | 2,6 GB | cache tokenów, biegi pilota |
| `2026-09-25_kontynuacja_5epok` | 8,4 | 8,3 GB | punkty kontrolne treningu 5 epok |
| `2026-09-26_petla_uczenia` | 7,0 | 2,8 GB | wagi pośrednie ramion `losowe`/`bez` pilota |
| `2026-09-23_fastgan` | 5,9 | 4,6 GB | punkty kontrolne FastGAN 5k–45k (końcowy 50k jest lokalnie) |
| `2026-09-14_tracking_nano/SmallViTAblation/results/calibration` | 0,75 | 0,75 GB | wyniki kalibracji Nano |
| `2026-09-25_kontynuacja_dobor`, `..._dobor150` | 4,5 | 4,5 GB | punkty kontrolne |

## D. Nie ViT — projekt grantowy (Favorita) — decyzja osobna (~21 GB)

| Ścieżka | GB | Tylko na Edenie |
|---|---|---|
| `~/grant_tft/outputs` (np. `tft_surowy_stan_sterowany/PROGNOZY_*.npz` po 1,1 GB) | 11,5 | 11,4 GB (0 % lokalnie) |
| `~/grant_tft/droga95` (poza `dane/`; w tym `pylib` — biblioteki do zadań) | 10,0 | ~4,7 GB |

## E. Narzędzia (nie zbiory, nie wyniki)

| Ścieżka | GB | Pliki | Zalecenie |
|---|---|---|---|
| `~/venvs/rfdetr_nano_clean_1_10_1` | 9,4 | 39 tys. | zostawić — bez niego żadne zadanie ViT na Edenie nie ruszy |
| `~/.cache` (huggingface, torch) | 0,5 | 0,4 tys. | zostawić — wagi DINOv2 i modeli |
| `~/.conda` | 0,3 | 10,7 tys. | zostawić |

## Zalecana kolejność (po zgodzie)

1. Pobrać lokalnie wszystkie małe pliki wyników (raporty, JSON, CSV, logi < 1 MB): ~1,4 GB, 12 tys. plików —
   wtedy z Edenu znikają tylko duże pliki binarne.
2. Pobrać wynik `1817712_2` (s43 bez k06) i domknąć tabelę pętli Large.
3. Decyzja o zadaniu 1810406 (równoległa sesja): `scancel`, jeśli usuwamy `2026-09-24_koopman_fastgan_petla`.
4. Usunąć B, potem C; D i E tylko po osobnej decyzji. Zbiory A przenieść do jednego katalogu `~/smallvit_data`.

## Wykonanie (2026-09-28, po zgodzie użytkownika „usuwaj to” dla grup B i C)

1. Małe pliki wyników (< 1 MB oraz pliki tekstowe < 50 MB i archiwa kodu) z usuwanych katalogów pobrano do
   `F:\Studia\PhD_projekt\VIT\eden_archiwum\smallvit_runs_raporty_2026-09-28.tar.gz` (1,3 GB, 10 435 plików;
   spis archiwum sprawdzony z listą: brak 0, zły rozmiar 0). W nim m.in. raport zadania 1817712_2 (Large s43 bez k06).
2. Końcowy FastGAN (`checkpoint_050000.pth`) skopiowano do `~/smallvit_runs/2026-09-28_maly_vit_od_zera/wejscia/fastgan/`
   (suma SHA256 zgodna z lokalną) przed usunięciem katalogu `2026-09-23_fastgan`.
3. Usunięto 14 348 plików z `~/smallvit_runs` według jawnej listy (bez plików pasujących do danych: `data/`,
   `datasets/`, `images/`, `probe/`, `pool_manifest`, `.jpg`), oraz całe `~/.vscode-server` i `~/.roboflow`.
   Z listy zdjęto dwa archiwa, które okazały się zbiorami uczącymi: `2026-09-22_dobor_klatek_stan_sieci/context_train.tar`
   (57 073 obrazy + pool_manifest, 4,9 GB) i `2026-09-23_fastgan/eden_inputs.tar` (1044 wycinki narzędzia dla FastGAN).
4. Anulowano zadanie 1810406 (równoległa sesja, wiszące od 2026-09-24) na wyraźną prośbę użytkownika.

Kwota po: **52,2 GB z 200 GB** (było 193,2), 109 429 plików (było 148 036). Nie ruszano: `~/grant_tft`, `~/venvs`,
`~/.cache`, `~/.conda`, plików w katalogu domowym.
