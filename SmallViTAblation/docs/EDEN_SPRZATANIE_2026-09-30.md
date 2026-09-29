# Eden: co można usunąć (analiza 2026-09-30, nic nie usunięto)

Pytanie użytkownika: „sprawdź, ile tam można usunąć”. Pomiar: `lfs quota -p 1299`, `du` na każdym katalogu
(bez `du -sh ~`), pełna lista plików ośmiu katalogów wyników (84 tys. plików) porównana z dyskiem lokalnym po ścieżce
i rozmiarze. Poprzednia analiza i sprzątanie: `EDEN_SPRZATANIE_2026-09-28.md` (wtedy 193 → 52 GB).

Stan: **117 GiB z 200 GiB, 168 499 plików z 200 000**. Od 28.09 przybyło 65 GB, prawie wszystko to zrzuty latentu
pętli małego ViT. Działa tylko uczenie dyfuzji (1818355, drugi fragment 1818356 czeka).

## A. Bez żadnej straty — ~4,9 GB i ~57 tys. plików

| Ścieżka (`~/smallvit_runs`, chyba że inaczej) | GB | Pliki | Dlaczego bez straty |
|---|---|---|---|
| `2026-09-15_tracking_static36/data/tracking_static36/` (rozpakowane) | 2,04 | 22 724 | to samo co `tracking_static36.tar` (22 680 obrazów); zadania rozpakowują archiwum do `/dev/shm`; do katalogu odwołuje się tylko stary pilot `eden/submit_tracking_static_pilot.sh` |
| `2026-09-14_tracking_nano/SmallViTAblation/data/tracking_calibration/` (rozpakowane) | 1,13 | 12 555 | to samo co `tracking_calibration.tar` |
| `~/.vscode-server` | 1,29 | 8 550 | serwer VS Code Remote, instaluje się sam przy połączeniu (używany 29.09, usuwać bez otwartego połączenia) |
| `2026-09-21_koopman_langevin/package/frames` | 0,43 | 4 800 | klatki wycięte z filmów Cataract-LMM (źródłowe archiwa filmów są lokalnie); resztki po sprzątaniu 28.09, nic ich nie używa |
| `~/.triton` | 0,11 | 8 337 | pamięć podręczna kompilatora, odtwarza się sama |

To zwalnia głównie **pliki**: z 168,5 tys. do około 111 tys.

## B. Wyniki z kopią lokalną — ~11,9 GB

| Ścieżka | Razem GB | Lokalnie GB | Tylko Eden | Uwagi |
|---|---|---|---|---|
| `2026-09-28_maly_vit_wektor` | 7,38 | 6,10 | 1,19 GB, 63 pliki (wagi `wspolny/k#.pt`) | lokalnie `experiments/maly_vit_wektor_tozsamosci/wyniki_eden` |
| `2026-09-28_maly_vit_od_zera/wyniki` | 7,99 | 5,80 | 2,19 GB, 77 plików | **zostawić `wejscia/` (0,73 GB) i `deps/`** — używają ich `rownanie_eden.slurm` i `dyfuzja_narzedzie/eden/generuj_eden.slurm` |

## C. Tylko na Edenie — usunięcie = utrata surowych danych (~48 GB)

| Ścieżka | GB | Tylko Eden | Co to jest |
|---|---|---|---|
| `2026-09-29_maly_vit_sila_it2` | 42,75 | 39,0 GB | `LATENT.npz` co obrót: 5 ramion × 10 ziaren × 28 = 1400 plików po ~26 MB (36,7 GB) + wagi `wspolny/k#.pt` (1,4 GB) |
| `2026-09-28_maly_vit_sila` | 11,59 | 10,3 GB | to samo dla pierwszej wersji (9,2 GB zrzutów); **`petla_s*/wspolny/k02.pt` zostawić** — to wagi ViT dla `generuj_eden.slurm` dyfuzji |

Lokalnie jest `experiments/maly_vit_petla_sila_tozsamosci` (2,3 GB: raporty, obrazy, analiza), ale nie zrzuty latentu.
Na dysku F: jest 14 GB wolnego, więc całych zrzutów nie da się pobrać. Zrzuty to surowiec do analiz Koopmana
i Langevina na tych pętlach (jak `maly_vit_rownanie_latentu/t1_bez_sterowania.py` na zrzutach wersji 2); bez nich
takie analizy wymagają ponownego biegu kampanii (~30 min na ziarno na Edenie).

## Zostają

`~/grant_tft` 29,2 GB (projekt grantowy, osobna decyzja), `~/venvs` 9,5 GB, `~/.conda`, `~/.cache`, archiwa zbiorów
(`tracking_static36.tar` 2,1 GB, `tracking_calibration.tar` 1,15 GB, `2026-09-22_dobor_klatek_stan_sieci/context_train.tar`
5,0 GB — z niego uczy się dyfuzja, `2026-09-23_fastgan/eden_inputs.tar`), sondy w `2026-09-24_koopman_gesta_historia/probe`,
trwająca `2026-09-29_dyfuzja_narzedzie` (2,9 GB).

## Podsumowanie

A: 4,9 GB i 57 tys. plików bez straty. A + B: 16,8 GB. A + B + C: około 65 GB, kwota spadłaby do ~52 GB.

## Wykonanie (2026-09-30, po zgodzie użytkownika „usuwaj”)

Usunięte: rozpakowane kopie `tracking_static36/` i `tracking_calibration/` (przed usunięciem sprawdzone: każdy ich plik
jest w archiwum `.tar`), `2026-09-21_koopman_langevin/package/frames`, `~/.triton`, oraz katalogi ramion
(`bez`, `koopman`, `losowe`, `current`, `kolejne`, `tozsamosc`) w `wyniki/petla_s*` czterech pętli:
`2026-09-28_maly_vit_od_zera` (30), `2026-09-28_maly_vit_wektor` (30), `2026-09-28_maly_vit_sila` (50),
`2026-09-29_maly_vit_sila_it2` (100).

Zostawione celowo: `wspolny/` każdej pętli (wagi rozgrzewki; `wspolny/k02.pt` z `maly_vit_sila` wczytuje
`dyfuzja_narzedzie/eden/generuj_eden.slurm`, a `wspolny` z `maly_vit_wektor` kopiują skrypty pętli siły jako start),
`WEKTOR.json` i `WEKTOR.npz`, `maly_vit_od_zera/wejscia` i `deps` (wejścia wielu skryptów), `~/.vscode-server`
(serwer VS Code był w chwili sprzątania połączony).

Kwota: 117 GiB → 56,7 GiB, pliki: 168 501 → 70 504.
