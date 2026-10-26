# Mosty do fizyki: replikacja na 24 seedach i ablacje (6 wariantów × 3 seedy)

**Data:** 2026-10-26, 13:04 (Europe/Warsaw). **Typ:** część 1 = pomiar na istniejących checkpointach (bez treningu); część 2 = 18 treningów
od zera na Edenie z pomiarem w tym samym zadaniu. **Predykcje:** `PREDYKCJE_PRZED_POMIAREM.md` (zapisane przed uruchomieniem).

## Cel

Sprawdzić na wielu niezależnych biegach i na klatkach spoza treningu, czy tezy o MałymViT jako „protokole mieszania na sferze”,
zmierzone dotąd na jednym biegu (seed 47) i jednej klatce, są ogólne, oraz sprawdzić je przyczynowo ablacjami:

- **A.** Trening czyni pole MLP nienormalnym (ścinanie, nie wir ani pułapka); każdy token jest siodłem; wokół narzędzia pole bardziej ściska.
- **B.** Uwaga ściska chmurę tokenów (konsensus na sferze, ruch styczny), MLP rozdziela klasy; trening przenosi ściskanie z uwagi do MLP;
  separacja klas w stanie końcowym przewiduje IoU.
- **C.** Kolejność uwaga→MLP ma znaczenie tylko w blokach 4–5 (defekt komutatora), tam, gdzie zapada decyzja o narzędziu.

## Rodzice i źródła

- `experiments/schur_qr_jakobiany_mlp_20261024__10-03-40` (teza A, definicje miar Schura i Q–R; tu bez scipy: ‖N‖² = ‖J‖² − Σ|λ|²).
- `experiments/kontrakcja_komutator_polkrokow_20261025__02-12-46` (tezy B i C, definicje kontrakcji, porządku, separacji, komutatora).
- Checkpointy części 1: kampania `outputs/kompensacja_funkcjonalna_dgx2_20261017` (lokalnie `download/unpacked_20261018_035119/replications`,
  na Edenie `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017/replications/s<seed>/weights`):
  24 seedy 3047, 4047, …, 26047; każdy 65 640 kroków (60 epok, epoka = 1094 kroki) tym samym przepisem co historia seedu 47;
  12 checkpointów: 0, 300, 1000, 3000, 6000, 10 000, 16 000, 24 000, 33 000, 44 000, 55 000, 65 640. SHA256 wag zapisuje każde zadanie
  do `Wyniki/Dane/pomiar/s<seed>/WAGI_SHA256.txt`. Model: `Kod_eksperymentu/model.py` (kopia `code/model.py` kampanii, SHA256 0570c957…).
- Klatki: `input/dataset.json` kampanii: 500 klatek TEST z 25 filmów spoza treningu (20 na film), w tym 50 diagnostycznych
  (indeksy 5 i 15 w filmie); używane są klatki z co najmniej jedną ramką GT (cystotom). Obrazy `images/*.npz` (klucz `rgb`, 320 × 480).
- Dane treningu części 2: `input/training_manifest.jsonl` kampanii (35 000 klatek TRAIN, SHA256 1c224daa…) i archiwa zip
  `smallvit_data/cataract_lmm/3_Object_Tracking/annotations` na Edenie.

## Procedura

**Część 1** (`Kod_eksperymentu/pomiar.slurm`, tablica 24 zadań, 1 × A100 + 16 CPU): dla każdego seedu i checkpointu
`pomiar_mosty_eden.py` liczy na 500 klatkach TEST: 13 stanów przejścia, miary B (odległości par na sferze i surowe, parametr porządku,
separacja klas, ranga efektywna, udział styczny ruchu, zachowanie 5 sąsiadów), miary C (S, R, P, tylko uwaga, tylko MLP; defekt
komutatora per token i skutek dla wyjścia), IoU i pewność ramki szczytu, AP@0,5; na 50 klatkach diagnostycznych miary A z pełnych
jakobianów 192 × 192 (wszystkie tokeny narzędzia + 60 losowych tła, 6 bloków): nienormalność Henriciego, ‖Ω‖/‖S‖, tr J, znaki widma,
|1+λ| > 1, Q, R. Wyjście: `Wyniki/Dane/pomiar/s<seed>/{wyniki.npz, podsumowanie.json}`.

**Część 2** (`Kod_eksperymentu/ablacje.slurm`, tablica 18 zadań, 1 × A100 + 16 CPU): `trening_ablacje.py` trenuje wariant
(base, parallel, frozen_mlp, frozen_attn, no_wd, shuffled_labels) na seedzie 3047, 4047 albo 5047 tym samym przepisem i siatką
checkpointów co kampania (kod wyprowadzony z `train_replicate.py` kampanii, ustawienia deterministyczne), potem ten sam pomiar na jego
12 checkpointach. Wyjście: `Wyniki/Dane/ablacje/<wariant>_s<seed>/{weights/, restart.pt, loss.jsonl, COMPLETE.json, pomiar/}`.

**Ocena:** `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` → `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}` (PASS/FAIL z liczbami).

## Metryki

Definicje w docstringu `pomiar_mosty_eden.py`; progi w `PREDYKCJE_PRZED_POMIAREM.md`.

## Ograniczenia

- Wszystkie seedy części 1 trenowano na jednym podziale danych (35 000 klatek TRAIN); klatki TEST pochodzą z 25 filmów jednego zbioru.
- Siatka 12 checkpointów jest rzadka po 10. epoce; trend wczesny (0–10 000 kroków) ma 6 punktów.
- Jakobiany liczone dla ≤ ~80 tokenów na klatkę (wszystkie narzędzia + 60 tła), nie dla wszystkich 600.
- Ablacje mają po 3 seedy: wystarczają do testu znaku 3/3 i do porównania par z base, nie do przedziałów ufności.
- Wariant `parallel` jest przy pomiarze „swoim” przejściem (MLP z wejścia bloku), więc defekt komutatora ‖S−P‖ jest w nim z definicji 0.
- Trening deterministyczny na A100 nie jest bitowo zgodny z seedami kampanii liczonymi na H100; base w części 2 jest własną kontrolą.

## Odtwarzanie

```
python experiments/eden.py init mosty_fizyki_24seedy_ablacje_20261026__13-04-31
python experiments/eden.py push mosty_fizyki_24seedy_ablacje_20261026__13-04-31
python experiments/eden.py submit mosty_fizyki_24seedy_ablacje_20261026__13-04-31 --script Kod_eksperymentu/pomiar.slurm
python experiments/eden.py submit mosty_fizyki_24seedy_ablacje_20261026__13-04-31 --script Kod_eksperymentu/ablacje.slurm
python experiments/eden.py watch mosty_fizyki_24seedy_ablacje_20261026__13-04-31 --pull-on-complete
python Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
```

Próby dymne przed tablicami: `--sbatch-arg=--array=0 --script-arg=proba` (pomiar) i `--sbatch-arg=--array=1 --script-arg=smoke` (ablacje).

## Stan

Zobacz `Wyniki/Dane/Eden/STATUS_EDEN.txt` i koniec tego pliku (uzupełniany po pomiarze).

- 2026-10-26 13:12–13:45: próby dymne na dgx-3. Pierwsza próba pomiaru (1930564_0) przekroczyła 30 min, bo 16 procesów puli × 16 wątków
  BLAS dławiło węzeł (0,8 s na macierz 192 × 192); po ustawieniu BLAS jednowątkowego próba 1930572_0 zajęła 42 s (A: 10 s na 4 klatki
  × 6 bloków). Próba ablacji `smoke` (1930573_1, wariant base, seed 4047, 64 klatki, 4 kroki, pomiar na 2 checkpointach) ukończona
  w 61 s. Wyniki prób: `Wyniki/Dane/proby_dymne/` i `Wyniki/Dane/ablacje_smoke/`; nie wchodzą do oceny.
- 2026-10-26 13:46: zgłoszone pełne tablice: pomiar 24 seedów (1930576, zadania _0…_23) i ablacje 18 biegów (1930577, _0…_17);
  partycja short, 1 × A100 + 16 CPU na zadanie.
