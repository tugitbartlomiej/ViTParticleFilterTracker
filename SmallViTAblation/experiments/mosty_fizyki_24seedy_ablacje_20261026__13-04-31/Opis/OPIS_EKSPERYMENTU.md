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
- 2026-10-26 14:25: węzły z wolnymi GPU (dgx-1, dgx-2, dgx-3) nie miały wolnych CPU, więc zadania z 16 CPU czekały. Oczekujące
  zakresy anulowane (`scancel 1930574_[8-23] 1930577_[0-17]`; ukończone i liczące się zadania 1930574_0…_7 zostały) i zgłoszone
  ponownie z mniejszą liczbą CPU: pomiar 1930605 (zadania 8–23, 2 CPU; pula 16 procesów na 2 rdzeniach wydłuża część A),
  ablacje 1930606 (0–17, 3 CPU). Numery zarejestrowane w narzędziu eden.py; anulowane zakresy nie policzyły niczego.
- 2026-10-26 15:30: wstępny odczyt na 8 ukończonych seedach (pobrane do `Wyniki/Dane/pomiar/`, przed końcem tablicy). Poprawka
  w `ocena_predykcji.py` (nie w progach): lista `miary_A` w npz zaczyna się od 5 etykiet wierszy, przez co pierwsza wersja oceny
  czytała dla P1/P2/P3/P10 złe kolumny (np. liczbę zespolonych λ zamiast ‖N‖/‖J‖). Po poprawce kolumny zgodne z docstringiem pomiaru.
  Progi z PREDYKCJE_PRZED_POMIAREM.md bez zmian; ocena końcowa po 24 seedach.

- 2026-10-26 17:25: część 1 ukończona (24/24 zadania COMPLETED, 19–40 min każde), wyniki pobrane. Ocena na 24 seedach (progi bez zmian):
  P1 24/24 (‖N‖/‖J‖ bloku 4: 0,899–0,939 przy k65640, 0,759–0,768 przy k0), P2 24/24, P4 24/24, P6 24/24 (Σ Δsep MLP 0,88–1,35 wobec
  uwagi 0,10–0,44), P8 24/24, P10 22/24 (2 seedy: ρ(sep, IoU) przy k0 = −0,35/−0,37, reszta warunków spełniona), P7b 1/1 (ρ = 0,50).
  FAIL: P3 11/24 (tr J narzędzia nie jest systematycznie niższy niż tła), P5 14/24 (uwaga bloków 1–5 styczna 0,94–0,99 we wszystkich
  seedach; blok 6 schodzi do 0,80–0,90 w 10 seedach), P7a 0/24 (ρ(sep₁₂, IoU) per klatka 0,34–0,48, próg 0,5 za wysoki), P9 10/24
  (szczyt defektu komutatora w bloku 3 w 13 seedach, 4–5 w 10, 2 w 1). AP@0,5 przy k65640: mediana 0,879 (0,827–0,902).
- 2026-10-26 17:25: część 2: 9/18 biegów ukończonych (trening ~55–75 min + pomiar ~30 min na A100), 9 się liczy. Wstępnie (dane pobrane bez wag):
  A2 0/2 (parallel AP 0,895/0,873 ≥ base 0,888/0,864: kolejność uwaga→MLP nie ma znaczenia dla AP), A3 0/1 (frozen_mlp s5047 AP 0,884
  = base; separację przejmuje uwaga: Σ Δsep uwaga 0,73, MLP −0,06; ‖N‖/‖J‖ bloku 4 zostaje 0,76), A4 0/3 (frozen_attn: iloczyn
  kontrakcji uwagi 0,62–0,66 → 0,92–0,97, czyli uwaga o zamrożonych wagach staje się prawie identycznością, bo zmienia się rozkład
  wejść; AP 0,861/0,861/0,786; MLP nadal rozdziela 0,52–0,85, ‖N‖/‖J‖ 0,95–0,97). Predykcja A4 była źle postawiona: kontrakcja uwagi
  zależy od stanu tokenów, nie tylko od wag. Ocena końcowa A1–A6 po 18 biegach.
- 2026-10-26 18:26: część 2 ukończona (18/18 COMPLETED, 1 h 09 – 1 h 46 każde). Ocena końcowa `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}`:
  część 1 jak wyżej; część 2: **A5 PASS 3/3** (no_wd: ‖N‖/‖J‖ bloku 4 0,90–0,93, nienormalność nie pochodzi z weight decay),
  A1 1/3 (base na A100 odtwarza P1, P4, P6; pada tylko P9, szczyt komutatora w bloku 3 i 5), A2 0/3 (parallel AP 0,895/0,869/0,873
  wobec base 0,888/0,885/0,864: kolejność uwaga→MLP bez znaczenia dla AP; separację dzielą uwaga i MLP po równo, +0,44…+0,67 / +0,27…+0,50),
  A3 0/3 (frozen_mlp AP 0,866/0,857/0,884 = base; Σ Δsep uwaga +0,42…+0,80, MLP −0,06…+0,01; ‖N‖/‖J‖ zostaje 0,76), A4 0/3
  (frozen_attn: iloczyn kontrakcji uwagi 0,62–0,66 → 0,92–0,97, AP 0,861/0,861/0,786; MLP rozdziela +0,52…+0,85, ‖N‖/‖J‖ 0,95–0,97),
  A6 0/3 (shuffled_labels: AP 0,005–0,009 zgodnie z oczekiwaniem, ale ‖N‖/‖J‖ bloku 4 rośnie do 0,92–0,96 i Σ Δsep MLP = +0,23…+0,28;
  nienormalność pola MLP rośnie także przy zapamiętywaniu losowych etykiet, więc nie jest sygnaturą nauczenia zadania).
- 2026-10-26 18:26: konfundator P7. ρ(sep₁₂, IoU) ≈ 0,34–0,48 w P7a liczone było po wszystkich 12 checkpointach razem i odzwierciedla
  trajektorię treningu (s3047: mediana sep₁₂ 1,13 → 2,33 i mediana IoU 0 → 0,80 rosną wspólnie). Per klatka przy k65640 ρ = −0,16…+0,11
  (mediana −0,035, 24 seedy; ablacje −0,07…+0,14, shuffled_labels +0,24). Separacja klas nie przewiduje IoU klatki; jest miarą postępu
  treningu. Dodatkowo sep₁₂ silnie zależy od liczby tokenów narzędzia (ρ = −0,71), a IoU od niej rośnie (ρ = +0,36). Wniosek do
  raportu: P7a odrzucona z powodu, a P7b (ρ = 0,50 między seedami, 24 punkty) to słaby dowód.
- Interpretacja całości: tezy A i B (nienormalne ścinanie w MLP, uwaga ściska / MLP rozdziela, trening przenosi ściskanie) replikują się
  na 24 seedach jako opis wyuczonego rozwiązania, ale ablacje pokazują, że to podział pracy z wyboru, nie z konieczności: sieć
  kompensuje zamrożenie MLP (uwaga przejmuje separację) i zamrożenie uwagi (MLP robi wszystko, uwaga staje się identycznością), a blok
  równoległy nie traci AP. Teza C (kolejność w blokach 4–5) odrzucona. Nienormalność rośnie także bez weight decay i przy losowych
  etykietach. Lokalizacja w narzędziu (P3) nie replikuje się.
- 2026-10-26 18:31: na zlecenie użytkownika zapisane `Wyniki/Krótki_Wniosek/WNIOSEK.md` i `Wyniki/Raport/RAPORT.md` (tabele predykcji, miar A/B/C po 24 seedach, ablacji, rozbiór konfundatora P7). Wykresy nie były zamawiane.
