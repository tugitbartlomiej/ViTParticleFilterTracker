# Session Summary: Assets ruchu cząsteczek, opis układu, ViT od zera na Imagenette (60 epok)

## Metadata
- **Date:** 2026-10-11
- **Time:** ok. 16:30–17:30
- **Status:** In Progress (bieg Eden 1826880)
- **Type:** Mixed (Analysis, SSH, Training)

## Objective
1. „Wszystko, co było robione i liczone, w szczególności zapisy przejść i ruchów cząsteczek, skopiować do odpowiednich folderów w `assets` i opisać.”
2. „W jednym pliku bardzo szczegółowy opis układu: jak uczony był ViT, jak analizowano tokeny jako cząsteczki, jakie siły (BatchNorm, AdamW…),
   zasady, wyniki; w kolejnym pliku model 60 epok, jego gęsty przebieg i odnośniki do plików trajektorii z Edena, żeby na nich liczyć” — folder `OpisUkładu`.
3. „Wytrenować ViT na Edenie od zera na Imagenette.”
(Wcześniej w tej sesji, przed kompaktowaniem: linki literatury, ocena publikowalności, metryki Imagenette, relacja τ–klatka; pytania o 768 wiązek i środek drgań — odpowiedź w opisie układu, sekcje 7.8 i 9.1.)

## Context
Badanie „ruch tokenów ViT jak w szkle” (`outputs/porownanie_szklo_20261010`), dane rozproszone po experiments/, outputs/ i Edenie (część kasowana dla miejsca).
Konwencja z pamięci: trwałe przebiegi do `assets/<Model_bieg>/` z README.

## Actions Taken
1. Inwentaryzacja: lokalnie 27 GB gęsty zapis 35k, 17 GB osie aplikacji, 0,7 GB analiz; na Edenie ~30 GB tylko tam (pokrętła 15 GB, pomiar60 5,5 GB, szkło BD 1,1 GB, wagi biegu gęstego 3,8 GB). Dysk F: 107 GB wolnego.
2. Asset `assets/RuchCzasteczek_ViT_szklo_2026-10`:
   - `zloz_lokalnie.py`: tekst kopiowany, binaria jako twarde dowiązania NTFS (40 456 plików, 48,5 GB bez zajmowania miejsca), rejestr `KOPIA_LOKALNA.tsv`.
   - `pobierz_z_edena.sh`: części kod, szklo_bd, bieg35000, pomiar60, pokretla, ruch60; tar przez ssh + SHA256SUMS liczone na Edenie.
   - `opisz.py`: README każdego podfolderu z manifestów/metadanych; `STARSZE_EKSPERYMENTY_RUCHU.md` (indeks, bez kopii); `.gitignore` binariów.
   - Odsyłacz dopisany do `assets/MalyViT_35000_od_zera_s47/README.md`.
3. Wykryto równolegle działającą inną sesję/Codex: zadanie Eden `ruch60_rp` 1826867 (trajektorie 806 ckpt) i plik `OpisUkładu/01_OPIS_UKLADU.md` (Codex). Nie nadpisano; własna wersja w `OpisUkładu/Claude/`.
4. Dwa agenty Explore: rozkład sił w `app/` (pchniecie.py, przyczyny_ruchu.py, przeplyw_sily.py…) i katalog eksperymentów 28.09–11.10.
5. `OpisUkładu/Claude/01_UKLAD_CZASTEK_ViT.md`, `02_TRAJEKTORIE_60_EPOK_I_DANE.md`, `wczytaj_trajektorie.py` (przetestowany na danych lokalnych), `README.md`.
6. Imagenette: `kod/trening_klasyfikacja.py` + flaga `--historia_wag` (wagi co 10 do 1000, co 100, końce epok); kopia starej wersji na Edenie `trening_klasyfikacja.py.przed_historia_wag_20261011`; próba 1826879 OK; bieg 1826880.

## Results

### Key Findings
- Rozkład sił (z raportów): pęd AdamW 64–68 %, gradient paczki 31–35 %, BatchNorm ~1 % na krok (duży na epoki: +47 % vs +2,5 % MSD kierunków), weight decay 0 %; v(t+1)=0,87 v(t)+S_nowa wyjaśnia 96–98 %.
- Werdykt substancji: osobne, zamrożone, sztywniejące dołki sprzężone wspólnymi wagami (Jakobian), nie szkło/żel/turbulencja.
- 768 to wymiar przestrzeni, nie liczba „wiązek”; środek dołka = położenie przy uśrednionych w czasie wagach, dryfuje z uczeniem.
- `ruch60`: rp128 (806,100,600,128) f16, pelne768 (806,4,600,768), srednia768, normy768, oceny, R_rp128, META.

### Issues Encountered
- Heredoc w Bash psuje się na dłuższych skryptach → pliki pisane narzędziem Write.
- Sumy części szklo_bd: jedyna niezgodność to log `ruch60_rp_1826867.out` rosnący w trakcie kopiowania (ponowne pobranie odrzucone przez użytkownika).
- Pobieranie z Edena zatrzymane przez użytkownika: `ruch60` pobrany częściowo (rp128 niepełny), części bieg35000, pomiar60, pokretla NIE pobrane; opisz.py nie przegenerowany po pobraniu.
- Agent z zapisem „waiter” wystartował ruch60 przed końcem pokretla (grep na log z polskim znakiem).

## Files Generated/Modified
- `SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10/` (README.md i README podfolderów, zloz_lokalnie.py, pobierz_z_edena.sh, opisz.py, KOPIA_LOKALNA.tsv, .gitignore, STARSZE_EKSPERYMENTY_RUCHU.md, dane)
- `SmallViTAblation/assets/MalyViT_35000_od_zera_s47/README.md` (odsyłacz)
- `SmallViTAblation/OpisUkładu/Claude/{README.md, 01_UKLAD_CZASTEK_ViT.md, 02_TRAJEKTORIE_60_EPOK_I_DANE.md, wczytaj_trajektorie.py}`
- `SmallViTAblation/experiments/maly_vit_szklo_pokretla_20261010/kod/trening_klasyfikacja.py` (+ `--historia_wag`), kopia na Edenie
- pamięć: `assets-smallvit-przebiegi.md` (stan assetu)

## Commands Used
```bash
bash pobierz_z_edena.sh [kod|szklo_bd|bieg35000|pomiar60|pokretla|ruch60]
F:/Python/3.13/python.exe -X utf8 zloz_lokalnie.py ; F:/Python/3.13/python.exe -X utf8 opisz.py
ANALIZY=- ZOSTAW=1 sbatch --job-name=klas_proba_hw --time=0:30:00 klasyfikacja.slurm klas_proba_historia_wag 5e-4 2 16 --proba --historia_wag
ANALIZY=300:2188,16576:17760 ZOSTAW=1 sbatch --job-name=klas_vit60 klasyfikacja.slurm klas_vit_od_zera_60ep_lr_5e-4 5e-4 60 24 --historia_wag --gesto_od 16576
```

## Next Steps
- [ ] Odebrać wynik biegu 1826880 (dokładność po epokach, ~320 wag, gesty, ANALIZA dwóch okien) i pobrać do `assets/` (nowy podfolder Imagenette).
- [ ] Na zgodę użytkownika dokończyć pobieranie: ruch60 (rp128 niepełny), bieg35000, pomiar60, pokretla; potem `opisz.py` i weryfikacja SHA.
- [ ] Ocena klasyfikacji nowego biegu (`ocena_klasyfikacji.py`: top-1/top-5/mAP/ECE) na wybranych epokach.
- [ ] Analizy na `ruch60`: środek drgań i jego dryf, efektywna liczba „wiązek” (widmo przyrostów 768D), MSD/Fs w skali epok.
