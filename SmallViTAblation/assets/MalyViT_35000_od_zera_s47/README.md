# Mały ViT od zera na 35 000 klatkach: pełna historia 60 epok, ziarno 47

Folder `wagi/`: 806 checkpointów `k<krok>.pt` jednego biegu uczenia od zera, od kroku 0 do 65 640.

## Trening

- Bieg na Edenie 2026-10-09, H100. Zadanie 1826056 (epoki 1–21) i wznowienie 1826076 (epoki 22–60) od
  `stan_e21.pt` z wagami, AdamW i generatorem. Kolejka permutacji zbioru po wznowieniu zaczęła się od nowa.
- Przepis biegu K1000 z gęstą historią bez zmian: `trening.Uczen(47)`, `default_rng(47)`, paczka 32, AdamW 5e-4,
  wd 0,05, rozgrzewka lr 300 kroków, potem stały lr, bf16, odbicie 0,5 i jasność/kontrast ±20 %.
- Dane: 35 000 klatek TRAIN Cataract-LMM Object Tracking z 117 filmów (`MANIFEST_35000.jsonl` eksperymentu), każda
  rozciągnięta do 480 × 320, ramki skalowane osobno w X i Y.
- Epoka = 1094 kroki (35 008 próbek). Koniec epoki e to krok 1094 · e.
- Kod i opis: `experiments/maly_vit_od_zera_35000_21epok_20261009` (README.md, STATUS.md, kod/).

## Zapisane kroki

| Kroki | Co ile | Plików |
|---|---|---|
| 0–1000 | 10 | 101 |
| 1000–65 600 | 100 | 646 |
| końce epok 1–60 (1094 · e) | 1 na epokę | 59 bez powtórzeń z siatką co 100 |

Każdy plik to `{'net': state_dict, 'krok': int, 'epoka': float}`, około 11,7 MB. Ładowanie:

```python
ck = torch.load(sciezka, map_location='cpu', weights_only=True)
model.load_state_dict(ck['net'], strict=True)      # MalyViT z experiments/.../kod/model.py
```

Ten sam format wczytuje aplikacja: historia „35 000 klatek od zera” w zakładce „Ruch zbioru” i w detekcji.

## Walidacja

mAP@0,5 na 3000 klatkach walidacji (żadna nie należy do treningu), `dane/OCENY.json`:

| Epoka | mAP@0,5 |
|---|---|
| 1 | 0,449 |
| 8–21, średnio | około 0,62 |
| 21 | 0,652 |
| 26 (najlepsza) | 0,653 |
| 60 | 0,608 |

Po około 8. epoce walidacja nie rośnie przy stałym lr, a od około 40. lekko spada. Strata uczenia dalej maleje.

## Integralność

- `SHA256SUMS_EDEN.txt`: sumy policzone na Edenie; `SHA256SUMS_LOKALNE.txt`: po pobraniu. Skrypt
  `pobierz_z_edena.sh` porównuje je plik po pliku i pomija pliki już zgodne.
- 61 plików (krok 0 i końce epok) pobrano wcześniej do katalogu eksperymentu z kontrolą sum i przeniesiono tutaj.
- Źródło na Edenie zostaje: `~/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok/wyniki/pelny/wagi`.

## Osobny bieg z gęstym zapisem

Bieg `gesty` (zadanie 1826161, 21 epok, ten sam przepis) ma własne 341 checkpointów na Edenie
(`wyniki/gesty/wagi`). Nie są tutaj: to inna trajektoria (inny przebieg na GPU), zapisana razem z ruchem sond.

## Zapisy ruchu tokenów tego modelu

Gęsty zapis tokenów, pomiary naturalne (4000 zdjęć do 21. epoki, 3000 + 1000 do 60. epoki), biegi pokręteł i analizy „ruch jak w szkle”
leżą w `../RuchCzasteczek_ViT_szklo_2026-10` (README tam).
