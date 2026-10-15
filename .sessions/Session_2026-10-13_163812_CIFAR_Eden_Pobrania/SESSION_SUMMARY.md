# Session Summary: CIFAR-10 na Edenie, pobrania i porządki

## Metadata
- **Date:** 2026-10-13
- **Time:** 11:40–16:40
- **Status:** In Progress
- **Type:** Training / SSH

## Objective
- Wytrenować drugi model na drugim, publicznym zbiorze i zapisać cały przebieg uczenia (checkpointy, kroki), nie tylko gotowy model.
- Użytkownik: „dlaczego nie taki ViT jaki ja liczyłem, tylko na publicznych danych?” → dodany MałyViT na CIFAR-10.
- Policzyć kontrole przed publikacją; gdy hopper zajęty, puszczać na DGX (`short`, A100).
- Pobierać lokalnie, potem usuwać na Edenie to, co pobrane, i zapisywać w sesji, co pobrano i co usunięto.

## Context
Wszystkie wyniki układu laserów/pułapek/koloidów pochodziły z jednego biegu (s47) MałegoViT na danych użytkownika.
Eksperyment: `SmallViTAblation/experiments/deit_cifar10_20261013/` (README, kod). Eden: `~/smallvit_runs/2026-10-13_deit_cifar10/`.

## Actions Taken
1. Oryginalny CIFAR-10 z Toronto pobrany (MD5 c58f3010…349a zgodny); porównanie z npz z HuggingFace: train 50 000/50 000,
   test 10 000/10 000 obrazów z tą samą etykietą (train w innej kolejności). Wpis w `assets/Dane_CIFAR10/README.md`.
2. Trener `kod/trening_cifar_deit.py`: `--model malyvit` (MałyViT bez zmian, CIFAR ×4 do 128 px → 64 tokeny, 2 890 402 param.),
   `--losowe_etykiety`, `--wagi_co_po1k` (rzadszy zapis wag dla biegów kontrolnych). Test lokalny OK. Commity 62243ddf, e2e1fff8.
3. Biegi (wszystkie przepis s47: AdamW 5e-4, wd 0,05, paczka 32, 60 epok = 93 780 kroków):

| bieg (wyniki/…) | zadanie | GPU | stan | dokładność |
|---|---|---|---|---|
| s47_obciecie (DeiT-Tiny) | 1827646 | H200 | koniec 93 780 | 0,7496 |
| s47_bez_obciecia (DeiT-Tiny) | 1827647 | H100 | koniec 93 780 | 0,753 |
| maly_s47_obciecie | 1827690 → wznowiony 1827779 | A100 → H200 | stop przez kwotę na 81 276, wznowiony od e52 | 0,848 |
| maly_s48_obciecie (rzadkie wagi) | 1827696 → 1827780 | A100 → H200 | stop 64 083, wznowiony od e41 | 0,849 |
| maly_s47_wd0 (rzadkie wagi) | 1827697 → 1827781 | A100 → H200 | stop 57 831, wznowiony od e37 | 0,852 |
| maly_s47_losowe_etykiety (rzadkie wagi) | 1827698 → 1827782 | A100 → H100 | stop 57 831, wznowiony od e37 | 0,100 |

4. Porządki na Edenie (szczegóły niżej).

## Results

### Key Findings
- MałyViT (6 bloków, łodyga konwolucyjna) bije DeiT-Tiny (12 bloków, łatka 4) na CIFAR-10: ~0,85 vs 0,75.
- Brak obcięcia gradientu nie zmienia dokładności DeiT (0,750 vs 0,753).

### Issues Encountered
- Kwota Edena (200 GB): 4 biegi MałegoViT zatrzymały się z zapisem stanu (zabezpieczenie `MIN_STOP_GB`); wznowione `--wznow`.
- Dopasowanie plików po nazwie i rozmiarze jest zawodne dla wag (k*.pt mają te same nazwy i rozmiary w różnych biegach) —
  weryfikować tylko MD5/SHA.
- O 16:20 katalog `2026-10-09_maly_vit_od_zera_35000_21epok/wyniki` (s47, 14 GB) został usunięty z Edena NIE przez tę sesję
  (prawdopodobnie inna sesja; brak notatki). Kopie lokalne potwierdzone: 806 wag `pelny/wagi` (SHA256SUMS_EDEN = SHA256SUMS_LOKALNE
  w `assets/MalyViT_35000_od_zera_s47/`), 341 wag `gesty/wagi` (MD5 zgodne z sumami policzonymi na Edenie tuż przed usunięciem,
  kopia w `assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis/eden/gesty/wagi`), stany e59/e60 w
  `01_bieg_35000_gesty_zapis/eden/pelny/pelny/`. Kod i deps tego biegu zostały na Edenie (używa ich PYTHONPATH biegów CIFAR).
- Skrypt pobierania porównywał listy MD5 jako całość i odmawiał przez kolejność sortowania i lokalne pliki dodatkowe; dwa katalogi
  zweryfikowano ręcznie (`md5sum -c`). Poprawiona wersja: scratchpad `pobierz_usun2.sh` (sprawdza każdy plik z Edena osobno).

## Pobrania i usunięcia na Edenie (2026-10-13)
Rejestr na Edenie: `~/smallvit_runs/USUNIETE_Z_EDENU.txt` i `~/smallvit_runs/2026-10-10_szklo_koloidalne/USUNIETE_Z_EDENU_2026-10-13.txt`.

| godz. | katalog na Edenie (`~/smallvit_runs/…`) | rozmiar | kopia lokalna (`SmallViTAblation/…`) | weryfikacja | usunięty |
|---|---|---|---|---|---|
| ~14:00 | 2026-10-10_szklo_koloidalne/wyniki/ruch60_rp128_50tr_50val | 15 GB | assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val (była wcześniej) | MD5 7/7 | tak |
| 16:20 | 2026-10-09_maly_vit_od_zera_35000_21epok/wyniki | 14 GB | patrz „Issues” (była wcześniej) | SHA/MD5 | tak, NIE przez tę sesję |
| 16:29 | 2026-10-10_szklo_koloidalne/wyniki/wiazki_koloidy | 3,5 GB | outputs/porownanie_szklo_20261010/eden/wyniki/wiazki_koloidy (pobrano u192_20zdjec.npy 3,7 GB) | MD5 4/4 | tak |
| 16:37 | 2026-10-10_szklo_pokretla/wyniki/dlugi_lr_5e-4_rp | 7,7 GB | assets/RuchCzasteczek_ViT_szklo_2026-10/05_biegi_pokretla_eden/wyniki/dlugi_lr_5e-4_rp (pobrano 8,2 GB) | MD5 26/26 | tak |

**Zostało na Edenie bez kopii lokalnej** (nie usuwać przed pobraniem): `2026-10-10_szklo_pokretla/wyniki/klas_vit_od_zera_60ep_lr_5e-4`
(16 GB), `lr_5e-4_rp`, `bezpoz_lr_5e-4_rp`, `lr_2e-3_rp_zostaw`, `lr_5e-4_rp_z48_zostaw`, `lr_5e-4_rp_z49_zostaw`, `cnn_lr_5e-4_rp_zostaw`
(po 4,5 GB), `klas_proba_historia_wag` (0,2 GB), `2026-09-15_tracking_static36/data/tracking_static36.tar` (2,1 GB) oraz wszystkie
biegi CIFAR w `2026-10-13_deit_cifar10/wyniki/` (~60 GB).
**Miejsce po porządkach:** Eden 33 GB wolne; lokalnie F: 17 GB, E: 19 GB, C: 6 GB — dalsze pobrania wymagają miejsca lokalnego.

## Files Generated/Modified
- `SmallViTAblation/experiments/deit_cifar10_20261013/README.md`, `kod/trening_cifar_deit.py`
- `SmallViTAblation/assets/Dane_CIFAR10/README.md` (+ `cifar-10-python.tar.gz`)
- pamięć: `deit-cifar10-drugi-model-2026-10-13.md`

## Commands Used
```
sbatch --job-name=maly_cifar --partition=hopper-2,hopper,short deit_cifar.slurm maly_s47_obciecie --model malyvit --epoki 60 --obserwuj --historia_wag --stan_co 10 [--wznow]
... --wagi_co_po1k 1000 --wagi_co_po10k 1000 [--ziarno 48 | --wd 0 | --losowe_etykiety]
ssh eden-cluster "cd ~/smallvit_runs/<rodzic> && tar cf - <katalog>" | tar xf - -C <lokalnie>; md5sum -c <lista z Edena>
```

## Next Steps
- [ ] Doczekać końca 4 wznowionych biegów MałegoViT (zadania 1827779–1827782).
- [ ] Analizy (obrót igieł, pułapka/Jakobian, koloidy) liczyć na Edenie i pobierać tylko wyniki (MB), bo brak miejsca lokalnie.
- [ ] Biegi czekające na miejsce: MałyViT ziarno 49, DeiT ziarna 48 i 49.
- [ ] Ustalić z użytkownikiem dysk na pobranie biegów CIFAR i pozostałych biegów szkła.
