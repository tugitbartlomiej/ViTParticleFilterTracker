# Pobrane z Edena 2026-10-16

Zasady: nic nie skasowano ani na Edenie, ani lokalnie; nic nie zapisano na dysku I:; limit F: >= 20 GB wolnego (po pobraniu ok. 24 GB).
Kazdy pobrany plik porownany SHA-256 (policzone na Edenie vs lokalnie), 0 niezgodnosci, 0 plikow "rozne". Skrypt, logi, listy pominietych: `POBRANE_Z_EDENA_2026-10-16_pliki/`.
Tryb: strumien `ssh ... tar cf - -T - | tar xf -`; pliki >= 5 MB juz obecne lokalnie (ta sama SHA) pominiete.

| katalog na Edenie | rozmiar Eden | status lokalny przed | decyzja | gdzie / co |
|---|---|---|---|---|
| 2026-10-13_deit_cifar10 | 64 GB | brak (tylko CIFAR i kod) | pobrac czesc 21 GB | `assets/DeiTTiny_CIFAR10_2026-10-13/` : 3 biegi glowne (gesty, pelny, dane, 148 wag) + 3 kontrole (gesty, pelny, wagi k<=100) |
| 2026-10-10_szklo_pokretla | 40 GB | czesciowo (dlugi_lr_5e-4_rp, lr_5e-4_rp, sgd_*) | pobrac 37 GB | `.../05_biegi_pokretla_eden/` : wszystkie male biegi + gesty 4 biegow *_zostaw/bezpoz + caly klas_vit_od_zera_60ep |
| 2026-10-08_k1000_podmiana150_sd | 669 MB | czesciowo (ziarna w 09_) | pobrac | `.../09_k1000_biegi_eden/` |
| vit_k1000_wzorce_pydmd_deeptime_20261009 | 648 MB (524 MB .venv, 110 MB .pip-cache) | brak | pobrac wyniki i kod (~30 MB) | 09_k1000_biegi_eden |
| 2026-10-10_szklo_koloidalne | 241 MB | w wiekszosci (pomiar60, szklo BD) | pobrac brakujace 107 MB | `.../04_szklo_koloidalne_BD/eden_symulacje/` (os_szklo_krysztal_60ep itd.), pomiar60 zgodny 7/7 |
| vit_k1000_replay900k_co5_20261009 | 224 MB | production lokalnie | pobrac bez smoke | 09_k1000_biegi_eden |
| vit_k1000_1500_pelne_latenty_90_20261008 | 120 MB | production lokalnie | pobrac bez smoke | 09_k1000_biegi_eden |
| vit_k1000_wspolna_korekta_patch_20261009 | 37 MB | brak | pobrac | 09_k1000_biegi_eden |
| 2026-10-09_maly_vit_od_zera_35000_21epok | 51 MB | w pelni (MalyViT_35000_od_zera_s47) | nie pobierac wiekszosc (6 plikow >=5 MB zgodnych lokalnie) | Eden_stare_male... |
| 2026-10-13_fdt_temperatura, koopman_pulapki | 5 MB | tylko kod | pobrac | `.../10_pomiary_pulapek_eden_2026-10-13/` |
| stare male (2026-09..10-07, outputs, experiments) | ok. 150 MB | w wiekszosci | pobrac to co brak | `assets/Eden_stare_male_przebiegi_2026-10-16/` |
| smallvit_data/imagenette | 386 MB | czesciowo (val 3925 plikow) | pobrac reszte 261 MB | `assets/Dane_Imagenette/` |
| smallvit_data/cataract_lmm | 48 GB | kompletny: 175 plikow Eden = lokalne po nazwie i rozmiarze (zip-y wideo identyczne rozmiarowo; brak lokalnie tylko SHA256SUMS 17 KB); sd576_* na Edenie puste (40 KB) | nie pobierac | Dataset/Cataract-LMM |
| venvs 9,7 GB, grant_tft | | | nie pobierac | |

## Zostaje tylko na Edenie
- deit_cifar10: ok. 43 GB wag k*.pt (po ~940 na bieg glowny, ~265 na kontrole; pobrano k<=1000 co 10/100 i co 2000), `wyniki/proba` (324 MB, bieg probny).
- szklo_pokretla: gesty `cnn_lr_5e-4_rp_zostaw` (4,1 GB; brak miejsca w budzecie F:).
- k1000: results/smoke*, wyniki/proba, deps/timm, .venv, .pip-cache.
- venvs, cataract_lmm (kopia lokalna jest).
