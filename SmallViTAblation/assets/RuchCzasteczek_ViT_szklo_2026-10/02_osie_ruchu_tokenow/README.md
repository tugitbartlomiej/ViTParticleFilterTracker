# Osie ruchu tokenów (format aplikacji, zakładka „Ruch zbioru”)

Każdy podfolder to jedna oś czasu w formacie aplikacji (schemat 2): `manifest.json` (kroki, SHA-256 klatek, zdjęcia, parametry), klatki `step_NNNN.q16` albo `.f32` (cząstka = token; [N, 4] = x, y, z w PCA3 z kroku 0 i norma przesunięcia w pełnych 768D), `metrics.npy` (pewność, IoU, długość kroku na zdjęcie i krok), `initial_pca.npz` (baza PCA), `images.json`, czasem `pca32_podzbior.npz` (32 składowe dla podzbioru zdjęć) i `OCENY_POMIAR.json`. Wczytanie: aplikacja `app/app.py`, zakładka „Ruch zbioru”, otwarcie osi z podfolderu; w Pythonie `app/ruch_zbioru.py` (kopia w `07_kod/lokalnie/app_format_osi`). Oryginały leżą w `outputs/ruch_czasteczek_gui/`.

## A_35000_4000zdjec_do_21_epoki

Model 35 000 od zera (asset `MalyViT_35000_od_zera_s47`). 3000 zdjęć TRAIN + 1000 VALID, 341 checkpointów do 21. epoki (co 10 kroków do 1000, potem co 100, plus końce epok). Pomiar na Edenie (`pomiar_eden.py`). Główny materiał analiz TRAIN/VALID.

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261010T_natural_eden_pelny_3000tr_1000val` | complete | 4000 | 2 400 000 | 341 | 0–22974 |  |  | q16 | 9.02 GB |

## B_35000_100zdjec_konce_epok_do_21

Model 35 000. 100 losowych klatek uczących (`outputs/vit_35000_losowe100_20261009`), końce epok 0–21.

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261009T162654_natural_78cd84bd6edd` | complete | 100 | 60 000 | 22 | 0–22974 | 1094 | 32 | q16 | 10.9 MB |

## C_35000_20zdjec_konce_epok_do_60

Model 35 000. 20 klatek, końce epok 0–60 (pełne 60 epok, wagi z assetu `MalyViT_35000_od_zera_s47/wagi`).

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261009T221404_natural_8083acc06418` | complete | 20 | 12 000 | 61 | 0–65640 | 1094 | 20 | q16 | 6.1 MB |

## D_35000_5zdjec_konce_epok_do_21

Model 35 000. 5 klatek, końce epok 0–21 (podgląd).

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261009T172905_natural_5de07a0979e3` | complete | 5 | 3 000 | 22 | 0–22974 | 1094 | 32 | q16 | 0.7 MB |

## E_35000_100zdjec_uklad_klatki_gesty

Oś pochodna, nie pomiar checkpointów: gęsty zapis 100 sond biegu 35 000 (folder 01) w układzie względem klatki (CR, odjęty wspólny ruch 6 sąsiadów), PCA32, kroki 300–22 970. Skrypt `eksport_osi_szkla.py`.

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261010T_natural_szklo_klatka_35k` | complete | 100 | 60 000 | 1362 | 300–22970 | 1094 | 32 | q16 | 0.66 GB |

## F_K1000_1500zdjec_kroki_0-1000

Inny model: gęsta historia K1000 (`experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi`, odniesienie `MalyViT_K1000_SD_conditioning`). 1500 zdjęć `MANIFEST_1500`, kroki 0–1000. Pomiary różnią się co ile kroków i wielkością paczki; `_q16` to kopie 16-bitowe tych samych osi.

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261009T080332_natural_0051d7971abb` | complete | 1500 | 900 000 | 101 | 0–1000 | 10 | 4 | f32 | 1.46 GB |
| `20261009T080332_natural_0051d7971abb_q16` | complete | 1500 | 900 000 | 101 | 0–1000 | 10 | 4 | q16 | 0.74 GB |
| `20261009T081936_natural_07e68c7cdb6f` | complete | 1500 | 900 000 | 101 | 0–1000 | 5 | 40 | f32 | 1.46 GB |
| `20261009T081936_natural_07e68c7cdb6f_q16` | complete | 1500 | 900 000 | 101 | 0–1000 | 5 | 40 | q16 | 0.74 GB |
| `20261009T094824_natural_41cd2ac1904b` | complete | 1500 | 900 000 | 101 | 0–1000 | 5 | 20 | f32 | 1.46 GB |
| `20261009T094824_natural_41cd2ac1904b_q16` | complete | 1500 | 900 000 | 101 | 0–1000 | 5 | 20 | q16 | 0.74 GB |
| `20261009T110313_natural_fa3c69aee473` | complete | 1500 | 900 000 | 101 | 0–1000 | 5 | 20 | q16 | 0.74 GB |

## G_K1000_3zdjecia_kroki_0-1000

Model K1000, 3 zdjęcia, kroki 0–1000 co 10 (podgląd).

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261009T172911_natural_ef7da43f8027` | complete | 3 | 1 800 | 101 | 0–1000 | 10 | 32 | q16 | 1.7 MB |

## Z_proby_i_nieudane

Próby i przerwane pomiary z 2026-10-09: optymalizacja latentu (3 kroki), pomiary przerwane lub nieudane (status w tabeli). Zachowane dla kompletności historii, bez wartości analitycznej.

| oś | status | zdjęć | cząstek | kroków | zakres kroków | co ile | paczka | format | rozmiar |
|---|---|---|---|---|---|---|---|---|---|
| `20261009T074155_187b922fb12b` | complete | 1 | 600 | 3 | 0–2 | 1 |  | f32 | 0.1 MB |
| `20261009T074314_3c016cb1525d` | complete | 1 | 600 | 3 | 0–2 | 1 |  | f32 | 0.1 MB |
| `20261009T074735_22f7cb6bd7f7` | cancelled | 1 | 600 | 0 | – | 1 |  | f32 | 0.0 MB |
| `20261009T074951_0adff7c0b622` | complete | 1 | 600 | 3 | 0–2 | 1 |  | f32 | 0.1 MB |
| `20261009T080005_natural_519c59df006e` | failed | 4 | 2 400 | 0 | – | 10 | 4 | f32 | 0.1 MB |
| `20261009T080232_natural_7b818f3952bd` | complete | 4 | 2 400 | 2 | 0–10 | 10 | 4 | f32 | 0.1 MB |
| `20261009T204245_natural_96f17eeb2d14` | failed | 0 | 0 | 0 | – | 10 | 20 | q16 | 0.0 MB |
| `20261009T204259_natural_0ee40a5919c8` | failed | 0 | 0 | 0 | – | 10 | 20 | q16 | 0.0 MB |
| `20261009T204351_natural_c0b3cf5becbc` | failed | 0 | 0 | 0 | – | 1 | 20 | q16 | 0.0 MB |
| `20261009T204416_natural_0e37e73b2e5f` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T210112_natural_c727f17418e7` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T210113_natural_239c065edf93` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T210113_natural_5a239a1cf248` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T210655_natural_e088b26d8d42` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T210746_natural_7c6dc0f903ab` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T211814_natural_6aee55070e8b` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T213135_natural_3ebd304954d9` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
| `20261009T215142_natural_823bb9be8e7f` | failed | 0 | 0 | 0 | – | 1094 | 20 | q16 | 0.0 MB |
