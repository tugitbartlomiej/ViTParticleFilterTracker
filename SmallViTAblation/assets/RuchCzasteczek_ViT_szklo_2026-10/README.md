# Ruch cząsteczek (tokenów) małego ViT a szkło koloidalne: wszystkie zapisy, 2026-10-09…11

Jedno miejsce na zapisy ruchu tokenów i przebiegi uczenia z badania „ruch cząsteczek jak w szkle” (`06_analizy/porownanie_szklo_20261010/README.md`). Cząstka = token obrazu (600 na zdjęcie 480 × 320), położenie = 768 aktywacji GELU warstwy ukrytej MLP ostatniego bloku, czas = krok uczenia. Wagi modelu 35 000 (60 epok, 806 checkpointów) są w `../MalyViT_35000_od_zera_s47`.

## Zawartość

| folder | co zawiera | plików | rozmiar |
|---|---|---|---|
| `01_bieg_35000_gesty_zapis/` | Gęsty zapis tokenów biegu 35 000 (co 1 krok, PCA32 i pełne 768D), checkpointy biegu gęstego, stany do wznowienia. | 40 | 28.76 GB |
| `02_osie_ruchu_tokenow/` | Pomiary naturalne ruchu tokenów po checkpointach w formacie aplikacji (4000 zdjęć do 21. epoki, 20 zdjęć do 60., K1000). | 2 738 | 17.02 GB |
| `03_pomiar_naturalny_60_epok_eden/` | Pomiar 3000 TRAIN + 1000 VALID do 60. epoki: PCA32 podzbioru, metryki, porównanie TRAIN/VALID. | 1 | 0.0 MB |
| `04_szklo_koloidalne_BD/` | Kontrola dodatnia: symulacje szkła koloidalnego (dynamika browna), osie aplikacji i surowe wyniki. | 68 372 | 2.20 GB |
| `05_biegi_pokretla_eden/` | Biegi pokręteł: lr, gęstość, ziarna, SGD, CNN, warstwy, Imagenette, okno po 60. epoce; stany, zapisy tokenów, analizy, test jądra. | 1 | 0.0 MB |
| `06_analizy/` | Wszystkie analizy, JSON-y wyników, rysunki i animacje badania. | 272 | 1.44 GB |
| `07_kod/` | Kod z Edena i lokalny, którym powstały zapisy; moduły czytające format osi. | 65 | 1.0 MB |
| `08_literatura/` | Zbiór literatury (102 linki, 66 PDF). | 85 | 0.17 GB |

Każdy folder ma własne README z tabelą zawartości wygenerowaną z manifestów i metadanych (`opisz.py`). Wcześniejsze eksperymenty z ruchem cząsteczek, których tu nie kopiowano: `STARSZE_EKSPERYMENTY_RUCHU.md`.

## Jak powstała ta kopia

- `zloz_lokalnie.py`: części z `experiments/` i `outputs/`. Pliki tekstowe skopiowane, pliki binarne jako twarde dowiązania NTFS (ta sama treść, zero dodatkowego miejsca; usunięcie oryginału nie usuwa danych stąd). Rejestr każdego pliku: `KOPIA_LOKALNA.tsv`.
- `pobierz_z_edena.sh`: części, które były tylko na Edenie, strumieniem tar. Każda część sprawdzona sumami SHA-256 policzonymi na Edenie (`SHA256SUMS_EDEN_*.txt`, wynik w `pobieranie_z_edena.log`). Na Edenie niczego nie zmieniano.
- Duże pliki binarne są wyłączone z gita (`.gitignore` tego folderu i główny).

## Uwaga o dowiązaniach

Plik binarny edytowany w miejscu zmieniłby się w obu lokalizacjach. Zapisy ruchu nie są edytowane (aplikacja i skrypty piszą nowe pliki), więc w praktyce to kopia. Niezależną kopię na innym dysku daje zwykłe skopiowanie tego folderu.
