# Istniejące surowe dane: gdzie są i co zawierają

Stan sprawdzenia: 11 października 2026. W tym katalogu zgromadzono kopie małych raportów i metadanych oraz skróty Windows `.lnk` do dużych danych. Dwuklik skrótu otwiera katalog lub plik źródłowy. Prawy przycisk → Właściwości pokazuje pełny cel; dla skrótu pliku można użyć „Otwórz lokalizację pliku”. Pliki źródłowe pozostają w swoich miejscach. Skróty nie są kopiami tablic; po przeniesieniu źródła trzeba je poprawić.

## Najważniejsze rozróżnienie

Model, o który chodzi, trenowano **60 epok na 35 000 zdjęć**. Zachowano **806 checkpointów**. Nie znalazłem istniejącej kompletnej tablicy wszystkich 768 współrzędnych dla wszystkich 806 checkpointów tego biegu. Kod audytu stworzył taki plik roboczy dla ośmiu zdjęć, a potem jawnie go usunął. Zachowały się natomiast pełne aktywacje 768D dla **75 wybranych checkpointów**, w tym końców wszystkich 60 epok. To najbardziej surowe zachowane aktywacje interesującego biegu, które znalazłem.

Nie uruchamiać `compute_grams.py`, aby tylko otworzyć te wyniki: ten skrypt wykonuje inferencję. Poniższe ścieżki i skróty pozwalają pracować na już istniejących plikach.

## A. Model 60 epok: surowe 768D dla wybranych checkpointów

**Skrót: `linki/DANE_60_EPOK_768D_KONTROLE.lnk`.**

Źródło: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/audyt_szkla_768d_20261010/precision_controls`.

75 plików `kNNNNN.npz`, razem około 2.47 GB. Każdy zawiera tablice:

- `fp32`: naturalny latent w FP32, **8 × 600 × 768**;
- `bf16`: odczyt z autocast bf16, zapisany w float16;
- `frozen_bn`: kontrolny FP32 z buforami BatchNorm zamrożonymi na kroku 300.

Do analizy naturalnego ruchu używać `fp32`; `frozen_bn` opisuje odrębną kontrolę interwencyjną. Krok bierze się z nazwy pliku. Osiem zdjęć to wiersze puli 144, 448, 922, 2741, 2786, 3200, 3438, 3698. Są to 4800 cząsteczek, każda ma 768 współrzędnych. Checkpointy końców epok mają krok `1094*epoka`. Dodatkowe wczesne zapisy są nieregularne.

Kopie `long_sources.json`, `long_bn_controls.json` i `SONDY.json` w `metadane/` opisują pochodzenie i sondy. `long_sources.json` jest historyczny i może wskazywać dawne położenie wag; aktualne checkpointy są w assets, opisane niżej.

Przykład odczytu, do samodzielnego wykonania:

```python
from pathlib import Path
import numpy as np
root = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')
p = root / 'outputs/audyt_szkla_768d_20261010/precision_controls/k65640.npz'
with np.load(p) as data:
    z = data['fp32']   # (8,600,768), naturalny latent po 60. epoce
    particle = z[0, 100]   # 768 współrzędnych tokenu 100 pierwszej sondy
```

Ten fragment jedynie czyta zapisane tablice. Nie ładuje modelu i nie wykonuje forwardu.

## B. Wszystkie 806 wag tego samego modelu

**Skrót: `linki/WAGI_60_EPOK_806_CHECKPOINTOW.lnk`.**

Źródło: `assets/MalyViT_35000_od_zera_s47/wagi`. Pliki `k*.pt`, od kroku 0 do 65 640, około 9.45 GB według liczby plików i typowego rozmiaru. Dokładne rozmiary są w `INWENTARZ_DANYCH.json`.

To wagi sieci i bufory, **nie gotowe trajektorie tokenów**. Zwykłe checkpointy nie przechowują pełnego stanu AdamW. Do obliczenia brakujących reprezentacji potrzebna byłaby osobna inferencja — nie jest częścią tego porządkowania.

Skróty do `dane/` i katalogu całego modelu prowadzą do LOG.json, OCENY.json, README i zachowanych sum kontrolnych. Kod treningu jest pod skrótem `KOD_TRENINGU_35K.lnk`.

## C. Najgęstsze pełne 768D: osobny bieg 21 epok

**Skrót: `linki/DANE_GESTE_21_EPOK_768D_I_PACZKI.lnk`.**

Źródło: `experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty`.

16 plików `pelne768_000.npz` … `pelne768_015.npz`. Klucze: `kroki` int32 oraz `latent` float16. Pełne porcje mają 100 × 8 × 600 × 768, ostatnia jest krótsza. Razem **1533 stany**, pełny zapis co krok 0–1094, następnie co 50 do 22 950. Przejście 1094→1100 ma 6 kroków. Koniec treningu jest w 22 974, ale nie ma na nim zapisu pełnych 768D.

Dodatkowo 9 dużych `pca32_*.npz` obejmuje 100 sond w rzucie 32D, z normami w pełnych 768D. **Te pliki nie zastępują pełnych aktywacji.** PACZKI.npz ma telemetrię aktualizacji i indeksy obrazów paczek; BAZA_PCA32.npz i SONDY.json opisują pomiar. Cały katalog ma około 28.72 GB. Każdy plik pełnych 768D ma również osobny skrót w `linki/pelne768_21epok/`.

```python
import numpy as np
with np.load(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_od_zera_35000_21epok_20261009\wyniki\gesty\gesty\pelne768_000.npz') as data:
    steps = data['kroki']
    z = data['latent']  # czas × 8 obrazów × 600 tokenów × 768 kanałów
```

Ten bieg ma te same dane treningowe i recepturę, lecz jest inną trajektorią niż `pelny` 60 epok. Nie dopisywać tych stanów do historii 60 epok.

## D. K1000: oryginalna bardzo gęsta tablica FP32

**Skrót: `linki/K1000_LATENT_GESTY_768D.lnk`.**

Źródło: `experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/LATENT_GESTY.npy`.

Kształt **1001 × 12 × 600 × 768**, float32. Rozmiar około **22.14 GB**. Każda aktualizacja od 0 do 1000, 12 sond. To najpełniejszy zachowany lokalnie gęsty zapis pierwotnego biegu K1000, ale nie historia modelu 60 epok. Można otworzyć przez `np.load(path,mmap_mode='r')`, bez wczytywania całej tablicy.

Obok są obrazy.npy, images.json, maski.npz, metadata.json, LOG.json i OCENY.json. `LATENT_SOND.npz` ma inne, uśrednione obserwacje i nie jest zamiennikiem LATENT_GESTY.npy.

## E. Pomiar 60 epok dla 4000 zdjęć: zachowane metryki, brak trajektorii q16

**Skrót: `linki/POMIAR_60_EPOK_4000_ZDJEC_METADANE.lnk`.**

Aktualne źródło lokalne: `outputs/porownanie_szklo_20261010/eden/wyniki/pomiar60`. W trakcie pracy wcześniejsza kopia `assets/MalyViT_35000_od_zera_s47/trajektorie_60_epok/pomiar_pelny60_3000tr_1000val` przestała istnieć. Skrót prowadzi do zachowanego katalogu w outputs.

Manifest opisuje 806 stanów 4000 zdjęć (3000 TRAIN + 1000 VALID), czyli 2.4 miliona tokenów w jednym stanie. `metrics.npy` ma kształt 4000 × 806 × 5: confidence, iou, delta_l2, training_step, both90. To **nie pełny latent 768D**. `initial_pca.npz` opisuje projekcję, nie całą historię aktywacji.

W manifeście jest 806 plików `step_*.q16`; nie ma ich w lokalnym katalogu. Read-only sprawdzenie na Edenie 11 października wykazało również brak tych plików w wskazanym źródle. Tam pozostał dodatkowo `pca32_podzbior.npz`. Nie ma podstaw, aby nazywać ten lokalny folder kompletną osią trajektorii. Manifest ma `full_latent_states_persisted=false`, więc nawet pełny zestaw q16 nie zawierałby wszystkich współrzędnych 768D.

Źródło historyczne na Edenie: `eden-cluster:/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/wyniki/pomiar_pelny60_3000tr_1000val`.

Istnieje także zbiorczy katalog `assets/RuchCzasteczek_ViT_szklo_2026-10` ze skopiowanymi danymi i dowiązaniami NTFS. Prowadzi do niego skrót `ZBIORCZE_ARCHIWUM_RUCHU.lnk`. Jego podkatalog `03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val` jest oznaczony jako RP128: nie traktować go jako pełnych 768 współrzędnych. Stan pobierania i kompletność tego osobnego archiwum należy sprawdzać w jego README i logu; ten pakiet nie uruchamia ani nie nadzoruje tamtych operacji.

## F. Analizy pochodne

**Skrót: `linki/AUDYT_768D_WYNIKI_POCHODNE.lnk`.**

`dense_image*.npz` i `long_image*.npz` przechowują macierze Grama, centroidy, normy i sąsiedztwa. Powstały z pełnych latentów, lecz nie zawierają pełnych współrzędnych wszystkich tokenów. Raporty i JSON-y umożliwiają odczyt istniejących wniosków; do nowego dowolnego obliczenia na kanale j użyć A, C lub D.

## Co jest w tym folderze

- `01_OPIS_UKLADU.md`: szczegółowa synteza układu, treningu, sił i wyników;
- `02_DANE_I_TRAJEKTORIE.md`: ten przewodnik;
- `03_KATALOG_EKSPERYMENTOW.md`: indeks rodzin i źródeł;
- `04_RAPORTY_SCALONE.md`: połączone treści zachowanych raportów, z identyfikacją źródeł;
- `linki/`: skróty do rzeczywistych danych i folderów;
- `metadane/`: kopie małych plików opisujących najważniejsze trajektorie;
- `zrodla/`: kopie raportów i opisów źródłowych;
- `INWENTARZ_DANYCH.json`: ścieżki, rozmiary i nagłówki tablic, bez wykonywania analiz ruchu;
- `KONTROLA_PAKIETU.json`: sprawdzenie celów skrótów i kompletności kopii.

Po korekcie polecenia zatrzymano rozpoczęty eksport inferencyjny. Niepełna tablica utworzona przez ten eksport została usunięta i nie należy do pakietu. Żaden historyczny wynik nie został usunięty ani przeniesiony.
