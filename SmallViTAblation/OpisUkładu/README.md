# Opis układu i dostęp do danych

**Czym jest układ.** MałyViT (detektor narzędzia w filmach z operacji zaćmy) oglądany jak układ fizyczny: 600 tokenów zdjęcia
to koloidy, 768 neuronów MLP w każdym z 6 bloków to reflektory, które pchają koloid wiązkami, a uwaga to siły między koloidami.
Ruch ma dwie osie czasu: głębokość sieci (jedno przejście, 12 podkroków) i uczenie (806 checkpointów biegu s47). Pole wiązek
w jednym przejściu nie trzyma koloidu, a w czasie uczenia koloid drga jak cząstka Browna w pęsecie wokół dryfującego środka
i dąży do celu zapisanego w stracie, w 5 liczbach odpowiedzi głowy.

## Od czego zacząć

1. [Opis układu](01_OPIS_UKLADU.md): czym jest każdy element, dwie osie czasu, pułapka optyczna, pułapka uczenia, cel koloidu,
   sieć i uczenie, co jest faktem, a co obrazem (stan 15 października 2026).
2. [Dane i trajektorie](02_DANE_I_TRAJEKTORIE.md): gdzie leżą surowe tablice, co zachowano, co odtworzono, czego brakuje.
3. [Katalog eksperymentów](03_KATALOG_EKSPERYMENTOW.md) i [scalone raporty](04_RAPORTY_SCALONE.md): 212 folderów rodzin
   i 243 raporty z `experiments/` i `outputs/` (stan 11 października). Kopie raportów są w `zrodla/`, metadane trajektorii
   w `metadane/`.

Pełne zasady Z1–Z29: `app/ZASADY_DZIALANIA_UKLADU.md`. Pomiary z 15 października: `experiments/pulapka_cel_koloidu_20261015/`.

## Skróty do danych (`linki/`)

- **DANE_60_EPOK_768D_KONTROLE.lnk** — pełne aktywacje 768D ośmiu zdjęć w 75 wybranych checkpointach biegu 60 epok (s47);
- **WAGI_60_EPOK_806_CHECKPOINTOW.lnk** — wszystkie checkpointy tego biegu (35 000 zdjęć);
- **DANE_GESTE_21_EPOK_768D_I_PACZKI.lnk** — osobny bieg 21 epok: pełne 768D ośmiu zdjęć i telemetria paczek;
- **K1000_LATENT_GESTY_768D.lnk** — tablica 1001 × 12 × 600 × 768 FP32 (~22 GB) historii K1000;
- **ZBIORCZE_ARCHIWUM_RUCHU.lnk** — archiwum danych ruchu w `assets`.

Dwuklik otwiera cel; Właściwości pokazują ścieżkę. Duże dane zostają w swoich miejscach, skróty nie zawierają ich bajtów.
Pakiet z 11 października skopiował 305 małych plików i przygotował 315 skrótów (`KONTROLA_PAKIETU.json`).

## Ważne zastrzeżenia o danych

- 768D po GELU bloku 6 to **jasności wiązek** padających na koloid, nie jego położenie (położenie: 192D strumienia rezydualnego).
- Dla biegu 60 epok nie zachowano pełnych aktywacji wszystkich 806 checkpointów; pełne 768D są w plikach kontrolnych wybranych
  stanów, w tym końców wszystkich 60 epok. Macierze Grama, PCA3/PCA32 i RP128 nie są pełnymi współrzędnymi.
- Stany 192D tokenów potrzebne do pomiarów pułapki liczy się na nowo z wag (aplikacja, `app/przejscia_zapis.py`,
  `experiments/pulapka_cel_koloidu_20261015/pomiar.py`).
