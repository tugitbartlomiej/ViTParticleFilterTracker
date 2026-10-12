# Biegi pokręteł (Eden, `~/smallvit_runs/2026-10-10_szklo_pokretla`)

Krótkie biegi uczenia od zera z obserwatorem tokenów: „temperatura” (lr), „gęstość” (liczba klatek), ziarna, optymalizator (SGD), architektura (CNN), warstwy i zadanie (klasyfikacja Imagenette). Każdy bieg `wyniki/<bieg>/`: `dane/` (metadata.json z pełnym przepisem, LOG.json strata, OCENY.json walidacja), `pelny/stan_eNN.pt` (dwa ostatnie stany: wagi, AdamW, generatory), `wagi/k*.pt` (wybrane kroki), `gesty/` (SONDY.json, BAZA_PCA32.npz, PACZKI.npz, porcje `pca32_NNN.npz`; przy rzucie `rp128` pod kluczem pca32 leży losowy rzut 128D; `pelne768_NNN.npz` z pełnym latentem 768D jednej sondy) oraz `ANALIZA*.json` (odciski szkła, heterogeniczność, testy substancji). Surowe porcje `gesty` większości biegów skasowano na Edenie po analizie. Pełne zostały dla `lr_5e-4_rp` i `dlugi_lr_5e-4_rp`. Opis eksperymentu: `07_kod/lokalnie/maly_vit_szklo_pokretla_20261010/README.md`. Wyniki: README porównania, sekcje 5d, 6, 6b, 6c, 6d.

| bieg | zadanie | model | opt | lr | paczka | ziarno | epoki | kroki | rzut | sondy | gęsto od | porcje gesty | stany | ANALIZ | rozmiar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Logi Slurm: `logi/`. Sumy kontrolne z Edena: `SHA256SUMS_EDEN_pokretla.txt`.
