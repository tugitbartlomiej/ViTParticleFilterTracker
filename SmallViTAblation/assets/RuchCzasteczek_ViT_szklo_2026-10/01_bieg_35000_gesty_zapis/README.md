# Bieg 35 000 od zera z gęstym zapisem tokenów (21 epok, ziarno 47)

Ten sam przepis co asset `../MalyViT_35000_od_zera_s47` (tam wagi 60 epok, tu nie duplikowane), ale osobny bieg z obserwatorem gęstym (`obserwator_gesty.py`). Zapis tokenów: 100 sond (zdjęć) × 600 tokenów. Latent = wyjście GELU MLP ostatniego bloku (768 neuronów na token).

- `lokalnie/gesty/gesty/`: `pca32_NNN.npz` (rzut na PCA32 z kroku 0, co 1 krok do 2188, potem co 10 do 22 970; 98,6 % wariancji statycznej, ale tylko 4–14 % energii ruchu, zob. README porównania 5b), `pelne768_NNN.npz` (pełny latent 768D 8 pierwszych sond, co 1 krok do 1094, potem co 50), `BAZA_PCA32.npz`, `PACZKI.npz` (które próbki były w każdej paczce), `SONDY.json`.
- `lokalnie/gesty/dane/`, `lokalnie/pelny_dane/`: LOG.json, OCENY.json (mAP@0,5), metadata.json biegu gęstego i biegu pełnego.
- `lokalnie/wejscia/`: `MANIFEST_35000.jsonl` i adnotacje walidacji.
- `eden/gesty/wagi/`: 341 checkpointów biegu gęstego `k<krok>.pt` (do kroku 22 974); `eden/gesty/pelny/`: `stan_e20.pt`, `stan_e21.pt`.
- `eden/pelny/pelny/`: `stan_e59.pt`, `stan_e60.pt`, pełne stany biegu 60 epok (wagi, AdamW, generatory) do wznowienia.
- `eden/proba`, `eden/gesty_proba`: biegi próbne, 512 klatek, 2 epoki.

Na Edenie surowy gęsty zapis (31 GB) skasowano 2026-10-10 po pobraniu. Ta kopia i oryginał w `experiments/` to jedyne pełne egzemplarze.

| podfolder | plików | rozmiar |
|---|---|---|
| `lokalnie/gesty/gesty` | 29 | 28.72 GB |
| `lokalnie/gesty/dane` | 3 | 0.1 MB |
| `lokalnie/pelny_dane` | 4 | 0.2 MB |
| `lokalnie/wejscia` | 3 | 35.3 MB |
