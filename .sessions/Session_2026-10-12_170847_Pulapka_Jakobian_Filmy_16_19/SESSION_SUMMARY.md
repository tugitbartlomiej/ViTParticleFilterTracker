# Session Summary: Pułapka bloku MLP — matematyka, Jakobian, filmy 16–19

## Metadata
- **Date:** 2026-10-12
- **Time:** ok. 16:07–17:08
- **Status:** Completed
- **Type:** Analysis (wizualizacje + literatura + pomiar)

## Objective
1. Matematyczne ugruntowanie obrazu z filmu 12: każdy element (sfera, szkło, reflektor, wiązka, cząsteczka, pułapka, temperatura) z odpowiednikiem i wzorem w ViT.
2. Zmierzyć i zwizualizować „pułapkę koloidalną” bloku MLP na prawdziwych danych.
3. Pokazać pułapkę w dwóch widokach (ogólny + zbliżenie), zsynchronizowanych krokiem uczenia.

## Context
Kontynuacja sesji `Session_2026-10-12_160647_Lasery_Reflektory_Sfera` (zasady Z1–Z14, filmy 8–12). Worktree `.claude/worktrees/laser-koloid-zasady`, gałąź `worktree-laser-koloid-zasady` (bez push); pliki kopiowane do `SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/`.

## Actions Taken
1. Undermind: deep search „Mathematical grounding of an optical tweezer picture of transformer MLP blocks…”; pełne teksty: Lu 2019 (transformer = konwekcja–dyfuzja, FFN = konwekcja), Mudarisov 2026 (sterowanie FFN, siodła), Pérez García 2018 (FORMA), Pesce 2009 (siły niezachowawcze), Riechers 2024 (geometria LayerNorm), Im 2016 (zachowawczość autoenkoderów), Yaida 2018 (FDR dla SGD), Gieseler 2020 (kalibracja pęset).
2. `pulapka_jakobian.py` (J odtwarzany z latentu; potem zastąpiony dokładnym J w filmie 18).
3. `matematyka_reflektorow/matematyka_reflektorow.pdf` (5 stron): słownik element ↔ wzór ↔ status, dwie sfery (LayerNorm vs szklana), pułapka = Jakobian, termodynamika, braki. Zastrzeżenie: FORMA/kalibracja pęset to metody fizyczne, nie stosowane dotąd do sieci.
4. Film 16 (siodło z wirem, cząstki próbne) — na wagach z niewłaściwego biegu.
5. Odkrycie niezgodności wag: forward z `01_bieg_35000_gesty_zapis` nie odtwarza latentu; `MalyViT_35000_od_zera_s47` odtwarza z R² = 1,0.
6. Film 17: ognisko 768 wiązek w tokenie 316 + 599 prawdziwych tokenów (przyciąganie/odpychanie po LN), 806 checkpointów.
7. Film 18: dwa widoki (ogólny + zbliżenie) z dokładnym Jakobianem J = W₂ diag(GELU′(a)) W₁ J_LN(x).
8. Film 19: jak 18 + wspólny pasek osi uczenia, licznik kroku nad oboma widokami, prędkość/strzałki w zbliżeniu ∝ rzeczywista siła pola.

## Results

### Key Findings
- Pułapka = siodło z wirem: ≈51 % kierunków S cofa; ‖A‖/‖S‖ 0,995 → 0,958 (macierz losowa ≈ 1, autoenkodery 0,10–0,23, pęseta optyczna 0,01–0,05); ślad J/192 ≈ −0,007; siła pola ×3,0 przez 60 epok.
- Liczby z filmu 16 i z `pulapka_jakobian.py` (×14, ślad −0,05/−0,076, symetryczność 55–56 %) wynikały z wag innego biegu — nieaktualne.
- Ostatni blok rozsuwa tokeny narzędzia względem tokenu 316 (×1,098) mocniej niż tło (×1,020); hipoteza: gęsta głowa potrzebuje różnych odległości do krawędzi ramki na każdym tokenie.
- Siły 768 wiązek w ognisku znoszą się: |wypadkowa| / Σ|siła| = 0,056 → 0,168.

### Issues Encountered
- Niezgodne wagi (patrz wyżej) — naprawione w filmach 17–19.
- Surowe odległości rosną dla wszystkich tokenów (strumień rezydualny rośnie) → przyciąganie liczone po LayerNorm.
- Ochrona worktree blokowała złożone komendy git/bash → rozbijanie na proste polecenia.

## Files Generated/Modified
W `SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/`:
- `matematyka_reflektorow/matematyka_reflektorow.tex` (+ `.pdf`), `pulapka_jakobian.py` + `PULAPKA_JAKOBIAN.json`
- `film_lasery_16_pulapka_siodlo.py`, `film_lasery_17_pulapka_ognisko.py`, `film_lasery_18_pulapka_dwa_widoki.py`, `film_lasery_19_pulapka_dwa_widoki_sync.py`
- filmy `lasery_16_pulapka_siodlo.mp4`, `lasery_17_pulapka_ognisko.mp4`, `lasery_18_pulapka_dwa_widoki.mp4`, `lasery_19_pulapka_dwa_widoki_sync.mp4` (+ klatki png)
- pamięci podręczne: `pulapka_16_cache.npz`, `pulapka_17_cache.npz`, `pulapka_18_cache.npz`
Pamięć: `wizualizacja-film12-reflektory.md`.

## Commands Used
- `F:/Python/3.13/python.exe -X utf8 film_lasery_19_pulapka_dwa_widoki_sync.py [proba]` (render GPU moderngl + h264_nvenc)
- `pdflatex matematyka_reflektorow.tex`
- Undermind: `launch_deep_search`, `read_pdfs`, `search_papers`.

## Next Steps
- [ ] Poprawić liczby pułapki w `matematyka_reflektorow.tex` (pkt 4) i w `ZASADY_LASER_KOLOID.md` na wartości z dokładnego J.
- [ ] Sprawdzić, które wcześniejsze wyniki (filmy 8–12, `PRZECIWLASERY.json`, `OBROT_REFLEKTORA.json`) używały wag z `01_bieg_35000_gesty_zapis` i przeliczyć na `MalyViT_35000_od_zera_s47`.
- [ ] Test hipotezy: czy rozsuwanie tokenów narzędzia odpowiada różnicom w celach l, t, r, b gęstej głowy.
- [ ] Uwaga jako wiązki między cząsteczkami; łańcuch 6 bloków.
