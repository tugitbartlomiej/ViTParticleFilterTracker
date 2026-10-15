# Session Summary: Lasery, reflektory i szklana sfera (szklo3_3D)

## Metadata
- **Date:** 2026-10-12
- **Time:** ok. 11:30–16:07
- **Status:** Completed
- **Type:** Analysis (wizualizacje + literatura)

## Objective
1. Ocenić, czy obraz „sfera 768 laserów + 600 koloidów w pułapkach” poprawnie fizycznie opisuje uczenie MałegoViT i czy jest publikowalny.
2. Znaleźć braki w badaniu na podstawie literatury (Undermind, pełne teksty).
3. Spisać zasady modelu i zrobić diagram architektury.
4. Zbudować poprawną wizualizację reflektorów i wiązek (filmy 8–12).

## Context
Kontynuacja badania `outputs/porownanie_szklo_20261010` (kryształ tokenów, filmy 1–7 laserów). Sesja jest forkiem sesji „szklo1”, praca w worktree `.claude/worktrees/laser-koloid-zasady` (gałąź `worktree-laser-koloid-zasady`, bazowana na `codex/eden-nano-training-20260914`); pliki kopiowane do katalogu roboczego użytkownika.

## Actions Taken
1. Undermind: deep search „Optical trap and colloid picture of representation learning…” + przegląd istniejących kwerend (kryształ, pułapki ze wspólnym napędem).
2. Pełne teksty (read_pdfs) 10 prac: Geshkovski 2023, Rigollet 2025, Herty & Liu 2026, Isobe 2026, Agazzi 2026, Pashakhanloo & Koulakov 2023, He & Su 2019, Qin 2021, Mannes 2026, Antognini & Sohl-Dickstein 2018 (Zhang 2021 nieczytelny).
3. Test τ vs lr na biegach pokręteł (RP128): τ_1e = 13, 6, 3 kroki dla lr 1,25e-4, 5e-4, 2e-3 → τ ∝ η^−0,5.
4. Pomiary na wagach fc2/fc1 bloku 6 (17 checkpointów): przeciwlasery, cos(w_c, v_c), |v_c|; obrót reflektora (Prokrustes + zero).
5. `ZASADY_LASER_KOLOID.md` (Z1–Z14), diagram TikZ `diagram_maly_vit/diagram_maly_vit.pdf` (3 strony).
6. Filmy: 8 (jeden reflektor matrycowy), 9 (kanały galwo), 10 (600 reflektorów = tokeny), 11 (768 reflektorów = neurony, ruchome), 12 (768 stałych reflektorów, galwa, jedno załamanie na sferze) — najlepszy.

## Results

### Key Findings
- Koloid „nadążający za pułapką” w czasie uczenia: błędne (zero opóźnienia z definicji). Poprawna pułapka w czasie uczenia = dolina straty (Pashakhanloo: fluktuacje w poprzek, dyfuzja wzdłuż).
- Token = cząstka wzdłuż głębokości (Geshkovski/Rigollet); neurony = cząstki w czasie uczenia (Mei, Herty & Liu).
- Blok MLP = 768 sił Σ z_c v_c; laser ma dwa wektory (celownik w_c, pchnięcie v_c).
- Przeciwlasery 0 % → 38 %; słaba siła zwrotna cos(w_c, v_c) = −0,16; brak wspólnego obrotu reflektora.
- Publikowalne są zjawiska (siatka z kodowania pozycji, heterogeniczność → IoU, wspólny napęd), nie sam obraz laserów; braki: test RPI, surogaty rzutu, profile głębokości, jądro lokalnej sprężystości, τ z autokorelacji.

### Issues Encountered
- Worktree z origin/master nie miał folderu → reset do gałęzi roboczej z `core.longpaths=true`.
- Wspólny `lasery_wspolne.py` zmieniony przez inną sesję (niezacommitowany) — filmy 10–12 go wymagają.
- Film 10 grupował wiązki po tokenach (poprawione w 11 i 12); film 12 v1 miał łuki w sferze (poprawione: jedno załamanie na powierzchni).

## Files Generated/Modified
W `SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/`:
- `ZASADY_LASER_KOLOID.md`, `sprawdz_przeciwlasery.py` + `PRZECIWLASERY.json`, `sprawdz_obrot_reflektora.py` + `OBROT_REFLEKTORA.json`
- `diagram_maly_vit/diagram_maly_vit.tex` (+ `.pdf`)
- `film_lasery_8_reflektor.py`, `film_lasery_9_reflektor_galwo.py`, `film_lasery_10_reflektory_sfera.py`, `film_lasery_11_reflektory_kanalow.py`, `film_lasery_12_reflektory_stale.py`
- filmy `lasery_8_reflektor.mp4`, `lasery_9_reflektor_galwo.mp4`, `lasery_10_reflektory_sfera.mp4`, `lasery_11_reflektory_kanalow.mp4`, `lasery_12_reflektory_stale.mp4` (+ klatki png), `reflektor_8_cache.npz` (186 MB, pamięć podręczna)
Pamięć: `zasady-laser-koloid.md`, `filmy-zawsze-nowe.md` (wpisy w MEMORY.md).
Commity (gałąź `worktree-laser-koloid-zasady`): 54c69e23, 2dd95ba9, 6525470a, df4d4b3a, a9ba6431, 7e9ece0b, 435fa4a3, 8b4fe1eb.

## Commands Used
- `F:/Python/3.13/python.exe -X utf8 film_lasery_12_reflektory_stale.py [proba]`
- `pdflatex diagram_maly_vit.tex`
- Undermind: `launch_deep_search`, `read_pdfs`, `search_papers`; Scite: kontrola editorialNotices.

## Next Steps
- [ ] Matematyczne ugruntowanie odpowiedników (sfera, szkło, reflektor, wiązki, pułapka) z wzorami + research (Undermind).
- [ ] Uwaga jako wiązki między cząsteczkami; łańcuch 6 bloków.
- [ ] Testy: RPI, surogaty rzutu, profile głębokości w 806 checkpointach, jądro lokalnej sprężystości, τ z autokorelacji.
