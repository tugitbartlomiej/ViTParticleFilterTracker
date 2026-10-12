# Session Summary: Szkło — 60 epok, wspólny napęd, wiązki i koloidy

## Metadata
- **Date:** 2026-10-11
- **Time:** ~14:00–17:34
- **Status:** In Progress (zadania Eden 1826876, 1826881)
- **Type:** Mixed (Analysis + SSH/Eden)

## Objective
Użytkownik chce: (1) sprawdzić ruch tokenów przez 60 epok i wykazać, że jest w nim zachowanie; (2) zdiagnozować klasę ruchu
(szkło / Mori–Zwanzig / koloid / gaz / ciecz) na podstawie charakterystyk z literatury, z użyciem piDMD, deeptime na Edenie;
(3) odpowiedzieć „czym jest centralny punkt drgań”; (4) sprawdzić hipotezę „768 wiązek lasera trzyma 600 koloidów”;
(5) trwałe przebiegi zapisywać w `SmallViTAblation/assets` z opisem; (6) liczyć na Edenie, nie lokalnie.

## Context
Wcześniej (sesje 2026-10-10/11): pokrętła lr/gęstości, test jądra (pole = Jakobian wspólnych wag), relacja τ–klatka (README 6d),
ocena klasyfikacji Imagenette, literatura (102 linki, 66 PDF w `literatura_szklo_tokeny_20261011`).

## Actions Taken
1. Inwentarz Edena: 806 checkpointów `2026-10-09_maly_vit_od_zera_35000_21epok/wyniki/pelny/wagi`, pomiar `pomiar_pelny60_3000tr_1000val`
   (metrics 4000×806×5); pobrano metryki lokalnie; |z−z0| 768D rośnie do ~16 w 5 epok, potem lekko maleje.
2. Eksport `eden/ruch60_rp.py` (zadanie 1826867, 17 min, 15,8 GB): rp128 (806,100,600,128), pelne768 dla 4 zdjęć, srednia768, normy, oceny;
   pobrany do `assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val` (SHA-256 OK, 7 plików).
3. Undermind deep search „Diagnosing the dynamical class of trapped particles with common drive from trajectory fingerprints” (162 prace)
   + 4 search_papers + Scite; 3. lista linków `literatura_szklo_tokeny_20261011/linki/klasa-dynamiki-markow-pamiec-operator-2026-10-11.txt` (33).
4. Lokalny test Markowa przerwany (polecenie użytkownika); przeniesiony na Eden: `markow_test_eden.py` (zad. 1826873, 3 okna),
   `ruch60_analiza.py` (zad. 1826874).
5. Odjęcie wspólnego napędu: `wspolny_naped_krok.py` (czynniki z pola innych zdjęć) i `wspolny_naped_wagi.py` (prawdziwe Δθ z 806 checkpointów),
   zadanie 1826876.
6. Test wiązek/kulek: `wiazki_koloidy.py` — z_i = GELU(W u_i + b), rozkład Δz na obrót wiązek W (fc1, 768×192) i ruch kulek u (192D),
   zadanie 1826881 (GPU, partycje hopper-2/short/hopper).

## Results

### Key Findings
- Skala epok (RUCH60.json): √MSD(100 kroków)/d_shell 2,0 → 1,0; α 0,40 → 0,20–0,36; Jaccard sąsiadów 0,64–0,71 (L=1) → 0,56–0,59 (L=40);
  operator co 100 kroków: błąd LS 0,65–0,77 trwania, symetryczny ≈ LS, ortogonalny 0,98–1,0, skalarny 0,73–0,81 (stała 0,45–0,63);
  768D/128D 0,94–1,01; mod zdjęcia 31–39 %; Spearman ruch–IoU od epoki 5: TRAIN −0,49…−0,59, VALID −0,43…−0,68.
- Skala kroku (MARKOW_*.json): operator jednokrokowy ≈ trwanie (0,98–0,99), symetryczny ≈ LS, kąty własne 0,2–0,4°; jądro pamięci
  K(1)/K0 0,67, K(2) 0,35–0,60; wcześnie ujemny ogon (sprężyste odbicie), po 11 epokach dodatni (całka 1,29 K0, tarcie lepkie);
  VAMP czasy implikowane maleją do τ≈8–12, potem rosną (niestacjonarność); β C(t) 0,51–0,83; centrum (W=100): drgania 32–46 % wariancji,
  dryf centrum 1,3–2,2 klatki w oknie, kurtoza przyrostów 1,3–1,8× gaussowskiej.
- Wspólny napęd (krok, lr 5e-4, okno 300–2188): czynniki z innych zdjęć K=1/3/10/30/100 → 0,10/0,25/0,42/0,56/0,72 (kontrola ≤0,055).
- Centralny punkt = f(θ̄(t), x)_i (token w sieci uśrednionej po szumie paczek); 768 = neurony warstwy ukrytej MLP ostatniego bloku.

### Issues Encountered
- Lokalny `markow_test.py` padał/wolny (pamięć) → przepisany strumieniowo i przeniesiony na Eden (`markow_test_eden.py`).
- Równoległa sesja buduje `assets/RuchCzasteczek_ViT_szklo_2026-10`; usunięto zdublowany folder w `assets/MalyViT_35000_od_zera_s47`,
  opis eksportu przeniesiony do `03_.../OPIS_ruch60_rp128_50tr_50val.md`.
- 2 duplikaty linków zakomentowane („JUZ NA LISCIE”).
- hopper-2 zajęty → `scontrol update Partition=hopper-2,short,hopper`.

## Files Generated/Modified
- `SmallViTAblation/outputs/porownanie_szklo_20261010/eden/`: ruch60_rp.py/.slurm, markow_test_eden.py, markow_test.slurm, ruch60_analiza.slurm,
  wspolny_naped_krok.py, wspolny_naped_wagi.py, wspolny_naped.slurm, wiazki_koloidy.py/.slurm
- `SmallViTAblation/outputs/porownanie_szklo_20261010/`: markow_test.py, ruch60_analiza.py
- `.../eden/wyniki/{pomiar60,markow,ruch60}/` (wyniki pobrane), `naped/` (w toku)
- `assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/{ruch60_rp128_50tr_50val/, OPIS_ruch60_rp128_50tr_50val.md}`
- `literatura_szklo_tokeny_20261011/linki/klasa-dynamiki-markow-pamiec-operator-2026-10-11.txt`
- Pamięć: `assets-smallvit-przebiegi.md` (nowa), `szklo-fala4-substancja-2026-10-11.md` (dopisek), MEMORY.md

## Commands Used
```bash
sbatch ruch60_rp.slurm            # 1826867 eksport 60 epok
sbatch markow_test.slurm          # 1826873
sbatch ruch60_analiza.slurm       # 1826874
sbatch wspolny_naped.slurm        # 1826876
sbatch wiazki_koloidy.slurm       # 1826881
bash assets/RuchCzasteczek_ViT_szklo_2026-10/pobierz_z_edena.sh ruch60
```

## Next Steps
- [ ] Pobrać `wyniki/naped` i `wyniki/wiazki_koloidy` (+ asset u192_20zdjec.npy, W_fc1.npy do assets/03_...)
- [ ] Werdykt „wiązki czy kulki” i odjęcie napędu z prawdziwych Δθ; dopisać README sekcje 6e–6g
- [ ] Test Chapmana–Kołmogorowa ze stanem rozszerzonym o napęd; minimalny model (OU + jądro + zamrożone κ_i) na pełnej karcie
- [ ] Spearman ruch–IoU na większej liczbie zdjęć (sygnał praktyczny pod NeurIPS)
