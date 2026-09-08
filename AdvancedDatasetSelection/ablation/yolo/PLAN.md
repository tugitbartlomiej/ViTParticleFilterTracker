# PLAN ABLACJI: EL2N YOLO ep170 + selekcja 20k + retrening YOLO do ep400

**Data:** 2026-04-25
**Worker:** clw1
**Status:** PLAN — do akceptacji przed uruchomieniem
**Powiązane:** `UWAGI_PROFESORA.md` uwagi #8 i #9, wzorzec metodologiczny w `AdvancedDatasetSelection/ablation/PLAN.md` (DETR ablation V1–V5)

---

## 1. Cel eksperymentu

Sprawdzić, czy **EL2N derived from YOLOv8 ep170** prowadzi do podobnie efektywnej kuracji jak obecny **EL2N derived from DETR ep170** — i tym samym osłabić zarzut o **circular reasoning** (DETR-based difficulty score → DETR wygrywa cross-dataset).

Konkretnie:

1. Wyciągnąć per-frame EL2N loss z **YOLOv8 ep170** dla pełnego pool-a 91,336 frames
2. Powtórzyć selekcję 20,000 frames przez Multi-Feature Cluster-Based Pipeline (4 cechy + K-Means) z **YOLO-EL2N zamiast DETR-EL2N** (pozostałe 3 cechy bez zmian)
3. Wytrenować YOLOv8 na nowym 20k subset **od ep170 do ~ep400** (230 epok dodatkowo, analogiczny zakres do DETR fine-tune który dotrwał do ep435)
4. Porównać z baseline: **YOLOv8 trenowany na DETR-curated 20k = 78.74% mAP@0.5 ep270** (cross-dataset)
5. Dodatkowo: porównać ranking subsetów (Jaccard overlap) — diagnoza czy DETR-bias istnieje

Wyniki zasilą Sekcję 6.5 **Ablation Study** (uwaga #2 + #9 promotora) i tabelę YE-A z `UWAGI_PROFESORA.md` sekcja 8.

---

## 2. Krok 1 — Ekstrakcja YOLO-EL2N z ep170

### Cel
Per-frame loss z YOLOv8 ep170 dla wszystkich 91,336 augmented frames z poolu treningowego.

### Skrypt (do napisania)
`AdvancedDatasetSelection/extract_yolo_el2n.py`:

```python
# Wejścia:
#   - YOLOv8 checkpoint: /mnt/f/.../Eden/Checkpoints/YOLO_EDEN_TRAIN/.../epoch170.pt
#       (lokalnie: /mnt/evafs/.../YOLO_Checkpoints/epoch170.pt na Edenie)
#   - Annotations: merged_20k_annotations_fixed.json (lub pełny pool)
#   - Images: 91,336-frame augmented pool
# Wyjście:
#   - yolo_ep170_el2n_scores.json: {frame_id: loss_value}
#   - per-frame loss = box_loss + cls_loss + dfl_loss (Ultralytics convention)
#   - normalizacja: z-score lub min-max [0,1] (ZGODNIE z normalizacją DETR-EL2N w obecnym pipeline)
```

### Wzorzec z DETR-EL2N (do zachowania)
DETR-EL2N (`DETR_EL2N_Scorer`, kod w `AdvancedDatasetSelection/`) używa:
- Query 81 confidence: `d_i = 0.7·(1 - s_q81) + 0.3·H_norm(p_q81)`
- threshold τ=0.3, default 1.0 dla "no detection above threshold"
- normalizacja [0,1]

YOLO odpowiednik:
- per-frame loss z forward pass (tryb training=False ale z computed loss)
- normalizacja **identyczna** (z-score lub [0,1] — sprawdzić config)
- **Brak** specific-query mechanism — YOLOv8 daje pojedynczy scalar loss per frame

### Weryfikacja
- Sprawdzić rozkład YOLO-EL2N (histogram) vs DETR-EL2N — czy podobny shape (long-tail)
- Spearman correlation ranking YOLO-EL2N vs DETR-EL2N (oczekiwane 0.4–0.7 — moderate)
- Sanity check: top-10 najtrudniejszych frames per ranking — czy zawierają anatomicznie trudne sceny

### Zasoby
- 1× H100 lub A100, 91,336 forward passes na batch 64 ≈ 1,427 batchy
- czas: ~30–60 min
- output: ~2 MB JSON

---

## 3. Krok 2 — Powtórna selekcja 20k z YOLO-EL2N

### Modyfikacja pipeline'u
`AdvancedDatasetSelection/main_selection_pipeline.py` — dodać parametr `--el2n_source {detr,yolo}`:
- `detr` → obecne `DETR_EL2N_Scorer` (default, backward-compatible)
- `yolo` → nowy loader `YOLO_EL2N_Scorer` wczytujący `yolo_ep170_el2n_scores.json`

### Konfiguracja (BEZ ZMIAN vs obecny pipeline)
- Wagi feature groups: **DINOv3 = 0.35, EL2N = 0.30, SAM = 0.20, Fourier = 0.15**
- PCA DINOv3: 1024 → 32 dim
- Standaryzacja per-group (z-score)
- Concatenation: 32 + 9 + 1 + 1 = 43-dim unified space
- K-Means: K = 20,000, FAISS GPU, 300 iterations, seed=42
- Representative selection: **centroid** (najbliższy klastrowi)

### Output
`output/selected_dataset_yolo_el2n/`:
- `selected_frames.json` — 20,000 frame_ids
- `selected_annotations.json` — COCO format
- `selection_metadata.json` — wagi, PCA explained variance, cluster stats

### Zasoby
- 1× H100, ~2–3 h (głównie K-Means na 91k × 43-dim)

---

## 4. Krok 3 — Porównanie subsetów (DETR-EL2N vs YOLO-EL2N)

### Metryki overlap
- **Jaccard index** = |A ∩ B| / |A ∪ B|, gdzie A, B = zbiory 20k frame_ids
- **IoU** identyczne dla zbiorów dyskretnych
- **Per-video overlap** — z których z 50 wideo każdy subset czerpie po ile frames

### Hipotezy
| Overlap | Interpretacja |
|---------|---------------|
| **>80%** | Pipeline robust to EL2N source — circular reasoning słaby. DINOv3+SAM+Fourier dominują selekcję, EL2N jest przyrostowy. |
| **50–80%** | EL2N wpływa znacząco, ale 3 pozostałe cechy zapewniają wspólny rdzeń. Bezpieczne do publikacji. |
| **<50%** | DETR-bias realny — ranking trudności drastycznie zależy od architektury detektora. Wymaga jawnego ujawnienia w Limitations. |

### Diagnostyka pomocnicza
- Histogram trudności per frame (DETR vs YOLO) na próbce — czy są frames "trudne dla DETR ale łatwe dla YOLO"?
- Klasy semantyczne (jeśli dostępne) — czy YOLO-EL2N preferuje inne sceny niż DETR-EL2N?

### Output
- `comparison_subsets_report.md` (tekstowy, ~50 linii)
- `overlap_diagram.png` — Venn diagram lub bar chart

### Zasoby
- Lokalnie (CPU), <30 min

---

## 5. Krok 4 — Trening YOLOv8 na YOLO-EL2N 20k subset

### Konfiguracja (zgodna z obecnym fine-tune YOLOv8 produkcyjnym)

Bazuje na `Eden/Checkpoints/YOLO_EDEN_TRAIN/YoloTreningSesnions/20KTreningFrom170epochStart/exp/args.yaml`:

| Parametr | Wartość | Uzasadnienie |
|----------|---------|--------------|
| Start checkpoint | `epoch170.pt` (z YOLO base 5×H100 hopper) | identyczna baza jak DETR-EL2N variant |
| Epochs total | 400 (start 170, +230 epok) | analogia do DETR fine-tune do ep435 |
| Batch | 32 per GPU | jak obecny |
| Image size | 640×640 | YOLOv8 default |
| Optimizer | auto (SGD + cosine) — **ALE** `cos_lr=false` w obecnej produkcji → linear decay | **identyczne** z baseline (zgodność porównania) |
| LR0 | **1e-4** (obniżone z 0.01 dla fine-tune, jak w sesji `2026-01-20_yolo_finetune_lr0001`) | sesja produkcyjna potwierdzona |
| LRF | 0.01 | per args.yaml |
| Patience | 50 | wczesne stopienia jeśli plateau |
| Warmup epochs | 3 | per args.yaml |
| AMP | true | mixed precision |
| Mosaic | 1.0 (close_mosaic=10) | jak obecny |
| HSV | h=0.015, s=0.7, v=0.4 | jak obecny |
| Workers | 24 | per args.yaml |

### Hardware (zgodny z aktualnym workflow)
- **Preferowane: hopper / 7×H100** (per najnowszy produkcyjny job 1524831 z sesji 2026-01-20)
- Alternative: dgx / 4× A100 lub hopper / 4×H100 (jeśli 7-GPU slot zajęty)

### Estymata czasu (na podstawie historycznych logów)

Bazuje na danych SLURM:
- Job 1524831 `yolo_20k_lr0001` hopper **7× H100**: 27h41m wallclock dla **~100 epok** (ep170→ep270 final_ep170_lr0001) → **~16 min/epoch** na 7× H100
- Job 1523692 `yolo_20k_ep170_500` hopper **4× H100**: 10h57m dla mniejszego zakresu (~30 epok) → **~22 min/epoch** na 4× H100
- Job 1500982 `yolo_20k_dgx1` dgx-4 **4× A100**: 4h57m (mniej epok)

| Wariant sprzętowy | Per-epoch | Dla 230 epok | Wallclock |
|-------------------|-----------|---------------|-----------|
| hopper / 7× H100 | ~16 min | 230 × 16 = 3680 min | **~2.5 dnia** |
| hopper / 4× H100 | ~22 min | 230 × 22 = 5060 min | **~3.5 dnia** |
| dgx / 4× A100 | ~30 min | 230 × 30 = 6900 min | **~4.5 dnia** |
| pascal / 4× P100 | ~50–60 min | 11.5–14k min | **8–10 dni** (długie partition timeouts) |

**Rekomendacja:** hopper 7× H100 lub 4× H100 (zależnie od dostępności quoty Eden — dysk obecnie 100% pełny, do cleanup-u przed startem).

### Checkpointy
- save_interval: **co 10 epok** (ep180, 190, ..., 400) — zgodnie z DETR ablation PLAN
- Bezstratny "best" zapisywany po każdej walidacji
- Output dir: `Yolo/20k_yolo_el2n_finetune/exp/weights/`

### Walidacja w trakcie
- Same-distribution: na video-level val split (zgodnie z `video_split.json` z DETR ablation)
- Co 10 epok automatic eval (Ultralytics default)

### Limity Eden
- Partition `hopper` time limit: 7 dni (sprawdzić aktualny limit w `sinfo`)
- Jeśli timeout → resume z ostatniego checkpointu, łatwe w Ultralytics (`--resume`)

---

## 6. Krok 5 — Benchmark cross-dataset (po treningu)

### Skrypt
Reuse `YOLO_DETR_Benchmarks/scripts/benchmark_yolo_vs_detr_q81_all_epoch.py` z modyfikacją: dodać ścieżkę do nowych checkpointów YOLO-EL2N.

### Zakres ewaluacji
- **Co 10 epok**: ep170, 180, 190, ..., 400 (24 punkty)
- **External benchmark = ONE undivided test set** (zgodnie z konwencją artykułu)
- Confidence threshold τ=0.3, IoU=0.5 (zgodnie z protokołem artykułu)
- Metryki: mAP@0.5, mAP@[0.5:0.95], precision, recall, F1, TP/FP/FN

### Output
- `BENCHMARK_YOLO_EL2N_<TIMESTAMP>/`:
  - `BENCHMARK_REPORT.md` — markdown summary (analogiczny do `BENCHMARK_ALL_EPOCHS_20260209_054404`)
  - `results_all_epochs.json`
  - `*_predictions.json` per epoch

### Wykres trajectory
- mAP@0.5 vs epoch dla:
  - YOLO 20k DETR-EL2N (obecne, baseline 78.74% ep270 — z `BENCHMARK_ALL_EPOCHS_20260209_054404`)
  - YOLO 20k YOLO-EL2N (nowy)
- Optional: dodać DETR 20k DETR-EL2N (ep270 = 85.74%) jako trzecia linia dla kontekstu

---

## 7. Oczekiwane wyniki i hipotezy

### H1: YOLO-EL2N daje YOLO podobny gain (~76–80%)
**Implikacja:** Pipeline jest model-agnostic. Circular reasoning to zarzut bezpodstawny. Można w Discussion napisać "feature curation generalizes across detector families when the difficulty score is computed from the matching architecture".

### H2: YOLO-EL2N daje YOLO LEPSZY gain (>80%, blisko DETR 85.74%)
**Implikacja:** DETR-EL2N **nie był optymalny** dla YOLOv8. To wzmacnia tezę "match difficulty source to target architecture". Wymaga update Limitation pkt 6 (architecture-specific difficulty).

### H3: YOLO-EL2N daje YOLO GORSZY gain (<70%)
**Implikacja:** Pipeline preferuje DETR — coś w kombinacji DINOv3+SAM+Fourier+EL2N synergizes lepiej z DETR. Wymagałoby głębszej analizy (per-feature contribution dla YOLO). Nie psuje claim 42× curation gain (który dotyczy DETR), ale zmienia narrację 7.2.3.

### H4 (skrajna): brak różnicy między subsetami (Jaccard >95%)
**Implikacja:** EL2N waga 0.30 jest "zbyt mała by zmienić selekcję" — DINOv3+SAM dominują. Sugeruje że artykuł powinien podnieść wagę EL2N w przyszłej iteracji albo osłabić jej rolę w opisie.

---

## 8. Powiązanie z innymi ablacjami z UWAGI_PROFESORA.md

### Ablation B (Difficulty source) z sekcji 3 UWAGI_PROFESORA
| Variant | Krok w tym planie | Status |
|---------|-------------------|--------|
| **B1** EL2N z DETR (current baseline) | nie wymaga nowego runu — istniejące `BENCHMARK_ALL_EPOCHS_20260209` | ✅ DONE |
| **B2** EL2N z YOLOv8 ep170 (nowy ranking) | **ten plan, kroki 1–6** | 📋 PLAN |
| **B3** brak EL2N (waga 0) | osobna ablacja — pokrywana przez DETR ablation V3 NO_EL2N (już wykonana, eval pending) | ⏳ pending eval |

### Ablation YE-A/B/C z sekcji 8 UWAGI_PROFESORA
| Variant | Pokrycie przez ten plan |
|---------|-------------------------|
| **YE-A** (YOLO-EL2N → YOLO 20k f.t.) | ✅ **bezpośrednio realizowany** — kroki 4–6 |
| **YE-B** (YOLO-EL2N → DETR 20k f.t.) | ❌ wymaga osobnego treningu DETR (nie objęte tym planem; rekomendacja: jako rozszerzenie po wynikach YE-A) |
| **YE-C** (DETR-EL2N → YOLO 20k f.t.) | ✅ = obecny baseline 78.74% (już DONE) |

### Feature-group ablacja A1–A6 (uwaga #2 promotora)
- **NIE jest pokryta** tym planem — to osobna seria 6 wariantów (różne wagi feature groups)
- Powiązana, ale niezależna — może być wykonana równolegle z YE-A jeśli zasoby pozwolą

---

## 9. Estymata zasobów (sumarycznie)

### GPU-hours
| Krok | Hardware | Wallclock | GPU-h |
|------|----------|-----------|-------|
| 1. Ekstrakcja YOLO-EL2N | 1× H100 | 0.5–1 h | ~1 |
| 2. Selekcja 20k YOLO-EL2N | 1× H100 | 2–3 h | ~3 |
| 3. Porównanie subsetów | CPU | <1 h | 0 |
| 4. Trening YOLO ep170→400 | 7× H100 hopper | 2.5 dnia | ~420 |
| 5. Benchmark 24 checkpointów | 1× H100 | 4–6 h | ~6 |
| **TOTAL (preferred)** | mix | **~3 dni** | **~430 GPU-h** |

Alternatywa "tani" (4× A100): ~5 dni, ~480 GPU-h.

### Storage
- Nowe checkpointy YOLO ep170-400 co 10 epok: 24 × ~155 MB = ~3.7 GB
- Benchmark predictions: ~50 MB
- **Razem: ~4 GB nowych danych** — mieści się w obecnym limicie Eden po cleanup

### Real time (kalendarz)
- **Optimistycznie** (hopper 7× H100, brak kolejki): **~3 dni**
- **Realnie** (z kolejkowaniem SLURM, dostępność hopper, cleanup dysku): **~5–7 dni**
- **Pessymistycznie** (pascal/dgx fallback, multiple resume): **~10–14 dni**

---

## 10. Ryzyka i open questions

### Ryzyka techniczne
1. **Format YOLO loss vs DETR EL2N nieporównywalny** — YOLOv8 zwraca trzy komponenty (box, cls, dfl); DETR Query-81 daje scalar 1−s+entropy. Wymaga decyzji która linearna kombinacja YOLO loss "matches" DETR formula.
   *Mitigation:* preliminary correlation test na próbce 1k frames przed pełną ekstrakcją.

2. **Normalizacja per-frame loss** — DETR-EL2N używa min-max [0,1] z default=1.0 dla brak detekcji. YOLO zawsze daje loss > 0; nie ma "no detection" case.
   *Mitigation:* zachować z-score per-pool, opisać różnicę explicit w Methods.

3. **Trening YOLO do ep400 może saturate przed ep270** — historyczne dane pokazują YOLO ep270 = 78.74% jako produkcyjny peak. Możliwe że ep400 da dokładnie to samo (lub gorsze przez overfitting).
   *Mitigation:* `patience=50` automatic early stop; checkpointy co 10 epok; analiza trajektorii post-hoc.

4. **Eden disk 100% full** (per snapshot 2026-04-23) — blocker dla nowych jobów.
   *Mitigation:* przed startem cleanup checkpointów co 5 ep → co 10 ep w `~/DETR/Checkpoints/ablation/` (~30–50 GB do odzyskania). Lub usunąć stare YOLO base epochs (ep0–80) z `~/Yolo/train_out_5gpu_hopper/`.

### Open questions do promotora / autora
1. **Czy normalizować YOLO-EL2N tak samo jak DETR-EL2N?**
   Opcja A: zachować formula `0.7·(1−s_q81) + 0.3·H_norm` analogicznie dla YOLO (problem: brak "Query 81" w YOLO)
   Opcja B: użyć surowy YOLO loss znormalizowany [0,1] — bardziej naturalne dla YOLO
   **Domyślna rekomendacja clw1: Opcja B**, z uzasadnieniem w Methods że formula musi być adekwatna dla architektury

2. **Czy YE-B (DETR trenowany na YOLO-curated 20k) jest must-have?**
   Dodaje +5 dni GPU. Bez YE-B nadal można powiedzieć coś o YE-A i circular reasoning.
   **Rekomendacja:** YE-B jako optional — uruchomić tylko jeśli YE-A da niejednoznaczny wynik

3. **Czy YOLO-EL2N ranking ma być raportowany w artykule jako figura?** (uwaga #8 punkt 4 w UWAGI_PROFESORA)
   Opcje: histogram trudności, top-N samples mosaic, heatmap per-video
   **Rekomendacja:** histogram (1 panel) + Spearman correlation w tekście

4. **Submisja przed czy po wynikach YE-A?**
   Per UWAGI_PROFESORA sekcja 10 pkt 1 — to pytanie otwarte do promotora. Jeśli "must-have" → submission delay ~2 tygodnie.

---

## 11. Checklist wykonawczy

### Pre-flight (przed startem)
- [ ] Cleanup Eden disk (target: <80% full)
- [ ] Potwierdzenie z promotorem: opcja A vs B normalizacji YOLO-EL2N (open Q #1)
- [ ] Sprawdzenie dostępności hopper 7× H100 (`squeue -p hopper`)

### Implementacja
- [ ] Krok 1: napisać `extract_yolo_el2n.py`
- [ ] Krok 1: uruchomić ekstrakcję, weryfikować histogram + Spearman vs DETR-EL2N
- [ ] Krok 2: dodać parametr `--el2n_source` do `main_selection_pipeline.py`
- [ ] Krok 2: uruchomić selekcję, output `selected_dataset_yolo_el2n/`
- [ ] Krok 3: wygenerować `comparison_subsets_report.md` + Jaccard
- [ ] Krok 4: napisać SLURM `run_yolo_el2n_finetune.slurm`, `sbatch`
- [ ] Krok 4: monitorować (resume jeśli timeout)
- [ ] Krok 5: uruchomić benchmark 24 checkpointów
- [ ] Krok 5: wygenerować trajectory plot

### Post-eksperyment
- [ ] Wpisać wyniki do tabeli YE-A w UWAGI_PROFESORA sekcja 8
- [ ] Update Sekcja 6.5 Ablation Study w `Uwagi/Sekcje/06_results/` (lub nowy fragment do merged_v1)
- [ ] Update Limitations pkt 6 (z wynikami można osłabić caveat o circular reasoning)
- [ ] Krótki raport `RESULTS_YOLO_EL2N_ABLATION.md` w `Uwagi/Merge/Md/PoprawkiProfesor/`

---

## 12. Powiązane pliki

### Wewnątrz tego repo (artykuł)
- `UWAGI_PROFESORA.md` — wytyczne źródłowe (sekcje 7, 8, 9)
- `Uwagi/Merge/DETR_IEEE_article_merged_v1.tex` — plik docelowy update'u Sekcji 6.5 i Limitations
- `CLAUDE.md` (root) — globalny przewodnik ścieżek (sekcja 3.A YOLO checkpointy, 3.B benchmarki)
- `Uwagi/Merge/Md/PierwszaSynteza/HARDWARE_INVESTIGATION_REPORT.md` — referencja sprzętu Eden

### Zewnętrzne (kod + dane)
- `/mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/ablation/PLAN.md` — wzorzec ablacji DETR (V1–V5), zachować spójność metodologiczną
- `/mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/main_selection_pipeline.py` — punkt modyfikacji (parametr `--el2n_source`)
- `/mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/YOLO_EDEN_TRAIN/YoloTreningSesnions/20KTreningFrom170epochStart/exp/args.yaml` — referencyjna konfiguracja YOLO
- `/mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/benchmark_yolo_vs_detr_q81_all_epoch.py` — punkt rozszerzenia o YOLO-EL2N branch

### Eden (zdalnie via SSH)
- `~/YOLO_Checkpoints/epoch170.pt` — start checkpoint YOLO base
- `~/Yolo/20k_finetune/` — workdir nowego treningu
- `~/DETR/logs/` — referencyjne logi historyczne (do estymat czasu)

---

**Koniec planu.** Szacunkowy całkowity koszt (GPU + kalendarz): **~430 GPU-h, ~5–7 dni real time** przy hopper 7× H100. Decyzja idzie/nie idzie należy do autora i promotora po akceptacji punktu 10 (open questions).
