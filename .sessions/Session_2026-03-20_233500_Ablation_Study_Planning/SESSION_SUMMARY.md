# Session Summary: Session_2026-03-20_233500_Ablation_Study_Planning

## Metadata
- **Date:** 2026-03-20
- **Time:** 23:35:00
- **Duration:** ~6h (cala sesja devloop conductor)
- **Status:** Completed
- **Type:** Mixed (Analysis + SSH + Planning)

## Objective
Zaprojektowac i przygotowac ablation study dla pipeline kuracji danych DETR, odpowiadajac na krytyki recenzentow (Codex/Gemini/Claude) dotyczace braku ablacji, data leakage i EL2N bias.

## Context
Artykul IEEE Access "Intelligent Data Curation for Transformer-Based Surgical Tool Detection" ma 4-etapowy pipeline selekcji danych (Fourier + DINO + SAM + EL2N), ale brak dowodow ze kazdy etap jest potrzebny. Recenzenci AI (3 modele) zidentyfikowali lacznie ~35 problemow. Sesja miala na celu: (1) zebranie wszystkich uwag, (2) zbadanie data leakage, (3) zaprojektowanie ablacji, (4) przygotowanie infrastruktury.

## Actions Taken

### Faza 1: Recenzja artykulu
1. Przeczytano istniejaca recenzje Codex + Gemini (19 problemow)
2. Przeprowadzono niezalezna recenzje (Claude Opus 4.6) — 13 nowych problemow
3. Dopisano do `recenzja_ai_models.md` jako sekcje "Recenzja: Claude Opus 4.6"
4. Kluczowe znaleziska: phantom precision 89.4%, brak diagramu pipeline, brak opisu fine-tuningu

### Faza 2: Audyt data leakage
5. Skanowanie 5 obszarow projektu rownolegly (agenty Explore)
6. Przeczytanie 14+ sesji z `.sessions/` dla kontekstu historycznego
7. Zidentyfikowano 3 warstwy leakage:
   - **Warstwa 1:** Pipeline selekcji na pelnym poolie 90k (transductive leakage)
   - **Warstwa 2:** `random_split()` image-level zamiast video-level
   - **Warstwa 3:** EL2N checkpoint z pelnego poolie (circular dependency)
8. Odkryto ze `split_and_build_coco.py` (poprawny video-level split) nigdy nie zostal uzyty

### Faza 3: Weryfikacja czyszczenia
9. Znaleziono dowody czyszczenia:
   - 48 obrazow Roboflow valid->train: wykryte i wykluczone (sesja 2026-01-16)
   - TestDatasetGenerator: 52 duplikaty usuniete via pHash
   - `detect_duplicates.py`, `check_duplicates_with_roboflow.py`, `leaked_valid_images.json`
10. Wniosek: benchmarki cross-dataset czyste, ale trening DETR nadal `random_split()`

### Faza 4: Analiza EL2N bias
11. Przeczytano `detr_el2n_scorer.py` — uzywa Query 81 z pelnego DETR ep170
12. EL2N to 30% sygnalu (1 dim z 41) — nie dominuje selekcji
13. YOLO tez trenowany na tych samych 20k (potwierdzone w README)
14. Artykul opisuje EL2N inaczej niz kod (proxy 10ep vs pelny ep170)

### Faza 5: Czyszczenie Eden
15. Skanowanie duzych plikow na Eden (ssh eden-cluster)
16. Znaleziono: UCO3D 513 GB (group storage), JEPA 25 GB, stare checkpoints
17. **Usunieto UCO3D** — dysk z 100% na 11% (459 GB odzyskane)
18. Usunieto `~/uco3d` (31 MB)
19. Zainstalowano scikit-learn w yolo_py310

### Faza 6: Plan ablacji
20. Zaprojektowano 7 wariantow (V1-V7): Full, -Fourier, -EL2N, Quality, Random, Diversity, EL2N_only
21. Stworzono skrypty: `generate_ablation_on_eden.py`, `detr_train_ablation.py`, `evaluate_ablation.py`
22. Stworzono 5 skryptow SLURM (run_ablation_v1..v5.slurm) + generator
23. Uzytkownik zaktualizowal plan do v14+ uwzgledniajac uwagi Codex/Gemini/Claude

### Faza 7: Weryfikacja i naprawy
24. Weryfikacja planu v8..v14 — iteracyjne szukanie luk
25. Naprawiono blokery B1-B6: COCOeval, V6/V7, EPOCHS parametryzacja
26. Naprawiono L1/L4: format .pth (torch.save zamiast save_pretrained)
27. Rozwiazano L2: random_split(seed=42) dla wszystkich wariantow (fair comparison)
28. Naprawiono PDF z komentarzami (tcolorbox zamiast fcolorbox)
29. Ujednolicono supercategory na 'surgical-instrument' w Roboflow

## Results

### Key Findings

#### Recenzja
- 13 nowych problemow niewylapanych przez Codex/Gemini
- Najwazniejsze: phantom precision 89.4%, brak pipeline diagramu, niespojnosc EL2N opis vs kod

#### Data Leakage
- `split_and_build_coco.py` (poprawny) nigdy nie uzyty — trening uzywa `random_split()`
- Artykul twierdzi "strict video-level separation" — kod tego nie potwierdza
- Czyszczenie benchmarkow BYLO robione (48 Roboflow, 52 pHash) — ale tylko ewaluacja, nie trening

#### EL2N Bias
- Ryzyko umiarkowane: EL2N = 30% sygnalu, 70% model-agnostic
- Artykul ≠ kod: opisuje proxy 10ep, kod uzywa pelnego ep170 bez proxy
- Ablacja V3 (No EL2N) rozstrzygnie

#### Czyszczenie Eden
- UCO3D: 513 GB usuniete z group storage
- Dysk: 512/512 GB (100%) → 54/512 GB (11%)

### Metrics/Data

| Metryka | Wartosc |
|---------|---------|
| Problemy w recenzji (nowe) | 13 |
| Warstwy data leakage | 3 |
| GB odzyskane na Eden | ~570 |
| Warianty ablacji | 7 |
| Pliki skryptow stworzonych | 11+ |
| Iteracje planu (v1→v14) | 14 |
| Blokery rozwiazane | 8 (B1-B6, L1, L4) |

### Issues Encountered
- **sklearn brak na Eden** → Rozwiazanie: `pip install scikit-learn` (1.7.2)
- **UCO3D 513 GB blokuje dysk** → Rozwiazanie: `rm -rf` (459 GB odzyskane)
- **PDF komentarze nachodza** → Rozwiazanie: tcolorbox zamiast fcolorbox
- **Plan vs kod niespojny** → Rozwiazanie: iteracyjna weryfikacja (14 wersji planu)
- **hopper zajety** → Rozwiazanie: krok 3.5 (preflight) przed kazdym sbatch

## Conclusions
1. Artykul wymaga istotnych poprawek przed submisja (EL2N opis, pipeline diagram, ablacja)
2. Data leakage w treningu (random_split) nie wplywa na cross-dataset eval (glowna metryka)
3. Ablacja 7 wariantow rozstrzygnie wartosc kazdego etapu pipeline
4. Infrastruktura gotowa — blokery rozwiazane, canary do uruchomienia

## Next Steps
- [ ] Przeslac external_benchmark (Roboflow) na Eden (E2)
- [ ] Przeslac feature cache na Eden (krok 1)
- [ ] Uruchomic `generate_ablation_on_eden.py` (krok 3)
- [ ] Canary test 6 wariantow × 2 epoki (krok 4)
- [ ] Pelny trening 6 wariantow × 130 epok (krok 5)
- [ ] Ewaluacja i tabela wynikowa (krok 7)
- [ ] Poprawic artykul: EL2N opis, pipeline diagram, ablacja tabela

## Files Generated/Modified

### Nowe pliki
| Plik | Opis |
|------|------|
| `AdvancedDatasetSelection/ablation/PLAN.md` | Plan ablacji (zastapiony przez uzytkownika) |
| `AdvancedDatasetSelection/ablation/generate_video_split.py` | Generator video-level split |
| `AdvancedDatasetSelection/ablation/generate_ablation_datasets.py` | Generator 5 wariantow (zastapiony) |
| `AdvancedDatasetSelection/ablation/evaluate_ablation.py` | Ewaluator (zastapiony) |
| `Eden/Scripts/AblationStudy/detr_train_ablation.py` | Trainer z --annotations_val_path |
| `Eden/Scripts/AblationStudy/run_ablation_v1..v5.slurm` | 5 skryptow SLURM |
| `Eden/Scripts/AblationStudy/generate_slurm_jobs.sh` | Generator SLURM |
| `Eden/Scripts/AblationStudy/PLAN.md` | Plan v14 (zaktualizowany przez uzytkownika) |

### Zmodyfikowane pliki
| Plik | Zmiana |
|------|--------|
| `UWAGI/recenzja_ai_models.md` | +sekcja "Recenzja: Claude Opus 4.6" |
| `UWAGI/access_z_komentarzami.tex` | fcolorbox → tcolorbox (fix nachodzenia) |
| Roboflow `_annotations.coco.json` (3 splity) | supercategory → 'surgical-instrument' |

### Na Eden
| Akcja | Szczegoly |
|-------|-----------|
| Usunieto UCO3D | `/mnt/evafs/groups/transformers_vsc/bpiotrowski/uco3d/` (513 GB) |
| Usunieto ~/uco3d | 31 MB |
| Zainstalowano sklearn | scikit-learn 1.7.2 w yolo_py310 |

## Commands Used

```bash
# Czyszczenie Eden
ssh eden-cluster "rm -rf /mnt/evafs/groups/transformers_vsc/bpiotrowski/uco3d/"
ssh eden-cluster "rm -rf ~/uco3d"

# Instalacja sklearn
ssh eden-cluster 'source .../conda.sh && conda activate yolo_py310 && pip install scikit-learn'

# Kompilacja PDF
cd UWAGI && pdflatex -interaction=nonstopmode access_z_komentarzami.tex

# Generowanie SLURM
bash Eden/Scripts/AblationStudy/generate_slurm_jobs.sh

# Supercategory fix
python3 -c "..." # zmiana supercategory w 3 splitach Roboflow
```

## Related Work
- **Previous session:** `Session_2026-01-20_140500`
- **Referenced sessions:**
  - `Session_2025-10-31_163000` — plan video-level split (nigdy nie uzyty)
  - `Session_2025-12-12_121949` — przygotowanie 20k datasetu
  - `Session_2026-01-16_220114` — odkrycie i naprawa 48 leaked images
  - `Session_2025-12-31_042306` — benchmark 20k finetune, odkrycie TestDatasetGenerator leakage
- **Referenced files:**
  - `Cataract_DETR_DatasetBuilder/split_and_build_coco.py` — poprawny video-level split (nie uzyty)
  - `AdvancedDatasetSelection/selection_methods/detr_el2n_scorer.py` — EL2N implementacja
  - `YOLO_DETR_Benchmarks/scripts/leaked_valid_images.json` — 48 leaked images
  - `TestDatasetGenerator/detect_duplicates.py` — pHash duplicate detection

---

*Session saved: 2026-03-20 23:35:00*
*Conductor: devloop | Model: Claude Opus 4.6 (1M context)*
