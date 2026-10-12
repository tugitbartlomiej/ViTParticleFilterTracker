# Indeks Klas Projektu (Serena MCP)

**Ostatnia aktualizacja:** 2026-10-12
**Metoda skanowania:** Serena MCP `get_symbols_overview` (pliki nowe/zmienione od 2026-01-16) + lista plików z klasami; reszta przeniesiona ze stanu 2026-01-16
**Uwaga:** `find_symbol` z pustym wzorcem nie działa (AssertionError) i jest bardzo wolne na całych katalogach - używaj `get_symbols_overview` per plik.

---

## AdvancedDatasetSelection/ (70 plików .py)

#### main_selection_pipeline.py
- `AdvancedDatasetSelectionPipeline` - główny pipeline selekcji: `load_datasets()`, `run()`, `_generate_output()`, `_generate_visualizations()`, `_generate_report()`

#### feature_extractors/
- `DINOExtractor` (dino_extractor.py) - cechy DINOv2 ViT-L/16: `extract_cls_features()`, `compute_features_batch()`, `compute_similarity_matrix()`
- `FastSAMExtractor` (fastsam_extractor.py) - złożoność sceny: `extract_masks()`, `compute_scene_complexity()`, `compute_complexity_batch()`
- `SAMExtractor` (sam_extractor.py) - SAM: `extract_masks()`, `compute_scene_complexity()`, `compute_features()`, `compute_features_from_path()`, `compute_complexity_batch()`, `compute_complexity_without_sam()`, `unload()`
- `FourierAnalyzer` (fourier_analyzer.py) - FFT: `compute_frequency_features()`, `compute_similarity_matrix()`, `filter_redundant()`, `analyze_diversity()`

#### selection_methods/
- `ClusterBasedSelector` (cluster_selector.py) - K-means FAISS GPU: `extract_all_features()`, `combine_features()`, `cluster_features()`, `select_representatives()`, `select_optimal_subset()`, `generate_selection_report()`
- `CombinedSelector` (combined_selector.py) - 4 kroki: `step1_fourier_prefilter()`, `step2_combine_features()`, `step3_k_center_select()`, `step4_el2n_rank()`, `select_optimal_subset()`, `compute_combined_score()`
- `KCenterGreedy` (k_center_greedy.py) - `compute_distances()`, `select()`, `select_with_diversity_guarantee()`, `compute_coverage_radius()`, `analyze_diversity()`; funkcja `select_diverse_subset()`
- `EL2NScorer` + `SimpleProxyModel` + `ImageDataset` (el2n_scorer.py) - `train_proxy_model()`, `compute_el2n_scores()`, `compute_el2n_for_detection()`, `rank_by_difficulty()`, `get_score_statistics()`, `save_scores()`/`load_scores()`; funkcja `compute_proxy_el2n_from_features()`
- `DETR_EL2N_Scorer` (detr_el2n_scorer.py) - EL2N z DETR Query 81: `compute_single_image_difficulty()`, `compute_el2n_scores()`, `save_selected_visualizations()`

#### utils/
- `COCOHandler` (coco_handler.py) - `load_annotations()`, `get_image_paths()`, `get_annotations_for_image()`, `filter_by_image_ids()`, `merge_datasets()`, `save_annotations()`, `copy_selected_images()`, `get_statistics()`
- visualization.py - klasy wizualizacji

#### inne (zawierają klasy, nieskanowane szczegółowo)
- `ablation/detr/evaluate_ablation.py`, `ablation/detr/local_eval_2026-06-11/common/ablation_eval_lib.py`
- `paper_visualizations/` - visualize_clustering(_3d), dino_features, discriminative_features, discriminative_per_image, fastsam_segmentation, fourier_spectrum

---

## YOLO_DETR_Benchmarks/scripts/ (49 plików)
- `benchmark_yolo_vs_detr_q81_all_epoch.py` - funkcje: `load_yolo_model()`, `load_detr_model()`, `prepare_coco_annotations(exclude_images=)`, `run_inference_on_split()`, `evaluate_predictions()`, `generate_report()`; klasa `DFLoss`
- `LEAKED_IMAGES_FILE` - JSON z 48 leaked images; `datasets_full` vs `datasets_clean`
- DETR_Background_Training: `BackgroundTaskExecutor`, `StatusManager`, `PipelineMonitor`

## BareDetr/ (10 plików)
- data/dataset.py: `CocoDetectionDataset`; data/transforms.py
- engine.py: `SmoothedValue`, `MetricLogger`, `train_one_epoch()`, `evaluate()`
- models/: backbone.py, detr.py, position_encoding.py, transformer.py

---

## SmallViTAblation/ (nowy moduł, ~75 plików .py poza outputs/assets)

Katalogi: `app/` (GUI PyQt + OpenGL, Python 3.13), `pipeline/` (curation, eval, features, koopman_langevin, transfer), `src/narzedzia/`, `scripts/`, `tools/`, `analysis/`, `eden/` (zadania klastra), `assets/` (trwałe przebiegi), `outputs/`, `experiments/`.

### app/ - aplikacja wizualizacji ruchu cząsteczek (natywne GPU, OpenGL 4.3)
- `Okno` (app.py) - główne okno; `ZAKLADKI_LENIWE`, `zbuduj_zakladke()`, ustawienia startowe/zapis/odczyt, `closeEvent()`; funkcja `main()`
- `ChmuraRuchuGPU` (ruch_gpu.py) - chmura cząsteczek ruchu na GPU (compute shadery, SSBO): `ustaw_os()`, `ustaw_budzet()`, `pompuj()`, `ustaw_klatke()`, `ustaw_koopmana()`, `przelicz_koopmana()`, `ustaw_fizyke()`, `ustaw_atencje()`, `zadaj_statystyki()`, `zadaj_wybor()`, `oblicz()`, `paint()`, `odczytaj()`, `zamknij()`
- `CzasteczkiGPU`, `PlaszczyznaGPU` (czasteczki3d_gpu.py) - cząsteczki 3D latentu: `ustaw_latent()`, `ustaw_obraz()`, `ustaw_wektory()`, `ustaw_mape()`, `oblicz()`, `paint()`, `odczytaj()`, `geometria()`
- ruch_zbioru.py - zbiór ruchu: `Config` (validate), `Timeline` (frame, read, read_raw, quant, stats, bounds, curves, bajty_klatki), `Engine` (run), `_Cancelled`; kwantyzacja Q16 (`kwantyzuj`, `odkwantyzuj`, `zapisz_klatke`, `konwertuj_q16`), `osie_szkla()`, `run()`
- pozostałe moduły z klasami: atlas_kanalow, czasteczki, czasteczki3d, detekcja, edytor_sd, galezie_uczenia, historie_checkpointow, korekta_latentu, krajobraz, makro_gpu, okno_zrodel_sily, opcje_atencji, panel_rownania, pchniecie, przestrzen_makro, przyczyny_ruchu, ruch_naturalny, stan_uczenia, uklad, widok_gl, widzety_edytora, wygoda
- zakładki: zakladka_detekcji, zakladka_filmu, zakladka_korekty, zakladka_pchniecia, zakladka_ruchu (+ _atencja, _fizyka, _koopman), zakladka_uczenia, zakladki_glowne
- testy/: test_ruch_gpu, test_ruch_naturalny, test_zakladka_ruchu (testy przeniesione do `app/testy/`)

### pipeline/
- `LatentLangevin` (koopman_langevin/latent_langevin.py) - `_choose_alpha()`, `summary()`, `known_mask()`, `scores()`; funkcja `_design()`
- curation/data.py, curation/features.py; eval/common.py; features/build_cache.py, features/common.py; transfer/check_iso_resource.py
- `src/narzedzia/deklaracja_wynikow.py`

---

## Statystyki (przybliżone)

| Moduł | Pliki .py | Uwagi |
|-------|-----------|-------|
| AdvancedDatasetSelection | 70 | 21 plików z klasami |
| YOLO_DETR_Benchmarks/scripts | 49 | |
| Annotators | 55 | nieskanowane |
| BareDetr | 10 | |
| SmallViTAblation | ~75 | app ~45 modułów |

---

## Ostatnie zmiany
- 2026-10-12: dodano SmallViTAblation (app GPU, pipeline Koopman/Langevin), SAMExtractor, CombinedSelector, KCenterGreedy, EL2NScorer, COCOHandler.
- 2026-01-16: mechanizm wykluczania leaked images w benchmarkach.

*Indeks wygenerowany przez Serena MCP: 2026-10-12*
*Projekt: ViTParticleFilterTracker*
