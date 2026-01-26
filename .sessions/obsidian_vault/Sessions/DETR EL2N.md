---
date: 2026-01-18
time: 01:25
type: Mixed
topics: [DETR, Dataset Selection, EL2N, GPU, Pipeline]
aliases: ["DETR EL2N"]
---

# DETR EL2N

> [!info] Session Info
> **Date:** 2026-01-18 01:25
> **Type:** Mixed
> **ID:** `Session_2026-01-18_012500_PyTorch_GPU_Fix`

## Objective

Naprawić problem z DETR EL2N scorer który nie używał GPU podczas przetwarzania obrazów w pipeline selekcji datasetu.

## Topics

[[DETR]] [[Dataset Selection]] [[EL2N]] [[GPU]] [[Pipeline]]

## Actions Taken

1. **Dodano diagnostykę GPU do `detr_el2n_scorer.py`:** - Funkcja `verify_gpu_available()` - pełna weryfikacja CUDA - Rozbudowane logowanie w `_initialize_model()` - Weryfikacja device przy pierwszym bat
2. **Zdiagnozowano problem:** - Test GPU pokazał: `PyTorch version: 2.9.1+cpu` - CUDA available: False - PyTorch był zainstalowany bez obsługi CUDA
3. **Naprawiono problem:** - Odinstalowano PyTorch CPU: `pip uninstall torch torchvision torchaudio` - Zainstalowano PyTorch z CUDA 12.4: `pip install torch torchvision torchaudio --index-url https://dow
4. **Weryfikacja:** - PyTorch: 2.6.0+cu124 - CUDA available: True - GPU: NVIDIA GeForce RTX 3070 - Model działa na cuda:0

## Key Findings

- Problem nie był w kodzie - PyTorch był zainstalowany w wersji CPU-only
- Użytkownik miał dwie instalacje Pythona z różnymi wersjami PyTorch
- Po reinstalacji model poprawnie ładuje się na GPU (0.18 GB memory)

## Files Modified

- `main_selection_pipeline.py`
- `detr_el2n_scorer.py`
- `AdvancedDatasetSelection/selection_methods/detr_el2n_scorer.py`

## Next Steps

- [ ] PyTorch z CUDA zainstalowany
- [ ] Model DETR działa na GPU
- [ ] Uruchomić pełny pipeline selekcji z GPU acceleration

## Related Sessions

- [[Session]] (2025-10-20)
- [[Session]] (2025-10-20)
- [[Session]] (2025-10-21)
- [[Session]] (2025-10-22)
- [[Session]] (2025-10-31)
- [[Dataset Selection]] (Unknown)
- [[DETR EL2N]] (2025-12-11)
- [[Dataset Selection]] (2025-12-11)
- [[Session]] (2025-12-11)
- [[Dataset Selection]] (2025-12-12)
- [[Dataset Selection]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Dataset Selection]] (2025-12-12)
- [[Visualization]] (2025-12-12)
- [[Visualization]] (2025-12-12)
- [[Query 81]] (2025-12-12)
- [[Visualization]] (2025-12-13)
- [[Visualization]] (2025-12-13)
- [[Query 81]] (2025-12-13)
- [[Session]] (2025-12-13)
- [[Dataset Selection]] (2025-12-31)
- [[Session]] (2026-01-11)
- [[YOLO Fix]] (2026-01-11)
- [[IEEE Article]] (2026-01-12)
- [[IEEE Article]] (2026-01-12)
- [[IEEE Article]] (2026-01-12)
- [[YOLO Resume]] (2026-01-12)
- [[YOLO Resume]] (2026-01-12)
- [[Dataset Selection]] (2026-01-16)
- [[DETR EL2N]] (2026-01-18)
- [[Dataset Selection]] (2026-01-18)
- [[YOLO Resume]] (2026-01-18)
- [[Visualization]] (2026-01-26)

---

> [!tip] Navigation
> - [[Sessions Index|Back to Index]]
> - [[Mixed|All Mixed Sessions]]
