# Session: Session_2025-12-12_121949

**Created:** 2025-12-12 12:19:49
**Type:** Training | SSH
**Status:** Active

## Quick Summary
Przygotowanie datasetu 20k do fine-tuningu DETR na klastrze EDEN. Połączenie adnotacji z dwóch źródeł COCO, naprawa duplikatów bounding boxów, wizualizacja i uruchomienie joba treningowego.

## Key Results
- Stworzono `merged_20k_annotations.json` - 20,000 obrazów, 20,000 adnotacji (1 bbox/obraz)
- Naprawiono 216 duplikatów bbox (zachowano największe)
- Job wysłany na EDEN: `1453424` (partycja `long`, 4 GPU, 5 dni)

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
- `ssh/` - Skrypty i komendy klastra

## Related Sessions
- Previous: `Session_2025-12-11_205605` - Advanced Dataset Selection
