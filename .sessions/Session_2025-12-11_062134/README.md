# Session: Session_2025-12-11_062134

**Created:** 2025-12-11 06:21:34
**Type:** Analysis | Development
**Status:** Completed

## Quick Summary
Aktualizacja Advanced Dataset Selection Pipeline - dodanie obsługi DETR Query 81 only, sekwencyjnego ładowania modeli (GPU memory optimization), oraz raportów wyjaśniających dlaczego każdy obraz został wybrany do re-treningu.

## Key Results
- DETR EL2N Scorer używa tylko Query 81 (najlepszy dla tooltip detection)
- Sekwencyjne ładowanie: DINO → unload → SAM3 → unload → DETR → unload
- Nowe raporty: `selection_reasons.json` + `selection_reasons.txt`
- Wizualizacje DETR Q81 bbox dla wybranych obrazów

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja

## Important TODO (Next Session)
**Problem:** Dla re-treningu trzeba wybierać obrazy z niskim Q81 score (<20%), ale obecna logika EL2N spłaszcza wszystkie próbki poniżej progu 0.3 do tej samej trudności.

**Rozwiązanie:** Dodać filtr `q81_max_score: 0.2` w config.

## Related Sessions
- Previous: Kontynuacja pracy nad AdvancedDatasetSelection
