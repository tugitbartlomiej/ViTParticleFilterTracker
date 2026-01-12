# Session: YOLO_Resume_Fix

**Created:** 2026-01-12 05:55:00 CET
**Type:** SSH / Training
**Status:** In Progress

## Quick Summary
Naprawiono błąd YOLO training gdzie `resume=True` nie działało na zakończonym checkpointcie (epoch 200). Stworzono nowy skrypt `yolo_train_20k_finetune_v2.py` używający pretrained weights zamiast resume. Job oczekuje w kolejce na Eden z powodu niskiego priorytetu (fair-share).

## Key Results
- Zidentyfikowano przyczynę błędu: Ultralytics `resume=True` nie działa dla zakończonego treningu
- Stworzono poprawiony skrypt v2 z logiką `--pretrained_path`
- Job 1509626 oczekuje w kolejce (priorytet 950 vs ~100,000 innych)
- dgx-2/dgx-3 niedostępne (awaria klimatyzacji)

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
