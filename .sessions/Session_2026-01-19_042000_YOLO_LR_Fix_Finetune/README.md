# Session: YOLO LR Fix & Fine-tune ep170-500

**Created:** 2026-01-19 04:20
**Type:** Training / SSH
**Status:** Active (Job Running)

## Quick Summary
Naprawiono krytyczny bug z Learning Rate w treningu YOLO - `optimizer=auto` ignorował ustawiony `lr0`. Zmieniono na explicit `optimizer=SGD` i uruchomiono fine-tuning od epoch 170 do 500 na 4x H100 GPU (partycja hopper). LR ustawiony na 0.001 (10x mniejszy niż oryginalny) dla stabilnego fine-tuningu.

## Key Results
- Fix: `optimizer='SGD'` zamiast `'auto'` - teraz LR jest respektowany
- Job 1523692 uruchomiony na 4x H100 (hopper partition)
- LR = 0.001 -> 0.0001 (lrf=0.1), 330 nowych epok
- Szacowany czas: ~2.75 dni

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
