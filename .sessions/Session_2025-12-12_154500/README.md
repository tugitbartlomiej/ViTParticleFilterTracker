# Session: Session_2025-12-12_154500

**Created:** 2025-12-12 15:45:00
**Type:** SSH | Training
**Status:** Active

## Quick Summary
Uruchomienie treningu DETR na 20k wybranych obrazach na klastrze Eden (Pascal GPU). Sesja obejmowała debugowanie problemów z SLURM, AMP, checkpoint resume i batch size. Utworzono dokumentację zasad SLURM dla Eden.

## Key Results
- Trening DETR uruchomiony na Pascal (4x P100 16GB) od epoch 170
- Rozwiązano problemy: AMP conflict, epochs < checkpoint, start_epoch +1, batch_size OOM
- Utworzono kompletny przewodnik `Eden/TrainingRules/EDEN_SLURM_GUIDE.md`

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
