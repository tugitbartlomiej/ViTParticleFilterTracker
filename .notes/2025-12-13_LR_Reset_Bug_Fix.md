# LR Reset Bug - Resume Training Problem

**Data:** 2025-12-13
**Dotyczy:** DETR fine-tuning na Eden
**Plik:** `detr_train_optimized.py`

---

## Problem

Przy wznawianiu treningu z checkpointu, **nowe wartosci LR byly nadpisywane starymi** z checkpointu.

### Objaw
```
[Checkpoint] Successfully loaded state from epoch 172
Current LR: 1.00e-04   # <-- ZLE! Powinno byc 5e-5
```

### Przyczyna
```python
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# ^ To nadpisuje param_groups[0]['lr'] starymi wartosciami!
```

Kolejnosc operacji:
1. Tworzysz optimizer z nowym LR (5e-5)
2. Ladowanie checkpointu -> `optimizer.load_state_dict()`
3. **LR wraca do starej wartosci (1e-4) z checkpointu!**

---

## Rozwiazanie

Dodaj **explicit LR reset** po wczytaniu checkpointu:

```python
# Po load_checkpoint():
if args.resume_training and latest_checkpoint_path:
    # CRITICAL: Reset LR to new values
    print(f"[LR Reset] Old LR: {optimizer.param_groups[0]['lr']:.2e}")

    optimizer.param_groups[0]['lr'] = args.lr  # main LR
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = args.lr_backbone  # backbone LR

    print(f"[LR Reset] New LR: {args.lr:.2e}")

    # Re-initialize scheduler for remaining epochs
    remaining_epochs = args.epochs - start_epoch
    if scheduler is not None and args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=args.lr_min
        )
        print(f"[Scheduler] Re-initialized for {remaining_epochs} epochs")
```

### Lokalizacja
`Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py` - linie 592-614

---

## Weryfikacja

Po naprawie logi pokazuja:
```
[LR Reset] Resetting learning rates to new fine-tuning values...
  Old LR (from checkpoint): 1.00e-04
  New LR: 5.00e-05 (backbone: 5.00e-06)
[Scheduler] Re-initialized cosine scheduler for 128 remaining epochs
Current LR: 5.00e-05   # <-- POPRAWNE!
```

---

## Dodatkowy Problem: Scheduler

Stary checkpoint **nie mial scheduler_state_dict** (trenowany bez schedulera).
Nowy cosine scheduler byl inicjalizowany dla 0->300 epochs, ale trening startowal z epoch 172.

**Rozwiazanie:** Re-init scheduler z `T_max = epochs - start_epoch`

---

## Lekcja

**Zawsze sprawdzaj LR w logach po resume!**

Checklist przed resume training:
- [ ] Sprawdz czy checkpoint ma scheduler_state_dict
- [ ] Zweryfikuj LR w logach po starcie
- [ ] Jesli LR zle -> dodaj explicit reset

---

## Powiazane pliki
- `Eden/TrainingRules/EDEN_SLURM_GUIDE.md` - Zasada #9
- `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py` - Fix
