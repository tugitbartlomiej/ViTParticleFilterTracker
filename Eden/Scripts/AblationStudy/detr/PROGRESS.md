# Ablation Study — Progress Tracker

**Plan**: PLAN.md v16 (7 wariantów: V1-V7)
**Start**: 2026-03-21 ~01:00 UTC
**Ostatnia aktualizacja**: 2026-03-22 20:00 UTC

---

## Postęp kroków

| Krok | Opis | Status | Data | Uwagi |
|------|------|--------|------|-------|
| E1 | sklearn na Eden | ✅ DONE | 2026-03-20 | `conda activate yolo_py310 && pip install scikit-learn` |
| E2 | External benchmark na Eden | ✅ DONE | 2026-03-20 | Roboflow dataset, poprawiona struktura train/valid/test, category_id naprawione |
| W1 | run_ablation_all.sh → 7 wariantów | ✅ DONE | 2026-03-20 | afterany → afterok |
| KROK 0 | Aktualizacja skryptów (B1-B6, G4-G5, L1-L5) | ✅ DONE | 2026-03-20 | Claude_worker naprawił wszystkie blokery |
| KROK 1 | Transfer feature cache na Eden | ✅ DONE | 2026-03-21 | 369 MB → ~/DETR/ablation/feature_cache/ |
| KROK 2 | Transfer skryptów na Eden | ✅ DONE | 2026-03-21 | 6 plików → ~/DETR/ |
| KROK 3 | Generowanie wariantów (CPU) | ✅ DONE | 2026-03-21 | 7 katalogów v1-v7, video_split.json |
| KROK 3.5 | Preflight | ✅ DONE | 2026-03-21 | sklearn OK, pycocotools OK, external OK |
| **KROK 4** | **Canary test (5 epok)** | **✅ DONE** | **2026-03-21/22** | **6/6 best models saved** |
| KROK 5 | Pełny trening z --use_amp (300 epok) | 🔄 W KOLEJCE | 2026-03-30 | 6 jobów sekwencyjnych na hopper-2, IDs: 1641331-1641336 |
| KROK 6 | Monitorowanie | ⏳ | — | — |
| KROK 7 | Ewaluacja (same-dist + cross-dataset) | ⏳ | — | — |

---

## Canary — szczegóły (KROK 4)

| Wariant | Node | Czas | Exit | Best val loss | Best model |
|---------|------|------|------|--------------|------------|
| V2 (NO_FOURIER) | hopper-2 → pascal → hopper-2 | ~6h (TIMEOUT, ale OK) | TIMEOUT | saved | ✅ 159M |
| V3 (NO_EL2N) | hopper-2 | ~1.5h | 0 | 0.1412 | ✅ 159M |
| V4 (QUALITY_ONLY) | hopper-2 | ~2h | 0 | saved | ✅ 159M |
| V5 (RANDOM) | dgx-4 → hopper-2 | ~2h | 0 | saved | ✅ 159M |
| V6 (DIVERSITY_ONLY) | pascal | ~21h (anulowany, best saved) | CANCELLED | saved | ✅ 159M |
| V7 (EL2N_ONLY) | hopper-2 | ~2h | 0 | saved | ✅ 159M |

**Kryteria canary:**
- [x] Train loss < 1.0 — ALL PASSED (0.15-0.22)
- [x] Best model saved — ALL 6/6
- [x] Brak OOM — ALL PASSED
- [x] Brak crashy — ALL PASSED (V2 timeout i V6 cancel nie są crashe)

---

## Problemy napotkane podczas canary

1. **V2/V6 na pascal** — za wolne (~18s/batch vs ~1s/batch na hopper-2). Rozwiązanie: przeniesione na szybsze nody.
2. **hopper-2 /tmp pełny** — stare TMPDIR nie wyczyszczone po TIMEOUT V2. V6 musiał jechać na pascal.
3. **MaxSubmitJobLimit** — konto `null` ma limit 0 jobów. Rozwiązanie: `--account=transformers_vsc`.
4. **VARIANT env var** — skrypt wymaga `--export=ALL,VARIANT=vN` lub `VARIANT=vN sbatch ...`.

---

## KROK 5 — Pełny trening z AMP (URUCHOMIONY 2026-03-30)

**Parametry:** epoch 170 → 300, LR 5e-5, cosine scheduler, patience 30, **--use_amp**
**Klaster:** Eden, partycja `hopper`, węzeł `hopper` (H100 GPU, MaxTime=5 dni)
**Tryb:** Sekwencyjny (afterany dependency), 1 job naraz

### Job IDs — kolejka (aktywna)

| Wariant | Job ID | Status | Dependency |
|---------|--------|--------|------------|
| V2 (NO_FOURIER) | **1641365** | PD (Priority — czeka na GPU) | — |
| V3 (NO_EL2N) | **1641366** | PD (Dependency) | afterany:1641365 |
| V4 (QUALITY_ONLY) | **1641367** | PD (Dependency) | afterany:1641366 |
| V5 (RANDOM) | **1641368** | PD (Dependency) | afterany:1641367 |
| V6 (DIVERSITY_ONLY) | **1641369** | PD (Dependency) | afterany:1641368 |
| V7 (EL2N_ONLY) | **1641370** | PD (Dependency) | afterany:1641369 |

```bash
squeue -u bpiotrowski                          # sprawdź status
tail -f ~/DETR/logs/ablation_v2_1641365.log    # logi V2
```

### ⚠️ Anulowane joby (błędna partycja)
IDs 1641331–1641336 były na `hopper-2` (MaxTime=24h < 3 dni) → `PartitionTimeLimit`. Anulowane i ponownie wysłane.

### Kluczowe odkrycia (2026-03-30)

- Partycja `standard` **nie istnieje** → zmieniono na `hopper`
- `hopper-2` MaxTime=**24h** — za krótkie dla 130 epok (~3 dni) → użyto `hopper` (MaxTime=5 dni)
- `/raid` na `hopper` = 14TB wolnego → skrypt auto-wybiera TMPDIR=/raid (identycznie jak hopper-2)
- `SLURM_TMPDIR` nie jest ustawiany automatycznie przez SLURM
- Poprzednie modele (bez AMP, Mar 23-25) zostaną nadpisane przez lepsze

### Po zakończeniu KROKU 5 — uruchom ewaluację

```bash
cd ~/DETR
sbatch --account=transformers_vsc --partition=hopper-2 \
    Eden/Scripts/AblationStudy/run_ablation_eval.slurm
```

**Uwaga:** Użyć `--account=transformers_vsc` dla wszystkich jobów!

---

## Best models na Eden

```
~/DETR/ablation_best/
├── v2/best_model.pth  (159M, 2026-03-23 07:17)  ← canary, zostanie nadpisany przez KROK 5
├── v3/best_model.pth  (159M, 2026-03-23 03:46)  ← canary, zostanie nadpisany
├── v4/best_model.pth  (159M, 2026-03-23 14:53)  ← canary, zostanie nadpisany
├── v5/best_model.pth  (159M, 2026-03-25 11:52)  ← canary, zostanie nadpisany
├── v6/best_model.pth  (159M, 2026-03-25 12:09)  ← canary, zostanie nadpisany
└── v7/best_model.pth  (159M, 2026-03-23 11:17)  ← canary, zostanie nadpisany
```

V1 (FULL) — istniejący checkpoint: `~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth`

**Uwaga:** Modele z canary (bez AMP) to wyniki ~5 epok. Pełny trening (KROK 5) powinien je znacznie poprawić.
