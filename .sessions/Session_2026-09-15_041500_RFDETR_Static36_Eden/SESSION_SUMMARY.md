# RF-DETR Nano Static36 — stan przekazania

Branch: `codex/eden-nano-training-20260914`.

Commit wdrożony na Eden: `4ec9c8b`.

Katalog kampanii:

```text
~/smallvit_runs/2026-09-15_tracking_static36/
```

Zadania SLURM:

- `1807632`: przygotowanie pełnej puli treningowej, cech DINOv2, czterech
  selektorów i pakietu danych; jedna H100;
- `1807633`: 36 treningów RF-DETR Nano, zależność `afterok:1807632`, osiem H100.

DINOv2 jest przypięty do commita
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Przy ostatniej kontroli zadanie
przygotowawcze działało na Hopperze, model i wagi zostały pobrane, a cache
zawierał pierwsze ukończone partie. Zadanie treningowe oczekiwało na zależność.

Sprawdzenie stanu:

```bash
squeue -j 1807632,1807633 -o "%i|%T|%j|%M|%R"
tail -f ~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation/slurm_logs/static_prep_1807632.out
python ~/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation/eden/queue.py status \
  ~/smallvit_runs/2026-09-15_tracking_static36/queue/static36 --json
```

Nie uruchomiono Koopmana, Langevina ani Mori–Zwanziga. Czwarty selektor to
jawnie oznaczona statyczna hybryda DINOv2 i kotwic czasowych.
