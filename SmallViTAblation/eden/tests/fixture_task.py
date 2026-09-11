#!/usr/bin/env python3
"""
Falszywe "zadanie treningowe" do testow kolejki (bez GPU).

Uzycie: fixture_task.py <run_id> <marker_dir>
Sterowanie przez zmienne srodowiskowe:
    TASK_SLEEP       czas snu w sekundach (domyslnie 0.1)
    FAIL_RUN_IDS     lista run_id (po przecinku) konczacych sie exit 1
    LONG_RUN_IDS     lista run_id spiacych LONG_SLEEP (domyslnie 30) s
                     — do symulowania ubicia workera w polowie
Kazde uruchomienie DOPISUJE wiersz do <marker_dir>/<run_id> — liczba wierszy
= liczba wykonan (test "dokladnie raz" / "wznowienie po kill").
"""
import os
import sys
import time
from pathlib import Path


def main() -> int:
    run_id, marker_dir = sys.argv[1], Path(sys.argv[2])
    marker_dir.mkdir(parents=True, exist_ok=True)
    with open(marker_dir / run_id, "a", encoding="utf-8") as f:
        f.write(f"run pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"t={time.time()}\n")

    fail_ids = {s.strip() for s in os.environ.get("FAIL_RUN_IDS", "").split(",") if s.strip()}
    long_ids = {s.strip() for s in os.environ.get("LONG_RUN_IDS", "").split(",") if s.strip()}
    if run_id in fail_ids:
        print(f"fixture: celowa awaria {run_id}", file=sys.stderr)
        return 1
    sleep_s = float(os.environ.get("LONG_SLEEP", "30")) if run_id in long_ids \
        else float(os.environ.get("TASK_SLEEP", "0.1"))
    time.sleep(sleep_s)
    print(f"fixture: {run_id} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
