"""
SMOKE TEST wariantu (wariant = nazwa folderu, w ktorym lezy ten skrypt).

Szybki test poprawnosci na 50 obrazach z KAZDEGO z dwoch datasetow:
  1. external benchmark, split test (dokladnie 50 obrazow),
  2. same-distribution, split test (pierwsze 50 obrazow).

Cel: sprawdzic ladowanie checkpointa, inferencje i niezerowe mAP,
ZANIM uruchomimy pelne, wielogodzinne ewaluacje.

Uruchomienie:
  py -3.11 run_smoke_test.py [--limit 50] [--batch-size 4] [--workers 2]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "common"))

VARIANT_ID = HERE.parent.name  # np. "v1"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Smoke test ablacji {VARIANT_ID}")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from ablation_eval_lib import run_smoke
    run_smoke(VARIANT_ID, limit=args.limit,
              batch_size=args.batch_size, num_workers=args.workers)
