"""
PELNA ewaluacja wariantu na SAME-DISTRIBUTION (dataset 2, wlasny zbior).

Ewaluacja na splitach val + test wlasnego zbioru (video-level split,
identyczny dla wszystkich wariantow; obrazy z puli 90,141 augmentowanych
klatek). To jest zrodlo drugiej kolumny tabeli Ablation Study.

Szacowany czas na RTX 3070: ~1.5-2.5 h (oba splity).
Mozna ograniczyc do jednego splitu: --splits test

Uruchomienie:
  py -3.11 run_eval_samedist.py [--splits val test] [--batch-size 4] [--workers 2]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "common"))

VARIANT_ID = HERE.parent.name

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Same-dist ewaluacja ablacji {VARIANT_ID}")
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from ablation_eval_lib import run_samedist
    run_samedist(VARIANT_ID, splits=tuple(args.splits),
                 batch_size=args.batch_size, num_workers=args.workers)
