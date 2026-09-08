"""
PELNA ewaluacja wariantu na EXTERNAL BENCHMARK (dataset 1, cross-dataset).

Inferencja JEDEN raz na scalonym zbiorze train+valid+test (2333 obrazy),
nastepnie metryki liczone z tych samych predykcji dla:
  - FULL  (2333 obrazow - protokol "ONE undivided test set" z artykulu),
  - CLEAN (FULL minus 48 leaked images),
  - per-split train/valid/test (debug, porownania z historycznymi benchmarkami).

To jest ZRODLO glownej kolumny tabeli Ablation Study w artykule.
Szacowany czas na RTX 3070: ~15-30 min.

Uruchomienie:
  py -3.11 run_eval_external.py [--batch-size 4] [--workers 2]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "common"))

VARIANT_ID = HERE.parent.name

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"External benchmark ablacji {VARIANT_ID}")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--protocol", choices=["all", "q81"], default="all",
                    help="all = wszystkie 100 queries; q81 = protokol artykulu "
                         "(1 predykcja ze slotu 81, prog 0.3)")
    args = ap.parse_args()

    from ablation_eval_lib import run_external
    run_external(VARIANT_ID, batch_size=args.batch_size, num_workers=args.workers,
                 protocol=args.protocol)
