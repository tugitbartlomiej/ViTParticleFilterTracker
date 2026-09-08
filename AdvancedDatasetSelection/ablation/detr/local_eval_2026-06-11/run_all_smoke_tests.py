"""
Orkiestrator smoke testow: SEKWENCYJNIE przechodzi przez v1 -> v2 -> ... -> v7
w jednym procesie (zadnych podprocesow ani niczego w tle).

Dla kazdego wariantu wywoluje run_smoke() z biblioteki, po czym zwalnia GPU
i przechodzi do nastepnego. Blad jednego wariantu jest lapany i raportowany,
a sekwencja idzie dalej (chyba ze --stop-on-error). Na koncu zbiorcze
podsumowanie: OK (niezerowe mAP na obu datasetach) / PODEJRZANE / BLAD.

Uruchomienie:
  py -3.11 run_all_smoke_tests.py
  py -3.11 run_all_smoke_tests.py --variants v1 v5 v6   (tylko wybrane)
  py -3.11 run_all_smoke_tests.py --stop-on-error       (przerwij po 1. bledzie)
  py -3.11 run_all_smoke_tests.py --limit 50 --batch-size 4
"""

import argparse
import io
import json
import sys
import time
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "common"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ALL_VARIANTS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]


def smoke_verdict(variant_id: str) -> str:
    """Czyta metrics.json zapisany przez smoke test i ocenia wynik."""
    metrics_path = BASE_DIR / "results" / variant_id / "smoke" / "metrics.json"
    if not metrics_path.exists():
        return "BRAK WYNIKOW"
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            results = json.load(f).get("results", {})
    except (json.JSONDecodeError, OSError):
        return "USZKODZONY metrics.json"
    if not results:
        return "PUSTE WYNIKI"
    zero = [tag for tag, r in results.items() if r.get("mAP_0.5", 0) <= 0.0]
    if zero:
        return f"PODEJRZANE (mAP=0: {', '.join(zero)})"
    maps = ", ".join(f"{tag}={r['mAP_0.5']*100:.1f}%" for tag, r in results.items())
    return f"OK ({maps})"


def main():
    ap = argparse.ArgumentParser(description="Smoke testy wszystkich ablacji po kolei")
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS, help="ktore warianty (domyslnie wszystkie)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stop-on-error", action="store_true",
                    help="przerwij sekwencje po pierwszym bledzie")
    args = ap.parse_args()

    from ablation_eval_lib import run_smoke

    print(f"=== SMOKE TESTY (sekwencyjnie): {' -> '.join(args.variants)} "
          f"(limit={args.limit}, batch={args.batch_size}) ===")

    statuses = {}
    t_start = time.time()
    for vid in args.variants:
        print(f"\n{'='*70}\n>>> {vid.upper()}\n{'='*70}")
        t0 = time.time()
        try:
            run_smoke(vid, limit=args.limit,
                      batch_size=args.batch_size, num_workers=args.workers)
            statuses[vid] = smoke_verdict(vid)
            print(f"\n>>> {vid.upper()}: zakonczony w {time.time()-t0:.0f}s")
        except (Exception, SystemExit) as e:
            statuses[vid] = f"BLAD ({type(e).__name__}: {e})"
            print(f"\n>>> {vid.upper()}: BLAD po {time.time()-t0:.0f}s")
            traceback.print_exc()
            if args.stop_on_error:
                print(">>> --stop-on-error: przerywam sekwencje.")
                break

    print(f"\n\n{'='*70}")
    print(f"  PODSUMOWANIE SMOKE TESTOW ({time.time()-t_start:.0f}s lacznie)")
    print(f"{'='*70}")
    failed = []
    for vid in args.variants:
        status = statuses.get(vid, "POMINIETY")
        print(f"  {vid.upper():4} | {status}")
        if not status.startswith("OK"):
            failed.append(vid)

    if failed:
        print(f"\nDo wyjasnienia przed pelnymi runami: {', '.join(failed)}")
        sys.exit(1)
    print("\nWszystkie smoke testy OK - mozna uruchamiac pelne ewaluacje "
          "(najpierw v1\\run_eval_external.py i sanity-check ~85.74%).")


if __name__ == "__main__":
    main()
