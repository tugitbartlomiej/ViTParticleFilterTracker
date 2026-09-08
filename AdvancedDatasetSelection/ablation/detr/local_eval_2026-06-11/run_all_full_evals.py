"""
Orkiestrator PELNYCH ewaluacji ablacji: SEKWENCYJNIE v1 -> v2 -> ... -> v7
w jednym procesie (bez podprocesow, niczego w tle).

Dwie fazy (kolejnosc celowa - gdy przerwiesz w polowie, masz komplet
najwazniejszych liczb):
  FAZA 1: external benchmark (cross-dataset, ~15-30 min/wariant) dla v1..v7
  FAZA 2: same-distribution  (~1.5-2.5 h/wariant) dla v1..v7

Blad ktoregokolwiek wariantu jest lapany i raportowany, a sekwencja
PRZECHODZI DO NASTEPNEGO (chyba ze --stop-on-error). GPU zwalniane miedzy
wariantami. Po wszystkim: tabela zbiorcza (aggregate_results -> SUMMARY.md)
i sanity-check V1 vs headline artykulu (~85.74%).

Uruchomienie:
  py -3.11 run_all_full_evals.py                        (obie fazy, v1..v7)
  py -3.11 run_all_full_evals.py --mode external        (tylko faza 1)
  py -3.11 run_all_full_evals.py --mode samedist        (tylko faza 2)
  py -3.11 run_all_full_evals.py --variants v1 v5 v6    (tylko wybrane)
  py -3.11 run_all_full_evals.py --stop-on-error
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
VARIANT_CHOICES = ["v0"] + ALL_VARIANTS  # v0 = uncurated ep170 (tylko external)
V1_HEADLINE_MAP05 = 85.74  # % - DETR 20k ep270, LEGACY benchmark 20260112 (artykul)
V0_HEADLINE_MAP05 = 2.04   # % - uncurated ep170, ten sam benchmark/protokol


def _read_results(variant_id: str, run_name: str) -> dict:
    path = BASE_DIR / "results" / variant_id / run_name / "metrics.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("results", {})
    except (json.JSONDecodeError, OSError):
        return {}


def external_verdict(variant_id: str, run_name: str = "external") -> str:
    r = _read_results(variant_id, run_name)
    if not r:
        return "BRAK WYNIKOW"
    full = r.get("FULL", {}).get("mAP_0.5")
    valid = r.get("split_valid", {}).get("mAP_0.5")
    if full is None:
        return "NIEKOMPLETNE (brak FULL)"
    if full <= 0.0:
        return "PODEJRZANE (FULL mAP=0)"
    valid_s = f", valid={valid*100:.2f}%" if valid is not None else ""
    return f"OK (FULL={full*100:.2f}%{valid_s})"


def samedist_verdict(variant_id: str, run_name: str = "samedist") -> str:
    r = _read_results(variant_id, run_name)
    if not r:
        return "BRAK WYNIKOW"
    parts = [f"{split}={r[split]['mAP_0.5']*100:.2f}%"
             for split in ("val", "test") if split in r]
    if not parts:
        return "NIEKOMPLETNE"
    zero = [s for s in ("val", "test") if s in r and r[s]["mAP_0.5"] <= 0.0]
    if zero:
        return f"PODEJRZANE (mAP=0: {', '.join(zero)})"
    return f"OK ({', '.join(parts)})"


def run_phase(phase_name: str, variants, run_fn, verdict_fn,
              statuses: dict, stop_on_error: bool, **kwargs) -> bool:
    """Sekwencyjna faza ewaluacji. Zwraca False, gdy przerwano przez
    --stop-on-error; blad pojedynczego wariantu NIE przerywa fazy."""
    for vid in variants:
        print(f"\n{'='*70}\n>>> {phase_name} {vid.upper()}\n{'='*70}")
        t0 = time.time()
        try:
            run_fn(vid, **kwargs)
            statuses[vid] = verdict_fn(vid)
            print(f"\n>>> {phase_name} {vid.upper()}: zakonczony w {time.time()-t0:.0f}s")
        except (Exception, SystemExit) as e:
            statuses[vid] = f"BLAD ({type(e).__name__}: {e})"
            print(f"\n>>> {phase_name} {vid.upper()}: BLAD po {time.time()-t0:.0f}s "
                  f"- przechodze do nastepnego wariantu")
            traceback.print_exc()
            if stop_on_error:
                print(">>> --stop-on-error: przerywam.")
                return False
    return True


def sanity_check(protocol: str, ext_run: str):
    """Porownanie z liczbami artykulu.

    Headline artykulu (85.74% / 2.04%) to mAP w protokole Q81 liczone na
    SPLICIE VALID (200 obrazow) - wiec wlasciwy sanity-check w trybie q81
    porownuje split_valid; FULL podajemy jako uczciwa liczbe pelnozbiorowa.
    """
    for vid, headline, label in (("v1", V1_HEADLINE_MAP05, "curated ep270"),
                                 ("v0", V0_HEADLINE_MAP05, "uncurated ep170")):
        r = _read_results(vid, ext_run)
        if not r:
            continue
        full = r.get("FULL", {}).get("mAP_0.5")
        valid = r.get("split_valid", {}).get("mAP_0.5")
        if protocol == "q81" and valid is not None:
            diff = valid * 100 - headline
            verdict = ("ZGODNE - protokol artykulu odtworzony"
                       if abs(diff) <= 1.5 else "ROZBIEZNOSC > 1.5pp - wyjasnic")
            print(f"\n[SANITY {vid.upper()} {label}] Q81 valid mAP@0.5 = "
                  f"{valid*100:.2f}% vs headline {headline}% ({diff:+.2f}pp) -> {verdict}")
            if full is not None:
                print(f"[SANITY {vid.upper()}] Q81 FULL (2333 obrazy) = "
                      f"{full*100:.2f}% <- uczciwa liczba pelnozbiorowa")
        elif full is not None:
            print(f"\n[SANITY {vid.upper()} {label}] FULL mAP@0.5 = {full*100:.2f}% "
                  f"(headline {headline}% liczono protokolem Q81 na valid - "
                  f"porownanie wprost niemiarodajne; uruchom --protocol q81)")
    # Curation gain na pelnym zbiorze (gdy sa oba warianty)
    r1 = _read_results("v1", ext_run).get("FULL", {}).get("mAP_0.5")
    r0 = _read_results("v0", ext_run).get("FULL", {}).get("mAP_0.5")
    if r1 and r0 and r0 > 0:
        print(f"[CURATION GAIN, FULL 2333, {protocol}] V1/V0 = "
              f"{r1*100:.2f}% / {r0*100:.2f}% = {r1/r0:.1f}x")


def main():
    ap = argparse.ArgumentParser(description="Pelne ewaluacje ablacji V0-V7 po kolei")
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=VARIANT_CHOICES,
                    help="ktore warianty (domyslnie v1-v7; v0=uncurated tylko external)")
    ap.add_argument("--mode", choices=["both", "external", "samedist"], default="both")
    ap.add_argument("--protocol", choices=["all", "q81"], default="all",
                    help="all = 100 queries; q81 = protokol artykulu (slot 81, prog 0.3)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    from functools import partial
    from ablation_eval_lib import run_external, run_samedist

    ext_run = "external" if args.protocol == "all" else f"external_{args.protocol}"
    sd_run = "samedist" if args.protocol == "all" else f"samedist_{args.protocol}"

    print(f"=== PELNE EWALUACJE (sekwencyjnie): {' -> '.join(args.variants)} "
          f"| mode={args.mode} | protocol={args.protocol} | batch={args.batch_size} ===")

    ext_statuses, sd_statuses = {}, {}
    t_start = time.time()
    aborted = False

    if args.mode in ("both", "external"):
        print(f"\n{'#'*70}\n#  FAZA 1: EXTERNAL BENCHMARK (cross-dataset, "
              f"protokol {args.protocol.upper()})\n{'#'*70}")
        aborted = not run_phase("EXTERNAL", args.variants, run_external,
                                partial(external_verdict, run_name=ext_run),
                                ext_statuses,
                                args.stop_on_error,
                                batch_size=args.batch_size,
                                num_workers=args.workers,
                                protocol=args.protocol)

    if args.mode in ("both", "samedist") and not aborted:
        sd_variants = [v for v in args.variants if v != "v0"]  # v0 nie ma anotacji ablacyjnych
        print(f"\n{'#'*70}\n#  FAZA 2: SAME-DISTRIBUTION (val+test, "
              f"protokol {args.protocol.upper()})\n{'#'*70}")
        run_phase("SAMEDIST", sd_variants, run_samedist,
                  partial(samedist_verdict, run_name=sd_run),
                  sd_statuses,
                  args.stop_on_error,
                  batch_size=args.batch_size,
                  num_workers=args.workers,
                  protocol=args.protocol)

    # --- Podsumowanie ---
    hours = (time.time() - t_start) / 3600
    print(f"\n\n{'='*70}")
    print(f"  PODSUMOWANIE PELNYCH EWALUACJI ({hours:.1f} h lacznie)")
    print(f"{'='*70}")
    failed = []
    for vid in args.variants:
        ext = ext_statuses.get(vid, "-")
        sd = sd_statuses.get(vid, "-")
        print(f"  {vid.upper():4} | external: {ext}")
        print(f"       | samedist: {sd}")
        for status in (ext, sd):
            if status not in ("-",) and not status.startswith("OK"):
                failed.append(vid)
                break

    sanity_check(args.protocol, ext_run)

    # Tabela zbiorcza do artykulu (z tego, co policzone)
    try:
        import aggregate_results
        argv_backup = sys.argv
        sys.argv = ["aggregate_results.py"]
        try:
            aggregate_results.main()
        finally:
            sys.argv = argv_backup
    except Exception as e:
        print(f"[WARN] Agregacja nie powiodla sie ({e}) - uruchom recznie: "
              f"py -3.11 common\\aggregate_results.py")

    if failed:
        print(f"\nWarianty do wyjasnienia: {', '.join(sorted(set(failed)))}")
        sys.exit(1)
    print("\nKomplet ewaluacji zakonczony poprawnie.")


if __name__ == "__main__":
    main()
