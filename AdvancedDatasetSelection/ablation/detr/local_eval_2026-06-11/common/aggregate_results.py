"""
Agregator wynikow ablacji V1-V7 -> jedna tabela porownawcza (markdown).

Zbiera results/<vid>/{external,samedist}/metrics.json ze wszystkich wariantow
i sklada tabele w ukladzie docelowym dla artykulu (sekcja Ablation Study):
  wariant x {external FULL/CLEAN mAP@0.5, same-dist test mAP@0.5} + delty vs V1.

Mozna uruchamiac w dowolnym momencie - warianty bez wynikow dostaja "-".

Uruchomienie:
  py -3.11 aggregate_results.py [--output ../results/SUMMARY.md]
"""

import argparse
import io
import json
import sys
from pathlib import Path

import paths_config as cfg

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

VARIANTS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]


def load_metrics(variant_id: str, run_name: str) -> dict:
    path = cfg.RESULTS_DIR / variant_id / run_name / "metrics.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("results", {})


def pct(results: dict, tag: str, key: str = "mAP_0.5"):
    r = results.get(tag)
    return r[key] * 100 if r and key in r else None


def fmt(value, suffix: str = "%") -> str:
    return f"{value:.2f}{suffix}" if value is not None else "-"


def fmt_delta(value, base) -> str:
    if value is None or base is None:
        return "-"
    return f"{value - base:+.2f}pp"


def main():
    ap = argparse.ArgumentParser(description="Agregacja wynikow ablacji V1-V7")
    ap.add_argument("--output", type=str,
                    default=str(cfg.RESULTS_DIR / "SUMMARY.md"))
    args = ap.parse_args()

    rows = []
    for vid in VARIANTS:
        ext = load_metrics(vid, "external")
        sd = load_metrics(vid, "samedist")
        rows.append({
            "vid": vid,
            "label": cfg.VARIANT_LABELS[vid],
            "ext_full": pct(ext, "FULL"),
            "ext_clean": pct(ext, "CLEAN"),
            "ext_full_5095": pct(ext, "FULL", "mAP_0.5:0.95"),
            "sd_test": pct(sd, "test"),
            "sd_val": pct(sd, "val"),
        })

    base = rows[0]  # V1 = baseline
    lines = ["# Ablation Study - zbiorcze wyniki (lokalna ewaluacja)", "",
             "Protokol: external = scalone 2333 obrazy (FULL) / minus 48 leaked (CLEAN); "
             "same-dist = video-level split, identyczny dla wszystkich wariantow. "
             "Progi TP/FP: conf=0.3, IoU=0.5; mAP z pycocotools.", "",
             "| Wariant | Opis | Ext FULL mAP@0.5 | Ext CLEAN mAP@0.5 | "
             "Ext FULL mAP@[.5:.95] | Same-dist test mAP@0.5 | Same-dist val mAP@0.5 | "
             "d Ext FULL vs V1 | d SD test vs V1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['vid'].upper()} | {r['label']} | {fmt(r['ext_full'])} "
            f"| {fmt(r['ext_clean'])} | {fmt(r['ext_full_5095'])} "
            f"| {fmt(r['sd_test'])} | {fmt(r['sd_val'])} "
            f"| {fmt_delta(r['ext_full'], base['ext_full'])} "
            f"| {fmt_delta(r['sd_test'], base['sd_test'])} |")

    lines += ["", "Sanity-check: V1 Ext FULL powinno wynosic ~85.74% "
              "(headline artykulu, LEGACY benchmark 20260112). "
              "Istotna rozbieznosc = inny protokol, wyjasnic przed wpisaniem do tekstu.", ""]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nZapisano: {out_path}")


if __name__ == "__main__":
    main()
