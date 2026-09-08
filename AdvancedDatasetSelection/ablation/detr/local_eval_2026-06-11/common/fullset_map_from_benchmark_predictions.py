"""
Pelnozbiorowe mAP@0.5 z ZAPISANYCH predykcji historycznych benchmarkow (bez GPU).

Benchmarki LEGACY (20260112, DETR_20k) i CLEAN (20260209, YOLO_20k) zapisaly
predykcje per split (train/valid/test) z progiem conf=0.3 (DETR: protokol Q81).
Ten skrypt scala je do JEDNEGO niedzielonego zbioru (ten sam id-remap co
merge_external w ablation_eval_lib) i liczy COCO mAP na pelnych 2333 obrazach
/ 2346 anotacjach + sanity na partycji valid (musi odtworzyc 85.74 / 78.74).

Cel: liczby do rewizji v2 artykulu (decyzja usera "a": headline pelnozbiorowy).

Uruchomienie:
  py -3.11 fullset_map_from_benchmark_predictions.py
  py -3.11 fullset_map_from_benchmark_predictions.py --models DETR_20k_ep270 YOLO_20k_ep270
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_config as cfg
from ablation_eval_lib import (merge_external, filter_coco, gt_category_id,
                               coco_map, detection_counts)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BENCHMARKS = {
    "LEGACY_20260112": Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                            r"\YOLO_DETR_Benchmarks\Benchmarks"
                            r"\BENCHMARK_ALL_EPOCHS_20260112_194440"),
    "CLEAN_20260209": Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                           r"\YOLO_DETR_Benchmarks\Benchmarks"
                           r"\BENCHMARK_ALL_EPOCHS_20260209_054404"),
}


def build_id_map():
    """(split, oryginalne_id) -> id w zbiorze scalonym.

    MUSI replikowac iteracje merge_external: splity w kolejnosci
    cfg.EXTERNAL_SPLITS, obrazy sortowane po id, numeracja od 1.
    """
    id_map = {}
    next_id = 1
    for split in cfg.EXTERNAL_SPLITS:
        with open(cfg.EXTERNAL_DIR / split / "_annotations.coco.json",
                  encoding="utf-8") as f:
            coco = json.load(f)
        for img in sorted(coco["images"], key=lambda i: i["id"]):
            id_map[(split, int(img["id"]))] = next_id
            next_id += 1
    return id_map


def discover_models(bench_dir: Path):
    names = set()
    for f in bench_dir.glob("*_predictions.json"):
        m = re.match(r"(.+)_(train|valid|test)_predictions\.json$", f.name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def load_merged_predictions(bench_dir: Path, model: str, id_map, cat_id: int):
    """Predykcje modelu ze wszystkich 3 splitow, przemapowane na scalone id.
    Zwraca None, gdy brakuje ktoregokolwiek pliku splitu."""
    preds = []
    for split in cfg.EXTERNAL_SPLITS:
        fp = bench_dir / f"{model}_{split}_predictions.json"
        if not fp.exists():
            return None
        with open(fp, encoding="utf-8") as f:
            split_preds = json.load(f)
        for p in split_preds:
            key = (split, int(p["image_id"]))
            if key not in id_map:
                continue  # predykcja dla obrazu spoza aktualnego GT
            preds.append({"image_id": id_map[key], "category_id": cat_id,
                          "bbox": p["bbox"], "score": float(p["score"])})
    return preds


def main():
    ap = argparse.ArgumentParser(description="Pelnozbiorowe mAP z zapisanych predykcji")
    ap.add_argument("--models", nargs="*", default=None,
                    help="konkretne modele (np. DETR_20k_ep270); domyslnie wszystkie")
    args = ap.parse_args()

    out_dir = cfg.RESULTS_DIR / "benchmark_fullset"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, _ = merge_external()
    cat_id = gt_category_id(merged)
    id_map = build_id_map()
    valid_ids = {i["id"] for i in merged["images"] if i.get("source_split") == "valid"}
    valid_gt = filter_coco(merged, valid_ids, "partycja valid (sanity)")

    rows = []
    for bench_name, bench_dir in BENCHMARKS.items():
        if not bench_dir.exists():
            print(f"[WARN] brak katalogu {bench_dir}")
            continue
        for model in discover_models(bench_dir):
            if args.models and model not in args.models:
                continue
            preds = load_merged_predictions(bench_dir, model, id_map, cat_id)
            if preds is None:
                print(f"[skip] {model} ({bench_name}): niekompletne pliki splitow")
                continue
            full_map = coco_map(preds, merged, out_dir, f"{model}_FULL")
            full_det = detection_counts(preds, merged)
            val_map = coco_map(preds, valid_gt, out_dir, f"{model}_valid")
            rows.append({"benchmark": bench_name, "model": model,
                         "n_preds": len(preds),
                         "FULL_mAP50": full_map["mAP_0.5"],
                         "FULL_mAP5095": full_map["mAP_0.5:0.95"],
                         "TP": full_det["TP"], "FP": full_det["FP"],
                         "FN": full_det["FN"],
                         "Precision": full_det["Precision"],
                         "Recall": full_det["Recall"], "F1": full_det["F1"],
                         "valid_mAP50": val_map["mAP_0.5"]})
            print(f"[{bench_name}] {model}: FULL mAP@0.5={full_map['mAP_0.5']*100:.2f}% "
                  f"(mAP@[.5:.95]={full_map['mAP_0.5:0.95']*100:.2f}%, TP={full_det['TP']}, "
                  f"F1={full_det['F1']*100:.1f}%) | sanity valid={val_map['mAP_0.5']*100:.2f}%")

    # zapis
    with open(out_dir / "fullset_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = ["# Pelnozbiorowe mAP z zapisanych predykcji benchmarkow", "",
             "Scalenie train+valid+test (2333 obrazy / 2346 anotacji); protokol jak w",
             "benchmarkach zrodlowych (DETR: Q81, conf 0.3). Kolumna 'valid' = sanity",
             "(musi odtwarzac liczby raportow zrodlowych: DETR ep270=85.74, YOLO ep270=78.74).", "",
             "| Benchmark | Model | FULL mAP@.5 | FULL mAP@.5:.95 | TP | FP | FN | Prec. | Recall | F1 | valid mAP@.5 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['benchmark']} | {r['model']} | {r['FULL_mAP50']*100:.2f}% "
                     f"| {r['FULL_mAP5095']*100:.2f}% | {r['TP']} | {r['FP']} | {r['FN']} "
                     f"| {r['Precision']*100:.1f}% | {r['Recall']*100:.1f}% "
                     f"| {r['F1']*100:.1f}% | {r['valid_mAP50']*100:.2f}% |")
    with open(out_dir / "FULLSET_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nZapisano: {out_dir / 'FULLSET_SUMMARY.md'} ({len(rows)} modeli)")


if __name__ == "__main__":
    main()
