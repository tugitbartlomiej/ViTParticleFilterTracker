"""
Per-query sweep DETR na EXTERNAL benchmarku (Cataract Surgery Instruments, FULL 2333).

Cel: ustalic, czy slot 81 (Query 81) jest NAJLEPSZYM pojedynczym query na external.

Metoda (dokladnie jak w narracji zadania):
  1. JEDEN forward pass na obraz (FULL 2333, batch=4, RTX 3070).
  2. Dla KAZDEGO slotu q=0..99 budujemy predykcje uzywajac WYLACZNIE slotu q,
     tym samym postprocessingiem co protokol Q81
     (softmax(logits)[:,q,0] = P(klasa instrument); ramka z pred_boxes[:,q]
     znormalizowana cxcywh -> xywh * oryginalny rozmiar obrazu).
  3. Metryki per slot na FULL 2333:
       - mAP@0.5 THRESHOLDED  (predykcja tylko gdy score>=0.3) = DOKLADNIE protokol
         artykulu (run_inference_q81) -> slot 81 powinien dac 26.04%
       - mAP@0.5 UNTHRESHOLDED (jedna predykcja/obraz, COCOeval po score) -> slot 81
         powinien dac 31.90% (zgodnie z wczesniejsza analiza protocol_analysis)
       - TP/FP/FN + Precision/Recall/F1 przy conf=0.3, IoU=0.5
  4. Ranking wszystkich 100 slotow; pozycja slotu 81, top-N, margines.

Checkpoint: V1 = curated 20k ep270 (checkpoint_epoch_270.pth) - model produkcyjny
(headline 85.74% valid / 26.04% FULL). Wybor wariantu: --variant (domyslnie v1).

Read-only na checkpointach i danych. Nie modyfikuje oryginalnych skryptow.

Uruchomienie:
  py -3.11 query_sweep_external.py --variant v1
  py -3.11 query_sweep_external.py --variant v0     # uncurated ep170 (opcjonalnie)
"""

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "common"))

import paths_config as cfg
from ablation_eval_lib import (
    load_model, load_processor, merge_external, gt_category_id,
    CocoInferenceDataset, filter_coco, detection_counts, _ensure_ann_fields,
    _collate_pad,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NUM_QUERIES = 100
CONF = cfg.CONF_THRESHOLD        # 0.3
IOU = cfg.IOU_THRESHOLD          # 0.5 (dla TP/FP/FN)
Q81 = cfg.DETR_QUERY_ID          # 81

OUT_DIR = Path(r"F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\Uwagi\v2_Ablacje\Dane\query_sweep_external")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1", choices=list(cfg.CHECKPOINTS.keys()))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    workdir = OUT_DIR / f"_work_{args.variant}"
    workdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("[WARN] Brak CUDA - bedzie wolno.")

    # --- Model + dane ---
    model, model_info = load_model(args.variant, device)
    processor = load_processor()

    full_gt, dir_by_img_id = merge_external()
    for a in full_gt["annotations"]:
        _ensure_ann_fields(a)
    cat_id = gt_category_id(full_gt)
    ds = CocoInferenceDataset(full_gt, None, processor, dir_by_img_id=dir_by_img_id)
    if ds.missing:
        full_gt = filter_coco(full_gt, ds.kept_image_ids(),
                              full_gt["info"]["description"] + " (istniejace pliki)")
    n_img = len(full_gt["images"])
    n_ann = len(full_gt["annotations"])
    print(f"FULL: {n_img} obrazow / {n_ann} anotacji, category_id={cat_id}")

    # --- JEDEN forward pass na obraz; zbieramy score+box dla WSZYSTKICH 100 slotow ---
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=_collate_pad,
                        num_workers=args.workers, shuffle=False)
    # per_slot_preds[q] = lista predykcji (po jednej na obraz, score NIEfiltrowany)
    per_slot_preds = [[] for _ in range(NUM_QUERIES)]
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            pv = batch["pixel_values"].to(device)
            pm = batch["pixel_mask"].to(device)
            outputs = model(pixel_values=pv, pixel_mask=pm)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # [B,Q,C+1]
            scores = probs[:, :, 0].cpu().numpy()                        # [B,Q] P(klasa 0)
            boxes = outputs.pred_boxes.cpu().numpy()                     # [B,Q,4] cxcywh norm
            for i, img_id in enumerate(batch["image_ids"]):
                n_done += 1
                h, w = batch["orig_sizes"][i]
                for q in range(NUM_QUERIES):
                    cx, cy, bw, bh = (float(v) for v in boxes[i, q])
                    per_slot_preds[q].append({
                        "image_id": int(img_id),
                        "category_id": int(cat_id),
                        "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                        "score": float(scores[i, q]),
                    })
            if (bi + 1) % 50 == 0:
                rate = n_done / max(1e-6, time.time() - t0)
                print(f"  ... {n_done}/{n_img} obrazow ({rate:.1f} img/s)")
    print(f"Forward pass: {n_img} obrazow w {time.time()-t0:.0f}s")

    # --- COCO GT zbudowane RAZ (wydajnie), reuse dla wszystkich 200 ewaluacji ---
    gt_path = workdir / "gt_FULL.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(full_gt, f)
    coco_gt = COCO(str(gt_path))
    gt_img_ids = set(coco_gt.getImgIds())

    def eval_map50(preds):
        preds_f = [p for p in preds if p["image_id"] in gt_img_ids]
        if not preds_f:
            return 0.0
        with redirect_stdout(io.StringIO()):
            dt = coco_gt.loadRes(preds_f)
            ev = COCOeval(coco_gt, dt, iouType="bbox")
            ev.evaluate(); ev.accumulate(); ev.summarize()
        return float(ev.stats[1])  # mAP@0.5

    # --- Sweep 100 slotow ---
    print("\nLiczenie metryk per slot (100 slotow x [unthr + thr] mAP + TP/FP/FN)...")
    rows = []
    ts = time.time()
    for q in range(NUM_QUERIES):
        preds_all = per_slot_preds[q]                                 # 1 pred/obraz
        preds_thr = [p for p in preds_all if p["score"] >= CONF]      # protokol Q81
        map_unthr = eval_map50(preds_all)
        map_thr = eval_map50(preds_thr)
        dc = detection_counts(preds_all, full_gt, conf=CONF, iou_thr=IOU)
        rows.append({
            "slot": q,
            "mAP50_thresholded": round(map_thr * 100, 4),
            "mAP50_unthresholded": round(map_unthr * 100, 4),
            "TP": dc["TP"], "FP": dc["FP"], "FN": dc["FN"],
            "Precision": round(dc["Precision"] * 100, 2),
            "Recall": round(dc["Recall"] * 100, 2),
            "F1": round(dc["F1"] * 100, 2),
            "n_pred_above_conf": len(preds_thr),
        })
        if (q + 1) % 20 == 0:
            print(f"  ... {q+1}/100 ({(q+1)/max(1e-6,time.time()-ts):.1f} slot/s)")
    print(f"Sweep: {time.time()-ts:.0f}s")

    # --- Rankingi ---
    by_thr = sorted(rows, key=lambda r: -r["mAP50_thresholded"])
    by_unthr = sorted(rows, key=lambda r: -r["mAP50_unthresholded"])
    rank_thr = {r["slot"]: i for i, r in enumerate(by_thr, 1)}
    rank_unthr = {r["slot"]: i for i, r in enumerate(by_unthr, 1)}
    for r in rows:
        r["rank_thresholded"] = rank_thr[r["slot"]]
        r["rank_unthresholded"] = rank_unthr[r["slot"]]

    q81 = next(r for r in rows if r["slot"] == Q81)
    best_thr = by_thr[0]
    second_thr = by_thr[1]
    best_unthr = by_unthr[0]
    second_unthr = by_unthr[1]

    verdict_thr = (q81["slot"] == best_thr["slot"])
    verdict_unthr = (q81["slot"] == best_unthr["slot"])

    summary = {
        "variant": args.variant,
        "label": cfg.VARIANT_LABELS[args.variant],
        "checkpoint": model_info["checkpoint"],
        "epoch": model_info["epoch"],
        "num_queries": model_info["num_queries"],
        "dataset": "External Cataract Surgery Instruments FULL (train+valid+test merged)",
        "dataset_path": str(cfg.EXTERNAL_DIR),
        "n_images": n_img, "n_annotations": n_ann,
        "protocol": {"conf_threshold": CONF, "iou_threshold": IOU,
                     "map_engine": "pycocotools COCOeval bbox",
                     "score_def": "softmax(logits)[slot,0] = P(instrument class)"},
        "Q81": q81,
        "verdict_Q81_best_thresholded": verdict_thr,
        "verdict_Q81_best_unthresholded": verdict_unthr,
        "best_thresholded": best_thr, "second_thresholded": second_thr,
        "margin_Q81_vs_second_thresholded_pp":
            round(q81["mAP50_thresholded"] - second_thr["mAP50_thresholded"], 4)
            if verdict_thr else None,
        "best_unthresholded": best_unthr, "second_unthresholded": second_unthr,
        "margin_Q81_vs_second_unthresholded_pp":
            round(q81["mAP50_unthresholded"] - second_unthr["mAP50_unthresholded"], 4)
            if verdict_unthr else None,
        "top10_thresholded": [{"slot": r["slot"], "mAP50": r["mAP50_thresholded"],
                               "TP": r["TP"], "FP": r["FP"], "FN": r["FN"]}
                              for r in by_thr[:10]],
        "top10_unthresholded": [{"slot": r["slot"], "mAP50": r["mAP50_unthresholded"]}
                                for r in by_unthr[:10]],
        "all_slots": rows,
    }

    tag = f"{args.variant}_ep{model_info['epoch']}"
    json_path = OUT_DIR / f"query_sweep_external_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = OUT_DIR / f"query_sweep_external_{tag}.csv"
    cols = ["slot", "rank_thresholded", "mAP50_thresholded", "rank_unthresholded",
            "mAP50_unthresholded", "TP", "FP", "FN", "Precision", "Recall", "F1",
            "n_pred_above_conf"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: x["rank_thresholded"]):
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # --- Konsola ---
    print("\n" + "=" * 72)
    print(f"WERDYKT ({args.variant} = {cfg.VARIANT_LABELS[args.variant]}, ep{model_info['epoch']}, external FULL {n_img})")
    print("=" * 72)
    print(f"Q81 mAP@0.5 THRESHOLDED (conf 0.3, protokol artykulu) = "
          f"{q81['mAP50_thresholded']:.2f}%  (sanity vs 26.04%)")
    print(f"Q81 mAP@0.5 UNTHRESHOLDED                             = "
          f"{q81['mAP50_unthresholded']:.2f}%  (sanity vs 31.90%)")
    print(f"Q81 TP/FP/FN @0.3 = {q81['TP']}/{q81['FP']}/{q81['FN']}  "
          f"(sanity TP vs 835), F1={q81['F1']:.1f}%")
    print(f"Q81 ranga THRESHOLDED   : #{q81['rank_thresholded']}/100")
    print(f"Q81 ranga UNTHRESHOLDED : #{q81['rank_unthresholded']}/100")
    print(f"\nQ81 NAJLEPSZY (thresholded)?   {'TAK' if verdict_thr else 'NIE'}")
    print(f"Q81 NAJLEPSZY (unthresholded)? {'TAK' if verdict_unthr else 'NIE'}")
    if verdict_thr:
        print(f"  margines nad #2 (slot {second_thr['slot']}): "
              f"{summary['margin_Q81_vs_second_thresholded_pp']:.2f}pp "
              f"({q81['mAP50_thresholded']:.2f}% vs {second_thr['mAP50_thresholded']:.2f}%)")
    else:
        print(f"  NAJLEPSZY = slot {best_thr['slot']} = {best_thr['mAP50_thresholded']:.2f}% "
              f"(Q81 = {q81['mAP50_thresholded']:.2f}%)")
    print("\nTOP-10 (thresholded, conf 0.3):")
    for i, r in enumerate(by_thr[:10], 1):
        mk = "  <== Q81" if r["slot"] == Q81 else ""
        print(f"  #{i:2d}  slot {r['slot']:3d}  mAP@0.5={r['mAP50_thresholded']:6.2f}%  "
              f"TP={r['TP']:4d} FP={r['FP']:4d} FN={r['FN']:4d}{mk}")
    print("\nTOP-10 (unthresholded):")
    for i, r in enumerate(by_unthr[:10], 1):
        mk = "  <== Q81" if r["slot"] == Q81 else ""
        print(f"  #{i:2d}  slot {r['slot']:3d}  mAP@0.5={r['mAP50_unthresholded']:6.2f}%{mk}")

    print(f"\nZapisano:\n  {json_path}\n  {csv_path}")
    return summary


if __name__ == "__main__":
    main()
