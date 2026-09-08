"""
Niezalezna weryfikacja YOLO 20k ep270 na pelnym external benchmarku.

Liczba 34.17% mAP@0.5 (FULL) pochodzi z zarchiwizowanych predykcji benchmarku
CLEAN_20260209. Ten skrypt robi SWIEZA inferencje lokalnie (ultralytics),
replikujac protokol oryginalnego benchmarku 1:1:
  model(path, conf=0.3, max_det=300, verbose=False) -> xyxy + conf,
i liczy COCO mAP na scalonym zbiorze (2333 obrazy / 2346 anotacji) + sanity
na partycji valid (oczekiwane ~78.74%).

Uruchomienie:
  py -3.11 verify_yolo_fullset.py
  py -3.11 verify_yolo_fullset.py --checkpoint <sciezka.pt>
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_config as cfg
from ablation_eval_lib import (merge_external, filter_coco, gt_category_id,
                               coco_map, detection_counts)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

YOLO_EP270 = (r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints"
              r"\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart"
              r"\final_ep170_lr0001_weights\epoch270.pt")
CONF = 0.3        # CONF_THRESHOLD_STANDARD z benchmarku
MAX_DET = 300     # YOLO_MAX_DETECTIONS z benchmarku


def main():
    ap = argparse.ArgumentParser(description="Swieza weryfikacja YOLO ep270 na FULL")
    ap.add_argument("--checkpoint", default=YOLO_EP270)
    ap.add_argument("--conf", type=float, default=CONF,
                    help="prog pewnosci (0.3 = protokol benchmarku; 0.001 = standard mAP)")
    args = ap.parse_args()
    conf = args.conf

    from ultralytics import YOLO

    out_dir = cfg.RESULTS_DIR / f"verify_yolo_conf{conf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, dir_by_img_id = merge_external()
    cat_id = gt_category_id(merged)

    print(f"Ladowanie YOLO: {args.checkpoint}")
    model = YOLO(args.checkpoint)

    preds = []
    t0 = time.time()
    images = sorted(merged["images"], key=lambda i: i["id"])
    for n, img in enumerate(images, 1):
        fp = dir_by_img_id[img["id"]] / img["file_name"]
        res = model(str(fp), conf=conf, max_det=MAX_DET, verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        if boxes is not None and boxes.shape[0] > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(xyxy, confs):
                preds.append({"image_id": int(img["id"]),
                              "category_id": int(cat_id),
                              "bbox": [float(x1), float(y1),
                                       float(x2 - x1), float(y2 - y1)],
                              "score": float(score)})
        if n % 500 == 0:
            print(f"  ... {n}/{len(images)} ({n/(time.time()-t0):.1f} img/s)")
    print(f"Inferencja: {len(images)} obrazow w {time.time()-t0:.0f}s, "
          f"{len(preds)} predykcji")

    stem = Path(args.checkpoint).stem  # np. epoch270 / epoch70
    with open(out_dir / f"predictions_yolo_{stem}.json", "w", encoding="utf-8") as f:
        json.dump(preds, f)

    full_map = coco_map(preds, merged, out_dir, f"yolo_{stem}_FULL")
    full_det = detection_counts(preds, merged)
    valid_ids = {i["id"] for i in merged["images"] if i.get("source_split") == "valid"}
    valid_gt = filter_coco(merged, valid_ids, "valid sanity")
    val_map = coco_map(preds, valid_gt, out_dir, f"yolo_{stem}_valid")

    print(f"\n=== WYNIK SWIEZEJ INFERENCJI YOLO {stem} (conf={conf}) ===")
    print(f"FULL  mAP@0.5 = {full_map['mAP_0.5']*100:.2f}%")
    print(f"FULL  mAP@[.5:.95] = {full_map['mAP_0.5:0.95']*100:.2f}%")
    print(f"FULL  TP={full_det['TP']} FP={full_det['FP']} FN={full_det['FN']} | "
          f"P={full_det['Precision']*100:.1f}% R={full_det['Recall']*100:.1f}% "
          f"F1={full_det['F1']*100:.1f}%")
    print(f"valid mAP@0.5 = {val_map['mAP_0.5']*100:.2f}%")

    with open(out_dir / f"verify_summary_{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "conf": conf, "max_det": MAX_DET,
                   "FULL": {**full_map, **full_det},
                   "valid_mAP50": val_map["mAP_0.5"]}, f, indent=2)


if __name__ == "__main__":
    main()
