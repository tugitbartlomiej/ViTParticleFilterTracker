#!/usr/bin/env python3
"""Analiza wrazliwosci NMS dla YOLO — Agent 4, zadanie 2.

Liczy wplyw progu IoU procedury NMS na precision/recall/F1 Z ZAPISANYCH
predykcji (conf 0.001), BEZ ponownej inferencji. NMS zaimplementowane czysto
w numpy, wiec analiza dziala bez zainstalowanego ultralytics.

Wyjście: tabela CSV + JSON (po jednym wierszu na prog IoU) oraz WYNIOSK
wypisany na stdout — jeśli NMS nie ma istotnego znaczenia, pokazujemy liczby,
nie pomijamy tematu.

Uzycie:
    # z zapisanych predykcji i GT:
    py -3.11 pipeline/transfer/nms_sensitivity.py \
        --predictions results/<run>/preds_val.json --gt results/<run>/gt_val.json \
        --protocol configs/protocol.yaml --out results/<run>/nms_sensitivity

    # demonstracja/fixture (syntetyczne nakladajace sie ramki), bez plikow:
    py -3.11 pipeline/transfer/nms_sensitivity.py --demo \
        --protocol configs/protocol.yaml --out /tmp/nms_demo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.transfer import transfer_common as tc  # noqa: E402


# --------------------------------------------------------------------------- #
# czysta implementacja NMS (numpy) — analiza progow, nie inferencja
# --------------------------------------------------------------------------- #
def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Zwraca indeksy zachowane po NMS. boxes w formacie x,y,w,h."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=int)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=int)


def apply_nms_per_image(preds_by_image, iou_thr, conf_floor):
    """NMS na kazdym obrazie; zwraca {img_id: (boxes, scores)} po stlumieniu."""
    out = {}
    for img_id, (boxes, scores) in preds_by_image.items():
        keep0 = scores >= conf_floor
        b, s = boxes[keep0], scores[keep0]
        keep = nms_numpy(b, s, iou_thr)
        out[img_id] = (b[keep], s[keep])
    return out


# --------------------------------------------------------------------------- #
# metryki P/R/F1 po dopasowaniu (wspolne z metrics.py)
# --------------------------------------------------------------------------- #
def prf_for(preds_by_image, gts_by_image, conf_thr, iou_match):
    from SmallViTAblation.pipeline.transfer import metrics as M
    tp, fp, fn = M.match_at_threshold(preds_by_image, gts_by_image, conf_thr, iou_match)
    prec, rec, f1 = M._prf(tp, fp, fn)
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6)}


# --------------------------------------------------------------------------- #
# rdzen analizy
# --------------------------------------------------------------------------- #
def run_sensitivity(gt_coco, preds, iso, sweep_iou, conf_operating):
    from SmallViTAblation.pipeline.transfer import metrics as M
    gts_arr, n_images, n_ann, _ = M.parse_gt(gt_coco)
    preds_raw = M.parse_preds(preds)
    conf_floor = iso["map_conf_threshold"]
    iou_match = iso["eval_iou_threshold"]
    rows = []
    for iou_thr in sweep_iou:
        suppressed = apply_nms_per_image(preds_raw, iou_thr, conf_floor)
        m = prf_for(suppressed, gts_arr, conf_operating, iou_match)
        n_boxes = int(sum(len(b) for b, _ in suppressed.values()))
        rows.append({"nms_iou_threshold": float(iou_thr),
                     "boxes_after_nms": n_boxes, **m})
    return rows, {"n_images": n_images, "n_annotations": n_ann,
                  "conf_operating": conf_operating, "iou_match": iou_match,
                  "conf_floor": conf_floor}


def write_outputs(rows, meta, conclusion, out_prefix):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    fields = ["nms_iou_threshold", "boxes_after_nms", "TP", "FP", "FN",
              "precision", "recall", "f1"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    tc.write_json(json_path, {"meta": meta, "rows": rows, "conclusion": conclusion})
    return csv_path, json_path


def make_conclusion(rows, eps=0.02):
    f1s = [r["f1"] for r in rows]
    precs = [r["precision"] for r in rows]
    recs = [r["recall"] for r in rows]
    spread_f1 = max(f1s) - min(f1s)
    istotne = spread_f1 > eps
    text = (
        f"Zakres F1 miedzy progami IoU NMS: {spread_f1:.4f} "
        f"(min {min(f1s):.4f}, max {max(f1s):.4f}); "
        f"precision [{min(precs):.4f}, {max(precs):.4f}], "
        f"recall [{min(recs):.4f}, {max(recs):.4f}]. "
    )
    if istotne:
        text += ("WNIOSEK: NMS MA istotny wplyw na wynik — prog IoU trzeba "
                 "odnotowac w tabeli i trzymac staly miedzy biegami.")
    else:
        text += (f"WNIOSEK: przy zadaniu z jedna klasa i zwykle jednym "
                 f"narzedziem w kadrze NMS NIE MA istotnego wplywu "
                 f"(zmiana F1 < {eps}). Pokazujemy liczby, nie pomijamy tematu.")
    return {"f1_spread": round(spread_f1, 6), "significant": istotne, "text": text}


# --------------------------------------------------------------------------- #
# fixture: syntetyczne predykcje COCO z nakladajacymi sie ramkami
# --------------------------------------------------------------------------- #
def synthetic_demo():
    """50 obrazow: GT = 1 ramka; predykcje = 1 mocna + kilka duplikatow o róznej
    ufności (nakladajacych sie), zeby NMS mial co tlumic. conf od 0.001."""
    rng = np.random.default_rng(42)
    images, annotations, preds = [], [], []
    ann_id = 1
    for img_id in range(1, 51):
        w, h = 512, 512
        images.append({"id": img_id, "file_name": f"img_{img_id:04d}.jpg",
                       "width": w, "height": h})
        gx, gy = float(rng.integers(80, 380)), float(rng.integers(80, 380))
        gw, gh = 40.0, 55.0
        annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                            "bbox": [gx, gy, gw, gh], "area": gw * gh, "iscrowd": 0})
        ann_id += 1
        # 1 prawdziwe trafienie (wysoka conf) + 4 duplikaty (niska/srednia conf)
        preds.append({"image_id": img_id, "category_id": 1,
                      "bbox": [gx + 1, gy + 1, gw, gh], "score": 0.92})
        for k in range(4):
            jx = gx + float(rng.integers(-6, 7))
            jy = gy + float(rng.integers(-6, 7))
            preds.append({"image_id": img_id, "category_id": 1,
                          "bbox": [jx, jy, gw, gh],
                          "score": float(rng.uniform(0.05, 0.6))})
        # 1 falszywe trafienie z dala od GT
        preds.append({"image_id": img_id, "category_id": 1,
                      "bbox": [10.0, 10.0, 30.0, 30.0], "score": 0.2})
    gt_coco = {"images": images, "annotations": annotations,
               "categories": [{"id": 1, "name": tc.CATEGORY_NAME}]}
    return gt_coco, preds


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Analiza wrazliwosci NMS dla YOLO.")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="plik COCO-results z surowymi predykcjami (conf 0.001)")
    ap.add_argument("--gt", type=Path, default=None, help="plik COCO ground-truth")
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="prefiks wyjsc (csv/json)")
    ap.add_argument("--iou-thresholds", type=str, default=None,
                    help="lista progow IoU po przecinku (nadpisuje protocol.transfer.nms.sweep_iou)")
    ap.add_argument("--conf-threshold", type=float, default=None,
                    help="operacyjny prog ufności do P/R/F1 (domyslnie: wybrany na val)")
    ap.add_argument("--demo", action="store_true",
                    help="uzyj wbudowanych syntetycznych predykcji (fixture, bez plikow)")
    args = ap.parse_args()

    protocol = tc.load_protocol(args.protocol)
    iso = tc.resolve_iso_resource(protocol)

    if args.iou_thresholds:
        sweep = [float(x) for x in args.iou_thresholds.split(",") if x.strip()]
    else:
        sweep = iso["nms_sweep_iou"]

    if args.demo:
        gt_coco, preds = synthetic_demo()
    else:
        if args.predictions is None or args.gt is None:
            ap.error("bez --demo wymagane sa --predictions i --gt")
        gt_coco = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    # prog operacyjny: podany albo wybrany na walidacji (maksymalizuje F1)
    if args.conf_threshold is not None:
        conf_op = args.conf_threshold
    else:
        from SmallViTAblation.pipeline.transfer import metrics as M
        gts_arr = M.parse_gt(gt_coco)[0]
        preds_arr = M.parse_preds(preds)
        conf_op, _ = M.select_operating_threshold(
            preds_arr, gts_arr, iso["eval_iou_threshold"])

    rows, meta = run_sensitivity(gt_coco, preds, iso, sweep, conf_op)
    conclusion = make_conclusion(rows)
    csv_path, json_path = write_outputs(rows, meta, conclusion, args.out)

    print(f"Analiza NMS: {len(rows)} progow IoU, conf_operating={conf_op}")
    print(f"{'IoU_NMS':>8} {'boxes':>7} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'prec':>7} {'rec':>7} {'F1':>7}")
    for r in rows:
        print(f"{r['nms_iou_threshold']:>8.2f} {r['boxes_after_nms']:>7d} "
              f"{r['TP']:>4d} {r['FP']:>4d} {r['FN']:>4d} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    print("-" * 60)
    print(conclusion["text"])
    print(f"Zapisano: {csv_path} , {json_path}")


if __name__ == "__main__":
    main()
