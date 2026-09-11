"""EL2N z modelu wstepnego (proxy): inferencja po CALEJ puli train.

AGENT_1 zadanie 3. Wyjscie: feature_cache/el2n_<seed>.pkl w tym samym
formacie co pozostale cechy (features (N,1) float32 + frame_ids + meta).

Definicja wyniku (zapisana tez w meta):
  score(frame) = sredni blad 1-confidence po jednokrotnym dopasowaniu
  predykcji proxy do GT przy IoU >= evaluation.iou_threshold. Brak dopasowania
  daje 1.0. Dla zweryfikowanej klatki negatywnej score jest najwyzsza pewnoscia
  falszywego alarmu (0.0, gdy predykcji brak).
  Na koniec normalizacja min-max do [0, 1] po calej puli:
      el2n = (score - min(score)) / (max(score) - min(score));
  parametry normalizacji (min, max, metoda) sa w meta pliku .pkl.

--fake: FIXTURE do testow dymnych - deterministyczna funkcja pikseli
(udzial 'trudnych', nieregularnych obszarow), ten sam interfejs i format.

Uzycie:
    py -3.11 pipeline/features/el2n_from_proxy.py --seed 42 [--fake]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.features.common import (  # noqa: E402
    load_protocol, read_manifest, resolve_under_root, save_feature_pkl,
    train_pool, utc_now, write_cache_manifest)


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def minmax01(scores: np.ndarray) -> tuple[np.ndarray, dict]:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores), {"method": "minmax_0_1",
                                       "raw_min": lo, "raw_max": hi,
                                       "note": "stale scores -> zera"}
    return (scores - lo) / (hi - lo), {"method": "minmax_0_1",
                                       "raw_min": lo, "raw_max": hi}


def prediction_arrays(predictions) -> tuple[np.ndarray, np.ndarray]:
    """Normalize current ``sv.Detections`` and legacy iterable outputs."""
    if isinstance(predictions, (list, tuple)) and len(predictions) == 1 \
            and hasattr(predictions[0], "xyxy"):
        predictions = predictions[0]
    if hasattr(predictions, "xyxy"):
        boxes = np.asarray(predictions.xyxy, dtype=float).reshape(-1, 4)
        confidence = getattr(predictions, "confidence", None)
        confidence = (np.ones(len(boxes), dtype=float) if confidence is None
                      else np.asarray(confidence, dtype=float).reshape(-1))
        if len(confidence) != len(boxes):
            raise ValueError("RF-DETR prediction fields have inconsistent lengths")
        return boxes, confidence
    boxes, confidence = [], []
    for item in predictions:
        if isinstance(item, dict):
            box = item.get("xyxy")
            if box is None:
                xywh = item.get("bbox")
                if xywh is None:
                    raise ValueError("prediction dictionary has no xyxy/bbox")
                x, y, w, h = map(float, xywh)
                box = [x, y, x + w, y + h]
            boxes.append(list(map(float, box)))
            confidence.append(float(item.get("confidence", item.get("score", 1.0))))
        else:
            values = list(map(float, item))
            if len(values) < 4:
                raise ValueError("prediction row has fewer than four coordinates")
            boxes.append(values[:4])
            confidence.append(values[4] if len(values) > 4 else 1.0)
    return (np.asarray(boxes, dtype=float).reshape(-1, 4),
            np.asarray(confidence, dtype=float))


def detection_error(gt_xywh: list[list[float]], boxes_xyxy: np.ndarray,
                    confidence: np.ndarray, iou_threshold: float) -> float:
    """Frame difficulty with one-to-one greedy IoU matching."""
    if not gt_xywh:
        return float(confidence.max()) if len(confidence) else 0.0
    candidates = []
    for gt_idx, gt in enumerate(gt_xywh):
        for pred_idx, box in enumerate(boxes_xyxy):
            pred_xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            overlap = iou_xywh(pred_xywh, gt)
            if overlap >= iou_threshold:
                candidates.append((overlap, float(confidence[pred_idx]), gt_idx, pred_idx))
    candidates.sort(reverse=True)
    matched_gt, matched_pred, matched_conf = set(), set(), {}
    for _, conf, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matched_conf[gt_idx] = conf
    return float(np.mean([1.0 - matched_conf.get(i, 0.0)
                          for i in range(len(gt_xywh))]))


def score_real(protocol: dict, pool: list[dict], images_root: Path,
               proxy_path: Path, seed: int) -> np.ndarray:
    """PRODUKCJA: inferencja rfdetr-proxy, EL2N po dopasowanych GT."""
    from rfdetr import RFDETRSmall  # leniwie

    conf_thr = float(protocol["evaluation"]["map_conf_threshold"])
    iou_thr = float(protocol["evaluation"]["iou_threshold"])
    # Let RF-DETR reconstruct and validate its own checkpoint schema. Loading a
    # state_dict with strict=False could silently leave task heads uninitialized.
    det = RFDETRSmall(pretrain_weights=str(proxy_path))

    scores = np.zeros((len(pool),), dtype=np.float32)
    for i, r in enumerate(pool):
        if r.get("label_status") == "unknown":
            raise ValueError(f"{r['frame_id']}: EL2N cannot score an unknown label")
        if not r.get("path"):
            raise ValueError(f"{r['frame_id']}: materialize archive-backed image first")
        pred = det.predict(str(images_root / r["path"]), threshold=conf_thr)
        boxes, confidence = prediction_arrays(pred)
        row_annotations = r.get("annotations")
        if row_annotations is None:
            row_annotations = ([{"bbox": r["bbox"]}]
                               if r.get("bbox") and r.get("has_instrument") else [])
        gt = [annotation["bbox"] for annotation in row_annotations]
        scores[i] = detection_error(gt, boxes, confidence, iou_thr)
        if (i + 1) % 200 == 0:
            print(f"[el2n real] {i + 1}/{len(pool)}", flush=True)
    return scores


def score_fake(pool: list[dict], images_root: Path) -> np.ndarray:
    """FIXTURE: deterministyczna 'trudnosc' z pikseli (ten sam interfejs)."""
    from PIL import Image
    scores = np.zeros((len(pool),), dtype=np.float32)
    for i, r in enumerate(pool):
        gray = np.asarray(
            Image.open(images_root / r["path"]).convert("L"),
            dtype=np.float32) / 255.0
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        hard = float((gx > 0.15).mean() + (gy > 0.15).mean()) / 2.0
        scores[i] = hard * (1.0 + 0.01 * (i % 7))
        if (i + 1) % 200 == 0:
            print(f"[el2n fake] {i + 1}/{len(pool)}", flush=True)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--images-root", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--fake", action="store_true",
                    help="FIXTURE/test dymny: zastepcze scores z pikseli")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    paths_cfg = protocol.get("paths", {})
    manifest = args.manifest or resolve_under_root(
        paths_cfg.get("pool_manifest", "pool_manifest.jsonl"))
    images_root = args.images_root or resolve_under_root(
        paths_cfg.get("images_root", "."))
    cache_dir = Path(args.cache_dir or resolve_under_root(
        paths_cfg.get("feature_cache_dir", "feature_cache")))

    pool = train_pool(read_manifest(manifest))
    frame_ids = [r["frame_id"] for r in pool]
    proxy_path = cache_dir / f"el2n_proxy_{args.seed}.pth"

    if args.fake:
        raw = score_fake(pool, images_root)
        model = f"fake-proxy-{args.seed}"
    else:
        if not proxy_path.exists():
            sys.exit(f"Brak proxy {proxy_path}. Najpierw: "
                     f"train_el2n_proxy.py --seed {args.seed}")
        raw = score_real(protocol, pool, images_root, proxy_path, args.seed)
        model = protocol["model"]["name"]

    norm, norm_meta = minmax01(raw)
    meta = {
        "model": f"EL2N proxy ({model})", "version": utc_now(),
        "seed": args.seed, "proxy_file": proxy_path.name,
        "proxy_epochs": int(protocol["el2n"]["proxy_epochs"]),
        "score": protocol["el2n"].get("score", "mean_error_norm_vs_gt"),
        "normalization": norm_meta, "fake": bool(args.fake),
    }
    out = cache_dir / f"el2n_{args.seed}.pkl"
    save_feature_pkl(out, norm.reshape(-1, 1), frame_ids, meta)
    write_cache_manifest(cache_dir, {f"el2n_{args.seed}": {
        "file": out.name, "n_frames": len(frame_ids), "dim": 1,
        **meta}})
    print(f"Zapisano {out}: {len(frame_ids)} klatek, "
          f"normalizacja {norm_meta['method']} "
          f"[{norm_meta['raw_min']:.4f}, {norm_meta['raw_max']:.4f}] -> [0,1]")


if __name__ == "__main__":
    main()
