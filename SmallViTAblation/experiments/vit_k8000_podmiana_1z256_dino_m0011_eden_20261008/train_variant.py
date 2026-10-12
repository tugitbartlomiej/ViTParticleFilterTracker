from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import model as M


W, H = 480, 320
BATCH = 32
STEPS = 500
LR = 2e-5
WD = 0.05
SEED = 20261008
EVAL_EVERY = 25


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, path)


def iou(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x0 = np.maximum(boxes[:, 0], gt[0]); y0 = np.maximum(boxes[:, 1], gt[1])
    x1 = np.minimum(boxes[:, 2], gt[2]); y1 = np.minimum(boxes[:, 3], gt[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    area_g = max(0.0, float((gt[2] - gt[0]) * (gt[3] - gt[1])))
    return inter / (area_b + area_g - inter + 1e-9)


def load_rows(data: Path, ids: list[str], by_id: dict[str, dict]):
    images, boxes = [], []
    for image_id in ids:
        row = by_id[image_id]
        img = cv2.imread(str(data / row["file"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["file"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        sx, sy = W / row["width"], H / row["height"]
        b = np.asarray([[x * sx, y * sy, (x + w) * sx, (y + h) * sy] for x, y, w, h in row["gt_xywh"]], np.float32)
        images.append(img); boxes.append(b)
    return np.stack(images), boxes


def targets(boxes):
    h, w = H // M.KROK, W // M.KROK
    normal = M.cele(boxes, h, w)
    flipped = [np.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1) if len(b) else b for b in boxes]
    return normal, M.cele(flipped, h, w)


@torch.no_grad()
def detect(net, images: np.ndarray, batch=64):
    net.eval(); out = []
    for s in range(0, len(images), batch):
        x = torch.from_numpy(images[s:s + batch]).cuda().permute(0, 3, 1, 2).float() / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, _ = net(x)
        b, sc = M.dekoduj(o)
        out.extend(zip(b.float().cpu().numpy(), sc.float().cpu().numpy()))
    return out


def ap50(detections, gt_boxes):
    all_det, n_gt = [], 0
    for image_idx, ((boxes, scores), gt) in enumerate(zip(detections, gt_boxes)):
        n_gt += len(gt)
        for b, score in zip(boxes, scores):
            all_det.append((float(score), image_idx, b))
    all_det.sort(key=lambda x: -x[0])
    used = [np.zeros(len(g), bool) for g in gt_boxes]
    tp = np.zeros(len(all_det), np.float64)
    for j, (_, image_idx, box) in enumerate(all_det):
        gt = gt_boxes[image_idx]
        if len(gt):
            values = np.array([iou(np.asarray([box]), g)[0] for g in gt]); values[used[image_idx]] = -1
            k = int(values.argmax())
            if values[k] >= 0.5:
                tp[j] = 1; used[image_idx][k] = True
    ctp = np.cumsum(tp); rec = ctp / max(n_gt, 1); prec = ctp / np.arange(1, len(tp) + 1)
    for j in range(len(prec) - 2, -1, -1):
        prec[j] = max(prec[j], prec[j + 1])
    return float(np.mean([prec[k] if (k := np.searchsorted(rec, r, side="left")) < len(prec) else 0.0 for r in np.linspace(0, 1, 101)]))


@torch.no_grad()
def probe_metrics(net, probe_image: np.ndarray, gt: np.ndarray, step: int):
    boxes, scores = detect(net, probe_image[None], batch=1)[0]
    values = iou(boxes, gt)
    top = int(np.argmax(scores)); best = int(np.argmax(values))
    return {
        "step": step,
        "top_confidence": float(scores[top]),
        "top_confidence_iou": float(values[top]),
        "best_iou": float(values[best]),
        "best_iou_confidence": float(scores[best]),
        "best_confidence_times_iou": float(np.max(scores * values)),
        "top_box": boxes[top].tolist(),
        "best_iou_box": boxes[best].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, required=True, help="0=control, 1..8=intervention")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    pool = json.loads((args.data / "pool.json").read_text(encoding="utf-8"))
    records = json.loads((args.data / "records.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    variants = [{"name": "control", "slot_id": pool["control_slot"]}] + [
        {"name": f"intervention_{i:02d}", "slot_id": image_id} for i, image_id in enumerate(pool["interventions"], 1)
    ]
    variant = variants[args.variant]
    out = args.output / variant["name"]
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "FINAL.json"
    if final_path.exists():
        print(f"{variant['name']} already complete", flush=True); return

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ids = pool["common_255"] + [variant["slot_id"]]
    X, boxes = load_rows(args.data, ids, by_id)
    (target0, target1) = targets(boxes)
    guard_X, guard_boxes = load_rows(args.data, pool["guard_512"], by_id)
    probe_cfg = json.loads((args.data / "probe.json").read_text(encoding="utf-8"))
    probe = cv2.cvtColor(cv2.imread(str(args.data / probe_cfg["file"])), cv2.COLOR_BGR2RGB)
    probe = cv2.resize(probe, (W, H), interpolation=cv2.INTER_AREA)
    probe_gt = np.asarray(probe_cfg["gt_xyxy_480x320"][0], np.float32)

    net = M.MalyViT().cuda()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    params_decay = [p for p in net.parameters() if p.ndim >= 2]
    params_no_decay = [p for p in net.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": params_decay, "weight_decay": WD}, {"params": params_no_decay, "weight_decay": 0.0}], lr=LR)

    provenance = {
        "variant": variant,
        "array_index": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "pool_size": len(ids),
        "common_images": 255,
        "changed_slot": 255,
        "steps": args.steps,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WD,
        "seed": SEED,
        "optimizer_note": "fresh AdamW because K8000 is a state_dict without optimizer state",
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    atomic_json(out / "PROVENANCE.json", provenance)

    trajectory = [probe_metrics(net, probe, probe_gt, 0)]
    guard_before = ap50(detect(net, guard_X), guard_boxes)
    rng = np.random.default_rng(SEED)
    queue = []
    start = time.time()
    net.train()
    for step in range(1, args.steps + 1):
        if len(queue) < BATCH:
            queue += rng.permutation(len(X)).tolist()
        idx = np.asarray(queue[:BATCH]); queue = queue[BATCH:]
        flip_np = rng.random(BATCH) < 0.5
        x = torch.from_numpy(X[idx]).cuda(non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        flip = torch.from_numpy(flip_np).cuda()
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        aug = torch.from_numpy(rng.uniform(0.8, 1.2, (BATCH, 2)).astype(np.float32)).cuda()
        mean = x.mean((1, 2, 3), keepdim=True)
        x = ((x - mean) * aug[:, 0, None, None, None] + mean * aug[:, 1, None, None, None]).clamp(0, 1)
        ii = torch.from_numpy(idx)
        heat = torch.where(torch.from_numpy(flip_np)[:, None, None], target1[0][ii], target0[0][ii]).cuda()
        ltrb = torch.where(torch.from_numpy(flip_np)[:, None, None, None], target1[1][ii], target0[1][ii]).cuda()
        pos = torch.where(torch.from_numpy(flip_np)[:, None, None], target1[2][ii], target0[2][ii]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = net(x)
        loss, parts = M.strata(pred, heat, ltrb, pos)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % EVAL_EVERY == 0:
            trajectory.append(probe_metrics(net, probe, probe_gt, step))
            atomic_json(out / "STATUS.json", {"state": "training", "step": step, "loss": float(loss), "probe": trajectory[-1], "elapsed_seconds": time.time() - start})
            net.train()

    if trajectory[-1]["step"] != args.steps:
        trajectory.append(probe_metrics(net, probe, probe_gt, args.steps))

    guard_after = ap50(detect(net, guard_X), guard_boxes)
    checkpoint = out / f"checkpoint_step{args.steps}.pt"
    torch.save(net.state_dict(), checkpoint)
    result = {
        "state": "complete",
        "variant": variant,
        "pool_ids": ids,
        "probe_trajectory": trajectory,
        "probe_before": trajectory[0],
        "probe_after": trajectory[-1],
        "guard_ap50_before": guard_before,
        "guard_ap50_after": guard_after,
        "guard_ap50_delta": guard_after - guard_before,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "elapsed_seconds": time.time() - start,
    }
    atomic_json(out / "RESULT.json", result)
    atomic_json(final_path, {"state": "complete", "variant": variant["name"], "step": args.steps, "result": "RESULT.json"})
    print(json.dumps({"variant": variant["name"], "probe_after": trajectory[-1], "guard_delta": guard_after - guard_before}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
