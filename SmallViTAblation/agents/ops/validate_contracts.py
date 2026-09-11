#!/usr/bin/env python3
"""Walidacja plikow kontraktowych (agents/README.md, sekcja 4).

Uzycie (z katalogu SmallViTAblation):
    py -3.11 agents/ops/validate_contracts.py --root .
    py -3.11 agents/ops/validate_contracts.py --root . --only variants
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

OK, FAIL = "[OK]  ", "[FAIL]"

METRIC_FIELDS = {
    "mAP_0.5", "mAP_0.5:0.95", "mAP_0.75", "AR@100",
    "TP", "FP", "FN", "precision", "recall", "f1",
    "n_images", "n_annotations",
}

MANIFEST_FIELDS = {
    "run_id", "git_commit", "protocol_sha256", "variant_file_sha256",
    "started_at", "finished_at", "hostname", "gpu",
    "python", "torch", "library_versions", "epochs_completed",
    "resolution", "seed",
}

POOL_FIELDS = {"frame_id", "source_dataset", "video_id", "frame_index", "split"}
ACTIVE_QUEUE_FIELDS = {
    "run_id", "run_key", "stage", "profile", "arch", "strategy", "budget",
    "split_seed", "selection_seed", "train_seed", "variant_file",
    "protocol_sha256", "variant_sha256",
}
LEGACY_QUEUE_FIELDS = {
    "run_id", "tier", "kind", "variant", "groups", "weights",
    "seed", "budget", "arch",
}


class Report:
    def __init__(self) -> None:
        self.errors = 0

    def ok(self, msg: str) -> None:
        print(f"{OK}{msg}")

    def fail(self, msg: str) -> None:
        self.errors += 1
        print(f"{FAIL}{msg}")

    def check(self, cond: bool, msg: str) -> bool:
        self.ok(msg) if cond else self.fail(msg)
        return cond


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: niepoprawny JSON: {e}") from e


def _valid_bbox(value) -> bool:
    if not (isinstance(value, list) and len(value) == 4):
        return False
    try:
        x, y, width, height = map(float, value)
    except (TypeError, ValueError):
        return False
    return x >= 0 and y >= 0 and width > 0 and height > 0


def _pool_train_ids(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    return {row["frame_id"] for _, row in read_jsonl(path)
            if row.get("split") == "train" and "frame_id" in row}


def validate_pool(root: Path, rep: Report) -> None:
    path = root / "pool_manifest.jsonl"
    if not path.exists():
        rep.fail(f"{path} nie istnieje (Master jeszcze nie dostarczyl)")
        return
    rep.ok(f"{path} istnieje")
    train_ids: set[str] = set()
    frame_ids: set[str] = set()
    n = 0
    for lineno, row in read_jsonl(path):
        n += 1
        missing = POOL_FIELDS - row.keys()
        if missing:
            rep.fail(f"{path}:{lineno} brakuje pol: {sorted(missing)}")
        frame_id = row.get("frame_id")
        if frame_id in frame_ids:
            rep.fail(f"{path}:{lineno} zduplikowany frame_id={frame_id!r}")
        elif frame_id is not None:
            frame_ids.add(frame_id)
        if row.get("split") not in ("train", "val", "test"):
            rep.fail(f"{path}:{lineno} split={row.get('split')!r}")
        if not row.get("group_id") and not row.get("patient_id"):
            rep.fail(f"{path}:{lineno} brak group_id/patient_id do rozłącznego splitu")
        direct = bool(row.get("path"))
        archived = bool(row.get("archive")) and bool(row.get("member"))
        if direct == archived:
            rep.fail(f"{path}:{lineno} wymagane dokładnie jedno źródło: path albo archive+member")
        label_origin = row.get("label_origin")
        label_status = row.get("label_status")
        if row.get("split") in ("val", "test"):
            verified = label_origin == "human" or label_status in {"provided", "verified_empty"}
            if not verified:
                rep.fail(f"{path}:{lineno} val/test bez zweryfikowanej etykiety")
        if row.get("label_origin") == "human" and row.get("seed_distance_s", 0) != 0.0:
            rep.fail(f"{path}:{lineno} human z seed_distance_s != 0")
        annotations = row.get("annotations")
        if annotations is not None:
            if not isinstance(annotations, list):
                rep.fail(f"{path}:{lineno} annotations nie jest listą")
            else:
                for ann_i, annotation in enumerate(annotations):
                    if not isinstance(annotation, dict) or not _valid_bbox(annotation.get("bbox")):
                        rep.fail(f"{path}:{lineno} annotations[{ann_i}] ma błędny bbox")
        elif row.get("has_instrument"):
            if not _valid_bbox(row.get("bbox")):
                rep.fail(f"{path}:{lineno} has_instrument bez poprawnego bbox")
        elif "bbox" in row:
            rep.fail(f"{path}:{lineno} bbox obecny przy has_instrument=false")
        if row.get("split") == "train":
            if frame_id is not None:
                train_ids.add(frame_id)
    rep.ok(f"{path}: {n} wierszy, {len(train_ids)} klatek train")


def validate_variants(root: Path, rep: Report) -> None:
    vdir = root / "variants"
    if not vdir.is_dir():
        rep.fail(f"{vdir} nie istnieje (Agent 1)")
        return
    train_ids = _pool_train_ids(root / "pool_manifest.jsonl")
    files = sorted(vdir.glob("*.json")) + sorted(vdir.glob("**/variant.json"))
    if not files:
        rep.fail(f"{vdir} pusty (Agent 1)")
        return
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rid = d.get("run_id")
        rep.check(bool(rid), f"{f.name}: run_id obecny")
        rep.check(f.stem == rid, f"{f.name}: nazwa pliku == run_id")
        ids = d.get("frame_ids", d.get("selected_frame_ids", []))
        budget = d.get("budget")
        if budget == "full_train_pool":
            rep.ok(f"{rid}: FULLPOOL, {len(ids)} klatek")
        else:
            rep.check(len(ids) == budget,
                      f"{rid}: len(frame_ids)={len(ids)} == budget={budget}")
        rep.check(len(set(ids)) == len(ids), f"{rid}: frame_ids bez duplikatow")
        if train_ids is not None:
            outside = set(ids) - train_ids
            rep.check(not outside,
                      f"{rid}: wszystkie frame_ids w split=train"
                      + (f" (poza: {len(outside)})" if outside else ""))
        for key in ("variant", "seed"):
            rep.check(key in d, f"{rid}: pole {key}")


def validate_queue(root: Path, rep: Report) -> None:
    path = root / "queue.jsonl"
    if not path.exists():
        rep.fail(f"{path} nie istnieje (tools/build_run_manifest.py --out queue.jsonl)")
        return
    n = 0
    for lineno, row in read_jsonl(path):
        n += 1
        req = ACTIVE_QUEUE_FIELDS if "run_key" in row or "stage" in row else LEGACY_QUEUE_FIELDS
        missing = req - row.keys()
        if missing:
            rep.fail(f"queue.jsonl:{lineno} brakuje pol: {sorted(missing)}")
        w = row.get("weights") or {}
        if w:
            s = sum(w.values())
            if abs(s - 1.0) > 1e-3:
                rep.fail(f"queue.jsonl:{lineno} suma wag {s} != 1")
        if req is ACTIVE_QUEUE_FIELDS:
            rep.check(row.get("stage") == "train",
                      f"queue.jsonl:{lineno} stage=train")
            rep.check(bool(row.get("variant_file")),
                      f"queue.jsonl:{lineno} variant_file obecny")
    rep.ok(f"queue.jsonl: {n} biegow")


def validate_run_manifests(root: Path, rep: Report) -> None:
    rdir = root / "results"
    if not rdir.is_dir():
        rep.fail(f"{rdir} nie istnieje (Agent 2)")
        return
    proto = root / "configs" / "protocol.yaml"
    want = sha256(proto) if proto.exists() else None
    found = sorted(rdir.glob("*/run_manifest.json"))
    if not found:
        rep.fail("brak run_manifest.json w results/*/ (Agent 2)")
        return
    for f in found:
        d = json.loads(f.read_text(encoding="utf-8"))
        rid = f.parent.name
        missing = MANIFEST_FIELDS - d.keys()
        rep.check(not missing, f"{rid}: pola manifestu"
                  + (f" (brak: {sorted(missing)})" if missing else ""))
        if want is not None and d.get("protocol_sha256") != want:
            rep.fail(f"{rid}: protocol_sha256 != hash biezacego protocol.yaml")
        leftover = [p.name for p in f.parent.iterdir()
                    if p.suffix == ".pth" and p.stem not in ("last", "best")]
        rep.check(not leftover,
                  f"{rid}: tylko last.pth/best.pth"
                  + (f" (zostaly: {leftover})" if leftover else ""))


def validate_metrics(root: Path, rep: Report) -> None:
    rdir = root / "results"
    found = sorted(rdir.glob("*/metrics.json")) if rdir.is_dir() else []
    if not found:
        rep.fail("brak metrics.json w results/*/ (Agent 3)")
        return
    for f in found:
        d = json.loads(f.read_text(encoding="utf-8"))
        rid = f.parent.name
        for key in ("run_id", "checkpoint_rule", "results",
                    "operating_threshold", "threshold_selected_on"):
            rep.check(key in d, f"{rid}: metrics pole {key}")
        rep.check(d.get("checkpoint_rule") in ("last_epoch", "best_val"),
                  f"{rid}: checkpoint_rule poprawne")
        rep.check(d.get("threshold_selected_on") == "val",
                  f"{rid}: prog wybrany na val")
        for sname, block in (d.get("results") or {}).items():
            missing = METRIC_FIELDS - block.keys()
            rep.check(not missing, f"{rid}/{sname}: komplet metryk"
                      + (f" (brak: {sorted(missing)})" if missing else ""))


def validate_ledger(root: Path, rep: Report) -> None:
    path = root / "external_access_ledger.jsonl"
    if not path.exists():
        rep.fail(f"{path} nie istnieje (wymagany od pierwszego dnia)")
        return
    n = sum(1 for _ in read_jsonl(path))
    rep.ok(f"{path}: {n} wpisow")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="katalog SmallViTAblation")
    ap.add_argument("--only", nargs="*",
                    choices=["pool", "variants", "queue", "manifests",
                             "metrics", "ledger"],
                    default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    sel = set(args.only) if args.only else None

    def want(name: str) -> bool:
        return sel is None or name in sel

    rep = Report()
    steps = [
        ("pool", validate_pool),
        ("queue", validate_queue),
        ("variants", validate_variants),
        ("manifests", validate_run_manifests),
        ("metrics", validate_metrics),
        ("ledger", validate_ledger),
    ]
    for name, fn in steps:
        if want(name):
            print(f"--- {name} ---")
            try:
                fn(root, rep)
            except (ValueError, OSError) as e:
                rep.fail(str(e))
    print(f"=== bledy: {rep.errors} ===")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
