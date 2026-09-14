#!/usr/bin/env python3
"""Create static selections or cumulative active rounds without launching training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from .acquisition import STRATEGIES, select_batch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_embeddings(path: Path, rows: list[dict]) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "frame_ids" not in data or "features" not in data:
        raise ValueError("embedding NPZ requires frame_ids and features")
    frame_ids = [str(x) for x in data["frame_ids"].tolist()]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("duplicate frame_ids in embedding cache")
    by_id = {frame_id: i for i, frame_id in enumerate(frame_ids)}
    missing = [row["frame_id"] for row in rows if row["frame_id"] not in by_id]
    if missing:
        raise ValueError(f"embedding cache misses {len(missing)} pool frames: {missing[:3]}")
    return np.asarray(data["features"])[[by_id[row["frame_id"]] for row in rows]]


def _load_scores(path: Path | None) -> dict[str, dict[str, float]] | None:
    if path is None:
        return None
    rows = _read_jsonl(path)
    return {str(row["frame_id"]): row for row in rows}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path,
                    help="real feature NPZ; optional only for static random/uniform_time")
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--static", action="store_true",
                    help="choose the first budget with the requested selector, without shared L0")
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 1500, 2500])
    ap.add_argument("--initial-selection", type=Path,
                    help="JSON with selected_frame_ids; shared L0 for every strategy")
    ap.add_argument("--initial-strategy", choices=["uniform_time", "random"],
                    default="uniform_time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gap-frames", type=float, default=5)
    ap.add_argument("--coverage-weight", type=float, default=.60)
    ap.add_argument("--auxiliary-weight", type=float, default=.40)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.static and args.strategy not in {"random", "uniform_time", "coverage"}:
        raise ValueError("static mode supports random, uniform_time and coverage")
    if args.static and args.initial_selection:
        raise ValueError("static mode does not accept an active initial selection")
    if not args.static and args.embeddings is None:
        raise ValueError("active rounds require --embeddings")
    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] < 1:
        raise ValueError("budgets must be positive")
    pool_all = _read_jsonl(args.pool)
    rows = [row for row in pool_all if row.get("split", "train") == "train"]
    excluded_non_train = len(pool_all) - len(rows)
    embeddings = _load_embeddings(args.embeddings, rows) if args.embeddings else None
    scores = _load_scores(args.scores)
    selected: list[str] = []
    initial_audit = None
    initial_strategy = args.strategy if args.static else args.initial_strategy
    if args.initial_selection:
        raw = json.loads(args.initial_selection.read_text(encoding="utf-8"))
        selected = list(map(str, raw.get("selected_frame_ids", raw.get("frame_ids", []))))
    else:
        selected, initial_audit = select_batch(
            rows, embeddings, batch_size=budgets[0], strategy=initial_strategy,
            seed=args.seed, min_gap_frames=args.min_gap_frames)
    if len(selected) != budgets[0]:
        raise ValueError("initial selection must exactly equal the first cumulative budget")
    if len(selected) != len(set(selected)):
        raise ValueError("initial selection contains duplicate frame_ids")
    pool_ids = {str(row["frame_id"]) for row in rows}
    missing_initial = sorted(set(selected) - pool_ids)
    if missing_initial:
        raise ValueError(f"initial selection contains frames outside train pool: {missing_initial[:3]}")

    args.out.mkdir(parents=True, exist_ok=True)
    rounds = []
    for round_number, budget in enumerate(budgets, 1):
        need = budget - len(selected)
        if need < 0:
            raise ValueError("budgets must be cumulative and non-decreasing")
        audit = initial_audit if args.static and round_number == 1 else None
        if need:
            batch, audit = select_batch(
                rows, embeddings, batch_size=need, strategy=args.strategy,
                seed=args.seed + round_number - 1, selected_ids=selected,
                scores=scores, min_gap_frames=args.min_gap_frames,
                coverage_weight=args.coverage_weight,
                auxiliary_weight=args.auxiliary_weight,
            )
            selected.extend(batch)
        variant = {
            "schema_version": 1,
            "variant_id": f"active-{args.strategy}-s{args.seed}-b{budget}",
            "run_id": f"active-{args.strategy}-s{args.seed}-b{budget}",
            "strategy": args.strategy,
            "selection_mode": "static" if args.static else "active",
            "variant": f"active-{args.strategy}",
            "seed": args.seed,
            "budget": budget,
            "round": round_number,
            "selected_frame_ids": list(selected),
            "frame_ids": list(selected),
            "parent_budget": rounds[-1]["budget"] if rounds else None,
            "pool_sha256": _sha256(args.pool),
            "embeddings_sha256": _sha256(args.embeddings) if args.embeddings else None,
            "scores_sha256": _sha256(args.scores) if args.scores else None,
            "selection_audit": audit,
            "initial_selection": {
                "strategy": initial_strategy if not args.initial_selection else "provided",
                "frame_ids": list(selected[:budgets[0]]),
                "audit": initial_audit,
            },
        }
        target = args.out / f"{variant['variant_id']}.json"
        _atomic_json(target, variant)
        rounds.append({"budget": budget, "path": str(target),
                       "n_selected": len(selected)})
    _atomic_json(args.out / "active_curation_manifest.json", {
        "schema_version": 1,
        "strategy": args.strategy,
        "seed": args.seed,
        "rounds": rounds,
        "same_initial_selection_required_across_strategies": not args.static,
        "selection_mode": "static" if args.static else "active",
        "labels_used_for_ranking": False,
        "pool_frames_total": len(pool_all),
        "train_pool_frames": len(rows),
        "non_train_frames_excluded": excluded_non_train,
    })
    print(json.dumps({"strategy": args.strategy, "rounds": rounds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
