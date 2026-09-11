#!/usr/bin/env python3
"""Build a bounded Eden training manifest from frozen active-curation variants."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _canonical_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _variants(path: Path) -> list[tuple[Path, dict]]:
    result = []
    for file in sorted(path.glob("active-*.json")):
        value = json.loads(file.read_text(encoding="utf-8"))
        if not all(key in value for key in ("strategy", "budget", "seed", "frame_ids")):
            continue
        if len(value["frame_ids"]) != int(value["budget"]):
            raise ValueError(f"{file}: frame_ids does not match budget")
        result.append((file.resolve(), value))
    return result


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "SmallViTAblation"],
            cwd=ROOT.parent, capture_output=True, text=True, check=True,
            timeout=15).stdout.strip())
        return {"commit": commit, "small_vit_ablation_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "small_vit_ablation_dirty": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants-dir", type=Path, required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--profiles-config", type=Path, default=ROOT / "configs/profiles.yaml")
    ap.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strategies", nargs="+")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--allow-over-max", action="store_true")
    args = ap.parse_args(argv)

    config = yaml.safe_load(args.profiles_config.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    profiles = config.get("profiles", {})
    if args.profile not in profiles:
        raise ValueError(f"unknown profile {args.profile!r}")
    profile = profiles[args.profile]
    models = args.models or profile["models"]
    unknown_models = sorted(set(models) - set(profile["models"]))
    if unknown_models:
        raise ValueError(f"models outside profile {args.profile}: {unknown_models}")
    allowed_budgets = set(map(int, profile["budgets"]))
    if profile.get("require_explicit_strategies") and not args.strategies:
        raise ValueError(f"profile {args.profile} requires --strategies with the frozen winner")
    allowed_strategies = set(args.strategies or profile.get(
        "strategies", config["active_curation"]["strategies"]))
    unknown_strategies = sorted(allowed_strategies - set(config["active_curation"]["strategies"]))
    if unknown_strategies:
        raise ValueError(f"unknown active-curation strategies: {unknown_strategies}")
    split_seed = int(protocol.get("split", {}).get("seed", 42))
    pairing = profile.get("seed_pairing", "matched")
    if pairing not in {"matched", "cross"}:
        raise ValueError(f"unknown seed_pairing={pairing!r}")

    runs = []
    for variant_path, variant in _variants(args.variants_dir):
        if int(variant["budget"]) not in allowed_budgets:
            continue
        if variant["strategy"] not in allowed_strategies:
            continue
        for model in models:
            train_seeds = ([int(variant["seed"])] if pairing == "matched"
                           else list(map(int, profile["seeds"])))
            if pairing == "matched" and int(variant["seed"]) not in set(map(int, profile["seeds"])):
                continue
            for train_seed in train_seeds:
                semantic = {
                    "schema_version": 1,
                    "stage": "train",
                    "profile": args.profile,
                    "arch": model,
                    "strategy": variant["strategy"],
                    "budget": int(variant["budget"]),
                    "split_seed": split_seed,
                    "selection_seed": int(variant["seed"]),
                    "train_seed": int(train_seed),
                    "variant_sha256": _file_hash(variant_path),
                    "protocol_sha256": _file_hash(args.protocol),
                }
                run_key = _canonical_hash(semantic)
                run_id = (f"{variant['strategy']}__b{variant['budget']}"
                          f"__sel{variant['seed']}__train{train_seed}__{model}")
                runs.append({**semantic, "run_id": run_id, "run_key": run_key,
                             "seed": int(train_seed),
                             "variant_file": str(variant_path)})
    ids = [run["run_id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run_id in active manifest")
    if not runs:
        raise ValueError("no variants match the selected profile, budgets, seeds and strategies")
    maximum = int(profile["max_runs"])
    if len(runs) > maximum and not args.allow_over_max:
        raise ValueError(
            f"profile {args.profile} permits at most {maximum} runs, generated {len(runs)}; "
            "filter --strategies/--models or use --allow-over-max explicitly")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for run in runs:
            stream.write(json.dumps(run, ensure_ascii=False) + "\n")
    lock = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_snapshot": profile,
        "active_curation_snapshot": config["active_curation"],
        "models": models,
        "strategies": sorted(allowed_strategies),
        "seed_pairing": pairing,
        "split_seed": split_seed,
        "profiles_config": str(args.profiles_config.resolve()),
        "profiles_sha256": _file_hash(args.profiles_config),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _file_hash(args.protocol),
        "variants": [{"path": str(path), "sha256": _file_hash(path)}
                     for path, value in _variants(args.variants_dir)
                     if int(value["budget"]) in allowed_budgets
                     and value["strategy"] in allowed_strategies],
        "manifest": str(args.out.resolve()),
        "manifest_sha256": _file_hash(args.out),
        "runs": len(runs),
        "git": _git_state(),
    }
    lock_path = args.out.with_suffix(args.out.suffix + ".lock.json")
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(json.dumps({"profile": args.profile, "runs": len(runs),
                      "out": str(args.out), "lock": str(lock_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
