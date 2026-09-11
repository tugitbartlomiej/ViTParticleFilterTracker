"""Cztery baseline'y selekcji (AGENT_1 zadanie 5). Wyjscie: variants/<run_id>.json.

  B-RANDAUG   losowe `budget` klatek z puli treningowej.
  B-RANDSRC   losowanie KLATEK ZRODLOWYCH (przed augmentacja), potem wszystkie
              ich warianty az do budget. Wzorzec nazw: `..._aug_N` to wariant
              klatki bez przyrostka. Gdy pula nie ma augmentacji (nowe wideo),
              kazda klatka jest wlasnym zrodlem - log wypisuje, ktora sciezka
              poszlismy.
  B-UNIFORM   rowna liczba klatek z kazdej zweryfikowanej jednostki podzialu:
              group_id, a dla starszych manifestow patient_id.
  B-FULLPOOL  wszystkie klatki train, bez wyboru (budget = rozmiar puli).

Wyjsciowy pliku identyczny z select_subset.py (agents/README.md 4.2).
Determinizm: wszystko z np.random.RandomState(seed), pula posortowana po frame_id.

Uzycie:
    py -3.11 pipeline/selection/baselines.py --queue queue.jsonl [--run-id X]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.features.common import (  # noqa: E402
    load_protocol, read_manifest, resolve_under_root, train_pool)

AUG_RE = re.compile(r"^(?P<src>.+)_aug_\d+$")

BASELINE_IDS = ("B-RANDAUG", "B-RANDSRC", "B-UNIFORM", "B-FULLPOOL")


def source_key(frame_id: str) -> str:
    m = AUG_RE.match(frame_id)
    return m.group("src") if m else frame_id


def sample_unique(rng: np.random.RandomState, n: int, k: int) -> list[int]:
    return sorted(rng.choice(n, size=k, replace=False).tolist())


def b_randaug(pool_ids: list[str], budget: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    return sample_unique(rng, len(pool_ids), budget)


def b_randsrc(pool_ids: list[str], budget: int, seed: int) -> list[int]:
    groups: dict[str, list[int]] = {}
    for i, fid in enumerate(pool_ids):
        groups.setdefault(source_key(fid), []).append(i)
    n_aug = sum(1 for fid in pool_ids if AUG_RE.match(fid))
    if n_aug > 0:
        print(f"[B-RANDSRC] sciezka: PULA AUGMENTOWANA - {n_aug} klatek "
              f"_aug_N w {len(groups)} grupach zrodlowych; losuje klatki "
              f"zrodlowe i biore wszystkie ich warianty")
    else:
        print(f"[B-RANDSRC] sciezka: PULA BEZ AUGMENTACJI (nowe wideo) - "
              f" kazda klatka jest wlasnym zrodlem ({len(groups)} grup); "
              f"losowanie po klatkach zrodlowych")
    rng = np.random.RandomState(seed)
    src_ids = sorted(groups)
    order = rng.permutation(len(src_ids))
    chosen: list[int] = []
    taken: set[int] = set()
    for oi in order:
        if len(chosen) >= budget:
            break
        members = sorted(groups[src_ids[oi]])
        if len(chosen) + len(members) <= budget:
            for m in members:
                chosen.append(m)
                taken.add(m)
        else:
            # grupa nie miesci sie w calosci - dopelnij deterministycznie
            # pierwszymi (wg frame_id) wariantami tej grupy
            for m in members:
                if len(chosen) >= budget:
                    break
                if m not in taken:
                    chosen.append(m)
                    taken.add(m)
    return sorted(chosen)


def b_uniform(pool: list[dict], pool_ids: list[str], budget: int,
              seed: int) -> list[int]:
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(pool):
        group = r.get("group_id") or r.get("patient_id")
        if group is None:
            raise ValueError("B-UNIFORM requires group_id or patient_id")
        by_group.setdefault(str(group), []).append(i)
    groups = sorted(by_group)
    p = len(groups)
    base, rem = divmod(budget, p)
    alloc = {group: base + (1 if j < rem else 0)
             for j, group in enumerate(groups)}
    # Mniejsze grupy oddaja nadwyzke; redystrybucja jest deterministyczna.
    rng = np.random.RandomState(seed)
    chosen: set[int] = set()
    leftover = 0
    for group in groups:
        avail = by_group[group]
        take = min(alloc[group], len(avail))
        leftover += alloc[group] - take
        for i in sample_unique(rng, len(avail), take):
            chosen.add(avail[i])
    # niedobor (pacjent mial mniej klatek niz alokacja) dopelniamy
    # deterministycznie: pierwsze niewybrane klatki w kolejnosci frame_id
    if leftover > 0:
        for i in range(len(pool_ids)):
            if leftover <= 0:
                break
            if i not in chosen:
                chosen.add(i)
                leftover -= 1
    if leftover > 0:
        raise RuntimeError(f"B-UNIFORM: pula za mala, brakuje {leftover} klatek")
    return sorted(chosen)


def select_baseline(variant: str, pool: list[dict], budget, seed: int,
                    run_id: str) -> dict:
    pool_ids = [r["frame_id"] for r in pool]
    n = len(pool_ids)
    if variant == "B-FULLPOOL":
        idxs = list(range(n))
        method = "full_train_pool"
        budget_out = n
    else:
        if not isinstance(budget, int):
            raise ValueError(f"{run_id}: {variant} wymaga budget int")
        if budget > n:
            raise ValueError(f"{run_id}: budget {budget} > pula train {n}")
        budget_out = budget
        if variant == "B-RANDAUG":
            idxs = b_randaug(pool_ids, budget, seed)
            method = "random_frames_from_train_pool"
        elif variant == "B-RANDSRC":
            idxs = b_randsrc(pool_ids, budget, seed)
            method = "random_source_frames_then_variants"
        elif variant == "B-UNIFORM":
            idxs = b_uniform(pool, pool_ids, budget, seed)
            method = "uniform_per_operation_group"
        else:
            raise ValueError(f"{run_id}: nieznany baseline '{variant}'")
    if len(idxs) != budget_out:
        raise RuntimeError(f"{run_id}: wybrano {len(idxs)} != {budget_out}")
    return {
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "budget": budget_out,
        "frame_ids": sorted(pool_ids[i] for i in idxs),
        "selection_meta": {
            "weights": {}, "pca_dim": None, "algorithm": method,
            "representative": None, "groups": [],
            "n_pool_train": n, "override": None,
            "baseline": variant,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--variants-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--run-id", action="append", default=None)
    # alternatywnie: bez queue, jeden bieg z reki
    ap.add_argument("--variant", choices=BASELINE_IDS, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--budget", default=None)
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    paths_cfg = protocol.get("paths", {})
    manifest = args.manifest or resolve_under_root(
        paths_cfg.get("pool_manifest", "pool_manifest.jsonl"))
    variants_dir = Path(args.variants_dir or resolve_under_root(
        paths_cfg.get("variants_dir", "variants")))
    variants_dir.mkdir(parents=True, exist_ok=True)
    pool = train_pool(read_manifest(manifest))
    print(f"Pula train: {len(pool)} klatek")

    if args.variant:
        if args.seed is None or args.budget is None:
            sys.exit("--variant wymaga --seed i --budget "
                     "(budget: int lub 'full_train_pool')")
        budget: object = args.budget
        try:
            budget = int(str(budget))
        except ValueError:
            if budget != "full_train_pool":
                sys.exit(f"--budget: ani int, ani 'full_train_pool': {budget}")
        runs = [{"run_id": f"{args.variant}__s{args.seed}__b{budget}",
                 "variant": args.variant, "seed": args.seed,
                 "budget": budget}]
    else:
        queue = args.queue or resolve_under_root(
            paths_cfg.get("queue_file", "queue.jsonl"))
        runs = []
        with open(queue, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if str(r.get("variant", "")).startswith("B-") \
                        or r.get("kind") == "baseline":
                    runs.append(r)
        if args.run_id:
            runs = [r for r in runs if r["run_id"] in set(args.run_id)]

    for run in runs:
        variant = run["variant"]
        if variant not in BASELINE_IDS:
            print(f"[skip] {run['run_id']}: '{variant}' nie jest baseline")
            continue
        result = select_baseline(variant, pool, run["budget"], int(run["seed"]),
                                 run["run_id"])
        out = variants_dir / f"{run['run_id']}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"[ok] {run['run_id']}: {len(result['frame_ids'])} klatek -> {out}")


if __name__ == "__main__":
    main()
