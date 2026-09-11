#!/usr/bin/env python3
"""
Rozwija configs/variants.yaml + configs/protocol.yaml do plaskiej listy biegow.

Kazdy bieg to jeden wiersz: id wariantu, wagi grup cech, ziarno, budzet,
architektura, poziom. Ta lista jest wejsciem dla generatora zbiorow i dla
kolejki SLURM na Edenie (jeden wiersz = jedno zadanie workera).

Uzycie:
    py -3.11 tools/build_run_manifest.py                 # podsumowanie
    py -3.11 tools/build_run_manifest.py --out runs.jsonl
    py -3.11 tools/build_run_manifest.py --tier A --out runs_a.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Brak pyyaml. Zainstaluj: py -3.11 -m pip install pyyaml")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load_configs() -> tuple[dict, dict]:
    protocol = yaml.safe_load((CONFIGS / "protocol.yaml").read_text(encoding="utf-8"))
    variants = yaml.safe_load((CONFIGS / "variants.yaml").read_text(encoding="utf-8"))
    return protocol, variants


def renormalize(groups: list[str], base: dict[str, float]) -> dict[str, float]:
    """Waga 0 dla grup wylaczonych, reszta przeskalowana PROPORCJONALNIE do sumy 1.

    To jest jedyna regula wylaczania grupy w calym eksperymencie (docs/02_MATRIX).
    """
    key = {"D": "dino", "E": "el2n", "S": "sam", "F": "fourier"}
    active = {key[g]: base[key[g]] for g in groups}
    total = sum(active.values())
    if total <= 0:
        raise ValueError(f"Zerowa suma wag dla grup {groups}")
    out = {name: 0.0 for name in base}
    for name, w in active.items():
        out[name] = round(w / total, 6)
    return out


def expand(protocol: dict, variants: dict, tier: str | None) -> list[dict]:
    base = protocol["features"]["weights_base"]
    budget_a = protocol["budgets"]["tier_a"]
    budget_b = protocol["budgets"]["tier_b"]
    seeds_matrix = protocol["seeds"]["matrix"]
    seeds_b = protocol["seeds"]["tier_b"]
    arch_a = protocol["model"]["name"]

    runs: list[dict] = []

    def add(**kw):
        runs.append(kw)

    # --- poziom A: macierz 15 kombinacji ---
    if tier in (None, "A"):
        for v in variants["matrix"]:
            for seed in seeds_matrix:
                add(run_id=f"{v['id']}__s{seed}__b{budget_a}", tier="A", kind="matrix",
                    variant=v["id"], groups=v["groups"],
                    weights=renormalize(v["groups"], base),
                    seed=seed, budget=budget_a, arch=arch_a, note=v["note"])

        # --- poziom A: baseline'y ---
        for b in variants["baselines"]:
            draws = b["draws"]
            budget = "full_train_pool" if b["kind"] == "full_train_pool" else budget_a
            for i in range(draws):
                seed = seeds_matrix[i] if i < len(seeds_matrix) else 1000 + i
                add(run_id=f"{b['id']}__s{seed}__b{budget}", tier="A", kind="baseline",
                    variant=b["id"], groups=[], weights={}, seed=seed,
                    budget=budget, arch=arch_a, note=b["note"])

        # --- poziom A: analiza czulosci ---
        sens = variants["sensitivity"]
        ref = sens["reference"]
        ref_groups = next(v["groups"] for v in variants["matrix"] if v["id"] == ref)
        for study in sens["studies"]:
            applies = study.get("applies_to", [None])
            for val in study["values"]:
                for target in applies:
                    sid = f"{study['id']}_{study['param']}_{val}" + (f"_{target}" if target else "")
                    sid = sid.replace("/", "-").replace(" ", "")
                    add(run_id=f"{sid}__s{sens['seeds'][0]}__b{budget_a}", tier="A",
                        kind="sensitivity", variant=ref, groups=ref_groups,
                        weights=renormalize(ref_groups, base), seed=sens["seeds"][0],
                        budget=budget_a, arch=arch_a,
                        override={study["param"]: val, "target_group": target},
                        note=f"czulosc: {study['param']}={val}")

        # --- poziom A: sweep budzetu (bez powtorzen tego, co juz jest) ---
        sw = variants["size_sweep"]
        seen = {(r["variant"], r["seed"], r["budget"]) for r in runs}
        for vid in sw["variants"]:
            groups = next((v["groups"] for v in variants["matrix"] if v["id"] == vid), [])
            for budget in sw["budgets"]:
                for seed in sw["seeds"]:
                    if (vid, seed, budget) in seen:
                        continue
                    add(run_id=f"{vid}__s{seed}__b{budget}", tier="A", kind="size_sweep",
                        variant=vid, groups=groups,
                        weights=renormalize(groups, base) if groups else {},
                        seed=seed, budget=budget, arch=arch_a, note="sweep budzetu")

    # --- poziom B: potwierdzenie + przeniesienie architektur ---
    if tier in (None, "B"):
        tb = variants["tier_b"]
        for vid in tb["variants"]:
            if str(vid).startswith("TBD_"):
                raise ValueError(
                    "Tier B contains unresolved TBD variants. Create and pass a locked "
                    "promotion decision before generating confirmation runs.")
            groups = next((v["groups"] for v in variants["matrix"] if v["id"] == vid), [])
            for seed in seeds_b:
                add(run_id=f"{vid}__s{seed}__b{budget_b}__{tb['architectures']['primary']}",
                    tier="B", kind="confirm", variant=vid, groups=groups,
                    weights=renormalize(groups, base) if groups else {},
                    seed=seed, budget=budget_b, arch=tb["architectures"]["primary"],
                    note="potwierdzenie na pelnym budzecie")
            for arch in tb["architectures"]["transfer"]:
                for seed in tb["architectures"]["transfer_seeds"]:
                    add(run_id=f"{vid}__s{seed}__b{budget_b}__{arch}", tier="B",
                        kind="transfer", variant=vid, groups=groups,
                        weights=renormalize(groups, base) if groups else {},
                        seed=seed, budget=budget_b, arch=arch,
                        note="przeniesienie na inna architekture, iso-resource")

    return runs


def summarize(runs: list[dict]) -> None:
    from collections import Counter
    print(f"Biegow razem: {len(runs)}")
    for field in ("tier", "kind", "arch"):
        c = Counter(r[field] for r in runs)
        print(f"  wg {field}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    budgets = Counter(str(r["budget"]) for r in runs)
    print("  wg budzetu: " + ", ".join(f"{k}={v}" for k, v in sorted(budgets.items())))
    # obrazo-epoki jako miara kosztu niezalezna od sprzetu
    img = sum(r["budget"] for r in runs if isinstance(r["budget"], int))
    print(f"  suma obrazow na epoke (bez FULLPOOL): {img:,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default="A",
                    help="safe default is A; B is rejected while variants contain TBD")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    protocol, variants = load_configs()
    runs = expand(protocol, variants, args.tier)
    summarize(runs)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in runs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Zapisano: {args.out}")


if __name__ == "__main__":
    main()
