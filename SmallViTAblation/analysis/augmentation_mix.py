#!/usr/bin/env python3
"""
Udzial typow augmentacji w kazdym wariancie vs udzial w calej puli (Agent 5, zadanie 2).

Odniesienie do zarzutu recenzenta 1: poprzednio 71,3% wybranych obrazow to kopie
powstale przez augmentacje. Czysta analiza nazw (wzorzec _aug_[typ_]N), bez GPU.
Obsluguje nowa puli wideo BEZ augmentacji — raport mowi wowczas
"brak augmentacji w puli".

Uzycie:
    py -3.11 analysis/augmentation_mix.py
    py -3.11 analysis/augmentation_mix.py --pool-manifest X --variants-dir Y --out Z.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _common as C

REVIEWER_CLAIM_PCT = 71.3  # zarzut recenzenta 1, wiersz "drobne" w 04_REVIEWER_MAP.md


def classify(frame_id: str) -> tuple[str, str]:
    """Zwraca (kategoria, typ): ('original','-') albo ('augmented', typ|'unspecified')."""
    stem = C.stem_of(frame_id)
    _, aug_type, _ = C.split_aug(stem)
    if C.AUG_RE.search(stem):
        return "augmented", aug_type or "unspecified"
    return "original", "-"


def mix(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: round(100.0 * v / total, 2) for k, v in counter.most_common()}


def fmt_table(name: str, cat: Counter, types: Counter, total: int) -> str:
    aug_n = cat.get("augmented", 0)
    aug_pct = 100.0 * aug_n / total if total else 0.0
    lines = [f"### {name}  (n={total})",
             f"  oryginaly: {cat.get('original', 0)} ({100 - aug_pct:.1f}%)",
             f"  kopie augmentowane: {aug_n} ({aug_pct:.1f}%)"]
    if types:
        lines.append("  typy przekszalcen:")
        for t, n in types.most_common():
            lines.append(f"    {t:20} {n:7}  ({100.0 * n / total:5.1f}%)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-manifest", type=Path, default=None)
    ap.add_argument("--variants-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    protocol = C.load_protocol(args.protocol)
    acfg = C.analysis_cfg(protocol)
    pool_path = args.pool_manifest or C.cfg_path(acfg, "pool_manifest")
    variants_dir = args.variants_dir or C.cfg_path(acfg, "variants_dir")
    if not pool_path or not pool_path.exists():
        raise SystemExit(f"Brak pool_manifest: {pool_path}")

    pool = C.load_pool_manifest(pool_path)
    train = [r for r in pool if r.get("split") == "train"]
    scope = train or pool
    scope_name = "pula treningowa (split=train)" if train else "cala pula"

    pool_cat, pool_types = Counter(), Counter()
    for r in scope:
        cat, typ = classify(r["frame_id"])
        pool_cat[cat] += 1
        if cat == "augmented":
            pool_types[typ] += 1

    pool_total = sum(pool_cat.values())
    pool_aug_pct = 100.0 * pool_cat.get("augmented", 0) / pool_total if pool_total else 0.0
    no_aug = pool_cat.get("augmented", 0) == 0

    print(fmt_table(f"PULA: {scope_name}", pool_cat, pool_types, pool_total))
    if no_aug:
        print("  >> BRAK AUGMENTACJI W PULI (np. nowa pula wideo) — udzialy augmentacji "
              "sa trywialnie 0%; zarzut 71,3% nie dotyczy tej puli.")

    by_id = {r["frame_id"]: r for r in pool}
    report = {
        "pool": {"scope": scope_name, "total": pool_total,
                 "categories": dict(pool_cat), "category_pct": mix(pool_cat),
                 "aug_types": dict(pool_types), "aug_type_pct": mix(pool_types),
                 "no_augmentation_in_pool": no_aug},
        "variants": {},
        "reviewer_claim_pct": REVIEWER_CLAIM_PCT,
    }

    variants = C.iter_variant_files(Path(variants_dir)) if variants_dir and Path(variants_dir).exists() else []
    if not variants:
        print(f"\nBrak wariantow w {variants_dir} — raportuje tylko sklad puli.")

    print(f"\n{'wariant':40} {'n':>6} {'aug%':>7} {'orig%':>7}   typy (udzial %)")
    for run_id, data in variants:
        frame_ids = data.get("frame_ids", [])
        cat, types = Counter(), Counter()
        unknown = 0
        for fid in frame_ids:
            if fid not in by_id:
                unknown += 1
            c, t = classify(fid)
            cat[c] += 1
            if c == "augmented":
                types[t] += 1
        total = sum(cat.values())
        aug_pct = 100.0 * cat.get("augmented", 0) / total if total else 0.0
        typ_str = ", ".join(f"{t}={100.0 * n / total:.1f}%" for t, n in types.most_common()) or "-"
        note = f"  ({unknown} spoza puli!)" if unknown else ""
        print(f"{run_id:40} {total:6} {aug_pct:6.1f}% {100 - aug_pct:6.1f}%   {typ_str}{note}")
        report["variants"][run_id] = {
            "total": total, "categories": dict(cat), "category_pct": mix(cat),
            "aug_types": dict(types), "aug_type_pct": mix(types),
            "frame_ids_not_in_pool": unknown,
            "augmented_pct": round(aug_pct, 2),
            "delta_vs_pool_pct": round(aug_pct - pool_aug_pct, 2),
            "delta_vs_reviewer_claim_pct": round(aug_pct - REVIEWER_CLAIM_PCT, 2),
        }

    if variants and not no_aug:
        deltas = {rid: v["augmented_pct"] for rid, v in report["variants"].items()}
        spread = max(deltas.values()) - min(deltas.values()) if deltas else 0.0
        report["aug_pct_spread_between_variants"] = round(spread, 2)
        print(f"\nRozstep udzialu augmentacji miedzy wariantami: {spread:.1f} pp "
              f"(vs pula {pool_aug_pct:.1f}%, zarzut recenzenta {REVIEWER_CLAIM_PCT}%)")

    out = args.out or C.cfg_path(acfg, "out_dir") / "augmentation_mix.json"
    C.write_json(out, report)


if __name__ == "__main__":
    main()
