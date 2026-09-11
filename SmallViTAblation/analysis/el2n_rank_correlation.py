#!/usr/bin/env python3
"""
Korelacja rankingow trudnosci klatki: EL2N z modelu DETR vs EL2N z modelu YOLO
(Agent 5, zadanie 3). Zamkniecie zarzutu o bledne kolo: wysoka korelacja
KOLEJNOSCI znaczy, ze miara opisuje trudnosc klatki, a nie preferencje architektury.

Wejscia:
  DETR: feature_cache/el2n_<seed>.pkl   (Agent 1; wzorzec: analysis.el2n.detr_pattern)
  YOLO: sciezka wg AdvancedDatasetSelection/ablation/yolo/PLAN.md, krok 1
        (yolo_ep170_el2n_scores.json = {frame_id: loss}; domyslnie z protocol.yaml:
        analysis.el2n.yolo_scores, albo --yolo-path)

Lazy import: pickle/json zawsze, scipy.stats tylko jesli jest (fallback: Spearman
bez p-value), torch YOLO NIGDY nie jest potrzebny — czytamy gotowe wyniki.

Uzycie:
    py -3.11 analysis/el2n_rank_correlation.py --seed 42
    py -3.11 analysis/el2n_rank_correlation.py --fake     # test na danych syntetycznych
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import _common as C


def _coerce_scores(obj) -> dict[str, float]:
    """dict frame_id->float; toleruje zagniezdzenie {'scores': {...}} i wartosci-dicty."""
    if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
        obj = obj["scores"]
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            v = next((v[key] for key in ("el2n", "score", "loss", "value") if key in v), None)
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_detr(path: Path) -> dict[str, float]:
    with path.open("rb") as f:
        return _coerce_scores(pickle.load(f))


def load_yolo(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".json":
        return _coerce_scores(json.loads(path.read_text(encoding="utf-8")))
    with path.open("rb") as f:  # .pkl tez dopuszczamy
        return _coerce_scores(pickle.load(f))


def make_fake(protocol, acfg) -> tuple[dict[str, float], dict[str, float]]:
    """Deterministyczne syntetyczne rankingi do testow hydraulicznych."""
    pool_path = C.cfg_path(acfg, "pool_manifest")
    if pool_path and pool_path.exists():
        ids = [r["frame_id"] for r in C.load_pool_manifest(pool_path)
               if r.get("split") == "train"]
    else:
        ids = [f"synth/frame_{i:06d}" for i in range(500)]
    seed = int(protocol.get("split", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    detr = rng.random(len(ids))
    # YOLO = skorelowana monotonicznie + szum: oczekiwany spearman ~0.8
    yolo = 0.8 * detr + 0.2 * rng.random(len(ids))
    return dict(zip(ids, detr.tolist())), dict(zip(ids, yolo.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None,
                    help="ziarno pliku el2n_<seed>.pkl (domyslnie pierwsze z protocol seeds.matrix)")
    ap.add_argument("--detr-path", type=Path, default=None)
    ap.add_argument("--yolo-path", type=Path, default=None)
    ap.add_argument("--fake", action="store_true",
                    help="syntetyczne rankingi zamiast plikow (test)")
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    protocol = C.load_protocol(args.protocol)
    acfg = C.analysis_cfg(protocol)
    el2n_cfg = acfg.get("el2n", {})

    if args.fake:
        detr, yolo = make_fake(protocol, acfg)
        src = "fake"
        print("Tryb --fake: syntetyczne rankingi (oczekiwany wysoki spearman).")
    else:
        seed = args.seed
        if seed is None:
            seed = int(protocol.get("seeds", {}).get("matrix", [42])[0])
        detr_path = args.detr_path or C.cfg_path(acfg, "feature_cache_dir") / \
            el2n_cfg.get("detr_pattern", "el2n_{seed}.pkl").format(seed=seed)
        yolo_path = args.yolo_path or C.resolve(el2n_cfg.get(
            "yolo_scores",
            "AdvancedDatasetSelection/ablation/yolo/yolo_ep170_el2n_scores.json"))
        if not detr_path.exists():
            raise SystemExit(f"Brak pliku DETR EL2N: {detr_path}\n"
                             f"  (Agent 1 jeszcze go nie wygenerowal; na testy uzyj --fake)")
        if not yolo_path.exists():
            raise SystemExit(f"Brak pliku YOLO EL2N: {yolo_path}\n"
                             f"  (plan: AdvancedDatasetSelection/ablation/yolo/PLAN.md krok 1; "
                             f"na testy uzyj --fake)")
        detr, yolo = load_detr(detr_path), load_yolo(yolo_path)
        src = {"detr": str(detr_path), "yolo": str(yolo_path), "seed": seed}
        print(f"DETR: {detr_path} ({len(detr)} klatek)")
        print(f"YOLO: {yolo_path} ({len(yolo)} klatek)")

    common = sorted(set(detr) & set(yolo))
    if len(common) < 3:
        raise SystemExit(f"Za malo wspolnych klatek: {len(common)} — nie ma czego korelowac.")
    x = np.array([detr[k] for k in common])
    y = np.array([yolo[k] for k in common])
    print(f"Wspolne klatki: {len(common)} (DETR {len(detr)}, YOLO {len(yolo)})")

    corr = C.correlations(x, y)
    corr["source"] = src
    only_detr, only_yolo = len(detr) - len(common), len(yolo) - len(common)
    if only_detr or only_yolo:
        corr["frames_only_detr"] = only_detr
        corr["frames_only_yolo"] = only_yolo

    n_scatter = int(el2n_cfg.get("scatter_sample", 2000))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(common), size=min(n_scatter, len(common)), replace=False)
    corr["scatter"] = {
        "xlabel": "EL2N (DETR)", "ylabel": "EL2N (YOLO)",
        "points": [{"frame_id": common[i], "detr": float(x[i]), "yolo": float(y[i])}
                   for i in sorted(idx)],
    }

    print(f"\nSpearman rho = {corr['spearman_r']:.4f}"
          + (f" (p={corr['spearman_p']:.3g})" if corr["spearman_p"] is not None
             else " (p-value niedostepne — brak scipy)"))
    print(f"Pearson  r  = {corr['pearson_r']:.4f}"
          + (f" (p={corr['pearson_p']:.3g})" if corr["pearson_p"] is not None else ""))
    rho = corr["spearman_r"]
    if rho >= 0.7:
        verdict = "WYSOKA — miara opisuje trudnosc klatki, nie preferencje architektury DETR."
    elif rho >= 0.4:
        verdict = "UMIARKOWANA (zakres 0.4-0.7 oczekiwany wg PLAN.md YOLO) — wspolny rdzen trudnosci istnieje."
    else:
        verdict = "NISKA — ranking trudnosci zalezy od rodziny detektora; ujawnic w Limitations."
    print(f"Interpretacja: {verdict}")
    corr["interpretation"] = verdict

    out = args.out or C.cfg_path(acfg, "out_dir") / "el2n_rank_correlation.json"
    C.write_json(out, corr)


if __name__ == "__main__":
    main()
