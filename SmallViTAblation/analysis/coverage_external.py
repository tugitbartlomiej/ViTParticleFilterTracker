#!/usr/bin/env python3
"""
Pokrycie rozkladu ZEWNETRZNEGO przez wybrane podzbiory treningowe (Agent 5, zadanie 1).

Mierzy wlasciwa rzecz: nie rozrzut wewnatrz wlasnego rozkladu, a srednia
odleglosc kazdego obrazu ZEWNETRZNEGO do NAJBLIZSZEGO obrazu z wybranego
podzbioru treningowego, plus rozklad tych odleglosci (percentyle, histogram).
Na koniec: korelacja miary pokrycia z wynikiem na external (jesli wyniki istnieja).

DYSCYPLINA ZERO-LOOK: liczymy TYLKO OSADZENIA obrazow zewnetrznych (DINO albo
cechy zastepcze z pikseli). Nie uzywamy etykiet, checkpointow ani zadnych wynikow
modelu na zbiorze zewnetrznym do obliczenia pokrycia — wiec nie wplywa to na
wybor epoki ani progu. Szczegoly: analysis/README.md.

Uzycie:
    py -3.11 analysis/coverage_external.py --external-manifest sciezka.jsonl --external-root katalog
    py -3.11 analysis/coverage_external.py --external-dir katalog_obrazow --fake-embeddings
    py -3.11 analysis/coverage_external.py ... --embeddings-cache analysis/out/emb_cache

Opcje:
    --fake-embeddings      deterministyczne cechy zastepcze z pikseli (testy, bez torch)
    --embeddings-cache KAT licz osadzenia raz i trzymaj w plikach .npz
    --max-external N       limit obrazow zewnetrznych (testy)
    --max-variant N        limit obrazow podzbioru treningowego (testy)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _common as C


def collect_external(args, acfg) -> list[tuple[str, Path]]:
    root = None
    if args.external_manifest:
        manifest = Path(args.external_manifest)
        recs = C.load_jsonl(manifest)
        root = Path(args.external_root) if args.external_root else manifest.parent
        entries = []
        for r in recs:
            key = r.get("frame_id") or r.get("path")
            entries.append((str(key), root / r["path"]))
    elif args.external_dir:
        root = Path(args.external_dir)
        entries = [(str(p.relative_to(root)), p)
                   for p in sorted(root.rglob("*")) if p.suffix.lower() in C.IMAGE_EXTS]
    else:
        dflt = C.cfg_path(acfg, "external", "manifest")
        if dflt is None or not dflt.exists():
            raise SystemExit("Podaj --external-manifest albo --external-dir "
                             "(analysis.external.manifest w protocol.yaml jest pusty — "
                             "zbior zewnetrzny jeszcze nie istnieje).")
        root = C.cfg_path(acfg, "external", "image_root") or dflt.parent
        entries = [(str(r.get("frame_id") or r["path"]), root / r["path"])
                   for r in C.load_jsonl(dflt)]
    if args.max_external:
        entries = entries[: args.max_external]
    if not entries:
        raise SystemExit("Zbior zewnetrzny: 0 obrazow — nie ma czego liczyc.")
    return entries


def collect_variants(args, acfg, protocol) -> list[tuple[str, list[tuple[str, Path]]]]:
    variants_dir = Path(args.variants_dir) if args.variants_dir else C.cfg_path(acfg, "variants_dir")
    image_root = Path(args.image_root) if args.image_root else C.cfg_path(acfg, "image_root")
    pool_path = Path(args.pool_manifest) if args.pool_manifest else C.cfg_path(acfg, "pool_manifest")

    if not variants_dir or not variants_dir.exists():
        print(f"UWAGA: brak katalogu wariantow ({variants_dir}) — nie ma czego analizowac.")
        return []
    if not pool_path or not pool_path.exists():
        raise SystemExit(f"Brak pool_manifest ({pool_path}) — potrzebny do mapowania frame_id -> path.")

    pool = {r["frame_id"]: r for r in C.load_pool_manifest(pool_path)}
    out = []
    for run_id, data in C.iter_variant_files(variants_dir):
        frame_ids = data.get("frame_ids", [])
        entries, missing = [], 0
        for fid in frame_ids:
            rec = pool.get(fid)
            if rec is None:
                missing += 1
                continue
            entries.append((fid, image_root / rec["path"]))
        if missing:
            print(f"  UWAGA [{run_id}]: {missing} frame_id spoza pool_manifest — pominieto")
        if args.max_variant:
            entries = entries[: args.max_variant]
        if entries:
            out.append((run_id, entries))
    return out


def dist_stats(d: np.ndarray, acfg) -> dict:
    cov = acfg.get("coverage", {})
    pcts = [int(p) for p in cov.get("percentiles", [50, 90, 95, 99])]
    bins = int(cov.get("histogram_bins", 20))
    counts, edges = np.histogram(d, bins=bins)
    return {
        "n_external": int(len(d)),
        "mean_min_dist": float(d.mean()),
        "std_min_dist": float(d.std()),
        "min": float(d.min()),
        "max": float(d.max()),
        "percentiles": {f"p{p}": float(np.percentile(d, p)) for p in pcts},
        "histogram": {"bin_edges": [float(x) for x in edges],
                      "counts": [int(x) for x in counts]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external-manifest", type=Path, default=None,
                    help="jsonl z obrazami zbioru zewnetrznego (frame_id/path)")
    ap.add_argument("--external-dir", type=Path, default=None,
                    help="alternatywnie: katalog z obrazami zewnetrznymi")
    ap.add_argument("--external-root", type=Path, default=None,
                    help="katalog bazowy dla wzglednych 'path' w manifeście")
    ap.add_argument("--pool-manifest", type=Path, default=None)
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--variants-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--embeddings-cache", type=Path, default=None,
                    help="katalog na .npz z osadzeniami (liczone raz)")
    ap.add_argument("--fake-embeddings", action="store_true",
                    help="deterministyczne cechy zastepcze z pikseli zamiast DINO")
    ap.add_argument("--max-external", type=int, default=None)
    ap.add_argument("--max-variant", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    protocol = C.load_protocol(args.protocol)
    acfg = C.analysis_cfg(protocol)
    cov = acfg.get("coverage", {})
    distance = cov.get("distance", "cosine")
    mode = "fake" if args.fake_embeddings else "dino"

    print(f"Tryb osadzen: {mode} | odleglosc: {distance}")
    ext_entries = collect_external(args, acfg)
    print(f"Zbior zewnetrzny: {len(ext_entries)} obrazow")

    cache = Path(args.embeddings_cache) if args.embeddings_cache else None
    ext_keys, ext_emb = C.compute_embeddings(
        ext_entries, acfg, mode, cache / "external.npz" if cache else None)

    variants = collect_variants(args, acfg, protocol)
    results_dir = Path(args.results_dir) if args.results_dir else C.cfg_path(acfg, "results_dir")

    report = {
        "mode": mode,
        "distance": distance,
        "n_external": len(ext_entries),
        "variants": {},
        "correlation": None,
    }

    pairs_cov, pairs_map, pair_ids = [], [], []
    for run_id, entries in variants:
        cache_file = cache / f"variant_{run_id}.npz" if cache else None
        _, emb = C.compute_embeddings(entries, acfg, mode, cache_file)
        d = C.min_distances(ext_emb, emb, distance)
        st = dist_stats(d, acfg)
        st["n_train_images"] = len(entries)

        metric, block = None, None
        mf = results_dir / run_id / "metrics.json" if results_dir else None
        if mf and mf.exists():
            metrics = json.loads(mf.read_text(encoding="utf-8"))
            metric, block = C.external_metric_from(metrics)
        st["external_metric"] = {"mAP_0.5": metric, "block": block} if metric is not None else None
        report["variants"][run_id] = st

        if metric is not None:
            pairs_cov.append(st["mean_min_dist"])
            pairs_map.append(metric)
            pair_ids.append(run_id)

        print(f"\n=== {run_id}  (n_train={st['n_train_images']}, n_ext={st['n_external']})")
        print(f"    mean_min_dist = {st['mean_min_dist']:.4f}  (std {st['std_min_dist']:.4f})")
        print("    percentyle: " + "  ".join(f"{k}={v:.4f}" for k, v in st["percentiles"].items()))
        if metric is not None:
            print(f"    external mAP_0.5 ({block}) = {metric:.4f}")

    if len(pairs_cov) >= 2:
        corr = C.correlations(pairs_cov, pairs_map)
        corr["run_ids"] = pair_ids
        corr["note"] = ("korelacja mean_min_dist (pokrycie) vs mAP_0.5 na external; "
                        "oczekiwany znak ujemny: lepsze pokrycie = mniejsza odleglosc = wyzszy mAP")
        report["correlation"] = corr
        print(f"\n=== Korelacja pokrycie vs wynik external (n={corr['n']})")
        print(f"    spearman r = {corr['spearman_r']:.4f} (p={corr['spearman_p']})")
        print(f"    pearson  r = {corr['pearson_r']:.4f} (p={corr['pearson_p']})")
        min_req = int(cov.get("min_variants_for_correlation", 8))
        if corr["n"] < min_req:
            warn = (f"OSTRZEZENIE O MOCY TESTU: n={corr['n']} < {min_req} wariantow — "
                    f"korelacja ma znikoma moc statystyczna, traktuj jako wskazowke, nie dowod.")
            print("    " + warn)
            corr["power_warning"] = warn
    else:
        print(f"\nKorelacja: potrzeba >=2 wariantow z metrics.json (jest {len(pairs_cov)}).")

    out = args.out or C.cfg_path(acfg, "out_dir") / "coverage_external.json"
    C.write_json(out, report)


if __name__ == "__main__":
    main()
