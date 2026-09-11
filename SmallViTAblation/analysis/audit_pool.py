#!/usr/bin/env python3
"""
Audyt wyciekow miedzy splitami — rozszerzenie tools/audit_leakage.py (Agent 5, zadanie 4).

Cztery poziomy naraz, bo kazdy lapie inny bledy:
  1. PACJENT        — ta sama osoba w dwoch czesciach podzialu,
  2. KLATKA ZRODLOWA — ta sama klatka pod roznymi wariantami augmentacji
                      (rozpoznawanie przez zrzucenie przyrostka _aug_[typ_]N),
  3. PLIK           — identyczna sciezka (opcjonalnie identyczny hash zawartosci),
  4. PODOBIENSTWO   — NOWOSC: ten sam material pod inna nazwa, miedzy zbiorami
                      publicznymi; perceptual dhash zaimplementowany w numpy/PIL
                      (bez ciezkich zaleznosci) + prog Hamminga z protocol.yaml.

Wejscie: pool_manifest.jsonl (kontrakt agents/README 4.1) ALBO katalogi COCO
z annotations_{train,val,test}.json (jak w tools/audit_leakage.py).

Kazdy NOWY zbior musi przejsc ten audyt PRZED wejsciem do puli:
    py -3.11 analysis/audit_pool.py --pool-manifest X --fail-on-leak   # exit != 0 przy wycieku
    py -3.11 analysis/audit_pool.py katalog_coco/ --fail-on-leak
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

import _common as C

AUG_SUFFIX = re.compile(r"_aug_(?:[A-Za-z][A-Za-z0-9]*_)?\d+$")
MAX_EXAMPLES = 12


def patient_of(rec: dict) -> str:
    if rec.get("patient_id"):
        return str(rec["patient_id"])
    stem = C.stem_of(str(rec.get("frame_id", "")))
    if "_frame_" in stem:
        return stem.split("_frame_")[0]
    parts = str(rec.get("frame_id", "")).split("/")
    return parts[1] if len(parts) > 1 else stem


def source_of(name: str) -> str:
    return AUG_SUFFIX.sub("", C.stem_of(name))


def collect_from_manifest(path: Path) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    for rec in C.load_pool_manifest(path):
        splits.setdefault(str(rec["split"]), []).append(rec)
    return splits


def collect_from_coco_dir(d: Path) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    for s in ("train", "val", "test"):
        f = d / f"annotations_{s}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        splits[s] = [{"frame_id": im["file_name"], "path": im["file_name"],
                      "patient_id": None, "split": s} for im in data["images"]]
    return splits


def file_hash(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def phash_level(splits: dict[str, list[dict]], image_root: Path | None,
                acfg: dict) -> dict:
    """Poziom 4: dhash per obraz, pary miedzysplitowe o hammingu <= prog."""
    aud = acfg.get("audit", {})
    hash_size = int(aud.get("dhash_size", 8))
    threshold = int(aud.get("hamming_threshold", 5))
    cap = int(aud.get("max_images_for_phash", 20000))

    per_split: dict[str, tuple[list[dict], np.ndarray]] = {}
    for s, recs in splits.items():
        recs = sorted(recs, key=lambda r: str(r.get("path", r["frame_id"])))
        if len(recs) > cap:
            stride = len(recs) / cap
            recs = [recs[int(i * stride)] for i in range(cap)]
        hashes, kept = [], []
        for r in recs:
            if image_root is None:
                break
            p = Path(r["path"])
            p = p if p.is_absolute() else image_root / p
            h = C.dhash(p, hash_size)
            if h is not None:
                hashes.append(h)
                kept.append(r)
        if hashes:
            per_split[s] = (kept, np.stack(hashes))

    result = {"threshold": threshold, "hash_size": hash_size,
              "hashed_per_split": {s: len(v[0]) for s, v in per_split.items()},
              "pairs": [], "n_pairs": 0}
    if image_root is None:
        result["skipped"] = "brak image_root — nie wczytano pikseli"
        return result
    if len(per_split) < 2:
        result["skipped"] = "obrazy dostepne dla <2 splitow"
        return result

    for a, b in combinations(sorted(per_split), 2):
        ra, ha = per_split[a]
        rb, hb = per_split[b]
        dist = C.hamming_matrix(ha, hb)
        ii, jj = np.nonzero(dist <= threshold)
        for i, j in zip(ii, jj):
            result["pairs"].append({
                "splits": f"{a}-{b}", "hamming": int(dist[i, j]),
                "a": str(ra[i].get("path", ra[i]["frame_id"])),
                "b": str(rb[j].get("path", rb[j]["frame_id"])),
            })
        result["n_pairs"] += len(ii)
    result["pairs"] = result["pairs"][: MAX_EXAMPLES * 3]
    return result


def audit(splits: dict[str, list[dict]], image_root: Path | None, acfg: dict,
          hash_files: bool) -> dict:
    if len(splits) < 2:
        return {"error": f"mniej niz dwa splity ({sorted(splits)}) — nie ma czego porownywac"}

    res = {"counts": {}, "levels": {}, "verdict": "CZYSTY"}
    index = {}
    for s, recs in splits.items():
        pats = {patient_of(r) for r in recs}
        srcs = {source_of(r["frame_id"]) for r in recs}
        files = {str(r.get("path", r["frame_id"])) for r in recs}
        hashes = set()
        if hash_files and image_root is not None:
            for r in recs:
                p = Path(r.get("path", ""))
                p = p if p.is_absolute() else image_root / p
                h = file_hash(p)
                if h:
                    hashes.add(h)
        index[s] = {"patients": pats, "sources": srcs, "files": files, "hashes": hashes}
        res["counts"][s] = {"obrazy": len(recs), "klatki_zrodlowe": len(srcs),
                            "pacjenci": len(pats)}

    leaks = []
    for a, b in combinations(sorted(index), 2):
        ia, ib = index[a], index[b]
        pair = {
            "pacjenci": sorted(ia["patients"] & ib["patients"]),
            "klatki_zrodlowe": sorted(ia["sources"] & ib["sources"]),
            "pliki": sorted(ia["files"] & ib["files"]),
            "hashe_plikow": sorted(ia["hashes"] & ib["hashes"]),
        }
        res["levels"][f"{a}-{b}"] = {
            "pacjenci_wspolni": len(pair["pacjenci"]),
            "pacjenci_przyklady": pair["pacjenci"][:MAX_EXAMPLES],
            "klatki_zrodlowe_wspolne": len(pair["klatki_zrodlowe"]),
            "klatki_zrodlowe_przyklady": pair["klatki_zrodlowe"][:MAX_EXAMPLES],
            "pliki_wspolne": len(pair["pliki"]),
            "pliki_przyklady": pair["pliki"][:MAX_EXAMPLES],
            "hashe_wspolne": len(pair["hashe_plikow"]),
        }
        if pair["pacjenci"] or pair["klatki_zrodlowe"] or pair["pliki"] or pair["hashe_plikow"]:
            leaks.append(f"{a}-{b}: pacjent/zrodlo/plik")

    res["poziom4_podobienstwo"] = phash_level(splits, image_root, acfg)
    if res["poziom4_podobienstwo"].get("n_pairs", 0) > 0:
        leaks.append(f"podobne obrazy miedzysplitowo: {res['poziom4_podobienstwo']['n_pairs']} par")

    if leaks:
        res["verdict"] = "WYCIEK"
        res["leaks"] = leaks
    return res


def report(r: dict, label: str) -> None:
    print(f"\n=== {label}")
    if "error" in r:
        print(f"    {r['error']}")
        return
    for s, c in r["counts"].items():
        print(f"    {s:5} obrazy={c['obrazy']:7}  klatki_zrodlowe={c['klatki_zrodlowe']:7}"
              f"  pacjenci={c['pacjenci']:4}")
    for pair, lv in r["levels"].items():
        bad = lv["pacjenci_wspolni"] or lv["klatki_zrodlowe_wspolne"] or lv["pliki_wspolne"] \
            or lv["hashe_wspolne"]
        print(f"    {'!!' if bad else 'OK'} {pair:12} pacjenci={lv['pacjenci_wspolni']}"
              f"  klatki_zrodlowe={lv['klatki_zrodlowe_wspolne']}"
              f"  pliki={lv['pliki_wspolne']}  hashe={lv['hashe_wspolne']}")
        for key, lbl in (("pacjenci_przyklady", "pacjenci"),
                         ("klatki_zrodlowe_przyklady", "klatki zrodlowe"),
                         ("pliki_przyklady", "pliki")):
            if lv[key]:
                print(f"        {lbl}: {', '.join(map(str, lv[key][:MAX_EXAMPLES]))}")
    p4 = r["poziom4_podobienstwo"]
    if "skipped" in p4:
        print(f"    -- poziom 4 (podobienstwo dhash): pominiety — {p4['skipped']}")
    else:
        print(f"    {'!!' if p4['n_pairs'] else 'OK'} poziom 4 (dhash, prog {p4['threshold']}): "
              f"{p4['n_pairs']} podejrzanych par miedzysplitowych "
              f"(zhashowano: {p4['hashed_per_split']})")
        for pr in p4["pairs"][:MAX_EXAMPLES]:
            print(f"        [{pr['splits']}] h={pr['hamming']}: {pr['a']}  <->  {pr['b']}")
    print(f"    WERDYKT: {r['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=Path,
                    help="katalogi COCO z annotations_{train,val,test}.json")
    ap.add_argument("--pool-manifest", type=Path, default=None,
                    help="pool_manifest.jsonl (domyslnie z protocol.yaml)")
    ap.add_argument("--image-root", type=Path, default=None,
                    help="katalog bazowy dla 'path' (domyslnie z protocol.yaml)")
    ap.add_argument("--hash-files", action="store_true",
                    help="poziom 3 dodatkowo po hashu zawartosci (sha256), nie tylko sciezce")
    ap.add_argument("--skip-phash", action="store_true", help="pomin poziom 4 (dhash)")
    ap.add_argument("--fail-on-leak", action="store_true",
                    help="exit code 1 przy werdykcie WYCIEK (brama wejscia nowego zbioru do puli)")
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    protocol = C.load_protocol(args.protocol)
    acfg = C.analysis_cfg(protocol)

    jobs: list[tuple[str, dict[str, list[dict]], Path | None]] = []
    if args.pool_manifest or (not args.dirs and C.cfg_path(acfg, "pool_manifest")
                              and C.cfg_path(acfg, "pool_manifest").exists()):
        pm = Path(args.pool_manifest) if args.pool_manifest else C.cfg_path(acfg, "pool_manifest")
        img_root = args.image_root or C.cfg_path(acfg, "image_root")
        jobs.append((str(pm), collect_from_manifest(pm), img_root))
    for d in args.dirs:
        jobs.append((str(d), collect_from_coco_dir(d), d))

    if not jobs:
        raise SystemExit("Podaj --pool-manifest albo katalogi COCO "
                         "(pool_manifest z protocol.yaml nie istnieje).")

    all_reports, verdicts = {}, []
    for label, splits, img_root in jobs:
        if args.skip_phash:
            img_root = None
        r = audit(splits, img_root, acfg, args.hash_files)
        report(r, label)
        all_reports[label] = r
        verdicts.append(r.get("verdict", "BLAD"))

    bad = [v for v in verdicts if v not in ("CZYSTY",)]
    print("\n" + "=" * 64)
    print(f"Sprawdzono: {len(verdicts)} | czystych: {len(verdicts) - len(bad)} | problem: {len(bad)}")

    out = args.out or C.cfg_path(acfg, "out_dir") / "audit_pool.json"
    C.write_json(out, {"verdicts": dict(zip((j[0] for j in jobs), verdicts)),
                       "reports": all_reports})

    if args.fail_on_leak and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
