#!/usr/bin/env python3
"""
Lista kontrolna Agent 5 (zadanie 5): czy kazdy zarzut recenzenta ma artefakt na dysku.

Wejscie:  docs/04_REVIEWER_MAP.md (tabela zarzut -> co go zamyka) + configs/*.yaml
Wyjscie:  raport JSON (analysis/out/checklist.json) + tabela w konsoli.
Stany:    ZAMKNIETE / W TOKU / BRAK.  Exit code 0 tylko gdy BRAK zer pozycji "brak".

Uruchamiac PRZED KAZDYM zgloszeniem artykułu:
    py -3.11 analysis/checklist.py

Pozycje czysto redakcyjne (tekst artykułu, bez artefaktu dyskowego) zamyka autor
recznie przez analysis/out/manual_confirmations.json = {"confirmed": {"r2_17": "poprawiono w v2"}}.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _common as C

STATES = ("brak", "w toku", "zamkniete")
MAP_ROOT = C.ROOT / "docs" / "04_REVIEWER_MAP.md"


def worse(a: str, b: str) -> str:
    return a if STATES.index(a) < STATES.index(b) else b


# ---------------------------------------------------------------- parsowanie mapy

def parse_reviewer_map(path: Path) -> list[dict]:
    rows, reviewer = [], None
    occurrence: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Recenzent"):
            m = re.search(r"Recenzent\s+(\d)", line)
            reviewer = int(m.group(1)) if m else None
            occurrence = {}
            continue
        if line.startswith("## "):
            reviewer = None
            continue
        if reviewer is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("#", "") or set(cells[0]) <= {"-", ":"}:
            continue
        label = cells[0].replace("\u2013", "-").replace("\u2014", "-")
        occurrence[label] = occurrence.get(label, 0) + 1
        key = f"r{reviewer}_{label}"
        if label == "drobne":
            key = f"r{reviewer}_drobne{occurrence[label]}"
        rows.append({"key": key, "reviewer": reviewer, "label": label,
                     "zarzut": cells[1], "closer": cells[2],
                     "where": cells[3] if len(cells) > 3 else ""})
    return rows


# ---------------------------------------------------------------- skanowanie wynikow

def scan_results(results_dir: Path | None) -> dict:
    out = {"exists": results_dir is not None and results_dir.exists(),
           "runs": {}, "run_ids": []}
    if not out["exists"]:
        return out
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        run_id = d.name
        out["run_ids"].append(run_id)
        out["runs"][run_id] = {
            "metrics": (d / "metrics.json").exists(),
            "manifest": (d / "run_manifest.json").exists(),
            "dir": d,
        }
    return out


def matrix_variant_ids(variants: dict) -> list[str]:
    return [v["id"] for v in variants.get("matrix", [])]


def expected_sensitivity_ids(protocol: dict, variants: dict) -> list[str]:
    sens = variants["sensitivity"]
    budget_a = protocol["budgets"]["tier_a"]
    seed = sens["seeds"][0]
    ids = []
    for study in sens["studies"]:
        for val in study["values"]:
            for target in study.get("applies_to", [None]):
                sid = f"{study['id']}_{study['param']}_{val}" + (f"_{target}" if target else "")
                sid = sid.replace("/", "-").replace(" ", "")
                ids.append(f"{sid}__s{seed}__b{budget_a}")
    return ids


# ---------------------------------------------------------------- sprawdzenia

def check_seeds(protocol, variants, res, n_results) -> tuple[str, str]:
    need = protocol["seeds"]["matrix"]
    seed_re = re.compile(r"__s(\d+)__")
    per_variant = {}
    for vid in matrix_variant_ids(variants):
        seeds = set()
        for rid, r in res["runs"].items():
            if rid.split("__")[0] == vid and r["metrics"]:
                m = seed_re.search(rid)
                if m and int(m.group(1)) in need:
                    seeds.add(int(m.group(1)))
        per_variant[vid] = seeds
    full = [v for v, s in per_variant.items() if len(s) >= len(need)]
    any_ = [v for v, s in per_variant.items() if s]
    detail = (f"ziarna {need}: {len(full)}/{len(per_variant)} wariantow z kompletem; "
              f"jakiekolwiek wyniki: {len(any_)}/{len(per_variant)}")
    if len(full) == len(per_variant) and per_variant:
        return "zamkniete", detail
    return ("w toku" if any_ else "brak"), detail


def check_matrix(protocol, variants, res, n_results) -> tuple[str, str]:
    ids = matrix_variant_ids(variants)
    have = {vid for vid in ids
            if any(r["metrics"] and rid.split("__")[0] == vid for rid, r in res["runs"].items())}
    detail = f"macierz {len(ids)} kombinacji: wyniki dla {len(have)}/{len(ids)}"
    if have == set(ids) and ids:
        return "zamkniete", detail
    return ("w toku" if have else "brak"), detail


def check_sensitivity(protocol, variants, res, n_results) -> tuple[str, str]:
    exp = expected_sensitivity_ids(protocol, variants)
    have = {e for e in exp if res["runs"].get(e, {}).get("metrics")}
    detail = f"perturbacje S-*: {len(have)}/{len(exp)} biegow z metrics.json"
    if have == set(exp) and exp:
        return "zamkniete", detail
    return ("w toku" if have else "brak"), detail


def check_sensitivity_subset(protocol, variants, res, study_ids) -> tuple[str, str]:
    exp = [e for e in expected_sensitivity_ids(protocol, variants)
           if e.split("_")[0] + "_" + e.split("_")[1] in study_ids
           or any(e.startswith(s + "_") for s in study_ids)]
    have = {e for e in exp if res["runs"].get(e, {}).get("metrics")}
    detail = f"studia {study_ids}: {len(have)}/{len(exp)} biegow"
    if have == set(exp) and exp:
        return "w toku", detail + " (eksperymenty sa; czesc redakcyjna wymaga potwierdzenia recznego)"
    return ("w toku" if have else "brak"), detail


def check_ledger(protocol, variants, res, n_results) -> tuple[str, str]:
    acfg = C.analysis_cfg(protocol)
    p = C.resolve(protocol["evaluation"]["zero_look"]["ledger"])
    max_entries = int(acfg["checklist"]["ledger_max_entries"])
    if not p.exists():
        return "brak", f"brak {p.name} (zewnetrzny jeszcze nie liczony albo rejestr nieprowadzony)"
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad = 0
    for ln in lines:
        try:
            json.loads(ln)
        except json.JSONDecodeError:
            bad += 1
    n = len(lines)
    if bad:
        return "brak", f"{bad}/{n} wierszy nie jest poprawnym JSON"
    if n == 0:
        return "brak", "rejestr pusty — zero-look wymaga >=1 udokumentowanego dostepu po ewaluacji finalnej"
    if n > max_entries:
        return "brak", f"{n} wpisow > {max_entries} — wyglada na wielokrotne ogladanie external"
    return "zamkniete", f"{n} wpisow (limit {max_entries}), wszystkie poprawne"


def check_metrics_fields(protocol, variants, res, n_results) -> tuple[str, str]:
    acfg = C.analysis_cfg(protocol)
    ck = acfg["checklist"]
    top = set(ck["metrics_top_fields"])
    blocks = set(ck["metrics_result_blocks"])
    fields = set(ck["metrics_block_fields"])
    files = [(rid, r["dir"] / "metrics.json") for rid, r in res["runs"].items() if r["metrics"]]
    if not files:
        return "brak", "nie ma zadnego metrics.json"
    ok, sigs, problems = 0, set(), []
    for rid, f in files:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append(f"{rid}: niepoprawny JSON")
            continue
        errs = []
        if not top.issubset(m.keys()):
            errs.append(f"brak pol {sorted(top - set(m))}")
        resd = m.get("results", {})
        if not blocks.issubset(resd.keys()):
            errs.append(f"brak blokow {sorted(blocks - set(resd))}")
        sig = (tuple(sorted(m.keys())),
               tuple(sorted((b, tuple(sorted(resd.get(b, {}).keys()))) for b in blocks)))
        sigs.add(sig)
        for b in blocks & set(resd):
            if not fields.issubset(set(resd[b])):
                errs.append(f"blok {b}: brak {sorted(fields - set(resd[b]))}")
        if m.get("threshold_selected_on") not in (None, "val"):
            errs.append(f"threshold_selected_on={m['threshold_selected_on']} != val")
        if errs:
            problems.append(f"{rid}: " + "; ".join(errs))
        else:
            ok += 1
    uniform = len(sigs) <= 1
    detail = f"{ok}/{len(files)} metrics.json zgodnych z kontraktem 4.5; identyczny komplet pol: {uniform}"
    if problems:
        detail += " | problemy: " + "; ".join(problems[:3])
    if ok == len(files) and uniform:
        return "zamkniete", detail
    return ("w toku" if ok else "brak"), detail


def check_run_manifests(protocol, variants, res, n_results) -> tuple[str, str]:
    if not res["run_ids"]:
        return "brak", "brak wynikow — nie ma czego sprawdzac"
    missing = [rid for rid, r in res["runs"].items() if not r["manifest"]]
    shas = set()
    for rid, r in res["runs"].items():
        f = r["dir"] / "run_manifest.json"
        if f.exists():
            try:
                shas.add(json.loads(f.read_text(encoding="utf-8")).get("protocol_sha256"))
            except json.JSONDecodeError:
                pass
    detail = (f"run_manifest.json: {len(res['run_ids']) - len(missing)}/{len(res['run_ids'])}; "
              f"rozne protocol_sha256: {len(shas - {None})}")
    if missing:
        detail += f" | brak w: {', '.join(missing[:5])}"
    if not missing and len(shas - {None}) <= 1 and shas - {None}:
        return "zamkniete", detail
    if not missing:
        return "w toku", detail + " | manifesty bez hashu protokolu albo wiele wersji protokolu"
    return ("w toku" if len(missing) < len(res["run_ids"]) else "brak"), detail


def check_baselines(protocol, variants, res, n_results) -> tuple[str, str]:
    parts, all_ok, any_ok = [], True, False
    for b in variants.get("baselines", []):
        need = int(b.get("draws", 1))
        have = sum(1 for rid, r in res["runs"].items()
                   if rid.startswith(b["id"] + "__") and r["metrics"])
        parts.append(f"{b['id']}: {have}/{need}")
        all_ok &= have >= need
        any_ok |= have > 0
    detail = ", ".join(parts)
    if all_ok and parts:
        return "zamkniete", detail
    return ("w toku" if any_ok else "brak"), detail


def check_size_sweep(protocol, variants, res, n_results) -> tuple[str, str]:
    sw = variants.get("size_sweep", {})
    exp = [f"{v}__s{s}__b{b}" for v in sw.get("variants", [])
           for b in sw.get("budgets", []) for s in sw.get("seeds", [])]
    have = {e for e in exp if res["runs"].get(e, {}).get("metrics")}
    detail = f"sweep budzetu: {len(have)}/{len(exp)} biegow"
    if have == set(exp) and exp:
        return "zamkniete", detail
    return ("w toku" if have else "brak"), detail


def check_transfer(protocol, variants, res, n_results) -> tuple[str, str]:
    archs = variants.get("tier_b", {}).get("architectures", {}).get("transfer", [])
    if not archs:
        return "brak", "brak zdefiniowanych architektur przeniesienia w variants.yaml"
    parts, all_ok, any_ok = [], True, False
    for a in archs:
        have = sum(1 for rid, r in res["runs"].items()
                   if rid.endswith("__" + a) and r["metrics"])
        parts.append(f"{a}: {have}")
        all_ok &= have > 0
        any_ok |= have > 0
    detail = "przeniesienie architektur (biegi z metrics.json): " + ", ".join(parts)
    if all_ok:
        return "zamkniete", detail
    return ("w toku" if any_ok else "brak"), detail


def check_iso_resource(protocol, variants, res, n_results) -> tuple[str, str]:
    archs = variants.get("tier_b", {}).get("architectures", {}).get("transfer", [])
    epochs = protocol["training"]["epochs"]
    reso = protocol["model"]["resolution"]
    checked, ok = 0, 0
    for rid, r in res["runs"].items():
        if not any(rid.endswith("__" + a) for a in archs) or not r["manifest"]:
            continue
        try:
            m = json.loads((r["dir"] / "run_manifest.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        checked += 1
        if m.get("epochs_completed") == epochs and m.get("resolution") == reso:
            ok += 1
    if checked == 0:
        return "brak", "brak biegow przeniesienia z manifestami — iso-resource niesprawdzalne"
    detail = f"iso-resource: {ok}/{checked} biegow transferowych: epochs={epochs}, resolution={reso}"
    if ok == checked:
        return "zamkniete", detail
    return "w toku", detail


def check_pool_split(protocol, variants, res, n_results) -> tuple[str, str]:
    acfg = C.analysis_cfg(protocol)
    p = C.cfg_path(acfg, "pool_manifest")
    want = protocol["split"]["videos"]
    if not p or not p.exists():
        return "brak", f"brak pool_manifest ({p})"
    recs = C.load_pool_manifest(p)
    per = {}
    for r in recs:
        pid = r.get("patient_id") or r.get("video_id")
        per.setdefault(str(r["split"]), set()).add(str(pid))
    got = {s: len(v) for s, v in per.items()}
    exp = {s: want[s] for s in ("train", "val", "test") if s in want}
    ok = all(got.get(s) == n for s, n in exp.items())
    detail = f"pacjenci per split: {got} (oczekiwano {exp})"
    if ok:
        return "zamkniete", detail + " — podzial po pacjencie, jeden plik dla wszystkich biegow"
    return "brak", detail


def check_inference_config(protocol, variants, res, n_results) -> tuple[str, str]:
    inf = protocol["model"].get("inference")
    if inf == "standard_topk":
        return "zamkniete", "protocol.yaml model.inference=standard_topk (ZERO Query 81); egzekwowanie w kodzie — Agent 3"
    return "brak", f"protocol.yaml model.inference={inf} — oczekiwano standard_topk"


def check_dataset_table(protocol, variants, res, n_results) -> tuple[str, str]:
    acfg = C.analysis_cfg(protocol)
    p = C.cfg_path(acfg, "out_dir") / "dataset_table.json"
    if not p.exists():
        return "brak", "brak analysis/out/dataset_table.json (uruchom dataset_table.py --verify)"
    data = json.loads(p.read_text(encoding="utf-8"))
    v = data.get("verify")
    if v and v.get("all_match"):
        return "zamkniete", "tabela podzialu wygenerowana i zweryfikowana z oczekiwanymi liczbami"
    if v:
        return "w toku", "tabela istnieje, ale --verify nie przechodzi (niezgodnosci: " + \
            ", ".join(v.get("mismatches", [])[:4]) + ")"
    return "w toku", "tabela istnieje bez weryfikacji — uruchom z --verify"


def check_aug_mix(protocol, variants, res, n_results) -> tuple[str, str]:
    acfg = C.analysis_cfg(protocol)
    p = C.cfg_path(acfg, "out_dir") / "augmentation_mix.json"
    if not p.exists():
        return "brak", "brak analysis/out/augmentation_mix.json (uruchom augmentation_mix.py)"
    data = json.loads(p.read_text(encoding="utf-8"))
    nv = len(data.get("variants", {}))
    return "zamkniete", f"analiza skadu augmentacji istnieje ({nv} wariantow przeanalizowanych)"


def check_nms(protocol, variants, res, n_results) -> tuple[str, str]:
    acfg = C.analysis_cfg(protocol)
    out_dir = C.cfg_path(acfg, "out_dir")
    hits = list(out_dir.glob("*nms*")) if out_dir and out_dir.exists() else []
    if hits:
        return "zamkniete", f"analiza NMS: {hits[0].name}"
    return "brak", "brak artefaktu analizy NMS (docs/06 T14)"


CHECKS = {
    "seeds": check_seeds,
    "matrix": check_matrix,
    "sensitivity": check_sensitivity,
    "sens_pca_alg": lambda p, v, r, n: check_sensitivity_subset(p, v, r, n, ("S-PCA", "S-ALG")),
    "ledger": check_ledger,
    "metrics_fields": check_metrics_fields,
    "run_manifests": check_run_manifests,
    "baselines": check_baselines,
    "size_sweep": check_size_sweep,
    "transfer": check_transfer,
    "iso_resource": check_iso_resource,
    "pool_split": check_pool_split,
    "inference_config": check_inference_config,
    "dataset_table": check_dataset_table,
    "aug_mix": check_aug_mix,
    "nms": check_nms,
}

# (reviewer, label) -> lista sprawdzen; None = pozycja czysto redakcyjna (manualna)
ROW_CHECKS: dict[str, list[str] | None] = {
    "r1_1": ["pool_split"],
    "r1_2": ["seeds"],
    "r1_3": ["sensitivity"],
    "r1_4": ["matrix", "run_manifests"],
    "r1_5": ["inference_config"],
    "r1_6": ["metrics_fields"],
    "r1_7": ["transfer"],
    "r1_8": ["iso_resource"],
    "r1_9": ["nms"],
    "r1_10": ["ledger"],
    "r1_drobne1": None,      # "nowosc to kombinacja" — redakcja
    "r1_drobne2": ["size_sweep"],
    "r1_drobne3": ["aug_mix"],
    "r2_1": ["sensitivity"],
    "r2_2": ["dataset_table"],
    "r2_3": ["transfer"],
    "r2_4": ["metrics_fields"],
    "r2_5": ["matrix"],
    "r2_6": ["baselines"],
    "r2_7": ["transfer"],
    "r2_8": ["dataset_table"],
    "r2_9": ["seeds"],        # ziarna; "jedna instytucja" zostaje jako ograniczenie w tekscie
    "r2_10": ["transfer"],
    "r2_11-16": ["sens_pca_alg"],
    "r2_17": None,            # interpretacja kliniczna — redakcja
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewer-map", type=Path, default=MAP_ROOT)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--variants-config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    protocol = C.load_protocol(args.protocol)
    variants = C.load_variants(args.variants_config)
    acfg = C.analysis_cfg(protocol)
    results_dir = args.results_dir or C.cfg_path(acfg, "results_dir")
    res = scan_results(results_dir)
    n_results = len(res["run_ids"])

    rows = parse_reviewer_map(args.reviewer_map)
    if not rows:
        raise SystemExit(f"Nie sparsowano zadnego wiersza z {args.reviewer_map}")

    manual_p = C.cfg_path(acfg, "out_dir") / "manual_confirmations.json"
    manual = {}
    if manual_p.exists():
        data = json.loads(manual_p.read_text(encoding="utf-8"))
        manual = data.get("confirmed", data)

    cache: dict[str, tuple[str, str]] = {}
    items = []
    for row in rows:
        checks = ROW_CHECKS.get(row["key"], "UNMAPPED")
        sub, state = [], "brak"
        if checks is None:
            if row["key"] in manual:
                state = "zamkniete"
                sub.append({"name": "manual", "state": "zamkniete",
                            "detail": f"potwierdzenie reczne autora: {manual[row['key']]}"})
            else:
                sub.append({"name": "manual", "state": "brak",
                            "detail": "pozycja redakcyjna — zamknij tekstem artykułu i "
                                      "wpisz do analysis/out/manual_confirmations.json"})
        elif checks == "UNMAPPED":
            sub.append({"name": "mapping", "state": "brak",
                        "detail": "brak mapowania wiersza na sprawdzenie — dopisz w ROW_CHECKS"})
        else:
            state = "zamkniete"
            for name in checks:
                if name not in cache:
                    cache[name] = CHECKS[name](protocol, variants, res, n_results)
                st, detail = cache[name]
                sub.append({"name": name, "state": st, "detail": detail})
                state = worse(state, st)
        items.append({**row, "state": state, "checks": sub})

    counts = {s: sum(1 for i in items if i["state"] == s) for s in STATES}
    print(f"Wynikow na dysku: {n_results} ({results_dir})\n")
    print(f"{'pozycja':14} {'stan':10} zarzut / szczegoly")
    print("-" * 100)
    for i in items:
        zarzut = i["zarzut"][:58]
        print(f"{i['key']:14} {i['state'].upper():10} {zarzut}")
        for c in i["checks"]:
            print(f"{'':14} {'':10}   [{c['name']}] {c['detail'][:110]}")
    print("-" * 100)
    print(f"PODSUMOWANIE: zamkniete={counts['zamkniete']}  w toku={counts['w toku']}  brak={counts['brak']}")
    if counts["brak"]:
        print("Artikel NIE gotowy do zgloszenia — pozycje 'brak' wroca jako zarzuty recenzenta.")

    out = args.out or C.cfg_path(acfg, "out_dir") / "checklist.json"
    C.write_json(out, {"summary": counts, "n_results": n_results,
                       "results_dir": str(results_dir), "items": items})
    raise SystemExit(0 if counts["brak"] == 0 else 1)


if __name__ == "__main__":
    main()
