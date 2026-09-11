#!/usr/bin/env python3
"""Test spojnosci iso-resource (Agent 4, test 3).

Uruchamia train_transfer.py --dry-run dla d-fine-s i yolo-current na TYM SAMYM
pliku wariantu i porownuje wygenerowane run_manifest.json:
  - rozdzielczosc, liczba epok, ziarno, plik wariantu (sha256) — IDENTYCZNE,
  - pole arch — ROZNE.
Opcjonalnie porownuje tez z wzorcem rf-detr-small (manifest Agenta 2), jesli
podano --reference-manifest.

Uruchamiane in-process, bez GPU i bez instalacji ultralytics/d-fine (dry-run).

Uzycie:
    py -3.11 pipeline/transfer/check_iso_resource.py \
        --fixture-root /tmp/fix --variant-file /tmp/fix/variants/FIXTURE__s42__b20.json \
        --protocol configs/protocol.yaml --out /tmp/iso_check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.transfer import train_transfer as tt  # noqa: E402

OK, FAIL = "[OK]  ", "[FAIL]"

# Pola, ktore MUSZA byc identyczne miedzy architekturami (iso-resource).
ISO_SAME = ["resolution", "epochs_completed", "seed"]
ISO_SAME_NESTED = [
    ("iso_resource", "resolution"),
    ("iso_resource", "epochs"),
    ("iso_resource", "seed"),
    ("iso_resource", "batch_size"),
    ("iso_resource", "variant_file_sha256"),
    ("iso_resource", "best_val_criterion"),
    ("protocol_sha256",),
    ("variant_file_sha256",),
]


class Report:
    def __init__(self):
        self.errors = 0

    def ok(self, m):
        print(f"{OK}{m}")

    def fail(self, m):
        self.errors += 1
        print(f"{FAIL}{m}")

    def check(self, cond, m):
        self.ok(m) if cond else self.fail(m)
        return cond


def _dig(man, path):
    cur = man
    for p in path:
        cur = cur[p]
    return cur


def dry_run_arch(arch, run_id, args):
    argv = ["--run-id", run_id, "--variant-file", str(args.variant_file),
            "--protocol", str(args.protocol), "--data-root", str(args.fixture_root),
            "--out", str(args.out), "--arch", arch, "--dry-run"]
    if args.repo_root:
        argv += ["--repo-root", str(args.repo_root)]
    rc = tt.main(argv)
    if rc != 0:
        raise SystemExit(f"dry-run zwrocil {rc} dla {arch}")
    man_path = Path(args.out) / run_id / "run_manifest.json"
    return json.loads(man_path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-root", type=Path, required=True)
    ap.add_argument("--variant-file", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--reference-manifest", type=Path, default=None,
                    help="run_manifest.json modelu glownego (rf-detr-small), jesli istnieje")
    args = ap.parse_args()

    rep = Report()
    run_base = Path(args.variant_file).stem
    man_dfine = dry_run_arch("d-fine-s", f"{run_base}__d-fine-s", args)
    man_yolo = dry_run_arch("yolo-current", f"{run_base}__yolo-current", args)
    manifests = {"d-fine-s": man_dfine, "yolo-current": man_yolo}

    if args.reference_manifest and args.reference_manifest.exists():
        manifests["rf-detr-small(ref)"] = json.loads(
            args.reference_manifest.read_text(encoding="utf-8"))
        print(f"--- dolaczono wzorzec: {args.reference_manifest} ---")
    else:
        print("--- wzorzec rf-detr-small niedostepny (Agent 2) — porownuje transfer ---")

    print("--- iso-resource: pola IDENTYCZNE miedzy architekturami ---")
    names = list(manifests)
    for field in ISO_SAME:
        vals = {n: manifests[n].get(field) for n in names}
        same = len(set(map(str, vals.values()))) == 1
        rep.check(same, f"{field}: {vals}")
    for path in ISO_SAME_NESTED:
        try:
            vals = {n: _dig(manifests[n], path) for n in names}
        except KeyError as e:
            rep.fail(f"{'.'.join(path)}: brak pola ({e})")
            continue
        same = len(set(map(str, vals.values()))) == 1
        rep.check(same, f"{'.'.join(path)}: identyczne "
                        f"({vals[names[0]]})" if same else
                        f"{'.'.join(path)}: ROZNE {vals}")

    print("--- pole arch: MUSI byc rozne ---")
    archs = {n: manifests[n].get("arch") for n in ("d-fine-s", "yolo-current")}
    rep.check(archs["d-fine-s"] != archs["yolo-current"], f"arch rozne: {archs}")
    rep.check(archs["d-fine-s"] == "d-fine-s", "arch d-fine-s poprawne")
    rep.check(archs["yolo-current"] == "yolo-current", "arch yolo-current poprawne")

    print("--- wymagane pola manifestu (README 4.4) ---")
    from SmallViTAblation.pipeline.transfer import transfer_common as tc
    for n, man in manifests.items():
        missing = tc.MANIFEST_REQUIRED_FIELDS - man.keys()
        rep.check(not missing, f"{n}: komplet pol 4.4"
                  + (f" (brak: {sorted(missing)})" if missing else ""))
        rep.check("arch" in man and "nms" in man, f"{n}: pola arch+nms obecne")

    print("--- NMS: D-FINE nie dotyczy, YOLO ustawienia obecne ---")
    rep.check(man_dfine["nms"]["applies"] is False, "d-fine-s: nms.applies=False")
    rep.check(man_yolo["nms"]["applies"] is True, "yolo-current: nms.applies=True")
    rep.check(man_yolo["nms"].get("iou_threshold") is not None,
              "yolo-current: nms.iou_threshold zapisany")
    rep.check(man_yolo["nms"].get("conf_threshold") is not None,
              "yolo-current: nms.conf_threshold zapisany")

    print(f"=== bledy: {rep.errors} ===")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
