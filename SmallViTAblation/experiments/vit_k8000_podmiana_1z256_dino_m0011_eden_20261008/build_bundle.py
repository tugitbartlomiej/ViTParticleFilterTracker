from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation")
EXP = ROOT / "experiments" / "vit_k8000_podmiana_1z256_dino_m0011_eden_20261008"
OUT = ROOT / "outputs" / "vit_k8000_podmiana_1z256_dino_m0011_eden_20261008"
DATA = OUT / "input_pack"
CKPT = ROOT / "assets" / "MalyViT_K8000_s47_losowe" / "MALY_VIT_K8000_s47_losowe.pth"
BUNDLE = OUT / "deploy_bundle.tar.gz"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    files = []
    for path in sorted(EXP.iterdir()):
        if path.is_file() and path.name != "build_bundle.py" and path.suffix != ".pyc":
            files.append((path, Path("code") / path.name))
    for path in sorted(DATA.rglob("*")):
        if path.is_file():
            files.append((path, Path("input_pack") / path.relative_to(DATA)))
    files.append((CKPT, Path("assets") / CKPT.name))

    manifest = {
        "files": [{"path": str(arc).replace("\\", "/"), "bytes": src.stat().st_size, "sha256": sha(src)} for src, arc in files],
        "file_count": len(files),
        "checkpoint_sha256_expected": "0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b",
    }
    manifest_path = OUT / "DEPLOY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append((manifest_path, Path("DEPLOY_MANIFEST.json")))

    with tarfile.open(BUNDLE, "w:gz", compresslevel=6) as tf:
        for src, arc in files:
            tf.add(src, arcname=str(arc).replace("\\", "/"), recursive=False)
    report = {"bundle": str(BUNDLE), "bytes": BUNDLE.stat().st_size, "sha256": sha(BUNDLE), **manifest}
    (OUT / "BUNDLE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"bundle": str(BUNDLE), "bytes": BUNDLE.stat().st_size, "sha256": report["sha256"], "files": manifest["file_count"]}, indent=2))


if __name__ == "__main__":
    main()
