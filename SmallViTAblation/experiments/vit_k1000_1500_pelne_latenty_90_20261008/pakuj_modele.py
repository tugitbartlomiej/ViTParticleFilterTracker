"""Buduje dwa samodzielne, wykonywalne pakiety K1000 + wspólna korekta."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pelne_latenty import GELU_FLOOR

EXPECTED_CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
EXPECTED_MANIFEST_SHA256 = "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7"
EXPECTED_COCO_SHA256 = "3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75"
RUNTIME_FILES = ("model_snapshot.py", "korekta_model.py", "artifact_runtime.py")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_existing(
    path: Path,
    *,
    method: str,
    checkpoint: Path,
    source_code: Path,
    source_model: Path,
) -> bool:
    manifest_path = path / "FILES_SHA256.json"
    if not path.exists():
        return False
    if not manifest_path.exists():
        raise RuntimeError(f"Niekompletny istniejący pakiet {path}")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        candidate = path / relative
        if not candidate.is_file() or sha256(candidate) != digest:
            raise RuntimeError(f"Niespójny istniejący pakiet: {candidate}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((path / "PROVENANCE.json").read_text(encoding="utf-8"))
    expected_runtime = {name: sha256(source_code / name) for name in RUNTIME_FILES}
    checks = {
        "method": config.get("method") == method,
        "checkpoint": config.get("checkpoint_sha256") == sha256(checkpoint),
        "source_model": provenance.get("source_regime_model_sha256") == sha256(source_model),
        "runtime": provenance.get("runtime_source_sha256") == expected_runtime,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Istniejący pakiet pochodzi z innych wejść: {path}: {checks}")
    return True


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32).copy())


def _package_readme(method: str) -> str:
    label = "A: mean600x768" if method == "mean600x768" else "B: inferred regime corrector"
    return f"""# Wykonywalny K1000 + korekta {label}

Pakiet zawiera pełny checkpoint K1000, kod architektury i zamrożoną warstwę
korekty naturalnego latentu post-GELU. Dla każdego obrazu model najpierw liczy
jego własny naturalny latent `600x768`, następnie stosuje wspólną regułę i
kończy ten sam pełny forward. Inferencja nie używa etykiety ani oracle delta.

```python
from pathlib import Path
from artifact_runtime import load_executable_artifact

model, config = load_executable_artifact(Path('.'), device='cuda')
output, natural_z0, corrected_z = model(image_fp32_0_1)
```

`FILES_SHA256.json` chroni wszystkie pliki wejściowe pakietu. Metoda została
dopasowana na oracle ruchach dokładnych 1500 obrazów K1000; wynik in-sample nie
jest dowodem generalizacji. Ocena group-holdout po filmach znajduje się poza
pakietem w raporcie eksperymentu.
"""


def _write_package(
    destination: Path,
    *,
    method: str,
    checkpoint: Path,
    correction_state: dict[str, torch.Tensor],
    source_code: Path,
    source_model: Path,
) -> dict:
    if _validate_existing(
        destination,
        method=method,
        checkpoint=checkpoint,
        source_code=source_code,
        source_model=source_model,
    ):
        return {"path": str(destination), "status": "existing_valid"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent)
    )
    try:
        shutil.copy2(checkpoint, temporary / "MODEL_VIT_K1000.pt")
        for name in RUNTIME_FILES:
            shutil.copy2(source_code / name, temporary / name)
        torch.save(correction_state, temporary / "CORRECTION.pt")
        correction_sha = sha256(temporary / "CORRECTION.pt")
        config = {
            "schema_version": 1,
            "method": method,
            "checkpoint_step": 1000,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "correction_sha256": correction_sha,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "coco_sha256": EXPECTED_COCO_SHA256,
            "latent_point": "last transformer block MLP GELU output",
            "latent_shape": [600, 768],
            "grid_shape": [20, 30],
            "gelu_floor": GELU_FLOOR,
            "precision": "FP32",
            "weights_and_buffers_frozen": True,
            "natural_latent_per_image": True,
            "target_or_oracle_used_at_inference": False,
            "fit_scope": "same exact 1500 K1000 base images; in-sample",
            "runtime_versions": {
                "python_package_torch": torch.__version__,
                "python_package_numpy": np.__version__,
                "python_package_timm": _package_version("timm"),
            },
        }
        _write_json(temporary / "config.json", config)
        (temporary / "README.md").write_text(_package_readme(method), encoding="utf-8")
        provenance = {
            **config,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(checkpoint),
            "source_regime_model": str(source_model),
            "source_regime_model_sha256": sha256(source_model),
            "runtime_source_sha256": {
                name: sha256(source_code / name) for name in RUNTIME_FILES
            },
        }
        _write_json(temporary / "PROVENANCE.json", provenance)
        hashes = {
            file.relative_to(temporary).as_posix(): sha256(file)
            for file in sorted(temporary.rglob("*"))
            if file.is_file()
        }
        _write_json(temporary / "FILES_SHA256.json", hashes)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "path": str(destination),
        "status": "created",
        "method": method,
        "correction_sha256": correction_sha,
        "files": len(hashes),
    }


def build_packages(
    checkpoint: Path, regime_model: Path, output: Path, source_code: Path
) -> dict:
    checkpoint = Path(checkpoint)
    regime_model = Path(regime_model)
    output = Path(output)
    source_code = Path(source_code)
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Nieprawidłowy checkpoint K1000")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"net", "krok"} or payload["krok"] != 1000:
        raise RuntimeError("Nieprawidłowy kontener checkpointa K1000")
    with np.load(regime_model, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    required = {
        "mean_delta",
        "z_mean",
        "z_components",
        "feature_mean",
        "feature_scale",
        "assignment_centroids",
        "correction_by_regime_token",
        "grid_shape",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"Brak pól modelu reżimów: {sorted(required - set(arrays))}")
    if arrays["mean_delta"].shape != (600, 768):
        raise RuntimeError(arrays["mean_delta"].shape)
    if arrays["correction_by_regime_token"].shape[1:] != (600, 768):
        raise RuntimeError(arrays["correction_by_regime_token"].shape)
    if arrays["grid_shape"].tolist() != [20, 30]:
        raise RuntimeError(arrays["grid_shape"])

    mean_state = {"mean_delta": _as_tensor(arrays["mean_delta"])}
    regime_names = (
        "z_mean",
        "z_components",
        "feature_mean",
        "feature_scale",
        "assignment_centroids",
        "correction_by_regime_token",
    )
    regime_state = {name: _as_tensor(arrays[name]) for name in regime_names}
    result = {
        "mean": _write_package(
            output / "model_mean",
            method="mean600x768",
            checkpoint=checkpoint,
            correction_state=mean_state,
            source_code=source_code,
            source_model=regime_model,
        ),
        "regime": _write_package(
            output / "model_regime",
            method="inferred_regime",
            checkpoint=checkpoint,
            correction_state=regime_state,
            source_code=source_code,
            source_model=regime_model,
        ),
    }
    _write_json(output / "PACKAGE_BUILD.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--regime-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(
        json.dumps(
            build_packages(args.checkpoint, args.regime_model, args.output, args.source_code),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
