"""Odczyt lokalnego środowiska bez modeli, tensorów, inferencji i treningu.

Python 3.11; sam skrypt wymaga wyłącznie biblioteki standardowej. Importy
zależności wykonuje w osobnych procesach, również przy awarii biblioteki DLL.
Nie instaluje pakietów, nie pobiera danych, nie zapisuje cache ani raportu.
JSON wypisuje na stdout. Opcjonalne --python można podać wielokrotnie.
CUDA sprawdza przez NVML/nvidia-smi, bez uruchamiania kerneli.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
VENV_ROOT = ROOT.parents[1] / "venvs"
FASTGAN = ROOT / "experiments/fastgan_narzedzie"
HUB = Path.home() / ".cache/torch/hub"
LOCAL_INPUTS = Path(__file__).resolve().parent / "local_inputs"
PACKAGES = ("torch", "torchvision", "rfdetr", "transformers", "numpy", "scipy",
            "scikit-learn", "Pillow", "faiss-cpu", "pytorch-lightning",
            "supervision", "lpips", "timm", "omegaconf")
PREFIX = "__PREFLIGHT_JSON__="


def inspect_file(path: Path) -> dict:
    result = {"path": path.as_posix(), "exists": path.is_file()}
    if path.is_file():
        result.update(bytes=path.stat().st_size)
        with path.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def import_check(python: str, body: str, timeout: int = 60) -> dict:
    """Import bez konstruowania modelu; awaria procesu pozostaje w raporcie."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0",
               NO_ALBUMENTATIONS_UPDATE="1", PYTORCH_NVML_BASED_CUDA_CHECK="1",
               PYTHONIOENCODING="utf-8")
    # Nie maskujemy konfliktu dwóch runtime OpenMP.
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    code = "import json\nresult = {}\n" + body
    code += "\nprint(" + repr(PREFIX) + " + json.dumps(result), flush=True)\n"
    try:
        proc = subprocess.run([python, "-B", "-c", code], cwd=ROOT, env=env,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}
    lines = proc.stdout.splitlines()
    marker = next((line[len(PREFIX):] for line in reversed(lines)
                   if line.startswith(PREFIX)), None)
    return {"ok": proc.returncode == 0 and marker is not None,
            "returncode": proc.returncode,
            "details": json.loads(marker) if marker else None,
            "stdout": "\n".join(line for line in lines if not line.startswith(PREFIX))[-4000:],
            "stderr": proc.stderr[-6000:]}


def inspect_environment(python: str) -> dict:
    metadata = """
import importlib.metadata as md
import platform, site, sys
from pathlib import Path
result.update(python=sys.executable, version=platform.python_version(),
              prefix=sys.prefix, base_prefix=sys.base_prefix,
              user_site_enabled=site.ENABLE_USER_SITE, packages={})
cfg = Path(sys.prefix) / 'pyvenv.cfg'
result['pyvenv_cfg'] = cfg.read_text() if cfg.exists() else None
for name in PACKAGES:
    try:
        dist = md.distribution(name)
        result['packages'][name] = dict(version=dist.version, root=str(dist.locate_file('')))
        if name == 'rfdetr':
            result['packages'][name]['requires'] = dist.requires
    except md.PackageNotFoundError:
        result['packages'][name] = None
""".replace("for name in PACKAGES:", f"for name in {PACKAGES!r}:")
    checks = {
        "metadata": metadata,
        "numerical_imports": "import numpy, scipy, PIL\nresult['numpy'] = numpy.__version__",
        "torch_cuda_metadata": """
import torch, torchvision
result.update(torch=torch.__version__, torchvision=torchvision.__version__,
              cuda_build=torch.version.cuda, cuda_built=torch.backends.cuda.is_built(),
              cuda_available_nvml=torch.cuda.is_available(),
              device_count_nvml=torch.cuda.device_count(),
              cuda_context_initialized=torch.cuda.is_initialized())
""",
        "rfdetr_generation_imports": """
from rfdetr import RFDETRNano
from rfdetr.utilities.tensors import NestedTensor
result.update(RFDETRNano_module=RFDETRNano.__module__, NestedTensor_module=NestedTensor.__module__)
""",
        "faiss_import": "import faiss\nresult['version'] = faiss.__version__",
        "torch_then_faiss_import": "import torch, faiss\nresult['version'] = faiss.__version__",
        "fastgan_vendor_import": f"""
import importlib.util
spec = importlib.util.spec_from_file_location('local_fastgan_preflight', {str(FASTGAN / 'vendor/deepfake_gi_fastGAN/models.py')!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result['Generator_present'] = hasattr(module, 'Generator')
""",
        "dino_hubconf_import": f"""
import sys
sys.path.insert(0, {str(LOCAL_INPUTS / 'dino')!r})
import hubconf
result['dinov2_vits14_present'] = callable(hubconf.dinov2_vits14)
result['hubconf_path'] = hubconf.__file__
""",
    }
    # Najwyżej dwa importy jednocześnie, bez uruchamiania obliczeń modeli.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(import_check, python, body)
                   for name, body in checks.items()}
        results = {name: future.result() for name, future in futures.items()}
    return {"python": python, "checks": results}


def inspect_assets() -> dict:
    download_root = FASTGAN / "wyniki/eden_20260923_50k"
    config = json.loads((download_root / "results_50k/CONFIG.json").read_text(encoding="utf-8-sig"))
    download = json.loads((download_root / "DOWNLOAD.json").read_text(encoding="utf-8-sig"))
    checkpoint = inspect_file(download_root / "results_50k/checkpoint_050000.pth")
    checkpoint["expected_sha256"] = download["files"]["results_50k/checkpoint_050000.pth"]["sha256"]
    checkpoint["download_sha256_matches"] = checkpoint.get("sha256") == checkpoint["expected_sha256"]
    vendor = inspect_file(FASTGAN / "vendor/deepfake_gi_fastGAN/models.py")
    vendor["training_architecture_sha256_matches"] = vendor.get("sha256") == config["architecture_sha256"]
    crops = inspect_file(FASTGAN / "dane/pilot_1000/MANIFEST.json")
    crop_records = json.loads(Path(crops["path"]).read_text(encoding="utf-8-sig"))["records"]
    crops.update(root=str(Path(crops["path"]).parent), records=len(crop_records),
                 missing_files=sum(not (Path(crops["path"]).parent / row["file"]).is_file()
                                   for row in crop_records),
                 training_manifest_sha256_matches=crops.get("sha256") == config["data_manifest_sha256"])
    manifest = ROOT / "feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl"
    pool = inspect_file(manifest)
    counts: dict[str, int] = {}
    train_rows = []
    with manifest.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            split = row.get("split", "unknown")
            counts[split] = counts.get(split, 0) + 1
            if split == "train":
                train_rows.append(row)
    roots = [manifest.parent, ROOT / "feature_cache/lokalny_static36"]
    pool.update(split_counts=counts, candidate_data_roots=[
        {"path": root.as_posix(), "train_files_present": sum((root / row["path"]).is_file()
                                                            for row in train_rows),
         "train_files_expected": len(train_rows)} for root in roots])
    dino = {"torch_hub_dir": (HUB / "facebookresearch_dinov2_main").as_posix(),
            "weights": inspect_file(HUB / "checkpoints/dinov2_vits14_pretrain.pth"),
            "eden_code_ref": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "local_code_matches_eden": None,
            "note": "Lokalny cache main nie ma potwierdzonej zgodności z kodem indeksu Edena."}
    return {"fastgan_checkpoint": checkpoint, "fastgan_vendor": vendor,
            "fastgan_training_config": config, "crop_manifest": crops, "pool": pool,
            "local_pool_manifest": inspect_file(ROOT / "feature_cache/lokalny_static36/pool_manifest.jsonl"),
            "source_annotation_archives": (ROOT.parent / "Dataset/Cataract-LMM/3_Object_Tracking/annotations").as_posix(),
            "dino": dino}


def integration_config(report: dict) -> dict:
    """Ścieżki dla koordynatora; ready nie oznacza testu inferencji/GPU."""
    selected_python = VENV_ROOT / "rfdetr_1_10_1/Scripts/python.exe"
    data_root = LOCAL_INPUTS / "data"
    selection = LOCAL_INPUTS / "initial_selection.json"
    pool_manifest = data_root / "pool_manifest.jsonl"
    initial_weights = LOCAL_INPUTS / "rf-detr-nano.pth"
    assets = report["assets"]
    report["assets"]["initial_selection"] = inspect_file(selection)
    report["assets"]["initial_weights"] = inspect_file(initial_weights)
    report["assets"]["initial_weights"]["expected_sha256"] = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"
    report["assets"]["initial_weights"]["sha256_matches"] = (
        report["assets"]["initial_weights"].get("sha256")
        == report["assets"]["initial_weights"]["expected_sha256"])
    config = {"python": selected_python.as_posix(), "data_root": data_root.as_posix(),
              "pool": pool_manifest.as_posix(),
              "initial_selection": selection.as_posix(),
              "initial_weights": initial_weights.as_posix(),
              "crop_manifest": assets["crop_manifest"]["path"],
              "crop_root": (FASTGAN / "dane/pilot_1000").as_posix(),
              "generator": assets["fastgan_checkpoint"]["path"],
              "vendor": (FASTGAN / "vendor/deepfake_gi_fastGAN").as_posix(),
              "hub": (LOCAL_INPUTS / "dino").as_posix(),
              "dino_weights": assets["dino"]["weights"]["path"], "pythonpath": []}
    selected = next((entry for entry in report["environments"]
                     if Path(entry["python"]) == selected_python), None)
    blockers = []
    if not selected or not all(check["ok"] for check in selected["checks"].values()):
        blockers.append("Nie wszystkie wymagane importy/metadane wybranego interpretera przeszły kontrolę.")
    for key, value in config.items():
        if isinstance(value, str) and not Path(value).exists():
            blockers.append(f"Brak ścieżki {key}: {value}")
    normalized_pool = inspect_file(pool_manifest)
    normalized_pool["expected_sha256"] = "10d84e513054b6287b3eb2a4864839fc5457942c9f636716b17aa9e2d709d91e"
    normalized_pool["expected_sha256_source"] = "SHA-256 puli Edena potwierdzone przez użytkownika."
    normalized_pool["sha256_matches"] = normalized_pool.get("sha256") == normalized_pool["expected_sha256"]
    assets["normalized_pool"] = normalized_pool
    if not normalized_pool["sha256_matches"]:
        blockers.append("Docelowa pula z Edena jest nieobecna lub ma niezgodne SHA-256.")
    else:
        with pool_manifest.open(encoding="utf-8-sig") as stream:
            train_rows = [row for line in stream if (row := json.loads(line))["split"] == "train"]
        present = sum((data_root / row["path"]).is_file() for row in train_rows)
        normalized_pool.update(train_files_present=present, train_files_expected=len(train_rows))
        if present != len(train_rows):
            blockers.append(f"Niepełna docelowa pula train: {present}/{len(train_rows)} plików.")
    dino_check = (selected or {}).get("checks", {}).get("dino_hubconf_import", {})
    dino_details = dino_check.get("details") or {}
    if not dino_check.get("ok") or Path(dino_details.get("hubconf_path", "")) != LOCAL_INPUTS / "dino/hubconf.py":
        blockers.append("Import dokładnego kodu DINO z local_inputs/dino wymaga sprawdzenia po skopiowaniu.")
    for ok, message in (
        (assets["initial_weights"]["sha256_matches"], "SHA-256 wag początkowych COCO"),
        (assets["fastgan_checkpoint"]["download_sha256_matches"], "SHA-256 checkpointu FastGAN"),
        (assets["fastgan_vendor"]["training_architecture_sha256_matches"], "SHA-256 architektury FastGAN"),
        (assets["crop_manifest"]["training_manifest_sha256_matches"], "SHA-256 manifestu wycinków FastGAN"),
        (assets["crop_manifest"]["missing_files"] == 0, "komplet wycinków FastGAN"),
    ):
        if not ok:
            blockers.append("Nie przeszła kontrola: " + message)
    config.update(ready=not blockers, ready_scope="Importy i kompletność lokalnych wejść; bez testu forward, gradientów i dopasowania do VRAM.",
                  blockers=blockers, inference_verified=False, gpu_execution_verified=False,
                  notes=[
                      "Oba venv korzystają z --system-site-packages i aktywnego user site; nie zmieniano żadnych pakietów.",
                      "Do lokalnego generowania wybrano RF-DETR 1.10.1; fastgan_local nie ma RF-DETR.",
                      "Lightning 2.2.2 nie spełnia rfdetr[train]: >=2.6,<3,!=2.6.2,!=2.6.3. Konstruktor/generowanie nie wymagają lokalnego treningu.",
                      "Podstawowe wymagania RF-DETR: torch>=2.2, torchvision>=0.17, transformers>=5.1,<6, supervision>=0.29, pydantic>=2,<3.",
                      "Docelowe dane, initial_selection i kod DINO pochodzą z kopii Edena w local_inputs; stare kontrole assets/pool i assets/dino opisują wcześniejszy lokalny cache.",
                      "Wagi DINO pozostają w lokalnym cache; ich SHA-256 zapisano w assets/dino/weights. Indeks i zapytania muszą używać tego samego źródła modelu.",
                      "Torch lokalny 2.6.0+cu124 różni się od zapisanego środowiska FastGAN 2.9.1+cu128; zgodność numeryczna forward/gradientów nie była badana.",
                      "config.json ma permutations=32; konfiguracji nie zmieniano.",
                  ])
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="append", dest="pythons",
                        help="Interpreter do kontroli; można powtarzać.")
    args = parser.parse_args()
    pythons = args.pythons or [str(VENV_ROOT / name / "Scripts/python.exe")
                              for name in ("fastgan_local", "rfdetr_1_10_1")]
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Importy, wersje, SHA-256 plików, istnienie ścieżek i metadane CUDA. Bez modeli, inferencji i treningu.",
              "preflight_python": sys.executable,
              "environments": [inspect_environment(python) for python in pythons],
              "assets": inspect_assets()}
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,compute_cap",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        report["gpu_metadata"] = {"ok": proc.returncode == 0,
                                  "columns": ["name", "driver", "total_MiB", "used_MiB", "free_MiB", "compute_capability"],
                                  "csv": proc.stdout.strip(), "stderr": proc.stderr.strip(),
                                  "observed_utc": datetime.now(timezone.utc).isoformat()}
    except (OSError, subprocess.TimeoutExpired) as error:
        report["gpu_metadata"] = {"ok": False, "error": str(error)}
    report = {**integration_config(report), **report}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
