"""Weryfikuje pakiet na świeżych 1500 forwardach; obala go dowolna niespójność z best."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from artifact_runtime import load_executable_artifact, verify_package_files
from dane import TarImageSource, load_manifest
from uruchom_joint import (
    REGULARIZATION_WEIGHT,
    _image_tensor,
    _record_batch,
    _target_from_box,
    json_atomic,
    joint_loss_per_image,
    sha256,
    state_fingerprint,
    summarize_records,
    write_jsonl,
)
from wspolna_korekta import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    GELU_FLOOR,
    tensor_sha256,
)

PACKAGE_FILES = ("model_snapshot.py", "artifact_runtime.py")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_package_manifest(package: Path) -> dict[str, str]:
    hashes = {
        path.name: sha256(path)
        for path in package.iterdir()
        if path.is_file() and path.name != "FILES_SHA256.json"
    }
    json_atomic(package / "FILES_SHA256.json", dict(sorted(hashes.items())))
    return hashes


def _package_readme() -> str:
    return """# K1000 z jedną wspólną korektą per patch

Pakiet zawiera dokładny checkpoint K1000 oraz jedną macierz `delta[600,768]`.
Dla każdego nowego obrazu model oblicza jego własny naturalny post-GELU `z0`,
stosuje `clamp_min(z0 + delta, -0.16997121)` i wykonuje oryginalny tail.

Delta została wybrana według mAP50:95 na tych samych 1500 obrazach, na których
ją uczono. Pakiet nie ma potwierdzonej generalizacji poza tą próbą.
"""


@torch.no_grad()
def fresh_load_evaluation(
    package: Path,
    manifest: Path,
    archive: Path,
    baseline_records_path: Path,
    best_summary: dict,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    rows = load_manifest(manifest)
    if len(rows) != 1500:
        raise RuntimeError(len(rows))
    baseline_records = _load_jsonl(baseline_records_path)
    model, config = load_executable_artifact(package, device)
    model_hash_before = state_fingerprint(model.base)
    records: list[dict] = []
    gt_all = torch.empty((1500, 4), dtype=torch.float32)
    loss_sum = 0.0
    component_sums = {name: 0.0 for name in ("positive", "negative", "rank", "box", "giou")}
    started = time.perf_counter()
    source = TarImageSource(archive)
    first_outputs: list[torch.Tensor] = []
    try:
        for start in range(0, 1500, batch_size):
            end = min(start + batch_size, 1500)
            batch_rows = rows[start:end]
            images = torch.stack([_image_tensor(source, row, device) for row in batch_rows])
            targets = [_target_from_box(row["bbox_xyxy_480"]) for row in batch_rows]
            gt = torch.tensor(
                [row["bbox_xyxy_480"] for row in batch_rows], device=device, dtype=torch.float32
            )
            target_index = torch.tensor(
                [target["token_index"] for target in targets], device=device, dtype=torch.long
            )
            target_ltrb = torch.tensor(
                np.stack([target["target_ltrb"] for target in targets]),
                device=device,
                dtype=torch.float32,
            )
            output = model(images)
            if start == 0:
                first_outputs.append(output.detach().cpu())
            losses, components = joint_loss_per_image(output, target_index, target_ltrb)
            loss_sum += float(losses.sum().item())
            for name, values in components.items():
                component_sums[name] += float(values.sum().item())
            gt_all[start:end].copy_(gt.cpu())
            records.extend(_record_batch(output, gt, target_index, batch_rows))
            print(json.dumps({"fresh_load": [end, 1500]}), flush=True)
    finally:
        source.close()

    second_model, second_config = load_executable_artifact(package, device)
    source = TarImageSource(archive)
    try:
        first_rows = rows[:batch_size]
        first_images = torch.stack([_image_tensor(source, row, device) for row in first_rows])
    finally:
        source.close()
    second_output = second_model(first_images).detach().cpu()
    two_loads_max_abs_error = float((first_outputs[0] - second_output).abs().max().item())
    delta = model.delta.detach()
    regularization = REGULARIZATION_WEIGHT * float(delta.square().mean().item())
    objective = {
        "data_loss": loss_sum / 1500,
        "regularization": regularization,
        "total_loss": loss_sum / 1500 + regularization,
        "components": {name: value / 1500 for name, value in component_sums.items()},
    }
    summary = summarize_records(records, gt_all, objective, delta.cpu(), baseline_records)
    expected = best_summary["detection"]
    differences = {
        key: float(summary["detection"][key] - expected[key])
        for key in ("AP50", "AP75", "mAP50_95")
    }
    if any(abs(value) > 1e-9 for value in differences.values()):
        raise RuntimeError(f"Fresh-load różni się od wybranego best: {differences}")
    model_hash_after = state_fingerprint(model.base)
    summary["fresh_load"] = {
        "status": "pass",
        "all_1500_retested": True,
        "two_independent_loads_max_abs_error_first_batch": two_loads_max_abs_error,
        "two_independent_loads_bitwise_equal_first_batch": two_loads_max_abs_error == 0.0,
        "config_equal_between_loads": config == second_config,
        "metrics_difference_vs_training_best": differences,
        "weights_buffers_hash_before": model_hash_before,
        "weights_buffers_hash_after": model_hash_after,
        "weights_buffers_unchanged": model_hash_before == model_hash_after,
        "all_parameters_require_grad_false": all(not p.requires_grad for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not all(
        (
            summary["fresh_load"]["two_independent_loads_bitwise_equal_first_batch"],
            summary["fresh_load"]["config_equal_between_loads"],
            summary["fresh_load"]["weights_buffers_unchanged"],
            summary["fresh_load"]["all_parameters_require_grad_false"],
        )
    ):
        raise RuntimeError(summary["fresh_load"])
    return summary, records


def _report(training: dict, fresh: dict) -> str:
    baseline = training["baseline"]
    best = training["best"]
    final = training["final"]
    old = training["old_mean_A_reference"]["metrics"]
    return f"""# Wynik: jedna wspólna korekta per patch K1000

## Werdykt

Wybór i ocena są wykonane na tych samych 1500 bazowych obrazach. Jest to
wynik in-sample i nie stanowi dowodu generalizacji.

Najlepszy checkpoint pochodzi z passu {training['best_pass']} spośród
{training['completed_pass']} wykonanych pełnych passów. Każdy pass obejmował
dokładnie 1500 obrazów i jedną aktualizację wspólnej delty Adam.

## Metryki

| wariant | AP50 | AP75 | mAP50:95 | both90 |
|---|---:|---:|---:|---:|
| baseline delta=0 | {baseline['detection']['AP50']:.9f} | {baseline['detection']['AP75']:.9f} | {baseline['detection']['mAP50_95']:.9f} | {baseline['both90_count']} |
| historyczna Mean A | {old['detection']['AP50']:.9f} | {old['detection']['AP75']:.9f} | {old['detection']['mAP50_95']:.9f} | {old['both90_count']} |
| final | {final['detection']['AP50']:.9f} | {final['detection']['AP75']:.9f} | {final['detection']['mAP50_95']:.9f} | {final['both90_count']} |
| best | {best['detection']['AP50']:.9f} | {best['detection']['AP75']:.9f} | {best['detection']['mAP50_95']:.9f} | {best['both90_count']} |
| fresh-load best | {fresh['detection']['AP50']:.9f} | {fresh['detection']['AP75']:.9f} | {fresh['detection']['mAP50_95']:.9f} | {fresh['both90_count']} |

## Diagnostyka

Baseline false positives przy score co najmniej 0,5: {baseline['false_positives']['score_0.5_iou_0.5']['count']}.
Best false positives przy score co najmniej 0,5: {best['false_positives']['score_0.5_iou_0.5']['count']}.
Baseline właściwy token był top-1 dla {baseline['top1_is_target_token_count']} obrazów;
best dla {best['top1_is_target_token_count']}.

Norma RMS najlepszej delty wynosi {best['delta']['rms']:.9g}, a L2
{best['delta']['l2']:.9g}. Model K1000 pozostał zamrożony. Fresh-load ponownie
ocenił 1500 obrazów i odtworzył metryki best.

## Interpretacja A/B

Historyczna Mean A mieszała poprawki tokenu obiektu z poprawkami tła dla tego
samego indeksu patcha w innych obrazach. Jest to zgodne ze spadkiem confidence
dla 1481 z 1500 obrazów. Historyczne B miało tylko 36,06 procent trafności
odtworzenia etykiet odkrytych reżimów na danych uczących. Te porażki nie
dowodzą niemożliwości wspólnego conditional correctora; obecny eksperyment
testuje wyłącznie najprostszy wariant jednej bezwarunkowej delty.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh-load wymaga CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Niewłaściwy K1000")
    if sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Niewłaściwy manifest")
    training_summary_path = args.training / "TRAINING_SUMMARY.json"
    training = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if training["status"] != "complete" or not training["weights_buffers_unchanged"]:
        raise RuntimeError("Trening nie jest kompletny lub naruszył K1000")
    best = torch.load(args.training / "BEST_DELTA.pt", map_location="cpu", weights_only=True)
    package = args.output / "package"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, package / "MODEL_VIT_K1000.pt")
    for name in PACKAGE_FILES:
        shutil.copy2(args.code_root / name, package / name)
    torch.save(
        {
            "delta": best["delta"].float().cpu(),
            "selected_pass": int(best["selected_pass"]),
            "selection_metric": "training_mAP50_95",
            "delta_sha256": tensor_sha256(best["delta"]),
        },
        package / "CORRECTION.pt",
    )
    config = {
        "artifact": "K1000 + one shared patch delta",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "delta_shape": [600, 768],
        "delta_sha256": tensor_sha256(best["delta"]),
        "gelu_floor": GELU_FLOOR,
        "intervention": "post-GELU of final MLP; natural z0 + shared delta",
        "selected_pass": int(best["selected_pass"]),
        "selection_metric": "mAP50_95 on the same 1500 training images",
        "in_sample": True,
        "generalization_claim": False,
    }
    json_atomic(package / "config.json", config)
    provenance = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "training_summary_sha256": sha256(training_summary_path),
        "best_delta_source_sha256": sha256(args.training / "BEST_DELTA.pt"),
        "source_code_sha256": {name: sha256(args.code_root / name) for name in PACKAGE_FILES},
        "selection_on_training_data": True,
        "old_oracle_deltas_used_for_training": False,
        "weights_unfrozen": False,
    }
    json_atomic(package / "PROVENANCE.json", provenance)
    (package / "README.md").write_text(_package_readme(), encoding="utf-8", newline="\n")
    _write_package_manifest(package)
    verify_package_files(package)

    fresh, records = fresh_load_evaluation(
        package,
        args.manifest,
        args.archive,
        args.training / "BASELINE_RECORDS.jsonl",
        training["best"],
        device,
        args.batch_size,
    )
    write_jsonl(args.output / "FRESH_LOAD_RECORDS_1500.jsonl", records)
    json_atomic(args.output / "FRESH_LOAD_1500.json", fresh)
    json_atomic(package / "FRESH_LOAD_1500.json", fresh)
    _write_package_manifest(package)
    verify_package_files(package)
    report = _report(training, fresh)
    (args.output / "RAPORT_KONCOWY.md").write_text(report, encoding="utf-8", newline="\n")
    final = {
        "status": "complete",
        "training_summary_sha256": sha256(training_summary_path),
        "fresh_load_sha256": sha256(args.output / "FRESH_LOAD_1500.json"),
        "report_sha256": sha256(args.output / "RAPORT_KONCOWY.md"),
        "package_manifest_sha256": sha256(package / "FILES_SHA256.json"),
        "best_pass": training["best_pass"],
        "baseline_mAP50_95": training["baseline"]["detection"]["mAP50_95"],
        "best_mAP50_95": training["best"]["detection"]["mAP50_95"],
        "final_mAP50_95": training["final"]["detection"]["mAP50_95"],
        "fresh_load_mAP50_95": fresh["detection"]["mAP50_95"],
        "in_sample": True,
        "generalization_claim": False,
    }
    json_atomic(args.output / "FINAL_STATUS.json", final)
    print(json.dumps(final), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
