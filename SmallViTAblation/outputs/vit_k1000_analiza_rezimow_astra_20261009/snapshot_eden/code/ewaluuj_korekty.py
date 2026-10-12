"""Fresh-load ocena baseline, wspólnych korekt A/B, holdoutu i oracle na 1500."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from artifact_runtime import load_executable_artifact, verify_package_files
from dane import LocalImageSource, TarImageSource, decode_resize, load_manifest
from korekta_model import MeanCorrection, RegimeCorrection
from metryki_source import detection_map
from model_snapshot import MalyViT, dekoduj
from pelne_latenty import GELU_FLOOR, capture_baseline, load_artifact, state_fingerprint, top1_metrics

EXPECTED_CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
EXPECTED_MANIFEST_SHA256 = "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7"
METHODS = (
    "baseline",
    "mean_in_sample",
    "regime_in_sample",
    "mean_group_holdout",
    "regime_group_holdout",
    "oracle_per_image",
)
EVALUATION_SOURCE_FILES = (
    "ewaluuj_korekty.py",
    "metryki_source.py",
    "dane.py",
    "pelne_latenty.py",
    "model_snapshot.py",
    "korekta_model.py",
    "artifact_runtime.py",
)
PACKAGE_RUNTIME_FILES = ("model_snapshot.py", "korekta_model.py", "artifact_runtime.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in EVALUATION_SOURCE_FILES}


def _package_runtime_hashes(package: Path) -> dict[str, str]:
    return {name: sha256(package / name) for name in PACKAGE_RUNTIME_FILES}


def _validate_resume_record(
    record: dict,
    row: dict,
    index: int,
    fold: int,
    evaluation_signature: str,
) -> None:
    identity = {
        "signature": record.get("evaluation_signature") == evaluation_signature,
        "index": record.get("index") == index,
        "image_id": record.get("image_id") == row["image_id"],
        "image_sha256": record.get("image_sha256") == row["image_sha256"],
        "frame_id": record.get("frame_id") == row["frame_id"],
        "legacy_bbox_flag": record.get("campaign_legacy_bbox_differs")
        == (row["bbox_xyxy_480"] != row["bbox_xyxy_480_campaign_legacy"]),
        "fold": record.get("fold") == fold,
        "no_individual_optimization": record.get(
            "individual_optimization_during_transfer_evaluation"
        )
        is False,
        "methods": set(record.get("methods", {})) == set(METHODS),
    }
    if not all(identity.values()):
        raise RuntimeError(f"Niespójny rekord wznowienia {index}: {identity}")
    for method in METHODS:
        detection = record["methods"][method]
        scalars = np.asarray([detection.get("confidence"), detection.get("iou")], dtype=np.float64)
        boxes = np.asarray(detection.get("all_boxes_xyxy"), dtype=np.float64)
        scores = np.asarray(detection.get("all_scores"), dtype=np.float64)
        output_hash = detection.get("output_tensor_sha256")
        if (
            not np.isfinite(scalars).all()
            or boxes.shape != (10, 4)
            or scores.shape != (10,)
            or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or not isinstance(output_hash, str)
            or len(output_hash) != 64
        ):
            raise RuntimeError(f"Uszkodzona detekcja {method} w rekordzie {index}")


def _load_base(checkpoint: Path, device: torch.device) -> MalyViT:
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Nieprawidłowy K1000")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"net", "krok"} or payload["krok"] != 1000:
        raise RuntimeError("Nieprawidłowy kontener K1000")
    model = MalyViT().float().eval().requires_grad_(False)
    model.load_state_dict(payload["net"], strict=True)
    return model.to(device)


def _corrections_from_npz(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if path.name.startswith("MEAN"):
        return MeanCorrection(torch.from_numpy(arrays["mean_delta"]), GELU_FLOOR).to(device).eval()
    return RegimeCorrection(
        z_mean=torch.from_numpy(arrays["z_mean"]),
        z_components=torch.from_numpy(arrays["z_components"]),
        feature_mean=torch.from_numpy(arrays["feature_mean"]),
        feature_scale=torch.from_numpy(arrays["feature_scale"]),
        assignment_centroids=torch.from_numpy(arrays["assignment_centroids"]),
        correction_by_regime_token=torch.from_numpy(arrays["correction_by_regime_token"]),
        grid_shape=tuple(int(v) for v in arrays["grid_shape"]),
        floor=GELU_FLOOR,
    ).to(device).eval().requires_grad_(False)


def _full_hook(model: MalyViT, image: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    hook = model.bloki[-1].mlp.act.register_forward_hook(
        lambda _module, _args, _natural: replacement
    )
    try:
        with torch.no_grad():
            output, _ = model(image)
    finally:
        hook.remove()
    return output


def _detection(output: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    top = top1_metrics(output, ground_truth)
    boxes, scores = dekoduj(output.detach(), k=10)
    return {
        **top,
        "all_boxes_xyxy": boxes[0].cpu().numpy().astype(float).tolist(),
        "all_scores": scores[0].cpu().numpy().astype(float).tolist(),
        "output_tensor_sha256": _tensor_sha(output),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"{q:.2f}": float(np.quantile(values, q))
        for q in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    }


def _comparison(records: list[dict], method: str, tolerance: float = 1e-6) -> dict:
    confidence_delta = np.asarray(
        [record["methods"][method]["confidence"] - record["methods"]["baseline"]["confidence"] for record in records]
    )
    iou_delta = np.asarray(
        [record["methods"][method]["iou"] - record["methods"]["baseline"]["iou"] for record in records]
    )

    def counts(values: np.ndarray) -> dict[str, int]:
        return {
            "improved": int((values > tolerance).sum()),
            "unchanged": int((np.abs(values) <= tolerance).sum()),
            "worse": int((values < -tolerance).sum()),
        }

    unchanged = (np.abs(confidence_delta) <= tolerance) & (np.abs(iou_delta) <= tolerance)
    pareto_improved = (
        (confidence_delta >= -tolerance)
        & (iou_delta >= -tolerance)
        & ~unchanged
    )
    pareto_worse = (
        (confidence_delta <= tolerance)
        & (iou_delta <= tolerance)
        & ~unchanged
    )
    mixed = ~(unchanged | pareto_improved | pareto_worse)
    both90 = np.asarray(
        [
            record["methods"][method]["confidence"] >= 0.90
            and record["methods"][method]["iou"] >= 0.90
            for record in records
        ],
        dtype=bool,
    )
    return {
        "confidence": counts(confidence_delta),
        "iou": counts(iou_delta),
        "joint_vs_baseline": {
            "pareto_improved": int(pareto_improved.sum()),
            "unchanged": int(unchanged.sum()),
            "pareto_worse": int(pareto_worse.sum()),
            "mixed": int(mixed.sum()),
        },
        "both90_count": int(both90.sum()),
        "both90_fraction": float(both90.mean()),
        "delta_confidence_quantiles": _quantiles(confidence_delta),
        "delta_iou_quantiles": _quantiles(iou_delta),
        "nonworse_on_both_for_every_image": bool(
            np.all((confidence_delta >= -tolerance) & (iou_delta >= -tolerance))
        ),
        "improves_both_for_every_image": bool(
            np.all((confidence_delta > tolerance) & (iou_delta > tolerance))
        ),
    }


def _per_image_relation(method: dict, baseline: dict, tolerance: float = 1e-6) -> dict:
    confidence_delta = float(method["confidence"] - baseline["confidence"])
    iou_delta = float(method["iou"] - baseline["iou"])

    def label(value: float) -> str:
        if value > tolerance:
            return "improved"
        if value < -tolerance:
            return "worse"
        return "unchanged"

    return {
        "delta_confidence": confidence_delta,
        "delta_iou": iou_delta,
        "confidence_relation": label(confidence_delta),
        "iou_relation": label(iou_delta),
    }


def _worst_exceptions(records: list[dict], method: str, count: int = 20) -> list[dict]:
    ranked = sorted(
        records,
        key=lambda record: (
            record["vs_baseline"][method]["delta_iou"],
            record["vs_baseline"][method]["delta_confidence"],
        ),
    )
    return [
        {
            "index": record["index"],
            "frame_id": record["frame_id"],
            "delta_iou": record["vs_baseline"][method]["delta_iou"],
            "delta_confidence": record["vs_baseline"][method]["delta_confidence"],
            "baseline": {
                "confidence": record["methods"]["baseline"]["confidence"],
                "iou": record["methods"]["baseline"]["iou"],
            },
            "corrected": {
                "confidence": record["methods"][method]["confidence"],
                "iou": record["methods"][method]["iou"],
            },
        }
        for record in ranked[:count]
    ]


def _attach_verification(package: Path, payload: dict) -> None:
    verification_path = package / "FRESH_LOAD_VERIFICATION.json"
    _json_atomic(verification_path, payload)
    manifest_path = package / "FILES_SHA256.json"
    hashes = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes[verification_path.name] = sha256(verification_path)
    _json_atomic(manifest_path, dict(sorted(hashes.items())))
    verify_package_files(package)


def _report(summary: dict) -> str:
    lines = [
        "# Ocena wspólnych korekt K1000 na dokładnych 1500 obrazach",
        "",
        "Metody A i B są dopasowane z oracle ruchów tej samej bazy. Ich wynik `in-sample` nie jest dowodem generalizacji. Wiersze group-holdout powstały bez użycia filmów testowych podczas dopasowania korekty.",
        "IoU używa prawidłowego skalowania GT osobno po osiach. Dla 203 klatek różni się ono od historycznej, stałej skali 2/3 użytej przez loader kampanii K1000; raport JSON podaje oba podzbiory osobno.",
        "",
        "| metoda | AP50 | AP75 | mAP50:95 | oba progi 0,90 | lepiej IoU | gorzej IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = summary["methods"][method]
        comparison = summary["comparisons"].get(method)
        both = comparison["both90_count"] if comparison else metrics["both90_count"]
        better = comparison["iou"]["improved"] if comparison else "-"
        worse = comparison["iou"]["worse"] if comparison else "-"
        lines.append(
            f"| {method} | {metrics['detection']['AP50']:.6f} | {metrics['detection']['AP75']:.6f} | "
            f"{metrics['detection']['mAP50_95']:.6f} | {both} | {better} | {worse} |"
        )
    lines.extend(
        [
            "",
            "`oracle_per_image` korzysta z etykiety każdego obrazu i jest wyłącznie górnym odniesieniem. Podczas oceny metod wspólnych nie wykonywano żadnej indywidualnej optymalizacji.",
            "",
            f"Czy metoda A pomaga wszystkim bez pogorszenia confidence lub IoU: `{summary['comparisons']['mean_in_sample']['nonworse_on_both_for_every_image']}`.",
            f"Czy metoda B pomaga wszystkim bez pogorszenia confidence lub IoU: `{summary['comparisons']['regime_in_sample']['nonworse_on_both_for_every_image']}`.",
            "",
        ]
    )
    for method in ("mean_in_sample", "regime_in_sample"):
        lines.extend([f"## Najgorsze wyjątki IoU: {method}", ""])
        for item in summary["worst_exceptions"][method][:10]:
            lines.append(
                f"- indeks {item['index']}, `{item['frame_id']}`: delta IoU "
                f"{item['delta_iou']:+.6f}, delta confidence {item['delta_confidence']:+.6f}."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--local-image-root", type=Path)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--regimes", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Ewaluacja wymaga GPU CUDA")
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    if sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Nieprawidłowy manifest")
    rows = load_manifest(args.manifest)
    if len(rows) != 1500:
        raise RuntimeError(len(rows))

    mean_package = args.packages / "model_mean"
    regime_package = args.packages / "model_regime"
    mean_model, mean_config = load_executable_artifact(mean_package, device)
    regime_model, regime_config = load_executable_artifact(regime_package, device)
    if mean_config["method"] != "mean600x768" or regime_config["method"] != "inferred_regime":
        raise RuntimeError("Zamienione pakiety korekt")
    base = _load_base(args.checkpoint, device)
    base_hash_before = state_fingerprint(base)
    mean_hash_before = state_fingerprint(mean_model.base)
    regime_hash_before = state_fingerprint(regime_model.base)

    folds_path = args.regimes / "FOLDS.json"
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    image_fold = [int(value) for value in folds["image_fold"]]
    if len(image_fold) != 1500:
        raise RuntimeError("Niewłaściwy podział holdout")
    fold_mean = []
    fold_regime = []
    fold_hashes = []
    for fold in range(int(folds["folds"])):
        mean_path = args.regimes / f"fold_{fold}" / "MEAN_DELTA.npz"
        regime_path = args.regimes / f"fold_{fold}" / "REGIME_MODEL.npz"
        fold_mean.append(_corrections_from_npz(mean_path, device))
        fold_regime.append(_corrections_from_npz(regime_path, device))
        fold_hashes.extend([sha256(mean_path), sha256(regime_path)])

    signature_payload = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "manifest": EXPECTED_MANIFEST_SHA256,
        "mean_correction": mean_config["correction_sha256"],
        "regime_correction": regime_config["correction_sha256"],
        "fold_models": fold_hashes,
        "fold_definition_sha256": sha256(folds_path),
        "evaluation_source_sha256": _source_hashes(),
        "package_runtime_sha256": {
            "mean": _package_runtime_hashes(mean_package),
            "regime": _package_runtime_hashes(regime_package),
        },
        "decoder_padding_policy": "scores <= 0 are excluded from AP/mAP",
        "no_individual_optimization": True,
    }
    evaluation_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_config = {**signature_payload, "signature": evaluation_signature}
    evaluation_config_path = args.output / "EVALUATION_CONFIG.json"
    if evaluation_config_path.exists():
        existing_config = json.loads(evaluation_config_path.read_text(encoding="utf-8"))
        if existing_config != evaluation_config:
            raise RuntimeError("Istniejąca ewaluacja używa innego kodu lub wejść")
    else:
        _json_atomic(evaluation_config_path, evaluation_config)
    image_source = LocalImageSource(args.local_image_root) if args.local_image_root else TarImageSource(args.archive)

    try:
        for index, row in enumerate(rows):
            record_path = args.output / "records" / f"{index:06d}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                _validate_resume_record(
                    record, row, index, image_fold[index], evaluation_signature
                )
                continue
            payload = image_source.read_bytes(row["image_name"])
            if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
                raise RuntimeError(f"Hash obrazu {index}")
            rgb = decode_resize(payload)
            image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(
                device=device, dtype=torch.float32
            ) / 255.0
            ground_truth = torch.tensor(row["bbox_xyxy_480"], device=device, dtype=torch.float32)

            baseline_output, z0, _q0 = capture_baseline(base, image)
            with torch.no_grad():
                mean_output, mean_natural, _mean_corrected = mean_model(image)
                regime_output, regime_natural, _regime_corrected = regime_model(image)
            torch.testing.assert_close(mean_natural, z0, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(regime_natural, z0, rtol=2e-5, atol=2e-5)

            fold = image_fold[index]
            with torch.no_grad():
                holdout_mean_z = fold_mean[fold](z0)
                holdout_regime_z = fold_regime[fold](z0)
            holdout_mean_output = _full_hook(base, image, holdout_mean_z)
            holdout_regime_output = _full_hook(base, image, holdout_regime_z)

            oracle_artifact = load_artifact(args.production / "latents" / f"{index:06d}.npz")
            stored_before = torch.from_numpy(oracle_artifact["before"])[None].to(device)
            torch.testing.assert_close(stored_before, z0, rtol=2e-5, atol=2e-5)
            oracle_z = torch.from_numpy(oracle_artifact["after"])[None].to(device)
            oracle_output = _full_hook(base, image, oracle_z)

            outputs = {
                "baseline": baseline_output,
                "mean_in_sample": mean_output,
                "regime_in_sample": regime_output,
                "mean_group_holdout": holdout_mean_output,
                "regime_group_holdout": holdout_regime_output,
                "oracle_per_image": oracle_output,
            }
            methods = {name: _detection(output, ground_truth) for name, output in outputs.items()}
            record = {
                "index": index,
                "image_id": row["image_id"],
                "image_name": row["image_name"],
                "image_sha256": row["image_sha256"],
                "frame_id": row["frame_id"],
                "bbox_xyxy_480": row["bbox_xyxy_480"],
                "bbox_xyxy_480_campaign_legacy": row[
                    "bbox_xyxy_480_campaign_legacy"
                ],
                "campaign_legacy_bbox_differs": row["bbox_xyxy_480"]
                != row["bbox_xyxy_480_campaign_legacy"],
                "ground_truth_resize": "per-axis from decoded source dimensions to 480x320",
                "fold": fold,
                "oracle_after_step": int(oracle_artifact["metadata"]["after_step"]),
                "oracle_status": oracle_artifact["metadata"]["status"],
                "methods": methods,
                "vs_baseline": {
                    name: _per_image_relation(methods[name], methods["baseline"])
                    for name in METHODS
                    if name != "baseline"
                },
                "evaluation_signature": evaluation_signature,
                "individual_optimization_during_transfer_evaluation": False,
            }
            _json_atomic(record_path, record)
            if (index + 1) % 25 == 0:
                _json_atomic(
                    args.output / "PROGRESS.json",
                    {"status": "running", "completed": index + 1, "total": 1500},
                )

        records = [
            json.loads((args.output / "records" / f"{index:06d}.json").read_text(encoding="utf-8"))
            for index in range(1500)
        ]
        ground_truths = [np.asarray([row["bbox_xyxy_480"]], np.float32) for row in rows]
        method_summaries = {}
        for method in METHODS:
            detections = [
                (
                    np.asarray(record["methods"][method]["all_boxes_xyxy"], np.float32),
                    np.asarray(record["methods"][method]["all_scores"], np.float32),
                )
                for record in records
            ]
            confidence = np.asarray([record["methods"][method]["confidence"] for record in records])
            iou = np.asarray([record["methods"][method]["iou"] for record in records])
            method_summaries[method] = {
                "detection": detection_map(detections, ground_truths),
                "confidence_quantiles": _quantiles(confidence),
                "iou_quantiles": _quantiles(iou),
                "both90_count": int(((confidence >= 0.90) & (iou >= 0.90)).sum()),
            }
        baseline_detection = method_summaries["baseline"]["detection"]
        for method in METHODS:
            current = method_summaries[method]["detection"]
            method_summaries[method]["aggregate_delta_vs_baseline"] = {
                key: float(current[key] - baseline_detection[key])
                for key in ("AP50", "AP75", "mAP50_95")
            }
        comparisons = {
            method: _comparison(records, method)
            for method in METHODS
            if method != "baseline"
        }
        resize_subgroups = {}
        for label, legacy_differs in (
            ("campaign_legacy_bbox_same", False),
            ("campaign_legacy_bbox_different_corrected_gt", True),
        ):
            selected = [
                index
                for index, record in enumerate(records)
                if bool(record["campaign_legacy_bbox_differs"]) is legacy_differs
            ]
            subgroup_records = [records[index] for index in selected]
            subgroup_ground_truths = [ground_truths[index] for index in selected]
            subgroup_methods = {}
            for method in METHODS:
                detections = [
                    (
                        np.asarray(record["methods"][method]["all_boxes_xyxy"], np.float32),
                        np.asarray(record["methods"][method]["all_scores"], np.float32),
                    )
                    for record in subgroup_records
                ]
                confidence = np.asarray(
                    [record["methods"][method]["confidence"] for record in subgroup_records]
                )
                iou = np.asarray(
                    [record["methods"][method]["iou"] for record in subgroup_records]
                )
                subgroup_methods[method] = {
                    "detection": detection_map(detections, subgroup_ground_truths),
                    "both90_count": int(((confidence >= 0.90) & (iou >= 0.90)).sum()),
                    "confidence_quantiles": _quantiles(confidence),
                    "iou_quantiles": _quantiles(iou),
                }
            resize_subgroups[label] = {
                "count": len(selected),
                "legacy_box_used_as_gt": False,
                "methods": subgroup_methods,
                "comparisons": {
                    method: _comparison(subgroup_records, method)
                    for method in METHODS
                    if method != "baseline"
                },
            }
        summary = {
            "status": "complete",
            "count": 1500,
            "evaluation_signature": evaluation_signature,
            "methods": method_summaries,
            "comparisons": comparisons,
            "ground_truth_resize_subgroups": resize_subgroups,
            "worst_exceptions": {
                method: _worst_exceptions(records, method)
                for method in ("mean_in_sample", "regime_in_sample", "mean_group_holdout", "regime_group_holdout")
            },
            "detection_matching": (
                "one class; all top-10 local maxima; global score ranking; one-to-one matching "
                "per image; score<=0 decoder padding excluded; COCO 101-point AP; "
                "mAP uses IoU 0.50:0.05:0.95"
            ),
            "in_sample_is_not_generalization": True,
            "group_holdout_unit": "source video",
            "individual_optimization_during_transfer_evaluation": False,
        }
        _json_atomic(args.output / "EVALUATION_SUMMARY.json", summary)
        (args.output / "REPORT.md").write_text(_report(summary), encoding="utf-8")

        # Zawsze wykonaj niezależny fresh-load na obrazie 0, także po resume.
        row = rows[0]
        payload = image_source.read_bytes(row["image_name"])
        rgb = decode_resize(payload)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(
            device=device, dtype=torch.float32
        ) / 255.0
        ground_truth = torch.tensor(row["bbox_xyxy_480"], device=device, dtype=torch.float32)
        stored_record = records[0]
        verification = {}
        for label, package in (("mean_in_sample", mean_package), ("regime_in_sample", regime_package)):
            first, first_config = load_executable_artifact(package, device)
            second, second_config = load_executable_artifact(package, device)
            with torch.no_grad():
                first_output, first_z0, first_corrected = first(image)
                second_output, second_z0, second_corrected = second(image)
            torch.testing.assert_close(first_output, second_output, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(first_z0, second_z0, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(first_corrected, second_corrected, rtol=2e-5, atol=2e-5)
            metrics = _detection(first_output, ground_truth)
            stored = stored_record["methods"][label]
            if abs(metrics["confidence"] - stored["confidence"]) > 2e-5 or abs(metrics["iou"] - stored["iou"]) > 2e-5:
                raise RuntimeError(f"Fresh-load zmienił metryki {label}")
            verification[label] = {
                "index": 0,
                "image_sha256": row["image_sha256"],
                "two_independent_loads_output_bitwise_equal": _tensor_sha(first_output) == _tensor_sha(second_output),
                "two_independent_loads_max_abs_error": float((first_output - second_output).abs().max().item()),
                "output_tensor_sha256": _tensor_sha(first_output),
                "natural_z0_sha256": _tensor_sha(first_z0),
                "corrected_z_sha256": _tensor_sha(first_corrected),
                "confidence": metrics["confidence"],
                "iou": metrics["iou"],
                "config_equal": first_config == second_config,
            }
            _attach_verification(package, verification[label])
        _json_atomic(args.output / "FRESH_LOAD_VERIFICATION.json", verification)

        fingerprints_after = {
            "baseline": state_fingerprint(base),
            "mean": state_fingerprint(mean_model.base),
            "regime": state_fingerprint(regime_model.base),
        }
        expected_fingerprints = {
            "baseline": base_hash_before,
            "mean": mean_hash_before,
            "regime": regime_hash_before,
        }
        if fingerprints_after != expected_fingerprints:
            raise RuntimeError("Wagi lub bufory zmieniły się podczas ewaluacji")
        _json_atomic(
            args.output / "MODEL_FREEZE_VERIFICATION.json",
            {
                "before": expected_fingerprints,
                "after": fingerprints_after,
                "unchanged": True,
                "all_parameters_require_grad_false": True,
            },
        )
        _json_atomic(args.output / "PROGRESS.json", {"status": "complete", "completed": 1500, "total": 1500})
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        image_source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
