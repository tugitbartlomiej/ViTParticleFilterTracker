"""Odkrywa reżimy z z0 i oracle ruchu; przypisanie eksportuje wyłącznie funkcję z0/kontekstu."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dane import load_manifest
from pelne_latenty import load_artifact


def _sample_positions(
    training_images: np.ndarray, tokens: int, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(count), len(training_images) * tokens)
    flat = rng.choice(len(training_images) * tokens, size=count, replace=False)
    return training_images[flat // tokens], flat % tokens


def _randomized_pca(samples: np.ndarray, components: int, rng: np.random.Generator):
    samples = np.asarray(samples, dtype=np.float32)
    components = min(int(components), samples.shape[0] - 1, samples.shape[1])
    if components < 1:
        raise ValueError((samples.shape, components))
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = samples - mean
    width = min(samples.shape[1], components + 8)
    omega = rng.standard_normal((samples.shape[1], width), dtype=np.float32)
    basis = centered @ omega
    basis, _ = np.linalg.qr(basis, mode="reduced")
    small = basis.T @ centered
    _left, singular, right = np.linalg.svd(small, full_matrices=False)
    return mean, right[:components].astype(np.float32), singular[:components].astype(np.float32)


def _squared_distance(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return (
        np.sum(features * features, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None]
        - 2.0 * features @ centers.T
    )


def _kmeans(features: np.ndarray, clusters: int, rng: np.random.Generator, iterations: int = 25):
    features = np.asarray(features, dtype=np.float32)
    clusters = min(int(clusters), len(features))
    centers = np.empty((clusters, features.shape[1]), np.float32)
    centers[0] = features[int(rng.integers(len(features)))]
    nearest = np.sum((features - centers[0]) ** 2, axis=1)
    for cluster in range(1, clusters):
        total = float(nearest.sum())
        index = int(rng.integers(len(features))) if total <= 0 else int(rng.choice(len(features), p=nearest / total))
        centers[cluster] = features[index]
        nearest = np.minimum(nearest, np.sum((features - centers[cluster]) ** 2, axis=1))
    labels = np.zeros(len(features), np.int32)
    for _ in range(iterations):
        labels = _squared_distance(features, centers).argmin(axis=1).astype(np.int32)
        new = np.empty_like(centers)
        for cluster in range(clusters):
            selected = features[labels == cluster]
            if len(selected):
                new[cluster] = selected.mean(axis=0)
            else:
                farthest = int(np.argmax(_squared_distance(features, centers).min(axis=1)))
                new[cluster] = features[farthest]
        shift = float(np.max(np.linalg.norm(new - centers, axis=1)))
        centers = new
        if shift < 1e-5:
            break
    return centers, labels


def _projection(values: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (values - mean) @ components.T


def _position_features(tokens: int, grid_shape: tuple[int, int]) -> np.ndarray:
    height, width = grid_shape
    if height * width != tokens:
        raise ValueError((tokens, grid_shape))
    index = np.arange(tokens)
    return np.column_stack(
        [
            (index // width) / max(height - 1, 1),
            (index % width) / max(width - 1, 1),
        ]
    ).astype(np.float32)


def fit_regime_model(
    z0: np.ndarray,
    delta: np.ndarray,
    training_images: np.ndarray,
    *,
    components: int = 16,
    regimes: int = 32,
    pca_sample: int = 50_000,
    cluster_sample: int = 100_000,
    seed: int = 20261008,
    grid_shape: tuple[int, int] = (20, 30),
    shrinkage: float = 20.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Ruch odkrywa etykiety; eksportowany klasyfikator etykiet widzi tylko z0."""
    if z0.shape != delta.shape or z0.ndim != 3:
        raise ValueError((z0.shape, delta.shape))
    training_images = np.asarray(training_images, dtype=np.int32)
    images, tokens, dimensions = z0.shape
    if not len(training_images):
        raise ValueError("Pusty zbiór treningowy")
    rng = np.random.default_rng(seed)
    image_sample, token_sample = _sample_positions(training_images, tokens, pca_sample, rng)
    z_sample = z0[image_sample, token_sample]
    delta_sample = delta[image_sample, token_sample]
    magnitude_sample = np.linalg.norm(delta_sample, axis=1)
    direction_sample = delta_sample / np.maximum(magnitude_sample[:, None], 1e-12)
    z_mean, z_components, z_singular = _randomized_pca(z_sample, components, rng)
    direction_mean, direction_components, direction_singular = _randomized_pca(
        direction_sample, components, rng
    )

    image_cluster, token_cluster = _sample_positions(training_images, tokens, cluster_sample, rng)
    z_cluster = z0[image_cluster, token_cluster]
    delta_cluster = delta[image_cluster, token_cluster]
    magnitude_cluster = np.linalg.norm(delta_cluster, axis=1)
    direction_cluster = delta_cluster / np.maximum(magnitude_cluster[:, None], 1e-12)
    discovery = np.column_stack(
        [
            _projection(z_cluster, z_mean, z_components),
            _projection(direction_cluster, direction_mean, direction_components),
            np.log1p(magnitude_cluster),
        ]
    ).astype(np.float32)
    discovery_mean = discovery.mean(axis=0)
    discovery_scale = discovery.std(axis=0)
    discovery_scale[discovery_scale < 1e-8] = 1.0
    discovery_standard = (discovery - discovery_mean) / discovery_scale
    discovery_centroids, _ = _kmeans(discovery_standard, regimes, rng)
    regimes = len(discovery_centroids)

    position = _position_features(tokens, grid_shape)
    z_projected_sample = _projection(z_cluster, z_mean, z_components)
    contexts = np.empty_like(z_projected_sample)
    for image in np.unique(image_cluster):
        context = _projection(z0[image], z_mean, z_components).mean(axis=0)
        contexts[image_cluster == image] = context
    assignment_sample = np.column_stack(
        [z_projected_sample, position[token_cluster], contexts]
    ).astype(np.float32)
    feature_mean = assignment_sample.mean(axis=0)
    feature_scale = assignment_sample.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0

    mean_delta = delta[training_images].mean(axis=0, dtype=np.float64).astype(np.float32)
    assignment_sum = np.zeros((regimes, assignment_sample.shape[1]), np.float64)
    regime_counts = np.zeros(regimes, np.int64)
    correction_sum = np.zeros((regimes, tokens, dimensions), np.float64)
    correction_count = np.zeros((regimes, tokens), np.int32)
    token_index = np.arange(tokens)
    discovery_label_by_image: dict[int, np.ndarray] = {}
    assignment_feature_by_image: dict[int, np.ndarray] = {}
    for image in training_images:
        zp = _projection(z0[image], z_mean, z_components)
        movement = delta[image]
        magnitude = np.linalg.norm(movement, axis=1)
        direction = movement / np.maximum(magnitude[:, None], 1e-12)
        discovery_features = np.column_stack(
            [
                zp,
                _projection(direction, direction_mean, direction_components),
                np.log1p(magnitude),
            ]
        ).astype(np.float32)
        discovery_features = (discovery_features - discovery_mean) / discovery_scale
        labels = _squared_distance(discovery_features, discovery_centroids).argmin(axis=1)
        context = np.broadcast_to(zp.mean(axis=0), (tokens, zp.shape[1]))
        assignment_features = np.column_stack([zp, position, context]).astype(np.float32)
        assignment_features = (assignment_features - feature_mean) / feature_scale
        discovery_label_by_image[int(image)] = labels.astype(np.int16)
        assignment_feature_by_image[int(image)] = assignment_features
        for regime in np.unique(labels):
            selected = token_index[labels == regime]
            assignment_sum[regime] += assignment_features[selected].sum(axis=0)
            regime_counts[regime] += len(selected)
            correction_sum[regime, selected] += movement[selected]
            correction_count[regime, selected] += 1
    assignment_centroids = (
        assignment_sum / np.maximum(regime_counts[:, None], 1)
    ).astype(np.float32)
    correction = (
        correction_sum + shrinkage * mean_delta[None]
    ) / (correction_count[..., None] + shrinkage)
    correction = correction.astype(np.float32)

    correct = 0
    total = 0
    confusion = np.zeros((regimes, regimes), np.int64)
    for image in training_images:
        truth = discovery_label_by_image[int(image)]
        predicted = _squared_distance(
            assignment_feature_by_image[int(image)], assignment_centroids
        ).argmin(axis=1)
        correct += int((truth == predicted).sum())
        total += tokens
        np.add.at(confusion, (truth, predicted), 1)

    model = {
        "mean_delta": mean_delta,
        "z_mean": z_mean,
        "z_components": z_components,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "assignment_centroids": assignment_centroids,
        "correction_by_regime_token": correction,
        "regime_counts": regime_counts,
        "regime_token_counts": correction_count,
        "discovery_mean": discovery_mean.astype(np.float32),
        "discovery_scale": discovery_scale.astype(np.float32),
        "discovery_centroids": discovery_centroids.astype(np.float32),
        "direction_mean": direction_mean,
        "direction_components": direction_components,
        "z_singular_values": z_singular,
        "direction_singular_values": direction_singular,
        "grid_shape": np.asarray(grid_shape, dtype=np.int32),
        "shrinkage": np.asarray([shrinkage], dtype=np.float32),
    }
    diagnostics = {
        "training_images": int(len(training_images)),
        "training_particles": int(len(training_images) * tokens),
        "dimensions": dimensions,
        "components": int(z_components.shape[0]),
        "regimes": regimes,
        "discovery_uses_oracle_delta": True,
        "discovery_inputs": ["natural_z0", "oracle_delta_direction", "oracle_delta_magnitude"],
        "assignment_inputs": ["natural_z0", "token_row_col", "image_z0_context"],
        "assignment_uses_oracle_delta": False,
        "training_assignment_accuracy_to_discovered_regime": correct / max(total, 1),
        "regime_counts": regime_counts.tolist(),
        "assignment_confusion": confusion.tolist(),
        "shrinkage_to_token_mean": shrinkage,
    }
    return model, diagnostics


def _save_model(path: Path, model: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model)


def _balanced_group_folds(groups: list[str], folds: int = 3) -> np.ndarray:
    unique, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    totals = np.zeros(folds, np.int64)
    mapping: dict[str, int] = {}
    for item in order:
        fold = int(np.argmin(totals))
        mapping[str(unique[item])] = fold
        totals[fold] += counts[item]
    return np.asarray([mapping[group] for group in groups], dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    if len(rows) != 1500:
        raise RuntimeError(len(rows))
    z0 = np.empty((1500, 600, 768), np.float32)
    delta = np.empty_like(z0)
    for index in range(1500):
        artifact = load_artifact(args.production / "latents" / f"{index:06d}.npz")
        z0[index] = artifact["before"]
        delta[index] = artifact["delta"]
    groups = [row["frame_id"].split("/")[2] for row in rows]
    fold_of_image = _balanced_group_folds(groups, args.folds)
    args.output.mkdir(parents=True, exist_ok=True)
    all_model, all_diagnostics = fit_regime_model(
        z0, delta, np.arange(1500), seed=20261008
    )
    _save_model(args.output / "REGIME_MODEL_ALL.npz", all_model)
    np.savez_compressed(args.output / "MEAN_DELTA_ALL.npz", mean_delta=all_model["mean_delta"])
    diagnostics = {"all": all_diagnostics, "folds": {}}
    for fold in range(args.folds):
        training = np.flatnonzero(fold_of_image != fold)
        held_out = np.flatnonzero(fold_of_image == fold)
        model, report = fit_regime_model(
            z0,
            delta,
            training,
            seed=20261008 + fold + 1,
            pca_sample=40_000,
            cluster_sample=80_000,
        )
        fold_dir = args.output / f"fold_{fold}"
        _save_model(fold_dir / "REGIME_MODEL.npz", model)
        np.savez_compressed(fold_dir / "MEAN_DELTA.npz", mean_delta=model["mean_delta"])
        report["held_out_images"] = int(len(held_out))
        report["held_out_groups"] = sorted(set(groups[index] for index in held_out))
        diagnostics["folds"][str(fold)] = report
    folds_payload = {
        "strategy": "balanced group holdout by source video; no video crosses folds",
        "folds": args.folds,
        "image_fold": fold_of_image.tolist(),
        "groups": groups,
    }
    (args.output / "FOLDS.json").write_text(
        json.dumps(folds_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "REGIME_DIAGNOSTICS.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "all": all_diagnostics, "folds": args.folds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
