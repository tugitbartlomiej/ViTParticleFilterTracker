"""Testuje DMD/DMDc na prawdziwych przejściach; wyciek delta(t+1) obala test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analiza_regimow import _projection, _randomized_pca
from pelne_latenty import load_artifact


def _ridge(features: np.ndarray, targets: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    gram = features.T @ features
    scale = float(np.trace(gram) / max(len(gram), 1))
    gram.flat[:: len(gram) + 1] += ridge * max(scale, 1e-12)
    return np.linalg.solve(gram, features.T @ targets)


def _metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    error = target - predicted
    mse = float(np.mean(error * error))
    baseline = float(np.mean((target - target.mean(axis=0)) ** 2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "r2": 1.0 - mse / max(baseline, 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--control-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    panel = json.loads(args.control_panel.read_text(encoding="utf-8"))
    trajectories = []
    controls = []
    trajectory_images = []
    for image_text, expected_tokens in panel.items():
        image = int(image_text)
        artifact = load_artifact(args.production / "latents" / f"{image:06d}.npz")
        tokens = artifact.get("control_tokens")
        if tokens is None or tokens.tolist() != expected_tokens:
            raise RuntimeError(f"Brak lub zły panel sterowania dla obrazu {image}")
        z = artifact["control_z"]
        gradient = artifact["control_gradient_before_step"]
        moment1 = artifact["control_adam_m1_before_step"]
        moment2 = artifact["control_adam_m2_before_step"]
        if z.shape[0] != gradient.shape[0] + 1:
            raise RuntimeError(f"Zła oś czasu sterowania dla obrazu {image}")
        for token_offset in range(len(tokens)):
            trajectories.append(z[:, token_offset])
            controls.append(
                np.concatenate(
                    [gradient[:, token_offset], moment1[:, token_offset], moment2[:, token_offset]],
                    axis=1,
                )
            )
            trajectory_images.append(image)
    z = np.stack(trajectories).astype(np.float32)
    u = np.stack(controls).astype(np.float32)
    trajectory_images = np.asarray(trajectory_images)
    if z.shape[1] != 501 or u.shape[1] != 500:
        raise RuntimeError((z.shape, u.shape))

    unique_images = np.unique(trajectory_images)
    held_out_images = unique_images[::5]
    held_out_trajectory = np.isin(trajectory_images, held_out_images)
    train_trajectory = ~held_out_trajectory
    rng = np.random.default_rng(20261008)
    state_train_rows = z[train_trajectory].reshape(-1, z.shape[-1])
    state_sample = state_train_rows[
        rng.choice(len(state_train_rows), size=min(50_000, len(state_train_rows)), replace=False)
    ]
    state_mean, state_components, singular = _randomized_pca(state_sample, args.rank, rng)
    control_train_rows = u[train_trajectory].reshape(-1, u.shape[-1])
    control_sample = control_train_rows[
        rng.choice(len(control_train_rows), size=min(50_000, len(control_train_rows)), replace=False)
    ]
    control_mean, control_components, control_singular = _randomized_pca(
        control_sample, args.rank, rng
    )

    x = _projection(z[:, :-1].reshape(-1, z.shape[-1]), state_mean, state_components)
    y = _projection(z[:, 1:].reshape(-1, z.shape[-1]), state_mean, state_components)
    control = _projection(u.reshape(-1, u.shape[-1]), control_mean, control_components)
    pair_train = np.repeat(train_trajectory, 500)
    pair_test = np.repeat(held_out_trajectory, 500)
    ones = np.ones((len(x), 1), np.float32)
    dmd_features = np.concatenate([x, ones], axis=1)
    dmdc_features = np.concatenate([x, control, ones], axis=1)
    dmd_coef = _ridge(dmd_features[pair_train], y[pair_train])
    dmdc_coef = _ridge(dmdc_features[pair_train], y[pair_train])

    prediction = {
        "persistence": x[pair_test],
        "dmd": dmd_features[pair_test] @ dmd_coef,
        "dmdc": dmdc_features[pair_test] @ dmdc_coef,
    }
    report = {
        "status": "complete",
        "implementation": "NumPy CPU reduced ridge least squares; no GPU claim",
        "equation": "z_(t+1) = A z_t + B u_t",
        "control_alignment": "raw gradient and Adam m1/m2 before optimizer.step at t predict z_(t+1)",
        "uses_delta_t_plus_1_as_control": False,
        "crosses_trajectory_boundaries": False,
        "trajectory_count": int(len(z)),
        "transition_count": int(len(z) * 500),
        "held_out_images": held_out_images.tolist(),
        "held_out_trajectories": int(held_out_trajectory.sum()),
        "state_rank": int(len(state_components)),
        "control_rank": int(len(control_components)),
        "held_out_metrics_reduced_state": {
            name: _metrics(y[pair_test], values) for name, values in prediction.items()
        },
        "not_used_for_transfer_rule": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "DMDC_MODEL.npz",
        state_mean=state_mean,
        state_components=state_components,
        state_singular_values=singular,
        control_mean=control_mean,
        control_components=control_components,
        control_singular_values=control_singular,
        dmd_coefficients=dmd_coef,
        dmdc_coefficients=dmdc_coef,
    )
    (args.output / "DMDC_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
