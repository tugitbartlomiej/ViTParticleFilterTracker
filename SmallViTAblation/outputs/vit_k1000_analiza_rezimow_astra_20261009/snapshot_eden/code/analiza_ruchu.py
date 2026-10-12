"""Strumieniowa analiza rzeczywistego ruchu 900 000 cząsteczek na wspólnej osi czasu."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pelne_latenty import load_artifact

SEGMENTS = ("k0_k100", "k100_k500", "k0_k500", "k0_after")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"{q:.2f}": float(np.quantile(values, q))
        for q in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    particle_norms = {
        name: np.empty((1500, 600), np.float32) for name in SEGMENTS
    }
    image_rms = {name: np.empty(1500, np.float32) for name in SEGMENTS}
    token_norm_sum = {name: np.zeros(600, np.float64) for name in SEGMENTS}
    coordinate_sum = {name: np.zeros(768, np.float64) for name in SEGMENTS}
    coordinate_square_sum = {name: np.zeros(768, np.float64) for name in SEGMENTS}
    cosine_early_late = np.empty((1500, 600), np.float32)
    after_steps = np.empty(1500, np.int32)

    for index in range(1500):
        artifact = load_artifact(args.production / "latents" / f"{index:06d}.npz")
        if artifact["trajectory_steps"].tolist() != [100, 500]:
            raise RuntimeError(f"Niezgodne kroki trajektorii obrazu {index}")
        z0 = artifact["before"].astype(np.float32, copy=False)
        z100, z500 = artifact["trajectory"].astype(np.float32, copy=False)
        after = artifact["after"].astype(np.float32, copy=False)
        movements = {
            "k0_k100": z100 - z0,
            "k100_k500": z500 - z100,
            "k0_k500": z500 - z0,
            "k0_after": after - z0,
        }
        for name, movement in movements.items():
            norms = np.linalg.norm(movement, axis=1)
            particle_norms[name][index] = norms
            image_rms[name][index] = np.sqrt(np.mean(movement.astype(np.float64) ** 2))
            token_norm_sum[name] += norms
            coordinate_sum[name] += movement.sum(axis=0, dtype=np.float64)
            coordinate_square_sum[name] += np.square(movement, dtype=np.float64).sum(
                axis=0, dtype=np.float64
            )
        early = movements["k0_k100"]
        late = movements["k100_k500"]
        numerator = np.sum(early * late, axis=1, dtype=np.float64)
        denominator = np.linalg.norm(early, axis=1) * np.linalg.norm(late, axis=1)
        cosine_early_late[index] = (numerator / np.maximum(denominator, 1e-12)).astype(
            np.float32
        )
        after_steps[index] = int(artifact["metadata"]["after_step"])

    count = 1500 * 600
    summary = {
        "status": "complete",
        "images": 1500,
        "particles": count,
        "dimensions": 768,
        "common_real_states": [0, 100, 500],
        "terminal_padding": False,
        "after_is_per_image_visited_state": True,
        "segments": {
            name: {
                "particle_l2_quantiles": _quantiles(particle_norms[name].reshape(-1)),
                "image_rms_quantiles": _quantiles(image_rms[name]),
                "zero_movement_particles": int((particle_norms[name] == 0).sum()),
            }
            for name in SEGMENTS
        },
        "early_late_cosine_quantiles": _quantiles(cosine_early_late.reshape(-1)),
        "early_late_opposite_direction_count": int((cosine_early_late < 0).sum()),
        "after_step_quantiles": _quantiles(after_steps),
        "interpretation": (
            "k0/k100/k500 are comparable actual shared steps for all images; k0_after uses "
            "each image's recorded best visited step and is not a shared-time state"
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "after_step": after_steps,
        "cosine_early_late_by_image_token": cosine_early_late,
    }
    for name in SEGMENTS:
        mean = coordinate_sum[name] / count
        variance = coordinate_square_sum[name] / count - mean * mean
        arrays[f"particle_l2_{name}"] = particle_norms[name]
        arrays[f"image_rms_{name}"] = image_rms[name]
        arrays[f"token_mean_l2_{name}"] = (token_norm_sum[name] / 1500).astype(np.float32)
        arrays[f"coordinate_mean_delta_{name}"] = mean.astype(np.float32)
        arrays[f"coordinate_std_delta_{name}"] = np.sqrt(np.maximum(variance, 0)).astype(
            np.float32
        )
    np.savez_compressed(args.output / "MOVEMENT_ARRAYS.npz", **arrays)
    (args.output / "MOVEMENT_STATS.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
