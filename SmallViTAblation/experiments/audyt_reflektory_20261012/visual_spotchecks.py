"""Read-only checks of existing film-17/18 caches and the two endpoint weights.

No model forward pass, training, or rendering. Only the requested JSON is written.
Run from any directory: python visual_spotchecks.py
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
FILMS = ROOT / "outputs/porownanie_szklo_20261010/film_krysztal"
WEIGHTS = ROOT / "assets/MalyViT_35000_od_zera_s47/wagi"
OUT = ROOT / "outputs/audyt_reflektory_20261012/visual_spotchecks.json"


def unit_cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def layer_standardize(a):
    centered = a - a.mean(-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)


def ratio(a0, a1, focal, gamma=None):
    before = a0 - a0[focal]
    after = a1 - a1[focal]
    if gamma is not None:
        before, after = before * gamma, after * gamma
    return np.linalg.norm(after, axis=-1) / np.maximum(np.linalg.norm(before, axis=-1), 1e-12)


def file_record(path):
    s = path.stat()
    return {"path": str(path), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def main():
    with np.load(FILMS / "szescian.npz") as d:
        steps = d["kroki"].astype(int)
        tool = d["narzedzie_0"].astype(bool)
    grid = np.stack(np.divmod(np.arange(600), 30)[::-1], axis=1)
    focal = int(np.flatnonzero(tool)[np.argmin(np.linalg.norm(grid[tool] - grid[tool].mean(0), axis=1))])
    selected = [0, len(steps) - 1]
    with np.load(FILMS / "pulapka_17_cache.npz") as c:
        dtype_info = {}
        data = {}
        for name in ("X", "F", "Zf", "V"):
            full = c[name]
            assert len(full) == len(steps), (name, full.shape, len(steps))
            dtype_info[name] = str(full.dtype)
            data[name] = full[selected].astype(np.float64)
            del full
    X, F, Z, V = (data[name] for name in ("X", "F", "Zf", "V"))
    # Equivalent to film 17's PCA of DEL[-1] - DEL[-1].mean(0).
    U, _, _ = np.linalg.svd((X[-1] - X[-1].mean(0)).T, full_matrices=False)
    P = U[:, :3].T
    keep = np.arange(600) != focal
    results = []
    for n, step in enumerate(steps[selected]):
        path = WEIGHTS / f"k{step:05d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)["net"]
        gamma = state["norm.weight"].double().numpy()
        bias = state["bloki.5.mlp.fc2.bias"].double().numpy()
        current_v = state["bloki.5.mlp.fc2.weight"].double().numpy()
        clipped_red = P @ V[n] @ np.maximum(Z[n], 0)
        signed_sum = P @ V[n] @ Z[n]
        # Actual unused Fw of film 17: cached full MLP forward output.
        full_f = P @ F[n, focal]
        a0, a1 = layer_standardize(X[n]), layer_standardize(X[n] + F[n])
        unweighted_r = ratio(a0, a1, focal)
        affine_r = ratio(a0, a1, focal, gamma)
        changed = (unweighted_r < 1) != (affine_r < 1)
        groups = {}
        for label, mask in (("all_except_focal", keep), ("tool_except_focal", tool & keep), ("background", ~tool)):
            groups[label] = {
                "count": int(mask.sum()),
                "changed_attraction_label_count": int(changed[mask].sum()),
                "changed_attraction_label_fraction": float(changed[mask].mean()),
                "attraction_fraction_no_gamma": float((unweighted_r[mask] < 1).mean()),
                "attraction_fraction_actual_final_gamma": float((affine_r[mask] < 1).mean()),
                "median_r_no_gamma": float(np.median(unweighted_r[mask])),
                "median_r_actual_final_gamma": float(np.median(affine_r[mask])),
            }
        results.append({
            "step": int(step), "weight_file": file_record(path),
            "negative_z_fraction_focal": float((Z[n] < 0).mean()),
            "red_relerr_vs_full_F_3d": float(np.linalg.norm(clipped_red - full_f) / np.linalg.norm(full_f)),
            "red_cos_vs_full_F_3d": unit_cos(clipped_red, full_f),
            "signed_sum_relerr_vs_full_F_3d": float(np.linalg.norm(signed_sum - full_f) / np.linalg.norm(full_f)),
            "bias_norm_over_full_F_192d": float(np.linalg.norm(bias) / np.linalg.norm(F[n, focal])),
            "negative_sum_norm_over_full_F_192d": float(np.linalg.norm(V[n] @ np.minimum(Z[n], 0)) / np.linalg.norm(F[n, focal])),
            "cached_F_vs_current_V_signed_z_plus_bias_relerr": float(np.linalg.norm(F[n, focal] - (current_v @ Z[n] + bias)) / np.linalg.norm(F[n, focal])),
            "cached_V_vs_current_V_relerr": float(np.linalg.norm(V[n] - current_v) / np.linalg.norm(current_v)),
            "final_gamma_range": [float(gamma.min()), float(gamma.max())],
            "distance_ratios": groups,
        })
    with np.load(FILMS / "pulapka_18_cache.npz") as c:
        J = c["JJ"][selected].astype(np.float64)
    sym, anti = (J[-1] + J[-1].T) / 2, (J[-1] - J[-1].T) / 2
    ev, eigvec = np.linalg.eigh(sym)
    e1, e2 = eigvec[:, 0], eigvec[:, -1]
    projector12 = np.outer(e1, e1) + np.outer(e2, e2)
    U3, _, _ = np.linalg.svd((np.eye(192) - projector12) @ anti @ np.stack([e1, e2], axis=1), full_matrices=False)
    Q = np.stack([e1, e2, U3[:, 0]], axis=1)
    j_diagnostics = []
    for n, step in enumerate(steps[selected]):
        j3 = Q.T @ J[n] @ Q
        leakage = (np.eye(192) - Q @ Q.T) @ J[n] @ Q
        j_diagnostics.append({
            "step": int(step),
            "diagonal_in_final_basis_e1_e2_e3": np.diag(j3).tolist(),
            "outside_basis_Frobenius_fraction_of_JQ": float(np.linalg.norm(leakage) / np.linalg.norm(J[n] @ Q)),
            "comment": "Closed 3D probe dynamics omit this out-of-plane component; the subspace is not invariant.",
        })
    payload = {
        "scope": "Existing-data audit only; no forward pass, training, or rendering.",
        "focal_token": focal,
        "cache_dtypes": dtype_info,
        "files": [file_record(FILMS / name) for name in ("szescian.npz", "pulapka_17_cache.npz", "pulapka_18_cache.npz")],
        "method": {
            "projection": "Final-checkpoint centered raw X PCA, identical subspace to film 17. Float64 audit arithmetic, original caches retained.",
            "red": "Movie-17 red = P V max(Zf,0). Full arrow = P cached F. Common render scaling cancels from relative error/cosine.",
            "metric": "Compare distances after standardization without affine parameters versus the same standardization multiplied by actual final net.norm.weight at that checkpoint. beta cancels from pair differences. eps=1e-5 is the final torch.nn.LayerNorm default.",
            "limitations": "Existing X/F/V caches are float16. This check measures the movie's cached data; it does not replace an independent full-precision forward validation.",
        },
        "endpoints": results,
        "right_panel_projection": j_diagnostics,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "endpoints": results, "right_panel_projection": j_diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
