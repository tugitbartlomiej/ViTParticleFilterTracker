"""Live raw patch recording and FIT-only cross-fitted Koopman–GLE diagnostics.

One particle is one fixed image-grid patch, not a pooled image representation
or a tracked physical object. D labels may score predictions but may never fit
scalers, the weak bbox readout, Koopman, or GLE. The detector itself is the live
training detector and is not cross-fitted; that limitation is recorded in audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class LivePatchRecorder:
    """Attach to the existing RF-DETR encoder; never construct another detector.

    Use as a context manager around inference on the current ``net``. After
    each forward call, ``pop()`` returns float32 ``[B, H*W, dim]`` in row-major
    patch order. Only channels are projected. The hook detaches its own copy
    and leaves the detector output, gradients, weights, and mode unchanged.
    """

    def __init__(self, net, dim=24, seed=42):
        if int(dim) != dim or dim < 1:
            raise ValueError("projected channel dimension must be positive")
        self.dim, self.seed = int(dim), int(seed)
        self.grid_shape = None
        self.raw_channels = None
        self.projection = None
        self._torch_projection = None
        self._pending = None
        self._closed = False
        try:
            encoder = net.backbone[0].encoder
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("live RF-DETR net must expose backbone[0].encoder") from exc
        self._handle = encoder.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        import torch

        if self._pending is not None:
            raise RuntimeError("consume each live patch batch with pop() before another forward")
        maps = output if isinstance(output, (list, tuple)) else (
            output.get("feature_maps") if isinstance(output, dict) else getattr(output, "feature_maps", None))
        if maps is None or len(maps) == 0:
            raise ValueError("encoder must emit nonempty raw ViT feature maps")
        raw = maps[-1]
        if not isinstance(raw, torch.Tensor) or raw.ndim != 4:
            raise ValueError("raw ViT feature map must be a B,C,H,W tensor")
        batch, channels, height, width = raw.shape
        if min(batch, channels, height, width) < 1:
            raise ValueError("raw ViT map cannot have an empty dimension")
        if self.projection is None:
            self.grid_shape = (int(height), int(width))
            self.raw_channels = int(channels)
            self.projection = (np.random.default_rng(self.seed).normal(size=(channels, self.dim))
                               / np.sqrt(self.dim)).astype(np.float32)
        if self.grid_shape != (height, width) or self.raw_channels != channels:
            raise ValueError("patch grid or raw channels changed within the live cache")
        with torch.no_grad():
            if self._torch_projection is None or self._torch_projection.device != raw.device:
                self._torch_projection = torch.as_tensor(self.projection, device=raw.device)
            # The only flatten joins H and W. No patch reduction is permitted.
            patches = raw.detach().float().permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            projected = patches @ self._torch_projection
            if not bool(torch.isfinite(projected).all()):
                raise ValueError("nonfinite live patch projection")
            self._pending = projected.float().cpu().numpy()

    def pop(self):
        if self._pending is None:
            raise RuntimeError("no unconsumed live forward output")
        result, self._pending = self._pending, None
        return result

    @property
    def metadata(self):
        return dict(feature="last_raw_vit_patch", particle="one fixed image-grid patch",
                    spatial_pooling=False, grid_shape=list(self.grid_shape) if self.grid_shape else None,
                    raw_channels=self.raw_channels, projected_channels=self.dim,
                    projection_seed=self.seed, live_detector=True,
                    projection_sha256=(hashlib.sha256(self.projection.tobytes()).hexdigest()
                                       if self.projection is not None else None))

    def close(self):
        if not self._closed:
            self._handle.remove()
            self._closed = True
        self._pending = None
        self._torch_projection = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("cannot reuse a closed patch recorder")
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


def positive_gain(e_obs, e_dyn, valid_history):
    """Nonnegative observed-readout minus predicted-readout BCE; unavailable = 0."""
    observed, dynamic = np.asarray(e_obs, dtype=float), np.asarray(e_dyn, dtype=float)
    valid = np.asarray(valid_history, dtype=bool)
    if observed.shape != dynamic.shape or observed.shape != valid.shape:
        raise ValueError("aligned observation, dynamics, and history arrays required")
    gain = np.zeros(observed.shape, dtype=np.float64)
    usable = valid & np.isfinite(observed) & np.isfinite(dynamic)
    gain[usable] = np.maximum(observed[usable] - dynamic[usable], 0.)
    return gain


def _check_partition(rows, pool_rows, support_ids, excluded_fit_groups):
    if not rows or not pool_rows:
        raise ValueError("nonempty train cache and scoring pool required")
    if any(row.get("split") != "train" for row in list(rows) + list(pool_rows)):
        raise ValueError("dynamics cannot read validation or test rows")
    row_index = {str(row["frame_id"]): row for row in rows}
    pool_index = {str(row["frame_id"]): row for row in pool_rows}
    if len(row_index) != len(rows) or len(pool_index) != len(pool_rows):
        raise ValueError("frame IDs must be unique")
    excluded = set(map(str, excluded_fit_groups))
    groups = {str(row["group_id"]) for row in rows}
    if not excluded <= groups:
        raise ValueError("excluded fit groups must occur in the cache")
    for row in list(rows) + list(pool_rows):
        group = str(row["group_id"])
        expected = "D" if group in excluded else "FIT"
        role = row.get("role", expected)
        if role != expected:
            raise ValueError(f"group {group} role {role!r} disagrees with FIT/D exclusion")
    for frame_id, row in pool_index.items():
        cached = row_index.get(frame_id)
        if cached is None:
            raise ValueError("scoring frame is absent from the live cache")
        if (str(row["group_id"]) != str(cached["group_id"])
                or str(row["video_id"]) != str(cached["video_id"])
                or float(row["frame_index"]) != float(cached["frame_index"])):
            raise ValueError("scoring/cache group, video, or frame time mismatch")
    support = list(map(str, support_ids))
    if not support or len(support) != len(set(support)) or not set(support) <= set(pool_index):
        raise ValueError("support must contain unique labeled FIT frame IDs from the scoring pool")
    if any(str(row_index[frame_id]["group_id"]) in excluded for frame_id in support):
        raise ValueError("support must contain FIT labels only; D support is forbidden")
    return support, excluded


def compute_dynamics(rows, tokens, inputs, pool_rows, support_ids, excluded_fit_groups, out,
                     *, grid_shape=(24, 24), seed=42, device="cuda", epochs=12,
                     max_windows=20000, context=8, step=5, folds=3):
    """Cross-fit raw patch dynamics and save immutable per-round artifacts.

    Cache rows and pool rows retain source ``split='train'`` and may optionally
    carry ``role='FIT'`` / ``role='D'``. ``pool_rows`` contains labeled FIT
    candidates and labeled D scoring frames in the desired output order;
    ``support_ids`` contains currently selected/labeled FIT frames only.
    D history can be used at scoring time but never to fit any model or scale.

    ``valid_history`` requires both actual regular temporal history and a model
    fitted on independent FIT groups. Missing history yields NaN ``e_dyn`` and
    zero ``positive_gain``. The return value includes the old fitter's errors,
    diagnostics and audit, plus these two aligned arrays.
    """
    support, excluded = _check_partition(rows, pool_rows, support_ids, excluded_fit_groups)
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(f"new immutable dynamics directory required: {destination}")
    from .vendor.token_dopasowanie import fit_token_cross_fitted

    result = fit_token_cross_fitted(
        rows, tokens, inputs, pool_rows, support, excluded_fit_groups=excluded,
        grid_shape=grid_shape, seed=seed, device=device, epochs=epochs,
        max_windows=max_windows, context=context, step=step, folds=folds,
        artifact_dir=destination,
    )
    valid = (np.asarray([item["has_temporal_context"] for item in result["diagnostics"]], dtype=bool)
             & np.isfinite(result["e_dyn"]) & np.isfinite(result["e_persistence"]))
    result["valid_history"] = valid
    result["positive_gain"] = positive_gain(result["e_obs"], result["e_dyn"], valid)
    for index, item in enumerate(result["diagnostics"]):
        item.update(valid_history=bool(valid[index]), positive_gain=float(result["positive_gain"][index]),
                    role="D" if str(item["group_id"]) in excluded else "FIT")
    result["audit"].update(
        schema_version=1, score_order=[str(row["frame_id"]) for row in pool_rows],
        support_semantics="currently labeled FIT only; support readout is group cross-fitted",
        gain="max(e_obs - e_dyn, 0) only with finite errors and valid temporal history",
        detector_weights="live current weights, detector itself is not cross-fitted",
        evaluation_rows_read=False, d_used_for_fitting=False,
        temporal_prediction="strict; target latent and metadata excluded from prediction",
    )
    with (destination / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result["audit"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (destination / "diagnostics.jsonl").open("x", encoding="utf-8") as stream:
        for item in result["diagnostics"]:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    with (destination / "scores.npz").open("xb") as stream:
        np.savez_compressed(stream, frame_ids=np.asarray([str(row["frame_id"]) for row in pool_rows]),
                            **{key: result[key] for key in
                               ("e_obs", "e_dyn", "e_persistence", "valid_history", "positive_gain")})
    return result
