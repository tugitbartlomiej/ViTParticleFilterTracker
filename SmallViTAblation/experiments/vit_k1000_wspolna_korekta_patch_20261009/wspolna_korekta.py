"""Rdzeń jednej współdzielonej korekty; test obala wzrost mAP względem delta zero."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GELU_FLOOR = -0.16997121
STRIDE = 16.0
TOKENS = 600
DIMENSIONS = 768
EXPECTED_CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
EXPECTED_MANIFEST_SHA256 = "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7"
EXPECTED_BASELINE = {
    "AP50": 0.626887827062123,
    "AP75": 0.13109693084564492,
    "mAP50_95": 0.24549623827449926,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_fingerprint(net: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class SharedPatchCorrection(nn.Module):
    """Dodaje tę samą macierz 600x768 do naturalnego latentu każdego obrazu."""

    def __init__(self, delta: torch.Tensor, floor: float = GELU_FLOOR):
        super().__init__()
        if tuple(delta.shape) != (TOKENS, DIMENSIONS):
            # Małe kształty są dozwolone w testach jednostkowych.
            if delta.ndim != 2:
                raise ValueError(tuple(delta.shape))
        self.register_buffer("delta", delta.detach().float().clone())
        self.floor = float(floor)

    def forward(self, natural_z0: torch.Tensor) -> torch.Tensor:
        if tuple(natural_z0.shape[-2:]) != tuple(self.delta.shape):
            raise ValueError((tuple(natural_z0.shape), tuple(self.delta.shape)))
        return (natural_z0 + self.delta).clamp_min(self.floor)


def capture_prefix(net: nn.Module, images: torch.Tensor):
    """Zwraca naturalny output, post-GELU z0 oraz wyjście ostatniego bloku q0."""
    cache: dict[str, torch.Tensor] = {}
    block = net.bloki[-1]
    hook_z = block.mlp.act.register_forward_hook(
        lambda _module, _args, output: cache.update(z=output.detach().clone())
    )
    hook_q = block.register_forward_hook(
        lambda _module, _args, output: cache.update(q=output.detach().clone())
    )
    try:
        with torch.no_grad():
            output, _ = net(images)
    finally:
        hook_z.remove()
        hook_q.remove()
    if set(cache) != {"z", "q"}:
        raise RuntimeError(f"Hooki zebrały {sorted(cache)}")
    return output.detach(), cache["z"], cache["q"]


def fast_tail(
    net: nn.Module,
    natural_z0: torch.Tensor,
    natural_q0: torch.Tensor,
    delta: torch.Tensor,
    floor: float = GELU_FLOOR,
) -> torch.Tensor:
    """Przelicza oryginalny tail po jednej wspólnej interwencji post-GELU."""
    if tuple(delta.shape) != tuple(natural_z0.shape[-2:]):
        raise ValueError((tuple(delta.shape), tuple(natural_z0.shape)))
    changed = (natural_z0 + delta).clamp_min(float(floor))
    block = net.bloki[-1]
    residual_change = block.ls2(
        F.linear(changed - natural_z0, block.mlp.fc2.weight, None)
    )
    flat = net.glowa(net.norm(natural_q0 + residual_change))
    if flat.shape[1] != TOKENS:
        raise ValueError(tuple(flat.shape))
    return flat.reshape(flat.shape[0], 20, 30, 5)


def choose_target_token(box_xyxy: np.ndarray, height: int = 20, width: int = 30) -> dict:
    """Wybiera token o najwyższej geometrycznej granicy IoU dla corrected GT."""
    box = np.asarray(box_xyxy, dtype=np.float64)
    x0, y0, x1, y1 = box.tolist()
    if not (x1 > x0 and y1 > y0):
        raise ValueError(box.tolist())
    xs = (np.arange(width, dtype=np.float64) + 0.5) * STRIDE
    ys = (np.arange(height, dtype=np.float64) + 0.5) * STRIDE
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    ux0, uy0 = np.minimum(cx, x0), np.minimum(cy, y0)
    ux1, uy1 = np.maximum(cx, x1), np.maximum(cy, y1)
    upper = ((x1 - x0) * (y1 - y0)) / ((ux1 - ux0) * (uy1 - uy0))
    distance = (cx - (x0 + x1) / 2.0) ** 2 + (cy - (y0 + y1) / 2.0) ** 2
    best = float(upper.max())
    candidates = np.flatnonzero(np.isclose(upper.reshape(-1), best, rtol=0, atol=1e-12))
    index = int(candidates[np.argmin(distance.reshape(-1)[candidates])])
    row, column = divmod(index, width)
    anchor_x, anchor_y = xs[column], ys[row]
    ltrb = np.maximum(
        np.asarray([anchor_x - x0, anchor_y - y0, x1 - anchor_x, y1 - anchor_y], np.float32)
        / STRIDE,
        np.float32(1e-4),
    )
    return {
        "token_index": index,
        "token_row": row,
        "token_col": column,
        "target_ltrb": ltrb,
        "geometry_iou_upper_bound": best,
    }


def _giou_ltrb_batch(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (
        torch.minimum(predicted[:, 0], target[:, 0])
        + torch.minimum(predicted[:, 2], target[:, 2])
    ) * (
        torch.minimum(predicted[:, 1], target[:, 1])
        + torch.minimum(predicted[:, 3], target[:, 3])
    )
    area_predicted = (predicted[:, 0] + predicted[:, 2]) * (
        predicted[:, 1] + predicted[:, 3]
    )
    area_target = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
    union = area_predicted + area_target - intersection
    cover = (
        torch.maximum(predicted[:, 0], target[:, 0])
        + torch.maximum(predicted[:, 2], target[:, 2])
    ) * (
        torch.maximum(predicted[:, 1], target[:, 1])
        + torch.maximum(predicted[:, 3], target[:, 3])
    )
    return intersection / union.clamp_min(1e-9) - (cover - union) / cover.clamp_min(1e-9)


def joint_loss_per_image(
    output: torch.Tensor,
    target_index: torch.Tensor,
    target_ltrb: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stały cel joint: właściwy token, tło, top-1, ltrb i GIoU."""
    batch = output.shape[0]
    flat = output.reshape(batch, -1, 5)
    if target_index.shape != (batch,) or target_ltrb.shape != (batch, 4):
        raise ValueError((tuple(output.shape), tuple(target_index.shape), tuple(target_ltrb.shape)))
    if bool(((target_index < 0) | (target_index >= flat.shape[1])).any()):
        raise ValueError("Indeks docelowego tokenu poza siatką")
    rows = torch.arange(batch, device=output.device)
    logits = flat[:, :, 0]
    target_logit = logits[rows, target_index]
    positive = F.softplus(5.5 - target_logit)
    all_negative = F.softplus(logits + 2.0)
    negative = (
        all_negative.sum(dim=1) - all_negative[rows, target_index]
    ) / max(logits.shape[1] - 1, 1)
    rank = F.cross_entropy(logits, target_index, reduction="none")
    raw_ltrb = flat[rows, target_index, 1:]
    box = F.smooth_l1_loss(
        raw_ltrb,
        target_ltrb.clamp_min(1e-4).log(),
        reduction="none",
    ).mean(dim=1)
    predicted_ltrb = raw_ltrb.clamp(-20, 20).exp()
    giou = _giou_ltrb_batch(predicted_ltrb, target_ltrb)
    components = {
        "positive": positive,
        "negative": negative,
        "rank": rank,
        "box": box,
        "giou": giou,
    }
    total = positive + negative + 0.5 * rank + 20.0 * box + 10.0 * (1.0 - giou)
    return total, components


def exact_mean_backward(
    parameter: torch.Tensor,
    batches: Iterable,
    total_count: int,
    objective_fn: Callable[[torch.Tensor, object], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    regularization_weight: float,
) -> dict:
    """Akumuluje dokładnie sumę loss_i/N dla nierównych mikropaczek."""
    if total_count <= 0:
        raise ValueError(total_count)
    examples = 0
    data_sum = 0.0
    component_sums: dict[str, float] = {}
    for batch in batches:
        losses, components = objective_fn(parameter, batch)
        if losses.ndim != 1 or not torch.isfinite(losses).all():
            raise RuntimeError("Niefinitywna albo nieper-image strata")
        count = int(losses.numel())
        (losses.sum() / float(total_count)).backward()
        examples += count
        data_sum += float(losses.detach().sum().item())
        for name, values in components.items():
            component_sums[name] = component_sums.get(name, 0.0) + float(
                values.detach().sum().item()
            )
    if examples != total_count:
        raise RuntimeError(f"Akumulacja objęła {examples}, oczekiwano {total_count}")
    regularization = float(regularization_weight) * parameter.square().mean()
    regularization.backward()
    return {
        "examples": examples,
        "weight_sum": examples / float(total_count),
        "data_loss": data_sum / total_count,
        "regularization": float(regularization.detach().item()),
        "total_loss": data_sum / total_count + float(regularization.detach().item()),
        "components": {name: value / total_count for name, value in component_sums.items()},
    }


def learning_rate(full_pass: int) -> float:
    if not 1 <= full_pass <= 100:
        raise ValueError(full_pass)
    if full_pass <= 40:
        return 0.01
    if full_pass <= 70:
        return 0.003
    return 0.001


def should_stop_for_plateau(
    completed_pass: int,
    evaluations_without_improvement: int,
    min_passes: int = 30,
    patience_evaluations: int = 6,
) -> bool:
    return (
        completed_pass >= min_passes
        and evaluations_without_improvement >= patience_evaluations
    )


def json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(path: Path, expected_config_sha256: str) -> dict:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "completed_pass",
        "delta",
        "optimizer",
        "torch_rng_state",
        "numpy_rng_state",
        "config_sha256",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise RuntimeError(f"Niepełny checkpoint {path}")
    if value["format_version"] != 1:
        raise RuntimeError("Nieobsługiwana wersja checkpointu")
    if value["config_sha256"] != expected_config_sha256:
        raise RuntimeError("Checkpoint pochodzi z innej konfiguracji")
    if tuple(value["delta"].shape) != (TOKENS, DIMENSIONS):
        # Test jednostkowy korzysta z małego tensora.
        if value["delta"].ndim != 2:
            raise RuntimeError(tuple(value["delta"].shape))
    if not torch.isfinite(value["delta"]).all():
        raise RuntimeError("Niefinitywna delta w checkpointcie")
    return value
