"""Wczytuje samodzielny K1000 z jedną wspólną korektą; obala go niezgodność fresh-load."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from model_snapshot import MalyViT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package_files(package: Path) -> dict[str, str]:
    package = Path(package)
    expected = json.loads((package / "FILES_SHA256.json").read_text(encoding="utf-8"))
    actual = {name: sha256(package / name) for name in expected}
    bad = {name: {"expected": expected[name], "actual": actual[name]} for name in expected if actual[name] != expected[name]}
    if bad:
        raise RuntimeError(f"Niezgodne pliki pakietu: {bad}")
    return actual


class ExecutableSharedPatchModel(nn.Module):
    """Pełny K1000; hook dodaje tę samą deltę do naturalnego post-GELU."""

    def __init__(self, base: MalyViT, delta: torch.Tensor, floor: float):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.register_buffer("delta", delta.detach().float().clone())
        self.floor = float(floor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        def replace(_module, _args, natural):
            return (natural + self.delta).clamp_min(self.floor)

        hook = self.base.bloki[-1].mlp.act.register_forward_hook(replace)
        try:
            output, _ = self.base(images)
        finally:
            hook.remove()
        return output


def load_executable_artifact(package: Path, device: torch.device | str = "cpu"):
    package = Path(package)
    verify_package_files(package)
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(package / "MODEL_VIT_K1000.pt", map_location="cpu", weights_only=True)
    if set(checkpoint) != {"net", "krok"} or int(checkpoint["krok"]) != 1000:
        raise RuntimeError("Pakiet nie zawiera dokładnego kontenera K1000")
    state = torch.load(package / "CORRECTION.pt", map_location="cpu", weights_only=True)
    delta = state["delta"]
    if tuple(delta.shape) != (600, 768) or not torch.isfinite(delta).all():
        raise RuntimeError("Nieprawidłowa wspólna delta")
    base = MalyViT().float().eval().requires_grad_(False)
    base.load_state_dict(checkpoint["net"], strict=True)
    model = ExecutableSharedPatchModel(base, delta, config["gelu_floor"])
    model = model.to(device).eval().requires_grad_(False)
    return model, config
