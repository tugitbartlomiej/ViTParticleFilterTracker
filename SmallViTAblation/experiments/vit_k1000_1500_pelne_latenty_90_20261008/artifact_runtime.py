"""Minimalny runtime samodzielnych pakietów K1000 + wspólna korekta latentu."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from korekta_model import MeanCorrection, RegimeCorrection
from model_snapshot import MalyViT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CorrectedK1000(nn.Module):
    """Pełny K1000; hook transformuje naturalne post-GELU osobno dla obrazu."""

    def __init__(self, base: MalyViT, correction: nn.Module):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.correction = correction.eval().requires_grad_(False)
        self.last_natural: torch.Tensor | None = None
        self.last_corrected: torch.Tensor | None = None

    def forward(self, image: torch.Tensor):
        self.last_natural = None
        self.last_corrected = None

        def replace(_module, _args, natural):
            corrected = self.correction(natural)
            self.last_natural = natural
            self.last_corrected = corrected
            return corrected

        hook = self.base.bloki[-1].mlp.act.register_forward_hook(replace)
        try:
            output, _ = self.base(image)
        finally:
            hook.remove()
        if self.last_natural is None or self.last_corrected is None:
            raise RuntimeError("Hook korekty nie został wykonany")
        return output, self.last_natural, self.last_corrected


def verify_package_files(path: Path) -> dict[str, str]:
    manifest_path = Path(path) / "FILES_SHA256.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Pakiet nie ma {manifest_path.name}")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        candidate = Path(path) / relative
        if not candidate.is_file() or sha256(candidate) != digest:
            raise RuntimeError(f"Nieprawidłowy plik pakietu: {relative}")
    return expected


def load_executable_artifact(path: Path, device: str | torch.device = "cpu"):
    """Świeżo buduje model i regułę wyłącznie z zawartości katalogu pakietu."""
    path = Path(path)
    verify_package_files(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    checkpoint = path / "MODEL_VIT_K1000.pt"
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("Hash checkpointa w pakiecie nie zgadza się z config")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("krok") != 1000:
        raise RuntimeError("Pakiet nie zawiera K1000")
    base = MalyViT().float().eval().requires_grad_(False)
    base.load_state_dict(payload["net"], strict=True)
    correction_path = path / "CORRECTION.pt"
    if sha256(correction_path) != config["correction_sha256"]:
        raise RuntimeError("Hash korekty w pakiecie nie zgadza się z config")
    state = torch.load(correction_path, map_location="cpu", weights_only=True)
    if config["method"] == "mean600x768":
        correction = MeanCorrection(state["mean_delta"], config["gelu_floor"])
    elif config["method"] == "inferred_regime":
        correction = RegimeCorrection(
            z_mean=state["z_mean"],
            z_components=state["z_components"],
            feature_mean=state["feature_mean"],
            feature_scale=state["feature_scale"],
            assignment_centroids=state["assignment_centroids"],
            correction_by_regime_token=state["correction_by_regime_token"],
            grid_shape=tuple(config["grid_shape"]),
            floor=config["gelu_floor"],
        )
    else:
        raise ValueError(config["method"])
    executable = CorrectedK1000(base, correction).to(device).eval().requires_grad_(False)
    return executable, config
