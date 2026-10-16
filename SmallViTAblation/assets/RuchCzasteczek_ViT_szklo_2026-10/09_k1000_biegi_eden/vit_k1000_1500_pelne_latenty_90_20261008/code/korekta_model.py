"""Wykonywalne warstwy wspólnej korekty naturalnego latentu K1000."""
from __future__ import annotations

import torch
from torch import nn


class MeanCorrection(nn.Module):
    """Metoda A: jedna wspólna delta dla każdego z 600 indeksów tokenu."""

    def __init__(self, mean_delta: torch.Tensor, floor: float):
        super().__init__()
        self.register_buffer("mean_delta", mean_delta.detach().float().clone())
        self.floor = float(floor)

    def forward(self, natural_z0: torch.Tensor) -> torch.Tensor:
        if natural_z0.shape[-2:] != self.mean_delta.shape:
            raise ValueError((natural_z0.shape, self.mean_delta.shape))
        return (natural_z0 + self.mean_delta).clamp_min(self.floor)


class RegimeCorrection(nn.Module):
    """Metoda B: reżim z naturalnego z0/kontekstu, potem delta per patch."""

    def __init__(
        self,
        z_mean: torch.Tensor,
        z_components: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        assignment_centroids: torch.Tensor,
        correction_by_regime_token: torch.Tensor,
        grid_shape: tuple[int, int],
        floor: float,
    ):
        super().__init__()
        self.register_buffer("z_mean", z_mean.detach().float().clone())
        self.register_buffer("z_components", z_components.detach().float().clone())
        self.register_buffer("feature_mean", feature_mean.detach().float().clone())
        self.register_buffer("feature_scale", feature_scale.detach().float().clone())
        self.register_buffer(
            "assignment_centroids", assignment_centroids.detach().float().clone()
        )
        self.register_buffer(
            "correction_by_regime_token",
            correction_by_regime_token.detach().float().clone(),
        )
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.floor = float(floor)
        self.last_regime_ids: torch.Tensor | None = None

    def assignment_features(self, natural_z0: torch.Tensor) -> torch.Tensor:
        batch, tokens, dimensions = natural_z0.shape
        height, width = self.grid_shape
        if tokens != height * width or dimensions != self.z_mean.numel():
            raise ValueError((natural_z0.shape, self.grid_shape, self.z_mean.shape))
        projection = (natural_z0 - self.z_mean) @ self.z_components.T
        context = projection.mean(dim=1, keepdim=True).expand(-1, tokens, -1)
        token = torch.arange(tokens, device=natural_z0.device)
        row = (token // width).float() / max(height - 1, 1)
        column = (token % width).float() / max(width - 1, 1)
        position = torch.stack([row, column], dim=-1)[None].expand(batch, -1, -1)
        features = torch.cat([projection, position, context], dim=-1)
        return (features - self.feature_mean) / self.feature_scale.clamp_min(1e-8)

    def assign(self, natural_z0: torch.Tensor) -> torch.Tensor:
        features = self.assignment_features(natural_z0)
        distance = (
            features[:, :, None, :] - self.assignment_centroids[None, None, :, :]
        ).square().sum(dim=-1)
        return distance.argmin(dim=-1)

    def forward(self, natural_z0: torch.Tensor) -> torch.Tensor:
        regimes = self.assign(natural_z0)
        self.last_regime_ids = regimes.detach()
        batch, tokens, _ = natural_z0.shape
        token_index = torch.arange(tokens, device=natural_z0.device)[None].expand(batch, -1)
        correction = self.correction_by_regime_token[regimes, token_index]
        return (natural_z0 + correction).clamp_min(self.floor)
