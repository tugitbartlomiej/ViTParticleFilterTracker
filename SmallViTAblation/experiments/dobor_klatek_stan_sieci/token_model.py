"""Czy pełne GLE ze skalą ruchu patcha przewiduje użyteczne tokeny?

Port skali z tft_langevin_optymalizacja_95/model_s.py na przyczynowe obrazy.
sigma liczone tylko z zaobserwowanego ruchu danego patcha; D,JP,W skalowane
sigma**gamma. Kotwica Z oraz filtr N pozostają liniowe w latencie. Tryb strict
nie zna zdarzenia jutra, więc nie uruchamia warunkowego skoku Z. Okno skali
to maksymalnie dostępne8 obserwacji, a nie28 dni z oryginału. Zmiana jednostek
czasu i metadanych jest adaptacją, nie identycznym modelem z TS.
"""
from __future__ import annotations
import torch
from torch import nn
from .model import ExtendedLangevin


class ScaledTokenLangevin(ExtendedLangevin):
    def __init__(self, config, step_scale, acc_scale):
        if config.conditional:
            raise ValueError('token pilot requires strict causal mode')
        super().__init__(config)
        for name, value in (('step_scale', step_scale), ('acc_scale', acc_scale)):
            scale = torch.as_tensor(value, dtype=torch.float32).clone().detach()
            if scale.shape != (config.latent_dim,) or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f'{name} must be positive FIT-only scale per channel')
            self.register_buffer(name, scale)
        self.gamma = nn.Parameter(torch.full((3,), .5))  # D, JP, W; no future-event jump

    def particle_scale(self, velocity):
        energy = velocity.square().sum(-1)
        cumulative = energy.cumsum(1)
        window = self.config.context
        if energy.shape[1] > window:
            cumulative = torch.cat((cumulative[:, :window], cumulative[:, window:]-cumulative[:, :-window]), 1)
        count = torch.arange(1, energy.shape[1]+1, device=velocity.device).clamp(max=window)
        return (cumulative/count/self.step_scale.square().sum()).clamp_min(1e-8).sqrt().clamp(.5, 50)[..., None]

    def forward(self, z, inputs, observed):
        if z.ndim != 3 or z.shape[1] < 2 or z.shape[-1] != self.config.latent_dim:
            raise ValueError('z must be B,T>=2,latent_dim')
        if inputs.shape != (*z.shape[:2], self.config.input_dim) or observed.shape != z.shape[:2] or observed.dtype != torch.bool:
            raise ValueError('metadata shape/type mismatch')
        origin = z[:, :-1]
        velocity = torch.cat((torch.zeros_like(origin[:, :1]), origin[:, 1:]-origin[:, :-1]), 1)
        acceleration = torch.cat((torch.zeros_like(velocity[:, :2]), velocity[:, 2:]-velocity[:, 1:-1]), 1)
        sigma = self.particle_scale(velocity)
        v = velocity/self.step_scale/sigma
        a = acceleration/self.acc_scale/sigma
        gains = [sigma**gamma for gamma in self.gamma]
        terms = {
            'D': self.D(torch.cat((origin, v, a), -1))*self.step_scale*gains[0],
            'JP': self.JP(self.token(torch.cat((origin, v), -1)))*self.step_scale*gains[1],
            'W': self.W(inputs, origin)*self.step_scale*gains[2] if self.config.forcing else torch.zeros_like(origin),
            'Z': self.Z(origin, inputs, observed) if self.config.events else torch.zeros_like(origin),
        }
        base_residual = z[:, 1:]-origin-sum(terms.values())
        terms['N'] = self.N(base_residual/self.step_scale)*self.step_scale if self.config.residuals else torch.zeros_like(origin)
        prediction = origin+sum(terms.values())
        return dict(prediction=prediction, terms=terms, base_residual=base_residual,
                    residual=z[:, 1:]-prediction, particle_scale=sigma)
