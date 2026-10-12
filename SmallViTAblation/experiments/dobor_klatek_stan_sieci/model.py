"""Czy D+JP+W+Z+N przewiduje latent poza grupa dopasowania lepiej od prostych kontroli?

Port struktury rownania Grant, bez zaleznosci od TFT. Zrodla (audyt 2026-09-22):
F:/Studia/MCHTR/Grant/Projekt/experiments/tft_langevin_latent_56/model.py
F:/Studia/MCHTR/Grant/Projekt/experiments/tft_langevin_czlon_zdarzen/model_z.py

z_hat[t+1] = z[t] + D[t] + JP[t] + W[t] + Z[t] + N[t].
Normalizacja latentu i wejsc nalezy do zewnetrznego FIT danego folda, stad s=1.
Domyslnie strict: prognoza celu t+1 widzi z i metadane tylko do t. Conditional
udostepnia W oraz Z metadane celu t+1, nigdy latent celu. Wszystkie wyjscia maja
indeks poczatku kroku: [B,T-1,d]. Pierwsze v i a sa zerowe, bez danych sprzed
odcinka; caller ocenia pelne, rowne okna i trenuje ostatni krok.

observed oznacza confidence>=0.5, NIE fizyczna obecnosc narzedzia. Z strict
jest oczekiwanym powrotem do kotwicy przy niskiej pewnosci, jak wariant Grant;
to hipoteza wymagajaca ablacji, a nie gwarancja poprawy. Pamiec resetuje sie
przy kazdym forward (osobne odcinki). Nie nalezy laczyc nieciaglych klipow.

JP kopiuje uwage z zanikiem i obrotem. N jest DETERMINISTYCZNA warunkowa srednia
dawnych reszt bazowych, nie losowanie szumu. Okresy startowe skaluja sie do
okna obserwacji, bez przenoszenia kalendarzowych dni/tygodni z TFT. Odczepy
reszt maja odstep 1 i 7 obserwacji, gdy mieszcza sie w oknie.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LangevinConfig:
    latent_dim: int = 24
    input_dim: int = 6
    context: int = 8
    hidden_dim: int = 64
    heads: int = 2
    residual_rank: int = 8
    events: bool = True
    forcing: bool = True
    residuals: bool = True
    conditional: bool = False

    def __post_init__(self):
        for name in ("latent_dim", "input_dim", "hidden_dim", "heads", "residual_rank"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.context < 2:
            raise ValueError("context must be at least 2")


class MLP(nn.Module):
    """Dwie warstwy ukryte i zerowy poczatkowy przyrost, jak w Grant."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _decay_logits(half_lives: Tensor) -> Tensor:
    rho = torch.exp(-math.log(2.) / half_lives)
    return torch.logit(rho)


class MemoryKernel(nn.Module):
    """JP: przyczynowa uwaga, wykładniczy zanik i uczone obroty par kanalow."""

    def __init__(self, config: LangevinConfig):
        super().__init__()
        self.context, self.heads = config.context, config.heads
        self.rank = max(2, 2 * (config.hidden_dim // config.heads // 2))
        width = self.heads * self.rank
        self.Wq = nn.Linear(config.hidden_dim, width, bias=False)
        self.Wk = nn.Linear(config.hidden_dim, width, bias=False)
        self.Wv = nn.Linear(config.hidden_dim, width, bias=False)
        self.Wo = nn.Linear(width, config.latent_dim)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)
        half_lives = torch.linspace(max(1., config.context / 4.), float(config.context), self.heads)
        self.theta = nn.Parameter(_decay_logits(half_lives))
        omega = torch.zeros(self.heads, self.rank // 2)
        omega[:, ::4] = 2. * math.pi / config.context
        self.omega = nn.Parameter(omega)

    def _rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        angle = self.omega[None, :, None, :] * positions[None, None, :, None]
        c, s = angle.cos(), angle.sin()
        first, second = x[..., 0::2], x[..., 1::2]
        return torch.stack((first * c - second * s, first * s + second * c), -1).flatten(-2)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        def split(layer):
            return layer(tokens).view(batch, length, self.heads, self.rank).transpose(1, 2)
        positions = torch.arange(length, dtype=tokens.dtype, device=tokens.device)
        q, k = self._rotate(split(self.Wq), positions), self._rotate(split(self.Wk), positions)
        lag = positions[:, None] - positions[None, :]
        mask = (lag >= 0) & (lag < self.context)
        envelope = self.theta.sigmoid()[:, None, None] ** lag.clamp_min(0)[None]
        weights = (F.elu(q) + 1.) @ (F.elu(k) + 1.).transpose(-1, -2)
        weights = weights * envelope[None] * mask[None, None]
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)
        memory = (weights @ split(self.Wv)).transpose(1, 2).reshape(batch, length, -1)
        return self.Wo(memory)


class Forcing(nn.Module):
    """W: dowolne FIR historii metadanych i osobne wejscie dostepne w chwili decyzji."""

    def __init__(self, config: LangevinConfig):
        super().__init__()
        self.conditional = config.conditional
        self.window = config.context + int(config.conditional)
        filters = max(4, config.hidden_dim // 4)
        self.fir = nn.Conv1d(config.input_dim, filters, self.window)
        nn.init.normal_(self.fir.weight, std=1. / self.window)
        self.head = MLP(config.input_dim + filters + config.latent_dim,
                        config.hidden_dim, config.latent_dim)

    def forward(self, inputs: Tensor, z_origin: Tensor) -> Tensor:
        available = inputs if self.conditional else inputs[:, :-1]
        history = self.fir(F.pad(available.transpose(1, 2), (self.window - 1, 0))).transpose(1, 2)
        if self.conditional:
            history, latest = history[:, 1:], inputs[:, 1:]
        else:
            latest = inputs[:, :-1]
        return self.head(torch.cat((latest, history, z_origin), -1))


class EventTerm(nn.Module):
    """Z: kotwica pewnosci; conditional zna przejscie, strict tylko szanse powrotu."""

    def __init__(self, config: LangevinConfig):
        super().__init__()
        self.conditional = config.conditional
        self.A = nn.Parameter(torch.full((config.latent_dim,), .9))
        self.pi = nn.Parameter(torch.zeros(4))  # przerwa 1, 2, 3, 4+ obserwacje
        self.jump = MLP(config.latent_dim + config.input_dim, config.hidden_dim, config.latent_dim)

    def forward(self, z_origin: Tensor, inputs: Tensor, observed: Tensor) -> Tensor:
        current = observed[:, :-1]
        batch, length = current.shape
        index = torch.arange(length, device=z_origin.device)[None].expand(batch, -1)
        last = torch.where(current, index, torch.full_like(index, -1)).cummax(1).values
        has_anchor = (last >= 0) & ~current
        anchor = torch.gather(z_origin, 1, last.clamp_min(0)[..., None].expand_as(z_origin))
        displacement = (z_origin - anchor) * has_anchor[..., None]
        correction = -self.A * displacement
        if not self.conditional:
            gap = (index - last).clamp(1, 4) - 1
            probability = self.pi.sigmoid()[gap] * has_anchor
            return probability[..., None] * correction
        returns = ~current & observed[:, 1:]
        disappears = current & ~observed[:, 1:]
        jump = self.jump(torch.cat((z_origin, inputs[:, :-1]), -1))
        return returns[..., None] * correction + disappears[..., None] * jump


class ColoredResidual(nn.Module):
    """N: niskorzędowy filtr r0[t-l], l>=1; nigdy reszta aktualnego celu."""

    def __init__(self, config: LangevinConfig):
        super().__init__()
        self.context, self.heads, self.rank = config.context, config.heads, config.residual_rank
        self.V = nn.Linear(config.latent_dim, self.heads * self.rank, bias=False)
        self.U = nn.Linear(self.heads * self.rank, config.latent_dim, bias=False)
        nn.init.zeros_(self.U.weight)
        half_lives = torch.linspace(1., max(1., config.context / 2.), self.heads)
        self.theta = nn.Parameter(_decay_logits(half_lives))
        self.omega = nn.Parameter(torch.linspace(0., 2. * math.pi / config.context, self.heads))
        self.a1 = nn.Parameter(torch.zeros(config.latent_dim))
        self.a7 = nn.Parameter(torch.zeros(config.latent_dim))

    def forward(self, residual: Tensor) -> Tensor:
        batch, length, _ = residual.shape
        values = self.V(residual).view(batch, length, self.heads, self.rank)
        memory = torch.zeros_like(values)
        for lag in range(1, min(self.context, length)):
            kernel = self.theta.sigmoid().pow(lag - 1) * torch.cos(self.omega * (lag - 1))
            shifted = torch.cat((values.new_zeros(batch, lag, self.heads, self.rank), values[:, :-lag]), 1)
            memory = memory + shifted * kernel[None, None, :, None]
        result = self.U(memory.reshape(batch, length, -1))
        for lag, coefficient in ((1, self.a1), (7, self.a7)):
            if lag < min(self.context, length):
                shifted = torch.cat((residual.new_zeros(batch, lag, residual.shape[-1]), residual[:, :-lag]), 1)
                result = result + coefficient * shifted
        return result


class ExtendedLangevin(nn.Module):
    """Jedno rownanie z niezaleznymi czlonami, prognoza poziomu i obie reszty."""

    def __init__(self, config: LangevinConfig):
        super().__init__()
        self.config = config
        self.D = MLP(3 * config.latent_dim, config.hidden_dim, config.latent_dim)
        self.token = nn.Linear(2 * config.latent_dim, config.hidden_dim)
        self.JP = MemoryKernel(config)
        self.W = Forcing(config)
        self.Z = EventTerm(config)
        self.N = ColoredResidual(config)

    def forward(self, z: Tensor, inputs: Tensor, observed: Tensor) -> dict:
        if z.ndim != 3 or z.shape[1] < 2 or z.shape[-1] != self.config.latent_dim:
            raise ValueError("z must have shape [B,T>=2,latent_dim]")
        if inputs.shape != (*z.shape[:2], self.config.input_dim):
            raise ValueError("inputs must have shape [B,T,input_dim]")
        if observed.shape != z.shape[:2] or observed.dtype != torch.bool:
            raise ValueError("observed must be boolean [B,T], derived from confidence")
        if inputs.device != z.device or observed.device != z.device:
            raise ValueError("z, inputs and observed must use the same device")
        origin = z[:, :-1]
        velocity = torch.cat((torch.zeros_like(origin[:, :1]), origin[:, 1:] - origin[:, :-1]), 1)
        acceleration = torch.cat((torch.zeros_like(velocity[:, :2]), velocity[:, 2:] - velocity[:, 1:-1]), 1)
        terms = {
            "D": self.D(torch.cat((origin, velocity, acceleration), -1)),
            "JP": self.JP(self.token(torch.cat((origin, velocity), -1))),
            "W": self.W(inputs, origin) if self.config.forcing else torch.zeros_like(origin),
            "Z": self.Z(origin, inputs, observed) if self.config.events else torch.zeros_like(origin),
        }
        deterministic = sum(terms.values())
        base_residual = z[:, 1:] - origin - deterministic
        terms["N"] = self.N(base_residual) if self.config.residuals else torch.zeros_like(origin)
        prediction = origin + deterministic + terms["N"]
        return {"prediction": prediction, "terms": terms, "base_residual": base_residual,
                "residual": z[:, 1:] - prediction}
