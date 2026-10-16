"""Mały hybrydowy ViT od zera z gęstą głową (PLAN.md).

Łodyga: 4 konwolucje 3×3 z krokiem 2 (jak CCT / VitTrack) → siatka o kroku 16 px. Pozycje: stałe 2D sin-cos
(działa dla każdej wielkości siatki: klatki 480×320 → 30×20, sondy 256×256 → 16×16). Bloki timm.
Głowa gęsta, na każdy token: logit „tu jest środek narzędzia” i odległości do krawędzi ramki (l, t, r, b) w jednostkach
kroku (styl CenterNet/FCOS, bez dopasowania węgierskiego).
Latent do Koopmana: aktywacje neuronów warstwy ukrytej MLP ostatniego bloku (768 na token, po GELU) — „neurony,
które się zapalają”.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Mlp(nn.Module):
    """Minimalny MLP zgodny stanem i forwardem z domyślnym Mlp timm."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class _Attention(nn.Module):
    """Domyślna wielogłowicowa atencja bloku ViT użytego w kampanii."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim musi być podzielne przez num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x, attn_mask=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Samodzielny odpowiednik parametrów domyślnego timm ViT Block.

    Kampania używała wyłącznie qkv_bias=True, bez dropoutu, LayerScale,
    stochastic depth ani QK norm. Nazwy modułów pozostają identyczne, dzięki
    czemu dokładny K1000 ładuje się ze strict=True.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads, qkv_bias)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

KROK = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sincos_2d(h, w, dim, device):
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    d = dim // 4
    om = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=torch.float32) / d))
    ox, oy = x.reshape(-1, 1) * om, y.reshape(-1, 1) * om
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)          # (h*w, dim)


def _cba(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.GELU())


class MalyViT(nn.Module):
    def __init__(self, dim=192, glebokosc=6, glowy=3, mlp=4.0):
        super().__init__()
        self.lodyga = nn.Sequential(_cba(3, 24), _cba(24, 48), _cba(48, 96), nn.Conv2d(96, dim, 3, 2, 1))
        self.bloki = nn.ModuleList([Block(dim, glowy, mlp, qkv_bias=True) for _ in range(glebokosc)])
        self.norm = nn.LayerNorm(dim)
        self.glowa = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 5))
        nn.init.constant_(self.glowa[-1].bias, 0.0)
        with torch.no_grad():
            self.glowa[-1].bias[0] = -4.6                                  # prior 0,01 dla mapy środków
            self.glowa[-1].bias[1:] = math.log(3.0)                         # ~3 kroki do krawędzi na starcie
        self._latent = None
        self._hak = None

    def zapisuj_latent(self, tak: bool):
        if tak and self._hak is None:
            self._hak = self.bloki[-1].mlp.act.register_forward_hook(lambda m, i, o: setattr(self, '_latent', o))
        if not tak and self._hak is not None:
            self._hak.remove(); self._hak = None; self._latent = None

    def forward(self, x):
        """x: (B,3,H,W) w [0,1]. Zwraca mapę (B,h,w,5) i latent (B,h*w,768) gdy zapisywany."""
        x = (x - MEAN.to(x.device, x.dtype)) / STD.to(x.device, x.dtype)
        f = self.lodyga(x)
        B, C, h, w = f.shape
        t = f.flatten(2).transpose(1, 2) + sincos_2d(h, w, C, x.device).to(f.dtype)
        for b in self.bloki:
            t = b(t)
        o = self.glowa(self.norm(t)).view(B, h, w, 5)
        return o, self._latent


def cele(ramki, h, w):
    """ramki: lista (n_i,4) xyxy w pikselach wejścia. Zwraca cel mapy (B,h,w), cele ltrb (B,h,w,4) w krokach, maskę."""
    B = len(ramki)
    heat = torch.zeros(B, h, w); ltrb = torch.zeros(B, h, w, 4); poz = torch.zeros(B, h, w)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
    cx_c, cy_c = (xs + .5) * KROK, (ys + .5) * KROK                        # środki komórek w pikselach
    for i, rr in enumerate(ramki):
        for x0, y0, x1, y1 in rr:
            cx, cy = (x0 + x1) / 2 / KROK - .5, (y0 + y1) / 2 / KROK - .5
            sx, sy = max(.7, (x1 - x0) / KROK / 6), max(.7, (y1 - y0) / KROK / 6)
            g = torch.exp(-((xs - cx) ** 2 / (2 * sx ** 2) + (ys - cy) ** 2 / (2 * sy ** 2)))
            ix, iy = min(max(int(round(cx)), 0), w - 1), min(max(int(round(cy)), 0), h - 1)
            g[iy, ix] = 1.0
            heat[i] = torch.maximum(heat[i], g)
            wew = (cx_c > x0) & (cx_c < x1) & (cy_c > y0) & (cy_c < y1) & (g > .3)
            wew[iy, ix] = True
            d = torch.stack([cx_c - x0, cy_c - y0, x1 - cx_c, y1 - cy_c], -1) / KROK
            ltrb[i][wew] = d[wew].clamp(min=.05); poz[i][wew] = torch.maximum(poz[i][wew], g[wew])
    return heat, ltrb, poz


def _giou(a, b):
    """a, b: (n,4) ltrb względem tego samego punktu."""
    ia = (torch.min(a[:, 0], b[:, 0]) + torch.min(a[:, 2], b[:, 2])) * (torch.min(a[:, 1], b[:, 1]) + torch.min(a[:, 3], b[:, 3]))
    pa = (a[:, 0] + a[:, 2]) * (a[:, 1] + a[:, 3]); pb = (b[:, 0] + b[:, 2]) * (b[:, 1] + b[:, 3])
    u = pa + pb - ia
    c = (torch.max(a[:, 0], b[:, 0]) + torch.max(a[:, 2], b[:, 2])) * (torch.max(a[:, 1], b[:, 1]) + torch.max(a[:, 3], b[:, 3]))
    return ia / u - (c - u) / c


def strata(o, heat, ltrb, poz):
    """Ogniskowa strata CenterNet na mapie środków + (1 − GIoU) na komórkach wewnątrz ramki."""
    p = o[..., 0].float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    l_pos = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    l_neg = -(torch.log(1 - p) * p ** 2 * (1 - heat) ** 4 * (1 - pos)).sum()
    n = pos.sum().clamp(min=1)
    l_heat = (l_pos + l_neg) / n
    m = poz > 0
    if m.any():
        pr = o[..., 1:].float().exp()[m]
        w = poz[m]
        l_box = ((1 - _giou(pr, ltrb[m])) * w).sum() / w.sum()
    else:
        l_box = o.sum() * 0
    return l_heat + 2 * l_box, dict(heat=float(l_heat), box=float(l_box))


@torch.no_grad()
def dekoduj(o, k=10):
    """Szczyty mapy (3×3) → do k ramek xyxy w pikselach wejścia i pewności. Zwraca listę (boxes (k,4), scores (k,))."""
    B, h, w, _ = o.shape
    p = o[..., 0].float().sigmoid()
    mx = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]
    p = p * (p == mx)
    s, idx = p.view(B, -1).topk(min(k, h * w), 1)
    ys, xs = (idx // w).float(), (idx % w).float()
    d = o[..., 1:].float().exp().view(B, -1, 4).gather(1, idx[..., None].expand(-1, -1, 4)) * KROK
    cx, cy = (xs + .5) * KROK, (ys + .5) * KROK
    bx = torch.stack([cx - d[..., 0], cy - d[..., 1], cx + d[..., 2], cy + d[..., 3]], -1)
    return bx, s
