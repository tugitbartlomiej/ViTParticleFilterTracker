"""DeiT-Tiny dla CIFAR-10 (obraz 32×32): drugi model i drugi zbiór do porównania z MałymViT (bieg s47).

Architektura DeiT-Tiny (timm `deit_tiny_patch16_224`): wymiar 192, 12 bloków pre-LN, 3 głowy, MLP 192 → 768 → 192 z GELU,
token klasy CLS, uczone kodowanie pozycji, głowa liniowa na CLS. Zmiana pod CIFAR: łatka 4×4, więc siatka 8×8 = 64 tokeny
obrazu (+ CLS). Bez tokenu destylacji (nie ma nauczyciela). Uczony od zera.

Interfejs jak MałyViT (obserwator_gesty / obserwator_rp): `net.bloki` (bloki timm z `mlp.act`), `zapisuj_latent(bool)`,
`forward(x) -> (logity, latent)`, gdzie latent = wyjście GELU w MLP ostatniego bloku dla 64 tokenów obrazu (bez CLS),
kształt (B, 64, 768) — ten sam obiekt co „natężenia z_jc” 768 reflektorów w MałymViT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)          # średnia i odchylenie CIFAR-10 (zbiór uczący)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


class DeitCifar(nn.Module):
    def __init__(self, n_klas=10, rozmiar=32, latka=4, dim=192, glebokosc=12, glowy=3):
        super().__init__()
        self.vit = VisionTransformer(img_size=rozmiar, patch_size=latka, in_chans=3, num_classes=n_klas, embed_dim=dim,
                                     depth=glebokosc, num_heads=glowy, mlp_ratio=4.0, qkv_bias=True, class_token=True,
                                     global_pool='token')
        self.siatka = rozmiar // latka
        self.tokenow = self.siatka ** 2
        self._latent = None
        self._hak = None

    @property
    def bloki(self):
        return self.vit.blocks

    def zapisuj_latent(self, tak: bool):
        if tak and self._hak is None:
            self._hak = self.vit.blocks[-1].mlp.act.register_forward_hook(lambda m, i, o: setattr(self, '_latent', o[:, 1:]))
        elif not tak and self._hak is not None:
            self._hak.remove(); self._hak = None; self._latent = None

    def forward(self, x):
        x = (x - MEAN.to(x.device, x.dtype)) / STD.to(x.device, x.dtype)
        return self.vit(x), self._latent
