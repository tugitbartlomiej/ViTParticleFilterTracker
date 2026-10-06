"""Konwerter pełnego latentu małego ViT (600 x 768) na cechy DINOv2 ViT-S/14 tego samego zdjęcia.

Wejście: naturalne aktywacje po GELU ukrytej warstwy MLP ostatniego bloku ViT z checkpointu (siatka 20x30).
Wyjście: CLS 384 po normalizacji L2 (zgodny z DINO_SPEC i indeksem FAISS) oraz 600 tokenów 384 na tej
samej siatce co DINO uruchomione na obrazie 280x420. Kolejność tokenów i ich amplituda zostają; jedyna
normalizacja to stała standaryzacja kanałów policzona na TRAIN przed uczeniem.
Kontrola w tym samym biegu: liniowa głowa na uśrednionym latencie (bez układu przestrzennego).
"""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.vision_transformer import Block

from model_vit import MalyViT

DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DINO_WAGI_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
DINO_KOD_SHA = '7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blok in iter(lambda: f.read(1 << 22), b''):
            h.update(blok)
    return h.hexdigest()


class Konwerter(nn.Module):
    def __init__(self, dim=384, glebokosc=6, glowy=6, we=768, wy=384, dropout=0.0, drop_path=0.0, maska=0.0):
        super().__init__()
        self.konf = dict(dim=dim, glebokosc=glebokosc, glowy=glowy, we=we, wy=wy, dropout=dropout, drop_path=drop_path, maska=maska)
        self.maska = maska
        self.token_maski = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer('x_mean', torch.zeros(we))
        self.register_buffer('x_std', torch.ones(we))
        self.wejscie = nn.Linear(we, dim)
        self.poz = nn.Parameter(torch.randn(1, 600, dim) * .02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.bloki = nn.ModuleList([Block(dim, glowy, 4.0, qkv_bias=True, proj_drop=dropout, attn_drop=dropout,
                                          drop_path=drop_path * i / max(1, glebokosc - 1)) for i in range(glebokosc)])
        self.norm = nn.LayerNorm(dim)
        self.glowa_cls = nn.Linear(dim, wy)
        self.glowa_patch = nn.Linear(dim, wy)

    def forward(self, z):
        if z.shape[1:] != (600, 768):
            raise ValueError(f'Oczekiwano Bx600x768, jest {tuple(z.shape)}')
        x = (z - self.x_mean) / self.x_std
        t = self.wejscie(x)
        if self.training and self.maska > 0:
            m = torch.rand(t.shape[:2], device=t.device) < self.maska
            t = torch.where(m[..., None], self.token_maski.to(t.dtype).expand_as(t), t)
        t = t + self.poz
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        for b in self.bloki:
            t = b(t)
        t = self.norm(t)
        return F.normalize(self.glowa_cls(t[:, 0]).float(), dim=-1), self.glowa_patch(t[:, 1:]).float()


class KontrolaLiniowa(nn.Module):
    """Średnia po tokenach (po tej samej standaryzacji) -> Linear -> CLS 384. Bez układu przestrzennego."""

    def __init__(self, we=768, wy=384):
        super().__init__()
        self.glowa = nn.Linear(we, wy)

    def forward(self, z, x_mean, x_std):
        return F.normalize(self.glowa(((z - x_mean) / x_std).mean(1)).float(), dim=-1)


def wczytaj_vit(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if 'net' not in ck:
        raise ValueError('Checkpoint ViT musi mieć klucz net')
    vit = MalyViT()
    vit.load_state_dict(ck['net'], strict=True)
    vit.to(device).eval().requires_grad_(False)
    vit.zapisuj_latent(True)
    return vit, ck.get('krok')


@torch.no_grad()
def latent_vit(vit, obraz_u8):
    """obraz_u8: Bx3x320x480 uint8 na GPU -> naturalny latent Bx600x768 float32 (zwykły forward, hak tylko czyta)."""
    _, z = vit(obraz_u8.float().div_(255))
    vit._latent = None
    if z is None or z.shape[1:] != (600, 768):
        raise ValueError('ViT nie zwrócił aktywacji 600x768')
    return z.float()


def wczytaj_dino(repo, wagi, device, pozwol_inny_kod=False):
    if sha256(wagi) != DINO_WAGI_SHA:
        raise ValueError('Wagi DINOv2 różnią się od DINO_SPEC')
    kod = sha256(Path(repo) / 'dinov2/models/vision_transformer.py')
    if kod != DINO_KOD_SHA and not pozwol_inny_kod:
        raise ValueError(f'Kod DINOv2 różni się od DINO_SPEC: {kod}')
    dino = torch.hub.load(str(repo), 'dinov2_vits14', source='local', pretrained=False)
    dino.load_state_dict(torch.load(wagi, map_location='cpu', weights_only=True), strict=True)
    return dino.to(device).eval().requires_grad_(False), kod


@torch.no_grad()
def cechy_dino(dino, cls_u8, patch_u8):
    """CLS 384 L2 z obrazu 224 (DINO_SPEC) i tokeny 20x30x384 L2 z obrazu 280x420."""
    m, s = DINO_MEAN.to(cls_u8.device), DINO_STD.to(cls_u8.device)
    y_cls = dino.forward_features((cls_u8.float() / 255 - m) / s)['x_norm_clstoken']
    y_patch = dino.forward_features((patch_u8.float() / 255 - m) / s)['x_norm_patchtokens']
    if y_patch.shape[1] != 600:
        raise ValueError(f'DINO dał {y_patch.shape[1]} tokenów zamiast 600')
    return F.normalize(y_cls.float(), dim=-1), F.normalize(y_patch.float(), dim=-1)


class KonwerterGotowy:
    """Użycie po treningu: KonwerterGotowy('KONWERTER.pt').predict(latent Nx600x768) -> (CLS Nx384, tokeny Nx600x384)."""

    def __init__(self, path, device='cpu'):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.meta = ck['meta']
        self.model = Konwerter(**ck['konf'])
        sd = dict(ck['model'])
        sd.setdefault('token_maski', torch.zeros(1, 1, ck['konf']['dim']))  # checkpointy wersji 1
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, latent):
        z = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        if z.dim() == 2:
            z = z[None]
        cls, patch = self.model(z)
        return cls, F.normalize(patch, dim=-1)
