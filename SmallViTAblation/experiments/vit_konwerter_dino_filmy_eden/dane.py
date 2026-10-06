"""Strumień klatek z filmów MP4 czytanych prosto z archiwów ZIP do RAM, bez zapisu na dysk.

Każda klatka trafia do pętli uczenia jako trzy obrazy uint8 policzone z TYCH SAMYCH pikseli:
  vit   3x320x480  cv2.INTER_AREA, jak w całym potoku małego ViT (latent 20x30x768),
  cls   3x224x224  dokładnie DINO_SPEC: PIL Resize(256, BICUBIC) + CenterCrop(224), jak indeks FAISS,
  patch 3x280x420  cv2.INTER_AREA; DINOv2 ViT-S/14 daje wtedy siatkę 20x30, tę samą co tokeny ViT.
Etykiety narzędzia nie są potrzebne: cel liczy się z obrazu, więc wyrównanie CSV/MP4 nie wpływa na uczenie.

Uczenie: każdy proces ładujący przeplata kilka filmów naraz i miesza klatki w buforze w RAM,
żeby paczka nie składała się z kolejnych, prawie identycznych klatek jednego filmu.
"""
import io
import json
import math
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms as T

_CLS = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])


def wczytaj_manifest(sciezka, zamiana=None):
    """Lista filmów z FILMY.json (format z vit_sd_pelny_latent_filmy_eden_20261006/build_manifest.py).

    zamiana=(stary_prefiks, nowy_prefiks) przepina ścieżki, np. na lokalny test dymny.
    """
    rows = json.loads(Path(sciezka).read_text(encoding='utf-8-sig'))
    out = []
    for r in rows:
        r = dict(r)
        if zamiana:
            r['zip_path'] = r['zip_path'].replace(zamiana[0], zamiana[1])
        r['split'] = r['split'].lower()
        out.append(r)
    out.sort(key=lambda r: r['video_id'])
    if len({r['video_id'] for r in out}) != len(out):
        raise ValueError('Powtórzony film w manifeście')
    return out


def przygotuj(rgb):
    """rgb: HxWx3 uint8 (natywna klatka). Zwraca trzy tensory uint8 CHW."""
    vit = cv2.resize(rgb, (480, 320), interpolation=cv2.INTER_AREA)
    patch = cv2.resize(rgb, (420, 280), interpolation=cv2.INTER_AREA)
    cls = np.array(_CLS(Image.fromarray(rgb)))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).contiguous()
    return t(vit), t(cls), t(patch)


def klatki_mp4(row, wybrane=None):
    """(numer od 1, rgb) z filmu MP4 czytanego z ZIP do RAM. wybrane: zbiór numerów albo None (wszystkie)."""
    import av
    with zipfile.ZipFile(row['zip_path']) as z:
        dane = z.read(row['member'])
    with io.BytesIO(dane) as buf, av.open(buf, mode='r') as kont:
        s = kont.streams.video[0]
        s.codec_context.thread_count = 1
        for n, kl in enumerate(kont.decode(s), 1):
            if wybrane is None or n in wybrane:
                yield n, kl.to_ndarray(format='rgb24')


def augmentuj(rgb, rng, hflip=0.0, crop_min=1.0, jitter=0.0):
    """Przekształca NATYWNĄ klatkę, zanim powstaną obrazy dla ViT i DINO, więc para latent-cel zostaje zgodna."""
    if hflip and rng.random() < hflip:
        rgb = rgb[:, ::-1]
    if crop_min < 1.0:
        H, W = rgb.shape[:2]
        s, a = rng.uniform(crop_min, 1.0), math.exp(rng.uniform(math.log(.85), math.log(1.18)))
        h, w = min(H, round(H * math.sqrt(s / a))), min(W, round(W * math.sqrt(s * a)))
        y, x = rng.randrange(H - h + 1), rng.randrange(W - w + 1)
        rgb = rgb[y:y + h, x:x + w]
    rgb = np.ascontiguousarray(rgb)
    if jitter:
        rgb = cv2.convertScaleAbs(rgb, alpha=rng.uniform(1 - jitter, 1 + jitter), beta=rng.uniform(-jitter, jitter) * 64)
    return rgb


def _probka(rgb, film_id, nr, rng=None, aug=None):
    if aug and rng is not None:
        rgb = augmentuj(rgb, rng, **aug)
    vit, cls, patch = przygotuj(rgb)
    return dict(vit=vit, cls=cls, patch=patch, film=film_id, klatka=nr)


class StrumienUczacy(IterableDataset):
    """Nieskończony strumień klatek TRAIN, filmy dzielone po rangach DDP i procesach ładujących."""

    def __init__(self, rows, rank, world, seed, przeplot=4, bufor=256, aug=None):
        super().__init__()
        self.rows = [r for r in rows if r['split'] == 'train']
        self.film_ids = {r['video_id']: i for i, r in enumerate(rows)}
        self.rank, self.world, self.seed = rank, world, seed
        self.przeplot, self.bufor, self.aug = przeplot, bufor, aug

    def __iter__(self):
        cv2.setNumThreads(1)
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        moje = self.rows[self.rank::self.world][wid::nw]
        if not moje:
            raise RuntimeError(f'Brak filmów dla rangi {self.rank}, procesu {wid}: zmniejsz liczbę procesów')
        rng = random.Random(self.seed * 7919 + self.rank * 1009 + wid)
        kolejka = []

        def nastepny():
            if not kolejka:
                kolejka.extend(rng.sample(moje, len(moje)))
            row = kolejka.pop()
            return row, klatki_mp4(row)

        aktywne = [nastepny() for _ in range(min(self.przeplot, len(moje)))]
        bufor = []
        while True:
            i = rng.randrange(len(aktywne))
            row, gen = aktywne[i]
            try:
                nr, rgb = next(gen)
            except StopIteration:
                aktywne[i] = nastepny()
                continue
            bufor.append(_probka(rgb, self.film_ids[row['video_id']], nr, rng, self.aug))
            if len(bufor) >= self.bufor:
                j = rng.randrange(len(bufor))
                bufor[j], bufor[-1] = bufor[-1], bufor[j]
                yield bufor.pop()


class ZbiorOceny(IterableDataset):
    """Skończony, deterministyczny zbiór VAL: co `krok`-ta klatka każdego filmu VAL."""

    def __init__(self, rows, rank, world, krok=15, max_filmow=None):
        super().__init__()
        val = [r for r in rows if r['split'] == 'val'][:max_filmow]
        self.rows = val[rank::world]
        self.film_ids = {r['video_id']: i for i, r in enumerate(rows)}
        self.krok = krok

    def __iter__(self):
        cv2.setNumThreads(1)
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for row in self.rows[wid::nw]:
            wybrane = set(range(1, int(row['frame_count']) + 1, self.krok))
            for nr, rgb in klatki_mp4(row, wybrane):
                yield _probka(rgb, self.film_ids[row['video_id']], nr)


def zloz(probki):
    out = {k: torch.stack([p[k] for p in probki]) for k in ('vit', 'cls', 'patch')}
    for k in ('film', 'klatka'):
        out[k] = torch.tensor([p[k] for p in probki], dtype=torch.long)
    return out
