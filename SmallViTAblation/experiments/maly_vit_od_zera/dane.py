"""Dane dla małego ViT (PLAN.md): baza 1500 klatek treningowych, próbka val, pula do dodawania, sondy.

Klatki 720×480 zmniejszane do 480×320 (dokładnie 2/3), trzymane w RAM jako uint8. Cele gęstej głowy liczone raz,
osobno dla odbicia poziomego. Źródła tylko do odczytu (koopman_fastgan_petla, koopman_tozsamosc_kierunki).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

import model as M

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PETLA = EXP / 'koopman_fastgan_petla'
RUN = PETLA / 'wyniki/local_1810448/run'
DS = RUN / 'datasets/pretrain'
INDEKS = RUN / 'index'
PROBE = RUN / 'probe'
DATA_ROOT = PETLA / 'local_inputs/data'
WYKL = EXP / 'koopman_tozsamosc_kierunki/wyniki/dobor_klatek/wykluczone_pretrain_1500_plus_kotwice.json'
CACHE = HERE / 'cache'
W_, H_ = 480, 320
SKALA = 2 / 3
N_WAL = 3000


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def frame_id(sciezka):
    s = Path(sciezka).stem                                   # TR_0002_S1_P03_00000001
    return f'cataract_lmm/tracking/{s[:14]}/{s[15:]}'


def wczytaj(sciezka):
    im = cv2.imread(str(sciezka), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(sciezka)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (W_, H_), interpolation=cv2.INTER_AREA)


def _coco(plik, wybor=None):
    a = jz(plik)
    ann = {}
    for x in a['annotations']:
        b = x['bbox']; ann.setdefault(x['image_id'], []).append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
    ims = a['images'] if wybor is None else [a['images'][i] for i in wybor]
    return [dict(frame_id=frame_id(im['file_name']), sciezka=im['file_name'],
                 ramki=(np.array(ann.get(im['id'], []), np.float32) * SKALA).reshape(-1, 4)) for im in ims]


def _laduj(wiersze, nazwa):
    f = CACHE / f'{nazwa}.npy'
    if f.exists():
        X = np.load(f, mmap_mode=None)
        if len(X) == len(wiersze):
            return X
    CACHE.mkdir(exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(lambda r: wczytaj(r['sciezka']), wiersze)))
    np.save(f, X)
    return X


class Zbior:
    """Obrazy uint8 (N,320,480,3) + ramki + cele (dla wersji zwykłej i odbitej)."""

    def __init__(self, wiersze, X):
        self.wiersze, self.X = wiersze, X
        h, w = H_ // M.KROK, W_ // M.KROK
        self.cele = [M.cele([r['ramki'] for r in wiersze], h, w)]
        odb = [np.stack([W_ - r['ramki'][:, 2], r['ramki'][:, 1], W_ - r['ramki'][:, 0], r['ramki'][:, 3]], 1)
               if len(r['ramki']) else r['ramki'] for r in wiersze]
        self.cele.append(M.cele(odb, h, w))

    def __len__(self):
        return len(self.wiersze)

    def dolacz(self, inny):
        self.wiersze = self.wiersze + inny.wiersze
        self.X = np.concatenate([self.X, inny.X])
        self.cele = [tuple(torch.cat([a, b]) for a, b in zip(c1, c2)) for c1, c2 in zip(self.cele, inny.cele)]


def baza():
    w = _coco(DS / 'train/_annotations.coco.json')
    return Zbior(w, _laduj(w, 'baza1500'))


def walidacja():
    a = jz(DS / 'valid/_annotations.coco.json')
    wyb = sorted(np.random.default_rng(0).choice(len(a['images']), N_WAL, replace=False).tolist())
    w = _coco(DS / 'valid/_annotations.coco.json', wyb)
    return w, _laduj(w, f'wal{N_WAL}')


def pula(wykluczone):
    """Wiersze puli treningowej (split train, z ramką), bez wykluczonych frame_id."""
    out = []
    for l in open(INDEKS / 'manifest.jsonl', encoding='utf-8'):
        r = json.loads(l)
        if r['split'] != 'train' or not r['annotations'] or r['frame_id'] in wykluczone:
            continue
        b = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                      for a in r['annotations']], np.float32) * SKALA
        out.append(dict(frame_id=r['frame_id'], sciezka=str(DATA_ROOT / r['path']), ramki=b, video=r['video_id'],
                        klatka=int(r['frame_index'])))
    return out


def zbior_z_puli(wiersze):
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(lambda r: wczytaj(r['sciezka']), wiersze))) if wiersze else np.zeros((0, H_, W_, 3), np.uint8)
    return Zbior(wiersze, X)


WARIANTY16 = ('original', 'rotation90', 'rotation180', 'rotation270', 'black', 'yellow', 'scale08', 'length08',
              'shift_left', 'shift_right', 'shift_up', 'shift_down', 'foreground_dim', 'foreground_bright',
              'perspective_mild', 'empty_black')


S_SONDY = 176     # wycinek 256 px z klatki natywnej → 176 px ≈ skala 2/3 klatek treningowych; siatka 11×11


def sondy():
    """192 sondy (12 źródeł × 16 wariantów) zmniejszone do 176×176: obrazy (192,3,176,176) float, maski tokenów
    narzędzia i tła (192,121) bool na siatce 11×11, ramka z maski (xyxy) albo None, obecność narzędzia."""
    man = jz(PROBE / 'manifest.json')
    obrazy, tn, tt, ramki, obecn, opis = [], [], [], [], [], []
    g = S_SONDY // M.KROK
    for m in man:
        im = cv2.cvtColor(cv2.imread(str(PROBE / m['image'])), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (S_SONDY, S_SONDY), interpolation=cv2.INTER_AREA)
        mk = cv2.resize((cv2.imread(str(PROBE / m['mask']), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8),
                        (S_SONDY, S_SONDY), interpolation=cv2.INTER_NEAREST) > 0
        obrazy.append(im)
        frac = mk.reshape(g, M.KROK, g, M.KROK).mean((1, 3)).reshape(-1)    # udział maski w każdym tokenie
        tn.append(frac >= .25)
        dyl = cv2.dilate(mk.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        tt.append(dyl.reshape(g, M.KROK, g, M.KROK).mean((1, 3)).reshape(-1) == 0)
        ys, xs = np.nonzero(mk)
        ramki.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) and m['semantic_presence'] else None)
        obecn.append(bool(m['semantic_presence']))
        opis.append(dict(obraz=m['image'], wariant=m['variant'], zrodlo=m['source_file'], rola=m['role'],
                         frame_id=m['frame_id'], indeks=len(opis)))
    X = torch.from_numpy(np.stack(obrazy)).permute(0, 3, 1, 2).float() / 255.
    return X, torch.from_numpy(np.stack(tn)), torch.from_numpy(np.stack(tt)), ramki, np.array(obecn), opis


def wykluczone_bazowe(baza_wiersze):
    return set(jz(WYKL)) | {r['frame_id'] for r in baza_wiersze}


os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
