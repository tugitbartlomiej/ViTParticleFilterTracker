#!/usr/bin/env python3
"""Uczenie modelu dyfuzyjnego wycinków narzędzia (PLAN.md). Jeden plik wyjściowy na wszystko, wznawialne.

Dane: katalog z przygotuj_filmy.py (okna.bin + okna_indeks.npy + okna.json; okna z klatek filmów treningowych)
albo okna240.npy z przygotuj_dane.py. Każdy przykład: losowy wycinek 176×176 z okna 240 (środek obrysu narzędzia
zostaje w wycinku z marginesem 16 px), losowe odbicie poziome (jak w uczeniu małego ViT); dekodowanie w procesach
DataLoadera. Każdy krok: losowe t, strata MSE przewidzianego szumu. bf16, AdamW, rozgrzewka lr, obcięcie normy
gradientu 1, średnia wykładnicza wag (EMA).

Przerwanie: plik STOP w katalogu wyjściowym (SLURM tworzy go przy SIGTERM) albo SIGTERM do procesu → zapis
ostatni.pt i wyjście. Ponowne uruchomienie z tym samym --out wznawia od ostatni.pt.
Zapis: ostatni.pt (model, EMA, optymalizator, krok), ckpt_XXXXXX.pt (same wagi EMA, co --zapis-co),
model_ema.pt (na końcu), podglad_XXXXXX.png, LOG.jsonl, STATUS.json.

    python trenuj.py --dane ../dane/filmy3 --out ../wyniki/trening --kroki 100000 --batch 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import model_dyfuzji as MD

S, MARG = 176, 16
_STOP = {'sygnal': False}


def _sygnal(*_):
    _STOP['sygnal'] = True


def zapisz_atomowo(obj, p):
    tmp = Path(str(p) + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, p)


def zapisz_json(obj, p):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, p)


def zakres_przesuniec(k, okno):
    """Dopuszczalne przesunięcie wycinka 176 w oknie, żeby punkt k (środek obrysu) był w wycinku z marginesem."""
    lo = int(np.clip(math.ceil(k - (S - MARG)), 0, okno - S)); hi = int(np.clip(math.floor(k - MARG), 0, okno - S))
    return (lo, hi) if lo <= hi else (0, okno - S)


class Okna(torch.utils.data.Dataset):
    """Okna 240×240 RGB+maska; zwraca losowy wycinek (4,176,176) uint8. Plik danych otwierany leniwie w procesie."""

    def __init__(self, sciezka, limit=0):
        p = Path(sciezka)
        self.npy = p.suffix == '.npy'
        if self.npy:
            self.plik, opisy = p, json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
            self.okno = int(np.load(p, mmap_mode='r').shape[1])
        else:
            self.plik, opisy = p / 'okna.bin', json.loads((p / 'okna.json').read_text(encoding='utf-8'))
            self.ind = np.load(p / 'okna_indeks.npy'); self.okno = 240
        if limit:
            opisy = opisy[:limit]
        self.n = len(opisy)
        pkt = [d.get('srodek', d.get('koncowka')) for d in opisy]
        self.zak = np.array([(*zakres_przesuniec(x, self.okno), *zakres_przesuniec(y, self.okno)) for x, y in pkt], np.int64)
        self._d = None

    def __len__(self):
        return self.n

    def _okno(self, i):
        if self._d is None:
            self._d = np.load(self.plik, mmap_mode='r') if self.npy else np.memmap(self.plik, dtype=np.uint8, mode='r')
        if self.npy:
            return np.asarray(self._d[i])
        a, la, b, lb = self.ind[i]
        bgr = cv2.imdecode(np.frombuffer(self._d[a:a + la], np.uint8), cv2.IMREAD_COLOR)
        m = cv2.imdecode(np.frombuffer(self._d[b:b + lb], np.uint8), cv2.IMREAD_GRAYSCALE)
        return np.dstack([bgr[..., ::-1], m])

    def __getitem__(self, i):
        o = self._okno(i)
        lx, hx, ly, hy = self.zak[i]
        ox, oy = int(torch.randint(lx, hx + 1, ())), int(torch.randint(ly, hy + 1, ()))
        w = o[oy:oy + S, ox:ox + S]
        if torch.rand(()) < .5:
            w = w[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(w)).permute(2, 0, 1)


def ladowarka(ds, B, workery, seed):
    g = torch.Generator().manual_seed(seed)
    los = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=1 << 40, generator=g)
    kw = dict(num_workers=workery, persistent_workers=True, prefetch_factor=4) if workery else {}
    return iter(torch.utils.data.DataLoader(ds, batch_size=B, sampler=los, pin_memory=True, drop_last=True, **kw))


def siatka_png(x, p, kolumny=4):
    """x (n,4,S,S) w [-1,1]: górny rząd RGB, dolny RGB z maską na czerwono."""
    x = ((x.float().clamp(-1, 1) + 1) / 2).cpu()
    rgb, m = x[:, :3], x[:, 3:4]
    nak = rgb * (1 - .45 * m) + .45 * m * torch.tensor([1., 0, 0]).view(1, 3, 1, 1)
    kafle = torch.cat([rgb, nak], 3)                                              # obok siebie
    n = kafle.shape[0]; w = (n + kolumny - 1) // kolumny
    kafle = torch.cat([kafle, torch.zeros(w * kolumny - n, *kafle.shape[1:])])
    g = kafle.view(w, kolumny, 3, S, 2 * S).permute(0, 3, 1, 4, 2).reshape(w * S, kolumny * 2 * S, 3)
    Image.fromarray((g.numpy() * 255).round().astype(np.uint8)).save(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dane', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kroki', type=int, default=100_000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rozgrzewka', type=int, default=1000)
    ap.add_argument('--ema', type=float, default=0.9995)
    ap.add_argument('--zapis-co', type=int, default=10_000)
    ap.add_argument('--ostatni-co', type=int, default=1000)
    ap.add_argument('--podglad-co', type=int, default=5000)
    ap.add_argument('--log-co', type=int, default=100)
    ap.add_argument('--baza', type=int, default=64, help='kanały pierwszego poziomu UNet')
    ap.add_argument('--limit', type=int, default=0, help='tylko pierwsze N okien (próba)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workery', type=int, default=6, help='procesy dekodujące dane (0 = w procesie głównym)')
    a = ap.parse_args()
    dev = 'cuda'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stop_plik = out / 'STOP'
    if stop_plik.exists():
        stop_plik.unlink()                                                        # pozostałość po poprzednim fragmencie
    signal.signal(signal.SIGTERM, _sygnal)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    ds = Okna(a.dane, a.limit)
    with open(ds.plik, 'rb') as f:
        sha_danych = hashlib.sha256(f.read(1 << 26)).hexdigest()                  # pierwsze 64 MB: tożsamość pliku
    print(f'dane: {ds.n} okien {ds.okno}×{ds.okno} z {ds.plik}', flush=True)

    model = MD.UNet(baza=a.baza).to(dev)
    ema = MD.EMA(model, a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    harm = MD.Harmonogram(dev=dev)
    parametrow = sum(p.numel() for p in model.parameters())
    krok = 0
    if (out / 'ostatni.pt').exists():
        p = torch.load(out / 'ostatni.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(p['model']); ema.m.load_state_dict(p['ema']); opt.load_state_dict(p['opt'])
        krok = p['krok']
        torch.manual_seed(a.seed + krok)
        print(f'wznowienie od kroku {krok}', flush=True)
    else:
        torch.manual_seed(a.seed)
    if krok >= a.kroki:
        print(f'już ukończone: krok {krok} >= {a.kroki}', flush=True)
        return 0
    torch.backends.cudnn.benchmark = True
    cfg = dict(model.cfg)
    arg = vars(a)

    def stan(etykieta, s_na_krok=None, strata=None):
        zapisz_json(dict(stan=etykieta, krok=krok, kroki=a.kroki, parametrow=parametrow, okien=ds.n,
                         s_na_krok=s_na_krok, godzin_do_konca=(a.kroki - krok) * s_na_krok / 3600 if s_na_krok else None,
                         strata=strata, gpu=torch.cuda.get_device_name(0), czas=time.strftime('%Y-%m-%d %H:%M:%S'),
                         argumenty=arg), out / 'STATUS.json')

    def ostatni():
        zapisz_atomowo(dict(model=model.state_dict(), ema=ema.m.state_dict(), opt=opt.state_dict(), krok=krok, cfg=cfg,
                            argumenty=arg, dane_sha256_64mb=sha_danych), out / 'ostatni.pt')

    def podglad():
        x = MD.probkuj(ema.m, harm, 16, kroki=50, seed=1234, dev=dev)
        siatka_png(x, out / f'podglad_{krok:06d}.png')

    log = open(out / 'LOG.jsonl', 'a', encoding='utf-8')
    kosze = torch.zeros(4, device=dev); licz = torch.zeros(4, device=dev); suma, n_sum = 0.0, 0
    tlog = time.time(); s_na_krok = None; wpis: dict = {}
    stan('trwa')
    dane = ladowarka(ds, a.batch, a.workery, a.seed * 1_000_003 + krok)
    model.train()
    while krok < a.kroki:
        x0 = next(dane).to(dev, non_blocking=True).float().div(127.5).sub(1)
        t = torch.randint(0, harm.T, (a.batch,), device=dev)
        eps = torch.randn_like(x0)
        xt = harm.zaszum(x0, t, eps)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(xt, t)
        strata_pr = (pred.float() - eps).pow(2).mean((1, 2, 3))
        strata = strata_pr.mean()
        opt.zero_grad(set_to_none=True)
        strata.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = a.lr * min(1.0, (krok + 1) / a.rozgrzewka)
        opt.step()
        ema.aktualizuj(model, krok)
        krok += 1
        with torch.no_grad():
            k = (t * 4 // harm.T).clamp(max=3)
            kosze.index_add_(0, k, strata_pr.detach()); licz.index_add_(0, k, torch.ones_like(strata_pr))
        suma += float(strata); n_sum += 1
        if krok % a.log_co == 0:
            s_na_krok = (time.time() - tlog) / a.log_co; tlog = time.time()
            wpis = dict(krok=krok, strata=suma / n_sum, lr=opt.param_groups[0]['lr'], s_na_krok=round(s_na_krok, 4),
                        strata_wg_t=[round(float(v), 5) for v in (kosze / licz.clamp(min=1))],
                        pamiec_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            log.write(json.dumps(wpis) + '\n'); log.flush()
            if krok % (a.log_co * 10) == 0:
                print(wpis, flush=True)
            kosze.zero_(); licz.zero_(); suma, n_sum = 0.0, 0
        if krok % a.ostatni_co == 0:
            ostatni(); stan('trwa', s_na_krok, wpis.get('strata'))
        if krok % a.zapis_co == 0:
            zapisz_atomowo(dict(cfg=cfg, ema=ema.m.state_dict(), krok=krok), out / f'ckpt_{krok:06d}.pt')
        if krok % a.podglad_co == 0:
            model.eval(); podglad(); model.train()
        if _STOP['sygnal'] or stop_plik.exists():
            ostatni(); stan('przerwany', s_na_krok)
            print(f'przerwano na kroku {krok}; wznowienie tym samym poleceniem', flush=True)
            return 0
    ostatni()
    zapisz_atomowo(dict(cfg=cfg, ema=ema.m.state_dict(), krok=krok, argumenty=arg, dane_sha256_64mb=sha_danych),
                   out / 'model_ema.pt')
    model.eval(); podglad()
    stan('ukonczony', s_na_krok)
    print(f'koniec, krok {krok}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
