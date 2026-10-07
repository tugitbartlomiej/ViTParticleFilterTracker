#!/usr/bin/env python3
"""Siły działające na KAŻDĄ cząstkę (token) w każdym kroku uczenia małego ViT (plan: ../PLAN.md).

Przepis uczenia identyczny jak gęsta historia (maly_vit_ruch_czastek_gesta_historia, ziarno 47): trening.Uczen(47),
rng = default_rng(47), baza 1500 klatek, paczka 32, AdamW 5e-4, wd 0,05, rozgrzewka lr 300, obcinanie gradientu 1,
bf16. Pomiar nie zmienia uczenia (no_grad, eval, fp32 bez TF32, bez generatorów losowych).

W każdym kroku t (wagi theta_t, bufory BatchNorm b_t -> theta_(t+1), b_(t+1)), dla 3 klatek próbnych i 600 tokenów:
  Z      z(theta_t, b_t)                               latent przed krokiem (wiersz T = stan końcowy)
  S_nowa z(theta_t + d_nowa, b_t) − z(theta_t, b_t)     d_nowa = −lr/(1−b1^k) (1−b1) g_t / mian   (nowa paczka)
  S_ped  to samo dla d_ped = dtheta − d_nowa − d_zanik  (pęd: pamięć poprzednich paczek w AdamW)
  S_zan  to samo dla d_zanik = −lr wd theta_t           (zanik wag)
  S_buf  z(theta_t, b_(t+1)) − z(theta_t, b_t)          (zmiana statystyk biegnących BatchNorm łodygi)
Rzeczywisty ruch dz_t = Z[t+1] − Z[t]; reszta nieliniowości = dz − (S_nowa + S_ped + S_zan + S_buf) (log).
Co --grupy-co kroków: S_grupa[g] = z(theta_t + dtheta_g, b_t) − z(theta_t, b_t) dla 13 grup wag
(łodyga; atencja i MLP każdego z 6 bloków).
Wszystko float16 w memmapach .npy (dane/), plus SKALARY.json (lr, strata, norma gradientu, reszta liniowości).

    python -u bieg_sily.py --wyniki <katalog> [--krokow 1000] [--klatki 2 3 6] [--proba]
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import dane as D
import trening as T

HERE = Path(__file__).resolve().parent
WEJ12 = Path(os.environ.get('CZASTKI_WEJSCIA', HERE.parent / 'wejscia'))
GRUPY = ['łodyga'] + [f'blok {b + 1} {c}' for b in range(6) for c in ('atencja', 'MLP')]


def grupa(n):
    if n.startswith('lodyga'):
        return 0
    if n.startswith('bloki.'):
        b = int(n.split('.')[1]); r = n.split('.', 2)[2]
        return 1 + 2 * b + (0 if r.startswith(('norm1', 'attn', 'ls1')) else 1)
    return None


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float) + '\n', encoding='utf-8')


class PomiarSil:
    def __init__(self, X, out, T_, grupy_co):
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().float().cuda() / 255.
        n = len(X); sh = (n, 600, 768)
        mm = lambda nazwa, k: np.lib.format.open_memmap(out / 'dane' / f'{nazwa}.npy', mode='w+', dtype=np.float16, shape=(k,) + sh)
        self.Z = mm('Z', T_ + 1)
        self.S = {k: mm(f'S_{k}', T_) for k in ('nowa', 'ped', 'zan', 'buf')}
        self.grupy_co = grupy_co
        self.kroki_grup = list(range(0, T_, grupy_co))
        self.SG = mm('S_grupy', len(self.kroki_grup) * len(GRUPY)) if grupy_co else None
        self.skal = []
        self.snap = None

    def czy(self, j):
        return True

    @torch.no_grad()
    def z(self, net, params, bufs):
        m32, c32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        net.eval(); net.zapisuj_latent(True)
        try:
            torch.func.functional_call(net, (params, bufs), (self.X,))
            return net._latent.float().clone()
        finally:
            net.zapisuj_latent(False)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m32, c32

    def przed(self, u, *a):
        self.snap = ({n: p.detach().clone() for n, p in u.net.named_parameters()},
                     {n: b.detach().clone() for n, b in u.net.named_buffers()})

    def po(self, u):
        t = u.krok - 1                                         # indeks kroku, który właśnie się wykonał
        net, opt = u.net, u.opt
        th0, b0 = self.snap
        th1 = {n: p.detach() for n, p in net.named_parameters()}
        b1_ = {n: b.detach() for n, b in net.named_buffers()}
        nazwy = {id(p): n for n, p in net.named_parameters()}
        d_nowa, d_zan, d_ped = {}, {}, {}
        with torch.no_grad():
            for grp in opt.param_groups:
                lr, (be1, be2), eps, wd = grp['lr'], grp['betas'], grp['eps'], grp['weight_decay']
                for p in grp['params']:
                    n = nazwy[id(p)]
                    if p.grad is None:
                        d_nowa[n] = torch.zeros_like(p); d_zan[n] = torch.zeros_like(p); d_ped[n] = th1[n] - th0[n]
                        continue
                    st = opt.state[p]
                    k = float(st['step'])
                    mian = st['exp_avg_sq'].sqrt() / math.sqrt(1 - be2 ** k) + eps
                    d_nowa[n] = -(lr / (1 - be1 ** k)) * (1 - be1) * p.grad / mian
                    d_zan[n] = -lr * wd * th0[n]
                    d_ped[n] = (th1[n] - th0[n]) - d_nowa[n] - d_zan[n]
            z0 = self.z(net, th0, b0)
            if t == 0:
                self.Z[0] = z0.half().cpu().numpy()
            sily = {}
            for kl, d in (('nowa', d_nowa), ('ped', d_ped), ('zan', d_zan)):
                sily[kl] = self.z(net, {n: th0[n] + d[n] for n in th0}, b0) - z0
            sily['buf'] = self.z(net, th0, b1_) - z0
            z1 = self.z(net, th1, b1_)
            for kl, s in sily.items():
                self.S[kl][t] = s.half().cpu().numpy()
            self.Z[t + 1] = z1.half().cpu().numpy()
            dz = z1 - z0
            reszta = float((dz - sum(sily.values())).norm() / dz.norm().clamp_min(1e-30))
            if self.SG is not None and t % self.grupy_co == 0:
                i = self.kroki_grup.index(t)
                for g in range(len(GRUPY)):
                    pg = {n: (th0[n] + (th1[n] - th0[n]) if grupa(n) == g else th0[n]) for n in th0}
                    self.SG[i * len(GRUPY) + g] = (self.z(net, pg, b0) - z0).half().cpu().numpy()
        self.skal.append(dict(krok=t, lr=opt.param_groups[0]['lr'], reszta_liniowosci=reszta,
                              dz=float(dz.norm()), **{f'S_{k}': float(s.norm()) for k, s in sily.items()}))
        if t % 100 == 0:
            for a in [self.Z, *self.S.values()] + ([self.SG] if self.SG is not None else []):
                a.flush()
            print(f'  krok {t}: |dz| {float(dz.norm()):.3f}, reszta liniowości {reszta:.3f}, '
                  + ', '.join(f'|S_{k}| {float(s.norm()):.3f}' for k, s in sily.items()), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wyniki', required=True)
    ap.add_argument('--ziarno', type=int, default=47)
    ap.add_argument('--krokow', type=int, default=1000)
    ap.add_argument('--klatki', type=int, nargs='+', default=[2, 3, 6])
    ap.add_argument('--grupy-co', type=int, default=10)
    ap.add_argument('--proba', action='store_true')
    a = ap.parse_args(argv)
    if a.proba:
        a.krokow, a.grupy_co = 20, 5
    out = Path(a.wyniki)
    if (out / 'dane' / 'Z.npy').exists():
        raise FileExistsError(f'Nie nadpisuję: {out / "dane" / "Z.npy"}')
    (out / 'dane').mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    X12 = np.load(WEJ12 / 'obrazy.npy'); maski = dict(np.load(WEJ12 / 'maski.npz'))
    opis = json.loads((WEJ12 / 'images.json').read_text(encoding='utf-8'))
    for nazwa in ('obrazy.npy', 'maski.npz', 'images.json'):
        shutil.copy2(WEJ12 / nazwa, out / 'dane' / nazwa)
    baza = D.baza()
    print(f's{a.ziarno}: dane gotowe ({time.time() - t0:.0f} s): baza {len(baza)}, klatki {a.klatki}', flush=True)
    pomiar = PomiarSil(X12[a.klatki], out, a.krokow, a.grupy_co)
    meta = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(), seed=a.ziarno, krokow=a.krokow,
                klatki=a.klatki, klatki_opis=[opis[i] for i in a.klatki], grupy=GRUPY, grupy_co=a.grupy_co,
                kroki_grup=pomiar.kroki_grup, narzedzie=maski['tool'][a.klatki].tolist(),
                hyper=dict(paczka=T.PACZKA, lr=T.LR, wd=T.WD, rozgrzewka_lr=T.ROZGRZEWKA),
                pomiar='eval, fp32, bez TF32; latent bloki[-1].mlp.act; siły przez podstawienie wag (functional_call)',
                device=torch.cuda.get_device_name(0), host=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'),
                torch=torch.__version__, proba=a.proba)
    zj(out / 'dane' / 'metadata.json', meta)
    u = T.Uczen(a.ziarno); rng = np.random.default_rng(a.ziarno)
    t1 = time.time()
    u.kroki(baza, a.krokow, rng, obserwator=pomiar)
    assert u.krok == a.krokow
    for arr in [pomiar.Z, *pomiar.S.values()] + ([pomiar.SG] if pomiar.SG is not None else []):
        arr.flush()
    torch.save(dict(net={k: v.detach().cpu() for k, v in u.net.state_dict().items()}, krok=u.krok), out / 'dane' / 'koniec.pt')
    zj(out / 'dane' / 'SKALARY.json', pomiar.skal); zj(out / 'dane' / 'LOG.json', u.log)
    r = np.array([s['reszta_liniowosci'] for s in pomiar.skal])
    meta.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), sekundy=time.time() - t1,
                reszta_liniowosci_mediana=float(np.median(r)), reszta_liniowosci_p90=float(np.percentile(r, 90)))
    zj(out / 'dane' / 'metadata.json', meta)
    print(f'KONIEC: {a.krokow} kroków, {time.time() - t1:.0f} s, reszta liniowości mediana {np.median(r):.3f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
