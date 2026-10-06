#!/usr/bin/env python3
"""Uczenie jednego ramienia małego ViT o zadanej wielkości latentu (plan: ../PLAN.md).

Ramię = wielkość modelu z model.RAMIONA (femto, pico, nano, tiny, small, tiny_tl8, nano_tl8). Przepis wspólny dla
wszystkich ramion i taki jak w biegach małego ViT (trening.Uczen): od zera, paczka 32, AdamW 5e-4, wd 0,05,
rozgrzewka lr 300 kroków, potem stały lr, bf16, odbicie poziome i jasność. Zmiana względem MODEL_VIT_K1000:
zbiór 5000 klatek (baza 1500 + 3500 z puli równo po 118 filmach, dane.zbior_n) i domyślnie 5000 kroków.

Zapis w --wyniki:
  MODEL_<ramie>.pt           wagi końcowe + konfig (do aplikacji: MalyViT(**konfig))
  wagi/k{krok:05d}.pt        same wagi co --zapis-wag kroków (z krokiem 0)
  pelny/k{krok:05d}.pt       wagi + stan AdamW na końcu
  dane/LATENT_KONTROLNY.npz  latent 12 klatek pomiaru na każdym checkpoincie: (K, 12, tokeny, mlp*dim) float16,
                             przy TokenLearnerze też mapy uwagi (K, 12, 600, S) float16
  dane/OCENY.json            mAP@0,5 na 3000 klatkach walidacji co --co-ocena kroków
  dane/LOG.json, dane/ZBIOR.json (frame_id z puli), dane/metadata.json

    python -u bieg.py --ramie tiny --wyniki <katalog> [--krokow 5000] [--proba]
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import dane as D
import model as M
import trening as T

HERE = Path(__file__).resolve().parent
WEJ12 = Path(os.environ.get('CZASTKI_WEJSCIA', HERE.parent / 'wejscia'))


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float) + '\n', encoding='utf-8')


class Checkpointy:
    """Obserwator trening.Uczen.kroki: co `co` kroków zapis wag i latentu 12 klatek pomiaru (bez wpływu na uczenie:
    no_grad, eval, nie zużywa generatorów losowych)."""

    def __init__(self, X12, out, co):
        self.X = torch.from_numpy(X12).permute(0, 3, 1, 2).contiguous().float().cuda() / 255.
        self.out, self.co = out, co
        self.kroki, self.latent, self.mapy = [], [], []

    def czy(self, j):
        return False                                         # przed(): niepotrzebne; zapis z po() przez zmierz()

    @torch.no_grad()
    def zmierz(self, u):
        u.net.eval(); u.net.zapisuj_latent(True)
        try:
            _, z = u.net(self.X)
            self.latent.append(z.float().cpu().numpy().astype(np.float16))
            if u.net.tokeny:
                self.mapy.append(u.net._mapy.float().cpu().numpy().astype(np.float16))
        finally:
            u.net.zapisuj_latent(False); u.net.train()
        self.kroki.append(u.krok)
        (self.out / 'wagi').mkdir(parents=True, exist_ok=True)
        torch.save(dict(net={k: v.detach().cpu().clone() for k, v in u.net.state_dict().items()}, krok=u.krok,
                        konfig=u.konfig), self.out / 'wagi' / f'k{u.krok:05d}.pt')

    def zapisz(self):
        d = dict(kroki=np.array(self.kroki), latent=np.stack(self.latent))
        if self.mapy:
            d['mapy'] = np.stack(self.mapy)
        np.savez_compressed(self.out / 'dane' / 'LATENT_KONTROLNY.npz', **d)
        return {k: list(v.shape) for k, v in d.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ramie', required=True, choices=sorted(M.RAMIONA))
    ap.add_argument('--wyniki', required=True)
    ap.add_argument('--ziarno', type=int, default=47)
    ap.add_argument('--krokow', type=int, default=5000)
    ap.add_argument('--n-zdjec', type=int, default=5000)
    ap.add_argument('--zapis-wag', type=int, default=500)
    ap.add_argument('--co-ocena', type=int, default=500)
    ap.add_argument('--proba', action='store_true', help='próba kodu: 20 kroków, zapis i ocena co 10')
    a = ap.parse_args(argv)
    if a.proba:
        a.krokow, a.zapis_wag, a.co_ocena = 20, 10, 10
    out = Path(a.wyniki)
    if (out / f'MODEL_{a.ramie}.pt').exists():
        raise FileExistsError(f'Nie nadpisuję istniejącego ramienia: {out}')
    (out / 'dane').mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    konfig = M.RAMIONA[a.ramie]

    t0 = time.time()
    X12 = np.load(WEJ12 / 'obrazy.npy')
    assert X12.shape == (12, 320, 480, 3) and X12.dtype == np.uint8
    zb, z_puli = D.zbior_n(a.n_zdjec)
    wal_w, wal_X = D.walidacja()
    zj(out / 'dane' / 'ZBIOR.json', dict(n=len(zb), baza=1500, z_puli=len(z_puli), frame_id_z_puli=z_puli,
                                         filmow_z_puli=len({f.split('/')[2] for f in z_puli})))
    print(f'{a.ramie}: dane gotowe ({time.time() - t0:.0f} s): uczenie {len(zb)} klatek (z puli {len(z_puli)}), '
          f'val {len(wal_w)}', flush=True)

    u = T.Uczen(a.ziarno, konfig); rng = np.random.default_rng(a.ziarno)
    n_param = sum(p.numel() for p in u.net.parameters())
    meta = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(), ramie=a.ramie, konfig=konfig,
                ksztalt_latentu=list(M.ksztalt_latentu(konfig)), parametry=n_param, seed=a.ziarno, krokow=a.krokow,
                n_zdjec=len(zb), zapis_wag=a.zapis_wag, co_ocena=a.co_ocena,
                hyper=dict(paczka=T.PACZKA, beta1=T.BETA1, lr=T.LR, wd=T.WD, rozgrzewka_lr=T.ROZGRZEWKA,
                           autocast_uczenia='bf16', augmentacja='odbicie poziome 0,5; jasność i kontrast 0,8-1,2'),
                latent_layer='bloki[-1].mlp.act (GELU)', klatki_pomiaru='CZASTKI_WEJSCIA/obrazy.npy (12 klatek)',
                code=[dict(path=p.name, sha256=sha256(p)) for p in sorted(HERE.glob('*.py'))],
                versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                device=torch.cuda.get_device_name(0), host=platform.node(),
                slurm_job=os.environ.get('SLURM_JOB_ID'), proba=a.proba)
    zj(out / 'dane' / 'metadata.json', meta)
    print(f'{a.ramie}: {konfig}, parametry {n_param / 1e6:.2f} M, latent {M.ksztalt_latentu(konfig)}', flush=True)

    oceny = []

    def ocena(uu):
        ap_ = T.ap50(uu.wykryj(wal_X), [r['ramki'] for r in wal_w])
        oceny.append(dict(krok=uu.krok, mAP50=ap_))
        print(f'  krok {uu.krok}: mAP50 {ap_:.4f}', flush=True)

    ck = Checkpointy(X12, out, a.zapis_wag)

    class Obs:                                               # zapis checkpointu po krokach podzielnych przez zapis_wag
        czy = staticmethod(lambda j: (u.krok + 1) % a.zapis_wag == 0)
        przed = staticmethod(lambda *x: None)
        po = staticmethod(lambda uu: ck.zmierz(uu))

    ocena(u); ck.zmierz(u)
    t1 = time.time()
    u.kroki(zb, a.krokow, rng, co_ocena=a.co_ocena, ocena=ocena, obserwator=Obs)
    czas = time.time() - t1
    assert u.krok == a.krokow
    if not oceny or oceny[-1]['krok'] != u.krok:
        ocena(u)
    if ck.kroki[-1] != u.krok:
        ck.zmierz(u)
    (out / 'pelny').mkdir(parents=True, exist_ok=True)
    torch.save(dict(**u.stan(), konfig=konfig), out / 'pelny' / f'k{u.krok:05d}.pt')
    torch.save(dict(net={k: v.detach().cpu() for k, v in u.net.state_dict().items()}, krok=u.krok, konfig=konfig,
                    ramie=a.ramie, mAP50=oceny[-1]['mAP50']), out / f'MODEL_{a.ramie}.pt')
    ksz = ck.zapisz()
    zj(out / 'dane' / 'OCENY.json', oceny); zj(out / 'dane' / 'LOG.json', u.log)
    meta.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), training_seconds=czas,
                total_seconds=time.time() - t0, s_na_krok=czas / a.krokow, mAP50_koniec=oceny[-1]['mAP50'],
                mAP50_najlepszy=max(o['mAP50'] for o in oceny), latent_kontrolny=ksz,
                wagi=sorted(p.name for p in (out / 'wagi').glob('*.pt')))
    zj(out / 'dane' / 'metadata.json', meta)
    print(f'KONIEC {a.ramie}: mAP50 {oceny[-1]["mAP50"]:.4f}, uczenie {czas:.0f} s ({czas / a.krokow * 1000:.0f} ms/krok)',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
