#!/usr/bin/env python3
"""Gęsta historia surowych tokenów małego ViT w uczeniu od zera (plan: ../../maly_vit_ruch_czastek_maska_1px/PLAN.md).

Przepis uczenia bez zmian względem wspólnej rozgrzewki pętli małego ViT (petla4.py): trening.Uczen(47),
rng = default_rng(47), baza 1500 klatek, paczka 32, AdamW 5e-4, rozgrzewka lr 300 kroków, bf16, odbicie i jasność,
ocena mAP@0,5 próbki val co 100 kroków, średnia telemetria 192 sond co 10 kroków.

Nowe: przed pierwszą i po KAŻDEJ aktualizacji 12 pełnych klatek 480×320 przechodzi przez sieć w trybie eval,
fp32 bez autocast i bez TF32; zapis wszystkich 600 tokenów × 768 neuronów (MLP ostatniego bloku po GELU), bez
uśredniania. Pomiar nie zmienia uczenia: no_grad, eval (BatchNorm nie aktualizuje statystyk), nie zużywa
generatorów losowych.

Zapis w --wyniki:
  dane/LATENT_GESTY.npy   (T, 12, 600, 768) float32, wiersz t = stan po t aktualizacjach, t = 0..N
  wagi/k{krok:04d}.pt     same wagi sieci (z buforami BatchNorm) co --zapis-wag kroków
  pelny/k{krok:04d}.pt    wagi + stan AdamW co --zapis-pelny kroków
  dane/LATENT_SOND.npz    średnia telemetria sond co 10 kroków (format LATENT.npz pętli)
  dane/OCENY.json, dane/LOG.json, dane/metadata.json

    python -u bieg.py --wyniki <katalog> [--krokow 1000] [--proba]
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import hashlib
import json
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


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float) + '\n', encoding='utf-8')


class Pomiar:
    """Obserwator kroków trening.Uczen.kroki: po każdej aktualizacji zapisuje surowe tokeny 12 klatek."""

    def __init__(self, X12, latent, out, zapis_wag, zapis_pelny):
        self.X = torch.from_numpy(X12).permute(0, 3, 1, 2).contiguous().float().cuda() / 255.
        self.latent, self.out = latent, out
        self.zapis_wag, self.zapis_pelny = zapis_wag, zapis_pelny
        self.ruch = []            # względna zmiana tokenów między kolejnymi krokami (tylko do logu i kontroli)
        self.poprzedni = None
        self.kontrola = None

    def czy(self, j):
        return True

    def przed(self, u, *a):
        pass

    def po(self, u):
        self.zmierz(u)

    @torch.no_grad()
    def zmierz(self, u):
        t = u.krok
        m32, c32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        u.net.eval(); u.net.zapisuj_latent(True)
        try:
            _, z = u.net(self.X)
            arr = z.float().cpu().numpy()
            if t == 0:
                _, z2 = u.net(self.X)
                self.kontrola = float((z2.float().cpu() - z.float().cpu()).abs().max())
        finally:
            u.net.zapisuj_latent(False)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m32, c32
            u.net.train()
        assert arr.dtype == np.float32 and arr.shape == self.latent.shape[1:], arr.shape
        assert np.isfinite(arr).all(), f'nieskończone tokeny po kroku {t}'
        self.latent[t] = arr
        if self.poprzedni is not None:
            d = float(np.linalg.norm(arr - self.poprzedni) / max(np.linalg.norm(self.poprzedni), 1e-30))
            assert d > 0, f'identyczne tokeny po krokach {t - 1} i {t}'
            self.ruch.append((t, d))
        self.poprzedni = arr
        if t % self.zapis_wag == 0:
            (self.out / 'wagi').mkdir(parents=True, exist_ok=True)
            torch.save(dict(net={k: v.detach().cpu().clone() for k, v in u.net.state_dict().items()}, krok=t),
                       self.out / 'wagi' / f'k{t:04d}.pt')
        if t % self.zapis_pelny == 0:
            (self.out / 'pelny').mkdir(parents=True, exist_ok=True)
            torch.save(u.stan(), self.out / 'pelny' / f'k{t:04d}.pt')
        if t % 100 == 0:
            self.latent.flush()
            ost = self.ruch[-1][1] if self.ruch else float('nan')
            print(f'  krok {t}: zapisano tokeny, względna zmiana ostatniego kroku {ost:.2e}', flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wyniki', required=True)
    ap.add_argument('--ziarno', type=int, default=47)
    ap.add_argument('--krokow', type=int, default=1000)
    ap.add_argument('--zapis-wag', type=int, default=10)
    ap.add_argument('--zapis-pelny', type=int, default=100)
    ap.add_argument('--co-ocena', type=int, default=100)
    ap.add_argument('--proba', action='store_true', help='próba kodu: 20 kroków, częstsze zapisy')
    a = ap.parse_args(argv)
    if a.proba:
        a.krokow, a.zapis_wag, a.zapis_pelny, a.co_ocena = 20, 5, 10, 10
    out = Path(a.wyniki)
    dest = out / 'dane' / 'LATENT_GESTY.npy'
    if dest.exists():
        raise FileExistsError(f'Nie nadpisuję istniejącego pomiaru: {dest}')
    (out / 'dane').mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False

    t0 = time.time()
    X12 = np.load(WEJ12 / 'obrazy.npy')
    maski = dict(np.load(WEJ12 / 'maski.npz'))
    obrazy_opis = json.loads((WEJ12 / 'images.json').read_text(encoding='utf-8'))
    assert X12.shape == (12, 320, 480, 3) and X12.dtype == np.uint8 and len(obrazy_opis) == 12
    assert maski['tool'].shape == (12, 600)
    for nazwa in ('obrazy.npy', 'maski.npz', 'images.json'):
        shutil.copy2(WEJ12 / nazwa, out / 'dane' / nazwa)
    baza = D.baza(); wal_w, wal_X = D.walidacja(); sondy = D.sondy()
    obecn = sondy[4]
    print(f's{a.ziarno}: dane gotowe ({time.time() - t0:.0f} s): baza {len(baza)}, val {len(wal_w)}, '
          f'sondy {len(sondy[0])}, klatki pomiaru {len(X12)}', flush=True)

    T_ = a.krokow + 1
    latent = np.lib.format.open_memmap(dest, mode='w+', dtype=np.float32, shape=(T_, 12, 600, 768))
    meta = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(), seed=a.ziarno,
                kroki=list(range(T_)), shape=[T_, 12, 600, 768], dtype='float32', images=obrazy_opis,
                training_recipe='trening.Uczen(ziarno) + numpy default_rng(ziarno), baza 1500 klatek; '
                                'jak wspólna rozgrzewka pętli małego ViT (petla4.py), bez dodawania klatek',
                hyper=dict(paczka=T.PACZKA, lr=T.LR, wd=T.WD, rozgrzewka_lr=T.ROZGRZEWKA, autocast_uczenia='bf16',
                           augmentacja='odbicie poziome 0,5; jasność i kontrast 0,8-1,2'),
                measurement=dict(mode='eval', no_grad=True, autocast=False, tf32=False, dtype='float32',
                                 latent_layer='bloki[-1].mlp.act (GELU)', token_order='row-major: token=y*30+x',
                                 grid_hw=[20, 30], input_size_wh=[480, 320], co_krokow=1,
                                 uwaga_batchnorm='W trybie eval łodyga używa statystyk biegnących BatchNorm; zmieniają '
                                                 'się w każdym kroku uczenia niezależnie od lr, więc ruch tokenów '
                                                 'zawiera także zmianę tych statystyk.'),
                zapis_wag=a.zapis_wag, zapis_pelny=a.zapis_pelny, co_ocena=a.co_ocena,
                code=[dict(path=p.name, sha256=sha256(p)) for p in sorted(HERE.glob('*.py'))],
                versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                device=torch.cuda.get_device_name(0), host=platform.node(),
                slurm_job=os.environ.get('SLURM_JOB_ID'), proba=a.proba)
    zj(out / 'dane' / 'metadata.json', meta)

    u = T.Uczen(a.ziarno); rng = np.random.default_rng(a.ziarno)
    oceny = []

    def ocena(uu):
        ap_ = T.ap50(uu.wykryj(wal_X), [r['ramki'] for r in wal_w])
        q = uu.jakosc_sond(sondy)
        oceny.append(dict(krok=uu.krok, mAP50=ap_, q_sond=float(np.nanmean(q[obecn])),
                          q_puste=float(np.nanmean(q[~obecn]))))
        print(f'  krok {uu.krok}: mAP50 {ap_:.4f}', flush=True)

    pomiar = Pomiar(X12, latent, out, a.zapis_wag, a.zapis_pelny)
    ocena(u)
    lat = [(0, u.latent_sond(sondy))]
    pomiar.zmierz(u)
    t1 = time.time()
    u.kroki(baza, a.krokow, rng, sondy=sondy, latent_out=lat, co_ocena=a.co_ocena, ocena=ocena, obserwator=pomiar)
    czas = time.time() - t1
    assert u.krok == a.krokow
    if a.krokow % a.zapis_pelny:                         # stan końcowy zawsze z wagami i optymalizatorem
        torch.save(u.stan(), out / 'pelny' / f'k{u.krok:04d}.pt')
    latent.flush(); del latent

    np.savez_compressed(out / 'dane' / 'LATENT_SOND.npz', kroki=np.array([x for x, _ in lat]),
                        latent=np.stack([l for _, l in lat]))
    zj(out / 'dane' / 'OCENY.json', oceny); zj(out / 'dane' / 'LOG.json', u.log)
    zj(out / 'dane' / 'RUCH_KROKOW.json', [dict(krok=t, wzgledna_zmiana=d) for t, d in pomiar.ruch])

    Z = np.load(dest, mmap_mode='r')
    assert Z.shape == (T_, 12, 600, 768)
    for t in range(T_):
        assert np.isfinite(Z[t]).all() and np.abs(Z[t]).sum() > 0, f'pusty lub zły wiersz {t}'
    meta.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                training_seconds=czas, total_seconds=time.time() - t0,
                repeated_forward_max_abs_error_step0=pomiar.kontrola,
                latent_bytes=dest.stat().st_size, latent_sha256=sha256(dest),
                wagi=sorted(p.name for p in (out / 'wagi').glob('*.pt')),
                pelny=sorted(p.name for p in (out / 'pelny').glob('*.pt')),
                mAP50_koniec=oceny[-1]['mAP50'] if oceny else None)
    zj(out / 'dane' / 'metadata.json', meta)
    print(f'KONIEC: {dest} {Z.shape}, {dest.stat().st_size / 1e9:.2f} GB, uczenie {czas:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
