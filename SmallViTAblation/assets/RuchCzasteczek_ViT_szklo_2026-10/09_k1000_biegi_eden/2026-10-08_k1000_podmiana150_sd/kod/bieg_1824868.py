#!/usr/bin/env python3
"""Kontynuacja uczenia małego ViT od K1000 (pełny stan z AdamW z gęstej historii) w ramionach (PLAN.md):
  C0    oryginalna baza 1500 klatek,
  A150  ta sama baza, 150 pozycji (PODMIANA.json) zastąpione klatkami TRAIN znalezionymi przez DINO dla obrazu SD,
  C0b   powtórzenie C0 (poziom szumu obliczeń; w trybie deterministycznym powinno dać różnicę 0).
Przepis bez zmian: trening.Uczen (AdamW 5e-4, wd 0,05/0, betas 0,9/0,999, rozgrzewka 300 kroków już minęła, clip 1,
bf16, odbicie i jasność/kontrast), paczka 32, kolejka permutacji jak w pętli. Jeden wywołany ciąg kroków na ramię
(kolejka paczek ciągła); RNG kolejki i augmentacji: default_rng(47) od nowa (stan RNG z kroku 1000 nie był zapisany),
ten sam dla wszystkich ramion, więc ramiona dostają te same indeksy pozycji i te same augmentacje.
Epoka = 47 aktualizacji (1504 ekspozycje, jak w pętli). Pomiar przed uczeniem i po każdej epoce (K1000, K1047, ...).
Obliczenia deterministyczne (SDPA MATH, use_deterministic_algorithms, cudnn.deterministic), jak galezie_uczenia.

Pomiar (eval, no_grad, fp32 bez TF32 dla latentu; detekcja jak w trening.wykryj):
  - zdjęcie celu 0055/4019: pełny latent 600 x 768 (bloki[-1].mlp.act), pewność i IoU najpewniejszej ramki;
  - grupy nowe150 / usuniete150 / sondy64: latent średni po tokenach i po tokenach ramki narzędzia (2 x 768),
    pewność i IoU najpewniejszej ramki;
  - mAP@0,5 na próbce 3000 klatek walidacji (D.walidacja, jak w oryginale).
    python -u bieg.py --wyniki <katalog> --ramie C0|A150|C0b [--epok 5] [--proba]
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
WEJ = Path(os.environ['PODMIANA_WEJSCIA'])
PELNY = Path(os.environ['K1000_PELNY'])
EPOKA = 47
ZIARNO = 47


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float) + '\n', encoding='utf-8')


def deterministycznie():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    k = sdpa_kernel(SDPBackend.MATH); k.__enter__()
    return k


def maska_ramki(ramki):
    """Tokeny siatki 20 x 30, których środek leży w ramce, plus token środka ramki (jak korekta_latentu)."""
    ys, xs = np.mgrid[0:20, 0:30]; cx, cy = (xs + .5) * 16, (ys + .5) * 16
    ob = np.zeros((20, 30), bool)
    for x0, y0, x1, y1 in np.asarray(ramki).reshape(-1, 4):
        ob |= (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
        ob[min(max(int((y0 + y1) / 32), 0), 19), min(max(int((x0 + x1) / 32), 0), 29)] = True
    return ob.reshape(-1)


def wczytaj_stan(asset):
    st = torch.load(PELNY, map_location='cpu', weights_only=False)
    a = torch.load(asset, map_location='cpu', weights_only=True)['net']
    assert st['krok'] == 1000 and 'opt' in st, 'brak pełnego stanu K1000'
    assert set(st['net']) == set(a), 'inne klucze wag niż asset K1000'
    rozn = max(float((st['net'][k].float() - a[k].float()).abs().max()) for k in a)
    assert rozn == 0.0, f'wagi pełnego K1000 różnią się od assetu: {rozn}'
    return st


class Pomiar:
    def __init__(self, cel, grupy, wal_w, wal_X, out):
        self.cel, self.grupy, self.wal_w, self.wal_X, self.out = cel, grupy, wal_w, wal_X, out
        self.kroki, self.z_cel, self.g_lat, self.metryki = [], [], {k: [] for k in grupy}, []

    def czy(self, j):
        return (j + 1) % EPOKA == 0

    def przed(self, u, *a):
        pass

    def po(self, u):
        self.zmierz(u)

    @torch.no_grad()
    def latent(self, u, X):
        m32, c32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        u.net.eval(); u.net.zapisuj_latent(True)
        out = []
        try:
            for s in range(0, len(X), 32):
                x = torch.from_numpy(X[s:s + 32]).to(u.dev).permute(0, 3, 1, 2).float() / 255.
                _, z = u.net(x); out.append(z.float().cpu())
        finally:
            u.net.zapisuj_latent(False)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m32, c32
        return torch.cat(out).numpy()

    def detekcja(self, u, X, ramki):
        wy = u.wykryj(X)
        conf, iou = [], []
        for (bx, sc), r in zip(wy, ramki):
            k = int(np.argmax(sc)); conf.append(float(sc[k]))
            iou.append(float(max(T.iou(bx[k], np.asarray(g, np.float32)) for g in r)) if len(r) else float('nan'))
        return np.array(conf), np.array(iou)

    def zmierz(self, u):
        t0 = time.time()
        zc = self.latent(u, self.cel['X'])[0]
        assert np.isfinite(zc).all()
        c, i = self.detekcja(u, self.cel['X'], [self.cel['ramki']])
        rek = dict(krok=u.krok, cel_pewnosc=float(c[0]), cel_iou=float(i[0]))
        for k, g in self.grupy.items():
            z = self.latent(u, g['X'])
            mt = g['maski'][..., None]
            self.g_lat[k].append(np.stack([z.mean(1), (z * mt).sum(1) / mt.sum(1).clip(min=1)], 1).astype(np.float32))
            c, i = self.detekcja(u, g['X'], g['ramki'])
            rek[f'{k}_pewnosc'] = float(c.mean()); rek[f'{k}_iou'] = float(np.nanmean(i))
            rek[f'{k}_iou_ponad_05'] = float(np.nanmean(i >= .5))
            g.setdefault('szczegoly', []).append(dict(krok=u.krok, pewnosc=c.tolist(), iou=i.tolist()))
        rek['mAP50_wal3000'] = T.ap50(u.wykryj(self.wal_X), [r['ramki'] for r in self.wal_w])
        u.net.train()
        rek['czas_pomiaru_s'] = time.time() - t0
        self.kroki.append(u.krok); self.z_cel.append(zc); self.metryki.append(rek)
        (self.out / 'pelny').mkdir(parents=True, exist_ok=True)
        torch.save(u.stan(), self.out / 'pelny' / f'k{u.krok:04d}.pt')
        print(f'  pomiar krok {u.krok}: ' + ', '.join(f'{a} {b:.4f}' for a, b in rek.items() if a != 'krok'), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wyniki', required=True)
    ap.add_argument('--ramie', required=True, choices=['C0', 'A150', 'C0b'])
    ap.add_argument('--epok', type=int, default=5)
    ap.add_argument('--proba', action='store_true', help='1 epoka, sprawdzenie kodu')
    a = ap.parse_args(argv)
    if a.proba:
        a.epok = 1
    out = Path(a.wyniki) / a.ramie
    if (out / 'POMIARY.npz').exists():
        raise FileExistsError(f'nie nadpisuję {out / "POMIARY.npz"}')
    out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    kern = deterministycznie()
    t0 = time.time()

    stan = wczytaj_stan(WEJ / 'MODEL_VIT_K1000.pt')
    baza = D.baza(); wal_w, wal_X = D.walidacja()
    assert len(baza) == 1500
    nowe = json.loads((WEJ / 'NOWE.json').read_text(encoding='utf-8'))
    cel = json.loads((WEJ / 'CEL.json').read_text(encoding='utf-8'))
    pod = json.loads((WEJ / 'PODMIANA.json').read_text(encoding='utf-8'))
    usun = [x['pozycja'] for x in pod['usuniete']]; sond = [x['pozycja'] for x in pod['sondy']]
    for x in pod['usuniete']:
        assert baza.wiersze[x['pozycja']]['frame_id'] == x['frame_id'], 'kolejność bazy inna niż w manifeście'
    X_nowe = np.stack([D.wczytaj(WEJ / n['plik']) for n in nowe])
    r_nowe = [np.asarray(n['ramki_480'], np.float32).reshape(-1, 4) for n in nowe]

    if a.ramie == 'A150':
        wiersze = list(baza.wiersze); X = baza.X.copy()
        for k, n in enumerate(nowe):
            p = n['pozycja_w_bazie']; assert p == usun[k]
            wiersze[p] = dict(frame_id=n['frame_id'], sciezka=str(WEJ / n['plik']), ramki=r_nowe[k])
            X[p] = X_nowe[k]
        zb = D.Zbior(wiersze, X)
    else:
        zb = baza
    assert len(zb) == 1500
    grupy = dict(nowe150=dict(X=X_nowe, ramki=r_nowe),
                 usuniete150=dict(X=baza.X[usun], ramki=[baza.wiersze[i]['ramki'] for i in usun]),
                 sondy64=dict(X=baza.X[sond], ramki=[baza.wiersze[i]['ramki'] for i in sond]))
    for g in grupy.values():
        g['maski'] = np.stack([maska_ramki(r) for r in g['ramki']])
    Xc = D.wczytaj(WEJ / cel['plik'])[None]
    pomiar = Pomiar(dict(X=Xc, ramki=np.asarray(cel['ramki_480'], np.float32).reshape(-1, 4)), grupy, wal_w, wal_X, out)
    print(f'{a.ramie}: dane gotowe ({time.time() - t0:.0f} s), zbiór {len(zb)}, podmienionych '
          f'{sum(zb.wiersze[i]["frame_id"] != baza.wiersze[i]["frame_id"] for i in range(1500))}', flush=True)

    u = T.Uczen(ZIARNO); u.wczytaj(stan)
    assert u.krok == 1000 and abs(u._lr() - T.LR) < 1e-12
    rng = np.random.default_rng(ZIARNO)
    pomiar.zmierz(u)
    t1 = time.time()
    u.kroki(zb, EPOKA * a.epok, rng, obserwator=pomiar)
    czas = time.time() - t1
    assert u.krok == 1000 + EPOKA * a.epok and len(pomiar.kroki) == a.epok + 1
    kern.__exit__(None, None, None)

    np.savez_compressed(out / 'POMIARY.npz', kroki=np.array(pomiar.kroki), z_cel=np.stack(pomiar.z_cel),
                        **{f'lat_{k}': np.stack(v) for k, v in pomiar.g_lat.items()})
    zj(out / 'METRYKI.json', pomiar.metryki)
    zj(out / 'SZCZEGOLY_GRUP.json', {k: g['szczegoly'] for k, g in grupy.items()})
    zj(out / 'LOG.json', u.log)
    zj(out / 'metadata.json', dict(
        ramie=a.ramie, status='complete', utc=datetime.now(timezone.utc).isoformat(), ziarno_rng=ZIARNO, epoka_krokow=EPOKA,
        epok=a.epok, kroki=pomiar.kroki, start=dict(plik=str(PELNY), sha256=sha256(PELNY), krok=1000),
        hyper=dict(paczka=T.PACZKA, lr=T.LR, wd=T.WD, rozgrzewka_lr=T.ROZGRZEWKA, autocast_uczenia='bf16'),
        deterministyczne=True, uczenie_s=czas, razem_s=time.time() - t0,
        wejscia={p.name: sha256(p) for p in sorted(WEJ.glob('*.json'))},
        kod=[dict(path=p.name, sha256=sha256(p)) for p in sorted(HERE.glob('*.py'))],
        wersje=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        gpu=torch.cuda.get_device_name(0), host=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'), proba=a.proba))
    print(f'KONIEC {a.ramie}: uczenie {czas:.0f} s, razem {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
