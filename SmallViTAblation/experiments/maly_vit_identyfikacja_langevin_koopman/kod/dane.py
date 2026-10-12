"""Dostarcza stan i wykonane paczki; obala poprawność przesunięcie wejścia o krok.

Wiersz paczki k powoduje przejście s[k] -> s[k+1]. U[t] zawsze opisuje
ukończoną paczkę t-1, a U[0] pozostaje NaN. Sonda != obraz paczki.
Duże oryginalne tablice tylko memmap, bez modyfikacji i bez kopiowania.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parents[1]
EXPERIMENTS = HERE.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(seed):
    root = EXPERIMENTS / 'maly_vit_rownanie_latentu' / f'nagranie_s{seed}_sily'
    cache = EXPERIMENTS / 'maly_vit_prognoza_tozsamosci_z_historii' / 'wyniki' / f'seria_s{seed}.npz'
    meta = json.loads((root / 'META.json').read_text(encoding='utf-8'))
    z = np.load(cache)
    S = z['s'].astype(float)
    # Recompute the original frozen calibration, available at step 1000.
    L = np.load(root / 'sondy_lat.npy', mmap_mode='r')
    ob = np.asarray(meta['obecn'], bool)
    cal = L[960:1001:10].astype(float).mean(0)
    X = cal.reshape(-1, cal.shape[-1])
    y = np.column_stack([ob, np.zeros(len(ob), bool)]).reshape(-1)
    mu, sd = X[~y].mean(0), X[~y].std(0) + 1e-6
    w = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit((X-mu)/sd, y).coef_[0]
    w /= np.linalg.norm(w)
    exact = (L[1000, ob, 0].astype(np.float32)-mu.astype(np.float32)) @ (w/sd).astype(np.float32)
    assert np.max(np.abs(exact-S[1000])) < 1e-4
    # P and N are indexed by the state AFTER completion, unlike source files.
    P, N = np.full_like(S, np.nan), np.full_like(S, np.nan)
    P[1:] = z['ped'][:-1]
    N[1:] = z['nowa'][:-1]
    descfile = HERE / 'wyniki' / f'DESKRYPTORY_s{seed}.npz'
    if descfile.exists():
        q = np.load(descfile)['q']
    else:
        lat = np.load(root / 'paczka_lat.npy', mmap_mode='r')
        mask = np.load(root / 'paczka_ma_narz.npy', mmap_mode='r')
        q = np.full((len(S), 10), np.nan)
        aug = np.load(root / 'paczka_a.npy', mmap_mode='r')
        flip = np.load(root / 'paczka_odb.npy', mmap_mode='r')
        loss = np.load(root / 'strata.npy', mmap_mode='r')
        norm = np.load(root / 'norma_grad.npy', mmap_mode='r')
        lr = np.load(root / 'lr.npy', mmap_mode='r')
        for a in range(0, len(lat), 64):
            b = min(a+64, len(lat))
            projected = (lat[a:b].astype(np.float32)-mu.astype(np.float32)) @ (w/sd).astype(np.float32)
            present = mask[a:b].astype(float)
            count = present.sum(1)
            toolmean = (projected[:,:,0]*present).sum(1)/np.maximum(count,1)
            toolvar = (((projected[:,:,0]-toolmean[:,None])**2)*present).sum(1)/np.maximum(count,1)
            q[a+1:b+1] = np.column_stack([toolmean, np.sqrt(toolvar), projected[:,:,1].mean(1),
                present.mean(1), aug[a:b,:,0].mean(1), aug[a:b,:,1].mean(1),
                flip[a:b].mean(1), loss[a:b], norm[a:b], lr[a:b]])
        np.savez_compressed(descfile, q=q)
    # Historical buffers diagnostic is available every 10 completed steps.
    rk = np.load(root / 'rozklad_kroki.npy')
    rb = np.load(root / 'rozklad_bufory.npy', mmap_mode='r')
    buffer_s = rb[:,ob].astype(np.float32) @ (w/sd).astype(np.float32)
    audit = dict(seed=seed, states=list(S.shape), calibration_steps=[960,970,980,990,1000],
        force_first_completed_step=meta['sily_od']+1, future_input='forbidden',
        cache=str(cache), cache_sha256=sha256(cache), meta_sha256=sha256(root/'META.json'),
        descriptors=['tool_mean','tool_sd','full_mean','tool_fraction','contrast_mean','brightness_mean',
                     'flip_fraction','loss','gradient_norm','lr'],
        history_rule='q[t],N[t],P[t] come from completed source row t-1',
        original_files={p.name:dict(bytes=p.stat().st_size, shape=list(np.load(p,mmap_mode='r').shape))
                        for p in root.glob('*.npy')},
        buffer_rms=float(np.sqrt(np.mean(buffer_s**2))), buffer_measured_steps=int(len(rk)))
    return dict(S=S,N=N,P=P,q=q,audit=audit)
