"""Trajektorie tokenów po wszystkich checkpointach biegu 35 000 (806 wag, 60 epok) do analizy ruchu w skali epok (DMD, VAMP, MSD po
epokach, centrum drgań). Dla n_rp zdjęć TRAIN + n_rp VALID: losowy rzut 768 → 128 (Johnson–Lindenstrauss, `torch.randn` z generatorem
o ziarnie 0 podzielone przez √128, dokładnie jak obserwator_rp fali 3 pokręteł, więc współrzędne są te same co w ANALIZA_*_rp),
średnia tokenów zdjęcia w 768D (mod zdjęcia), normy |z| i |z − z(krok 0)| w 768D, pewność i IoU mapy. Dla n_768 + n_768 pierwszych
zdjęć dodatkowo pełny latent 768D. Zdjęcia: ten sam ciąg losowy co pomiar_eden.py (ziarno 2026: 3000 TRAIN z manifestu, 1000 VALID),
z niego pierwsze n_rp z każdej części, czyli te same zdjęcia co początek podzbioru PCA32 pomiaru pelny60.
Zapis: rp128.npy (K, N, 600, 128) f16, pelne768.npy (K, N768, 600, 768) f16, srednia768.npy (K, N, 768) f32, normy768.npy (K, N, 600, 2) f16,
oceny.npy (K, N, 2) f32 [pewność, IoU], R_rp128.npy, META.json. Uruchomienie (katalog kodu biegu 35 000, zmienne MALY_VIT_* jak w treningu):
  python ruch60_rp.py --wagi .../pelny/wagi --manifest .../MANIFEST_35000.jsonl --zipy ... --out ... [--n_rp 50 --n_768 2 --max_krok 65640]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dane as D  # noqa: E402
import model as M  # noqa: E402
import trening as T  # noqa: E402
from trening35000 import wczytaj_z_zipow, wiersze_35000  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--wagi', required=True)
ap.add_argument('--manifest', required=True)
ap.add_argument('--zipy', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n_train', type=int, default=3000, help='jak w pomiar_eden.py (ten sam ciąg losowy)')
ap.add_argument('--n_val', type=int, default=1000)
ap.add_argument('--ziarno', type=int, default=2026)
ap.add_argument('--n_rp', type=int, default=50, help='zdjęć z rzutem 128D na część (TRAIN i VALID)')
ap.add_argument('--n_768', type=int, default=2, help='zdjęć z pełnym 768D na część')
ap.add_argument('--rzut_dim', type=int, default=128)
ap.add_argument('--ziarno_rzutu', type=int, default=0)
ap.add_argument('--paczka', type=int, default=50)
ap.add_argument('--wszystkie', action='store_true')
ap.add_argument('--max_krok', type=int, default=65640)
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
TOK = 600
dev = 'cuda'
t0 = time.time()


def log(*s):
    print(*s, f'[{time.time() - t0:.0f} s]', flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- checkpointy (ten sam wybór co pomiar_eden.py)
kroki_pliki = {}
for p in Path(a.wagi).iterdir():
    m = re.fullmatch(r'k([0-9]+)\.pt', p.name)
    if m:
        kroki_pliki[int(m[1])] = p
kmax = min(max(kroki_pliki), a.max_krok)
wybrane = sorted(k for k in kroki_pliki if k <= kmax and (a.wszystkie or (k <= 1000 and k % 10 == 0) or k % 100 == 0 or k % 1094 == 0 or k == kmax))
K = len(wybrane)
log(f'checkpointów na dysku {len(kroki_pliki)}, wybranych {K} (0..{kmax})')

# ---------------------------------------------------------------- zdjęcia: ten sam ciąg losowy co pomiar_eden.py, pierwsze n_rp z każdej części
rng = np.random.default_rng(a.ziarno)
wt = wiersze_35000(a.manifest)
idx_tr = sorted(rng.choice(len(wt), a.n_train, replace=False).tolist())
wal_w, wal_X = D.walidacja()
idx_val = sorted(rng.choice(len(wal_w), a.n_val, replace=False).tolist())
n = a.n_rp
rows_tr = [wt[i] for i in idx_tr[:n]]
rows_val = [wal_w[i] for i in idx_val[:n]]
X = np.concatenate([wczytaj_z_zipow(rows_tr, a.zipy), wal_X[idx_val[:n]]])
del wal_X
N = len(X)
pelne = np.concatenate([np.arange(a.n_768), n + np.arange(a.n_768)])
pelne_set = set(pelne.tolist())
ramki = [r['ramki'] for r in rows_tr] + [r['ramki'] for r in rows_val]
zdjecia = [dict(index=j, split='TRAIN' if j < n else 'VALID', frame_id=r['frame_id'], pool_row=int(idx_tr[j]) if j < n else None,
                val_row=None if j < n else int(idx_val[j - n]), n_gt=int(len(r['ramki'])), pelne768=bool(j in pelne_set))
           for j, r in enumerate(rows_tr + rows_val)]
log(f'zdjęcia: {n} TRAIN + {n} VALID = {N}, pełne 768D dla {len(pelne)}; cząsteczek z rzutem {N * TOK}')

# ---------------------------------------------------------------- sieć, rzut, pliki wyjściowe (memmap, bo 806 × 100 × 600 × 128 f16 ≈ 12 GB)
net = M.MalyViT().to(dev)
net.eval()
net.zapisuj_latent(True)
Xg = torch.from_numpy(X).to(dev)
g = torch.Generator().manual_seed(a.ziarno_rzutu)
R = (torch.randn(768, a.rzut_dim, generator=g) / a.rzut_dim ** 0.5).to(dev)
np.save(OUT / 'R_rp128.npy', R.cpu().numpy())
rp = np.lib.format.open_memmap(OUT / 'rp128.npy', mode='w+', dtype=np.float16, shape=(K, N, TOK, a.rzut_dim))
p768 = np.lib.format.open_memmap(OUT / 'pelne768.npy', mode='w+', dtype=np.float16, shape=(K, len(pelne), TOK, 768))
sr = np.lib.format.open_memmap(OUT / 'srednia768.npy', mode='w+', dtype=np.float32, shape=(K, N, 768))
normy = np.lib.format.open_memmap(OUT / 'normy768.npy', mode='w+', dtype=np.float16, shape=(K, N, TOK, 2))
oceny = np.zeros((K, N, 2), np.float32)
z0 = torch.zeros((N, TOK, 768), dtype=torch.float16, device=dev)

for ik, k in enumerate(wybrane):
    st = torch.load(kroki_pliki[k], map_location='cpu', weights_only=False)
    net.load_state_dict(st['net'], strict=True)
    with torch.no_grad():
        for s in range(0, N, a.paczka):
            x = Xg[s:s + a.paczka].permute(0, 3, 1, 2).float() / 255.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, lat = net(x)
            lat = lat.float()                                                   # (B, 600, 768)
            B = len(lat)
            if ik == 0:
                z0[s:s + B] = lat.half()
            rp[ik, s:s + B] = (lat @ R).half().cpu().numpy()
            sr[ik, s:s + B] = lat.mean(1).cpu().numpy()
            normy[ik, s:s + B, :, 0] = lat.norm(dim=2).half().cpu().numpy()
            normy[ik, s:s + B, :, 1] = (lat - z0[s:s + B].float()).norm(dim=2).half().cpu().numpy()
            for j in range(len(pelne)):
                if s <= pelne[j] < s + B:
                    p768[ik, j] = lat[pelne[j] - s].half().cpu().numpy()
            bx, sc = M.dekoduj(o)
            bx, sc = bx.cpu().numpy(), sc.cpu().numpy()
            for j in range(B):
                oceny[ik, s + j, 0] = float(sc[j, 0])
                oceny[ik, s + j, 1] = float(max((T.iou(bx[j, 0], gg) for gg in ramki[s + j]), default=0.0))
            del lat, o
    if ik % 50 == 0 or ik == K - 1:
        log(f'krok {k} ({ik + 1}/{K}): |z−z0| {float(normy[ik, :, :, 1].astype(np.float32).mean()):.2f}, |z| {float(normy[ik, :, :, 0].astype(np.float32).mean()):.2f}, '
            f'pewność {oceny[ik, :, 0].mean():.3f}, IoU {oceny[ik, :, 1].mean():.3f}, VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB')
    del st

for arr in (rp, p768, sr, normy):
    arr.flush()
np.save(OUT / 'oceny.npy', oceny)
meta = dict(opis=__doc__, kroki=[int(k) for k in wybrane], K=K, N=N, tokenow=TOK, zdjecia=zdjecia, pelne768_indeksy=[int(v) for v in pelne],
            rzut=dict(dim=a.rzut_dim, ziarno=a.ziarno_rzutu, wzor='torch.randn(768, dim, generator=Generator().manual_seed(ziarno)) / dim**0.5',
                      plik='R_rp128.npy', jak='obserwator_rp.py fali 3 pokręteł (te same współrzędne co ANALIZA_*_rp)'),
            pliki=dict(rp128='(K, N, 600, 128) float16', pelne768='(K, N768, 600, 768) float16', srednia768='(K, N, 768) float32, średnia tokenów zdjęcia',
                       normy768='(K, N, 600, 2) float16: |z|, |z − z(krok 0)|', oceny='(K, N, 2) float32: pewność szczytu mapy, IoU jego ramki z GT'),
            latent='GELU MLP ostatniego bloku MalyViT (dim 192, 6 bloków), 600 tokenów × 768, forward eval w bf16 autocast jak pomiar_eden.py',
            wybor_zdjec=f'default_rng({a.ziarno}): {a.n_train} z manifestu, {a.n_val} z dane.walidacja(); pierwsze {n} z każdej części',
            checkpoint_ostatni=dict(krok=int(wybrane[-1]), sha256=sha(kroki_pliki[wybrane[-1]])), config=vars(a),
            pochodzenie=dict(skrypt='ruch60_rp.py', data=time.strftime('%Y-%m-%d'), maszyna='Eden', zadanie=os.environ.get('SLURM_JOB_ID')))
(OUT / 'META.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
log('GOTOWE', OUT, f'{sum(os.path.getsize(OUT / f) for f in os.listdir(OUT)) / 1e9:.2f} GB')
