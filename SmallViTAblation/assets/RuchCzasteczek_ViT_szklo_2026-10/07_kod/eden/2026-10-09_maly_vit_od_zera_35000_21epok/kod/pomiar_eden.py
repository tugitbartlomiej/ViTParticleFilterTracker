"""Pomiar naturalny na Edenie: trajektorie tokenów wielu zdjęć (uczących i walidacyjnych) po checkpointach biegu 35 000,
w formacie osi czasu aplikacji „Ruch zbioru” (manifest v2, klatki q16: x, y, z = rzut PCA3 latentu, norma = |z − z(krok 0)|
w 768D). Bez uczenia: każdy checkpoint wczytany strict, forward w eval z bf16, latent = GELU MLP ostatniego bloku.
Dodatkowo PCA32 (stała baza z kroku 0) dla podzbioru zdjęć do analizy fizyki ruchu i oceny generalizacji (TRAIN vs VALID).

Uruchomienie (w katalogu kodu biegu 35 000, ze zmiennymi MALY_VIT_* jak w treningu):
  python pomiar_eden.py --wagi .../wyniki/pelny/wagi --manifest .../MANIFEST_35000.jsonl --zipy ... --out ...
         [--n_train 3000 --n_val 1000 --ziarno 2026 --pca32_n 100 --paczka 128 --wszystkie]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
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
ap.add_argument('--n_train', type=int, default=3000)
ap.add_argument('--n_val', type=int, default=1000)
ap.add_argument('--ziarno', type=int, default=2026)
ap.add_argument('--pca32_n', type=int, default=100)
ap.add_argument('--paczka', type=int, default=128)
ap.add_argument('--wszystkie', action='store_true', help='wszystkie checkpointy zamiast co 10 do 1000, co 100 i końców epok')
ap.add_argument('--max_krok', type=int, default=22974, help='ostatni checkpoint (domyślnie koniec 21. epoki; bieg ma 60 epok)')
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
TOK = 600
Q16 = 65535.0
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


# ---------------------------------------------------------------- checkpointy
kroki_pliki = {}
for p in Path(a.wagi).iterdir():
    m = re.fullmatch(r'k([0-9]+)\.pt', p.name)
    if m:
        kroki_pliki[int(m[1])] = p
kmax = min(max(kroki_pliki), a.max_krok)
wybrane = sorted(k for k in kroki_pliki if k <= kmax and (a.wszystkie or (k <= 1000 and k % 10 == 0) or k % 100 == 0 or k % 1094 == 0 or k == kmax))
log(f'checkpointów na dysku {len(kroki_pliki)}, wybranych {len(wybrane)} (0..{kmax})')

# ---------------------------------------------------------------- zdjęcia: losowe uczące + walidacyjne
rng = np.random.default_rng(a.ziarno)
wt = wiersze_35000(a.manifest)
idx_tr = sorted(rng.choice(len(wt), a.n_train, replace=False).tolist())
rows_tr = [wt[i] for i in idx_tr]
X_tr = wczytaj_z_zipow(rows_tr, a.zipy)
wal_w, wal_X = D.walidacja()
idx_val = sorted(rng.choice(len(wal_w), a.n_val, replace=False).tolist())
rows_val = [wal_w[i] for i in idx_val]
X_val = wal_X[idx_val]
X = np.concatenate([X_tr, X_val])
N = len(X)
images = []
for j, (i, r) in enumerate(zip(idx_tr, rows_tr)):
    w, h = r['wh']
    b = r['ramki'][0] if len(r['ramki']) else np.zeros(4, np.float32)
    images.append(dict(index=j, image_id=j, annotation_id=j, image_name=r['frame_id'].split('/')[-2] + '_' + r['frame_id'].split('/')[-1][-6:] + '.jpg',
                       image_sha256=r['sha'], frame_id=r['frame_id'], pool_row=int(i), split='TRAIN', category='narzędzie',
                       source_width=int(w), source_height=int(h), bbox_xyxy_480=[float(v) for v in b], resize_scale_xy=[D.W_ / w, D.H_ / h],
                       n_gt=int(len(r['ramki']))))
for j, (i, r) in enumerate(zip(idx_val, rows_val), start=len(rows_tr)):
    im = cv2.imread(r['sciezka'], cv2.IMREAD_COLOR)
    h, w = (im.shape[0], im.shape[1]) if im is not None else (480, 720)
    b = r['ramki'][0] if len(r['ramki']) else np.zeros(4, np.float32)
    images.append(dict(index=j, image_id=j, annotation_id=j, image_name=Path(r['sciezka']).name, image_sha256=sha(r['sciezka']) if im is not None else '',
                       frame_id=r['frame_id'], val_row=int(i), split='VALID', category='narzędzie', source_width=int(w), source_height=int(h),
                       bbox_xyxy_480=[float(v) for v in b], resize_scale_xy=[D.W_ / w, D.H_ / h], n_gt=int(len(r['ramki']))))
ramki_gt = [r['ramki'] for r in rows_tr] + [r['ramki'] for r in rows_val]
split = np.array([0] * len(rows_tr) + [1] * len(rows_val))
del X_tr, X_val, wal_X
log(f'zdjęcia: {len(rows_tr)} TRAIN + {len(rows_val)} VALID = {N}, cząsteczek {N * TOK}')
json.dump(dict(images=images), open(OUT / 'images.json', 'w', encoding='utf-8'), ensure_ascii=False)

# ---------------------------------------------------------------- sieć i przebieg po checkpointach
net = M.MalyViT().to(dev)
net.eval()
net.zapisuj_latent(True)
Xg = torch.from_numpy(X).to(dev)                                               # uint8 (N, 320, 480, 3) ~ 1,8 GB przy 4000
pod = np.concatenate([np.arange(min(a.pca32_n, len(rows_tr))), len(rows_tr) + np.arange(min(a.pca32_n, len(rows_val)))])
rez = None
mean3 = comp3 = eig3 = None
mean32 = comp32 = None
z0 = None
files, stats, curves = {}, {}, {k: [] for k in ('steps', 'confidence', 'iou', 'both90', 'real_step')}
lo_n, hi_n, lo_m, hi_m = np.full(3, np.inf), np.full(3, -np.inf), np.full(3, np.inf), np.full(3, -np.inf)
norm_scale = 0.0
xyz0 = None
metr = np.full((N, len(wybrane), 5), np.nan)
pca32_pod = np.zeros((len(wybrane), len(pod), TOK, 32), np.float16)
oceny = []


@torch.no_grad()
def paczki():
    """Kolejne paczki: (s, latent (B, 600, 768) float32 na karcie, mapa o)."""
    for s in range(0, N, a.paczka):
        x = Xg[s:s + a.paczka].permute(0, 3, 1, 2).float() / 255.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, lat = net(x)
        yield s, lat.float(), o


@torch.no_grad()
def baza_pca():
    """Krok 0, przebieg 1: próbka 4096 tokenów → PCA3 (jak w pomiarze aplikacji), losowe 5 % tokenów → PCA32; z0 w half."""
    global rez, mean3, comp3, eig3, mean32, comp32, z0
    g = torch.Generator(device='cpu').manual_seed(a.ziarno)
    rez = torch.randperm(N * TOK, generator=g)[:4096].sort().values
    rez_set = rez.numpy()
    z0 = torch.empty((N, TOK, 768), dtype=torch.float16, device=dev)
    S = []
    suma = torch.zeros(768, dtype=torch.float64, device=dev)
    C = torch.zeros((768, 768), dtype=torch.float64, device=dev)
    n_c = 0
    for s, lat, _ in paczki():
        z0[s:s + len(lat)] = lat.half()
        lo, hi = s * TOK, (s + len(lat)) * TOK
        w = rez_set[(rez_set >= lo) & (rez_set < hi)] - lo
        if len(w):
            S.append(lat.reshape(-1, 768)[torch.as_tensor(w, device=dev)].double())
        Z = lat.reshape(-1, 768)
        ii = torch.randperm(len(Z), generator=g)[:max(1, len(Z) // 20)].to(dev)
        Zs = Z[ii].double()
        suma += Zs.sum(0)
        C += Zs.T @ Zs
        n_c += len(Zs)
    S = torch.cat(S)
    mean3 = S.mean(0)
    _, sv, V = torch.linalg.svd(S - mean3, full_matrices=False)
    comp3 = V[:3].T.float().contiguous()                                        # (768, 3)
    eig3 = (sv[:3] ** 2 / (len(S) - 1)).float()
    mean32 = (suma / n_c)
    Cc = C / n_c - mean32[:, None] * mean32[None, :]
    w, Vv = torch.linalg.eigh(Cc)
    comp32 = Vv[:, -32:].flip(1).T.float().contiguous()                         # (32, 768)
    mean32 = mean32.float()
    np.savez(OUT / 'initial_pca.npz', mean=mean3.float().cpu().numpy(), components=comp3.cpu().numpy(), eigenvalues=eig3.cpu().numpy(),
             reservoir_token_indices=rez.numpy(), mean32=mean32.cpu().numpy(), comp32=comp32.cpu().numpy(),
             udzial32=float(w[-32:].sum() / w.sum()), pca32_z=f'losowe 5 % tokenów kroku 0 ({n_c})')
    mean3 = mean3.float()


pod_t = None
for ik, k in enumerate(wybrane):
    st = torch.load(kroki_pliki[k], map_location='cpu', weights_only=False)
    net.load_state_dict(st['net'], strict=True)
    if ik == 0:
        baza_pca()
        pod_t = torch.as_tensor(pod, device=dev)
    frame = np.empty((N * TOK, 4), np.float32)
    conf = np.zeros(N)
    iou_v = np.zeros(N)
    p32 = torch.empty((len(pod), TOK, 32), dtype=torch.float16, device=dev)
    for s, lat, o in paczki():
        B = len(lat)
        xyz = (lat - mean3) @ comp3
        norma = (lat - z0[s:s + B].float()).norm(dim=2)
        frame[s * TOK:(s + B) * TOK] = torch.cat([xyz, norma[..., None]], 2).reshape(-1, 4).cpu().numpy()
        w = torch.nonzero((pod_t >= s) & (pod_t < s + B)).reshape(-1)
        if len(w):
            p32[w] = ((lat[pod_t[w] - s] - mean32) @ comp32.T).half()
        bx, sc = M.dekoduj(o)
        bx, sc = bx.cpu().numpy(), sc.cpu().numpy()
        for j in range(B):
            conf[s + j] = float(sc[j, 0])
            g = ramki_gt[s + j]
            iou_v[s + j] = float(max((T.iou(bx[j, 0], gg) for gg in g), default=0.0))
        del lat, o, xyz, norma
    lo = frame[:, :3].min(0)
    span = (frame[:, :3].max(0).astype(np.float64) - lo).astype(np.float32)
    nm = np.float32(frame[:, 3].max())
    q = np.empty(frame.shape, '<u2')
    sk = np.where(span > 0, Q16 / np.maximum(span.astype(np.float64), 1e-300), 0.)
    q[:, :3] = np.clip(np.rint((frame[:, :3].astype(np.float64) - lo) * sk), 0, Q16)
    q[:, 3] = np.clip(np.rint(frame[:, 3].astype(np.float64) / float(nm) * Q16), 0, Q16) if nm > 0 else 0
    raw = q.tobytes()
    (OUT / f'step_{k:04d}.q16').write_bytes(raw)
    files[str(k)] = {'name': f'step_{k:04d}.q16', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'format': 'q16',
                     'quant': {'lo': [float(v) for v in lo], 'span': [float(v) for v in span], 'norm_max': float(nm)}}
    if xyz0 is None:
        xyz0 = frame[:, :3].copy()
    lo_n, hi_n = np.minimum(lo_n, frame[:, :3].min(0)), np.maximum(hi_n, frame[:, :3].max(0))
    mv = frame[:, :3] - xyz0
    lo_m, hi_m = np.minimum(lo_m, mv.min(0)), np.maximum(hi_m, mv.max(0))
    norm_scale = max(norm_scale, float(nm))
    both = (conf >= .9) & (iou_v >= .9)
    nimg = frame[:, 3].reshape(N, TOK).mean(1)
    metr[:, ik, 0], metr[:, ik, 1], metr[:, ik, 2], metr[:, ik, 3], metr[:, ik, 4] = conf, iou_v, nimg, k, both
    s = dict(confidence=float(conf.mean()), iou=float(iou_v.mean()), both90=int(both.sum()), norm_median=float(np.median(frame[:, 3])),
             norm_max=float(nm), real_step=int(k))
    stats[str(k)] = s
    for kk in ('confidence', 'iou', 'both90', 'real_step'):
        curves[kk].append(s[kk])
    curves['steps'].append(int(k))
    oceny.append(dict(krok=int(k), conf_train=float(conf[split == 0].mean()), conf_val=float(conf[split == 1].mean()),
                      iou_train=float(iou_v[split == 0].mean()), iou_val=float(iou_v[split == 1].mean()),
                      norma_train=float(nimg[split == 0].mean()), norma_val=float(nimg[split == 1].mean())))
    pca32_pod[ik] = p32.cpu().numpy()
    if ik % 20 == 0 or ik == len(wybrane) - 1:
        log(f'krok {k} ({ik + 1}/{len(wybrane)}): pewność {s["confidence"]:.3f} (train {oceny[-1]["conf_train"]:.3f} / val {oceny[-1]["conf_val"]:.3f}), '
            f'IoU {s["iou"]:.3f} (train {oceny[-1]["iou_train"]:.3f} / val {oceny[-1]["iou_val"]:.3f}), norma {s["norm_median"]:.2f}, '
            f'VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB')
    del st, p32

np.save(OUT / 'metrics.npy', metr)
np.savez_compressed(OUT / 'pca32_podzbior.npz', pca32=pca32_pod, kroki=np.array(wybrane), zdjecia=pod, split=split[pod],
                    frame_id=np.array([images[i]['frame_id'] for i in pod]), uwaga='PCA32 w stałej bazie z kroku 0 (initial_pca.npz: mean32, comp32)')
json.dump(oceny, open(OUT / 'OCENY_POMIAR.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
meta = {
    'schema_version': 2, 'status': 'complete', 'time_axis': 'training_checkpoint', 'baseline_step': int(wybrane[0]),
    'steps': [int(k) for k in wybrane], 'files': files, 'stats': stats, 'point_count': N * TOK, 'image_count': N, 'tokens_per_image': TOK,
    'image_names': [im['image_name'] for im in images], 'images': images, 'frame_format': 'q16',
    'bounds': {'natural': np.column_stack([lo_n, hi_n]).tolist(), 'movement': np.column_stack([lo_m, hi_m]).tolist()},
    'norm_scale': norm_scale, 'checkpoint_sha256': sha(kroki_pliki[wybrane[-1]]), 'checkpoint_step': int(wybrane[-1]),
    'curves': dict(curves, label='Pewność (najwyższy szczyt mapy) i IoU jego ramki z najlepszą ramką GT, średnio po wszystkich zdjęciach'),
    'metric_columns': ['confidence', 'iou', 'delta_l2', 'training_step', 'both90'], 'metric_steps': [int(k) for k in wybrane],
    'metrics_file': {'name': 'metrics.npy', 'sha256': sha(OUT / 'metrics.npy'), 'shape': [N, len(wybrane), 5], 'dtype': 'float64',
                     'unobserved_values': 'brak (wszystkie zdjęcia we wszystkich krokach)'},
    'pca_file': {'name': 'initial_pca.npz', 'sha256': sha(OUT / 'initial_pca.npz')},
    'coordinates': '(z − mean) @ components; image-major then token-major', 'norm': 'L2(z(step) − z(first selected checkpoint)) w 768D',
    'interpolation': False, 'full_latent_states_persisted': False, 'inference_only': True, 'precision': 'bf16 autocast (jak obserwator gęsty)',
    'config': dict(vars(a), n_images=N, n_train=len(rows_tr), n_val=len(rows_val), checkpoint_dir=a.wagi),
    'split': {'TRAIN': len(rows_tr), 'VALID': len(rows_val), 'uwaga': 'VALID = próbka 3000 klatek walidacji biegu (dane.walidacja, rng 0), z niej losowe n_val'},
    'producer_provenance': {'skrypt': 'pomiar_eden.py', 'data': '2026-10-10', 'maszyna': 'Eden', 'image_count': N, 'tokens_per_image': TOK},
}
(OUT / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
log('GOTOWE', OUT, len(wybrane), 'kroków,', round(sum(f['bytes'] for f in files.values()) / 1e9, 2), 'GB')
