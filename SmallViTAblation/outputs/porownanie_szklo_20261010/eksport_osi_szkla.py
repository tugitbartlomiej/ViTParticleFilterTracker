"""Eksport osi czasu „względem klatki” (układ szkła) dla wszystkich 60 000 cząsteczek biegu 35 000 do formatu
aplikacji (manifest v2, klatki q16), żeby zakładka „Ruch zbioru” odtwarzała grzechotanie w klatce z całym
zestawem kolorowań, fizyki ruchu i obszarów oka.

Układ: z_i^CR(t) = z_i(t) − średnia_{j ∈ N6(i)} [z_j(t) − z_j(300)], sąsiedzi = 6 najbliższych w latencie (PCA32)
w obrębie zdjęcia w kroku 300. Współrzędne ekranu: stała baza PCA3 policzona z z^CR w krokach 300, 2188, 10 000.
Norma (4. kolumna) = |z^CR(t) − z^CR(300)| w PCA32 (odpowiednik „zmiany od pierwszego kroku” w pomiarze).
Kroki: co 2 w 300–2188 (część gęsta co 1 krok), co 50 w 2200–22 950 oraz 22 970. Statystyki pewności/IoU
interpolowane liniowo między 22 checkpointami pomiaru naturalnego (metrics.npy i stats), oznaczone w manifeście.
Uruchomienie: F:/Python/3.13/python.exe eksport_osi_szkla.py
"""
import glob
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

G = Path(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty')
E35K = Path(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd')
OUT = Path(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261010T_natural_szklo_klatka_35k')
NI, TOK = 100, 600
N = NI * TOK
T_REF = 300
KROKI = sorted(set(range(300, 2189, 2)) | set(range(2200, 22951, 50)) | {22970})
Q16 = 65535.0
t0 = time.time()


def log(*a):
    print(*a, f'[{time.time() - t0:.0f} s]', flush=True)


def kwantyzuj(frame):
    frame = np.asarray(frame, np.float32)
    assert frame.ndim == 2 and frame.shape[1] == 4 and np.isfinite(frame).all() and (frame[:, 3] >= 0).all()
    lo = frame[:, :3].min(axis=0)
    span = (frame[:, :3].max(axis=0).astype(np.float64) - lo).astype(np.float32)
    nmax = np.float32(frame[:, 3].max())
    q = np.empty(frame.shape, '<u2')
    skala = np.where(span > 0, Q16 / np.maximum(span.astype(np.float64), 1e-300), 0.)
    q[:, :3] = np.clip(np.rint((frame[:, :3].astype(np.float64) - lo) * skala), 0, Q16)
    q[:, 3] = np.clip(np.rint(frame[:, 3].astype(np.float64) / float(nmax) * Q16), 0, Q16) if nmax > 0 else 0
    return q, {'lo': [float(v) for v in lo], 'span': [float(v) for v in span], 'norm_max': float(nmax)}


def zapisz_klatke(path, frame):
    dane, quant = kwantyzuj(frame)
    raw = dane.tobytes()
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {'name': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'format': 'q16', 'quant': quant}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- kontrola zgodności sond z osią pomiaru
m_src = json.loads((E35K / 'manifest.json').read_text(encoding='utf-8'))
sondy = json.load(open(G / 'SONDY.json', encoding='utf-8'))['sondy']
assert [r['pool_row'] for r in m_src['images']] == sondy, 'kolejność sond w gęstym zapisie różni się od osi pomiaru'
pliki = sorted(glob.glob(str(G / 'pca32_*.npz')))
pozycje = {}
for p in pliki:
    kr = np.load(p)['kroki']
    for j, k in enumerate(kr):
        pozycje[int(k)] = (p, j)
brak = [k for k in KROKI if k not in pozycje]
assert not brak, f'brak kroków w zapisie: {brak[:10]}'

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# ---------------------------------------------------------------- przebieg wstępny: sąsiedzi w kroku 300, baza PCA3
def klatka32(krok):
    p, j = pozycje[krok]
    return np.load(p)['pca32'][j].astype(np.float32).reshape(N, 32)


Z300 = klatka32(T_REF)
N6 = np.empty((N, 6), int)
d_nn, d_sh = np.empty(N), np.empty(N)
for im in range(NI):
    x = Z300[im * TOK:(im + 1) * TOK]
    n2 = (x ** 2).sum(1)
    D = n2[:, None] + n2[None, :] - 2 * x @ x.T
    np.maximum(D, 0, out=D)
    np.fill_diagonal(D, np.inf)
    D = np.sqrt(D)
    nn = np.argpartition(D, 6, axis=1)[:, :6]
    N6[im * TOK:(im + 1) * TOK] = nn + im * TOK
    d_nn[im * TOK:(im + 1) * TOK] = D.min(1)
    d_sh[im * TOK:(im + 1) * TOK] = np.take_along_axis(D, nn, 1).mean(1)
log('sąsiedzi policzeni; d_nn', round(float(np.median(d_nn)), 3), 'd_shell', round(float(np.median(d_sh)), 3))


def cr(frame32):
    return frame32 - (frame32[N6] - Z300[N6]).mean(1)


probka = np.concatenate([cr(klatka32(k)) for k in (T_REF, 2188, 10000)])
mean32 = probka.mean(0)
_, S, Vt = np.linalg.svd(probka - mean32, full_matrices=False)
comp = np.ascontiguousarray(Vt[:3].T.astype(np.float32))       # (32, 3)
wariancja = (S ** 2) / (len(probka) - 1)
log('baza PCA3: udział', [round(float(v), 3) for v in (wariancja[:3] / wariancja.sum())])
del probka
np.savez(OUT / 'initial_pca.npz', mean=mean32.astype(np.float32), components=comp, eigenvalues=wariancja[:3].astype(np.float32),
         reservoir_token_indices=np.arange(0, N, N // 4096)[:4096], baza='PCA32 gęstego zapisu (BAZA_PCA32.npz) → 3 składowe z^CR')

# ---------------------------------------------------------------- statystyki z pomiaru naturalnego do interpolacji
st_steps = np.array(sorted(int(s) for s in m_src['stats']))
st = {k: np.array([m_src['stats'][str(s)][k] for s in st_steps], float) for k in ('confidence', 'iou', 'both90')}
metr_src = np.load(E35K / 'metrics.npy')                           # (100, 22, 5): confidence, iou, delta_l2, training_step, both90
ms = np.array(m_src['metric_steps'], float)
metr = np.zeros((NI, len(KROKI), 5))
for i in range(NI):
    for c in (0, 1, 4):
        metr[i, :, c] = np.interp(KROKI, ms, metr_src[i, :, c])
metr[:, :, 3] = KROKI

# ---------------------------------------------------------------- główny przebieg strumieniowy
files, stats, curves = {}, {}, {k: [] for k in ('steps', 'confidence', 'iou', 'both90', 'real_step')}
lo_n, hi_n = np.full(3, np.inf), np.full(3, -np.inf)
lo_m, hi_m = np.full(3, np.inf), np.full(3, -np.inf)
norm_scale = 0.0
xyz0 = None
CR300 = Z300.copy()
do_zrobienia = {}
for k in KROKI:
    do_zrobienia.setdefault(pozycje[k][0], []).append(k)
for p in pliki:
    if p not in do_zrobienia:
        continue
    z = np.load(p)
    pca = z['pca32']
    for k in do_zrobienia[p]:
        frame32 = pca[pozycje[k][1]].astype(np.float32).reshape(N, 32)
        c = cr(frame32)
        xyz = (c - mean32) @ comp
        norma = np.linalg.norm(c - CR300, axis=1)
        if xyz0 is None:
            xyz0 = xyz.copy()
        frame = np.concatenate([xyz, norma[:, None]], 1).astype(np.float32)
        files[str(k)] = zapisz_klatke(OUT / f'step_{k:04d}.q16', frame)
        lo_n, hi_n = np.minimum(lo_n, xyz.min(0)), np.maximum(hi_n, xyz.max(0))
        mv = xyz - xyz0
        lo_m, hi_m = np.minimum(lo_m, mv.min(0)), np.maximum(hi_m, mv.max(0))
        norm_scale = max(norm_scale, float(norma.max()))
        i_k = KROKI.index(k)
        metr[:, i_k, 2] = norma.reshape(NI, TOK).mean(1)
        s = dict(confidence=float(np.interp(k, st_steps, st['confidence'])), iou=float(np.interp(k, st_steps, st['iou'])),
                 both90=int(round(float(np.interp(k, st_steps, st['both90'])))), norm_median=float(np.median(norma)), norm_max=float(norma.max()),
                 real_step=int(k), interpolated_metrics=True)
        stats[str(k)] = s
        for kk in ('confidence', 'iou', 'both90', 'real_step'):
            curves[kk].append(s[kk])
        curves['steps'].append(int(k))
    del z, pca
    log(os.path.basename(p), 'klatek gotowych', len(files))

np.save(OUT / 'metrics.npy', metr)
shutil.copy2(E35K / 'obszary_oka.npz', OUT / 'obszary_oka.npz')
cfg = json.loads((E35K / 'config.json').read_text(encoding='utf-8'))
cfg['uwaga'] = 'oś pochodna: układ względem klatki z gęstego zapisu, nie pomiar checkpointów'
(OUT / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
meta = {
    'schema_version': 2, 'status': 'complete', 'time_axis': 'training_checkpoint', 'baseline_step': T_REF,
    'steps': KROKI, 'files': files, 'stats': stats, 'point_count': N, 'image_count': NI, 'tokens_per_image': TOK,
    'image_names': m_src['image_names'], 'images': m_src['images'], 'frame_format': 'q16',
    'bounds': {'natural': np.column_stack([lo_n, hi_n]).tolist(), 'movement': np.column_stack([lo_m, hi_m]).tolist()},
    'norm_scale': norm_scale, 'checkpoint_sha256': None, 'checkpoint_step': KROKI[-1],
    'curves': dict(curves, label='Pewność i IoU interpolowane liniowo między 22 checkpointami pomiaru naturalnego'),
    'metric_columns': ['confidence', 'iou', 'delta_l2', 'training_step', 'both90'], 'metric_steps': KROKI,
    'metrics_file': {'name': 'metrics.npy', 'sha256': sha(OUT / 'metrics.npy'), 'shape': [NI, len(KROKI), 5], 'dtype': 'float64',
                     'unobserved_values': 'interpolowane (confidence, iou, both90); delta_l2 = |z^CR(t) − z^CR(300)| średnio po zdjęciu'},
    'pca_file': {'name': 'initial_pca.npz', 'sha256': sha(OUT / 'initial_pca.npz')},
    'uklad': {'nazwa': 'względem klatki (cage-relative)', 'sasiedzi': '6 najbliższych w PCA32 w obrębie zdjęcia, krok 300',
              'wzor': 'z_i^CR(t) = z_i(t) − mean_j∈N6(i) [z_j(t) − z_j(300)]', 'd_nn_mediana': float(np.median(d_nn)), 'd_shell_mediana': float(np.median(d_sh)),
              'baza_ekranu': 'PCA3 z z^CR w krokach 300, 2188, 10000', 'zrodlo': str(G), 'pliki_zrodla': [os.path.basename(p) for p in pliki],
              'os_pomiaru': str(E35K), 'sondy': sondy},
    'coordinates': '(z^CR − mean) @ components; image-major then token-major', 'norm': 'L2(z^CR(step) − z^CR(300)) w PCA32',
    'interpolation': False, 'full_latent_states_persisted': False, 'inference_only': True, 'config': cfg,
    'producer_provenance': {'skrypt': 'outputs/porownanie_szklo_20261010/eksport_osi_szkla.py', 'data': '2026-10-10'},
}
(OUT / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
razem = sum(f['bytes'] for f in files.values())
log('GOTOWE:', OUT, len(KROKI), 'kroków,', round(razem / 1e6), 'MB')
