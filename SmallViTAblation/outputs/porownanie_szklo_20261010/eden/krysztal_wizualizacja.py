"""Wizualizacja kryształu tokenów (Eden, CPU) z eksportu ruch60 (rp128: 806 checkpointów × 100 zdjęć × 600 tokenów × 128).
(A) oś czasu w formacie aplikacji „Ruch zbioru” (manifest v2, klatki q16) do animacji natywnej na GPU: współrzędne = stała baza PCA3
    położeń względnych (z − średnia tokenów zdjęcia) z kroków 0, 1094, 65 640; 4. kolumna = amplituda drgań wokół węzła
    |x(t) − c(t)|, c = średnia krocząca ±5 checkpointów (kolorowanie normą = mapa „temperatury” węzłów).
(B) panel krysztal.png: siatka jednego zdjęcia w kolejnych krokach (wiązania między sąsiednimi patchami), mapa amplitudy drgań
    i mapa „sąsiedzi w latencie = sąsiednie patche” na siatce 20 × 30 z ramką narzędzia, krzywe numeru patcha / Jaccarda,
    wiązki kontra kulki, ruchliwość kontra IoU; plus test narzędzie kontra tło (amplituda drgań w ramce GT i poza nią).
Uruchomienie: python krysztal_wizualizacja.py <eksport> <pomiar60> <wyniki> <oś_wyjściowa> <panel_wyjściowy>"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

EKS, POM, WYN, OS_, PAN = (Path(p) for p in sys.argv[1:6])
OS_.mkdir(parents=True, exist_ok=True); PAN.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
META = json.load(open(EKS / 'META.json', encoding='utf-8'))
kroki = [int(k) for k in META['kroki']]
rp = np.load(EKS / 'rp128.npy', mmap_mode='r')
oceny = np.load(EKS / 'oceny.npy')
K, NI, TOK, D = rp.shape
H, W, EP, Q16, PL = 20, 30, 1094, 65535.0, 16
N = NI * TOK
mp = json.loads((POM / 'manifest.json').read_text(encoding='utf-8'))
assert [int(s) for s in mp['steps']] == kroki, 'kroki eksportu różne od pomiaru'
idx_pom = [z['index'] for z in []]
n_half = NI // 2
wiersze = list(range(n_half)) + list(range(3000, 3000 + n_half))             # zdjęcia eksportu = 0..49 TRAIN i 3000..3049 VALID pomiaru
images = [dict(mp['images'][r], index=j, image_id=j, annotation_id=j) for j, r in enumerate(wiersze)]
assert [im['frame_id'] for im in images] == [z['frame_id'] for z in META['zdjecia']], 'zdjęcia eksportu różne od pomiaru'


def wzgl(t):
    X = np.asarray(rp[t]).astype(np.float32)
    return X - X.mean(1, keepdims=True)


# ---------------------------------------------------------------- (A) oś aplikacji
probka = np.concatenate([wzgl(kroki.index(k)).reshape(N, D) for k in (0, 1094, 65640)])
mean = probka.mean(0); _, S, Vt = np.linalg.svd(probka[::3] - mean, full_matrices=False)
comp = np.ascontiguousarray(Vt[:3].T.astype(np.float32)); war = S ** 2 / (len(probka[::3]) - 1)
log('PCA3 udział', [round(float(v), 3) for v in war[:3] / war.sum()])
del probka
np.savez(OS_ / 'initial_pca.npz', mean=mean.astype(np.float32), components=comp, eigenvalues=war[:3].astype(np.float32),
         reservoir_token_indices=np.arange(0, N, max(1, N // 4096))[:4096], baza='PCA3 położeń względnych rp128 z kroków 0, 1094, 65640')


def kwantyzuj(frame):
    lo = frame[:, :3].min(0); span = (frame[:, :3].max(0).astype(np.float64) - lo).astype(np.float32); nmax = np.float32(frame[:, 3].max())
    q = np.empty(frame.shape, '<u2'); sk = np.where(span > 0, Q16 / np.maximum(span.astype(np.float64), 1e-300), 0.)
    q[:, :3] = np.clip(np.rint((frame[:, :3].astype(np.float64) - lo) * sk), 0, Q16)
    q[:, 3] = np.clip(np.rint(frame[:, 3].astype(np.float64) / float(nmax) * Q16), 0, Q16) if nmax > 0 else 0
    return q, {'lo': [float(v) for v in lo], 'span': [float(v) for v in span], 'norm_max': float(nmax)}


R = 5
files, stats = {}, {}
curves = {k: [] for k in ('steps', 'confidence', 'iou', 'both90', 'real_step')}
metr = np.zeros((NI, K, 5))
lo_n, hi_n, lo_m, hi_m = np.full(3, np.inf), np.full(3, -np.inf), np.full(3, np.inf), np.full(3, -np.inf)
norm_scale = 0.0; xyz0 = None
okno = {}                                                                        # bufor położeń względnych ±R
amp_sum = np.zeros((NI, TOK)); amp_n = 0                                         # amplituda drgań w epokach 50–60
for t in range(K):
    for u in range(max(0, t - R), min(K, t + R + 1)):
        if u not in okno:
            okno[u] = wzgl(u)
    for u in [u for u in okno if u < t - R]:
        del okno[u]
    c = np.mean([okno[u] for u in range(max(0, t - R), min(K, t + R + 1))], 0)
    x = okno[t]
    amp = np.linalg.norm(x - c, axis=-1)                                          # (NI, 600)
    if kroki[t] >= 50 * EP:
        amp_sum += amp; amp_n += 1
    xyz = (x.reshape(N, D) - mean) @ comp
    frame = np.concatenate([xyz, amp.reshape(N, 1)], 1).astype(np.float32)
    q, quant = kwantyzuj(frame); raw = q.tobytes(); nazwa = f'step_{kroki[t]:05d}.q16'
    (OS_ / nazwa).write_bytes(raw)
    files[str(kroki[t])] = {'name': nazwa, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'format': 'q16', 'quant': quant}
    if xyz0 is None:
        xyz0 = xyz.copy()
    lo_n, hi_n = np.minimum(lo_n, xyz.min(0)), np.maximum(hi_n, xyz.max(0)); mv = xyz - xyz0
    lo_m, hi_m = np.minimum(lo_m, mv.min(0)), np.maximum(hi_m, mv.max(0)); norm_scale = max(norm_scale, float(amp.max()))
    conf, iou = oceny[t, :, 0], oceny[t, :, 1]; both = (conf >= .9) & (iou >= .9)
    metr[:, t] = np.stack([conf, iou, amp.mean(1), np.full(NI, kroki[t]), both], 1)
    s = dict(confidence=float(conf.mean()), iou=float(iou.mean()), both90=int(both.sum()), norm_median=float(np.median(amp)), norm_max=float(amp.max()), real_step=kroki[t])
    stats[str(kroki[t])] = s
    for kk in ('confidence', 'iou', 'both90', 'real_step'):
        curves[kk].append(s[kk])
    curves['steps'].append(kroki[t])
    if t % 100 == 0:
        log(f'oś: krok {kroki[t]} ({t + 1}/{K}), amplituda mediana {np.median(amp):.3f}')
np.save(OS_ / 'metrics.npy', metr)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta = {'schema_version': 2, 'status': 'complete', 'time_axis': 'training_checkpoint', 'baseline_step': 0, 'steps': kroki, 'files': files, 'stats': stats,
        'point_count': N, 'image_count': NI, 'tokens_per_image': TOK, 'image_names': [im['image_name'] for im in images], 'images': images, 'frame_format': 'q16',
        'bounds': {'natural': np.column_stack([lo_n, hi_n]).tolist(), 'movement': np.column_stack([lo_m, hi_m]).tolist()}, 'norm_scale': norm_scale,
        'checkpoint_sha256': mp.get('checkpoint_sha256'), 'checkpoint_step': kroki[-1],
        'curves': dict(curves, label='Pewność i IoU szczytu mapy (eksport ruch60), średnio po 100 zdjęciach'),
        'metric_columns': ['confidence', 'iou', 'delta_l2', 'training_step', 'both90'], 'metric_steps': kroki,
        'metrics_file': {'name': 'metrics.npy', 'sha256': sha(OS_ / 'metrics.npy'), 'shape': [NI, K, 5], 'dtype': 'float64',
                         'unobserved_values': 'brak; delta_l2 = średnia amplituda drgań wokół węzła w zdjęciu'},
        'pca_file': {'name': 'initial_pca.npz', 'sha256': sha(OS_ / 'initial_pca.npz')},
        'uklad': {'nazwa': 'kryształ: położenia względne, norma = amplituda drgań wokół węzła', 'zrodlo': str(EKS),
                  'wzor': 'x = z − średnia tokenów zdjęcia (rp128); norma = |x(t) − średnia x po ±5 checkpointach|', 'baza_ekranu': 'PCA3 z kroków 0, 1094, 65640'},
        'coordinates': '(x − mean) @ components; image-major then token-major', 'norm': 'amplituda drgań wokół węzła w rp128',
        'interpolation': False, 'full_latent_states_persisted': False, 'inference_only': True, 'split': {'TRAIN': n_half, 'VALID': NI - n_half},
        'producer_provenance': {'skrypt': 'krysztal_wizualizacja.py', 'data': time.strftime('%Y-%m-%d'), 'maszyna': 'Eden', 'image_count': NI, 'tokens_per_image': TOK}}
(OS_ / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
log('oś gotowa', OS_, round(sum(f['bytes'] for f in files.values()) / 1e6), 'MB')

# ---------------------------------------------------------------- narzędzie kontra tło
amp_late = amp_sum / max(amp_n, 1)                                               # (NI, 600)
siatka = np.stack([np.arange(TOK) // W, np.arange(TOK) % W], 1)
w_ramce = np.zeros((NI, TOK), bool)
for j, im in enumerate(images):
    b = im.get('bbox_xyxy_480') or [0, 0, 0, 0]
    if im.get('n_gt', 0) and b[2] > b[0]:
        cx = (siatka[:, 1] + 0.5) * PL; cy = (siatka[:, 0] + 0.5) * PL
        w_ramce[j] = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
maj = w_ramce.any(1)
wzgl_amp = amp_late / amp_late.mean(1, keepdims=True)
narz = dict(zdjec_z_ramka=int(maj.sum()), amplituda_w_ramce=float(wzgl_amp[w_ramce].mean()), amplituda_poza=float(wzgl_amp[maj][~w_ramce[maj]].mean()),
            udzial_zdjec_ramka_ruchliwsza=float(np.mean([wzgl_amp[j][w_ramce[j]].mean() > wzgl_amp[j][~w_ramce[j]].mean() for j in np.where(maj)[0]])))
log('narzędzie kontra tło (amplituda względna, epoki 50–60):', narz)
json.dump(narz, open(PAN / 'NARZEDZIE_TLO.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- (B) panel
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
plt.rcParams.update({'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8',
                     'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
j0 = int(np.where(maj)[0][0])                                                    # pierwsze zdjęcie TRAIN z ramką
wiaz = [(i, i + 1) for i in range(TOK) if (i % W) < W - 1] + [(i, i + W) for i in range(TOK - W)]
kol = plt.cm.twilight((siatka[:, 1] / (W - 1)) * 0.85)
kol[:, :3] *= (0.55 + 0.45 * siatka[:, 0:1] / (H - 1))
fig = plt.figure(figsize=(20, 11.5), dpi=100)
gs = fig.add_gridspec(3, 5, height_ratios=[1.15, 1, 1])
for c_, kk in enumerate((0, 100, 1094, 10940, 65640)):
    a = fig.add_subplot(gs[0, c_])
    X = wzgl(kroki.index(kk))[j0]
    _, _, V = np.linalg.svd(X - X.mean(0), full_matrices=False); P2 = (X - X.mean(0)) @ V[:2].T
    if P2[np.argmax(siatka[:, 1]), 0] < P2[np.argmin(siatka[:, 1]), 0]:
        P2[:, 0] *= -1
    a.add_collection(LineCollection([[P2[i], P2[k]] for i, k in wiaz], colors='#8a8f99', linewidths=0.3, alpha=0.5))
    a.scatter(P2[:, 0], P2[:, 1], s=7, c=kol, zorder=3)
    if w_ramce[j0].any():
        a.scatter(P2[w_ramce[j0], 0], P2[w_ramce[j0], 1], s=16, facecolors='none', edgecolors='#ff4433', lw=0.7, zorder=4)
    a.set_title(f'krok {kk} (epoka {kk / EP:.1f})', fontsize=10); a.set_xticks([]); a.set_yticks([])
fig.text(0.01, 0.97, f'Siatka tokenów jednego zdjęcia w latencie (PCA2 z rp128); linie = sąsiednie patche na obrazie, kolor = kolumna × wiersz, czerwone obwódki = ramka narzędzia', fontsize=10)
a = fig.add_subplot(gs[1, 0])
im_ = a.imshow(amp_late[j0].reshape(H, W), cmap='inferno'); plt.colorbar(im_, ax=a, fraction=0.03)
b = images[j0]['bbox_xyxy_480']; a.add_patch(Rectangle((b[0] / PL - 0.5, b[1] / PL - 0.5), (b[2] - b[0]) / PL, (b[3] - b[1]) / PL, fill=False, ec='#4af0ff', lw=1.5))
a.set_title('amplituda drgań wokół węzła (epoki 50–60)', fontsize=10); a.set_xticks([]); a.set_yticks([])
a = fig.add_subplot(gs[1, 1])
im_ = a.imshow((amp_late / amp_late.mean(1, keepdims=True)).mean(0).reshape(H, W), cmap='inferno'); plt.colorbar(im_, ax=a, fraction=0.03)
a.set_title('amplituda względna, średnio po 100 zdjęciach', fontsize=10); a.set_xticks([]); a.set_yticks([])
a = fig.add_subplot(gs[1, 2])
Dg = cdist(siatka.astype(float), siatka.astype(float)); sas_s = (Dg <= np.sqrt(2) + 1e-9) & (Dg > 0)
mapa = np.zeros(TOK)
Xl = wzgl(K - 1)
for j in range(NI):
    Dm = cdist(Xl[j], Xl[j]); np.fill_diagonal(Dm, np.inf); N6 = np.argsort(Dm, 1)[:, :6]
    mapa += sas_s[np.arange(TOK)[:, None], N6].mean(1)
im_ = a.imshow((mapa / NI).reshape(H, W), cmap='viridis', vmin=0, vmax=1); plt.colorbar(im_, ax=a, fraction=0.03)
a.set_title('ile z 6 sąsiadów w latencie to sąsiednie patche (epoka 60)', fontsize=10); a.set_xticks([]); a.set_yticks([])
a = fig.add_subplot(gs[1, 3])
a.bar(['w ramce\nnarzędzia', 'poza ramką'], [narz['amplituda_w_ramce'], narz['amplituda_poza']], color=['#ff4433', '#8a8f99'])
a.set_ylabel('amplituda względna'); a.grid(True, axis='y', alpha=0.4)
a.set_title(f'narzędzie kontra tło: ruchliwsze w {narz["udzial_zdjec_ramka_ruchliwsza"]:.0%} zdjęć', fontsize=10)
a = fig.add_subplot(gs[1, 4])
try:
    pz = json.load(open(WYN / 'pozycja_patchy' / 'POZYCJA_PATCHY.json', encoding='utf-8'))['punkty']
    x_ = [max(p['krok'], 1) for p in pz]
    for kl, c, et in (('r2_szablon_pozycji_VALID', '#ffffff', 'numer patcha (R²)'), ('sasiedzi_to_sasiednie_patche', '#ffd166', 'sąsiedzi = sąsiednie patche'),
                      ('jaccard_z_krokiem_0', '#ff4433', 'sąsiedzi z kroku 0')):
        a.plot(x_, [p[kl] for p in pz], 'o-', ms=3, color=c, label=et)
    a.axhline(8 / 599, color='#8a8f99', ls=':', lw=1); a.set_xscale('log'); a.set_ylim(0, 1.02); a.legend(fontsize=7); a.grid(True, alpha=0.4)
except Exception as e:  # noqa: BLE001
    a.text(0.1, 0.5, f'brak POZYCJA_PATCHY: {e}', fontsize=7)
a.set_xlabel('krok'); a.set_title('węzły siatki przez 60 epok', fontsize=10)
a = fig.add_subplot(gs[2, 0:2])
try:
    wz = json.load(open(WYN / 'wiazki_koloidy' / 'WIAZKI.json', encoding='utf-8'))
    nm = list(wz['etapy']); xx = np.arange(len(nm))
    for L, c in zip(('1', '10', '40'), ('#ffffff', '#ffd166', '#ff4433')):
        y = [wz['etapy'][e]['lagi'].get(L, {}).get('udzial_wiazek', np.nan) for e in nm]
        kr_ = [wz['etapy'][e]['lagi'].get(L, {}).get('krokow', 0) for e in nm]
        a.plot(xx, y, 'o-', color=c, label=f'opóźnienie {L} checkpointów (~{max(kr_)} kroków)')
    a.axhline(0.5, color='#8a8f99', ls='--', lw=1); a.set_xticks(xx); a.set_xticklabels(nm, fontsize=8); a.set_ylim(0, 1); a.legend(fontsize=7); a.grid(True, alpha=0.4)
    a.set_ylabel('udział obrotu wiązek W'); a.set_title(f'kto rusza tokenem: obrót osi odczytu (wiązki) kontra ruch atomów (kulki); cała droga e1→e60: {wz["cala_droga_e1_e60"]["udzial_wiazek"]:.2f}', fontsize=10)
except Exception as e:  # noqa: BLE001
    a.text(0.1, 0.5, f'brak WIAZKI: {e}', fontsize=7)
a = fig.add_subplot(gs[2, 2:4])
try:
    r6 = json.load(open(WYN / 'ruch60' / 'RUCH60.json', encoding='utf-8'))['etapy']
    nm = list(r6); xx = np.arange(len(nm))
    a.plot(xx, [r6[e]['sygnal']['TRAIN']['spearman_ruch_iou'] for e in nm], 'o-', color='#ffffff', label='Spearman ruch–IoU, TRAIN')
    a.plot(xx, [r6[e]['sygnal']['VALID']['spearman_ruch_iou'] for e in nm], 's--', color='#4af0ff', label='VALID')
    a.plot(xx, [r6[e]['przesuniecie_1_do_dshell'] / 3 for e in nm], '^:', color='#ffd166', label='przesunięcie na 100 kroków / odstęp sąsiadów (÷3)')
    a.axhline(0, color='#8a8f99', lw=0.8); a.set_xticks(xx); a.set_xticklabels([f'ep. {e}' for e in nm], fontsize=8); a.legend(fontsize=7); a.grid(True, alpha=0.4)
except Exception as e:  # noqa: BLE001
    a.text(0.1, 0.5, f'brak RUCH60: {e}', fontsize=7)
a.set_title('miękkość kryształu zdjęcia a jego błąd detekcji', fontsize=10)
a = fig.add_subplot(gs[2, 4])
pr = amp_late.mean(1); io = oceny[[t for t in range(K) if kroki[t] >= 50 * EP]].mean(0)[:, 1]
a.scatter(pr[:n_half], io[:n_half], s=14, color='#ffffff', label='TRAIN'); a.scatter(pr[n_half:], io[n_half:], s=14, color='#4af0ff', label='VALID')
a.set_xlabel('średnia amplituda drgań zdjęcia (ep. 50–60)'); a.set_ylabel('IoU (ep. 50–60)'); a.legend(fontsize=7); a.grid(True, alpha=0.4)
a.set_title('każdy punkt = zdjęcie', fontsize=10)
fig.suptitle('Kryształ tokenów małego ViT: węzły = pozycje patchy, drgania wokół węzłów, odkształcenie przez treść obrazu', fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(PAN / 'krysztal.png')
log('zapisano panel', PAN / 'krysztal.png')
