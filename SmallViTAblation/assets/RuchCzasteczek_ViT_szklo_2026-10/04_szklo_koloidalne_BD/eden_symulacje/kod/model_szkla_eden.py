"""Model szkła koloidalnego na Edenie: dynamika browna N polidyspersyjnych miękkich kul harmonicznych (ε = σ = 1),
pudło periodyczne 3D, B gęstości naraz na jednej karcie (lista Verleta, koszt kroku = narzut kerneli).
Tryby: stala (stała T, skan φ) i stygnace (T(t) = T0 (1 + t/t_zal)^-0,79 do t1, potem ∝ t^-0,30, jak energia drgań
tokenów ViT przy stałym lr). Odciski palca jak dla danych ViT (odciski.py), eksport osi czasu w formacie aplikacji.

Uruchomienie: python model_szkla_eden.py --tryb=stala --out=... [--phi=0.56,...] [--N=4096] [--T=3e-4] [--dt=0.3]
              [--n_komp=300000 --n_rown=1000000 --n_rec=10000000 --co=200 --seed=100]
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import odciski as O


def _arg(nazwa, domyslna, typ=float):
    for a in sys.argv:
        if a.startswith(f'--{nazwa}='):
            return typ(a.split('=', 1)[1])
    return domyslna


TRYB = _arg('tryb', 'stala', str)
OUT = Path(_arg('out', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wyniki', TRYB), str))
N = _arg('N', 4096, int)
T_STALA = _arg('T', 3e-4)
DT = _arg('dt', 0.3)
CO = _arg('co', 200, int)
N_KOMP, N_ROWN, N_REC = _arg('n_komp', 300_000, int), _arg('n_rown', 1_000_000, int), _arg('n_rec', 10_000_000, int)
PHI = tuple(float(x) for x in _arg('phi', '0.56,0.58,0.60,0.61,0.62,0.63,0.64,0.66', str).split(','))
SEED = _arg('seed', 100, int)
T0_STYG, T_ZAL, T1 = _arg('T0', 0.01), _arg('tzal', 24_000, int), _arg('t1', 2_000_000, int)
CO_OS = _arg('co_os', 25, int)                  # klatka osi aplikacji co tyle rekordów
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(*a):
    print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
    with open(OUT / 'log.txt', 'a', encoding='utf-8') as f:
        print(*a, f'[{time.time() - t0:.0f} s]', file=f)


class Pudla:
    def __init__(self, phis, seed):
        self.B = B = len(phis)
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.sig = 0.85 + 0.30 * torch.rand(B, N, generator=self.g, device=dev)
        V = (np.pi / 6) * (self.sig ** 3).sum(1)
        self.L_cel = (V / torch.tensor(phis, device=dev, dtype=torch.float32)) ** (1 / 3)
        self.L = 1.6 * self.L_cel
        self.x = self.L[:, None, None] * torch.rand(B, N, 3, generator=self.g, device=dev)
        self.x_unw = self.x.clone()
        self.r_cut = float(self.sig.max())
        self.skin = 1.0
        self.lista = None
        self.odswiezen = 0

    def zbuduj_liste(self):
        listy, sij = [], []
        K = 8
        for b in range(self.B):                                   # po jednym układzie, żeby nie trzymać (B, N, N, 3)
            dx = self.x[b][:, None, :] - self.x[b][None, :, :]
            dx -= self.L[b] * torch.round(dx / self.L[b])
            r = dx.norm(dim=2)
            r.fill_diagonal_(float('inf'))
            K = max(K, int((r < self.r_cut + self.skin).sum(1).max().item()))
            listy.append(torch.argsort(r, dim=1))
            del dx, r
        self.lista = torch.stack([l[:, :K] for l in listy])
        sj = torch.gather(self.sig, 1, self.lista.reshape(self.B, -1)).reshape(self.B, N, K)
        self.sij_l = 0.5 * (self.sig[:, :, None] + sj)
        self.x_lista = self.x_unw.clone()
        self.odswiezen += 1

    def sily(self):
        if self.lista is None or float((self.x_unw - self.x_lista).norm(dim=2).max()) > 0.5 * self.skin:
            self.zbuduj_liste()
        K = self.lista.shape[2]
        xj = torch.gather(self.x, 1, self.lista.reshape(self.B, -1, 1).expand(-1, -1, 3)).reshape(self.B, N, K, 3)
        dx = self.x[:, :, None, :] - xj
        Lb = self.L[:, None, None, None]
        dx -= Lb * torch.round(dx / Lb)
        r = dx.norm(dim=3).clamp_min(1e-9)
        ov = (1 - r / self.sij_l).clamp_min(0)
        return ((ov / self.sij_l / r)[..., None] * dx).sum(2)

    def krok(self, T, dt):
        F = self.sily()
        amp = float(np.sqrt(2 * T * dt))
        d = F * dt + amp * torch.randn(self.B, N, 3, generator=self.g, device=dev)
        self.x += d
        self.x_unw += d
        self.x -= self.L[:, None, None] * torch.floor(self.x / self.L[:, None, None])

    def sciskaj(self, n, T):
        L0 = self.L.clone()
        for i in range(n):
            self.L = L0 + (self.L_cel - L0) * (i + 1) / n
            self.x -= self.L[:, None, None] * torch.floor(self.x / self.L[:, None, None])
            if i % 20 == 0:
                self.lista = None
            self.krok(T, DT)
        self.L = self.L_cel.clone()
        self.x -= self.L[:, None, None] * torch.floor(self.x / self.L[:, None, None])
        self.x_unw = self.x.clone()
        self.lista = None


def T_styg(t):
    if t <= T1:
        return T0_STYG * (1 + t / T_ZAL) ** (-0.79)
    return T0_STYG * (1 + T1 / T_ZAL) ** (-0.79) * (t / T1) ** (-0.30)


T_fun = (lambda t: T_STALA) if TRYB == 'stala' else T_styg
log(f'tryb {TRYB}, N {N}, φ {PHI}, T {T_STALA}, dt {DT}, kroki {N_KOMP}/{N_ROWN}/{N_REC}, co {CO}, urządzenie {dev}',
    torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
p = Pudla(PHI, SEED)
p.sciskaj(N_KOMP, T_fun(0))
for i in range(N_ROWN):
    p.krok(T_fun(0), DT)
    if i % 200_000 == 0:
        log(f'równoważenie {i}/{N_ROWN}')
log('ściśnięte i zrównoważone, L =', [round(float(v), 3) for v in p.L], 'K listy', p.lista.shape[2] if p.lista is not None else None)
n_rec = N_REC // CO
P = np.empty((n_rec, p.B, N, 3), np.float32)
Pw_co = 20
Pw = np.empty((n_rec // Pw_co + 1, p.B, N, 3), np.float32)
Ts = np.empty(n_rec, np.float64)
for i in range(N_REC):
    T = T_fun(i)
    p.krok(T, DT)
    if i % CO == 0:
        j = i // CO
        P[j] = p.x_unw.cpu().numpy()
        if j % Pw_co == 0:
            Pw[j // Pw_co] = p.x.cpu().numpy()
        Ts[j] = T
    if i % 500_000 == 0:
        log(f'zapis {i}/{N_REC}, T = {T:.2e}, odświeżeń listy {p.odswiezen}')
log('symulacja gotowa, rekordów', n_rec)
L = p.L.cpu().numpy()
sig = p.sig.cpu().numpy()
del p
torch.cuda.empty_cache()
np.savez(OUT / 'przebieg.npz', P=P[::5].astype(np.float16), Pw=Pw.astype(np.float16), Ts=Ts, L=L, sig=sig.astype(np.float16), phi=np.array(PHI),
         co=CO, dt=DT, T=T_STALA, uwaga='P co 5 rekordów (float16, nieowinięte), Pw co 20 rekordów (owinięte)')
log('przebieg.npz zapisany')

# ---------------------------------------------------------------- odciski
Q16 = 65535.0


def eksport_osi(Pb, nazwa, opis, co=CO_OS):
    out = OUT / 'osie' / nazwa
    out.mkdir(parents=True, exist_ok=True)
    files, stats, curves = {}, {}, {k: [] for k in ('steps', 'confidence', 'iou', 'both90', 'real_step')}
    lo_n, hi_n, lo_m, hi_m = np.full(3, np.inf), np.full(3, -np.inf), np.full(3, np.inf), np.full(3, -np.inf)
    nmax = 0.0
    X0 = Pb[0] - Pb[0].mean(0)
    steps = list(range(0, len(Pb), co))
    for s in steps:
        xyz = (Pb[s] - Pb[s].mean(0)).astype(np.float32)
        norma = np.linalg.norm(xyz - X0, axis=1).astype(np.float32)
        frame = np.concatenate([xyz, norma[:, None]], 1)
        lo = frame[:, :3].min(0)
        span = (frame[:, :3].max(0).astype(np.float64) - lo).astype(np.float32)
        nm = np.float32(norma.max())
        q = np.empty(frame.shape, '<u2')
        sk = np.where(span > 0, Q16 / np.maximum(span.astype(np.float64), 1e-300), 0.)
        q[:, :3] = np.clip(np.rint((frame[:, :3].astype(np.float64) - lo) * sk), 0, Q16)
        q[:, 3] = np.clip(np.rint(norma.astype(np.float64) / float(nm) * Q16), 0, Q16) if nm > 0 else 0
        raw = q.tobytes()
        (out / f'step_{s:04d}.q16').write_bytes(raw)
        files[str(s)] = {'name': f'step_{s:04d}.q16', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'format': 'q16',
                         'quant': {'lo': [float(v) for v in lo], 'span': [float(v) for v in span], 'norm_max': float(nm)}}
        stats[str(s)] = dict(confidence=0.0, iou=0.0, both90=0, norm_median=float(np.median(norma)), norm_max=float(nm), real_step=int(s))
        for k in ('confidence', 'iou', 'both90', 'real_step'):
            curves[k].append(stats[str(s)][k])
        curves['steps'].append(int(s))
        lo_n, hi_n = np.minimum(lo_n, xyz.min(0)), np.maximum(hi_n, xyz.max(0))
        mv = xyz - X0
        lo_m, hi_m = np.minimum(lo_m, mv.min(0)), np.maximum(hi_m, mv.max(0))
        nmax = max(nmax, float(nm))
    meta = {'schema_version': 2, 'status': 'complete', 'steps': steps, 'files': files, 'stats': stats, 'point_count': N, 'image_count': 0,
            'frame_format': 'q16', 'bounds': {'natural': np.column_stack([lo_n, hi_n]).tolist(), 'movement': np.column_stack([lo_m, hi_m]).tolist()},
            'norm_scale': nmax, 'curves': dict(curves, label='symulacja: brak detekcji (0)'), 'baseline_step': 0,
            'coordinates': 'położenia nieowinięte minus środek masy, jednostka σ', 'norm': 'L2(x(t) − x(0))', 'interpolation': False,
            'uklad': opis, 'producer_provenance': {'skrypt': 'model_szkla_eden.py', 'data': '2026-10-10', 'maszyna': 'Eden'}}
    (out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')


wyniki = {}
tau_B_rek = 1 / T_STALA / DT / CO
taus = np.unique(np.round(np.logspace(0, np.log10(min(n_rec - 1, 20000) * 0.9), 40)).astype(int))
def zapisz_wyniki():
    json.dump(dict(opis=__doc__, tryb=TRYB, N=N, dt=DT, co=CO, kroki=dict(kompresja=N_KOMP, rownowaga=N_ROWN, zapis=N_REC), phi=list(PHI), seed=SEED,
                   biegi=wyniki, czas_s=round(time.time() - t0)), open(OUT / 'MODEL_SZKLA_EDEN.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


kroki_Pw = np.arange(0, n_rec, Pw_co)[:len(Pw)]
for b, phi in enumerate(PHI):
    try:
        eksport_osi(P[:, b], f'szklo_BD_eden_{TRYB}_phi_{phi}',
                    dict(model='dynamika browna, miękkie kule harmoniczne, polidyspersja 8,7 %, Eden', nazwa=f'szkło BD Eden, φ={phi}, {TRYB}', phi=phi,
                         T=T_STALA if TRYB == 'stala' else dict(T0=T0_STYG, t_zal=T_ZAL, t1=T1), dt=DT, rekord_co_krokow=CO, klatka_co_rekordow=CO_OS, N=N))
        okno = (n_rec // 5, n_rec - 1) if TRYB == 'stala' else (n_rec // 10, n_rec - 1)
        od_w = (okno[0] // Pw_co) * Pw_co
        nn, dnn, dsh = O.knn_3d_pudlo(Pw[od_w // Pw_co, b], float(L[b]), 6)
        pary = np.stack([np.arange(N), nn[:, 0]], 1)
        tw = (okno[0],) if TRYB == 'stala' else tuple(int(x) for x in np.unique(np.round(np.logspace(np.log10(max(20, n_rec // 500)), np.log10(n_rec // 2), 9))))
        kr = (okno[0], okno[1] - 30) if TRYB == 'stala' else tuple(int(x) for x in np.linspace(okno[0], n_rec - 40, 6))
        w = O.odciski(P[:, b], np.arange(n_rec), okno, float(np.median(dsh)), float(np.median(dnn)), taus=taus, pary=pary, tw_lista=tw, krotkie_tw=kr,
                      odejmij_pole=True, nazwa=f'BD {TRYB} φ={phi}')
        trw = O.trwalosc_sasiedztwa(Pw[:, b], kroki_Pw, od_w, [od_w + k for k in (20, 100, 300, 1000, 3000, 10000) if od_w + k < n_rec], L=float(L[b]))
        w['trwalosc'] = {int(k - od_w): v for k, v in trw.items()}
        w['karta'], w['spelnione'] = O.karta(w, w['trwalosc'].get(1000))
        w.update(phi=phi, T=T_STALA, L=float(L[b]), tau_B_w_rekordach=tau_B_rek, tryb=TRYB)
        if TRYB != 'stala':
            w['T_rec'] = dict(rekord=[int(x) for x in np.arange(n_rec)[::max(1, n_rec // 200)]], T=[float(x) for x in Ts[::max(1, n_rec // 200)]])
        wyniki[f'phi_{phi}'] = w
        log(f'φ={phi}: Lindemann_shell {w["lindemann_shell"]:.3f} c {w["c_plateau"]:.3f} α plateau {w["alfa_plateau"]:.2f} a2 {w["a2_szczyt"]:.2f}@{w["a2_szczyt_tau"]} '
            f'f_q {w["f_q"]:.2f} β {w["beta"]:.2f} τ_1e {w["tau_1e"]} ucieczki@10 {w["uciekajacy_tau10"]:.3f} trwałość {w["trwalosc"]} μ {w["mu"]} karta {w["spelnione"]}')
    except Exception as e:                                        # jedna gęstość nie może zepsuć reszty
        import traceback
        log(f'φ={phi}: BŁĄD analizy: {type(e).__name__}: {e}')
        traceback.print_exc()
        wyniki[f'phi_{phi}'] = dict(phi=phi, blad=f'{type(e).__name__}: {e}')
    zapisz_wyniki()
log('GOTOWE')
