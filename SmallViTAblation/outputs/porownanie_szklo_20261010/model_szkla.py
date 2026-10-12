"""Model szkła koloidalnego dopasowany do ruchu tokenów ViT: dynamika browna N polidyspersyjnych miękkich kul
(odpychanie harmoniczne U = ε/2 (1 − r/σ_ij)², model Duriana / O'Herna / Berthiera i Wittena) w pudle periodycznym 3D,
na karcie (torch). Ruch przetłumiony jak w zawiesinie: dx = F dt + sqrt(2 T dt) ξ, γ = 1, D0 = T.
Kilka ułamków objętości φ liczy się naraz jako paczka układów (koszt kroku to narzut kerneli, nie arytmetyka).

Tryby:
  (a) stała temperatura, skan φ — koloid Weeksa; wybór φ, dla którego odciski (rozmiar klatki względem odstępu
      sąsiadów i antykorelacja c) są najbliższe tokenom ViT w układzie klatki (UKLAD_SZKLA.json, S1);
  (b) stygnięcie własne T(t) ∝ (1 + t/t0)^-0,79, potem ∝ t^-0,30 — jak energia drgań tokenów przy stałym lr;
      to jest „szkło, które samo ostygło”.
Te same odciski co dla danych (odciski.py). Eksport dwóch osi czasu do formatu aplikacji (manifest v2, q16).
Uruchomienie: F:/Python/3.13/python.exe model_szkla.py [--szybko | --test | --skan]
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

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
OSIE = Path(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui')
SZYBKO = '--szybko' in sys.argv
TEST = '--test' in sys.argv
SKAN = '--skan' in sys.argv
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 2048


def _arg(nazwa, domyslna, typ=float):
    for a in sys.argv:
        if a.startswith(f'--{nazwa}='):
            return typ(a.split('=', 1)[1])
    return domyslna


# ε = σ = 1. Skan przy T = 3e-3 (dt 0,15) dał ciecz do φ = 0,68: energia cieplna 15× większa od sprężystej kontaktów.
# Dlatego T = 3e-4 (przekrycie termiczne sqrt(T) ≈ 0,017 σ, prawie twarde kule) i dt = 0,3 (błąd wariancji kontaktu ≈ 18 %).
T_STALA = _arg('T', 3e-4)
DT = _arg('dt', 0.3)
CO = 100                       # zapis co CO kroków = 1 „rekord”; τ_B = σ²/D0 = 1/T czasu = 1/(T·dt·CO) rekordów
N_KOMP, N_ROWN, N_REC = ((2_000, 2_000, 30_000) if TEST else (30_000, 30_000, 150_000) if SKAN else
                         (50_000, 50_000, 200_000) if SZYBKO else (150_000, 300_000, 700_000))
PHI = tuple(float(x) for x in _arg('phi', '0.60,0.66' if TEST else '0.56,0.60,0.64,0.68', str).split(','))
T0_STYG, T_ZAL = _arg('T0', 0.01), _arg('tzal', 2000, int)
if TEST:
    OSIE = OSIE / '_test'
t0 = time.time()


def log(*a):
    print(*a, f'[{time.time() - t0:.0f} s]', flush=True)


class Pudla:
    """B układów naraz: x (B, N, 3), własne L i średnice; lista Verleta (B, N, K)."""

    def __init__(self, phis, seed):
        self.B = B = len(phis)
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.sig = 0.85 + 0.30 * torch.rand(B, N, generator=self.g, device=dev)           # polidyspersja ≈ 8,7 %
        V = (np.pi / 6) * (self.sig ** 3).sum(1)
        self.L_cel = (V / torch.tensor(phis, device=dev, dtype=torch.float32)) ** (1 / 3)
        self.L = 1.6 * self.L_cel
        self.x = self.L[:, None, None] * torch.rand(B, N, 3, generator=self.g, device=dev)
        self.x_unw = self.x.clone()
        self.oko = torch.eye(N, dtype=torch.bool, device=dev)
        self.r_cut = float(self.sig.max())
        self.skin = 1.0
        self.lista = None
        self.odswiezen = 0

    def _L(self):
        return self.L[:, None, None, None]

    def zbuduj_liste(self):
        dx = self.x[:, :, None, :] - self.x[:, None, :, :]
        dx -= self._L() * torch.round(dx / self._L())
        r = dx.norm(dim=3)
        r.masked_fill_(self.oko, float('inf'))
        K = max(8, int((r < self.r_cut + self.skin).sum(2).max().item()))
        self.lista = torch.argsort(r, dim=2)[:, :, :K]
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
        dx -= self._L() * torch.round(dx / self._L())
        r = dx.norm(dim=3).clamp_min(1e-9)
        ov = (1 - r / self.sij_l).clamp_min(0)
        return ((ov / self.sij_l / r)[..., None] * dx).sum(2)

    def krok(self, T, dt):
        F = self.sily()
        amp = torch.as_tensor(np.sqrt(2 * np.asarray(T, dtype=np.float64) * dt), device=dev, dtype=torch.float32)
        amp = amp.reshape(-1, 1, 1) if amp.ndim else amp
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


def bieg(phis, T_fun, n_rec, seed, nazwa):
    """Zwraca P_unw (n_rec/CO, B, N, 3), P_wrap, L (B,), T_rec."""
    p = Pudla(phis, seed)
    p.sciskaj(N_KOMP, T_fun(0))
    for i in range(N_ROWN):
        p.krok(T_fun(0), DT)
    log(nazwa, 'ściśnięte i zrównoważone, L =', [round(float(v), 3) for v in p.L], 'K listy', p.lista.shape[2] if p.lista is not None else None)
    P, Pw, Ts = [], [], []
    for i in range(n_rec):
        T = T_fun(i)
        p.krok(T, DT)
        if i % CO == 0:
            P.append(p.x_unw.cpu().numpy().copy())
            Pw.append(p.x.cpu().numpy().copy())
            Ts.append(float(T))
        if i % 200_000 == 0:
            log(nazwa, f'krok {i}/{n_rec}, T = {float(T):.2e}, odświeżeń listy {p.odswiezen}')
    return np.stack(P), np.stack(Pw), p.L.cpu().numpy(), np.array(Ts)


def odciski_biegu(P, Pw, L, okno, nazwa, tw_lista, krotkie_tw):
    kroki = np.arange(len(P))
    nn, dnn, dsh = O.knn_3d_pudlo(Pw[okno[0]], L, 6)
    pary = np.stack([np.arange(N), nn[:, 0]], 1)
    w = O.odciski(P, kroki, okno, float(np.median(dsh)), float(np.median(dnn)), pary=pary, tw_lista=tw_lista, krotkie_tw=krotkie_tw,
                  odejmij_pole=True, nazwa=nazwa)
    trw = O.trwalosc_sasiedztwa(Pw, kroki, okno[0], [okno[0] + k for k in (10, 100, 300, 1000, 3000) if okno[0] + k < len(P)], L=L)
    w['trwalosc'] = {int(k - okno[0]): v for k, v in trw.items()}
    w['karta'], w['spelnione'] = O.karta(w, w['trwalosc'].get(1000))
    return w


Q16 = 65535.0


def eksport_osi(P, nazwa, opis, co=10):
    out = OSIE / nazwa
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob('step_*.q16'):
        f.unlink()
    files, stats, curves = {}, {}, {k: [] for k in ('steps', 'confidence', 'iou', 'both90', 'real_step')}
    lo_n, hi_n, lo_m, hi_m = np.full(3, np.inf), np.full(3, -np.inf), np.full(3, np.inf), np.full(3, -np.inf)
    nmax = 0.0
    X0 = P[0] - P[0].mean(0)
    steps = list(range(0, len(P), co))
    for s in steps:
        xyz = (P[s] - P[s].mean(0)).astype(np.float32)
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
            'uklad': opis, 'producer_provenance': {'skrypt': 'outputs/porownanie_szklo_20261010/model_szkla.py', 'data': '2026-10-10'}}
    (out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    log('oś czasu zapisana:', out, len(steps), 'kroków')


def cel_vit():
    try:
        u = json.load(open(OUT / 'UKLAD_SZKLA.json', encoding='utf-8'))['etapy']['S1']
        return dict(lindemann_shell=u['lindemann_shell'], c_plateau=u['c_plateau'], a2_plateau=u['a2_plateau'])
    except Exception:
        return None


def odleglosc_do_celu(w, cel):
    return float(abs(np.log(max(w['lindemann_shell'], 1e-6) / cel['lindemann_shell'])) + abs(w['c_plateau'] - cel['c_plateau']))


# ---------------------------------------------------------------- (a) stała temperatura, wszystkie φ naraz
wyniki = {}
P, Pw, L, Ts = bieg(PHI, lambda t: T_STALA, N_REC, 100, 'stała T')
n_rec = len(P)
okno_a = (n_rec // 5, n_rec - 1)
for b, phi in enumerate(PHI):
    w = odciski_biegu(P[:, b], Pw[:, b], float(L[b]), okno_a, f'BD φ={phi}, T={T_STALA}', tw_lista=(okno_a[0],), krotkie_tw=(okno_a[0], okno_a[1] - 30))
    w.update(phi=phi, T=T_STALA, L=float(L[b]), tau_B_w_rekordach=1 / T_STALA / DT / CO)
    wyniki[f'phi_{phi}'] = w
    i_b = int(np.argmin(abs(np.array(w['taus']) - w['tau_B_w_rekordach'])))
    log(f'φ={phi}: MSD(τ_B)={w["msd"][i_b]:.4f} Lindemann_shell {w["lindemann_shell"]:.3f} c {w["c_plateau"]:.3f} α plateau {w["alfa_plateau"]:.2f} '
        f'a2 {w["a2_szczyt"]:.2f}@{w["a2_szczyt_tau"]} ucieczki@10 {w["uciekajacy_tau10"]:.3f} trwałość {w["trwalosc"]} karta {w["spelnione"]}')
if SKAN:
    sys.exit(0)
cel = cel_vit()
najblizszy = min(wyniki, key=lambda k: odleglosc_do_celu(wyniki[k], cel)) if cel else f'phi_{PHI[len(PHI) // 2]}'
log('cel ViT S1:', cel, '→ najbliższy bieg', najblizszy)
P_a = P[:, PHI.index(wyniki[najblizszy]['phi'])].copy()
del P, Pw

# ---------------------------------------------------------------- (b) stygnięcie własne, wszystkie φ naraz
T0, t_zal, t1 = T0_STYG, T_ZAL, N_REC // 5           # T(t1) ≈ T0 (1 + t1/t_zal)^-0,79 ≈ T_STALA, potem wolniej t^-0,30


def T_styg(t):
    if t <= t1:
        return T0 * (1 + t / t_zal) ** (-0.79)
    return T0 * (1 + t1 / t_zal) ** (-0.79) * (t / t1) ** (-0.30)


P, Pw, L, Ts = bieg(PHI, T_styg, N_REC, 7, 'stygnięcie')
n_rec = len(P)
okno_b = (n_rec // 10, n_rec - 1)
tw_b = tuple(int(x) for x in np.unique(np.round(np.logspace(np.log10(max(20, n_rec // 200)), np.log10(n_rec // 2), 7))))
for b, phi in enumerate(PHI):
    w = odciski_biegu(P[:, b], Pw[:, b], float(L[b]), okno_b, f'BD stygnące φ={phi}', tw_lista=tw_b,
                      krotkie_tw=tuple(int(x) for x in np.linspace(okno_b[0], n_rec - 40, 5)))
    w.update(phi=phi, T0=T0, t_zal=t_zal, t1=t1, L=float(L[b]),
             T_rec=dict(rekord=[int(x) for x in np.arange(n_rec)[::max(1, n_rec // 100)]], T=[float(x) for x in Ts[::max(1, n_rec // 100)]]))
    wyniki[f'stygnace_phi_{phi}'] = w
    log(f'stygnące φ={phi}: μ {w["mu"]} krótkie {w["stosunek_krotkich"]:.3f} Lindemann {w["lindemann_shell"]:.3f} c {w["c_plateau"]:.3f} karta {w["spelnione"]}')
styg_wybrany = f'stygnace_phi_{wyniki[najblizszy]["phi"]}'
P_b = P[:, PHI.index(wyniki[najblizszy]['phi'])].copy()
del P, Pw

json.dump(dict(opis=__doc__, N=N, dt=DT, co=CO, kroki=dict(kompresja=N_KOMP, rownowaga=N_ROWN, zapis=N_REC), phi=list(PHI), cel_ViT=cel,
               najblizszy=najblizszy, stygnace_wybrane=styg_wybrany, biegi=wyniki, czas_s=round(time.time() - t0)),
          open(OUT / 'MODEL_SZKLA.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- eksport osi do aplikacji
eksport_osi(P_a, f'20261010T_szklo_BD_{najblizszy}',
            dict(model='dynamika browna, miękkie kule harmoniczne, polidyspersja 8,7 %', nazwa=f'szkło koloidalne BD, φ={wyniki[najblizszy]["phi"]}, stała T',
                 phi=wyniki[najblizszy]['phi'], T=T_STALA, dt=DT, rekord_co_krokow=CO, klatka_co_rekordow=10, jednostka='σ (średnia średnica)', N=N))
eksport_osi(P_b, f'20261010T_szklo_BD_{styg_wybrany}',
            dict(model='dynamika browna, miękkie kule, stygnięcie własne T ∝ (1+t/t0)^-0,79 potem t^-0,30',
                 nazwa=f'szkło koloidalne BD, φ={wyniki[najblizszy]["phi"]}, stygnące jak tokeny', phi=wyniki[najblizszy]['phi'], T0=T0, t_zal=t_zal, t1=t1,
                 dt=DT, rekord_co_krokow=CO, klatka_co_rekordow=10, jednostka='σ', N=N))

# ---------------------------------------------------------------- rysunek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
vit = None
try:
    vit = json.load(open(OUT / 'UKLAD_SZKLA.json', encoding='utf-8'))['etapy']
except Exception:
    pass
fig, ax = plt.subplots(2, 3, figsize=(19, 11), dpi=100)
kol = matplotlib.colormaps['plasma'](np.linspace(0.15, 0.85, len(PHI)))
stale = [(k, v) for k, v in wyniki.items() if not k.startswith('stygnace')]
for (k, w), c in zip(stale, kol):
    ds, tb = w['d_shell'], w['tau_B_w_rekordach']
    ax[0, 0].loglog(np.array(w['taus']) / tb, np.array(w['msd']) / ds ** 2, color=c, lw=2.2 if k == najblizszy else 1.3, label=f"BD φ={w['phi']}")
    ax[0, 1].semilogx(np.array(w['taus']) / tb, w['a2'], color=c, lw=1.6, label=f"BD φ={w['phi']}")
    ax[0, 2].semilogx(np.array(w['taus']) / tb, w['fs'], color=c, lw=1.6, label=f"BD φ={w['phi']}")
    cx = sorted(w['c_tau'])
    ax[1, 0].semilogx(np.array(cx) / tb, [w['c_tau'][x] for x in cx], 'o-', color=c, lw=1.2, ms=4, label=f"BD φ={w['phi']}")
skala = 1.0 / 100.0
if vit:
    for k, c in (('S1', '#ffd166'), ('S4', '#ff4433')):
        w = vit[k]
        ds = w['d_shell']
        ax[0, 0].loglog(np.array(w['taus']) * skala, np.array(w['msd']) / ds ** 2, '--', color=c, lw=2, label=f"ViT {w['nazwa']} (τ/100)")
        ax[0, 1].semilogx(np.array(w['taus']) * skala, w['a2'], '--', color=c, lw=2, label=f"ViT {w['nazwa']}")
        ax[0, 2].semilogx(np.array(w['taus']) * skala, w['fs'], '--', color=c, lw=2, label=f"ViT {w['nazwa']}")
        cx = sorted(int(x) for x in w['c_tau'])
        ax[1, 0].semilogx(np.array(cx) * skala, [w['c_tau'][str(x)] for x in cx], 's--', color=c, lw=1.4, ms=4, label=f"ViT {w['nazwa']}")
ax[0, 0].axhline(0.15 ** 2, color='#ff4433', ls=':', lw=0.8)
ax[0, 0].text(1e-2, 0.15 ** 2 * 1.2, '(0,15 d_shell)²', fontsize=9, color='#ff4433')
ax[0, 0].set(xlabel='τ / τ_B (BD) albo τ/100 kroków (ViT)', ylabel='MSD / d_shell²', title='MSD w jednostkach odstępu sąsiadów')
ax[0, 1].set(xlabel='τ / τ_B', ylabel='α₂', title='parametr niegaussowski')
ax[0, 2].axhline(np.exp(-1), color='w', ls=':', lw=0.8)
ax[0, 2].set(xlabel='τ / τ_B', ylabel='Fs(q, τ)', title='funkcja rozpraszania, q = 2π/d_shell', ylim=(-0.05, 1.05))
ax[1, 0].axhspan(0.25, 0.55, color='#ffd166', alpha=0.12)
ax[1, 0].set(xlabel='τ / τ_B', ylabel='c(τ)', title='antykorelacja kolejnych przesunięć')
w = wyniki[styg_wybrany]
a = ax[1, 1]
for tau, c in ((10, '#4cd964'), (100, '#ffd166'), (1000, '#ff4433')):
    pts = sorted((tw, v[tau]) for tw, v in w['starzenie'].items() if tau in v)
    if pts:
        a.loglog([p[0] for p in pts], [p[1] / w['d_shell'] ** 2 for p in pts], 'o-', color=c, lw=1.6,
                 label=f'BD stygnące φ={w["phi"]} MSD(τ={tau}; t_w)' + (f', μ={w["mu"][tau]:.2f}' if tau in w['mu'] else ''))
if vit:
    v1 = vit['S1']
    pts = sorted((tw, v[10]) for tw, v in v1['starzenie'].items() if 10 in v)
    a.loglog([p[0] * skala for p in pts], [p[1] / v1['d_shell'] ** 2 for p in pts], 's--', color='#4a8cff', lw=1.4, label='ViT S1 MSD(τ=10; t_w/100)')
a.plot(w['T_rec']['rekord'], np.array(w['T_rec']['T']) / w['T_rec']['T'][0] * 0.3, color='#8a8f99', lw=1, label='T(t)/T0 × 0,3')
a.set(xlabel='t_w (rekordy BD) / (kroki ViT / 100)', ylabel='MSD / d_shell²', title='starzenie szkła stygnącego a tokenów')
for qx in ax.flat[:5]:
    qx.grid(True, which='both', alpha=0.4)
    qx.legend(fontsize=8)
a = ax[1, 2]
nazwy = [o['kryterium'] for o in w['karta']]
kolumny = [k for k in wyniki] + ([f'ViT {k}' for k in ('S1', 'S4')] if vit else [])
mat = np.array([[1 if o['ok'] else 0 for o in wyniki[k]['karta']] for k in wyniki] +
               ([[1 if o['ok'] else 0 for o in O.karta(vit[k], None)[0]] for k in ('S1', 'S4')] if vit else [])).T
a.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
a.set_xticks(range(len(kolumny)))
a.set_xticklabels([f"{k.replace('stygnace_', 'styg ')}\n{int(mat[:, i].sum())}/{len(nazwy)}" for i, k in enumerate(kolumny)], fontsize=7, rotation=30)
a.set_yticks(range(len(nazwy)))
a.set_yticklabels([n[:58] for n in nazwy], fontsize=7)
a.set_title('karta zgodności: model BD i tokeny ViT')
fig.suptitle(f'Model szkła koloidalnego (dynamika browna, N={N}, {len(PHI)} gęstości naraz) obok tokenów ViT w układzie klatki', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / 'model_szkla.png')
log('GOTOWE')
