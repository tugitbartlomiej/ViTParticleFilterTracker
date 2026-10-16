"""Do czego zbiega pułapka uczenia: dolina zerowego błędu zamiast punktu.

Głowa czyta z tokenu x tylko 5 liczb: o = głowa(LN(x)). Kierunki, które głowa widzi, to wiersze jakobianu J_h = ∂o/∂x (5 × 192)
w danym checkpoincie; reszta (187 kierunków) to dno doliny — tam zmiana x nie zmienia odpowiedzi.
Hipoteza: w kierunkach czytanych przez głowę pułapka uczenia jest sztywna i koloid zbiega (prędkość środka maleje, drgania małe),
wzdłuż dna doliny koloid wędruje (dyfuzja, brak punktu końcowego).
Dla tokenów w ramce i 226 w każdym checkpoincie: J_h (z LN), stan x; potem z pomiar.npz (X_wyb) i J_h:
  środek pułapki m_t (średnia ruchoma 21 ckpt co 100 kroków), drgania y = x − m;
  rozkład prędkości środka i drgań na podprzestrzeń czytaną (5 wymiarów) i dno (187), na wymiar;
  odległość od końcowego środka w obu podprzestrzeniach w czasie.
Zapis: outputs/pulapka_cel_koloidu_20261015/DOLINA.md i dolina.npz (J_h tokenów wybranych).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

APP, KAT_H = sys.argv[1], sys.argv[2]
KAT = Path(__file__).resolve().parents[2] / 'outputs' / 'pulapka_cel_koloidu_20261015'
sys.path.insert(0, APP)
import lasery_dane as LD

f = np.load(KAT / 'pomiar.npz')
wyb = f['wybrane']; X = f['X_wyb']; kr = f['kroki']
plik_j = KAT / 'dolina.npz'
if plik_j.exists():
    JH = np.load(plik_j)['JH']
else:
    dev = torch.device('cuda'); M = LD.modul_modelu(); h = LD.Historia(KAT_H)
    net = M.MalyViT().to(dev).eval()
    for p_ in net.parameters():
        p_.requires_grad_(False)
    JH = np.zeros((h.K, len(wyb), 5, 192), np.float32); t0 = time.time()
    for i in range(h.K):
        st = torch.load(Path(h.wagi) / f'k{int(h.kroki[i]):05d}.pt', map_location=dev, weights_only=True)
        net.load_state_dict(st.get('net', st), strict=True)
        x = torch.from_numpy(X[i]).to(dev).requires_grad_(True)       # stany tokenów wybranych (z pomiaru)
        o = net.glowa(net.norm(x))                                     # (n, 5): token zależy tylko od siebie
        for c in range(5):
            g, = torch.autograd.grad(o[:, c].sum(), x, retain_graph=True)
            JH[i, :, c] = g.cpu().numpy()
        if i % 200 == 0:
            print(f'{i}/{h.K} {time.time() - t0:.0f} s', flush=True)
    np.savez_compressed(plik_j, JH=JH)

reg = np.flatnonzero((kr >= 1100) & (np.r_[np.diff(kr), 0] == 100)); reg = np.r_[reg, reg[-1] + 1]
t = kr[reg]; Xr = X[reg]; Jr = JH[reg]; hw = 10
m = np.stack([Xr[max(0, i - hw):i + hw + 1].mean(0) for i in range(len(reg))])
y = Xr - m
linie = []


def pisz(s=''):
    print(s); linie.append(s)


def rozklad(v, J):
    """v (n, 192), J (n, 5, 192) → (‖część czytana‖², ‖część dna‖²) na token."""
    Q = np.linalg.qr(np.transpose(J, (0, 2, 1)))[0]                     # (n, 192, 5) baza czytanych kierunków
    r = np.einsum('nd,ndk->nk', v, Q)
    czyt = (r ** 2).sum(-1)
    return czyt, (v ** 2).sum(-1) - czyt


pisz('# Do czego zbiega pułapka uczenia: dolina zerowego błędu')
pisz('\nKierunki czytane przez głowę: 5 wierszy jakobianu ∂o/∂x; dno doliny: pozostałe 187. Wartości na jeden wymiar '
     '(część czytana / 5, dno / 187), żeby porównać sztywność.')
okresy = {'wczesne (1 100–8 000)': (t >= 1100) & (t < 8000), 'środkowe (8 000–30 000)': (t >= 8000) & (t < 30000),
          'późne (30 000–65 000)': t >= 30000}
for nazwa, kol in (('226', [list(wyb).index(226)]), ('w ramce', list(range(len(wyb))))):
    pisz(f'\n## {nazwa}')
    for on, s in okresy.items():
        ii = np.flatnonzero(s)
        ii = ii[(ii >= hw) & (ii < len(reg) - hw - 1)]
        vc, vd, yc, yd = [], [], [], []
        for i in ii:
            a, b = rozklad(m[i + 1, kol] - m[i, kol], Jr[i, kol]); vc.append(a); vd.append(b)
            a, b = rozklad(y[i, kol], Jr[i, kol]); yc.append(a); yd.append(b)
        vc, vd, yc, yd = (np.mean(z) for z in (vc, vd, yc, yd))
        pisz(f'- {on}: prędkość środka na wymiar: czytane {np.sqrt(vc / 5):.2e}, dno {np.sqrt(vd / 187):.2e} '
             f'(stosunek {np.sqrt(vc / 5) / np.sqrt(vd / 187):.2f}); drgania na wymiar: czytane {np.sqrt(yc / 5):.2e}, '
             f'dno {np.sqrt(yd / 187):.2e} (stosunek {np.sqrt(yc / 5) / np.sqrt(yd / 187):.2f})')
    # odległość od końcowego środka w obu podprzestrzeniach (baza czytana z końca uczenia)
    iT = len(reg) - hw - 1
    dc, dd = [], []
    for i in range(hw, iT, 20):
        a, b = rozklad(m[i, kol] - m[iT, kol], Jr[iT, kol]); dc.append(np.sqrt(np.mean(a) / 5)); dd.append(np.sqrt(np.mean(b) / 187))
    tt = t[hw:iT:20]
    pisz(f'- odległość od końcowego środka na wymiar (krok {tt[0]} → {tt[len(tt) // 2]} → {tt[-1]}): czytane '
         f'{dc[0]:.2e} → {dc[len(dc) // 2]:.2e} → {dc[-1]:.2e}; dno {dd[0]:.2e} → {dd[len(dd) // 2]:.2e} → {dd[-1]:.2e}')
(KAT / 'DOLINA.md').write_text('\n'.join(linie) + '\n', encoding='utf-8')
print('zapisano', KAT / 'DOLINA.md')
