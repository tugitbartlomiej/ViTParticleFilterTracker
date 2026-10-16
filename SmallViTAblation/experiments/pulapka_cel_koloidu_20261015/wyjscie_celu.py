"""Do jakiego punktu dąży koloid: cel w przestrzeni wyjścia głowy (5 liczb) tokenu środka ramki.

Cel z kodu uczenia: mapa środków heat = 1 (ogniskowa strata CenterNet: p → 1, czyli logit → +∞, brak skończonego minimum)
i odległości ramki ltrb (1 − GIoU: skończony cel). Dla każdego checkpointu: wyjście głowy tokenu 226 (logit i ltrb),
strata tej części, siła błędu na logit ∂L/∂logit; dopasowanie logit ~ a·ln t + b (jak regresja logistyczna na danych
rozdzielnych: logit rośnie logarytmicznie, siła maleje jak 1/t); normy wag głowy (weight decay ściąga je do zera).
Zapis: outputs/pulapka_cel_koloidu_20261015/WYJSCIE_CELU.md
"""
import sys
from pathlib import Path

import numpy as np
import torch

APP, KAT_H = sys.argv[1], sys.argv[2]
KAT = Path(__file__).resolve().parents[2] / 'outputs' / 'pulapka_cel_koloidu_20261015'
sys.path.insert(0, APP)
import lasery_dane as LD

f = np.load(KAT / 'pomiar.npz'); wyb = list(f['wybrane']); X = f['X_wyb']; j = wyb.index(226)
dev = torch.device('cuda'); M = LD.modul_modelu(); h = LD.Historia(KAT_H)
_, ltrb, _ = M.cele([[[float(v) for v in h.gt]]], 20, 30); cel = ltrb.view(600, 4)[226].numpy()
net = M.MalyViT().to(dev).eval()
K = h.K; O = np.zeros((K, 5)); W = np.zeros((K, 3))
with torch.no_grad():
    for i in range(K):
        st = torch.load(Path(h.wagi) / f'k{int(h.kroki[i]):05d}.pt', map_location=dev, weights_only=True)
        net.load_state_dict(st.get('net', st), strict=True)
        O[i] = net.glowa(net.norm(torch.from_numpy(X[i, j:j + 1]).to(dev)))[0].cpu().numpy()
        W[i] = [float(net.glowa[0].weight.norm()), float(net.glowa[2].weight.norm()), float(X[i, j] @ X[i, j]) ** 0.5]
kr = h.kroki.astype(float); z = O[:, 0]; p = 1 / (1 + np.exp(-z))
sila = (1 - p) ** 2 * (1 - p - 2 * p * np.log(p))                    # −∂/∂z [−(1−p)² ln p] (siła błędu na logit)
pr = np.exp(O[:, 1:]); bl_ltrb = np.abs(pr - cel).mean(1)
s = kr >= 1000
a, b = np.polyfit(np.log(kr[s]), z[s], 1); r2 = 1 - np.var(z[s] - (a * np.log(kr[s]) + b)) / np.var(z[s])
a2, b2 = np.polyfit(np.log(kr[s]), np.log(sila[s]), 1)
linie = []


def pisz(x=''):
    print(x); linie.append(x)


pisz('# Do jakiego punktu dąży koloid: wyjście głowy tokenu 226 (środek ramki)')
pisz(f'\nCel: heat = 1 (logit → +∞), ltrb = {np.round(cel, 2).tolist()} (w krokach siatki 16 px).')
for i in [0, 50, 102, 200, 400, 600, K - 1]:
    pisz(f'- krok {int(kr[i])}: logit {z[i]:+.2f} (p = {p[i]:.4f}), siła na logit {sila[i]:.2e}; ltrb {np.round(pr[i], 2).tolist()} '
         f'(średni błąd {bl_ltrb[i]:.3f}); |W1| {W[i, 0]:.2f}, |W2| {W[i, 1]:.2f}, |x| {W[i, 2]:.1f}')
pisz(f'\nlogit ~ {a:.2f}·ln t + {b:.2f} (R² {r2:.3f}, kroki ≥ 1000); siła na logit ~ t^{a2:.2f}.')
pisz(f'Błąd ltrb: krok 1000 {bl_ltrb[np.argmin(np.abs(kr - 1000))]:.3f} → koniec {bl_ltrb[-1]:.3f}.')
(KAT / 'WYJSCIE_CELU.md').write_text('\n'.join(linie) + '\n', encoding='utf-8')
print('zapisano', KAT / 'WYJSCIE_CELU.md')
