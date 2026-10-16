"""Gdzie dokładnie leży cel tokenu 226 i czy da się go przenieść do przestrzeni tokenu (latentu, 192 wymiary).

Cel w przestrzeni wyjścia głowy o = (logit, ln l, ln t, ln r, ln b): logit → +∞ (p = 1), exp(o[1:]) = ltrb celu.
Dla głów z kilku checkpointów (wagi głowy stałe) szukamy w przestrzeni tokenu x (Adam, start w prawdziwym x tokenu 226):
  A) najwyższego osiągalnego logitu przy ramce równej celowi (kara za odchylenie ltrb) — czy p = 1 jest osiągalne;
  B) najbliższego x, który daje cel ramki i logit = obecny + 1 (przesunięcie |Δx| i jego udział w 5 kierunkach czytanych);
  C) ile kierunków nie zmienia odpowiedzi (wymiar zbioru przeciwobrazu = 192 − rząd jakobianu).
Zapis: outputs/pulapka_cel_koloidu_20261015/CEL_W_LATENCIE.md
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
_, ltrb, _ = M.cele([[[float(v) for v in h.gt]]], 20, 30); cel = ltrb.view(600, 4)[226].to(dev)
net = M.MalyViT().to(dev).eval()
for p_ in net.parameters():
    p_.requires_grad_(False)
linie = []


def pisz(s=''):
    print(s); linie.append(s)


def glowa(x):
    return net.glowa(net.norm(x))


pisz('# Gdzie leży cel tokenu 226 i dlaczego nie ma go jako punktu w latencie')
pisz(f'\nCel w wyjściu głowy: logit → +∞ (p = 1); ltrb = {np.round(cel.cpu().numpy(), 2).tolist()} kratek, '
     f'czyli o[1:] = ln ltrb = {np.round(np.log(cel.cpu().numpy()), 3).tolist()}.')
for i in [102, 200, 400, 600, len(h.kroki) - 1]:
    st = torch.load(Path(h.wagi) / f'k{int(h.kroki[i]):05d}.pt', map_location=dev, weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    x0 = torch.from_numpy(X[i, j]).to(dev)
    o0 = glowa(x0[None])[0]
    # C) jakobian i wymiar przeciwobrazu
    Jh = torch.autograd.functional.jacobian(lambda x: glowa(x[None])[0], x0)
    sv = torch.linalg.svdvals(Jh); rz = int((sv > sv[0] * 1e-6).sum())
    Q = torch.linalg.qr(Jh.T)[0]
    # A) najwyższy logit przy ramce = cel
    x = x0.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(3000):
        o = glowa(x[None])[0]
        loss = -o[0] + 100 * ((o[1:].exp() - cel) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    oA = glowa(x.detach()[None])[0]
    # B) najbliższy x z ramką = cel i logitem = obecny + 1
    x = x0.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=0.02)
    for _ in range(3000):
        o = glowa(x[None])[0]
        loss = ((x - x0) ** 2).sum() + 1e3 * ((o[1:].exp() - cel) ** 2).sum() + 1e3 * (o[0] - (o0[0] + 1)) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    dx = (x.detach() - x0); oB = glowa(x.detach()[None])[0]
    udz = float((dx @ Q).pow(2).sum() / dx.pow(2).sum().clamp_min(1e-30))
    pA = float(torch.sigmoid(oA[0]))
    pisz(f'\n## krok {int(h.kroki[i])} (|x| = {float(x0.norm()):.1f})')
    pisz(f'- teraz: logit {float(o0[0]):+.2f} (p = {float(torch.sigmoid(o0[0])):.4f}), ltrb {np.round(o0[1:].exp().cpu().numpy(), 2).tolist()}')
    pisz(f'- A) najwyższy logit osiągalny samym ruchem tokenu (ramka = cel): {float(oA[0]):+.2f} (p = {pA:.4f}), '
         f'ltrb {np.round(oA[1:].exp().cpu().numpy(), 2).tolist()}')
    pisz(f'- B) najbliższy punkt z ramką = cel i logitem +1: |Δx| = {float(dx.norm()):.2f} ({float(dx.norm() / x0.norm()):.1%} |x|), '
         f'{udz:.0%} w 5 kierunkach czytanych; osiągnięte: logit {float(oB[0]):+.2f}, ltrb {np.round(oB[1:].exp().cpu().numpy(), 2).tolist()}')
    pisz(f'- C) rząd jakobianu głowy {rz} → zbiór punktów o tej samej odpowiedzi ma {192 - rz} wymiarów')
(KAT / 'CEL_W_LATENCIE.md').write_text('\n'.join(linie) + '\n', encoding='utf-8')
print('zapisano', KAT / 'CEL_W_LATENCIE.md')
