"""Klatki A, B, C na ostatnim checkpoincie seedu 3047 (krok 437500, epoka 400), etykieta z obrysu (udział pikseli ≥ 0,5,
mieszane pominięte): obraz z łatami narzędzia i mieszanymi, rzut LDA (oś 1 × oś 2) dla narzędzia i dla tej samej maski w tle
(mediana i najlepsze położenie wg Fishera, LDA od nowa)."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
E = Path(__file__).resolve().parents[2]
APP = E.parents[1] / 'app'
sys.path.insert(0, str(APP))
import lasery_dane as LD, rzuty_dane as RD, etykiety_narzedzia as EN
CK = E.parent / 'vit_kontynuacja_do_400epok_20261020__23-50-57/Wyniki/Dane/checkpoints/k437500.pt'
KLATKI = {'A': '140854f7a6dbcbe5', 'B': '2cca2743e1c74b14', 'C': 'c584ef4647149c7c'}
dev = RD._dev()
M_ = LD.modul_modelu(); net = M_.MalyViT().to(dev).eval().requires_grad_(False)
net.load_state_dict(torch.load(CK, map_location='cpu', weights_only=False)['net'])
pam = {}
net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))
fig, ax = plt.subplots(3, 4, figsize=(17, 11.5))
for w, (nazwa, hid) in enumerate(KLATKI.items()):
    h = LD.Historia(APP / 'cache_uczenia/lasery' / hid)
    with torch.no_grad():
        o, _ = net(torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(dev))
        bx, sc = M_.dekoduj(o.float().cpu(), k=1)
    X = pam['x'][0].float()
    narz, mies, u = h.narz, h.mieszane, h.udzial
    keep = ~mies
    R = EN.przesuniecia(narz, u > 0)
    M = torch.as_tensor(np.vstack([narz[None], R])[:, keep], device=dev)
    Xk = X[torch.as_tensor(keep, device=dev)]
    kk = RD.k_sasiadow(narz.sum())
    P = Xk[None] @ RD.baza_L_t(Xk, M)
    s = {q: v.cpu().numpy() for q, v in RD.miary_separacji_t(P, M, kk).items()}
    tlo = s['fisher'][1:]
    med = 1 + int(np.argsort(tlo)[len(tlo) // 2]); best = 1 + int(np.argmax(tlo))
    p = (1 + (tlo >= s['fisher'][0]).sum()) / (1 + len(tlo))
    a = ax[w, 0]; a.imshow(h.rgb)
    for t in range(LD.TOK):
        r_, c_ = divmod(t, LD.W)
        if narz[t] or mies[t]:
            a.add_patch(Rectangle((c_ * 16, r_ * 16), 16, 16, fill=False, ec='lime' if narz[t] else 'yellow', lw=1.2))
        for j, kol in ((med, 'orange'), (best, 'red')):
            if np.vstack([narz[None], R])[j, t]:
                a.add_patch(Rectangle((c_ * 16, r_ * 16), 16, 16, fill=False, ec=kol, lw=1, ls='--'))
    g = h.gt; a.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec='white', lw=0.8, ls=':'))
    rb = bx[0, 0].tolist(); a.add_patch(Rectangle((rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1], fill=False, ec='cyan', lw=1))
    a.set_title(f'{nazwa}: {h.etykieta}\nIoU {LD.iou(rb, g):.2f}, pewność {float(sc[0, 0]):.2f}; zielone = narzędzie ({narz.sum()}), '
                f'żółte = mieszane ({mies.sum()})\nbiała kropkowana = stara ramka GT, cyjan = ramka modelu', fontsize=8.5)
    a.axis('off')
    for kol, (j, opis, c) in enumerate(((0, 'narzędzie (obrys)', 'lime'), (med, f'ta sama maska w tle: mediana z {len(R)}', 'orange'),
                                         (best, 'ta sama maska w tle: najlepsza', 'red'))):
        a = ax[w, kol + 1]; Q = P[j].cpu().numpy(); m = M[j].cpu().numpy()
        a.scatter(Q[~m, 0], Q[~m, 1], s=6, c='0.6', lw=0)
        a.scatter(Q[m, 0], Q[m, 1], s=30, c=c, ec='k', lw=0.4)
        a.set_title(f'LDA dopasowana do: {opis}\nFisher {s["fisher"][j]:.2f}, sylwetka {s["sil"][j]:.2f}, {kk}NN {s["hit"][j]:.2f}' +
                    (f'\np przestrzenne (Fisher) = {p:.3f}' if j == 0 else ''), fontsize=9)
        a.set_xlabel('oś 1 (LDA)'); a.set_ylabel('oś 2')
fig.suptitle('Seed 3047, epoka 400 (krok 437500), etykieta z obrysu: narzędzie a ta sama maska przesunięta w tło', fontsize=12)
fig.tight_layout()
out = E / 'Wyniki/Wykresy/klatki_ABC_obrys_epoka400.png'
fig.savefig(out, dpi=110); print(out)
