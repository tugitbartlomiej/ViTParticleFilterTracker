"""Klatki A, B, C: obraz z ramką GT i ramką modelu oraz rzut LDA (oś 1 × oś 2, jak R9) dla prawdziwej ramki narzędzia
i dla ramek tła tego samego kształtu (mediana i najlepsza wg sylwetki LDA z ponownym dopasowaniem). Ostatni checkpoint."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
E = Path(__file__).resolve().parents[2]
APP = E.parents[1] / 'app'
sys.path.insert(0, str(APP)); sys.path.insert(0, str(E / 'Kod_eksperymentu'))
import lasery_dane as LD, rzuty_dane as RD
from bateria import KLATKI, ramki
dev = RD._dev()
i = 805
fig, ax = plt.subplots(3, 4, figsize=(17, 11.5))
for w, (nazwa, hid) in enumerate(KLATKI.items()):
    h = LD.Historia(APP / 'cache_uczenia/lasery' / hid); d = RD.wczytaj(h)
    X = torch.as_tensor(np.asarray(d.X[i], np.float32), device=dev)
    narz = np.asarray(h.narz, bool); R, ks = ramki(narz)
    M = torch.as_tensor(np.vstack([narz[None], R]), device=dev)
    B = RD.baza_L_t(X, M)
    P = X[None] @ B
    s = RD.miary_separacji_t(P, M); sil = s['sil'].cpu().numpy(); hit = s['hit'].cpu().numpy()
    tlo = sil[1:]; med = 1 + int(np.argsort(tlo)[len(tlo) // 2]); best = 1 + int(np.argmax(tlo))
    p = (1 + (tlo >= sil[0]).sum()) / (1 + len(tlo))
    a = ax[w, 0]; a.imshow(h.rgb)
    def prost(m, kol, ls='-'):
        idx = np.flatnonzero(m); r, c = np.divmod(idx, LD.W)
        a.add_patch(Rectangle((c.min() * 16, r.min() * 16), (c.max() - c.min() + 1) * 16, (r.max() - r.min() + 1) * 16,
                              fill=False, ec=kol, lw=2, ls=ls))
    prost(narz, 'lime'); prost(R[med - 1], 'orange', '--'); prost(R[best - 1], 'red', '--')
    g, r_ = h.gt, h.ramka[i]
    a.add_patch(Rectangle((r_[0], r_[1]), r_[2] - r_[0], r_[3] - r_[1], fill=False, ec='cyan', lw=1.2))
    a.set_title(f'{nazwa}: IoU modelu {float(h.iou[i]):.2f}, pewność {float(h.pew[i]):.2f}\n'
                f'zielona = tokeny narzędzia ({narz.sum()}), cyjan = ramka modelu\npomarańczowa/czerwona = ramka tła mediana/najlepsza', fontsize=9)
    a.axis('off')
    for kol, (j, opis) in enumerate(((0, 'prawdziwa ramka narzędzia'), (med, 'ramka tła: mediana z %d położeń' % len(R)),
                                     (best, 'ramka tła: najlepsza'))):
        a = ax[w, kol + 1]; Q = P[j].cpu().numpy(); m = M[j].cpu().numpy()
        a.scatter(Q[~m, 0], Q[~m, 1], s=6, c='0.6', lw=0)
        a.scatter(Q[m, 0], Q[m, 1], s=18, c='lime' if j == 0 else ('orange' if j == med else 'red'), ec='k', lw=0.3)
        a.set_title(f'LDA dopasowana do: {opis}\nsylwetka {sil[j]:.2f}, trafienia kNN {hit[j]:.2f}' +
                    (f'\np przestrzenne (sylwetka) = {p:.3f}' if j == 0 else ''), fontsize=9)
        a.set_xlabel('oś 1 (LDA)'); a.set_ylabel('oś 2')
fig.suptitle('Krok 65640: LDA rozdziela narzędzie, ale rozdziela też prawie każdy prostokąt tła tego samego kształtu', fontsize=12)
fig.tight_layout()
out = E / 'Wyniki/Wykresy/klatki_ABC_lda_zero_przestrzenne.png'
fig.savefig(out, dpi=110); print(out)
