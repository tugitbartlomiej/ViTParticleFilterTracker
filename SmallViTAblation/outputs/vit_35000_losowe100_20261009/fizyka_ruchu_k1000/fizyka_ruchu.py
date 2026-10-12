"""Fizyczne właściwości ruchu cząsteczek (K1000, 101 stanów co 10 kroków, rzut PCA3 + norma 768D):
temperatura (energia szumu), wykładnik MSD, pamięć prędkości (VACF), prostoliniowość, tempo powrotu OU, promień
bezwładności, ruch w rozgrzewce. Potem: czy w tej przestrzeni cząsteczki się grupują i czy grupy pokrywają się z
narzędziem / tłem / obrazem."""
import os, sys, json, time
import numpy as np
APP = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/app'
sys.path.insert(0, APP); os.chdir(APP)
import ruch_zbioru as R
OUT = os.path.dirname(os.path.abspath(__file__))
SC = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T080332_natural_0051d7971abb_q16'
N_IMG = 300                                     # 300 obrazów × 600 = 180 000 cząsteczek, 101 stanów

t0 = time.time()
t = R.Timeline(SC)
n = N_IMG * 600
X = np.stack([np.asarray(t.frame(s), np.float32)[:n] for s in t.steps])            # T × n × 4
T = len(t.steps)
P, nrm = X[..., :3].astype(np.float64), X[..., 3].astype(np.float64)
print('wczytano', X.shape, round(time.time() - t0), 's', flush=True)

# narzędzie: patch, którego środek leży w ramce GT (jak maska w aplikacji)
im = t.metadata['images']
tool = np.zeros(n, bool)
for i in range(N_IMG):
    x0, y0, x1, y1 = im[i]['bbox_xyxy_480']
    cx = (np.arange(600) % 30) * 16 + 8; cy = (np.arange(600) // 30) * 16 + 8
    tool[i * 600:(i + 1) * 600] = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
obraz = np.repeat(np.arange(N_IMG), 600)
patch = np.tile(np.arange(600), N_IMG)

K0 = 6                                          # po rozgrzewce (krok 60): reżim dyfuzyjny
D = np.diff(P, axis=0)                          # T-1 × n × 3 przyrosty
Dl = D[K0:]                                     # przyrosty w reżimie 2
pole = Dl.mean(1, keepdims=True)                # pole kolektywne (średnie przesunięcie populacji)
Dw = Dl - pole                                  # ruch własny (po odjęciu pola)
temp = (Dw ** 2).sum(2).mean(0)                 # temperatura: energia szumu na krok
dn = np.linalg.norm(Dl, axis=2)
cos1 = ((Dl[1:] * Dl[:-1]).sum(2) / (dn[1:] * dn[:-1] + 1e-12)).mean(0)     # VACF lag 1
# MSD względem punktu po rozgrzewce, wykładnik z log-log dla τ = 1..12
Q = P[K0:]
taus = np.arange(1, 13)
msd = np.stack([((Q[tau:] - Q[:-tau]) ** 2).sum(2).mean(0) for tau in taus])     # 12 × n
lt = np.log(taus)[:, None]; lm = np.log(msd + 1e-12)
alpha = ((lt - lt.mean()) * (lm - lm.mean(0))).sum(0) / ((lt - lt.mean()) ** 2).sum()
prosto = np.linalg.norm(Q[-1] - Q[0], axis=1) / (dn.sum(0) + 1e-12)
# OU: Δ_k = -κ (x_k - x̄) + szum  (κ z regresji, po składowych, uśrednione)
xc = Q[:-1] - Q.mean(0, keepdims=True)
kappa = -((xc * Dl).sum((0, 2))) / ((xc ** 2).sum((0, 2)) + 1e-12)
rg = np.sqrt(((Q - Q.mean(0)) ** 2).sum(2).mean(0))
rozgrz = np.linalg.norm(P[K0] - P[0], axis=1)
n768 = nrm[-1]
cechy = np.stack([np.log(temp + 1e-9), alpha, cos1, prosto, kappa, np.log(rg + 1e-9), np.log(rozgrz + 1e-9), np.log(n768 + 1e-9)], 1)
nazwy = ['log temperatura', 'wykładnik MSD α', 'VACF lag1', 'prostoliniowość', 'tempo powrotu κ', 'log R_g',
         'log ruch rozgrzewki', 'log norma 768D (k=1000)']
print('cechy policzone', round(time.time() - t0), 's', flush=True)


def opis(nazwa, v):
    q = np.percentile(v, [5, 50, 95])
    return f'{nazwa:26s} mediana {q[1]:8.3f}  (5–95%: {q[0]:8.3f} .. {q[2]:8.3f})  narzędzie {np.median(v[tool]):8.3f}  tło {np.median(v[~tool]):8.3f}'


print('\n== Właściwości ruchu (reżim po rozgrzewce, kroki 60–1000)')
for nm, v in zip(nazwy, cechy.T):
    print(opis(nm, np.exp(v) if nm.startswith('log') else v))
print(f'udział energii pola kolektywnego w ruchu: {((pole ** 2).sum(2).mean()) / ((Dl ** 2).sum(2).mean()):.3f}')
print(f'MSD populacji: α mediana {np.median(alpha):.2f} (1 = dyfuzja, <1 subdyfuzja/uwięzienie, 2 = balistyczny)')

# grupowanie w przestrzeni cech: GMM + BIC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
Z = (cechy - cechy.mean(0)) / cechy.std(0)
rng = np.random.default_rng(0); sub = rng.choice(n, 40000, replace=False)
bic = {}
for k in range(1, 9):
    g = GaussianMixture(k, covariance_type='full', random_state=0, n_init=2).fit(Z[sub])
    bic[k] = g.bic(Z[sub])
kbest = min(bic, key=bic.get)
g = GaussianMixture(kbest, covariance_type='full', random_state=0, n_init=3).fit(Z[sub])
lab = g.predict(Z)
print('\n== Grupowanie (GMM, BIC):', {k: round(v / 1e3) for k, v in bic.items()}, '-> k =', kbest)
print('ARI grup z narzędzie/tło:', round(adjusted_rand_score(tool, lab), 3))
print('ARI grup z numerem obrazu:', round(adjusted_rand_score(obraz, lab), 3))
print('ARI grup z numerem patcha:', round(adjusted_rand_score(patch, lab), 3))
for c in range(kbest):
    m = lab == c
    print(f'  grupa {c}: {m.mean() * 100:5.1f}%  narzędzie {tool[m].mean() * 100:5.1f}%  ' +
          '  '.join(f'{nm.split()[0][:6]}={np.median(cechy[m, j]):.2f}' for j, nm in enumerate(nazwy)))
# ile wariancji cech tłumaczy obraz (średnia po obrazie) i patch
for nm, j in (('obraz', 0), ('patch', 1)):
    gr = obraz if j == 0 else patch
    expl = []
    for c in range(cechy.shape[1]):
        v = cechy[:, c]; sr = np.zeros(gr.max() + 1); np.add.at(sr, gr, v); cnt = np.bincount(gr); sr /= cnt
        expl.append(1 - ((v - sr[gr]) ** 2).mean() / v.var())
    print(f'udział wariancji cech tłumaczony przez {nm}:', ' '.join(f'{e:.2f}' for e in expl))
# AUC narzędzie vs tło dla każdej cechy
from sklearn.metrics import roc_auc_score
print('AUC narzędzie vs tło:', ' '.join(f'{nm.split()[0][:6]}={max(a := roc_auc_score(tool, cechy[:, j]), 1 - a):.2f}' for j, nm in enumerate(nazwy)))

# rysunek
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(16, 9), facecolor='#111418')
for a in ax.ravel():
    a.set_facecolor('#171b20'); a.tick_params(colors='#ddd'); [s.set_color('#555') for s in a.spines.values()]
    a.xaxis.label.set_color('#ddd'); a.yaxis.label.set_color('#ddd'); a.title.set_color('#eee')
s2 = rng.choice(n, 30000, replace=False)
kol = np.where(tool[s2], '#ff5544', '#6fb7ff')
ax[0, 0].scatter(cechy[s2, 1], cechy[s2, 0], s=1, c=kol, alpha=.35); ax[0, 0].set_xlabel('wykładnik MSD α'); ax[0, 0].set_ylabel('log temperatura'); ax[0, 0].set_title('ciepło a rodzaj dyfuzji (czerwone = narzędzie)')
ax[0, 1].scatter(cechy[s2, 2], cechy[s2, 4], s=1, c=kol, alpha=.35); ax[0, 1].set_xlabel('VACF lag 1'); ax[0, 1].set_ylabel('tempo powrotu κ'); ax[0, 1].set_title('pamięć prędkości a siła powrotu')
ax[0, 2].scatter(cechy[s2, 6], cechy[s2, 7], s=1, c=kol, alpha=.35); ax[0, 2].set_xlabel('log ruch w rozgrzewce'); ax[0, 2].set_ylabel('log norma 768D po 1000'); ax[0, 2].set_title('rozgrzewka a całkowita zmiana')
mt = msd.mean(1); mn = msd[:, tool].mean(1); mb = msd[:, ~tool].mean(1)
ax[1, 0].loglog(taus * 10, mt, 'w-', label='wszystkie'); ax[1, 0].loglog(taus * 10, mn, '-', color='#ff5544', label='narzędzie'); ax[1, 0].loglog(taus * 10, mb, '-', color='#6fb7ff', label='tło')
ax[1, 0].loglog(taus * 10, mt[0] * taus, ':', color='#aaa', label='α = 1'); ax[1, 0].set_xlabel('τ (kroki treningu)'); ax[1, 0].set_ylabel('MSD w PCA3'); ax[1, 0].set_title('średnie przesunięcie kwadratowe'); ax[1, 0].legend(facecolor='#222', labelcolor='#eee')
vac = [1.0] + [((Dl[l:] * Dl[:-l]).sum(2).mean() / (Dl ** 2).sum(2).mean()) for l in range(1, 11)]
vw = [1.0] + [((Dw[l:] * Dw[:-l]).sum(2).mean() / (Dw ** 2).sum(2).mean()) for l in range(1, 11)]
ax[1, 1].plot(np.arange(11) * 10, vac, 'o-', color='#ffd166', label='pełny ruch'); ax[1, 1].plot(np.arange(11) * 10, vw, 's-', color='#9be', label='po odjęciu pola kolektywnego')
ax[1, 1].axhline(0, color='#666'); ax[1, 1].set_xlabel('opóźnienie (kroki)'); ax[1, 1].set_ylabel('VACF'); ax[1, 1].set_title('autokorelacja prędkości populacji'); ax[1, 1].legend(facecolor='#222', labelcolor='#eee')
kr = np.array(t.steps[1:]); ax[1, 2].plot(kr, (D ** 2).sum(2).mean(1), '-', color='#ffd166', label='energia kroku (wszystkie)')
ax[1, 2].plot(kr, (D.mean(1) ** 2).sum(1), '-', color='#9be', label='energia pola kolektywnego'); ax[1, 2].set_yscale('log'); ax[1, 2].set_xlabel('krok treningu'); ax[1, 2].set_ylabel('⟨|Δ|²⟩'); ax[1, 2].set_title('temperatura w czasie'); ax[1, 2].legend(facecolor='#222', labelcolor='#eee')
fig.suptitle('Właściwości ruchu 180 000 cząsteczek (300 zdjęć × 600), historia K1000 co 10 kroków, rzut PCA3', color='#eee')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fizyka_ruchu.png'), dpi=110, facecolor=fig.get_facecolor())
np.savez_compressed(os.path.join(OUT, 'fizyka_cechy.npz'), cechy=cechy, nazwy=nazwy, tool=tool, obraz=obraz, patch=patch, lab=lab, msd=msd, vac=vac, vw=vw)
print('zapisano', round(time.time() - t0), 's')
