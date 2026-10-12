"""Bezpośredni pomiar oddziaływania między tokenami (Eden, GPU): funkcja odpowiedzi G(r), jak pchnięcie jednej cząstki w krysztale.
Do tokenu i zaraz po lodydze (przed blokami, w miejscu dodania kodowania pozycji) dodajemy ±δ (losowy kierunek 192D, |δ| = ε·|t_i|)
i mierzymy zmianę latentu (GELU fc1 ostatniego bloku, 768D) WSZYSTKICH tokenów: Δz_j = (z_j(+δ) − z_j(−δ)) / 2. Pchamy w przestrzeni
tokenów, nie pikseli, bo pola recepcyjne lodygi zachodzą na sąsiednie patche — tak mierzymy tylko sprzężenie przez bloki (uwagę i MLP).
Przewidywania: kryształ sprężysty — odpowiedź maleje z odległością NA SIATCE patchy i jest liniowa w δ; niezależne pułapki ze wspólnym
napędem — odpowiedź innych tokenów ≈ 0; brak adresu (bieg bez kodowania pozycji) — odpowiedź nie zależy od odległości na siatce.
Wynik: ODPOWIEDZ.json: dla każdego modelu G(r) (średnia |Δz_j| / |Δz_i| w pierścieniach odległości), udział odpowiedzi w r ≤ √2,
wykładnik spadku, nieliniowość (ε 0,02 vs 0,08), anizotropia (poziomo vs pionowo), korelacja G z odległością w latencie (kontrola).
Uruchomienie (katalog kodu biegu 35 000): python odpowiedz_tokenow.py --manifest ... --zipy ... --out ... --modele nazwa=ścieżka[:bezpoz],..."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dane as D  # noqa: E402
import model as M  # noqa: E402
from trening35000 import wczytaj_z_zipow, wiersze_35000  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', required=True)
ap.add_argument('--zipy', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--modele', required=True, help='nazwa=plik.pt[:bezpoz],...')
ap.add_argument('--n_zdjec', type=int, default=10, help='na część (TRAIN i VALID)')
ap.add_argument('--zrodla', type=int, default=40, help='pchanych tokenów na zdjęcie')
ap.add_argument('--eps', default='0.02,0.08')
ap.add_argument('--piksele', action='store_true', help='pchamy piksele patcha 16×16 (przez lodygę) zamiast tokenu po lodydze')
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
dev = 'cuda'
t0 = time.time()
log = lambda *s: print(*s, f'[{time.time() - t0:.0f} s]', flush=True)
H, W, TOK = 20, 30, 600
EPS = [float(v) for v in a.eps.split(',')]

# zdjęcia: ten sam ciąg co ruch60_rp.py (ziarno 2026), pierwsze n z TRAIN i VALID
rng = np.random.default_rng(2026)
wt = wiersze_35000(a.manifest)
idx_tr = sorted(rng.choice(len(wt), 3000, replace=False).tolist())
wal_w, wal_X = D.walidacja()
idx_val = sorted(rng.choice(len(wal_w), 1000, replace=False).tolist())
n = a.n_zdjec
X = np.concatenate([wczytaj_z_zipow([wt[i] for i in idx_tr[:n]], a.zipy), wal_X[idx_val[:n]]]); del wal_X
N = len(X)
Xg = torch.from_numpy(X).to(dev).permute(0, 3, 1, 2).float() / 255.
log(f'zdjęć {N}')

siatka = np.stack([np.arange(TOK) // W, np.arange(TOK) % W], 1).astype(float)
Dg = np.sqrt(((siatka[:, None] - siatka[None]) ** 2).sum(-1))
PIERSCIENIE = [(0.5, 1.01), (1.01, 1.5), (1.5, 2.3), (2.3, 3.2), (3.2, 4.5), (4.5, 6.5), (6.5, 9.5), (9.5, 14), (14, 40)]
rz = np.random.default_rng(0)
zrodla = np.sort(rz.choice(TOK, a.zrodla, replace=False))

orig_pe = M.sincos_2d
STAN = dict(delta=None, bezpoz=False)


def pe(h, w, dim, device):
    base = torch.zeros(h * w, dim, device=device) if STAN['bezpoz'] else orig_pe(h, w, dim, device)
    return base if STAN['delta'] is None else base + STAN['delta']


M.sincos_2d = pe
net = M.MalyViT().to(dev).eval()
net.zapisuj_latent(True)

# kierunki pchnięć: bierzemy normę tokenu po lodydze z hooka
tokeny = {}
net.lodyga.register_forward_hook(lambda m, i, o: tokeny.__setitem__('f', o.detach()))

wyniki = {}
for spec in a.modele.split(','):
    nazwa, sciezka = spec.split('=', 1)
    STAN['bezpoz'] = sciezka.endswith(':bezpoz'); sciezka = sciezka.removesuffix(':bezpoz')
    net.load_state_dict(torch.load(sciezka, map_location='cpu', weights_only=False)['net'], strict=True)
    G = {e: np.zeros(len(PIERSCIENIE)) for e in EPS}; Gh = np.zeros(len(PIERSCIENIE)); Gv = np.zeros(len(PIERSCIENIE)); cnt = np.zeros(len(PIERSCIENIE))
    lok, samo, nielin, kor_lat, wyk = [], [], [], [], []
    for im in range(N):
        x = Xg[im:im + 1]
        with torch.no_grad():
            STAN['delta'] = None
            _, z0 = net(x)
            t_norm = tokeny['f'].flatten(2).transpose(1, 2)[0].norm(dim=1)          # (600,) normy tokenów po lodydze
            z0 = z0[0].float()
            C = tokeny['f'].shape[1]                                                # 192 kanały tokenu
            odp = {}
            for e in EPS:
                kier = torch.randn(len(zrodla), C, generator=torch.Generator().manual_seed(im)).to(dev)
                kier = kier / kier.norm(dim=1, keepdim=True) * e * t_norm[torch.from_numpy(zrodla).to(dev)][:, None]
                if a.piksele:                                                      # szum na pikselach jednego patcha, amplituda ε · odchylenie obrazu
                    xp = x.expand(2 * len(zrodla), -1, -1, -1).clone()
                    gp = torch.Generator().manual_seed(1000 + im)
                    for q, i in enumerate(zrodla):
                        r0, c0 = (i // W) * 16, (i % W) * 16
                        sz = torch.randn(3, 16, 16, generator=gp).to(dev) * e * float(x.std())
                        xp[2 * q, :, r0:r0 + 16, c0:c0 + 16] += sz; xp[2 * q + 1, :, r0:r0 + 16, c0:c0 + 16] -= sz
                    STAN['delta'] = None
                    _, z = net(xp)
                else:
                    dl = torch.zeros(2 * len(zrodla), TOK, C, device=dev)
                    for q, i in enumerate(zrodla):
                        dl[2 * q, i] = kier[q]; dl[2 * q + 1, i] = -kier[q]
                    STAN['delta'] = dl
                    _, z = net(x.expand(2 * len(zrodla), -1, -1, -1))
                z = z.float()
                dz = ((z[0::2] - z[1::2]) / 2).norm(dim=2).cpu().numpy()            # (zrodla, 600)
                odp[e] = dz / e                                                    # na jednostkę ε
            STAN['delta'] = None
        z0n = z0.cpu().numpy()
        for q, i in enumerate(zrodla):
            for e in EPS:
                r = odp[e][q]; s = r[i]
                for p, (lo, hi) in enumerate(PIERSCIENIE):
                    m = (Dg[i] >= lo) & (Dg[i] < hi)
                    if m.any():
                        G[e][p] += r[m].mean() / s
                        if e == EPS[0]:
                            cnt[p] += 1
                            mh = m & (siatka[:, 0] == siatka[i, 0]); mv = m & (siatka[:, 1] == siatka[i, 1])
                            if mh.any() and mv.any():
                                Gh[p] += r[mh].mean() / s; Gv[p] += r[mv].mean() / s
            r = odp[EPS[0]][q].copy(); s = r[i]; r[i] = 0
            samo.append(float(s))
            lok.append(float(r[(Dg[i] <= np.sqrt(2) + 1e-9) & (Dg[i] > 0)].sum() / max(r.sum(), 1e-12)))
            nielin.append(float(np.abs(odp[EPS[1]][q] - odp[EPS[0]][q]).sum() / max(np.abs(odp[EPS[0]][q]).sum(), 1e-12)))
            inne = np.arange(TOK) != i
            dl_lat = np.linalg.norm(z0n - z0n[i], axis=1)

            kor_lat.append((float(spearmanr(r[inne], -Dg[i][inne])[0]), float(spearmanr(r[inne], -dl_lat[inne])[0])))
            mm = inne & (Dg[i] <= 10)
            wyk.append(float(np.polyfit(np.log(Dg[i][mm]), np.log(r[mm] + 1e-12), 1)[0]))
        if im % 5 == 0:
            log(f'{nazwa} zdjęcie {im + 1}/{N}')
    G = {str(e): (G[e] / np.maximum(cnt, 1)).tolist() for e in EPS}
    kl = np.array(kor_lat)
    wyniki[nazwa] = dict(plik=sciezka, bez_pozycji=STAN['bezpoz'], pierscienie=PIERSCIENIE, G_wzgl_samo=G,
                         G_poziomo=(Gh / np.maximum(cnt, 1)).tolist(), G_pionowo=(Gv / np.maximum(cnt, 1)).tolist(),
                         udzial_w_r_le_sqrt2=float(np.mean(lok)), udzial_losowy=8 / 599,
                         spearman_odpowiedz_bliskosc_siatki=float(np.nanmean(kl[:, 0])), spearman_odpowiedz_bliskosc_latentu=float(np.nanmean(kl[:, 1])),
                         wykladnik_spadku_r_le_10=float(np.median(wyk)), nieliniowosc_eps=float(np.median(nielin)), odpowiedz_wlasna_na_eps=float(np.median(samo)))
    w = wyniki[nazwa]
    log(f'{nazwa:>16s}: G(r) {[round(v, 3) for v in G[str(EPS[0])]]}, udział r≤√2 {w["udzial_w_r_le_sqrt2"]:.3f} (losowo {8 / 599:.3f}), '
        f'Spearman siatka {w["spearman_odpowiedz_bliskosc_siatki"]:.2f} / latent {w["spearman_odpowiedz_bliskosc_latentu"]:.2f}, wykładnik {w["wykladnik_spadku_r_le_10"]:.2f}, nieliniowość {w["nieliniowosc_eps"]:.3f}')

json.dump(dict(opis=__doc__.split('\n')[0], tryb='piksele' if a.piksele else 'token', eps=EPS, zrodla=zrodla.tolist(), zdjec=N, modele=wyniki), open(OUT / ('ODPOWIEDZ_PIKSELE.json' if a.piksele else 'ODPOWIEDZ.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('zapisano', OUT / 'ODPOWIEDZ.json')
