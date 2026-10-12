"""Właściwości ruchu liczone bezpośrednio z gęstego zapisu (bieg „gesty”: 100 sond × 600 tokenów, PCA32 i normy 768D
co 1 krok do kroku 2188, potem co 10; skład paczki, strata przed/po, norma aktualizacji wag co krok).

Wynik: trajektorie_wlasnosci.npz i PODSUMOWANIE.json (do animacji w edit/animations i do raportu).
Strumieniowo po plikach pca32_NNN.npz (2 GB każdy); pełna trajektoria tylko dla podpróbki 10 sond (6000 tokenów).
"""
import glob
import json
import os
import time

import numpy as np

G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
E35K = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd'
OUT = os.path.dirname(os.path.abspath(__file__))
PODPROBKA = 10                                   # sond w pełnej trajektorii (10 × 600 = 6000 tokenów)
OBSZARY = ('tło', 'rogówka', 'źrenica', 'narzędzie')

t0 = time.time()
sondy = json.load(open(os.path.join(G, 'SONDY.json'), encoding='utf-8'))
pool = np.asarray(sondy['sondy'])                # wiersze zbioru = indeksy paczki
klasy = np.load(os.path.join(E35K, 'obszary_oka.npz'))['klasy']          # (100, 600) ta sama kolejność sond
assert klasy.shape == (100, 600)
kl = klasy.reshape(-1)
P = np.load(os.path.join(G, 'PACZKI.npz'))
krok_p, idx_p = P['krok'], P['idx']
w_paczce = np.zeros((len(krok_p) + 1, 100), bool)                        # [krok, sonda]: obraz sondy w paczce kroku
for t, row in zip(krok_p, idx_p):
    w_paczce[t] = np.isin(pool, row)

# --- statystyki populacji co zapis (strumieniowo)
pliki = sorted(glob.glob(os.path.join(G, 'pca32_*.npz')))
kroki, energia, pole, temp_ob, temp768_ob, cos_ob, n_ob, e_sonda768 = [], [], [], [], [], [], [], []
traj, traj_kroki = [], []
prev_p = prev_d = None; prev_k = None
for plik in pliki:
    z = np.load(plik)
    kr, pca, nr = z['kroki'], z['pca32'], z['normy']
    n = len(kr)
    for a in range(0, n, 50):
        blok = pca[a:a + 50].astype(np.float32).reshape(-1, 60000, 32)          # (b, tokeny, 32)
        nrm = nr[a:a + 50].astype(np.float32).reshape(-1, 60000, 2)
        for j in range(len(blok)):
            k = int(kr[a + j]); p = blok[j]
            if k < PODPROBKA * 0 + 999999 and (prev_k is None or k - prev_k in (1, 10)):
                pass
            traj.append(p[:PODPROBKA * 600].copy()); traj_kroki.append(k)
            if prev_p is not None and k - prev_k in (1, 10):
                d = p - prev_p
                f = d.mean(0)
                e = (d ** 2).sum(1)
                dw = d - f
                ew = (dw ** 2).sum(1)
                e768 = nrm[j, :, 1] ** 2                                        # |z_t - z_poprz|² w 768D
                kroki.append(k); energia.append(float(e.mean())); pole.append(float((f ** 2).sum()))
                temp_ob.append([float(ew[kl == c].mean()) for c in range(4)])
                temp768_ob.append([float(e768[kl == c].mean()) for c in range(4)])
                e_sonda768.append(e768.reshape(100, 600).mean(1))
                if prev_d is not None:
                    ln = np.linalg.norm(d, axis=1) * np.linalg.norm(prev_d, axis=1)
                    cs = (d * prev_d).sum(1) / np.maximum(ln, 1e-20)
                    cos_ob.append([float(cs[(kl == c) & (ln > 1e-20)].mean()) for c in range(4)])
                else:
                    cos_ob.append([np.nan] * 4)
                prev_d = d
            else:
                prev_d = None
            prev_p, prev_k = p, k
    print(os.path.basename(plik), 'kroki', kr[0], '..', kr[-1], f'{time.time() - t0:.0f} s', flush=True)
kroki = np.asarray(kroki); energia = np.asarray(energia); pole = np.asarray(pole)
temp_ob = np.asarray(temp_ob); temp768_ob = np.asarray(temp768_ob); cos_ob = np.asarray(cos_ob)
e_sonda768 = np.asarray(e_sonda768)                                             # (zapisy, 100)
traj = np.stack(traj); traj_kroki = np.asarray(traj_kroki)                     # (zapisy, 6000, 32)
kl_sub = kl[:PODPROBKA * 600]
print('populacja policzona', kroki.shape, traj.shape, f'{time.time() - t0:.0f} s', flush=True)

# --- efekt paczki: energia kroku sondy, gdy jej obraz JEST w paczce vs nie jest (część gęsta, co 1 krok)
geste = kroki <= 2188
wp = w_paczce[kroki[geste]]                                                     # (zapisy gęste, 100)
e_g = e_sonda768[geste]
w_tak = float(e_g[wp].mean()); w_nie = float(e_g[~wp].mean())
# ten sam efekt w czasie (okna po 200 kroków)
okna = []
for a in range(1, 2188, 200):
    m = (kroki >= a) & (kroki < a + 200) & geste
    sub_w = w_paczce[kroki[m]]; sub_e = e_sonda768[m]
    if sub_w.any() and (~sub_w).any():
        okna.append(dict(od=int(a), do=int(min(a + 199, 2188)), w_paczce=float(sub_e[sub_w].mean()),
                         poza=float(sub_e[~sub_w].mean()), stosunek=float(sub_e[sub_w].mean() / sub_e[~sub_w].mean())))

# --- związek z siłą kroku: energia populacji vs spadek straty, norma aktualizacji wag, lr (część gęsta)
kp = kroki[geste]
spadek = (P['strata_przed'] - P['strata_po'])[kp - 1]
akt = P['norma_aktualizacji'][kp - 1]; lr = P['lr'][kp - 1]
eg = energia[geste]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


kor = dict(energia_vs_norma_aktualizacji=spearman(eg, akt), energia_vs_spadek_straty=spearman(eg, spadek),
           energia_vs_lr=spearman(eg, lr), energia_vs_norma_aktualizacji_po_rozgrzewce=spearman(eg[kp > 300], akt[kp > 300]),
           energia_vs_spadek_straty_po_rozgrzewce=spearman(eg[kp > 300], spadek[kp > 300]))

# --- MSD(τ), lokalny wykładnik i starzenie (podpróbka, część gęsta co 1 krok)
ig = traj_kroki <= 2188
Tg = traj[ig]                                                                    # (2189, 6000, 32)
taus = np.unique(np.round(np.logspace(0, np.log10(1500), 36)).astype(int))
msd = {}
for tw in (0, 30, 100, 300, 1000):
    rows = []
    for tau in taus:
        if tw + tau >= len(Tg):
            rows.append([np.nan] * 4); continue
        dd = ((Tg[tw + tau] - Tg[tw]) ** 2).sum(1)
        rows.append([float(dd[kl_sub == c].mean()) for c in range(4)])
    msd[tw] = np.asarray(rows)
# MSD uśrednione po oknach startowych (stacjonarne przybliżenie) w reżimie po rozgrzewce t ≥ 300
rows = []
for tau in taus:
    st = np.arange(300, len(Tg) - tau, max(1, (len(Tg) - 300 - tau) // 40)) if 300 + tau < len(Tg) else []
    if len(st) == 0:
        rows.append([np.nan] * 4); continue
    dd = np.mean([((Tg[s + tau] - Tg[s]) ** 2).sum(1) for s in st], axis=0)
    rows.append([float(dd[kl_sub == c].mean()) for c in range(4)])
msd_stac = np.asarray(rows)
lt = np.log(taus.astype(float)); lm = np.log(np.maximum(msd_stac, 1e-20))
alfa_lok = np.full_like(lm, np.nan)
alfa_lok[1:-1] = (lm[2:] - lm[:-2]) / (lt[2:] - lt[:-2])[:, None]

# --- VACF(opóźnienie) po rozgrzewce (podpróbka, przyrosty co 1 krok, t ≥ 300)
D = np.diff(Tg[300:], axis=0)                                                    # (n-1, 6000, 32)
D = D - D.mean(1, keepdims=True)                                                 # bez pola kolektywnego
lags = np.unique(np.concatenate([np.arange(1, 21), np.round(np.logspace(np.log10(25), np.log10(800), 14)).astype(int)]))
vacf = []
den = (D * D).sum(2).mean(0)
for l in lags:
    num = (D[l:] * D[:-l]).sum(2).mean(0)
    v = num / np.maximum(den, 1e-20)
    vacf.append([float(v[kl_sub == c].mean()) for c in range(4)])
vacf = np.asarray(vacf)

np.savez_compressed(os.path.join(OUT, 'trajektorie_wlasnosci.npz'), kroki=kroki, energia=energia, pole=pole,
                    temp_ob=temp_ob, temp768_ob=temp768_ob, cos_ob=cos_ob, e_sonda768=e_sonda768, w_paczce=w_paczce,
                    taus=taus, msd_stac=msd_stac, alfa_lok=alfa_lok, lags=lags, vacf=vacf,
                    **{f'msd_tw{tw}': v for tw, v in msd.items()},
                    strata_przed=P['strata_przed'], strata_po=P['strata_po'], norma_aktualizacji=P['norma_aktualizacji'],
                    lr=P['lr'], krok_paczki=P['krok'])
pods = dict(
    zapisy=int(len(kroki)), geste_do=2188, podprobka_tokenow=int(PODPROBKA * 600), obszary=OBSZARY,
    efekt_paczki=dict(energia768_gdy_obraz_w_paczce=w_tak, energia768_gdy_poza=w_nie, stosunek=w_tak / w_nie,
                      udzial_krokow_w_paczce=float(wp.mean()), okna=okna),
    korelacje_spearmana=kor,
    temperatura_pca32=dict(kroki=[int(k) for k in (30, 100, 300, 1000, 2000)],
                           **{OBSZARY[c]: [float(np.median(temp_ob[(kroki >= k - 20) & (kroki <= k + 20), c])) for k in (30, 100, 300, 1000, 2000)]
                              for c in range(4)}),
    vacf_lag1_po_rozgrzewce={OBSZARY[c]: float(vacf[0, c]) for c in range(4)},
    vacf_lag10_po_rozgrzewce={OBSZARY[c]: float(vacf[list(lags).index(10), c]) for c in range(4)},
    alfa_lokalna={f'tau_{int(taus[i])}': [float(x) for x in alfa_lok[i]] for i in range(1, len(taus) - 1, 4)},
    pole_kolektywne_udzial=dict(kroki=[int(k) for k in (5, 30, 60, 100, 300, 1000, 2000)],
                                udzial=[float(np.median((pole / np.maximum(energia, 1e-20))[(kroki >= k - 5) & (kroki <= k + 5)])) for k in (5, 30, 60, 100, 300, 1000, 2000)]),
    czas_s=round(time.time() - t0))
json.dump(pods, open(os.path.join(OUT, 'PODSUMOWANIE.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(pods, ensure_ascii=False, indent=1)[:3000], flush=True)
print('GOTOWE', round(time.time() - t0), 's', flush=True)
