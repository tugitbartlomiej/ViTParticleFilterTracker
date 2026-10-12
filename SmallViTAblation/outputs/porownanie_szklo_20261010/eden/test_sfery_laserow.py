"""Jak dobrze model „sfera 768 laserów + 600 koloidów” opisuje MalyViT? Testy lokalne na zapisanych danych biegu 35 000 (zdjęcie TRAIN 0).
Latent z_j = GELU(W u_j + b): u_j ∈ R^192 (wejście fc1 ostatniego bloku po LayerNorm) = „położenie cząsteczki”, wiersze W ∈ R^{768×192} = „lasery”.
1. TRIANGULACJA: czy ognisko jest przecięciem wiązek? Z natężeń z_j > 0 (GELU odwracalne jednoznacznie) odtwarzamy u_j najmniejszymi kwadratami
   (W_S u = GELU^{-1}(z_S) − b_S) i przewidujemy WSZYSTKIE 768 natężeń; R² na laserach spoza S = czy 768 liczb to naprawdę jeden punkt w 192D.
2. WYMIAR: ile laserów świeci (z > 0) na cząsteczkę vs 192 niewiadomych; wymiar efektywny (participation ratio) chmury z i rząd W.
3. Rozkład ruchu „wiązki vs kulki” z WIAZKI.json (Eden, wiazki_koloidy.py, 100 zdjęć): udział zmiany laserów w energii ruchu.
Zapis: TEST_SFERY.json. Uruchomienie: F:/Python/3.13/python.exe -X utf8 test_sfery_laserow.py"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

HERE = Path(__file__).parent
Z = np.load(HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy', mmap_mode='r')
WK = HERE / 'wyniki/wiazki_koloidy'
Wa = np.load(WK / 'W_fc1.npy', mmap_mode='r'); Ba = np.load(WK / 'b_fc1.npy', mmap_mode='r')
kroki = json.load(open(WK / 'META_WIAZKI.json', encoding='utf-8'))['kroki']
gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))
# odwrotność GELU na gałęzi z > 0 (tablica + interpolacja)
xs = np.linspace(0, 12, 200001); gs = gelu(xs)
ginv = lambda z: np.interp(z, gs, xs)
wyn = {}
for t in [0, 101, 300, 805]:                                                       # krok 0, koniec ep. 1, ~ep. 21, ep. 60
    z = np.asarray(Z[t, 0]).astype(np.float64); W = np.asarray(Wa[t]).astype(np.float64); b = np.asarray(Ba[t]).astype(np.float64)
    sv = np.linalg.svd(W, compute_uv=False)
    on = z > 0.02                                                                  # bezpiecznie na gałęzi rosnącej (fp16)
    r2, r2_all, n_on = [], [], on.sum(1)
    rng = np.random.default_rng(0)
    for j in range(600):
        S = np.flatnonzero(on[j])
        if len(S) < 200:
            r2.append(np.nan); continue
        rng.shuffle(S); ucz, test = S[: int(0.8 * len(S))], S[int(0.8 * len(S)):]   # trzymamy 20 % świecących do sprawdzenia
        u, *_ = np.linalg.lstsq(W[ucz], ginv(z[j, ucz]) - b[ucz], rcond=None)
        zp = gelu(W @ u + b)
        inne = np.setdiff1d(np.arange(768), ucz)
        r2.append(1 - ((zp[inne] - z[j, inne]) ** 2).sum() / ((z[j, inne] - z[j, inne].mean()) ** 2).sum())
    zc = z - z.mean(0); ev = np.linalg.eigvalsh(zc.T @ zc)[::-1]; pr = ev.sum() ** 2 / (ev ** 2).sum()
    wyn[str(kroki[t])] = dict(lasery_swiecace_na_czastke_med=float(np.median(n_on)), lasery_swiecace_min=int(n_on.min()),
                              czastek_z_mniej_niz_192=int((n_on < 192).sum()),
                              R2_przewidywania_natezen_spoza_dopasowania_med=float(np.nanmedian(r2)), R2_p10=float(np.nanpercentile(r2, 10)),
                              wymiar_efektywny_chmury_z=float(pr), rzad_W=int((sv > sv[0] * 1e-4).sum()), uwarunkowanie_W=float(sv[0] / sv[-1]))
    print(kroki[t], wyn[str(kroki[t])], flush=True)
w = json.load(open(WK / 'WIAZKI.json', encoding='utf-8'))
roz = {et: {lag: round(v['udzial_wiazek'], 3) for lag, v in d['lagi'].items()} for et, d in w['etapy'].items()}
roz['cala_droga_e1_e60'] = round(w['cala_droga_e1_e60']['udzial_wiazek'], 3)
roz['obrot_laserow_e1_e60_cos'] = round(w['cala_droga_e1_e60']['cos_wiazek'], 3)
roz['kor_sasiadow_tylko_lasery_vs_tylko_kulki_e1_e60'] = [round(w['cala_droga_e1_e60']['kor_wiazki']['sasiedzi'], 3), round(w['cala_droga_e1_e60']['kor_kulki']['sasiedzi'], 3)]
print(json.dumps(roz, ensure_ascii=False))
json.dump(dict(opis=__doc__.split('\n')[0], triangulacja_i_wymiar=wyn, udzial_laserow_w_energii_ruchu=roz), open(HERE / 'wyniki/TEST_SFERY.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
