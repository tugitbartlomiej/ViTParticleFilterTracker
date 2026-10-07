"""Etap 8 (prośba użytkownika): korekta na wszystkich 768 kanałach, ale każda współrzędna w naturalnym przedziale
swojego kanału, wyznaczonym z prawdziwych obrazów (400 obrazów manifestu przy k1000, 100 losowych tokenów z obrazu).
Trzy poziomy przedziału: [min, max] danych, percentyle 0,1-99,9 i 1-99. Rzut na pudło po każdym kroku Adama (tryb
„zakresy” stanowiska z podmienionymi granicami). Cel detekcja jak w etapie 1. Miary naturalności jak w etapie 5.
Latent po korekcie zapisany do zb_m<i>_<poziom>.npy dla SD (sd_prowadzone.py box:<poziom>). Argumenty m1 m2 ...:
zamiast trzech poziomów przedział [min − m szer, max + m szer] (szer = max − min z danych), plik *_marginesy.json.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty_naturalne')
sys.path.insert(0, str(ROOT / 'app'))
import numpy as np
import torch

import korekta_latentu as KL

MARGINESY = [float(a) for a in sys.argv[1:]]          # np. 0.25 0.5 1.0: margines jako ułamek szerokości min-max
PLIK = 'ETAP8_ZAKRESY_NATURALNE' + ('_marginesy' if MARGINESY else '') + '.json'
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main(n_cel=8):
    st = KL.Stanowisko(dict(KL.checkpointy())['gęsta historia k1000'])
    rows = KL.manifest(); rng = np.random.default_rng(0)
    T = []
    for i in range(len(rows)):
        z = st.przebieg(KL.obraz_manifestu(rows[i])[0])['z']
        T.append(z[torch.from_numpy(rng.choice(600, 100, replace=False)).to(st.dev)])
    T = torch.cat(T)
    mu = T.mean(0); _, S_, Vt = torch.linalg.svd(T - mu, full_matrices=False)
    U = Vt[:64].T; s = S_[:64] / (len(T) - 1) ** .5
    nrm_nat = T.norm(dim=1)

    def nienat(z):
        y = z - mu; p = y @ U
        return dict(poza=float(((y - p @ U.T) ** 2).sum(1).sqrt().mean()), maha=float(((p / s) ** 2).sum(1).sqrt().mean()),
                    norma_tokenu_max=float(z.norm(dim=1).max()), norma_tokenu_sr=float(z.norm(dim=1).mean()))
    poziomy = {'minmax': (0.0, 1.0), 'p0.1': (.001, .999), 'p1': (.01, .99)}
    zakresy = {k: (torch.quantile(T, a, dim=0), torch.quantile(T, b, dim=0)) for k, (a, b) in poziomy.items()}
    if MARGINESY:
        lo0, hi0 = zakresy['minmax']; szer = hi0 - lo0
        zakresy = {f'mm_m{m}': (lo0 - m * szer, hi0 + m * szer) for m in MARGINESY}
    lo_ck, hi_ck = st.h_min.clone(), st.h_max.clone()
    rap = dict(prawdziwe=dict(nienat=nienat(T), norma_tokenu_p99=float(torch.quantile(nrm_nat, .99))),
               szerokosc_przedzialu={k: float((b - a).mean()) for k, (a, b) in zakresy.items()},
               szerokosc_ckpt=float((hi_ck - lo_ck).mean()), wyniki=[])
    log('szerokość przedziału kanału: checkpoint', round(rap['szerokosc_ckpt'], 2), {k: round(v, 3) for k, v in rap['szerokosc_przedzialu'].items()})
    cele = json.loads((OUT / 'ETAP1_CELE.json').read_text(encoding='utf-8'))['cele'][:n_cel]
    W2 = st.blok.mlp.fc2.weight.detach()
    for nazwa, (lo, hi) in zakresy.items():
        st.h_min, st.h_max = torch.minimum(lo, lo_ck), torch.maximum(torch.minimum(hi, hi_ck), lo + 1e-6)
        for c in cele:
            rgb, ram = KL.obraz_manifestu(rows[c['i']])
            z0 = st.przebieg(rgb)['z']
            # punkt startu: obraz może sam wychodzić poza wąski przedział; rzut przy pierwszym kroku go tam wprowadza
            spec = dict(tryb='zakresy', kanaly=np.arange(768), tokeny=0, cel='detekcja', cel_pewnosc=.95, norma='brak',
                        kroki=300, lr=.05, zapis_co=300, ziarno=0)
            w = KL.oblicz_korekte(st, rgb, ram, spec, dict(id=f'm{c["i"]:04d}_nat_{nazwa}'))
            kn = np.load(Path(w['kat']) / 'KONIEC.npz')
            zn = torch.from_numpy(kn['z_baza'] + kn['d_najlepszy']).to(st.dev)
            np.save(OUT / f'zb_m{c["i"]:04d}_{nazwa}.npy', zn.cpu().numpy().astype(np.float32))
            rek = dict(cel=c['i'], poziom=nazwa, iou=(w['przed']['iou_top'], w['najlepszy']['iou_top']),
                       pewnosc=(w['przed']['pewnosc_top'], w['najlepszy']['pewnosc_top']), fp=w['najlepszy']['fp'],
                       strata=(w['przed']['strata'], w['najlepszy']['strata']), norma_dz=float((zn - z0).norm()),
                       W2_zmiana_wzgl=float(((zn - z0) @ W2.T).norm() / (z0 @ W2.T).norm()),
                       nienat_przed=nienat(z0), nienat_po=nienat(zn), niewidoczna=w['najlepszy']['fc2']['udzial_niewidocznej'])
            rap['wyniki'].append(rek)
            log(nazwa, c['i'], 'IoU', [round(x, 3) for x in rek['iou']], 'pewn', [round(x, 3) for x in rek['pewnosc']],
                'poza', round(rek['nienat_po']['poza'], 2), 'maha', round(rek['nienat_po']['maha'], 1), '||dz||', round(rek['norma_dz'], 1),
                'W2', round(rek['W2_zmiana_wzgl'], 2))
            (OUT / PLIK).write_text(json.dumps(rap, indent=1, ensure_ascii=False), encoding='utf-8')
    st.h_min, st.h_max = lo_ck, hi_ck
    pods = {}
    for nazwa in zakresy:
        w = [x for x in rap['wyniki'] if x['poziom'] == nazwa]
        pods[nazwa] = dict(iou=float(np.mean([x['iou'][1] for x in w])), pewnosc=float(np.mean([x['pewnosc'][1] for x in w])),
                           iou_powyzej_05=int(sum(x['iou'][1] >= .5 for x in w)), poza=float(np.mean([x['nienat_po']['poza'] for x in w])),
                           maha=float(np.mean([x['nienat_po']['maha'] for x in w])), norma_dz=float(np.mean([x['norma_dz'] for x in w])),
                           W2=float(np.mean([x['W2_zmiana_wzgl'] for x in w])), n=len(w))
        log(nazwa, {a: round(b, 3) for a, b in pods[nazwa].items()})
    rap['podsumowanie'] = pods
    (OUT / PLIK).write_text(json.dumps(rap, indent=1, ensure_ascii=False), encoding='utf-8')
    log('KONIEC')


if __name__ == '__main__':
    main()
