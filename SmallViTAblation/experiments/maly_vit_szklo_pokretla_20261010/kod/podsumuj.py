"""Podsumowanie analiz biegów z pokrętłami w jednej linii na analizę (odciski S1, heterogeniczność, nowe testy „substancji”).
python podsumuj.py [wzorzec_nazwy_biegu ...]   (domyślnie wszystkie wyniki/*/ANALIZA*.json)"""
import glob
import json
import os
import sys

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wz = sys.argv[1:]
f = lambda v, d=3: ('   -' if v is None else (f'{v:.{d}f}' if isinstance(v, (int, float)) and abs(v) < 1e6 else str(v)))
for p in sorted(glob.glob(os.path.join(EXP, 'wyniki', '*', 'ANALIZA*.json'))):
    b = os.path.basename(os.path.dirname(p)); n = os.path.basename(p)[len('ANALIZA'):-len('.json')]
    if wz and not any(w in b or w in n for w in wz):
        continue
    j = json.load(open(p, encoding='utf-8'))
    S1 = j['S1']; h = j['heterogenicznosc_CR']; oc = j.get('oceny') or [{}]
    linia = (f"{b + n:<36} Lind {S1['lindemann_shell']:.3f} τ1e {f(S1.get('tau_1e'), 0):>5} β {f(S1.get('beta'), 2)} f_q {S1['f_q']:.2f} uc10 {S1['uciekajacy_tau10']:.3f} "
             f"E768 {j['energia_768']['E_300_400']:.3g} | C1 {h['szczyt']['C1']:.3f}@{h['szczyt']['tau']} pow {[round(c, 2) for c in h['szczyt']['C_powloki']]} "
             f"cross {f(max(h['C_cross']) if h['C_cross'] else None)}")
    if 'trwalosc_ruchliwosci_CR' in j:
        t = j['trwalosc_ruchliwosci_CR']; kk = j['korelacja_krzyzowa_blok'].get('werdykt', {}); kp = j['korelacja_krzyzowa_populacja'].get('werdykt', {})
        rv = j['rozklad_wariancji']['udzial']; rp = j['rozbieganie_par']
        linia += (f" | trw lag1 {f(t.get('spearman_wzgledny_po_lagu', [None])[0], 2)} ost {f(t.get('pierwsze_ostatnie_wzgledne'), 2)} zdj {f(t.get('pierwsze_ostatnie_zdjec'), 2)}"
                  f" | kk sąs0 {f(kk.get('C_sasiedzi_0'))} min {f(kk.get('C_sasiedzi_min'))} los0 {f(kk.get('C_losowe_zdjecie_0'))} | pop sąs0 {f(kp.get('C_sasiedzi_0'))} los0 {f(kp.get('C_losowe_zdjecie_0'))} inne0 {f(kp.get('C_inne_zdjecia_0'))}"
                  f" | E τ1: wsp {rv['wspolny'][0]:.2f} zdj {rv['zdjecie'][0]:.2f} sąs {rv['sasiedztwo'][0]:.2f} wł {rv['wlasny'][0]:.2f}"
                  f" | rozb r2k {f(rp['r2_wzgledne'][-1] if rp['r2_wzgledne'] else None, 2)} wykł {f(rp.get('wykladnik_10_1000'), 2)}")
    linia += f" | ocena {f(oc[-1].get('mAP50'), 3)}"
    print(linia)
