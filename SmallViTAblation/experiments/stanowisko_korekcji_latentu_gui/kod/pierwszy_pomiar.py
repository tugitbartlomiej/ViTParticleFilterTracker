"""Pierwszy pomiar stanowiska korekcji latentu: po jednym działającym pomiarze każdego etapu (SPECYFIKACJA, sekcja 8).

1. Import historycznych par przed/po (6 obrazów manifestu): część niewidoczna dla fc2, naruszenie zakresów z checkpointu.
2. Nowa korekta oracle (zakresy, J128, wszystkie tokeny, Adam 0,3, 200 kroków, zapis co krok) przy k1000 i k500
   dla 3 obrazów, z losowymi kontrolami.
3. Ruch uczenia tych obrazów (101 checkpointów) wobec korekty; modele dynamiki na torze korekty i torze uczenia.
4. Gałęzie uczenia w kroku 500: waga obrazu w paczce (ε = 0,5, liniowość), 3 przyszłe sekwencje, deterministycznie,
   oraz poziom szumu bez determinizmu; wzrost zaburzeń wag.
5. Obserwator Kalmana na tokenach narzędzia klatki próbnej 0 (co krok, 1001 kroków).
Uruchomienie z SmallViTAblation: F:/Python/python.exe experiments/stanowisko_korekcji_latentu_gui/kod/pierwszy_pomiar.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'app'))
import numpy as np
import torch

import dynamika_korekty as DK
import galezie_uczenia as GU
import korekta_latentu as KL
from stan_uczenia import Odtworzenie

OBRAZY_HIST = [0, 5, 60, 150, 250, 399]
OBRAZY = [5, 60, 250]
t0 = time.time()
kat = KL.WYJSCIE / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pierwszy_pomiar"
kat.mkdir(parents=True)
W = dict(opis=__doc__, kat=str(kat))
ck = dict(KL.checkpointy())
rows = KL.manifest()


def zap():
    KL._json(kat / 'PODSUMOWANIE.json', W)


# 1 ------------------------------------------------------------------------------------------------
st = KL.Stanowisko(ck['gęsta historia k1000'])
W['fc2'] = dict(rzad=st.fc2['rzad'], jadro=st.fc2['jadro'], tol=st.fc2['tol'],
                J128={k: v for k, v in st.fc2_podzbioru(KL.kanaly_j128()).items() if k != 'S'})
W['historia'] = {}
for i in OBRAZY_HIST:
    h = KL.import_historyczny(st, i)
    W['historia'][i] = dict(iou=(h['przed']['iou_top'], h['po']['iou_top']), strata=(h['przed']['strata'], h['po']['strata']),
                            udzial_niewidocznej=h['fc2']['udzial_niewidocznej'], poza_zakresami=h['naruszenia']['ulamek'],
                            max_przekroczenie=h['naruszenia']['max'], ostrzezenia=h['ostrzezenia'],
                            w_bazie_uczenia=h['w_bazie_uczenia'])
    print('hist', i, W['historia'][i], flush=True)
W['ap_historyczne'] = KL.ap_historyczne()
zap()

# 2-3 ------------------------------------------------------------------------------------------------
W['korekty'] = {}
for kk in ('k1000', 'k0500'):
    s = KL.Stanowisko(ck[f'gęsta historia {kk}'])
    for i in OBRAZY:
        rgb, ram = KL.obraz_manifestu(rows[i])
        spec = dict(tryb='zakresy', kanaly=KL.kanaly_j128(), tokeny=0, cel='strata', norma='brak', lam=0.0, kroki=200,
                    lr=0.3, optymalizator='adam', ziarno=0, zapis_co=1)
        w = KL.oblicz_korekte(s, rgb, ram, spec, dict(id=f'm{i:04d}', zrodlo='manifest J128', indeks=i))
        tor = KL.tor_uczenia(rgb, ram)
        baza = s.przebieg(rgb)
        sp = KL.spec_z_zapisu(s, w['konfiguracja'], ram)
        kn = np.load(Path(w['kat']) / 'KONIEC.npz')
        with torch.no_grad():
            zp, _, _ = s.wyjscie(torch.from_numpy(kn['d_najlepszy']).to(s.dev), baza, sp)
        dz = (zp[0] - baza['z']).cpu().numpy()
        por = KL.porownaj_z_uczeniem(tor, s.krok, dz)
        ic = int(np.flatnonzero(tor['kroki'] == s.krok)[0])
        _, D = KL.wczytaj_trajektorie(w['kat'], 'd'); _, G = KL.wczytaj_trajektorie(w['kat'], 'g')
        X = D.reshape(len(D), -1)
        dyn_k = DK.ocen_modele(X)
        wej = DK.porownaj_wejscie(X, DK.krok_adama(G.reshape(len(G), -1), .3))
        Xu = tor['z'][:, :, KL.kanaly_j128()].reshape(len(tor['kroki']), -1)
        dyn_u = DK.ocen_modele(Xu)
        rek = dict(kat=w['kat'], przed=w['przed'], najlepszy={k: w['najlepszy'][k] for k in (
            'krok', 'strata', 'iou_top', 'pewnosc_top', 'fp', 'norma_dz', 'fc2', 'naruszenia')},
            kontrole=[{k: c[k] for k in ('rodzaj', 'ziarno', 'strata', 'iou_top', 'norma_d')} for c in w['kontrole']],
            sprawdzenia=w['sprawdzenia'],
            porownanie=dict(cos_predkosci_przed=por.get('cos_predkosci_przed'),
                            cos_przeszlosc_srednio=float(np.nanmean(por['cos'][:ic])) if ic else None,
                            cos_przyszlosc=({int(k): float(c) for k, c in zip(tor['kroki'][ic + 1:], por['cos'][ic + 1:])
                                             if int(k) % 100 == 0} if ic + 1 < len(tor['kroki']) else None),
                            alfa_przyszlosc=({int(k): float(c) for k, c in zip(tor['kroki'][ic + 1:], por['alfa'][ic + 1:])
                                              if int(k) % 100 == 0} if ic + 1 < len(tor['kroki']) else None),
                            iou_sieci_k1000=tor['miary'][-1]['iou_top'], iou_sieci_kc=tor['miary'][ic]['iou_top']),
            dynamika_korekty=dyn_k['tabela'], wejscie_optymalizatora=wej, dynamika_uczenia=dyn_u['tabela'],
            parametry_korekty=dyn_k['parametry'], parametry_uczenia=dyn_u['parametry'])
        W['korekty'][f'{kk}_m{i:04d}'] = rek
        print(kk, i, rek['przed']['iou_top'], '->', rek['najlepszy']['iou_top'], rek['porownanie'], flush=True)
        zap()

# 4 ------------------------------------------------------------------------------------------------
b = GU.bieg_gestej_historii(); od = Odtworzenie(b); od.przygotuj()
X, R, Mk, N = GU.sondy_gestej_historii(b)
rgb, ram = KL.obraz_manifestu(rows[5])
G = GU.Galezie(od, np.concatenate([X, rgb[None]]), R + [ram], np.concatenate([Mk, KL.maski_gt(ram)['obiekt'][None]]),
               N + ['wybrany obraz m0005'])
hs = (1, 2, 5, 10, 20)
W['galezie'] = {}
for det in (True, False):
    g = GU.licz_odpowiedz(G, 500, dict(rodzaj='waga', sloty=[0], eps_slotow=[0.5]), H=20, horyzonty=hs,
                          przyszlosci=(None, 1, 2), liniowosc=True, deterministyczne=det,
                          nazwa='waga_det' if det else 'waga_niedet')
    W['galezie'][f'waga_deterministycznie_{det}'] = dict(kat=g['kat'], podsumowanie={h: dict(
        odpowiedz_dz_srednio=float(np.mean(z['odpowiedz_dz']['srednia'])), szum_dz_srednio=float(np.mean(z['szum_dz']['srednia'])),
        odpowiedz_dstrata_srednio=float(np.mean(z['odpowiedz_dstrata']['srednia'])),
        szum_dstrata_srednio_abs=float(np.mean(np.abs(z['szum_dstrata']['srednia']))),
        liniowosc=z.get('liniowosc')) for h, z in g['podsumowanie'].items()})
    print('galezie', det, W['galezie'][f'waga_deterministycznie_{det}']['podsumowanie'], flush=True)
    zap()
x1 = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255
c = [t.cuda() for t in od.M.cele([ram], 20, 30)]
g = GU.licz_odpowiedz(G, 500, dict(rodzaj='podmiana', sloty=[0], x=x1, cel=c), H=20, horyzonty=hs, przyszlosci=(None, 1, 2),
                      liniowosc=False, nazwa='podmiana')
W['galezie']['podmiana_slotu_0_na_m0005'] = dict(kat=g['kat'], podsumowanie={h: dict(
    odpowiedz_dz_wybrany=float(z['odpowiedz_dz']['srednia'][-1]), odpowiedz_dstrata_wybrany=float(z['odpowiedz_dstrata']['srednia'][-1]),
    odpowiedz_dstrata_wybrany_min_max=(float(z['odpowiedz_dstrata']['min'][-1]), float(z['odpowiedz_dstrata']['max'][-1])),
    odpowiedz_diou_wybrany=float(z['odpowiedz_diou']['srednia'][-1]),
    odpowiedz_dstrata_12_klatek=float(np.mean(z['odpowiedz_dstrata']['srednia'][:-1]))) for h, z in g['podsumowanie'].items()})
print('podmiana', W['galezie']['podmiana_slotu_0_na_m0005'], flush=True)
wz = GU.licz_wzrost(G, 500, skale=(1e-4, 1e-3, 1e-2), H=20, horyzonty=hs, przyszlosci=(None, 1))
W['wzrost'] = dict(kat=wz['kat'], wyniki=wz['wyniki'])
print('wzrost', wz['wyniki'], flush=True)
zap()

# 5 ------------------------------------------------------------------------------------------------
import analiza_uczenia as AU
Z = AU.wczytaj_klatke(b, 0)
op = AU.opis_biegu(b)
Xs = np.asarray(Z[:, op['narzedzie'][0]], np.float32)
W['obserwator'] = {}
for u, co in ((1.0, 1), (0.25, 1), (0.1, 1), (0.25, 5), (0.25, 20)):
    k = DK.obserwator(Xs, r=6, ulamek_tokenow=u, co_ile=co)
    W['obserwator'][f'tokeny{u}_co{co}'] = dict(
        blad_modow={n: float(np.nanmean(v)) for n, v in k['blad_modow'].items()},
        blad_latentu={n: float(np.mean(v)) for n, v in k['blad_latentu'].items()}, pokrycie_90=k['pokrycie_90'],
        obserwowalnosc=k['obserwowalnosc']['rzad'], koszt=k['koszt']['lacznie'], wariancja_r_modow=k['wariancja_r_modow'])
    print('kalman', u, co, W['obserwator'][f'tokeny{u}_co{co}'], flush=True)
W['czas_s'] = time.time() - t0
zap()
print('KONIEC', kat, W['czas_s'])
