"""Czy uwaga ściąga cząsteczki do środka chmury, a MLP je rozpycha? (pytanie z 2026-10-12, zakładka „Lasery i koloidy”)
Dla każdego podkroku przejścia w przód (6 bloków × uwaga, MLP) na surowych stanach strumienia rezydualnego x_i (600 tokenów, 192D):
  - rozrzut chmury r = średnia |x_i − x̄| (x̄ = średnia tokenów w tym stanie); stosunek r_po / r_przed (< 1: chmura się kurczy);
  - składowa promieniowa przesunięcia względem środka: ⟨(Δx_i − Δx̄) · u_i⟩ / r_przed, u_i = (x_i − x̄)/|x_i − x̄| (< 0: ciągnie do środka);
  - udział wspólnego przesunięcia ‖Δx̄‖² / ⟨‖Δx_i‖²⟩ (część podkroku, która przesuwa całą chmurę naraz);
  - średni kosinus par tokenów (względem środka) przed i po (rośnie: tokeny bardziej podobne, „wygładzanie” uwagi).
Checkpointy: 40 równo w indeksach historii 35 000 (0–65 640), zdjęcia: TRAIN 0 i 3 zdjęcia z app/zdjeciatestowe.
Wynik: KURCZENIE_UWAGA_MLP.json. Uruchomienie (z katalogu app): F:/Python/3.13/python.exe -X utf8 ../outputs/porownanie_szklo_20261010/film_krysztal/analiza_kurczenie_uwaga_mlp.py"""
import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[3] / 'app'
sys.path.insert(0, str(APP))
import lasery_dane as LD  # noqa: E402
from historie_checkpointow import historia  # noqa: E402

h = historia('35k')
kroki = h.kroki()
wyb = [kroki[i] for i in np.linspace(0, len(kroki) - 1, 40).round().astype(int)]
zdjecia = [LD.ZDJECIE_FILMU] + sorted((APP / 'zdjeciatestowe').glob('*.jpg'))[1:8:3]
NAZWY = [f'b{b + 1}_{"uwaga" if k % 2 == 0 else "MLP"}' for b in range(6) for k in (0, 1)]
wyn = {n: dict(stosunek=[], promien=[], wspolne=[], cos_przed=[], cos_po=[]) for n in NAZWY}
wiersze = []
for zd in zdjecia:
    rgb, _ = LD.wczytaj_zdjecie(zd)
    for k in wyb:
        d = LD.podkroki(h.wagi, k, rgb)
        Xc = d['xn'] * d['rms'][:, None, None]                                  # (13, 600, 192) względem środka chmury
        r = d['rms']
        U = Xc / np.maximum(np.linalg.norm(Xc, axis=2, keepdims=True), 1e-12)
        for s in range(12):
            dX = Xc[s + 1] - Xc[s]                                             # przesunięcie względem środka (wspólne usunięte)
            dX_sur = d['pos'][s + 1] - d['pos'][s]                              # tylko do udziału wspólnego (rzut 3D surowy, przybliżenie)
            promien = float((dX * U[s]).sum(1).mean() / r[s])
            Un = U[s]; cp = float((Un @ Un.T).mean()); Un2 = U[s + 1]; cpo = float((Un2 @ Un2.T).mean())
            wsp = float(np.linalg.norm(dX_sur.mean(0)) ** 2 / max((dX_sur ** 2).sum(1).mean(), 1e-12))
            w = wyn[NAZWY[s]]
            w['stosunek'].append(float(r[s + 1] / r[s])); w['promien'].append(promien); w['wspolne'].append(wsp)
            w['cos_przed'].append(cp); w['cos_po'].append(cpo)
            wiersze.append(dict(zdjecie=zd.name, krok=int(k), podkrok=NAZWY[s], stosunek=float(r[s + 1] / r[s]), promien=promien))
    print('zdjęcie', zd.name, 'gotowe', flush=True)

podsum = {}
for n, w in wyn.items():
    st = np.array(w['stosunek']); pr = np.array(w['promien'])
    podsum[n] = dict(stosunek_mediana=float(np.median(st)), stosunek_p10_p90=[float(np.percentile(st, 10)), float(np.percentile(st, 90))],
                     kurczy_odsetek=float((st < 1).mean()), promien_mediana=float(np.median(pr)), do_srodka_odsetek=float((pr < 0).mean()),
                     wspolne_mediana=float(np.median(w['wspolne'])), cos_przed=float(np.median(w['cos_przed'])), cos_po=float(np.median(w['cos_po'])))
    print(f'{n:12s} r_po/r_przed {podsum[n]["stosunek_mediana"]:.3f} (kurczy w {podsum[n]["kurczy_odsetek"]:.0%}), '
          f'promieniowo {podsum[n]["promien_mediana"]:+.3f} (do środka w {podsum[n]["do_srodka_odsetek"]:.0%}), '
          f'cos par {podsum[n]["cos_przed"]:.3f} → {podsum[n]["cos_po"]:.3f}')
uw = [n for n in NAZWY if 'uwaga' in n]; ml = [n for n in NAZWY if 'MLP' in n]
razem = dict(uwaga_stosunek=float(np.median([v for n in uw for v in wyn[n]['stosunek']])), mlp_stosunek=float(np.median([v for n in ml for v in wyn[n]['stosunek']])),
             uwaga_kurczy=float(np.mean([v < 1 for n in uw for v in wyn[n]['stosunek']])), mlp_kurczy=float(np.mean([v < 1 for n in ml for v in wyn[n]['stosunek']])))
# w czasie uczenia: krok 0 vs reszta
for etap, war in (('krok_0', lambda k: k == 0), ('epoki_1_60', lambda k: k >= 1094)):
    for typ, lista in (('uwaga', uw), ('mlp', ml)):
        v = [w['stosunek'] for w in wiersze if war(w['krok']) and w['podkrok'] in lista]
        razem[f'{typ}_stosunek_{etap}'] = float(np.median(v)) if v else None
print('razem', json.dumps(razem, ensure_ascii=False))
json.dump(dict(opis=__doc__.split('\n')[0], checkpointy=wyb, zdjecia=[z.name for z in zdjecia], podkroki=podsum, razem=razem, wiersze=wiersze),
          open(Path(__file__).with_name('KURCZENIE_UWAGA_MLP.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
