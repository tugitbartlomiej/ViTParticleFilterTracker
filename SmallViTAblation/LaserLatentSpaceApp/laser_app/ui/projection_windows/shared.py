"""Stałe okienek rzutów: kolory, lista okienek R0–R12, formatowanie liczb, strzałki."""
from __future__ import annotations

import numpy as np


TLO = '#03060a'
S_EKR = 10.0                                         # promień RMS chmury na ekranie (jednostki sceny)
KOL_NARZ, KOL_TLO = (1.0, 0.36, 0.30), (0.56, 0.64, 0.72)
KOL_WYBR, KOL_DOD = (1.0, 1.0, 1.0), (1.0, 0.82, 0.40)
KOL_SILA, KOL_ZGODNA, KOL_PRZECIW = (1.0, 0.62, 0.20), (1.0, 0.78, 0.35), (0.35, 0.68, 1.0)
KOL_BAZ = {'P3': '#e8e8e8', 'A': '#ffb000', 'A_t': '#ff7a45', 'S': '#4af0ff', 'W': '#c264ff', 'D': '#7ee081'}
OKIENKA = (('R0', 'zdjęcie', 'R0. Zdjęcie: ramka GT, ramka modelu i IoU, patche narzędzia (obrys) i grupa wykryta bez etykiet'),
           ('R1', 'A', 'R1. A — mapa klas: oś narzędzie–tło (100 % separacji) + reszta chmury'),
           ('R2', 'S', 'R2. S — siły: 3 kierunki największej energii sił netto bloku 6'),
           ('R3', 'W', 'R3. W — wiązki ważone aktywacją (następca P3)'),
           ('R4', 'B', 'R4. B — rama siły wybranej cząsteczki (lokalna)'),
           ('R5', 'C', 'R5. C — rama Schura: podprzestrzeń niezmiennicza jakobianu (lokalna)'),
           ('R6', 'D', 'R6. D — wędrówka w czasie uczenia'),
           ('R7', 'F', 'R7. F — 7 widoków Fano: każda para kierunków razem raz'),
           ('R8', 'bez rzutu', 'R8. Bez rzutu: miary 192D i wierność widoków'),
           ('R9', 'L', 'R9. L — separacja narzędzia od tła: oś LDA ze skurczem + własne parametry'),
           ('R12', 'kNN 3D', 'R12. kNN na chmurze 3D: kolor tokenu narzędzia = ułamek sąsiadów, którzy też są narzędziem, z liniami do sąsiadów'),
           ('R11', 'miary w czasie', 'R11. Miary separacji w czasie uczenia: Fisher LDA i 192D, kNN PCA3, sylwetka 192D z zerem przestrzennym'),
           ('R10', 'LDA na żywo', 'R10. LDA na żywo: narzędzie vs tło oraz ta sama maska przesunięta w tło (zero przestrzenne)'))
pl = lambda x, n=2: '—' if x is None or not np.isfinite(x) else f'{x:.{n}f}'.replace('.', ',')
def proc(x):
    if x is None or not np.isfinite(x):
        return '—'
    v = 100 * float(x)
    return (f'{v:.1f}' if abs(v) < 10 else f'{v:.0f}').replace('.', ',') + ' %'
def strzalki(O, W, kol):
    """Odcinki O → O + W (n, 3) dla GLLinePlotItem (mode='lines'), kolory wierzchołków i końce (groty)."""
    O = np.asarray(O, np.float32); W = np.asarray(W, np.float32)
    seg = np.empty((2 * len(O), 3), np.float32)
    seg[0::2] = O; seg[1::2] = O + W
    return seg, np.repeat(np.asarray(kol, np.float32), 2, 0), O + W
