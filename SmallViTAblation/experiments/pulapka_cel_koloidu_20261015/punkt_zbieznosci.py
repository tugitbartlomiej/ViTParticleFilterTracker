"""Do czego zbiega pułapka uczenia: ekstrapolacja środka pułapki m_t = m_∞ + A·t^−α z części uczenia i sprawdzenie na końcu.

Środek pułapki m_t = średnia ruchoma stanu koloidu (okno 21 checkpointów co 100 kroków; drgania OU się uśredniają).
Dla odcięcia t_c dopasowujemy α (siatka) oraz m_∞, A (najmniejsze kwadraty, osobno w każdym z 192 wymiarów) na krokach ≤ t_c,
potem porównujemy prognozę m_∞ z prawdziwym środkiem na końcu uczenia m_T:
  błąd względny = |m_∞ − m_T| / |m(t_c) − m_T|  (1 = nie lepiej niż „zostaje tam, gdzie jest”, 0 = trafiona meta).
Zapis: outputs/pulapka_cel_koloidu_20261015/PUNKT_ZBIEZNOSCI.md
"""
from pathlib import Path

import numpy as np

KAT = Path(__file__).resolve().parents[2] / 'outputs' / 'pulapka_cel_koloidu_20261015'
f = np.load(KAT / 'pomiar.npz')
kr = f['kroki']; wyb = list(f['wybrane']); X = f['X_wyb']
reg = np.flatnonzero((kr >= 1100) & (np.r_[np.diff(kr), 0] == 100)); reg = np.r_[reg, reg[-1] + 1]
t = kr[reg].astype(float); Xr = X[reg]
h = 10
m = np.stack([Xr[max(0, i - h):i + h + 1].mean(0) for i in range(len(reg))])     # środek pułapki
mT = m[-h - 1]                                                                     # środek na końcu (pełne okno)
linie = []


def pisz(s=''):
    print(s); linie.append(s)


pisz('# Do czego zbiega pułapka uczenia: prognoza punktu końcowego z części uczenia')
pisz('\nm(t) = m_∞ + A·t^−α dopasowane na krokach ≤ t_c; błąd względny = |m_∞ − m_T| / |m(t_c) − m_T| (1 = brak zysku z prognozy).')
for nazwa, kol in (('226', [wyb.index(226)]), ('w ramce (średnio)', list(range(len(wyb))))):
    pisz(f'\n## {nazwa}')
    for tc in (8000, 16000, 32000):
        s = (t >= 2000) & (t <= tc)
        najl = None
        for a in np.linspace(0.1, 1.5, 29):
            B = np.stack([np.ones(s.sum()), t[s] ** -a], 1)
            res = 0.0; prog = []
            for k in kol:
                C, r_, *_ = np.linalg.lstsq(B, m[s, k], rcond=None)
                res += float(((B @ C - m[s, k]) ** 2).sum()); prog.append(C[0])
            if najl is None or res < najl[0]:
                najl = (res, a, np.array(prog))
        _, a, minf = najl
        ic = np.flatnonzero(s)[-1]
        bl = [np.linalg.norm(minf[i] - mT[k]) / max(np.linalg.norm(m[ic, k] - mT[k]), 1e-30) for i, k in enumerate(kol)]
        # ile drogi od t_c do końca prognoza „przewidziała” (rzut na kierunek prawdziwego ruchu)
        cz = [float((minf[i] - m[ic, k]) @ (mT[k] - m[ic, k]) / max(np.linalg.norm(mT[k] - m[ic, k]) ** 2, 1e-30)) for i, k in enumerate(kol)]
        pisz(f'- t_c = {tc}: α = {a:.2f}; błąd względny prognozy {np.median(bl):.2f}; prognoza pokrywa {np.median(cz):.0%} '
             f'prawdziwej drogi do końca (w jej kierunku)')
(KAT / 'PUNKT_ZBIEZNOSCI.md').write_text('\n'.join(linie) + '\n', encoding='utf-8')
print('zapisano', KAT / 'PUNKT_ZBIEZNOSCI.md')
