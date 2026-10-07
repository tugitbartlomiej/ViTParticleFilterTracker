"""Czwarty panel zakładki „Uczenie ViT”: równanie ruchu wybranych cząstek z ich własnej historii.

Dla listy cząstek (patchy) liczone są trzy modele z rownanie_czastki (Langevin, GLE z pamięcią, Koopman),
ich błąd prognozy względem persystencji na krokach testowych oraz rozkład równania GLE na części: bezwładność,
pamięć (jądro K), sprężynę do wolnej średniej i szum. Wykresy (Jung i in. 2017, Vroylandt i in. 2022,
Brunton i in. 2017, Kunin i in. 2021): błąd w funkcji horyzontu, jądro pamięci (regresja i Volterra), rozkład
kroku wokół bieżącego kroku, portret fazowy, autokorelacja i rozkład szumu, VACF danych i modelu, wartości
własne Koopmana. Liczenie w tle; rozkład kroku nadąża za suwakiem uczenia.
"""
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (QComboBox, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QVBoxLayout,
                             QWidget)
from matplotlib.figure import Figure

import analiza_uczenia as AU
import rownanie_czastki as RC
from styl import rola
from uklad import Przeplyw, dodaj
from zakladka_detekcji import PlotnoCiemne

KOLORY = ['#ff9f1c', '#2ec4b6', '#e71d36', '#a06cd5', '#7bd389', '#f15bb5', '#4cc9f0', '#fee440']
STYLE = ['-', '--', ':']
MAKS_CZASTEK = 20


def lista_patchy(tekst):
    tok = []
    for cz in tekst.replace(';', ',').split(','):
        cz = cz.strip()
        try:
            a, _, b = cz.partition('-')
            tok += list(range(int(a), int(b or a) + 1))
        except ValueError:
            continue
    wynik = []
    for t in tok:
        if 0 <= t < 600 and t not in wynik:
            wynik.append(t)
    return wynik[:MAKS_CZASTEK]


class PanelRownania(QWidget):
    def __init__(self, zakladka):
        super().__init__()
        self.z = zakladka
        self.wyniki = {}                      # tok -> {'modele': [mdl0, mdl1, mdl2], 'ocena': [...], 'gle': analiza}
        self.watek = None
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        w = Przeplyw(); lay.addLayout(w)
        self.edit = QLineEdit(); self.edit.setPlaceholderText('np. 255, 300, 45, 512'); self.edit.setMinimumWidth(150)
        dodaj(w, 'Cząstki:', self.edit, 'Numery patchy (wiersz · 30 + kolumna), do 20. Puste: czubek / wybrany, środek\n'
              'narzędzia, środek obrazu i róg tła. Każda cząstka ma swój kolor w przestrzeni makro.')
        b = rola(QPushButton('+ wybrany'), 'narzedzie'); b.setToolTip('Dopisz wybrany patch do listy.')
        b.clicked.connect(self._dodaj_wybrany); w.addWidget(b)
        b = rola(QPushButton('zaznaczone'), 'narzedzie')
        b.setToolTip(f'Wpisuje zaznaczone patche (Ctrl + klik, prostokąt; do {MAKS_CZASTEK}) i od razu liczy równanie\n'
                     'Langevina, GLE i Koopmana osobno dla każdej z tych cząstek, z jej własnej historii.\n'
                     'Kolory jak zaznaczenie na obrazie i w przestrzeni makro. Policzone raz wracają z zapisu.')
        b.clicked.connect(self._zaznaczone); w.addWidget(b)
        self.combo_model = QComboBox(); self.combo_model.addItems(RC.MODELE); self.combo_model.setCurrentIndex(1)
        dodaj(w, 'Model:', self.combo_model,
              'Model prognozy rysowany w przestrzeni makro (wszystkie trzy są liczone i porównane w tabeli):\n'
              '- Langevin: v(t+1) = beta v(t) − kappa (z − m) + szum;\n'
              '- GLE z pamięcią: v(t+1) = suma a_l v(t − l) − kappa (z − m) + c + szum, jedno jądro dla 768 kanałów;\n'
              '- Koopman (Hankel-DMD): liniowy operator na r głównych składowych toru z d opóźnieniami,\n'
              '  reszta ruchu poza tymi składowymi: oscylator.\n'
              'Każdy model widzi tylko przeszłe stany tej cząstki (bez siły paczki, bez wag).')
        self.spin_L = QSpinBox(); self.spin_L.setRange(1, 40); self.spin_L.setValue(10)
        dodaj(w, 'pamięć L', self.spin_L, 'Długość jądra pamięci GLE (kroków). L = 1 to dokładnie oscylator.')
        self.spin_d = QSpinBox(); self.spin_d.setRange(1, 30); self.spin_d.setValue(8)
        dodaj(w, 'Koopman d', self.spin_d, 'Liczba opóźnień w stanie Koopmana (Hankel).')
        self.spin_r = QSpinBox(); self.spin_r.setRange(1, 32); self.spin_r.setValue(8)
        dodaj(w, 'r', self.spin_r, 'Liczba głównych składowych toru cząstki w modelu Koopmana.')
        self.spin_od = QSpinBox(); self.spin_od.setRange(20, 990); self.spin_od.setValue(300)
        self.spin_do = QSpinBox(); self.spin_do.setRange(40, 1000); self.spin_do.setValue(700)
        dodaj(w, 'dopasuj na krokach', self.spin_od, 'Początek okna dopasowania (indeks kroku).')
        dodaj(w, '–', self.spin_do, 'Koniec okna dopasowania; test na krokach od tego miejsca do końca.')
        self.btn = rola(QPushButton('Licz równania'), 'glowny'); self.btn.clicked.connect(self.licz); w.addWidget(self.btn)
        self.combo_szcz = QComboBox(); self.combo_szcz.setMinimumWidth(110)
        dodaj(w, 'Wykresy dla:', self.combo_szcz, 'Której cząstki dotyczą wykresy rozkładu równania.')
        self.etykieta = QLabel('Wpisz cząstki i „Licz równania” (po wczytaniu historii). Około 2 s na cząstkę.')
        self.etykieta.setWordWrap(True); self.etykieta.setTextFormat(self._rich())
        from PyQt6.QtCore import Qt
        self.etykieta.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.etykieta.setContentsMargins(4, 2, 4, 2)
        # równanie i tabela u góry, wykresy pod spodem: dwa osobno przewijane pola z przeciąganym podziałem
        sc_t = QScrollArea(); sc_t.setWidget(self.etykieta); sc_t.setWidgetResizable(True)
        self.fig = Figure(figsize=(6, 13)); self.canvas = PlotnoCiemne(self.fig); self.canvas.setMinimumHeight(1250)
        sc = QScrollArea(); sc.setWidget(self.canvas); sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.podzial = QSplitter(Qt.Orientation.Vertical); self.podzial.addWidget(sc_t); self.podzial.addWidget(sc)
        self.podzial.setChildrenCollapsible(False); self.podzial.setHandleWidth(7)
        self.podzial.setStretchFactor(0, 2); self.podzial.setStretchFactor(1, 3)
        self.podzial.setToolTip('Przeciągnij pasek między tabelą a wykresami, żeby zmienić podział.')
        lay.addWidget(self.podzial, 1)
        self.combo_szcz.currentIndexChanged.connect(lambda *_: self.rysuj())
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(150)
        self._timer.timeout.connect(self._rysuj_rozklad)

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, '_podzielono', False):              # pierwszy raz: tabela 40 %, wykresy 60 %
            self._podzielono = True
            QTimer.singleShot(0, lambda: self.podzial.setSizes(
                [self.podzial.height() * 2 // 5, self.podzial.height() - self.podzial.height() * 2 // 5]))

    @staticmethod
    def _rich():
        from PyQt6.QtCore import Qt
        return Qt.TextFormat.RichText

    def tokeny(self):
        return lista_patchy(self.edit.text())

    def kolor(self, tok):
        zazn = getattr(self.z, 'zaznaczenie', None)
        if zazn is not None and tok in list(zazn):                       # kolor zaznaczenia (obraz, makro, wykresy)
            from matplotlib.colors import to_hex
            from wspolne import paleta_zaznaczenia
            return to_hex(paleta_zaznaczenia(len(zazn))[list(zazn).index(tok)][:3])
        t = self.tokeny()
        return KOLORY[t.index(tok) % len(KOLORY)] if tok in t else '#ffffff'

    def _zaznaczone(self):
        zazn = getattr(self.z, 'zaznaczenie', None)
        if zazn is None:
            self.z._status('Najpierw zaznacz patche: Ctrl + klik na obrazie albo w przestrzeni makro.'); return
        self.edit.setText(', '.join(str(int(t)) for t in list(zazn)[:MAKS_CZASTEK]))
        if len(zazn) > MAKS_CZASTEK:
            self.z._status(f'Panel równania liczy najwyżej {MAKS_CZASTEK} cząstek; wzięte pierwsze {MAKS_CZASTEK} zaznaczone.')
        self.licz()

    def _dodaj_wybrany(self):
        if self.z.token is None:
            return
        t = self.tokeny()
        if self.z.token not in t:
            t.append(self.z.token)
        self.edit.setText(', '.join(map(str, t[:MAKS_CZASTEK])))

    def parametry(self):
        return dict(od=self.spin_od.value(), do=self.spin_do.value(), L=self.spin_L.value(), d=self.spin_d.value(),
                    r=self.spin_r.value())

    # ------------------------------------------------------------ liczenie
    def licz(self):
        z = self.z
        if z.Z is None or self.watek is not None:
            return
        tok = self.tokeny()
        if not tok:
            narz = np.flatnonzero(z.narz)
            tok = [z.token if z.token is not None else int(narz[0]), int(narz[len(narz) // 2]), 315, 45]
            tok = list(dict.fromkeys(tok))
            self.edit.setText(', '.join(map(str, tok)))
        par = self.parametry()
        Z = z.Z
        plik = self._plik()
        from zakladka_filmu import WatekFilmu

        def zadanie(wt):
            zap = self._wczytaj_zapis(plik)
            klucz_par = tuple(sorted(par.items()))
            wynik, nowe = {}, 0
            for i, t in enumerate(tok):
                x = np.asarray(Z[:, t], np.float32)
                modele = [RC.dopasuj(x, m, **par) for m in range(3)]              # szybkie (około 0,6 s razem)
                r = zap['wpisy'].get((t, klucz_par))
                if r is None:                                                    # ocena i rozkład: wolne, zapisywane
                    r = dict(ocena=[RC.ocena(mdl, par['do']) for mdl in modele], gle=RC.analiza_gle(modele[1]))
                    zap['wpisy'][(t, klucz_par)] = r; nowe += 1
                wynik[t] = dict(modele=modele, ocena=r['ocena'], gle=r['gle'])
                wt.postep.emit(int(100 * (i + 1) / len(tok)), f'równanie cząstki {t}: {i + 1}/{len(tok)}')
            zap['ostatnie'] = (list(tok), par)
            if plik is not None:
                try:
                    import pickle
                    tmp = plik.with_suffix('.tmp')
                    tmp.write_bytes(pickle.dumps(zap, protocol=4)); tmp.replace(plik)
                except OSError:
                    pass
            self._z_zapisu = len(tok) - nowe
            return wynik

        self.watek = WatekFilmu(zadanie, self)
        self.watek.postep.connect(lambda k, t: (z.pasek.setValue(k), z._status(t, 3000)))
        self.watek.gotowy.connect(self._gotowe)
        self.watek.blad.connect(lambda t: (setattr(self, 'watek', None), QMessageBox.critical(self, 'Błąd', t[-2000:])))
        self.btn.setEnabled(False)
        self.watek.start()

    # ------------------------------------------------------------ zapis policzonych równań (app/cache_uczenia)
    def _plik(self):
        b = self.z._bieg()
        return None if b is None else AU.plik_wynikow(b, self.z.i_kl, 'rownania', 'pkl')

    def _wczytaj_zapis(self, plik):
        """Zapisane oceny i rozkłady GLE {(patch, parametry): ...} i ostatnio liczona lista; inny odcisk biegu = pusty."""
        znacznik = AU.znacznik_biegu(self.z._bieg())
        try:
            import pickle
            d = pickle.loads(plik.read_bytes())
            if d.get('znacznik') == znacznik:
                return d
        except Exception:
            pass
        return dict(znacznik=znacznik, wpisy={}, ostatnie=None)

    def po_wczytaniu(self):
        """Po wczytaniu historii: ostatnio policzone równania tej klatki wracają same (w tle, z zapisu)."""
        self.wyniki = {}; self.combo_szcz.clear(); self.fig.clear(); self.canvas.draw_idle()
        plik = self._plik()
        if plik is None or not plik.exists():
            return
        ost = self._wczytaj_zapis(plik).get('ostatnie')
        if not ost:
            return
        tok, par = ost
        spiny = dict(od=self.spin_od, do=self.spin_do, L=self.spin_L, d=self.spin_d, r=self.spin_r)
        for k, sp in spiny.items():
            sp.blockSignals(True); sp.setValue(int(par.get(k, sp.value()))); sp.blockSignals(False)
        self.edit.blockSignals(True); self.edit.setText(', '.join(map(str, tok))); self.edit.blockSignals(False)
        self.etykieta.setText('Wczytuję zapisane równania cząstek...')
        self.licz()

    def _gotowe(self, w):
        self.watek = None; self.btn.setEnabled(True)
        self.wyniki = w
        self.combo_szcz.blockSignals(True); self.combo_szcz.clear()
        self.combo_szcz.addItems([f'patch {t}' for t in w]); self.combo_szcz.blockSignals(False)
        self.z._tory = {}
        self.rysuj()
        self.z.idz(self.z.t)
        zz = getattr(self, '_z_zapisu', 0)
        self.z._status(f'Równania {len(w)} cząstek' + (f', {zz} z zapisu w cache_uczenia' if zz else ', zapisane w cache_uczenia')
                       + '.', 8000)

    def model_dla(self, tok):
        """Dopasowany model wybranego typu dla patcha z listy (albo None)."""
        r = self.wyniki.get(tok)
        return None if r is None else r['modele'][self.combo_model.currentIndex()]

    # ------------------------------------------------------------ rysunki
    def _tabela(self):
        hs = (1, 5, 20, 50)
        w = ['<table cellspacing="0" cellpadding="2"><tr><th align="left">cząstka</th><th align="left">model</th>'
             + ''.join(f'<th>h={h}</th>' for h in hs) + '</tr>']
        for t, r in self.wyniki.items():
            for m, o in enumerate(r['ocena']):
                naj = min(r['ocena'], key=lambda q: q.get(1, 9)).get(1)
                w.append(f'<tr><td><span style="color:{self.kolor(t)}">■</span> {t}</td><td>{RC.MODELE[m].split(" (")[0]}</td>'
                         + ''.join(f'<td align="right">{"<b>" if (h == 1 and o.get(h) == naj) else ""}'
                                   f'{o.get(h, float("nan")):.2f}</td>' for h in hs) + '</tr>')
        return ''.join(w) + '</table>'

    def _opis(self, tok):
        r = self.wyniki[tok]; m0, m1, m2 = r['modele']; g = r['gle']
        a = m1['a']
        pam = ' '.join(f'{"+" if a[l] >= 0 else "−"} {abs(a[l]):.2f}·v(t−{l})' for l in range(1, min(len(a), 4)))
        u = g['udzialy']
        lam = m2['lam']; i = int(np.argmax(np.abs(lam)))
        return (f'<b>Patch {tok}</b> (tylko własna historia, dopasowanie na krokach {m1["od"]}–{m1["do"]}):<br>'
                f'Langevin: v(t+1) = {m0["beta"]:.2f}·v(t) − {m0["kappa"]:.3f}·(z − m) + η<br>'
                f'GLE: v(t+1) = {a[0]:.2f}·v(t) {pam}{" …" if len(a) > 4 else ""} − {m1["kappa"]:.3f}·(z − m) + η;'
                f' wyjaśnia {m1["R2"] * 100:.0f} % prędkości<br>'
                f'udział w ruchu: bezwładność {u["bezwładność"] * 100:.0f} %, pamięć {u["pamięć"] * 100:.0f} %, sprężyna '
                f'{u["sprężyna"] * 100:.0f} %, szum η {u["szum"] * 100:.0f} %; szum: kurtoza {g["kurtoza"]:.1f} '
                f'({"grube ogony: rzadkie silne pchnięcia, jak paczki" if g["kurtoza"] > 1 else "prawie gaussowski"}), '
                f'autokorelacja w kroku 1: {g["acf"][1]:.2f}<br>'
                f'Koopman (r = {m2["r"]}, d = {m2["d"]}): największa |λ| = {abs(lam[i]):.3f}, okres '
                f'{(2 * np.pi / abs(np.angle(lam[i]))) if abs(np.angle(lam[i])) > 1e-6 else float("inf"):.0f} kroków<br>'
                f'Błąd prognozy / błąd „bez ruchu” na krokach testowych (mniej = lepiej; 1 = tak źle jak '
                f'założenie, że cząstka stoi). h = na ile kroków uczenia do przodu przewiduje prognoza:<br>'
                + self._tabela())

    def rysuj(self):
        f = self.fig; f.clear()
        if not self.wyniki:
            self.canvas.draw_idle(); return
        i = max(0, self.combo_szcz.currentIndex())
        tok = list(self.wyniki)[min(i, len(self.wyniki) - 1)]
        self.etykieta.setText(self._opis(tok))
        r = self.wyniki[tok]; g = r['gle']; m1, m2 = r['modele'][1], r['modele'][2]
        osie = f.subplots(4, 2)
        f.subplots_adjust(left=.11, right=.97, top=.96, bottom=.04, hspace=.6, wspace=.35)
        self._osie = osie
        ax = osie[0, 0]
        for t, rr in self.wyniki.items():
            for m, o in enumerate(rr['ocena']):
                hs = sorted(o)
                ax.plot(hs, [o[h] for h in hs], STYLE[m], marker='o', ms=3, color=self.kolor(t),
                        lw=1.6 if t == tok else .7, alpha=1 if t == tok else .45)
        ax.axhline(1, color='gray', lw=.7); ax.set_xscale('log')
        ax.set_title('błąd / „bez ruchu” vs horyzont h\nciągła: Langevin, przerywana: GLE,\nkropki: Koopman', fontsize=7.5)
        ax = osie[0, 1]
        n = len(g['K_reg'])
        ax.bar(np.arange(n), g['K_reg'], color=self.kolor(tok), alpha=.7, label='regresja (δ − a_l)')
        ax.plot(np.arange(n), g['K_volterra'], 'w.-', lw=1, label='z korelacji (Volterra)')
        ax.axhline(0, color='gray', lw=.6); ax.legend(fontsize=6, framealpha=.3)
        ax.set_title('jądro pamięci K(s)\n(tarcie z opóźnieniem s)', fontsize=7.5); ax.set_xlabel('opóźnienie s (kroków)', fontsize=7)
        self._rysuj_rozklad(rysuj=False)
        ax = osie[1, 1]
        y, M, v, u1 = m1['y'], m1['M'], m1['v'], m1['u1']
        od, do = m1['od'], m1['do']
        px, pv = (y - M)[od:do] @ u1, v[od:do] @ u1
        ax.plot(px, pv, color=self.kolor(tok), lw=.5, alpha=.7)
        m0 = r['modele'][0]
        X, V = np.meshgrid(np.linspace(px.min(), px.max(), 13), np.linspace(pv.min(), pv.max(), 13))
        ax.quiver(X, V, m0['beta'] * V - m0['kappa'] * X, (m0['beta'] - 1) * V - m0['kappa'] * X,
                  color='#9aa', alpha=.6, width=.004)
        k = self.z.idx[self.z.t] if self.z.Z is not None else od
        ax.plot([(y[k] - M[k]) @ u1], [v[k] @ u1], 'o', color='white', ms=6)
        ax.set_title('portret fazowy (1. składowa toru)\nstrzałki: pole oscylatora', fontsize=7.5)
        ax.set_xlabel('z − m', fontsize=7); ax.set_ylabel('v', fontsize=7)
        ax = osie[2, 0]
        ax.bar(np.arange(len(g['acf'])), g['acf'], color=self.kolor(tok), alpha=.8)
        ax.axhspan(-g['pasmo'], g['pasmo'], color='gray', alpha=.3)
        ax.set_title('autokorelacja szumu η\n(szare: pasmo szumu białego)', fontsize=7.5)
        ax = osie[2, 1]
        e = g['eta'].reshape(-1) / max(g['sigma'], 1e-12)
        ax.hist(e, bins=80, range=(-6, 6), density=True, color=self.kolor(tok), alpha=.75)
        xs = np.linspace(-6, 6, 200); ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'w', lw=1)
        ax.set_yscale('log'); ax.set_ylim(1e-4, 1)
        ax.set_title(f'rozkład szumu η / σ (biała: Gauss)\nkurtoza {g["kurtoza"]:.1f}', fontsize=7.5)
        ax = osie[3, 0]
        ax.plot(g['vacf'], 'o-', ms=3, color=self.kolor(tok), label='dane')
        ax.plot(g['vacf_model'], 'w--', lw=1, label='model GLE + szum biały')
        ax.axhline(0, color='gray', lw=.6); ax.legend(fontsize=6, framealpha=.3)
        ax.set_title('autokorelacja prędkości (VACF)', fontsize=7.5); ax.set_xlabel('opóźnienie (kroków)', fontsize=7)
        ax = osie[3, 1]
        th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), color='gray', lw=.7)
        lam = m2['lam']
        ax.scatter(lam.real, lam.imag, s=8, color=self.kolor(tok))
        ax.set_aspect('equal'); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_title('wartości własne Koopmana\n(koło jednostkowe)', fontsize=7.5)
        for a in f.axes:
            a.tick_params(labelsize=6)
        self.canvas.draw_idle()

    def przy_kroku(self):
        """Wołane przy zmianie kroku: rozkład kroku wokół bieżącego kroku (z opóźnieniem, żeby nie dławić odtwarzania)."""
        if self.wyniki and self.isVisible():
            self._timer.start()

    def _rysuj_rozklad(self, rysuj=True):
        if not self.wyniki or getattr(self, '_osie', None) is None:
            return
        i = max(0, self.combo_szcz.currentIndex())
        tok = list(self.wyniki)[min(i, len(self.wyniki) - 1)]
        g = self.wyniki[tok]['gle']['rozklad']
        k = int(self.z.idx[self.z.t]) if self.z.Z is not None else 0
        ax = self._osie[1, 0]; ax.clear()
        t = g['t']; w = (t > k - 40) & (t <= k)
        x = t[w] + 1
        dol_p = np.zeros(w.sum()); dol_u = np.zeros(w.sum())
        for nazwa, klucz, kol in (('bezwładność', 'bezwladnosc', '#4cc9f0'), ('pamięć', 'pamiec', '#a06cd5'),
                                  ('sprężyna', 'sprezyna', '#7bd389'), ('szum η', 'szum', '#e71d36')):
            val = g[klucz][w]
            dol = np.where(val >= 0, dol_p, dol_u)
            ax.bar(x, val, bottom=dol, color=kol, width=.85, label=nazwa)
            dol_p = dol_p + np.clip(val, 0, None); dol_u = dol_u + np.clip(val, None, 0)
        ax.plot(x, g['v'][w], 'w.-', lw=.8, ms=3, label='rzeczywisty krok')
        ax.legend(fontsize=5, ncol=3, framealpha=.3, loc='upper left')
        ax.set_title(f'rozkład kroku v(t) na części GLE\n(1. składowa toru), kroki {max(0, k - 39)}–{k}', fontsize=7.5)
        ax.tick_params(labelsize=6)
        if rysuj:
            self.canvas.draw_idle()
