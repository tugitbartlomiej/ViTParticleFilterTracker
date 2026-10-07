"""Zakładka „Pchnięcie obrazów”: jak pojedynczy obraz w paczce przesuwa cząstki sondy w jednym kroku uczenia.

Łączy łańcuch użytkownika „sztuczny latent -> obrazy SD -> prawdziwe obrazy odszukane dla SD (MSE latentu, DINO,
konwerter latent -> DINO)” z równaniem ruchu cząstek: dla każdego kandydata liczy dokładne pchnięcie jednego kroku
AdamW (pchniecie.py) przy stanie odtworzonym z ziarna (stan_uczenia.py, paczki_uczenia.py) i rozkłada je na:
zmianę straty detekcji sondy (czy pomaga detekcji), zmianę siły tożsamości (metoda wektora tożsamości) i kosinus do
stanu końcowego (tylko wyjaśnienie: używa przyszłości). Sprawdzian: przewidziany krok sondy wobec rzeczywistego z gęstej
historii (98 % na krokach 100, 500, 900) i suma pchnięć 32 obrazów paczki wobec pchnięcia całej paczki (99 %).
"""
import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QLabel, QProgressBar, QPushButton, QSpinBox, QSplitter, QVBoxLayout,
                             QWidget)
from matplotlib.figure import Figure

from styl import rola
from uklad import Przeplyw, dodaj
from zakladka_detekcji import PlotnoCiemne

KOL_ZRODLA = {'paczka': '#d0d0d0', 'baza': '#7f9fbf', 'odszukane_sd': '#2ca02c', 'odszukane_kontrola': '#8c564b',
              'sd': '#ff7f0e', 'dino_sd': '#e377c2', 'konwerter_sd': '#17becf'}
KROTKIE = {'paczka': 'paczka', 'baza': 'baza', 'odszukane_sd': 'odsz. SD', 'odszukane_kontrola': 'losowa',
           'sd': 'SD', 'dino_sd': 'DINO', 'konwerter_sd': 'konwerter'}
NAZWY_ZRODEL = {'paczka': 'paczka tego kroku', 'baza': 'baza uczenia', 'odszukane_sd': 'odszukane dla SD (MSE latentu)',
                'odszukane_kontrola': 'losowa 32 (kontrola)', 'sd': 'obrazy SD (pseudo-etykieta)',
                'dino_sd': 'DINO dla obrazów SD', 'konwerter_sd': 'konwerter latent -> DINO'}
MIARY = [('strata', 'zmiana straty sondy (ujemna = pomaga detekcji)', 'RdBu'),
         ('tozsamosc', 'zmiana siły tożsamości narzędzia', 'RdBu_r'),
         ('do_konca', 'kosinus do stanu końcowego (wyjaśnienie, używa przyszłości)', 'RdBu_r'),
         ('dlugosc', 'długość pchnięcia', 'viridis')]
WYNIKI = Path(__file__).resolve().parent.parent / 'experiments' / 'maly_vit_bezwladnosc_a_ped_adamw' / 'wyniki'


def _ramki_sond(bieg):
    import analiza_uczenia as AU
    out = []
    for o in AU._json(Path(bieg['dane']) / 'images.json', []):
        r = []
        for poly in o.get('cystotome_seg') or []:
            p = np.asarray(poly, np.float32).reshape(-1, 2) * (480 / 720)
            r.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
        out.append(r)
    return out


def _miniatura(x):
    return np.ascontiguousarray(np.asarray(x)[::5, ::5])


class ZakladkaPchniecia(QWidget):
    def __init__(self, okno, uczenie):
        super().__init__()
        self.okno, self.u = okno, uczenie
        self.od = self.P = None
        self._cache = {}
        self.wynik = None; self.wybrany = 0; self.watek = None
        lay = QVBoxLayout(self)
        w = Przeplyw(); lay.addLayout(w)
        self.chk_sledz = QCheckBox('krok i klatka z zakładki Uczenie'); self.chk_sledz.setChecked(True)
        self.chk_sledz.setToolTip('Bierze bieżący krok uczenia, klatkę próbną i zaznaczone patche z zakładki „Uczenie ViT”.')
        w.addWidget(self.chk_sledz)
        self.spin_k = QSpinBox(); self.spin_k.setRange(0, 999)
        dodaj(w, 'krok', self.spin_k, 'Krok uczenia k: pchnięcie aktualizacji k -> k+1 (stan AdamW odtworzony z ziarna).')
        self.spin_sonda = QSpinBox(); self.spin_sonda.setRange(0, 11)
        dodaj(w, 'klatka sondy', self.spin_sonda, 'Która z 12 klatek gęstej historii jest sondą (jej cząstki pchamy).')
        self.combo_tok = QComboBox(); self.combo_tok.addItems(['tokeny narzędzia', 'zaznaczone w zakładce Uczenie', 'wszystkie 600'])
        dodaj(w, 'mierz na', self.combo_tok, 'Na których cząstkach sondy liczyć siłę tożsamości, kosinus i długość '
                                              '(zmiana straty zawsze z całej sondy).')
        w.addWidget(QLabel('Źródła:'))
        self.chk_zr = {}
        for k, opis, wl in (('baza', 'baza', True), ('odszukane_sd', 'odszukane dla SD', True), ('sd', 'obrazy SD', True),
                            ('dino_sd', 'DINO dla SD', True), ('konwerter_sd', 'konwerter -> DINO', False)):
            c = QCheckBox(opis); c.setChecked(wl); w.addWidget(c); self.chk_zr[k] = c
        self.chk_zr['konwerter_sd'].setToolTip('Pierwszy raz około minuty: latent obrazów SD z ViT k1000, konwerter na cechy '
                                               'DINO, najbliższe z 35 000 klatek i odczyt z archiwów ZIP.')
        self.spin_baza = QSpinBox(); self.spin_baza.setRange(4, 1500); self.spin_baza.setValue(48)
        dodaj(w, 'baza N', self.spin_baza, 'Ile losowych klatek bazy (zawsze te same dla danego N).')
        self.combo_miara = QComboBox(); self.combo_miara.addItems([m[1] for m in MIARY])
        dodaj(w, 'sortuj / mapa', self.combo_miara, 'Miara do sortowania galerii i mapy na sondzie.')
        self.btn = rola(QPushButton('Licz pchnięcia'), 'glowny'); w.addWidget(self.btn)
        self.btn_zapisz = QPushButton('Zapisz pomiar'); w.addWidget(self.btn_zapisz)
        self.btn_zapisz.setToolTip(f'JSON z miarami wszystkich kandydatów w {WYNIKI}.')
        self.pasek = QProgressBar(); self.pasek.setMaximumWidth(200); w.addWidget(self.pasek)
        self.etykieta = QLabel('Pchnięcie: jak jeden obraz w paczce przesuwa cząstki sondy w jednym kroku AdamW. '
                               'Wczytaj historię w zakładce „Uczenie ViT” i kliknij „Licz pchnięcia”.')
        self.etykieta.setWordWrap(True); lay.addWidget(self.etykieta)
        self.pod = pod = QSplitter(Qt.Orientation.Horizontal)
        from PyQt6.QtWidgets import QScrollArea
        self._przew = QScrollArea(); self._przew.setWidget(pod); self._przew.setWidgetResizable(True)
        lay.addWidget(self._przew, 1)
        self.fig_s = Figure(figsize=(5, 6)); self.can_s = PlotnoCiemne(self.fig_s); pod.addWidget(self.can_s)
        self.fig_g = Figure(figsize=(7, 6)); self.can_g = PlotnoCiemne(self.fig_g); pod.addWidget(self.can_g)
        self.fig_w = Figure(figsize=(5, 6)); self.can_w = PlotnoCiemne(self.fig_w); pod.addWidget(self.can_w)
        pod.setSizes([520, 760, 560])
        self.btn.clicked.connect(self.licz)
        self.btn_zapisz.clicked.connect(self.zapisz)
        self.combo_miara.currentIndexChanged.connect(lambda *_: self.rysuj())
        self.combo_tok.currentIndexChanged.connect(lambda *_: self._przelicz_miary())
        self.can_g.mpl_connect('button_press_event', self._klik_galerii)
        self.can_w.mpl_connect('pick_event', self._pick)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        o = Qt.Orientation.Vertical if self.width() < 1500 else Qt.Orientation.Horizontal   # wąskie okno: panele pod sobą
        if self.pod.orientation() != o:
            self.pod.setOrientation(o)
            pion = o == Qt.Orientation.Vertical          # pod sobą: każdy panel około 750 px, całość się przewija
            self.pod.setMinimumHeight(2300 if pion else 0)
            self.pod.setSizes([700, 900, 700] if pion else [520, 760, 560])

    # ------------------------------------------------------------ dane z zakładki uczenia
    def _z_zakladki(self):
        u = self.u
        if u.Z is None:
            return False
        if self.chk_sledz.isChecked():
            self.spin_k.setValue(min(int(u.idx[u.t]), 999)); self.spin_sonda.setValue(int(u.i_kl))
        return True

    def _maska(self, sonda):
        i = self.combo_tok.currentIndex()
        if i == 1 and self.u.zaznaczenie is not None:
            m = np.zeros(600, bool); m[self.u.zaznaczenie] = True; return m
        if i == 2:
            return np.ones(600, bool)
        return None

    # ------------------------------------------------------------ liczenie
    def licz(self):
        if self.watek is not None:
            return
        if not self._z_zakladki():
            self.etykieta.setText('Najpierw wczytaj historię uczenia w zakładce „Uczenie ViT”.'); return
        bieg = self.u._bieg()
        if not (Path(bieg['dane']).parents[2] / 'kod' / 'trening.py').exists():
            self.etykieta.setText('Ten bieg nie ma kodu uczenia (kod/trening.py), więc nie da się odtworzyć stanu AdamW.'); return
        k, sonda = self.spin_k.value(), self.spin_sonda.value()
        zr = [z for z, c in self.chk_zr.items() if c.isChecked()]
        nb = self.spin_baza.value()
        maska = self._maska(sonda)
        from zakladka_filmu import WatekFilmu

        def zadanie(wt):
            import kandydaci as KD
            from pchniecie import Pchniecie
            from stan_uczenia import Odtworzenie
            import analiza_uczenia as AU
            if self.od is None or self.od.bieg['klucz'] != bieg['klucz']:
                wt.postep.emit(2, 'wczytuję bazę uczenia (pierwszy raz około minuty)...')
                self.od = Odtworzenie(bieg)
                op = AU.opis_biegu(bieg)
                self.P = Pchniecie(self.od, np.asarray(op['obrazy']), op['narzedzie'], _ramki_sond(bieg))
                self._cache = {}
            wt.postep.emit(5, 'powtórka uczenia i stan AdamW (pierwszy raz około 2 minut)...')
            self.od.przygotuj(lambda a, n: wt.postep.emit(5 + int(25 * a / n), f'powtórka uczenia: krok {a}/{n}'))
            P = self.P
            wt.postep.emit(32, f'stan kroku {k}...')
            cz = P.przygotuj(k, sonda)[sonda]
            L = np.load(Path(bieg['dane']) / 'LATENT_GESTY.npy', mmap_mode='r')
            P.kierunki_miar(np.asarray(L[k]), np.asarray(L[-1]), sonda)
            import torch
            rzecz = torch.from_numpy(np.asarray(L[k + 1, sonda], np.float32) - np.asarray(L[k, sonda], np.float32)).cuda()
            sk = lambda y, yh: 1 - float(((y - yh) ** 2).sum() / (y ** 2).sum())
            pred = cz['nowa'] + cz['ped'] + cz['zanik'] + cz['bn']
            kand = []
            obr = P.pchniecia_paczki(sonda)
            x = P.paczka['x'].permute(0, 2, 3, 1).cpu().numpy()
            for i, S in enumerate(obr):
                j = int(P.paczka['idx'][i])
                kand.append(dict(zrodlo='paczka', nazwa=f'paczka #{i + 1} (baza {j})', etykieta='zbiór',
                                 S=S, obraz=(x[i] * 255).astype(np.uint8)))
            suma_obr = torch.stack(obr).sum(0)
            zbiory = []
            if 'baza' in zr:
                zbiory.append(('baza', lambda: KD.baza(self.od, nb)))
            if 'odszukane_sd' in zr:
                zbiory.append(('odszukane_sd', KD.odszukane_sd))
            if 'sd' in zr:
                import detekcja as D
                zbiory.append(('sd', lambda: KD.sd_z_pseudoetykieta(D.Detektor(self.od.wagi / f'k{self.od.kroki:04d}.pt'))))
            if 'dino_sd' in zr:
                zbiory.append(('dino_sd', KD.dino_sd))
            if 'konwerter_sd' in zr:
                zbiory.append(('konwerter_sd', KD.konwerter_sd))
            for zi, (nazwa, f) in enumerate(zbiory):
                kl = (nazwa, nb if nazwa == 'baza' else 0)
                if kl not in self._cache:
                    wt.postep.emit(40 + int(40 * zi / max(len(zbiory), 1)), f'kandydaci: {nazwa}...')
                    self._cache[kl] = f()
                for c in self._cache[kl]:
                    kand.append(dict(c, S=P.pchniecie_kandydata(c['obraz'], c['ramki'], sonda)))
            wt.postep.emit(90, 'miary...')
            for c in kand:
                c['miary'] = P.miary(c['S'], sonda, maska)
                c['S'] = None if c['zrodlo'] != 'paczka' else None
                c['mini'] = _miniatura(c['obraz'])
            czesci = {n: dict(norma=float(v.norm() / rzecz.norm()), **{a: b for a, b in P.miary(v, sonda, maska).items()
                                                                   if not a.startswith('mapa')}) for n, v in cz.items()}
            return dict(k=k, sonda=sonda, kand=kand, czesci=czesci, skill=sk(rzecz, pred), skill_bez_bn=sk(rzecz, pred - cz['bn']),
                        skill_bez_paczki=sk(rzecz, cz['ped'] + cz['zanik'] + cz['bn']), skill_suma=sk(cz['nowa'], suma_obr),
                        lr=P.lr, rgb=np.asarray(AU.opis_biegu(bieg)['obrazy'][sonda]), maska=maska,
                        maska_nar=P.maski[sonda])

        self.btn.setEnabled(False)
        self.watek = WatekFilmu(zadanie, self)
        self.watek.postep.connect(lambda a, t: (self.pasek.setValue(a), self.etykieta.setText(t)))
        self.watek.gotowy.connect(self._gotowe)
        self.watek.blad.connect(lambda t: (setattr(self, 'watek', None), self.btn.setEnabled(True),
                                           self.etykieta.setText('Błąd: ' + t[-800:])))
        self.watek.start()

    def _gotowe(self, w):
        self.watek = None; self.btn.setEnabled(True); self.pasek.setValue(100)
        self.wynik = w; self.wybrany = 0
        c = w['czesci']
        self.etykieta.setText(
            f'Krok {w["k"]} -> {w["k"] + 1}, klatka sondy {w["sonda"]}, lr {w["lr"]:.2e}. Sprawdzian: przewidziany krok sondy '
            f'(paczka + pęd + zanik + BN) wobec rzeczywistego z gęstej historii: trafność {w["skill"]:.3f} (bez BN '
            f'{w["skill_bez_bn"]:.3f}, bez paczki tylko {w["skill_bez_paczki"]:.3f}); suma pchnięć 32 obrazów = pchnięcie '
            f'paczki w {w["skill_suma"] * 100:.1f} %. Długość części / ruch: paczka {c["nowa"]["norma"]:.2f}, pęd '
            f'{c["ped"]["norma"]:.2f}, BN {c["bn"]["norma"]:.2f}, zanik {c["zanik"]["norma"]:.3f}. Kandydatów {len(w["kand"])}.')
        self.rysuj()

    def _przelicz_miary(self):
        if self.wynik is not None:
            self.etykieta.setText(self.etykieta.text() + '  (Zmiana tokenów: kliknij „Licz pchnięcia”, żeby przeliczyć miary.)')

    # ------------------------------------------------------------ rysunki
    def _miara(self):
        return MIARY[self.combo_miara.currentIndex()]

    def _kolejnosc(self):
        klucz = self._miara()[0]
        v = np.array([c['miary'][klucz] for c in self.wynik['kand']])
        return np.argsort(v) if klucz == 'strata' else np.argsort(-v)

    def rysuj(self):
        if self.wynik is None:
            return
        self._rysuj_galerie(); self._rysuj_wykresy(); self._rysuj_sonde()

    def _rysuj_galerie(self):
        f = self.fig_g; f.clear()
        kand = self.wynik['kand']; kol = self._kolejnosc()[:60]
        klucz, opis, _ = self._miara()
        n = len(kol); nk = 6; nw = int(np.ceil(n / nk))
        self._osie_gal = {}
        for p, i in enumerate(kol):
            ax = f.add_axes([(p % nk) / nk + .003, 1 - (p // nk + 1) / max(nw, 1) * .96 - .02, 1 / nk - .006, .96 / max(nw, 1) - .03])
            c = kand[i]
            ax.imshow(c['mini']); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(KOL_ZRODLA.get(c['zrodlo'], 'white')); s.set_linewidth(3 if i == self.wybrany else 1.5)
            ax.set_title(f'{c["miary"][klucz]:+.4f}', fontsize=6.5, color='white', pad=1)
            self._osie_gal[ax] = i
        f.suptitle(f'Kandydaci posortowani wg: {opis} (ramka = źródło; klik wybiera)', fontsize=8, color='white', y=.999)
        self.can_g.draw_idle()

    def _rysuj_wykresy(self):
        w = self.wynik; f = self.fig_w; f.clear()
        osie = f.subplots(3, 1, gridspec_kw=dict(height_ratios=[1, 1.5, 1.1]))
        f.subplots_adjust(left=.14, right=.97, top=.95, bottom=.09, hspace=.85)
        ax = osie[0]
        cz = w['czesci']; nazwy = ['nowa', 'ped', 'bn', 'zanik']
        ax.bar(['paczka', 'pęd', 'BN', 'zanik'], [cz[n]['norma'] for n in nazwy], color=['#d0d0d0', '#ffd030', '#9467bd', '#555555'])
        ax.axhline(1, color='white', lw=.7, ls=':')
        ax.set_title(f'części kroku sondy / ruch (trafność {w["skill"] * 100:.1f} %)', fontsize=8)
        ax = osie[1]
        x = np.array([c['miary']['tozsamosc'] for c in w['kand']]); y = np.array([c['miary']['strata'] for c in w['kand']])
        zr = np.array([c['zrodlo'] for c in w['kand']])
        for z in dict.fromkeys(zr):
            m = zr == z
            ax.scatter(x[m], y[m], s=14, c=KOL_ZRODLA.get(z, 'white'), label=KROTKIE.get(z, z), picker=4, alpha=.85)
        ax.scatter([x[self.wybrany]], [y[self.wybrany]], s=90, facecolors='none', edgecolors='white', lw=1.5)
        ax.axhline(0, color='white', lw=.6, ls=':'); ax.axvline(0, color='white', lw=.6, ls=':')
        ax.set_xlabel('zmiana siły tożsamości', fontsize=7); ax.set_ylabel('zmiana straty sondy', fontsize=7)
        ax.set_title('obraz: tożsamość (w prawo = więcej)\nwobec straty sondy (w dół = pomaga)', fontsize=8)
        ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.28), ncol=4, framealpha=.3)
        self._ax_sc = ax
        ax = osie[2]
        klucz, opis, _ = self._miara()
        zrodla = list(dict.fromkeys(zr))
        for i, z in enumerate(zrodla):
            v = np.array([c['miary'][klucz] for c in w['kand'] if c['zrodlo'] == z])
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.18, .18, len(v)), v, s=8,
                       c=KOL_ZRODLA.get(z, 'white'), alpha=.7)
            ax.plot([i - .3, i + .3], [np.mean(v)] * 2, color='white', lw=2)
        ax.axhline(0, color='white', lw=.6, ls=':')
        ax.set_xticks(range(len(zrodla))); ax.set_xticklabels([KROTKIE.get(z, z) for z in zrodla], fontsize=6.5)
        ax.set_title(f'{opis.split(" (")[0]}: obrazy i średnia (biała)', fontsize=8)
        for a in f.axes:
            a.tick_params(labelsize=6.5)
        self.can_w.draw_idle()

    def _rysuj_sonde(self):
        w = self.wynik; f = self.fig_s; f.clear()
        c = w['kand'][self.wybrany]
        klucz, opis, cmap = self._miara()
        ax = f.add_axes([.02, .45, .96, .5])
        ax.imshow(w['rgb'], extent=(0, 480, 320, 0))
        mapa = c['miary'].get('mapa_' + klucz, c['miary']['mapa_dlugosc']).reshape(20, 30)
        v = max(float(np.abs(mapa).max()), 1e-12)
        ax.imshow(mapa, extent=(0, 480, 320, 0), cmap=cmap, vmin=-v if cmap != 'viridis' else 0, vmax=v, alpha=.6,
                  interpolation='nearest')
        m = w['maska'] if w['maska'] is not None else w['maska_nar']
        r, k = np.divmod(np.flatnonzero(m), 30)
        ax.scatter(k * 16 + 8, r * 16 + 8, s=3, c='white', alpha=.6)
        ax.set_xlim(0, 480); ax.set_ylim(320, 0); ax.axis('off')
        ax.set_title(f'sonda: {opis.split(" (")[0]}\nw każdym tokenie od wybranego obrazu', fontsize=8)
        ax2 = f.add_axes([.02, .04, .46, .34]); ax2.imshow(c['obraz']); ax2.axis('off')
        for r_ in c.get('ramki', []):
            from matplotlib.patches import Rectangle
            ax2.add_patch(Rectangle((r_[0], r_[1]), r_[2] - r_[0], r_[3] - r_[1], fill=False, ec='#00ff66', lw=1))
        import textwrap
        ax2.set_title(textwrap.fill(c['nazwa'], 34), fontsize=6.5)
        mi = c['miary']
        f.text(.52, .38, f'{NAZWY_ZRODEL.get(c["zrodlo"], c["zrodlo"])}\netykieta: {c["etykieta"]}\n\n'
                        f'zmiana straty sondy {mi["strata"]:+.5f}\nzmiana siły tożsamości {mi["tozsamosc"]:+.4f}\n'
                        f'kosinus do końca {mi["do_konca"]:+.3f}\ndługość pchnięcia {mi["dlugosc"]:.4f}',
               fontsize=8, va='top', color='#e0e0e0')
        self.can_s.draw_idle()

    def _klik_galerii(self, e):
        i = getattr(self, '_osie_gal', {}).get(e.inaxes)
        if i is not None:
            self.wybrany = i; self.rysuj()

    def _pick(self, e):
        if self.wynik is None or e.mouseevent.inaxes is not getattr(self, '_ax_sc', None):
            return
        x = np.array([c['miary']['tozsamosc'] for c in self.wynik['kand']])
        y = np.array([c['miary']['strata'] for c in self.wynik['kand']])
        sx, sy = np.ptp(x) or 1, np.ptp(y) or 1
        self.wybrany = int(np.argmin(((x - e.mouseevent.xdata) / sx) ** 2 + ((y - e.mouseevent.ydata) / sy) ** 2))
        self.rysuj()

    # ------------------------------------------------------------ zapis
    def zapisz(self):
        if self.wynik is None:
            return
        w = self.wynik
        WYNIKI.mkdir(parents=True, exist_ok=True)
        p = WYNIKI / f'PCHNIECIE_k{w["k"]:04d}_sonda{w["sonda"]:02d}.json'
        rek = [dict(zrodlo=c['zrodlo'], nazwa=c['nazwa'], etykieta=c['etykieta'], ramki=c.get('ramki', []),
                    **{a: b for a, b in c['miary'].items() if not a.startswith('mapa')}) for c in w['kand']]
        p.write_text(json.dumps(dict(krok=w['k'], sonda=w['sonda'], lr=w['lr'], trafnosc_kroku=w['skill'],
                                     trafnosc_bez_bn=w['skill_bez_bn'], trafnosc_bez_paczki=w['skill_bez_paczki'],
                                     suma_obrazow_wobec_paczki=w['skill_suma'],
                                     czesci={n: {a: b for a, b in v.items()} for n, v in w['czesci'].items()},
                                     kandydaci=rek), indent=1, ensure_ascii=False), encoding='utf-8')
        self.etykieta.setText(f'Zapisano {p}')
