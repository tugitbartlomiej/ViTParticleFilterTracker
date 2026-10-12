"""Zakładka „Ruch zbioru”, część Koopmana: strona opcji „Koopman” (Hankel-DMD liczone na karcie), tabela widma,
ręczny i automatyczny wybór osi oraz dobór osi pod klastry (k-means na próbce z karty).

Klasa bazowa dla `zakladka_ruchu.ZakladkaRuchu` (korzysta z jej `cloud`, `opcje`, `_int`, `_float`, `_budz`)."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel

from styl import rola
from uklad import dodaj, pole_wyboru, przycisk


class KoopmanRuchu:
    def _strona_koopmana(self):
        """Strona „Koopman”: dopasowanie operatora, wybór osi i dobór osi pod klastry."""
        self.opcje.dodaj_strone('Koopman')
        row = self.opcje.wiersz('Koopman')
        self.spin_kd = self._int(row, 'opóźnienia', 1, 4, 3,
                                 'Zanurzenie opóźnieniowe (Hankel-DMD): stan kroku to d ostatnich zapisów (PC1, PC2, PC3, '
                                 'norma 768D), czyli 4 · d liczb. Więcej opóźnień = więcej modów i lepiej widoczne '
                                 'oscylacje, ale potrzeba d zapisów wstecz (na początku osi czasu powtarzany jest pierwszy).')
        self.combo_kxy = QComboBox()
        self.combo_kxy.addItem('zespolony niezmiennik najbliższy λ = 1: X = Re, Y = Im', 0)
        self.combo_kxy.addItem('dwa mody najbliższe λ = 1: części rzeczywiste', 1)
        self.combo_kxy.addItem('niezmiennik dodatni (λ ≈ +1) na X i ujemny (λ ≈ −1) na Y', 2)
        self.combo_kxy.setCurrentIndex(1)                     # w pomiarach naturalnych niezmienniki są rzeczywiste
        dodaj(row, 'X, Y', self.combo_kxy,
              'Które funkcje własne Koopmana trafiają na osie X i Y. Wartość własna bliska 1 oznacza wielkość prawie\n'
              'stałą wzdłuż toru cząsteczki (niezmiennik), więc położenie w X, Y mówi, do jakiej rodziny ruchów należy\n'
              'cząsteczka. Zespolony niezmiennik obraca się powoli: część rzeczywista i urojona to jego dwie osie.\n'
              'λ ≈ −1: wielkość, która co zapis zmienia znak, ale zachowuje wartość bezwzględną (niezmiennik ujemny).')
        self.combo_kz = QComboBox()
        self.combo_kz.addItem('najszybszy: największe |λ − 1|', 0)
        self.combo_kz.addItem('najsilniej oscylujący: największy |arg λ|', 1)
        self.combo_kz.addItem('najsilniej gasnący lub rosnący: największe |ln |λ||', 2)
        dodaj(row, 'Z', self.combo_kz, 'Który mod (spośród nieużytych na X, Y) trafia na oś Z jako część najbardziej '
              'dynamiczna: najdalej od λ = 1 (najszybsza zmiana na zapis), o największej częstotliwości albo najszybciej '
              'gasnący lub rosnący.')
        self.combo_kczesc = QComboBox()
        for tekst, v in (('część rzeczywista', 0), ('część urojona', 1), ('moduł |ψ|', 2)):
            self.combo_kczesc.addItem(tekst, v)
        dodaj(row, 'część Z', self.combo_kczesc, 'Jak pokazać zespoloną funkcję własną na osi Z: część rzeczywista, '
              'urojona albo moduł (amplituda modu, bez fazy). Dla modu rzeczywistego część urojona to 0.')
        self.btn_koop = przycisk(row, 'Przelicz Koopmana', 'Ponownie dopasowuje operator Koopmana na karcie do wszystkich '
                                 'stanów osi czasu (zwykle dzieje się to samo po wczytaniu osi i zmianie opóźnień).')
        self.btn_koop_auto = przycisk(row, 'Osie automatycznie', 'Wraca do automatycznego wyboru osi z pól „X, Y”, „Z” i '
                                      '„część Z” (tabela niżej pokaże, które składowe zostały wybrane).')
        # --- ręczny wybór składowych na osie
        row = self.opcje.wiersz('Koopman')
        self.combo_ksort = QComboBox()
        for tekst, v in (('bliskość do 1: |λ − 1| rosnąco (niezmienniki najpierw)', 0), ('|λ| malejąco (najwolniej gasnące)', 1),
                         ('|arg λ| rosnąco (najwolniejsze oscylacje)', 2), ('|arg λ| malejąco (najszybsze oscylacje)', 3),
                         ('Re λ malejąco', 4), ('Im λ malejąco', 5), ('|λ − 1| malejąco (najbardziej dynamiczne)', 6)):
            self.combo_ksort.addItem(tekst, v)
        dodaj(row, 'sortuj mody', self.combo_ksort,
              'Kolejność modów w tabeli i w szybkim przypisaniu („od”, „do” liczone w tej kolejności). Bliskość do 1: '
              'najpierw niezmienniki (λ ≈ 1, wielkości prawie stałe wzdłuż toru).')
        self.combo_kre = QComboBox()
        for tekst, v in (('wszystkie', 0), ('Re λ > 0', 1), ('Re λ < 0', 2)):
            self.combo_kre.addItem(tekst, v)
        dodaj(row, 'znak Re λ', self.combo_kre, 'Pokaż tylko mody o dodatniej (λ po prawej stronie zera, bez zmiany znaku '
              'co zapis) albo ujemnej części rzeczywistej wartości własnej (mod zmienia znak co zapis).')
        self.combo_kim = QComboBox()
        for tekst, v in (('wszystkie', 0), ('Im λ > 0', 1), ('Im λ < 0', 2), ('Im λ = 0 (rzeczywiste)', 3)):
            self.combo_kim.addItem(tekst, v)
        dodaj(row, 'znak Im λ', self.combo_kim, 'Pokaż mody o dodatniej albo ujemnej części urojonej (pary sprzężone '
              'oscylują w przeciwne strony; Im ψ ma u nich przeciwny znak) albo tylko mody rzeczywiste.')
        row = self.opcje.wiersz('Koopman')
        self.combo_kos = QComboBox()
        for tekst, v in (('X', 'x'), ('Y', 'y'), ('Z', 'z')):
            self.combo_kos.addItem(tekst, v)
        dodaj(row, 'oś', self.combo_kos, 'Na którą oś trafiają mody z szybkiego przypisania.')
        self.spin_kod = self._int(row, 'mody od', 1, 16, 1, 'Pierwszy mod zakresu (numer w tabeli, po sortowaniu i filtrach).')
        self.spin_kdo = self._int(row, 'do', 1, 16, 1, 'Ostatni mod zakresu (włącznie).')
        self.combo_kczesci = QComboBox()
        for tekst, v in (('część rzeczywista', ('re',)), ('część urojona', ('im',)), ('rzeczywista i urojona', ('re', 'im'))):
            self.combo_kczesci.addItem(tekst, v)
        dodaj(row, 'części', self.combo_kczesci, 'Które części funkcji własnych ψ wybranych modów rzutować na oś.')
        self.btn_kustaw = przycisk(row, 'Ustaw na osi', 'Zastępuje składowe osi wybranymi częściami modów z zakresu.')
        self.btn_kdodaj = przycisk(row, 'Dodaj do osi', 'Dokłada wybrane części modów z zakresu do składowych osi.')
        self.btn_kczysc = przycisk(row, 'Wyczyść oś', 'Usuwa wszystkie składowe osi (oś stoi wtedy w zerze).')
        row = self.opcje.wiersz('Koopman')
        self.combo_kkomb = {}
        for os_ in ('x', 'y', 'z'):
            c = QComboBox(); c.addItem('suma (rzut liniowy)', 0); c.addItem('moduł √Σ²', 1)
            dodaj(row, f'łączenie {os_.upper()}', c,
                  f'Jak łączyć składowe osi {os_.upper()}: suma = zwykły rzut liniowy na sumę wybranych części ψ (mody '
                  f'mogą się wzmacniać albo znosić); moduł = długość wektora składowych √Σ², zawsze ≥ 0 (np. Re i Im '
                  f'jednego modu dają jego amplitudę |ψ| bez fazy).')
            self.combo_kkomb[os_] = c
        from PyQt6.QtWidgets import QTableWidget, QAbstractItemView
        self.tab_koop = QTableWidget(0, 13)
        self.tab_koop.setHorizontalHeaderLabels(['nr', 'X Re', 'X Im', 'Y Re', 'Y Im', 'Z Re', 'Z Im',
                                                 'Re λ', 'Im λ', '|λ|', '|λ−1|', 'arg°', 'okres'])
        self.tab_koop.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tab_koop.setMinimumHeight(220)
        self.tab_koop.verticalHeader().setVisible(False)
        self.tab_koop.verticalHeader().setDefaultSectionSize(20)
        self.tab_koop.setToolTip('Widmo dopasowanego operatora Koopmana. Zaznacz, które części (Re, Im) funkcji własnych '
                                 'których modów rzutować na osie X, Y, Z (łączenie osi wyżej). Zmiana pola przełącza na '
                                 'ręczny wybór osi; „Osie automatycznie” wraca do wyboru z pól powyżej.')
        # tabela widma w prawej kolumnie (pod pomiarami), widoczna tylko w układzie Koopmana: tam mieści się całe widmo
        self._koop_mody, self._osie_reczne = [], None
        self.koop_label = rola(QLabel('Wybierz układ „Koopman”, aby dopasować operator i zobaczyć widmo.'), 'opis')
        self.koop_label.setWordWrap(True)
        self.koop_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.opcje._strony['Koopman'].addWidget(self.koop_label)
        # --- dobór osi pod klastry: wiersze na górze strony, zaraz pod pierwszym
        from uklad import Przeplyw
        lay_k = self.opcje._strony['Koopman']

        def wiersz_gora(n):
            w = Przeplyw(); lay_k.insertLayout(n, w)
            return w
        row = wiersz_gora(1)
        self.btn_kklastry = przycisk(row, 'Osie pod klastry',
                                     'Sam dobiera osie X, Y (i Z) spośród składowych Koopmana tak, żeby cząsteczki '
                                     'układały się w wyraźne klastry z jak najmniejszą liczbą outlierów. Karta zwraca '
                                     'próbkę stanów cząsteczek widoku, potem k-means (na karcie, torch) ocenia każdą parę '
                                     'Re ψ, Im ψ i |ψ| modów dla k = 2..k max: zwartość i rozdzielenie klastrów (sylwetka) '
                                     'minus kara za outliery. Wynik trafia do tabeli jak ręczne osie.', glowny=True)
        self.spin_kl_m = self._int(row, 'próbka', 500, 16384, 4096,
                                   'Liczba cząsteczek widoku (równomiernie po zakresie) użytych do oceny. Więcej = '
                                   'pewniej, ale wolniej (4096 to około 2-5 s na RTX 3070).')
        self.combo_kl_kiedy = QComboBox()
        self.combo_kl_kiedy.addItem('bieżący zapis', False); self.combo_kl_kiedy.addItem('cały ruch (8 zapisów)', True)
        dodaj(row, 'oceniaj w', self.combo_kl_kiedy,
              'Bieżący zapis: klastry takie, jakie widać teraz. Cały ruch: ta sama próbka cząsteczek w 8 zapisach '
              'rozłożonych po osi czasu i w bieżącym; każdy zapis oceniany osobno, wynik uśredniony. Wygrywają osie, na '
              'których cząsteczki tworzą klastry przez cały trening (a nie takie, które tylko oddzielają wczesny trening '
              'od późnego). Środki klastrów do kolorów: z bieżącego zapisu.')
        self.chk_kl_z = pole_wyboru(row, 'także oś Z', 'Dobierz również oś Z: składową, z którą klastry w 3D są '
                                    'najlepsze. Bez zaznaczenia Z zostaje bez zmian.', zaznaczone=True)
        self.chk_kl_kolor = pole_wyboru(row, 'koloruj klastry', 'Po doborze osi przełącza kolor na „Klaster Koopmana”: '
                                        'każdy klaster ma swoją barwę, outliery są szare. Przydział liczy shader dla '
                                        'wszystkich cząsteczek w każdej klatce (najbliższy środek klastra).',
                                        zaznaczone=True)
        row = wiersz_gora(2)
        self.spin_kl_k = self._int(row, 'k max', 2, 10, 6, 'Największa sprawdzana liczba klastrów (k = 2..k max).')
        self.spin_kl_prog = self._float(row, 'outlier od × σ', 2, 6, 3, 1,
                                        'Punkt jest outlierem, gdy leży dalej od środka swojego klastra niż ta '
                                        'wielokrotność σ klastra (σ z mediany odległości, skalibrowanej na 2D albo 3D), '
                                        'albo należy do klastra mniejszego niż 2 % próbki. Dla gaussowskiego klastra 3σ '
                                        'odcina około 1 % punktów w 2D i 3 % w 3D.')
        self.spin_kl_waga = self._float(row, 'kara za outliery', 0, 20, 4, 1,
                                        'Wynik = sylwetka − kara × udział outlierów. Większa kara = mniej outlierów '
                                        'kosztem słabiej rozdzielonych klastrów.')
        self.kl_label = rola(QLabel('Osie pod klastry: kliknij przycisk w układzie „Koopman”.'), 'opis')
        self.kl_label.setWordWrap(True)
        self.kl_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay_k.insertWidget(3, self.kl_label)

    def _koopman_opis(self, info):
        """Po dopasowaniu albo zmianie osi: tabela modów i opis osi."""
        mody = info.get('mody') or []
        if not mody:
            self.koop_label.setText(info.get('opis', '')); return
        self._koop_mody = mody
        osie = info.get('osie') or {}
        lam = {m['j']: m for m in mody}

        def opis(os_):
            o = osie.get(os_, {'skladowe': [], 'kombinacja': 0})
            if not o['skladowe']:
                return '0'
            cz = [f'{c.capitalize()} ψ(λ = {lam[j]["re"]:+.3f}{lam[j]["im"]:+.3f}i)' for j, c in o['skladowe'] if j in lam]
            return ('√(' + '² + '.join(cz) + '²)') if o['kombinacja'] == 1 and len(cz) > 1 else \
                   ('|' + cz[0] + '|' if o['kombinacja'] == 1 else ' + '.join(cz))
        tryb = 'automatyczne' if info.get('automatyczne') else 'ręczne'
        self.koop_label.setText(f'Operator Koopmana (Hankel-DMD, {info["d"]} opóźnienia, {info["pary"]:,} par stanów, '
                                f'zapis co {info["krok"]:.0f} kroków treningu)'
                                + (f'; pominięto {info["zdegenerowane"]} modów o zerowej wariancji (artefakty zanurzenia)'
                                   if info.get('zdegenerowane') else '') + f'. Osie {tryb}:\n'
                                f'X = {opis("x")}\nY = {opis("y")}\nZ = {opis("z")}')
        self._koop_osie = osie
        for os_, c in self.combo_kkomb.items():
            c.blockSignals(True); c.setCurrentIndex(int(osie.get(os_, {}).get('kombinacja', 0))); c.blockSignals(False)
        self._tabela_koopmana()

    def _mody_widoczne(self):
        """Mody w kolejności sortowania i po filtrach znaków (lista słowników z koop_info)."""
        tol = 1e-7
        mody = [m for m in self._koop_mody
                if {0: True, 1: m['re'] > 0, 2: m['re'] < 0}[self.combo_kre.currentData()]
                and {0: True, 1: m['im'] > tol, 2: m['im'] < -tol, 3: abs(m['im']) <= tol}[self.combo_kim.currentData()]]
        klucze = {0: lambda m: abs(complex(m['re'], m['im']) - 1), 1: lambda m: -m['modul'], 2: lambda m: abs(m['kat']),
                  3: lambda m: -abs(m['kat']), 4: lambda m: -m['re'], 5: lambda m: -m['im'],
                  6: lambda m: -abs(complex(m['re'], m['im']) - 1)}
        return sorted(mody, key=klucze[self.combo_ksort.currentData()])

    def _tabela_koopmana(self):
        from PyQt6.QtWidgets import QTableWidgetItem
        mody = self._mody_widoczne()
        osie = getattr(self, '_koop_osie', {}) or {}
        wybrane = {(os_, j, c) for os_, o in osie.items() for j, c in o['skladowe']}
        t = self.tab_koop
        t.blockSignals(True)
        t.setRowCount(len(mody))
        for r, m in enumerate(mody):
            okres = f'{m["okres"]:.0f}' if m['okres'] else '—'
            for k, tekst in zip((0, 7, 8, 9, 10, 11, 12),
                                (str(r + 1), f'{m["re"]:+.3f}', f'{m["im"]:+.3f}', f'{m["modul"]:.3f}',
                                 f'{abs(complex(m["re"], m["im"]) - 1):.3f}', f'{m["kat"]:+.0f}', okres)):
                it = QTableWidgetItem(tekst); it.setData(Qt.ItemDataRole.UserRole, m['j']); t.setItem(r, k, it)
            for k, (os_, c) in enumerate((('x', 're'), ('x', 'im'), ('y', 're'), ('y', 'im'), ('z', 're'), ('z', 'im'))):
                it = QTableWidgetItem()
                it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                it.setCheckState(Qt.CheckState.Checked if (os_, m['j'], c) in wybrane else Qt.CheckState.Unchecked)
                it.setData(Qt.ItemDataRole.UserRole, (os_, m['j'], c))
                t.setItem(r, 1 + k, it)
        t.resizeColumnsToContents()
        t.blockSignals(False)
        self.spin_kod.setMaximum(max(len(mody), 1)); self.spin_kdo.setMaximum(max(len(mody), 1))

    def _osie_koopmana(self, osie):
        """Nowe przypisanie osi (None = automatyczne) -> karta przelicza wagi i zakres osi."""
        self._osie_reczne = osie
        self._display_options()

    def _z_tabeli_koopmana(self):
        """Przypisanie osi ze stanu tabeli (pola zaznaczone także w modach ukrytych przez filtry zostają)."""
        if not self._koop_mody:
            return
        osie = {os_: dict(skladowe=list(o['skladowe']), kombinacja=self.combo_kkomb[os_].currentData())
                for os_, o in (getattr(self, '_koop_osie', {}) or {}).items()}
        for os_ in ('x', 'y', 'z'):
            osie.setdefault(os_, dict(skladowe=[], kombinacja=self.combo_kkomb[os_].currentData()))
        t = self.tab_koop
        for r in range(t.rowCount()):
            for k in range(1, 7):
                it = t.item(r, k)
                if it is None:
                    continue
                os_, j, c = it.data(Qt.ItemDataRole.UserRole)
                sk = osie[os_]['skladowe']
                if it.checkState() == Qt.CheckState.Checked and (j, c) not in sk:
                    sk.append((j, c))
                elif it.checkState() != Qt.CheckState.Checked and (j, c) in sk:
                    sk.remove((j, c))
        self._osie_koopmana(osie)

    def _przypisz_koopmana(self, akcja):
        """Szybkie przypisanie: mody od..do (w kolejności tabeli) i wybrane części na wybraną oś."""
        if not self._koop_mody:
            self._set_text(self.koop_label, 'Najpierw wybierz układ „Koopman”, aby dopasować operator.'); return
        self._z_tabeli_koopmana()                       # punkt wyjścia: bieżące przypisanie
        osie = {k: dict(skladowe=list(v['skladowe']), kombinacja=v['kombinacja']) for k, v in self._osie_reczne.items()}
        os_ = self.combo_kos.currentData()
        mody = self._mody_widoczne()[self.spin_kod.value() - 1:self.spin_kdo.value()]
        nowe = [(m['j'], c) for m in mody for c in self.combo_kczesci.currentData()]
        if akcja == 'czysc':
            osie[os_]['skladowe'] = []
        elif akcja == 'ustaw':
            osie[os_]['skladowe'] = nowe
        else:
            osie[os_]['skladowe'] += [n for n in nowe if n not in osie[os_]['skladowe']]
        self._osie_koopmana(osie)

    def _klastry_koopmana(self):
        """Krok 1: próbka stanów Hankela cząsteczek widoku z karty (wynik przez sygnał probka_koopmana)."""
        if not self.timeline:
            self._set_text(self.kl_label, 'Osie pod klastry: najpierw wczytaj oś czasu.'); return
        if self.watek is not None:
            self._set_text(self.kl_label, 'Osie pod klastry: poczekaj, aż skończy się bieżące zadanie w tle.'); return
        if self.combo_uklad.currentData() != 3:
            self.combo_uklad.setCurrentIndex(self.combo_uklad.findData(3))      # dopasowanie Koopmana, potem próbka
        self._set_text(self.kl_label, 'Osie pod klastry: pobieram próbkę cząsteczek z karty…')
        self.cloud.zadaj_probke_koopmana(self.spin_kl_m.value(), self.combo_kl_kiedy.currentData())
        self._budz()

    def _klastry_probka(self, p):
        """Krok 2: ocena par i trójek osi w tle (k-means na karcie przez torch), potem nowe osie."""
        if 'blad' in p:
            self._set_text(self.kl_label, p['blad']); return
        if len(p['H']) < 50:
            self._set_text(self.kl_label, f'Osie pod klastry: w widoku tylko {len(p["H"])} widocznych cząsteczek '
                           '(potrzeba co najmniej 50). Zmień filtry albo zakres.'); return
        if self.watek is not None:
            self._set_text(self.kl_label, 'Osie pod klastry: poczekaj, aż skończy się bieżące zadanie w tle.'); return
        import klastry_koopmana as KK
        ustaw = dict(z_tez=self.chk_kl_z.isChecked(), k_max=self.spin_kl_k.value(), prog=self.spin_kl_prog.value(),
                     waga_out=self.spin_kl_waga.value())
        d = p['d']

        def licz(progress, cancel):
            grupy = p['grupy'] if len(p['kroki']) > 1 else None
            return KK.dobierz(p['H'], p['W'], p['war'], p['prog'], p['vals'], p['zdeg'], osie_obecne=p['osie'],
                              grupy=grupy, grupa_modelu=p['biezacy'],
                              postep=lambda t, f: progress({'phase': 'clusters', 'message': t, 'fraction': f}), **ustaw)

        def gotowe(w):
            if w is None:
                return
            if self.spin_kd.value() != d:                  # w międzyczasie zmieniono opóźnienia: inne mody
                self._set_text(self.kl_label, 'Osie pod klastry: zmieniono liczbę opóźnień, kliknij ponownie.'); return
            kroki = p['kroki']
            gdzie = f'zapis {kroki[0]}' if len(kroki) == 1 else f'{len(kroki)} zapisów ({kroki[0]}..{kroki[-1]})'
            self._set_text(self.kl_label, KK.opis(w, p['vals']) + f'\nOceniano: {gdzie}, k = 2..{ustaw["k_max"]}, '
                           f'outlier od {ustaw["prog"]:.1f} σ, kara {ustaw["waga_out"]:.1f}.')
            if w['model'] is None:
                return
            self.cloud.ustaw_klastry(w['model'])
            self._osie_koopmana(w['osie'])
            if self.chk_kl_kolor.isChecked():
                self.combo_color.setCurrentIndex(self.combo_color.findData(7))
            self._set_text(self.status, 'Osie pod klastry ustawione (opis na stronie „Koopman”).')
        self._set_text(self.kl_label, f'Osie pod klastry: oceniam {len(p["H"]):,} cząsteczek…')
        self._job(licz, gotowe)
