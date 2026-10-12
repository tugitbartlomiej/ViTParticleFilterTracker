"""Zakładka „Ruch zbioru”, część atencji: strona opcji „Atencja” i heatmapy atencji ViT z wag checkpointu
bieżącego kroku (jeden obraz od razu, cały zbiór przyciskiem, liczone na karcie).

Klasa bazowa dla `zakladka_ruchu.ZakladkaRuchu` (korzysta z jej `cloud`, `timeline`, `opcje`, `_int`, `_float`)."""
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QComboBox, QLabel

import ruch_naturalny as N
from styl import rola
from uklad import dodaj, pole_wyboru, przycisk


class AtencjaRuchu:
    def _strona_atencji(self):
        """Strona „Atencja”: heatmapy atencji ViT na cząsteczkach i zdjęciu."""
        self.opcje.dodaj_strone('Atencja')
        row = self.opcje.wiersz('Atencja')
        self.chk_att = pole_wyboru(row, 'Heatmapa atencji',
                                   'Nakłada mapę atencji ViT z wag zapisanego checkpointu bieżącego kroku (pomiary naturalne). '
                                   'Widok jednego obrazu: liczona od razu przy każdym kroku. Cały zbiór: przyciskiem „Policz '
                                   'dla całego zbioru” (1500 obrazów na karcie, kilka sekund). Rodzaj mapy, blok, głowę i '
                                   'skalę wybiera się niżej, jak w Uczenie ViT.')
        self.combo_att_gdzie = QComboBox()
        for tekst, v in (('na cząsteczkach', 1), ('na zdjęciu w przestrzeni', 2), ('na cząsteczkach i zdjęciu', 3)):
            self.combo_att_gdzie.addItem(tekst, v)
        dodaj(row, 'nakładanie', self.combo_att_gdzie,
              'Gdzie pokazać mapę: kolor (inferno) i rozmiar cząsteczek według wartości mapy patcha; plama ciepła na '
              'zdjęciu rozpiętym na cząsteczkach (strona „Wygląd”, tylko widok jednego obrazu); albo oba naraz.')
        self.spin_att_wzm = self._float(row, 'powiększ wg mapy', 0, 10, 2.5, 1,
                                        'Rozmiar cząsteczki = rozmiar · (0,4 + ta wartość · mapa). 0 = sam kolor.')
        self.spin_att_krycie = self._float(row, 'krycie na zdjęciu', 0, 1, .6, 2,
                                           'Jak mocno heatmapa przykrywa zdjęcie w przestrzeni (0 = wcale, 1 = sama mapa).')
        self.btn_att_zbior = przycisk(row, 'Policz dla całego zbioru',
                                      'Liczy na karcie mapy atencji wszystkich obrazów zbioru z wag bieżącego zapisanego '
                                      'kroku i nakłada je na cząsteczki. Przy zmianie kroku mapa zostaje z kroku, dla którego '
                                      'ją policzono (napis obok); kliknij ponownie, aby przeliczyć.')
        self.att_label = rola(QLabel('Atencja: wyłączona.'), 'opis'); self.att_label.setWordWrap(True)
        from opcje_atencji import OpcjeAtencji
        self.opcje_att = OpcjeAtencji(z_kryciem=False)
        for w in (self.opcje_att.combo_strz, self.opcje_att.spin_k, self.opcje_att.spin_grubosc):
            w.setEnabled(False)
            w.setToolTip('Strzałki atencji są w przestrzeni makrowektorów (Uczenie ViT); tutaj tylko heatmapy.')
        self.opcje._strony['Atencja'].addWidget(self.opcje_att)
        self.opcje._strony['Atencja'].addWidget(self.att_label)
        self._att_klucz = self._att_zbior = None
        self._det = self._det_sciezka = self._att_obrazy = None

    def _krok_atencji(self):
        """Zapisany krok, z którego wag liczona jest atencja (lewy zapis bieżącej pary)."""
        i, _ = self._indices()
        return self.timeline.steps[i]

    def _sciezka_wag(self, krok):
        """Plik wag kroku z metadanych pomiaru; gdy przeniesiono go (np. do assets), ten sam plik z historii
        checkpointów, rozpoznany po nazwie i sumie SHA-256 zapisanej w metadanych."""
        for r in (self.timeline.metadata.get('checkpoints') or []):
            if r.get('step') == krok:
                p = r.get('path')
                if p and Path(p).is_file():
                    return p
                import hashlib
                import historie_checkpointow as HC
                for h in HC.HISTORIE:
                    q = h.wagi / Path(str(p)).name
                    if q.is_file() and (not r.get('sha256') or hashlib.sha256(q.read_bytes()).hexdigest() == r['sha256']):
                        return str(q)
                return p
        return None

    def _detektor(self, sciezka):
        """Detektor ViT z wagami checkpointu (moduł detekcji ładowany dopiero przy pierwszym użyciu atencji)."""
        import torch
        import detekcja as D
        if self._det is None:
            self._det = D.Detektor(sciezka)
        elif self._det_sciezka != sciezka:
            ck = torch.load(sciezka, map_location='cpu', weights_only=True)
            self._det.net.load_state_dict(ck.get('net', ck), strict=True)
        self._det_sciezka = sciezka
        return self._det

    def _token_atencji(self):
        if self.zaznaczone:
            return np.array(sorted(self.zaznaczone))
        w = self.cloud.wybrany
        return None if w is None else int(w % (self.cloud.tpi or 600))

    def _atencja_odswiez(self, sila=False):
        tryb = self.combo_att_gdzie.currentData() if self.chk_att.isChecked() else 0
        self.cloud.ustaw_atencje(tryb=tryb, wzm=self.spin_att_wzm.value(), krycie=self.spin_att_krycie.value())
        if not tryb or not self.timeline:
            self._set_text(self.att_label, 'Atencja: wyłączona.' if not tryb else 'Atencja: wczytaj oś czasu.')
            self._budz(); return
        krok = self._krok_atencji()
        sciezka = self._sciezka_wag(krok)
        if sciezka is None:
            self._set_text(self.att_label, 'Atencja potrzebuje checkpointu każdego kroku: dostępna dla pomiarów '
                           'naturalnych (zapisane modele kolejnych kroków treningu).')
            return
        if self.combo_scope.currentIndex() != 1:
            z = self._att_zbior
            self._set_text(self.att_label, 'Cały zbiór: kliknij „Policz dla całego zbioru”.' if z is None else
                           f'Atencja całego zbioru z wag kroku {z[0]}' + ('' if z[0] == krok else
                           f' (bieżący krok {krok}: kliknij ponownie, aby przeliczyć)') + f'. {z[1]}')
            self._budz(); return
        if self.watek is not None:
            return                                         # model zajęty liczeniem w tle
        obraz = min(self.spin_image.value(), max(self.timeline.image_count - 1, 0))
        u = None
        try:
            import atencja_zbioru as AZ
            u = AZ.ustawienia(self.opcje_att)
            token = self._token_atencji()
            klucz = (str(self.timeline.root), krok, obraz, tuple(sorted(u.items())), None if token is None else str(token))
            if klucz == self._att_klucz and not sila:
                return
            rgb = self._zdjecie_obrazu(obraz)
            if rgb is None:
                self._set_text(self.att_label, f'Brak zdjęcia obrazu {obraz}: atencji nie da się policzyć.'); return
            det = self._detektor(sciezka)
            import torch
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().to(det.dev) / 255
            v = AZ.mapy(det.net, x, u, token)
            if v is None:
                self._set_text(self.att_label, 'Ta mapa potrzebuje wybranego patcha: kliknij cząsteczkę albo zaznacz '
                               'patche (Ctrl + klik).'); return
            self.cloud.ustaw_atencje(v[0], first=obraz * (self.cloud.tpi or 600))
            self._att_klucz = klucz
            self._set_text(self.att_label, f'Atencja obrazu {obraz} z wag kroku {krok}: {self.opcje_att.opis(token)} (znormowana w obrazie do 0..1).')
        except Exception as e:                             # brak CUDA, pliku wag albo zdjęcia: komunikat, bez przerwy
            self._set_text(self.att_label, f'Atencja niedostępna: {type(e).__name__}: {e}')
        self._budz()

    def _atencja_zbioru(self):
        """W tle: mapy atencji wszystkich obrazów z wag bieżącego kroku (obrazy dekodowane raz, na karcie)."""
        if not self.timeline:
            return
        krok = self._krok_atencji()
        sciezka = self._sciezka_wag(krok)
        m = self.timeline.metadata
        if sciezka is None or not m.get('resolved_image_paths') or not m.get('images'):
            self._set_text(self.att_label, 'Atencja zbioru potrzebuje checkpointów kroków i ścieżek zdjęć (pomiar '
                           'naturalny).'); return
        import atencja_zbioru as AZ
        u = AZ.ustawienia(self.opcje_att)
        token = self._token_atencji()
        if not self.chk_att.isChecked():
            self.chk_att.setChecked(True)

        def licz(progress, cancel):
            import torch
            if self._att_obrazy is None:
                progress({'phase': 'attention', 'message': 'Dekodowanie zdjęć zbioru (raz, do pamięci karty)…', 'fraction': 0})
                obrazy = [dict(r, index=i) for i, r in enumerate(m['images'])]
                sciezki = [Path(p) for p in m['resolved_image_paths']]
                self._att_obrazy = N._Obrazy(obrazy, sciezki, torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                             torch.cuda.is_available(), cancel).tensor
            det = self._detektor(sciezka)
            wynik = AZ.mapy_zbioru(det.net, self._att_obrazy, u, token,
                                   postep=lambda a, n: progress({'phase': 'attention', 'fraction': a / n,
                                                                  'message': f'Atencja zbioru: {a}/{n} obrazów'}),
                                   przerwij=cancel)
            return wynik

        def gotowe(wynik):
            if wynik is None:
                self._set_text(self.att_label, 'Ta mapa potrzebuje wybranego patcha: kliknij cząsteczkę albo zaznacz '
                               'patche (Ctrl + klik).'); return
            self.cloud.ustaw_atencje(wynik.reshape(-1), first=0)
            self._att_zbior = (krok, self.opcje_att.opis(token))
            self._att_klucz = None
            self._atencja_odswiez()
        self._job(licz, gotowe)
