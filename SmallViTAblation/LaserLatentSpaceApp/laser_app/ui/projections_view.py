"""Okienka R1–R8 zakładki „Lasery i koloidy”: rzuty z 192 wymiarów do 3D z miernikiem wierności (obliczenia: projections.py,
uzasadnienie: reports/Rzutowanie wielowymiarowego ruchu do 3D.md).

Po lewej zwijana sekcja „Rzuty z 192 wymiarów”: liczenie stanów 192D (raz na historię i zdjęcie, potem z dysku), przyciski
okienek i opcje. Jedno okienko = jeden rzut:
  R1 A — mapa klas: oś 1 = Δ narzędzie − tło (100 % separacji), osie 2–3 = PCA reszty chmury (wspólna albo w każdym checkpoincie);
  R2 S — siły: 3 kierunki największej energii sił netto bloku 6;
  R3 W — wiązki ważone aktywacją (następca P3) i wiązki wybranej cząsteczki (sieć silna);
  R4 B — rama siły wybranej cząsteczki: F̂ i płaszczyzna, w której jej wiązki się znoszą;
  R5 C — rama Schura: podprzestrzeń niezmiennicza pełnego jakobianu, pole lokalnej deformacji, płaszczyzna wiru i kierunek własny;
  R6 D — wędrówka w czasie uczenia: wszystkie, narzędzie, wybrane albo jedna cząsteczka, z torami zerowymi (błądzenie losowe);
  R7 F — 7 widoków płaszczyzny Fano nad 7 kierunkami (każda para kierunków razem w dokładnie jednym widoku);
  R8 — bez rzutu: wierność widoków w czasie, separacja klas wobec permutacji etykiet, kosinusy w 192D, pary wiązek, widmo J;
  R9 L — separacja narzędzia od tła: oś 1 = LDA ze skurczem (Δ wybielone kowariancją wewnątrz klas, dopasowane na puli ostatnich
       checkpointów), osie 2–3 do wyboru; w podpisie sylwetka, trafienia kNN, margines i stosunek Fishera z zerem permutacyjnym.
W okienkach globalnych (R1–R3, R6, R7) ślady to wędrówka cząsteczek przez checkpointy uczenia, a strzałki to siła netto bloku 6
(dokładny rzut; krycie = widoczna część długości ρ = |Pu|/|u|). Wszystkie pokazują chwilę osi czasu zakładki.
Rysowanie na karcie (projections_gl.ProjectionGPU, GLSL 4.30): cała historia w bazie okienka idzie do bufora raz (przy zmianie bazy, środka,
skali, wygładzania), w klatce tylko czas; ramy lokalne (wiązki, pole Schura) idą raz na checkpoint i cząsteczkę. Rzutowanie
192D → 3D całej historii liczy karta (projections.ProjectionData.wsp, torch)."""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from laser_app.compute import history as hist, projections
from laser_app.ui.projection_windows.bases import BasesMixin
from laser_app.ui.projection_windows.drawing import DrawingMixin
from laser_app.ui.projection_windows.frames import FramesMixin
from laser_app.ui.projection_windows.knn import KnnMixin
from laser_app.ui.projection_windows.metrics import MetricsMixin
from laser_app.ui.projection_windows.panel import PanelMixin
from laser_app.ui.projection_windows.photo_lda import PhotoLDAMixin
from laser_app.ui.projection_windows.scene import Scene
from laser_app.ui.projection_windows.shared import OKIENKA, pl, S_EKR
from laser_app.ui.projection_windows.workers import FrameCWorker, ProjectionWorker
from laser_app.ui.widgets import NarrowCombo


class ProjectionWindows(PhotoLDAMixin, MetricsMixin, KnnMixin, PanelMixin, FramesMixin, BasesMixin, DrawingMixin, QWidget):
    """Sekcja „Rzuty z 192 wymiarów” w lewym panelu zakładki i okienka R1–R8 (budowane przez buduj_okienka)."""

    def __init__(self, zakladka):
        super().__init__()
        from laser_app.ui.widgets import CollapsibleSection, _spin
        self._z = zakladka
        self.dane = None
        self._h = None
        self.t = 0.0
        self._watek = None
        self._sceny, self._lb, self._ostatnie = {}, {}, {}
        self._pam, self._u, self._frB = {}, {}, {}
        self._C = None; self._C_zlecony = None
        self._B = None
        self._blok = False
        self._i_panel = None
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        g = CollapsibleSection('Rzuty z 192 wymiarów: okienka R1–R8'); f = QFormLayout(g); lay.addWidget(g)
        self.sekcja = g
        opis = QLabel('Liczone w pełnych 192 wymiarach, rysowane w 3D. Pod każdym okienkiem: ile czego ten rzut pokazuje; losowy rzut '
                      '3D pokazuje średnio 1,6 % energii i 12,5 % długości wektora. Ślady = wędrówka cząsteczek przez checkpointy.')
        opis.setWordWrap(True); f.addRow(opis)
        hb = QHBoxLayout()
        self.bt_rz = QPushButton('Policz / wczytaj stany 192D'); self.bt_rz_stop = QPushButton('Przerwij')
        self.bt_rz.setToolTip('Jedno przejście w przód na każdy checkpoint wybranej historii (GPU): stany 192D przed MLP bloku 6, siły,\n'
                              'kolumny fc2, bazy rzutów i mierniki wierności. Wynik zostaje w katalogu historii (rzuty/), potem wczytuje się od razu.')
        self.bt_rz.clicked.connect(self._licz); self.bt_rz_stop.clicked.connect(self._przerwij); self.bt_rz_stop.setEnabled(False)
        hb.addWidget(self.bt_rz); hb.addWidget(self.bt_rz_stop); f.addRow(hb)
        self.pasek_rz = QProgressBar(); self.pasek_rz.setTextVisible(True); f.addRow(self.pasek_rz)
        self.lb_rz = QLabel('Najpierw „Policz / wczytaj” historię w sekcji Dane.'); self.lb_rz.setWordWrap(True); f.addRow(self.lb_rz)
        w_ = QWidget(); s_ = QGridLayout(w_); s_.setContentsMargins(0, 0, 0, 0); s_.setSpacing(2)
        for n, (k, sk, tyt) in enumerate(OKIENKA):
            b = QPushButton(f'{k} {sk}'); b.setToolTip(tyt + '\nKlik: pokaż to okienko na wierzchu.')
            b.clicked.connect(lambda _=False, k=k: self.pokaz(k))
            s_.addWidget(b, n // 4, n % 4)
        f.addRow('Okienka', w_)

        gw = CollapsibleSection('Wspólne: cząsteczki, ślady, strzałki (R1–R3, R6, R7)'); fw = QFormLayout(gw); f.addRow(gw)
        self.cb_kogo = NarrowCombo(); self.cb_kogo.addItems(['wszystkie cząsteczki', 'tylko narzędzie', 'tylko wybrane', 'narzędzie i wybrane'])
        self.cb_kogo.setToolTip('Które cząsteczki rysować. Wybrane = „Cząsteczka” i „+ cząsteczki” z sekcji sfery (biała = pierwsza).')
        self.cb_slad = NarrowCombo()
        self.cb_slad.addItems(['ślady: brak', 'ślady wybranych', 'ślady narzędzia', 'ślady narzędzia i wybranych', 'ślady wszystkich'])
        self.cb_slad.setCurrentIndex(3)
        self.cb_slad.setToolTip('Wędrówka: droga cząsteczki przez checkpointy uczenia w tym samym rzucie (rzut liniowy, więc droga rzutu =\n'
                                'rzut drogi). Jaśniejsza = nowsza.')
        self.sp_slad = _spin(0, 5000, 60, 5, sufiks=' ckpt'); self.sp_slad.setSpecialValueText('od początku uczenia')
        self.sp_slad.setToolTip('Długość śladu w checkpointach (0 = od pierwszego checkpointu).')
        self.sp_wyg = _spin(1, 200, 5, 1, sufiks=' ckpt'); self.sp_wyg.setSpecialValueText('bez wygładzania')
        self.sp_wyg.setToolTip('Ślady i tory zerowe jako średnia krocząca z tylu ostatnich checkpointów (tylko przeszłość). Szum SGD\n'
                               'między gęstymi checkpointami (co 10 kroków na początku) zamienia surowy ślad w gąszcz; wygładzony ślad\n'
                               'pokazuje dryf. Punkty cząsteczek zostają surowe. Odstępy checkpointów: 10 kroków do kroku 1000, potem 100.')
        self.cb_strz = NarrowCombo()
        self.cb_strz.addItems(['strzałki: brak', 'siła netto bloku 6: pokazane cząsteczki', 'siła netto bloku 6: tylko wybrane'])
        self.cb_strz.setCurrentIndex(2)
        self.cb_strz.setToolTip('Siła = przesunięcie przez MLP bloku 6 względem średniej po tokenach (suma 768 wiązek), dokładny rzut.')
        self.sp_strz = _spin(0.02, 50, 1.0, 0.1, 2, ' ×'); self.sp_strz.setToolTip('Długość strzałek sił i wiązek (te same jednostki co położenia).')
        self.ch_rho = QCheckBox('krycie strzałki = widoczna część długości ρ = |Pu|/|u|'); self.ch_rho.setChecked(True)
        self.ch_rho.setToolTip('Strzałka prawie niewidoczna = wektor leży prawie cały poza tym rzutem (jego długość na ekranie kłamie).')
        self.cb_srodek = NarrowCombo()
        self.cb_srodek.addItems(['środek: środek chmury w checkpoincie', 'środek: środek tła w checkpoincie (ruch względem tła)'])
        self.cb_srodek.setToolTip('Środek tła: widać, jak narzędzie odsuwa się od tła, a nie jak dryfuje cała chmura.')
        self.cb_skala = NarrowCombo()
        self.cb_skala.addItems(['skala: promień chmury w każdym checkpoincie (kształt)', 'skala: stała z końca uczenia (widać wzrost chmury)'])
        self.cb_skala.setToolTip('R = promień RMS chmury (średnia odległość tokenu od środka) w 192D.')
        self.sp_rozm = _spin(1, 20, 6, 0.5, 1, ' px')
        self.ch_kamera = QCheckBox('kamera śledzi wybraną cząsteczkę'); self.ch_kamera.setChecked(True)
        self.ch_kamera.setToolTip('Gdy jest wybrana cząsteczka, środek widoku jest na niej: LPM obraca wokół niej, kółko zbliża do niej.\n'
                                  'Bez wybranej cząsteczki (albo po wyłączeniu) kamera stoi na środku chmury.')
        self.ch_osie = QCheckBox('osie z nazwami'); self.ch_osie.setChecked(True)
        for nazwa, w in (('Pokazuj', self.cb_kogo), ('Wędrówka', self.cb_slad), ('Długość śladu', self.sp_slad),
                         ('Wygładzanie śladu', self.sp_wyg), ('Strzałki', self.cb_strz),
                         ('Długość strzałek', self.sp_strz), ('Środek', self.cb_srodek), ('Skala', self.cb_skala),
                         ('Wielkość punktów', self.sp_rozm)):
            fw.addRow(nazwa, w)
        fw.addRow(self.ch_rho); fw.addRow(self.ch_kamera); fw.addRow(self.ch_osie)

        ga = CollapsibleSection('R1 A, R3 W, R7 F'); fa = QFormLayout(ga); f.addRow(ga)
        self.cb_A = NarrowCombo()
        self.cb_A.addItems(['A: baza wspólna dla wszystkich checkpointów (stabilna)', 'A: baza w każdym checkpoincie (osie 2–3 wyrównane Procrustesem)'])
        self.cb_A.setToolTip('Wspólna: jedna kamera przez całe uczenie (najstabilniejsza wg benchmarku Verniera i in. 2020).\n'
                             'W każdym checkpoincie: oś 1 = Δ tego checkpointu, osie 2–3 obrócone do osi wspólnych; różnica między nimi\n'
                             'pokazuje, ile ruchu wnosi sama kamera.')
        self.ch_W = QCheckBox('R3: wiązki wybranej cząsteczki (sieć silna jak w R4)'); self.ch_W.setChecked(True)
        self.cb_F = NarrowCombo()
        self.cb_F.addItems([f'F {n + 1}: ' + ' · '.join(projections.NAZWY_F7[a] for a in tr) for n, tr in enumerate(projections.FANO)])
        self.cb_F.setToolTip('7 kierunków (Δ, 2 × reszta chmury, 2 × siły, 2 × wiązki), ortonormalizowanych w tej kolejności;\n'
                             'w 7 widokach płaszczyzny Fano każda para tych kierunków jest razem w dokładnie jednym widoku.')
        fa.addRow(self.cb_A); fa.addRow(self.ch_W); fa.addRow('R7 widok', self.cb_F)

        gb = CollapsibleSection('R4 B: rama siły'); fb = QFormLayout(gb); f.addRow(gb)
        self.sp_B_prog = _spin(0.2, 1.0, 0.8, 0.05, 2)
        self.sp_B_prog.setToolTip('Sieć silna: najsilniejsze wiązki, które razem niosą ten ułamek energii Σ|u_c|²; reszta jako jedna\n'
                                  'szara strzałka. Suma narysowanych = siła netto dokładnie (Radjai i in. 1998).')
        self.sp_B_skala = _spin(0.1, 20, 1.0, 0.1, 1, ' ×')
        fb.addRow('Sieć silna: ułamek energii', self.sp_B_prog); fb.addRow('Długość wiązek', self.sp_B_skala)

        gc = CollapsibleSection('R5 C: rama Schura'); fc = QFormLayout(gc); f.addRow(gc)
        self.ch_C = QCheckBox('licz przy zmianie checkpointu i cząsteczki (gdy R5 albo R8 widać)'); self.ch_C.setChecked(True)
        self.cb_C_para = NarrowCombo()
        self.cb_C_para.addItems(['para zespolona: najszybszy obrót |Im λ|', 'para zespolona: najmniej stabilna (max Re λ)',
                                 'para zespolona: najbliżej siły F̂'])
        self.cb_C_rzecz = NarrowCombo()
        self.cb_C_rzecz.addItems(['mod rzeczywisty: najsilniej wypycha (max Re λ)', 'mod rzeczywisty: najsilniej ściąga (min Re λ)',
                                  'mod rzeczywisty: najbliżej siły F̂'])
        self.sp_C_G = _spin(3, 15, 7, 2); self.sp_C_G.setToolTip('Węzłów siatki pola na bok (G³ punktów).')
        self.sp_C_zas = _spin(0.02, 3.0, 0.5, 0.05, 2, ' × odl. sąsiada')
        self.sp_C_zas.setToolTip('Pół boku sześcianu pola jako ułamek odległości do najbliższej innej cząsteczki w 192D.')
        self.cb_C_pole = NarrowCombo()
        self.cb_C_pole.addItems(['pole dokładne: F(x + Qu) − F(x)', 'część liniowa: (QᵀJQ) u'])
        self.sp_C_strz = _spin(0.1, 5, 1.0, 0.1, 1, ' ×')
        fc.addRow(self.ch_C); fc.addRow(self.cb_C_para); fc.addRow(self.cb_C_rzecz); fc.addRow('Siatka', self.sp_C_G)
        fc.addRow('Pół boku', self.sp_C_zas); fc.addRow('Pole', self.cb_C_pole); fc.addRow('Długość strzałek', self.sp_C_strz)

        gd = CollapsibleSection('R6 D: wędrówka'); fd = QFormLayout(gd); f.addRow(gd)
        self.cb_D = NarrowCombo()
        self.cb_D.addItems(['PCA przemieszczeń wszystkich cząsteczek (wspólna)', 'PCA przemieszczeń pokazanych cząsteczek',
                            'rama wybranej cząsteczki: start → koniec + odchylenia', 'mapa klas A (wspólna)'])
        self.cb_D.setToolTip('Kierunki ruchu (przemieszczeń między checkpointami), a nie rozrzutu chmury. Rama cząsteczki: oś 1 = jej\n'
                             'przesunięcie netto (widoczne w 100 %), osie 2–3 = główne zboczenia toru z tej prostej.')
        self.ch_D_zero = QCheckBox('tory zerowe wybranej: błądzenie losowe z tymi samymi krokami'); self.ch_D_zero.setChecked(True)
        self.ch_D_zero.setToolTip('Szare tory: te same długości kroków, losowe kierunki w 192D. W ramie cząsteczki każdy tor zerowy ma własną\n'
                                  'ramę (tę samą przewagę). PCA samego błądzenia daje łuki i pętle (Antognini i Sohl-Dickstein 2018):\n'
                                  'kształt toru znaczy coś tylko wtedy, gdy różni się od torów zerowych.')
        self.sp_D_zero = _spin(1, 30, 6, 1)
        fd.addRow(self.cb_D); fd.addRow(self.ch_D_zero); fd.addRow('Torów zerowych', self.sp_D_zero)

        gp = CollapsibleSection('R8: bez rzutu'); fp = QFormLayout(gp); f.addRow(gp)
        self.cb_P = NarrowCombo()
        self.cb_P.addItems(['wierność: rozrzut chmury', 'wierność: energia sił netto', 'wierność: energia wiązek',
                            'wierność: długość separacji Δ', 'wierność: ruch między checkpointami'])
        fp.addRow('Wykres wierności', self.cb_P)

        gl_ = CollapsibleSection('R9 L: separacja narzędzia od tła'); fl = QFormLayout(gl_); f.addRow(gl_)
        self.sp_L_lam = _spin(0.0, 10.0, 0.1, 0.05, 2)
        self.sp_L_lam.setToolTip('Skurcz LDA: w ∝ (S_w + λ·tr(S_w)/192·I)⁻¹ Δ. 0 = czysta LDA (przy 18 tokenach narzędzia przeucza),\n'
                                 'duże λ → sam kierunek Δ (jak oś 1 widoku A).')
        self.sp_L_N = _spin(1, 2000, 100, 10, sufiks=' ckpt')
        self.sp_L_N.setToolTip('Pula dopasowania: tyle ostatnich checkpointów (do checkpointu dopasowania) razem; więcej = stabilniejsza oś,\n'
                               'mniej = lepiej dopasowana do jednej chwili. 1 = tylko checkpoint dopasowania.')
        self.cb_L_kiedy = NarrowCombo(); self.cb_L_kiedy.addItems(['dopasuj na końcu uczenia (stała kamera)', 'dopasuj na bieżącym checkpoincie'])
        self.cb_L_osie = NarrowCombo()
        self.cb_L_osie.addItems(['osie 2–3: największy rozrzut wewnątrz klas', 'osie 2–3: najmniejszy rozrzut wewnątrz klas (klasy jak kropki)',
                                 'osie 2–3: PCA tła', 'osie 2–3: PCA narzędzia'])
        self.cb_L_etyk = NarrowCombo(); self.cb_L_etyk.addItems(['obrys narzędzia (udział pikseli w łacie)', 'ramka GT (stara etykieta)',
                                 'grupa wykryta bez etykiet (TokenCut na latencie MLP 768D)'])
        self.cb_L_etyk.setToolTip('Obrys: wielokąty narzędzia z adnotacji Cataract-LMM; narzędzie = łaty z udziałem pikseli ≥ próg,\n'
                                  'mieszane (0 < udział < próg) na żółto, poza dopasowaniem i miarami. Ramka GT obejmuje głównie tło\n'
                                  '(cienki cystotom: z 18 łat ramki 3 są w co najmniej połowie narzędziem).\n'
                                  'Grupa wykryta bez etykiet: TokenCut na aktywacjach MLP bloku 6 w danym checkpoincie (bez adnotacji);\n'
                                  'oś LDA dopasowana do tej grupy, obrys tylko do sprawdzenia (Dice, precyzja; żółte = narzędzie pominięte).')
        self.cb_L_os = NarrowCombo(); self.cb_L_os.addItems(['oś LDA z etykiety (obrys)', 'oś LDA z wykrytej grupy (bez etykiet)'])
        self.cb_L_os.setToolTip('Dla etykiety „grupa wykryta bez etykiet”: do czego dopasować oś LDA. Z obrysu: widok jak przy etykiecie,\n'
                                'a kolor narzędzia = grupa wykryta bez etykiet, więc widać, czy grupa leży tam, gdzie oznaczone narzędzie.\n'
                                'Z grupy: widok całkowicie bez etykiet. Przy innych etykietach oś zawsze z wybranej etykiety.')
        self.sp_L_prog = _spin(0.05, 1.0, 0.5, 0.05, 2)
        self.sp_L_prog.setToolTip('Próg udziału pikseli narzędzia w łacie (etykieta z obrysu).')
        self.ch_L_przestrz = QCheckBox('zero przestrzenne (ta sama maska w innych miejscach tła)'); self.ch_L_przestrz.setChecked(True)
        self.ch_L_przestrz.setToolTip('Maska narzędzia przesunięta w każde położenie siatki rozłączne z narzędziem, LDA od nowa dla każdego.\n'
                                      'Dowolny spójny kawałek tła też się „oddziela” (podobne sąsiednie łaty), więc to jest właściwe zero;\n'
                                      'p = odsetek położeń tła z miarą ≥ narzędzia. Liczone na pauzie, LDA dopasowana na bieżącym checkpoincie.')
        self.sp_L_brzeg = _spin(0, 3, 0, 1, sufiks=' kom.')
        self.sp_L_brzeg.setToolTip('Pomiń tyle skrajnych kolumn i wierszy ramki przy dopasowaniu i w miarach: prostokątna ramka obejmuje\n'
                                   'na brzegach piksele tła. Pominięte tokeny są rysowane na żółto.')
        self.sp_L_perm = _spin(0, 500, 100, 50); self.sp_L_perm.setToolTip('Losowych etykiet do zera miar (95. percentyl); 0 = bez zera.')
        self.ch_L_refit = QCheckBox('zero z ponownym dopasowaniem osi (DiProPerm)'); self.ch_L_refit.setChecked(True)
        self.ch_L_refit.setToolTip('Dla każdej losowej etykiety oś jest dopasowywana od nowa (na bieżącym checkpoincie), więc zero obejmuje\n'
                                   'przeuczenie 18 punktów w 192D. Uczciwsze od zera przy stałych osiach (dolne ograniczenie).')
        self.ch_L_cv = QCheckBox('walidacja blokowana po checkpointach (oś z innych bloków)'); self.ch_L_cv.setChecked(True)
        self.ch_L_cv.setToolTip('Pula checkpointów dzielona na bloki ciągłe; oś dopasowana bez bloku jest oceniana na jego środkowym checkpoincie.\n'
                                'Chroni przed korelacją czasową; nie chroni przed tymi samymi tokenami (te są w każdym checkpoincie).')
        self.sp_L_bl = _spin(2, 20, 5, 1, sufiks=' bloków')
        for nazwa, w in (('Skurcz λ', self.sp_L_lam), ('Pula dopasowania', self.sp_L_N), ('Kiedy', self.cb_L_kiedy),
                         ('Osie 2–3', self.cb_L_osie), ('Etykieta', self.cb_L_etyk), ('Oś LDA', self.cb_L_os), ('Próg udziału', self.sp_L_prog),
                         ('Brzeg ramki', self.sp_L_brzeg), ('Permutacji', self.sp_L_perm)):
            fl.addRow(nazwa, w)
        fl.addRow(self.ch_L_przestrz); fl.addRow(self.ch_L_refit); fl.addRow(self.ch_L_cv); fl.addRow('Bloków CV', self.sp_L_bl)
        g0 = CollapsibleSection('R0 i R10: zdjęcie z patchami i LDA na żywo'); f0 = QFormLayout(g0); f.addRow(g0)
        self.ch_R0_grupa = QCheckBox('R0: pokaż grupę wykrytą bez etykiet (czerwone patche)'); self.ch_R0_grupa.setChecked(True)
        self.ch_R0_grupa.setToolTip('TokenCut na aktywacjach MLP bloku 6 w bieżącym checkpoincie. Podczas odtwarzania liczy się tylko to, co już\n'
                                    'jest w pamięci; reszta na pauzie.')
        self.ch_R0_siatka = QCheckBox('R0: siatka patchy 16 × 16'); self.ch_R0_siatka.setChecked(False)
        self.cb_L10 = NarrowCombo(); self.cb_L10.addItems(['1. panel: oś jak w R9 (stała kamera)', '1. panel: oś dopasowana na tym checkpoincie',
                              '1. panel: PCA3 chmury (bez etykiet)'])
        self.cb_L10.setToolTip('Osie 2. i 3. panelu zawsze dopasowane od nowa do przesuniętej maski (zero przestrzenne).')
        self.ch_L10_tlo = QCheckBox('R10: ta sama maska w tle (mediana i najlepsze położenie)'); self.ch_L10_tlo.setChecked(True)
        self.ch_L10_tlo.setToolTip('Maska narzędzia przesunięta w każde położenie tła, LDA od nowa. Liczy się na pauzie (ok. 1 s).')
        self.cb_R12_widok = NarrowCombo(); self.cb_R12_widok.addItems(['PCA3 chmury (bez etykiet)', 'oś LDA jak w R9'])
        self.cb_R12_przestrzen = NarrowCombo(); self.cb_R12_przestrzen.addItems(['sąsiedzi w widoku 3D', 'sąsiedzi w pełnych 192D (rysunek w 3D)'])
        self.cb_R12_przestrzen.setToolTip('Gdzie szukać najbliższych sąsiadów: w tych 3 wymiarach, które widać, czy w pełnych 192D (wtedy linie\n'
                                          'mogą prowadzić do tokenów, które w rzucie są daleko).')
        self.sp_R12_k = _spin(0, 50, 0, 1); self.sp_R12_k.setSpecialValueText('auto (min(10, n − 1))')
        self.ch_R12_kr = QCheckBox('R12: linie do sąsiadów'); self.ch_R12_kr.setChecked(True)
        self.cb_R12_baza = NarrowCombo()
        self.cb_R12_baza.addItems(['baza PCA: wspólna dla całej historii (stabilna)',
                                   'baza PCA: w każdym checkpoincie, wyrównana do wspólnej (Procrustes 3 × 3)',
                                   'baza PCA: w każdym checkpoincie surowa (skacze)'])
        self.cb_R12_baza.setToolTip('Dotyczy widoku „PCA3 chmury”. Wspólna: osie z kowariancji zebranej po próbce checkpointów całej historii, liczone\n'
                                    'raz; kamera nie obraca się w czasie (domyślna, najstabilniejsza). W każdym checkpoincie wyrównana: osie PCA tego\n'
                                    'checkpointu obrócone/odbite Procrustesem 3 × 3 do osi wspólnych (pokazuje kształt dopasowany na bieżąco, bez\n'
                                    'przeskoków znaku i kolejności osi). Surowa: PCA od nowa w każdym checkpoincie, bez wyrównania; znaki osi i ich\n'
                                    'kolejność przeskakują, gdy wartości własne się mijają, więc chmura skacze (stare zachowanie; bez interpolacji).')
        self.sp_R12_bwyg = _spin(0, 50, 0, 1, sufiks=' ckpt'); self.sp_R12_bwyg.setSpecialValueText('wyłączone')
        self.sp_R12_bwyg.setToolTip('Tylko dla bazy „w każdym checkpoincie” (wyrównanej albo surowej): kowariancja przed liczeniem osi uśredniana po\n'
                                    'checkpointach od t − n do t + n, więc osie zmieniają się powoli. 0 = bez uśredniania.')
        self.ch_R12_interp = QCheckBox('R12: interpolacja między checkpointami'); self.ch_R12_interp.setChecked(True)
        self.ch_R12_interp.setToolTip('Położenie tokenu = (1 − f)·P_i + f·P_j między sąsiednimi checkpointami i i j (każdy w swojej bazie, więc bez\n'
                                      'skoku przy zmianie checkpointu), a kolor mieszany tak samo z ułamków sąsiadów. Wyłączona: ruch skokowy co\n'
                                      'checkpoint (najbliższy). Linie do sąsiadów zawsze z najbliższego checkpointu. Nie dotyczy bazy surowej.')
        self.cb_R12_metryka = NarrowCombo(); self.cb_R12_metryka.addItems(['metryka sąsiadów: euklidesowa', 'metryka sąsiadów: kosinusowa'])
        self.cb_R12_metryka.setToolTip('Jak mierzyć bliskość tokenów przy szukaniu sąsiadów: odległością euklidesową albo kosinusem kąta między\n'
                                       'wektorami (w widoku 3D: współrzędnych wyśrodkowanych, w 192D: stanów tokenów; długość nie ma znaczenia).')
        self.sp_R12_fwyg = _spin(0, 50, 0, 1, sufiks=' ckpt'); self.sp_R12_fwyg.setSpecialValueText('wyłączone')
        self.sp_R12_fwyg.setToolTip('Ułamek sąsiadów-narzędzi (kolor tokenu) uśredniony po checkpointach od t − n do t + n: sąsiedzi liczeni w każdym\n'
                                    'z nich osobno (w bieżącej bazie rzutu), potem średnia. Uspokaja migotanie koloru. 0 = bez uśredniania.')
        f0.addRow(self.ch_R0_grupa); f0.addRow(self.ch_R0_siatka); f0.addRow('R10', self.cb_L10); f0.addRow(self.ch_L10_tlo)
        f0.addRow('R12 widok', self.cb_R12_widok); f0.addRow('R12 baza PCA', self.cb_R12_baza); f0.addRow('R12 wygładzanie bazy ±', self.sp_R12_bwyg)
        f0.addRow(self.ch_R12_interp); f0.addRow('R12 sąsiedzi', self.cb_R12_przestrzen); f0.addRow('R12 metryka', self.cb_R12_metryka)
        f0.addRow('R12 k', self.sp_R12_k); f0.addRow('R12 wygładzanie ułamka ±', self.sp_R12_fwyg); f0.addRow(self.ch_R12_kr)
        for w_ in (self.cb_R12_widok, self.cb_R12_przestrzen, self.cb_R12_baza, self.cb_R12_metryka):
            w_.currentIndexChanged.connect(self._opcje)
        for w_ in (self.sp_R12_k, self.sp_R12_bwyg, self.sp_R12_fwyg):
            w_.valueChanged.connect(self._opcje)
        self.ch_R12_kr.toggled.connect(self._opcje); self.ch_R12_interp.toggled.connect(self._opcje)
        for w_ in (self.ch_R0_grupa, self.ch_R0_siatka, self.ch_L10_tlo):
            w_.toggled.connect(self._opcje)
        self.cb_L10.currentIndexChanged.connect(self._opcje)
        for z_ in (gw, ga, gb, gc, gd, gp, gl_, g0):
            z_.naglowek.setChecked(False)                                  # podsekcje zwinięte: panel nie rośnie
        # okienko → podsekcje z jego ustawieniami (prawy klik w okienko R otwiera je w popupie, tab.menu_okienka)
        self._sekcje_R = {'R0': (g0,), 'R1': (ga, gw), 'R2': (gw,), 'R3': (ga, gw), 'R4': (gb,), 'R5': (gc,), 'R6': (gd, gw),
                          'R7': (ga, gw), 'R8': (gp,), 'R9': (gl_,), 'R10': (g0, gl_), 'R11': (gl_,), 'R12': (g0,)}
        for w in (self.cb_kogo, self.cb_slad, self.cb_strz, self.cb_srodek, self.cb_skala, self.cb_A, self.cb_F, self.cb_D,
                  self.cb_L_kiedy, self.cb_L_osie, self.cb_L_etyk, self.cb_L_os):
            w.currentIndexChanged.connect(self._opcje)
        for w in (self.sp_slad, self.sp_wyg, self.sp_strz, self.sp_rozm, self.sp_B_prog, self.sp_B_skala, self.sp_C_strz, self.sp_D_zero,
                  self.sp_L_lam, self.sp_L_N, self.sp_L_brzeg, self.sp_L_perm, self.sp_L_bl, self.sp_L_prog):
            w.valueChanged.connect(self._opcje)
        for w in (self.ch_rho, self.ch_kamera, self.ch_osie, self.ch_W, self.ch_D_zero, self.ch_L_refit, self.ch_L_cv, self.ch_L_przestrz):
            w.toggled.connect(self._opcje)
        for w in (self.cb_C_para, self.cb_C_rzecz, self.cb_C_pole):
            w.currentIndexChanged.connect(lambda *_: self._rysuj_widoczne())
        for w in (self.sp_C_G, self.sp_C_zas):
            w.valueChanged.connect(lambda *_: self._rysuj_widoczne())
        self.ch_C.toggled.connect(lambda *_: self._rysuj_widoczne())
        self.cb_P.currentIndexChanged.connect(lambda *_: self._panel_wiernosc())

    def sekcje_okienka(self, k):
        """Podsekcje panelu z ustawieniami okienka k (R0–R12); pusta lista dla nieznanego."""
        return list(self._sekcje_R.get(k, ()))

    # ------------------------------------------------------------------ budowa okienek
    def buduj_okienka(self):
        z = self._z
        for k, _, tyt in OKIENKA:
            if k == 'R8':
                w = self._buduj_panel()
            elif k == 'R0':
                w = self._buduj_zdjecie()
            elif k == 'R10':
                w = self._buduj_lda()
            elif k == 'R11':
                w = self._buduj_miary()
            elif k == 'R12':
                w = self._buduj_knn()
            else:
                sc = Scene()
                self._sceny[k] = sc
                sc.v.klik_px.connect(lambda x, y, k=k: self._klik(k, x, y))
                w = self._z_napisem(sc.v, k)
            d = z._dok(k, tyt, w)
            d.visibilityChanged.connect(lambda on, k=k: self._rysuj(k) if on else None)
        z.sp_tok.valueChanged.connect(lambda *_: self._wybor())
        z.ed_tok.editingFinished.connect(self._wybor)
        self._watek_C = FrameCWorker()
        self._watek_C.gotowe.connect(self._C_gotowe)

    def uklad_domyslny(self):
        """R1–R7 w kartach obok sfery reflektorów, R8 w karcie obok wykresu 6."""
        d, okna = self._z.doki, self._z.okna
        prev = d['s']
        for k in ('R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R9', 'R10', 'R11', 'R12'):
            okna.tabifyDockWidget(prev, d[k])
            prev = d[k]
        okna.tabifyDockWidget(d['6'], d['R8'])
        d['s'].raise_(); d['6'].raise_()

    def _z_napisem(self, widok, k):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(widok, 1)
        lb = QLabel('')
        lb.setStyleSheet('color:#ffd166; padding:2px 6px; background:#0b1014')
        lb.setWordWrap(True)
        lb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        lb.setFixedHeight(4 * lb.fontMetrics().height() + 10)              # stała wysokość: zmiana napisu nie przelicza doków
        lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        from laser_app.ui.dock_windows import bez_ukladu
        v.addWidget(bez_ukladu(lb))                                         # setText nie przelicza układu okna (dock_windows._LayoutShield)
        self._lb[k] = lb
        return w

    # ------------------------------------------------------------------ dane
    def ustaw_historie(self, h):
        self._h = h
        self.dane = None
        self._pam.clear(); self._u.clear(); self._frB.clear(); self._C = None; self._C_zlecony = None; self._B = None; self._i_panel = None
        for sc in self._sceny.values():
            sc.kadr = None
        try:
            self.dane = projections.wczytaj(h)
        except Exception:                                                # noqa: BLE001
            traceback.print_exc()
        if self.dane is None:
            self.lb_rz.setText('Stany 192D tej historii nie są jeszcze policzone: „Policz / wczytaj stany 192D” '
                               f'({h.K} przejść w przód na GPU, potem z dysku).')
            for lb in self._lb.values():
                lb.setText('Brak stanów 192D: sekcja „Rzuty z 192 wymiarów” → „Policz / wczytaj stany 192D”.')
            return
        self._po_wczytaniu()

    def _po_wczytaniu(self):
        d = self.dane
        self.lb_rz.setText(f'Stany 192D: {d.K} checkpointów. Zgodność z rzutem sfery P3: błąd {pl(float(d.zgodnosc_P3) * 100, 2)} % '
                           f'(float16). Wymiar efektywny chmury (współczynnik uczestnictwa) {pl(float(d.PR[0]), 1)} → '
                           f'{pl(float(d.PR[-1]), 1)}.')
        self._panel_wiernosc()
        self._rysuj_widoczne()

    def _licz(self):
        h = self._z.h
        if h is None or self._watek is not None:
            return
        self._watek = ProjectionWorker(h)
        self._watek.postep.connect(lambda i, n, o: (self.pasek_rz.setRange(0, n), self.pasek_rz.setValue(i), self.lb_rz.setText(o)))
        self._watek.gotowe.connect(self._policzone)
        self._watek.blad.connect(lambda s: (self.lb_rz.setText(f'Nie policzono: {s}'), self._koniec_liczenia()))
        self._watek.finished.connect(lambda: setattr(self, '_watek', None))
        self.bt_rz.setEnabled(False); self.bt_rz_stop.setEnabled(True)
        self.lb_rz.setText('Liczenie stanów 192D na GPU…')
        self._watek.start()

    def _koniec_liczenia(self):
        self.bt_rz.setEnabled(True); self.bt_rz_stop.setEnabled(False)

    def _przerwij(self):
        if self._watek is not None:
            self._watek.stop = True

    def _policzone(self, katalog, d):
        self._koniec_liczenia()
        if self._h is None or str(self._h.katalog) != katalog:
            return                                                       # w międzyczasie wczytano inną historię
        self.dane = d
        self._pam.clear(); self._u.clear(); self._frB.clear(); self._i_panel = None
        for sc in self._sceny.values():
            sc.kadr = None
        self._po_wczytaniu()

    # ------------------------------------------------------------------ czas, wybór, opcje
    def ustaw_czas(self, t):
        self.t = float(t)
        if self.dane is None:
            return
        self._rysuj_widoczne()

    def _rysuj_widoczne(self):
        for k, _, _ in OKIENKA:                                          # okienka w nieaktywnych kartach nie liczą (widac)
            if self._z.widac(k):
                self._rysuj(k)

    def _wybor(self):
        for k in [k for k in self._pam if k[0] in ('D1', 'Dr', 'zero', 'zero3', 'ruch', 'ruchj')]:
            del self._pam[k]
        self._B = None
        self._rysuj_widoczne()

    def _opcje(self, *_):
        for k in [k for k in self._pam if k[0] in ('D1', 'Dr', 'zero', 'zero3', 'ruch', 'ruchj', 'L', 'Lm')]:
            del self._pam[k]
        self._rysuj_widoczne()

    def pokaz(self, k):
        d = self._z.doki.get(k)
        if d is not None:
            d.show(); d.raise_()
            self._rysuj(k)

    def _indeksy(self):
        K = self.dane.K
        t = float(np.clip(self.t, 0, K - 1))
        i = int(np.floor(t)); j = min(i + 1, K - 1)
        return i, j, t - i, int(np.clip(round(t), 0, K - 1))

    def _wybrane(self):
        return self._z._wybrane_tok()

    def _maski(self):
        d = self.dane
        narz = np.asarray(d.narz, bool)
        wyb = np.zeros(hist.TOK, bool); wyb[self._wybrane()] = True
        pok = [np.ones(hist.TOK, bool), narz, wyb, narz | wyb][self.cb_kogo.currentIndex()]
        sl = [np.zeros(hist.TOK, bool), wyb, narz, narz | wyb, np.ones(hist.TOK, bool)][self.cb_slad.currentIndex()]
        return pok, wyb, sl & pok

    def _skala_k(self):
        R = np.asarray(self.dane.R, np.float64)
        return S_EKR / np.maximum(R if self.cb_skala.currentIndex() == 0 else np.full_like(R, R[-1]), 1e-12)

    def _kursor(self, l):
        if self._blok or self._z.h is None:
            return
        self._z.ustaw_czas(float(l.value()))

    def _klik(self, k, x, y):
        P = self._ostatnie.get(k)
        if P is None:
            return
        P, pok = P
        px, gl_ = self._sceny[k].v.na_ekran(P)
        dd = np.hypot(px[:, 0] - x, px[:, 1] - y)
        dd[~pok | ~np.isfinite(dd) | ~np.isfinite(gl_)] = np.inf
        j = int(np.argmin(dd))
        if dd[j] < 25:
            self._z.sp_tok.setValue(j)
