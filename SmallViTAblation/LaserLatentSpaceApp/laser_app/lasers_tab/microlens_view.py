"""Okienko M: mikrosoczewki (microlens_gl.MicrolensScene) razem z drugą sferą reflektorów w jednym widoku 3D.

Zdjęcie na dole naświetla przez płyty łodygi 600 półfabrykatów soczewek (stan 0), w sferze lasery bloków i gwiazda uwagi
formują ich kształt (12 podkroków), na górze lampa prześwietla matrycę gotowych soczewek na ekran: każda soczewka tylko na
własną komórkę (głowa jest na token), jasność = ciepło (pewność), ramka = prostokąt plamki najjaśniejszej soczewki.
Dane: przejście bieżącego checkpointu (history.podkroki, klucz 'glowa' = głowa na każdym z 13 stanów), to samo co okienko P."""
from __future__ import annotations

from collections import ChainMap

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QFormLayout

from laser_app.compute import history as hist
from laser_app.gpu.microlens_gl import MicrolensScene
from laser_app.gpu.sphere_gl import ReflectorSphere
from laser_app.lasers_tab.shared import TLO
from laser_app.ui.gl_view import GLView
from laser_app.ui.widgets import CollapsibleSection, NarrowCombo, _spin

FILTRY = ['brak', 'tokeny narzędzia', 'wybrane cząsteczki', 'wszystkie 600']


class _Mieszanka:
    """Leniwe mieszanie (1 − f)·A[i] + f·B[i] dużych tablic przejścia (z, ah): liczy tylko indeks, o który prosi sfera."""

    def __init__(self, A, B, f):
        self.A, self.B, self.f = A, B, float(f)

    def __getitem__(self, i):
        return ((1.0 - self.f) * np.asarray(self.A[i], np.float32) + self.f * np.asarray(self.B[i], np.float32)).astype(np.float32)


def mieszaj_przejscia(da, db, f):
    """Przejście „między” checkpointami a i b (f = 0..1): położenia 13 stanów, głowa, kierunki laserów i natężenia, uwaga.
    Oba przejścia są w tym samym rzucie 3D (baza.npz), więc mieszanie położeń jest ciągłe; reflektory stoją (slot z a)."""
    if db is None or f <= 1e-4:
        return da
    if f >= 1 - 1e-4:
        return db
    m = dict(da)
    for k in ('pos_n', 'glowa'):
        if k in da and k in db:
            m[k] = ((1 - f) * np.asarray(da[k], np.float32) + f * np.asarray(db[k], np.float32)).astype(np.float32)
    dn = (1 - f) * np.asarray(da['dirs_n'], np.float32) + f * np.asarray(db['dirs_n'], np.float32)
    m['dirs_n'] = (dn / np.maximum(np.linalg.norm(dn, axis=2, keepdims=True), 1e-12)).astype(np.float32)
    m['z'] = _Mieszanka(da['z'], db['z'], f); m['ah'] = _Mieszanka(da['ah'], db['ah'], f)
    m['krok'] = da['krok']
    return m


class MicrolensMixin:
    """Okienko M: mikrosoczewki. Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_mikro(self):
        """Sekcja panelu „Mikrosoczewki” (okienko M)."""
        g = CollapsibleSection('Mikrosoczewki (okienko M): zdjęcie → łodyga → formowanie w sferze → lampa i ekran'); f = QFormLayout(g)
        self.ch_mk_obraz = QCheckBox('zdjęcie na dole: naświetla półfabrykaty soczewek'); self.ch_mk_obraz.setChecked(True)
        self.ch_mk_obraz.setToolTip('Klatka wideo pod sferą. W alegorii światło zdjęcia nie przechodzi na ekran: przez filtry łodygi\n'
                                    'odciska początkowy kształt każdej mikrosoczewki (stan 0 = łodyga + kodowanie pozycji).')
        self.ch_mk_plyty = QCheckBox('płyty łodygi: 4 konwolucje (3 główne składowe cech jako RGB)'); self.ch_mk_plyty.setChecked(True)
        self.ch_mk_plyty.setToolTip('Cztery płyty między zdjęciem a sferą: wyjścia kolejnych konwolucji łodygi (240×160 → 30×20), każde\n'
                                    'jako obraz PCA-RGB. Pokazują, co z obrazu przechodzi do kształtu soczewek.')
        self.sp_mk_krycie_plyt = _spin(0.05, 1.0, 0.55, 0.05, 2); self.sp_mk_krycie_plyt.setToolTip('Krycie płyt łodygi (1 = nieprzezroczyste).')
        self.cb_mk_prom = NarrowCombo(); self.cb_mk_prom.addItems([f"promienie naświetlenia: {x}" for x in FILTRY]); self.cb_mk_prom.setCurrentIndex(3)
        self.cb_mk_prom.setToolTip('Linie od komórki zdjęcia do jej koloidu w sferze (kolor = kolor łaty): które soczewki pokazać.\n'
                                   'Wszystkie 600 naraz zasłaniają sferę; narzędzie albo wybrane cząsteczki odsłaniają szczegóły.')
        self.cb_mk_linki = NarrowCombo(); self.cb_mk_linki.addItems([f"linki koloid → soczewka: {x}" for x in FILTRY]); self.cb_mk_linki.setCurrentIndex(0)
        self.cb_mk_linki.setToolTip('Linie od koloidu w sferze do jego mikrosoczewki w matrycy nad sferą (jaśniejsze = większa transmisja).\n'
                                    'Soczewka przyjmuje kształt koloidu: jego położenie w 192 wymiarach.')
        self.cb_mk_rown = NarrowCombo()
        self.cb_mk_rown.addItems(['ogniskowanie: stałe pole równoległe, koloid odchyla światło w środek ramki przewidzianej przez głowę',
                                  'skupianie: koloid zbiera pęk równoległych promieni spod siebie i skupia go w swojej soczewce',
                                  'jedna wiązka: równolegle z oryginalnego zdjęcia, załamanie przy wejściu w szkło do koloidu, linkiem dalej',
                                  'pod koloidem: równolegle prosto na koloid (zdjęcie zmniejszone pod chmurę)'])
        self.cb_mk_rown.setCurrentIndex(0)
        self.cb_mk_rown.setToolTip('Ogniskowanie: zdjęcie świeci stałym polem równoległym (pionowo; kąt i położenie promieni nie zależą od koloidów).\n'
                                   'Koloid stoi w tym polu, zbiera pęk spod siebie i wypuszcza go pod własnym kątem w punkt ekranu, który głowa\n'
                                   'tego tokenu podaje jako środek ramki (komórka + (r − l)/2, (b − t)/2); jasność wiązki wyjściowej = ciepło.\n'
                                   'Koloidy narzędzia celują w to samo miejsce: tam powstaje jasna plama. Ruch koloidu zmienia tylko kąt wyjścia.\n'
                                   'Skupianie: oryginalne zdjęcie (mieści się w obrysie szklanej sfery) świeci równolegle w górę; pod każdym koloidem\n'
                                   'pęk kilku pionowych promieni przechodzi przez płyty łodygi do płaszczyzny koloidu (apertura), a koloid skupia go\n'
                                   'w ognisku: swojej mikrosoczewce w matrycy. Kolor promienia = kolor zdjęcia w miejscu, skąd wychodzi.\n'
                                   'Z oryginalnej łaty: jedna wiązka z własnej łaty, pionowo do wejścia w szkło, tam załamanie do koloidu.\n'
                                   'Pod koloidem: zdjęcie zmniejszone pod chmurę, jedna wiązka prosto na koloid.')
        self.sp_mk_wzm = _spin(0.1, 60.0, 6.0, 0.5, 2)
        self.sp_mk_wzm.setToolTip('Wzmocnienie wiązek wychodzących z koloidów w trybie ogniskowania (jasność × ciepło × to wzmocnienie).')
        self.ch_mk_pole = QCheckBox('stałe pole równoległe ze zdjęcia (pionowo do powierzchni szklanej sfery)'); self.ch_mk_pole.setChecked(True)
        self.ch_mk_pole.setToolTip('Słabe pionowe promienie z każdej łaty zdjęcia aż do powierzchni szklanej sfery: pole światła, które się NIE zmienia,\n'
                                   'niezależnie od tego, gdzie stoją koloidy. Na szkle pole się kończy: dalej biegnie tylko światło, które przeszło\n'
                                   'przez koloidy (promienie naświetlenia, wiązki wychodzące z koloidów).')
        self.sp_mk_jas_pola = _spin(0.0, 2.0, 0.12, 0.02, 2); self.sp_mk_jas_pola.setToolTip('Jasność stałego pola równoległego.')
        self.ch_mk_zal = QCheckBox('załamanie na wejściu i wyjściu ze szklanej sfery (jak na soczewce)'); self.ch_mk_zal.setChecked(True)
        self.ch_mk_zal.setToolTip('Wejście: pionowy promień stałego pola trafia w dolną półkulę i łamie się do szkła (powietrze → szkło); punkt wejścia jest\n'
                                  'taki, że załamany promień dochodzi do koloidu (zasada Fermata dla fali płaskiej), kolor = zdjęcie pod punktem wejścia.\n'
                                  'Wyjście: wiązka z koloidu biegnie w szkle do powierzchni sfery i tam łamie się (prawo Snella, szkło → powietrze),\n'
                                  'jak na soczewce kulistej. Punkt wyjścia wynika z zasady Fermata (najkrótsza droga optyczna n·|w szkle| + |w powietrzu|),\n'
                                  'więc po załamaniu wiązka dalej trafia w swój cel: środek ramki z głowy (ogniskowanie) albo soczewkę (skupianie).')
        self.sp_mk_n = _spin(1.0, 3.0, 1.5, 0.05, 2); self.sp_mk_n.setToolTip('Współczynnik załamania szkła sfery (1 = bez załamania, szkło ok. 1,5, diament 2,4).')
        self.sp_mk_pek_n = _spin(1, 24, 6, 1); self.sp_mk_pek_n.setToolTip('Promieni w pęku skupianym przez jeden koloid (tryb skupiania).')
        self.sp_mk_pek_ap = _spin(0.1, 6.0, 1.0, 0.1, 2)
        self.sp_mk_pek_ap.setToolTip('Apertura koloidu: promień pęku równoległych promieni, który koloid zbiera i skupia w ognisku (jednostki sfery).')
        self.sp_mk_skala = _spin(0.3, 3.0, 1.4, 0.1, 2, ' ×')
        self.sp_mk_skala.setToolTip('Wielkość chmury koloidów w okienku M względem dużej sfery. Zdjęcie, płyty łodygi, matryca soczewek\n'
                                    'i ekran dostają tę samą podstawę co chmura, więc wiązki idą równolegle w górę, załamują się na\n'
                                    'koloidach i dochodzą do soczewek pod niewielkim kątem (kolumna optyczna).')
        self.sp_mk_jas_prom = _spin(0.1, 10.0, 0.7, 0.1, 2); self.sp_mk_jas_prom.setToolTip('Jasność promieni naświetlenia i linków (mnożnik).')
        self.sp_mk_krycie_prom = _spin(0.02, 1.0, 0.25, 0.02, 2); self.sp_mk_krycie_prom.setToolTip('Krycie promieni naświetlenia i linków.')
        self.ch_mk_jas_zdj = QCheckBox('jasność promieni ze zdjęcia (tylko ilustracja: sieć nie przenosi jasności łat)'); self.ch_mk_jas_zdj.setChecked(False)
        self.ch_mk_jas_zdj.setToolTip('Wyłączone (domyślnie): promień naświetlenia ma barwę swojej łaty, ale jasność niezależną od zdjęcia (stała luminancja,\n'
                                      'dalej mnożona przez jasność promieni, krycie, wzmocnienie i ciepło jak dotąd). Luminancja łaty nie ma odpowiednika\n'
                                      'w sieci: łata wchodzi do konwolucji łodygi, nie do szkła, więc jasna łata nie daje „jaśniejszego” koloidu.\n'
                                      'Włączone: jasność promienia = jasność zdjęcia w miejscu, skąd wychodzi (dawny wygląd, tylko ilustracja).')
        self.ch_mk_sfera = QCheckBox('sfera reflektorów w środku: koloidy, lasery bloku, linie uwagi'); self.ch_mk_sfera.setChecked(True)
        self.ch_mk_sfera.setToolTip('Druga sfera reflektorów (te same ustawienia co duża sfera, sekcja „Sfera”) w trybie przejścia przez sieć:\n'
                                    'w podkroku MLP świecą lasery bloku, w podkroku uwagi linie między cząsteczkami. Formowanie soczewek.')
        self.cb_mk_las = NarrowCombo()
        self.cb_mk_las.addItems(['lasery: jak na dużej sferze', 'lasery: brak', 'lasery: wszystkie wiązki (768 × 600)',
                                 'lasery: do wybranych cząsteczek', 'lasery: z wybranych reflektorów'])
        self.cb_mk_las.setToolTip('Wiązki laserów MLP w sferze okienka M (w podkrokach MLP świecą lasery bloku): jak na dużej sferze (sekcja „Sfera”)\n'
                                  'albo własny wybór tego okienka. Wybrane cząsteczki i reflektory: sekcja „Sfera” (wybór).')
        self.sp_mk_las_gain = _spin(0.0, 20.0, 0.0, 0.01, 3); self.sp_mk_las_gain.setSpecialValueText('jak na dużej sferze')
        self.sp_mk_las_gain.setToolTip('Jasność wiązek laserów w okienku M (0 = jak na dużej sferze).')
        self.sp_mk_las_prog = _spin(0.0, 5.0, 0.0, 0.01, 3); self.sp_mk_las_prog.setSpecialValueText('jak na dużej sferze')
        self.sp_mk_las_prog.setToolTip('Próg jasności wiązki lasera: słabsze nie są rysowane (0 = jak na dużej sferze).')
        self.cb_mk_uw = NarrowCombo()
        self.cb_mk_uw.addItems(['linie uwagi: jak na dużej sferze', 'linie uwagi: brak', 'linie uwagi: na wybraną cząsteczkę', 'linie uwagi: wszystkie pary'])
        self.cb_mk_uw.setToolTip('Linie uwagi (gwiazda) w podkrokach uwagi w sferze okienka M.')
        self.ch_mk_refl = QCheckBox('reflektory na zewnętrznej sferze'); self.ch_mk_refl.setChecked(True)
        self.ch_mk_refl.setToolTip('Punkty 768 reflektorów bloku na zewnętrznej sferze okienka M.')
        self.sp_mk_las_eksp = _spin(0.05, 20.0, 1.0, 0.1, 2); self.sp_mk_las_eksp.setToolTip('Ekspozycja światła laserów w sferze okienka M.')
        self.ch_mk_sw = QCheckBox('wiązki jak światło: suma HDR, poświata, mapowanie tonalne'); self.ch_mk_sw.setChecked(True)
        self.ch_mk_sw.setToolTip('Promienie zdjęcia, pole równoległe, wiązki z koloidów i z soczewek sumują się bez nasycenia w buforze HDR;\n'
                                 'jasne miejsca dostają rozmytą poświatę (bloom), całość jest mapowana tonalnie (1 − e^−x), więc skupione światło\n'
                                 'jarzy się zamiast przepalać na biało, a słabe wiązki zostają widoczne.')
        self.sp_mk_sw_eksp = _spin(0.05, 20.0, 1.0, 0.1, 2); self.sp_mk_sw_eksp.setToolTip('Ekspozycja wiązek (mnożnik światła przed mapowaniem tonalnym).')
        self.sp_mk_sw_blum = _spin(0.0, 10.0, 1.0, 0.1, 2); self.sp_mk_sw_blum.setToolTip('Siła poświaty (bloom) jasnych miejsc; 0 = bez poświaty.')
        self.sp_mk_sw_szer = _spin(0.5, 8.0, 2.0, 0.25, 2); self.sp_mk_sw_szer.setToolTip('Szerokość poświaty (rozmycie w pikselach połowy rozdzielczości).')
        self.sp_mk_sw_prog = _spin(0.0, 2.0, 0.15, 0.05, 2); self.sp_mk_sw_prog.setToolTip('Od jakiej jasności miejsca dostają poświatę.')
        self.sp_mk_sw_grub = _spin(1.0, 6.0, 1.6, 0.2, 1); self.sp_mk_sw_grub.setToolTip('Grubość wiązek w trybie światła [px].')
        self.ch_mk_pole_las = QCheckBox('pole laserów formuje mikrosoczewki (kształt soczewki = siła laserów bloku MLP)'); self.ch_mk_pole_las.setChecked(True)
        self.ch_mk_pole_las.setToolTip('Każda mikrosoczewka w matrycy wydłuża się i obraca wzdłuż wypadkowej siły laserów bloku MLP bieżącego podkroku na swój\n'
                                       'koloid: F_j = Σ_c (z_jc − z̄_c) v_c (dokładny rozkład MLP, w rzucie sceny). Do soczewki wchodzą wiązki jej K najsilniejszych\n'
                                       'laserów: z kierunku −v_c (pchają wzdłuż v_c), kolor = kolor reflektora, długość i jasność = |z| (błękit = ujemna jasność).\n'
                                       'W podkroku uwagi pole pokazuje lasery bloku MLP, który świeci w tej chwili na sferze (przenikanie podkroków).')
        self.cb_mk_pole_dla = NarrowCombo(); self.cb_mk_pole_dla.addItems([f'wiązki formujące: {x}' for x in FILTRY[1:]]); self.cb_mk_pole_dla.setCurrentIndex(2)
        self.cb_mk_pole_dla.setToolTip('Do których soczewek rysować wiązki formujących laserów: narzędzie, wybrane cząsteczki albo wszystkie 600.')
        self.sp_mk_pole_k = _spin(1, 16, 4, 1); self.sp_mk_pole_k.setToolTip('Ile najsilniejszych laserów (|z − z̄|) formuje każdą soczewkę na rysunku.')
        self.sp_mk_pole_def = _spin(0.0, 3.0, 0.6, 0.05, 2); self.sp_mk_pole_def.setToolTip('Siła odkształcenia soczewki przez pole (0 = okrągłe).')
        self.sp_mk_pole_dl = _spin(0.1, 5.0, 1.0, 0.1, 2); self.sp_mk_pole_dl.setToolTip('Długość wiązek formujących laserów.')
        self.sp_mk_pole_jas = _spin(0.05, 10.0, 1.0, 0.05, 2); self.sp_mk_pole_jas.setToolTip('Jasność wiązek formujących laserów.')
        self.cb_mk_wyglad = NarrowCombo()
        self.cb_mk_wyglad.addItems(['wygląd szkła: szklane kulki koloidów',
                                    'wygląd szkła: kropla w ścinaniu (maź odkształcana polem laserów, światło skupiane kontrakcją uwagi)'])
        self.cb_mk_wyglad.setCurrentIndex(1)
        self.cb_mk_wyglad.setToolTip('Kropla w ścinaniu: wokół każdego koloidu chmura drobin w elipsoidzie, jak kropla w przepływie liniowym (Taylor 1934;\n'
                                     'model elipsoidalny Maffettone–Minale 1998). Przepływem jest pole MLP bloku w koloidzie: jakobian J = P·∂MLP/∂x·Pᵀ\n'
                                     '(dokładnie z wag). Elipsoida jest rozciągana wzdłuż osi rozciągania części symetrycznej S = (J + Jᵀ)/2 (λ > 0),\n'
                                     'ściskana wzdłuż osi ściskania (λ < 0) i obracana wirowością ω (część antysymetryczna); ruch w podkroku (uwaga albo MLP)\n'
                                     'ją wydłuża i zostawia ogon. Pomiar z 24.10.2026 (eksperyment schur_qr_jakobiany_mlp_20261024__10-03-40): 86 % energii S\n'
                                     'i 91 % energii ω pochodzi z części nienormalnej jakobianu (‖N‖/‖J‖ ≈ 0,93 wobec 0,71 dla macierzy losowej), więc to\n'
                                     'ŚCINANIE, nie wir ani soczewka; pole nie ma też zera w pobliżu koloidów (brak pułapki). S i ω to kompresje pełnego\n'
                                     'jakobianu do rzutu (dokładne), ale ok. 80 % odpowiedzi pola ucieka z rzutu: to cień odkształcenia, nie całe odkształcenie.\n'
                                     'Drobiny zlewają się w półprzezroczystą maź, która ugina światło wiązek na krawędziach i je skupia; skąd bierze się\n'
                                     'skupianie, ustala lista „skupianie mazi” (domyślnie kontrakcja uwagi). Szklane kulki: koloid jako kulka, bez pola.')
        self.sp_mk_mz_r = _spin(0.5, 10.0, 2.5, 0.25, 2); self.sp_mk_mz_r.setToolTip('Promień mazi wokół koloidu (jednostki sfery).')
        self.sp_mk_mz_def = _spin(0.0, 4.0, 0.8, 0.1, 2); self.sp_mk_mz_def.setToolTip('Odkształcenie mazi i soczewek polem (siodło: rozciąganie i ściskanie).')
        self.sp_mk_mz_wir = _spin(0.0, 10.0, 1.0, 0.1, 2); self.sp_mk_mz_wir.setToolTip('Tempo obrotu mazi i soczewek w wirze pola (0 = bez animacji).')
        self.sp_mk_mz_n = _spin(8, 160, 48, 8); self.sp_mk_mz_n.setToolTip('Drobin mazi na koloid (więcej = gładsza maź, wolniej).')
        self.sp_mk_mz_prog = _spin(0.05, 5.0, 0.6, 0.05, 2); self.sp_mk_mz_prog.setToolTip('Próg gęstości powierzchni mazi (mniej = maź grubsza i bardziej zlana).')
        self.sp_mk_mz_kr = _spin(0.05, 1.0, 0.7, 0.05, 2); self.sp_mk_mz_kr.setToolTip('Krycie mazi (półprzezroczystość).')
        self.sp_mk_mz_skup = _spin(0.0, 10.0, 1.5, 0.1, 2); self.sp_mk_mz_skup.setToolTip('Skupianie światła wiązek w mazi (0 = maź nie świeci światłem wiązek).')
        self.cb_mk_skup = NarrowCombo()
        self.cb_mk_skup.addItems(['skupianie mazi: kontrakcja uwagi względem chmury (na sferze: sąsiedztwo tokenu ściska się bardziej niż przeciętnie)',
                                  'skupianie mazi: kontrakcja uwagi bezwzględna (na sferze: świeci tylko, gdy sąsiedzi się zbliżają)',
                                  'skupianie mazi: ciepło głowy (odczyt, jak dotąd)'])
        self.cb_mk_skup.setCurrentIndex(0)
        self.cb_mk_skup.setToolTip('Skąd bierze się jasność drobin mazi i skupianie w niej światła wiązek.\n'
                                   'Kontrakcja uwagi: dla każdego tokenu 5 najbliższych sąsiadów w stanie przed półkrokiem uwagi bloku (stan 2b), na sferze\n'
                                   'chmury (stany środkowane po tokenach i znormowane per token: liczy się kierunek, nie norma); d_przed = średnia odległość\n'
                                   'do nich, d_po = do TYCH SAMYCH sąsiadów po półkroku uwagi (stan 2b + 1); κ = 1 − d_po/d_przed. „Względem chmury” dzieli\n'
                                   'd_po/d_przed przez średni stosunek wszystkich tokenów: świeci token, którego sąsiedztwo ściska się bardziej niż przeciętnie\n'
                                   '(historia 90f58d4f8b263d8b, k65640: w bloku 2 narzędzie κ ≈ 0,08 wobec 0,007 tła; w bloku 4 sąsiedztwa się rozchodzą,\n'
                                   'narzędzia najmniej). „Bezwzględna” świeci tylko przy κ > 0. Ciepło głowy to odczyt wyniku, nie przyczyna: tak maź\n'
                                   'świeciła dotąd. Po wyborze liczy się od nowa przy następnej zmianie bloku, pary checkpointów albo ułamka.')
        self.sp_mk_mz_refr = _spin(0.0, 5.0, 1.0, 0.1, 2); self.sp_mk_mz_refr.setToolTip('Ugięcie światła na krawędziach mazi (jak soczewka).')
        self.ch_mk_szklo = QCheckBox('koloidy jako szklane kulki (i drobiny super szkła jako szkło)'); self.ch_mk_szklo.setChecked(True)
        self.ch_mk_szklo.setToolTip('Każdy koloid rysowany jako szklana kulka: przezroczysty środek, jaśniejszy brzeg (odbicie Fresnela), połysk od światła\n'
                                    'z góry i jasna plamka skupionego światła po drugiej stronie (tym jaśniejsza, im większe ciepło). Zabarwienie =\n'
                                    'kolor łaty, rozjaśniony ciepłem. Drobiny super szkła (maź) dostają ten sam szklany wygląd.')
        self.cb_mk_szklo_dla = NarrowCombo(); self.cb_mk_szklo_dla.addItems([f'szklane kulki: {x}' for x in FILTRY[1:]]); self.cb_mk_szklo_dla.setCurrentIndex(2)
        self.cb_mk_szklo_dla.setToolTip('Które koloidy rysować jako szklane kulki: narzędzie, wybrane cząsteczki albo wszystkie 600.')
        self.sp_mk_szklo_r = _spin(4.0, 80.0, 22.0, 1.0, 1); self.sp_mk_szklo_r.setToolTip('Wielkość szklanej kulki koloidu na ekranie [px].')
        self.sp_mk_szklo_kr = _spin(0.1, 1.0, 0.9, 0.05, 2); self.sp_mk_szklo_kr.setToolTip('Krycie szklanych kulek (1 = pełne odbicia i połysk).')
        self.ch_mk_maz = QCheckBox('super szkło: drobne cząstki krążące wokół koloidów (płynna maź)'); self.ch_mk_maz.setChecked(False)
        self.ch_mk_maz.setToolTip('Wokół każdego koloidu krąży chmurka drobnych cząstek super szkła. W bieżącym podkroku są ciągnięte za ruchem\n'
                                  'koloidu (wypadkowa laserów MLP albo gwiazdy uwagi): rozciągają się wzdłuż niego i zostają z tyłu jak lepka maź.\n'
                                  'Animacja płynie w czasie rzeczywistym także przy zatrzymanym podkroku. To obraz, nie pomiar: wielkość chmurki\n'
                                  'nie ma odpowiednika w sieci; kierunek i długość rozciągnięcia biorą się z prawdziwego przesunięcia w podkroku.')
        self.cb_mk_maz_dla = NarrowCombo(); self.cb_mk_maz_dla.addItems([f'super szkło: {x}' for x in FILTRY[1:]]); self.cb_mk_maz_dla.setCurrentIndex(2)
        self.cb_mk_maz_dla.setToolTip('Wokół których koloidów rysować cząstki super szkła: narzędzie, wybrane cząsteczki albo wszystkie 600.')
        self.sp_mk_maz_n = _spin(4, 96, 24, 4); self.sp_mk_maz_n.setToolTip('Cząstek super szkła na jeden koloid (600 koloidów × n punktów).')
        self.sp_mk_maz_r = _spin(0.2, 10.0, 1.6, 0.2, 2); self.sp_mk_maz_r.setToolTip('Promień chmurki cząstek wokół koloidu (jednostki sfery).')
        self.sp_mk_maz_lep = _spin(0.0, 5.0, 1.0, 0.1, 2)
        self.sp_mk_maz_lep.setToolTip('Lepkość mazi: jak mocno cząstki rozciągają się wzdłuż ruchu koloidu w podkroku i zostają za nim (0 = sztywna kula).')
        self.sp_mk_maz_tempo = _spin(0.0, 10.0, 1.0, 0.1, 2); self.sp_mk_maz_tempo.setToolTip('Tempo krążenia cząstek wokół koloidu (0 = stoją).')
        self.sp_mk_maz_jas = _spin(0.05, 5.0, 0.6, 0.05, 2); self.sp_mk_maz_jas.setToolTip('Jasność cząstek super szkła (sumują się addytywnie).')
        self.sp_mk_maz_pkt = _spin(1.0, 12.0, 3.0, 0.5, 1); self.sp_mk_maz_pkt.setToolTip('Wielkość punktu cząstki super szkła na ekranie [px].')
        self.ch_mk_socz = QCheckBox('matryca mikrosoczewek nad sferą: jasność = transmisja (ciepło)'); self.ch_mk_socz.setChecked(True)
        self.ch_mk_socz.setToolTip('600 kółek w siatce 30 × 20 nad sferą, jedna soczewka na łatę zdjęcia. Jasność = ile światła lampy\n'
                                   'przepuszcza (sigmoid ciepła z głowy). Żółty obrys = wybrane cząsteczki.')
        self.ch_mk_lampa = QCheckBox('lampa pod matrycą i wiązki przez soczewki na ekran'); self.ch_mk_lampa.setChecked(True)
        self.ch_mk_lampa.setToolTip('Jednostajna lampa świeci przez matrycę; każda soczewka przepuszcza wiązkę tylko na własną komórkę ekranu\n'
                                    '(głowa działa na każdy token osobno, bez mieszania). Jasność wiązki = ciepło.')
        self.sp_mk_jas_lampy = _spin(0.1, 10.0, 1.0, 0.1, 2); self.sp_mk_jas_lampy.setToolTip('Jasność wiązek lampy (mnożnik ciepła).')
        self.sp_mk_prog = _spin(0.0, 1.0, 0.02, 0.01, 3); self.sp_mk_prog.setToolTip('Wiązki lampy słabsze niż ten próg ciepła nie są rysowane.')
        self.cb_mk_ekran = NarrowCombo()
        self.cb_mk_ekran.addItems(['ekran: cień soczewek (ciepło = pewność komórki)', 'ekran: różnica cień − kalka (czerwony za jasno, niebieski za ciemno)',
                                   'ekran: kalka (cel ciepła z etykiety, Gauss wokół środka narzędzia)'])
        self.cb_mk_ekran.setToolTip('Co pokazuje ekran na górze: cień (mapa ciepła 20 × 30 z głowy), różnicę między cieniem a kalką\n'
                                    '(to prowadzi uczenie: gdzie soczewki mają się jeszcze zmienić) albo samą kalkę z etykiety.')
        self.cb_mk_sonda = NarrowCombo()
        self.cb_mk_sonda.addItems(['głowa czyta stan po bieżącym podkroku (sonda; sieć tak nie robi)', 'głowa czyta tylko stan końcowy (jak sieć)'])
        self.cb_mk_sonda.setToolTip('Sieć czyta głową wyłącznie stan po 12. podkroku. Sonda przykłada tę samą głowę do stanu po bieżącym\n'
                                    'podkroku, żeby zobaczyć, jak cień powstaje w głąb sieci (zabieg diagnostyczny, nie działanie sieci).')
        self.sp_mk_gamma = _spin(0.2, 3.0, 1.0, 0.1, 2); self.sp_mk_gamma.setToolTip('Wykładnik jasności ciepła na ekranie, soczewkach i wiązkach (0,5 rozjaśnia słabe).')
        self.ch_mk_ramki = QCheckBox('ramki na ekranie: błękitna z głowy (komórka szczytu ciepła), biała GT'); self.ch_mk_ramki.setChecked(True)
        self.ch_mk_ramki.setToolTip('Ramka z głowy: prostokąt plamki najjaśniejszej soczewki (l, t, r, b z jej komórki). Biała: ramka z etykiety.')
        f.addRow(self.ch_mk_obraz); f.addRow(self.ch_mk_plyty); f.addRow('  krycie płyt', self.sp_mk_krycie_plyt)
        f.addRow(self.cb_mk_prom); f.addRow(self.cb_mk_rown); f.addRow('  promieni w pęku', self.sp_mk_pek_n); f.addRow('  apertura koloidu', self.sp_mk_pek_ap); f.addRow(self.ch_mk_zal); f.addRow('  współczynnik załamania', self.sp_mk_n); f.addRow('  wzmocnienie wyjścia', self.sp_mk_wzm);
        f.addRow(self.ch_mk_pole); f.addRow('  jasność pola', self.sp_mk_jas_pola); f.addRow('  wielkość chmury', self.sp_mk_skala); f.addRow(self.cb_mk_linki); f.addRow('  jasność promieni', self.sp_mk_jas_prom)
        f.addRow('  krycie promieni', self.sp_mk_krycie_prom); f.addRow(self.ch_mk_jas_zdj); f.addRow(self.ch_mk_sfera)
        f.addRow(self.cb_mk_las); f.addRow('  jasność laserów', self.sp_mk_las_gain); f.addRow('  próg laserów', self.sp_mk_las_prog)
        f.addRow('  ekspozycja laserów', self.sp_mk_las_eksp); f.addRow(self.cb_mk_uw); f.addRow(self.ch_mk_refl)
        f.addRow(self.ch_mk_sw); f.addRow('  ekspozycja wiązek', self.sp_mk_sw_eksp); f.addRow('  poświata', self.sp_mk_sw_blum)
        f.addRow('  szerokość poświaty', self.sp_mk_sw_szer); f.addRow('  próg poświaty', self.sp_mk_sw_prog); f.addRow('  grubość wiązek', self.sp_mk_sw_grub)
        f.addRow(self.ch_mk_pole_las); f.addRow(self.cb_mk_pole_dla); f.addRow('  laserów na soczewkę', self.sp_mk_pole_k)
        f.addRow('  odkształcenie soczewki', self.sp_mk_pole_def); f.addRow('  długość wiązek', self.sp_mk_pole_dl); f.addRow('  jasność wiązek', self.sp_mk_pole_jas)
        f.addRow(self.ch_mk_szklo); f.addRow(self.cb_mk_wyglad); f.addRow('  promień mazi', self.sp_mk_mz_r)
        f.addRow('  odkształcenie polem', self.sp_mk_mz_def); f.addRow('  tempo wiru', self.sp_mk_mz_wir)
        f.addRow('  drobin na koloid', self.sp_mk_mz_n); f.addRow('  próg powierzchni', self.sp_mk_mz_prog); f.addRow('  krycie mazi', self.sp_mk_mz_kr)
        f.addRow(self.cb_mk_skup); f.addRow('  skupianie światła', self.sp_mk_mz_skup); f.addRow('  ugięcie na krawędzi', self.sp_mk_mz_refr)
        f.addRow(self.cb_mk_szklo_dla); f.addRow('  wielkość kulki', self.sp_mk_szklo_r)
        f.addRow('  krycie kulek', self.sp_mk_szklo_kr)
        f.addRow(self.ch_mk_maz); f.addRow(self.cb_mk_maz_dla); f.addRow('  cząstek na koloid', self.sp_mk_maz_n)
        f.addRow('  promień chmurki', self.sp_mk_maz_r); f.addRow('  lepkość mazi', self.sp_mk_maz_lep)
        f.addRow('  tempo krążenia', self.sp_mk_maz_tempo); f.addRow('  jasność', self.sp_mk_maz_jas); f.addRow('  wielkość punktu', self.sp_mk_maz_pkt)
        f.addRow(self.ch_mk_socz); f.addRow(self.ch_mk_lampa); f.addRow('  jasność lampy', self.sp_mk_jas_lampy); f.addRow('  próg wiązek', self.sp_mk_prog)
        f.addRow(self.cb_mk_ekran); f.addRow(self.cb_mk_sonda); f.addRow('  wykładnik jasności', self.sp_mk_gamma); f.addRow(self.ch_mk_ramki)
        for w in (self.ch_mk_pole_las, self.ch_mk_zal, self.ch_mk_refl, self.ch_mk_sw, self.ch_mk_szklo, self.ch_mk_pole, self.ch_mk_obraz, self.ch_mk_plyty, self.ch_mk_sfera, self.ch_mk_maz, self.ch_mk_socz, self.ch_mk_lampa, self.ch_mk_ramki, self.ch_mk_jas_zdj):
            w.toggled.connect(self._opcje_mikro)
        for w in (self.cb_mk_wyglad, self.cb_mk_skup,self.cb_mk_pole_dla, self.cb_mk_las, self.cb_mk_uw, self.cb_mk_prom, self.cb_mk_szklo_dla, self.cb_mk_maz_dla, self.cb_mk_rown, self.cb_mk_linki, self.cb_mk_ekran, self.cb_mk_sonda):
            w.currentIndexChanged.connect(self._opcje_mikro)
        for w in (self.sp_mk_mz_r, self.sp_mk_mz_def, self.sp_mk_mz_wir, self.sp_mk_mz_n, self.sp_mk_mz_prog,
                  self.sp_mk_mz_kr, self.sp_mk_mz_skup, self.sp_mk_mz_refr, self.sp_mk_pole_k, self.sp_mk_pole_def, self.sp_mk_pole_dl, self.sp_mk_pole_jas, self.sp_mk_n, self.sp_mk_las_gain, self.sp_mk_las_prog, self.sp_mk_las_eksp, self.sp_mk_sw_eksp, self.sp_mk_sw_blum,
                  self.sp_mk_sw_szer, self.sp_mk_sw_prog, self.sp_mk_sw_grub, self.sp_mk_skala, self.sp_mk_szklo_r, self.sp_mk_szklo_kr, self.sp_mk_wzm, self.sp_mk_jas_pola, self.sp_mk_pek_n, self.sp_mk_pek_ap, self.sp_mk_maz_n, self.sp_mk_maz_r, self.sp_mk_maz_lep, self.sp_mk_maz_tempo, self.sp_mk_maz_jas, self.sp_mk_maz_pkt,
                  self.sp_mk_krycie_plyt, self.sp_mk_jas_prom, self.sp_mk_krycie_prom, self.sp_mk_jas_lampy, self.sp_mk_prog, self.sp_mk_gamma):
            w.valueChanged.connect(self._opcje_mikro)
        self._sek['mikro'] = g

    def _buduj_okienko_mikro(self):
        """Okienko M: widok 3D ze sceną mikrosoczewek i drugą sferą reflektorów (wspólne opcje z dużą sferą)."""
        self.vm = GLView(); self.vm.setBackgroundColor(TLO); self.vm.setCameraPosition(distance=330, elevation=30, azimuth=-55)
        self.emk = MicrolensScene(); self.emk.R = float(hist.R_SFERY); self.vm.addItem(self.emk)
        self.em = ReflectorSphere(hist.R_SFERY); self._mk_opcje_sfery = {}
        self.em.opcje = ChainMap(self._mk_opcje_sfery, self.es.opcje)      # wspólne z dużą sferą, poza skalą chmury
        self.vm.addItem(self.em)
        self._dok('m', 'M. Mikrosoczewki: zdjęcie naświetla, lasery i gwiazda formują, lampa rzuca cień na ekran',
                  self._z_napisem(self.vm, 'lbm'))
        self.doki['m'].visibilityChanged.connect(self._mk_widoczne)
        self._mk_klucz = None; self._mk_zdjecie = None; self._mk_plyty = None
        from PyQt6.QtCore import QElapsedTimer, QTimer
        self._mk_zegar = QElapsedTimer(); self._mk_zegar.start()
        self._mk_anim = QTimer(self); self._mk_anim.setInterval(33); self._mk_anim.timeout.connect(self._mk_klatka)

    # ------------------------------------------------------------------ zasilanie: przejście bieżącego checkpointu (jak okienko P)
    def _mk_widoczne(self, on):
        """Okienko M odsłonięte: po chwili (gdy widać już jego obszar) bierze gotowe przejście albo zleca policzenie."""
        if not on or self.h is None:
            return
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._mk_odsloniete)

    def _mk_odsloniete(self):
        h = self.h
        if h is None or not self.widac('m'):
            return
        self._zlec_podkroki()
        self._mk_odswiez()

    def _mk_przejscie(self, d):
        """Nowe przejście w pamięci (bieżące albo policzone z wyprzedzeniem): okienko M odświeża mieszankę."""
        self._mk_odswiez()

    def _mk_czas(self, t):
        """Oś kroków uczenia: okienko M miesza przejścia checkpointów a = ⌊t⌋ i b = a + 1 (ułamek t − a), więc koloidy, ciepło,
        lasery i uwaga zmieniają się płynnie między checkpointami; brakujące przejścia liczą się w tle z wyprzedzeniem."""
        h = self.h
        if h is None or not hasattr(self, 'emk') or not self.widac('m'):
            return
        a = int(np.clip(np.floor(t), 0, h.K - 1)); b = min(a + 1, h.K - 1)
        for j in (a, b, min(b + 1, h.K - 1)):
            self._z_wyprzedzeniem(j)
        self._mk_odswiez()

    def _mk_odswiez(self):
        """Okienko M w chwili osi uczenia self.t i podkroku self.s_pod: przejście a, b albo ich mieszanka (co jest w pamięci)."""
        h = self.h
        if h is None or not hasattr(self, 'emk') or not self.widac('m'):
            return
        pam = self._pamiec_pod()
        t = float(np.clip(self.t, 0, h.K - 1)); a = int(np.floor(t)); b = min(a + 1, h.K - 1); f = t - a
        da, db = pam.get(int(h.kroki[a])), pam.get(int(h.kroki[b]))
        if da is None and db is None:
            return
        if da is None:
            da, db, f = db, None, 0.0                                  # tylko b w pamięci: na razie b
        elif db is None:
            f = 0.0                                                    # b liczy się w tle: na razie a
        if self._mk_zdjecie != id(h):
            self.emk.ustaw_zdjecie(h.rgb, h.gt, h.narz); self._mk_zdjecie = id(h)
        klucz = (id(da), id(db), round(f, 4), id(h))
        if self._mk_klucz == klucz:
            return
        blizszy = db if (db is not None and f >= 0.5) else da
        m = dict(da)                                                   # scena: tylko głowa mieszana (mała); sfera: para na GPU
        if db is not None and f > 1e-4:
            m['glowa'] = ((1 - f) * np.asarray(da['glowa'], np.float32) + f * np.asarray(db['glowa'], np.float32)).astype(np.float32)
        m.pop('lodyga', None)
        if getattr(self, '_mk_plyty', None) != id(blizszy):
            m['lodyga'] = blizszy.get('lodyga')                         # płyty łodygi z bliższego checkpointu (tylko przy zmianie)
        self._mk_plyty = id(blizszy)
        self._mk_klucz = klucz
        self._mk_para = (da, db, f)
        self.em.ustaw_pare_podkrokow(da, db, f, h.narz)
        self.emk.ustaw_przejscie(m, self.em._P_surowe[..., :3])
        self._opcje_mikro()
        self._mk_podkrok(self.s_pod)

    def _mk_pole(self, s):
        """Pole laserów bloku MLP rysowanego w chwili s na koloidy (dokładny rozkład MLP): wypadkowa siła i K najsilniejszych laserów
        tokenu; przy mieszaniu checkpointów natężenia i kierunki z obu (ułamek f). Liczone tylko przy zmianie bloku, pary lub f."""
        para = getattr(self, '_mk_para', None)
        if para is None or not self.ch_mk_pole_las.isChecked():
            return
        from laser_app.gpu.base import przenikanie, reflektory_przejscia
        da, db, f = para; fpl = self._plynnie(); _, bm, _ = przenikanie(s, fpl)
        K = int(self.sp_mk_pole_k.value())
        skup = int(self.emk.opcje.get('skup_z', 0))
        klucz = (id(da), id(db), round(f, 3), bm, K, skup)
        if klucz == getattr(self, '_mk_pole_klucz', None) or 'pv_n' not in da:
            return
        self._mk_pole_klucz = klucz
        z = np.asarray(da['z'][bm], np.float32); pv = np.asarray(da['pv_n'][bm], np.float32)
        if db is not None and f > 1e-4 and 'pv_n' in db:
            z = (1 - f) * z + f * np.asarray(db['z'][bm], np.float32); pv = (1 - f) * pv + f * np.asarray(db['pv_n'][bm], np.float32)
        zc = z - z.mean(0, keepdims=True)
        F = zc @ pv                                                    # (600, 3): Σ_c (z_jc − z̄_c) v_c, w rzucie sceny
        K = min(K, zc.shape[1]); idx = np.argpartition(-np.abs(zc), K - 1, axis=1)[:, :K]
        zt = np.take_along_axis(zc, idx, axis=1)
        _, kol, _ = reflektory_przejscia(da, s, fpl)
        kappa = self._mk_kontrakcja(da, db, f, bm, skup)              # skupianie mazi: kontrakcja w półkroku uwagi bloku bm (2 = ciepło: brak)
        self.emk.ustaw_pole_laserow(F, idx, zt, pv, kol, self._mk_pole_jakobianu(da, db, f, bm, F, kappa))

    @staticmethod
    def kontrakcja_uwagi(xa, xb, k=5, sfera=True, wzgledem_chmury=True):
        """Kontrakcja sąsiedztwa tokenu w półkroku uwagi: xa, xb (N, D) pełne stany przed i po półkroku (xn[2b], xn[2b+1]). sfera:
        stany środkowane po tokenach i znormowane per token (liczy się kierunek, jak po LayerNorm bez afinicznych; jednorodne
        skalowanie chmury daje κ = 0). Dla każdego tokenu k najbliższych sąsiadów w xa (bez niego samego); d_przed = średnia odległość
        do nich w xa, d_po = do TYCH SAMYCH sąsiadów w xb; κ = 1 − d_po/d_przed (> 0 zbliżanie, < 0 oddalanie). wzgledem_chmury:
        d_po/d_przed dzielone przez średni stosunek wszystkich tokenów (κ > 0 = sąsiedztwo ściska się bardziej niż przeciętnie), bo
        w bloku 4 wszystkie sąsiedztwa się rozchodzą i bez tego maź byłaby ciemna. Zwraca float64 (N,)."""
        xa = np.asarray(xa, np.float64); xb = np.asarray(xb, np.float64); N = len(xa); k = int(min(k, N - 1))
        if sfera:
            xa = xa - xa.mean(1, keepdims=True); xa = xa / np.maximum(np.linalg.norm(xa, axis=1, keepdims=True), 1e-12)
            xb = xb - xb.mean(1, keepdims=True); xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
        sq = (xa * xa).sum(1); D2 = sq[:, None] + sq[None, :] - 2.0 * (xa @ xa.T)   # tylko do wyboru sąsiadów
        np.fill_diagonal(D2, np.inf)
        idx = np.argpartition(D2, k - 1, axis=1)[:, :k]                              # (N, k)
        d_przed = np.linalg.norm(xa[:, None, :] - xa[idx], axis=2).mean(1)          # odległości liczone wprost (dokładnie)
        d_po = np.linalg.norm(xb[:, None, :] - xb[idx], axis=2).mean(1)
        r = d_po / np.maximum(d_przed, 1e-12)
        if wzgledem_chmury:
            r = r / max(float(d_po.mean() / max(float(d_przed.mean()), 1e-12)), 1e-12)
        return 1.0 - r

    def _mk_kontrakcja(self, da, db, f, bm, tryb=0):
        """κ bloku bm (kontrakcja_uwagi stanów xn[2·bm] → xn[2·bm + 1]; tryb 0 = względem chmury, 1 = bezwzględna, 2 = ciepło głowy:
        None); przy mieszaniu checkpointów κ obu mieszane ułamkiem f. Pamięć na (przejście, blok, tryb), bo f zmienia się płynnie.
        None także, gdy przejście nie ma pełnych stanów 'xn' (stare zapisy)."""
        if int(tryb) >= 2:
            return None
        pam = getattr(self, '_mk_kap_pam', None)
        if pam is None:
            pam = self._mk_kap_pam = {}

        def jeden(d):
            if d is None or 'xn' not in d:
                return None
            kl = (id(d), d.get('krok'), int(bm), int(tryb))             # krok: ten sam id po zwolnieniu przejścia ≠ ten sam checkpoint
            if kl not in pam:
                if len(pam) >= 12:
                    pam.clear()
                pam[kl] = self.kontrakcja_uwagi(d['xn'][2 * bm], d['xn'][2 * bm + 1], wzgledem_chmury=(int(tryb) == 0))
            return pam[kl]
        ka = jeden(da)
        if ka is None:
            return None
        kb = jeden(db) if (db is not None and f > 1e-4) else None
        return ka if kb is None else (1 - f) * ka + f * kb

    @staticmethod
    def pole_z_jakobianu(J, F, kappa=None):
        """Pole wokół koloidów z jakobianu J (N, 3, 3) pola MLP w rzucie '_n' i skoku F (N, 3): S = (J + Jᵀ)/2 (osie rozciągania
        λ > 0 i ściskania λ < 0 kropli w ścinaniu) i ω (wirowość z części antysymetrycznej (J − Jᵀ)/2), oba względem 95. percentyla
        max|λ(S)|, |F| względem 99. percentyla i kontrakcja uwagi κ (N,) względem 95. percentyla |κ|, obcięta do [−1.5, 1.5]
        (bez κ: 0). Zwraca (N, 3, 4) do bufora JK: [S00 S01 S02 ω0], [S11 S12 S22 ω1], [ω2 |F| κ 0].
        S i ω są kompresjami P·S₁₉₂·Pᵀ, P·Ω₁₉₂·Pᵀ pełnego jakobianu (dokładne, wartości własne S przeplatają się z pełnymi).
        Ogniska u* = −J⁻¹F tu nie ma: pomiar 2026-10-24 (README, okienko M) pokazał, że to zero przybliżenia w rzucie leżące
        dziesiątki komórek od koloidu, bez związku z zerem pełnego pola."""
        J = np.asarray(J, np.float64); F = np.asarray(F, np.float64); N = len(J)
        S = 0.5 * (J + np.swapaxes(J, 1, 2)); A = 0.5 * (J - np.swapaxes(J, 1, 2))
        om = np.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]], 1)
        sc = float(np.percentile(np.abs(np.linalg.eigvalsh(S)).max(1), 95)) or 1.0
        fm = float(np.percentile(np.linalg.norm(F, axis=1), 99)) or 1.0
        Sn = S / sc; omn = om / sc
        JK = np.zeros((N, 3, 4), np.float32)
        JK[:, 0] = np.stack([Sn[:, 0, 0], Sn[:, 0, 1], Sn[:, 0, 2], omn[:, 0]], 1)
        JK[:, 1] = np.stack([Sn[:, 1, 1], Sn[:, 1, 2], Sn[:, 2, 2], omn[:, 1]], 1)
        JK[:, 2, 0] = omn[:, 2]; JK[:, 2, 1] = np.linalg.norm(F, axis=1) / fm
        if kappa is not None:
            kp = np.asarray(kappa, np.float64); km = float(np.percentile(np.abs(kp), 95)) or 1.0
            JK[:, 2, 2] = np.clip(kp / km, -1.5, 1.5)
        return JK

    def _mk_pole_jakobianu(self, da, db, f, bm, F, kappa=None):
        """Bufor JK dla bloku bm (mieszanie checkpointów jak reszta), z kontrakcją uwagi κ (albo None); None, gdy przejście nie ma
        jakobianów."""
        if 'jak_n' not in da:
            return None
        J = np.asarray(da['jak_n'][bm], np.float64)
        if db is not None and f > 1e-4 and 'jak_n' in db:
            J = (1 - f) * J + f * np.asarray(db['jak_n'][bm], np.float64)
        return self.pole_z_jakobianu(J, F, kappa)

    def _mk_podkrok(self, s):
        """Chwila podkroku 0..12 (ułamkowa) w okienku M: sfera i scena; napis pod widokiem."""
        if not hasattr(self, 'emk') or not self.widac('m'):
            return
        self.em.ustaw_podkrok(s); self.emk.ustaw_czas(s); self._mk_pole(s)
        self.lbm.setText(self.emk.opis(s, hist.iou))

    def _mk_animuj(self):
        """Czy okienko M potrzebuje animacji: drobiny super szkła albo wir pola (maź, soczewki)."""
        wir = self.sp_mk_mz_wir.value() > 0 and ((self.ch_mk_szklo.isChecked() and self.cb_mk_wyglad.currentIndex() == 1)
                                                 or (self.ch_mk_pole_las.isChecked() and self.ch_mk_socz.isChecked()))
        return self.ch_mk_maz.isChecked() or wir

    def _mk_klatka(self):
        """Animacja (~30 kl./s): super szkło, wir mazi i soczewek; tylko gdy okienko M widać."""
        if self._mk_animuj() and self.widac('m') and self.emk.G is not None:
            self.emk.czas = self._mk_zegar.elapsed() / 1000.0; self.emk.update()

    def _opcje_mikro(self, *_):
        if not hasattr(self, 'emk'):
            return
        o = self.emk.opcje
        o.update(obraz=self.ch_mk_obraz.isChecked(), plyty=self.ch_mk_plyty.isChecked(), krycie_plyt=self.sp_mk_krycie_plyt.value(),
                 promienie=self.cb_mk_prom.currentIndex(), rownolegle=self.cb_mk_rown.currentIndex(), pek_n=self.sp_mk_pek_n.value(), pek_ap=self.sp_mk_pek_ap.value(), wzm_wyj=self.sp_mk_wzm.value(),
                 pole=self.ch_mk_pole.isChecked(), jas_pola=self.sp_mk_jas_pola.value(), maz=self.ch_mk_maz.isChecked(), wyglad=self.cb_mk_wyglad.currentIndex(), mz_r=self.sp_mk_mz_r.value(),
                 mz_def=self.sp_mk_mz_def.value(), mz_wir=self.sp_mk_mz_wir.value(), mz_n=self.sp_mk_mz_n.value(),
                 mz_prog=self.sp_mk_mz_prog.value(), mz_kr=self.sp_mk_mz_kr.value(), mz_skup=self.sp_mk_mz_skup.value(), mz_refr=self.sp_mk_mz_refr.value(),
                 skup_z=self.cb_mk_skup.currentIndex(), jas_zdj=self.ch_mk_jas_zdj.isChecked(),
                 pole_las=self.ch_mk_pole_las.isChecked(), pole_dla=self.cb_mk_pole_dla.currentIndex() + 1,
                 pole_def=self.sp_mk_pole_def.value(), pole_dl=self.sp_mk_pole_dl.value(), pole_jas=self.sp_mk_pole_jas.value(), zalamanie=self.ch_mk_zal.isChecked(), n_szkla=self.sp_mk_n.value(), swiatlo=self.ch_mk_sw.isChecked(), ekspozycja=self.sp_mk_sw_eksp.value(),
                 blum=self.sp_mk_sw_blum.value(), blum_szer=self.sp_mk_sw_szer.value(), blum_prog=self.sp_mk_sw_prog.value(),
                 grub_sw=self.sp_mk_sw_grub.value(), szklo=self.ch_mk_szklo.isChecked(), szklo_dla=self.cb_mk_szklo_dla.currentIndex() + 1,
                 szklo_r=self.sp_mk_szklo_r.value(), szklo_krycie=self.sp_mk_szklo_kr.value(), maz_filtr=self.cb_mk_maz_dla.currentIndex() + 1,
                 maz_n=self.sp_mk_maz_n.value(), maz_r=self.sp_mk_maz_r.value(), maz_lep=self.sp_mk_maz_lep.value(),
                 maz_tempo=self.sp_mk_maz_tempo.value(), maz_jas=self.sp_mk_maz_jas.value(), maz_pkt=self.sp_mk_maz_pkt.value(), linki=self.cb_mk_linki.currentIndex(), jas_prom=self.sp_mk_jas_prom.value(),
                 krycie_prom=self.sp_mk_krycie_prom.value(), soczewki=self.ch_mk_socz.isChecked(), lampa=self.ch_mk_lampa.isChecked(),
                 jas_lampy=self.sp_mk_jas_lampy.value(), prog=self.sp_mk_prog.value(), ekran=self.cb_mk_ekran.currentIndex(),
                 sonda=self.cb_mk_sonda.currentIndex(), gamma=self.sp_mk_gamma.value(), ramki=self.ch_mk_ramki.isChecked(),
                 skala=float(self.es.opcje.get('skala_chmury', 1.0)) * self.sp_mk_skala.value())
        so = self._mk_opcje_sfery; so.clear(); so['skala_chmury'] = o['skala']      # własne ustawienia sfery okienka M
        il = self.cb_mk_las.currentIndex()
        if il:
            so['wiazki'] = (0, 1, 2, 3)[il - 1]
        if self.sp_mk_las_gain.value() > 0:
            so['gain'] = self.sp_mk_las_gain.value()
        if self.sp_mk_las_prog.value() > 0:
            so['prog'] = self.sp_mk_las_prog.value()
        iu = self.cb_mk_uw.currentIndex()
        if iu:
            so['sily'] = iu - 1
        so['reflektory'] = self.ch_mk_refl.isChecked(); so['ekspozycja'] = self.sp_mk_las_eksp.value()
        self.em.setVisible(self.ch_mk_sfera.isChecked())
        anim = self._mk_animuj()
        if anim and not self._mk_anim.isActive():
            self._mk_anim.start()
        elif not anim and self._mk_anim.isActive():
            self._mk_anim.stop()
        if self.h is not None:
            tok = self._wybrane_tok()
            self.emk.ustaw_wybor(tok); self.em.ustaw_wybor(self._wybrane_refl(), tok)
            ks = self.cb_kol_sciezki.currentIndex()
            if ks in (1, 2, 3, 5) and getattr(self, '_mk_ks', None) != (ks, id(self.h)):
                self._mk_ks = (ks, id(self.h)); self.em.ustaw_kolory_sciezek(self._kolory_sfery(ks - 1 if ks < 4 else 3))
        self.emk.update(); self.em.update()
        if self.widac('m') and self.emk.G is not None:
            self._mk_pole(self.s_pod)
            self.lbm.setText(self.emk.opis(self.s_pod, hist.iou))
