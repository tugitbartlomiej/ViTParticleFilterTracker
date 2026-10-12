"""Animacje „ruch tokenów jak w szkle”: nagrania narzędziami aplikacji (ZakladkaRuchu.nagraj_film) z osi czasu
układu szkła (względem klatki, 60 000 cząsteczek biegu 35 000) i z symulowanego szkła koloidalnego (dynamika browna),
plus porównanie z oryginalnym układem (względem populacji). Klipy pomijają osie, których jeszcze nie ma na dysku.
Uruchomienie: F:/Python/3.13/python.exe -u animacje_szkla.py [G1,G2,...]
"""
import glob
import json
import os
import sys
import time

import numpy as np

APP = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/app'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'animacje')
OSIE = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui'
E35K = os.path.join(OSIE, '20261009T162654_natural_78cd84bd6edd')          # pomiar naturalny: 22 końce epok (względem populacji)
SZKLO = os.path.join(OSIE, '20261010T_natural_szklo_klatka_35k')           # układ względem klatki, co 2 kroki do 2188
def _wybierz(wzorzec, bez=None):
    """Osie szkła BD (lokalne albo z Edena); gdy jest wybór gęstości z MODEL_EDEN_szybko.json, bierze najbliższą tokenom."""
    kand = sorted(p for p in glob.glob(os.path.join(OSIE, wzorzec)) if not bez or bez not in os.path.basename(p))
    try:
        naj = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODEL_EDEN_szybko.json'), encoding='utf-8'))['najblizszy']
        phi = naj.split('phi_')[-1]
        pref = [p for p in kand if p.endswith(f'phi_{phi}')]
        if pref:
            return pref
    except Exception:
        pass
    return kand


BD = _wybierz('20261010T_szklo_eden_szklo_BD_eden_stala_phi_*') or _wybierz('20261010T_szklo_BD_phi_*', bez='stygnace')
BD_STYG = _wybierz('20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_*') or _wybierz('20261010T_szklo_BD_stygnace_*')
OBRAZ = 0                                        # zdjęcie 0 osi szkła: TR_0001_S1_P03_000838 (cystotome)
FPS = 30
TYLKO = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None
os.makedirs(OUT, exist_ok=True)
os.chdir(APP); sys.path.insert(0, APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
q = QApplication(sys.argv)
from zakladka_ruchu import ZakladkaRuchu  # noqa: E402
from PIL import Image  # noqa: E402

z = ZakladkaRuchu(None); z.resize(1500, 1150); z.show(); q.processEvents()
z.spin_vram.setValue(4); z.btn_panel.setChecked(True)


def czekaj(warunek, limit):
    t = time.perf_counter()
    while not warunek() and time.perf_counter() - t < limit:
        q.processEvents(); time.sleep(.01)
    return warunek()


def wczytaj(sciezka, obszary=True):
    z.edit_timeline.setText(sciezka)
    cel = os.path.normcase(os.path.abspath(sciezka))
    czekaj(lambda: getattr(z, 'watek', None) is None, 180)        # poprzednie zadanie tła (obszary, wczytanie) musi się skończyć
    z._load_timeline()
    assert czekaj(lambda: z.timeline is not None and os.path.normcase(os.path.abspath(str(z.timeline.root))) == cel, 300), \
        f'nie wczytano {sciezka}: status „{z.status.text()[:200]}”'
    z.cloud.wczytaj_teraz(z.timeline.steps); assert czekaj(lambda: z.cloud.na_karcie() == z.cloud.T, 900)
    if obszary:
        czekaj(lambda: z._obszary_klasy is not None, 180)
    print('wczytano', os.path.basename(sciezka), 'stanów', z.cloud.T, 'cząsteczek', z.cloud.N, flush=True)


def ustaw(uklad=0, kolor=11, scope=0, obraz=OBRAZ, co=1, krycie=.3, rozmiar=1.5, slad=0, okno=5, cecha=3, bez_poz=False,
          zdj=0, att=False, obszar=0, kamera=(3.6, 22, 40), moc=1.3, ruch=False, kadr=0):
    z.combo_uklad.setCurrentIndex(z.combo_uklad.findData(uklad))
    z.combo_color.setCurrentIndex(z.combo_color.findData(kolor))
    z.combo_scope.setCurrentIndex(scope); z.spin_image.setValue(obraz)
    z.combo_co.setCurrentIndex(z.combo_co.findData(co))
    z.spin_opacity.setValue(krycie); z.spin_size.setValue(rozmiar); z.spin_trail.setValue(slad); z.spin_moc.setValue(moc)
    z.spin_fz_okno.setValue(okno); z.combo_cecha.setCurrentIndex(cecha); z.chk_bez_poz.setChecked(bez_poz)
    z.combo_zdj.setCurrentIndex(zdj); z.combo_fz_obszar.setCurrentIndex(obszar)
    z.combo_wyg.setCurrentIndex(0); z.chk_smooth.setChecked(True); z.chk_petla.setChecked(False)
    z.chk_movement.setChecked(ruch); z.combo_kadr.setCurrentIndex(kadr)
    z.chk_att.setChecked(att)
    z.view.setCameraPosition(distance=kamera[0], elevation=kamera[1], azimuth=kamera[2])
    z._fz_wynik = None; z._fz_klucz = None
    z._seek_index(0); czekaj(lambda: False, .8)


def nearest(krok):
    st = z.timeline.steps
    return int(np.argmin(np.abs(np.asarray(st) - krok)))


def nagraj(nazwa, od, do, sekundy, napis, obrot=0.0, arkusz=6):
    if TYLKO is not None and nazwa.split('_')[0] not in TYLKO:
        print(f'{nazwa}: pominięte', flush=True); return None
    ramki = []
    n = int(sekundy * FPS)
    co = max(1, n // (arkusz - 1))

    def kl(a, krok):
        if obrot:
            z.view.orbit(obrot / FPS, 0)
        kl.licznik += 1
        if (kl.licznik - 1) % co == 0 and len(ramki) < arkusz:
            ramki.append((krok, a.copy()))
        return a
    kl.licznik = 0
    t0 = time.perf_counter()
    plik = os.path.join(OUT, f'{nazwa}.mp4')
    z._seek_index(nearest(od)); czekaj(lambda: z._fz_wynik is not None or z.combo_uklad.currentData() != 4 and
                                       z.combo_color.currentData() not in (8, 9, 10, 15, 16, 17), 60)
    nk = z.nagraj_film(plik, fps=FPS, sekundy=sekundy, od=od, do=do, napis=napis, klatka=kl)
    print(f'{nazwa}: {nk} klatek w {time.perf_counter() - t0:.0f} s -> {plik}', flush=True)
    assert z.cloud.blad == '', z.cloud.blad
    if ramki:
        w, h = ramki[0][1].shape[1], ramki[0][1].shape[0]
        kol = min(3, len(ramki)); wie = -(-len(ramki) // kol)
        sk = 0.5
        ark = Image.new('RGB', (int(w * sk) * kol, int(h * sk) * wie), (16, 23, 28))
        for i, (krok, a) in enumerate(ramki):
            im = Image.fromarray(a).resize((int(w * sk), int(h * sk)))
            ark.paste(im, ((i % kol) * int(w * sk), (i // kol) * int(h * sk)))
        ark.save(os.path.join(OUT, f'{nazwa}_arkusz.png'))
    return plik


# ====================================================================== ViT: układ względem klatki (szkło)
if os.path.isdir(SZKLO):
    wczytaj(SZKLO)
    # G1: jedno zdjęcie, 600 cząsteczek, ślady: grzechotanie w klatce bez przepływu sąsiedztw
    ustaw(uklad=0, kolor=11, scope=1, obraz=OBRAZ, krycie=.85, rozmiar=4.0, slad=4, okno=5, kamera=(1.9, 22, 30), ruch=False, kadr=1)
    nagraj('G1_szklo_klatka_jedno_zdjecie', 300, 1100, 16, obrot=3.0,
           napis=lambda k: (f'G1 · Układ względem klatki (minus ruch 6 sąsiadów): każda cząsteczka grzechocze wokół swojego miejsca, '
                            f'jak w szkle koloidalnym   |   krok {k:.0f}, zdjęcie {OBRAZ} (600 cząsteczek), co 2 kroki\n'
                            'kolory: tło szare, rogówka zielona, źrenica niebieska, narzędzie czerwone · '
                            'rms drgań 0,35 = 0,45 odstępu do sąsiada, antykorelacja kolejnych przesunięć 0,4'))
    # G2: cała chmura 60 000 cząsteczek, kolor po obszarach, wolny obrót
    ustaw(uklad=0, kolor=11, scope=0, krycie=.25, rozmiar=1.5, slad=1, okno=5, kamera=(2.6, 22, 40), ruch=False, kadr=1)
    nagraj('G2_szklo_klatka_cala_chmura', 300, 2150, 24, obrot=2.0,
           napis=lambda k: (f'G2 · 60 000 cząsteczek w układzie klatki: brak wspólnego przepływu, zostaje drganie termiczne i rzadkie '
                            f'przegrupowania   |   krok {k:.0f} (co 2 kroki, lr stały od kroku 300)\n'
                            'energia drgań maleje jak t^-0,79 przy stałym lr: szkło stygnie samo, bo słabnie gradient'))
    # G3: kolor po temperaturze (energia drgań własnych), żeby widać było, że narzędzie i źrenica są cieplejsze
    ustaw(uklad=0, kolor=17, scope=0, krycie=.3, rozmiar=1.6, slad=0, okno=5, kamera=(2.6, 22, 40), ruch=False, moc=1.3, kadr=1)
    nagraj('G3_szklo_klatka_temperatura', 300, 2150, 18, obrot=2.0,
           napis=lambda k: (f'G3 · Układ klatki, kolor = temperatura (energia drgań własnych w oknie ±5 zapisów)   |   krok {k:.0f}\n'
                            'temperatura klatki: tło 0,11, rogówka 0,15, źrenica 0,20, narzędzie 0,46 (MSD względem klatki przy τ ≈ 50)'))

# ====================================================================== ViT: ten sam fragment względem populacji (kontrast)
if os.path.isdir(E35K):
    pass                                                        # oś pomiaru ma tylko końce epok; kontrast daje klip A2 z outputs/animacje_ruchu_20261010

# ====================================================================== symulowane szkło koloidalne (dynamika browna)
for sciezka, kod, tytul in [(p, 'G4', 'stała temperatura') for p in BD[-1:]] + [(p, 'G5', 'stygnięcie własne jak tokeny') for p in BD_STYG[-1:]]:
    wczytaj(sciezka, obszary=False)
    u = json.load(open(os.path.join(sciezka, 'manifest.json'), encoding='utf-8')).get('uklad', {})
    st = z.timeline.steps
    ustaw(uklad=0, kolor=9, scope=0, krycie=.8, rozmiar=4.0, slad=2, okno=5, kamera=(2.2, 20, 30), ruch=False)
    nagraj(f'{kod}_szklo_koloidalne_BD', st[0], st[-1], 20, obrot=2.5,
           napis=lambda k, u=u, tytul=tytul: (f'{kod} · Szkło koloidalne z symulacji: dynamika browna {u.get("N", "")} miękkich kul, φ = {u.get("phi", "")}, '
                                              f'{tytul}   |   klatka {k:.0f} (co {u.get("klatka_co_rekordow", 10)} rekordów po {u.get("rekord_co_krokow", 100)} kroków)\n'
                                              'kolor = prędkość chwilowa; ten sam zestaw odcisków palca co dla tokenów (odciski.py)'))
print('GOTOWE', flush=True)
