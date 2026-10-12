"""Animacje najważniejszych typów ruchu cząsteczek w latencie, nagrane narzędziami aplikacji (zakładka „Ruch zbioru”,
metoda ZakladkaRuchu.nagraj_film), plus wskaźniki, po których widać, że sieć nauczyła się narzędzia.

Dane: historia K1000 (1500 zdjęć × 600 cząsteczek, 101 checkpointów co 10 kroków) i historia 35 000 klatek
(100 zdjęć uczących, 22 końce epok). Wynik: pliki MP4, arkusze klatek PNG i WSKAZNIKI.json w tym katalogu.
Uruchomienie: F:/Python/3.13/python.exe -u generuj_animacje.py (z katalogu app albo dowolnego)."""
import json
import os
import sys
import time

import numpy as np

APP = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/app'
OUT = os.path.dirname(os.path.abspath(__file__))
K1000 = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T080332_natural_0051d7971abb_q16'
E35K = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd'
OCENY35K = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/MalyViT_35000_od_zera_s47/dane/OCENY.json'
OBRAZ = 1247                                    # TR_0139_S1_P03_00001726: 16 patchy narzędzia, IoU 0,44 (k100) -> 0,94 (k1000)
FPS = 30
TYLKO = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None          # np. A2,A4,A5: nagraj tylko te klipy
os.chdir(APP); sys.path.insert(0, APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
q = QApplication(sys.argv)
from zakladka_ruchu import ZakladkaRuchu  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

z = ZakladkaRuchu(None); z.resize(1500, 1150); z.show(); q.processEvents()
z.spin_vram.setValue(4); z.btn_panel.setChecked(True)
WSK = {}


def czekaj(warunek, limit):
    t = time.perf_counter()
    while not warunek() and time.perf_counter() - t < limit:
        q.processEvents(); time.sleep(.01)
    return warunek()


def wczytaj(sciezka):
    z.edit_timeline.setText(sciezka)
    z._load_timeline(); assert czekaj(lambda: z.timeline is not None, 120)
    z.cloud.wczytaj_teraz(z.timeline.steps); assert czekaj(lambda: z.cloud.na_karcie() == z.cloud.T, 300)
    assert czekaj(lambda: z._obszary_klasy is not None, 180), 'obszary oka'
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


def metryki():
    return np.load(os.path.join(z.timeline.root, 'metrics.npy'))


def nearest(krok):
    st = z.timeline.steps
    return int(np.argmin(np.abs(np.asarray(st) - krok)))


def nagraj(nazwa, od, do, sekundy, napis, klatka=None, obrot=0.0, arkusz=6):
    """Nagranie MP4 + arkusz klatek PNG; obrot: stopnie obrotu kamery na sekundę filmu."""
    if TYLKO is not None and nazwa.split('_')[0] not in TYLKO:
        print(f'{nazwa}: pominięte', flush=True); return None
    ramki = []
    n = int(sekundy * FPS)
    co = max(1, n // (arkusz - 1))

    def kl(a, krok):
        if obrot:
            z.view.orbit(obrot / FPS, 0)
        if klatka is not None:
            a = klatka(a, krok)
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


def auc(a, b):
    x = np.concatenate([a, b]); r = np.empty(len(x)); r[np.argsort(x, kind='mergesort')] = np.arange(1, len(x) + 1)
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


# ====================================================================== K1000
wczytaj(K1000)
M = metryki()                                   # (1500, 101, 5): confidence, iou, delta, krok, both90
ST = z.timeline.steps

# --- wskaźniki wytrenowania w czasie: detekcja + fizyka ruchu (okno 5 zapisów wokół kroku)
ustaw(uklad=4, kolor=11, co=1, okno=5)
seria = None
wiersze = []
for krok in (30, 60, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
    z._fz_wynik = None; z._seek_index(nearest(krok)); assert czekaj(lambda: z._fz_wynik is not None, 60)
    w = z._fz_wynik; wl, kl = w['wl'], w['klasy']; seria = w['seria']
    i = nearest(krok)
    narz, tlo = kl == 3, kl == 0
    lt = np.log(np.maximum(wl[:, 0], 1e-12)); l7 = np.log(np.maximum(wl[:, 6], 1e-12))
    rez = wl[:, 7]
    wiersze.append(dict(krok=ST[i], iou_srednie=float(M[:, i, 1].mean()), pewnosc_srednia=float(M[:, i, 0].mean()),
                        iou_ge_0_5=float((M[:, i, 1] >= .5).mean()),
                        auc_temperatura_narz_vs_tlo=auc(lt[narz], lt[tlo]), auc_zmiana768_narz_vs_tlo=auc(l7[narz], l7[tlo]),
                        temperatura_narz=float(np.median(wl[narz, 0])), temperatura_tlo=float(np.median(wl[tlo, 0])),
                        rezim_balistyczny=float((rez == 0).mean()), rezim_dyfuzyjny=float((rez == 1).mean()),
                        rezim_uwieziony=float((rez == 2).mean()), alfa_mediana=float(np.median(wl[:, 1])),
                        vacf_mediana=float(np.median(wl[:, 2]))))
    print(f'krok {ST[i]:5d}: IoU {wiersze[-1]["iou_srednie"]:.3f}, AUC temp {wiersze[-1]["auc_temperatura_narz_vs_tlo"]:.2f}, '
          f'bal/dyf/uw {wiersze[-1]["rezim_balistyczny"]:.2f}/{wiersze[-1]["rezim_dyfuzyjny"]:.2f}/{wiersze[-1]["rezim_uwieziony"]:.2f}',
          flush=True)
WSK['k1000'] = dict(obraz_animacji=OBRAZ, wiersze=wiersze,
                    seria=dict(kroki=[int(k) for k in seria['kroki']],
                               temperatura_obszary=np.nan_to_num(seria['temperatura']).round(5).tolist(),
                               cos_obszary=np.nan_to_num(seria['cos']).round(4).tolist(),
                               pole_kolektywne=np.nan_to_num(seria['pole']).round(5).tolist(),
                               kolektywnosc=np.nan_to_num(seria['kolektywnosc']).round(4).tolist()))
json.dump(WSK, open(os.path.join(OUT, 'WSKAZNIKI.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def udzialy_rezimow():
    w = z._fz_wynik
    if w is None:
        return ''
    r = w['wl'][:, 7]
    return f'reżim: balistyczny {100 * (r == 0).mean():.0f} %, dyfuzyjny {100 * (r == 1).mean():.0f} %, uwięziony {100 * (r == 2).mean():.0f} %'


# --- A1: rozgrzewka = przepływ kolektywny (balistyczny), potem przejście do dyfuzji uwięzionej
ustaw(uklad=0, kolor=9, co=2, krycie=.14, rozmiar=1.6, slad=4, okno=3, kamera=(3.8, 24, 35), ruch=True)
nagraj('A1_rozgrzewka_przeplyw_kolektywny', 0, 200, 14, obrot=4.0,
       napis=lambda k: (f'A1 · Rozgrzewka: przepływ kolektywny (balistyczny) → drgania w klatce   |   krok {k:.0f} / 1000, 225 000 cząsteczek (co 2. patch, 1500 zdjęć)\n'
                        f'kolor = reżim ruchu w oknie ±3 zapisów: żółty balistyczny, niebieski dyfuzyjny, fiolet uwięziony   |   {udzialy_rezimow()}\n'
                        f'IoU średnie na 1500 zdjęciach: {M[:, nearest(k), 1].mean():.3f}'))

# --- A2: drgania w klatce (uwięzienie) na jednym obrazie, ślady
ustaw(uklad=0, kolor=11, scope=1, krycie=.6, rozmiar=3.0, slad=3, okno=5, kamera=(3.0, 20, 30), ruch=False)
nagraj('A2_drgania_w_klatce_jeden_obraz', 300, 700, 14, obrot=3.0,
       napis=lambda k: (f'A2 · Drgania w klatce: cząsteczki zawracają co zapis wokół wolno dryfującego położenia   |   krok {k:.0f}, zdjęcie {OBRAZ} (600 cząsteczek)\n'
                        f'kolor = obszar oka: czerwony narzędzie, niebieski źrenica, zielony rogówka, szary tło; ślad 3 zapisów\n'
                        + (f'w oknie ±5: α = {np.median(z._fz_wynik["wl"][:, 1]):.2f} (1 = dyfuzja), VACF = {np.median(z._fz_wynik["wl"][:, 2]):+.2f} (ujemne = zawracanie)   |   '
                           if z._fz_wynik is not None else '') + f'IoU tego zdjęcia {M[OBRAZ, nearest(k), 1]:.2f}, pewność {M[OBRAZ, nearest(k), 0]:.2f}'))

# --- A3: pozycja przed treścią: ruch w rozgrzewce na siatce zdjęcia
ustaw(uklad=1, kolor=11, scope=1, krycie=.9, rozmiar=4.0, slad=0, okno=5, cecha=3, bez_poz=False, zdj=1, kamera=(3.2, 38, -75), kadr=2)
nagraj('A3_rozgrzewka_na_siatce_zdjecia', 0, 150, 12, obrot=0.0,
       napis=lambda k: (f'A3 · Pozycja przed treścią: ruch w rozgrzewce na siatce zdjęcia   |   krok {k:.0f}, zdjęcie {OBRAZ}\n'
                        'X, Y = położenie patcha na zdjęciu, Z = norma zmiany latentu 768D od kroku 0; zdjęcie rozpięte na cząsteczkach\n'
                        'położenie patcha tłumaczy 84 % wariancji ruchu w rozgrzewce (stałe kodowanie pozycji sin-cos), treść zdjęcia prawie nic'))

# --- A4: temperatura rozdziela narzędzie od tła (układ fizyki, okno ruchome; lewa: tło, prawa: narzędzie)
ustaw(uklad=4, kolor=11, co=1, krycie=.16, rozmiar=1.8, okno=5, kamera=(3.0, 26, -62))


def para(a, krok):
    z.combo_fz_obszar.setCurrentIndex(4); z.spin_opacity.setValue(1.0); z.spin_size.setValue(2.6); z._render(); q.processEvents()
    b = z.view.grabFramebuffer().convertToFormat(z.view.grabFramebuffer().format())
    from PyQt6.QtGui import QImage
    b = b.convertToFormat(QImage.Format.Format_RGB888)
    buf = b.constBits(); buf.setsize(b.sizeInBytes())
    bb = np.frombuffer(buf, np.uint8).reshape(b.height(), b.bytesPerLine())[:, :b.width() * 3].reshape(b.height(), b.width(), 3).copy()
    z.combo_fz_obszar.setCurrentIndex(1); z.spin_opacity.setValue(.16); z.spin_size.setValue(1.8); z._render(); q.processEvents()
    im = Image.new('RGB', (a.shape[1] * 2 + 8, a.shape[0]), (40, 44, 50))
    im.paste(Image.fromarray(a), (0, 0)); im.paste(Image.fromarray(bb), (a.shape[1] + 8, 0))
    d = ImageDraw.Draw(im)
    d.text((14, a.shape[0] - 28), 'TŁO (83 % patchy)', fill=(230, 230, 230))
    d.text((a.shape[1] + 22, a.shape[0] - 28), 'NARZĘDZIE (1,7 % patchy)', fill=(255, 120, 100))
    return np.asarray(im)


def auc_teraz():
    w = z._fz_wynik
    if w is None:
        return ''
    wl, kl = w['wl'], w['klasy']
    lt = np.log(np.maximum(wl[::3, 0], 1e-12)); k3 = kl[::3]
    return f'AUC temperatury narzędzie vs tło = {auc(lt[k3 == 3], lt[k3 == 0]):.2f}'


z.combo_fz_obszar.setCurrentIndex(1)
nagraj('A4_temperatura_narzedzie_vs_tlo', 60, 1000, 16, klatka=para,
       napis=lambda k: (f'A4 · Temperatura rozdziela narzędzie od tła   |   krok {k:.0f}, okno ±5 zapisów, 900 000 cząsteczek\n'
                        'X = log temperatury (energia drgań własnych), Y = log zmiany 768D od kroku 0, Z = wykładnik MSD α\n'
                        f'{auc_teraz()}   |   narzędzie drga mocniej, ale w sumie przesuwa się mniej niż tło'))
z.combo_fz_obszar.setCurrentIndex(0)

# --- A5: jak poznać, że sieć rozpoznaje narzędzie: atencja + IoU na jednym zdjęciu
ustaw(uklad=1, kolor=11, scope=1, krycie=.95, rozmiar=6.0, slad=0, okno=5, cecha=3, bez_poz=True, zdj=1, kamera=(3.1, 42, -70), kadr=1)
z.opcje_att.combo_mapa.setCurrentIndex(0); z.combo_att_gdzie.setCurrentIndex(2); z.spin_att_wzm.setValue(4.0)
z.spin_att_krycie.setValue(.75); z.chk_att.setChecked(True)
czekaj(lambda: z._att_klucz is not None, 90)
print('atencja:', z.att_label.text()[:120], flush=True)
nagraj('A5_atencja_i_IoU_rozpoznanie_narzedzia', 0, 1000, 16, obrot=0.0,
       napis=lambda k: (f'A5 · Skąd wiemy, że sieć rozpoznaje narzędzie   |   krok {k:.0f}, zdjęcie {OBRAZ}\n'
                        'heatmapa atencji (średnia otrzymana) z wag tego checkpointu: kolor i rozmiar cząsteczek + plama na zdjęciu; '
                        'Z = zmiana latentu bez składnika pozycji\n'
                        f'pewność detekcji {M[OBRAZ, nearest(k), 0]:.2f}, IoU z etykietą {M[OBRAZ, nearest(k), 1]:.2f}   |   '
                        f'średnio na 1500 zdjęciach: IoU {M[:, nearest(k), 1].mean():.3f}, pewność {M[:, nearest(k), 0].mean():.3f}'))
z.chk_att.setChecked(False)

# ====================================================================== 35 000 klatek, 22 końce epok
wczytaj(E35K)
M2 = metryki(); ST2 = z.timeline.steps
oceny = {int(x['epoka']): float(x['mAP50']) for x in json.load(open(OCENY35K, encoding='utf-8')) if x.get('epoka')}
# rozrzut chmury (PCA3) w każdej epoce
rozrzut, promien = [], []
for s in ST2:
    P = np.asarray(z.timeline.frame(s), np.float64)[:, :3]
    c = P - P.mean(0); rozrzut.append(float(np.sqrt((c ** 2).sum(1).mean())))
    promien.append(float(np.median(np.linalg.norm(P - np.median(P, 0), axis=1))))
WSK['e35k'] = dict(epoki=[int(s // 1094) for s in ST2], kroki=[int(s) for s in ST2], rozrzut_pca3=rozrzut,
                   mediana_promienia_pca3=promien,
                   iou_srednie_100_uczacych=[float(M2[:, i, 1].mean()) for i in range(len(ST2))],
                   pewnosc_srednia=[float(M2[:, i, 0].mean()) for i in range(len(ST2))],
                   map_walidacja=[oceny.get(int(s // 1094)) for s in ST2])
json.dump(WSK, open(os.path.join(OUT, 'WSKAZNIKI.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def ep(k):
    return k / 1094


ustaw(uklad=0, kolor=11, co=1, krycie=.3, rozmiar=1.8, slad=2, okno=3, kamera=(3.6, 22, 40), ruch=False)
nagraj('A6_epoki_kondensacja_chmury', 0, ST2[-1], 16, obrot=4.0,
       napis=lambda k: (f'A6 · 21 epok na 35 000 klatkach: chmura najpierw rośnie, potem się kurczy (kondensacja)   |   epoka {ep(k):.1f}, krok {k:.0f}\n'
                        f'60 000 cząsteczek ze 100 zdjęć uczących; rozrzut chmury {np.interp(k, ST2, rozrzut):.2f} (epoka 1: {rozrzut[1]:.2f}, epoka 21: {rozrzut[-1]:.2f})\n'
                        f'mAP@0,5 walidacji po epoce {max(1, int(ep(k)))}: {oceny.get(max(1, int(ep(k))), float("nan")):.3f} (plateau od ~8. epoki)   |   '
                        f'IoU na tych zdjęciach: {M2[:, nearest(k), 1].mean():.2f}'))

print('GOTOWE', flush=True)
os._exit(0)
