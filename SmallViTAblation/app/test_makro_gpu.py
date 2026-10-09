"""Zgodność przestrzeni makro liczonej na GPU (makro_gpu.py) ze wzorami CPU.

Uruchomienie: python test_makro_gpu.py (albo pytest). Otwiera na chwilę okno z widokiem OpenGL, odczytuje
z karty bufory po kernelach compute i porównuje je z _biezace() i _v_biezace() widoku (wzory sprzed refaktoru).
"""
import os
import sys

import numpy as np

os.environ.setdefault('EDYTOR_BEZ_USTAWIEN', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _sekwencja(T=80, ziarno=0):
    rng = np.random.default_rng(ziarno)
    baza = rng.standard_normal((600, 768)).astype(np.float32)
    kroki = rng.standard_normal((T, 600, 768)).astype(np.float32) * .15
    Z = np.abs(baza[None] + np.cumsum(kroki, 0) * .3).astype(np.float16)
    maski = np.zeros((T, 600), bool); maski[:, 250:262] = True
    yy, xx = np.mgrid[0:320, 0:480]
    rgb = np.stack([xx * 255 // 479, yy * 255 // 319, (xx + yy) * 255 // 798], -1).astype(np.uint8)
    return Z, maski, rgb


def _odczyt(w):
    """Klatka narysowana teraz -> (CUR [600, 3], mikrowektory [K, 3], łańcuch [K + 1, 3]) z buforów karty."""
    w.widok.grabFramebuffer()                              # wymusza paintGL (repaint() nie rysuje synchronicznie)
    w.widok.makeCurrent()
    try:
        g = w.gpu
        assert g.blad == '', g.blad
        cur = g.odczytaj('CUR', 600)[:, :3]
        mv = g.odczytaj('MV', 2 * g.K + 1)[:, :3] if g.K else None
    finally:
        w.widok.doneCurrent()
    return cur, (None if mv is None else mv[:g.K]), (None if mv is None else mv[g.K:])


def _ustaw_postep(w, u):
    """Postęp przelotu u (0..1) jak w _krok_przelotu, bez zegara: CPU i GPU dostają to samo u."""
    w._timer.stop()
    w._u = u
    w._rysuj(tylko_punkty=True)


def sprawdz_gpu_zgodny_z_cpu():
    """Zwraca słownik z wynikami (do wypisania przy uruchomieniu skryptu)."""
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(sys.argv)
    from przestrzen_makro import WidokPrzestrzeniMakro

    Z, maski, rgb = _sekwencja()
    w = WidokPrzestrzeniMakro()
    w.resize(900, 700); w.show(); app.processEvents()
    w.combo_mikro.setCurrentIndex(2)                       # mikrowektory ze środka i łańcuch
    w.spin_slad.setValue(20)
    w.ustaw_obraz(rgb)
    w.ustaw_sekwencje(Z, maski)
    wyniki = {}
    try:
        # 1. skok do kroku: położenia bitowo równe P[t]
        w.pokaz_klatke(10, 255); w._rysuj(skok=True)
        cur, mv, lan = _odczyt(w)
        assert np.array_equal(cur, w.P[10]), 'skok: CUR != P[t]'
        v_cpu = w._v_biezace()
        np.testing.assert_allclose(mv, v_cpu, rtol=0, atol=1e-4)
        wyniki['skok'] = 'bitowo'

        # 2. przelot do nowego kroku, kilka chwil
        w.pokaz_klatke(20, 255)
        bledy = []
        for u in (.1, .37, .8):
            _ustaw_postep(w, u)
            cur, mv, lan = _odczyt(w)
            p_cpu = w._biezace(); v_cpu = w._v_biezace()
            bledy.append(float(np.abs(cur - p_cpu).max()))
            np.testing.assert_allclose(cur, p_cpu, rtol=1e-6, atol=1e-4)
            np.testing.assert_allclose(mv, v_cpu, rtol=1e-5, atol=1e-3)
            np.testing.assert_allclose(mv.sum(0), w._biezace()[255], atol=1e-2)   # wypadkowa trafia w cząstkę
            kolej = np.argsort(-np.linalg.norm(w.sily_tokenu(20, 255)[:, None] * w.D * w.s, axis=1))
            np.testing.assert_allclose(lan[1:], np.cumsum(mv[kolej], 0), atol=1e-2)
        wyniki['przelot_max_blad'] = max(bledy)

        # 3. przelot przerwany w połowie nowym krokiem: start z bieżącego położenia
        w.pokaz_klatke(30, 255)
        _ustaw_postep(w, .5)
        cur, mv, _ = _odczyt(w)
        np.testing.assert_allclose(cur, w._biezace(), rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(mv, w._v_biezace(), rtol=1e-5, atol=1e-3)

        # 4. koniec przelotu: znów dokładnie P[t]
        _ustaw_postep(w, 1.0)
        cur, _, _ = _odczyt(w)
        assert np.array_equal(cur, w.P[30]), 'koniec przelotu: CUR != P[t]'
        wyniki['renderer'] = w.gpu.renderer
    finally:
        w.close()
    return wyniki


def test_gpu_zgodny_z_cpu():
    sprawdz_gpu_zgodny_z_cpu()


if __name__ == '__main__':
    print(sprawdz_gpu_zgodny_z_cpu())
    print('OK: GPU zgodne z CPU')
