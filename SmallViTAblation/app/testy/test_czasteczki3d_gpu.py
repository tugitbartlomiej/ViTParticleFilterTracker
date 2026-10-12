"""Zgodność widoku „Cząsteczki 3D” liczonego na GPU (czasteczki3d_gpu.py) ze wzorami CPU sprzed refaktoru.

Uruchomienie: F:\\Python\\3.13\\python.exe test_czasteczki3d_gpu.py (albo pytest). Otwiera na chwilę okno z widokiem
OpenGL, wymusza klatkę (grabFramebuffer), odczytuje z karty bufory po kernelach compute, teksturę płaszczyzny obrazu
i wierzchołki policzone shaderem rysowania (transform feedback) i porównuje je z funkcjami `ref_*` poniżej, które
są przepisanymi bez zmian wzorami z czasteczki3d.py sprzed przeniesienia na GPU (_wektory, _rysuj, _mieszanka,
_podloze, _luki).
"""
import os
import sys

import numpy as np

os.environ.setdefault('EDYTOR_BEZ_USTAWIEN', '1')
os.environ.setdefault('EDYTOR_BEZ_AUTOSTARTU', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wspolne import (CZASTECZKI_CO, KOLUMNY, KOMORKA, PCA_SKLADOWE, WIERSZE, WYS, kierunki, kolor_pca_rgb,  # noqa: E402
                     kolor_podobienstwa, kolory_kanalow, tokeny_co)


# ------------------------------------------------------------ wzory CPU sprzed refaktoru
def ref_dane(w):
    """_dane: (przed, po, indeksy wektorów, kolory, kierunki) w przestrzeni wektorów (kanały albo PCA)."""
    polsfera = w.combo_ksztalt.currentIndex() == 1
    kan = w._kanaly()
    i = w.combo_pca.currentIndex()
    k = PCA_SKLADOWE[i - 1] if i > 0 else 0
    if not k:
        return w.przed, w.po, kan, kolory_kanalow(), kierunki(polsfera)
    mu, W, _ = w._baza_pca(kan)
    k = min(k, W.shape[1])
    W = W[:, :k]
    przed = (w.przed[:, kan] - mu) @ W
    po = None if w.po is None else (w.po[:, kan] - mu) @ W
    return przed, po, np.arange(k), kolory_kanalow(k), kierunki(polsfera, k)


def ref_latent(w, przed=None, po=None):
    przed = w.przed if przed is None else przed
    po = w.po if po is None else po
    tryb = w.combo_tryb.currentIndex() if po is not None else 0
    if tryb == 0:
        return przed, tryb
    if tryb == 2:
        return po - przed, tryb
    if tryb == 4:
        t = w.suwak.value() / 100
        return przed + t * (po - przed), tryb
    return po, tryb


def ref_srodki(w, A):
    t = np.arange(600)
    s = np.stack([(t % KOLUMNY) * KOMORKA + KOMORKA / 2, WYS - ((t // KOLUMNY) * KOMORKA + KOMORKA / 2),
                  np.zeros(600)], 1).astype(np.float32)
    h = w.spin_wys.value()
    if h > 0 and A is not None:
        s[:, 2] = h * A.reshape(-1)
    return s


def ref_tokeny(w):
    tok = np.arange(600)
    zazn = w.chk_zazn.isChecked() and w.zaznaczenie is not None and len(w.zaznaczenie) > 0
    if w.chk_tylko.isChecked() and w.token is not None:
        tok = np.array([w.token])
    elif zazn:
        tok = np.sort(np.asarray(w.zaznaczenie, int))
    elif w.maska_tokenow is not None and w.combo_patche.currentIndex():
        m = w.maska_tokenow if w.combo_patche.currentIndex() == 1 else ~w.maska_tokenow
        tok = tok[m]
    if not (w.chk_tylko.isChecked() and w.token is not None) and not zazn and w.combo_ile.currentIndex():
        tok = np.intersect1d(tok, tokeny_co(CZASTECZKI_CO[w.combo_ile.currentIndex()]))
    return tok


def ref_wektory(w, A, szare=False):
    """_rysuj + _wektory: (m, odcinki [T*K*2, 3], końce [T*K, 3], kolory odcinków [T*K*2, 4])."""
    przed, po, kan, kolory, d = ref_dane(w)
    L, tryb = ref_latent(w, przed, po)
    if szare:
        L = przed
    tok = ref_tokeny(w)
    polowa = KOMORKA / 2
    r0 = polowa * w.spin_r0.value() / 100
    if tryb == 2:
        zakres = np.abs(L[:, kan]) if not szare else np.abs((po - przed)[:, kan])
    else:
        zakres = np.abs(np.concatenate([przed[:, kan]] + ([po[:, kan]] if po is not None and tryb in (1, 3, 4) else [])))
    m = max(float(np.percentile(zakres, 99.5)), 1e-6)
    skala = (polowa - r0) / m * w.spin_skala.value() / 100 if polowa > r0 else polowa / m
    srodki = ref_srodki(w, A)
    maks = KOMORKA * w.spin_maks.value() / 100 or None
    dl = np.clip(r0 + skala * L[np.ix_(tok, kan)], 0, maks)
    konce = srodki[tok, None, :] + dl[..., None] * d[None, kan, :]
    odc = np.empty((len(tok), len(kan), 2, 3), np.float32)
    odc[:, :, 0] = srodki[tok, None, :]
    odc[:, :, 1] = konce
    kol = kolory[kan]
    kol_odc = np.broadcast_to(kol[None, :, None, :], (len(tok), len(kan), 2, 4)).reshape(-1, 4)
    return m, odc.reshape(-1, 3), konce.reshape(-1, 3), kol_odc


def ref_zera(w, A):
    tok = ref_tokeny(w)
    r0 = KOMORKA / 2 * w.spin_r0.value() / 100
    srodki = ref_srodki(w, A)
    th = np.linspace(0, 2 * np.pi, 33)
    o = srodki[tok, None, :] + r0 * np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)[None]
    return np.stack([o[:, :-1], o[:, 1:]], 2).reshape(-1, 3)


def ref_mieszanka(w):
    import matplotlib  # noqa: F401
    i = w.combo_waga.currentIndex()
    if i in (4, 5):
        L, _ = ref_latent(w)
        return kolor_pca_rgb(L, w.baza_rgb()) if i == 4 else kolor_podobienstwa(L, w.token)
    przed, po, kan, kolory, _ = ref_dane(w)
    L, _ = ref_latent(w, przed, po)
    X = np.asarray(L, np.float32)[:, kan]
    W = np.abs(X) if i == 0 else X ** 2 if i == 1 else X ** 4 if i == 2 else np.clip(X, 0, None)
    sila = W.sum(1)
    C = W @ np.asarray(kolory, np.float32)[kan, :3] / np.maximum(sila, 1e-12)[:, None]
    g = C.mean(1, keepdims=True)
    ref = max(float(np.percentile(np.abs(C - g).max(1), 98)), 1e-6)
    C = np.clip(g + (C - g) * (.45 / ref) * w.spin_nasyc.value(), 0, 1)
    if w.chk_mies_jasn.isChecked():
        C = C * np.clip(sila / max(float(np.percentile(sila, 99)), 1e-12), 0, 1)[:, None]
    return C.astype(np.float32)


def ref_podloze(w, A, mies):
    """_podloze: RGBA [320, 480, 4] uint8 w kolejności wierszy tekstury (y w górę)."""
    import matplotlib
    from wspolne import SZER
    img = np.zeros((WYS, SZER, 3), np.float32)
    if w.rgb is not None and w.chk_obraz.isChecked():
        img = np.asarray(w.rgb, np.float32)[:WYS, :SZER, :3] / 255 * w.spin_alfa.value()
    if w.chk_atencja.isChecked() and A is not None:
        kol = matplotlib.colormaps['inferno'](A)[..., :3].astype(np.float32)
        kol = np.repeat(np.repeat(kol, KOMORKA, 0), KOMORKA, 1)
        a = w.opcje.krycie()
        img = img * (1 - a) + kol * a
    if w.chk_mies.isChecked() and mies is not None:
        M = mies.reshape(WIERSZE, KOLUMNY, 3)
        if w.chk_mies_gladko.isChecked():
            from scipy import ndimage
            M = ndimage.zoom(M, (KOMORKA, KOMORKA, 1), order=1, mode='nearest')[:WYS, :SZER]
        else:
            M = np.repeat(np.repeat(M, KOMORKA, 0), KOMORKA, 1)
        a = w.spin_mies_krycie.value()
        img = img * (1 - a) + M * a
    alfa = int(round(255 * w.spin_krycie.value())) if (w.chk_obraz.isChecked() or w.chk_atencja.isChecked()
                                                       or w.chk_mies.isChecked()) else 0
    rgba = np.concatenate([np.clip(img * 255, 0, 255).astype(np.uint8), np.full((WYS, SZER, 1), alfa, np.uint8)], 2)
    return np.flipud(rgba)


def ref_luki(w, A, strz):
    import matplotlib
    srodki = ref_srodki(w, A)
    src, dst, wg = strz
    a, b = srodki[src], srodki[dst]
    n = 24
    u = np.linspace(0, 1, n + 1, dtype=np.float32)
    h = .35 * np.linalg.norm(b - a, axis=1) + 8
    p = a[:, None] * (1 - u)[None, :, None] + b[:, None] * u[None, :, None]
    p[..., 2] += 4 * h[:, None] * u * (1 - u)
    odc = np.stack([p[:, :-1], p[:, 1:]], 2).reshape(-1, 3)
    kol = matplotlib.colormaps['cool'](wg).astype(np.float32)
    alfa = .15 + .85 * np.repeat(np.stack([u[:-1], u[1:]], 1)[None], len(src), 0)
    c = np.repeat(np.repeat(kol[:, None, None, :], n, 1), 2, 2).copy()
    c[..., 3] = alfa * (.35 + .65 * wg[:, None, None])
    kg = kol.copy(); kg[:, 3] = 1
    return odc, c.reshape(-1, 4), b + np.array([0, 0, .5], np.float32), kg, 4 + 8 * wg


# ------------------------------------------------------------ odczyt z karty
def _z_karty(w, fn):
    """Wymusza narysowanie klatki i wywołuje fn(gpu) przy aktywnym kontekście widoku."""
    w.widok.grabFramebuffer()                              # repaint() nie rysuje synchronicznie
    w.widok.makeCurrent()
    try:
        assert w.gpu.blad == '', w.gpu.blad
        return fn(w.gpu)
    finally:
        w.widok.doneCurrent()


def _dane(ziarno=0):
    rng = np.random.default_rng(ziarno)
    przed = (rng.standard_normal((600, 768)) * .6 + .2).astype(np.float32)
    po = (przed + rng.standard_normal((600, 768)).astype(np.float32) * .3).astype(np.float32)
    yy, xx = np.mgrid[0:320, 0:480]
    rgb = np.stack([xx * 255 // 479, yy * 255 // 319, (xx + yy) * 255 // 798], -1).astype(np.uint8)
    A = rng.random((4, 600, 600)).astype(np.float32) ** 8
    A /= A.sum(2, keepdims=True)
    return przed, po, rgb, (lambda b: A)


def _porownaj_wektory(w, A, opis, wyniki):
    m_ref, odc, konce, kol = ref_wektory(w, A)
    poz, k, _ = _z_karty(w, lambda g: g.geometria(0))
    m_gpu = w.gpu.m
    np.testing.assert_allclose(m_gpu, m_ref, rtol=2e-6, err_msg=f'{opis}: 99,5. centyl')
    np.testing.assert_allclose(poz, odc, rtol=1e-5, atol=2e-3, err_msg=f'{opis}: odcinki')
    np.testing.assert_array_equal(k, kol, err_msg=f'{opis}: kolory')
    pk, _, rozm = _z_karty(w, lambda g: g.geometria(1))
    np.testing.assert_allclose(pk, konce, rtol=1e-5, atol=2e-3, err_msg=f'{opis}: końce')
    assert np.all(rozm == np.float32(w.spin_gr.value() + 1.5))
    wyniki[opis] = dict(n=len(odc) // 2, m_wzgl=abs(m_gpu / m_ref - 1), maks_blad=float(np.abs(poz - odc).max()))


def test_gpu_zgodny_z_cpu():
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(sys.argv)
    from czasteczki3d import WidokCzasteczek3D

    przed, po, rgb, fn = _dane()
    w = WidokCzasteczek3D(np.arange(0, 768, 6))
    w.resize(1000, 700); w.show(); app.processEvents()
    wyniki = {}
    try:
        w.combo_tryb.setCurrentIndex(1)
        w.pokaz(przed, po, rgb, fn)
        _porownaj_wektory(w, None, 'PO, 768 kanałów', wyniki)

        w.combo_tryb.setCurrentIndex(4)                     # przejście: na kartę idzie tylko t
        _z_karty(w, lambda g: None)
        for t in (0, 37, 100):
            w.suwak.setValue(t)
            # ani latent, ani wartości, ani centyl nie są liczone od nowa (zmienia się tylko uniform t)
            assert w.gpu._lat is None and w.gpu._brudne <= {'mieszanka', 'podloze'}
            _porownaj_wektory(w, None, f'przejście t = {t} %', wyniki)

        w.combo_tryb.setCurrentIndex(2); w.spin_maks.setValue(60); w.spin_r0.setValue(0)
        _porownaj_wektory(w, None, 'różnica, przycięcie, r0 = 0', wyniki)
        w.spin_maks.setValue(0); w.spin_r0.setValue(25)

        w.combo_tryb.setCurrentIndex(3)                     # nałożone: szare PRZED
        _porownaj_wektory(w, None, 'nałożone', wyniki)
        _, odc_sz, _, _ = ref_wektory(w, None, szare=True)
        poz, k, _ = _z_karty(w, lambda g: g.geometria(0, zrodlo=0))
        np.testing.assert_allclose(poz, odc_sz, rtol=1e-5, atol=2e-3)
        assert np.allclose(k, [.55, .55, .55, .5])

        # PCA na kanałach J128, półsfera, co 2. patch, podniesienie o atencję, okręgi zera
        w.combo_tryb.setCurrentIndex(1)
        w.combo_kan.setCurrentIndex(4); w.combo_pca.setCurrentIndex(3); w.combo_ksztalt.setCurrentIndex(1)
        w.combo_ile.setCurrentIndex(1); w.spin_wys.setValue(40); w.chk_zero.setChecked(True)
        A = w.opcje.mapa(w.token)[0]
        _porownaj_wektory(w, A, 'PCA 8 z J128, półsfera, co 2., podniesienie', wyniki)
        poz, k, _ = _z_karty(w, lambda g: g.geometria(2))
        np.testing.assert_allclose(poz, ref_zera(w, A), rtol=1e-5, atol=1e-3)

        # wybrany patch i obrys narzędzia
        w.combo_pca.setCurrentIndex(0); w.combo_kan.setCurrentIndex(1); w.combo_ile.setCurrentIndex(0)
        maska = np.zeros(600, bool); maska[200:260] = True
        w.ustaw_obrysy([], maska); w.combo_patche.setCurrentIndex(2)
        _porownaj_wektory(w, A, 'co 2. kanał, poza obrysem', wyniki)
        w.combo_patche.setCurrentIndex(0)
        w._klik(8 * 16 + 3, WYS - (5 * 16 + 3)); w.chk_tylko.setChecked(True)
        assert w.token == 5 * 30 + 8
        _porownaj_wektory(w, A, 'tylko wybrany patch', wyniki)
        w.chk_tylko.setChecked(False); w.spin_wys.setValue(0); w.chk_zero.setChecked(False)

        # łuki atencji: każdy patch do najsilniejszego celu (600 łuków), z podniesieniem
        w.chk_atencja.setChecked(True); w.opcje.combo_strz.setCurrentIndex(3); w.spin_wys.setValue(25)
        A = w.opcje.mapa(w.token)[0]
        strz = w.opcje.strzalki(w.token)
        odc, c, groty, kg, rozm = ref_luki(w, A, strz)
        poz, k, _ = _z_karty(w, lambda g: g.geometria(3))
        np.testing.assert_allclose(poz, odc, rtol=1e-5, atol=2e-3)
        np.testing.assert_allclose(k, c, atol=1e-6)
        poz, k, r = _z_karty(w, lambda g: g.geometria(4))
        np.testing.assert_allclose(poz, groty, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(k, kg, atol=1e-6)
        np.testing.assert_allclose(r, rozm, rtol=1e-6)
        wyniki['łuki'] = len(strz[0])

        # płaszczyzna obrazu: zdjęcie, mapa atencji, mieszanka kolorów (wszystkie wagi, kwadraty i gładko)
        tex = _z_karty(w, lambda g: g.odczytaj_podloze())
        _porownaj_podloze(tex, ref_podloze(w, A, None), 'obraz + atencja', wyniki)
        w.chk_atencja.setChecked(False); w.spin_wys.setValue(0)
        w.chk_mies.setChecked(True); w.combo_tryb.setCurrentIndex(4); w.suwak.setValue(63)
        w.combo_kan.setCurrentIndex(0)
        for waga, jasn, gladko in ((0, False, False), (1, True, False), (2, False, True), (3, True, True),
                                   (4, False, False), (5, False, True)):
            w.combo_waga.setCurrentIndex(waga); w.chk_mies_jasn.setChecked(jasn); w.chk_mies_gladko.setChecked(gladko)
            w.spin_nasyc.setValue(1.7)
            M_ref = ref_mieszanka(w)
            M, tex = _z_karty(w, lambda g: (g.odczytaj('MIES', 600, kolumny=4)[:, :3], g.odczytaj_podloze()))
            np.testing.assert_allclose(M, M_ref, atol=2e-4, err_msg=f'mieszanka waga {waga}')
            _porownaj_podloze(tex, ref_podloze(w, None, M_ref), f'płaszczyzna, waga {waga}, gładko {gladko}', wyniki)
        # „Kopia w oknie”: nowy widok, dane przychodzą przed pierwszą klatką (bufory tworzone przy 1. rysowaniu)
        w.chk_mies.setChecked(False); w.combo_tryb.setCurrentIndex(1)
        w._osobne()
        kopia = w._okna[-1]
        app.processEvents()
        try:
            _porownaj_wektory(kopia, None, 'kopia w oknie (dane przed 1. klatką)', wyniki)
        finally:
            kopia.close()
        wyniki['renderer'] = w.gpu.renderer
    finally:
        w.close()
    return wyniki


def _porownaj_podloze(tex, ref, opis, wyniki):
    """Tekstura RGBA8 z karty a obraz z CPU: różnica najwyżej 1 poziom (zaokrąglenia float32 przed obcięciem
    do uint8), prawie wszystkie piksele identyczne."""
    roznica = np.abs(tex.astype(int) - ref.astype(int))
    zgodne = float((roznica == 0).mean())
    assert roznica.max() <= 1, f'{opis}: różnica {roznica.max()}'
    assert zgodne > .98, f'{opis}: identycznych pikseli {zgodne:.4f}'
    wyniki[opis] = f'identycznych {100 * zgodne:.2f} % kanałów, max różnica {roznica.max()}'


if __name__ == '__main__':
    for k, v in test_gpu_zgodny_z_cpu().items():
        print(f'{k}: {v}')
    print('OK: GPU zgodne z CPU')
