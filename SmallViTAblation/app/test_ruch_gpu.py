"""Zgodność ruchu zbioru liczonego na GPU (ruch_gpu.py) z referencją numpy, przez odczyt buforów karty.

Uruchomienie: F:/Python/3.13/python.exe -m pytest test_ruch_gpu.py (albo python test_ruch_gpu.py).
Otwiera na chwilę okno z widokiem OpenGL; klatkę wymusza grabFramebuffer() (repaint() nie rysuje synchronicznie),
potem makeCurrent() + glGetBufferSubData + doneCurrent().
Sprawdza: zapisane stany bitowo równe plikowi (także przesunięcie od pierwszego kroku), stany pośrednie i kolory,
granice osi i skale kolorów, statystyki obrazów w krokach, mapę 20 × 30, dokładną medianę normy 768D, wybór punktu
kliknięciem i tryb pierścienia (budżet VRAM mniejszy niż oś czasu).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('EDYTOR_BEZ_USTAWIEN', '1')
os.environ.setdefault('EDYTOR_BEZ_AUTOSTARTU', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOKENY = 600
_ZYWE = []


class OsSyntetyczna:
    """Atrapa ruch_zbioru.Timeline: losowy ruch 6 obrazów × 600 cząsteczek w 6 zapisanych krokach."""

    def __init__(self, obrazy=6, kroki=(0, 3, 7, 12, 20, 31), ziarno=0):
        rng = np.random.default_rng(ziarno)
        self.steps = list(kroki)
        self.image_count = obrazy
        self.count = obrazy * TOKENY
        self.metadata = {'norm_scale': 1e-6}
        self.image_names = [f'obraz_{i}.png' for i in range(obrazy)]
        start = rng.normal(size=(self.count, 3)).astype(np.float32) * 3
        self.dane = {}
        p = start.copy()
        for k, s in enumerate(self.steps):
            if k:
                p = (p + rng.normal(size=p.shape).astype(np.float32) * .4).astype(np.float32)
            a = np.empty((self.count, 4), np.float32)
            a[:, :3] = p
            a[:, 3] = np.abs(rng.normal(size=self.count)).astype(np.float32) * (k + 1)
            self.dane[s] = a
        self.dane[self.steps[0]][:, 3] = 0

    def frame(self, step):
        return self.dane[step].copy()


@pytest.fixture(scope='module')
def app():
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication.instance() or QApplication(sys.argv)


def _scena(app, os_czasu, budzet=None):
    from ruch_gpu import ChmuraRuchuGPU
    from widok_gl import WidokGL
    widok = WidokGL(); widok.resize(640, 480)
    widok.setCameraPosition(distance=3.8, elevation=18, azimuth=35)
    chmura = ChmuraRuchuGPU()
    widok.addItem(chmura)
    widok.show(); app.processEvents()
    chmura.ustaw_os(os_czasu, budzet=budzet)
    return widok, chmura


def _zamknij(app, widok, chmura):
    """Zwolnij bufory przy aktywnym kontekście, potem usuń widok (bez tego kolejny widok może trafić na
    usuwany kontekst)."""
    chmura.zamknij()
    widok.makeCurrent()
    try:
        chmura.release()
    finally:
        widok.doneCurrent()
    widok.close()
    widok.deleteLater()
    app.processEvents()


def _klatka(widok, chmura, **p):
    """Ustaw parametry, wymuś rysowanie i zwróć bufory CUR [N, 4] i FR [8, 4] z karty."""
    domyslne = dict(a=chmura.steps[0], b=chmura.steps[0], s=0.0, ruch=False, kolor=0, kadr=0, first=0,
                    count=chmura.N, rozmiar=2.0, krycie=.5, slad=0)
    domyslne.update(p)
    chmura.ustaw_klatke(**domyslne)
    widok.grabFramebuffer()
    widok.makeCurrent()
    try:
        assert chmura.blad == '', chmura.blad
        cur = chmura.odczytaj('CUR', chmura.N)
        fr = chmura.odczytaj('FR', 8)
    finally:
        widok.doneCurrent()
    return cur, fr


def _wszystkie(widok, chmura, app):
    chmura.wczytaj_teraz(chmura.steps)
    for _ in range(50):
        widok.grabFramebuffer(); app.processEvents()
        if chmura.na_karcie() == min(chmura.T, chmura.n_slotow) and not chmura._gotowe:
            break


def _granice(xyz):
    return np.stack([xyz.min(0), xyz.max(0)], 1)


def test_kinematyka_bitowo_i_interpolacja(app):
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        assert ch.cala_os and ch.na_karcie() == o.steps.__len__()
        s0 = o.steps[0]
        for k in range(len(o.steps) - 1):
            a, b = o.steps[k], o.steps[k + 1]
            cur, _ = _klatka(widok, ch, a=a, b=b, s=0.0)
            assert np.array_equal(cur, o.dane[a]), f'stan {a} nie jest bitowo równy zapisowi'
            cur, _ = _klatka(widok, ch, a=a, b=b, s=1.0)
            assert np.array_equal(cur, o.dane[b]), f'stan {b} nie jest bitowo równy zapisowi'
            cur, _ = _klatka(widok, ch, a=a, b=b, s=0.0, ruch=True)
            assert np.array_equal(cur[:, :3], o.dane[a][:, :3] - o.dane[s0][:, :3]), 'przesunięcie od startu'
            assert np.array_equal(cur[:, 3], o.dane[a][:, 3])
            for s in (.25, .61):
                cur, _ = _klatka(widok, ch, a=a, b=b, s=s)
                ref = o.dane[a] + (o.dane[b] - o.dane[a]) * np.float32(s)
                np.testing.assert_allclose(cur, ref, rtol=1e-5, atol=1e-5)
                cur, _ = _klatka(widok, ch, a=a, b=b, s=s, kolor=1)
                v = np.linalg.norm(o.dane[b][:, :3] - o.dane[a][:, :3], axis=1) / (b - a)
                np.testing.assert_allclose(cur[:, 3], v, rtol=1e-5, atol=1e-6)
                cur, _ = _klatka(widok, ch, a=a, b=b, s=s, kolor=2, ruch=True)
                np.testing.assert_allclose(cur[:, :3], ref[:, :3] - o.dane[s0][:, :3], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(cur[:, 3], np.linalg.norm(cur[:, :3], axis=1), rtol=1e-5, atol=1e-5)
        # jeden obraz: dokładna tożsamość cząsteczek 600..1199
        cur, _ = _klatka(widok, ch, a=o.steps[2], b=o.steps[3], s=0.0, first=600, count=600)
        assert np.array_equal(cur[600:1200], o.dane[o.steps[2]][600:1200])
    finally:
        _zamknij(app, widok, ch)


def test_granice_skale_statystyki_i_mapa(app):
    o = OsSyntetyczna(ziarno=1)
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        P = np.stack([o.dane[s] for s in o.steps])                       # [T, N, 4]
        D = P[:, :, :3] - P[:1, :, :3]
        # kadr 0: cały zbiór, wszystkie kroki (bez granic manifestu)
        for ruch, xyz in ((False, P[:, :, :3]), (True, D)):
            _, fr = _klatka(widok, ch, ruch=ruch, kadr=0)
            g = _granice(xyz.reshape(-1, 3))
            assert np.array_equal(fr[1, :3], g[:, 0]) and np.array_equal(fr[2, :3], g[:, 1])
            np.testing.assert_allclose(fr[0, :3], g.mean(1), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(fr[0, 3], 2 / max(np.max(g[:, 1] - g[:, 0]), .001), rtol=1e-6)
        # kadr 1: widoczne cząsteczki (obraz 2), wszystkie kroki
        _, fr = _klatka(widok, ch, kadr=1, first=1200, count=600)
        g = _granice(P[:, 1200:1800, :3].reshape(-1, 3))
        assert np.array_equal(fr[1, :3], g[:, 0]) and np.array_equal(fr[2, :3], g[:, 1])
        # kadr 2: bieżący stan widocznych cząsteczek (także pośredni)
        a, b = o.steps[1], o.steps[2]
        cur, fr = _klatka(widok, ch, a=a, b=b, s=.4, kadr=2, first=1200, count=600)
        g = _granice(cur[1200:1800, :3])
        assert np.array_equal(fr[1, :3], g[:, 0]) and np.array_equal(fr[2, :3], g[:, 1])
        # skale kolorów: norma 768D, prędkość między zapisami, |Δ| w PCA
        dt = np.diff(o.steps).astype(np.float32)
        v = np.linalg.norm(np.diff(P[:, :, :3], axis=0), axis=2) / dt[:, None]
        np.testing.assert_allclose(fr[3, :3], [P[:, :, 3].max(), v.max(), np.linalg.norm(D, axis=2).max()],
                                   rtol=1e-5)
        # statystyki obrazów w krokach (PIB) i mapa ruchu
        widok.makeCurrent()
        try:
            pib = ch.odczytaj('PIB', ch.T * ch.images * 4).reshape(ch.T, ch.images, 4, 4)
            heat = ch.odczytaj('HEAT', ch.T * ch.tpi, kolumny=1).reshape(ch.T, ch.tpi)
            known = ch.odczytaj('KNOWN', ch.T, dtype=np.uint32, kolumny=1)
        finally:
            widok.doneCurrent()
        assert known.tolist() == [1] * ch.T
        Pi = P.reshape(ch.T, ch.images, TOKENY, 4)
        Di = D.reshape(ch.T, ch.images, TOKENY, 3)
        assert np.array_equal(pib[:, :, 0, :3], Pi[..., :3].min(2)) and np.array_equal(pib[:, :, 1, :3], Pi[..., :3].max(2))
        assert np.array_equal(pib[:, :, 2, :3], Di.min(2)) and np.array_equal(pib[:, :, 3, :3], Di.max(2))
        assert np.array_equal(pib[:, :, 0, 3], Pi[..., 3].max(2))
        np.testing.assert_allclose(heat, Pi[..., 3].mean(1), rtol=1e-5, atol=1e-6)
    finally:
        _zamknij(app, widok, ch)


def test_mediana_dokladna_i_mapa_widoku(app):
    o = OsSyntetyczna(obrazy=5, ziarno=2)
    widok, ch = _scena(app, o)
    wyniki = []
    ch.statystyki_gotowe.connect(wyniki.append)
    try:
        _wszystkie(widok, ch, app)
        for krok, first, count in ((o.steps[3], 0, o.count), (o.steps[4], 1800, 600), (o.steps[1], 600, 601)):
            ch.zadaj_statystyki(krok, first, count)
            _klatka(widok, ch, a=krok, b=krok, first=first, count=count)
            w = wyniki[-1]
            assert w['krok'] == krok and w['first'] == first
            x = o.dane[krok][first:first + count, 3]
            srt = np.sort(x)
            dol, gora = srt[(count - 1) // 2], srt[count // 2]
            assert w['bity'][0] == dol.view(np.uint32) and w['bity'][1] == gora.view(np.uint32), 'mediana nie jest dokładna'
            assert w['bity'][2] == x.max().view(np.uint32)
            assert np.float32(w['mediana']) == np.float32(np.median(x.astype(np.float64)))
            if count == o.count:
                np.testing.assert_allclose(w['mapa'], x.reshape(o.image_count, TOKENY).mean(0), rtol=1e-5, atol=1e-6)
            else:
                assert np.array_equal(w['mapa'], x)
    finally:
        _zamknij(app, widok, ch)


def test_wybor_punktu_zgodny_z_cpu(app):
    o = OsSyntetyczna(obrazy=3, ziarno=3)
    widok, ch = _scena(app, o)
    wybory = []
    ch.wybrano.connect(wybory.append)
    try:
        _wszystkie(widok, ch, app)
        cur, fr = _klatka(widok, ch, a=o.steps[2], b=o.steps[3], s=.5)
        scena = (cur[:, :3] - fr[0, :3]) * fr[0, 3]
        ekran, glebia = widok.na_ekran(scena)
        for i in (17, 905, 1777):
            x, y = ekran[i]
            ch.zadaj_wybor(x + .4, y - .3, widok.width(), widok.height())
            widok.grabFramebuffer()
            d = ((ekran - np.array([x + .4, y - .3])) ** 2).sum(1)
            d[~np.isfinite(glebia)] = np.inf
            k = int(d.argmin())
            got, d2 = ch.ostatni_wybor
            assert got == k or abs(d[got] - d[k]) < 1e-3, (got, k, d[got], d[k])
            assert wybory[-1] == got
    finally:
        _zamknij(app, widok, ch)


def test_pierscien_przy_malym_budzecie(app):
    o = OsSyntetyczna(obrazy=4, kroki=(0, 2, 4, 6, 8, 10, 12, 14), ziarno=4)
    widok, ch = _scena(app, o, budzet=3 * o.count * 16)
    try:
        widok.grabFramebuffer()
        assert ch.n_slotow == 3 and not ch.cala_os
        for k in range(len(o.steps) - 1):
            a, b = o.steps[k], o.steps[k + 1]
            ch.ustaw_klatke(a, b, 0.0, False, 0, 0, 0, ch.N, 2.0, .5)
            for _ in range(200):
                ch.pompuj()
                if ch.gotowy(a) and ch.gotowy(b) and ch.gotowy(o.steps[0]):
                    break
                app.processEvents()
            cur, _ = _klatka(widok, ch, a=a, b=b, s=0.0)
            assert np.array_equal(cur, o.dane[a])
            cur, _ = _klatka(widok, ch, a=a, b=b, s=1.0, ruch=True)
            assert np.array_equal(cur[:, :3], o.dane[b][:, :3] - o.dane[o.steps[0]][:, :3])
            assert ch.na_karcie() <= 3
        # po przejściu całej osi granice obejmują wszystkie kroki, mimo że na karcie są tylko 3 stany
        _, fr = _klatka(widok, ch, a=o.steps[-1], b=o.steps[-1], kadr=0)
        g = _granice(np.concatenate([o.dane[s][:, :3] for s in o.steps]))
        assert np.array_equal(fr[1, :3], g[:, 0]) and np.array_equal(fr[2, :3], g[:, 1])
    finally:
        _zamknij(app, widok, ch)


def test_slady_i_wyrozniona_czasteczka_bez_bledow_gl(app):
    o = OsSyntetyczna(obrazy=3, ziarno=5)
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        ch.ustaw_wybrany(700)
        for first, count in ((600, 600), (0, o.count)):            # ślad obrazu i ślad wybranej cząsteczki
            _klatka(widok, ch, a=o.steps[3], b=o.steps[4], s=.5, first=first, count=count, slad=4, ruch=True)
            assert ch.blad == ''
        _klatka(widok, ch, a=o.steps[3], b=o.steps[4], s=.5, slad=0)
    finally:
        _zamknij(app, widok, ch)


class OsZakladki(OsSyntetyczna):
    """Atrapa z metadanymi, których używa zakładka (statystyki, granice, krzywe)."""
    checkpoint_sha256 = 'abc'

    def stats(self, step):
        return {'confidence': .5, 'iou': .4, 'both90': 0}

    def bounds(self, movement):
        raise ValueError('brak granic producenta')

    def curves(self):
        return {'steps': self.steps, 'confidence': [.5] * len(self.steps), 'iou': [.4] * len(self.steps)}


def test_zakladka_odtwarza_z_karty(app):
    """Cała zakładka z widokiem GL: odtwarzanie przez zapisy, stan zapisany bitowo, mapa i kliknięcie z karty."""
    from zakladka_ruchu import ZakladkaRuchu
    o = OsZakladki(obrazy=2, kroki=(0, 5, 10, 15), ziarno=6)
    tab = ZakladkaRuchu(); tab.resize(1000, 700); tab.show(); app.processEvents()
    try:
        assert not tab.timer.isActive() and tab.cloud.timeline is None
        tab._loaded((o, o.frame(0), o.frame(5)))
        for _ in range(100):
            tab._tick(); tab.view.grabFramebuffer(); app.processEvents()
            if tab.cloud.na_karcie() == len(o.steps) and tab._gpu_stats is not None:
                break
        assert tab.cloud.na_karcie() == len(o.steps) and tab.cloud.blad == ''
        tab.spin_tempo.setValue(1000)
        tab._toggle_play()
        for _ in range(100):
            tab._tick(); tab.view.grabFramebuffer(); app.processEvents()
            if not tab._playing:
                break
        assert tab._visual_step == 15.0 and 'Zapisany stan 15' in tab.frame_label.text()
        widok, ch = tab.view, tab.cloud
        widok.grabFramebuffer(); widok.makeCurrent()
        try:
            cur = ch.odczytaj('CUR', ch.N)
        finally:
            widok.doneCurrent()
        assert np.array_equal(cur[:, :3], o.dane[15][:, :3] - o.dane[0][:, :3])
        tab._seek_index(1); tab._visual_step = 7.5; tab._render()
        assert 'NIEZAPISANY' in tab.frame_label.text()
        for _ in range(20):
            tab._tick(); widok.grabFramebuffer(); app.processEvents()
            if tab._gpu_stats and tab._gpu_stats['krok'] == 5:
                break
        assert tab._gpu_stats['krok'] == 5
        np.testing.assert_allclose(tab.heat.getImageItem().image.T, o.dane[5][:, 3].reshape(2, 20, 30).mean(0),
                                   rtol=1e-5, atol=1e-6)
        assert 'mediana' in tab.metrics.text()
        widok.makeCurrent()
        try:
            cur = ch.odczytaj('CUR', ch.N); fr = ch.odczytaj('FR', 8)
        finally:
            widok.doneCurrent()
        ekran, _ = widok.na_ekran((cur[:, :3] - fr[0, :3]) * fr[0, 3])
        x, y = ekran[777]
        tab._pick(x, y)
        for _ in range(10):
            widok.grabFramebuffer(); app.processEvents()
            if 'Obraz 1' in tab.image_name.text():
                break
        assert tab.image_name.text().startswith('Obraz 1 · patch 177') and ch.wybrany == 777
    finally:
        assert tab.shutdown()
        tab.close()
        _ZYWE.append(tab)     # bez deleteLater: pyqtgraph trzyma globalny rejestr ViewBox, a usunięcie pokazanych
                              # wykresów zostawia w nim martwe menu, które psują kolejne testy w tym procesie


def test_prawdziwa_os_czasu_bitowo(app):
    import ruch_zbioru as R
    if not Path(R.DEFAULT_TIMELINE, 'manifest.json').is_file():
        pytest.skip('brak lokalnej osi czasu DEFAULT_TIMELINE')
    tl = R.Timeline(R.DEFAULT_TIMELINE)
    widok, ch = _scena(app, tl, budzet=8 * tl.count * 16)
    try:
        kroki = [tl.steps[0], tl.steps[1], tl.steps[len(tl.steps) // 2], tl.steps[-1]]
        ch.wczytaj_teraz(kroki)
        for a in kroki[1:]:
            cur, _ = _klatka(widok, ch, a=a, b=a, s=0.0, ruch=True)
            ref = np.array(tl.frame(a)); ref0 = np.array(tl.frame(tl.steps[0]))
            assert np.array_equal(cur[:, :3], ref[:, :3] - ref0[:, :3])
            assert np.array_equal(cur[:, 3], ref[:, 3])
    finally:
        _zamknij(app, widok, ch)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))


class OsQ16(OsSyntetyczna):
    """Ta sama atrapa w formacie q16 (jak ruch_zbioru.Timeline po zapisie 16-bitowym): read_raw + quant."""

    def __init__(self, **kw):
        super().__init__(**kw)
        import ruch_zbioru as R
        self.frame_format = 'q16'
        self.q = {s: R.kwantyzuj(a) for s, a in self.dane.items()}
        self.dane = {s: R.odkwantyzuj(q, quant) for s, (q, quant) in self.q.items()}   # referencja = odkwantowane

    def read_raw(self, step):
        return self.q[step][0].copy()

    def quant(self, step):
        import ruch_zbioru as R
        return R.parametry_kwantyzacji(self.q[step][1])


def test_q16_na_karcie_polowa_pamieci_i_wartosci(app):
    """Stany 16-bitowe: połowa bajtów na karcie, shader odkwantowuje tak jak ruch_zbioru.odkwantyzuj."""
    o = OsQ16()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        assert ch.q16 and ch.frame_bytes == o.count * 8 and ch.na_karcie() == len(o.steps)
        for krok in o.steps:
            for ruch in (False, True):
                cur, _ = _klatka(widok, ch, a=krok, b=krok, ruch=ruch)
                ref = o.dane[krok][:, :3] - (o.dane[o.steps[0]][:, :3] if ruch else 0)
                np.testing.assert_allclose(cur[:, :3], ref, rtol=0, atol=1e-5)   # 1-2 bity float32 (FMA na karcie)
                if not ruch:
                    np.testing.assert_allclose(cur[:, 3], o.dane[krok][:, 3], rtol=0, atol=2e-6)
        a, b = o.steps[1], o.steps[2]
        cur, _ = _klatka(widok, ch, a=a, b=b, s=.3)
        ref = o.dane[a][:, :3] + (o.dane[b][:, :3] - o.dane[a][:, :3]) * np.float32(.3)
        np.testing.assert_allclose(cur[:, :3], ref, rtol=0, atol=2e-5)
    finally:
        _ZYWE.append((widok, ch))
