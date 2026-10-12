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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _siatka_ref(n, tpi, mozaika, kafle):
    i = np.arange(n); im, t = i // tpi, i % tpi
    xy = np.stack([t % 30, -(t // 30)], 1).astype(np.float32)
    if mozaika:
        xy += np.stack([(im % kafle) * 31.0, -(im // kafle) * 21.0], 1).astype(np.float32)
    return xy


def test_uklady_siatka_i_przestrzen_fazowa(app):
    """Układy widoku liczone na karcie: siatka zdjęcia + cecha na Z (nałożona i mozaika), przestrzeń fazowa."""
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        s0, a, b = o.steps[0], o.steps[2], o.steps[3]
        A, B, P0 = o.dane[a], o.dane[b], o.dane[s0]
        for mozaika in (False, True):
            for cecha in range(5):
                for ruch in (False, True):
                    ch.ustaw_uklad(1, cecha, mozaika)
                    cur, fr = _klatka(widok, ch, a=a, b=a, ruch=ruch)
                    xy = _siatka_ref(o.count, TOKENY, mozaika, ch.kafle())
                    if cecha < 3:
                        z = A[:, cecha] - (P0[:, cecha] if ruch else 0)
                    elif cecha == 3:
                        z = A[:, 3]
                    else:
                        z = np.linalg.norm(A[:, :3] - (P0[:, :3] if ruch else 0), axis=1)
                    np.testing.assert_array_equal(cur[:, :2], xy)
                    np.testing.assert_allclose(cur[:, 2], z, rtol=1e-6, atol=1e-6)
                    assert np.isfinite(fr).all() and (fr[4, :3] > 0).all()
        ch.ustaw_uklad(2, 0, False)                            # faza: PC1, jego prędkość na odcinku a -> b, norma
        cur, _ = _klatka(widok, ch, a=a, b=b, s=0.0)
        np.testing.assert_allclose(cur[:, 0], A[:, 0], atol=1e-6)
        np.testing.assert_allclose(cur[:, 1], (B[:, 0] - A[:, 0]) / (b - a), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(cur[:, 2], A[:, 3], atol=1e-6)
        ostatni, przed = o.steps[-1], o.steps[-2]               # ostatni zapis: prędkość z odcinka poprzedni -> ostatni
        cur, _ = _klatka(widok, ch, a=ostatni, b=ostatni)
        np.testing.assert_allclose(cur[:, 1], (o.dane[ostatni][:, 0] - o.dane[przed][:, 0]) / (ostatni - przed),
                                   rtol=1e-5, atol=1e-6)
        ch.ustaw_uklad(0, 0, False)
        cur, _ = _klatka(widok, ch, a=a, b=a)
        np.testing.assert_array_equal(cur[:, :3], A[:, :3])      # powrót do PCA: znów bitowo z zapisu
    finally:
        _ZYWE.append((widok, ch))


def _jasne(widok):
    """Liczba jasnych pikseli klatki (rysowanie wymuszone przez grabFramebuffer)."""
    img = widok.grabFramebuffer().convertToFormat(widok.grabFramebuffer().format().Format_RGB32)
    a = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(img.height(), -1, 4)[..., :3]
    return int((a.max(-1) > 90).sum())


def test_wyglad_kolory_filtry_zaznaczenie_zdjecie(app):
    """Kolory 0-6, co n-ty patch, tylko narzędzie, zaznaczenie i zdjęcie w przestrzeni rysują się bez błędów GL;
    filtry zmniejszają liczbę rysowanych punktów, a wybór punktu pomija ukryte cząsteczki."""
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        maska = np.zeros(o.count, np.uint32); maska[np.arange(o.count) % 600 < 60] = 1          # dwa górne wiersze
        ch.ustaw_dane_pomocnicze(maska, np.linspace(0, 1, len(o.steps) * o.image_count, dtype=np.float32).reshape(len(o.steps), -1))
        for kolor in range(7):
            _klatka(widok, ch, a=o.steps[1], b=o.steps[1], kolor=kolor, krycie=1.0, rozmiar=3.0)
            assert ch.blad == '' and _jasne(widok) > 0, kolor
        pelne = _jasne(widok)
        ch.ustaw_wyglad(co=5); n5 = _jasne(widok)
        ch.ustaw_wyglad(co=1, tylko_narz=True); nn = _jasne(widok)
        ch.ustaw_zaznaczenie({0, 1, 2}); ch.ustaw_wyglad(co=1, tylko_zazn=True); nz = _jasne(widok)
        assert ch.blad == '' and 0 < nz < nn < pelne and n5 < pelne
        # wybór punktu: przy filtrze „tylko zaznaczone” wynik zawsze jest jednym z zaznaczonych patchy
        widok.grabFramebuffer()
        wybrane = []
        ch.wybrano.connect(wybrane.append)
        for x, y in ((320, 240), (200, 200), (420, 300)):
            ch.zadaj_wybor(x, y, widok.width(), widok.height()); widok.grabFramebuffer(); app.processEvents()
        assert all(i < 0 or i % 600 in (0, 1, 2) for i in wybrane)
        ch.ustaw_zaznaczenie(set()); ch.ustaw_wyglad()
        # zdjęcie rozpięte na cząsteczkach jednego obrazu: powierzchnia, linie i brzeg
        rgb = (np.arange(320 * 480 * 3) % 251).astype(np.uint8).reshape(320, 480, 3)
        for tryb in (1, 2, 3):
            for kolor in (0, 1, 2):
                ch.ustaw_zdjecie(rgb, tryb, kolor, .6, 2, brzeg=True)
                _klatka(widok, ch, a=o.steps[1], b=o.steps[1], first=600, count=600, krycie=1.0)
                assert ch.blad == '', (tryb, kolor, ch.blad)
        ch.ustaw_zdjecie(None, 0)
    finally:
        _ZYWE.append((widok, ch))


def test_cecha_bez_polozenia_patcha(app):
    """Siatka zdjęcia z cechą bez położenia: Z = cecha - średnia tej cechy dla tego patcha po obrazach w kroku."""
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        s0, a = o.steps[0], o.steps[3]
        A, P0 = o.dane[a], o.dane[s0]
        sr = lambda X: np.tile(X.reshape(o.image_count, TOKENY, -1).mean(0), (o.image_count, 1))
        ch.ustaw_bez_polozenia(True)
        for cecha in range(5):
            for ruch in (False, True):
                ch.ustaw_uklad(1, cecha, False)
                cur, fr = _klatka(widok, ch, a=a, b=a, ruch=ruch)
                if cecha < 3:
                    z = (A[:, cecha] - (P0[:, cecha] if ruch else 0)) - (sr(A)[:, cecha] - (sr(P0)[:, cecha] if ruch else 0))
                elif cecha == 3:
                    z = A[:, 3] - sr(A)[:, 3]
                else:
                    z = np.linalg.norm(A[:, :3] - sr(A)[:, :3] - ((P0[:, :3] - sr(P0)[:, :3]) if ruch else 0), axis=1)
                np.testing.assert_allclose(cur[:, 2], z, rtol=1e-5, atol=1e-5)
                lo, hi = fr[1, 2], fr[2, 2]                      # oś Z obejmuje wszystkie wartości
                assert lo <= z.min() + 1e-5 and hi >= z.max() - 1e-5, (cecha, ruch, lo, hi, z.min(), z.max())
        ch.ustaw_bez_polozenia(False)
    finally:
        _ZYWE.append((widok, ch))


class OsLiniowa(OsSyntetyczna):
    """Cząsteczki o znanej dynamice liniowej s(k+1) = A s(k): mod zespolony 0,99·e^(±6°i), λ = -0,9 i λ = 0,5.
    Pary ±s0 dają średnią dokładnie 0 w każdym kroku (standaryzacja nie przesuwa widma)."""

    def __init__(self, obrazy=6, kroki=12, ziarno=1):
        rng = np.random.default_rng(ziarno)
        self.steps = list(range(0, 10 * kroki, 10))
        self.image_count, self.count = obrazy, obrazy * TOKENY
        self.metadata = {'norm_scale': 1e-6}
        self.image_names = [f'obraz_{i}.png' for i in range(obrazy)]
        th = np.radians(6.0)
        B = np.array([[.99 * np.cos(th), -.99 * np.sin(th), 0, 0], [.99 * np.sin(th), .99 * np.cos(th), 0, 0],
                      [0, 0, -.9, 0], [0, 0, 0, .5]])
        Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        self.A = Q @ B @ Q.T
        pol = rng.normal(size=(self.count // 2, 4))
        s = np.concatenate([pol, -pol])
        self.dane = {}
        for k in self.steps:
            self.dane[k] = s.astype(np.float32).copy()
            s = s @ self.A.T


def test_koopman_widmo_i_wspolrzedne(app):
    o = OsLiniowa()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        ch.ustaw_uklad(3); ch.ustaw_koopmana(d=1, xy=0, z=0, czesc=0)
        cur, fr = _klatka(widok, ch, a=o.steps[4], b=o.steps[4])
        assert ch.blad == '' and ch.koop_info and ch.koop_info.get('mody'), ch.koop_info
        _, A, vals, W, Cxx, npar, mu, sd = ch._koop_A
        oczekiwane = np.linalg.eigvals(o.A)
        for lam in oczekiwane:
            assert np.min(np.abs(vals - lam)) < 2e-3, (vals, oczekiwane)
        wyb = ch.koop_info
        x, y, jz = wyb['x'][1], wyb['y'][1], wyb['z']
        assert abs(abs(vals[x]) - .99) < 2e-3 and abs(np.degrees(abs(np.angle(vals[x]))) - 6) < .2   # niezmiennik zespolony
        assert abs(vals[jz] + .9) < 2e-3                                                            # najszybszy: λ = -0,9
        # współrzędne z karty = rzut standaryzowanego stanu na wybrane funkcje własne (numpy)
        h = (o.dane[o.steps[4]].astype(np.float64) - mu) / sd
        sk = lambda j: 1 / np.sqrt(np.real(np.conj(W[:, j]) @ Cxx @ W[:, j]) / npar)
        psi = lambda j: h @ (W[:, j] * sk(j))
        np.testing.assert_allclose(cur[:, 0], np.real(psi(x)), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(cur[:, 1], np.imag(psi(x)), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(cur[:, 2], np.real(psi(jz)), rtol=1e-3, atol=1e-3)
        # niezmiennik: moduł funkcji własnej |ψ| maleje dokładnie jak |λ|^k (0,99 na zapis)
        cur8, _ = _klatka(widok, ch, a=o.steps[8], b=o.steps[8])
        r4, r8 = np.hypot(cur[:, 0], cur[:, 1]), np.hypot(cur8[:, 0], cur8[:, 1])
        np.testing.assert_allclose(r8 / np.maximum(r4, 1e-9), .99 ** 4, rtol=2e-3, atol=0)
        for xy in range(3):                                     # pozostałe tryby osi i części Z rysują się bez błędów
            for cz in range(3):
                ch.ustaw_koopmana(d=2, xy=xy, z=xy, czesc=cz)
                cur, fr = _klatka(widok, ch, a=o.steps[5], b=o.steps[6], s=.5)
                assert ch.blad == '' and np.isfinite(cur).all() and np.isfinite(fr).all()
    finally:
        _ZYWE.append((widok, ch))


def _wygladz_ref(o, k, tryb, okno, alfa):
    """Referencja numpy wygładzania stanu w zapisie o indeksie k (te same wagi co shader)."""
    T = len(o.steps)
    if tryb == 1:
        h = (okno | 1) // 2
        q = range(max(k - h, 0), min(k + h, T - 1) + 1)
        w = {j: 1.0 for j in q}
    else:
        L = 1 if alfa >= 1 else int(min(T, np.ceil(np.log(1e-4) / np.log(1 - alfa)) + 1))
        q = range(max(k - L + 1, 0), k + 1)
        w = {j: ((1 - alfa) ** (k - j) if j == 0 else alfa * (1 - alfa) ** (k - j)) for j in q}
    acc = sum(w[j] * o.dane[o.steps[j]].astype(np.float64) for j in q)
    return acc / sum(w.values())


def test_wygladzanie_srednia_i_ema(app):
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        for tryb, okno, alfa in ((1, 3, .3), (1, 5, .3), (2, 5, .3), (2, 5, .7)):
            ch.ustaw_wygladzanie(tryb, okno, alfa)
            for k in (0, 2, 5):
                for ruch in (False, True):
                    cur, _ = _klatka(widok, ch, a=o.steps[k], b=o.steps[k], ruch=ruch)
                    ref = _wygladz_ref(o, k, tryb, okno, alfa)
                    if ruch:
                        ref = ref - _wygladz_ref(o, 0, tryb, okno, alfa)
                    np.testing.assert_allclose(cur[:, :3], ref[:, :3], rtol=1e-5, atol=2e-5)
                    if not ruch:
                        np.testing.assert_allclose(cur[:, 3], ref[:, 3], rtol=1e-5, atol=2e-5)
        ch.ustaw_wygladzanie(0)
        cur, _ = _klatka(widok, ch, a=o.steps[2], b=o.steps[2])
        np.testing.assert_array_equal(cur, o.dane[o.steps[2]])          # bez wygładzania znów bitowo z zapisu
    finally:
        _ZYWE.append((widok, ch))


def test_atencja_w_masce_nie_psuje_narzedzia(app):
    """Atencja w górnych bitach MASK: filtr „tylko narzędzie” i kolor narzędzia działają jak bez niej."""
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        maska = np.zeros(o.count, np.uint32); maska[np.arange(o.count) % 600 < 60] = 1
        ch.ustaw_dane_pomocnicze(maska, None)
        _klatka(widok, ch, a=o.steps[1], b=o.steps[1], krycie=1.0, rozmiar=3.0)
        ch.ustaw_wyglad(tylko_narz=True); bez = _jasne(widok)
        ch.ustaw_atencje(np.linspace(0, 1, o.count), 0, tryb=0)          # same dane, bez nakładania
        z_att = _jasne(widok)
        assert bez == z_att and bez > 0
        for tryb in (1, 2, 3):
            ch.ustaw_atencje(tryb=tryb, wzm=2.0, krycie=.6)
            assert _jasne(widok) > 0 and ch.blad == ''
        ch.ustaw_wyglad(); ch.wyczysc_atencje(); ch.ustaw_atencje(tryb=0)
    finally:
        _ZYWE.append((widok, ch))


def test_koopman_osie_reczne(app):
    """Osie z dowolnych składowych modów: X = Re ψa + Re ψb (suma), Y = |ψc| (moduł Re i Im), Z = Im ψc."""
    o = OsLiniowa()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        ch.ustaw_uklad(3); ch.ustaw_koopmana(d=2)
        _klatka(widok, ch, a=o.steps[3], b=o.steps[3])
        _, A, vals, W, Cxx, npar, mu, sd = ch._koop_A
        mody = ch.koop_info['mody']                                            # bez modów zdegenerowanych
        zesp = max(mody, key=lambda m: m['im'])['j']                           # mod zespolony (Im > 0)
        rz = [m['j'] for m in sorted(mody, key=lambda m: -m['modul']) if abs(m['im']) < 1e-9][:2]
        osie = dict(x=dict(skladowe=[(rz[0], 're'), (rz[1], 're')], kombinacja=0),
                    y=dict(skladowe=[(zesp, 're'), (zesp, 'im')], kombinacja=1),
                    z=dict(skladowe=[(zesp, 'im')], kombinacja=0))
        ch.ustaw_koopmana(d=2, osie=osie)
        k = 3
        cur, _ = _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        assert ch.blad == '' and ch.koop_info['osie'] == osie and not ch.koop_info['automatyczne']
        h = np.concatenate([(o.dane[o.steps[k - j]].astype(np.float64) - mu) / sd for j in range(2)], 1)
        sk = lambda j: 1 / np.sqrt(np.real(np.conj(W[:, j]) @ Cxx @ W[:, j]) / npar)
        psi = lambda j: h @ (W[:, j] * sk(j))
        np.testing.assert_allclose(cur[:, 0], np.real(psi(rz[0])) + np.real(psi(rz[1])), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(cur[:, 1], np.abs(psi(zesp)), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(cur[:, 2], np.imag(psi(zesp)), rtol=1e-3, atol=1e-3)
        ch.ustaw_koopmana(d=2, osie=dict(x=dict(skladowe=[], kombinacja=0), y=osie['y'], z=osie['z']))
        cur, _ = _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        assert np.all(cur[:, 0] == 0)                                          # pusta oś stoi w zerze
        ch.ustaw_koopmana(d=2)                                                 # powrót do automatycznych
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        assert ch.koop_info['automatyczne']
    finally:
        _ZYWE.append((widok, ch))


def test_koopman_probka_do_klastrow(app):
    """Próbka stanów Hankela z karty = numpy dla tych samych cząsteczek; dobór osi pod klastry działa na niej."""
    o = OsLiniowa()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        ch.ustaw_uklad(3); ch.ustaw_koopmana(d=2)
        k = 5
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        wynik = []
        ch.probka_koopmana.connect(wynik.append)
        ch.zadaj_probke_koopmana(m=1000)
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        assert len(wynik) == 1 and 'blad' not in wynik[0], wynik
        p = wynik[0]
        _, A, vals, W, Cxx, npar, mu, sd = ch._koop_A
        i = p['czastki']
        assert len(i) == 1000 and p['kroki'] == [o.steps[k]] and len(np.unique(i)) == 1000
        h = np.concatenate([(o.dane[o.steps[k - j]][i].astype(np.float64) - mu) / sd for j in range(2)], 1)
        np.testing.assert_allclose(p['H'], h, rtol=1e-4, atol=1e-4)
        ch.zadaj_probke_koopmana(m=800, caly_ruch=True)
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        assert len(wynik) == 2 and len(wynik[1]['kroki']) == 8 and len(wynik[1]['H']) == 800 and o.steps[k] in wynik[1]['kroki']
        import klastry_koopmana as KK
        w = KK.dobierz(p['H'], p['W'], p['war'], p['prog'], p['vals'], p['zdeg'], osie_obecne=p['osie'],
                       urzadzenie='cpu', k_max=3)
        assert set(w['osie']) == {'x', 'y', 'z'} and np.isfinite(w['ocena']['wynik'])
        ch.ustaw_koopmana(d=2, osie=w['osie'])
        cur, _ = _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        assert ch.blad == '' and np.isfinite(cur).all()

        def nasycone():                                  # piksele w barwie klastra (szary = brak modelu / outlier)
            img = widok.grabFramebuffer()
            a_ = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(img.height(), -1, 4)
            rgb = a_[..., :3].astype(int)
            return int(((rgb.max(-1) - rgb.min(-1)) > 60).sum())
        ch.ustaw_klastry(w['model'])
        cur, _ = _klatka(widok, ch, a=o.steps[k], b=o.steps[k], kolor=7, krycie=1.0, rozmiar=4.0)
        assert ch.blad == '' and nasycone() > 50
        # barwa_klastra z shadera (kernel testowy, wynik w części roboczej KG) = przydział modelu w numpy
        import ruch_gpu as RG
        from OpenGL import GL
        widok.makeCurrent()
        try:
            src = RG._NAGLOWEK + '''
layout(local_size_x = 256) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_N) return;
    vec3 c = barwa_klastra(i);
    KG[2048u + 3u * i] = c.x; KG[2048u + 3u * i + 1u] = c.y; KG[2048u + 3u * i + 2u] = c.z;
}
'''
            ch._prog['test_kl'] = ch._program(cs=src)
            ch._wiaz()
            ch._uniformy('test_kl', u_N=ch.N, **ch._wspolne())
            GL.glDispatchCompute((ch.N + 255) // 256, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            rgb = ch._czytaj_bufor('KG', np.float32, 3 * ch.N, 2048 * 4).reshape(-1, 3)
        finally:
            widok.doneCurrent()
        m = w['model']
        Z = (cur[:, :3].astype(np.float64) - m['mediana']) / m['skala']
        if m['wym'] < 3:
            Z[:, 2] = 0
        d2 = (((Z[:, None, :] - m['srodki'][None]) / m['sigma'][None]) ** 2).sum(2)
        c = (m['logpi'][None] - np.log(m['sigma']).sum(1)[None] - .5 * d2).argmax(1)
        out = ~m['duzy'][c] | (np.sqrt(d2[np.arange(len(c)), c]) > m['prog'])
        szary = np.abs(rgb - .33).max(1) < 1e-4
        assert (szary == out).mean() > .999 and (~out).sum() > 0
        barwy = {}
        for kl, kolor in zip(c[~out], np.round(rgb[~out], 4)):
            barwy.setdefault(int(kl), set()).add(tuple(kolor))
        assert all(len(v) == 1 for v in barwy.values()) and len({next(iter(v)) for v in barwy.values()}) == len(barwy)
        n1 = nasycone()
        ch.ustaw_moc_koloru(0.0)                         # moc koloru 0: same odcienie szarości
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k], kolor=7, krycie=1.0, rozmiar=4.0)
        assert ch.blad == '' and nasycone() == 0
        ch.ustaw_moc_koloru(2.0)
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k], kolor=7, krycie=1.0, rozmiar=4.0)
        assert nasycone() >= n1
        ch.ustaw_moc_koloru(1.0)
        ch.ustaw_koopmana(d=2)                           # inne osie: model nieważny, wszystko szare
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k], kolor=7, krycie=1.0, rozmiar=4.0)
        assert ch.blad == '' and nasycone() == 0
        ch.ustaw_uklad(0); ch.zadaj_probke_koopmana()
        _klatka(widok, ch, a=o.steps[k], b=o.steps[k])
        assert 'blad' in wynik[-1] and ch._kprobka is None
    finally:
        _ZYWE.append((widok, ch))


def _fizyka_ref(P, pole, k0, k1, kw, tau, prog_a, prog_b, nrm):
    """Referencja numpy właściwości ruchu (te same definicje co CS_WLASC). P: T × n × 3."""
    X = P[k0:k1 + 1]
    d = np.diff(X, axis=0)
    dw = d - pole[k0 + 1:k1 + 1, None, :]
    temp = (dw ** 2).sum(2).mean(0)
    ln = np.linalg.norm(d, axis=2)
    cs = (d[1:] * d[:-1]).sum(2) / np.maximum(ln[1:] * ln[:-1], 1e-30)
    vacf = cs.mean(0)
    msd1 = (d ** 2).sum(2).mean(0)
    msdt = ((X[tau:] - X[:-tau]) ** 2).sum(2).mean(0)
    alfa = np.log(msdt / msd1) / np.log(tau)
    prosto = np.linalg.norm(X[-1] - X[0], axis=1) / ln.sum(0)
    xm = X.mean(0)
    c = X[:-1] - xm
    kappa = -(c * d).sum((0, 2)) / (c ** 2).sum((0, 2))
    rozg = np.linalg.norm(P[kw] - P[0], axis=1)
    rez = np.where((vacf > prog_b) & (prosto > prog_b), 0, np.where(alfa >= prog_a, 1, 2))
    # widmo przyrostów (najwyżej 128 ze środka okna): okres dominujący, udział pasma [2, 4], energie
    m0 = k1 - k0; m = min(m0, 128); j0 = k0 + (m0 - m) // 2
    dd = np.diff(P[j0:j0 + m + 1], axis=0)                       # m × n × 3
    F = np.fft.fft(dd, axis=0)
    f = np.arange(1, m // 2 + 1)
    pw = (np.abs(F[f]) ** 2).sum(2) * np.where(2 * f != m, 2.0, 1.0)[:, None]
    okres = m / f
    dom = okres[pw.argmax(0)]
    pas = pw[(okres >= 2) & (okres <= 4)].sum(0) / pw.sum(0)
    ekin = .5 * msd1
    epot = .5 * np.maximum(kappa, 0) * (c ** 2).sum((0, 2)) / len(d)
    return np.stack([temp, alfa, vacf, prosto, kappa, rozg, nrm[k1], rez, dom, pas, ekin, epot], 1)


def test_fizyka_ruchu_zgodna_z_numpy(app):
    """Właściwości ruchu z karty (temperatura, α, VACF, prostoliniowość, κ, rozgrzewka, zmiana 768D, reżim) i serie
    populacji w krokach z podziałem na obszary oka = numpy; układ 4 i kolory 8-11 rysują się bez błędów."""
    o = OsSyntetyczna(obrazy=6, kroki=tuple(range(0, 120, 10)))
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        rng = np.random.default_rng(3)
        klasy = rng.integers(0, 4, o.count).astype(np.uint8)
        ch.ustaw_obszary(klasy)
        wynik = []
        ch.fizyka_gotowa.connect(wynik.append)
        ch.ustaw_uklad(4)
        ch.ustaw_fizyke(okno=0, rozgrzewka=30, tau=3, prog_a=.7, prog_b=.3, osie=(0, 6, 1))
        ch.zadaj_odczyt_fizyki()
        _klatka(widok, ch, a=o.steps[5], b=o.steps[5])
        assert ch.blad == '' and len(wynik) == 1, (ch.blad, ch.fiz_info)
        w = wynik[0]
        T = len(o.steps)
        P = np.stack([o.dane[s][:, :3].astype(np.float64) for s in o.steps])
        nrm = np.stack([o.dane[s][:, 3].astype(np.float64) for s in o.steps])
        d = np.diff(P, axis=0)
        pole = np.vstack([np.zeros((1, 3)), d.mean(1)])
        ref = _fizyka_ref(P, pole, 0, T - 1, 3, 3, .7, .3, nrm)
        got = w['wl'].astype(np.float64)
        for c, (rtol, atol) in enumerate([(1e-3, 1e-5), (1e-3, 1e-3), (1e-3, 1e-4), (1e-3, 1e-4), (1e-3, 1e-4),
                                          (1e-4, 1e-4), (1e-5, 1e-5)]):
            np.testing.assert_allclose(got[:, c], ref[:, c], rtol=rtol, atol=atol, err_msg=f'właściwość {c}')
        assert (got[:, 7] == ref[:, 7]).mean() > .999
        assert (np.abs(got[:, 8] - ref[:, 8]) < 1e-4).mean() > .995                 # okres dominujący (remisy widma)
        for c in (9, 10, 11):
            np.testing.assert_allclose(got[:, c], ref[:, c], rtol=2e-3, atol=1e-4, err_msg=f'właściwość {c}')
        # serie populacji w krokach: temperatura i cos osobno dla obszarów
        s = w['seria']
        for c in range(4):
            m = klasy == c
            t_ref = ((d[:, m] - pole[1:, None, :]) ** 2).sum(2).mean(1)
            np.testing.assert_allclose(s['temperatura'][1:, c], t_ref, rtol=1e-3)
            ln = np.linalg.norm(d[:, m], axis=2)
            cs_ref = ((d[1:, m] * d[:-1, m]).sum(2) / np.maximum(ln[1:] * ln[:-1], 1e-30)).mean(1)
            np.testing.assert_allclose(s['cos'][2:, c], cs_ref, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(s['pole'][1:], (pole[1:] ** 2).sum(1), rtol=1e-4)
        # okno ruchome wokół bieżącego kroku
        ch.ustaw_fizyke(okno=3); ch.zadaj_odczyt_fizyki()
        _klatka(widok, ch, a=o.steps[6], b=o.steps[6])
        ref = _fizyka_ref(P, pole, 3, 9, 3, 3, .7, .3, nrm)
        np.testing.assert_allclose(wynik[-1]['wl'][:, 0], ref[:, 0], rtol=1e-3, atol=1e-5)
        assert wynik[-1]['okno'] == (o.steps[3], o.steps[9])
        cur, _ = _klatka(widok, ch, a=o.steps[6], b=o.steps[6])
        np.testing.assert_allclose(cur[:, 0], np.log(np.maximum(wynik[-1]['wl'][:, 0], 1e-12)), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(cur[:, 2], wynik[-1]['wl'][:, 1], rtol=1e-5, atol=1e-5)
        # płynna animacja: w połowie między zapisami 6 i 7 = średnia właściwości okien wokół 6 i wokół 7
        ref7 = _fizyka_ref(P, pole, 4, 10, 3, 3, .7, .3, nrm)
        cur, _ = _klatka(widok, ch, a=o.steps[6], b=o.steps[7], s=.5)
        sr = .5 * (np.log(np.maximum(ref[:, 0], 1e-12)) + np.log(np.maximum(ref7[:, 0], 1e-12)))
        np.testing.assert_allclose(cur[:, 0], sr, rtol=2e-3, atol=2e-3)
        ch.ustaw_kolory_fizyki(energia=2, zak_c=0, zak=[(0, 1e9, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0)])
        for kolor in (8, 9, 10, 11, 15, 16, 17):
            _klatka(widok, ch, a=o.steps[6], b=o.steps[6], kolor=kolor)
            assert ch.blad == ''
        ch.ustaw_filtr_obszaru(4)                         # tylko narzędzie: rysuje się bez błędów
        _klatka(widok, ch, a=o.steps[6], b=o.steps[6], kolor=11)
        assert ch.blad == ''
    finally:
        _ZYWE.append((widok, ch))


def test_przyspieszenie_chwilowe(app):
    """Kolor 12: |v1 - v0| wokół zapisu najbliższego bieżącej chwili, nierówne odstępy kroków (0, 3, 7, 12, 20, 31)."""
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        P = {s: o.dane[s][:, :3].astype(np.float64) for s in o.steps}
        st = o.steps
        for kc, s, a, b in ((2, 0.0, st[2], st[3]), (3, 0.7, st[2], st[3]), (1, 0.0, st[1], st[1])):
            cur, _ = _klatka(widok, ch, a=a, b=b, s=s, kolor=12)
            v0 = (P[st[kc]] - P[st[kc - 1]]) / (st[kc] - st[kc - 1])
            v1 = (P[st[kc + 1]] - P[st[kc]]) / (st[kc + 1] - st[kc])
            np.testing.assert_allclose(cur[:, 3], np.linalg.norm(v1 - v0, axis=1), rtol=1e-4, atol=1e-6)
        cur, _ = _klatka(widok, ch, a=st[0], b=st[0], kolor=12)          # pierwszy zapis: brak v0, kolor 0
        assert np.all(cur[:, 3] == 0) and ch.blad == ''
    finally:
        _ZYWE.append((widok, ch))


def test_predkosc_chwilowa_i_wzgledna(app):
    """Kolor 13: |x(k+1) - x(k-1)| / (t(k+1) - t(k-1)); kolor 14: ta prędkość / średnia |Δx|/Δt cząsteczki po osi."""
    o = OsSyntetyczna()
    widok, ch = _scena(app, o)
    try:
        _wszystkie(widok, ch, app)
        st = o.steps
        P = np.stack([o.dane[s][:, :3].astype(np.float64) for s in st])
        sr = (np.linalg.norm(np.diff(P, axis=0), axis=2) / np.diff(st)[:, None]).mean(0)
        for kc in (1, 3, 4):
            v = np.linalg.norm(P[kc + 1] - P[kc - 1], axis=1) / (st[kc + 1] - st[kc - 1])
            cur, _ = _klatka(widok, ch, a=st[kc], b=st[kc], kolor=13)
            np.testing.assert_allclose(cur[:, 3], v, rtol=1e-4, atol=1e-6)
            cur, _ = _klatka(widok, ch, a=st[kc], b=st[kc], kolor=14)
            np.testing.assert_allclose(cur[:, 3], v / sr, rtol=1e-4, atol=1e-6)
        assert ch.blad == ''
    finally:
        _ZYWE.append((widok, ch))
