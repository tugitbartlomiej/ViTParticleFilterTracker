"""Testy modułów aplikacji bez GPU i bez pokazywania okien: common.izopowierzchnia, history (sciezka_lokalna, zbior_uczacy,
Historia), tool_labels.maski, tab.lista_liczb, ustawienia, dock_windows.bez_ukladu, settings_panel.ParameterPanel,
gpu.base._GPUElement.wyslij_porcjami. Widżety Qt tylko na QApplication z platformą offscreen (nic się nie wyświetla).

Dane rzeczywiste tylko do odczytu: historia `cache_uczenia/lasery/2cca2743e1c74b14` i manifest zbioru uczącego. Zapisy testów (pickle
manifestu) idą do katalogu tymczasowego (LD.CACHE podmienione).
Uruchomienie: cd LaserLatentSpaceApp; $env:PYTHONPATH='.'; python -m unittest testy.test_moduly -v"""
import json
import os
import pickle
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')            # przed utworzeniem QApplication
os.environ.setdefault('EDYTOR_BEZ_USTAWIEN', '1')
os.environ.setdefault('EDYTOR_BEZ_AUTOSTARTU', '1')

from laser_app.compute import tool_labels  # noqa: E402
from laser_app import paths  # noqa: E402
from laser_app.compute import history as hist  # noqa: E402
from laser_app.compute import common  # noqa: E402

HISTORIA = hist.CACHE / '2cca2743e1c74b14'


def _qapp():
    """QApplication offscreen (jedna na proces)."""
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv[:1])


# ====================================================================================================== 1. izopowierzchnia
def _elipsoid(polosie, n=41, krok=0.5, odwrocony=False):
    """Pole V = √((x/a)² + (y/b)² + (z/c)²) na siatce n³ o boku `krok`, środek w środku siatki: poziom 1 to elipsoida o półosiach
    (a, b, c); V rośnie od środka (odwrocony: V = −to, maleje od środka)."""
    o = (np.arange(n) - (n - 1) / 2) * krok
    X, Y, Z = np.meshgrid(o, o, o, indexing='ij')
    V = np.sqrt((X / polosie[0]) ** 2 + (Y / polosie[1]) ** 2 + (Z / polosie[2]) ** 2)
    srodek = np.full(3, (n - 1) / 2 * krok)
    return (-V if odwrocony else V).astype(np.float32), srodek


class _BezSkimage:
    """Wymusza ścieżkę awaryjną izopowierzchni: `import skimage.measure` kończy się ImportError."""

    def __enter__(self):
        self._p = mock.patch.dict(sys.modules, {'skimage': None, 'skimage.measure': None})
        self._p.start()
        return self

    def __exit__(self, *a):
        self._p.stop()


class TestIzopowierzchnia(unittest.TestCase):
    def test_kula_zasieg_wierzcholkow_jak_promien(self):
        """Bez scikit-image: wierzchołki kuli o promieniu 3 mieszczą się w promieniu ± jeden krok siatki, a zasięg w każdej osi = promień."""
        krok = 0.5
        V, c = _elipsoid((1, 1, 1), n=41, krok=krok)
        with _BezSkimage():
            v, f, n = common.izopowierzchnia(V, 1.0, krok)
        self.assertEqual(v.shape[1], 3)
        self.assertGreater(len(v), 50)
        r = np.linalg.norm(v - c, axis=1)
        self.assertLess(np.abs(r - 1.0).max(), krok)                    # wszystkie na sferze (w obrębie kroku siatki)
        for os_ in range(3):
            self.assertAlmostEqual(float(v[:, os_].max() - c[os_]), 1.0, delta=krok)
            self.assertAlmostEqual(float(c[os_] - v[:, os_].min()), 1.0, delta=krok)

    def test_elipsoida_zasieg_w_osiach_jak_polosie(self):
        """Bez scikit-image: elipsoida o półosiach (1, 2, 3): rozpiętość wierzchołków wzdłuż każdej osi = półoś ± krok."""
        krok = 0.5
        polosie = (1.0, 2.0, 3.0)
        V, c = _elipsoid(polosie, n=41, krok=krok)
        with _BezSkimage():
            v, f, n = common.izopowierzchnia(V, 1.0, krok, 'descent')
        for os_, a in enumerate(polosie):
            self.assertAlmostEqual(float(v[:, os_].max() - c[os_]), a, delta=krok)
            self.assertAlmostEqual(float(c[os_] - v[:, os_].min()), a, delta=krok)

    def test_skala_kroku_siatki(self):
        """Wierzchołki są w jednostkach `krok` (od rogu siatki): ta sama indeksowa siatka z krokiem 2 daje współrzędne 2× większe."""
        V, _ = _elipsoid((1, 1.5, 2), n=33, krok=1.0)
        with _BezSkimage():
            v1, f1, _ = common.izopowierzchnia(V, 1.0, 1.0)
            v2, f2, _ = common.izopowierzchnia(V, 1.0, 2.0)
        np.testing.assert_allclose(v2, 2 * v1, atol=1e-5)
        np.testing.assert_array_equal(f1, f2)

    def test_normalne_descent_do_wnetrza_ascent_na_zewnatrz(self):
        """V rośnie od środka: 'descent' = normalne w stronę malejącego V (do środka), 'ascent' = na zewnątrz; normalne jednostkowe."""
        krok = 0.5
        V, c = _elipsoid((1.5, 1.0, 2.0), n=45, krok=krok)
        with _BezSkimage():
            v, f, nd = common.izopowierzchnia(V, 1.0, krok, 'descent')
            _, _, na = common.izopowierzchnia(V, 1.0, krok, 'ascent')
        np.testing.assert_allclose(np.linalg.norm(nd, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(na, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(nd, -na, atol=1e-6)
        # normalna elipsoidy w punkcie p to gradient (x/a², y/b², z/c²); 'ascent' zgodna z nim, 'descent' przeciwna
        p = v - c
        gr = p / np.array([1.5, 1.0, 2.0]) ** 2
        gr /= np.linalg.norm(gr, axis=1, keepdims=True)
        zgodnosc = (na * gr).sum(1)
        self.assertGreater(float(np.median(zgodnosc)), 0.9)
        self.assertGreater(float((zgodnosc > 0).mean()), 0.98)
        self.assertLess(float(((nd * p).sum(1) / np.linalg.norm(p, axis=1)).max()), 0)   # każda do wnętrza

    def test_normalne_odwrocone_pole(self):
        """V maleje od środka (V = −odległość): 'descent' wskazuje teraz na zewnątrz, 'ascent' do środka."""
        krok = 0.5
        V, c = _elipsoid((1, 1, 1), n=41, krok=krok, odwrocony=True)
        with _BezSkimage():
            v, _, nd = common.izopowierzchnia(V, -1.0, krok, 'descent')
            _, _, na = common.izopowierzchnia(V, -1.0, krok, 'ascent')
        radialnie = ((v - c) * nd).sum(1)
        self.assertGreater(float((radialnie > 0).mean()), 0.98)
        self.assertGreater(float((((v - c) * na).sum(1) < 0).mean()), 0.98)

    def test_indeksy_trojkatow_w_zakresie(self):
        """Trójkąty: dtype całkowity, indeksy w 0..len(wierzchołków)−1, kształt (m, 3), brak zdegenerowanych (3 różne indeksy)."""
        V, _ = _elipsoid((1, 2, 3), n=41, krok=0.5)
        with _BezSkimage():
            v, f, n = common.izopowierzchnia(V, 1.0, 0.5)
        self.assertEqual(f.ndim, 2); self.assertEqual(f.shape[1], 3)
        self.assertTrue(np.issubdtype(f.dtype, np.integer))
        self.assertGreaterEqual(int(f.min()), 0)
        self.assertLess(int(f.max()), len(v))
        self.assertEqual(len(n), len(v))
        self.assertTrue((f[:, 0] != f[:, 1]).all() and (f[:, 1] != f[:, 2]).all() and (f[:, 0] != f[:, 2]).all())
        self.assertTrue(np.isfinite(v).all() and np.isfinite(n).all())

    def test_poziom_poza_zakresem_danych_ValueError(self):
        """Poziom ponad maksimum i poniżej minimum pola: ValueError (brak powierzchni), a nie pusta siatka ani inny wyjątek."""
        V, _ = _elipsoid((1, 1, 1), n=21, krok=0.5)
        with _BezSkimage():
            for poziom in (float(V.max()) + 1, float(V.min()) - 1, 1e9):
                with self.assertRaises(ValueError, msg=f'poziom {poziom}'):
                    common.izopowierzchnia(V, poziom, 0.5)

    def test_pole_float64_i_niesasiadujace_w_pamieci(self):
        """Pole float64 i widok niecią­gły (transpozycja) dają tę samą powierzchnię co kopia ciągła."""
        V, _ = _elipsoid((1, 2, 1.5), n=31, krok=0.5)
        Vt = np.asfortranarray(V.astype(np.float64))
        with _BezSkimage():
            v1, f1, n1 = common.izopowierzchnia(V, 1.0, 0.5)
            v2, f2, n2 = common.izopowierzchnia(Vt, 1.0, 0.5)
        np.testing.assert_allclose(v1, v2, atol=1e-5)
        np.testing.assert_array_equal(f1, f2)

    def test_porownanie_ze_skimage_gdy_zainstalowany(self):
        """Gdy scikit-image jest zainstalowany: obie ścieżki dają ten sam kształt (zasięgi osi i orientacja normalnych)."""
        try:
            import skimage.measure  # noqa: F401
        except ImportError:
            self.skipTest('brak scikit-image: porównanie obu ścieżek pominięte')
        krok = 0.5
        V, c = _elipsoid((1, 2, 3), n=41, krok=krok)
        vs, fs, ns = common.izopowierzchnia(V, 1.0, krok, 'descent')
        with _BezSkimage():
            vf, ff, nf = common.izopowierzchnia(V, 1.0, krok, 'descent')
        for os_ in range(3):
            self.assertAlmostEqual(float(vs[:, os_].max()), float(vf[:, os_].max()), delta=krok)
            self.assertAlmostEqual(float(vs[:, os_].min()), float(vf[:, os_].min()), delta=krok)
        self.assertGreater(float(((vs - c) * ns).sum(1).mean()) * float(((vf - c) * nf).sum(1).mean()), 0)   # ten sam znak


# ====================================================================================================== 2. sciezka_lokalna
class TestSciezkaLokalna(unittest.TestCase):
    def test_istniejaca_sciezka_bez_zmian(self):
        """Ścieżka, która istnieje, wraca bez zmian (nawet z 'SmallViTAblation' w częściach)."""
        for p in (paths.APP_DIR, hist.ROOT, hist.ROOT / 'assets', Path(tempfile.gettempdir())):
            self.assertEqual(hist.sciezka_lokalna(p), Path(p))
            self.assertEqual(hist.sciezka_lokalna(str(p)), Path(p))

    def test_inny_dysk_mapowany_do_ROOT(self):
        """Nieistniejące X:\\...\\SmallViTAblation\\assets\\<istniejący podfolder> → LD.ROOT/assets/<podfolder>."""
        podfoldery = [d.name for d in (hist.ROOT / 'assets').iterdir() if d.is_dir()] if (hist.ROOT / 'assets').is_dir() else []
        if not podfoldery:
            self.skipTest('brak ROOT/assets z podfolderami')
        nazwa = podfoldery[0]
        obca = Path(r'X:\Nieistniejacy\Katalog\SmallViTAblation\assets') / nazwa
        self.assertFalse(obca.exists())
        self.assertEqual(hist.sciezka_lokalna(obca), hist.ROOT / 'assets' / nazwa)
        self.assertEqual(hist.sciezka_lokalna(str(obca)), hist.ROOT / 'assets' / nazwa)        # także jako napis
        glebiej = obca / 'cokolwiek_czego_nie_ma'
        self.assertEqual(hist.sciezka_lokalna(glebiej), glebiej)                              # brak celu w ROOT: bez zmian

    def test_inny_dysk_eksperymenty(self):
        """To samo dla podfolderu 'experiments' (istniejące wagi / kod w ROOT)."""
        e = hist.ROOT / 'experiments'
        if not e.is_dir() or not any(e.iterdir()):
            self.skipTest('brak ROOT/experiments')
        nazwa = next(iter(sorted(d.name for d in e.iterdir())))
        self.assertEqual(hist.sciezka_lokalna(rf'Y:\gdzies\SmallViTAblation\experiments\{nazwa}'), e / nazwa)

    def test_asset_foo_nieistniejacy_zostaje(self):
        """X:\\...\\SmallViTAblation\\assets\\foo, gdy w ROOT też nie ma 'foo': ścieżka bez zmian (nie zgadujemy)."""
        p = Path(r'X:\a\SmallViTAblation\assets\foo_nie_istnieje_123')
        self.assertEqual(hist.sciezka_lokalna(p), p)

    def test_poza_SmallViTAblation_bez_zmian(self):
        """Nieistniejąca ścieżka bez 'SmallViTAblation' wśród części wraca bez zmian."""
        for s in (r'X:\Dane\assets\foo', r'Z:\x\y.txt', 'relatywna/sciezka/foo.npy'):
            self.assertEqual(hist.sciezka_lokalna(s), Path(s))

    def test_niedostepny_dysk_nie_rzuca(self):
        """Ścieżka na nieistniejącym / niedostępnym dysku nie rzuca wyjątku (także gdy Path.exists zgłasza OSError)."""
        for s in (r'Q:\SmallViTAblation\assets\foo', r'\\serwer_nie_istnieje\udzial\SmallViTAblation\assets\a',
                  r'Z:\bez_znaczenia'):
            try:
                hist.sciezka_lokalna(s)
            except Exception as e:                                     # noqa: BLE001
                self.fail(f'{s}: wyjątek {e!r}')
        orig = Path.exists

        def exists(self, *a, **k):
            if self.drive.upper() == 'F:':
                raise PermissionError('brak uprawnień')                    # dysk F: bez dostępu
            return orig(self, *a, **k)
        with mock.patch.object(Path, 'exists', exists):
            p = hist.sciezka_lokalna(r'F:\x\SmallViTAblation\assets\foo')
            self.assertIsInstance(p, Path)
            q = hist.sciezka_lokalna(r'F:\x\SmallViTAblation\assets')                             # po błędzie szuka w ROOT
        self.assertIsInstance(q, Path)


# ====================================================================================================== 3. zbior_uczacy
@unittest.skipUnless(hist.MANIFEST_UCZACY.exists(), 'brak manifestu zbioru uczącego')
class TestZbiorUczacy(unittest.TestCase):
    KLUCZE = {'nr', 'film', 'klatka', 'member', 'archiwum', 'sha', 'wh', 'ramka', 'kategoria'}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / 'lasery'
        self._p = mock.patch.object(hist, 'CACHE', self.cache)
        self._p.start()
        self._stary = hist._uczacy
        hist._uczacy = None

    def tearDown(self):
        hist._uczacy = self._stary
        self._p.stop()
        self._tmp.cleanup()

    def _pkl(self):
        return sorted(self.cache.glob('_manifest_35000_*.pkl')) if self.cache.exists() else []

    def test_pierwsze_wywolanie_parsuje_i_zapisuje_pickle(self):
        """Pierwsze wywołanie czyta manifest (jsonl) i zapisuje _manifest_35000_<rozmiar>_<mtime>.pkl w LD.CACHE (tu: katalog tymczasowy)."""
        self.assertEqual(self._pkl(), [])
        wiersze = hist.zbior_uczacy()
        st = hist.MANIFEST_UCZACY.stat()
        self.assertEqual([p.name for p in self._pkl()], [f'_manifest_35000_{st.st_size}_{int(st.st_mtime)}.pkl'])
        n_linii = sum(1 for _ in open(hist.MANIFEST_UCZACY, encoding='utf-8'))
        self.assertEqual(len(wiersze), n_linii)
        self.assertEqual([r['nr'] for r in wiersze[:5]], [0, 1, 2, 3, 4])
        self.assertEqual(wiersze[-1]['nr'], n_linii - 1)
        self.assertIs(hist.zbior_uczacy(), wiersze)                      # drugie wywołanie w tym samym procesie: ten sam obiekt

    def test_drugie_wywolanie_z_pickle_bez_parsowania(self):
        """Po wyzerowaniu LD._uczacy wiersze wracają z pickle (json.loads nie jest wołane) i są identyczne."""
        pierwsze = hist.zbior_uczacy()
        hist._uczacy = None
        with mock.patch.object(hist.json, 'loads', side_effect=AssertionError('parsowanie manifestu mimo pickle')):
            drugie = hist.zbior_uczacy()
        self.assertEqual(len(drugie), len(pierwsze))
        self.assertEqual(drugie[:50], pierwsze[:50])
        self.assertEqual(drugie[-50:], pierwsze[-50:])
        self.assertEqual(drugie, pierwsze)

    def test_uszkodzony_pickle_ignorowany_i_nadpisany(self):
        """Uszkodzone pickle (śmieci, pusty plik, ucięty) jest ignorowane: wiersze liczone od nowa, a plik zapisany poprawnie."""
        wzorzec = hist.zbior_uczacy()
        plik = self._pkl()[0]
        dobry = plik.read_bytes()
        for zly in (b'to nie jest pickle', b'', dobry[:len(dobry) // 2], dobry[:10]):
            plik.write_bytes(zly)
            hist._uczacy = None
            wiersze = hist.zbior_uczacy()
            self.assertEqual(len(wiersze), len(wzorzec))
            self.assertEqual(wiersze[:20], wzorzec[:20])
            self.assertEqual(pickle.loads(plik.read_bytes()), wzorzec)   # przepisany poprawnie
            self.assertEqual(self._pkl(), [plik])

    def test_inny_manifest_inny_plik_cache(self):
        """Nazwa pliku cache zawiera rozmiar i mtime manifestu: zmiana mtime → nowy plik, stary nie jest używany."""
        hist.zbior_uczacy()
        stary = self._pkl()[0]
        stary.write_bytes(pickle.dumps([dict(nr=0, nieprawda=True)]))     # „stary” zapis z innym rozmiarem/mtime nie może zostać użyty
        st = hist.MANIFEST_UCZACY.stat()
        manifest = hist.MANIFEST_UCZACY

        class Przesuniety(type(manifest)):                           # ten sam plik, ale mtime o 100 s późniejszy
            def stat(self, *a, **k):
                o = super().stat(*a, **k)
                return mock.Mock(st_size=o.st_size, st_mtime=o.st_mtime + 100)
        hist._uczacy = None
        with mock.patch.object(hist, 'MANIFEST_UCZACY', Przesuniety(manifest)):
            wiersze = hist.zbior_uczacy()
        self.assertEqual(len(self._pkl()), 2)
        self.assertGreater(len(wiersze), 1000)
        self.assertNotIn('nieprawda', wiersze[0])

    def test_pola_wierszy(self):
        """Wiersze mają klucze nr, film, klatka, member, archiwum (sama nazwa, bez katalogu i bez \\), sha, wh, ramka (4 liczby w 480×320), kategoria."""
        wiersze = hist.zbior_uczacy()
        krok = max(1, len(wiersze) // 1500)
        for r in wiersze[::krok] + wiersze[-5:]:
            self.assertEqual(set(r), self.KLUCZE, r)
            self.assertIsInstance(r['nr'], int)
            self.assertIsInstance(r['klatka'], int)
            self.assertTrue(r['film'].startswith('TR_'), r['film'])
            self.assertNotIn('\\', r['archiwum']); self.assertNotIn('/', r['archiwum'])
            self.assertTrue(r['archiwum'])
            self.assertEqual(len(r['sha']), 64); int(r['sha'], 16)
            self.assertEqual(len(r['wh']), 2); self.assertTrue(all(isinstance(v, int) and v > 0 for v in r['wh']))
            self.assertEqual(len(r['ramka']), 4)
            x0, y0, x1, y1 = r['ramka']
            self.assertTrue(all(isinstance(v, float) for v in r['ramka']))
            self.assertTrue(0 <= x0 < x1 <= 480 + 1e-3, r['ramka'])
            self.assertTrue(0 <= y0 < y1 <= 320 + 1e-3, r['ramka'])
            self.assertTrue(r['kategoria'])

    def test_ramka_skalowana_osobno_w_x_i_y(self):
        """Ramka = bbox_xywh z manifestu przeskalowana osobno w X (480 / width) i Y (320 / height); sprawdzone na pierwszej linii."""
        with open(hist.MANIFEST_UCZACY, encoding='utf-8') as f:
            r = json.loads(f.readline())
        x, y, bw, bh = r['gt'][0]['bbox_xywh']
        sx, sy = 480 / r['width'], 320 / r['height']
        oczekiwana = [x * sx, y * sy, (x + bw) * sx, (y + bh) * sy]
        np.testing.assert_allclose(hist.zbior_uczacy()[0]['ramka'], oczekiwana, rtol=1e-9)
        self.assertEqual(hist.zbior_uczacy()[0]['member'], r['member'])

    def test_czy_w_zbiorze_uczacym(self):
        """Plik nazwany jak TR_xxxx_Sx_Pxx_klatka_nnnnnn.jpg z wiersza manifestu → ten wiersz; obca nazwa → None."""
        r = hist.zbior_uczacy()[123]
        nazwa = f"{r['film']}_klatka_{r['klatka']:06d}.jpg"
        znal = hist.czy_w_zbiorze_uczacym(Path('gdzies') / nazwa)
        self.assertIsNotNone(znal)
        self.assertEqual((znal['film'], znal['klatka']), (r['film'], r['klatka']))
        self.assertIsNone(hist.czy_w_zbiorze_uczacym('zdjecie.jpg'))


# ====================================================================================================== 4. Historia
def _stan_plikow(katalog):
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in Path(katalog).iterdir() if p.is_file()}


@unittest.skipUnless((HISTORIA / 'dane.npz').exists() and (HISTORIA / 'Z.npy').exists(), 'brak historii w cache_uczenia')
class TestHistoria(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.przed = _stan_plikow(HISTORIA)
        cls.h = hist.History(HISTORIA)
        cls.narz0 = cls.h.narz.copy()
        cls.mies0 = cls.h.mieszane.copy()

    @classmethod
    def tearDownClass(cls):
        assert _stan_plikow(HISTORIA) == cls.przed, 'Historia zmieniła pliki w cache (miał być tylko odczyt)'

    def tearDown(self):
        self.h.ustaw_etykiete('dotkniete')                 # tryb domyślny z konstruktora

    def test_Z_jest_mmap(self):
        """Z to mmap (nie wczytane w całości): float16 (K, 600, 768)."""
        h = self.h
        self.assertIsInstance(h.Z, np.memmap)
        self.assertEqual(h.Z.shape, (h.K, 600, 768))
        self.assertEqual(h.Z.dtype, np.float16)
        self.assertEqual(h.K, len(h.kroki))

    def test_wagi_zmapowane_na_istniejacy_katalog(self):
        """Cache policzony na dysku F: → h.wagi wskazuje istniejący katalog w bieżącym ROOT (a nie nieistniejące F:\\...)."""
        h = self.h
        self.assertIsInstance(h.wagi, str)
        self.assertTrue(Path(h.wagi).is_dir(), h.wagi)
        surowe = str(np.load(HISTORIA / 'dane.npz')['wagi'])
        self.assertEqual(Path(h.wagi), hist.sciezka_lokalna(surowe))
        if not Path(surowe).exists():                               # inny komputer / dysk: zmapowane do ROOT
            self.assertTrue(Path(h.wagi).is_relative_to(hist.ROOT))

    def test_tryby_etykiety_spojne_liczby(self):
        """ustaw_etykiete dla każdego trybu z TRYBY: opis tekstowy, ≥ 1 łata narzędzia, dotknięte ≥ środek i ≥ pokrycie(0,5), środek ⊆ dotknięte."""
        h = self.h
        liczby = {}
        for tryb, _ in tool_labels.TRYBY:
            opis = h.ustaw_etykiete(tryb)
            self.assertIsInstance(opis, str); self.assertEqual(opis, h.etykieta)
            self.assertGreaterEqual(int(h.narz.sum()), 1, tryb)
            self.assertEqual(h.narz.dtype, bool); self.assertEqual(h.narz.shape, (600,))
            self.assertEqual(h.mieszane.shape, (600,))
            self.assertFalse((h.narz & h.mieszane).any(), f'{tryb}: łata jednocześnie narzędzie i mieszana')
            liczby[tryb] = (int(h.narz.sum()), h.tryb_etykiety)
        self.assertEqual(liczby['dotkniete'][1], 'dotkniete')
        if liczby['srodek'][1] == 'srodek':
            self.assertGreaterEqual(liczby['dotkniete'][0], liczby['srodek'][0])
        if liczby['pokrycie'][1] == 'pokrycie':
            self.assertGreaterEqual(liczby['dotkniete'][0], liczby['pokrycie'][0])
        self.assertEqual(liczby['ramka'][0], int(h.narz_ramka.sum()))
        if h.udzial is not None and h.srodek is not None:
            self.assertFalse((h.srodek & ~(h.udzial > 0)).any(), 'środek łaty w obrysie, a udział piksel. = 0')

    def test_pokrycie_maleje_z_progiem(self):
        """Tryb 'pokrycie': im większy próg, tym mniej (albo tyle samo) łat narzędzia; łaty wyższego progu ⊆ niższego."""
        h = self.h
        poprzednia, p_prog = None, None
        for prog in (0.05, 0.25, 0.5, 0.75):
            h.ustaw_etykiete('pokrycie', prog)
            if h.tryb_etykiety != 'pokrycie':                       # próg za wysoki: powrót do ramki GT, dalej nie porównujemy
                break
            self.assertAlmostEqual(h.prog_etykiety, prog)
            if poprzednia is not None:
                self.assertTrue((h.narz <= poprzednia).all(), f'prog {prog} ⊄ prog {p_prog}')
            poprzednia, p_prog = h.narz.copy(), prog
        self.assertIsNotNone(poprzednia)

    def test_fok_narz_lok_i_nowe_tryby(self):
        """fok to łata narzędzia (należy do h.narz) w każdym trybie, narz_lok ma kształt (K,) i jest skończone, dodatnie."""
        h = self.h
        for tryb, _ in tool_labels.TRYBY:
            h.ustaw_etykiete(tryb)
            self.assertTrue(0 <= h.fok < 600)
            self.assertTrue(h.narz[h.fok], f'{tryb}: fok={h.fok} nie jest łatą narzędzia')
            self.assertEqual(h.narz_lok.shape, (h.K,))
            self.assertTrue(np.isfinite(h.narz_lok).all())
            self.assertTrue((h.narz_lok > 0).all())
            n = h.narz
            np.testing.assert_allclose(h.narz_lok, h.lok[:, n].mean(1) / np.maximum(h.lok[:, ~n].mean(1), 1e-9), rtol=1e-5)

    def test_nieznany_tryb_wraca_do_ramki(self):
        """Nieznany tryb nie rzuca: etykieta z ramki GT (opis o 0 łat), tryb_etykiety 'ramka', narz = narz_ramka."""
        h = self.h
        opis = h.ustaw_etykiete('bzdura')
        self.assertIn('ramka GT', opis)
        self.assertEqual(h.tryb_etykiety, 'ramka')
        np.testing.assert_array_equal(h.narz, h.narz_ramka)
        self.assertFalse(h.mieszane.any())

    def test_obszary_klasy_i_pierwszenstwo_narzedzia(self):
        """obszary(): uint8 (600,) w {0,1,2,3}; narzędzie (3) ma pierwszeństwo, potem źrenica (2), potem rogówka (1)."""
        h = self.h
        if h.oko is None:
            self.skipTest('historia bez danych oka')
        for tryb, _ in tool_labels.TRYBY:
            h.ustaw_etykiete(tryb)
            k = h.obszary()
            self.assertEqual(k.dtype, np.uint8); self.assertEqual(k.shape, (600,))
            self.assertTrue(set(np.unique(k)) <= {0, 1, 2, 3})
            np.testing.assert_array_equal(k == 3, h.narz)
            rog, zrz = h.oko['rogowka'] >= 0.5, h.oko['zrenica'] >= 0.5
            np.testing.assert_array_equal(k[~h.narz & zrz], 2)
            np.testing.assert_array_equal(k[~h.narz & rog & ~zrz], 1)
            np.testing.assert_array_equal(k[~h.narz & ~rog & ~zrz], 0)
        k_wysoki = h.obszary(prog_oka=1.01)                         # próg ponad maksimum udziału: zostaje samo narzędzie
        self.assertTrue(set(np.unique(k_wysoki)) <= {0, 3})
        k_zero = h.obszary(prog_oka=0.0)                            # każda łata ma udział ≥ 0 → oko wszędzie poza narzędziem
        self.assertEqual(int((k_zero == 0).sum()), 0)

    def test_obszary_bez_danych_oka(self):
        """Bez danych oka (oko=None) obszary() zwraca tylko 0 i 3."""
        h = self.h
        stare = h.oko
        try:
            h.oko = None
            k = h.obszary()
            self.assertTrue(set(np.unique(k)) <= {0, 3})
            np.testing.assert_array_equal(k == 3, h.narz)
        finally:
            h.oko = stare

    def test_powrot_trybu_przywraca_oryginalne_narz(self):
        """Po przejściu przez wszystkie tryby i powrocie do 'dotkniete' narz i mieszane są takie jak po konstruktorze."""
        h = self.h
        for tryb, _ in tool_labels.TRYBY:
            h.ustaw_etykiete(tryb)
        h.ustaw_etykiete('pokrycie', 0.9)
        h.ustaw_etykiete('dotkniete')
        np.testing.assert_array_equal(h.narz, self.narz0)
        np.testing.assert_array_equal(h.mieszane, self.mies0)
        self.assertEqual(h.tryb_etykiety, 'dotkniete')

    def test_indeks_krok_interp(self):
        """indeks(krok) i krok(t) są wzajemnie odwrotne na węzłach i interpolują liniowo między nimi; interp skrajne t przycina."""
        h = self.h
        for i in (0, 1, h.K // 2, h.K - 1):
            self.assertAlmostEqual(h.indeks(h.kroki[i]), float(i), places=6)
            self.assertAlmostEqual(h.krok(float(i)), float(h.kroki[i]), places=6)
        srodek = (h.kroki[3] + h.kroki[4]) / 2
        self.assertAlmostEqual(h.indeks(srodek), 3.5, places=6)
        X = np.arange(h.K, dtype=np.float64)
        self.assertAlmostEqual(float(h.interp(X, 2.25)), 2.25)
        self.assertAlmostEqual(float(h.interp(X, -5.0)), 0.0)
        self.assertAlmostEqual(float(h.interp(X, h.K + 5.0)), float(h.K - 1))


# ====================================================================================================== 5. maski (skrajne przypadki)
class TestMaskiSkrajne(unittest.TestCase):
    def test_same_zera(self):
        """Udział 0 wszędzie: żadnych łat narzędzia ani mieszanych w żadnym trybie obrysu."""
        u = np.zeros(600, np.float32)
        for tryb in ('pokrycie', 'dotkniete'):
            n, m = tool_labels.maski(u, tryb=tryb)
            self.assertFalse(n.any(), tryb); self.assertFalse(m.any(), tryb)
            self.assertEqual(n.shape, (600,)); self.assertEqual(n.dtype, bool); self.assertEqual(m.dtype, bool)
        n, m = tool_labels.maski(u, tryb='srodek', srodek=np.zeros(600, bool))
        self.assertFalse(n.any()); self.assertFalse(m.any())

    def test_same_jedynki(self):
        """Udział 1 wszędzie: wszystkie łaty to narzędzie, żadnych mieszanych (także przy progu 1)."""
        u = np.ones(600, np.float32)
        for tryb, prog in (('pokrycie', 0.5), ('pokrycie', 1.0), ('dotkniete', 0.5)):
            n, m = tool_labels.maski(u, prog=prog, tryb=tryb)
            self.assertTrue(n.all(), (tryb, prog)); self.assertFalse(m.any(), (tryb, prog))
        n, m = tool_labels.maski(u, tryb='srodek', srodek=np.ones(600, bool))
        self.assertTrue(n.all()); self.assertFalse(m.any())

    def test_progi_dokladnie_na_progu(self):
        """Udział równy progowi liczy się do narzędzia (≥), tuż pod progiem do mieszanych; dla liczb niedokładnych w float32 też."""
        for prog in (0.5, 0.25, 0.3, 0.1, 0.7, 1.0):
            u = np.array([0.0, np.float32(prog) - np.float32(1e-3) if prog < 1 else 0.999, np.float32(prog)], np.float32)
            n, m = tool_labels.maski(u, prog=prog, tryb='pokrycie')
            self.assertEqual(list(n), [False, False, True], f'prog={prog}')
            self.assertEqual(list(m), [False, True, False], f'prog={prog}')

    def test_prog_zero_i_dotkniete_na_zerze(self):
        """'dotkniete' to ściśle > 0 (udział 0 nie jest dotknięty, najmniejszy dodatni jest); prog=0 w 'pokrycie' daje wszystkie łaty."""
        u = np.array([0, np.finfo(np.float32).tiny, 1e-6, 0.4], np.float32)
        n, _ = tool_labels.maski(u, tryb='dotkniete')
        self.assertEqual(list(n), [False, True, True, True])
        n0, m0 = tool_labels.maski(u, prog=0.0, tryb='pokrycie')
        self.assertTrue(n0.all()); self.assertFalse(m0.any())

    def test_nieznany_tryb_ValueError(self):
        """Nieznany tryb → ValueError z nazwą trybu w komunikacie."""
        with self.assertRaises(ValueError) as c:
            tool_labels.maski(np.zeros(600), tryb='cos_innego')
        self.assertIn('cos_innego', str(c.exception))
        with self.assertRaises(ValueError):
            tool_labels.maski(np.zeros(600), tryb='')

    def test_brak_srodka_i_ramki_ValueError(self):
        """Tryby 'srodek' i 'ramka' bez swoich masek → ValueError; z maskami nie."""
        u = np.full(600, 0.3, np.float32)
        with self.assertRaises(ValueError):
            tool_labels.maski(u, tryb='srodek')
        with self.assertRaises(ValueError):
            tool_labels.maski(u, tryb='srodek', ramka=np.ones(600, bool))
        with self.assertRaises(ValueError):
            tool_labels.maski(u, tryb='ramka')
        with self.assertRaises(ValueError):
            tool_labels.maski(u, tryb='ramka', srodek=np.ones(600, bool))
        tool_labels.maski(u, tryb='srodek', srodek=np.ones(600, bool))
        tool_labels.maski(u, tryb='ramka', ramka=np.ones(600, bool))

    def test_wyniki_to_kopie_nie_widoki_wejscia(self):
        """Wynik 'srodek' i 'ramka' nie dzieli pamięci z wejściem: zmiana wyniku nie psuje masek Historii."""
        u = np.full(600, 0.3, np.float32)
        s = np.zeros(600, bool); s[:10] = True
        r = np.zeros(600, bool); r[5:20] = True
        n1, _ = tool_labels.maski(u, tryb='srodek', srodek=s); n1[:] = True
        n2, _ = tool_labels.maski(u, tryb='ramka', ramka=r); n2[:] = True
        self.assertEqual(int(s.sum()), 10)
        self.assertEqual(int(r.sum()), 15)

    def test_srodek_mieszane_i_ramka_mieszane_puste(self):
        """'srodek': mieszane = dotknięte poza środkiem; 'ramka': mieszane puste niezależnie od udziału."""
        u = np.zeros(600, np.float32); u[[1, 2, 3]] = 0.2
        s = np.zeros(600, bool); s[[2, 4]] = True              # środek w łacie 4 bez udziału: narzędzie, ale nie mieszana
        n, m = tool_labels.maski(u, tryb='srodek', srodek=s)
        self.assertEqual(list(np.flatnonzero(n)), [2, 4])
        self.assertEqual(list(np.flatnonzero(m)), [1, 3])
        n, m = tool_labels.maski(u, tryb='ramka', ramka=s)
        self.assertEqual(list(np.flatnonzero(n)), [2, 4])
        self.assertFalse(m.any())


# ====================================================================================================== 6. lista_liczb
class TestListaLiczb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _qapp()
        from laser_app.ui import widgets
        cls.f = staticmethod(widgets.lista_liczb)

    def test_przecinki_spacje_srednik(self):
        """Separatory: przecinek, spacja, średnik, ich mieszanki; kolejność wpisania zostaje."""
        self.assertEqual(self.f('12, 40 100', 768), [12, 40, 100])
        self.assertEqual(self.f('3;1;2', 10), [3, 1, 2])
        self.assertEqual(self.f('  5 ,,  7;; 9  ', 10), [5, 7, 9])
        self.assertEqual(self.f('', 10), [])
        self.assertEqual(self.f('   ', 10), [])

    def test_zakresy(self):
        """'a-b' to a..b włącznie; zakres odwrócony 'b-a' też; zakres jednoelementowy."""
        self.assertEqual(self.f('100-105', 768), [100, 101, 102, 103, 104, 105])
        self.assertEqual(self.f('5-3', 10), [3, 4, 5])
        self.assertEqual(self.f('4-4', 10), [4])
        self.assertEqual(self.f('1, 3-5, 9', 10), [1, 3, 4, 5, 9])

    def test_poza_zakresem_odrzucone(self):
        """Tylko 0..n−1: wartości ujemne w zakresie, równe n i większe odpadają; zakres obcinany na brzegu."""
        self.assertEqual(self.f('0 9 10 11 768', 10), [0, 9])
        self.assertEqual(self.f('7-15', 10), [7, 8, 9])
        self.assertEqual(self.f('-3', 10), [])                       # brak liczby przed '-' → śmieć
        self.assertEqual(self.f('10', 10), [])
        self.assertEqual(self.f('5', 0), [])

    def test_duplikaty(self):
        """Powtórzenia (także przez nakładające się zakresy) pojawiają się raz, na pierwszej pozycji."""
        self.assertEqual(self.f('3 3 3', 10), [3])
        self.assertEqual(self.f('1-4 3-6 2', 10), [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.f('5, 1, 5, 1', 10), [5, 1])

    def test_smieci(self):
        """Nierozpoznane tokeny są pomijane, reszta zostaje; nie rzuca na dziwnych wejściach."""
        self.assertEqual(self.f('abc 5 x-y 7 1.5 2-', 10), [5, 7, 2])
        self.assertEqual(self.f('1-2-3 4', 10), [4])
        self.assertEqual(self.f('--- ,,, ;;;', 10), [])
        self.assertEqual(self.f('١٢ 3', 10)[-1], 3)                   # cyfry spoza ASCII nie psują reszty

    def test_ogromny_zakres_nie_zawiesza(self):
        """Zakres '0-100000000' przy n=10 nie może zajmować sekund (pole tekstowe w GUI): wynik [0..9] szybko."""
        t0 = time.perf_counter()
        w = self.f('0-100000000', 10)
        dt = time.perf_counter() - t0
        self.assertEqual(w, list(range(10)))
        self.assertLess(dt, 0.5, f'lista_liczb przeglądała cały zakres: {dt:.1f} s')


# ====================================================================================================== 7. ustawienia
def _drzewo_kontrolek():
    """Minimalne drzewo jak w aplikacji: korzeń z listą, polem liczbowym, checkboxem, zakładką (klasa z modułu z MODULY) i słownikiem."""
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QPushButton, QSpinBox, QWidget

    class Zakladka(QWidget):
        pass
    Zakladka.__module__ = 'laser_app.lasers_tab.tab'                        # `settings._kontener` patrzy na moduł klasy

    class Obca(QWidget):
        pass                                                           # moduł testowy: nie jest kontenerem, jej kontrolek nie widać

    class Korzen:
        pass

    k = Korzen()
    k.combo = QComboBox(); k.combo.addItems(['A', 'B', 'C'])
    k.spin = QSpinBox(); k.spin.setRange(0, 100); k.spin.setValue(7)
    k.dspin = QDoubleSpinBox(); k.dspin.setRange(0, 10); k.dspin.setDecimals(3); k.dspin.setValue(1.5)
    k.check = QCheckBox(); k.check.setChecked(True)
    k.przycisk = QPushButton()                                         # bez przycisków
    k.okno = QSpinBox(); k.suwak = QSpinBox(); k.pasek = QSpinBox()    # POMIN
    k._prywatny = QSpinBox()
    z = Zakladka()
    z.combo2 = QComboBox(); z.combo2.addItems(['x', 'y', 'z']); z.combo2.setCurrentIndex(2)
    z.checki = {'a': QCheckBox(), 'b': QCheckBox()}; z.checki['b'].setChecked(True)
    k.zakladka_x = z
    k.obca = Obca(); k.obca.ukryty = QSpinBox()
    return k


class TestUstawienia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _qapp()
        from laser_app import settings
        cls.US = settings

    def setUp(self):
        self.k = _drzewo_kontrolek()

    def test_kontrolki_i_stan(self):
        """kontrolki znajduje listy, pola, checkboxy także w zakładce i słowniku; pomija przyciski, POMIN, prywatne i obce kontenery."""
        ks = self.US.kontrolki(self.k)
        self.assertEqual(set(ks), {'combo', 'spin', 'dspin', 'check', 'zakladka_x.combo2', 'zakladka_x.checki.a', 'zakladka_x.checki.b'})
        s = self.US.stan(self.k)
        self.assertEqual(s['combo'], dict(indeks=0, tekst='A'))
        self.assertEqual(s['spin'], 7)
        self.assertAlmostEqual(s['dspin'], 1.5)
        self.assertIs(s['check'], True)
        self.assertEqual(s['zakladka_x.combo2'], dict(indeks=2, tekst='z'))
        self.assertEqual((s['zakladka_x.checki.a'], s['zakladka_x.checki.b']), (False, True))
        json.dumps(s)                                                  # stan musi się dać zapisać do JSON

    def test_ustaw_combo_po_tekscie_potem_po_indeksie(self):
        """Lista: najpierw dopasowanie po tekście (ważniejszy od indeksu), potem indeks; brak obu → pominięte."""
        ok, pom = self.US.ustaw(self.k, {'combo': dict(indeks=0, tekst='C')})
        self.assertEqual((ok, pom), (1, 0))
        self.assertEqual(self.k.combo.currentIndex(), 2)               # tekst 'C' wygrał z indeksem 0
        ok, pom = self.US.ustaw(self.k, {'combo': dict(indeks=1, tekst='nie ma takiego')})
        self.assertEqual((ok, pom), (1, 0))
        self.assertEqual(self.k.combo.currentText(), 'B')              # indeks jako zapas
        ok, pom = self.US.ustaw(self.k, {'combo': dict(indeks=9, tekst='nie ma')})
        self.assertEqual((ok, pom), (0, 1))
        self.assertEqual(self.k.combo.currentText(), 'B')              # bez zmian
        ok, pom = self.US.ustaw(self.k, {'combo': dict(indeks=-1, tekst='')})
        self.assertEqual((ok, pom), (0, 1))

    def test_ustaw_spinbox_checkbox(self):
        """Pola liczbowe (int i double) i checkboxy ustawiane z wartości; zmiana tylko, gdy inna."""
        ok, pom = self.US.ustaw(self.k, {'spin': 42, 'dspin': 2.25, 'check': False, 'zakladka_x.checki.a': True})
        self.assertEqual((ok, pom), (4, 0))
        self.assertEqual(self.k.spin.value(), 42)
        self.assertAlmostEqual(self.k.dspin.value(), 2.25)
        self.assertFalse(self.k.check.isChecked())
        self.assertTrue(self.k.zakladka_x.checki['a'].isChecked())

    def test_nieznane_klucze_pominiete(self):
        """Klucze bez kontrolki liczone jako pominięte (też w zagnieżdżonych ścieżkach); poprawne w tym samym słowniku się stosują."""
        ok, pom = self.US.ustaw(self.k, {'nie.ma': 1, 'inny': True, 'zakladka_x.brak': 3, 'obca.ukryty': 5, 'spin': 11})
        self.assertEqual((ok, pom), (1, 4))
        self.assertEqual(self.k.spin.value(), 11)

    def test_zla_wartosc_liczona_jako_pominieta(self):
        """Wartość nieobsługiwanego typu dla kontrolki (np. tekst w spinboksie) nie przewraca ustaw, tylko zwiększa liczbę pominiętych."""
        ok, pom = self.US.ustaw(self.k, {'spin': 'abc', 'combo': 5})
        self.assertEqual(ok, 0)
        self.assertEqual(pom, 2)

    def test_stan_ustaw_roundtrip_w_pamieci(self):
        """ustaw(stan(x)) na zmienionych kontrolkach przywraca zapisany stan."""
        s = self.US.stan(self.k)
        self.k.combo.setCurrentIndex(2); self.k.spin.setValue(0); self.k.dspin.setValue(9.0); self.k.check.setChecked(False)
        self.k.zakladka_x.combo2.setCurrentIndex(0); self.k.zakladka_x.checki['b'].setChecked(False)
        ok, pom = self.US.ustaw(self.k, s)
        self.assertEqual((ok, pom), (len(s), 0))
        self.assertEqual(self.US.stan(self.k), s)

    def test_oczekujace_ustawienia_dopisywane_do_stanu(self):
        """Ustawienia zakładek jeszcze niezbudowanych (`_ustawienia_oczekujace`) trafiają do stanu, żeby plik ich nie gubił."""
        self.k._ustawienia_oczekujace = {'zakladka_leniwa.spin': 5}
        s = self.US.stan(self.k)
        self.assertEqual(s['zakladka_leniwa.spin'], 5)
        self.k._leniwe = {'zakladka_leniwa': object()}
        ok, pom = self.US.ustaw(self.k, {'zakladka_leniwa.spin': 8})
        self.assertEqual((ok, pom), (1, 0))                            # zakładka jeszcze nie istnieje: czeka
        self.assertEqual(self.k._ustawienia_oczekujace['zakladka_leniwa.spin'], 8)

    def test_uklad_okienek_stan_i_ustaw(self):
        """Zakładka z uklad_okienek / ustaw_uklad_okienek: układ trafia do stanu i wraca przez ustaw (zliczony jako ustawiony)."""
        class Dok:
            wartosc = {'doki': [1, 2, 3]}

            def uklad_okienek(self):
                return dict(self.wartosc)

            def ustaw_uklad_okienek(self, v):
                self.wartosc = v
        self.k.zakladka_y = Dok()
        s = self.US.stan(self.k)
        self.assertEqual(s['zakladka_y.uklad_okienek'], {'doki': [1, 2, 3]})
        ok, pom = self.US.ustaw(self.k, {'zakladka_y.uklad_okienek': {'doki': [9]}})
        self.assertEqual((ok, pom), (1, 0))
        self.assertEqual(self.k.zakladka_y.wartosc, {'doki': [9]})

    def test_zapisz_wczytaj_roundtrip_do_pliku(self):
        """zapisz → plik JSON (wersja, zapisano, ustawienia) → wczytaj na zmienionych kontrolkach przywraca stan."""
        with tempfile.TemporaryDirectory() as d:
            plik = Path(d) / 'ust.json'
            n = self.US.zapisz(self.k, plik)
            s0 = self.US.stan(self.k)
            self.assertEqual(n, len(s0))
            dane = json.loads(plik.read_text(encoding='utf-8'))
            self.assertEqual(dane['wersja'], self.US.WERSJA)
            self.assertEqual(dane['ustawienia'], json.loads(json.dumps(s0)))
            self.k.combo.setCurrentIndex(1); self.k.spin.setValue(99); self.k.check.setChecked(False)
            self.k.zakladka_x.checki['b'].setChecked(False)
            ok, pom = self.US.wczytaj(self.k, plik)
            self.assertEqual((ok, pom), (n, 0))
            self.assertEqual(self.US.stan(self.k), s0)

    def test_wczytaj_plik_bez_ustawien(self):
        """Plik JSON bez klucza 'ustawienia' to zero ustawionych i zero pominiętych; nie rzuca."""
        with tempfile.TemporaryDirectory() as d:
            plik = Path(d) / 'puste.json'
            plik.write_text('{"wersja": 1}', encoding='utf-8')
            self.assertEqual(self.US.wczytaj(self.k, plik), (0, 0))

    def test_plik_domyslny_nie_jest_ruszany(self):
        """zapisz/wczytaj z jawnym plikiem nie dotykają settings.json obok aplikacji."""
        przed = self.US.PLIK.stat().st_mtime_ns if self.US.PLIK.exists() else None
        with tempfile.TemporaryDirectory() as d:
            self.US.zapisz(self.k, Path(d) / 'a.json')
        po = self.US.PLIK.stat().st_mtime_ns if self.US.PLIK.exists() else None
        self.assertEqual(przed, po)


# ====================================================================================================== 8. dock_windows.bez_ukladu
class _Licznik:
    """Filtr zdarzeń zliczający LayoutRequest dla obserwowanego widżetu."""

    def __new__(cls, typ):
        from PyQt6.QtCore import QEvent, QObject

        class F(QObject):
            def __init__(self):
                super().__init__()
                self.n = 0

            def eventFilter(self, obj, e):
                if e.type() == QEvent.Type.LayoutRequest:
                    self.n += 1
                return False
        return F()


class TestOkna(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _qapp()
        from laser_app.ui import dock_windows
        cls.okna = dock_windows

    def _przetworz(self):
        from PyQt6.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, 0)
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, int(QEvent.Type.LayoutRequest.value))
        self.app.processEvents()

    def test_stala_wysokosc_osłony(self):
        """Osłona ma stałą wysokość (min = max = podana albo z napisu), politykę Ignored × Fixed i sizeHint z tą wysokością."""
        from PyQt6.QtWidgets import QLabel, QSizePolicy
        lb = QLabel('krótki')
        o = self.okna.bez_ukladu(lb, 23)
        self.assertEqual(o.minimumHeight(), 23); self.assertEqual(o.maximumHeight(), 23)
        self.assertEqual(o.height(), 23)
        self.assertEqual(o.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Ignored)
        self.assertEqual(o.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Fixed)
        self.assertEqual(o.sizeHint().height(), 23)
        self.assertEqual(o.minimumSizeHint().height(), 23)
        self.assertIs(lb.parentWidget(), o)
        o2 = self.okna.bez_ukladu(QLabel('x'))                           # bez wysokości: z sizeHint napisu
        self.assertGreater(o2.height(), 5)
        self.assertEqual(o2.minimumHeight(), o2.maximumHeight())

    def test_sizeHint_stały_przy_zmianie_napisu(self):
        """sizeHint (i minimumSizeHint) osłony nie zależy od długości napisu."""
        from PyQt6.QtWidgets import QLabel
        lb = QLabel('x')
        o = self.okna.bez_ukladu(lb, 20)
        h0, m0 = o.sizeHint(), o.minimumSizeHint()
        for tekst in ('x' * 5000, 'bardzo długi napis ' * 100, '', 'a\nb\nc\nd'):
            lb.setText(tekst)
            self.assertEqual(o.sizeHint(), h0, repr(tekst[:20]))
            self.assertEqual(o.minimumSizeHint(), m0)
            self.assertEqual(o.height(), 20)

    def test_napis_wypelnia_oslone_po_zmianie_rozmiaru(self):
        """resizeEvent osłony ustawia geometrię napisu na cały jej prostokąt."""
        from PyQt6.QtCore import QSize
        from PyQt6.QtGui import QResizeEvent
        from PyQt6.QtWidgets import QLabel
        lb = QLabel('abc')
        o = self.okna.bez_ukladu(lb, 18)
        o.resize(333, 18)
        o.resizeEvent(QResizeEvent(QSize(333, 18), QSize(100, 18)))     # okno nie jest pokazane: zdarzenie wprost
        self.assertEqual(lb.geometry(), o.rect())

    def test_setText_nie_wysyla_LayoutRequest_do_rodzica(self):
        """Zmiana napisu w osłonie nie zgłasza do widżetu-rodzica (z QVBoxLayout) żadnego LayoutRequest; z zwykłym QLabel jest odwrotnie."""
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

        def zlicz(zwykly):
            rodzic = QWidget(); lay = QVBoxLayout(rodzic)
            lb = QLabel('x')
            lay.addWidget(lb if zwykly else self.okna.bez_ukladu(lb, 18))
            f = _Licznik('LayoutRequest')
            rodzic.installEventFilter(f)
            rodzic.show()                                            # offscreen: bez widocznego okna, ale układ rodzica działa jak w aplikacji
            self._przetworz()
            f.n = 0
            for i in range(30):
                lb.setText('napis o zmiennej długości ' * (1 + (i * 7) % 40))
            self._przetworz()
            rodzic.close()
            return f.n
        self.assertGreater(zlicz(True), 0, 'kontrola: zwykły QLabel powinien zgłaszać LayoutRequest (test nieczuły)')
        self.assertEqual(zlicz(False), 0)

    def test_zmiana_napisu_nie_zmienia_sizeHint_rodzica(self):
        """Rodzic z osłoną ma stały sizeHint układu niezależnie od długości napisu."""
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
        rodzic = QWidget(); lay = QVBoxLayout(rodzic)
        lb = QLabel('x')
        lay.addWidget(self.okna.bez_ukladu(lb, 18))
        h0 = lay.sizeHint()
        lb.setText('b' * 3000)
        self._przetworz()
        self.assertEqual(lay.sizeHint(), h0)


# ====================================================================================================== 9. ParameterPanel
class TestPanelParametrow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _qapp()
        from laser_app.ui import settings_panel
        cls.PU = settings_panel

    def _strona(self, tekst='x'):
        from PyQt6.QtWidgets import QLabel
        return [QLabel(tekst)]

    def test_sizeHint_stały_niezależnie_od_stron(self):
        """sizeHint / minimumSizeHint panelu są stałe: nie rosną z zawartością stron ani ze zmianą napisów."""
        from PyQt6.QtWidgets import QLabel, QPushButton
        p = self.PU.ParameterPanel()
        h0, m0 = p.sizeHint(), p.minimumSizeHint()
        self.assertEqual((h0.width(), h0.height()), (440, 600))
        lb = QLabel('x')
        p.dodaj_strone('a', 'Strona A', [lb])
        self.assertEqual(p.sizeHint(), h0)
        p.dodaj_strone('b', 'Strona B', [QPushButton('duży') for _ in range(40)] + [QLabel('szeroki ' * 400)])
        self.assertEqual(p.sizeHint(), h0); self.assertEqual(p.minimumSizeHint(), m0)
        for i in range(5):
            lb.setText('inny napis ' * (1 + i * 50))
            self.assertEqual(p.sizeHint(), h0)

    def test_dodaj_strone_pokaz_biezaca(self):
        """Pusty panel: biezaca() = None; pierwsza dodana strona staje się bieżąca; pokaz(klucz) przełącza; nieznany klucz nic nie zmienia."""
        p = self.PU.ParameterPanel()
        self.assertIsNone(p.biezaca())
        v = p.dodaj_strone('a', 'A', self._strona('1'))
        self.assertEqual(p.biezaca(), 'a')                             # pierwsza strona wybrana sama
        self.assertEqual(p.strony.count(), 1); self.assertEqual(p.lista.count(), 1)
        self.assertEqual(p.lista.item(0).text(), 'A')
        self.assertEqual(p.lista.item(0).toolTip(), 'A')               # bez dymka: nazwa
        p.dodaj_strone('b', 'B', self._strona('2'), dymek='opis B')
        p.dodaj_strone('c', 'C', self._strona('3'))
        self.assertEqual(p.biezaca(), 'a')                             # dalsze strony nie zmieniają bieżącej
        self.assertEqual(p.lista.item(1).toolTip(), 'opis B')
        p.pokaz('c')
        self.assertEqual(p.biezaca(), 'c')
        self.assertEqual(p.strony.currentIndex(), 2)
        p.pokaz('nie ma')
        self.assertEqual(p.biezaca(), 'c')
        p.pokaz('b')
        self.assertEqual((p.biezaca(), p.strony.currentIndex()), ('b', 1))
        self.assertTrue(hasattr(v, 'addWidget'))                       # zwraca układ strony (do dopisywania)

    def test_sygnal_zmiana_strony(self):
        """Wybór strony (pokaz albo klik w listę) emituje zmiana_strony z kluczem; pierwsza strona też."""
        p = self.PU.ParameterPanel()
        odebrane = []
        p.zmiana_strony.connect(odebrane.append)
        p.dodaj_strone('a', 'A', self._strona())
        p.dodaj_strone('b', 'B', self._strona())
        p.pokaz('b'); p.pokaz('a'); p.pokaz('a')                        # powtórne pokaz tej samej: bez sygnału
        self.assertEqual(odebrane, ['a', 'b', 'a'])

    def test_widgety_strony_w_ukladzie_pionowym(self):
        """Widżety strony trafiają do jednego pionowego stosu w kolejności podania; zwrócony układ pozwala dopisać kolejne."""
        from PyQt6.QtWidgets import QLabel
        p = self.PU.ParameterPanel()
        a, b = QLabel('a'), QLabel('b')
        v = p.dodaj_strone('k', 'K', [a, b])
        self.assertIs(v.itemAt(0).widget(), a); self.assertIs(v.itemAt(1).widget(), b)
        c = QLabel('c')
        v.insertWidget(2, c)
        self.assertIs(v.itemAt(2).widget(), c)


# ====================================================================================================== 10. wyslij_porcjami
class TestWyslijPorcjami(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _qapp()
        try:
            from laser_app.gpu import base as gpu_base
        except ImportError as e:                                       # noqa: BLE001 (brak PyOpenGL)
            raise unittest.SkipTest(f'brak gpu.base: {e}')
        cls.LG = gpu_base

    def setUp(self):
        self._watki_przed = set(threading.enumerate())
        # _GPUElement dziedziczy po QObject: instancja bez __init__ nie pozwala nawet na getattr (sip), więc metody (zwykłe funkcje)
        # wołamy na zastępniku z samymi polami, których dotykają (`__dict__` na _duze oraz update)
        gpu_base = self.LG

        class Zastepca:
            def update(self):
                pass
        Zastepca.wyslij_porcjami = gpu_base._GPUElement.wyslij_porcjami
        Zastepca.duzy_gotowy = gpu_base._GPUElement.duzy_gotowy
        self.el = Zastepca()

    def _drenuj(self, st, timeout=10):
        """Wszystkie porcje z kolejki stanu aż do nbytes bajtów; zwraca skonkatenowane bajty i liczbę porcji."""
        kawalki, razem, t0 = [], 0, time.monotonic()
        while razem < st['nbytes']:
            self.assertLess(time.monotonic() - t0, timeout, 'wątek czytający nie dostarczył wszystkich porcji')
            try:
                k = st['kolejka'].get(timeout=0.5)
            except queue.Empty:
                continue
            kawalki.append(k); razem += k.nbytes
        return (np.concatenate(kawalki) if kawalki else np.zeros(0, np.uint8)), len(kawalki)

    def _watki(self, nazwa='ZH'):
        """Wątki wysyłki uruchomione w tym teście (bez wątków okien z wcześniejszych testów w tym samym procesie)."""
        przed = getattr(self, '_watki_przed', set())
        return [t for t in threading.enumerate() if t.name == f'wysylka_{nazwa}' and t.is_alive() and t not in przed]

    def test_bajty_z_kolejki_rowne_zrodlu(self):
        """Porcje z kolejki sklejone w całość dają dokładnie bajty źródła (float16 (50, 7, 8)); stan ma poprawne nbytes i flagi."""
        zr = np.random.default_rng(0).standard_normal((50, 7, 8)).astype(np.float16)
        self.el.wyslij_porcjami('ZH', zr, 16)
        st = self.el._duze['ZH']
        self.assertEqual(st['nbytes'], zr.nbytes)
        self.assertEqual((st['wyslane'], st['przydzielony'], st['gotowy'], st['stop']), (0, False, False, False))
        bajty, n = self._drenuj(st)
        self.assertEqual(n, 4)                                          # 16 + 16 + 16 + 2 wiersze
        self.assertEqual(bajty.dtype, np.uint8)
        self.assertEqual(bajty.tobytes(), zr.tobytes())
        self.assertFalse(self.el.duzy_gotowy('ZH'))                    # gotowy ustawia dopiero rysowanie (GL)
        self.assertFalse(self.el.duzy_gotowy('inna'))

    def test_wiele_porcji_wiecej_niz_kolejka(self):
        """Gdy porcji jest więcej niż miejsc w kolejce (maxsize 4), wątek czeka na odbiór i nic nie ginie ani nie zmienia kolejności."""
        zr = np.arange(50 * 7 * 8, dtype=np.float32).astype(np.float16).reshape(50, 7, 8)
        self.el.wyslij_porcjami('ZH', zr, 3)                            # 17 porcji
        st = self.el._duze['ZH']
        time.sleep(0.3)
        self.assertLessEqual(st['kolejka'].qsize(), 4)
        bajty, n = self._drenuj(st)
        self.assertEqual(n, 17)
        self.assertEqual(bajty.tobytes(), zr.tobytes())

    def test_zrodlo_mmap_i_nieciagle(self):
        """Źródło jako mmap z pliku .npy i jako widok nieciągły (transpozycja) daje poprawne bajty (w kolejności wierszy źródła)."""
        zr = np.random.default_rng(1).standard_normal((20, 5, 6)).astype(np.float16)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:     # Windows: mmap blokuje plik do zamknięcia
            p = Path(d) / 'z.npy'
            np.save(p, zr)
            mm = np.load(p, mmap_mode='r')
            self.el.wyslij_porcjami('M', mm, 7)
            bajty, _ = self._drenuj(self.el._duze['M'])
            self.assertEqual(bajty.tobytes(), zr.tobytes())
            t0 = time.monotonic()
            while self._watki('M') and time.monotonic() - t0 < 5:
                time.sleep(0.05)
            del mm
            import gc; gc.collect()
        nc = np.ascontiguousarray(zr.transpose(0, 2, 1)).transpose(0, 2, 1)   # widok: kolejność osi 1 i 2 zamieniona w pamięci
        self.assertFalse(nc.flags['C_CONTIGUOUS'])
        self.el.wyslij_porcjami('N', nc, 6)
        bajty, _ = self._drenuj(self.el._duze['N'])
        self.assertEqual(bajty.tobytes(), np.ascontiguousarray(nc).tobytes())

    def test_drugie_wywolanie_przerywa_pierwsze(self):
        """Drugie wywołanie dla tej samej nazwy ustawia stop=True w starym stanie, podmienia stan, a stary wątek się kończy."""
        zr = np.random.default_rng(2).standard_normal((50, 7, 8)).astype(np.float16)
        self.el.wyslij_porcjami('ZH', zr, 1)                            # 50 porcji, wątek zablokuje się na pełnej kolejce
        st1 = self.el._duze['ZH']
        time.sleep(0.2)
        self.assertEqual(len(self._watki()), 1)
        zr2 = (zr + 1).astype(np.float16)
        self.el.wyslij_porcjami('ZH', zr2, 16)
        st2 = self.el._duze['ZH']
        self.assertTrue(st1['stop'])
        self.assertFalse(st2['stop'])
        self.assertIsNot(st1, st2)
        t0 = time.monotonic()
        while len(self._watki()) > 1 and time.monotonic() - t0 < 5:     # stary wątek wychodzi po najwyżej ~0,2 s
            time.sleep(0.05)
        self.assertLessEqual(len(self._watki()), 1, 'wątek pierwszej wysyłki nie zakończył się po stop=True')   # nowy mógł już skończyć
        bajty, _ = self._drenuj(st2)
        self.assertEqual(bajty.tobytes(), zr2.tobytes())                # dane z nowego źródła
        t0 = time.monotonic()
        while self._watki() and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        self.assertEqual(self._watki(), [])                             # po wysłaniu całości wątek kończy pracę

    def test_osobne_bufory_niezalezne(self):
        """Wysyłki pod różnymi nazwami nie przerywają się nawzajem."""
        a = np.ones((10, 4), np.float16); b = np.full((10, 4), 2, np.float16)
        self.el.wyslij_porcjami('A', a, 4)
        self.el.wyslij_porcjami('B', b, 4)
        self.assertFalse(self.el._duze['A']['stop'])
        ba, _ = self._drenuj(self.el._duze['A']); bb, _ = self._drenuj(self.el._duze['B'])
        self.assertEqual(ba.tobytes(), a.tobytes()); self.assertEqual(bb.tobytes(), b.tobytes())

    def test_stop_zatrzymuje_czytanie(self):
        """Ręczne stop=True w stanie kończy wątek bez dalszego wypełniania kolejki."""
        zr = np.zeros((200, 4), np.float16)
        self.el.wyslij_porcjami('Q', zr, 1)
        st = self.el._duze['Q']
        st['stop'] = True
        t0 = time.monotonic()
        while self._watki('Q') and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        self.assertEqual(self._watki('Q'), [])
        self.assertLessEqual(st['kolejka'].qsize(), 4)


class TestJakobianPulapki(unittest.TestCase):
    """focal_trap._jakobian (analityczny, Newton pułapki) = autograd tej samej funkcji f(u) = PV (GELU(W1 LN(xj + P3ᵀu) + b1) − z̄)."""

    def test_zgodny_z_autogradem(self):
        import torch
        import torch.nn.functional as Fn
        from laser_app.compute import focal_trap
        g = torch.Generator().manual_seed(0)
        d, n = 192, 768
        r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
        xj, g_, be_, W1, b1, PV = r(d) * 2 + 0.3, 1 + 0.1 * r(d), 0.1 * r(d), r(n, d) / d ** 0.5, 0.1 * r(n), r(3, n)
        P3 = torch.linalg.qr(r(d, 3))[0].T.contiguous()
        eps = 1e-6

        def f(u):
            x = xj + u @ P3
            m = x.mean(); v = ((x - m) ** 2).mean()
            return PV @ Fn.gelu(((x - m) / torch.sqrt(v + eps) * g_ + be_) @ W1.T + b1)

        for u in (torch.zeros(3, dtype=torch.float64), 0.7 * r(3)):
            J = focal_trap._jakobian(xj, u, P3, g_, be_, eps, W1, b1, PV)
            Ja = torch.autograd.functional.jacobian(f, u)
            self.assertEqual(tuple(J.shape), (3, 3))
            self.assertLess(float((J - Ja).abs().max() / Ja.abs().max()), 1e-10)


class TestPulapkiWsadowo(unittest.TestCase):
    """focal_trap.rozwiaz3 (odwracanie 3 × 3 na GPU), _jakobian_wsadowo = _jakobian dla każdej cząsteczki, trap_map.charakter."""

    def test_rozwiaz3_jak_solve_i_osobliwe(self):
        import torch
        from laser_app.compute import focal_trap
        g = torch.Generator().manual_seed(1)
        J = torch.randn(50, 3, 3, generator=g, dtype=torch.float64); f = torch.randn(50, 3, generator=g, dtype=torch.float64)
        x = focal_trap.rozwiaz3(J, f)
        self.assertLess(float((x - torch.linalg.solve(J, f)).abs().max()), 1e-9)
        J[0] = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]], dtype=torch.float64)   # osobliwa: pseudoodwrotność
        x = focal_trap.rozwiaz3(J, f)
        self.assertTrue(bool(torch.isfinite(x).all()))
        self.assertLess(float((x[0] - torch.linalg.pinv(J[0]) @ f[0]).abs().max()), 1e-9)

    def test_jakobian_wsadowo_zgodny_z_pojedynczym(self):
        import torch
        from laser_app.compute import focal_trap
        g = torch.Generator().manual_seed(2)
        d, n = 192, 768
        r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
        g_, be_, W1, b1, PV = 1 + 0.1 * r(d), 0.1 * r(d), r(n, d) / d ** 0.5, 0.1 * r(n), r(3, n)
        P3 = torch.linalg.qr(r(d, 3))[0].T.contiguous()
        X = r(7, d) * 2
        Jw = focal_trap._jakobian_wsadowo(X, P3, g_, be_, 1e-6, W1, b1, PV)
        for b in range(7):
            Jb = focal_trap._jakobian(X[b], torch.zeros(3, dtype=torch.float64), P3, g_, be_, 1e-6, W1, b1, PV)
            self.assertLess(float((Jw[b] - Jb).abs().max()), 1e-12)

    def test_charakter_pulapki(self):
        from laser_app.compute import trap_map
        lam = np.array([[[-1, -2, -3], [-1, 2, 3], [1, 2, 3], [-1 + 2j, -1 - 2j, -3]]], np.complex64)
        typ, wir = trap_map.charakter(lam, np.ones((1, 4), bool))
        self.assertEqual(typ[0].tolist(), [1, 2, 3, 1])                          # ściąga, siodło, wypycha, ściąga (spirala)
        self.assertEqual(wir[0].tolist(), [False, False, False, True])
        typ, _ = trap_map.charakter(lam, np.array([[True, False, True, True]]))
        self.assertEqual(typ[0].tolist(), [1, 0, 3, 1])                          # bez zera w komórce: 0


class TestEksportObliczen(unittest.TestCase):
    """compute/exports.py i disk_cache.py: lista obliczeń, eksport do zip, import do innej historii o tym samym kluczu przenośnym,
    kontrola SHA256, brak historii, pamięć dyskowa wyników."""

    def setUp(self):
        from laser_app.compute import history
        self.tmp = Path(tempfile.mkdtemp())
        self.kroki = np.arange(0, 50, 10)
        self.rgb = (np.arange(320 * 480 * 3) % 251).astype(np.uint8).reshape(320, 480, 3)
        self.zrodlo = self._historia(self.tmp / 'cacheA', 'aaaa', r'F:\x\MalyViT_s47\wagi')
        (self.zrodlo / 'ogniskowa_t226_G9_z100.npz').write_bytes(b'pole' * 100)
        (self.zrodlo / 'ogniskowa_t226_G9_z100.json').write_text('{}', encoding='utf-8')
        (self.zrodlo / 'pulapka_podkrokow_t012_G5.npz').write_bytes(b'pomiar' * 50)
        from laser_app.compute import disk_cache
        disk_cache.zapisz(disk_cache.plik(self.zrodlo, 'dolina', t=226, baza='sfera', G=40, z=0.6, k=10), dict(Z=np.ones(3), baza='sfera', i=1))
        self.history = history

    def _historia(self, cache, klucz, wagi):
        kat = Path(cache) / klucz
        kat.mkdir(parents=True)
        np.savez(kat / 'dane.npz', kroki=self.kroki, rgb=self.rgb)
        (kat / 'Z.npy').write_bytes(b'Z' * 1000)
        (kat / 'opis.json').write_text(json.dumps(dict(wagi=wagi, kroki=len(self.kroki), okno=300, gt=[1.0, 2.0, 3.0, 4.0], wersja=3)),
                                       encoding='utf-8')
        return kat

    def test_lista_grupuje_obliczenia(self):
        from laser_app.compute import exports
        w = {(x['rodzaj'], x['tok']) for x in exports.lista(self.zrodlo)}
        self.assertTrue({('historia', None), ('pulapka', 226), ('pomiar', 12), ('pamiec', 226)} <= w, w)
        pul = next(x for x in exports.lista(self.zrodlo) if x['rodzaj'] == 'pulapka')
        self.assertEqual(sorted(pul['pliki']), ['ogniskowa_t226_G9_z100.json', 'ogniskowa_t226_G9_z100.npz'])

    def test_eksport_import_do_historii_z_innym_dyskiem(self):
        """Obliczenia cząsteczki 226 z historii na F: trafiają do tej samej historii policzonej na W: (inny klucz, ten sam przenośny)."""
        from laser_app.compute import exports
        cel_hist = self._historia(self.tmp / 'cacheB', 'bbbb', r'W:\y\MalyViT_s47\wagi')
        wpisy = [x for x in exports.lista(self.zrodlo) if x['tok'] == 226]
        z = exports.eksportuj(self.zrodlo, wpisy, self.tmp / 'e.zip')
        kat, zapisane, pominiete, bledy = exports.importuj(z, self.tmp / 'cacheB')
        self.assertEqual(kat, cel_hist)
        self.assertEqual(bledy, [])
        self.assertEqual((cel_hist / 'ogniskowa_t226_G9_z100.npz').read_bytes(), b'pole' * 100)
        kat, zapisane2, pominiete2, _ = exports.importuj(z, self.tmp / 'cacheB')       # drugi raz: nic nie nadpisuje
        self.assertEqual(zapisane2, []); self.assertEqual(sorted(pominiete2), sorted(zapisane))

    def test_inny_bieg_uczenia_to_inna_historia(self):
        """Te same kroki i zdjęcie, ale inne wagi (inny bieg): klucz przenośny inny, import nie trafia do cudzej historii."""
        from laser_app.compute import exports
        self._historia(self.tmp / 'cacheC', 'cccc', r'W:\y\MalyViT_s99\wagi')
        z = exports.eksportuj(self.zrodlo, [x for x in exports.lista(self.zrodlo) if x['tok'] == 226], self.tmp / 'e.zip')
        kat, zapisane, _, bledy = exports.importuj(z, self.tmp / 'cacheC')
        self.assertEqual(zapisane, [])
        self.assertTrue(bledy and 'brak historii' in bledy[0], bledy)

    def test_uszkodzony_plik_wykryty(self):
        import zipfile
        from laser_app.compute import exports
        cel_hist = self._historia(self.tmp / 'cacheD', 'aaaa', r'F:\x\MalyViT_s47\wagi')
        z = exports.eksportuj(self.zrodlo, [x for x in exports.lista(self.zrodlo) if x['rodzaj'] == 'pulapka'], self.tmp / 'e.zip')
        z2 = self.tmp / 'zly.zip'
        with zipfile.ZipFile(z) as a, zipfile.ZipFile(z2, 'w') as b:
            for n in a.namelist():
                b.writestr(n, b'zepsute' if n.endswith('.npz') else a.read(n))
        _, zapisane, _, bledy = exports.importuj(z2, self.tmp / 'cacheD')
        self.assertTrue(any('SHA256' in b_ for b_ in bledy), bledy)
        self.assertFalse((cel_hist / 'ogniskowa_t226_G9_z100.npz').exists())

    def test_pamiec_dyskowa(self):
        from laser_app.compute import disk_cache
        p = disk_cache.plik(self.zrodlo, 'pole_podkrokow', t=5, G=17, z=1.0, kc=np.ones((3, 3), np.float32), k=120)
        self.assertTrue(disk_cache.zapisz(p, dict(pole=np.arange(6.0).reshape(2, 3), G=17, opis='x')))
        d = disk_cache.wczytaj(p)
        self.assertEqual(d['G'], 17); self.assertEqual(d['opis'], 'x'); self.assertEqual(d['pole'].shape, (2, 3))
        z = dict(lista=[np.ones(2), np.ones(3)], prog=dict(b=1.5, db=np.float32(2)), krotka=(1, 'a'), brak=None)   # jak mechanika
        self.assertTrue(disk_cache.zapisz(p.with_name('zagn.npz'), z))
        w = disk_cache.wczytaj(p.with_name('zagn.npz'))
        self.assertEqual(w['lista'][1].shape, (3,)); self.assertEqual(w['prog'], dict(b=1.5, db=2.0))
        self.assertEqual(w['krotka'], (1, 'a')); self.assertIsNone(w['brak'])
        self.assertFalse(disk_cache.zapisz(p.with_name('obj.npz'), dict(x=object())))         # obiekt klasy: nie zapisuje
        self.assertIsNone(disk_cache.wczytaj(p.with_name('obj.npz')))


if __name__ == '__main__':
    unittest.main()
