"""Testy etykiet narzędzia i oka (tool_labels.py, History.ustaw_etykiete / obszary): dane syntetyczne, bez zipów i GPU."""
import unittest

import numpy as np

from laser_app.compute import history as hist, tool_labels

H, W, TOK, K = 20, 30, 600, 5


def _cienki_wielokat():
    """Cienki ukośny prostokąt (jak cystotom) w pikselach 480 × 320: szerokość ~5 px, długość ~200 px."""
    return np.array([[40, 40], [240, 200], [244, 196], [44, 36]], np.float32)


def _historia(udzial=None, srodek=None, oko=None, narz_ramka=None):
    """Minimalna Historia bez dysku: tylko pola, których używają ustaw_etykiete i obszary."""
    rng = np.random.default_rng(0)
    h = object.__new__(hist.History)
    h.narz_ramka = np.zeros(TOK, bool) if narz_ramka is None else np.asarray(narz_ramka, bool)
    h.narz = h.narz_ramka.copy()
    h.mieszane = np.zeros(TOK, bool)
    h.udzial, h.srodek, h.oko = udzial, srodek, oko
    h.film_klatka = None if udzial is None else ('TR_0001', 870)
    h._kat_obrysu = ['cystotome']
    h.K = K
    h.lok, h.vib = rng.random((K, TOK)).astype(np.float32) + 0.1, rng.random((K, TOK)).astype(np.float32)
    h.g = rng.random((TOK, 2)).astype(np.float32)
    return h


class TestMaski(unittest.TestCase):
    def setUp(self):
        self.u = np.zeros(TOK, np.float32)
        self.u[[10, 11, 12, 13]] = [0.05, 0.3, 0.5, 0.9]
        self.srodek = np.zeros(TOK, bool); self.srodek[[11, 13]] = True
        self.ramka = np.zeros(TOK, bool); self.ramka[[10, 11, 12, 13, 14]] = True

    def test_pokrycie_jak_dotad(self):
        n, m = tool_labels.maski(self.u)
        self.assertEqual(list(np.flatnonzero(n)), [12, 13])
        self.assertEqual(list(np.flatnonzero(m)), [10, 11])
        n2, m2 = tool_labels.maski(self.u, 0.25)                                   # wywołanie pozycyjne z rzuty_okienka
        self.assertEqual(list(np.flatnonzero(n2)), [11, 12, 13])
        self.assertEqual(list(np.flatnonzero(m2)), [10])
        n3, _ = tool_labels.maski(self.u, tryb='pokrycie', prog=0.9)
        self.assertEqual(list(np.flatnonzero(n3)), [13])

    def test_dotkniete(self):
        n, m = tool_labels.maski(self.u, tryb='dotkniete')
        self.assertEqual(list(np.flatnonzero(n)), [10, 11, 12, 13])
        self.assertFalse(m.any())

    def test_srodek(self):
        n, m = tool_labels.maski(self.u, tryb='srodek', srodek=self.srodek)
        self.assertEqual(list(np.flatnonzero(n)), [11, 13])
        self.assertEqual(list(np.flatnonzero(m)), [10, 12])               # dotknięte, ale środek poza obrysem
        with self.assertRaises(ValueError):
            tool_labels.maski(self.u, tryb='srodek')

    def test_ramka(self):
        n, m = tool_labels.maski(self.u, tryb='ramka', ramka=self.ramka)
        self.assertEqual(list(np.flatnonzero(n)), [10, 11, 12, 13, 14])
        self.assertFalse(m.any())
        with self.assertRaises(ValueError):
            tool_labels.maski(self.u, tryb='ramka')

    def test_zwracane_kopie_i_ksztalty(self):
        for tryb, _ in tool_labels.TRYBY:
            n, m = tool_labels.maski(self.u, tryb=tryb, srodek=self.srodek, ramka=self.ramka)
            self.assertEqual((n.shape, m.shape, n.dtype, m.dtype), ((TOK,), (TOK,), np.dtype(bool), np.dtype(bool)))
            self.assertFalse((n & m).any())
        with self.assertRaises(ValueError):
            tool_labels.maski(self.u, tryb='cokolwiek')

    def test_tryby_kolejnosc(self):
        self.assertEqual([k for k, _ in tool_labels.TRYBY], ['dotkniete', 'pokrycie50', 'pokrycie', 'srodek', 'ramka'])


class TestCienkiObrys(unittest.TestCase):
    def test_dotkniete_nadzbior_srodka_nadzbior_pokrycia(self):
        wiel = [_cienki_wielokat()]
        u = tool_labels.udzial_z_wielokatow(wiel)
        s = tool_labels.srodek_z_wielokatow(wiel)
        dot, _ = tool_labels.maski(u, tryb='dotkniete')
        sr, _ = tool_labels.maski(u, tryb='srodek', srodek=s)
        pok, _ = tool_labels.maski(u, 0.5)
        self.assertEqual((u.shape, s.shape), ((TOK,), (TOK,)))
        self.assertGreater(dot.sum(), sr.sum())                           # cienki obrys dotyka wielu łat, środkiem trafia w mniej
        self.assertGreater(sr.sum(), pok.sum())                           # a pokrycie ≥ 0,5 zostawia (prawie) nic
        self.assertLessEqual(int(pok.sum()), 2)
        self.assertFalse((sr & ~dot).any())                               # środek w obrysie => łata dotknięta
        self.assertFalse((pok & ~dot).any())
        self.assertEqual(int(dot.sum()), int((u > 0).sum()))
        self.assertLessEqual(float(u.max()), 1.0)

    def test_grubszy_obrys_porzadek_liczebnosci(self):
        gruby = np.array([[40, 40], [240, 200], [262, 172], [62, 12]], np.float32)           # szerokość ~28 px
        u = tool_labels.udzial_z_wielokatow([gruby])
        s = tool_labels.srodek_z_wielokatow([gruby])
        n = [int(tool_labels.maski(u, tryb='dotkniete')[0].sum()), int(tool_labels.maski(u, tryb='srodek', srodek=s)[0].sum()),
             int(tool_labels.maski(u, 0.5)[0].sum())]
        self.assertTrue(n[0] >= n[1] >= n[2] >= 1, n)

    def test_piksel_srodka(self):
        lata = [np.array([[0, 0], [15, 0], [15, 15], [0, 15]], np.float32)]                  # fillPoly włącznie z brzegiem: łata 0
        self.assertEqual(int(tool_labels.maska_pikseli(lata).sum()), 256)
        self.assertEqual(list(np.flatnonzero(tool_labels.srodek_z_wielokatow(lata))), [0])
        self.assertEqual(list(np.flatnonzero(tool_labels.udzial_z_wielokatow(lata))), [0])
        malo = [np.array([[0, 0], [3, 0], [3, 3], [0, 3]], np.float32)]                      # dotyka łaty 0, ale nie jej środka
        self.assertFalse(tool_labels.srodek_z_wielokatow(malo).any())
        self.assertGreater(tool_labels.udzial_z_wielokatow(malo)[0], 0)

    def test_pusty_wielokat(self):
        self.assertFalse(tool_labels.udzial_z_wielokatow([]).any())
        self.assertFalse(tool_labels.srodek_z_wielokatow([]).any())


class TestHistoria(unittest.TestCase):
    def _obrys(self):
        wiel = [_cienki_wielokat()]
        oko = dict(rogowka=np.zeros(TOK, np.float32), zrenica=np.zeros(TOK, np.float32))
        oko['rogowka'][100:200] = 0.8                                     # rogówka obejmuje też łaty narzędzia
        oko['zrenica'][130:150] = 0.6
        oko['zrenica'][190:195] = 0.3                                     # poniżej progu
        ramka = np.zeros(TOK, bool); ramka[[100, 101, 102, 130, 131]] = True
        return tool_labels.udzial_z_wielokatow(wiel), tool_labels.srodek_z_wielokatow(wiel), oko, ramka

    def test_ustaw_etykiete_tryby(self):
        u, s, oko, ramka = self._obrys()
        h = _historia(u, s, oko, ramka)
        n_dot = h.ustaw_etykiete()
        self.assertEqual(int(h.narz.sum()), int((u > 0).sum()))
        self.assertIn('dotknięte', n_dot)
        self.assertIn(f'{int(h.narz.sum())} łat', h.etykieta)
        self.assertTrue(h.etykieta.startswith('obrys TR_0001 kl. 870 (cystotome)'))
        self.assertEqual((h.tryb_etykiety, h.prog_etykiety), ('dotkniete', 0.5))
        liczby = {}
        for tryb in ('dotkniete', 'srodek', 'pokrycie', 'ramka'):
            h.ustaw_etykiete(tryb, 0.5)
            liczby[tryb] = int(h.narz.sum())
            # cienki obrys: pokrycie ≥ 0,5 nie daje łat, więc wraca ramka GT
            self.assertEqual(h.tryb_etykiety, 'ramka' if tryb == 'pokrycie' else tryb)
            if tryb == 'pokrycie':
                self.assertTrue(h.etykieta.startswith('ramka GT (obrys, pokrycie ≥ 0.5: 0 łat)'), h.etykieta)
            self.assertEqual(h.narz_lok.shape, (K,))
            self.assertTrue(h.narz[h.fok] if h.narz.any() else True)
        self.assertGreaterEqual(liczby['dotkniete'], liczby['srodek'])
        self.assertEqual(liczby['ramka'], 5)
        self.assertEqual(liczby['pokrycie'], 5)                           # fallback do ramki GT
        h.ustaw_etykiete('srodek')
        self.assertTrue(np.array_equal(h.mieszane, (u > 0) & ~s))
        h.ustaw_etykiete('dotkniete')
        self.assertFalse(h.mieszane.any())
        self.assertTrue(np.array_equal(h.narz_ramka, ramka))              # ramka GT nietknięta

    def test_przeliczenie_narz_lok_i_fok(self):
        u, s, oko, ramka = self._obrys()
        h = _historia(u, s, oko, ramka)
        h.ustaw_etykiete('dotkniete')
        n = h.narz
        oczek = h.lok[:, n].mean(1) / h.lok[:, ~n].mean(1)
        self.assertTrue(np.allclose(h.narz_lok, oczek))
        idx = np.flatnonzero(n)
        self.assertEqual(h.fok, int(idx[np.argmin(np.linalg.norm(h.g[n] - h.g[n].mean(0), axis=1))]))

    def test_ustaw_etykiete_pokrycie_prog(self):
        u, s, oko, ramka = self._obrys()
        h = _historia(u, s, oko, ramka)
        h.ustaw_etykiete('pokrycie', 0.25)
        self.assertEqual(int(h.narz.sum()), int((u >= 0.25).sum()))
        self.assertEqual(h.prog_etykiety, 0.25)
        self.assertIn('pokrycie', h.etykieta)

    def test_fallback_do_ramki(self):
        u, s, oko, ramka = self._obrys()
        h = _historia(np.zeros(TOK, np.float32), np.zeros(TOK, bool), oko, ramka)   # obrys bez łat
        txt = h.ustaw_etykiete('dotkniete')
        self.assertTrue(np.array_equal(h.narz, ramka))
        self.assertTrue(txt.startswith('ramka GT'))
        self.assertEqual(h.tryb_etykiety, 'ramka')
        h = _historia(None, None, None, ramka)                            # w ogóle bez obrysu
        self.assertEqual(h.ustaw_etykiete('srodek'), 'ramka GT')
        self.assertTrue(np.array_equal(h.narz, ramka))
        self.assertFalse(h.mieszane.any())
        h = _historia(u, None, oko, ramka)                                # stary cache bez środków
        h.ustaw_etykiete('srodek')
        self.assertTrue(np.array_equal(h.narz, ramka))
        self.assertTrue(h.etykieta.startswith('ramka GT'))

    def test_fallback_bez_narzedzia_nigdzie(self):
        h = _historia()                                                   # ramka też pusta: fok z drgań, narz_lok = 1
        h.ustaw_etykiete()
        self.assertFalse(h.narz.any())
        self.assertTrue(np.allclose(h.narz_lok, 1))
        self.assertEqual(h.fok, int(np.argmax(h.vib.mean(0))))

    def test_obszary(self):
        u, s, oko, ramka = self._obrys()
        h = _historia(u, s, oko, ramka)
        h.ustaw_etykiete('dotkniete')
        k = h.obszary()
        self.assertEqual((k.shape, k.dtype), ((TOK,), np.dtype(np.uint8)))
        self.assertTrue((k[h.narz] == 3).all())                           # narzędzie ma pierwszeństwo
        poza = ~h.narz
        self.assertTrue((k[poza & (oko['zrenica'] >= 0.5)] == 2).all())
        self.assertTrue((k[poza & (oko['rogowka'] >= 0.5) & (oko['zrenica'] < 0.5)] == 1).all())
        self.assertTrue((k[poza & (oko['rogowka'] < 0.5) & (oko['zrenica'] < 0.5)] == 0).all())
        self.assertTrue((k[190:195][~h.narz[190:195]] == 1).all())        # źrenica 0,3 < próg: zostaje rogówka
        self.assertTrue((h.obszary(0.9)[poza] == 0).all())                # wyższy próg: żadnego obszaru oka
        self.assertEqual(len(hist.KLASY_OKA), 4)

    def test_obszary_bez_oka(self):
        h = _historia(None, None, None, np.eye(1, TOK, 7, dtype=bool)[0])
        h.ustaw_etykiete()
        k = h.obszary()
        self.assertEqual(set(np.unique(k)), {0, 3})
        self.assertEqual(int((k == 3).sum()), 1)


class TestStaraEtykieta(unittest.TestCase):
    def test_pokrycie50_to_pokrycie_z_progiem_pol(self):
        """Tryb „≥ 50 % jak dawniej” = pokrycie z progiem 0,5, niezależnie od własnego progu."""
        u = np.array([0.0, 0.1, 0.49, 0.5, 0.9], np.float32)
        n, m = tool_labels.maski(u, 0.2, 'pokrycie50')
        self.assertEqual(n.tolist(), [False, False, False, True, True])
        self.assertEqual(m.tolist(), [False, True, True, False, False])


if __name__ == '__main__':
    unittest.main()
