"""Testy obliczeń rzutów z 192 wymiarów (rzuty_dane.py), bez GPU i bez okna: własności, które okienka R1–R8 obiecują w opisach."""
import unittest

import numpy as np

import rzuty_dane as RD


def _parametry(rng, d=24, n=64):
    return dict(g=1 + 0.1 * rng.standard_normal(d), b=0.1 * rng.standard_normal(d), W1=rng.standard_normal((n, d)) / np.sqrt(d),
                b1=0.1 * rng.standard_normal(n), V=rng.standard_normal((d, n)) / np.sqrt(n), eps=1e-5)


class TestRzutow(unittest.TestCase):
    def test_fano_kazda_para_raz(self):
        self.assertTrue(RD.fano_pokrywa_pary())
        self.assertEqual(len(RD.FANO), 7)

    def test_jakobian_jak_roznice_skonczone(self):
        rng = np.random.default_rng(1)
        par = _parametry(rng)
        x = rng.standard_normal(24) * 3 + 1
        J = RD.jakobian(par, x)
        h = 1e-6
        Jn = np.stack([(RD.pole(par, x + h * e)[0] - RD.pole(par, x - h * e)[0]) / (2 * h) for e in np.eye(24)], 1)
        self.assertLess(np.abs(J - Jn).max() / np.abs(Jn).max(), 1e-6)

    def test_rama_schura_niezmiennicza(self):
        rng = np.random.default_rng(2)
        J = rng.standard_normal((40, 40)) / np.sqrt(40)
        for para in (0, 1, 2):
            for rzecz in (0, 1, 2):
                r = RD.rama_schura(J, para, rzecz, F=rng.standard_normal(40))
                Q, T = r['Q'], r['T']
                self.assertLess(np.abs(Q.T @ Q - np.eye(3)).max(), 1e-10)
                self.assertTrue(r['niezmiennicza'])
                self.assertLess(r['reszta'], 1e-8)                        # J Q = Q T
                self.assertTrue(np.allclose(np.sort_complex(np.linalg.eigvals(T)), np.sort_complex(np.concatenate(
                    [r['wybrane'][:1], np.conj(r['wybrane'][:1]), r['wybrane'][1:]])), atol=1e-8))

    def test_rama_sily_sumy(self):
        rng = np.random.default_rng(3)
        u = rng.standard_normal((768, 192)) * rng.random((768, 1))
        fr = RD.rama_sily(u)
        B = fr['B']
        self.assertLess(np.abs(B.T @ B - np.eye(3)).max(), 1e-10)
        self.assertTrue(np.allclose(fr['F'] @ B, [fr['nF'], 0, 0], atol=1e-8))
        self.assertTrue(np.allclose((u @ B[:, 1:]).sum(0), 0, atol=1e-8))   # składowe poprzeczne znoszą się dokładnie
        silne, reszta = RD.siec_silna(u, 0.8)
        self.assertTrue(np.allclose(u[silne].sum(0) + reszta, fr['F']))
        e = (u ** 2).sum(1)
        self.assertGreaterEqual(e[silne].sum() / e.sum(), 0.8)

    def test_pary_przeciwne(self):
        v = np.random.default_rng(4).standard_normal((5, 192))
        self.assertAlmostEqual(RD.pary_przeciwne(np.concatenate([v[:1], -v[:1]])), 1.0)
        self.assertLess(RD.pary_przeciwne(v), 0.5)

    def test_baza_A_i_F7(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((300, 192)) @ rng.standard_normal((192, 192))
        delta = rng.standard_normal(192)
        A = RD.baza_A(delta, X.T @ X)
        self.assertLess(np.abs(A.T @ A - np.eye(3)).max(), 1e-10)
        self.assertTrue(np.allclose(A[:, 0], delta / np.linalg.norm(delta)))
        self.assertAlmostEqual(np.linalg.norm(A.T @ delta) / np.linalg.norm(delta), 1.0)   # 100 % separacji
        S6 = RD.glowne(rng.standard_normal((192, 192)) @ rng.standard_normal((192, 192)).T, 6)
        W6 = RD.glowne(np.cov(rng.standard_normal((192, 400))), 6)
        F7 = RD.baza_F7(A, S6, W6)
        self.assertEqual(F7.shape, (192, 7))
        self.assertLess(np.abs(F7.T @ F7 - np.eye(7)).max(), 1e-10)
        self.assertTrue(np.allclose(F7[:, 0], A[:, 0]))

    def test_tory_zerowe_i_rama_toru(self):
        rng = np.random.default_rng(6)
        tor = np.cumsum(rng.standard_normal((30, 192)) * np.linspace(2, 0.2, 30)[:, None], 0)
        Z = RD.tory_zerowe(tor, 5, ziarno=1)
        self.assertEqual(Z.shape, (5, 30, 192))
        self.assertTrue(np.allclose(Z[:, 0], tor[0]))
        self.assertTrue(np.allclose(np.linalg.norm(np.diff(Z, axis=1), axis=2), np.linalg.norm(np.diff(tor, axis=0), axis=1)[None]))
        B = RD.rama_toru(tor)
        self.assertLess(np.abs(B.T @ B - np.eye(3)).max(), 1e-10)
        netto = tor[-1] - tor[0]
        self.assertAlmostEqual(float((netto @ B)[0]), float(np.linalg.norm(netto)))       # przesunięcie netto w 100 %
        en, dl = RD.udzial_ruchu(tor[None], B)
        self.assertTrue(0 < en <= 1 and 0 < dl <= 1)
        p1, p3 = RD.udzialy_pca(Z[0])
        self.assertTrue(0 < p1 <= p3 <= 1)

    def test_separacja_L(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((600, 192))
        narz = np.zeros(600, bool); narz[7 * 30 + 12:7 * 30 + 21] = True; narz[8 * 30 + 12:8 * 30 + 21] = True   # ramka 2 × 9
        X[narz] += 3 * rng.standard_normal(192)
        B, cosd = RD.baza_L(X, narz, 0.1, 0)
        self.assertLess(np.abs(B.T @ B - np.eye(3)).max(), 1e-10)
        self.assertGreater(cosd, 0.0)
        m3 = RD.miary_separacji(X @ B, narz); z = RD.zero_separacji(X @ B, int(narz.sum()), 20)
        self.assertGreater(m3['hit'], z['hit']); self.assertGreater(m3['sil'], z['sil'])
        for osie in (1, 2, 3):
            B2, _ = RD.baza_L(X, narz, 0.1, osie)
            self.assertLess(np.abs(B2.T @ B2 - np.eye(3)).max(), 1e-10)
            self.assertTrue(np.allclose(B2[:, 0], B[:, 0]))
        r1 = RD.rdzen_narzedzia(narz, 1)
        self.assertEqual(int(r1.sum()), 14)                                  # 2 wiersze zostają, kolumny 12 i 20 odpadają
        self.assertEqual(int(RD.rdzen_narzedzia(narz, 0).sum()), 18)
        self.assertTrue(RD.rdzen_narzedzia(narz, 3).any())

    def test_zero_z_refitem_i_cv(self):
        rng = np.random.default_rng(8)
        narz = np.zeros(600, bool); narz[200:218] = True
        Xk = rng.standard_normal((12, 600, 192))
        Xk[:, narz] += 4 * rng.standard_normal(192)
        z = RD.zero_z_refitem_gpu(Xk[0], narz, 0.1, 0, 20)
        # przy n < d dopasowanie do losowych etykiet daje wysoką separację: zero z refitem >> zero ze stałymi osiami
        B, _ = RD.baza_L(Xk[0], narz, 0.1, 0)
        z0 = RD.zero_separacji(Xk[0] @ B, int(narz.sum()), 20)
        self.assertGreater(z['fisher'], z0['fisher'])
        cv, wyn = RD.cv_blokowane_gpu(Xk, narz, 0.1, 0, 4)
        self.assertEqual(len(wyn), 4)
        self.assertGreater(cv['hit'], z0['hit'])                              # prawdziwy sygnał przeżywa walidację blokowaną


if __name__ == '__main__':
    unittest.main()
