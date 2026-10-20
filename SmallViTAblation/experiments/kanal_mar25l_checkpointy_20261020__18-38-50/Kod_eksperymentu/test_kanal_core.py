"""Testy jednostkowe rdzenia (CPU, float64). Uruchomienie: python -m unittest test_kanal_core -v"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kanal_core as K  # noqa: E402

torch.set_default_dtype(torch.float64)


class TestKanal(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(7)

    def rnd(self, *shape):
        return torch.randn(*shape, generator=self.g, dtype=torch.float64)

    def test_gelu_matches_torch(self):
        z = self.rnd(1000) * 4
        self.assertLess(float((K.gelu(z) - torch.nn.functional.gelu(z)).abs().max()), 1e-12)
        eps = 1e-6
        fd = (K.gelu(z + eps) - K.gelu(z - eps)) / (2 * eps)
        self.assertLess(float((K.gelu_prime(z) - fd).abs().max()), 1e-7)

    def test_effective_input_reproduces_layernorm_linear(self):
        D, N, T = 12, 9, 50
        ln = torch.nn.LayerNorm(D).double(); lin = torch.nn.Linear(D, N).double()
        with torch.no_grad():
            ln.weight.copy_(self.rnd(D)); ln.bias.copy_(self.rnd(D))
        x = self.rnd(T, D) * 3 + 1
        z = lin(ln(x)).detach()
        th_eff, th_raw, b_hat = K.effective_input(lin.weight.detach(), lin.bias.detach(), ln.weight.detach(), ln.bias.detach())
        xhat = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, unbiased=False, keepdim=True) + ln.eps)
        z2 = xhat @ th_eff[:, :D].T + math.sqrt(D) * th_eff[:, D]
        self.assertLess(float((z - z2).abs().max()), 1e-10)
        self.assertLess(float(th_eff[:, :D].sum(1).abs().max()), 1e-12)
        self.assertTrue(torch.equal(th_raw[:, :D], lin.weight.detach()))

    def test_energy_cancellation_identity(self):
        T, N, D = 40, 6, 5
        g = self.rnd(T, N).clamp(min=-0.17); V = self.rnd(N, D)
        V[1] = -0.9 * V[0] + 0.01 * self.rnd(D)
        G = g.T @ g; A = V @ V.T
        kappa, c_g, c_out, beta, e = K.energy_cancellation(G, A)
        for i, j in [(0, 1), (2, 5), (3, 4)]:
            y = g[:, i:i + 1] * V[i] + g[:, j:j + 1] * V[j]
            num = (y * y).sum(); den = ((g[:, i:i + 1] * V[i]) ** 2).sum() + ((g[:, j:j + 1] * V[j]) ** 2).sum()
            self.assertAlmostEqual(float(kappa[i, j]), float(num / den), places=10)
        self.assertTrue(torch.all(kappa >= -1e-12) and torch.all(kappa <= 2 + 1e-12))

    def test_pair_token_sums_match_explicit_vectors(self):
        T, N, D = 30, 8, 7
        z = self.rnd(T, N) * 2; g = K.gelu(z); V = self.rnd(N, D)
        I = torch.tensor([0, 2, 5]); J = torch.tensor([1, 3, 7])
        A = V @ V.T
        const = K.pair_constants(A, I, J)
        mlp_norm = torch.linalg.vector_norm(g @ V, dim=1) + 0.5
        thr = torch.full((N,), 0.05)
        s = K.pair_token_sums(z, g, mlp_norm, I, J, const, thr)
        y2 = torch.zeros(3); err2 = torch.zeros(3); ynorm = torch.zeros(3)
        for t in range(T):
            y, yg = K.pair_vectors(z[t], g[t], V, I, J)
            y2 += (y * y).sum(1); err2 += ((y - yg) ** 2).sum(1); ynorm += torch.linalg.vector_norm(y, dim=1)
        self.assertLess(float((s['y2'] - y2).abs().max()), 1e-9)
        self.assertLess(float((s['err2'] - err2).abs().max()), 1e-9)
        self.assertLess(float((s['y_norm'] - ynorm).abs().max()), 1e-9)
        self.assertEqual(set(s), set(K.TOKEN_SUMS))

    def test_exact_channel_limit(self):
        """Para Mar25l: w_j = w_i + ε d, a_j = -a_i, ‖a‖ = c/ε → suma skończona, błąd GLU → 0, znoszenie → 0."""
        T, D = 400, 6
        z0 = self.rnd(T) * 1.5; dirx = self.rnd(T)
        a = self.rnd(D); a = a / a.norm()
        errs, rhos = [], []
        for eps in [1e-1, 1e-2, 1e-3]:
            z = torch.stack([z0 + eps * dirx, z0], 1); g = K.gelu(z)
            V = torch.stack([a / eps, -a / eps])
            A = V @ V.T
            I, J = torch.tensor([0]), torch.tensor([1])
            s = K.pair_token_sums(z, g, torch.ones(T), I, J, K.pair_constants(A, I, J), torch.zeros(2))
            errs.append(float(torch.sqrt(s['err2'] / s['y2'])))
            rhos.append(float(s['y_norm'] / s['u_sum']))
        self.assertTrue(errs[0] > errs[1] > errs[2])
        self.assertLess(errs[2], 1e-3)
        self.assertLess(rhos[2], 1e-2)

    def test_update_fraction_channel_motion(self):
        N, Din, D = 10, 5, 4
        th0 = self.rnd(N, Din); a0 = self.rnd(N, D)
        I, J = torch.tensor([0, 3]), torch.tensor([1, 4])
        th1, a1 = th0.clone(), a0.clone()
        m = a0[I] - a0[J]; mh = m / m.norm(dim=1, keepdim=True)
        dl = th0[I] - th0[J]; dh = dl / dl.norm(dim=1, keepdim=True)
        a1[I] += 0.3 * mh; a1[J] -= 0.3 * mh; th1[I] -= 0.2 * dh; th1[J] += 0.2 * dh
        blk = ((th1 - th0) ** 2).sum() + ((a1 - a0) ** 2).sum()
        u = K.update_fractions(th0, a0, th1, a1, I, J, blk)
        self.assertLess(float((u['f_channel'] - 1).abs().max()), 1e-12)
        self.assertTrue(torch.all(u['proj_m'] > 0) and torch.all(u['proj_delta'] < 0))
        # losowy ruch: średnia bliska 2/dim
        th2 = th0 + self.rnd(N, Din); a2 = a0 + self.rnd(N, D)
        fr = []
        for _ in range(300):
            th2 = th0 + self.rnd(N, Din); a2 = a0 + self.rnd(N, D)
            fr.append(float(K.update_fractions(th0, a0, th2, a2, I, J, torch.tensor(1.0))['f_channel'].mean()))
        self.assertAlmostEqual(np.mean(fr), 2.0 / (2 * Din + 2 * D), delta=0.03)

    def test_spearman_and_ranks(self):
        x = np.array([3.0, 1.0, 2.0, 2.0, 5.0])
        self.assertTrue(np.allclose(K.rank_average(x), [3, 0, 1.5, 1.5, 4]))
        y = np.exp(x)
        self.assertAlmostEqual(K.spearman(x, y), 1.0)
        self.assertAlmostEqual(K.spearman(x, -y), -1.0)

    def test_correlation_from_moments(self):
        Z = self.rnd(200, 5); Z[:, 4] = 2 * Z[:, 0] + 0.1 * Z[:, 1]
        R = K.correlation_from_moments(Z.sum(0), Z.T @ Z, 200)
        self.assertLess(float((R - torch.as_tensor(np.corrcoef(Z.numpy().T))).abs().max()), 1e-10)

    def test_select_top_and_literal(self):
        v = torch.tensor([0.1, -0.9, -0.95, 0.3, -0.6])
        idx, total, trunc = K.select_top(v, v < -0.5, 2, largest=False)
        self.assertEqual(idx.tolist(), [2, 1]); self.assertEqual(total, 3); self.assertTrue(trunc)
        wm = dict(pair_param_norm=torch.tensor([2e3, 10.0]), c_in_raw=torch.tensor([0.9999, 0.9999]),
                  c_out=torch.tensor([-0.95, -0.95]))
        upd = dict(f_channel=torch.tensor([0.95, float('nan')]))
        f = K.literal_flags(wm, upd, dict(norm_gt=1e3, cos_distance_lt=1e-3, cout_lt=-0.9, update_fraction_gt=0.9))
        self.assertEqual(f['all'].tolist(), [True, False])

    def test_enrichment_and_hist(self):
        c_in = np.array([0.95, 0.1, 0.2, 0.97]); c_out = np.array([-0.8, -0.7, 0.3, 0.5])
        e = K.enrichment(c_in, c_out, -0.5, 0.9)
        self.assertAlmostEqual(e['conditional'], 0.5); self.assertAlmostEqual(e['base'], 0.5); self.assertAlmostEqual(e['ratio'], 1.0)
        h, ex, ey = K.hist2d(c_in, c_out, 4)
        self.assertEqual(int(h.sum()), 4)


if __name__ == '__main__':
    unittest.main()
