"""Przyczyny ruchu cząsteczek w uczeniu: wagi (grupy parametrów), gradient straty i atencja, liczone
w przedziałach między zapisanymi checkpointami wag (zwykle co 10 kroków) na klatce próbnej.

Dla przedziału i (wagi W_i -> W_(i+1)):
- wagi: delta_g = z(W_i z grupą g podmienioną na W_(i+1)) − z(W_i); udział grupy w ruchu tokenu
  = <delta_g, dz> / ||dz||^2, gdzie dz = z(W_(i+1)) − z(W_i). Bez interakcji grup udziały sumują się do 1;
  zmiana wag grupy ||dW_g|| / ||W_g||;
- gradient: g = dL/dz (strata uczenia na tej klatce, ramka z obrysu narzędzia) w W_i; cos(−g, dz) mówi,
  czy cząstka rusza się tam, gdzie strata chce ją przesunąć; ||g|| = nacisk straty na token;
  normy gradientu wag każdej grupy;
- atencja ostatniego bloku (średnia głów): zmiana tego, na co patrzy token, 0,5 ||A_(i+1)[q] − A_i[q]||_1,
  i cos(dz_q, sum_j A_qj dz_j): czy token rusza się razem z tokenami, na które patrzy.
"""
import numpy as np

GRUPY = ['łodyga'] + [f'blok {b + 1} {c}' for b in range(6) for c in ('atencja', 'MLP')]


def grupa_parametru(nazwa):
    """Indeks grupy w GRUPY albo None (głowica i norma końcowa nie wpływają na latent)."""
    if nazwa.startswith('lodyga'):
        return 0
    if nazwa.startswith('bloki.'):
        b = int(nazwa.split('.')[1]); reszta = nazwa.split('.', 2)[2]
        return 1 + 2 * b + (0 if reszta.startswith(('norm1', 'attn', 'ls1')) else 1)
    return None


def _model_biegu(dane):
    """Moduł model.py biegu (funkcje cele i strata użyte w uczeniu)."""
    import importlib.util
    p = dane.parents[2] / 'kod' / 'model.py'
    spec = importlib.util.spec_from_file_location('model_historii_uczenia', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Przyczyny:
    def __init__(self, bieg, opis, i_klatki, rgb):
        import torch
        import detekcja as D
        self.opis, self.rgb = opis, np.ascontiguousarray(rgb)
        self.det = D.Detektor(opis['wagi'][0]); self.net = self.det.net; self.dev = self.det.dev
        self.x = torch.from_numpy(np.array(self.rgb)).permute(2, 0, 1)[None].float().to(self.dev) / 255
        mi = np.load(bieg['dane'] / 'maski.npz')['mask_input'][i_klatki]
        ys, xs = np.nonzero(mi)
        self.ramka = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) else None
        self.M = _model_biegu(bieg['dane'])
        self.klucze = list(torch.load(opis['wagi'][0], map_location='cpu', weights_only=True)['net'].keys())
        self.grupy = [grupa_parametru(k) for k in self.klucze]
        self._sd = {}

    def sd(self, i):
        import torch
        if i not in self._sd:
            if len(self._sd) > 3:
                self._sd.pop(next(iter(self._sd)))
            ck = torch.load(self.opis['wagi'][i], map_location='cpu', weights_only=True)
            self._sd[i] = {k: v.to(self.dev) for k, v in ck.get('net', ck).items()}
        return self._sd[i]

    def _z(self, sd):
        import torch
        self.net.load_state_dict(sd, strict=True)
        pam = {}
        h = self.net.bloki[-1].mlp.act.register_forward_hook(lambda m, a, o: pam.update(z=o))
        try:
            with torch.no_grad():
                self.net(self.x)
        finally:
            h.remove(); self.net._latent = None
        return pam['z'][0].float()                                                   # [600, 768]

    def delty(self, i):
        """z(W_i) [600, 768], dz [600, 768] i delta_g każdej grupy [G, 600, 768] dla przedziału i."""
        import torch
        a, b = self.sd(i), self.sd(i + 1)
        z0 = self._z(a); dz = self._z(b) - z0
        D = torch.zeros((len(GRUPY), 600, 768), device=self.dev)
        for g in range(len(GRUPY)):
            kl = [k for k, gg in zip(self.klucze, self.grupy) if gg == g]
            D[g] = self._z({**a, **{k: b[k] for k in kl}}) - z0
        return z0, dz, D

    def gradient(self, i):
        """dL/dz [600, 768], normy gradientu wag grup [G] i strata w wagach checkpointu i."""
        import torch
        self.net.load_state_dict(self.sd(i), strict=True)
        self.net.requires_grad_(True); self.net.zero_grad(set_to_none=True)
        pam = {}

        def hak(m, a, o):
            o.retain_grad(); pam['z'] = o
        h = self.net.bloki[-1].mlp.act.register_forward_hook(hak)
        try:
            o, _ = self.net(self.x)
            heat, ltrb, poz = self.M.cele([[self.ramka]] if self.ramka else [[]], o.shape[1], o.shape[2])
            L, _ = self.M.strata(o, heat.to(self.dev), ltrb.to(self.dev), poz.to(self.dev))
            L.backward()
        finally:
            h.remove(); self.net._latent = None
        g = pam['z'].grad[0].float().detach()
        gw = np.zeros(len(GRUPY), np.float32)
        for (k, p), gg in zip(self.net.named_parameters(), [grupa_parametru(k) for k, _ in self.net.named_parameters()]):
            if gg is not None and p.grad is not None:
                gw[gg] += float(p.grad.float().pow(2).sum())
        self.net.requires_grad_(False); self.net.zero_grad(set_to_none=True)
        return g, np.sqrt(gw), float(L)

    def atencja(self, i):
        self.net.load_state_dict(self.sd(i), strict=True)
        return self.det.atencja(self.rgb).mean(0)                                     # [600, 600]

    def licz(self, postep=None):
        """Wszystkie przedziały: słownik tablic (n = liczba checkpointów, n − 1 przedziałów)."""
        import torch
        n = len(self.opis['wagi']); G = len(GRUPY)
        udz = np.zeros((n - 1, G, 600), np.float32); dw = np.zeros((n - 1, G), np.float32)
        cos_g = np.zeros((n - 1, 600), np.float32); g_norma = np.zeros((n, 600), np.float32)
        gw = np.zeros((n, G), np.float32); strata = np.zeros(n, np.float32)
        tv = np.zeros((n - 1, 600), np.float32); cos_a = np.zeros((n - 1, 600), np.float32)
        cos_u = np.zeros((n - 1, 600), np.float32)
        dz_norma = np.zeros((n - 1, 600), np.float32)
        A_prev = None
        for i in range(n):
            g, gw[i], strata[i] = self.gradient(i)
            g_norma[i] = g.norm(dim=1).cpu().numpy()
            A = torch.from_numpy(self.atencja(i)).to(self.dev)
            if i > 0:
                tv[i - 1] = (.5 * (A - A_prev).abs().sum(1)).cpu().numpy()
            if i < n - 1:
                z0, dz, D = self.delty(i)
                nn = (dz * dz).sum(1).clamp_min(1e-12)
                udz[i] = ((D * dz[None]).sum(-1) / nn[None]).cpu().numpy()
                cos_g[i] = torch.nn.functional.cosine_similarity(-g, dz, dim=1).cpu().numpy()
                cos_a[i] = torch.nn.functional.cosine_similarity(dz, A @ dz, dim=1).cpu().numpy()
                # kontrola: ruch razem ze średnim ruchem wszystkich tokenów (atencja równomierna)
                cos_u[i] = torch.nn.functional.cosine_similarity(dz, dz.mean(0, keepdim=True), dim=1).cpu().numpy()
                dz_norma[i] = dz.norm(dim=1).cpu().numpy()
                a, b = self.sd(i), self.sd(i + 1)
                wn = np.zeros(G, np.float32)
                for k, gg in zip(self.klucze, self.grupy):
                    if gg is not None and a[k].is_floating_point():
                        dw[i, gg] += float((b[k].float() - a[k].float()).pow(2).sum())
                        wn[gg] += float(a[k].float().pow(2).sum())
                dw[i] = np.sqrt(dw[i]) / np.maximum(np.sqrt(wn), 1e-12)
            A_prev = A
            if postep:
                postep(i + 1, n)
        return dict(kroki_wag=self.opis['kroki_wag'], udzialy=udz, zmiana_wag=dw, cos_grad=cos_g, grad_norma=g_norma,
                    grad_wag=gw, strata_klatki=strata, zmiana_atencji=tv, cos_atencji=cos_a, cos_rowno=cos_u, dz_norma=dz_norma,
                    ramka=self.ramka)
