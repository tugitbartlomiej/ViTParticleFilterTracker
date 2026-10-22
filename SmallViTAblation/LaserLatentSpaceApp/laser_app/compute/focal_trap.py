"""Pułapka optyczna z laserów (ogniskowa wiązek) wokół wybranego koloidu: zakładka „Lasery i koloidy”, okienko 11 i sfera.

Koloid j w checkpoincie k to stan x̃_j przed MLP bloku 6 (history.policz); na sferze p_j = P3 (x̃_j − x̄), gdzie wiersze P3
(3 × 192) są ortonormalne. Pole wszystkich 768 wiązek bloku 6 w otoczeniu koloidu, w przekroju 3D rozpiętym przez P3:
    x(u) = x̃_j + P3ᵀ u,     f(u) = P3 Σ_c (z_c(x(u)) − z̄_c) v_c,     z_c(x) = GELU(w_c · LN(x) + b_c),
z̄_c = średnia jasność lasera c po 600 tokenach zdjęcia. Położenia na sferze są względem środka chmury, więc pchnięcie też
(jak wiązki na sferze, ZASADY Z20). f(0) = dokładny skok koloidu w bloku 6 (x̃ → x̃ + MLP, względem środka chmury).
Pole liczone jest TYLKO w komórce tokenu: sześcian wokół koloidu o połowie boku L = zasięg · r_kom, gdzie r_kom = połowa
odległości do najbliższego innego tokenu w rzucie 3D (w tym checkpoincie). Przy zasięgu ≤ 1 żaden inny token nie leży w sześcianie
(najdalszy róg √3 · r_kom < 2 r_kom), więc pole nie wchodzi na sąsiednie tokeny (wersja 4; wcześniej L = zasięg · r99 i siatka rosła
aż do pułapki, do 3 r99).
Pułapka (ogniskowa) = punkt u* komórki, w którym pchnięcia wszystkich wiązek się znoszą: f(u*) = 0 (metoda Newtona od koloidu,
tylko wewnątrz sześcianu). Jej charakter z wartości własnych λ Jakobianu 3 × 3 ∂f/∂u w u*: Re λ < 0 ściąga,
Re λ > 0 wypycha, Im λ ≠ 0 obraca (siodło z wirem, ZASADY Z15, Z23). Gdy zera nie ma w komórce, podawane jest minimum |f| na
siatce (zb = False).
Kolor punktu siatki = wiązki o największym wkładzie |z_c − z̄_c| · |P3 v_c| (kolory reflektorów jak na sferze), mieszane wagami
trzech największych; „dominacja” = udział największej w sumie wkładów.
Jednostki: surowe jednostki rzutu P3 (jak h.pos); ekran przelicza je skalą sfery 0,55 R / r99.
Do zbliżenia na ognisko jednej cząsteczki zapisywane są też jej wiązki: P3 v_c (kierunek i moc w rzucie sfery), jasność z_jc
i średnia jasność z̄_c po tokenach; wkład wiązki c do skoku to (z_jc − z̄_c) P3 v_c, a suma 768 wkładów = f(0).
Wynik w katalogu historii: ogniskowa_t<tok>_G<G>_z<zasięg·100>.npz.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from laser_app.compute import history as hist

WERSJA = 4                                   # 2: wiązki cząsteczki (P3 v_c, z_jc, z̄_c); 3: kontrola skoku, G = 1 = tylko wiązki; 4: pole tylko w komórce tokenu


class Interrupted(Exception):
    pass


def plik(h, tok, G, zasieg):
    return Path(h.katalog) / f'ogniskowa_t{int(tok):03d}_G{int(G)}_z{int(round(zasieg * 100)):03d}.npz'


def wczytaj(h, tok, G, zasieg):
    p = plik(h, tok, G, zasieg)
    if not p.exists():
        return None
    d = dict(np.load(p, allow_pickle=False))
    return d if int(d.get('wersja', 0)) == WERSJA else None


def _rzut_P3(net, dev, wagi, krok_ost):
    """P3 jak w history.policz: 3 główne kierunki znormowanych kolumn fc2 OSTATNIEGO checkpointu (float64)."""
    import torch
    st = torch.load(Path(wagi) / f'k{int(krok_ost):05d}.pt', map_location=dev, weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    V = net.bloki[-1].mlp.fc2.weight.detach().double()
    Vn = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)
    U_, _, _ = torch.linalg.svd(Vn - Vn.mean(1, keepdim=True), full_matrices=False)
    return U_[:, :3].T.contiguous()                                             # (3, 192) float64


def policz(h, tok, G=17, zasieg=1.0, postep=None, przerwij=None, urzadzenie='cuda'):
    """Pole wiązek i pułapka wokół koloidu `tok` we wszystkich checkpointach historii h (history.History).
    Zwraca słownik tablic (i zapisuje go obok historii). postep(i, n, opis); przerwij() → True kończy (wyjątek Przerwano)."""
    return policz_wiele(h, [tok], G, zasieg, postep, przerwij, urzadzenie)[int(tok)]


def _jakobian(xj, u, P3, g_, be_, eps, W1, b1, PV):
    """∂f/∂u (3 × 3) analitycznie dla jednej cząsteczki: f(u) = PV (GELU(W1 LN(xj + P3ᵀu) + b1) − z̄); pochodna LayerNorm, GELU (erf)
    i rzutu. Zgodne z torch.autograd.functional.jacobian do błędu zaokrągleń (testy/test_moduly.py)."""
    return _jakobian_wsadowo((xj + u @ P3)[None], P3, g_, be_, eps, W1, b1, PV)[0]


def _jakobian_wsadowo(X, P3, g_, be_, eps, W1, b1, PV):
    """∂f/∂u (B, 3, 3) w punktach X (B, 192) naraz (wszystkie cząsteczki jednym mnożeniem macierzy na GPU)."""
    import math
    import torch
    n = X.shape[1]
    xc = X - X.mean(1, keepdim=True); sd = torch.sqrt((xc * xc).mean(1, keepdim=True) + eps); xh = xc / sd   # (B, 192)
    a = (xh * g_ + be_) @ W1.T + b1                                             # (B, 768)
    Tc = P3 - P3.mean(1, keepdim=True)                                          # (3, 192) styczne: kierunki P3 w 192D
    dxh = (Tc[None] - xh[:, None, :] * (xh @ Tc.T)[:, :, None] / n) / sd[:, :, None]   # (B, 3, 192) pochodna LN
    dg = 0.5 * (1 + torch.erf(a / math.sqrt(2))) + a * torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    dz = ((dxh * g_) @ W1.T) * dg[:, None, :]                                    # (B, 3, 768): ∂z/∂u_k
    return (dz @ PV.T).transpose(1, 2)                                           # J[b, i, k] = ∂f_i/∂u_k


class _Block6Params:
    """Wagi bloku 6 w jednym checkpoincie w dwóch dokładnościach: float64 do reszty f (pchnięcia wiązek znoszą się w pułapce,
    więc f to mała różnica dużych liczb) i `dt` (float32: jakobian, siatka pola, kolory; na kartach GeForce float64 liczy się
    kilkadziesiąt razy wolniej, a pole i tak zapisujemy w float16)."""

    def __init__(self, b6, P3, dt):
        import torch
        self.eps = float(b6.norm2.eps)
        d64 = dict(g=b6.norm2.weight.double(), be=b6.norm2.bias.double(), W1=b6.mlp.fc1.weight.double(),
                   b1=b6.mlp.fc1.bias.double(), W2=b6.mlp.fc2.weight.double(), P3=P3)
        d64['PV'] = P3 @ d64['W2']
        self.p64 = d64
        self.dt = dt
        self.pdt = d64 if dt == torch.float64 else {k: v.to(dt) for k, v in d64.items()}

    def ustaw_P3(self, P3):
        import torch
        self.p64['P3'] = P3; self.p64['PV'] = P3 @ self.p64['W2']
        self.pdt = self.p64 if self.dt == torch.float64 else {k: v.to(self.dt) for k, v in self.p64.items()}

    @staticmethod
    def _z(p, eps, X):
        import torch
        import torch.nn.functional as Fn
        m = X.mean(-1, keepdim=True); v = ((X - m) ** 2).mean(-1, keepdim=True)
        return Fn.gelu(((X - m) / torch.sqrt(v + eps) * p['g'] + p['be']) @ p['W1'].T + p['b1'])

    def jasnosci(self, X, dokladnie=True):
        """z = GELU(W1 LN(X) + b1) dla X (..., 192): float64 (dokladnie) albo w dokładności roboczej."""
        p = self.p64 if dokladnie else self.pdt
        return self._z(p, self.eps, X if dokladnie else X.to(self.dt))

    def jakobian(self, X):
        p = self.pdt
        return _jakobian_wsadowo(X.to(self.dt), p['P3'], p['g'], p['be'], self.eps, p['W1'], p['b1'], p['PV']).double()


def checkpointy(h, dev, przerwij=None, dt=None):
    """Kolejne checkpointy historii: wagi wczytywane z wyprzedzeniem (4 wątki), jedno przejście sieci, stan przed MLP bloku 6
    i rzut P3 zgodny ze znakami z cache historii. Zwraca (i, krok, kontekst) dla każdego checkpointu; kontekst: xs (600, 192)
    float64, par (_Block6Params), Zt, zbar, pc (600, 3), Fm (ruch w bloku 6, 192D), blad_z, blad_pos."""
    import torch
    dt = dt or torch.float32
    M = hist.modul_modelu()
    net = M.MalyViT().to(dev).eval().requires_grad_(False)
    pam = {}
    net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))
    net.bloki[-1].mlp.act.register_forward_hook(lambda m, a, o: pam.__setitem__('z', o.detach()))
    net.norm.register_forward_hook(lambda m, a, o: pam.__setitem__('xa', a[0].detach()))         # stan PO bloku 6 (wejście końcowej LN)
    kroki = [int(k) for k in h.kroki]; K = len(kroki)
    P3 = _rzut_P3(net, dev, h.wagi, kroki[-1])
    rgb = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
    znaki = None
    from concurrent.futures import ThreadPoolExecutor
    pula = ThreadPoolExecutor(4)
    odczyt = lambda k: (lambda st: st.get('net', st))(torch.load(Path(h.wagi) / f'k{k:05d}.pt', map_location='cpu', weights_only=True))
    kolejka = [pula.submit(odczyt, k) for k in kroki[:6]]
    try:
        for i, k in enumerate(kroki):
            if przerwij is not None and przerwij():
                raise Interrupted()
            net.load_state_dict(kolejka.pop(0).result(), strict=True)
            if i + 6 < K:
                kolejka.append(pula.submit(odczyt, kroki[i + 6]))
            with torch.no_grad():
                net(rgb)
            xs = pam['x'][0].double()                                             # (600, 192) stan przed MLP bloku 6
            par = _Block6Params(net.bloki[-1], P3, dt)
            Zt = par.jasnosci(xs); zbar = Zt.mean(0)
            blad_z = float((Zt.float() - pam['z'][0].float()).abs().max())          # zgodność z przejściem sieci
            # kontrola, że wiązki naprawdę przesuwają koloid: Σ_c (z_jc − z̄_c) v_c = (stan po bloku 6 − stan przed MLP), względem środka chmury, 192D
            Fm = pam['xa'][0].double() - xs; Fm = Fm - Fm.mean(0, keepdim=True)
            pc = (xs - xs.mean(0, keepdim=True)) @ P3.T                           # (600, 3) położenia jak h.pos
            pos_h = torch.from_numpy(np.asarray(h.pos[i], np.float64)).to(dev)
            if znaki is None:                                                     # znak kierunków SVD jak w cache historii
                znaki = torch.sign((pc * pos_h).sum(0)); znaki[znaki == 0] = 1
                P3 = P3 * znaki[:, None]; pc = pc * znaki
                if float((pc - pos_h).abs().max() / pos_h.abs().max().clamp_min(1e-12)) > 1e-3:
                    # rzut niezgodny z cache (np. inny ostatni checkpoint): dopasowanie do zapisanych położeń + ortonormalizacja wierszy
                    Xc = xs - xs.mean(0, keepdim=True)
                    A = torch.linalg.lstsq(Xc, pos_h).solution.T                  # (3, 192)
                    U_, _, Vt_ = torch.linalg.svd(A, full_matrices=False)
                    P3 = U_ @ Vt_; pc = Xc @ P3.T
                par.ustaw_P3(P3)
            blad_pos = float(np.abs(pc.float().cpu().numpy() - h.pos[i]).max() / max(np.abs(h.pos[i]).max(), 1e-9))
            yield i, k, dict(xs=xs, par=par, Zt=Zt, zbar=zbar, pc=pc, Fm=Fm, blad_z=blad_z, blad_pos=blad_pos)
    finally:
        for f_ in kolejka:
            f_.cancel()
        pula.shutdown(wait=False)


def komorki(pc, toki, zasieg):
    """Komórka każdego tokenu (połowa odległości do najbliższego innego tokenu w rzucie 3D): r_kom (B,), sąsiad (B,), L = zasięg · r_kom."""
    import torch
    odl = torch.cdist(pc[toki], pc)                                               # (B, 600)
    odl[torch.arange(len(toki), device=pc.device), toki] = float('inf')
    rk, sas = odl.min(1)
    rk = 0.5 * rk
    return rk, sas, zasieg * rk


def rozwiaz3(J, f):
    """x = J⁻¹ f dla stosu macierzy 3 × 3 (B, 3, 3) i wektorów (B, 3): wzór z dopełnień algebraicznych, elementowo na GPU
    (bez SVD i bez pętli). Prawie osobliwe J (|det| < 1e-12 · ‖J‖³): pseudoodwrotność, jak lstsq."""
    import torch
    a, b, c = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
    d, e, g_ = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
    p, q, r = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]
    A_ = torch.stack([torch.stack([e * r - g_ * q, c * q - b * r, b * g_ - c * e], 1),
                      torch.stack([g_ * p - d * r, a * r - c * p, c * d - a * g_], 1),
                      torch.stack([d * q - e * p, b * p - a * q, a * e - b * d], 1)], 1)   # macierz dołączona
    det = a * A_[:, 0, 0] + b * A_[:, 1, 0] + c * A_[:, 2, 0]
    x = (A_ @ f[:, :, None])[:, :, 0] / det[:, None]
    zle = det.abs() < 1e-12 * J.flatten(1).norm(dim=1) ** 3
    if bool(zle.any()):
        x[zle] = (torch.linalg.pinv(J[zle]) @ f[zle][:, :, None])[:, :, 0]
    return x


def newton_wsadowo(kt, toki, Li, szukaj=True, iteracje=60, sprawdzaj_co=4):
    """Zero pola f(u) = 0 wokół każdego koloidu z `toki` naraz (B cząsteczek na GPU), tylko wewnątrz sześcianu komórki |u|∞ ≤ L.
    Krok Newtona z jakobianem w dokładności roboczej (float32) i resztą f w float64 (niedokładny Newton: zbieżność do dokładności
    reszty). Krok przycinany do 0,3 L; cząsteczka, która wyjdzie z komórki albo się zbiegnie (|f| ≤ 1e-6 |f(0)|), przestaje się
    ruszać; po 15 iteracjach kończy też ta, której |f| wciąż > 1e-3 |f(0)| (krąży w komórce bez zera; pomiar na 18 000 parach
    checkpoint × cząsteczka: 8 iteracji daje te same zera co 60). Synchronizacja z kartą co `sprawdzaj_co` iteracji. Zwraca u (B, 3), zb (B,) bool,
    f(0) (B, 3), z − z̄ w koloidzie (B, 768) i liczbę wykonanych iteracji."""
    import torch
    par, xs, zbar = kt['par'], kt['xs'], kt['zbar']
    P3, PV = par.p64['P3'], par.p64['PV']
    xj = xs[toki]                                                                 # (B, 192)
    B = len(toki); dev = xs.device
    dz0 = par.jasnosci(xj) - zbar
    f0 = dz0 @ PV.T
    u = torch.zeros(B, 3, dtype=torch.float64, device=dev)
    n0 = f0.norm(dim=1).clamp_min(1e-30)
    zb = torch.zeros(B, dtype=torch.bool, device=dev)
    akt = torch.ones(B, dtype=torch.bool, device=dev) if szukaj else torch.zeros(B, dtype=torch.bool, device=dev)
    it = 0
    idx = torch.arange(B, device=dev) if szukaj else torch.zeros(0, dtype=torch.long, device=dev)   # cząsteczki jeszcze w ruchu
    for it in range(1, (iteracje if szukaj else 0) + 1):
        if idx.numel() == 0:
            break
        ui, Lii = u[idx], Li[idx]
        X = xj[idx] + ui @ P3
        fu = (par.jasnosci(X) - zbar) @ PV.T
        zb_teraz = akt[idx] & (fu.norm(dim=1) <= 1e-6 * n0[idx])
        zb[idx] |= zb_teraz
        a = akt[idx] & ~zb_teraz
        if it > 15:                                                               # krąży w komórce bez zera: Newton tak blisko
            a &= fu.norm(dim=1) <= 1e-3 * n0[idx]                                 # zera zbiega się kwadratowo, więc to nie pułapka
        du = -rozwiaz3(par.jakobian(X), fu)
        du = du * torch.clamp(0.3 * Lii / du.norm(dim=1).clamp_min(1e-300), max=1.0)[:, None]
        ui = torch.where(a[:, None], ui + du, ui)
        a &= ~(ui.abs().amax(1) > Lii)                                            # wyszła z komórki: koniec
        u[idx] = ui; akt[idx] = a
        if it % sprawdzaj_co == 0:
            idx = idx[akt[idx]]                                                   # jedna synchronizacja co kilka iteracji
    if szukaj:
        fu = (par.jasnosci(xj + u @ P3) - zbar) @ PV.T
        zb |= (~zb) & (fu.norm(dim=1) <= 1e-4 * n0)
        zb &= u.abs().amax(1) <= Li                                               # zero liczy się tylko w komórce tokenu
    return u, zb, f0, dz0, it


def policz_wiele(h, toki, G=17, zasieg=1.0, postep=None, przerwij=None, urzadzenie='cuda', dokladnosc='float32'):
    """Pułapki kilku koloidów naraz: wczytanie wag i przejście sieci są wspólne, Newton liczy wszystkie cząsteczki jednocześnie
    (newton_wsadowo), osobno liczy się tylko siatka pola w komórce każdej cząsteczki. dokladnosc: 'float32' (jakobian i siatka
    w float32, reszta f w float64) albo 'float64' (wszystko jak dawniej). Tokeny z gotowym plikiem są brane z dysku.
    Zwraca {tok: wynik} i zapisuje każdy wynik osobno (ten sam plik co policz dla jednego tokenu)."""
    import torch
    G = int(G)
    toki = list(dict.fromkeys(int(t) for t in toki))
    wyniki = {}
    for t in toki:
        d = wczytaj(h, t, G, zasieg)
        if d is not None:
            wyniki[t] = d
    licz = [t for t in toki if t not in wyniki]
    if not licz:
        return wyniki
    dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
    dt = torch.float64 if dokladnosc == 'float64' else torch.float32
    szukaj = G > 1                                                                # G = 1: tylko wiązki wybranej cząsteczki, bez pola i bez zera
    kroki = [int(k) for k in h.kroki]; K = len(kroki)
    kol = torch.from_numpy(np.asarray(h.kol, np.float64)).to(dev)                 # (768, 3) kolory reflektorów jak na sferze
    lin = torch.linspace(-1.0, 1.0, G, dtype=torch.float64, device=dev) if G > 1 else torch.zeros(1, dtype=torch.float64, device=dev)
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing='ij')
    SIATKA = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], 1)      # (N, 3), indeks i = ix + G·iy + G²·iz
    N = G ** 3

    def tablice():
        return dict(pole=np.zeros((K, N, 4), np.float16), kolor=np.zeros((K, N, 4), np.float16), srodek=np.zeros((K, 3), np.float32),
                    srodek_siatki=np.zeros((K, 3), np.float32), ogn=np.zeros((K, 3), np.float32), zb=np.zeros(K, bool),
                    skok=np.zeros((K, 3), np.float32), L=np.zeros(K, np.float32), fmed=np.zeros(K, np.float32), fogn=np.zeros(K, np.float32),
                    lam=np.zeros((K, 3), np.complex64), wek=np.zeros((K, 3, 3), np.complex64), lam0=np.zeros((K, 3), np.complex64),
                    wek0=np.zeros((K, 3, 3), np.complex64), ogn_lin=np.zeros((K, 3), np.float32),
                    top_ogn=np.zeros((K, 5), np.int16), top_ogn_u=np.zeros((K, 5), np.float32), pv=np.zeros((K, 768, 3), np.float32),
                    zj=np.zeros((K, 768), np.float32), zbar=np.zeros((K, 768), np.float32), top_kol=np.zeros((K, 5), np.int16),
                    top_kol_u=np.zeros((K, 5), np.float32), blad_z=np.zeros(K, np.float32), blad_pos=np.zeros(K, np.float32),
                    blad_skok=np.zeros(K, np.float32), r_kom=np.zeros(K, np.float32), sasiad=np.zeros(K, np.int16))

    T = {t: tablice() for t in licz}
    t0 = time.time()
    opis_tok = str(licz[0]) if len(licz) == 1 else f'{len(licz)} cząsteczek ({", ".join(map(str, licz))})'
    tk = torch.as_tensor(licz, device=dev)
    for i, k, kt in checkpointy(h, dev, przerwij, dt):
        par, xs, zbar, pc = kt['par'], kt['xs'], kt['zbar'], kt['pc']
        P3, PV, W2 = par.p64['P3'], par.p64['PV'], par.p64['W2']
        nv = PV.norm(dim=0)                                                       # |P3 v_c|
        rk, sas, Li = komorki(pc, tk, float(zasieg))
        with torch.no_grad():
            U_, ZB, F0, DZ0, _ = newton_wsadowo(kt, tk, Li, szukaj)
            Fw = (kt['Zt'][tk] - zbar) @ W2.T
            Fm = kt['Fm'][tk]
            BS = ((Fw - Fm).norm(dim=1) / Fm.norm(dim=1).clamp_min(1e-30)).cpu().numpy()
            Jcz = par.jakobian(xs[tk] + U_ @ P3)                                   # (B, 3, 3) w zerach (albo w końcu Newtona)
            J0 = par.jakobian(xs[tk])                                              # (B, 3, 3) w koloidach
        rk_, sas_, Li_ = rk.cpu().numpy(), sas.cpu().numpy(), Li.cpu().numpy()
        ZB_ = ZB.cpu().numpy(); PVT = PV.T.float().cpu().numpy(); ZBAR_i = zbar.float().cpu().numpy()
        J0n = J0.cpu().numpy()
        for b, tok in enumerate(licz):                                            # osobno tylko siatka pola w komórce tokenu
            A_ = T[tok]
            A_['blad_z'][i] = kt['blad_z']; A_['blad_pos'][i] = kt['blad_pos']; A_['blad_skok'][i] = BS[b]
            A_['r_kom'][i] = rk_[b]; A_['sasiad'][i] = sas_[b]; Li_b = float(Li_[b])
            xj = xs[tok]
            u = U_[b]; zb = bool(ZB_[b]); f0 = F0[b:b + 1]; dz0 = DZ0[b:b + 1]
            with torch.no_grad():
                c_off = torch.zeros(3, dtype=torch.float64, device=dev)          # siatka zawsze wokół koloidu, nie rośnie
                U = c_off + SIATKA * Li_b
                dz = par.jasnosci(xj + U @ P3, dokladnie=dt == torch.float64) - zbar.to(dt)
                f = dz @ par.pdt['PV'].T
                w = dz.abs() * nv.to(dt)
                tw, ti = w.topk(3, dim=1)
                suma = w.sum(1).clamp_min(1e-30)
                mix = (tw[..., None] * kol.to(dt)[ti]).sum(1) / tw.sum(1, keepdim=True).clamp_min(1e-30)
                fm = f.norm(dim=1)
                A_['pole'][i, :, :3] = f.float().cpu().numpy(); A_['pole'][i, :, 3] = fm.float().cpu().numpy()
                A_['kolor'][i, :, :3] = mix.float().cpu().numpy(); A_['kolor'][i, :, 3] = (tw[:, 0] / suma).float().cpu().numpy()
                A_['fmed'][i] = float(fm.median())
                A_['skok'][i] = f0[0].float().cpu().numpy()
                A_['srodek'][i] = pc[tok].float().cpu().numpy(); A_['srodek_siatki'][i] = (pc[tok] + c_off).float().cpu().numpy()
                Ju = Jcz[b]
                if not zb:                                                        # zastępczo: minimum |f| na siatce
                    u = U[int(fm.argmin())].clone()
                    Ju = par.jakobian((xj + u @ P3)[None])[0]
                dzu = par.jasnosci((xj + u @ P3)[None]) - zbar
                fu = dzu @ PV.T
                A_['ogn'][i] = (pc[tok] + u).float().cpu().numpy(); A_['zb'][i] = zb; A_['fogn'][i] = float(fu.norm())
                ev, evec = np.linalg.eig(Ju.cpu().numpy())
                o = np.argsort(ev.real); A_['lam'][i] = ev[o]; A_['wek'][i] = evec[:, o]
                ev, evec = np.linalg.eig(J0n[b])
                o = np.argsort(ev.real); A_['lam0'][i] = ev[o]; A_['wek0'][i] = evec[:, o]
                # liniowe oszacowanie pułapki z kształtu pola w koloidzie: u = −J₀⁻¹ f(0) (bez ograniczenia zasięgu)
                A_['ogn_lin'][i] = (pc[tok] - torch.linalg.lstsq(J0[b], f0[0][:, None]).solution[:, 0]).float().cpu().numpy()
                for (dzz, ti_, tu_) in ((dzu[0], 'top_ogn', 'top_ogn_u'), (dz0[0], 'top_kol', 'top_kol_u')):
                    ww = dzz.abs() * nv; tv, tix = ww.topk(5)
                    A_[ti_][i] = tix.cpu().numpy(); A_[tu_][i] = (tv / ww.sum().clamp_min(1e-30)).float().cpu().numpy()
            A_['L'][i] = Li_b
            A_['pv'][i] = PVT; A_['zj'][i] = kt['Zt'][tok].float().cpu().numpy(); A_['zbar'][i] = ZBAR_i
        if postep is not None:
            postep(i + 1, K, f'pułapka {opis_tok}: checkpoint {i + 1}/{K} (krok {k}), {time.time() - t0:.0f} s')
    czas = time.time() - t0
    for tok in licz:
        A_ = T[tok]
        wynik = dict(wersja=WERSJA, tok=tok, G=G, zasieg=float(zasieg), kroki=np.asarray(kroki, np.int64), r99=np.asarray(h.r99, np.float32), **A_)
        p = plik(h, tok, G, zasieg)
        tmp = p.with_name(p.stem + '_tmp.npz')
        np.savez(tmp, **wynik)
        tmp.replace(p)
        (p.with_suffix('.json')).write_text(json.dumps(dict(tok=tok, G=G, zasieg=zasieg, K=K, czas_s=round(czas), razem_z=licz,
                                                            dokladnosc=dokladnosc,
                                                            blad_z_maks=float(A_['blad_z'].max()), blad_pos_maks=float(A_['blad_pos'].max()),
                                                            blad_skok_maks=float(A_['blad_skok'].max()), zbiezne=float(A_['zb'].mean())),
                                                       ensure_ascii=False, indent=1), encoding='utf-8')
        wyniki[tok] = wynik
    return wyniki


def gotowe(h):
    """Pułapki policzone na dysku dla historii h: lista (tok, G, zasięg) z plików ogniskowa_t*_G*_z*.npz (bez sprawdzania wersji)."""
    import re
    out = []
    for p in sorted(Path(h.katalog).glob('ogniskowa_t*_G*_z*.npz')):
        m = re.fullmatch(r'ogniskowa_t(\d+)_G(\d+)_z(\d+)', p.stem)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)) / 100.0))
    return out


def opis_lambda(lam):
    """Słowny charakter pułapki z 3 wartości własnych (ściąga / wypycha / obraca)."""
    s = []
    lam = np.asarray(lam)
    uz = np.zeros(3, bool)
    for a in range(3):
        if uz[a]:
            continue
        l = lam[a]
        if abs(l.imag) > 1e-9 * max(abs(l), 1e-30):
            b = next((c for c in range(a + 1, 3) if not uz[c] and abs(lam[c] - np.conj(l)) <= 1e-6 * max(abs(l), 1e-30)), None)
            if b is not None:
                uz[b] = True
            s.append(f'obraca {abs(l.imag):.2f} ({"zwija" if l.real < 0 else "rozwija"} {l.real:+.2f})'.replace('.', ','))
        else:
            s.append((f'ściąga {l.real:+.2f}' if l.real < 0 else f'wypycha {l.real:+.2f}').replace('.', ','))
        uz[a] = True
    return ', '.join(s)
