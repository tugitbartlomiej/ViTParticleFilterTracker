"""Pułapka optyczna z laserów (ogniskowa wiązek) wokół wybranego koloidu: zakładka „Lasery i koloidy”, okienko 11 i sfera.

Koloid j w checkpoincie k to stan x̃_j przed MLP bloku 6 (lasery_dane.policz); na sferze p_j = P3 (x̃_j − x̄), gdzie wiersze P3
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

import lasery_dane as LD

WERSJA = 4                                   # 2: wiązki cząsteczki (P3 v_c, z_jc, z̄_c); 3: kontrola skoku, G = 1 = tylko wiązki; 4: pole tylko w komórce tokenu


class Przerwano(Exception):
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
    """P3 jak w lasery_dane.policz: 3 główne kierunki znormowanych kolumn fc2 OSTATNIEGO checkpointu (float64)."""
    import torch
    st = torch.load(Path(wagi) / f'k{int(krok_ost):05d}.pt', map_location=dev, weights_only=True)
    net.load_state_dict(st.get('net', st), strict=True)
    V = net.bloki[-1].mlp.fc2.weight.detach().double()
    Vn = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)
    U_, _, _ = torch.linalg.svd(Vn - Vn.mean(1, keepdim=True), full_matrices=False)
    return U_[:, :3].T.contiguous()                                             # (3, 192) float64


def policz(h, tok, G=17, zasieg=1.0, postep=None, przerwij=None, urzadzenie='cuda'):
    """Pole wiązek i pułapka wokół koloidu `tok` we wszystkich checkpointach historii h (lasery_dane.Historia).
    Zwraca słownik tablic (i zapisuje go obok historii). postep(i, n, opis); przerwij() → True kończy (wyjątek Przerwano)."""
    return policz_wiele(h, [tok], G, zasieg, postep, przerwij, urzadzenie)[int(tok)]


def _jakobian(xj, u, P3, g_, be_, eps, W1, b1, PV):
    """∂f/∂u (3 × 3) analitycznie: f(u) = PV (GELU(W1 LN(xj + P3ᵀu) + b1) − z̄); pochodna LayerNorm, GELU (erf) i rzutu.
    Zamiast torch.autograd.functional.jacobian (3 przejścia wstecz z budową grafu w każdym kroku Newtona); zgodne z autogradem
    do błędu zaokrągleń float64 (testy/test_moduly.py)."""
    import math
    import torch
    x = xj + u @ P3
    n = x.shape[0]
    xc = x - x.mean(); sd = torch.sqrt((xc * xc).mean() + eps); xh = xc / sd
    a = (xh * g_ + be_) @ W1.T + b1
    Tc = P3 - P3.mean(1, keepdim=True)                                         # styczne: kierunki P3 w 192D
    dxh = (Tc - xh[None] * (Tc @ xh)[:, None] / n) / sd                         # pochodna LN wzdłuż każdego kierunku
    dg = 0.5 * (1 + torch.erf(a / math.sqrt(2))) + a * torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    dz = ((dxh * g_) @ W1.T) * dg                                               # (3, 768): ∂z/∂u_k
    return (dz @ PV.T).T                                                        # J[i, k] = ∂f_i/∂u_k


def policz_wiele(h, toki, G=17, zasieg=1.0, postep=None, przerwij=None, urzadzenie='cuda'):
    """Pułapki kilku koloidów naraz: wczytanie wag i przejście sieci są wspólne dla wszystkich tokenów w checkpoincie, osobno
    liczy się tylko pole w komórce każdego tokenu (Newton, siatka G³, wartości własne). Tokeny z gotowym plikiem są brane z dysku.
    Zwraca {tok: wynik} i zapisuje każdy wynik osobno (ten sam plik co policz dla jednego tokenu)."""
    import torch
    import torch.nn.functional as Fn
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
    M = LD.modul_modelu()
    net = M.MalyViT().to(dev).eval().requires_grad_(False)
    pam = {}
    net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))
    net.bloki[-1].mlp.act.register_forward_hook(lambda m, a, o: pam.__setitem__('z', o.detach()))
    net.norm.register_forward_hook(lambda m, a, o: pam.__setitem__('xa', a[0].detach()))         # stan PO bloku 6 (wejście końcowej LN)
    szukaj = G > 1                                                                # G = 1: tylko wiązki wybranej cząsteczki, bez pola i bez zera
    kroki = [int(k) for k in h.kroki]; K = len(kroki)
    P3 = _rzut_P3(net, dev, h.wagi, kroki[-1])
    rgb = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
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
    znaki = None
    from concurrent.futures import ThreadPoolExecutor
    pula = ThreadPoolExecutor(4)
    odczyt = lambda k: (lambda st: st.get('net', st))(torch.load(Path(h.wagi) / f'k{k:05d}.pt', map_location='cpu', weights_only=True))
    kolejka = [pula.submit(odczyt, k) for k in kroki[:6]]
    t0 = time.time()
    opis_tok = str(licz[0]) if len(licz) == 1 else f'{len(licz)} cząsteczek ({", ".join(map(str, licz))})'
    try:
        for i, k in enumerate(kroki):
            if przerwij is not None and przerwij():
                raise Przerwano()
            net.load_state_dict(kolejka.pop(0).result(), strict=True)
            if i + 6 < K:
                kolejka.append(pula.submit(odczyt, kroki[i + 6]))
            with torch.no_grad():
                net(rgb)
            xs = pam['x'][0].double()                                             # (600, 192) stan przed MLP bloku 6
            b6 = net.bloki[-1]
            g_, be_ = b6.norm2.weight.double(), b6.norm2.bias.double(); eps = float(b6.norm2.eps)
            W1, b1 = b6.mlp.fc1.weight.double(), b6.mlp.fc1.bias.double()
            W2 = b6.mlp.fc2.weight.double()
            PV = P3 @ W2                                                          # (3, 768): wiązki w rzucie sfery

            def jasnosci(X):
                m = X.mean(1, keepdim=True); v = ((X - m) ** 2).mean(1, keepdim=True)
                return Fn.gelu(((X - m) / torch.sqrt(v + eps) * g_ + be_) @ W1.T + b1)

            Zt = jasnosci(xs); zbar = Zt.mean(0)
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
                PV = P3 @ W2
            blad_pos = float(np.abs(pc.float().cpu().numpy() - h.pos[i]).max() / max(np.abs(h.pos[i]).max(), 1e-9))
            nv = PV.norm(dim=0)                                                   # |P3 v_c|
            PVT = PV.T.float().cpu().numpy(); ZBAR_i = zbar.float().cpu().numpy()
            for tok in licz:                                                      # osobno tylko komórka każdego tokenu
                A_ = T[tok]
                A_['blad_z'][i] = blad_z; A_['blad_pos'][i] = blad_pos
                Fw = (Zt[tok] - zbar) @ W2.T
                A_['blad_skok'][i] = float((Fw - Fm[tok]).norm() / Fm[tok].norm().clamp_min(1e-30))
                xj = xs[tok]
                odl = (pc - pc[tok]).norm(dim=1); odl[tok] = float('inf')            # komórka tokenu: połowa odległości do sąsiada
                sas = int(odl.argmin()); rk = 0.5 * float(odl[sas])
                Li = float(zasieg * rk); A_['r_kom'][i] = rk; A_['sasiad'][i] = sas

                def pole(U, xj=xj):                                               # U (n, 3) → f (n, 3), z − z̄ (n, 768)
                    dz = jasnosci(xj + U @ P3) - zbar
                    return dz @ PV.T, dz

                def jac(u, xj=xj):
                    return _jakobian(xj, u, P3, g_, be_, eps, W1, b1, PV)

                with torch.no_grad():
                    f0, dz0 = pole(torch.zeros(1, 3, dtype=torch.float64, device=dev))
                    # Newton od koloidu: f(u) = 0, tylko wewnątrz sześcianu komórki (|u_x|, |u_y|, |u_z| ≤ L)
                    u = torch.zeros(3, dtype=torch.float64, device=dev)
                    n0 = float(f0.norm()); zb = False
                    for _ in range(60 if szukaj else 0):
                        fu = pole(u[None])[0][0]
                        if float(fu.norm()) <= 1e-6 * max(n0, 1e-30):
                            zb = True; break
                        du = -torch.linalg.lstsq(jac(u), fu[:, None]).solution[:, 0]
                        if float(du.norm()) > 0.3 * Li:
                            du = du * (0.3 * Li / float(du.norm()))
                        u = u + du
                        if float(u.abs().max()) > Li:
                            break
                    fu = pole(u[None])[0][0]
                    if szukaj and not zb and float(fu.norm()) <= 1e-4 * max(n0, 1e-30):
                        zb = True
                    zb = zb and float(u.abs().max()) <= Li                           # zero liczy się tylko w komórce tokenu
                    c_off = torch.zeros(3, dtype=torch.float64, device=dev)          # siatka zawsze wokół koloidu, nie rośnie
                    U = c_off + SIATKA * Li
                    f, dz = pole(U)
                    w = dz.abs() * nv
                    tw, ti = w.topk(3, dim=1)
                    suma = w.sum(1).clamp_min(1e-30)
                    mix = (tw[..., None] * kol[ti]).sum(1) / tw.sum(1, keepdim=True).clamp_min(1e-30)
                    fm = f.norm(dim=1)
                    A_['pole'][i, :, :3] = f.float().cpu().numpy(); A_['pole'][i, :, 3] = fm.float().cpu().numpy()
                    A_['kolor'][i, :, :3] = mix.float().cpu().numpy(); A_['kolor'][i, :, 3] = (tw[:, 0] / suma).float().cpu().numpy()
                    A_['fmed'][i] = float(fm.median())
                    A_['skok'][i] = f0[0].float().cpu().numpy()
                    A_['srodek'][i] = pc[tok].float().cpu().numpy(); A_['srodek_siatki'][i] = (pc[tok] + c_off).float().cpu().numpy()
                    if not zb:                                                        # zastępczo: minimum |f| na siatce
                        u = U[int(fm.argmin())].clone()
                    fu, dzu = pole(u[None])
                    A_['ogn'][i] = (pc[tok] + u).float().cpu().numpy(); A_['zb'][i] = zb; A_['fogn'][i] = float(fu.norm())
                    ev, evec = np.linalg.eig(jac(u).cpu().numpy())
                    o = np.argsort(ev.real); A_['lam'][i] = ev[o]; A_['wek'][i] = evec[:, o]
                    J0 = jac(torch.zeros(3, dtype=torch.float64, device=dev))
                    ev, evec = np.linalg.eig(J0.cpu().numpy())
                    o = np.argsort(ev.real); A_['lam0'][i] = ev[o]; A_['wek0'][i] = evec[:, o]
                    # liniowe oszacowanie pułapki z kształtu pola w koloidzie: u = −J₀⁻¹ f(0) (bez ograniczenia zasięgu)
                    A_['ogn_lin'][i] = (pc[tok] - torch.linalg.lstsq(J0, f0[0][:, None]).solution[:, 0]).float().cpu().numpy()
                    for (dzz, ti_, tu_) in ((dzu[0], 'top_ogn', 'top_ogn_u'), (dz0[0], 'top_kol', 'top_kol_u')):
                        ww = dzz.abs() * nv; tv, tix = ww.topk(5)
                        A_[ti_][i] = tix.cpu().numpy(); A_[tu_][i] = (tv / ww.sum().clamp_min(1e-30)).float().cpu().numpy()
                A_['L'][i] = Li
                A_['pv'][i] = PVT; A_['zj'][i] = Zt[tok].float().cpu().numpy(); A_['zbar'][i] = ZBAR_i
            if postep is not None:
                postep(i + 1, K, f'pułapka {opis_tok}: checkpoint {i + 1}/{K} (krok {k}), {time.time() - t0:.0f} s')
    finally:
        for f_ in kolejka:
            f_.cancel()
        pula.shutdown(wait=False)
    czas = time.time() - t0
    for tok in licz:
        A_ = T[tok]
        wynik = dict(wersja=WERSJA, tok=tok, G=G, zasieg=float(zasieg), kroki=np.asarray(kroki, np.int64), r99=np.asarray(h.r99, np.float32), **A_)
        p = plik(h, tok, G, zasieg)
        tmp = p.with_name(p.stem + '_tmp.npz')
        np.savez(tmp, **wynik)
        tmp.replace(p)
        (p.with_suffix('.json')).write_text(json.dumps(dict(tok=tok, G=G, zasieg=zasieg, K=K, czas_s=round(czas), razem_z=licz,
                                                            blad_z_maks=float(A_['blad_z'].max()), blad_pos_maks=float(A_['blad_pos'].max()),
                                                            blad_skok_maks=float(A_['blad_skok'].max()), zbiezne=float(A_['zb'].mean())),
                                                       ensure_ascii=False, indent=1), encoding='utf-8')
        wyniki[tok] = wynik
    return wyniki


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
