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
    import torch
    import torch.nn.functional as Fn
    d = wczytaj(h, tok, G, zasieg)
    if d is not None:
        return d
    tok, G = int(tok), int(G)
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
    POLE = np.zeros((K, N, 4), np.float16); KOLOR = np.zeros((K, N, 4), np.float16)
    SRODEK = np.zeros((K, 3), np.float32); SIATKA_S = np.zeros((K, 3), np.float32); OGN = np.zeros((K, 3), np.float32); ZB = np.zeros(K, bool)
    SKOK = np.zeros((K, 3), np.float32); L = np.zeros(K, np.float32); FMED = np.zeros(K, np.float32)
    FOGN = np.zeros(K, np.float32); LAM = np.zeros((K, 3), np.complex64); WEK = np.zeros((K, 3, 3), np.complex64)
    LAM0 = np.zeros((K, 3), np.complex64); WEK0 = np.zeros((K, 3, 3), np.complex64); OGN_LIN = np.zeros((K, 3), np.float32)
    TOPO = np.zeros((K, 5), np.int16); TOPO_U = np.zeros((K, 5), np.float32)
    PVK = np.zeros((K, 768, 3), np.float32); ZJ = np.zeros((K, 768), np.float32); ZBAR = np.zeros((K, 768), np.float32)
    TOPK = np.zeros((K, 5), np.int16); TOPK_U = np.zeros((K, 5), np.float32); BLAD = np.zeros(K, np.float32)
    POS_BLAD = np.zeros(K, np.float32); BLAD_SKOK = np.zeros(K, np.float32); RKOM = np.zeros(K, np.float32); SASIAD = np.zeros(K, np.int16)
    znaki = None
    from concurrent.futures import ThreadPoolExecutor
    pula = ThreadPoolExecutor(4)
    odczyt = lambda k: (lambda st: st.get('net', st))(torch.load(Path(h.wagi) / f'k{k:05d}.pt', map_location='cpu', weights_only=True))
    kolejka = [pula.submit(odczyt, k) for k in kroki[:6]]
    t0 = time.time()
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
            BLAD[i] = float((Zt.float() - pam['z'][0].float()).abs().max())          # zgodność z przejściem sieci
            # kontrola, że wiązki naprawdę przesuwają koloid: Σ_c (z_jc − z̄_c) v_c = (stan po bloku 6 − stan przed MLP), względem środka chmury, 192D
            Fm = pam['xa'][0].double() - xs; Fm = Fm - Fm.mean(0, keepdim=True)
            Fw = (Zt[tok] - zbar) @ W2.T
            BLAD_SKOK[i] = float((Fw - Fm[tok]).norm() / Fm[tok].norm().clamp_min(1e-30))
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
            POS_BLAD[i] = float(np.abs(pc.float().cpu().numpy() - h.pos[i]).max() / max(np.abs(h.pos[i]).max(), 1e-9))
            xj = xs[tok]
            r99 = float(h.r99[i])
            odl = (pc - pc[tok]).norm(dim=1); odl[tok] = float('inf')                # komórka tokenu: połowa odległości do sąsiada
            sas = int(odl.argmin()); rk = 0.5 * float(odl[sas])
            Li = float(zasieg * rk); RKOM[i] = rk; SASIAD[i] = sas
            nv = PV.norm(dim=0)                                                   # |P3 v_c|

            def pole(U):                                                          # U (n, 3) → f (n, 3), z − z̄ (n, 768)
                dz = jasnosci(xj + U @ P3) - zbar
                return dz @ PV.T, dz

            jac = lambda u: torch.autograd.functional.jacobian(lambda v: pole(v[None])[0][0], u)
            with torch.no_grad():
                f0, dz0 = pole(torch.zeros(1, 3, dtype=torch.float64, device=dev))
            # Newton od koloidu: f(u) = 0, tylko wewnątrz sześcianu komórki (|u_x|, |u_y|, |u_z| ≤ L)
            u = torch.zeros(3, dtype=torch.float64, device=dev)
            n0 = float(f0.norm()); zb = False
            for _ in range(60 if szukaj else 0):
                with torch.no_grad():
                    fu = pole(u[None])[0][0]
                if float(fu.norm()) <= 1e-6 * max(n0, 1e-30):
                    zb = True; break
                du = -torch.linalg.lstsq(jac(u), fu[:, None]).solution[:, 0]
                if float(du.norm()) > 0.3 * Li:
                    du = du * (0.3 * Li / float(du.norm()))
                u = u + du
                if float(u.abs().max()) > Li:
                    break
            with torch.no_grad():
                fu = pole(u[None])[0][0]
            if szukaj and not zb and float(fu.norm()) <= 1e-4 * max(n0, 1e-30):
                zb = True
            zb = zb and float(u.abs().max()) <= Li                               # zero liczy się tylko w komórce tokenu
            c_off = torch.zeros(3, dtype=torch.float64, device=dev)              # siatka zawsze wokół koloidu, nie rośnie
            with torch.no_grad():
                U = c_off + SIATKA * Li
                f, dz = pole(U)
                w = dz.abs() * nv
                tw, ti = w.topk(3, dim=1)
                suma = w.sum(1).clamp_min(1e-30)
                mix = (tw[..., None] * kol[ti]).sum(1) / tw.sum(1, keepdim=True).clamp_min(1e-30)
                fm = f.norm(dim=1)
                POLE[i, :, :3] = f.float().cpu().numpy(); POLE[i, :, 3] = fm.float().cpu().numpy()
                KOLOR[i, :, :3] = mix.float().cpu().numpy(); KOLOR[i, :, 3] = (tw[:, 0] / suma).float().cpu().numpy()
                FMED[i] = float(fm.median())
            SKOK[i] = f0[0].float().cpu().numpy()
            SRODEK[i] = pc[tok].float().cpu().numpy(); SIATKA_S[i] = (pc[tok] + c_off).float().cpu().numpy()
            if not zb:                                                            # zastępczo: minimum |f| na siatce
                u = U[int(fm.argmin())].clone()
            with torch.no_grad():
                fu, dzu = pole(u[None])
            OGN[i] = (pc[tok] + u).float().cpu().numpy(); ZB[i] = zb; FOGN[i] = float(fu.norm())
            ev, evec = np.linalg.eig(jac(u).cpu().numpy())
            o = np.argsort(ev.real); LAM[i] = ev[o]; WEK[i] = evec[:, o]
            J0 = jac(torch.zeros(3, dtype=torch.float64, device=dev))
            ev, evec = np.linalg.eig(J0.cpu().numpy())
            o = np.argsort(ev.real); LAM0[i] = ev[o]; WEK0[i] = evec[:, o]
            # liniowe oszacowanie pułapki z kształtu pola w koloidzie: u = −J₀⁻¹ f(0) (bez ograniczenia zasięgu)
            OGN_LIN[i] = (pc[tok] - torch.linalg.lstsq(J0, f0[0][:, None]).solution[:, 0]).float().cpu().numpy()
            for (dzz, TI, TU) in ((dzu[0], TOPO, TOPO_U), (dz0[0], TOPK, TOPK_U)):
                ww = dzz.abs() * nv; tv, tix = ww.topk(5)
                TI[i] = tix.cpu().numpy(); TU[i] = (tv / ww.sum().clamp_min(1e-30)).float().cpu().numpy()
            L[i] = Li
            PVK[i] = PV.T.float().cpu().numpy(); ZJ[i] = Zt[tok].float().cpu().numpy(); ZBAR[i] = zbar.float().cpu().numpy()
            if postep is not None:
                postep(i + 1, K, f'pułapka: checkpoint {i + 1}/{K} (krok {k}), {time.time() - t0:.0f} s')
    finally:
        for f_ in kolejka:
            f_.cancel()
        pula.shutdown(wait=False)
    wynik = dict(wersja=WERSJA, tok=tok, G=G, zasieg=float(zasieg), kroki=np.asarray(kroki, np.int64), pole=POLE, kolor=KOLOR,
                 srodek=SRODEK, srodek_siatki=SIATKA_S, ogn=OGN, zb=ZB, skok=SKOK, L=L, fmed=FMED, fogn=FOGN, lam=LAM, wek=WEK, lam0=LAM0, wek0=WEK0,
                 ogn_lin=OGN_LIN, r99=np.asarray(h.r99, np.float32), pv=PVK, zj=ZJ, zbar=ZBAR,
                 top_ogn=TOPO, top_ogn_u=TOPO_U, top_kol=TOPK, top_kol_u=TOPK_U, blad_z=BLAD, blad_pos=POS_BLAD, blad_skok=BLAD_SKOK,
                 r_kom=RKOM, sasiad=SASIAD)
    p = plik(h, tok, G, zasieg)
    tmp = p.with_name(p.stem + '_tmp.npz')
    np.savez(tmp, **wynik)
    tmp.replace(p)
    (p.with_suffix('.json')).write_text(json.dumps(dict(tok=tok, G=G, zasieg=zasieg, K=K, czas_s=round(time.time() - t0),
                                                        blad_z_maks=float(BLAD.max()), blad_pos_maks=float(POS_BLAD.max()), blad_skok_maks=float(BLAD_SKOK.max()),
                                                        zbiezne=float(ZB.mean())), ensure_ascii=False, indent=1), encoding='utf-8')
    return wynik


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
