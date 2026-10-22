"""Dolina pewności i IoU wybranej cząsteczki (okienko 12 i sfera zakładki „Lasery i koloidy”).

Położenie cząsteczki na sferze to rzut 3D stanu x̃_j (192 wymiary) przed MLP bloku 6: p_j = P3 (x̃_j − x̄) (lasery_dane.policz).
Ze stanu x̃ odpowiedź głowy wynika dokładnie i bez udziału innych tokenów (uwaga bloku 6 już zadziałała, dalej każdy token osobno):
    x = x̃ + MLP₆(LN₂(x̃))          (768 laserów bloku 6)
    o = głowa(LN(x)) = (logit, ln l, ln t, ln r, ln b)
pewność p = σ(logit), ramka z komórki tokenu j: (cx − 16 l, cy − 16 t, cx + 16 r, cy + 16 b), IoU z prawdziwą ramką.

Dolina = zbiór położeń x̃, w których p ≥ p_min i IoU ≥ IoU_min. Jakobian J = ∂o/∂x̃ (5 × 192) ma rząd 5: głowa czyta z położenia
tylko 5 kierunków (ściany doliny), a w pozostałych 187 (jądro J, dno doliny) odpowiedź z bliska się nie zmienia. Dno nie jest płaskie:
LayerNorm i GELU je zakrzywiają, więc daleko od x̃_j odpowiedź jednak się zmienia.

Rysujemy przekrój 3D przez x̃_j: x̃_j + a e₁ + b e₂ + c e₃ na siatce G³ (a, b, c ∈ [−Z, Z]) w jednej z baz (BAZY):
  'sfera'  osie sfery (wiersze P3): przekrój leży dokładnie w rysowanym rzucie, punkt siatki ma na sferze położenie p_j + (a, b, c);
  'sciany' 3 najsilniejsze kierunki czytane przez głowę (prawe wektory osobliwe J): ściany doliny z bliska;
  'dno'    2 najsilniejsze kierunki czytane i 1 kierunek dna (oś sfery rzucona na jądro J): przekrój w poprzek doliny;
  'dno3'   3 kierunki dna (osie sfery rzucone na jądro J): z bliska nic się nie zmienia, z daleka widać zakrzywienie dna.
Dla baz innych niż 'sfera' sfera pokazuje cień przekroju: rzut P3 punktów siatki (M3 = P3 Eᵀ).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import lasery_dane as LD

BAZY = ('sfera', 'sciany', 'dno', 'dno3')
OPIS_BAZ = ('osie sfery (przekrój w rysowanym rzucie)', 'ściany: 3 kierunki czytane przez głowę',
            'w poprzek: 2 kierunki czytane + 1 kierunek dna', 'dno: 3 kierunki, których głowa nie czyta')


class Liczydlo:
    """Sieć na GPU dla jednej historii: stany x̃ checkpointów (z pamięcią podręczną) i przekroje doliny."""

    def __init__(self, h, urzadzenie='cuda'):
        import torch
        self.torch = torch
        self.h = h
        self.dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
        M = LD.modul_modelu()
        self.net = M.MalyViT().to(self.dev).eval().requires_grad_(False)
        self.pam = {}
        self.net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: self.pam.__setitem__('x', a[0].detach()))
        self.rgb = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(self.dev)
        self.gt = None if h.gt is None else torch.tensor(h.gt, device=self.dev, dtype=torch.float32)
        self.krok_wag = None
        self._X = {}
        self._poprz = {}
        self.P3 = self._rzut()

    def _wczytaj(self, krok):
        if self.krok_wag != int(krok):
            st = self.torch.load(Path(self.h.wagi) / f'k{int(krok):05d}.pt', map_location=self.dev, weights_only=True)
            self.net.load_state_dict(st.get('net', st), strict=True)
            self.krok_wag = int(krok)

    def _rzut(self):
        """P3 jak w lasery_dane.policz (3 główne kierunki znormowanych kolumn fc2 ostatniego checkpointu), znaki osi zgodne z h.pos."""
        torch = self.torch
        i = self.h.K - 1
        self._wczytaj(int(self.h.kroki[i]))
        V = self.net.bloki[-1].mlp.fc2.weight.detach().double()
        Vn = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)
        U_, _, _ = torch.linalg.svd(Vn - Vn.mean(1, keepdim=True), full_matrices=False)
        P3 = U_[:, :3].T.float()
        X = self.stan(i)
        pos =((X - X.mean(0, keepdim=True)) @ P3.T).cpu().numpy()
        ref = np.asarray(self.h.pos[i], np.float32)
        znak = np.sign((pos * ref).sum(0)); znak[znak == 0] = 1
        self.zgodnosc = float(1 - ((pos * znak - ref) ** 2).sum() / max((ref ** 2).sum(), 1e-12))   # R² względem zapisanych położeń
        return P3 * torch.as_tensor(znak, device=self.dev, dtype=P3.dtype)[:, None]

    def stan(self, i):
        """Stany x̃ (600, 192) przed MLP bloku 6 w checkpoincie i (pamięć ostatnich 48)."""
        i = int(i)
        if i not in self._X:
            self._wczytaj(int(self.h.kroki[i]))
            with self.torch.no_grad():
                self.net(self.rgb)
            if len(self._X) >= 48:
                self._X.pop(next(iter(self._X)))
            self._X[i] = self.pam['x'][0].float().clone()
        return self._X[i]

    def odpowiedz(self, xt):
        """xt (n, 192) stan przed MLP bloku 6 → o (n, 5) dokładnie jak sieć (MLP bloku 6, końcowa LN, głowa)."""
        blk = self.net.bloki[-1]
        x = xt + blk.ls2(blk.mlp(blk.norm2(xt)))
        return self.net.glowa(self.net.norm(x))

    def jakosc(self, o, tok):
        """Pewność p i IoU ramki tokenu tok (z jego komórki) z prawdziwą ramką (bez ramki: IoU = 1)."""
        torch = self.torch
        p = torch.sigmoid(o[:, 0])
        if self.gt is None:
            return p, torch.ones_like(p)
        cx, cy = ((tok % LD.W) + 0.5) * LD.KOMORKA, ((tok // LD.W) + 0.5) * LD.KOMORKA
        d = o[:, 1:].exp() * LD.KOMORKA
        g = self.gt
        x0 = torch.maximum(cx - d[:, 0], g[0]); y0 = torch.maximum(cy - d[:, 1], g[1])
        x1 = torch.minimum(cx + d[:, 2], g[2]); y1 = torch.minimum(cy + d[:, 3], g[3])
        I = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
        U = (d[:, 0] + d[:, 2]) * (d[:, 1] + d[:, 3]) + (g[2] - g[0]) * (g[3] - g[1]) - I
        return p, I / U.clamp_min(1e-9)

    def _jakosc_pkt(self, pts, tok, paczka=32768):
        torch = self.torch
        P, Q = [], []
        with torch.no_grad():
            for s in range(0, len(pts), paczka):
                p, q = self.jakosc(self.odpowiedz(pts[s:s + paczka]), tok)
                P.append(p); Q.append(q)
        return torch.cat(P), torch.cat(Q)

    def policz(self, i, tok, baza='sfera', G=40, zasieg=0.6):
        """Przekrój doliny w checkpoincie i dla cząsteczki tok. zasieg = połowa boku sześcianu jako ułamek promienia chmury r99[i]."""
        torch = self.torch
        i, tok, G = int(i), int(tok), int(G)
        X = self.stan(i); xj = X[tok]
        self._wczytaj(int(self.h.kroki[i]))                           # stany mogą być z pamięci: wagi głowy i MLP₆ z tego checkpointu
        J =torch.autograd.functional.jacobian(lambda x: self.odpowiedz(x[None])[0], xj)          # (5, 192)
        _, S, Vh = torch.linalg.svd(J, full_matrices=False)
        rzad = int((S > S[0] * 1e-6).sum())
        Qc = Vh[:rzad]                                                                             # kierunki czytane (wiersze)
        Np = torch.eye(192, device=self.dev) - Qc.T @ Qc                                           # rzut na jądro J (dno)
        dno = Np @ self.P3.T                                                                       # osie sfery rzucone na dno
        dno = torch.linalg.qr(dno)[0].T                                                            # (3, 192) ortonormalne
        czyt_sfery = (self.P3 @ Qc.T).pow(2).sum(1)                                                # ile każdej osi sfery widzi głowa
        if baza == 'sfera':
            E = self.P3.clone()
        elif baza == 'sciany':
            E = Vh[:3].clone()
        elif baza == 'dno':
            if (tok, baza) in self._poprz:                                                         # ciągłość: poprzedni kierunek dna
                n = Np @ self._poprz[(tok, baza)][2].to(self.dev)                                  # rzucony na nowe jądro J
            else:
                n = Np @ self.P3[int(torch.argmin(czyt_sfery))]                                    # oś sfery najmniej widoczna dla głowy
            E = torch.stack([Vh[0], Vh[1], n / n.norm().clamp_min(1e-12)])
        else:
            E = dno
        klucz = (tok, baza)
        if baza != 'sfera' and klucz in self._poprz:                                                # znaki osi: ciągłość w czasie
            Ep = self._poprz[klucz].to(self.dev)
            E = E * torch.sign((E * Ep).sum(1, keepdim=True) + 1e-30)
        self._poprz[klucz] = E.detach().cpu()
        r = float(self.h.r99[i]); Z = float(zasieg) * r
        a = torch.linspace(-Z, Z, G, device=self.dev)
        A, B, C = torch.meshgrid(a, a, a, indexing='ij')
        off = torch.stack([A.reshape(-1), B.reshape(-1), C.reshape(-1)], 1)                      # (G³, 3)
        p, q = self._jakosc_pkt(xj[None] + off @ E, tok)
        p0, q0 = self.jakosc(self.odpowiedz(xj[None]), tok)
        p0, q0 = float(p0[0]), float(q0[0])
        # zasięg doliny: jak daleko od x̃_j w losowych kierunkach dna i ścian jakość p·IoU spada do połowy
        g = torch.Generator(device=self.dev).manual_seed(0)
        ts = torch.linspace(0, 1.5 * r, 121, device=self.dev)
        zas = {}
        for nazwa, D in (('dno', torch.randn(48, 192, device=self.dev, generator=g) @ Np),
                         ('sciany', torch.randn(48, rzad, device=self.dev, generator=g) @ Qc)):
            D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
            pts = xj[None, None] + ts[None, :, None] * D[:, None]
            pp, qq = self._jakosc_pkt(pts.reshape(-1, 192), tok)
            jak = (pp * qq).reshape(len(D), len(ts)) < 0.5 * p0 * q0
            k = torch.where(jak.any(1), jak.float().argmax(1), torch.full((len(D),), len(ts) - 1, device=self.dev))
            zas[nazwa] = float(ts[k].median())
        pos0 = ((xj - X.mean(0)) @ self.P3.T).cpu().numpy()
        cpu = lambda t: t.detach().float().cpu().numpy()
        return dict(i=i, krok=int(self.h.kroki[i]), tok=tok, baza=baza, G=G, Z=Z, r99=r, zasieg=float(zasieg),
                    p=cpu(p).reshape(G, G, G), iou=cpu(q).reshape(G, G, G), p0=p0, iou0=q0,
                    E=cpu(E), M3=cpu(self.P3 @ E.T), pos0=pos0.astype(np.float32), sv=cpu(S), rzad=rzad,
                    czyt_osi=cpu((E @ Qc.T).pow(2).sum(1)), czyt_sfery=cpu(czyt_sfery), zas=zas, norma=float(xj.norm()),
                    zgodnosc=self.zgodnosc)


def pole_jakosci(d, p_min, iou_min):
    """Pole q ≥ 0 w dolinie: mniejszy z dwóch zapasów (pewności i IoU) względem progów, w ułamku drogi do 1."""
    a = (d['p'] - p_min) / max(1.0 - p_min, 1e-3)
    b = (d['iou'] - iou_min) / max(1.0 - iou_min, 1e-3)
    return np.minimum(a, b)


def obrys(d, p_min, iou_min):
    """Powierzchnia brzegu doliny (marching cubes pola jakości na poziomie 0, sześcian domknięty na brzegach przekroju).
    Zwraca (trójkąty (3n, 3) w układzie przekroju, normalne (3n, 3)) albo None, gdy w przekroju nie ma brzegu."""
    from wspolne import izopowierzchnia
    q = np.pad(pole_jakosci(d, p_min, iou_min), 1, constant_values=-1.0)
    if q.max() <= 0 or q.min() >= 0:
        return None
    G, Z = d['G'], d['Z']
    h_ = 2 * Z / (G - 1)
    try:
        v, f, n = izopowierzchnia(q, 0.0, h_, 'ascent')
    except (ValueError, RuntimeError):
        return None
    v = v - Z - h_                                                       # indeks 1 (po dopełnieniu) = −Z
    return v[f].reshape(-1, 3).astype(np.float32), -n[f].reshape(-1, 3).astype(np.float32)


def opis(d, p_min, iou_min):
    """Krótki opis przekroju do napisu okienka 12."""
    q = pole_jakosci(d, p_min, iou_min)
    nazwy = {'sfera': ('oś sfery 1', 'oś sfery 2', 'oś sfery 3'), 'sciany': ('ściana 1', 'ściana 2', 'ściana 3'),
             'dno': ('ściana 1', 'ściana 2', 'dno'), 'dno3': ('dno 1', 'dno 2', 'dno 3')}[d['baza']]
    czyt = ', '.join(f'{n} {c:.0%}' for n, c in zip(nazwy, d['czyt_osi']))
    pl = lambda x, k=2: f'{x:.{k}f}'.replace('.', ',')
    return (f'krok {d["krok"]}, cząsteczka {d["tok"]}: pewność {pl(d["p0"])}, IoU {pl(d["iou0"])};  głowa czyta {d["rzad"]} kierunków z 192, '
            f'dno ma {192 - d["rzad"]} wymiarów.  W sześcianie {(q >= 0).mean():.0%} położeń daje p ≥ {pl(p_min)} i IoU ≥ {pl(iou_min)}.  '
            f'Jakość spada o połowę po ~{d["zas"]["dno"]:.0f} wzdłuż dna i ~{d["zas"]["sciany"]:.0f} w poprzek '
            f'(promień chmury {d["r99"]:.0f}).  Ile osi czyta głowa: {czyt}.  Osie: czerwona 1, zielona 2, niebieska 3.')
