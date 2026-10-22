"""Rysunek techniczny reflektora c (okienko 13): geometria odcinków, punktów i napisów z wyniku reflector_mechanics.policz.

Trzy zespoły na wspólnej płycie: celownik w_c (lewo), igła pchnięcia v_c (środek), suwak progu b_c na śrubie (prawo).
Każdy przegub ma własny układ lokalny (reflector_mechanics._uklad): e₁ = kierunek części w spoczynku → oś pionowa rysunku,
e₂ = kierunek wykonanego obrotu → oś X, e₃ = reszta odrzutu → oś Y. Kąt obrotu jest prawdziwy, pomnożony przez przesadę ×N.
u ∈ [0, 1] = postęp między checkpointem i a następnym: część obraca się o u · (wykonany ruch) · N.
Linki: wkład tokenu zdjęcia w odrzut (jasność × siła pola celu), narysowany w swoim prawdziwym kierunku w układzie lokalnym.
"""
from __future__ import annotations

import numpy as np

KOL = dict(rama=(0.55, 0.62, 0.7, 0.9), slabe=(0.5, 0.56, 0.62, 0.35), igla=(1.0, 0.85, 0.35, 1.0), cel=(0.75, 0.6, 1.0, 1.0),
           prog=(0.4, 0.9, 1.0, 1.0), ruch=(0.35, 1.0, 0.5, 1.0), odrzut=(1.0, 0.35, 0.3, 1.0), sprezyna=(1.0, 0.6, 0.2, 1.0),
           narz=(0.29, 0.94, 1.0), tlo=(0.75, 0.78, 0.82), wybrany=(1.0, 0.9, 0.2))


def _rys(e):
    """Wektor w układzie lokalnym (e1, e2, e3) → oś rysunku (X = e2, Y = e3, Z = e1)."""
    e = np.asarray(e, float)
    return np.array([e[1], e[2], e[0]])


class ReflectorSchematic:
    def __init__(self):
        self.seg, self.pkt, self.nap = [], [], []

    def linia(self, a, b, kol, szer=1.5):
        self.seg.append((np.asarray(a, float), np.asarray(b, float), kol, szer))

    def lamana(self, P, kol, szer=1.5, przerywana=False):
        for k in range(len(P) - 1):
            if not przerywana or k % 2 == 0:
                self.linia(P[k], P[k + 1], kol, szer)

    def okrag(self, c, n1, n2, r, kol, szer=1.5, n=48):
        t = np.linspace(0, 2 * np.pi, n + 1)
        self.lamana(np.asarray(c) + r * (np.cos(t)[:, None] * n1 + np.sin(t)[:, None] * n2), kol, szer)

    def strzalka(self, a, b, kol, szer=2.0, grot=0.28):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a; L = np.linalg.norm(d)
        if L < 1e-6:
            return
        self.linia(a, b, kol, szer)
        t = d / L; n = np.cross(t, [0, 0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0, 0]); n /= np.linalg.norm(n)
        for s in (1, -1):
            self.linia(b, b - grot * L * t + s * 0.5 * grot * L * n, kol, szer)

    def punkt(self, p, kol, rozm):
        self.pkt.append((np.asarray(p, float), kol, rozm))

    def napis(self, p, tekst, kol=(220, 225, 230, 255)):
        self.nap.append((np.asarray(p, float), tekst, kol))

    def sprezyna(self, a, b, r, zwoje, kol, szer=1.4):
        a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = np.linalg.norm(d); t = d / max(L, 1e-9)
        n1 = np.cross(t, [0, 1.0, 0] if abs(t[1]) < 0.9 else [1.0, 0, 0]); n1 /= np.linalg.norm(n1); n2 = np.cross(t, n1)
        s = np.linspace(0, 1, zwoje * 16 + 1); f = 2 * np.pi * zwoje * s
        self.lamana(a + s[:, None] * d + r * (np.cos(f)[:, None] * n1 + np.sin(f)[:, None] * n2), kol, szer)

    def rura(self, a, b, r, kol, szer=1.2, n=6):
        """Rura: n tworzących i dwa pierścienie."""
        a, b = np.asarray(a, float), np.asarray(b, float); t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n1 = np.cross(t, [0, 1.0, 0] if abs(t[1]) < 0.9 else [1.0, 0, 0]); n1 /= np.linalg.norm(n1); n2 = np.cross(t, n1)
        for k in range(n):
            f = 2 * np.pi * k / n; o = r * (np.cos(f) * n1 + np.sin(f) * n2)
            self.linia(a + o, b + o, kol, szer)
        self.okrag(a, n1, n2, r, kol, szer, 24); self.okrag(b, n1, n2, r, kol, szer, 24)


def _podstawa(S, x0, wys=6.0):
    """Kolumna i płytka pod przegubem w (x0, 0, 0)."""
    for dx, dy in ((-0.35, -0.35), (0.35, -0.35), (0.35, 0.35), (-0.35, 0.35)):
        S.linia((x0 + dx, dy, -wys), (x0 + dx, dy, -1.9), KOL['rama'], 1.2)
    P = np.array([[-1.6, -1.6], [1.6, -1.6], [1.6, 1.6], [-1.6, 1.6], [-1.6, -1.6]])
    S.lamana(np.c_[x0 + P[:, 0], P[:, 1], np.full(5, -wys)], KOL['rama'], 1.6)


def _przegub(S, x0, kier):
    """Zawieszenie kardanowe: pierścień zewnętrzny stały (płaszczyzna X–Z), wewnętrzny obraca się z częścią (oś obrotu = Y)."""
    c = np.array([x0, 0.0, 0.0])
    S.okrag(c, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 1.9, KOL['rama'], 1.6)
    S.linia(c + (0, -1.9, 0), c + (0, -2.4, 0), KOL['rama'], 2.0); S.linia(c + (0, 1.9, 0), c + (0, 2.4, 0), KOL['rama'], 2.0)
    S.okrag(c, kier, np.array([0, 1.0, 0]), 1.5, KOL['rama'], 1.2)


def _linki(S, tip, V, li, n_max, pelne=True):
    """Linki tokenów: od kotwicy (kierunek siły) do końca części; grubość i krycie ~ siła."""
    if li is None:
        return
    nv = np.linalg.norm(V, axis=1); m = max(float(nv.max()), 1e-30)
    porz = np.argsort(-nv)
    for k in list(porz[:n_max]) + [len(V) - 1]:
        t = int(li['tokeny'][k]); wyb = t == int(li['tok_wybrany'])
        if nv[k] <= 0 or (k == len(V) - 1 and not wyb and k in porz[:n_max]):
            continue
        d = _rys(V[k]) / nv[k]; s = nv[k] / m
        kol = KOL['wybrany'] if wyb else KOL['narz'] if li['narz'][k] else KOL['tlo']
        kotwica = tip + (4.0 + 2.5 * s) * d
        S.linia(kotwica, tip, (*kol, 0.25 + 0.75 * s if not wyb else 1.0), 0.8 + 2.6 * s + (1.0 if wyb else 0))
        S.punkt(kotwica, (*kol, 1.0), 5 + 5 * s + (4 if wyb else 0))
        if pelne and (wyb or k in porz[:4]):
            S.napis(kotwica + d * 0.6, f't{t}', tuple(int(255 * c) for c in kol) + (255,))


def schemat(d, u, przes, n_linek=20, celownik=True, prog=True, linki=True):
    """Geometria całego reflektora w chwili u (0..1 między checkpointami) z przesadą ×przes."""
    S = ReflectorSchematic()
    ig, ce, pr, li = d['igla'], d['cel'], d['prog'], d['linki']
    if li is not None:
        li = dict(li, tok_wybrany=d['tok'])
    # płyta i siatka podłoża
    for x in np.arange(-20, 21, 2.5):
        S.linia((x, -7, -6), (x, 7, -6), (0.3, 0.36, 0.42, 0.25), 1.0)
    for y in np.arange(-7, 7.1, 2.5):
        S.linia((-20, y, -6), (20, y, -6), (0.3, 0.36, 0.42, 0.25), 1.0)

    def zespol(x0, cz, dl, kol, nazwa, rura_r, koncowka):
        _podstawa(S, x0)
        fi = np.radians(cz['kat']) * przes
        kier = lambda f: np.array([np.sin(f), 0.0, np.cos(f)])
        c = np.array([x0, 0.0, 0.0])
        _przegub(S, x0, kier(fi * u))
        n_t = kier(fi * u); tip = c + dl * n_t
        S.linia(c, c + dl * kier(0.0), KOL['slabe'], 1.0)                                         # spoczynek
        S.lamana(c + dl * np.array([kier(f) for f in np.linspace(0, fi, 16)]), (*KOL['ruch'][:3], 0.9), 1.3, True)
        if u > 0:                                                                                  # przebyta droga końca
            S.lamana(c + np.array([kier(f) for f in np.linspace(0.0, fi * u, 12)]) * dl, KOL['ruch'], 2.6)
        S.linia(c, c + dl * kier(fi), (*KOL['ruch'][:3], 0.55), 1.0)                              # cel ruchu
        S.rura(c, c + 0.55 * dl * n_t, rura_r, kol, 1.4)                                          # część ruchoma
        S.rura(c + 0.5 * dl * n_t, tip, 0.6 * rura_r, kol, 1.2)
        koncowka(tip, n_t)
        S.sprezyna(c + 0.15 * dl * n_t, c + 0.45 * dl * n_t, rura_r + 0.35, 6, KOL['sprezyna'], 1.3)   # weight decay
        f3 = _rys(cz['f3']); nf = np.linalg.norm(f3)
        if nf > 0:
            S.strzalka(tip, tip + 3.2 * f3 / nf, KOL['odrzut'], 2.6)
        S.napis(c + (0, 0, -6.6), nazwa)
        return tip

    if celownik:
        def kon_cel(tip, n_t):
            n1 = np.cross(n_t, [0, 1.0, 0]); n1 /= np.linalg.norm(n1); n2 = np.cross(n_t, n1)
            S.okrag(tip, n1, n2, 0.7, KOL['cel'], 1.6); S.linia(tip - 0.9 * n1, tip + 0.9 * n1, KOL['cel'], 1.2)
            S.linia(tip - 0.9 * n2, tip + 0.9 * n2, KOL['cel'], 1.2)
        tip_w = zespol(-13.0, ce, 5.5, KOL['cel'], 'celownik w_c (na co reaguje)', 0.32, kon_cel)
        if linki and li is not None:
            _linki(S, tip_w, li['w3'], li, max(n_linek // 2, 4), pelne=False)

    def kon_igla(tip, n_t):
        S.punkt(tip, KOL['igla'], 12)
    tip_v = zespol(0.0, ig, 7.0, KOL['igla'], 'igła pchnięcia v_c (w którą stronę pcha)', 0.3, kon_igla)
    if linki and li is not None:
        _linki(S, tip_v, li['v3'], li, n_linek)
    S.napis(tip_v + (0.4, 0, 0.9), 'odrzut', tuple(int(255 * c) for c in KOL['odrzut']))
    S.napis(np.array([0.0, 0, 7.0 * 0.3]) + (1.0, 0, 0), 'sprężyna wd', tuple(int(255 * c) for c in KOL['sprezyna']))

    if prog:
        x0 = 13.0
        _podstawa(S, x0, 6.0)
        S.sprezyna((x0, 0, -5.5), (x0, 0, 5.5), 0.28, 22, KOL['rama'], 1.0)                     # śruba
        S.linia((x0, 0, -5.8), (x0, 0, 5.8), KOL['rama'], 1.6)
        zb = lambda b: float(np.clip(b * 12.0, -5.2, 5.2))
        z0, z1 = zb(pr['b']), zb(pr['b'] + pr['db'] * przes)
        zt = z0 + (z1 - z0) * u
        for z, kol, sz in ((z0, KOL['slabe'], 1.0), (zt, KOL['prog'], 2.2)):
            P = np.array([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8], [-0.8, -0.8]])
            S.lamana(np.c_[x0 + P[:, 0], P[:, 1], np.full(5, z)], kol, sz)
        S.lamana(np.array([[x0 + 1.1, 0, z0], [x0 + 1.1, 0, z1]]), KOL['ruch'], 2.4)
        if pr['fb'] != 0:
            S.strzalka((x0 - 1.3, 0, zt), (x0 - 1.3, 0, zt + 2.0 * np.sign(pr['fb'])), KOL['odrzut'], 2.4)
        for z in range(-5, 6):
            S.linia((x0 - 0.25, -0.3, z), (x0 + 0.25, -0.3, z), KOL['slabe'], 1.0)
        S.napis((x0, 0, -6.6), 'próg b_c (od kiedy świeci); bez sprężyny')
    return S
