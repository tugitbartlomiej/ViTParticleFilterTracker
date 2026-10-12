# -*- coding: utf-8 -*-
"""Model 5/6: paczka, ped i powrot (Manim CE 0.20.1).

Lewa polowa: dolina straty (rzut 3D liczony recznie, kamera stoi), kulka = latent sieci.
Prawa polowa: zmierzona funkcja odpowiedzi R(h) na jedna paczke (dane_animacji.json).
Wychylenie kulki od polozenia wyjsciowego jest proporcjonalne do R(h).
"""
import json
import random
from pathlib import Path

import numpy as np
from manim import *

TLO = "#0E1A24"
PANEL = "#16293A"
LINIA = "#24425A"
TEKST = "#EAF2F5"
CIEMNY = "#7C93A3"
TEAL = "#2BB3C0"
POMAR = "#F08A2E"
POMAR_CIEMNY = "#B8601C"
POMAR_JASNY = "#F7B877"
ZIEL = "#5BC27A"

FONT = "Segoe UI"
MONO = "Consolas"
TU = Path(__file__).resolve().parent
EDIT = TU.parents[1]

config.background_color = TLO

# ---------------- dane ----------------
with open(EDIT / "dane_animacji.json", encoding="utf-8") as fh:
    _odp = json.load(fh)["odpowiedz_na_paczke"]
HOR = [0.0] + [float(v) for v in _odp["horyzonty"]]
RR = [0.0] + [float(v) for v in _odp["R"]]
R_500 = RR[-1]


def pchip(x, y):
    """Monotoniczna interpolacja Hermite'a (Fritsch-Carlson), bez przestrzalow."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    h = np.diff(x)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def koniec(h0, h1, d0, d1):
        mm = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(mm) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(mm) > abs(3 * d0):
            return 3 * d0
        return mm

    m[0] = koniec(h[0], h[1], d[0], d[1])
    m[-1] = koniec(h[-1], h[-2], d[-1], d[-2])

    def f(t):
        t = np.clip(np.atleast_1d(np.asarray(t, float)), x[0], x[-1])
        k = np.clip(np.searchsorted(x, t) - 1, 0, n - 2)
        s = (t - x[k]) / h[k]
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]

    return f


_f = pchip(HOR, RR)


def R(h):
    return float(_f(h)[0])


C_SPR = 0.11  # schematyczna sztywnosc sprezyny: S = -C * wychylenie


def pchniecie(h):
    """Schemat: P = dz - S, czyli pochodna odpowiedzi plus sila sprezyny (bez liczb na ekranie)."""
    e = 0.05
    a, b = max(0.0, h - e), h + e
    return max(0.0, (R(b) - R(a)) / (b - a) + C_SPR * R(h))


# ---------------- dolina (rzut 3D, kamera nieruchoma) ----------------
AZ = np.radians(12)
EL = np.radians(38)
SKALA = 1.2
SRODEK = np.array([-3.55, -1.55, 0.0])
X0 = -0.2          # polozenie kulki przed paczka (blisko dna)
K_WYCH = 0.5       # wychylenie (wsp. x doliny) na jednostke R
R_KULI = 0.17


def zf(x, y):
    return 0.28 * x**2 + 0.2 * y**2 + 0.045 * np.sin(2.1 * x + 0.4) * np.cos(1.7 * y)


def rzut(x, y, z):
    u = x * np.cos(AZ) - y * np.sin(AZ)
    d = x * np.sin(AZ) + y * np.cos(AZ)
    return SRODEK + SKALA * np.array([u, z * np.cos(EL) + d * np.sin(EL), 0.0])


def glebia(x, y, z):
    d = x * np.sin(AZ) + y * np.cos(AZ)
    return d * np.cos(EL) - z * np.sin(EL)


def zbuduj_doline():
    nr, nf, rmax = 22, 52, 2.35
    rs = np.linspace(0.0, rmax, nr + 1)
    fs = np.linspace(0.0, TAU, nf + 1)
    zmax = zf(rmax, 0.0)
    c_dol = np.array(color_to_rgb("#123A4D"))
    c_gora = np.array(color_to_rgb(TEAL))
    swiatlo = np.array([-0.45, -0.55, 1.0])
    swiatlo /= np.linalg.norm(swiatlo)
    sciany = []
    for i in range(nr):
        for j in range(nf):
            narozniki = []
            for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                x, y = rs[a] * np.cos(fs[b]), rs[a] * np.sin(fs[b])
                narozniki.append((x, y, zf(x, y)))
            xc = np.mean([p[0] for p in narozniki])
            yc = np.mean([p[1] for p in narozniki])
            zc = zf(xc, yc)
            e = 1e-3
            zx = (zf(xc + e, yc) - zf(xc - e, yc)) / (2 * e)
            zy = (zf(xc, yc + e) - zf(xc, yc - e)) / (2 * e)
            nrm = np.array([-zx, -zy, 1.0])
            nrm /= np.linalg.norm(nrm)
            cien = 0.55 + 0.5 * max(0.0, float(nrm @ swiatlo))
            t = np.clip(zc / zmax, 0, 1) ** 0.85
            kol = np.clip((c_dol * (1 - t) + c_gora * t) * cien, 0, 1)
            pts = [rzut(*p) for p in narozniki]
            pol = Polygon(*pts, stroke_width=0.7, stroke_color=TLO, stroke_opacity=0.35)
            pol.set_fill(rgb_to_color(kol), opacity=1.0)
            sciany.append((glebia(xc, yc, zc), pol))
    sciany.sort(key=lambda s: -s[0])
    return VGroup(*[p for _, p in sciany])


def punkt_kuli(xw):
    return rzut(xw, 0.0, zf(xw, 0.0)) + np.array([0.0, R_KULI * 0.9, 0.0])


def kierunek(xw):
    a, b = punkt_kuli(xw - 0.02), punkt_kuli(xw + 0.02)
    v = b - a
    return v / np.linalg.norm(v)


def napis(s, size=24, color=TEKST, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def liczba(s, size=24, color=CIEMNY):
    return Text(s, font=MONO, font_size=size, color=color)


def miniatury():
    """32 + 32 male kopie klatek z puli (cache w folderze slotu)."""
    from PIL import Image
    sciezki = [l.strip() for l in (EDIT / "klatki_puli.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    kat = TU / "miniatury"
    kat.mkdir(exist_ok=True)
    wyn = []
    for i, p in enumerate(sciezki):
        cel = kat / f"m{i:02d}.png"
        if not cel.exists():
            Image.open(p).convert("RGB").resize((144, 96), Image.LANCZOS).save(cel)
        wyn.append(cel)
    return wyn[:32], wyn[16:48]


class Model5(Scene):
    def do(self, t):
        dt = t - self.renderer.time
        if dt > 0.02:
            self.wait(dt)

    # ---------- stan ----------
    def h_akt(self):
        return self.hx.get_value() if self.faza == 1 else self.h2.get_value()

    def wych(self):
        """Wychylenie kulki w jednostkach R (od polozenia wyjsciowego)."""
        if self.faza == 1:
            return R(self.hx.get_value())
        return R_500 + R(self.h2.get_value())

    def x_kuli(self):
        drg = self.amp.get_value() * (0.05 * np.sin(6.1 * self.tj) + 0.035 * np.sin(9.7 * self.tj + 1.3)
                                      + 0.02 * np.sin(15.3 * self.tj + 0.4))
        return X0 + K_WYCH * self.wych() + drg

    def paczka(self, sciezki, srodek):
        tw, th, gap = 0.30, 0.20, 0.03
        grupa = Group()
        anims = []
        for i, p in enumerate(sciezki):
            r, c = divmod(i, 8)
            cel = srodek + np.array([(c - 3.5) * (tw + gap), (1.5 - r) * (th + gap), 0])
            im = ImageMobject(str(p))
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
            im.stretch_to_fit_width(tw).stretch_to_fit_height(th)
            kat = random.uniform(np.radians(160), np.radians(205))
            start = cel + np.array([np.cos(kat) * 5, np.sin(kat) * 0.6, 0])
            im.move_to(start)
            im.target = im.copy().move_to(cel)
            grupa.add(im)
            anims.append(MoveToTarget(im, rate_func=rate_functions.ease_out_cubic))
        return grupa, anims

    def construct(self):
        random.seed(5)
        self.hx = ValueTracker(0.0)
        self.h2 = ValueTracker(0.0)
        self.amp = ValueTracker(0.0)
        self.op_P = ValueTracker(0.0)
        self.op_ped = ValueTracker(0.0)
        self.op_S = ValueTracker(0.0)
        self.faza = 1
        self.tj = 0.0
        LS = 1.6  # dlugosc strzalek na jednostke

        # ---------- naglowek (0-1 s) ----------
        krok = Text("MODEL 5 / 6", font=MONO, font_size=22, color=CIEMNY).to_corner(UL, buff=0.5)
        tytul = Text("Paczka, pęd i powrót", font=FONT, font_size=40, weight=BOLD, color=TEKST)
        tytul.next_to(krok, DOWN, buff=0.14, aligned_edge=LEFT)
        self.play(FadeIn(krok, shift=0.2 * RIGHT), FadeIn(tytul, shift=0.2 * RIGHT),
                  run_time=0.9, rate_func=rate_functions.ease_out_cubic)
        self.do(1.0)

        # ---------- dolina ----------
        dolina = zbuduj_doline()
        self.play(FadeIn(dolina, shift=0.25 * UP), run_time=1.3, rate_func=rate_functions.ease_out_cubic)

        # kulka (latent)
        kula = VGroup(
            Circle(radius=R_KULI, stroke_color=TEKST, stroke_width=2).set_fill(TEKST, 1.0),
            Circle(radius=R_KULI * 0.38, stroke_width=0).set_fill(WHITE, 0.9).shift(
                np.array([-0.06, 0.06, 0])),
        )
        kula[0].set_fill(color=[TEKST, "#9FB6C2"])
        kula.move_to(punkt_kuli(X0))

        def ruch_kuli(m, dt):
            self.tj += dt
            m.move_to(punkt_kuli(self.x_kuli()))
        self.play(GrowFromCenter(kula), run_time=0.45, rate_func=rate_functions.ease_out_back)
        kula.add_updater(ruch_kuli)
        lat_t = napis("latent sieci", 24, CIEMNY)
        lat_t.next_to(punkt_kuli(X0), DOWN, buff=0.38)
        self.play(FadeIn(lat_t, shift=0.1 * UP), self.amp.animate.set_value(1.0), run_time=0.6,
                  rate_func=rate_functions.ease_out_sine)
        self.do(4.6)

        # ---------- wykres R(h) (prawa polowa) ----------
        panel = RoundedRectangle(width=6.45, height=5.55, corner_radius=0.15, fill_color=PANEL,
                                 fill_opacity=1, stroke_color=LINIA, stroke_width=2)
        panel.move_to([3.55, -0.55, 0])
        ax = Axes(x_range=[0, 50, 10], y_range=[0, 3.5, 1], x_length=5.15, y_length=3.9, tips=False,
                  axis_config={"color": CIEMNY, "stroke_width": 2.5, "include_ticks": True,
                               "tick_size": 0.06})
        ax.shift(np.array([1.05, -2.45, 0]) - ax.c2p(0, 0))
        prow = VGroup(*[Line(ax.c2p(0, v), ax.c2p(50, v), color=LINIA, stroke_width=1.5) for v in (1, 2, 3)])
        tx = VGroup(*[liczba(str(v), 24).next_to(ax.c2p(v, 0), DOWN, buff=0.14) for v in range(0, 51, 10)])
        ty = VGroup(*[liczba(str(v), 24).next_to(ax.c2p(0, v), LEFT, buff=0.14) for v in range(0, 4)])
        xl = napis("kroki po paczce", 24, CIEMNY)
        xl.next_to(tx, DOWN, buff=0.06).align_to(ax.c2p(50, 0), RIGHT)
        yl = napis("skutek / 1. krok", 24, CIEMNY)
        yl.next_to(ax.c2p(0, 3.5), UP, buff=0.2).align_to(ty, LEFT)
        self.play(FadeIn(panel), Create(ax), FadeIn(prow), run_time=0.9,
                  rate_func=rate_functions.ease_in_out_cubic)
        self.play(FadeIn(tx), FadeIn(ty), FadeIn(xl), FadeIn(yl), run_time=0.5,
                  rate_func=rate_functions.ease_out_sine)
        self.do(6.2)
        self.play(self.amp.animate.set_value(0.0), run_time=0.5, rate_func=rate_functions.ease_in_out_sine)
        # NIERUCHOME 1: kulka w dolinie przed paczka
        self.do(8.0)

        # ---------- paczka 1 ----------
        sc1, sc2 = miniatury()
        p1, anims = self.paczka(sc1, np.array([-3.55, 1.86, 0]))
        self.add(p1)
        self.play(FadeOut(lat_t, run_time=0.4),
                  LaggedStart(*anims, lag_ratio=0.04, run_time=1.4))
        self.do(9.75)
        cel = punkt_kuli(X0)
        uderz = [im.animate.move_to(cel).scale(0.12).set_opacity(0.0) for im in p1]
        self.play(LaggedStart(*uderz, lag_ratio=0.015, run_time=0.65, rate_func=rate_functions.ease_in_cubic))
        self.remove(p1)

        # strzalki (zawsze rysowane od kulki)
        def strzalka(dl, kier, kolor, op):
            if dl < 0.3 or op < 0.01:
                return VGroup()
            c = kula.get_center()
            a = c + kier * (R_KULI + 0.04)
            b = a + kier * dl
            return Arrow(a, b, buff=0, color=kolor, stroke_width=7, tip_length=0.22,
                         max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=40
                         ).set_opacity(op)

        def rysuj_P():
            return strzalka(LS * pchniecie(0.0), kierunek(self.x_kuli()), POMAR, self.op_P.get_value())

        def rysuj_ped():
            return strzalka(LS * pchniecie(self.h_akt()), kierunek(self.x_kuli()), POMAR_CIEMNY,
                            self.op_ped.get_value())

        def rysuj_S():
            return strzalka(LS * C_SPR * (R(self.h_akt())), -kierunek(self.x_kuli()), TEAL,
                            self.op_S.get_value())

        sP = always_redraw(rysuj_P)
        sPed = always_redraw(rysuj_ped)
        sS = always_redraw(rysuj_S)

        def nad(m, kier_znak, dl_fun, odst):
            k = kierunek(self.x_kuli())
            nrm = np.array([-k[1], k[0], 0])
            c = kula.get_center() + kier_znak * k * (R_KULI + 0.04 + 0.5 * dl_fun()) + nrm * odst
            m.move_to(c)

        lP = MathTex("P", font_size=44, color=POMAR)
        lP.add_updater(lambda m: nad(m, 1, lambda: LS * pchniecie(0.0), 0.38))
        lPed = napis("pęd", 26, POMAR)
        lPed.add_updater(lambda m: (nad(m, 1, lambda: LS * pchniecie(self.h_akt()), 0.36),
                                    m.set_opacity(self.op_ped.get_value()
                                                  * float(np.clip((LS * pchniecie(self.h_akt()) - 0.3) / 0.3, 0, 1)))))
        lS = MathTex("S", font_size=44, color=TEAL)
        lS.add_updater(lambda m: (nad(m, -1, lambda: LS * C_SPR * R(self.h_akt()), -0.36),
                                  m.set_opacity(self.op_S.get_value()
                                                * float(np.clip((LS * C_SPR * R(self.h_akt()) - 0.3) / 0.25, 0, 1)))))

        blysk = Circle(radius=R_KULI, color=POMAR, stroke_width=5).move_to(kula.get_center())
        self.add(sP)
        self.play(self.op_P.animate.set_value(1.0),
                  blysk.animate.scale(3.2).set_stroke(opacity=0.0),
                  run_time=0.45, rate_func=rate_functions.ease_out_cubic)
        self.remove(blysk)
        self.play(FadeIn(lP, scale=0.8), run_time=0.3, rate_func=rate_functions.ease_out_sine)
        # NIERUCHOME 2: chwila uderzenia (strzalka P)
        self.do(12.5)

        # slad polozenia wyjsciowego
        duch = Circle(radius=R_KULI, stroke_color=CIEMNY, stroke_width=2.5).move_to(punkt_kuli(X0))
        duch.set_stroke(opacity=0.8)
        self.add(sPed, sS, lS)
        lPed.set_opacity(0)
        self.add(lPed)
        self.play(self.op_P.animate.set_value(0.0), self.op_ped.animate.set_value(1.0),
                  FadeOut(lP), run_time=0.45, rate_func=rate_functions.ease_in_out_sine)
        self.remove(sP)

        # licznik krokow
        def rysuj_licznik():
            h = self.h_akt()
            return liczba(f"krok h = {int(round(h))}", 28, TEKST).move_to([4.35, 1.8, 0]).align_to(
                np.array([6.45, 0, 0]), RIGHT)
        licznik = always_redraw(rysuj_licznik)
        self.add(duch)
        self.play(FadeIn(licznik), FadeIn(duch), run_time=0.4, rate_func=rate_functions.ease_out_sine)

        # krzywa na zywo + kropka
        def rysuj_krzywa(tr, przes=0.0, kolor=POMAR, hmax_os=50.0):
            def f():
                hc = min(tr.get_value(), hmax_os - przes)
                if hc < 0.02:
                    return VMobject()
                hh = np.linspace(0, hc, max(3, int(hc * 12)))
                pts = [ax.c2p(przes + a, R(a)) for a in hh]
                m = VMobject(stroke_color=kolor, stroke_width=6)
                m.set_points_smoothly(pts) if len(pts) > 3 else m.set_points_as_corners(pts)
                return m
            return f

        krz1 = always_redraw(rysuj_krzywa(self.hx))
        kropka1 = always_redraw(lambda: Dot(ax.c2p(min(self.hx.get_value(), 50), R(min(self.hx.get_value(), 50))),
                                            radius=0.09, color="#FFF4D6"))
        self.add(krz1, kropka1)
        self.op_S.set_value(1.0)

        # ---------- kroki 0 -> 5: rozped, wjazd na zbocze ----------
        self.play(self.hx.animate.set_value(5.0), run_time=4.2, rate_func=rate_functions.ease_in_out_sine)

        # szczyt na wykresie
        k2 = Circle(radius=0.16, color=TEKST, stroke_width=2.5).move_to(ax.c2p(5, R(5)))
        a2 = napis("3,2 w 5. kroku", 26, TEKST)
        a2.next_to(ax.c2p(5, R(5)), RIGHT, buff=0.55).shift(0.05 * UP)
        self.play(Create(k2), run_time=0.35, rate_func=rate_functions.ease_out_sine)
        self.play(FadeIn(a2, shift=0.1 * RIGHT), run_time=0.4, rate_func=rate_functions.ease_out_sine)
        rown = MathTex(r"\Delta z", "=", "P", "+", "S", font_size=46, color=TEKST)
        rown[2].set_color(POMAR)
        rown[4].set_color(TEAL)
        rown.move_to([-3.55, -3.3, 0])
        self.play(FadeIn(rown, shift=0.1 * UP), run_time=0.5, rate_func=rate_functions.ease_out_sine)
        # NIERUCHOME 3: szczyt w 5. kroku
        self.do(20.3)

        # ---------- powrot 5 -> 50 ----------
        self.play(self.hx.animate.set_value(50.0), run_time=5.0, rate_func=rate_functions.ease_in_out_sine)
        # ogon 50 -> 500: kulka zjezdza do ~0,4
        slad = DashedLine(ax.c2p(20, 0.4), ax.c2p(50, 0.4), color=ZIEL, stroke_width=3, dash_length=0.1)
        slad_t = liczba("~0,4 po 500", 24, ZIEL)
        slad_t.next_to(ax.c2p(20, 0.4), UR, buff=0.1)
        self.play(Create(slad), self.hx.animate.set_value(500.0), run_time=1.3,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeIn(slad_t, shift=0.08 * UP), run_time=0.4, rate_func=rate_functions.ease_out_sine)
        # NIERUCHOME 4: po powrocie, krzywa cala
        self.do(28.6)

        # ---------- finał: druga paczka ----------
        krz1.clear_updaters()
        kropka1.clear_updaters()
        self.play(krz1.animate.set_stroke(opacity=0.35), FadeOut(kropka1), FadeOut(a2), FadeOut(k2),
                  FadeOut(slad_t), slad.animate.set_stroke(opacity=0.35), FadeOut(rown), FadeOut(licznik),
                  run_time=0.5, rate_func=rate_functions.ease_in_out_sine)
        # przelaczenie stanu na paczke 2 (to samo polozenie: R_500 + R(0))
        self.faza = 2
        self.h2.set_value(0.0)
        self.op_ped.set_value(0.0)
        p2, anims2 = self.paczka(sc2, np.array([-3.55, 1.86, 0]))
        self.add(p2)
        self.play(LaggedStart(*anims2, lag_ratio=0.03, run_time=1.1))
        cel2 = kula.get_center()
        uderz2 = [im.animate.move_to(cel2).scale(0.12).set_opacity(0.0) for im in p2]
        self.play(LaggedStart(*uderz2, lag_ratio=0.012, run_time=0.55, rate_func=rate_functions.ease_in_cubic))
        self.remove(p2)
        blysk2 = Circle(radius=R_KULI, color=POMAR, stroke_width=5).move_to(kula.get_center())
        self.play(self.op_ped.animate.set_value(1.0), blysk2.animate.scale(3.2).set_stroke(opacity=0.0),
                  run_time=0.35, rate_func=rate_functions.ease_out_cubic)
        self.remove(blysk2)

        P0 = 20.0
        krz2 = always_redraw(rysuj_krzywa(self.h2, przes=P0, kolor=POMAR_JASNY))
        kropka2 = always_redraw(lambda: Dot(ax.c2p(P0 + min(self.h2.get_value(), 50 - P0),
                                                   R(min(self.h2.get_value(), 50 - P0))),
                                            radius=0.09, color="#FFF4D6"))
        self.add(krz2, kropka2)
        self.play(self.h2.animate.set_value(30.0), run_time=3.6, rate_func=rate_functions.ease_in_out_sine)

        pas = Rectangle(width=ax.c2p(P0 + 20, 0)[0] - ax.c2p(P0, 0)[0],
                        height=ax.c2p(0, 3.5)[1] - ax.c2p(0, 0)[1],
                        stroke_width=0, fill_color=POMAR, fill_opacity=0.12)
        pas.move_to(ax.c2p(P0 + 10, 1.75))
        pas_t = liczba("~10-20 kroków", 24, POMAR_JASNY)
        pas_t.move_to([0, 1.95, 0]).align_to(np.array([6.45, 0, 0]), RIGHT)
        self.add(pas)
        pas.set_fill(opacity=0)
        self.bring_to_back(pas)
        self.bring_to_back(panel)
        self.play(pas.animate.set_fill(opacity=0.12), FadeIn(pas_t, shift=0.08 * UP), run_time=0.6,
                  rate_func=rate_functions.ease_out_sine)
        # NIERUCHOME 5: po drugiej paczce
        self.do(36.8)

        # ---------- wyciemnienie ----------
        for m in self.mobjects:
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6, rate_func=rate_functions.ease_in_sine)
