# -*- coding: utf-8 -*-
"""Scena 6/7: paczka zdjec i rownanie Langevina (Manim CE 0.20.1)."""
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
ZIEL = "#5BC27A"
CZERW = "#E0565B"

FONT = "Segoe UI"
MONO = "Consolas"
TU = Path(__file__).resolve().parent

config.background_color = TLO

# Pomiar: odpowiedz latentu sondy na pchniecie jednej paczki (jednostka: skutek 1. kroku)
H = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
R = [0.98, 1.97, 2.72, 3.13, 3.22, 3.08, 2.86, 2.66, 2.40, 2.18, 1.68, 1.23, 1.02, 0.97, 0.78, 0.64]


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
        t = np.atleast_1d(np.asarray(t, float))
        k = np.clip(np.searchsorted(x, t) - 1, 0, n - 2)
        s = (t - x[k]) / h[k]
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]

    return f


def napis(s, size=24, color=TEKST, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def liczba(s, size=24, color=CIEMNY):
    return Text(s, font=MONO, font_size=size, color=color)


class Scena6(Scene):
    def do(self, t):
        """Czekaj do bezwzglednego czasu t (s)."""
        dt = t - self.renderer.time
        if dt > 0.02:
            self.wait(dt)

    def construct(self):
        random.seed(6)

        # ---------- naglowek (0-1 s) ----------
        krok = Text("KROK 6 / 7", font=MONO, font_size=22, color=CIEMNY)
        krok.to_corner(UL, buff=0.5)
        tytul = Text("Paczka zdjęć i równanie Langevina", font=FONT, font_size=40,
                     weight=BOLD, color=TEKST)
        tytul.next_to(krok, DOWN, buff=0.14, aligned_edge=LEFT)
        self.play(FadeIn(krok, shift=0.2 * RIGHT), FadeIn(tytul, shift=0.2 * RIGHT),
                  run_time=0.9, rate_func=rate_functions.ease_out_cubic)
        self.do(1.0)

        # ---------- FAZA A (1-10 s) ----------
        sciezki = sorted((TU / "miniatury").glob("m*.png"))[:32]
        tw, th, gap = 0.56, 0.373, 0.05
        lewy, gora = -6.6, 1.95
        mini = Group()
        cele = []
        for i, p in enumerate(sciezki):
            r, c = divmod(i, 8)
            im = ImageMobject(str(p))
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
            im.stretch_to_fit_width(tw)
            im.stretch_to_fit_height(th)
            cel = np.array([lewy + tw / 2 + c * (tw + gap), gora - th / 2 - r * (th + gap), 0])
            cele.append(cel)
            # start poza kadrem, rozrzucony
            kat = random.uniform(0, TAU)
            start = cel + np.array([np.cos(kat) * 9, np.sin(kat) * 6, 0])
            im.move_to(start).scale(1.6).rotate(random.uniform(-0.6, 0.6))
            mini.add(im)
        self.add(mini)
        anims = []
        for i, (im, cel) in enumerate(zip(mini, cele)):
            dst = ImageMobject(str(sciezki[i]))
            dst.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
            dst.stretch_to_fit_width(tw).stretch_to_fit_height(th).move_to(cel)
            im.target = dst
            anims.append(MoveToTarget(im, rate_func=rate_functions.ease_out_cubic))
        self.play(LaggedStart(*anims, lag_ratio=0.05, run_time=1.9))
        ramka = SurroundingRectangle(mini, color=LINIA, buff=0.1, stroke_width=2, corner_radius=0.08)
        etyk_paczka = napis("paczka: 32 zdjęcia", 26, TEKST)
        etyk_paczka.next_to(ramka, DOWN, buff=0.18)
        self.play(Create(ramka), FadeIn(etyk_paczka, shift=0.15 * UP), run_time=0.5,
                  rate_func=rate_functions.ease_out_sine)
        self.do(3.4)

        yc = gora - (4 * th + 3 * gap) / 2
        st1 = Arrow([-1.65, yc, 0], [-1.12, yc, 0], buff=0, color=CIEMNY, stroke_width=4,
                    max_tip_length_to_length_ratio=0.3)
        pudlo = RoundedRectangle(width=2.35, height=0.95, corner_radius=0.12,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=TEAL, stroke_width=2.5)
        pudlo.move_to([0.16, yc, 0])
        pudlo_t = napis("krok AdamW", 26, TEKST).move_to(pudlo)
        st2 = Arrow([1.42, yc, 0], [1.95, yc, 0], buff=0, color=CIEMNY, stroke_width=4,
                    max_tip_length_to_length_ratio=0.3)
        self.play(GrowArrow(st1), run_time=0.4, rate_func=rate_functions.ease_out_cubic)
        self.play(FadeIn(pudlo, scale=0.85), FadeIn(pudlo_t, scale=0.85), run_time=0.5,
                  rate_func=rate_functions.ease_out_back)
        self.play(GrowArrow(st2), run_time=0.4, rate_func=rate_functions.ease_out_cubic)

        # panel latentu
        pan = RoundedRectangle(width=4.4, height=2.75, corner_radius=0.15, fill_color=PANEL,
                               fill_opacity=1, stroke_color=LINIA, stroke_width=2)
        pan.move_to([4.25, yc, 0])
        siatka = VGroup()
        for k in range(1, 6):
            x = pan.get_left()[0] + k * 4.4 / 6
            siatka.add(Line([x, pan.get_bottom()[1] + 0.12, 0], [x, pan.get_top()[1] - 0.12, 0],
                            color=LINIA, stroke_width=1))
        for k in range(1, 4):
            y = pan.get_bottom()[1] + k * 2.75 / 4
            siatka.add(Line([pan.get_left()[0] + 0.12, y, 0], [pan.get_right()[0] - 0.12, y, 0],
                            color=LINIA, stroke_width=1))
        pan_t = napis("latent sondy", 24, CIEMNY)
        pan_t.next_to(pan.get_corner(UL), DR, buff=0.12)
        p0 = pan.get_center() + np.array([-1.45, -0.75, 0])
        kropka = Dot(p0, radius=0.11, color=TEKST)
        self.play(FadeIn(pan), Create(siatka), FadeIn(pan_t), run_time=0.6,
                  rate_func=rate_functions.ease_out_sine)
        self.play(GrowFromCenter(kropka), run_time=0.35, rate_func=rate_functions.ease_out_back)

        vS = np.array([1.75, 0.35, 0])
        vP = np.array([0.95, 1.0, 0])
        strS = Arrow(p0, p0 + vS, buff=0.0, color=POMAR, stroke_width=6,
                     max_tip_length_to_length_ratio=0.18)
        lS = MathTex("S", font_size=40, color=POMAR).next_to(strS.get_center(), DOWN, buff=0.18)
        self.play(GrowArrow(strS), FadeIn(lS), run_time=0.6, rate_func=rate_functions.ease_out_cubic)
        self.do(6.3)
        strP = Arrow(p0 + vS, p0 + vS + vP, buff=0.0, color=TEAL, stroke_width=6,
                     max_tip_length_to_length_ratio=0.22)
        lP = MathTex("P", font_size=40, color=TEAL).next_to(strP.get_center(), RIGHT, buff=0.15)
        self.play(GrowArrow(strP), FadeIn(lP), run_time=0.6, rate_func=rate_functions.ease_out_cubic)
        suma = DashedLine(p0, p0 + vS + vP, color=TEKST, stroke_width=2.5, dash_length=0.1)
        duch = Dot(p0, radius=0.09, color=CIEMNY, fill_opacity=0.6)
        self.add(duch)
        self.play(Create(suma), kropka.animate.move_to(p0 + vS + vP), run_time=0.8,
                  rate_func=rate_functions.ease_in_out_cubic)

        rown = MathTex(r"\Delta z(t+1)", "=", "P(t)", "+", "S(t)", font_size=44, color=TEKST)
        rown[2].set_color(TEAL)
        rown[4].set_color(POMAR)
        rown.move_to([0, -1.75, 0])
        self.play(Write(rown), run_time=0.9, rate_func=rate_functions.ease_in_out_sine)
        leg1 = napis("P: pęd, pamięć poprzednich kroków", 24, TEKST, t2c={"P": TEAL})
        leg2 = napis("S: pchnięcie bieżącej paczki", 24, TEKST, t2c={"S": POMAR})
        leg1.next_to(rown, DOWN, buff=0.35)
        leg2.next_to(leg1, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(FadeIn(leg1, shift=0.1 * UP), run_time=0.4, rate_func=rate_functions.ease_out_sine)
        self.play(FadeIn(leg2, shift=0.1 * UP), run_time=0.4, rate_func=rate_functions.ease_out_sine)
        self.do(10.0)

        # ---------- FAZA B (10-27 s) ----------
        fazaA = Group(mini, ramka, etyk_paczka, st1, pudlo, pudlo_t, st2, pan, siatka, pan_t,
                      kropka, duch, strS, lS, strP, lP, suma, rown, leg1, leg2)
        self.play(FadeOut(fazaA, shift=0.2 * DOWN), run_time=0.7, rate_func=rate_functions.ease_in_sine)

        ax = Axes(x_range=[0, 50, 10], y_range=[0, 4.5, 1], x_length=11.2, y_length=3.55,
                  tips=True,
                  axis_config={"color": CIEMNY, "stroke_width": 2.5, "include_ticks": True,
                               "tick_size": 0.06},
                  x_axis_config={"include_tip": True, "tip_length": 0.18, "tip_width": 0.14},
                  y_axis_config={"include_tip": True, "tip_length": 0.18, "tip_width": 0.14})
        ax.shift(np.array([-5.7, -1.85, 0]) - ax.c2p(0, 0))
        prowadnice = VGroup(*[
            Line(ax.c2p(0, v), ax.c2p(50, v), color=LINIA, stroke_width=1.5) for v in (1, 2, 3, 4)
        ])
        tx = VGroup(*[liczba(str(v), 24).next_to(ax.c2p(v, 0), DOWN, buff=0.15) for v in range(0, 51, 10)])
        ty = VGroup(*[liczba(str(v), 24).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in range(0, 5)])
        xl = napis("kroki po podaniu paczki", 24, CIEMNY)
        xl.next_to(tx, DOWN, buff=0.08).align_to(ax.c2p(50, 0), RIGHT)
        yl = napis("skutek / pierwszy krok", 24, CIEMNY)
        yl.next_to(ax.c2p(0, 4.5), UP, buff=0.2).align_to(ty, LEFT)
        self.play(Create(ax), FadeIn(prowadnice), run_time=1.0, rate_func=rate_functions.ease_in_out_cubic)
        self.play(FadeIn(tx), FadeIn(ty), FadeIn(xl), FadeIn(yl), run_time=0.5,
                  rate_func=rate_functions.ease_out_sine)

        # krzywa odniesienia: (1 - 0.9^h)/0.1, tylko do 4.5
        hmax = np.log(0.55) / np.log(0.9)
        ref = ax.plot(lambda h: (1 - 0.9 ** h) / 0.1, x_range=[0, hmax, hmax / 120],
                      color=CIEMNY, stroke_width=3)
        ref_d = DashedVMobject(ref, num_dashes=22, dashed_ratio=0.55)
        ref_t = napis("gdyby pęd tylko powtarzał pchnięcie", 24, CIEMNY)
        ref_t.next_to(ax.c2p(hmax, 4.1), RIGHT, buff=0.35)
        self.play(Create(ref_d), run_time=1.0, rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeIn(ref_t, shift=0.1 * LEFT), run_time=0.4, rate_func=rate_functions.ease_out_sine)
        self.do(13.9)

        # krzywa zmierzona: PCHIP przez (0,0) i punkty pomiaru, rownomiernie po dlugosci luku
        f = pchip([0] + H, [0.0] + R)
        hh = np.linspace(0, 50, 3000)
        pts = np.array([ax.c2p(a, b) for a, b in zip(hh, f(hh))])
        dl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        s = np.linspace(0, dl[-1], 500)
        rs = np.stack([np.interp(s, dl, pts[:, k]) for k in range(3)], axis=1)
        krzywa = VMobject(color=POMAR, stroke_width=6)
        krzywa.set_points_as_corners(rs)
        kropki = VGroup(*[Dot(ax.c2p(a, b), radius=0.06, color=POMAR) for a, b in zip(H, R)])
        for d in kropki:
            d.set_opacity(0)
        czub = Dot(ax.c2p(0, 0), radius=0.1, color="#FFF4D6")
        poswiata = Dot(ax.c2p(0, 0), radius=0.22, color=POMAR, fill_opacity=0.35)

        def sledz(m):
            m.move_to(krzywa.get_end())

        def odkryj(m):
            xk = krzywa.get_end()[0]
            for d in kropki:
                if d.get_center()[0] <= xk + 1e-3:
                    d.set_opacity(1)

        czub.add_updater(sledz)
        poswiata.add_updater(sledz)
        kropki.add_updater(odkryj)
        self.add(kropki, poswiata, czub)
        self.play(Create(krzywa), run_time=3.5, rate_func=rate_functions.ease_in_out_sine)
        czub.clear_updaters()
        poswiata.clear_updaters()
        kropki.clear_updaters()
        for d in kropki:
            d.set_opacity(1)
        self.play(FadeOut(poswiata), czub.animate.scale(0.6).set_color(POMAR), run_time=0.4)
        self.do(18.0)

        # adnotacja 1: pierwsze 3 kroki pokrywaja sie z odniesieniem
        a1 = napis("przez 3 kroki pęd powtarza pchnięcie", 26, TEAL)
        a1.move_to(ax.c2p(0, 0.45)).align_to(ax.c2p(9, 0), LEFT)
        w1 = Line(a1.get_corner(UL) + np.array([-0.08, 0.02, 0]), ax.c2p(3, 2.72),
                  color=TEAL, stroke_width=2, buff=0.1)
        k1 = Circle(radius=0.13, color=TEAL, stroke_width=2.5).move_to(ax.c2p(3, 2.72))
        self.play(Create(w1), Create(k1), run_time=0.5, rate_func=rate_functions.ease_out_sine)
        self.play(FadeIn(a1, shift=0.1 * RIGHT), run_time=0.5, rate_func=rate_functions.ease_out_sine)
        self.do(20.6)

        # adnotacja 2: szczyt
        a2 = napis("szczyt: 3,2 w 5. kroku", 26, TEKST)
        a2.move_to(ax.c2p(0, 3.3)).align_to(ax.c2p(11, 0), LEFT)
        w2 = Line(ax.c2p(5, 3.22), a2.get_left() + np.array([-0.1, 0, 0]), color=TEKST,
                  stroke_width=2, buff=0.12)
        k2 = Circle(radius=0.15, color=TEKST, stroke_width=2.5).move_to(ax.c2p(5, 3.22))
        self.play(Create(k2), Create(w2), run_time=0.5, rate_func=rate_functions.ease_out_sine)
        self.play(FadeIn(a2, shift=0.1 * RIGHT), run_time=0.5, rate_func=rate_functions.ease_out_sine)
        self.do(23.0)

        # adnotacja 3: powrot i trwaly slad ~0,4
        a3 = napis("potem wraca: 0,64 po 50 krokach, ~0,4 po 500", 26, TEKST)
        a3.move_to(ax.c2p(0, 2.1)).align_to(ax.c2p(50, 0), RIGHT)
        w3 = Line(a3.get_bottom() + np.array([1.9, 0, 0]), ax.c2p(50, 0.64), color=TEKST,
                  stroke_width=2, buff=0.12)
        slad = DashedLine(ax.c2p(43, 0.4), ax.c2p(50, 0.4), color=ZIEL, stroke_width=3,
                          dash_length=0.1)
        slad_t = liczba("~0,4", 24, ZIEL).next_to(ax.c2p(50, 0.4), RIGHT, buff=0.12)
        self.play(Create(w3), run_time=0.4, rate_func=rate_functions.ease_out_sine)
        self.play(FadeIn(a3, shift=0.1 * LEFT), run_time=0.5, rate_func=rate_functions.ease_out_sine)
        self.play(Create(slad), FadeIn(slad_t), run_time=0.6, rate_func=rate_functions.ease_out_sine)
        self.do(25.6)

        zrodlo = VGroup(napis("pomiar: 180 sond, 2 ziarna,", 24, CIEMNY),
                        napis("kroki 1000–2500", 24, CIEMNY))
        zrodlo.arrange(DOWN, aligned_edge=RIGHT, buff=0.08)
        zrodlo.to_corner(DR, buff=0.5)
        self.play(FadeIn(zrodlo), run_time=0.5, rate_func=rate_functions.ease_out_sine)
        self.do(27.0)

        # ---------- wniosek (27-29.2 s) ----------
        wniosek = Text("Jedna paczka pcha mocno, ale krótko.", font=FONT, font_size=30,
                       weight=BOLD, color=POMAR)
        wniosek.to_corner(DL, buff=0.5)
        self.play(FadeIn(wniosek, shift=0.15 * UP), run_time=0.7, rate_func=rate_functions.ease_out_cubic)
        self.do(29.2)

        # ---------- wyciemnienie ----------
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8, rate_func=rate_functions.ease_in_sine)
