# -*- coding: utf-8 -*-
"""MODEL 2 / 6: wyznaczanie kierunku tozsamosci (LDA Fishera) i jego dryf (Manim CE 0.20.1)."""
import json
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
SZARY = "#9AA8B2"

FONT = "Segoe UI"
MONO = "Consolas"
TU = Path(__file__).resolve().parent
DANE = TU.parents[1] / "dane_animacji.json"

config.background_color = TLO

ease_out = rate_functions.ease_out_cubic
ease_io = rate_functions.ease_in_out_sine


def napis(s, size=24, color=TEKST, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def liczba(s, size=24, color=TEKST):
    return Text(s, font=MONO, font_size=size, color=color)


def pl(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


# ---------------- chmury schematyczne (ziarno stale) ----------------
def chmury():
    rng = np.random.default_rng(12)
    C0 = rot(np.radians(38)) @ np.diag([1.0, 0.30]) ** 2 @ rot(np.radians(38)).T
    C1 = rot(np.radians(30)) @ np.diag([0.55, 0.17]) ** 2 @ rot(np.radians(30)).T
    m0 = np.array([-0.35, -0.25])
    dl = np.array([np.cos(np.radians(38)), np.sin(np.radians(38))])
    pp = np.array([-dl[1], dl[0]])
    m1 = m0 + 1.1 * dl + 0.75 * pp
    X0 = rng.multivariate_normal(m0, C0, 150)
    X1 = rng.multivariate_normal(m1, C1, 40)
    srodek = np.vstack([X0, X1]).mean(0)
    return X0 - srodek, X1 - srodek


def fisher(X0, X1):
    Sw = np.cov(X0.T) * (len(X0) - 1) + np.cov(X1.T) * (len(X1) - 1)
    w = np.linalg.solve(Sw, X1.mean(0) - X0.mean(0))
    return w / np.linalg.norm(w)


def dprim(X0, X1, u):
    p0, p1 = X0 @ u, X1 @ u
    return abs(p1.mean() - p0.mean()) / np.sqrt((p0.var() + p1.var()) / 2)


class Model2(Scene):
    def do(self, t):
        dt = t - self.renderer.time
        if dt > 0.02:
            self.wait(dt)

    def construct(self):
        dane = json.load(open(DANE, encoding="utf-8"))["kierunek_tozsamosci"]
        KR = np.array(dane["kroki"], float)
        DS = np.array(dane["d_staly"], float)
        DO = np.array(dane["d_odsw"], float)
        COS = np.array(dane["cos_1000"], float)

        X0b, X1b = chmury()
        wF = fisher(X0b, X1b)
        thF = np.arctan2(wF[1], wF[0])

        # ---------------- naglowek ----------------
        krok = Text("MODEL 2 / 6", font=MONO, font_size=22, color=CIEMNY)
        krok.to_corner(UL, buff=0.5)
        tytul = Text("Wyznaczanie kierunku tożsamości", font=FONT, font_size=40,
                     weight=BOLD, color=TEKST)
        tytul.next_to(krok, DOWN, buff=0.14, aligned_edge=LEFT)
        self.play(FadeIn(krok, shift=0.2 * RIGHT), FadeIn(tytul, shift=0.2 * RIGHT),
                  run_time=0.9, rate_func=ease_out)

        # ---------------- plaszczyzna ----------------
        PC = np.array([-3.55, -0.78, 0])  # srodek plaszczyzny
        pan = RoundedRectangle(width=6.2, height=5.5, corner_radius=0.15, fill_color=PANEL,
                               fill_opacity=1, stroke_color=LINIA, stroke_width=2).move_to(PC)
        siatka = VGroup(
            Line(PC + LEFT * 2.95, PC + RIGHT * 2.95),
            Line(PC + DOWN * 2.6, PC + UP * 2.6),
        ).set_stroke(LINIA, width=1.5)
        pan_t = napis("latent, rzut 2D", 24, CIEMNY)
        pan_t.next_to(pan.get_corner(DR), UL, buff=0.15)

        # stan: kat prostej (rad), krok uczenia
        kat = ValueTracker(np.radians(-20))
        krokT = ValueTracker(50.0)
        fazaB = {"on": False}

        def phi(s):
            """Obrot chmur po kroku 1000: kat, ktorego cosinus to zmierzony cos_1000."""
            if s <= 1000:
                return 0.0
            c = float(np.interp(s, KR, COS))
            return -np.arccos(np.clip(c, -1, 1))

        def przesun(s):
            if s <= 1000:
                return np.zeros(2)
            q = (s - 1000) / 3500
            return np.array([0.22 * np.sin(np.pi * q), -0.18 * np.sin(2 * np.pi * q)])

        def stan():
            s = krokT.get_value()
            R = rot(phi(s))
            sh = przesun(s)
            return X0b @ R.T + sh, X1b @ R.T + sh, sh

        def p3(v):
            return PC + np.array([v[0], v[1], 0])

        def kropki_tla():
            X0, _, _ = stan()
            return VGroup(*[Dot(p3(p), radius=0.045, color=TEAL, fill_opacity=0.75) for p in X0])

        def kropki_narz():
            _, X1, _ = stan()
            return VGroup(*[Dot(p3(p), radius=0.05, color=POMAR, fill_opacity=0.95) for p in X1])

        # statyczne wersje do wejscia
        tlo0 = kropki_tla()
        narz0 = kropki_narz()
        self.play(FadeIn(pan), FadeIn(siatka), FadeIn(pan_t), run_time=0.6, rate_func=ease_out)
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in tlo0], lag_ratio=0.01),
                  run_time=1.0, rate_func=ease_out)
        et_tlo = napis("tło", 26, TEAL).move_to(p3(rot(np.radians(38)) @ np.array([-2.05, -0.15])
                                                   + np.array([0, -0.45])))
        self.play(FadeIn(et_tlo), run_time=0.35, rate_func=ease_out)
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in narz0], lag_ratio=0.03),
                  run_time=0.8, rate_func=ease_out)
        m1v = X1b.mean(0)
        et_narz = napis("narzędzie", 26, POMAR).move_to(p3(m1v + np.array([1.45, 0.35])))
        self.play(FadeIn(et_narz), run_time=0.35, rate_func=ease_out)

        # ---------------- prosta-kandydat ----------------
        L = 2.55

        def u_kat():
            a = kat.get_value()
            return np.array([np.cos(a), np.sin(a)])

        def prosta():
            u = u_kat()
            return Line(p3(-L * u), p3(L * u), color=TEKST, stroke_width=3.5)

        def rzuty():
            u = u_kat()
            g = VGroup()
            for X, col in ((X0b, TEAL), (X1b, POMAR)):
                for p in X:
                    q = (p @ u) * u
                    g.add(Dot(p3(q), radius=0.03, color=col, fill_opacity=0.55))
            return g

        pr0 = prosta()
        self.play(Create(pr0), run_time=0.7, rate_func=ease_io)
        pr = always_redraw(prosta)
        rz = always_redraw(rzuty)
        self.remove(pr0)
        self.add(pr)

        # ---------------- histogramy rzutow + d' ----------------
        HC = np.array([3.55, -0.55, 0])  # srodek panelu histogramu
        hpan = RoundedRectangle(width=6.0, height=3.9, corner_radius=0.15, fill_color=PANEL,
                                fill_opacity=1, stroke_color=LINIA, stroke_width=2).move_to(HC)
        BASE = HC[1] - 1.25
        SX = 0.82
        os_h = Line([HC[0] - 2.75, BASE, 0], [HC[0] + 2.75, BASE, 0], color=CIEMNY, stroke_width=2)
        os_t = napis("rzut na prostą", 24, CIEMNY).next_to(os_h, DOWN, buff=0.15)
        bins = np.linspace(-3.2, 3.2, 29)
        bw = bins[1] - bins[0]

        def histogramy():
            u = u_kat()
            g = VGroup()
            for X, col, op in ((X0b, TEAL, 0.55), (X1b, POMAR, 0.7)):
                h, _ = np.histogram(X @ u, bins=bins)
                h = h / len(X)
                for k, v in enumerate(h):
                    if v <= 0:
                        continue
                    hh = min(v * 7.0, 2.35)
                    x = HC[0] + (bins[k] + bw / 2) * SX
                    r = Rectangle(width=bw * SX * 0.92, height=hh, stroke_width=0,
                                  fill_color=col, fill_opacity=op)
                    r.move_to([x, BASE + hh / 2, 0])
                    g.add(r)
            return g

        dlab = Text("d′", font=FONT, font_size=40, color=TEKST, slant=ITALIC)
        dlab.move_to(HC + np.array([-0.55, 2.55, 0]))

        def dval():
            v = dprim(X0b, X1b, u_kat())
            t = liczba(pl(v), 40, TEKST)
            t.next_to(dlab, RIGHT, buff=0.3).align_to(dlab, DOWN)
            return t

        hist = always_redraw(histogramy)
        dv = always_redraw(dval)
        self.play(FadeIn(hpan), FadeIn(os_h), FadeIn(os_t), FadeIn(hist), FadeIn(rz),
                  run_time=0.8, rate_func=ease_out)
        self.play(FadeIn(dlab), FadeIn(dv), run_time=0.5, rate_func=ease_out)
        # NIERUCHOME UJECIE 1: chmury, pierwsza prosta, nakladajace sie histogramy
        self.wait(1.5)

        # ---------------- obrot prostej ----------------
        self.play(kat.animate.set_value(np.radians(55)), run_time=3.2, rate_func=ease_io)
        self.play(kat.animate.set_value(thF + np.radians(10)), run_time=2.4, rate_func=ease_io)
        self.play(kat.animate.set_value(thF), run_time=1.0, rate_func=ease_io)

        # ---------------- kierunek Fishera ----------------
        uF = np.array([np.cos(thF), np.sin(thF)])
        self.remove(pr)
        prF = Line(p3(-L * uF), p3(L * uF), color=TEKST, stroke_width=3.5)
        self.add(prF)
        strF = Arrow(p3(-L * uF), p3(L * uF), buff=0, color=POMAR, stroke_width=7,
                     max_tip_length_to_length_ratio=0.07)
        self.play(ReplacementTransform(prF, strF), dv.animate.set_color(POMAR),
                  run_time=0.8, rate_func=ease_out)
        dv.clear_updaters()
        dv.set_color(POMAR)
        et_F = napis("kierunek Fishera", 26, POMAR)
        et_F.next_to(p3(L * uF), RIGHT, buff=0.12).shift(0.1 * UP)
        self.play(FadeIn(et_F, shift=0.1 * RIGHT), run_time=0.45, rate_func=ease_out)
        # NIERUCHOME UJECIE 2: dopasowany kierunek, duze d'
        self.wait(1.7)

        # ---------------- przejscie do uczenia ----------------
        rz.clear_updaters()
        hist.clear_updaters()
        self.play(FadeOut(VGroup(hpan, os_h, os_t, hist, dlab, dv, rz, et_F)),
                  run_time=0.7, rate_func=rate_functions.ease_in_sine)

        # wykres z pomiaru
        ax = Axes(x_range=[0, 4500, 1000], y_range=[0, 30, 10], x_length=4.3, y_length=3.7,
                  tips=False, axis_config={"color": CIEMNY, "stroke_width": 2.5,
                                           "include_ticks": True, "tick_size": 0.06})
        ax.shift(np.array([0.5, -2.45, 0]) - ax.c2p(0, 0))
        prow = VGroup(*[Line(ax.c2p(0, v), ax.c2p(4500, v), color=LINIA, stroke_width=1.5)
                        for v in (10, 20, 30)])
        tx = VGroup(*[liczba(str(v), 22, CIEMNY).next_to(ax.c2p(v, 0), DOWN, buff=0.14)
                      for v in (0, 2000, 4000)])
        ty = VGroup(*[liczba(str(v), 22, CIEMNY).next_to(ax.c2p(0, v), LEFT, buff=0.14)
                      for v in (0, 10, 20, 30)])
        xl = napis("krok uczenia", 24, CIEMNY).next_to(tx, DOWN, buff=0.1).align_to(ax.c2p(4500, 0), RIGHT)
        yl = Text("d′", font=FONT, font_size=30, color=CIEMNY, slant=ITALIC)
        yl.next_to(ax.c2p(0, 30), UP, buff=0.15).align_to(ty, LEFT)
        self.play(Create(ax), FadeIn(prow), FadeIn(tx), FadeIn(ty), FadeIn(xl), FadeIn(yl),
                  run_time=0.9, rate_func=ease_io)

        def krzywa(Y, col, sw):
            def f():
                s = krokT.get_value()
                m = KR <= s
                xs = list(KR[m])
                ys = list(Y[m])
                if s > KR[0] and (len(xs) == 0 or xs[-1] < s):
                    xs.append(s)
                    ys.append(float(np.interp(s, KR, Y)))
                pts = [ax.c2p(a, b) for a, b in zip(xs, ys)]
                v = VMobject(stroke_color=col, stroke_width=sw)
                if len(pts) >= 2:
                    v.set_points_as_corners(pts)
                else:
                    v.set_points_as_corners([ax.c2p(KR[0], Y[0])] * 2)
                return v
            return f

        k_st = always_redraw(krzywa(DS, SZARY, 4.5))
        k_od = always_redraw(krzywa(DO, POMAR, 5))

        def kursor():
            s = krokT.get_value()
            return DashedLine(ax.c2p(s, 0), ax.c2p(s, 30), color=CIEMNY, stroke_width=2,
                              dash_length=0.08)

        def kursor_t():
            s = krokT.get_value()
            t = liczba(f"krok {int(round(s / 50) * 50)}", 24, TEKST)
            t.next_to(ax.c2p(4500, 30), UP, buff=0.15).align_to(ax.c2p(4500, 0), RIGHT)
            return t

        kur = always_redraw(kursor)
        kur_t = always_redraw(kursor_t)
        self.add(k_st, k_od)
        self.play(FadeIn(kur), FadeIn(kur_t), run_time=0.3)
        # historia do chwili dopasowania (krok 1000)
        self.play(krokT.animate.set_value(1000), run_time=1.6, rate_func=ease_io)
        szczyt = liczba("27,4", 24, SZARY).next_to(ax.c2p(1000, 27.4), LEFT, buff=0.1)
        self.play(FadeIn(szczyt), run_time=0.35, rate_func=ease_out)

        # etykiety krzywych sledza czubki
        def et_kr(tekst, Y, Yinny, col, gora):
            def f():
                s = krokT.get_value()
                y = ax.c2p(s, float(np.interp(s, KR, Y)))[1]
                yi = ax.c2p(s, float(np.interp(s, KR, Yinny)))[1]
                mid = (y + yi) / 2
                if abs(y - yi) < 0.34:
                    y = mid + (0.17 if (y > yi or (y == yi and gora)) else -0.17)
                t = napis(tekst, 24, col)
                t.next_to(np.array([ax.c2p(s, 0)[0], y, 0]), RIGHT, buff=0.12)
                return t
            return f

        e_st = always_redraw(et_kr("stały", DS, DO, SZARY, False))
        e_od = always_redraw(et_kr("odświeżany", DO, DS, POMAR, True))

        # ---------------- dwa kierunki na plaszczyznie ----------------
        def srodek():
            return p3(przesun(krokT.get_value()))

        def u_odsw():
            X0, X1, _ = stan()
            w = fisher(X0, X1)
            return w

        Ls = 2.4

        def strz_st():
            c = srodek()
            return Arrow(c, c + Ls * np.array([uF[0], uF[1], 0]), buff=0, color=SZARY,
                         stroke_width=6, max_tip_length_to_length_ratio=0.09)

        def strz_od():
            c = srodek()
            w = u_odsw()
            return Arrow(c, c + Ls * np.array([w[0], w[1], 0]), buff=0, color=POMAR,
                         stroke_width=7, max_tip_length_to_length_ratio=0.09)

        def luk():
            w = u_odsw()
            a1 = np.arctan2(w[1], w[0])
            d = thF - a1
            c = srodek()
            if abs(d) < 0.02:
                return VMobject()
            return Arc(radius=0.85, start_angle=a1, angle=d, arc_center=c, color=TEKST,
                       stroke_width=3)

        def cos_t():
            s = krokT.get_value()
            c = float(np.interp(s, KR, COS)) if s >= 1000 else 1.0
            g = VGroup(Text("cos", font=MONO, font_size=26, color=CIEMNY),
                       liczba(pl(c), 30, TEKST)).arrange(RIGHT, buff=0.15)
            bg = BackgroundRectangle(g, color=PANEL, fill_opacity=0.85, buff=0.08)
            out = VGroup(bg, g)
            out.move_to(pan.get_corner(UR) + np.array([-0.95, -0.4, 0]))
            return out

        # przejscie: plaska prosta Fishera -> dwie strzalki
        s_st = strz_st()
        self.add(s_st)
        self.bring_to_front(strF)
        e_kst = napis("stały", 24, SZARY)
        e_kst.next_to(p3(Ls * uF), UP, buff=0.12)
        self.play(Transform(strF, strz_od()), FadeIn(e_kst), FadeIn(e_st), FadeIn(e_od),
                  run_time=0.6, rate_func=ease_out)
        self.remove(strF)
        s_od = always_redraw(strz_od)
        self.add(s_od)
        lk = always_redraw(luk)
        ct = always_redraw(cos_t)
        self.play(FadeIn(ct), run_time=0.4, rate_func=ease_out)

        # chmury zaczynaja sie ruszac
        self.remove(tlo0, narz0, *tlo0, *narz0)
        tlo = always_redraw(kropki_tla)
        narz = always_redraw(kropki_narz)
        self.add(tlo, narz, lk)
        self.bring_to_front(s_st, s_od, e_kst, ct)
        self.play(FadeOut(et_tlo), FadeOut(et_narz), run_time=0.3)

        # ---------------- dryf: 1000 -> 2000 ----------------
        self.play(krokT.animate.set_value(2000), run_time=5.0, rate_func=ease_io)
        # NIERUCHOME UJECIE 3: srodek dryfu, staly slabnie
        self.wait(1.5)

        # ---------------- dryf: 2000 -> 4500 ----------------
        self.play(krokT.animate.set_value(4500), run_time=7.0, rate_func=ease_io)
        # NIERUCHOME UJECIE 4: koniec, duzy kat
        self.wait(2.0)

        for m in list(self.mobjects):
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7,
                  rate_func=rate_functions.ease_in_sine)
