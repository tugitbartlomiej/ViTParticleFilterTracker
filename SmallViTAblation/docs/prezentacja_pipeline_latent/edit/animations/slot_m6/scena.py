import json
from pathlib import Path

import numpy as np
from manim import *

BG = "#0E1A24"
PANEL = "#16293A"
GUIDE = "#24425A"
TXT = "#EAF2F5"
DIM = "#7C93A3"
TEAL = "#2BB3C0"
ORANGE = "#F08A2E"
GREEN = "#5BC27A"
RED = "#E0565B"
FONT = "Segoe UI"
MONO = "Consolas"

DANE = Path(__file__).resolve().parents[2] / "dane_animacji.json"

ease_out = rate_functions.ease_out_cubic
ease_io = rate_functions.ease_in_out_sine


def napis(s, size=24, color=TXT, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def liczba(s, size=24, color=DIM):
    return Text(s, font=MONO, font_size=size, color=color)


def pl(v, n=3):
    return f"{v:.{n}f}".replace(".", ",")


class Model6(Scene):
    def do(self, t):
        dt = t - self.renderer.time
        if dt > 0.02:
            self.wait(dt)

    def construct(self):
        self.camera.background_color = BG
        d = json.load(open(DANE, encoding="utf-8"))
        tw = d["trajektoria_wag"]
        kroki = np.array(tw["kroki"], dtype=float)  # 0..1000 co 10
        XY = np.array(tw["xyz"], dtype=float)[:, :2]
        mu = d["map_usrednianie"]
        sur_k = np.array(sorted(int(k) for k in mu["surowe"]), dtype=float)
        sur_v = np.array([mu["surowe"][str(int(k))] for k in sur_k])
        usr = {int(k): v for k, v in mu["usrednione_okno100"].items()}
        sr_sur = {int(k): v for k, v in mu["srednia_surowych_okno100"].items()}

        # ---------------- naglowek ----------------
        nr = Text("MODEL 6 / 6", font=MONO, font_size=22, color=DIM)
        nr.to_corner(UL, buff=0.5)
        tytul = Text("Trajektoria wag i uśrednianie", font=FONT, weight=BOLD,
                     font_size=40, color=TXT)
        tytul.next_to(nr, DOWN, buff=0.14, aligned_edge=LEFT)
        self.play(FadeIn(nr, shift=0.15 * RIGHT), FadeIn(tytul, shift=0.15 * RIGHT),
                  run_time=0.8, rate_func=ease_out)
        self.do(0.9)

        # ---------------- plaszczyzna kierunkow ----------------
        ax = Axes(x_range=[-1.5, 23, 5], y_range=[-6.5, 6.5, 2], x_length=6.5,
                  y_length=5.2, tips=True,
                  axis_config={"color": DIM, "stroke_width": 2.2, "include_ticks": False},
                  x_axis_config={"tip_length": 0.16, "tip_width": 0.12},
                  y_axis_config={"tip_length": 0.16, "tip_width": 0.12})
        ax.move_to(np.array([-3.4, -0.85, 0]))
        lab_x = napis("kierunek 1", 24, DIM).next_to(ax.x_axis.get_end(), DOWN, buff=0.22)
        lab_x.align_to(ax.x_axis.get_end(), RIGHT)
        lab_y = napis("kierunek 2", 24, DIM).next_to(ax.y_axis.get_end(), RIGHT, buff=0.18)

        def P(x, y):
            return ax.c2p(x, y)

        def poz(s):
            """polozenie wag w ciaglym kroku s (interpolacja liniowa danych)."""
            s = float(np.clip(s, 0, 1000))
            return np.array([np.interp(s, kroki, XY[:, 0]), np.interp(s, kroki, XY[:, 1])])

        def srednia(s):
            """srednia z 10 ostatnich zapisanych stanow (okno 100 krokow) w chwili s."""
            pts = [poz(s - 10 * j) for j in range(10) if s - 10 * j >= 0]
            return np.mean(pts, axis=0)

        # dolina: schematyczne kontury, wzdluz trajektorii, srodek poza droga
        c0 = P(11.0, -0.4)
        kat = 14 * DEGREES
        dolina = VGroup()
        for i, r in enumerate([0.35, 0.75, 1.2, 1.7, 2.25, 2.85]):
            e = Ellipse(width=2.25 * r, height=1.05 * r)
            e.rotate(kat).move_to(c0)
            e.set_stroke(GUIDE, width=2.0, opacity=0.85 - 0.08 * i).set_fill(opacity=0)
            dolina.add(e)
        dolina.set_z_index(-2)

        self.play(Create(ax), run_time=0.8, rate_func=ease_io)
        self.play(FadeIn(lab_x), FadeIn(lab_y), run_time=0.5, rate_func=ease_out)
        self.play(LaggedStart(*[FadeIn(e, scale=0.9) for e in dolina], lag_ratio=0.15),
                  run_time=0.9, rate_func=ease_out)

        # ---------------- trajektoria ----------------
        s = ValueTracker(0.0)

        def droga():
            sv = s.get_value()
            g = VGroup()
            ks = [k for k in kroki if k < sv] + [sv]
            pts = [P(*poz(k)) for k in ks]
            for a, b, k in zip(pts[:-1], pts[1:], ks[1:]):
                wiek = sv - k
                op = 0.28 + 0.72 * np.exp(-wiek / 260.0)
                g.add(Line(a, b, stroke_width=3.2, color=TEAL, stroke_opacity=op))
            return g

        sciezka = always_redraw(droga)
        kropka = always_redraw(lambda: Dot(P(*poz(s.get_value())), radius=0.085,
                                           color=TEAL).set_stroke(TXT, width=1.5))
        licznik = always_redraw(
            lambda: Text(f"krok {int(round(s.get_value())):4d}", font=MONO,
                         font_size=28, color=TXT).to_corner(UR, buff=0.5))

        self.play(FadeIn(kropka, scale=1.6), run_time=0.5, rate_func=ease_out)
        self.add(sciezka)
        self.play(FadeIn(licznik), run_time=0.4, rate_func=ease_out)
        self.play(s.animate.set_value(200), run_time=3.0, rate_func=linear)
        self.wait(1.4)  # NIERUCHOME 1: poczatek trajektorii
        self.play(s.animate.set_value(1000), run_time=5.5, rate_func=linear)
        self.wait(1.4)  # NIERUCHOME 2: pelna trajektoria

        # zamrozenie sciezki (statyczna kopia)
        sciezka_st = droga()
        kropka_st = Dot(P(*poz(1000)), radius=0.085, color=TEAL).set_stroke(TXT, width=1.5)
        licznik_st = licznik.copy()
        licznik_st.clear_updaters()
        self.remove(sciezka, kropka, licznik)
        self.add(sciezka_st, kropka_st, licznik_st)

        # ---------------- okno 100 krokow i srednia ----------------
        idx = list(range(91, 101))
        okno_pts = [P(*XY[i]) for i in idx]
        okno = VGroup(*[Dot(p, radius=0.06, color=ORANGE) for p in okno_pts])
        okno_lin = VGroup(*[Line(a, b, color=ORANGE, stroke_width=5)
                            for a, b in zip(okno_pts[:-1], okno_pts[1:])])
        self.play(Create(okno_lin), FadeIn(okno, lag_ratio=0.1), run_time=0.8, rate_func=ease_out)

        cm = XY[idx].mean(axis=0)
        cP = P(*cm)

        # lupa: okno powiekszone 5 razy w pustym rogu plaszczyzny
        LC = np.array([-1.35, -2.58, 0])
        ZM = 5.0

        def lupa(p):
            return LC + ZM * (np.array(p) - cP)

        ramka_okna = SurroundingRectangle(VGroup(*okno), buff=0.08, color=ORANGE,
                                          stroke_width=1.6)
        panel = RoundedRectangle(width=2.3, height=1.85, corner_radius=0.12)
        panel.set_fill(PANEL, opacity=1).set_stroke(GUIDE, width=2).move_to(LC)
        laczniki = VGroup(
            DashedLine(ramka_okna.get_corner(DL), panel.get_corner(UL), dash_length=0.06),
            DashedLine(ramka_okna.get_corner(DR), panel.get_corner(UR), dash_length=0.06),
        ).set_stroke(ORANGE, width=1.2, opacity=0.6)
        kawalek = VMobject(stroke_color=TEAL, stroke_width=3)
        kawalek.set_points_as_corners([lupa(P(*poz(k))) for k in np.arange(905, 1001, 5)])
        m_pts = [lupa(p) for p in okno_pts]
        m_dots = VGroup(*[Dot(p, radius=0.07, color=ORANGE) for p in m_pts])
        lupa_g = VGroup(panel, kawalek, m_dots)
        self.play(Create(ramka_okna), Create(laczniki), FadeIn(lupa_g, scale=0.85),
                  run_time=0.7, rate_func=ease_out)

        zbieg = VGroup(*[Line(p, LC, path_arc=(0.9 if j % 2 else -0.9), color=ORANGE,
                              stroke_width=1.6, stroke_opacity=0.8)
                         for j, p in enumerate(m_pts)])
        self.play(LaggedStart(*[Create(l) for l in zbieg], lag_ratio=0.08),
                  run_time=0.8, rate_func=ease_io)
        gw = Star(n=5, outer_radius=0.19, inner_radius=0.08, color=ORANGE)
        gw.set_fill(ORANGE, opacity=1).set_stroke(TXT, width=1.2).move_to(cP)
        gw_m = Star(n=5, outer_radius=0.24, inner_radius=0.1, color=ORANGE)
        gw_m.set_fill(ORANGE, opacity=1).set_stroke(TXT, width=1.2).move_to(LC)
        lab_gw = napis("średnia wag", 24, ORANGE).next_to(gw, LEFT, buff=0.2).shift(0.12 * DOWN)
        self.play(GrowFromCenter(gw_m), GrowFromCenter(gw), run_time=0.6, rate_func=ease_out)
        self.play(FadeIn(lab_gw, shift=0.1 * LEFT), run_time=0.4, rate_func=ease_out)
        self.wait(1.4)  # NIERUCHOME 3: gwiazda sredniej

        # ---------------- wykres mAP ----------------
        self.play(FadeOut(zbieg), FadeOut(okno), FadeOut(okno_lin), FadeOut(gw), FadeOut(gw_m),
                  FadeOut(lupa_g), FadeOut(laczniki), FadeOut(ramka_okna),
                  FadeOut(lab_gw), sciezka_st.animate.set_stroke(opacity=0.45),
                  run_time=0.6, rate_func=ease_io)

        ay = Axes(x_range=[500, 1000, 100], y_range=[0.1, 0.45, 0.1], x_length=5.0,
                  y_length=4.2, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2.2, "include_ticks": True,
                               "tick_size": 0.05})
        ay.move_to(np.array([4.1, -0.75, 0]))
        prow = VGroup(*[Line(ay.c2p(500, v), ay.c2p(1000, v), color=GUIDE, stroke_width=1.4)
                        for v in (0.2, 0.3, 0.4)])
        tx = VGroup(*[liczba(str(v), 24).next_to(ay.c2p(v, 0.1), DOWN, buff=0.14)
                      for v in (500, 750, 1000)])
        ty = VGroup(*[liczba(pl(v, 1), 24).next_to(ay.c2p(500, v), LEFT, buff=0.14)
                      for v in (0.2, 0.3, 0.4)])
        yl = Text("mAP", font=MONO, font_size=24, color=DIM).next_to(ay.c2p(500, 0.45), UP, buff=0.18)
        yl.align_to(ty, LEFT)
        xl = napis("krok", 24, DIM).next_to(tx, DOWN, buff=0.08).align_to(ay.c2p(1000, 0.1), RIGHT)
        self.play(Create(ay), FadeIn(prow), run_time=0.8, rate_func=ease_io)
        self.play(FadeIn(tx), FadeIn(ty), FadeIn(yl), FadeIn(xl), run_time=0.4, rate_func=ease_out)

        # kursor wag przechodzi ponownie kroki 500..1000
        s.set_value(500)
        kursor = always_redraw(lambda: Dot(P(*poz(s.get_value())), radius=0.085,
                                           color=TEAL).set_stroke(TXT, width=1.5))
        licznik2 = always_redraw(
            lambda: Text(f"krok {int(round(s.get_value())):4d}", font=MONO,
                         font_size=28, color=TXT).to_corner(UR, buff=0.5))

        def surowa():
            sv = s.get_value()
            ks = [k for k in sur_k if k < sv]
            vs = [mu["surowe"][str(int(k))] for k in ks]
            ks = ks + [sv]
            vs = vs + [float(np.interp(sv, sur_k, sur_v))]
            if len(ks) < 2:
                return VMobject()
            m = VMobject(stroke_color=TEAL, stroke_width=2.2)
            m.set_points_as_corners([ay.c2p(k, v) for k, v in zip(ks, vs)])
            return m

        kursor_wyk = always_redraw(lambda: Dot(
            ay.c2p(s.get_value(), float(np.interp(s.get_value(), sur_k, sur_v))),
            radius=0.05, color=TEAL))
        linia_sur = always_redraw(surowa)
        self.remove(kropka_st, licznik_st)
        self.add(kursor, licznik2, linia_sur, kursor_wyk)
        lab_sur = napis("surowe", 24, TEAL).next_to(ay.c2p(540, 0.18), DOWN, buff=0.12)

        punkty = VGroup()
        gwiazdki = VGroup()
        lab_us = None
        for i, k in enumerate([600, 700, 800, 900, 1000]):
            run = 1.1
            if i == 0:
                self.play(s.animate.set_value(k), FadeIn(lab_sur, run_time=0.6),
                          run_time=run, rate_func=linear)
            else:
                self.play(s.animate.set_value(k), run_time=run, rate_func=linear)
            pk = Dot(ay.c2p(k, usr[k]), radius=0.085, color=ORANGE).set_stroke(TXT, width=1.2)
            gm = Star(n=5, outer_radius=0.12, inner_radius=0.05, color=ORANGE)
            gm.set_fill(ORANGE, opacity=1).set_stroke(width=0).move_to(P(*srednia(k)))
            punkty.add(pk)
            gwiazdki.add(gm)
            anims = [GrowFromCenter(pk), GrowFromCenter(gm)]
            if i == 0:
                lab_us = napis("średnia wag", 24, ORANGE)
                lab_us.move_to(ay.c2p(500, 0.37), aligned_edge=LEFT).shift(0.1 * RIGHT)
                anims.append(FadeIn(lab_us, shift=0.08 * UP))
            self.play(*anims, run_time=0.3, rate_func=ease_out)

        # zatrzymanie: statyczne kopie
        lin_st = surowa()
        kur_st = Dot(P(*poz(1000)), radius=0.085, color=TEAL).set_stroke(TXT, width=1.5)
        kw_st = Dot(ay.c2p(1000, sur_v[-1]), radius=0.05, color=TEAL)
        lic_st = Text("krok 1000", font=MONO, font_size=28, color=TXT).to_corner(UR, buff=0.5)
        self.remove(kursor, licznik2, linia_sur, kursor_wyk)
        self.add(lin_st, kur_st, kw_st, lic_st)

        # srednia surowych z okna 900..1000 (kreska) i podpis
        kreska = DashedLine(ay.c2p(900, sr_sur[1000]), ay.c2p(1000, sr_sur[1000]),
                            dash_length=0.08, color=TXT, stroke_width=2.5)
        self.play(Create(kreska), run_time=0.6, rate_func=ease_io)
        podpis = Text(f"{pl(usr[1000])} wobec {pl(sr_sur[1000])}", font=MONO,
                      font_size=26, color=TXT)
        podpis.next_to(ay.c2p(1000, 0.45), UP, buff=0.2).align_to(ay.c2p(1000, 0), RIGHT)
        self.play(FadeIn(podpis, shift=0.1 * DOWN), run_time=0.6, rate_func=ease_out)
        self.wait(1.6)  # NIERUCHOME 4: wykres mAP

        # ---------------- final: przesuwane okno ----------------
        self.play(FadeOut(gwiazdki), FadeOut(kur_st), FadeOut(lic_st),
                  sciezka_st.animate.set_stroke(opacity=0.22), run_time=0.6, rate_func=ease_io)
        s.set_value(0.0)

        def tor(f, kolor, w):
            def _():
                sv = s.get_value()
                ks = list(np.arange(0, sv, 10.0)) + [sv]
                if len(ks) < 2:
                    return VMobject()
                m = VMobject(stroke_color=kolor, stroke_width=w)
                m.set_points_smoothly([P(*f(k)) for k in ks]) if len(ks) > 2 else \
                    m.set_points_as_corners([P(*f(k)) for k in ks])
                return m
            return _

        tor_w = always_redraw(tor(poz, TEAL, 2.6))
        tor_s = always_redraw(tor(srednia, ORANGE, 3.4))
        kursor3 = always_redraw(lambda: Dot(P(*poz(s.get_value())), radius=0.085,
                                            color=TEAL).set_stroke(TXT, width=1.5))
        gw3 = always_redraw(lambda: Star(n=5, outer_radius=0.17, inner_radius=0.07)
                            .set_fill(ORANGE, opacity=1).set_stroke(TXT, width=1.0)
                            .move_to(P(*srednia(s.get_value()))))
        licznik3 = always_redraw(
            lambda: Text(f"krok {int(round(s.get_value())):4d}", font=MONO,
                         font_size=28, color=TXT).to_corner(UR, buff=0.5))
        self.add(tor_w, tor_s, kursor3, gw3, licznik3)
        self.play(s.animate.set_value(1000), run_time=4.5, rate_func=linear)
        # statyczne kopie na ujecie koncowe
        st = VGroup(tor(poz, TEAL, 2.6)(), tor(srednia, ORANGE, 3.4)(),
                    Dot(P(*poz(1000)), radius=0.085, color=TEAL).set_stroke(TXT, width=1.5),
                    Star(n=5, outer_radius=0.17, inner_radius=0.07).set_fill(ORANGE, opacity=1)
                    .set_stroke(TXT, width=1.0).move_to(P(*srednia(1000))),
                    Text("krok 1000", font=MONO, font_size=28, color=TXT).to_corner(UR, buff=0.5))
        self.remove(tor_w, tor_s, kursor3, gw3, licznik3)
        self.add(st)
        self.wait(1.5)  # NIERUCHOME 5: final

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6, rate_func=ease_io)
