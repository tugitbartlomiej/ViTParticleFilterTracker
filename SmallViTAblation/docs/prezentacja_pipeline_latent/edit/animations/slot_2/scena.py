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

MAT = Path(__file__).resolve().parents[3] / "materialy"
OBRAZ = MAT / "d519ec8b_WEJSCIE_VIT.png"

ease_out = rate_functions.ease_out_cubic


class Scena2(Scene):
    def czekaj_do(self, t):
        dt = t - self.time
        if dt > 1e-3:
            self.wait(dt)

    def construct(self):
        self.camera.background_color = BG
        rng = np.random.default_rng(7)

        # ---------------- naglowek (0.0 - 1.0) ----------------
        krok = Text("KROK 2 / 7", font="Consolas", font_size=22, color=DIM)
        tytul = Text("Jak sieć widzi zdjęcie", font="Segoe UI", weight=BOLD,
                     font_size=40, color=TXT)
        krok.to_corner(UL, buff=0.5)
        tytul.next_to(krok, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(FadeIn(krok, shift=0.1 * DOWN), FadeIn(tytul, shift=0.1 * DOWN),
                  run_time=0.8, rate_func=smooth)
        self.czekaj_do(1.0)

        # ---------------- FAZA A (1 - 9 s) ----------------
        img = ImageMobject(str(OBRAZ))
        img.height = 3.6
        img.move_to(np.array([-2.25, -0.15, 0]))
        ramka_img = SurroundingRectangle(img, buff=0.0, color=GUIDE, stroke_width=2)
        self.play(FadeIn(img, scale=0.96), FadeIn(ramka_img), run_time=0.8, rate_func=ease_out)

        W, H = img.width, img.height
        L, R = img.get_left()[0], img.get_right()[0]
        T, B = img.get_top()[1], img.get_bottom()[1]
        s = W / 30.0
        linie = VGroup()
        for c in range(31):
            x = L + c * s
            linie.add(Line([x, T, 0], [x, B, 0]))
        for r in range(21):
            y = T - r * s
            linie.add(Line([L, y, 0], [R, y, 0]))
        linie.set_stroke(TEAL, width=1.3, opacity=0.6)
        self.play(LaggedStart(*[Create(l) for l in linie], lag_ratio=0.04),
                  run_time=2.0, rate_func=smooth)
        self.wait(0.3)

        podpis_a = Text("480 × 320 pikseli → 30 × 20 = 600 tokenów",
                        font="Consolas", font_size=26, color=TXT)
        podpis_a.next_to(img, DOWN, buff=0.35)
        if podpis_a.get_left()[0] < -6.5:
            podpis_a.shift((-6.5 - podpis_a.get_left()[0]) * RIGHT)
        self.play(FadeIn(podpis_a, shift=0.1 * UP), run_time=0.7, rate_func=ease_out)
        self.czekaj_do(5.6)

        # komorka nad koncem cystotomu (kolumna 12, wiersz 7)
        kol, wier = 12, 7
        cx = L + (kol + 0.5) * s
        cy = T - (wier + 0.5) * s
        kom = Square(side_length=s)
        kom.move_to([cx, cy, 0])
        kom.set_stroke(ORANGE, width=3).set_fill(ORANGE, opacity=0.45)
        self.play(FadeIn(kom, scale=1.6), run_time=0.5, rate_func=ease_out)
        self.play(kom.animate.scale(4.0).shift(0.4 * UP + 0.3 * RIGHT)
                  .set_fill(ORANGE, opacity=0.8),
                  run_time=0.6, rate_func=smooth)

        # wektor 768 liczb: pionowy pasek z poziomych pasow
        n_pas = 96
        bar_h, bar_w = 3.6, 0.5
        pas_h = bar_h / n_pas
        pasy = VGroup()
        for i in range(n_pas):
            t = rng.random()
            mix = rng.random() ** 2.2
            kol_p = interpolate_color(ManimColor(TEAL), ManimColor(ORANGE), mix)
            r_ = Rectangle(width=bar_w, height=pas_h * 0.86)
            r_.set_stroke(width=0).set_fill(kol_p, opacity=0.25 + 0.75 * t)
            r_.move_to([0, bar_h / 2 - (i + 0.5) * pas_h, 0])
            pasy.add(r_)
        obrys = SurroundingRectangle(pasy, buff=0.06, color=GUIDE, stroke_width=2)
        wektor = VGroup(obrys, pasy)
        wektor.move_to(np.array([4.4, img.get_center()[1], 0]))

        self.play(ReplacementTransform(kom, wektor, path_arc=-0.6),
                  run_time=1.1, rate_func=smooth)
        lab_w = Text("1 token = 768 liczb", font="Consolas", font_size=26, color=TXT)
        lab_w.next_to(wektor, DOWN, buff=0.35)
        lab_w.match_y(podpis_a)
        # wskaznik komorki zostaje na obrazie
        znacznik = Square(side_length=s).move_to([cx, cy, 0])
        znacznik.set_stroke(ORANGE, width=3).set_fill(ORANGE, opacity=0.35)
        self.play(FadeIn(lab_w, shift=0.1 * UP), FadeIn(znacznik),
                  run_time=0.6, rate_func=ease_out)
        self.czekaj_do(9.0)

        # ---------------- FAZA B (9 - 19 s) ----------------
        y0 = -0.4
        maly_h = 0.9
        maly_w = maly_h * 1.5
        cel_img = np.array([-6.05 + maly_w / 2 - 0.0, y0, 0])
        cel_img[0] = -7.11 + 0.55 + maly_w / 2
        self.play(
            FadeOut(linie), FadeOut(podpis_a), FadeOut(wektor), FadeOut(lab_w),
            FadeOut(znacznik), FadeOut(ramka_img),
            img.animate.set(height=maly_h).move_to(cel_img),
            run_time=0.7, rate_func=smooth,
        )
        ramka_m = SurroundingRectangle(img, buff=0.0, color=GUIDE, stroke_width=2)
        self.add(ramka_m)

        def strz(a, b):
            return Arrow(a, b, buff=0.12, stroke_width=4, color=TEAL,
                         max_tip_length_to_length_ratio=0.35, tip_length=0.16)

        # lodyga: 4 zwezajace sie trapezy
        hs = [1.25, 1.0, 0.78, 0.6, 0.45]
        tw, tg = 0.26, 0.07
        lodyga = VGroup()
        x = 0.0
        for i in range(4):
            p = Polygon([x, hs[i] / 2, 0], [x + tw, hs[i + 1] / 2, 0],
                        [x + tw, -hs[i + 1] / 2, 0], [x, -hs[i] / 2, 0])
            p.set_stroke(TEAL, width=2.5).set_fill(TEAL, opacity=0.15 + 0.12 * i)
            lodyga.add(p)
            x += tw + tg
        lodyga.move_to([-3.85, y0, 0])

        lab_lod = Text("stem: 4 sploty", font="Segoe UI", font_size=24, color=TXT)
        lab_lod.next_to(lodyga, DOWN, buff=0.35).set_y(y0 - 1.15)

        a1 = strz(img.get_right(), lodyga.get_left())
        self.play(GrowArrow(a1), run_time=0.3, rate_func=ease_out)
        self.play(LaggedStart(*[FadeIn(p, shift=0.1 * RIGHT) for p in lodyga], lag_ratio=0.25),
                  run_time=0.6, rate_func=ease_out)
        self.play(FadeIn(lab_lod, shift=0.08 * UP), run_time=0.35, rate_func=ease_out)

        # 6 blokow transformera
        bw, bh, bgap = 0.55, 1.4, 0.2
        bloki = VGroup()
        for i in range(6):
            rr = RoundedRectangle(width=bw, height=bh, corner_radius=0.1)
            rr.set_stroke(TEAL, width=3).set_fill(PANEL, opacity=1)
            nr = Text(str(i + 1), font="Consolas", font_size=24, color=DIM)
            nr.move_to(rr)
            bloki.add(VGroup(rr, nr))
        bloki.arrange(RIGHT, buff=bgap).move_to([-0.25, y0, 0])
        lab_blk = Text("6 bloków transformera", font="Segoe UI", font_size=24, color=TXT)
        lab_blk.next_to(bloki, DOWN, buff=0.35).set_y(y0 - 1.15)

        a2 = strz(lodyga.get_right(), bloki.get_left())
        self.play(GrowArrow(a2), run_time=0.3, rate_func=ease_out)
        self.play(LaggedStart(*[FadeIn(b, shift=0.1 * UP) for b in bloki], lag_ratio=0.18),
                  run_time=0.8, rate_func=ease_out)
        self.play(FadeIn(lab_blk, shift=0.08 * UP), run_time=0.35, rate_func=ease_out)

        # latent: znacznik na ostatnim bloku
        ost = bloki[-1][0]
        mlp = RoundedRectangle(width=bw - 0.16, height=0.34, corner_radius=0.06)
        mlp.set_stroke(ORANGE, width=2).set_fill(ORANGE, opacity=0.85)
        mlp.move_to(ost.get_center() + 0.33 * DOWN)
        obw = SurroundingRectangle(ost, buff=0.08, corner_radius=0.14,
                                   color=ORANGE, stroke_width=3)
        lab_lat = Text("latent: 600 × 768", font="Consolas", font_size=26, color=ORANGE)
        lab_lat.next_to(obw, UP, buff=0.3)
        nota = Text("GELU w MLP ostatniego bloku", font="Segoe UI", font_size=24, color=DIM)
        nota.next_to(lab_lat, UP, buff=0.15)
        # trzymaj w kadrze
        for m in (lab_lat, nota):
            if m.get_right()[0] > 6.6:
                m.shift((6.6 - m.get_right()[0]) * RIGHT)
        self.play(Create(obw), FadeIn(mlp, scale=0.6), bloki[-1][1].animate.set_opacity(0),
                  run_time=0.5, rate_func=ease_out)
        self.play(FadeIn(lab_lat, shift=0.08 * UP), run_time=0.45, rate_func=ease_out)
        self.play(FadeIn(nota, shift=0.08 * UP), run_time=0.35, rate_func=ease_out)
        self.wait(0.8)

        # glowa
        glowa = Polygon([0, 0.5, 0], [0.7, 0.3, 0], [0.7, -0.3, 0], [0, -0.5, 0])
        glowa.set_stroke(TEAL, width=3).set_fill(TEAL, opacity=0.2)
        glowa.move_to([3.1, y0, 0])
        lab_gl = Text("głowa", font="Segoe UI", font_size=24, color=TXT)
        lab_gl.next_to(glowa, DOWN, buff=0.35).set_y(y0 - 1.15)
        a3 = strz(obw.get_right(), glowa.get_left())
        self.play(GrowArrow(a3), run_time=0.3, rate_func=ease_out)
        self.play(FadeIn(glowa, shift=0.1 * RIGHT), FadeIn(lab_gl), run_time=0.45,
                  rate_func=ease_out)

        # wyjscie: ikona ramki
        ikona_tlo = RoundedRectangle(width=1.1, height=0.8, corner_radius=0.08)
        ikona_tlo.set_stroke(GUIDE, width=2).set_fill(PANEL, opacity=1)
        rogi = Rectangle(width=0.5, height=0.34).set_stroke(GREEN, width=3)
        pkt = Dot(radius=0.05, color=GREEN)
        rogi.move_to(ikona_tlo.get_center() + 0.08 * RIGHT + 0.04 * DOWN)
        pkt.move_to(rogi.get_corner(UL) + 0.12 * UP + 0.02 * LEFT)
        wyj = VGroup(ikona_tlo, rogi, pkt).move_to([5.5, y0, 0])
        lab_wy = Text("ramka + pewność", font="Segoe UI", font_size=24, color=TXT)
        lab_wy.next_to(wyj, DOWN, buff=0.35).set_y(y0 - 1.15)
        lab_wy.set_x(min(wyj.get_x(), 6.6 - lab_wy.width / 2))
        a4 = strz(glowa.get_right(), wyj.get_left())
        self.play(GrowArrow(a4), run_time=0.3, rate_func=ease_out)
        self.play(FadeIn(wyj, scale=0.8), FadeIn(lab_wy), run_time=0.45, rate_func=ease_out)

        # impuls danych
        t_imp = self.time
        tor = VMobject().set_points_as_corners([
            img.get_right() + 0.05 * RIGHT, wyj.get_center()])
        swiatlo = VGroup(*[
            Dot(radius=r, color=ORANGE).set_opacity(o)
            for r, o in [(0.26, 0.10), (0.18, 0.2), (0.12, 0.45), (0.075, 1.0)]
        ])
        swiatlo.move_to(tor.get_start())
        slad = TracedPath(swiatlo.get_center, stroke_color=ORANGE, stroke_width=5,
                          dissipating_time=0.22)
        self.add(slad, swiatlo)
        self.play(FadeIn(swiatlo, scale=0.5), run_time=0.15)
        self.play(MoveAlongPath(swiatlo, tor), run_time=1.8, rate_func=smooth)
        self.play(FadeOut(swiatlo, scale=1.8), Indicate(rogi, color=GREEN, scale_factor=1.15),
                  run_time=0.4)
        self.remove(slad)
        print("impuls start", t_imp, "koniec", self.time)
        self.czekaj_do(19.0)

        # ---------------- FAZA C (19 - 27.2 s) ----------------
        faza_b = [img, ramka_m, a1, a2, a3, a4, lodyga, lab_lod, bloki, lab_blk, obw, mlp,
                  lab_lat, nota, glowa, lab_gl, wyj, lab_wy]
        self.play(*[FadeOut(m) for m in faza_b], run_time=0.6, rate_func=smooth)

        panel = RoundedRectangle(width=12.2, height=4.2, corner_radius=0.18)
        panel.set_stroke(GUIDE, width=2).set_fill(PANEL, opacity=1)
        panel.move_to([0, -0.5, 0])
        self.play(FadeIn(panel, scale=0.98), run_time=0.6, rate_func=ease_out)

        tyt_p = Text("przestrzeń latentu (rzut 2D)", font="Segoe UI", font_size=28, color=DIM)
        tyt_p.next_to(panel, UP, buff=0.18, aligned_edge=LEFT)
        self.play(FadeIn(tyt_p, shift=0.08 * UP), run_time=0.5, rate_func=ease_out)

        # trajektorie: dryf wspolnego srodka + szybki szum OU wokol niego
        N, K = 14, 1000
        u = np.linspace(0, 1, K)
        srodek = np.stack([-3.2 + 6.2 * u + 0.35 * np.sin(2.2 * np.pi * u),
                           -0.6 + 0.8 * u + 0.45 * np.sin(1.3 * np.pi * u + 0.4)], axis=1)
        srodek[:, 1] += panel.get_center()[1]
        baza = rng.normal(0, 1, size=(N, 2)) * np.array([0.55, 0.42])
        theta, sigma, dt = 12.0, 0.8, 1.0 / K
        ou = np.zeros((N, K, 2))
        for k in range(1, K):
            ou[:, k] = ou[:, k - 1] - theta * ou[:, k - 1] * dt \
                + sigma * np.sqrt(dt) * rng.normal(size=(N, 2))
        traj = srodek[None, :, :] + baza[:, None, :] + ou
        # przytnij do panelu
        traj[..., 0] = np.clip(traj[..., 0], panel.get_left()[0] + 0.3, panel.get_right()[0] - 0.3)
        traj[..., 1] = np.clip(traj[..., 1], panel.get_bottom()[1] + 0.3, panel.get_top()[1] - 0.3)

        tracker = ValueTracker(0.0)

        def poz(i, v):
            f = v * (K - 1)
            k0 = int(np.floor(f))
            k1 = min(k0 + 1, K - 1)
            a = f - k0
            p = (1 - a) * traj[i, k0] + a * traj[i, k1]
            return np.array([p[0], p[1], 0.0])

        kropki = VGroup()
        for i in range(N):
            d = Dot(radius=0.07, color=TEAL)
            d.move_to(poz(i, 0.0))
            kropki.add(d)
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in kropki], lag_ratio=0.08),
                  run_time=0.8, rate_func=ease_out)

        licznik = always_redraw(lambda: Text(
            f"krok uczenia: {int(round(tracker.get_value() * 1000))}",
            font="Consolas", font_size=26, color=TXT,
        ).next_to(panel, UP, buff=0.18).align_to(panel, RIGHT))
        self.play(FadeIn(licznik), run_time=0.4, rate_func=ease_out)

        slady = VGroup()
        for i, d in enumerate(kropki):
            d.add_updater(lambda m, i=i: m.move_to(poz(i, tracker.get_value())))
            tp = TracedPath(d.get_center, stroke_color=TEAL, stroke_width=2.2,
                            stroke_opacity=0.8, dissipating_time=0.6)
            slady.add(tp)
        self.add(slady, kropki)

        podpis_c = Text("Każdy token to cząstka, która porusza się z każdym krokiem uczenia.",
                        font="Segoe UI", font_size=28, color=TXT)
        podpis_c.next_to(panel, DOWN, buff=0.3)
        print("ruch start", self.time)
        self.play(AnimationGroup(
            tracker.animate(run_time=5.0, rate_func=smooth).set_value(1.0),
            Succession(Wait(0.7), FadeIn(podpis_c, shift=0.08 * UP, run_time=0.8,
                                         rate_func=ease_out)),
        ), run_time=5.0)
        for d in kropki:
            d.clear_updaters()
        print("ruch koniec", self.time)
        self.czekaj_do(27.2)

        # ---------------- wygaszenie (27.2 - 28.0) ----------------
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8, rate_func=smooth)
        print("koniec", self.time)
