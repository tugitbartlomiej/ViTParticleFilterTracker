from pathlib import Path

import numpy as np
from manim import *

TU = "#0E1A24"
PANEL = "#16293A"
GUIDE = "#24425A"
TXT = "#EAF2F5"
DIM = "#7C93A3"
TEAL = "#2BB3C0"
ORANGE = "#F08A2E"
GREEN = "#5BC27A"
RED = "#E0565B"

SANS = "Segoe UI"
MONO = "Consolas"
HERE = Path(__file__).resolve().parent

config.background_color = TU

EI = rate_functions.ease_in_out_cubic
EO = rate_functions.ease_out_cubic


def pl(v):
    return f"{v:.2f}".replace(".", ",")


class Scena3(Scene):
    def build_card(self, img, label, x):
        photo = ImageMobject(str(HERE / img))
        photo.height = 2.7
        photo.move_to([x, 0.35, 0])
        frame = SurroundingRectangle(photo, buff=0, color=GUIDE, stroke_width=2)
        lab = Text(label, font=SANS, font_size=24, color=DIM)
        lab.next_to(photo, UP, buff=0.14).align_to(photo, LEFT)
        bg = RoundedRectangle(
            corner_radius=0.12,
            width=photo.width + 0.36,
            height=photo.height + 0.36 + 0.44 + 0.9,
            fill_color=PANEL, fill_opacity=1, stroke_color=GUIDE, stroke_width=1.5,
        )
        bg.move_to(photo).align_to(lab, UP).shift(UP * 0.18)
        return bg, photo, frame, lab

    def stat_line(self, key, val, anchor, color=TXT):
        t = Text(f"{key}{pl(val)} %", font=MONO, font_size=28, color=color)
        t.move_to(anchor, aligned_edge=LEFT)
        return t

    def construct(self):
        # ---------- header (0-1 s)
        krok = Text("KROK 3 / 7", font=MONO, font_size=22, color=DIM)
        krok.to_corner(UL, buff=0.55)
        title = Text("Cel w latencie", font=SANS, weight=BOLD, font_size=40, color=TXT)
        title.next_to(krok, DOWN, buff=0.12, aligned_edge=LEFT)
        self.play(FadeIn(krok, shift=0.1 * DOWN), run_time=0.4, rate_func=EO)
        self.play(FadeIn(title, shift=0.15 * RIGHT), run_time=0.6, rate_func=EO)

        # ---------- left card (1-6 s)
        xl, xr = -4.35, 4.35
        bgL, phL, frL, labL = self.build_card("przed.png", "przed korektą", xl)
        cardL = Group(bgL, phL, frL)
        self.play(FadeIn(cardL, shift=0.2 * UP), run_time=1.0, rate_func=EO)
        self.play(FadeIn(labL), run_time=0.5, rate_func=EO)
        aL1 = phL.get_corner(DL) + DOWN * 0.32
        aL2 = aL1 + DOWN * 0.45
        sL1 = self.stat_line("pewność  ", 86.79, aL1)
        sL2 = self.stat_line("IoU      ", 85.66, aL2)
        self.play(FadeIn(sL1, shift=0.1 * UP), run_time=0.5, rate_func=EO)
        self.play(FadeIn(sL2, shift=0.1 * UP), run_time=0.5, rate_func=EO)
        self.wait(2.5)  # t = 6.0

        # ---------- middle: 128 latent channels (6-12 s)
        pw, ph = 2.7, 1.5
        strip = RoundedRectangle(corner_radius=0.08, width=pw, height=ph,
                                 fill_color=PANEL, fill_opacity=1,
                                 stroke_color=GUIDE, stroke_width=1.5)
        strip.move_to([0, 1.45, 0])
        strip.set_z_index(2)
        base_y = strip.get_bottom()[1] + 0.12
        inner = pw - 0.24
        n = 128
        step = inner / n
        bw = step * 0.62
        rng = np.random.default_rng(3)
        h0 = rng.uniform(0.08, 0.55, n)
        big = rng.choice(n, 22, replace=False)
        h1 = np.clip(h0 + rng.normal(0, 0.06, n), 0.05, 0.6)
        h1[big] = rng.uniform(0.82, 1.12, len(big))

        def bars_for(hs, tint):
            g = VGroup()
            for i, h in enumerate(hs):
                col = ORANGE if (tint and i in set(big.tolist())) else TEAL
                r = Rectangle(width=bw, height=float(h), stroke_width=0,
                              fill_color=col, fill_opacity=1)
                r.move_to([strip.get_left()[0] + 0.12 + step * (i + 0.5), base_y + h / 2, 0])
                g.add(r)
            g.set_z_index(3)
            return g

        bars = bars_for(h0, False)
        bars_new = bars_for(h1, True)
        base_line = Line([strip.get_left()[0] + 0.1, base_y, 0],
                         [strip.get_right()[0] - 0.1, base_y, 0],
                         color=GUIDE, stroke_width=2).set_z_index(3)

        cap = VGroup(*[Text(t, font=SANS, font_size=26, color=TXT)
                       for t in ("zmieniamy tylko 128", "z 768 kanałów latentu")]
                     ).arrange(DOWN, buff=0.12)
        cap.next_to(strip, DOWN, buff=0.3)
        note = VGroup(*[Text(t, font=SANS, font_size=24, color=DIM)
                        for t in ("piksele i wagi sieci", "bez zmian")]
                      ).arrange(DOWN, buff=0.1)
        note.next_to(cap, DOWN, buff=0.25)

        self.play(FadeIn(strip), Create(base_line), run_time=0.6, rate_func=EO)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.02),
                  run_time=1.0, rate_func=EO)
        self.play(FadeIn(cap, shift=0.1 * UP), run_time=0.6, rate_func=EO)
        self.play(FadeIn(note), run_time=0.5, rate_func=EO)
        self.wait(0.3)
        self.play(Transform(bars, bars_new), run_time=2.0, rate_func=EI)
        self.wait(1.0)  # t = 12.0

        # ---------- right card (12-18 s)
        bgR, phR, frR, labR = self.build_card("po.png", "po korekcie", xr)
        cardR = Group(bgR, phR, frR)
        self.play(FadeIn(cardR, shift=0.2 * UP), run_time=1.0, rate_func=EO)
        self.play(FadeIn(labR), run_time=0.5, rate_func=EO)

        aR1 = phR.get_corner(DL) + DOWN * 0.32
        aR2 = aR1 + DOWN * 0.45
        tr = ValueTracker(0.0)

        def mk(key, a, b, anchor):
            def f():
                p = tr.get_value()
                v = a + (b - a) * p
                col = interpolate_color(ManimColor(TXT), ManimColor(GREEN), p)
                return self.stat_line(key, v, anchor, color=col)
            return f

        sR1 = always_redraw(mk("pewność  ", 86.79, 93.17, aR1))
        sR2 = always_redraw(mk("IoU      ", 85.66, 99.93, aR2))
        self.play(FadeIn(sR1), FadeIn(sR2), run_time=0.5, rate_func=EO)
        self.play(tr.animate.set_value(1.0), run_time=2.0, rate_func=EO)
        sR1.clear_updaters()
        sR2.clear_updaters()

        ay = strip.get_center()[1] - 0.05
        arrow = Arrow([frL.get_right()[0] + 0.05, ay, 0], [frR.get_left()[0] - 0.05, ay, 0],
                      buff=0, color=TEAL, stroke_width=6,
                      max_tip_length_to_length_ratio=0.04, max_stroke_width_to_length_ratio=10)
        arrow.set_z_index(1)
        self.play(GrowArrow(arrow), run_time=1.0, rate_func=EI)
        self.wait(1.0)  # t = 18.0

        # ---------- bottom note (18-23.2 s)
        l1 = Text("Na 400 zdjęciach: AP50 0,44 → 0,88", font=MONO, font_size=28, color=TXT)
        l2 = Text("To interwencja z etykietą: pokazuje, czego potrzebuje głowa, nie gotowy detektor.",
                  font=SANS, font_size=24, color=DIM)
        body = VGroup(l1, l2).arrange(DOWN, buff=0.16)
        box = RoundedRectangle(corner_radius=0.1, width=max(body.width + 0.6, 10.0),
                               height=body.height + 0.36, fill_color=PANEL, fill_opacity=1,
                               stroke_color=GUIDE, stroke_width=1.5)
        box.move_to([0, -4 + 0.55 + box.height / 2, 0])
        body.move_to(box)
        self.play(FadeIn(box), run_time=0.6, rate_func=EO)
        self.play(FadeIn(l1, shift=0.1 * UP), run_time=0.6, rate_func=EO)
        self.play(FadeIn(l2), run_time=0.6, rate_func=EO)
        self.wait(3.4)  # t = 23.2

        # ---------- fade out (23.2-24 s)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8, rate_func=EI)
