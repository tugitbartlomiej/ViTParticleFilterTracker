# -*- coding: utf-8 -*-
"""Model 3/6: Stable Diffusion prowadzone celem w latencie ViT (Manim CE 0.20.1).

Liczby z pomiaru (brief sceny, nie ma ich w dane_animacji.json):
  najwieksza wartosc celu 51,35; granica osiagalna dla obrazu 10,25;
  85 kanalow i 15 658 wartosci poza zasiegiem.
Plaszczyzna latentu, obszar osiagalny i sciezki sa schematyczne (bez liczb).
"""
from pathlib import Path

import numpy as np
from PIL import Image
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
VIOLET = "#B48EE0"
FONT = "Segoe UI"
MONO = "Consolas"

CEL_MAKS = 51.35
GRANICA = 10.25

TU = Path(__file__).resolve().parent
KLATKI = TU.parents[1] / "klatki_puli.txt"

ease_out = rate_functions.ease_out_cubic
ease_io = rate_functions.ease_in_out_sine

config.background_color = BG

# ---------------- geometria plaszczyzny ----------------
C_REG = np.array([2.55, -0.15, 0.0])        # srodek obszaru osiagalnego
CEL = np.array([5.85, 1.45, 0.0])           # gwiazda celu (poza obszarem)


def promien(th):
    return (1.28 + 0.20 * np.sin(3 * th + 0.5) + 0.12 * np.sin(5 * th + 1.3)
            + 0.08 * np.cos(2 * th))


TH = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
BRZEG = np.stack([C_REG[0] + promien(TH) * np.cos(TH),
                  C_REG[1] + promien(TH) * np.sin(TH), np.zeros_like(TH)], axis=1)


def wewnatrz(p):
    d = p - C_REG
    th = np.arctan2(d[1], d[0])
    return np.hypot(d[0], d[1]) <= promien(th)


def rzut(p):
    if wewnatrz(p):
        return p.copy()
    i = np.argmin(np.sum((BRZEG - p) ** 2, axis=1))
    return BRZEG[i].copy()


Q_STAR = BRZEG[np.argmin(np.sum((BRZEG - CEL) ** 2, axis=1))].copy()


def punkty_drogi(start, n=8, eta=0.34):
    pts = [np.array(start, float)]
    p = pts[0]
    for _ in range(n):
        p = rzut(p + eta * (CEL - p))
        pts.append(p)
    pts[-1] = Q_STAR.copy()
    return pts


def pozycja(pts, s):
    k = int(np.clip(np.floor(s), 0, len(pts) - 2))
    a = s - k
    return rzut((1 - a) * pts[k] + a * pts[k + 1])


# ---------------- obrazy: szum -> klatka ----------------
def wczytaj_klatki(n):
    sciezki = [l.strip() for l in KLATKI.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for s in sciezki[:n]:
        im = Image.open(s).convert("RGB").resize((240, 160), Image.BILINEAR)
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    return out


def szum(seed):
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + 0.28 * rng.standard_normal((160, 240, 3)), 0, 1).astype(np.float32)


def mieszanka(czysty, n, s, kroki=8):
    a = np.sin(0.5 * np.pi * np.clip(s / kroki, 0, 1)) ** 2   # udzial obrazu
    x = np.sqrt(a) * czysty + np.sqrt(1 - a) * (n - 0.5) + (1 - np.sqrt(a)) * 0.5
    x = np.clip(x, 0, 1)
    rgba = np.concatenate([x, np.ones((160, 240, 1), np.float32)], axis=2)
    return (rgba * 255).astype(np.uint8)


class Model3(Scene):
    def czekaj_do(self, t):
        dt = t - self.time
        if dt > 1e-3:
            self.wait(dt)

    def obraz(self, czysty, n, tracker, srodek, wys):
        img = ImageMobject(mieszanka(czysty, n, 0.0))
        img.height = wys
        img.move_to(srodek)

        def upd(m):
            m.pixel_array = mieszanka(czysty, n, tracker.get_value())
        img.add_updater(upd)
        return img

    def construct(self):
        self.camera.background_color = BG
        klatki = wczytaj_klatki(4)

        # ---------------- naglowek ----------------
        krok = Text("MODEL 3 / 6", font=MONO, font_size=22, color=DIM)
        tytul = Text("Stable Diffusion prowadzone celem", font=FONT, weight=BOLD,
                     font_size=40, color=TXT)
        krok.to_corner(UL, buff=0.5)
        tytul.next_to(krok, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(FadeIn(krok, shift=0.1 * DOWN), FadeIn(tytul, shift=0.1 * DOWN),
                  run_time=0.8, rate_func=smooth)
        self.czekaj_do(1.0)

        # ---------------- kafelek obrazu (szum) ----------------
        SR_IMG = np.array([-3.75, 0.05, 0])
        W_IMG = 2.9
        s1 = ValueTracker(0.0)
        self._s_akt = s1
        img = self.obraz(klatki[0], szum(11), s1, SR_IMG, W_IMG)
        ramka = Rectangle(width=img.width, height=img.height).move_to(SR_IMG)
        ramka.set_stroke(GUIDE, width=2)
        lab_gen = Text("generator SD", font=FONT, font_size=24, color=DIM)
        lab_gen.next_to(ramka, UP, buff=0.2)
        licznik = always_redraw(lambda: Text(
            f"krok {int(round(self._s_akt.get_value()))} / 8", font=MONO, font_size=24,
            color=TXT).next_to(ramka, DOWN, buff=0.22))
        self.play(FadeIn(img, scale=0.96), FadeIn(ramka), FadeIn(lab_gen),
                  run_time=0.8, rate_func=ease_out)
        self.play(FadeIn(licznik), run_time=0.4, rate_func=smooth)

        # ---------------- plaszczyzna latentu ----------------
        panel = RoundedRectangle(corner_radius=0.15, width=6.7, height=4.55)
        panel.move_to(np.array([3.45, 0.05, 0]))
        panel.set_fill(PANEL, 1).set_stroke(GUIDE, 1.5)
        siatka = VGroup()
        for x in np.arange(0.6, 6.8, 0.6):
            siatka.add(Line([x, -2.2, 0], [x, 2.3, 0]))
        for y in np.arange(-1.9, 2.3, 0.6):
            siatka.add(Line([0.12, y, 0], [6.78, y, 0]))
        siatka.set_stroke(GUIDE, width=1, opacity=0.45)
        lab_pl = Text("latent ViT (rzut 2D)", font=FONT, font_size=22, color=DIM)
        lab_pl.move_to(panel.get_corner(UL) + np.array([0.2, -0.28, 0]), aligned_edge=LEFT)

        obszar = Polygon(*BRZEG[::6], color=TEAL)
        obszar.set_fill(TEAL, opacity=0.16).set_stroke(TEAL, width=2.5, opacity=0.75)
        lab_obs = Text("obrazy osiągalne", font=FONT, font_size=24, color=TEAL)
        lab_obs.move_to(np.array([2.05, -1.95, 0]))

        self.play(FadeIn(panel), FadeIn(siatka), FadeIn(lab_pl), run_time=0.6, rate_func=smooth)
        self.play(DrawBorderThenFill(obszar), run_time=0.9, rate_func=ease_out)
        self.play(FadeIn(lab_obs, shift=0.08 * UP), run_time=0.4, rate_func=ease_out)

        # strzalka ViT od kafelka do plaszczyzny
        str_vit = Arrow(np.array([-1.45, 0.05, 0]), np.array([0.0, 0.05, 0]), buff=0.05,
                        color=DIM, stroke_width=4, max_tip_length_to_length_ratio=0.18)
        lab_vit = Text("ViT", font=FONT, font_size=24, color=DIM).next_to(str_vit, UP, buff=0.1)

        # gwiazda celu
        gw = Star(n=5, outer_radius=0.24, inner_radius=0.1, color=ORANGE)
        gw.set_fill(ORANGE, 1).set_stroke(ORANGE, 1).move_to(CEL)
        lab_cel = Text("cel", font=FONT, font_size=24, color=ORANGE)
        lab_cel.next_to(gw, UP, buff=0.12)
        self.play(GrowFromCenter(gw), run_time=0.6, rate_func=ease_out)
        self.play(FadeIn(lab_cel, shift=0.08 * DOWN), run_time=0.35, rate_func=ease_out)

        # punkt latentu biezacego obrazu (szum)
        start1 = np.array([1.75, 0.35, 0])
        pts1 = punkty_drogi(start1)
        dot1 = Dot(start1, radius=0.09, color=TEAL)
        dot1.add_updater(lambda m: m.move_to(pozycja(pts1, s1.get_value())))
        self.play(GrowArrow(str_vit), FadeIn(lab_vit), run_time=0.5, rate_func=ease_out)
        self.play(FadeIn(dot1, scale=0.5), run_time=0.4, rate_func=ease_out)

        # NIERUCHOME 1: szum i cel poza obszarem
        print("STOP1", self.time)
        self.wait(1.4)

        # ---------------- bieg 1: odszumianie z gradientem ----------------
        def strzalka(dot):
            def f():
                p = dot.get_center()
                u = CEL - p
                u = u / (np.linalg.norm(u) + 1e-9)
                return Arrow(p, p + 0.8 * u, buff=0.0, color=ORANGE, stroke_width=5,
                             max_tip_length_to_length_ratio=0.3)
            return always_redraw(f)

        grad1 = strzalka(dot1)
        lab_grad = always_redraw(lambda: Text("gradient", font=FONT, font_size=22, color=ORANGE)
                                 .next_to(grad1.get_end(), RIGHT, buff=0.1))
        slad1 = TracedPath(dot1.get_center, stroke_color=TEAL, stroke_width=3.5,
                           stroke_opacity=0.9)
        self.add(slad1)
        self.play(GrowArrow(grad1), FadeIn(lab_grad), run_time=0.6, rate_func=ease_out)
        for k in range(8):
            self.play(s1.animate.set_value(k + 1), run_time=0.68, rate_func=ease_io)
            self.wait(0.06)
        self.play(FadeOut(lab_grad), run_time=0.4, rate_func=smooth)
        # NIERUCHOME 2: obraz odszumiony, punkt na brzegu
        print("STOP2", self.time)
        self.wait(1.4)

        # ---------------- biegi 2-4: inne szumy, szybciej ----------------
        starty = [np.array([2.35, -1.05, 0]), np.array([3.25, 0.75, 0]),
                  np.array([1.55, -0.55, 0])]
        kolory = [GREEN, VIOLET, TXT]
        seedy = [23, 37, 51]
        img_akt, dot_akt, grad_akt = img, dot1, grad1
        for r in range(3):
            s = ValueTracker(0.0)
            nowy = self.obraz(klatki[r + 1], szum(seedy[r]), s, SR_IMG, W_IMG)
            pts = punkty_drogi(starty[r])
            d = Dot(starty[r], radius=0.09, color=kolory[r])
            d.add_updater(lambda m, pts=pts, s=s: m.move_to(pozycja(pts, s.get_value())))
            # poprzedni bieg: przygas, zatrzymaj
            dot_akt.clear_updaters()
            self.play(FadeIn(nowy), FadeOut(grad_akt),
                      dot_akt.animate.set_opacity(0.55), run_time=0.4, rate_func=smooth)
            self.remove(img_akt)
            img_akt.clear_updaters()
            self._s_akt = s
            self.add(ramka, licznik)
            self.play(FadeIn(d, scale=0.5), run_time=0.25, rate_func=ease_out)
            g = strzalka(d)
            tp = TracedPath(d.get_center, stroke_color=kolory[r], stroke_width=3.5,
                            stroke_opacity=0.9)
            self.add(tp)
            self.play(GrowArrow(g), run_time=0.25, rate_func=ease_out)
            for k in range(8):
                self.play(s.animate.set_value(k + 1), run_time=0.27, rate_func=ease_io)
            img_akt, dot_akt, grad_akt = nowy, d, g

        dot_akt.clear_updaters()
        self.play(FadeOut(grad_akt), run_time=0.35, rate_func=smooth)
        pierscien = Circle(radius=0.22, color=TXT).move_to(Q_STAR).set_stroke(TXT, 2.5)
        self.play(Create(pierscien), run_time=0.5, rate_func=ease_out)
        # NIERUCHOME 3: cztery sciezki na brzegu
        print("STOP3", self.time)
        self.wait(1.4)

        # ---------------- luka: brzeg -> cel ----------------
        luka_pl = DashedLine(Q_STAR, CEL, dash_length=0.12, color=RED, stroke_width=4,
                             buff=0.22)
        self.play(Create(luka_pl), run_time=0.7, rate_func=ease_out)

        X0, X1, Y_PAS = -5.6, 6.4, -3.15

        def xs(v):
            return X0 + (X1 - X0) * v / 55.0

        pas = Line([X0, Y_PAS, 0], [X1, Y_PAS, 0], color=DIM, stroke_width=4)
        kreski = VGroup(*[Line([xs(v), Y_PAS - 0.08, 0], [xs(v), Y_PAS + 0.08, 0])
                          for v in range(0, 56, 5)]).set_stroke(DIM, 2, opacity=0.7)
        n0 = Text("0", font=MONO, font_size=24, color=DIM).next_to([xs(0), Y_PAS, 0], DOWN, buff=0.18)
        n55 = Text("55", font=MONO, font_size=24, color=DIM).next_to([xs(55), Y_PAS, 0], DOWN, buff=0.18)
        self.play(Create(pas), FadeIn(kreski), FadeIn(n0), FadeIn(n55),
                  run_time=0.8, rate_func=ease_out)

        zn_g = Line([xs(GRANICA), Y_PAS - 0.2, 0], [xs(GRANICA), Y_PAS + 0.2, 0],
                    color=TEAL, stroke_width=5)
        lab_g = Text("granica 10,25", font=MONO, font_size=24, color=TEAL)
        lab_g.next_to(zn_g, UP, buff=0.12)
        zn_c = Line([xs(CEL_MAKS), Y_PAS - 0.2, 0], [xs(CEL_MAKS), Y_PAS + 0.2, 0],
                    color=ORANGE, stroke_width=5)
        lab_c = Text("cel 51,35", font=MONO, font_size=24, color=ORANGE)
        lab_c.next_to(zn_c, UP, buff=0.12)
        osiag = Line([xs(0), Y_PAS, 0], [xs(GRANICA), Y_PAS, 0], color=TEAL, stroke_width=7)
        self.play(Create(osiag), GrowFromCenter(zn_g), FadeIn(lab_g, shift=0.08 * UP),
                  run_time=0.7, rate_func=ease_out)
        self.play(GrowFromCenter(zn_c), FadeIn(lab_c, shift=0.08 * UP),
                  run_time=0.6, rate_func=ease_out)

        luka = Line([xs(GRANICA), Y_PAS, 0], [xs(CEL_MAKS), Y_PAS, 0], color=RED, stroke_width=8)
        lab_luka = VGroup(
            Text("luka", font=FONT, font_size=24, color=RED).next_to(luka, UP, buff=0.14),
            Text("85 kanałów, 15 658 wartości", font=MONO, font_size=24, color=RED)
            .next_to(luka, DOWN, buff=0.18))
        self.play(Create(luka), run_time=0.8, rate_func=ease_out)
        self.play(FadeIn(lab_luka, shift=0.08 * UP), run_time=0.5, rate_func=ease_out)
        # NIERUCHOME 4: skala z luka
        print("STOP4", self.time)
        self.wait(1.6)

        # ---------------- wygaszenie ----------------
        for m in self.mobjects:
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6, rate_func=smooth)
        print("KONIEC", self.time)
