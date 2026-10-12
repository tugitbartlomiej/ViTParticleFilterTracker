from manim import *
from manim.utils.rate_functions import ease_out_cubic, ease_in_out_cubic, ease_in_cubic

BG = "#0E1A24"
PANEL = "#16293A"
GUIDE = "#24425A"
TXT = "#EAF2F5"
DIM = "#7C93A3"
TEAL = "#2BB3C0"
ORANGE = "#F08A2E"
GREEN = "#5BC27A"
RED = "#E0565B"

MAT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/prezentacja_pipeline_latent/materialy/"

config.background_color = BG


def T(s, size=28, color=TXT, weight=NORMAL):
    return Text(s, font="Segoe UI", font_size=size, color=color, weight=weight)


def N(s, size=24, color=TXT):
    return Text(s, font="Consolas", font_size=size, color=color)


def box(label, stroke=TEAL, fill=PANEL, txt=TXT):
    t = T(label, 24, txt)
    r = RoundedRectangle(corner_radius=0.12, width=t.width + 0.5, height=t.height + 0.42,
                         stroke_color=stroke, stroke_width=3, fill_color=fill, fill_opacity=1)
    t.move_to(r)
    return VGroup(r, t)


class Scena4(Scene):
    def construct(self):
        # ---------- naglowek (0-1 s) ----------
        krok = N("KROK 4 / 7", 22, DIM).to_corner(UL, buff=0.5)
        tytul = T("Stable Diffusion: od celu do obrazu", 40, TXT, BOLD)
        tytul.next_to(krok, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(FadeIn(krok, shift=0.2 * RIGHT), FadeIn(tytul, shift=0.2 * RIGHT),
                  run_time=1.0, rate_func=ease_out_cubic)

        # ---------- czesc 1: karta z morfem + petla (1-10 s) ----------
        karta = RoundedRectangle(corner_radius=0.2, width=6.25, height=4.95,
                                 stroke_color=GUIDE, stroke_width=2,
                                 fill_color=PANEL, fill_opacity=1).move_to([-3.55, -0.62, 0])
        img_pos = karta.get_top() + DOWN * (0.18 + 1.9)
        im_szum = ImageMobject(MAT + "822922eb_SZUM_STARTOWY.png").set(height=3.8).move_to(img_pos)
        im_ddim = ImageMobject(MAT + "4d47060f_OBRAZ_SD_PO_DDIM.png").set(height=3.8).move_to(img_pos)
        im_obr = ImageMobject(MAT + "3a00feec_OBRAZ_SD.png").set(height=3.8).move_to(img_pos)
        lab_y = karta.get_bottom()[1] + 0.42

        def etykieta(s):
            return T(s, 28, DIM).move_to([karta.get_center()[0], lab_y, 0])

        lab = etykieta("szum startowy")
        self.play(FadeIn(karta), FadeIn(im_szum), FadeIn(lab, shift=0.1 * UP),
                  run_time=0.8, rate_func=ease_out_cubic)

        # schemat petli
        cel = box("cel w latencie", stroke=ORANGE, fill="#3A2614", txt=ORANGE).move_to([1.95, 1.45, 0])
        gen = box("generator SD").move_to([1.95, 0.0, 0])
        obr = box("obraz").move_to([5.15, 0.0, 0])
        vit = box("ViT: latent obrazu").move_to([5.15, -1.45, 0])

        a1 = Arrow(cel.get_bottom(), gen.get_top(), buff=0.06, color=DIM, stroke_width=4,
                   max_tip_length_to_length_ratio=0.35)
        a2 = Arrow(gen.get_right(), obr.get_left(), buff=0.08, color=DIM, stroke_width=4)
        a3 = Arrow(obr.get_bottom(), vit.get_top(), buff=0.06, color=DIM, stroke_width=4,
                   max_tip_length_to_length_ratio=0.35)
        p0 = vit.get_left() + LEFT * 0.06
        p1 = np.array([gen.get_center()[0], p0[1], 0])
        p2 = gen.get_bottom() + DOWN * 0.06
        sprz_linia = Line(p0, p1, color=RED, stroke_width=4)
        sprz_strz = Arrow(p1 + DOWN * 0.02, p2, buff=0, color=RED, stroke_width=4,
                          max_tip_length_to_length_ratio=0.4)
        sprz = VGroup(sprz_linia, sprz_strz)
        blad = T("błąd = latent obrazu − cel", 24, RED)
        blad.move_to([(p0[0] + p1[0]) / 2 + 0.1, p0[1] - 0.55, 0])

        self.play(FadeIn(cel, shift=0.15 * DOWN), run_time=0.6, rate_func=ease_out_cubic)
        self.play(GrowArrow(a1), FadeIn(gen), run_time=0.6, rate_func=ease_out_cubic)
        self.wait(0.4)

        lab2 = etykieta("odszumianie DDIM")
        self.play(FadeTransform(im_szum, im_ddim), ReplacementTransform(lab, lab2),
                  run_time=1.0, rate_func=ease_in_out_cubic)
        self.play(GrowArrow(a2), FadeIn(obr), run_time=0.6, rate_func=ease_out_cubic)
        self.wait(0.6)

        lab3 = etykieta("obraz")
        self.play(FadeTransform(im_ddim, im_obr), ReplacementTransform(lab2, lab3),
                  run_time=1.0, rate_func=ease_in_out_cubic)
        self.play(GrowArrow(a3), FadeIn(vit), run_time=0.6, rate_func=ease_out_cubic)
        self.play(Create(sprz_linia, rate_func=ease_in_cubic), run_time=0.55)
        self.play(GrowArrow(sprz_strz), run_time=0.45, rate_func=ease_out_cubic)
        self.play(FadeIn(blad, shift=0.1 * UP), run_time=0.6, rate_func=ease_out_cubic)
        self.wait(1.2)  # -> 10.0 s

        czesc1 = Group(karta, im_obr, lab3, cel, gen, obr, vit, a1, a2, a3, sprz, blad)
        self.play(FadeOut(czesc1), run_time=0.6, rate_func=ease_in_cubic)

        # ---------- czesc 2: cztery prowadzone obrazy (10.6-16 s) ----------
        pliki = ["16c93047_OBRAZ_2026100511_prowadzony.png",
                 "a10f402c_OBRAZ_2026100523_prowadzony.png",
                 "240c0564_OBRAZ_2026100537_prowadzony.png",
                 "ce3f7f4a_OBRAZ_2026100549_prowadzony.png"]
        ims = [ImageMobject(MAT + p).set(height=1.9) for p in pliki]
        siatka = Group(*ims).arrange_in_grid(rows=2, cols=2, buff=0.28).move_to([0, -0.35, 0])
        ramki = [SurroundingRectangle(im, buff=0.0, color=GUIDE, stroke_width=2) for im in ims]
        for im, rm in zip(ims, ramki):
            self.play(FadeIn(im, scale=0.92), Create(rm), run_time=0.6, rate_func=ease_out_cubic)
            self.wait(0.25)
        podpis = T("cztery różne szumy, ten sam cel", 28, TXT)
        podpis.next_to(siatka, DOWN, buff=0.3)
        self.play(FadeIn(podpis, shift=0.1 * UP), run_time=0.7, rate_func=ease_out_cubic)
        self.wait(0.7)  # -> 15.4 s
        self.play(FadeOut(Group(*ims, *ramki, podpis)), run_time=0.6, rate_func=ease_in_cubic)

        # ---------- czesc 3: wykres slupkowy (16-27.2 s) ----------
        x0, sk = -5.5, 0.2  # 0 -> x0, 55 -> x0 + 11
        oy = -0.5

        def X(v):
            return x0 + v * sk

        os_l = Line([X(0), oy, 0], [X(55), oy, 0], color=DIM, stroke_width=3)
        ticks = VGroup()
        for v in [0, 10, 20, 30, 40, 50]:
            ticks.add(Line([X(v), oy - 0.1, 0], [X(v), oy + 0.1, 0], color=DIM, stroke_width=3))
            ticks.add(N(str(v), 24, DIM).move_to([X(v), oy - 0.42, 0]))
        os_g = VGroup(os_l, ticks)
        self.play(Create(os_g), run_time=0.8, rate_func=ease_in_out_cubic)
        self.wait(0.4)

        gr = 10.25
        kres = DashedLine([X(gr), oy - 0.15, 0], [X(gr), 1.55, 0], color=TEAL,
                          stroke_width=4, dash_length=0.12)
        kres_lab = T("granica osiągalna dla obrazu: 10,25", 26, TEAL)
        kres_lab.next_to(kres.get_top(), UP, buff=0.12).align_to(kres, LEFT).shift(LEFT * 0.15)
        self.play(Create(kres), run_time=0.8, rate_func=ease_in_out_cubic)
        self.play(FadeIn(kres_lab, shift=0.1 * DOWN), run_time=0.7, rate_func=ease_out_cubic)
        self.wait(0.3)

        b_bot, b_h = -0.3, 0.72
        vt = ValueTracker(0.001)

        def slupek():
            w = max(vt.get_value() * sk, 0.001)
            return Rectangle(width=w, height=b_h, stroke_width=0, fill_color=ORANGE,
                             fill_opacity=1).move_to([X(0) + w / 2, b_bot + b_h / 2, 0])

        bar = always_redraw(slupek)
        self.add(bar)
        self.bring_to_front(kres)
        self.play(vt.animate.set_value(51.35), run_time=2.5, rate_func=ease_out_cubic)
        bar.clear_updaters()

        bar_lab = T("największa wartość celu: 51,35", 26, ORANGE)
        bar_lab.next_to([X(51.35), b_bot + b_h, 0], UP, buff=0.15).align_to(
            np.array([X(51.35), 0, 0]), RIGHT)
        self.play(FadeIn(bar_lab, shift=0.1 * DOWN), run_time=0.6, rate_func=ease_out_cubic)

        w_red = (51.35 - gr) * sk
        czerwony = Rectangle(width=w_red, height=b_h, stroke_width=0, fill_color=RED,
                             fill_opacity=1).move_to([X(gr) + w_red / 2, b_bot + b_h / 2, 0])
        self.play(FadeIn(czerwony), run_time=0.8, rate_func=ease_in_out_cubic)
        self.bring_to_front(kres)
        self.wait(0.4)

        fakt = T("85 kanałów, 15 658 wartości poza zasięgiem", 28, TXT).move_to([0, -1.85, 0])
        self.play(FadeIn(fakt, shift=0.1 * UP), run_time=0.7, rate_func=ease_out_cubic)
        self.wait(0.4)
        wniosek = T("Żaden obraz nie trafi dokładnie w ten cel.", 32, ORANGE, BOLD).move_to([0, -2.7, 0])
        self.play(FadeIn(wniosek, scale=0.95), run_time=0.8, rate_func=ease_out_cubic)
        self.wait(2.0)  # -> 27.2 s

        # ---------- wygaszenie (27.2-28 s) ----------
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8, rate_func=ease_in_cubic)
