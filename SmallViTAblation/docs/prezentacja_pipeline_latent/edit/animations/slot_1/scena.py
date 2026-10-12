from pathlib import Path

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

KARTY = [
    ("4146be44_ZDJECIE_ZRODLOWE.jpg", "Sonda: zdjęcie z narzędziem"),
    ("3a00feec_OBRAZ_SD.png", "Obraz ze Stable Diffusion"),
    ("2862ad10_NAJBLIZSZE_ZDJECIE.png", "Prawdziwa klatka|z puli uczącej"),
]

IMG_H = 2.3
GAP = 0.9


class Scena1(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- naglowek (0.0 - 1.0) ---
        krok = Text("KROK 1 / 7", font="Consolas", font_size=22, color=DIM)
        tytul = Text("Pytanie", font="Segoe UI", weight=BOLD, font_size=40, color=TXT)
        krok.to_corner(UL, buff=0.5)
        tytul.next_to(krok, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(
            FadeIn(krok, shift=0.1 * DOWN),
            FadeIn(tytul, shift=0.1 * DOWN),
            run_time=0.8,
            rate_func=smooth,
        )
        self.wait(0.2)

        # --- karty (1.0 - 7.5) ---
        karty = []
        for plik, podpis in KARTY:
            img = ImageMobject(str(MAT / plik))
            img.height = IMG_H
            ramka = RoundedRectangle(
                width=img.width + 0.12,
                height=img.height + 0.12,
                corner_radius=0.1,
                stroke_color=GUIDE,
                stroke_width=3,
                fill_color=PANEL,
                fill_opacity=1,
            )
            ramka.move_to(img)
            cap = VGroup(*[
                Text(w, font="Segoe UI", font_size=24, color=TXT)
                for w in podpis.split("|")
            ]).arrange(DOWN, buff=0.1)
            cap.next_to(ramka, DOWN, buff=0.18)
            karty.append(Group(ramka, img, cap))

        szer = karty[0][0].width
        x0 = -(szer + GAP)
        y_kart = 1.05
        for i, k in enumerate(karty):
            k.shift(np.array([x0 + i * (szer + GAP), y_kart, 0]) - k[0].get_center())

        def strzalka(a, b):
            return Arrow(
                a[0].get_right(),
                b[0].get_left(),
                buff=0.12,
                color=TEAL,
                stroke_width=6,
                max_tip_length_to_length_ratio=0.35,
            )

        # karta 1
        self.play(FadeIn(karty[0], shift=0.25 * UP), run_time=0.8, rate_func=smooth)
        self.wait(0.8)
        # karta 2 + strzalka
        self.play(FadeIn(karty[1], shift=0.25 * UP), run_time=0.8, rate_func=smooth)
        s1 = strzalka(karty[0], karty[1])
        self.play(GrowArrow(s1), run_time=0.5, rate_func=rate_functions.ease_out_cubic)
        self.wait(0.3)
        # karta 3 + strzalka
        self.play(FadeIn(karty[2], shift=0.25 * UP), run_time=0.8, rate_func=smooth)
        s2 = strzalka(karty[1], karty[2])
        self.play(GrowArrow(s2), run_time=0.5, rate_func=rate_functions.ease_out_cubic)
        self.wait(2.0)  # -> 7.5 s

        # --- pytanie (7.5 - 13.5) ---
        l1 = Text("Czy da się tak dobrać zdjęcia do uczenia,", font="Segoe UI",
                  font_size=36, color=TXT)
        l2 = Text("żeby detektor uczył się szybciej?", font="Segoe UI",
                  font_size=36, color=TXT)
        slowo = l2[-9:]  # "szybciej?" (spacje nie sa glifami)
        slowo.set_color(ORANGE)
        pytanie = VGroup(l1, l2).arrange(DOWN, buff=0.22)
        dol_kart = min(k[2].get_bottom()[1] for k in karty)
        pytanie.next_to(np.array([0, dol_kart, 0]), DOWN, buff=0.45)
        pytanie.set_x(0)

        self.play(Write(pytanie), run_time=2.5, rate_func=smooth)
        self.wait(1.0)

        opis = Text(
            "Mały ViT (2,9 mln parametrów) wykrywa narzędzie w nagraniach operacji zaćmy",
            font="Segoe UI", font_size=24, color=DIM,
        )
        opis.next_to(pytanie, DOWN, buff=0.4)
        self.play(FadeIn(opis, shift=0.1 * UP), run_time=0.8, rate_func=smooth)
        self.wait(1.7)  # -> 13.5 s

        # --- przytrzymanie (13.5 - 17.2) ---
        for _ in range(2):
            self.play(
                slowo.animate(rate_func=there_and_back).scale(1.03),
                run_time=1.0,
            )
            self.wait(0.4)
        self.wait(0.9)  # -> 17.2 s

        # --- wygaszenie (17.2 - 18.0) ---
        self.play(
            FadeOut(Group(*self.mobjects)),
            run_time=0.8,
            rate_func=smooth,
        )
