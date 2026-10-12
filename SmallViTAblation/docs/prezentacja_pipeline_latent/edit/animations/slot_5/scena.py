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

SLOT = Path(__file__).resolve().parent
MAT = SLOT.parents[2] / "materialy"
MINI = SLOT / "miniatury"

ZAPYTANIE = MAT / "6377ae0a_OBRAZ_ZAPYTANIA.png"
ZNALEZIONE = MAT / "2862ad10_NAJBLIZSZE_ZDJECIE.png"

KOL, WIER = 8, 6
CEL_W = 0.68
CEL_H = CEL_W * 2 / 3
ODST = 0.07
WYBRANY = 29  # wiersz 3, kolumna 5 (od zera)
LICZBA = 461547

KARTA_H = 3.2
Y_KART = 0.55
X_LEWA = -3.6
X_PRAWA = 3.6


def tysiace(n):
    return f"{n:,}".replace(",", " ")


def karta(sciezka, kolor=GUIDE):
    img = ImageMobject(str(sciezka))
    img.height = KARTA_H
    ramka = RoundedRectangle(
        width=img.width + 0.14,
        height=img.height + 0.14,
        corner_radius=0.1,
        stroke_color=kolor,
        stroke_width=3,
        fill_color=PANEL,
        fill_opacity=1,
    )
    ramka.move_to(img)
    return ramka, img


class Scena5(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- naglowek (0.0 - 1.0) ---
        krok = Text("KROK 5 / 7", font="Consolas", font_size=22, color=DIM)
        tytul = Text(
            "Wyszukiwanie prawdziwych zdjęć",
            font="Segoe UI",
            weight=BOLD,
            font_size=40,
            color=TXT,
        )
        krok.to_corner(UL, buff=0.5)
        tytul.next_to(krok, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(
            FadeIn(krok, shift=0.1 * DOWN),
            FadeIn(tytul, shift=0.1 * DOWN),
            run_time=0.8,
            rate_func=smooth,
        )
        self.wait(0.2)

        # --- zapytanie (1.0 - 4.0) ---
        q_ramka, q_img = karta(ZAPYTANIE)
        q_karta = Group(q_ramka, q_img).move_to([X_LEWA, Y_KART, 0])
        q_podpis = Text("zapytanie: obraz z SD", font="Segoe UI", font_size=26, color=TXT)
        q_podpis.next_to(q_ramka, DOWN, buff=0.25)
        self.play(FadeIn(q_karta, shift=0.25 * UP), run_time=0.9, rate_func=smooth)
        self.play(FadeIn(q_podpis, shift=0.1 * UP), run_time=0.6, rate_func=smooth)
        self.wait(1.5)

        # --- pula (4.0 - 12.0) ---
        sciezki = [MINI / f"m{i:02d}.png" for i in range(KOL * WIER)]
        sciezki[WYBRANY] = MINI / "znalezione.png"
        komorki = []
        for i, p in enumerate(sciezki):
            img = ImageMobject(str(p))
            img.width = CEL_W
            r, c = divmod(i, KOL)
            img.move_to(
                [
                    (c - (KOL - 1) / 2) * (CEL_W + ODST),
                    -(r - (WIER - 1) / 2) * (CEL_H + ODST),
                    0,
                ]
            )
            komorki.append(img)
        siatka = Group(*komorki)
        tlo_siatki = RoundedRectangle(
            width=siatka.width + 0.3,
            height=siatka.height + 0.3,
            corner_radius=0.12,
            stroke_color=GUIDE,
            stroke_width=2,
            fill_color=PANEL,
            fill_opacity=1,
        )
        pula = Group(tlo_siatki, siatka).move_to([X_PRAWA, 0.0, 0])

        self.play(
            FadeIn(tlo_siatki, run_time=0.4),
            LaggedStart(
                *[FadeIn(k, scale=0.85) for k in komorki],
                lag_ratio=0.04,
                run_time=1.5,
            ),
            rate_func=smooth,
        )
        p_podpis = Text("pula uczącego zbioru", font="Segoe UI", font_size=26, color=TXT)
        p_podpis.next_to(tlo_siatki, DOWN, buff=0.22)
        self.play(FadeIn(p_podpis, shift=0.1 * UP), run_time=0.5, rate_func=smooth)
        nota = Text(
            "każde przez ten sam zamrożony ViT",
            font="Segoe UI",
            font_size=24,
            color=DIM,
        )
        nota.next_to(p_podpis, DOWN, buff=0.18)
        self.play(FadeIn(nota), run_time=0.5, rate_func=smooth)

        # licznik i podswietlenie sterowane jednym trackerem
        koniec = KOL * WIER + WYBRANY
        k = ValueTracker(0)
        lewy = tlo_siatki.get_left()[0]

        def licznik_tekst():
            n = int(round(LICZBA * k.get_value() / koniec))
            napis = f"przeszukano: {tysiace(n)}"
            if n >= LICZBA:
                napis += " zdjęć"
            t = Text(napis, font="Consolas", font_size=28, color=TEAL)
            t.next_to(tlo_siatki, UP, buff=0.22)
            t.align_to(tlo_siatki, LEFT)
            return t

        licznik = always_redraw(licznik_tekst)

        def ramka_podsw():
            i = int(k.get_value()) % (KOL * WIER)
            if k.get_value() >= koniec:
                i = WYBRANY
            r = Rectangle(
                width=CEL_W + 0.07,
                height=CEL_H + 0.07,
                stroke_color=TEAL,
                stroke_width=5,
            )
            r.move_to(komorki[i])
            return r

        podsw = always_redraw(ramka_podsw)
        self.play(FadeIn(licznik), FadeIn(podsw), run_time=0.4, rate_func=smooth)
        self.play(
            k.animate.set_value(koniec),
            run_time=4.5,
            rate_func=rate_functions.ease_out_cubic,
        )
        self.wait(0.6)

        # --- wybor i wynik (12.0 - 21.2) ---
        podsw.clear_updaters()
        licznik.clear_updaters()
        self.play(
            podsw.animate.set_stroke(ORANGE, width=7),
            run_time=0.5,
            rate_func=smooth,
        )
        wybrana = komorki[WYBRANY]
        inne = [c for i, c in enumerate(komorki) if i != WYBRANY]
        self.play(
            *[FadeOut(c) for c in inne],
            FadeOut(tlo_siatki),
            FadeOut(p_podpis),
            FadeOut(nota),
            FadeOut(licznik),
            run_time=0.8,
            rate_func=smooth,
        )

        z_ramka, z_img = karta(ZNALEZIONE, kolor=ORANGE)
        Group(z_ramka, z_img).move_to([X_PRAWA, Y_KART, 0])
        self.play(
            wybrana.animate.match_height(z_img).move_to(z_img),
            ReplacementTransform(podsw, z_ramka),
            run_time=1.0,
            rate_func=smooth,
        )
        self.play(FadeIn(z_img), run_time=0.3)
        self.remove(wybrana)

        z_podpis = Text(
            "najbliższa prawdziwa klatka", font="Segoe UI", font_size=26, color=TXT
        )
        z_podpis.next_to(z_ramka, DOWN, buff=0.25)
        self.play(FadeIn(z_podpis, shift=0.1 * UP), run_time=0.6, rate_func=smooth)
        z_id = Text("TR_0006_S1_P03, klatka 1193", font="Consolas", font_size=24, color=DIM)
        z_id.next_to(z_podpis, DOWN, buff=0.15)
        self.play(FadeIn(z_id, shift=0.1 * UP), run_time=0.6, rate_func=smooth)
        self.wait(0.4)

        strz = DoubleArrow(
            [q_ramka.get_right()[0] + 0.15, Y_KART, 0],
            [z_ramka.get_left()[0] - 0.15, Y_KART, 0],
            buff=0,
            color=TEAL,
            stroke_width=5,
            tip_length=0.22,
        )
        self.play(GrowFromCenter(strz), run_time=0.5, rate_func=smooth)

        mse = Text("MSE 768: 0,0296", font="Consolas", font_size=28, color=TXT)
        cos = Text("cosinus: 0,875", font="Consolas", font_size=28, color=TXT)
        mse[:7].set_color(TEAL)
        cos[:8].set_color(TEAL)
        staty = VGroup(mse, cos).arrange(RIGHT, buff=1.2).move_to([0, -2.55, 0])
        self.play(FadeIn(mse, shift=0.1 * UP), run_time=0.6, rate_func=smooth)
        self.wait(0.3)
        self.play(FadeIn(cos, shift=0.1 * UP), run_time=0.6, rate_func=smooth)
        self.wait(0.5)

        wniosek = Text(
            "Uczymy na prawdziwych zdjęciach z poprawnymi etykietami.",
            font="Segoe UI",
            font_size=26,
            color=GREEN,
        )
        wniosek.move_to([0, -3.3, 0])
        self.play(FadeIn(wniosek, shift=0.1 * UP), run_time=0.8, rate_func=smooth)
        self.wait(1.6)

        # --- wygaszenie (21.2 - 22.0) ---
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8, rate_func=smooth)
