from manim import *
import numpy as np

BG = "#0E1A24"
PANEL = "#16293A"
GUIDE = "#24425A"
TXT = "#EAF2F5"
DIM = "#7C93A3"
TEAL = "#2BB3C0"
ORANGE = "#F08A2E"
GREEN = "#5BC27A"

KROKI = [
    ("1. Checkpoint + sondy", ["stan sieci, etykiety,", "pamięć optymalizatora"]),
    ("2. Cel w latencie", ["ograniczona korekta", "128 kanałów"]),
    ("3. Obrazy SD", ["różne szumy,", "zgodność z celem"]),
    ("4. Wyszukiwanie", ["podobne prawdziwe", "zdjęcia z puli"]),
    ("5. Paczki + prognoza", ["siły, kolejność,", "czas zbliżenia"]),
    ("6. Uczenie + ocena", ["aktualizacja i niezależna", "ocena detekcji"]),
]

# uklad petli: srodki i szerokosci wezlow (zgodnie z ruchem wskazowek od lewej gory)
CX, CY = 0.0, -0.6
H_NODE = 1.4
GAP_A = 0.08
WEZLY_POZ = [
    ((-2.6, 1.55), 3.95),
    ((2.6, 1.55), 3.3),
    ((4.95, -0.6), 3.3),
    ((2.6, -2.75), 3.3),
    ((-2.6, -2.75), 3.7),
    ((-4.95, -0.6), 3.3),
]


def _krawedz(i, strona):
    (x, y), w = WEZLY_POZ[i]
    h = H_NODE
    g = GAP_A
    return {
        "L": np.array([x - w / 2 - g, y, 0.0]),
        "R": np.array([x + w / 2 + g, y, 0.0]),
        "U": np.array([x, y + h / 2 + g, 0.0]),
        "D": np.array([x, y - h / 2 - g, 0.0]),
    }[strona]


# (wyjscie z wezla i, wejscie do wezla i+1, kat luku)
LUKI = [
    ("R", "L", -0.25),
    ("R", "U", -PI / 2),
    ("D", "R", -PI / 2),
    ("L", "R", -0.25),
    ("L", "D", -PI / 2),
    ("U", "L", -PI / 2),
]


def luk_miedzy(i, kolor):
    a, b, kat = LUKI[i]
    m = ArcBetweenPoints(_krawedz(i, a), _krawedz((i + 1) % 6, b), angle=kat,
                         color=kolor, stroke_width=5)
    m.add_tip(tip_length=0.24, tip_width=0.24)
    m.get_tip().set_color(kolor)
    return m


def sciezka_petli():
    """Zamknieta sciezka kropki: luki + przejscia pod wezlami (od srodka wezla 1)."""
    pts = []
    for i in range(6):
        c = np.array([*WEZLY_POZ[i][0], 0.0])
        a, b, kat = LUKI[i]
        wyj = _krawedz(i, a)
        luk = ArcBetweenPoints(wyj, _krawedz((i + 1) % 6, b), angle=kat)
        pts += [c + (wyj - c) * s for s in np.linspace(0, 1, 30)[:-1]]
        pts += [luk.point_from_proportion(s) for s in np.linspace(0, 1, 60)[:-1]]
        wej = _krawedz((i + 1) % 6, b)
        c2 = np.array([*WEZLY_POZ[(i + 1) % 6][0], 0.0])
        pts += [wej + (c2 - wej) * s for s in np.linspace(0, 1, 30)[:-1]]
    pts.append(np.array([*WEZLY_POZ[0][0], 0.0]))
    p = VMobject()
    p.set_points_as_corners(pts)
    return p


def odl_od_wezla(p, i):
    (x, y), w = WEZLY_POZ[i]
    dx = max(0.0, abs(p[0] - x) - w / 2)
    dy = max(0.0, abs(p[1] - y) - H_NODE / 2)
    return float(np.hypot(dx, dy))


class Scena7(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- naglowek (0.0 - 1.0) ---
        krok = Text("KROK 7 / 7", font="Consolas", font_size=22, color=DIM)
        tytul = Text("Pętla zamknięta", font="Segoe UI", weight=BOLD, font_size=40, color=TXT)
        krok.to_corner(UL, buff=0.5)
        tytul.next_to(krok, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(
            FadeIn(krok, shift=0.1 * DOWN),
            FadeIn(tytul, shift=0.1 * DOWN),
            run_time=0.8,
            rate_func=smooth,
        )
        self.wait(0.2)

        # --- wezly ---
        wezly = []
        for (nazwa, opis), ((x, y), wn) in zip(KROKI, WEZLY_POZ):
            rect = RoundedRectangle(
                width=wn, height=H_NODE, corner_radius=0.18,
                fill_color=PANEL, fill_opacity=1, stroke_color=TEAL, stroke_width=3,
            )
            t = Text(nazwa, font="Segoe UI", weight=BOLD, font_size=26, color=TXT)
            d = VGroup(*[Text(l, font="Segoe UI", font_size=22, color=DIM) for l in opis])
            d.arrange(DOWN, buff=0.08)
            tresc = VGroup(t, d).arrange(DOWN, buff=0.14)
            if tresc.width > wn - 0.25:
                tresc.scale_to_fit_width(wn - 0.25)
            g = VGroup(rect, tresc).move_to(np.array([x, y, 0.0]))
            g.set_z_index(3)
            wezly.append(g)

        strzalki = []
        for i in range(6):
            kolor = ORANGE if i == 5 else TEAL
            s = luk_miedzy(i, kolor)
            s.set_z_index(0)
            strzalki.append(s)

        powrot = Text("powrót do kroku 1", font="Segoe UI", font_size=24, color=ORANGE)
        powrot.move_to(np.array([-4.45, 0.47, 0]), aligned_edge=LEFT)

        # --- 1 - 9 s: wezly po kolei ---
        for i in range(6):
            self.play(FadeIn(wezly[i], scale=0.92), run_time=0.5, rate_func=smooth)
            self.wait(0.1)
            self.play(Create(strzalki[i]), run_time=0.5, rate_func=smooth)
            if i == 5:
                self.play(FadeIn(powrot, shift=0.1 * RIGHT), run_time=0.23, rate_func=smooth)
            else:
                self.wait(0.23)

        # --- 9 - 16 s: kropka okraza petle dwa razy ---
        srodek = VGroup(
            Text("krótkie uczenie → nowy stan", font="Segoe UI", font_size=28, color=TXT),
            Text("→ nowy cel → nowe paczki", font="Segoe UI", font_size=28, color=TXT),
        ).arrange(DOWN, buff=0.18).move_to(np.array([CX, CY - 0.35, 0]))
        self.play(FadeIn(srodek, shift=0.1 * UP), run_time=0.5, rate_func=smooth)

        sciezka = sciezka_petli()
        cel = strzalki[5].point_from_proportion(0.35)
        probki = np.linspace(0, 1, 3001)
        START = float(probki[np.argmin([np.linalg.norm(sciezka.point_from_proportion(s) - cel) for s in probki])])
        kropka = VGroup(
            Dot(radius=0.30, color=ORANGE, fill_opacity=0.12),
            Dot(radius=0.20, color=ORANGE, fill_opacity=0.25),
            Dot(radius=0.12, color=ORANGE, fill_opacity=1.0),
            Dot(radius=0.05, color="#FFE2C4", fill_opacity=1.0),
        ).move_to(sciezka.point_from_proportion(START))
        kropka.set_z_index(2)
        self.play(FadeIn(kropka, scale=0.5), run_time=0.2)

        skala = [1.0] * 6

        def ustaw(s):
            p = sciezka.point_from_proportion(s % 1.0)
            kropka.move_to(p)
            for k, g in enumerate(wezly):
                w = max(0.0, 1.0 - odl_od_wezla(p, k) / 0.45)
                w = w * w * (3 - 2 * w)
                nowa = 1.0 + 0.06 * w
                g.scale(nowa / skala[k])
                skala[k] = nowa
                g[0].set_stroke(color=interpolate_color(ManimColor(TEAL), ManimColor(ORANGE), w))

        def okrazenie(m, a):
            ustaw(START + a)

        self.play(UpdateFromAlphaFunc(kropka, okrazenie), run_time=3.9, rate_func=smooth)
        self.play(UpdateFromAlphaFunc(kropka, okrazenie), run_time=2.4, rate_func=smooth)
        ustaw(START)

        # --- 16 - 21 s: karta zamykajaca ---
        def przyciemnij(kolor):
            return interpolate_color(ManimColor(BG), ManimColor(kolor), 0.4)

        self.play(
            FadeOut(srodek), FadeOut(kropka), FadeOut(powrot),
            *[g.animate.set_opacity(0.4) for g in wezly],
            *[s.animate.set_color(przyciemnij(ORANGE if i == 5 else TEAL)) for i, s in enumerate(strzalki)],
            run_time=0.8, rate_func=smooth,
        )
        naglowek_k = Text("Test rozstrzygający", font="Segoe UI", weight=BOLD, font_size=34, color=ORANGE)
        pytanie = VGroup(
            Text("Czy wybrane zdjęcia uczą szybciej niż losowe,", font="Segoe UI", font_size=28, color=TXT),
            Text("przy tym samym budżecie kroków?", font="Segoe UI", font_size=28, color=TXT),
        ).arrange(DOWN, buff=0.16)
        tresc_k = VGroup(naglowek_k, pytanie).arrange(DOWN, buff=0.4)
        panel = RoundedRectangle(
            width=tresc_k.width + 1.0, height=tresc_k.height + 0.8, corner_radius=0.2,
            fill_color=PANEL, fill_opacity=1, stroke_color=GUIDE, stroke_width=3,
        )
        karta = VGroup(panel, tresc_k)
        karta.move_to(np.array([CX, CY, 0]))
        panel.set_z_index(5)
        tresc_k.set_z_index(6)
        self.play(FadeIn(panel, scale=0.95), run_time=0.5, rate_func=smooth)
        self.play(FadeIn(naglowek_k, shift=0.1 * UP), run_time=0.5, rate_func=smooth)
        self.wait(0.3)
        self.play(FadeIn(pytanie, shift=0.1 * UP), run_time=0.7, rate_func=smooth)
        self.wait(2.2)

        # --- 21 - 22 s: wygaszenie ---
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0, rate_func=smooth)
