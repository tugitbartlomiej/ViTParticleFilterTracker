# -*- coding: utf-8 -*-
"""Model 4/6: wyszukiwanie najbliższej prawdziwej klatki w przestrzeni cech ViT (Manim CE 0.20.1).

Sfera to schemat: 600 punktów udaje znormalizowane wektory cech (768 liczb) puli uczącej.
Liczby pokazane jako zmierzone pochodzą z briefu sceny (dane_animacji.json nie ma sekcji
wyszukiwania): pula 461 547 klatek, najbliższa TR_0006_S1_P03 klatka 1193, cosinus 0,875.
Progi stożka (0,35 i 0,86) to parametry rysunku, nie pomiar.
"""
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

SLOT = Path(__file__).resolve().parent
MAT = SLOT.parents[2] / "materialy"
OBRAZ_SD = MAT / "3a00feec_OBRAZ_SD.png"  # ten sam obraz co zapytanie w slot_5
ZNALEZIONE = MAT / "2862ad10_NAJBLIZSZE_ZDJECIE.png"  # TR_0006_S1_P03, klatka 1193

# z briefu sceny
LICZBA_PULI = 461547
COS_NAJBLIZSZY = 0.875

# parametry rysunku
R = 2.5
N_PKT = 600
PROG_SZEROKI = 0.35
PROG_WASKI = 0.86
PHI = 68 * DEGREES
TH_END = -62 * DEGREES
RATE = 0.06
# czas obrotu: R1 + R2 + R3 (patrz construct)
T_ROT = 5.5 + 6.1 + 4.7

C_TEAL = np.array(color_to_rgb(TEAL))
C_ORANGE = np.array(color_to_rgb(ORANGE))
C_DIM = np.array(color_to_rgb(DIM))


def tysiace(n):
    return f"{n:,}".replace(",", " ")


def przecinek(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def baza_styczna(q):
    a = np.array([0.0, 0.0, 1.0]) if abs(q[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = norm(np.cross(q, a))
    e2 = np.cross(q, e1)
    return e1, e2


def obroc_wokol(v, os_, kat):
    os_ = norm(os_)
    return (
        v * np.cos(kat)
        + np.cross(os_, v) * np.sin(kat)
        + os_ * np.dot(os_, v) * (1 - np.cos(kat))
    )


def chmura(q_v, t_v):
    """Punkty w układzie kamery na końcu obrotu (x w prawo, y w górę, z do widza)."""
    rng = np.random.default_rng(20261005)

    def wokol(c, sigma, n):
        c = norm(c)
        g = rng.normal(size=(n, 3)) * sigma
        g -= (g @ c)[:, None] * c
        return norm(c + g)

    blisko = norm(np.cos(np.radians(40)) * q_v + np.sin(np.radians(40)) * t_v)
    grupy = [
        wokol(blisko, 0.13, 80),
        wokol([-0.75, 0.30, 0.45], 0.20, 105),
        wokol([0.25, -0.85, 0.25], 0.17, 95),
        wokol([-0.30, -0.25, -0.90], 0.22, 105),
        wokol([0.80, 0.45, -0.40], 0.17, 85),
        wokol([-0.20, 0.90, -0.35], 0.15, 70),
    ]
    tlo = norm(rng.normal(size=(N_PKT - sum(len(g) for g in grupy), 3)))
    p = np.vstack(grupy + [tlo])

    # nikt poza wyznaczonymi nie może być bliżej niż cos 0,85
    gr = np.arccos(0.85)
    for i in range(len(p)):
        c = p[i] @ q_v
        if c > 0.85:
            w = p[i] - c * q_v
            w = norm(w) if np.linalg.norm(w) > 1e-6 else t_v
            kat = gr + rng.uniform(0.01, 0.06)
            p[i] = np.cos(kat) * q_v + np.sin(kat) * w

    # wyznaczone: najbliższy (cos 0,875) i cztery tuż za nim
    def na_kacie(cos_, azymut_st):
        kier = obroc_wokol(t_v, q_v, np.radians(azymut_st))
        kat = np.arccos(cos_)
        return np.cos(kat) * q_v + np.sin(kat) * kier

    p[0] = na_kacie(COS_NAJBLIZSZY, 0)
    for j, (c, az) in enumerate(
        [(0.872, -38), (0.870, 30), (0.868, -70), (0.866, 62)], start=1
    ):
        p[j] = na_kacie(c, az)
    return p


class Model4(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        self.camera.should_apply_shading = False
        # kamera patrzy nieco ponad środek sfery, więc sfera schodzi niżej od tytułu
        self.camera.frame_center = np.array([0.0, 0.0, 0.42])

        # orientacja końcowa -> układ świata
        self.set_camera_orientation(phi=PHI, theta=TH_END)
        M_end = self.camera.generate_rotation_matrix()
        TH0 = TH_END - RATE * T_ROT
        self.set_camera_orientation(phi=PHI, theta=TH0)

        q_v = norm([0.30, 0.42, 0.86])
        t_v = norm(np.array([1.0, -1.0, 0.0]) - (np.array([1.0, -1.0, 0.0]) @ q_v) * q_v)
        P_v = chmura(q_v, t_v)
        q = M_end.T @ q_v
        P = P_v @ M_end  # wiersze: M_end.T @ p
        cosq = P @ q

        rng = np.random.default_rng(11)
        ranga_poj = rng.permutation(N_PKT) / N_PKT
        ranga_skan = rng.permutation(N_PKT) / N_PKT

        # --- nagłówek (0.0 - 0.8) ---
        krok = Text("MODEL 4 / 6", font=MONO, font_size=22, color=DIM)
        tytul = Text(
            "Wyszukiwanie w przestrzeni cech",
            font=FONT,
            weight=BOLD,
            font_size=40,
            color=TXT,
        )
        krok.to_corner(UL, buff=0.5)
        tytul.next_to(krok, DOWN, buff=0.15, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(krok, tytul)
        self.remove(krok, tytul)
        self.play(
            FadeIn(krok, shift=0.1 * DOWN),
            FadeIn(tytul, shift=0.1 * DOWN),
            run_time=0.8,
            rate_func=smooth,
        )

        # --- sfera (R1: 5.5 s z obrotem) ---
        siatka = VGroup()
        for th in np.radians([30, 60, 90, 120, 150]):
            siatka.add(
                ParametricFunction(
                    lambda t, th=th: R
                    * np.array([np.sin(th) * np.cos(t), np.sin(th) * np.sin(t), np.cos(th)]),
                    t_range=[0, TAU, TAU / 72],
                    stroke_color=GUIDE,
                    stroke_width=1.6,
                    stroke_opacity=0.9,
                )
            )
        for ph in np.radians(np.arange(0, 180, 30)):
            siatka.add(
                ParametricFunction(
                    lambda t, ph=ph: R
                    * np.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)]),
                    t_range=[0, TAU, TAU / 72],
                    stroke_color=GUIDE,
                    stroke_width=1.6,
                    stroke_opacity=0.9,
                )
            )

        self.begin_ambient_camera_rotation(rate=RATE)
        self.play(
            LaggedStart(*[Create(s) for s in siatka], lag_ratio=0.08),
            run_time=1.5,
            rate_func=smooth,
        )

        # punkty: kropki o stałej orientacji, kolor i jasność z jednego updatera
        kropki = [
            Dot(R * p, radius=0.034, fill_color=TEAL, fill_opacity=0.0, stroke_width=0)
            for p in P
        ]
        self.add_fixed_orientation_mobjects(*kropki)

        pojaw = ValueTracker(0.0)
        skan = ValueTracker(0.0)
        prog = ValueTracker(PROG_SZEROKI)

        skale = np.ones(N_PKT)

        def stan(_m):
            Mrot = self.camera.get_rotation_matrix()
            z = (P @ Mrot.T)[:, 2]
            glebia = 0.28 + 0.72 * np.clip((z + 1) / 2, 0, 1) ** 1.3
            a_poj = np.clip((pojaw.get_value() - ranga_poj) / 0.12, 0, 1)
            w_skan = np.clip((skan.get_value() - ranga_skan) / 0.06, 0, 1)
            w_in = np.clip((cosq - prog.get_value() + 0.003) / 0.006, 0, 1)
            w_in = w_in * w_in * (3 - 2 * w_in)
            cel = C_DIM[None, :] * (1 - w_in[:, None]) + C_ORANGE[None, :] * w_in[:, None]
            kol = C_TEAL[None, :] * (1 - w_skan[:, None]) + cel * w_skan[:, None]
            op_cel = 0.22 * (1 - w_in) + 1.0 * w_in
            op = (0.85 * (1 - w_skan) + op_cel * w_skan) * a_poj
            # orientacja na wyjściu z przedniej półkuli nie dotyczy pomarańczowych
            op = op * (glebia * (1 - 0.6 * w_in * w_skan) + 0.6 * w_in * w_skan)
            skala = 1.0 + 0.8 * w_in * w_skan
            for i, d in enumerate(kropki):
                d.fill_rgbas = np.array([[kol[i, 0], kol[i, 1], kol[i, 2], op[i]]])
                if abs(skala[i] - skale[i]) > 1e-3:
                    d.scale(skala[i] / skale[i])
                    skale[i] = skala[i]

        sterownik = Mobject()
        sterownik.add_updater(stan)
        self.add(sterownik)

        self.play(pojaw.animate.set_value(1.15), run_time=2.0, rate_func=smooth)
        self.wait(0.3)

        # wektor zapytania
        strzalka = Arrow3D(
            start=ORIGIN,
            end=R * 1.02 * q,
            color=ORANGE,
            thickness=0.022,
            height=0.24,
            base_radius=0.075,
        )
        self.play(GrowFromPoint(strzalka, ORIGIN), run_time=1.0, rate_func=smooth)

        # kafelek obrazu SD (lewy dół)
        TILE_H = 1.75
        sd_img = ImageMobject(str(OBRAZ_SD))
        sd_img.height = TILE_H
        sd_ramka = RoundedRectangle(
            width=sd_img.width + 0.12,
            height=sd_img.height + 0.12,
            corner_radius=0.08,
            stroke_color=ORANGE,
            stroke_width=3,
            fill_color=PANEL,
            fill_opacity=1,
        )
        sd_img.move_to(sd_ramka)
        sd_kafel = Group(sd_ramka, sd_img)
        sd_kafel.move_to([-6.55 + sd_ramka.width / 2, -1.6, 0])
        sd_podpis = Text("zapytanie: obraz SD", font=FONT, font_size=24, color=TXT)
        sd_podpis.next_to(sd_ramka, DOWN, buff=0.18)
        sd_grupa = Group(sd_kafel, sd_podpis)
        self.add_fixed_in_frame_mobjects(sd_grupa)
        self.remove(sd_grupa)
        self.play(FadeIn(sd_grupa, shift=0.15 * UP), run_time=0.7, rate_func=smooth)

        # nieruchome ujęcie 1: sfera, punkty, zapytanie
        self.stop_ambient_camera_rotation()
        self.wait(1.5)

        # --- stożek cosinusa (R2: 6.1 s z obrotem) ---
        self.begin_ambient_camera_rotation(rate=RATE)
        e1, e2 = baza_styczna(q)
        stozek_a = ValueTracker(0.0)

        def rysuj_stozek():
            al = np.arccos(prog.get_value())
            a = stozek_a.get_value()

            def kier(v):
                return np.cos(al) * q + np.sin(al) * (np.cos(v) * e1 + np.sin(v) * e2)

            pow_ = Surface(
                lambda u, v: u * R * kier(v),
                u_range=[0, 1],
                v_range=[0, TAU],
                resolution=(1, 48),
                fill_color=ORANGE,
                fill_opacity=0.14 * a,
                checkerboard_colors=False,
                stroke_width=0,
            )
            obw = ParametricFunction(
                lambda v: R * kier(v),
                t_range=[0, TAU, TAU / 96],
                stroke_color=ORANGE,
                stroke_width=3,
                stroke_opacity=0.95 * a,
            )
            tworzace = VGroup(
                *[
                    Line(ORIGIN, R * kier(v), stroke_color=ORANGE, stroke_width=1.4,
                         stroke_opacity=0.45 * a)
                    for v in np.linspace(0, TAU, 6, endpoint=False)
                ]
            )
            return VGroup(pow_, tworzace, obw)

        stozek = always_redraw(rysuj_stozek)
        self.add(stozek)
        self.play(stozek_a.animate.set_value(1.0), run_time=1.0, rate_func=smooth)

        # liczniki (prawy górny róg, stałe w kadrze)
        X_PR = 6.6
        lab_skan = Text("przeszukano", font=MONO, font_size=26, color=DIM)
        lab_prog = Text("próg cos", font=MONO, font_size=26, color=DIM)
        wzor_liczby = Text(tysiace(LICZBA_PULI), font=MONO, font_size=28, color=TXT)
        x_lab = X_PR - wzor_liczby.width - 0.35 - lab_skan.width
        lab_skan.move_to([x_lab, 3.45, 0], aligned_edge=LEFT)
        lab_prog.move_to([x_lab, 2.95, 0], aligned_edge=LEFT)

        def staly(fn):
            m = fn()

            def upd(mob):
                mob.become(fn())
                self.camera.add_fixed_in_frame_mobjects(mob)

            m.add_updater(upd)
            return m

        def liczba_prog():
            t = Text(przecinek(prog.get_value()), font=MONO, font_size=28, color=ORANGE)
            t.move_to([X_PR, 2.95, 0], aligned_edge=RIGHT)
            return t

        def liczba_skan():
            n = int(round(LICZBA_PULI * min(skan.get_value(), 1.0)))
            t = Text(tysiace(n), font=MONO, font_size=28, color=TEAL)
            t.move_to([X_PR, 3.45, 0], aligned_edge=RIGHT)
            return t

        num_prog = staly(liczba_prog)
        g_prog = VGroup(lab_prog, num_prog)
        self.add_fixed_in_frame_mobjects(g_prog)
        self.remove(g_prog)
        self.play(FadeIn(g_prog, shift=0.1 * DOWN), run_time=0.5, rate_func=smooth)

        num_skan = staly(liczba_skan)
        g_skan = VGroup(lab_skan, num_skan)
        self.add_fixed_in_frame_mobjects(g_skan)
        self.remove(g_skan)
        self.play(FadeIn(g_skan, shift=0.1 * DOWN), run_time=0.4, rate_func=smooth)

        self.play(skan.animate.set_value(1.08), run_time=4.0, rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.2)

        # nieruchome ujęcie 2: szeroki stożek, wiele trafień
        self.stop_ambient_camera_rotation()
        self.wait(1.5)

        # --- zwężanie (R3: 4.7 s z obrotem) ---
        self.begin_ambient_camera_rotation(rate=RATE)
        self.play(
            prog.animate.set_value(PROG_WASKI),
            run_time=4.5,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(0.2)

        # nieruchome ujęcie 3: wąski stożek, kilka trafień
        self.stop_ambient_camera_rotation()
        self.wait(1.5)

        # --- najbliższa klatka (bez obrotu) ---
        for m in (stozek, num_prog, num_skan):
            m.clear_updaters()

        p0 = R * P[0]
        xy = self.camera.project_point(p0)
        xy = np.array([xy[0], xy[1], 0.0])
        rdzen = Dot(xy, radius=0.07, fill_color=ORANGE, stroke_color=TXT, stroke_width=2.5)
        self.add_fixed_in_frame_mobjects(rdzen)
        self.remove(rdzen)
        self.play(FadeIn(rdzen, scale=0.4), run_time=0.3, rate_func=smooth)
        for _ in range(2):
            fala = Circle(radius=0.1, stroke_color=ORANGE, stroke_width=4.5).move_to(xy)
            self.add_fixed_in_frame_mobjects(fala)
            self.play(
                fala.animate(rate_func=rate_functions.ease_out_cubic)
                .scale(5.0)
                .set_stroke(opacity=0),
                rdzen.animate(rate_func=there_and_back).scale(1.7),
                run_time=0.6,
            )
            self.remove(fala)

        zn_img = ImageMobject(str(ZNALEZIONE))
        zn_img.height = TILE_H
        zn_ramka = RoundedRectangle(
            width=zn_img.width + 0.12,
            height=zn_img.height + 0.12,
            corner_radius=0.08,
            stroke_color=TEAL,
            stroke_width=3,
            fill_color=PANEL,
            fill_opacity=1,
        )
        zn_img.move_to(zn_ramka)
        zn_kafel = Group(zn_ramka, zn_img)
        zn_kafel.move_to([6.55 - zn_ramka.width / 2, -1.6, 0])
        self.add_fixed_in_frame_mobjects(zn_kafel)
        self.remove(zn_kafel)
        self.play(FadeIn(zn_kafel, shift=0.15 * UP), run_time=0.7, rate_func=smooth)

        cel = zn_ramka.get_corner(UL) + np.array([0.0, -0.35, 0.0])
        kier_l = cel - xy
        kier_l /= np.linalg.norm(kier_l)
        linia = Line(xy + 0.1 * kier_l, cel, stroke_color=TEAL, stroke_width=2.2)
        self.add_fixed_in_frame_mobjects(linia)
        self.remove(linia)
        self.play(Create(linia), run_time=0.7, rate_func=smooth)

        cos_t = Text("cos " + przecinek(COS_NAJBLIZSZY, 3), font=MONO, font_size=30, color=TXT)
        cos_t[:3].set_color(TEAL)
        cos_t.next_to(zn_ramka, DOWN, buff=0.18)
        cos_t.align_to(zn_ramka, RIGHT)
        self.add_fixed_in_frame_mobjects(cos_t)
        self.remove(cos_t)
        self.play(FadeIn(cos_t, shift=0.1 * UP), run_time=0.5, rate_func=smooth)

        id_t = Text("TR_0006, klatka 1193", font=MONO, font_size=24, color=DIM)
        id_t.next_to(cos_t, DOWN, buff=0.12)
        id_t.align_to(zn_ramka, RIGHT)
        self.add_fixed_in_frame_mobjects(id_t)
        self.remove(id_t)
        self.play(FadeIn(id_t, shift=0.1 * UP), run_time=0.5, rate_func=smooth)

        # nieruchome ujęcie 4: znaleziona klatka
        self.wait(2.5)

        # --- wygaszenie ---
        sterownik.clear_updaters()
        self.play(
            *[
                FadeOut(m)
                for m in self.mobjects
                if m is not sterownik and not isinstance(m, ValueTracker)
            ],
            run_time=0.6,
            rate_func=smooth,
        )
        self.wait(0.05)
