# -*- coding: utf-8 -*-
"""MODEL 1 / 6: jak siec sie uczy (Manim CE 0.20.1).

Schemat jednego kroku uczenia: paczka zdjec -> tokeny -> stem -> 6 blokow -> glowa -> ramka,
strata, gradient wstecz zmieniajacy mape wag w kazdym bloku, przesuniecie chmury tokenow w latencie.
Ramki przewidywane, dlugosc slupka straty i chmura punktow sa schematyczne (bez liczb).
Ramka etykiety pochodzi z prawdziwej adnotacji klatki TR_0001_S1_P03_00000006.
"""
from pathlib import Path

import numpy as np
from manim import *
from PIL import Image

BG = "#0E1A24"
PANEL = "#16293A"
GUIDE = "#24425A"
TXT = "#EAF2F5"
DIM = "#7C93A3"
TEAL = "#2BB3C0"
ORANGE = "#F08A2E"
GREEN = "#5BC27A"
RED = "#E0565B"
W_CIEMNY = "#164A57"
W_JASNY = "#8FE6EE"

TU = Path(__file__).resolve().parent
EDIT = TU.parents[1]
KLATKI = [l.strip() for l in (EDIT / "klatki_puli.txt").read_text(encoding="utf-8").splitlines()
          if l.strip()][:4]
# adnotacja pierwszej klatki (pool_manifest.jsonl): bbox xywh w pikselach 720 x 480
GT_BOX = (340.0, 208.0, 221.0, 52.0)
IMG_PX = (720.0, 480.0)

ease_out = rate_functions.ease_out_cubic
sine = rate_functions.ease_in_out_sine

config.background_color = BG


class Lamana:
    """Lamana parametryzowana dlugoscia luku, s w [0, 1]."""

    def __init__(self, pts):
        self.p = np.array([[q[0], q[1], 0.0] for q in pts], float)
        seg = np.diff(self.p, axis=0)
        self.l = np.linalg.norm(seg, axis=1)
        self.c = np.concatenate([[0.0], np.cumsum(self.l)])
        self.L = float(self.c[-1])

    def pos(self, s):
        d = float(np.clip(s, 0.0, 1.0)) * self.L
        i = int(np.clip(np.searchsorted(self.c, d, side="right") - 1, 0, len(self.l) - 1))
        a = (d - self.c[i]) / self.l[i]
        return self.p[i] + a * (self.p[i + 1] - self.p[i])

    def s_na_odcinku(self, i, punkt):
        """Proporcja dlugosci dla punktu lezacego na odcinku i."""
        d = self.c[i] + np.linalg.norm(np.array([punkt[0], punkt[1], 0.0]) - self.p[i])
        return d / self.L


def gladko(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def kolor_wagi(v):
    return interpolate_color(ManimColor(W_CIEMNY), ManimColor(W_JASNY), float(np.clip(v, 0, 1)))


class Model1(Scene):
    def czekaj_do(self, t):
        dt = t - self.time
        if dt > 1e-3:
            self.wait(dt)

    def construct(self):
        rng = np.random.default_rng(11)
        Y = 0.55  # os sieci

        # ---------------- naglowek ----------------
        nr = Text("MODEL 1 / 6", font="Consolas", font_size=22, color=DIM)
        tytul = Text("Jak sieć się uczy", font="Segoe UI", weight=BOLD, font_size=40, color=TXT)
        nr.to_corner(UL, buff=0.5)
        tytul.next_to(nr, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play(FadeIn(nr, shift=0.1 * DOWN), FadeIn(tytul, shift=0.1 * DOWN),
                  run_time=0.8, rate_func=smooth)
        self.czekaj_do(1.0)

        # ---------------- 1. paczka 4 zdjec ----------------
        IMG_C = np.array([-5.25, Y, 0])
        IMG_W = 2.7
        kafle = []
        for i, p in enumerate(KLATKI):
            im = ImageMobject(p)
            im.width = 0.95
            kafle.append(im)
        xs = [-6.45, -5.4, -4.35, -3.3]
        for im, x in zip(kafle, xs):
            im.move_to([x - 9.0, Y, 0])
        # kolejnosc rysowania: pierwsza klatka na wierzchu
        for im in reversed(kafle):
            self.add(im)
        self.play(LaggedStart(*[im.animate.move_to([x, Y, 0]) for im, x in zip(kafle, xs)],
                              lag_ratio=0.12),
                  run_time=1.1, rate_func=ease_out)

        # zbierz w talie: przednie zdjecie duze, trzy za nim
        anims = []
        for k, im in enumerate(kafle):
            cel = IMG_C + k * np.array([-0.1, 0.1, 0])
            anims.append(im.animate.set(width=IMG_W).move_to(cel).set_opacity(1.0 if k == 0 else 0.45))
        # play dodaje obiekty w kolejnosci animacji: przednie zdjecie ostatnie
        self.play(*reversed(anims), run_time=0.9, rate_func=smooth)
        img = kafle[0]
        self.bring_to_front(img)
        ramka_img = SurroundingRectangle(img, buff=0.0, color=GUIDE, stroke_width=2)
        self.add(ramka_img)

        L, R = img.get_left()[0], img.get_right()[0]
        T, B = img.get_top()[1], img.get_bottom()[1]
        W, H = R - L, T - B

        def px(x, y):
            return np.array([L + x / IMG_PX[0] * W, T - y / IMG_PX[1] * H, 0.0])

        # siatka tokenow (rzadsza: 15 x 10 linii zamiast 30 x 20)
        linie = VGroup()
        for c in range(1, 15):
            x = L + c * W / 15
            linie.add(Line([x, T, 0], [x, B, 0]))
        for r in range(1, 10):
            y = T - r * H / 10
            linie.add(Line([L, y, 0], [R, y, 0]))
        linie.set_stroke(TEAL, width=1.2, opacity=0.55)
        self.play(LaggedStart(*[Create(l) for l in linie], lag_ratio=0.05),
                  run_time=1.0, rate_func=smooth)

        # ---------------- 2. siec ----------------
        hs = [1.5, 1.22, 0.98, 0.78, 0.6]
        tw, tg = 0.2, 0.06
        stem = VGroup()
        x = 0.0
        for i in range(4):
            p = Polygon([x, hs[i] / 2, 0], [x + tw, hs[i + 1] / 2, 0],
                        [x + tw, -hs[i + 1] / 2, 0], [x, -hs[i] / 2, 0])
            p.set_stroke(TEAL, width=2.5).set_fill(TEAL, opacity=0.15 + 0.1 * i)
            stem.add(p)
            x += tw + tg
        stem.move_to([-2.84, Y, 0])
        lab_stem = Text("stem", font="Segoe UI", font_size=24, color=TXT)
        lab_stem.move_to([stem.get_x(), Y - 1.15, 0])

        os_linia = Line([-3.62, Y, 0], [2.12, Y, 0]).set_stroke(GUIDE, width=2)
        self.add(os_linia)
        os_linia.set_stroke(opacity=0)
        self.play(os_linia.animate.set_stroke(opacity=1), LaggedStart(*[FadeIn(p, shift=0.1 * RIGHT) for p in stem], lag_ratio=0.25),
                  FadeIn(lab_stem, shift=0.08 * UP), run_time=0.7, rate_func=ease_out)

        bw, bh, bgap = 0.56, 1.5, 0.1
        n_w = 6
        cel_w = 0.068
        krok_w = 0.08
        bloki, blaski, mapy = VGroup(), VGroup(), []
        for i in range(6):
            rr = RoundedRectangle(width=bw, height=bh, corner_radius=0.08)
            rr.set_stroke(TEAL, width=2.5).set_fill(PANEL, opacity=1)
            bloki.add(rr)
        bloki.arrange(RIGHT, buff=bgap).move_to([-0.22, Y, 0])
        wagi = [rng.random((n_w, n_w)) * 0.8 + 0.1 for _ in range(6)]
        for i, rr in enumerate(bloki):
            bl = RoundedRectangle(width=bw, height=bh, corner_radius=0.08)
            bl.set_stroke(width=0).set_fill(TEAL, opacity=0.0).move_to(rr)
            blaski.add(bl)
            m = VGroup()
            for a in range(n_w):
                for b in range(n_w):
                    sq = Square(side_length=cel_w).set_stroke(width=0)
                    sq.set_fill(kolor_wagi(wagi[i][a, b]), opacity=1)
                    sq.move_to(rr.get_center() + np.array([(b - 2.5) * krok_w, (2.5 - a) * krok_w, 0]))
                    m.add(sq)
            mapy.append(m)
        mapy_g = VGroup(*mapy)
        lab_blk = Text("6 bloków transformera", font="Segoe UI", font_size=24, color=TXT)
        lab_blk.move_to([bloki.get_x(), Y - 1.15, 0])
        self.play(LaggedStart(*[FadeIn(VGroup(b, m), shift=0.1 * UP) for b, m in zip(bloki, mapy)],
                              lag_ratio=0.15),
                  FadeIn(lab_blk, shift=0.08 * UP),
                  run_time=1.0, rate_func=ease_out)
        self.add(blaski)
        self.bring_to_front(mapy_g)

        glowa = Polygon([0, 0.45, 0], [0.45, 0.25, 0], [0.45, -0.25, 0], [0, -0.45, 0])
        glowa.set_stroke(TEAL, width=2.5).set_fill(TEAL, opacity=0.2)
        glowa.move_to([2.12, Y, 0])
        lab_gl = Text("głowa", font="Segoe UI", font_size=24, color=TXT)
        lab_gl.move_to([2.1, Y - 1.15, 0])
        self.play(FadeIn(glowa, shift=0.1 * RIGHT), FadeIn(lab_gl, shift=0.08 * UP),
                  run_time=0.5, rate_func=ease_out)

        # panel latentu
        panel = RoundedRectangle(width=3.85, height=4.5, corner_radius=0.15)
        panel.set_stroke(GUIDE, width=2).set_fill(PANEL, opacity=1)
        panel.move_to([4.85, 0.05, 0])
        lab_lat = Text("latent", font="Segoe UI", font_size=26, color=DIM)
        lab_lat.next_to(panel, UP, buff=0.15, aligned_edge=LEFT)
        leg_n = VGroup(Dot(radius=0.06, color=ORANGE),
                       Text("narzędzie", font="Segoe UI", font_size=24, color=TXT)).arrange(RIGHT, buff=0.12)
        leg_t = VGroup(Dot(radius=0.06, color=TEAL),
                       Text("tło", font="Segoe UI", font_size=24, color=TXT)).arrange(RIGHT, buff=0.12)
        leg = VGroup(leg_n, leg_t).arrange(RIGHT, buff=0.45)
        leg.next_to(panel, DOWN, buff=0.18)
        self.play(FadeIn(panel, scale=0.98), FadeIn(lab_lat, shift=0.08 * UP),
                  FadeIn(leg, shift=0.08 * UP), run_time=0.6, rate_func=ease_out)

        # chmura tokenow w latencie (schematyczna)
        PC = panel.get_center()
        N_ALL, N_TOOL = 120, 15
        baza = np.clip(rng.normal(0, 1, size=(N_ALL, 2)), -2.2, 2.2)
        narz = np.zeros(N_ALL, bool)
        narz[:N_TOOL] = True
        # etapy rozdzielenia 0..4
        c_tool = [np.array([0.15, 0.05]), np.array([0.45, 0.4]), np.array([0.7, 0.7]),
                  np.array([0.88, 0.95]), np.array([0.98, 1.12])]
        c_bg = [np.array([0.0, 0.0]), np.array([-0.12, -0.15]), np.array([-0.22, -0.28]),
                np.array([-0.3, -0.38]), np.array([-0.34, -0.45])]
        s_tool = [0.62, 0.48, 0.38, 0.3, 0.25]
        s_bg = [0.72, 0.7, 0.67, 0.65, 0.63]

        def poz_etap(e):
            out = np.zeros((N_ALL, 3))
            for j in range(N_ALL):
                if narz[j]:
                    q = c_tool[e] + s_tool[e] * baza[j]
                else:
                    q = c_bg[e] + s_bg[e] * baza[j]
                q = np.clip(q, [-1.6, -1.95], [1.6, 1.95])
                out[j] = [PC[0] + q[0], PC[1] + q[1], 0.0]
            return out

        ETAPY = [poz_etap(e) for e in range(5)]
        kropki = VGroup()
        for j in range(N_ALL):
            d = Dot(radius=0.05, color=ORANGE if narz[j] else TEAL)
            d.move_to(ETAPY[0][j])
            d.set_opacity(0.0)
            kropki.add(d)
        # narzedzie na wierzchu
        kropki.submobjects = kropki.submobjects[N_TOOL:] + kropki.submobjects[:N_TOOL]
        kropki_idx = list(range(N_TOOL, N_ALL)) + list(range(N_TOOL))
        self.add(kropki)

        # licznik krokow
        def licznik_txt(k):
            t = Text(f"krok {k}", font="Consolas", font_size=30, color=TXT)
            t.move_to([0, 3.45, 0]).align_to(panel, RIGHT)
            return t

        licznik = licznik_txt(1)
        self.play(FadeIn(licznik, shift=0.08 * DOWN), run_time=0.4, rate_func=ease_out)

        # ---------------- tory impulsow ----------------
        tor_fw = Lamana([[-3.62, Y], [2.12, Y]])
        s_blok_fw = [tor_fw.s_na_odcinku(0, b.get_center()) for b in bloki]
        s_stem_fw = [tor_fw.s_na_odcinku(0, p.get_center()) for p in stem]
        s_glowa_fw = tor_fw.s_na_odcinku(0, glowa.get_center())

        def impuls(kolor):
            g = VGroup()
            for j in range(7):  # ogon
                g.add(Dot(radius=0.07 * (1 - j / 9), color=kolor).set_opacity(0.0))
            for r, o in [(0.24, 0.12), (0.16, 0.25), (0.11, 0.5), (0.07, 1.0)]:
                g.add(Dot(radius=r, color=kolor).set_opacity(0.0))
            return g

        def ustaw_impuls(g, tor, s, widok, ds=0.012):
            for j in range(7):
                g[j].move_to(tor.pos(s - (j + 1) * ds))
                g[j].set_opacity(widok * 0.7 * (1 - j / 7) if s - (j + 1) * ds > 0 else 0.0)
            for k, o in enumerate([0.12, 0.25, 0.5, 1.0]):
                g[7 + k].move_to(tor.pos(s))
                g[7 + k].set_opacity(widok * o)

        # stan gradientu dla mapy wag
        stan = {"stare": [w.copy() for w in wagi], "nowe": [w.copy() for w in wagi]}

        def ustaw_mapy(alfy):
            for i, m in enumerate(mapy):
                a = gladko(alfy[i])
                v = (1 - a) * stan["stare"][i] + a * stan["nowe"][i]
                for idx, sq in enumerate(m):
                    sq.set_fill(kolor_wagi(v[idx // n_w, idx % n_w]), opacity=1)

        # ---------------- tokeny wplywaja do sieci ----------------
        arr = np.asarray(Image.open(KLATKI[0]).convert("RGB"))
        komorki = rng.choice(150, size=12, replace=False)

        def tokeny_wplyw(tempo, extra=()):
            sqs = VGroup()
            for k, idx in enumerate(sorted(komorki)):
                cx, cy = idx % 15, idx // 15
                pxx = int((cx + 0.5) / 15 * arr.shape[1])
                pyy = int((cy + 0.5) / 10 * arr.shape[0])
                col = rgb_to_color(arr[pyy, pxx] / 255.0)
                sq = Square(side_length=0.12).set_stroke(TXT, width=1, opacity=0.6)
                sq.set_fill(col, opacity=1)
                sq.move_to([L + (cx + 0.5) * W / 15, T - (cy + 0.5) * H / 10, 0])
                sqs.add(sq)
            self.play(LaggedStart(*[FadeIn(s, scale=1.6) for s in sqs], lag_ratio=0.04), *extra,
                      run_time=0.4 * tempo, rate_func=ease_out)
            kol_y = np.linspace(Y + 0.82, Y - 0.82, 12)
            self.play(LaggedStart(*[s.animate.move_to([-3.62, y, 0]) for s, y in zip(sqs, kol_y)],
                                  lag_ratio=0.05),
                      run_time=0.8 * tempo, rate_func=smooth)
            self.play(LaggedStart(*[s.animate.move_to([stem.get_left()[0] + 0.05, Y, 0])
                                    .scale(0.3).set_opacity(0) for s in sqs],
                                  lag_ratio=0.04),
                      run_time=0.6 * tempo, rate_func=rate_functions.ease_in_cubic)
            self.remove(sqs)

        def wygas():
            an = []
            for i, p in enumerate(stem):
                an.append(p.animate.set_fill(TEAL, opacity=0.15 + 0.1 * i))
            an.append(glowa.animate.set_fill(TEAL, opacity=0.2))
            for b, bl in zip(bloki, blaski):
                an.append(b.animate.set_stroke(TEAL, width=2.5))
                an.append(bl.animate.set_fill(TEAL, opacity=0.0))
            return an

        # ---------------- przejscie w przod ----------------
        def w_przod(tempo, pokaz_chmure=False, etap=0):
            tr = ValueTracker(0.0)
            imp = impuls(TEAL)
            self.add(imp)

            def upd(m):
                s = tr.get_value()
                widok = gladko(min(s / 0.04, (1 - s) / 0.03, 1.0))
                ustaw_impuls(imp, tor_fw, s, widok)
                for i, bl in enumerate(blaski):
                    g = np.exp(-((s - s_blok_fw[i]) * tor_fw.L / 0.45) ** 2)
                    bl.set_fill(TEAL, opacity=0.5 * g)
                    bloki[i].set_stroke(interpolate_color(ManimColor(TEAL), ManimColor(W_JASNY), g),
                                        width=2.5 + 2.5 * g)
                for i, p in enumerate(stem):
                    g = np.exp(-((s - s_stem_fw[i]) * tor_fw.L / 0.3) ** 2)
                    p.set_fill(TEAL, opacity=0.15 + 0.1 * i + 0.5 * g)
                g = np.exp(-((s - s_glowa_fw) * tor_fw.L / 0.35) ** 2)
                glowa.set_fill(TEAL, opacity=0.2 + 0.55 * g)
                if pokaz_chmure:
                    a = gladko((s - s_blok_fw[5]) / 0.16)
                    zrodlo = bloki[5].get_center()
                    for d, j in zip(kropki, kropki_idx):
                        aj = gladko(a * 1.6 - (j % 23) / 23 * 0.6)
                        d.move_to(zrodlo + aj * (ETAPY[etap][j] - zrodlo))
                        d.set_opacity(aj)

            imp.add_updater(upd)
            self.play(tr.animate.set_value(1.0), run_time=1.9 * tempo, rate_func=sine)
            imp.clear_updaters()
            self.remove(imp)

        # luk: glowa -> ramka na zdjeciu
        def tor_luk(cel):
            p0 = np.array([2.12, Y + 0.45, 0])
            p1 = np.array([2.0, 2.55, 0])
            p2 = np.array([-3.2, 2.6, 0])
            p3 = np.array(cel)
            ts = np.linspace(0, 1, 60)
            pts = [((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3
                   for t in ts]
            return Lamana([[q[0], q[1]] for q in pts])

        def lot_luk(cel, tempo, extra=()):
            tor = tor_luk(cel)
            tr = ValueTracker(0.0)
            imp = impuls(ORANGE)
            self.add(imp)
            imp.add_updater(lambda m: ustaw_impuls(m, tor, tr.get_value(),
                                                   gladko(min(tr.get_value() / 0.05, 1.0)), ds=0.01))
            self.play(tr.animate.set_value(1.0), *extra, run_time=0.8 * tempo, rate_func=sine)
            imp.clear_updaters()
            self.remove(imp)

        # ramki
        gx, gy, gw, gh = GT_BOX
        dxy = np.array([-130.0, 95.0])
        dwh = np.array([-80.0, 70.0])
        F_PRED = [1.0, 0.55, 0.26, 0.07]

        def ramka_pred(k):
            f = F_PRED[k]
            cx = gx + gw / 2 + f * dxy[0]
            cy = gy + gh / 2 + f * dxy[1]
            w_ = gw + f * dwh[0]
            h_ = gh + f * dwh[1]
            r = Rectangle(width=w_ / IMG_PX[0] * W, height=h_ / IMG_PX[1] * H)
            r.set_stroke(ORANGE, width=4).set_fill(opacity=0)
            r.move_to(px(cx, cy))
            return r

        ramka_gt = Rectangle(width=gw / IMG_PX[0] * W, height=gh / IMG_PX[1] * H)
        ramka_gt.set_stroke(GREEN, width=4).set_fill(opacity=0)
        ramka_gt.move_to(px(gx + gw / 2, gy + gh / 2))

        lab_gt = VGroup(Line(ORIGIN, 0.35 * RIGHT).set_stroke(GREEN, width=4),
                        Text("etykieta", font="Segoe UI", font_size=24, color=TXT)).arrange(RIGHT, buff=0.15)
        lab_pr = VGroup(Line(ORIGIN, 0.35 * RIGHT).set_stroke(ORANGE, width=4),
                        Text("przewidywanie", font="Segoe UI", font_size=24, color=TXT)).arrange(RIGHT,
                                                                                               buff=0.15)
        lab_pr.move_to([0, Y - 1.15, 0]).align_to(img, LEFT)
        lab_gt.move_to([0, Y - 1.55, 0]).align_to(img, LEFT)

        # strata
        STR_Y = Y - 2.65
        lab_str = Text("strata", font="Segoe UI", font_size=24, color=TXT)
        lab_str.move_to([0, STR_Y + 0.42, 0]).align_to(img, LEFT)
        STR_MAX = W
        tlo_str = Rectangle(width=STR_MAX, height=0.24).set_stroke(GUIDE, width=1.5)
        tlo_str.set_fill(PANEL, opacity=1).move_to([L + STR_MAX / 2, STR_Y, 0])
        DL_STR = [1.0, 0.64, 0.36, 0.13]
        dl = ValueTracker(0.0)

        def slupek():
            frac = max(dl.get_value(), 1e-3)
            col = interpolate_color(ManimColor(RED), ManimColor(GREEN), float(np.clip(1 - frac, 0, 1)) ** 1.3)
            r = Rectangle(width=STR_MAX * frac, height=0.24).set_stroke(width=0).set_fill(col, opacity=1)
            r.move_to([L + STR_MAX * frac / 2, STR_Y, 0])
            return r

        bar = always_redraw(slupek)

        # strzalki gradientu nad blokami
        strz = VGroup()
        for b in bloki:
            a = Arrow([b.get_x(), b.get_top()[1] + 0.5, 0], [b.get_x(), b.get_top()[1] + 0.04, 0],
                      buff=0, stroke_width=4, color=ORANGE, tip_length=0.14,
                      max_tip_length_to_length_ratio=0.4)
            a.set_opacity(0)
            strz.add(a)
        self.add(strz)

        # tor wstecz: koniec slupka -> w prawo pod siecia -> w gore przy glowie -> w lewo do stemu
        X_PION = 2.62

        def tor_wstecz():
            x0 = L + STR_MAX * dl.get_value()
            return Lamana([[x0 + 0.05, STR_Y], [X_PION, STR_Y], [X_PION, Y], [stem.get_x(), Y]])

        def wstecz(tempo, k):
            tor = tor_wstecz()
            s_bl = [tor.s_na_odcinku(2, b.get_center()) for b in bloki]
            s_gl = tor.s_na_odcinku(2, glowa.get_center())
            s_st = [tor.s_na_odcinku(2, p.get_center()) for p in stem]
            # nowe wagi: niewielka zmiana, malejaca z krokami
            stan["stare"] = [w.copy() for w in stan["nowe"]]
            skala = [0.22, 0.16, 0.12, 0.09][k]
            stan["nowe"] = [np.clip(w + rng.normal(0, skala, w.shape), 0.04, 0.98) for w in stan["stare"]]
            tr = ValueTracker(0.0)
            imp = impuls(ORANGE)
            self.add(imp)
            self.bring_to_front(imp)

            def upd(m):
                s = tr.get_value()
                widok = gladko(min(s / 0.03, (1 - s) / 0.03, 1.0))
                ustaw_impuls(imp, tor, s, widok, ds=0.008)
                alfy = []
                for i, bl in enumerate(blaski):
                    g = np.exp(-((s - s_bl[i]) * tor.L / 0.45) ** 2)
                    bl.set_fill(ORANGE, opacity=0.45 * g)
                    bloki[i].set_stroke(interpolate_color(ManimColor(TEAL), ManimColor(ORANGE), g),
                                        width=2.5 + 2.5 * g)
                    a = (s - s_bl[i]) * tor.L / 0.6 + 0.5
                    alfy.append(a)
                    strz[i].set_opacity(gladko(a * 2.0))
                ustaw_mapy(alfy)
                g = np.exp(-((s - s_gl) * tor.L / 0.35) ** 2)
                glowa.set_fill(interpolate_color(ManimColor(TEAL), ManimColor(ORANGE), g), opacity=0.2 + 0.5 * g)
                for i, p in enumerate(stem):
                    g = np.exp(-((s - s_st[i]) * tor.L / 0.3) ** 2)
                    p.set_fill(interpolate_color(ManimColor(TEAL), ManimColor(ORANGE), g),
                               opacity=0.15 + 0.1 * i + 0.5 * g)

            imp.add_updater(upd)
            self.play(tr.animate.set_value(1.0), run_time=2.6 * tempo, rate_func=sine)
            imp.clear_updaters()
            ustaw_mapy([1.0] * 6)
            self.remove(imp)

        def przesun_latent(e_z, e_do, tempo, extra=()):
            tr = ValueTracker(0.0)

            def upd(m):
                a = tr.get_value()
                for d, j in zip(kropki, kropki_idx):
                    d.move_to((1 - a) * ETAPY[e_z][j] + a * ETAPY[e_do][j])

            kropki.add_updater(upd)
            self.play(tr.animate.set_value(1.0), *extra, run_time=1.4 * tempo, rate_func=smooth)
            kropki.clear_updaters()

        # ================= KROK 1 =================
        tokeny_wplyw(1.0)
        w_przod(1.0, pokaz_chmure=True, etap=0)
        rp = ramka_pred(0)
        lot_luk(rp.get_center(), 1.0, extra=[*wygas(), linie.animate.set_stroke(opacity=0.22)])
        self.play(GrowFromCenter(rp), FadeIn(lab_pr, shift=0.08 * UP), run_time=0.6, rate_func=ease_out)
        self.play(Create(ramka_gt), FadeIn(lab_gt, shift=0.08 * UP), run_time=0.6, rate_func=ease_out)
        self.add(tlo_str, bar)
        self.play(FadeIn(lab_str), FadeIn(tlo_str), run_time=0.3)
        self.play(dl.animate.set_value(DL_STR[0]), run_time=0.8, rate_func=ease_out)
        print("ZAMROZENIE 1 (ramka i strata)", round(self.time, 2))
        self.wait(1.4)

        wstecz(1.0, 0)
        self.play(*wygas(), run_time=0.25)
        print("ZAMROZENIE 2 (zmienione wagi)", round(self.time, 2))
        self.wait(1.4)
        self.play(strz.animate.set_opacity(0), run_time=0.35, rate_func=smooth)

        przesun_latent(0, 1, 1.0)
        print("ZAMROZENIE 3 (latent przesuniety)", round(self.time, 2))
        self.wait(1.3)

        # ================= KROKI 2-4: coraz szybciej =================
        tempa = [0.4, 0.28, 0.2]
        for k, tm in zip([1, 2, 3], tempa):
            nowy = licznik_txt(k + 1)
            tokeny_wplyw(max(tm, 0.3), extra=[FadeTransform(licznik, nowy)])
            licznik = nowy
            w_przod(tm)
            rn = ramka_pred(k)
            lot_luk(rn.get_center(), max(tm, 0.35), extra=wygas())
            self.play(Transform(rp, rn), dl.animate.set_value(DL_STR[k]),
                      run_time=0.8 * max(tm, 0.4), rate_func=smooth)
            wstecz(tm, k)
            przesun_latent(k, k + 1, max(tm, 0.35), extra=[strz.animate.set_opacity(0), *wygas()])

        print("ZAMROZENIE 4 (koniec)", round(self.time, 2))
        self.wait(2.0)

        bar.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6, rate_func=smooth)
        print("koniec", round(self.time, 2))
