# -*- coding: utf-8 -*-
"""Animacja „Trzy reżimy ruchu cząsteczki: MSD(τ)” (1920x1080, 30 kl./s, H.264).

Czyta wyłącznie gotowe tablice z ../../dane/trajektorie_wlasnosci.npz
(taus, msd_stac, alfa_lok, lags, vacf) oraz ../../dane/PODSUMOWANIE.json.
Niczego nie liczy z surowych trajektorii.

Uruchomienie (Python 3.13):
    F:\\Python\\3.13\\python.exe script.py                 # pełny render + klatki kontrolne
    F:\\Python\\3.13\\python.exe script.py --podglad 16,30 # tylko PNG wybranych chwil [s]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
DANE = HERE.parents[1] / "dane"

# ----------------------------------------------------------------------------- spec
W, H, DPI, FPS = 1920, 1080, 72, 30          # DPI=72 -> 1 pt == 1 px
DURATION = 30.5                               # s

BG, TXT, ACC = "#10171c", "#e8e8e8", "#ffd166"
REGIONS = ["tło", "rogówka", "źrenica", "narzędzie"]
COLORS = ["#8a8f99", "#4cd964", "#4a8cff", "#ff4433"]

# ----------------------------------------------------------------------------- czcionki
FONT_DIR = Path("C:/Windows/Fonts")
F_REG, F_SEMI = FONT_DIR / "segoeui.ttf", FONT_DIR / "seguisb.ttf"
for _f in (F_REG, F_SEMI):
    if not _f.exists():
        sys.exit(f"Brak czcionki {_f}")
    fm.fontManager.addfont(str(_f))
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = True
_found = Path(fm.findfont(fm.FontProperties(family="Segoe UI"), fallback_to_default=False))
if _found.name.lower() != "segoeui.ttf":
    sys.exit(f"Segoe UI nie wczytała się (findfont -> {_found})")


def fp(size: float, semi: bool = False) -> fm.FontProperties:
    return fm.FontProperties(fname=str(F_SEMI if semi else F_REG), size=size)


# ----------------------------------------------------------------------------- pomocnicze
def clamp01(x: float) -> float:
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x


def ease_out(t0: float, t1: float, t: float) -> float:
    u = clamp01((t - t0) / (t1 - t0))
    return 1.0 - (1.0 - u) ** 3


def ease_io(t0: float, t1: float, t: float) -> float:
    u = clamp01((t - t0) / (t1 - t0))
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def pl(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def kroki(n: int) -> str:
    if n == 1:
        return "krok"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "kroki"
    return "kroków"


def fx(px: float) -> float:            # piksel od lewej -> współrzędna figury
    return px / W


def fy(px: float) -> float:            # piksel od góry -> współrzędna figury
    return 1.0 - px / H


# ----------------------------------------------------------------------------- dane
def wczytaj(dane: Path):
    z = np.load(dane / "trajektorie_wlasnosci.npz")
    taus = z["taus"].astype(float)
    msd = z["msd_stac"].astype(float)
    alfa = z["alfa_lok"].astype(float)
    lags = z["lags"].astype(float)
    vacf = z["vacf"].astype(float)
    pods = json.loads((dane / "PODSUMOWANIE.json").read_text(encoding="utf-8"))
    return taus, msd, alfa, lags, vacf, pods


class Krzywe:
    """Interpolacja log-log MSD(τ) i α(τ) z gotowych punktów (NaN pomijane)."""

    def __init__(self, taus, msd, alfa):
        self.lt, self.msd, self.alfa = np.log(taus), msd, alfa
        self.vm = [np.isfinite(msd[:, k]) & (msd[:, k] > 0) for k in range(4)]
        self.va = [np.isfinite(alfa[:, k]) for k in range(4)]
        ok = np.any(np.stack(self.vm, 1), 1)
        self.tmin, self.tmax = float(taus[ok].min()), float(taus[ok].max())

    def msd_at(self, tau, k):
        v = self.vm[k]
        return np.exp(np.interp(np.log(tau), self.lt[v], np.log(self.msd[v, k])))

    def alfa_at(self, tau, k):          # poza zakresem α: najbliższa zdefiniowana wartość
        v = self.va[k]
        return np.interp(np.log(tau), self.lt[v], self.alfa[v, k])


# ----------------------------------------------------------------------------- oś czasu [s]
T = dict(
    sub=(0.2, 1.0), ctx=(0.5, 1.3),            # zdanie o MSD + kontekst (ease-out)
    head=(2.8, 3.6),                           # tytuł jedzie do nagłówka (ease-in-out)
    axes=(3.3, 4.1), panel=(3.6, 4.4),         # osie i licznik α (ease-out)
    draw=(4.3, 12.3),                          # rysowanie krzywych (ease-in-out)
    reg=(14.0, 17.0, 20.2),                    # starty trzech reżimów
    vacf_in=(23.6, 24.2), vacf_draw=(24.0, 25.4), vacf_ann=(25.4, 26.0),
    wniosek1=(27.6, 28.3), wniosek2=(27.8, 28.5),
)
GLIDE = 1.1                                    # przejazd markera do reżimu


# ----------------------------------------------------------------------------- scena
class Scena:
    def __init__(self, dane: Path):
        taus, msd, alfa, lags, vacf, pods = wczytaj(dane)
        self.K = Krzywe(taus, msd, alfa)
        self.taus = taus
        self.pods = pods
        K = self.K

        self.fig = fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)

        # ---------------- nagłówek
        self.t_title = fig.text(0.5, 0, "Trzy reżimy ruchu cząsteczki: MSD(τ)",
                                fontproperties=fp(56, True), color=TXT, ha="center", va="top", zorder=1001)
        self.t_sub = fig.text(0.5, 0, "MSD(τ) to średni kwadrat przesunięcia tokenu "
                              "po τ krokach uczenia; tokeny ViT są „cząsteczkami” w przestrzeni latentu.",
                              fontproperties=fp(28), color=TXT, ha="center", va="top", zorder=1001)
        self.t_ctx = fig.text(0.5, 0, "zapis co 1 krok: 100 zdjęć × 600 tokenów, uczenie od zera na 35 000 klatkach"
                              "  ·  rzut PCA32 (98,6 % wariancji latentu)  ·  średnia po oknach startowych t ≥ 300",
                              fontproperties=fp(20), color=TXT, ha="center", va="top", zorder=1001)

        # ---------------- główny wykres MSD (log-log)
        L, R, TOP, BOT = 128, 1180, 196, 112
        self.ax = ax = fig.add_axes([fx(L), BOT / H, fx(R - L), (H - TOP - BOT) / H])
        ax.set_facecolor(BG)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.75, 2400); ax.set_ylim(0.015, 30)
        self._styl_osi(ax, 18)
        xt = [1, 3, 10, 30, 100, 300, 1000]
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in xt]))
        yt = [0.03, 0.1, 0.3, 1, 3, 10, 30]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_major_formatter(FixedFormatter([pl(v, 2).rstrip("0").rstrip(",") for v in yt]))
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=TXT, alpha=0.15, lw=0.9)
        ax.set_xlabel("opóźnienie τ  [kroki uczenia]", fontproperties=fp(20), color=TXT, labelpad=8)
        ax.set_ylabel("MSD(τ)  [jedn. PCA32²]", fontproperties=fp(20), color=TXT, labelpad=10)

        # gęsta siatka τ do płynnego rysowania (interpolacja liniowa w log-log między punktami danych)
        self.TD = np.geomspace(K.tmin, K.tmax, 900)
        self.MD = [K.msd_at(self.TD, k) for k in range(4)]

        # ---------------- reżimy: pasma, linie nachylenia, podpisy
        def gmean_msd(tc):
            return float(np.exp(np.mean([np.log(K.msd_at(tc, k)) for k in range(4)])))

        self.REG = [
            dict(lo=1, hi=3, slope=1.8, tc=math.sqrt(3), span=(0.8, 3.9), factor=2.6, tau=2,
                 title="balistyczny: α ≈ 1,8", sub="τ = 1–3 kroków: ruch z rozpędem",
                 xy=(0.80, 13.0), ha="left", edges=[(0.0, 0.83)]),
            dict(lo=30, hi=100, slope=0.1, tc=math.sqrt(3000), span=(19, 160), factor=0.5, tau=80,
                 title="klatka: α ≈ 0,1", sub="τ = 30–100 kroków: cząsteczka drga w miejscu",
                 xy=(math.sqrt(3000), 0.40), ha="center", edges=[(0.0, 0.37), (0.50, 0.80)]),
            dict(lo=700, hi=1400, slope=1.0, tc=990.0, span=(430, 2200), factor=0.5, tau=988,
                 title="ucieczka dyfuzyjna: α ≈ 1", sub="τ ≈ 1000 kroków: cząsteczka opuszcza klatkę",
                 xy=(640, 13.0), ha="right", edges=[(0.0, 0.83)]),
        ]
        for r in self.REG:
            r["band"] = ax.axvspan(r["lo"], r["hi"], color=ACC, alpha=0, lw=0, zorder=1)
            r["edges"] = [ax.axvline(v, ymin=a, ymax=b, color=ACC, lw=1.2, ls=(0, (4, 4)), alpha=0, zorder=1)
                          for v in (r["lo"], r["hi"]) for a, b in r["edges"]]
            gt = np.geomspace(*r["span"], 200)
            r["gt"], r["gy"] = gt, gmean_msd(r["tc"]) * r["factor"] * (gt / r["tc"]) ** r["slope"]
            r["guide"], = ax.plot([], [], color=ACC, lw=2.6, ls=(0, (7, 5)), zorder=4,
                                  dash_capstyle="round", alpha=0)
            r["a_title"] = ax.annotate(r["title"], xy=r["xy"], xytext=(0, 0), textcoords="offset points",
                                       fontproperties=fp(26, True), color=ACC, ha=r["ha"], va="bottom",
                                       annotation_clip=False, alpha=0, zorder=9)
            r["a_sub"] = ax.annotate(r["sub"], xy=r["xy"], xytext=(0, -6), textcoords="offset points",
                                     fontproperties=fp(20), color=TXT, ha=r["ha"], va="top",
                                     annotation_clip=False, alpha=0, zorder=9)

        # ---------------- krzywe i markery
        self.lines, self.dots, self.heads, self.halos = [], [], [], []
        for k in range(4):
            self.halos.append(ax.plot([], [], "o", ms=30, color=COLORS[k], alpha=0, mec="none", zorder=6)[0])
            self.lines.append(ax.plot([], [], color=COLORS[k], lw=3.4, solid_capstyle="round", zorder=5)[0])
            self.dots.append(ax.plot([], [], "o", ms=5, color=COLORS[k], mec="none", zorder=5)[0])
            self.heads.append(ax.plot([], [], "o", ms=14, mfc=COLORS[k], mec=BG, mew=2.5, zorder=7)[0])

        # ---------------- licznik α(τ) (pełni rolę legendy)
        X0 = 1262
        self.panel = []
        self.panel.append(fig.text(fx(X0), fy(206), "lokalny wykładnik  α(τ) = d log MSD / d log τ",
                                   fontproperties=fp(20), color=TXT, va="top", alpha=0))
        self.t_tau = fig.text(fx(X0), fy(240), "", fontproperties=fp(36, True), color=TXT, va="top", alpha=0)
        self.panel.append(self.t_tau)
        for txt, x, ha in (("obszar", X0 + 36, "left"), ("MSD", 1650, "right"), ("α(τ)", 1830, "right")):
            self.panel.append(fig.text(fx(x), fy(306), txt, fontproperties=fp(18), color=TXT,
                                       ha=ha, va="top", alpha=0))
        self.t_msd, self.t_alfa, self.p_dots = [], [], []
        for k in range(4):
            yy = 342 + 44 * k
            d = Line2D([fx(X0 + 12)], [fy(yy + 15)], transform=fig.transFigure, marker="o", ms=14,
                       mfc=COLORS[k], mec="none", ls="none", alpha=0)
            fig.add_artist(d); self.p_dots.append(d)
            self.panel.append(fig.text(fx(X0 + 36), fy(yy), REGIONS[k], fontproperties=fp(24, True),
                                       color=COLORS[k], va="top", alpha=0))
            self.t_msd.append(fig.text(fx(1650), fy(yy), "", fontproperties=fp(24), color=TXT,
                                       ha="right", va="top", alpha=0))
            self.t_alfa.append(fig.text(fx(1830), fy(yy), "", fontproperties=fp(24, True), color=TXT,
                                        ha="right", va="top", alpha=0))
        self.panel += self.t_msd + self.t_alfa

        # ---------------- panel VACF
        self.v_texts = [
            fig.text(fx(X0), fy(536), "Autokorelacja prędkości VACF (opóźnienie w krokach)",
                     fontproperties=fp(21, True), color=TXT, va="top", alpha=0),
            fig.text(fx(X0), fy(566), "po rozgrzewce, bez pola kolektywnego",
                     fontproperties=fp(18), color=TXT, va="top", alpha=0),
        ]
        self.ax2 = ax2 = fig.add_axes([fx(1336), fy(870), fx(1850 - 1336), (870 - 604) / H])
        ax2.set_facecolor(BG)
        ax2.set_xscale("log"); ax2.set_xlim(0.8, 1000); ax2.set_ylim(-0.42, 0.95)
        self._styl_osi(ax2, 17)
        ax2.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
        ax2.xaxis.set_major_formatter(FixedFormatter(["1", "10", "100"]))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        yv = [-0.2, 0, 0.4, 0.8]
        ax2.yaxis.set_major_locator(FixedLocator(yv))
        ax2.yaxis.set_major_formatter(FixedFormatter([pl(v, 1) if v else "0" for v in yv]))
        ax2.grid(True, which="major", color=TXT, alpha=0.15, lw=0.9)
        ax2.axhline(0, color=TXT, alpha=0.3, lw=1.2, zorder=1)
        ax2.set_xlabel("opóźnienie [kroki uczenia]", fontproperties=fp(18), color=TXT, labelpad=4)
        self.LG = np.geomspace(lags.min(), lags.max(), 500)
        self.VD = [np.interp(np.log(self.LG), np.log(lags), vacf[:, k]) for k in range(4)]
        self.lags, self.vacf = lags, vacf
        self.vlines = [ax2.plot([], [], color=COLORS[k], lw=2.6, zorder=4)[0] for k in range(4)]
        v1 = float(np.mean(list(pods["vacf_lag1_po_rozgrzewce"].values())))
        v10 = float(np.mean(list(pods["vacf_lag10_po_rozgrzewce"].values())))
        self.v_ann = []
        for lag, v, txt, off, ha in ((1, v1, f"+{pl(v1)} po 1 kroku", (20, 0), "left"),
                                     (10, v10, f"≈ {pl(v10, 1)} po 10 krokach", (26, 44), "left")):
            ring = ax2.plot([lag], [v], "o", ms=20, mfc="none", mec=ACC, mew=2.2, alpha=0, zorder=6)[0]
            a = ax2.annotate(txt, xy=(lag, v), xytext=off, textcoords="offset points",
                             fontproperties=fp(20, True), color=ACC, ha=ha, va="center" if off[1] == 0 else "bottom",
                             annotation_clip=False, alpha=0, zorder=7,
                             arrowprops=None if off[1] == 0 else dict(arrowstyle="-", color=ACC, lw=1.4,
                                                                       shrinkA=2, shrinkB=11, alpha=0))
            self.v_ann.append((ring, a))

        # ---------------- wniosek
        self.t_w1 = fig.text(fx(X0), fy(958), "To dynamika szklista:", fontproperties=fp(26, True),
                             color=ACC, va="top", alpha=0)
        self.t_w2 = fig.text(fx(X0), fy(996), "szybkie drgania w klatce, rzadkie przebudowy klatki.",
                             fontproperties=fp(22), color=TXT, va="top", alpha=0)

        # ---------------- zasłony do wygaszania całych obszarów (fade-in osi)
        self.veil_main = Rectangle((0, 0), fx(1222), fy(170), transform=fig.transFigure,
                                   facecolor=BG, edgecolor="none", zorder=1000)
        self.veil_vacf = Rectangle((fx(1240), 0), 1 - fx(1240), fy(590), transform=fig.transFigure,
                                   facecolor=BG, edgecolor="none", zorder=1000)
        fig.add_artist(self.veil_main); fig.add_artist(self.veil_vacf)

    # -------------------------------------------------------------------------
    @staticmethod
    def _styl_osi(ax, ls):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(TXT); ax.spines[s].set_alpha(0.3); ax.spines[s].set_linewidth(1.2)
        rgba = matplotlib.colors.to_rgba(TXT, 0.3)
        ax.tick_params(which="both", colors=rgba, labelcolor=TXT, labelsize=ls)
        ax.tick_params(which="major", length=6, width=1.1)
        ax.tick_params(which="minor", length=3, width=0.8)

    # ------------------------------------------------------------------------- stan w chwili t
    def tau_marker(self, t: float) -> float:
        K = self.K
        lo, hi = math.log(K.tmin), math.log(K.tmax)
        tau = math.exp(lerp(lo, hi, ease_io(*T["draw"], t)))
        prev = K.tmax
        for s, r in zip(T["reg"], self.REG):
            if t >= s:
                u = ease_io(s, s + GLIDE, t)
                tau = math.exp(lerp(math.log(prev), math.log(r["tau"]), u))
            prev = r["tau"]
        return tau

    def update(self, t: float):
        K = self.K

        # nagłówek: start na środku ekranu, potem wjazd do nagłówka
        g = ease_io(*T["head"], t)
        self.t_title.set_position((0.5, fy(lerp(408, 28, g)))); self.t_title.set_fontsize(lerp(56, 34, g))
        self.t_sub.set_position((0.5, fy(lerp(500, 78, g)))); self.t_sub.set_fontsize(lerp(28, 22, g))
        self.t_ctx.set_position((0.5, fy(lerp(552, 114, g)))); self.t_ctx.set_fontsize(lerp(20, 18, g))
        self.t_sub.set_alpha(ease_out(*T["sub"], t))
        self.t_ctx.set_alpha(0.72 * ease_out(*T["ctx"], t))

        # osie
        self.veil_main.set_alpha(1 - ease_out(*T["axes"], t))
        self.veil_main.set_visible(self.veil_main.get_alpha() > 0.002)

        # krzywe: rysowanie od τ=1 w prawo
        head = math.exp(lerp(math.log(K.tmin), math.log(K.tmax), ease_io(*T["draw"], t)))
        idx = self.TD <= head
        sel_t = self.taus <= head * (1 + 1e-9)
        for k in range(4):
            xs = np.append(self.TD[idx], head)
            ys = np.append(self.MD[k][idx], K.msd_at(head, k))
            self.lines[k].set_data(xs, ys)
            v = K.vm[k] & sel_t
            self.dots[k].set_data(self.taus[v], K.msd[v, k])

        # markery + licznik
        pa = ease_out(*T["panel"], t)
        tm = self.tau_marker(t)
        for k in range(4):
            y = float(K.msd_at(tm, k))
            self.heads[k].set_data([tm], [y]); self.heads[k].set_alpha(pa)
            self.halos[k].set_data([tm], [y]); self.halos[k].set_alpha(0.16 * pa)
            self.t_msd[k].set_text(pl(y, 3 if y < 1 else 2 if y < 10 else 1))
            self.t_alfa[k].set_text(pl(float(K.alfa_at(tm, k)), 2))
            self.p_dots[k].set_alpha(pa)
        n = max(1, int(round(tm)))
        self.t_tau.set_text(f"τ = {n} {kroki(n)}")
        for a in self.panel:
            a.set_alpha(pa * (0.72 if a in self.panel[:1] or a in self.panel[2:5] else 1.0))

        # reżimy
        e4 = ease_out(0, 0.6, t - T["vacf_in"][0])
        for i, (s, r) in enumerate(zip(T["reg"], self.REG)):
            nxt = T["reg"][i + 1] if i < 2 else None
            dim = ease_out(nxt, nxt + 0.5, t) if nxt else 0.0
            band = 0.14 * ease_out(s + 0.2, s + 0.8, t)
            band = lerp(band, 0.05, dim); band = lerp(band, 0.06, e4)
            lab = ease_out(s + 0.8, s + 1.4, t)
            lab = lab * (1 - 0.45 * dim)
            lab = lerp(lab, 0.92, e4) if t >= s + 1.4 else lab
            r["band"].set_alpha(band)
            for e in r["edges"]:
                e.set_alpha(min(1.0, band * 3.2))
            gp = ease_io(s + 0.6, s + 1.2, t)
            m = int(round(gp * (len(r["gt"]) - 1))) + 1
            r["guide"].set_data(r["gt"][:m] if gp > 0 else [], r["gy"][:m] if gp > 0 else [])
            r["guide"].set_alpha(min(1.0, 0.95 * (1 - 0.4 * dim) if gp > 0 else 0))
            slide = 12 * (1 - ease_out(s + 0.8, s + 1.4, t))
            r["a_title"].set_alpha(lab); r["a_sub"].set_alpha(lab * 0.95)
            r["a_title"].xyann = (0, -slide); r["a_sub"].xyann = (0, -6 - slide)

        # VACF
        va = ease_out(*T["vacf_in"], t)
        self.veil_vacf.set_alpha(1 - va); self.veil_vacf.set_visible(va < 0.998)
        for tx in self.v_texts:
            tx.set_alpha(va * (1.0 if tx is self.v_texts[0] else 0.72))
        vp = ease_io(*T["vacf_draw"], t)
        lhead = math.exp(lerp(math.log(self.LG[0]), math.log(self.LG[-1]), vp))
        li = self.LG <= lhead
        for k in range(4):
            if vp <= 0:
                self.vlines[k].set_data([], [])
            else:
                self.vlines[k].set_data(np.append(self.LG[li], lhead),
                                        np.append(self.VD[k][li], np.interp(math.log(lhead), np.log(self.LG), self.VD[k])))
        aa = ease_out(*T["vacf_ann"], t)
        for ring, a in self.v_ann:
            ring.set_alpha(aa); a.set_alpha(aa)
            if a.arrow_patch is not None:
                a.arrow_patch.set_alpha(0.8 * aa)

        # wniosek
        self.t_w1.set_alpha(ease_out(*T["wniosek1"], t))
        self.t_w2.set_alpha(ease_out(*T["wniosek2"], t))

    # ------------------------------------------------------------------------- render
    def frame(self, t: float) -> np.ndarray:
        self.update(t)
        self.fig.canvas.draw()
        a = np.asarray(self.fig.canvas.buffer_rgba())
        assert a.shape == (H, W, 4), a.shape
        return a

    # ------------------------------------------------------------------------- audyt układu
    def audyt(self, t: float) -> list[str]:
        """Zwraca listę problemów: nachodzenie tekstów na siebie, na krzywe i wyjście poza kadr."""
        self.frame(t)
        rr = self.fig.canvas.get_renderer()
        texts = []
        for tx in list(self.fig.texts) + list(self.ax.texts) + list(self.ax2.texts):
            if tx.get_visible() and (tx.get_alpha() is None or tx.get_alpha() > 0.05) and tx.get_text():
                texts.append(tx)
        for ax in (self.ax, self.ax2):
            texts += [ax.xaxis.label, ax.yaxis.label]
        bbs = [(tx.get_text()[:38], matplotlib.text.Text.get_window_extent(tx, rr)) for tx in texts]
        out = []
        for n, b in bbs:
            if b.x0 < 4 or b.y0 < 4 or b.x1 > W - 4 or b.y1 > H - 4:
                out.append(f"poza kadrem: '{n}' {b}")
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                if bbs[i][1].overlaps(bbs[j][1]):
                    out.append(f"tekst-tekst: '{bbs[i][0]}' x '{bbs[j][0]}'")
        lines = [ln for ax in (self.ax, self.ax2) for ln in ax.lines
                 if ln.get_visible() and (ln.get_alpha() is None or ln.get_alpha() > 0.05)
                 and len(ln.get_xdata()) > 0]
        for n, b in bbs:
            bb = b.expanded(1.0, 1.0)
            for ln in lines:
                xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float))
                if ln.get_linestyle() not in ("None", "", " ") and len(xy) > 1:   # zagęść odcinki co ~2 px
                    seg = [np.linspace(xy[i], xy[i + 1], max(2, int(np.hypot(*(xy[i + 1] - xy[i])) / 2)))
                           for i in range(len(xy) - 1)]
                    xy = np.vstack(seg)
                ins = ((xy[:, 0] > bb.x0 - 4) & (xy[:, 0] < bb.x1 + 4) & (xy[:, 1] > bb.y0 - 4) & (xy[:, 1] < bb.y1 + 4))
                if ins.any():
                    out.append(f"tekst-krzywa: '{n}' x linia {ln.get_color()} ({int(ins.sum())} pkt)")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dane", type=Path, default=DANE)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--podglad", type=str, default="", help="lista chwil [s], np. 16,30 (tylko PNG)")
    ap.add_argument("--audyt", type=str, default="", help="lista chwil [s] do audytu układu")
    args = ap.parse_args()

    sc = Scena(args.dane)
    N = int(round(DURATION * FPS))

    if args.audyt:
        for t in [float(s) for s in args.audyt.split(",")]:
            pr = sc.audyt(t)
            print(f"t={t:5.2f}s: " + ("OK" if not pr else f"{len(pr)} problem(ów)"))
            for p in pr:
                print("   ", p)
        return
    if args.podglad:
        for t in [float(s) for s in args.podglad.split(",")]:
            plt.imsave(args.out / f"podglad_{t:05.2f}.png", sc.frame(t)[..., :3])
        return

    args.out.mkdir(parents=True, exist_ok=True)
    mp4 = args.out / "render.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(mp4)]
    kontrolne = {0: 0, int(round(0.33 * (N - 1))): 1, int(round(0.66 * (N - 1))): 2, N - 1: 3}
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(N):
            a = sc.frame(i / FPS)
            proc.stdin.write(a.tobytes())
            if i in kontrolne:
                plt.imsave(args.out / f"verify_{kontrolne[i]:02d}.png", a[..., :3])
            if i % 60 == 0:
                print(f"klatka {i}/{N}", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        sys.exit(f"ffmpeg zakończył się kodem {rc}")
    print(f"zapisano {mp4} ({N} klatek, {N / FPS:.2f} s)")


if __name__ == "__main__":
    main()
