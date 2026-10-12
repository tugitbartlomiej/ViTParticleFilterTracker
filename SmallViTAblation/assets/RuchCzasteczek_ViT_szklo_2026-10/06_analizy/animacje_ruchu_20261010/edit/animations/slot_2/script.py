# -*- coding: utf-8 -*-
"""Zegar uczenia: co ruch cząsteczek mówi o treningu ViT.

Animacja MP4 (1920x1080, 30 kl./s, H.264 CRF 18, yuv420p, bez dźwięku) zbudowana
wyłącznie z policzonych wcześniej serii:
  - edit/dane/trajektorie_wlasnosci.npz  (energia, pole, temperatury, cos, log paczek)
  - edit/dane/PODSUMOWANIE.json           (liczby do napisów)
  - assets/MalyViT_35000_od_zera_s47/dane/OCENY.json  (mAP@0,5 walidacji, inny bieg)

Uruchomienie (Python 3.13 z matplotlib 3.10, ffmpeg w PATH):
  F:\\Python\\3.13\\python.exe script.py               -> render.mp4 + verify_00..03.png
  F:\\Python\\3.13\\python.exe script.py --preview 5 13  -> podgląd PNG dla podanych sekund
"""
import json
import math
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory, offset_copy  # noqa: E402

# ------------------------------------------------------------------ ścieżki
HERE = Path(__file__).resolve().parent
EDIT = HERE.parents[1]
DANE = EDIT / "dane"
SMALLVIT = HERE.parents[4]
NPZ = DANE / "trajektorie_wlasnosci.npz"
PODS = DANE / "PODSUMOWANIE.json"
OCENY = SMALLVIT / "assets" / "MalyViT_35000_od_zera_s47" / "dane" / "OCENY.json"
OUT = HERE / "render.mp4"

# ------------------------------------------------------------------ spec
FPS = 30
W_PX, H_PX = 1920, 1080
DPI = 100

BG = "#10171c"
PANEL = "#131d23"
SPINE = "#2b3842"
TICK = "#7f8c97"
GRID = "#1a262e"
TXT = "#e8e8e8"
TXT2 = "#aab4bd"
DIM = "#6f7b85"
YEL = "#ffd166"
WHITE = "#e8e8e8"
REG_COL = ["#8a8f99", "#4cd964", "#4a8cff", "#ff4433"]
REG_NAMES = ["tło", "rogówka", "źrenica", "narzędzie"]
PH_COL = ["#ffd166", "#e8e8e8", "#ff9f43", "#7cc8ff"]

# ------------------------------------------------------------------ czcionki
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


def _font(name):
    p = FONT_DIR / name
    return str(p) if p.exists() else None


F_REG, F_BOLD, F_SEMI = _font("segoeui.ttf"), _font("segoeuib.ttf"), _font("seguisb.ttf")
if F_REG:
    fm.fontManager.addfont(F_REG)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
else:  # fallback
    print("UWAGA: brak Segoe UI, używam DejaVu Sans", file=sys.stderr)
    plt.rcParams["font.family"] = ["DejaVu Sans"]


def pt(px):
    """Rozmiar w pikselach -> punkty (dpi 100)."""
    return px * 72.0 / DPI


def fprop(kind, px):
    path = {"bold": F_BOLD, "semi": F_SEMI or F_BOLD}.get(kind)
    if path:
        return fm.FontProperties(fname=path, size=pt(px))
    return fm.FontProperties(weight="bold", size=pt(px))


def rgba(hexcol, a):
    r, g, b = matplotlib.colors.to_rgb(hexcol)
    return (r, g, b, float(np.clip(a, 0, 1)))


def fy(px):
    return 1.0 - px / H_PX


def fx(px):
    return px / W_PX


def pl(v, nd=2):
    """Liczba z polskim przecinkiem."""
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


# ------------------------------------------------------------------ easing
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t, t0, t1):
    return ease((t - t0) / (t1 - t0))


def keyed(t, keys):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            if t1 <= t0:
                return v1
            return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0))
    return keys[-1][1]


def lerp(a, b, u):
    return a + (b - a) * u


def glerp(a, b, u):
    return a * (b / a) ** u


# ------------------------------------------------------------------ dane
def rmed(y, w):
    """Mediana ruchoma (centrowana) odporna na NaN."""
    y = np.asarray(y, dtype=float)
    lo, hi = w // 2, w - 1 - w // 2
    yp = np.pad(y, [(lo, hi)] + [(0, 0)] * (y.ndim - 1), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(yp, w, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


d = np.load(NPZ)
pods = json.loads(PODS.read_text(encoding="utf-8"))
oceny = json.loads(OCENY.read_text(encoding="utf-8"))

kroki = d["kroki"].astype(float)
DENSE_END = int(pods.get("geste_do", 2188))
MD = kroki <= DENSE_END
MS = ~MD
xd, xs = kroki[MD], kroki[MS]

share = d["pole"] / d["energia"]
temp = d["temp_ob"]
cosv = d["cos_ob"]

share_d_raw, share_s_raw = share[MD], share[MS]
share_d, share_s = rmed(share[MD], 15), rmed(share[MS], 9)
temp_d, temp_s = rmed(temp[MD], 15), rmed(temp[MS], 9)
cos_d, cos_s = rmed(cosv[MD], 15), rmed(cosv[MS], 9)

steps = d["krok_paczki"].astype(float)
lr = d["lr"].astype(float)
LR_MAX = float(np.nanmax(lr))
lr_n = lr / LR_MAX
loss_raw = d["strata_po"].astype(float)
loss_sm = rmed(loss_raw, 50)

ep = [(e["krok"], e["epoka"], e["mAP50"]) for e in oceny if e.get("epoka") is not None]
ep = [e for e in ep if e[0] <= 23000]
ep_x = np.array([e[0] for e in ep], float)
ep_y = np.array([e[2] for e in ep], float)

# liczby do napisów (z danych, nie wpisane z ręki)
batch_ratio = pods["efekt_paczki"]["stosunek"]
pk = pods["pole_kolektywne_udzial"]
share_at = dict(zip(pk["kroki"], pk["udzial"]))
warm_shares = [v for s, v in share_at.items() if s <= 100]
tp = pods["temperatura_pca32"]
i1000 = tp["kroki"].index(1000)
tool_bg_1000 = tp["narzędzie"][i1000] / tp["tło"][i1000]


def win_ratio(a, b, j):
    m = (kroki >= a) & (kroki <= b)
    return float(np.median(temp[m, j] / temp[m, 0]))


tool_bg_late = win_ratio(10000, 22970, 3)
pup_ratios = [win_ratio(a, b, 2) for a, b in [(150, 300), (300, 600), (600, 1000), (1000, 2188),
                                                (2189, 10000), (10000, 22970)]]
cos_sparse = float(np.nanmedian(cosv[(kroki > DENSE_END) & (kroki <= 10000)]))
cos_warm = float(np.nanmedian(cosv[(kroki >= 5) & (kroki <= 60)]))
t_peak = [int(xd[np.argmax(temp_d[:, j])]) for j in range(4)]
cool_fac = float(np.median(temp_d[xd == 600][0] / np.median(temp_d[xd >= DENSE_END - 100], axis=0)))
t_peak_lo, t_peak_hi = int(round(min(t_peak), -1)), int(round(max(t_peak), -1))
i_ep8 = [e[1] for e in ep].index(8)

print(f"[dane] wpisy={len(kroki)} geste_do={DENSE_END} epok_mAP={len(ep)}")
print(f"[dane] udział pola: rozgrzewka {min(warm_shares):.2f}-{max(warm_shares):.2f}, "
      f"300={share_at[300]:.2f}, 1000={share_at[1000]:.2f}, 2000={share_at[2000]:.2f}")
print(f"[dane] narzędzie/tło @1000={tool_bg_1000:.2f}, późno={tool_bg_late:.2f}; "
      f"źrenica/tło {min(pup_ratios):.2f}-{max(pup_ratios):.2f}; cos co 10 kr.={cos_sparse:.2f}")
print(f"[dane] szczyt temp. kroki {t_peak}; chłodzenie 600->{DENSE_END}: x{cool_fac:.1f}; "
      f"paczka={batch_ratio:.3f}")

# ------------------------------------------------------------------ teksty
TITLE = "Zegar uczenia: co ruch cząsteczek mówi o treningu ViT"
SUBTITLE = ("60 000 cząsteczek (100 zdjęć × 600 tokenów, zapis co krok) · log paczek uczenia · "
            "bieg od zera na 35 000 klatkach")
INTRO = ("Wszystko poniżej policzono wyłącznie z ruchu 60 000 cząsteczek\n"
         "(100 zdjęć × 600 tokenów) i z logu paczek uczenia.")
INTRO2 = "Fazy uczenia czytamy z samego ruchu; walidacja (mAP) pojawia się dopiero na końcu jako tło."

PHASES = [  # (od, do, nazwa w pasku, nagłówek, opis)
    (0, 150, "rozgrzewka", "1 · Rozgrzewka (kroki 0–150)",
     f"lr rośnie liniowo od zera. {min(warm_shares)*100:.0f}–{max(warm_shares)*100:.0f} % energii "
     f"kroku to pole wspólne: cząsteczki płyną razem, cos kolejnych przyrostów ≈ {pl(cos_warm, 1)}."),
    (150, 300, "przejście", "2 · Przejście (150–300)",
     f"Udział pola spada do {share_at[300]*100:.0f} % przy kroku 300. Obszary rozchodzą się, "
     f"temperatura szczytuje przy kroku ~{t_peak_lo}–{t_peak_hi}."),
    (300, 600, "drgania w klatce, nagrzanie", "3 · Drgania w klatce (300–600)",
     "lr stałe 5 × 10⁻⁴. Ruch własny przeważa nad polem; cząsteczki są nagrzane i drgają "
     "wokół swoich miejsc."),
    (600, 23000, "chłodzenie", "4 · Chłodzenie (od 600)",
     f"Temperatura spada ok. {cool_fac:.0f} × do kroku {DENSE_END}, pole to już tylko "
     f"ok. {share_at[1000]*100:.0f}–{share_at[2000]*100:.0f} % energii."),
]

NOTE1_H = f"Od kroku {DENSE_END}: zapis co 10 kroków"
NOTE1_D = ("Przyrost Δx obejmuje teraz 10 kroków, więc temperatura i cos mają inną skalę. "
           "Skok na linii przerywanej to zmiana zapisu, nie fizyka.")
NOTE2_H = "Klatka widoczna co 10 kroków"
NOTE2_D = (f"cos przyrostów co 10 kroków ≈ {pl(cos_sparse, 1)}…0: cząsteczka zawraca, "
           "zamiast płynąć dalej.")
CAPTION = ("Walidacja przestaje rosnąć od ~8. epoki, a temperatura i strata dalej "
           "powoli maleją.")

ITEMS = [
    ("Rozgrzewka = balistyczny ruch wspólny.",
     f"{min(warm_shares)*100:.0f}–{max(warm_shares)*100:.0f} % energii kroku siedzi w polu kolektywnym, "
     f"kolejne przyrosty mają cos ≈ {pl(cos_warm, 1)}, a lr rośnie przez 300 kroków."),
    ("Po ~300 krokach: drgania w klatce.",
     f"Udział pola spada do {share_at[300]*100:.0f} % przy kroku 300 i do "
     f"{share_at[1000]*100:.0f}–{share_at[2000]*100:.0f} % w krokach 1000–2000; "
     f"co 10 kroków przyrosty zawracają (cos ≈ {pl(cos_sparse, 1)})."),
    ("Źrenica i narzędzie są cieplejsze niż tło.",
     f"Narzędzie do {pl(tool_bg_1000, 1)} × (krok ~1000), pod koniec już tylko o "
     f"~{(tool_bg_late-1)*100:.0f} %; źrenica {pl(min(pup_ratios), 1)}–{pl(max(pup_ratios), 1)} × "
     "przez cały trening po rozgrzewce."),
    (f"Własne zdjęcie w paczce dodaje tylko {(batch_ratio-1)*100:.0f} % ruchu.",
     f"Energia kroku sondy w paczce / poza paczką = {pl(batch_ratio, 2)} → ruch wyznacza wspólna "
     "aktualizacja wag (pole średnie), nie własny obraz."),
]

# ------------------------------------------------------------------ oś czasu (s)
T_PANEL = [2.8, 3.3, 3.8]          # pojawianie się paneli A, B, C (0,7 s każdy)
CUR = [(4.8, 0), (6.8, 150), (8.3, 150), (9.9, 300), (11.4, 300), (13.0, 600),
       (14.5, 600), (15.9, 1000), (18.1, 1000), (19.9, DENSE_END),
       (25.6, DENSE_END), (29.0, 22970)]
T_DYMEK = (15.9, 16.5)
T_RELAYOUT = (20.4, 21.2)         # A, B, C kurczą się
T_DIN = (21.0, 21.7)              # panel D wjeżdża
T_ZOOM = (21.6, 23.8)
T_BREAK = (23.4, 24.0)
T_NOTE1 = (23.6, 24.2)
T_PLATEAU = (29.0, 29.6)
T_CARD = (31.2, 31.9)
T_ITEMS = [32.2, 33.7, 35.2, 36.7]
DUR = T_ITEMS[-1] + 0.6 + 1.9
NFR = int(round(DUR * FPS))


def cursor(t):
    return keyed(t, CUR)


_tt = np.linspace(0, DUR, 20001)
_cc = np.array([cursor(t) for t in _tt])


def t_when(v):
    """Pierwsza chwila, gdy kursor osiąga v."""
    i = int(np.argmax(_cc >= v - 1e-6))
    return float(_tt[i])


T_NOTE2 = t_when(6000.0)

# ------------------------------------------------------------------ układ (px)
LPX, RPX = 118, 1430
RCOL = 1517
L1 = {"ph": (112, 176), "A": (210, 385), "B": (440, 680), "C": (735, 905), "D": (736, 905)}
L2 = {"ph": (112, 176), "A": (210, 318), "B": (362, 530), "C": (574, 690), "D": (736, 905)}


def rect(top, bot, dy=0.0):
    return [fx(LPX), fy(bot + dy), fx(RPX - LPX), (bot - top) / H_PX]


# ------------------------------------------------------------------ figura
fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
canvas = fig.canvas

ax_a = fig.add_axes(rect(*L1["A"]))
ax_ph = fig.add_axes(rect(*L1["ph"]), sharex=ax_a)
ax_b = fig.add_axes(rect(*L1["B"]), sharex=ax_a)
ax_c = fig.add_axes(rect(*L1["C"]), sharex=ax_a)
ax_d = fig.add_axes(rect(*L1["D"]), sharex=ax_a)
ax_d2 = ax_d.twinx()
PANELS = {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}

ax_ph.axis("off")
ax_ph.set_ylim(0, 1)

fmt_steps = FuncFormatter(lambda v, _: f"{int(round(v)):,}".replace(",", " "))
for ax in (ax_a, ax_b, ax_c, ax_d, ax_d2):
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=pt(15), length=4, width=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10], integer=True))
    ax.xaxis.set_major_formatter(fmt_steps)
ax_d2.spines["right"].set_visible(True)
ax_d2.spines["left"].set_visible(False)
ax_d2.patch.set_visible(False)
for ax in (ax_a, ax_b):
    ax.tick_params(labelbottom=False)
ax_d.tick_params(labelbottom=True)

ax_a.set_ylim(0, 1.12)
ax_a.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f} %"))

ax_b.set_yscale("log")
ax_b.yaxis.set_major_locator(FixedLocator([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]))
ax_b.yaxis.set_minor_locator(NullLocator())
ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))

ax_c.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))

ax_d.set_yscale("log")
ax_d.set_ylim(0.2, 8)
ax_d.yaxis.set_major_locator(FixedLocator([0.3, 1, 3]))
ax_d.yaxis.set_minor_locator(NullLocator())
ax_d.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
ax_d2.set_ylim(0, 0.9)
ax_d2.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8]))
ax_d2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))

# zarejestrowane artysty z alfą bazową i "właścicielem" (panel, którego alfa mnoży)
ART = []


def reg(artist, owner, base=1.0, kind="alpha"):
    ART.append((artist, owner, base, kind))
    return artist


def header(ax, owner, title, handles, labels):
    tr = offset_copy(ax.transAxes, fig=fig, x=0, y=7, units="dots")
    reg(ax.text(0, 1, title, transform=tr, ha="left", va="bottom", color=TXT,
                fontproperties=fprop("semi", 18)), owner)
    if handles:
        leg = ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles),
                        frameon=False, fontsize=pt(16.5), handlelength=1.5, columnspacing=1.3,
                        handletextpad=0.5, borderaxespad=0.15, labelcolor=TXT2)
        for tx in leg.get_texts():
            reg(tx, owner)
        for h in leg.legend_handles:
            reg(h, owner)
        return leg
    return None


def hline(color, lw=2.2, **kw):
    return Line2D([], [], color=color, lw=lw, **kw)


reg_handles = [hline(c) for c in REG_COL]

# --- panel A
header(ax_a, "A", "A · learning rate i udział pola kolektywnego w energii kroku",
       [hline(YEL, 1.6), hline(WHITE, 2.2)], ["lr (100 % = 5 × 10⁻⁴)", "pole / energia (mediana 15 zapisów)"])
l_lr = reg(ax_a.plot([], [], color=YEL, lw=1.6, zorder=4)[0], "A")
l_sh_raw = reg(ax_a.plot([], [], color=WHITE, lw=0.7, zorder=3)[0], "A", 0.22)
l_sh_d = reg(ax_a.plot([], [], color=WHITE, lw=2.0, zorder=5)[0], "A")
l_sh_s = reg(ax_a.plot([], [], color=WHITE, lw=2.0, zorder=5)[0], "A")
l_sh_raw_s = reg(ax_a.plot([], [], color=WHITE, lw=0.7, zorder=3)[0], "A", 0.22)
dot_a = reg(ax_a.plot([], [], "o", ms=7, color=WHITE, mec=BG, mew=1.2, zorder=8)[0], "CUR")
dot_lr = reg(ax_a.plot([], [], "o", ms=6, color=YEL, mec=BG, mew=1.2, zorder=8)[0], "CUR")

# --- panel B
header(ax_b, "B", "B · temperatura = energia drgań własnych po odjęciu pola (PCA32, skala log)",
       reg_handles, REG_NAMES)
l_t_d = [reg(ax_b.plot([], [], color=c, lw=2.0, zorder=4 + (j == 3))[0], "B") for j, c in enumerate(REG_COL)]
l_t_s = [reg(ax_b.plot([], [], color=c, lw=1.8, zorder=4 + (j == 3))[0], "B") for j, c in enumerate(REG_COL)]
dot_b = [reg(ax_b.plot([], [], "o", ms=6.5, color=c, mec=BG, mew=1.2, zorder=8)[0], "CUR") for c in REG_COL]

# --- panel C
header(ax_c, "C", "C · cos kolejnych przyrostów (1 = ruch po prostej, < 0 = zawracanie; kolory jak w B)",
       [], [])
reg(ax_c.axhline(0, color=SPINE, lw=1.0, zorder=1), "C")
l_c_d = [reg(ax_c.plot([], [], color=c, lw=1.5, zorder=4)[0], "C") for c in REG_COL]
l_c_s = [reg(ax_c.plot([], [], color=c, lw=1.4, zorder=4)[0], "C") for c in REG_COL]

# --- panel D
header(ax_d, "D", "D · strata na paczce i mAP@0,5 walidacji po epokach",
       [hline(YEL, 2.2), Line2D([], [], color=WHITE, lw=1.2, marker="o", ms=6)],
       ["strata po aktualizacji (mediana 50 kroków)", "mAP@0,5 (inny bieg, ten sam przepis)"])
l_loss_raw = reg(ax_d.plot([], [], color=YEL, lw=0.6, zorder=2)[0], "D", 0.16)
l_loss = reg(ax_d.plot([], [], color=YEL, lw=2.0, zorder=4)[0], "D")
l_map = reg(ax_d2.plot([], [], color=WHITE, lw=1.3, marker="o", ms=6, mec=BG, mew=1.0, zorder=6)[0], "D")
dot_d = reg(ax_d.plot([], [], "o", ms=6.5, color=YEL, mec=BG, mew=1.2, zorder=8)[0], "CURD")
pl_band = reg(ax_d2.axhspan(0.62, 0.65, color=WHITE, lw=0, zorder=1), "PLAT", 0.13)
pl_vline = reg(ax_d2.axvline(ep_x[i_ep8], color=WHITE, lw=1.1, ls=(0, (4, 4)), zorder=3), "PLAT", 0.7)
pl_txt = reg(ax_d2.text(ep_x[i_ep8], 0.86, "  8. epoka", color=TXT, ha="left", va="top",
                        fontproperties=fprop("semi", 16.5), zorder=7), "PLAT")
pl_txt2 = reg(ax_d2.text(22900, 0.705, "plateau mAP@0,5 ≈ 0,62–0,65", color=TXT, ha="right",
                         va="bottom", fontsize=pt(16.5), zorder=7), "PLAT")
ax_d2.set_ylabel("mAP@0,5", fontsize=pt(15), color=TICK, labelpad=6)
ax_d.set_ylabel("strata", fontsize=pt(15), color=TICK, labelpad=4)
YLAB = [ax_d.yaxis.label, ax_d2.yaxis.label]

# --- pasma faz (rosną za kursorem)
BANDS = []
for i, (x0, x1, *_rest) in enumerate(PHASES):
    for key, ax in PANELS.items():
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        r = Rectangle((x0, 0), 0, 1, transform=tr, color=PH_COL[i], lw=0, zorder=0.5)
        ax.add_patch(r)
        BANDS.append((r, key, i))
PH_BAR, PH_LBL = [], []
for i, (x0, x1, short, *_r) in enumerate(PHASES):
    r = Rectangle((x0, 0.0), 0, 0.15, color=PH_COL[i], lw=0)
    ax_ph.add_patch(r)
    PH_BAR.append(r)
    row_y = 0.76 if i % 2 == 0 else 0.40
    tr = offset_copy(ax_ph.transData, fig=fig, x=4, y=0, units="dots")
    PH_LBL.append(ax_ph.text(x0, row_y, f"{i + 1} · {short}", transform=tr, color=PH_COL[i], ha="left",
                             va="center", fontproperties=fprop("semi", 19)))
T_PH = [t_when(p[0] + 0.5) for p in PHASES]

# --- kursor czasu
CUR_LINES = [ax.axvline(0, color=WHITE, lw=1.3, zorder=7) for ax in (ax_a, ax_b, ax_c, ax_d)]

# --- linia zmiany zapisu
BRK = [ax.axvline(DENSE_END, color=WHITE, lw=1.1, ls=(0, (4, 4)), zorder=6) for ax in (ax_a, ax_b, ax_c)]
brk_txt = ax_b.text(DENSE_END, 1.25, "zapis co 10 kroków →",
                    transform=offset_copy(ax_b.transData, fig=fig, x=8, y=0, units="dots"),
                    color=TXT, ha="left", va="center", fontproperties=fprop("semi", 16.5), zorder=9)

# --- dymek przy kroku 1000
j1000 = int(np.argmin(np.abs(xd - 1000)))
Tbg, Ttool = temp_d[j1000, 0], temp_d[j1000, 3]
TR_BR = offset_copy(ax_b.transData, fig=fig, x=16, y=0, units="dots")
dym_br = ax_b.annotate("", xy=(1000, Tbg), xytext=(1000, Ttool), xycoords=TR_BR, textcoords=TR_BR,
                       arrowprops=dict(arrowstyle="<->", color=WHITE, lw=1.8, shrinkA=0, shrinkB=0,
                                       mutation_scale=14), zorder=9)
dym = ax_b.annotate(f"narzędzie {pl(tool_bg_1000, 1)} × cieplejsze od tła",
                    xy=(1000, math.sqrt(Tbg * Ttool)), xycoords=offset_copy(ax_b.transData, fig=fig, x=22, y=0,
                                                                          units="dots"),
                    xytext=(1240, 0.26), textcoords="data",
                    color=TXT, ha="left", va="center", fontproperties=fprop("semi", 20), zorder=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#1a242b", ec=REG_COL[3], lw=1.5),
                    arrowprops=dict(arrowstyle="-", color=REG_COL[3], lw=1.3, shrinkA=0, shrinkB=3,
                                    connectionstyle="arc3,rad=0.25"))

# --- tytuł, podtytuł, intro
t_title = fig.text(fx(67), fy(50), TITLE, ha="left", va="center", color=TXT, fontproperties=fprop("bold", 34),
                   zorder=60)
t_sub = fig.text(fx(68), fy(88), SUBTITLE, ha="left", va="center", color=TXT2, fontsize=pt(18), zorder=60)
t_intro = fig.text(0.5, fy(500), INTRO, ha="center", va="center", color=TXT, fontsize=pt(30),
                   linespacing=1.45)
t_intro2 = fig.text(0.5, fy(610), INTRO2, ha="center", va="center", color=YEL, fontsize=pt(22))
t_xlab = fig.text(fx((LPX + RPX) / 2), fy(958), "krok uczenia  (1 epoka = 1094 kroki)", ha="center",
                  va="center", color=TICK, fontsize=pt(16))

# --- prawa kolumna: zegar
WRAP = 40
c_lbl = fig.text(fx(RCOL), fy(140), "KROK UCZENIA", ha="left", va="center", color=DIM,
                 fontproperties=fprop("semi", 15))
c_num = fig.text(fx(RCOL), fy(188), "0", ha="left", va="center", color=TXT, fontproperties=fprop("bold", 50))
c_sub = fig.text(fx(RCOL), fy(236), "", ha="left", va="center", color=TXT2, fontsize=pt(18))
c_sep = Line2D([fx(RCOL), fx(1890)], [fy(266), fy(266)], color=SPINE, lw=1.2, transform=fig.transFigure)
fig.add_artist(c_sep)

R_BLOCKS = []
for i, p in enumerate(PHASES):
    y0 = 290 + i * 142
    h = fig.text(fx(RCOL), fy(y0), p[3], ha="left", va="top", color=PH_COL[i], fontproperties=fprop("semi", 20))
    dsc = fig.text(fx(RCOL), fy(y0 + 32), textwrap.fill(p[4], WRAP), ha="left", va="top", color=TXT2,
                   fontsize=pt(19), linespacing=1.32)
    R_BLOCKS.append((h, dsc))

n1h = fig.text(fx(RCOL), fy(292), NOTE1_H, ha="left", va="top", color=TXT, fontproperties=fprop("semi", 20))
n1d = fig.text(fx(RCOL), fy(324), textwrap.fill(NOTE1_D, WRAP), ha="left", va="top", color=TXT2,
               fontsize=pt(19), linespacing=1.32)
n2h = fig.text(fx(RCOL), fy(470), NOTE2_H, ha="left", va="top", color=TXT, fontproperties=fprop("semi", 20))
n2d = fig.text(fx(RCOL), fy(502), textwrap.fill(NOTE2_D, WRAP), ha="left", va="top", color=TXT2,
               fontsize=pt(19), linespacing=1.32)
cap = fig.text(fx(RCOL + 12), fy(632), textwrap.fill(CAPTION, 31), ha="left", va="top", color=TXT,
               fontproperties=fprop("semi", 21), linespacing=1.32,
               bbox=dict(boxstyle="round,pad=0.7", fc="#1a242b", ec=YEL, lw=1.6))

# --- karta wniosków
overlay = Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=BG, lw=0, zorder=50)
fig.add_artist(overlay)
k_title = fig.text(fx(170), fy(205), "Po czym poznać fazy uczenia z samego ruchu cząsteczek",
                   ha="left", va="center", color=TXT, fontproperties=fprop("bold", 32), zorder=61)
k_rule = Line2D([fx(170), fx(1750)], [fy(245), fy(245)], color=YEL, lw=2.0, transform=fig.transFigure, zorder=61)
fig.add_artist(k_rule)
K_ITEMS = []
for i, (lead, det) in enumerate(ITEMS):
    y = 315 + i * 150
    b = fig.text(fx(195), fy(y + 20), str(i + 1), ha="center", va="center", color=BG,
                 fontproperties=fprop("bold", 24), zorder=62,
                 bbox=dict(boxstyle="circle,pad=0.38", fc=YEL, ec="none"))
    tl = fig.text(fx(250), fy(y), lead, ha="left", va="center", color=TXT, fontproperties=fprop("semi", 28),
                  zorder=62)
    td = fig.text(fx(250), fy(y + 44), det, ha="left", va="center", color=TXT2, fontsize=pt(21.5), zorder=62)
    K_ITEMS.append((b, tl, td))
k_foot = fig.text(fx(170), fy(985),
                  "Dane: trajektorie_wlasnosci.npz, PODSUMOWANIE.json (ten bieg) · OCENY.json "
                  "(mAP@0,5: inny bieg tego samego przepisu). Temperatura i pole w PCA32.",
                  ha="left", va="center", color=DIM, fontsize=pt(16), zorder=62)


# ------------------------------------------------------------------ pomocnicze
def chrome(ax, a, xlab_a=0.0, ylab_side="left"):
    ax.set_visible(a > 0.002)
    ax.patch.set_alpha(a if ax is not ax_d2 else 0)
    for name, s in ax.spines.items():
        s.set_edgecolor(rgba(SPINE, a))
    ax.tick_params(axis="both", colors=rgba(SPINE, a))
    ax.tick_params(axis="y", labelcolor=rgba(TICK, a))
    ax.tick_params(axis="x", labelcolor=rgba(TICK, a * xlab_a), labelbottom=xlab_a > 0.002)
    if ax is not ax_d2:
        ax.yaxis.grid(True, color=rgba(GRID, a), lw=1.0)
        ax.set_axisbelow(True)


def upto(x, c):
    return int(np.searchsorted(x, c, side="right"))


def set_text(tobj, a, dx_px=0.0, base_xy=None):
    tobj.set_alpha(float(np.clip(a, 0, 1)))
    tobj.set_visible(a > 0.002)
    bp = tobj.get_bbox_patch()
    if bp is not None:
        bp.set_alpha(float(np.clip(a, 0, 1)))
    if base_xy is not None:
        tobj.set_position((base_xy[0] + fx(dx_px), base_xy[1]))


BASE_POS = {}
for tobj in [*[x for blk in R_BLOCKS for x in blk], n1h, n1d, n2h, n2d, cap,
             *[x for it in K_ITEMS for x in it], k_title]:
    BASE_POS[id(tobj)] = tobj.get_position()


def place(tobj, a, slide_px, prog=None):
    prog = a if prog is None else prog
    set_text(tobj, a, dx_px=(1 - min(prog, 1)) * slide_px, base_xy=BASE_POS[id(tobj)])


# ------------------------------------------------------------------ klatka
def update(t):
    c = cursor(t)
    p2 = ramp(t, *T_RELAYOUT)
    z = ramp(t, *T_ZOOM)
    card = ramp(t, *T_CARD)

    # --- intro / tytuł
    intro_a = 1 - ramp(t, 2.2, 2.9)
    set_text(t_intro, intro_a)
    set_text(t_intro2, ramp(t, 0.6, 1.3) * intro_a)
    set_text(t_title, 1.0)
    set_text(t_sub, ramp(t, 2.5, 3.2))

    # --- widoczność paneli
    pa = {k: ramp(t, s, s + 0.7) for k, s in zip("ABC", T_PANEL)}
    pa["D"] = ramp(t, *T_DIN)
    for key, ax in PANELS.items():
        top = lerp(L1[key][0], L2[key][0], p2)
        bot = lerp(L1[key][1], L2[key][1], p2)
        slide = (1 - pa[key]) * 26
        ax.set_position(rect(top, bot, dy=slide))
    ax_d2.set_position(ax_d.get_position())
    ax_ph.set_position(rect(*L1["ph"]))

    chrome(ax_a, pa["A"])
    chrome(ax_b, pa["B"])
    chrome(ax_c, pa["C"], xlab_a=1 - p2)
    chrome(ax_d, pa["D"], xlab_a=pa["D"])
    chrome(ax_d2, pa["D"])
    for yl in YLAB:
        yl.set_alpha(pa["D"])
    set_text(t_xlab, pa["C"])

    # --- osie
    xmax = glerp(2200.0, 23000.0, z)
    ax_a.set_xlim(0, xmax)
    ax_b.set_ylim(0.0015, glerp(1.0, 2.0, z))
    ax_c.set_ylim(lerp(0.3, -0.45, z), 1.03)

    # --- serie do kursora
    cur_on = ramp(t, 4.3, 4.8) * (1 - card)
    owners = dict(pa)
    owners["CUR"] = cur_on
    owners["CURD"] = cur_on * pa["D"]
    plat = ramp(t, *T_PLATEAU)
    owners["PLAT"] = plat
    for art, owner, base, _k in ART:
        a = base * owners.get(owner, 1.0)
        art.set_alpha(float(np.clip(a, 0, 1)))
        art.set_visible(a > 0.002)

    n_lr = upto(steps, c)
    l_lr.set_data(steps[:n_lr], lr_n[:n_lr])
    nd = upto(xd, c)
    ns = upto(xs, c)
    l_sh_raw.set_data(xd[:nd], share_d_raw[:nd])
    l_sh_d.set_data(xd[:nd], share_d[:nd])
    l_sh_s.set_data(xs[:ns], share_s[:ns])
    l_sh_raw_s.set_data(xs[:ns], share_s_raw[:ns])
    for j in range(4):
        l_t_d[j].set_data(xd[:nd], temp_d[:nd, j])
        l_t_s[j].set_data(xs[:ns], temp_s[:ns, j])
        l_c_d[j].set_data(xd[:nd], cos_d[:nd, j])
        l_c_s[j].set_data(xs[:ns], cos_s[:ns, j])
    l_loss_raw.set_data(steps[:n_lr], loss_raw[:n_lr])
    l_loss.set_data(steps[:n_lr], loss_sm[:n_lr])
    ne = upto(ep_x, c + 8)
    l_map.set_data(ep_x[:ne], ep_y[:ne])

    # kropki na kursorze
    if ns > 0:
        xi, sh_i, t_i = xs[ns - 1], share_s[ns - 1], temp_s[ns - 1]
    elif nd > 0:
        xi, sh_i, t_i = xd[nd - 1], share_d[nd - 1], temp_d[nd - 1]
    else:
        xi, sh_i, t_i = None, None, None
    if xi is not None:
        dot_a.set_data([xi], [sh_i])
        for j in range(4):
            dot_b[j].set_data([xi], [t_i[j]])
    else:
        dot_a.set_data([], [])
        for j in range(4):
            dot_b[j].set_data([], [])
    if n_lr > 0:
        dot_lr.set_data([steps[n_lr - 1]], [lr_n[n_lr - 1]])
        dot_d.set_data([steps[n_lr - 1]], [loss_sm[n_lr - 1]])
    else:
        dot_lr.set_data([], [])
        dot_d.set_data([], [])

    for ln in CUR_LINES:
        ln.set_xdata([c, c])
        ln.set_alpha(0.85 * cur_on)
        ln.set_visible(cur_on > 0.002)
    CUR_LINES[3].set_alpha(0.85 * cur_on * pa["D"])

    # --- pasma faz i pasek faz
    lbl_fade = 1 - ramp(t, T_RELAYOUT[0], T_RELAYOUT[0] + 0.6)
    band_base = [0.075, 0.05, 0.075, 0.06]
    for r, key, i in BANDS:
        x0, x1 = PHASES[i][0], PHASES[i][1]
        r.set_width(max(0.0, min(c, x1) - x0))
        r.set_alpha(band_base[i] * pa[key] * (1 - 0.35 * z))
    for i, (r, lbl) in enumerate(zip(PH_BAR, PH_LBL)):
        x0, x1 = PHASES[i][0], PHASES[i][1]
        r.set_width(max(0.0, min(c, x1) - x0))
        r.set_alpha(0.85 * (1 - card))
        set_text(lbl, ramp(t, T_PH[i], T_PH[i] + 0.5) * lbl_fade)

    # --- dymek
    da = ramp(t, *T_DYMEK) * (1 - ramp(t, T_RELAYOUT[0], T_RELAYOUT[0] + 0.5))
    set_text(dym, da)
    dym.arrow_patch.set_alpha(da)
    dym.set_visible(da > 0.002)
    dym_br.arrow_patch.set_alpha(da)
    dym_br.set_visible(da > 0.002)

    # --- linia zmiany zapisu
    ba = ramp(t, *T_BREAK) * (1 - card)
    for ln in BRK:
        ln.set_alpha(0.75 * ba)
        ln.set_visible(ba > 0.002)
    set_text(brk_txt, ba)

    # --- zegar
    clk = ramp(t, 2.8, 3.5)
    set_text(c_lbl, clk)
    set_text(c_num, clk)
    set_text(c_sub, clk)
    c_sep.set_alpha(clk)
    k_int = int(round(c))
    c_num.set_text(f"{k_int:,}".replace(",", " "))
    lr_now = lr[min(max(k_int, 1), len(lr)) - 1] if k_int >= 1 else 0.0
    c_sub.set_text(f"epoka {pl(c / 1094.0, 2)}   ·   lr = {pl(lr_now * 1e4, 2)} × 10⁻⁴")

    # --- bloki faz w prawej kolumnie
    out1 = 1 - ramp(t, T_RELAYOUT[0], T_RELAYOUT[0] + 0.6)
    app = [ramp(t, T_PH[i], T_PH[i] + 0.6) for i in range(4)]
    for i, (h, dsc) in enumerate(R_BLOCKS):
        nxt = app[i + 1] if i + 1 < 4 else 0.0
        a = app[i] * (1 - 0.5 * nxt) * out1
        place(h, a, 24, app[i])
        place(dsc, a, 24, app[i])

    # --- notatki części 2
    place(n1h, ramp(t, *T_NOTE1), 24)
    place(n1d, ramp(t, *T_NOTE1), 24)
    a2 = ramp(t, T_NOTE2, T_NOTE2 + 0.6)
    place(n2h, a2, 24)
    place(n2d, a2, 24)
    place(cap, plat, 24)

    # --- karta wniosków
    overlay.set_alpha(0.985 * card)
    overlay.set_visible(card > 0.002)
    ka = ramp(t, T_CARD[0] + 0.3, T_CARD[1] + 0.3)
    place(k_title, ka, -30)
    k_rule.set_alpha(ka)
    k_rule.set_visible(ka > 0.002)
    for i, (b, tl, td) in enumerate(K_ITEMS):
        a = ramp(t, T_ITEMS[i], T_ITEMS[i] + 0.6)
        place(b, a, -30)
        place(tl, a, -30)
        place(td, a, -30)
    set_text(k_foot, ramp(t, T_ITEMS[-1], T_ITEMS[-1] + 0.6))


def frame_rgba(t):
    update(t)
    canvas.draw()
    return canvas.buffer_rgba()


# ------------------------------------------------------------------ main
def main():
    args = sys.argv[1:]
    if args and args[0] == "--preview":
        outdir = Path(os.environ.get("PREVIEW_DIR", HERE))
        for s in args[1:]:
            update(float(s))
            fig.savefig(outdir / f"preview_{float(s):05.1f}.png", dpi=DPI, facecolor=BG)
            print("preview", s)
        return

    print(f"[render] {NFR} klatek, {DUR:.2f} s -> {OUT}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W_PX}x{H_PX}", "-r", str(FPS), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NFR):
        buf = frame_rgba(i / FPS)
        assert canvas.get_width_height() == (W_PX, H_PX)
        proc.stdin.write(buf)
        if i % 150 == 0:
            print(f"  klatka {i}/{NFR}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg zakończył się błędem")

    # klatki kontrolne z gotowego MP4: 0 %, 33 %, 66 %, 100 % czasu
    for n, q in enumerate([0.0, 0.33, 0.66, 1.0]):
        idx = int(round(q * (NFR - 1)))
        png = HERE / f"verify_{n:02d}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT), "-vf",
                        f"select=eq(n\\,{idx})", "-vsync", "0", "-frames:v", "1", str(png)], check=True)
        print(f"  {png.name}: klatka {idx} (t = {idx / FPS:.2f} s)")
    print("[render] gotowe")


if __name__ == "__main__":
    main()
