"""Film 19 = film 18 z wyraźną synchronizacją widoków (film 18 zostaje bez zmian). Od początku uczenia oba widoki pokazują ten sam
krok: wspólny pasek osi czasu przez całą szerokość, licznik kroku nad każdym widokiem, a w zbliżeniu prędkość cząstek i długość strzałek
rosną z rzeczywistą siłą pola w danym kroku (film 18 normował pole, więc zbliżenie wyglądało na stałe).
Opis filmu 18:
Pułapka bloku MLP 6 w dwóch widokach, oba z TEGO SAMEGO biegu (MalyViT_35000_od_zera_s47, 806 checkpointów do 60. epoki).
Łączy film 17 (widok ogólny) i film 16 (zbliżenie), ale film 16 liczył Jakobian na wagach innego biegu — tu jest przeliczony.

Lewy widok (ogólny, jak film 17): 768 wiązek krzyżuje się w tokenie narzędzia 316; wokół 599 prawdziwych tokenów zdjęcia, kolor =
przyciąganie (niebieski) / odpychanie (pomarańczowy) względem tokenu 316, liczone w 192D po LayerNorm.
Prawy widok (zbliżenie, jak film 16): lokalne pole bloku wokół tokenu 316, J = ∂F/∂x = W₂ diag(GELU′(a)) W₁ J_LN(x) — DOKŁADNY
Jakobian w punkcie x (strumień rezydualny przed MLP z przejścia modelu, bez odtwarzania z latentu). J = S + A; kierunki e₁ (najmocniej
cofa), e₂ (najmocniej wypycha), e₃ (płaszczyzna wiru) z 60. epoki; cząstki próbne dδ/dt = J₃ δ (prędkość unormowana).
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_18_pulapka_dwa_widoki.py [proba]
"""
import subprocess
import sys
import time

import moderngl
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.special import erf

from lasery_wspolne import EP, H, HERE, TOK, W, dane, oko

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
FPS, WY, HY, SS = 30, 1920, 1080, 2
WH = WY // 2                                                                       # szerokość jednego widoku
NK = 768
ROOT = HERE.parents[2]
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
C17 = HERE / 'pulapka_17_cache.npz'
CACHE = HERE / 'pulapka_18_cache.npz'
gelu = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
dgelu = lambda a: 0.5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)

# ---------------- dane ----------------
D = dane(0)
kroki, iou, narz = D['kroki'].astype(int), D['iou'], D['narz']
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
OKO_F = oko()
if not C17.exists():
    raise SystemExit('Brak pulapka_17_cache.npz: najpierw uruchom film_lasery_17_pulapka_ognisko.py')
c17 = np.load(C17)
X, F, Zf, V = (c17[k].astype(np.float32) for k in ('X', 'F', 'Zf', 'V'))
NT = len(kroki)

if CACHE.exists():
    JJ = np.load(CACHE)['JJ'].astype(np.float64)
else:
    JJ, kontrola = [], []
    for n, k in enumerate(kroki):
        s = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
        g = lambda nm: s[f'bloki.5.{nm}'].double().numpy()
        W1, b1, W2, gam, bet = g('mlp.fc1.weight'), g('mlp.fc1.bias'), g('mlp.fc2.weight'), g('norm2.weight'), g('norm2.bias')
        x = X[n, FOK].astype(np.float64); d = len(x)
        u = x - x.mean(); sd = np.sqrt((u * u).mean() + 1e-5)
        a = W1 @ (gam * u / sd + bet) + b1
        Jln = gam[:, None] * ((np.eye(d) - 1.0 / d) / sd - np.outer(u, u) / (d * sd ** 3))
        JJ.append(W2 @ (dgelu(a)[:, None] * W1) @ Jln)
        if n % 100 == 0:
            kontrola.append(float(1 - ((gelu(a) - Zf[n]) ** 2).sum() / ((Zf[n] - Zf[n].mean()) ** 2).sum()))
            log('checkpoint', n, k, 'R² GELU(a) vs zapisany latent', round(kontrola[-1], 5))
    JJ = np.array(JJ)
    np.savez_compressed(CACHE, JJ=JJ.astype(np.float32))
log('Jakobiany', JJ.shape)


def statystyki(J):
    S, A = (J + J.T) / 2, (J - J.T) / 2
    ev = np.linalg.eigvalsh(S)
    return dict(ujemne=float((ev < 0).mean()), slad=float(np.trace(J) / len(J)), sym=float(np.linalg.norm(S) ** 2 / np.linalg.norm(J) ** 2),
                rot=float(np.linalg.norm(A) / np.linalg.norm(S)), ev=ev, norma=float(np.linalg.norm(J, 2)))


STAT = [statystyki(JJ[i]) for i in range(NT)]
NORMA0 = STAT[0]['norma']
NORMA_T = np.array([s_['norma'] for s_ in STAT])
for i in (0, 100, 300, NT - 1):
    log(kroki[i], {k: round(v, 3) for k, v in STAT[i].items() if k != 'ev'})

# baza zbliżenia z 60. epoki
Je = JJ[-1]; Se, Ae = (Je + Je.T) / 2, (Je - Je.T) / 2
w_, V_ = np.linalg.eigh(Se)
e1, e2 = V_[:, 0], V_[:, -1]
P12 = np.outer(e1, e1) + np.outer(e2, e2)
U3, _, _ = np.linalg.svd((np.eye(len(e1)) - P12) @ Ae @ np.stack([e1, e2], 1), full_matrices=False)
Qb = np.stack([e1, e2, U3[:, 0]], 1)
J3 = np.einsum('ai,tij,jb->tab', Qb.T, JJ, Qb)

# widok ogólny (jak film 17)
DEL = X - X[:, FOK:FOK + 1]


def _ln(a):
    a = a - a.mean(-1, keepdims=True)
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-5)


_A0, _A1 = _ln(X), _ln(X + F)
R_ = np.linalg.norm(_A1 - _A1[:, FOK:FOK + 1], axis=2) / np.maximum(np.linalg.norm(_A0 - _A0[:, FOK:FOK + 1], axis=2), 1e-9)
R_[:, FOK] = 1.0
del _A0, _A1
U_, _, _ = np.linalg.svd((DEL[-1] - DEL[-1].mean(0)).T, full_matrices=False)
P = U_[:, :3].T
D3, DZ3 = DEL @ P.T, (F - F[:, FOK:FOK + 1]) @ P.T
SKALA_T = np.percentile(np.linalg.norm(D3, axis=2), 95, axis=1)
POW = SKALA_T[-1] / SKALA_T
PV = np.einsum('ad,tdc->tca', P, V)
DIR = PV / np.maximum(np.linalg.norm(PV, axis=2, keepdims=True), 1e-9)
STAT_P = dict(nar=(R_[:, narz & (np.arange(TOK) != FOK)] < 1).mean(1), tlo=(R_[:, ~narz] < 1).mean(1))
sys.path.insert(0, str(ROOT / 'app'))
from wspolne import kolory_kanalow                                                 # noqa: E402
KOL = kolory_kanalow()[:, :3]

# ---------------- oś czasu ----------------
FAZY = [(0, 'start'), (8, 'tokeny'), (16, 'S'), (22, 'A'), (28, 'razem'), (31, 'uczenie')]
T_U0 = 32.0
_kr, kr_ = [], 0.0
while kr_ < kroki[-1]:
    _kr.append(kr_); kr_ += min(1.5 + kr_ / 120, 300)
KR_U = np.array(_kr + [kroki[-1]])
T_KON = T_U0 + len(_kr) / FPS
FAZY += [(T_KON + 0.5, 'koniec'), (T_KON + 6.5, None)]
log('uczenie trwa', round(len(_kr) / FPS, 1), 's; film', round(T_KON + 6.5, 1), 's')
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def krok_t(t):
    if t < T_U0:
        return float(kroki[-1]) if t < 31 else float(kroki[-1]) * (1 - sm((t - 31) / 1.0))
    return float(KR_U[int(min((t - T_U0) * FPS, len(KR_U) - 1))])


def stan(t):
    kr = krok_t(t); fi = float(np.interp(kr, kroki, np.arange(NT))); i = int(min(fi, NT - 2)); u = fi - i
    L = lambda A: A[i] * (1 - u) + A[i + 1] * u
    s = 11.0 / L(SKALA_T)
    d = L(DIR); d /= np.linalg.norm(d, axis=1, keepdims=True)
    J = L(J3); S_, A_ = (J + J.T) / 2, (J - J.T) / 2
    if 16 <= t < 22:
        J = S_ * sm((t - 16) / 1.0) + J * (1 - sm((t - 16) / 1.0))
    elif 22 <= t < 28:
        J = A_ * sm((t - 22) / 1.0) + S_ * (1 - sm((t - 22) / 1.0))
    elif 28 <= t < 30:
        J = J * sm((t - 28) / 1.5) + A_ * (1 - sm((t - 28) / 1.5))
    sila = float(np.interp(kr, kroki, NORMA_T)) / NORMA_T[-1]                    # rzeczywista siła pola względem 60. epoki
    return dict(kr=kr, ii=i if u < 0.5 else i + 1, D=L(D3) * s, DZ=L(DZ3) * s, R=L(R_), dir=d, z=L(Zf), PV=L(PV),
                pow=L(POW), J=J / max(np.linalg.norm(J, 2), 1e-9), sila=sila)


# ---------------- kamery ----------------
def kamera(az, el, di):
    a, e = np.radians(az), np.radians(el)
    oko_ = di * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = -oko_ / np.linalg.norm(oko_); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    Vm = np.eye(4); Vm[0, :3], Vm[1, :3], Vm[2, :3] = s, up, -f; Vm[:3, 3] = -Vm[:3, :3] @ oko_
    fov, asp, n, fa = np.radians(40), WH / HY, 0.1, 300.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n) / (n - fa); Pm[2, 3] = 2 * fa * n / (n - fa); Pm[3, 2] = -1
    return Pm @ Vm, 1 / np.tan(fov / 2) * HY * SS / 2


def rzut(mvp, p, x0):
    q = mvp @ np.append(p, 1.0)
    return None if q[3] <= 0.1 else (x0 + (q[0] / q[3] * 0.5 + 0.5) * WH, (0.5 - q[1] / q[3] * 0.5) * HY)


# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
W4, H4 = WY * SS, HY * SS
hdr = ctx.texture((W4, H4), 4, dtype='f2'); fb_hdr = ctx.framebuffer([hdr])
ldr = ctx.texture((W4, H4), 4); fb_ldr = ctx.framebuffer([ldr])
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
kropki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform float skala; in vec3 pos; in vec4 kol; in float rozm; out vec4 v_k;
void main() { gl_Position = mvp * vec4(pos, 1.0); gl_PointSize = clamp(rozm * skala / gl_Position.w, 1.5, 60.0); v_k = kol; }''',
                     fragment_shader='''#version 430
uniform int ksztalt; in vec4 v_k; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard;
  float a = ksztalt == 1 ? smoothstep(0.55, 0.75, sqrt(r2)) * (1.0 - smoothstep(0.85, 1.0, sqrt(r2))) : exp(-r2 * 3.0);
  o = vec4(v_k.rgb * a * v_k.a, a * v_k.a); }''')
linie = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; in vec3 pos; in vec4 kol; out vec4 g_k;
void main() { gl_Position = mvp * vec4(pos, 1.0); g_k = kol; }''', geometry_shader='''#version 430
layout(lines) in; layout(triangle_strip, max_vertices = 4) out;
uniform vec2 vp; uniform float szer; in vec4 g_k[]; out vec4 f_k;
void main() {
  vec4 p0 = gl_in[0].gl_Position, p1 = gl_in[1].gl_Position;
  if (p0.w < 0.05 || p1.w < 0.05) return;
  vec2 s0 = p0.xy / p0.w * vp * 0.5, s1 = p1.xy / p1.w * vp * 0.5, d = s1 - s0;
  d = length(d) < 1e-4 ? vec2(1.0, 0.0) : normalize(d);
  vec2 n = vec2(-d.y, d.x) * szer * 0.5 / (vp * 0.5);
  f_k = g_k[0]; gl_Position = p0 + vec4(n * p0.w, 0, 0); EmitVertex(); gl_Position = p0 - vec4(n * p0.w, 0, 0); EmitVertex();
  f_k = g_k[1]; gl_Position = p1 + vec4(n * p1.w, 0, 0); EmitVertex(); gl_Position = p1 - vec4(n * p1.w, 0, 0); EmitVertex();
  EndPrimitive(); }''', fragment_shader='''#version 430
in vec4 f_k; out vec4 o; void main() { o = vec4(f_k.rgb * f_k.a, f_k.a); }''')
linie['vp'].value = (WH * SS, H4)
ton = ctx.program(vertex_shader='''#version 430
in vec2 p; out vec2 uv; void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0, 1); }''', fragment_shader='''#version 430
uniform sampler2D tex; in vec2 uv; out vec4 o;
void main() { vec3 c = texture(tex, uv).rgb; c = 1.0 - exp(-c); o = vec4(pow(c, vec3(0.85)), 1.0); }''')
quad = ctx.vertex_array(ton, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4')), '2f', 'p')])


def rysuj(prog, tryb, poz, kol, rozm=None):
    if len(poz) == 0:
        return
    dane_ = [poz.astype('f4'), kol.astype('f4')] + ([rozm.astype('f4')[:, None]] if rozm is not None else [])
    b = ctx.buffer(np.concatenate(dane_, 1).tobytes())
    fmt = '3f 4f 1f' if rozm is not None else '3f 4f'
    v = ctx.vertex_array(prog, [(b, fmt, *(['pos', 'kol', 'rozm'] if rozm is not None else ['pos', 'kol']))])
    v.render(tryb); v.release(); b.release()


def strzalki(p0, p1, kol, glow=0.3):
    d = p1 - p0; L = np.linalg.norm(d, axis=1, keepdims=True); dn = d / np.maximum(L, 1e-9)
    bok = np.cross(dn, [0, 0, 1.0]); bok /= np.maximum(np.linalg.norm(bok, axis=1, keepdims=True), 1e-9)
    hl = np.minimum(glow, 0.4 * L); t_ = p1 - dn * hl
    seg = np.concatenate([np.stack([p0, p1], 1), np.stack([p1, t_ + bok * hl * 0.5], 1), np.stack([p1, t_ - bok * hl * 0.5], 1)]).reshape(-1, 3)
    return seg, np.repeat(np.tile(kol, (3, 1)), 2, 0)


def barwa_r(R):
    lr = np.clip(np.log(np.clip(R, 1e-6, None)) / 0.12, -1, 1)[:, None]
    return np.where(lr < 0, np.array([[0.35, 0.62, 1.0]]), np.array([[1.0, 0.55, 0.25]])), np.abs(lr)


# zbliżenie: siatka strzałek, osie, cząstki próbne
R0 = 10.0
g_ = np.linspace(-1, 1, 7)
SIAT = np.array([[x, y, z] for x in g_ for y in g_ for z in g_ if 0.25 < x * x + y * y + z * z <= 1.0]) * R0
OSIE = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float) * R0 * 1.25
KOL_OSI = np.array([[0.35, 0.6, 1.0], [0.35, 0.6, 1.0], [1.0, 0.4, 0.3], [1.0, 0.4, 0.3], [0.8, 0.5, 1.0], [0.8, 0.5, 1.0]])
rng = np.random.default_rng(1)
NP_, HIST = 500, 24


def losuj(n):
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * R0 * rng.uniform(0.15, 0.9, (n, 1)) ** (1 / 3)


TR = losuj(NP_); WIEK = rng.uniform(0, 3, NP_)
SLAD = np.repeat(TR[None], HIST, 0)


def krok_czastek(J, dt):
    global TR, WIEK, SLAD
    TR = TR + (TR @ J.T) * dt + rng.normal(scale=0.002 * R0, size=TR.shape) * np.sqrt(dt * 30)
    WIEK += dt
    r = np.linalg.norm(TR, axis=1)
    nowe = (r > 1.25 * R0) | (r < 0.05 * R0) | (WIEK > 4.0)
    if nowe.any():
        TR[nowe] = losuj(int(nowe.sum())); WIEK[nowe] = 0.0
    SLAD = np.roll(SLAD, 1, 0); SLAD[0] = TR
    SLAD[:, nowe] = TR[nowe]


# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
TEKSTY = {
    'start': ('Pułapka bloku MLP 6 w dwóch widokach (jeden bieg: wagi i aktywacje zgodne)',
              'Z lewej: 768 wiązek krzyżuje się w tokenie narzędzia.  Z prawej: zbliżenie — lokalne pole tych wiązek wokół tego tokenu (Jakobian J).'),
    'tokeny': ('Z lewej prawdziwe tokeny, z prawej cząstki próbne w polu J',
               'Lewy: niebieski = blok przyciąga do narzędzia, pomarańczowy = odpycha.  Prawy: e₁ cofa, e₂ wypycha, część antysymetryczna J skręca tory.'),
    'S': ('Zbliżenie: tylko część symetryczna S = (J + Jᵀ)/2 — czyste siodło', 'Ujemne wartości własne S cofają, dodatnie wypychają; bez wiru tory są proste.'),
    'A': ('Zbliżenie: tylko część antysymetryczna A = (J − Jᵀ)/2 — czysty wir', 'A tylko obraca; siła z wirem nie jest gradientem żadnego potencjału.'),
    'razem': ('Zbliżenie: siodło + wir', ''),
    'uczenie': ('Uczenie: oba widoki pokazują TEN SAM krok (wspólny pasek u dołu)',
                'Pierwsze kroki wolno, potem epoki szybko.  W zbliżeniu prędkość cząstek i długość strzałek rosną z rzeczywistą siłą pola w tym kroku.'),
    'koniec': ('Po 60 epokach', ''),
}
_k = STAT[-1]
TEKSTY['koniec'] = ('Po 60 epokach: siodło z wirem prawie jak w macierzy losowej',
                    f'‖A‖/‖S‖ = {pl(_k["rot"], 2)}, symetryczność {_k["sym"]:.0%}, ślad J/192 = {pl(_k["slad"], 3)}, siła pola ×{pl(_k["norma"] / NORMA0, 1)}.  '
                    'Film 16 (wagi z innego biegu) przeszacował kontrakcję i wzrost pola.')


def nakladka(img, faza, t, S, mvpL, mvpR):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    d.text((WY / 2, 18), tyt, font=FN(30, True), fill=(235, 240, 245, 255), anchor='mt')
    if pod:
        d.text((WY / 2, 60), pod, font=FN(19), fill=(255, 190, 120, 255), anchor='mt')
    d.line((WH, 95, WH, HY - 120), fill=(60, 70, 80, 255), width=2)
    d.text((20, 100), 'WIDOK OGÓLNY: 600 tokenów zdjęcia wokół ogniska', font=FN(20, True), fill=(200, 225, 255, 255))
    d.text((WH + 20, 100), 'ZBLIŻENIE: pole bloku w samej pułapce (J)', font=FN(20, True), fill=(200, 225, 255, 255))
    # wspólny licznik i pasek kroku (od początku uczenia)
    ucz = faza in ('uczenie', 'koniec')
    lab = f'krok {S["kr"]:,.0f}'.replace(',', ' ') + f'  ·  epoka {pl(S["kr"] / EP, 2)}'
    kol_k = (255, 225, 120, 255) if ucz else (150, 155, 160, 255)
    for x_ in (WH - 20, WY - 20):
        d.text((x_, 100), lab, font=FN(22, True), fill=kol_k, anchor='ra')
    x0, x1, y0 = 60, WY - 60, HY - 100
    d.rectangle((x0 - 14, y0 - 30, x1 + 14, y0 + 30), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
    d.line((x0, y0, x1, y0), fill=(80, 90, 100, 255), width=3)
    poz = lambda kr: x0 + (x1 - x0) * np.log1p(kr / 50) / np.log1p(kroki[-1] / 50)    # oś logarytmiczna: pierwsze kroki szeroko
    for ep in (0, 1, 2, 5, 10, 20, 40, 60):
        xe = poz(ep * EP)
        d.line((xe, y0 - 7, xe, y0 + 7), fill=(150, 160, 170, 255), width=2)
        d.text((xe, y0 + 10), f'ep {ep}', font=FN(14), fill=(170, 180, 190, 255), anchor='mt')
    xm = poz(S['kr'])
    d.line((x0, y0, xm, y0), fill=kol_k, width=5)
    d.ellipse((xm - 9, y0 - 9, xm + 9, y0 + 9), fill=kol_k)
    d.text((x0, y0 - 26), 'wspólna oś uczenia dla obu widoków (skala logarytmiczna)' + ('' if ucz else ' — przed uczeniem: pokaz w kroku końcowym'),
           font=FN(15), fill=(200, 205, 210, 255))
    d.text((22, HY - 52), 'Dane: MalyViT_35000_od_zera_s47, 806 checkpointów, zdjęcie TRAIN 0, token narzędzia 316, blok 6.  J dokładny w punkcie x (przejście modelu).',
           font=FN(16), fill=(205, 210, 215, 255))
    d.text((22, HY - 28), 'Rzuty do 3D to ilustracja (lewy: 3 główne składowe położeń, prawy: e₁ e₂ e₃ z 60. epoki); kolory i liczby z rachunku w 192D.',
           font=FN(16), fill=(205, 210, 215, 255))
    ii = S['ii']; st = STAT[ii]
    q = rzut(mvpL, np.zeros(3), 0)
    if q and t > 2:
        d.ellipse((q[0] - 26, q[1] - 26, q[0] + 26, q[1] + 26), outline=(255, 255, 255, 160), width=2)
        d.text((q[0] + 30, q[1] - 34), '→ zbliżenie', font=FN(18, True), fill=(255, 255, 255, 200))
    for p, s, k in ((OSIE[1] * 1.08, 'e₁: cofa', (120, 170, 255)), (OSIE[3] * 1.08, 'e₂: wypycha', (255, 130, 110)), (OSIE[5] * 1.08, 'e₃: wir', (210, 150, 255))):
        q = rzut(mvpR, p, WH)
        if q and WH + 10 < q[0] < WY - 10:
            d.text(q, s, font=FN(20, True), fill=k + (255,), anchor='mm')
    # panel lewy
    x0, y0 = 22, 140
    d.rectangle((x0 - 8, y0 - 6, x0 + 330, y0 + 140), fill=(5, 9, 12, 215), outline=(42, 51, 58, 255))
    for n, (s, k) in enumerate(((f'siła pola teraz: {S["sila"]:.0%} końcowej', (255, 225, 120)),
                                (f'narzędzie przyciągane: {STAT_P["nar"][ii]:.0%}', (120, 175, 255)),
                                (f'tło przyciągane: {STAT_P["tlo"][ii]:.0%}', (255, 170, 110)),
                                (f'IoU: {pl(float(iou[ii]), 2)}', (74, 240, 255)),
                                (f'powiększenie chmury: ×{pl(float(S["pow"]), 1)}', (190, 210, 230)))):
        d.text((x0, y0 + 26 * n), s, font=FN(18, True), fill=k + (255,))
    # panel prawy: wartości własne S
    x0, y0, w, h = WY - 330, 175, 300, 120
    d.rectangle((x0 - 8, y0 - 26, x0 + w + 8, y0 + h + 122), fill=(5, 9, 12, 215), outline=(42, 51, 58, 255))
    d.text((x0, y0 - 22), 'wartości własne S (192)', font=FN(15), fill=(220, 225, 230, 255))
    ev = st['ev']; m = max(np.abs(ev).max(), 1e-9); bw = w / len(ev)
    d.line((x0, y0 + h / 2, x0 + w, y0 + h / 2), fill=(70, 80, 90, 255))
    for n, e in enumerate(ev):
        y1 = y0 + h / 2 - e / m * h / 2
        d.rectangle((x0 + n * bw, min(y1, y0 + h / 2), x0 + (n + 1) * bw - 0.3, max(y1, y0 + h / 2)), fill=(90, 150, 255, 255) if e < 0 else (255, 110, 90, 255))
    for n, (s, k) in enumerate(((f'cofające kierunki: {st["ujemne"]:.0%}', (120, 170, 255)),
                                (f'średnia λ (ślad J / 192): {st["slad"]:+.3f}'.replace('.', ','), (200, 230, 200)),
                                (f'symetryczność ‖S‖²/‖J‖²: {st["sym"]:.0%}', (255, 200, 120)),
                                (f'siła pola względem kroku 0: ×{pl(st["norma"] / NORMA0, 1)}', (160, 220, 255)))):
        d.text((x0, y0 + h + 8 + 26 * n), s, font=FN(17, True), fill=k + (255,))
    a = 1.0 if faza == 'koniec' else 0.0
    if a > 0:
        x0, y0 = WH + 40, HY - 360
        d.rectangle((x0 - 12, y0 - 36, x0 + 600, y0 + 210), fill=(5, 9, 12, 230), outline=(42, 51, 58, 255))
        d.text((x0, y0 - 30), 'wir / siodło ‖A‖/‖S‖ (0 = pole zachowawcze)', font=FN(17, True), fill=(230, 235, 240, 255))
        for n, (s, lo, hi, k) in enumerate((('macierz losowa', 1.0, 1.0, (150, 150, 150)), ('MałyViT, blok 6', st['rot'], st['rot'], (255, 200, 120)),
                                            ('autoenkodery (Im 2016)', 0.10, 0.23, (140, 220, 160)), ('pęseta optyczna (Pesce 2009)', 0.01, 0.05, (120, 180, 255)))):
            y = y0 + 6 + n * 48; bx = x0 + 250; L = 220
            d.text((x0, y), s, font=FN(16), fill=k + (255,))
            d.rectangle((bx, y + 2, bx + L, y + 20), outline=(60, 70, 80, 255))
            d.rectangle((bx, y + 2, bx + L * min(hi, 1.0), y + 20), fill=k + (140,))
            d.rectangle((bx, y + 2, bx + L * min(lo, 1.0), y + 20), fill=k + (255,))
            d.text((bx + L + 8, y + 1), pl(lo, 2) if lo == hi else f'{pl(lo, 2)}–{pl(hi, 2)}', font=FN(15), fill=k + (255,))


def widok_lewy(t, S, mvp, skala):
    mb = mvp.T.astype('f4').tobytes()
    linie['mvp'].write(mb); kropki['mvp'].write(mb); kropki['skala'].value = skala
    z = np.clip(S['z'], 0, None); a = np.clip(z / max(np.percentile(z, 97), 1e-6), 0, 1)
    on = sm(t / 2.5) * (1 - 0.5 * sm((t - 8) / 2))
    p0 = -S['dir'] * 26.0
    kk = np.repeat(np.concatenate([0.3 + 0.7 * KOL, (a * 0.7 * on)[:, None]], 1), 2, 0); kk[0::2, 3] *= 0.1
    linie['szer'].value = 2.0
    rysuj(linie, moderngl.LINES, np.stack([p0, np.zeros_like(p0)], 1).reshape(-1, 3), kk)
    tk = sm((t - 8) / 2)
    if tk > 0:
        idx = np.arange(TOK) != FOK
        kol, si = barwa_r(S['R'])
        rysuj(kropki, moderngl.POINTS, S['D'][idx], np.concatenate([kol[idx] * (0.5 + 0.8 * si[idx]), np.full((idx.sum(), 1), 0.9 * tk)], 1), np.full(int(idx.sum()), 0.45))
        sg, kk = strzalki(S['D'][idx], S['D'][idx] + S['DZ'][idx], np.concatenate([kol[idx], (0.25 + 0.6 * si[idx]) * tk], 1), 0.35)
        linie['szer'].value = 1.4
        rysuj(linie, moderngl.LINES, sg, kk)
    rysuj(kropki, moderngl.POINTS, np.zeros((1, 3)), np.array([[0.85, 1.0, 0.9, 2.5]]), np.array([1.6]))


def widok_prawy(t, S, mvp, skala):
    mb = mvp.T.astype('f4').tobytes()
    linie['mvp'].write(mb); kropki['mvp'].write(mb); kropki['skala'].value = skala
    J = S['J']
    linie['szer'].value = 2.5
    rysuj(linie, moderngl.LINES, OSIE, np.concatenate([KOL_OSI, np.full((6, 1), 0.7)], 1))
    Fv = SIAT @ J.T
    rad = (Fv * SIAT).sum(1) / (np.linalg.norm(SIAT, axis=1) * np.linalg.norm(Fv, axis=1) + 1e-9)
    kol = np.where(rad[:, None] < 0, np.array([[0.35, 0.6, 1.0]]), np.array([[1.0, 0.4, 0.3]])) * (0.4 + 0.6 * np.abs(rad))[:, None]
    L = 0.12 * R0 * Fv / np.maximum(np.linalg.norm(Fv, axis=1, keepdims=True), 1e-9) * np.clip(np.linalg.norm(Fv, axis=1, keepdims=True) / R0, 0.3, 1.0)
    L = L * (0.25 + 0.75 * min(S['sila'], 1.2))                                   # długość strzałek rośnie z siłą pola w tym kroku
    a_str = 0.55 * (1 - 0.45 * sm((t - 8) / 2))
    linie['szer'].value = 2.0
    rysuj(linie, moderngl.LINES, np.stack([SIAT, SIAT + L], 1).reshape(-1, 3), np.repeat(np.concatenate([kol, np.full((len(kol), 1), a_str)], 1), 2, 0))
    if t >= 8:
        seg = np.stack([SLAD[:-1], SLAD[1:]], 2).reshape(-1, 3)
        kc = np.where((SLAD[0] * (SLAD[0] - SLAD[1])).sum(1)[:, None] < 0, np.array([[0.4, 0.65, 1.0]]), np.array([[1.0, 0.5, 0.35]]))
        al = np.repeat(np.linspace(0.6, 0.05, HIST - 1), NP_)
        kk = np.concatenate([np.tile(kc, (HIST - 1, 1)), al[:, None] * sm((t - 8) / 1.5)], 1)
        linie['szer'].value = 1.6
        rysuj(linie, moderngl.LINES, seg, np.repeat(kk, 2, 0))
        rysuj(kropki, moderngl.POINTS, SLAD[0], np.concatenate([kc * 1.2, np.full((NP_, 1), 0.9)], 1), np.full(NP_, 0.25))
    rysuj(kropki, moderngl.POINTS, np.zeros((1, 3)), np.array([[0.75, 1.0, 0.8, 2.5]]), np.array([2.2]))


def klatka(t):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    S = stan(t)
    if t >= 8:
        krok_czastek(S['J'], 1.6 / FPS * max(S['sila'], 0.12))
    mvpL, skL = kamera(-60 + 7 * t, 20 + 6 * np.sin(t / 8), 44.0)
    mvpR, skR = kamera(-50 + 8 * t, 22 + 8 * np.sin(t / 9), 36.0)
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    ctx.viewport = (0, 0, WH * SS, H4); widok_lewy(t, S, mvpL, skL)
    ctx.viewport = (WH * SS, 0, WH * SS, H4); widok_prawy(t, S, mvpR, skR)
    fb_ldr.use(); ctx.viewport = (0, 0, W4, H4); ctx.disable(moderngl.BLEND); hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    tk = sm((t - 8) / 2)
    if tk > 0:
        ctx.viewport = (0, 0, WH * SS, H4)
        kropki['mvp'].write(mvpL.T.astype('f4').tobytes()); kropki['skala'].value = skL; kropki['ksztalt'].value = 1
        nz = narz & (np.arange(TOK) != FOK)
        rysuj(kropki, moderngl.POINTS, S['D'][nz], np.tile([0.29, 0.94, 1.0, 0.85 * tk], (int(nz.sum()), 1)), np.full(int(nz.sum()), 1.2))
        kropki['ksztalt'].value = 0
    ctx.viewport = (0, 0, W4, H4)
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, faza, t, S, mvpL, mvpR)
    return img


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    zapisz = {4.0, 12.0, 19.0, 25.0, 40.0, round(T_KON + 3, 1)}
    for t in CZASY:
        if t < 8:
            if round(t, 2) in zapisz:
                klatka(float(t)).save(HERE / f'lasery_19_proba_{t:04.1f}.png')
            continue
        if any(abs(t - z) < 0.5 / FPS for z in zapisz):
            klatka(float(t)).save(HERE / f'lasery_19_proba_{t:04.1f}.png')
        else:
            S_ = stan(float(t)); krok_czastek(S_['J'], 1.6 / FPS * max(S_['sila'], 0.12))
else:
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WY}x{HY}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '20', '-pix_fmt', 'yuv420p', str(HERE / 'lasery_19_pulapka_dwa_widoki_sync.mp4')],
                          stdin=subprocess.PIPE)
    for n, t in enumerate(CZASY):
        ff.stdin.write(klatka(float(t)).tobytes())
        if n % 300 == 0:
            log(f'klatka {n}/{len(CZASY)}')
    ff.stdin.close(); ff.wait()
log('gotowe')
