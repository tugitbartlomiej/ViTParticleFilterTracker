"""Film 16: „pułapka” bloku MLP wokół jednej cząsteczki = siodło z wirem (nowy film; pomiar w pulapka_jakobian.py, opis w
matematyka_reflektorow/matematyka_reflektorow.pdf, pkt 4).

Lokalne pole sił 768 wiązek ostatniego bloku wokół tokenu narzędzia 316: F(x + δ) ≈ F(x) + J δ,
    J = W₂ diag(GELU'(a)) W₁ J_LN(x),  J = S + A  (S symetryczna: cofanie/wypychanie, A antysymetryczna: wir).
Pokazujemy J w 3 kierunkach z kroku 22 974: e₁ = wektor własny S o najmniejszej wartości (cofa), e₂ = o największej (wypycha),
e₃ = kierunek, w który wir A najmocniej wyprowadza ruch z płaszczyzny e₁e₂. J₃ = Qᵀ J Q (3 × 3) w każdym z 341 checkpointów.
Cząstki próbne płyną według dδ/dt = J₃ δ (prędkość unormowana normą J₃; licznik pokazuje rzeczywisty wzrost siły pola).
Fazy: pole strzałek; przepływ; tylko S (czyste siodło); tylko A (czysty wir); razem; uczenie 0 → 22 974; porównanie z pęsetą optyczną.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_16_pulapka_siodlo.py [proba]
"""
import subprocess
import sys
import time

import moderngl
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.special import erf

from lasery_wspolne import EP, HERE, TOK, dane

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
FPS, WY, HY, SS = 30, 1920, 1080, 2
ASSETS = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10'
WAGI = ASSETS / '01_bieg_35000_gesty_zapis/eden/gesty/wagi'
LATENT = ASSETS / '03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
CACHE = HERE / 'pulapka_16_cache.npz'

# ---------------- dane: J dla tokenu narzędzia w 341 checkpointach ----------------
D = dane(0)
kroki_all, narz = D['kroki'].astype(int), D['narz']
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
gelu = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
dgelu = lambda a: 0.5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
_A = np.linspace(-0.75, 12, 40000); _Z = gelu(_A)


def odtworz_u(z, W1, b1):
    m = z > -0.12
    u, *_ = np.linalg.lstsq(W1[m], np.interp(z[m], _Z, _A) - b1[m], rcond=None)
    for _ in range(12):
        a = W1 @ u + b1
        Jg = dgelu(a)[:, None] * W1
        u -= np.linalg.solve(Jg.T @ Jg + 1e-3 * np.eye(len(u)), Jg.T @ (gelu(a) - z))
    return u


if CACHE.exists():
    c_ = np.load(CACHE); KW, JJ = c_['KW'], c_['JJ']
else:
    import os
    KW = np.array(sorted(int(f[1:6]) for f in os.listdir(WAGI) if f.endswith('.pt')))
    idx = np.searchsorted(kroki_all, KW)
    Zf = np.asarray(np.load(LATENT, mmap_mode='r')[idx, 0, FOK]).astype(np.float64)   # (341, 768)
    JJ = []
    for n, k in enumerate(KW):
        s = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
        g = lambda nm: s[f'bloki.5.{nm}'].double().numpy()
        W1, b1, W2, gam, bet = g('mlp.fc1.weight'), g('mlp.fc1.bias'), g('mlp.fc2.weight'), g('norm2.weight'), g('norm2.bias')
        u = odtworz_u(Zf[n], W1, b1); a = W1 @ u + b1
        xh = (u - bet) / gam; xh -= xh.mean(); nn_ = xh / np.linalg.norm(xh); d = len(u)
        Pln = np.eye(d) - np.ones((d, d)) / d - np.outer(nn_, nn_)
        JJ.append(Pln @ W2 @ (dgelu(a)[:, None] * W1) @ (gam[:, None] * Pln))   # obie strony w podprzestrzeni widzianej przez LN
    JJ = np.array(JJ, np.float32)
    np.savez_compressed(CACHE, KW=KW, JJ=JJ)
NT = len(KW)
log('Jakobiany', JJ.shape, 'token', FOK)

# baza 3D z końca uczenia
Je = JJ[-1].astype(np.float64); Se, Ae = (Je + Je.T) / 2, (Je - Je.T) / 2
w_, V_ = np.linalg.eigh(Se)
e1, e2 = V_[:, 0], V_[:, -1]
P12 = np.outer(e1, e1) + np.outer(e2, e2)
U3, _, _ = np.linalg.svd((np.eye(len(e1)) - P12) @ Ae @ np.stack([e1, e2], 1), full_matrices=False)
e3 = U3[:, 0]
Q = np.stack([e1, e2, e3], 1)
J3 = np.einsum('ai,tij,jb->tab', Q.T, JJ.astype(np.float64), Q)                    # (341, 3, 3)


def statystyki(J):
    S, A = (J + J.T) / 2, (J - J.T) / 2
    ev = np.linalg.eigvalsh(S)
    nz = np.abs(ev) > 1e-6 * np.abs(ev).max()                                          # bez dwóch kierunków wyciętych przez LN
    return dict(ujemne=float((ev[nz] < 0).mean()), slad=float(np.trace(J) / nz.sum()), sym=float(np.linalg.norm(S) ** 2 / np.linalg.norm(J) ** 2),
                rot=float(np.linalg.norm(A) / np.linalg.norm(S)), ev=np.sort(ev[nz]), norma=float(np.linalg.norm(J, 2)))


STAT = [statystyki(JJ[i].astype(np.float64)) for i in range(NT)]
NORMA0 = STAT[0]['norma']
log('koniec:', {k: round(v, 3) for k, v in STAT[-1].items() if k != 'ev'})

# ---------------- oś czasu ----------------
FAZY = [(0, 'pole'), (6, 'przeplyw'), (17, 'S'), (23, 'A'), (29, 'razem'), (33, 'uczenie'), (54, 'porownanie'), (60, None)]
T_U0, T_U1 = 34.0, 53.0
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def indeks(t):
    if t < T_U0:
        return float(NT - 1) if t < 33 else float(NT - 1) * (1 - sm((t - 33) / 1.0))
    u = np.clip((t - T_U0) / (T_U1 - T_U0), 0, 1)
    kr = KW[-1] * u ** 2.2                                                          # pierwsze kroki wolno, epoki szybko
    return float(np.interp(kr, KW, np.arange(NT)))


def macierz_J(t):
    fi = indeks(t); i = int(min(fi, NT - 2)); u = fi - i
    J = J3[i] * (1 - u) + J3[i + 1] * u
    S, A = (J + J.T) / 2, (J - J.T) / 2
    if 17 <= t < 23:
        J = S * sm((t - 17) / 1.0) + J * (1 - sm((t - 17) / 1.0))
    elif 23 <= t < 29:
        J = A * sm((t - 23) / 1.0) + S * (1 - sm((t - 23) / 1.0))
    elif 29 <= t < 31:
        J = J * sm((t - 29) / 1.5) + A * (1 - sm((t - 29) / 1.5))
    return J / max(np.linalg.norm(J, 2), 1e-9), i if u < 0.5 else i + 1


# ---------------- kamera ----------------
R0 = 10.0


def mvp_kamery(t):
    az, el, di = np.radians(-50 + 8 * t), np.radians(22 + 8 * np.sin(t / 9)), 34.0
    oko = di * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    f = -oko / np.linalg.norm(oko); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    V = np.eye(4); V[0, :3], V[1, :3], V[2, :3] = s, up, -f; V[:3, 3] = -V[:3, :3] @ oko
    fov, asp, n, fa = np.radians(38), WY / HY, 0.1, 300.0
    P = np.zeros((4, 4)); P[0, 0] = 1 / np.tan(fov / 2) / asp; P[1, 1] = 1 / np.tan(fov / 2)
    P[2, 2] = (fa + n) / (n - fa); P[2, 3] = 2 * fa * n / (n - fa); P[3, 2] = -1
    return P @ V, 1 / np.tan(fov / 2) * HY * SS / 2


def rzut(mvp, p):
    q = mvp @ np.append(p, 1.0)
    return None if q[3] <= 0.1 else ((q[0] / q[3] * 0.5 + 0.5) * WY, (0.5 - q[1] / q[3] * 0.5) * HY)


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
in vec4 v_k; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard; float a = exp(-r2 * 3.0);
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
linie['vp'].value = (W4, H4)
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


# siatka strzałek w kuli
g_ = np.linspace(-1, 1, 7)
SIAT = np.array([[x, y, z] for x in g_ for y in g_ for z in g_ if 0.25 < x * x + y * y + z * z <= 1.0]) * R0
OSIE = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float) * R0 * 1.25
KOL_OSI = np.array([[0.35, 0.6, 1.0], [0.35, 0.6, 1.0], [1.0, 0.4, 0.3], [1.0, 0.4, 0.3], [0.8, 0.5, 1.0], [0.8, 0.5, 1.0]])

# cząstki próbne
rng = np.random.default_rng(1)
NP_, HIST = 650, 26


def losuj(n):
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * R0 * rng.uniform(0.15, 0.9, (n, 1)) ** (1 / 3)


TR = losuj(NP_); WIEK = rng.uniform(0, 3, NP_)
SLAD = np.repeat(TR[None], HIST, 0)

# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
TEKSTY = {
    'pole': ('Pułapka bloku MLP wokół jednej cząsteczki: lokalne pole 768 wiązek',
             ['F(x + δ) ≈ F(x) + J δ,   J = W₂ diag(GELU′) W₁ J_LN.   Strzałki: niebieskie = siła do środka, czerwone = od środka.',
              'Trzy kierunki z Jakobianu (krok 22 974): e₁ najmocniej cofa, e₂ najmocniej wypycha, e₃ = płaszczyzna wiru.']),
    'przeplyw': ('Cząstki próbne płyną w tym polu: siodło z wirem',
                 ['Wzdłuż e₁ wracają do cząsteczki, wzdłuż e₂ uciekają, a część antysymetryczna J skręca ich tory.',
                  'To nie jest studnia (wszystkie kierunki do środka), tylko siodło: połowa kierunków cofa, połowa wypycha.']),
    'S': ('Tylko część symetryczna S = (J + Jᵀ)/2: czyste siodło',
          ['S to „sztywność” pułapki: ujemne wartości własne cofają, dodatnie wypychają.  Bez wiru tory są proste.', '']),
    'A': ('Tylko część antysymetryczna A = (J − Jᵀ)/2: czysty wir',
          ['A nie zbliża ani nie oddala, tylko obraca.  Siła z wirem nie jest gradientem żadnego potencjału (pole niezachowawcze).', '']),
    'razem': ('Razem: siodło + wir', ['', '']),
    'uczenie': ('Uczenie: pole rośnie, pojawia się słaba kontrakcja, wir zostaje',
                ['Prędkość na ekranie jest unormowana; licznik pokazuje rzeczywisty wzrost siły pola względem kroku 0.',
                 'Pierwsze kroki wolno, potem epoki szybko.  Panel z lewej: wartości własne S (niebieskie < 0 cofają, czerwone > 0 wypychają).']),
    'porownanie': ('Jak bardzo „kręci” ta pułapka?  Porównanie z prawdziwymi pułapkami',
                   ['Pęseta optyczna: część rotacyjna 1–4 % sztywności (Pesce i in. 2009).  Wyuczone autoenkodery: pole w 95–99 % symetryczne (Im i in. 2016).',
                    'Blok MałegoViT: wir prawie tak silny jak siodło — bliżej „fontanny prawdopodobieństwa” (Roichman i in. 2008) niż klasycznej pułapki.']),
}


def nakladka(img, faza, t, ii, mvp):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    d.text((WY / 2, 24), tyt, font=FN(34, True), fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        if s:
            d.text((WY / 2, 76 + 30 * n), s, font=FN(21), fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), f'Dane: blok 6 MałegoViT, token narzędzia {FOK}, 341 checkpointów (kroki 0–22 974). J liczone w podprzestrzeni widzianej przez LayerNorm, skala σ = 1.',
           font=FN(17), fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Rzut 190D → 3D na kierunki e₁, e₂, e₃ z końca uczenia (ilustracja); cząstki próbne płyną według dδ/dt = J₃ δ, prędkość unormowana.',
           font=FN(17), fill=(205, 210, 215, 255))
    for p, s, k in ((OSIE[1] * 1.08, 'e₁: cofa', (120, 170, 255)), (OSIE[3] * 1.08, 'e₂: wypycha', (255, 130, 110)), (OSIE[5] * 1.08, 'e₃: wir', (210, 150, 255))):
        q = rzut(mvp, p)
        if q:
            d.text(q, s, font=FN(22, True), fill=k + (255,), anchor='mm')
    st = STAT[ii]
    # panel wartości własnych S
    x0, y0, w, h = 30, 190, 420, 230
    d.rectangle((x0 - 10, y0 - 34, x0 + w + 10, y0 + h + 120), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
    d.text((x0, y0 - 30), 'wartości własne S (190, posortowane)', font=FN(16), fill=(220, 225, 230, 255))
    ev = st['ev']; m = max(np.abs(ev).max(), 1e-9); bw = w / len(ev)
    d.line((x0, y0 + h / 2, x0 + w, y0 + h / 2), fill=(70, 80, 90, 255))
    for n, e in enumerate(ev):
        y1 = y0 + h / 2 - e / m * h / 2
        d.rectangle((x0 + n * bw, min(y1, y0 + h / 2), x0 + (n + 1) * bw - 0.5, max(y1, y0 + h / 2)), fill=(90, 150, 255, 255) if e < 0 else (255, 110, 90, 255))
    for n, (s, k) in enumerate(((f'krok {KW[ii]:,}'.replace(',', ' ') + f'   (epoka {pl(KW[ii] / EP, 2)})', (235, 240, 245)),
                                (f'cofające kierunki (λ < 0): {st["ujemne"]:.0%}', (120, 170, 255)),
                                (f'średnia λ (ślad J): {st["slad"]:+.3f}'.replace('.', ','), (200, 230, 200)),
                                (f'symetryczność ‖S‖²/‖J‖²: {st["sym"]:.0%}', (255, 200, 120)),
                                (f'siła pola względem kroku 0: ×{pl(st["norma"] / NORMA0, 1)}', (160, 220, 255)))):
        d.text((x0, y0 + h + 8 + 22 * n), s, font=FN(18, True), fill=k + (255,))
    # porównanie: stosunek wir / siodło
    a = 1.0 if faza == 'porownanie' else (0.55 if faza in ('uczenie',) else 0.0)
    if a > 0:
        x0, y0 = WY - 640, HY - 330
        d.rectangle((x0 - 12, y0 - 40, WY - 26, y0 + 230), fill=(5, 9, 12, int(230 * a)), outline=(42, 51, 58, int(255 * a)))
        d.text((x0, y0 - 34), 'wir / siodło  ‖A‖ / ‖S‖  (0 = pole zachowawcze)', font=FN(17, True), fill=(230, 235, 240, int(255 * a)))
        wiersze = [('macierz losowa', 1.0, 1.0, (150, 150, 150)), ('MałyViT, blok 6 (teraz)', st['rot'], st['rot'], (255, 200, 120)),
                   ('autoenkodery (Im 2016)', 0.10, 0.23, (140, 220, 160)), ('pęseta optyczna (Pesce 2009)', 0.01, 0.05, (120, 180, 255))]
        for n, (s, lo, hi, k) in enumerate(wiersze):
            y = y0 + 8 + n * 52
            d.text((x0, y), s, font=FN(17), fill=k + (int(255 * a),))
            bx = x0 + 250; L = 230
            d.rectangle((bx, y + 2, bx + L, y + 22), outline=(60, 70, 80, int(255 * a)))
            d.rectangle((bx, y + 2, bx + L * min(hi, 1.0), y + 22), fill=k + (int(140 * a),))
            d.rectangle((bx, y + 2, bx + L * min(lo, 1.0), y + 22), fill=k + (int(255 * a),))
            d.text((bx + L + 8, y + 2), (pl(lo, 2) if lo == hi else f'{pl(lo, 2)}–{pl(hi, 2)}'), font=FN(16), fill=k + (int(255 * a),))


def krok_czastek(J, dt):
    global TR, WIEK, SLAD
    v = TR @ J.T
    TR = TR + v * dt + rng.normal(scale=0.002 * R0, size=TR.shape) * np.sqrt(dt * 30)    # odrobina szumu „termicznego” (ilustracja)
    WIEK += dt
    r = np.linalg.norm(TR, axis=1)
    nowe = (r > 1.25 * R0) | (r < 0.05 * R0) | (WIEK > 4.0)
    if nowe.any():
        TR[nowe] = losuj(int(nowe.sum())); WIEK[nowe] = 0.0
    SLAD = np.roll(SLAD, 1, 0); SLAD[0] = TR
    SLAD[:, nowe] = TR[nowe]


def klatka(t, dt):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    J, ii = macierz_J(t)
    if t >= 6:
        krok_czastek(J, dt * 1.4)
    mvp, skala = mvp_kamery(t)
    mb = mvp.T.astype('f4').tobytes()
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mb); kropki['mvp'].write(mb); kropki['skala'].value = skala
    linie['szer'].value = 2.5
    rysuj(linie, moderngl.LINES, OSIE, np.concatenate([KOL_OSI, np.full((6, 1), 0.7)], 1))
    # strzałki pola
    F = SIAT @ J.T
    rad = (F * SIAT).sum(1) / (np.linalg.norm(SIAT, axis=1) * np.linalg.norm(F, axis=1) + 1e-9)
    kol = np.where(rad[:, None] < 0, np.array([[0.35, 0.6, 1.0]]), np.array([[1.0, 0.4, 0.3]])) * (0.4 + 0.6 * np.abs(rad))[:, None]
    L = 0.12 * R0 * F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9) * np.clip(np.linalg.norm(F, axis=1, keepdims=True) / R0, 0.3, 1.0)
    a_str = 0.55 * (1 - 0.45 * sm((t - 6) / 2))
    seg = np.stack([SIAT, SIAT + L], 1).reshape(-1, 3)
    linie['szer'].value = 2.0
    rysuj(linie, moderngl.LINES, seg, np.repeat(np.concatenate([kol, np.full((len(kol), 1), a_str)], 1), 2, 0))
    # ślady cząstek
    if t >= 6:
        sp = np.linalg.norm(SLAD[0] - SLAD[1], axis=1)
        seg = np.stack([SLAD[:-1], SLAD[1:]], 2).reshape(-1, 3)
        r = np.linalg.norm(SLAD[0], axis=1) / R0
        kc = np.where((SLAD[0] * (SLAD[0] - SLAD[1])).sum(1)[:, None] < 0, np.array([[0.4, 0.65, 1.0]]), np.array([[1.0, 0.5, 0.35]]))
        al = np.repeat(np.linspace(0.6, 0.05, HIST - 1), NP_)                       # kolejność odcinków: historia, potem cząstka
        kk = np.concatenate([np.tile(kc, (HIST - 1, 1)), al[:, None] * sm((t - 6) / 1.5)], 1)
        linie['szer'].value = 1.6
        rysuj(linie, moderngl.LINES, seg, np.repeat(kk, 2, 0))
        rysuj(kropki, moderngl.POINTS, SLAD[0], np.concatenate([kc * 1.2, np.full((NP_, 1), 0.9)], 1), np.full(NP_, 0.25))
    rysuj(kropki, moderngl.POINTS, np.zeros((1, 3)), np.array([[0.75, 1.0, 0.8, 2.5]]), np.array([2.2]))
    fb_ldr.use(); ctx.disable(moderngl.BLEND); hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, faza, t, ii, mvp)
    return img


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    for t in np.arange(0, 47.0, 1 / FPS):                                         # rozbieg cząstek do klatek próbnych
        if t in (4.0,) or abs(t - 12) < 1e-6 or abs(t - 20) < 1e-6 or abs(t - 26) < 1e-6 or abs(t - 46) < 1e-6:
            klatka(t, 1 / FPS).save(HERE / f'lasery_16_proba_{t:04.1f}.png')
        else:
            if t >= 6:
                krok_czastek(macierz_J(t)[0], 1.4 / FPS)
    klatka(57.0, 1 / FPS).save(HERE / 'lasery_16_proba_57.0.png')
else:
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WY}x{HY}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '20', '-pix_fmt', 'yuv420p', str(HERE / 'lasery_16_pulapka_siodlo.mp4')],
                          stdin=subprocess.PIPE)
    for n, t in enumerate(CZASY):
        ff.stdin.write(klatka(float(t), 1 / FPS).tobytes())
        if n % 300 == 0:
            log(f'klatka {n}/{len(CZASY)}')
    ff.stdin.close(); ff.wait()
log('gotowe')
