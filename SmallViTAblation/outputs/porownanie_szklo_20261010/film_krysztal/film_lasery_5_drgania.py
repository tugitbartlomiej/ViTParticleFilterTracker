"""Film 5: co trzęsie cząsteczkami. Ta sama sfera 768 laserów i 600 cząsteczek co w filmie 4, plus GENERATOR nad sferą = wektor wag θ,
jedyny element układu z własnym szumem. Każdy krok uczenia: paczka 32 klatek (prawdziwe miniatury z PACZKI.npz) wpada do generatora, generator
błyska z siłą = długość kroku wag |δθ|, impuls biegnie 768 światłowodami do laserów, lasery migoczą (prawdziwa zmiana natężeń 768 neuronów),
ogniska skaczą (prawdziwe przesunięcie tokenów w tym kroku). Temperatura = długość kroku: w rozgrzewce lr krok wag rośnie i drgania rosną.

Dane: gęsty zapis biegu 35 000 (ziarno 47, osobny bieg z obserwatorem gęstym), sonda 0 = TR_0001_S1_P03 klatka 838, pełny latent 768D co krok
dla kroków 0–1094 (drgania_prep.py → drgania_X.npy, drgania.npz). Rzut 768 → 3: trzy główne kierunki PRZYROSTÓW na krok (kroki 300–1094),
bez odejmowania średniej zdjęcia (ruch wspólny zostaje widoczny), 1 jednostka = odstęp sąsiadów w kroku 0 w 768D.
Ilustracja: generator, światłowody, iskry paczki, sfera, kolory.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_5_drgania.py <drgania_X.npy> [proba]"""
import sys
import time

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from lasery_wspolne import H, HERE, TOK, W, ffmpeg

PROBA = len(sys.argv) > 2 and sys.argv[2] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
NK, FPS, WY, HY, SS = 768, 30, 1920, 1080, 2
X = np.load(sys.argv[1]).astype(np.float16)                                         # (1095, 600, 768)
dd = np.load(HERE / 'drgania.npz')
dth, lr, strata, WOLNE, mini, filmy, zdj = dd['dth'], dd['lr'], dd['strata'], dd['wolne'], dd['mini'], dd['filmy'], dd['zdjecie']
bb = dd['bbox480']; NS = X.shape[0] - 1
log('wczytano', X.shape)

# ---------------- analiza (768D) ----------------
x0 = X[0].astype(np.float32)
Dm = cdist(x0, x0); np.fill_diagonal(Dm, np.inf); a0 = float(np.median(np.sort(Dm, 1)[:, :4].mean(1)))
Ew, Es, Eo, Et, ZM = (np.zeros(NS) for _ in range(5))
kl = np.zeros(NS + 1)
proj_dl = []
for c in range(0, NK, 96):                                                           # porcjami po neuronach (pamięć)
    x = X[:, :, c:c + 96].astype(np.float32); dl = np.diff(x, axis=0)
    wsp = dl.mean(1, keepdims=True); r = (dl - wsp).reshape(NS, H, W, -1)
    pg = np.pad(r, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')
    smo = sum(pg[:, 1 + a:1 + a + H, 1 + b:1 + b + W] for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9
    Et += (dl ** 2).sum((1, 2)); Ew += 600 * (wsp ** 2).sum((1, 2)); Es += (smo ** 2).sum((1, 2, 3)); Eo += ((r - smo) ** 2).sum((1, 2, 3))
    ZM += np.abs(dl).sum((1, 2))
    kl += ((x - uniform_filter1d(x, 21, axis=0, mode='nearest')) ** 2).sum((1, 2))
kl = np.sqrt(kl / TOK) / a0                                                          # klatka: RMS odchylenia od średniej ±10 kroków
UDZ = np.stack([Ew, Es, Eo]) / Et                                                    # udziały energii kroku
krok_dl = np.sqrt(Et / TOK) / a0
from scipy.stats import spearmanr  # noqa: E402
RHO = spearmanr(dth, krok_dl)[0]
log(f'a0 {a0:.3f}; udziały (mediana 300+) {np.median(UDZ[:, 300:], 1).round(2)}; Spearman |δθ| vs |Δx| {RHO:.2f}')
# rzut: 3 główne kierunki przyrostów
Dl_s = np.diff(X[300::1, ::3].astype(np.float32), axis=0).reshape(-1, NK)
_, sv, Vt = np.linalg.svd(Dl_s[::5], full_matrices=False); D = Vt[:3]; UDZ_RZUT = float((sv[:3] ** 2).sum() / (sv ** 2).sum())
siatka = np.stack([np.arange(TOK) % W, np.arange(TOK) // W], 1).astype(float)
g3 = np.concatenate([siatka - siatka.mean(0), np.zeros((TOK, 1))], 1); g3[:, 1] *= -1
P = np.empty((NS + 1, TOK, 3), np.float32)
for t in range(NS + 1):
    q = (X[t].astype(np.float32) - x0) @ D.T / a0
    P[t] = g3 + q[:, [1, 2, 0]] * np.array([1, -1, 1])
SKALA = 2.0
cx = (siatka[:, 0] + 0.5) * 16; cy = (siatka[:, 1] + 0.5) * 16
narz = (cx >= bb[0]) & (cx <= bb[2]) & (cy >= bb[1]) & (cy <= bb[3])
log(f'rzut niesie {UDZ_RZUT:.0%} energii kroku; krok w rzucie med {np.median(np.linalg.norm(np.diff(P[300:], axis=0), axis=2)):.3f}')

R = 34.0
i_ = np.arange(NK) + 0.5
fi, th = np.arccos(1 - 2 * i_ / NK), np.pi * (1 + 5 ** 0.5) * i_
sfera = np.stack([np.cos(th) * np.sin(fi), np.sin(th) * np.sin(fi), np.cos(fi)], 1)
xk = X[-1].astype(np.float32); w = np.maximum(xk - xk.mean(0), 0)
c_ = (w.T @ g3[:, :2]) / np.maximum(w.sum(0)[:, None], 1e-9) / np.array([15.0, 10.0]) * 0.9
rz, kol = linear_sum_assignment(((sfera[:, None, :2] - c_[None]) ** 2).sum(-1))
LAS = np.zeros((NK, 3)); LAS[kol] = sfera[rz] * R
GEN = np.array([0.0, 0.0, R + 20.0])
ZRODLO = np.array([-62.0, 0.0, R + 26.0])                                           # skąd wlatują paczki


# ---------------- oś czasu ----------------
T_A, T_B, T_C, T_D, T_E = 10.0, 40.0, 72.0, 84.0, 84.0
KR_WOLNY = 2.0
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def stan(t):
    """Zwraca: faza, ciągły krok s (położenia/natężenia), numer przejścia tw, faza wewnątrz kroku u (wolna część), puls generatora."""
    if t < T_A:
        return 'wstep', 300.0, None, None
    if t < T_B:
        n = int((t - T_A) // KR_WOLNY); u = ((t - T_A) % KR_WOLNY) / KR_WOLNY; tw = int(WOLNE[n])
        return 'wolno', tw + sm((u - 0.6) / 0.25), (n, tw), u
    if t < T_C:
        return 'szybko', min(NS, NS * (t - T_B) / (T_C - T_B - 2.0)), None, None
    return 'koniec', float(NS), None, None


KAM = [(0, -70, 22, 178), (8, -55, 22, 170), (T_A + 8, -35, 24, 165), (T_A + 16, -10, 34, 66), (T_B - 1, 25, 38, 60),
       (T_B + 3, 30, 28, 105), (T_C, 230, 28, 105), (T_D, 260, 28, 160)]


def kamera(t):
    for (ta, *pa), (tb, *pb) in zip(KAM[:-1], KAM[1:]):
        if t <= tb:
            u = sm((t - ta) / (tb - ta)); break
    else:
        (ta, *pa), (tb, *pb), u = KAM[-2], KAM[-1], 1.0
    az, el, di = [pa[i] + (pb[i] - pa[i]) * u for i in range(3)]
    cel = np.array([0, 0, 16.0]) * np.clip((di - 66) / 90, 0, 1)                     # z daleka celujemy między siatkę a generator
    return az, el, di, cel


def macierz(t):
    az, el, di, cel_ = kamera(t)
    a, e = np.radians(az), np.radians(el)
    oko = cel_ + di * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = cel_ - oko; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); u = np.cross(s, f)
    V = np.eye(4); V[0, :3], V[1, :3], V[2, :3] = s, u, -f; V[:3, 3] = -V[:3, :3] @ oko
    fov, asp, n, fa = np.radians(38), WY / HY, 0.1, 800.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n) / (n - fa); Pm[2, 3] = 2 * fa * n / (n - fa); Pm[3, 2] = -1
    return Pm @ V, f, 1 / np.tan(fov / 2) * HY * SS / 2


def ekran(M, p):
    q = M @ np.append(p, 1.0)
    return None if q[3] <= 0 else ((q[0] / q[3] * 0.5 + 0.5) * WY, (0.5 - q[1] / q[3] * 0.5) * HY)


# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
W4, H4 = WY * SS, HY * SS
hdr = ctx.texture((W4, H4), 4, dtype='f2'); fb_hdr = ctx.framebuffer([hdr])
ldr = ctx.texture((W4, H4), 4); fb_ldr = ctx.framebuffer([ldr])
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
wiazki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform int tryb; uniform float gain; uniform vec3 srodek, gen;
layout(std430, binding=0) readonly buffer B0 { vec4 L[]; };
layout(std430, binding=1) readonly buffer B1 { vec4 P[]; };
layout(std430, binding=2) readonly buffer B2 { float I[]; };
layout(std430, binding=3) readonly buffer B3 { float M[]; };
out float v_t; out float v_i;
void main() {
    int id = gl_InstanceID; int k = tryb == 0 ? id / 600 : id; int j = id % 600;
    vec3 a = tryb == 2 ? gen : L[k].xyz;
    vec3 b = tryb == 0 ? P[j].xyz : (tryb == 1 ? srodek : L[k].xyz);
    float s = tryb == 0 ? I[j * 768 + k] : (tryb == 1 ? M[k] : 1.0 + L[k].w);
    v_i = s * gain; v_t = float(gl_VertexID);
    gl_Position = mvp * vec4(gl_VertexID == 0 ? a : b, 1.0);
}''', fragment_shader='''#version 430
uniform vec3 kolor; uniform int tryb; uniform float faza, puls; in float v_t; in float v_i; out vec4 o;
void main() {
  float r = tryb == 2 ? (0.25 + puls * 6.0 * exp(-pow((v_t - faza) / 0.045, 2.0))) : (0.10 + 0.90 * v_t * v_t);
  o = vec4(kolor * v_i * r, 1.0); }''')
kropki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform float skala;
in vec3 pos; in vec4 kol; in float rozm; out vec4 v_k;
void main() { gl_Position = mvp * vec4(pos, 1.0); gl_PointSize = max(1.5, rozm * skala / gl_Position.w); v_k = kol; }''',
                     fragment_shader='''#version 430
uniform int pierscien; in vec4 v_k; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard;
  float a = pierscien == 1 ? smoothstep(0.55, 0.75, sqrt(r2)) * (1.0 - smoothstep(0.85, 1.0, sqrt(r2))) : exp(-r2 * 4.0);
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
buf_L = ctx.buffer(reserve=NK * 16); buf_L.bind_to_storage_buffer(0)
buf_P = ctx.buffer(reserve=TOK * 16); buf_P.bind_to_storage_buffer(1)
buf_I = ctx.buffer(reserve=TOK * NK * 4); buf_I.bind_to_storage_buffer(2)
buf_M = ctx.buffer(reserve=NK * 4); buf_M.bind_to_storage_buffer(3)
vao_w = ctx.vertex_array(wiazki, [])


def rysuj(prog, poz, kol, rozm=None, tryb=moderngl.POINTS):
    cz = [poz.astype('f4'), kol.astype('f4')] + ([rozm.astype('f4')[:, None]] if rozm is not None else [])
    b = ctx.buffer(np.concatenate(cz, 1).tobytes())
    v = ctx.vertex_array(prog, [(b, '3f 4f 1f' if rozm is not None else '3f 4f', *(['pos', 'kol', 'rozm'] if rozm is not None else ['pos', 'kol']))])
    v.render(tryb); v.release(); b.release()


seg = []
for lon in np.radians(np.arange(0, 180, 15)):
    t_ = np.linspace(0, 2 * np.pi, 97); p = R * np.stack([np.cos(t_) * np.cos(lon), np.cos(t_) * np.sin(lon), np.sin(t_)], 1); seg += [p[:-1], p[1:]]
for lat in np.radians(np.arange(-75, 90, 15)):
    t_ = np.linspace(0, 2 * np.pi, 97); p = R * np.stack([np.cos(t_) * np.cos(lat), np.sin(t_) * np.cos(lat), np.full_like(t_, np.sin(lat))], 1); seg += [p[:-1], p[1:]]
SF = np.stack([np.concatenate(seg[0::2]), np.concatenate(seg[1::2])], 1).reshape(-1, 3)

# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_S, F_M, F_P = FN(33, True), FN(21), FN(17), FN(15)
TEKSTY = {
    'wstep': ('Co trzęsie cząsteczkami? Generator nad sferą = wagi sieci θ',
              ['Lasery nie świecą same: natężenie każdego z 768 neuronów dla każdego tokenu liczą te same wspólne wagi θ.',
               'Wagi to jedyny element układu z własnym szumem. Pomarańczowe światłowody: z generatora do każdego lasera.']),
    'wolno': ('Jeden krok uczenia w zwolnionym tempie',
              ['Paczka 32 klatek (lewy panel, prawdziwe klatki tego kroku) → gradient → generator błyska z siłą |δθ| = długość kroku wag',
               '→ impuls biegnie do 768 laserów → natężenia się zmieniają → WSZYSTKIE ogniska skaczą naraz. Strzałki = skok × 4.']),
    'szybko': ('Cała 1. epoka: 1094 kroki w 30 s — rozgrzewka lr podgrzewa układ',
               ['Przez pierwsze 300 kroków lr rośnie, krok wag |δθ| rośnie 0,003 → 0,17 (termometr po prawej), a z nim drgania.',
                'Każda paczka jest inna, więc kierunek kroku skacze — to jest „szum termiczny” tego układu.']),
    'koniec': ('Temperatura = długość kroku wag, ale to nie jest ciepło', []),
}
KONIEC_TXT = None
UDZ_MED = np.median(UDZ[:, 300:], 1)
Q = lambda a, b: kl[a:b].mean()
pl = lambda v, n=2: f'{v:.{n}f}'.replace('.', ',')
KL_S = uniform_filter1d(kl, 9)


def pasek(d, x, y, sz, wys, war, kol, opis):
    d.rectangle((x, y, x + sz, y + wys), fill=(25, 30, 36, 230))
    d.rectangle((x, y, x + sz * float(np.clip(war, 0, 1)), y + wys), fill=kol)
    d.text((x + sz + 10, y + wys / 2), opis, font=F_M, fill=(225, 230, 235, 255), anchor='lm')


def nakladka(img, t, faza, s, wol, u, M):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    d.text((WY / 2, 24), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for n, l_ in enumerate(pod):
        d.text((WY / 2, 74 + 29 * n), l_, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 56), f'Dane: gęsty zapis biegu 35 000 (ziarno 47), zdjęcie TR_0001_S1_P03 klatka 838, latent 768D co krok; |δθ| i paczki z logu optymalizatora. '
           f'Rzut 768 → 3 = główne kierunki skoków ({UDZ_RZUT:.0%} energii kroku).', font=F_P, fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Ilustracja: generator, światłowody, iskry, sfera i kolory. Zielone = wiązki do cząsteczek, niebieskie = do środka masy (wspólnej części), '
           'błękitne pierścienie = narzędzie.', font=F_P, fill=(205, 210, 215, 255))
    pg = ekran(M, GEN)
    if pg and faza != 'koniec' and t > 4:
        d.text((pg[0] + 40, pg[1]), 'θ — wagi sieci (generator)', font=FN(20, True), fill=(255, 175, 80, int(255 * sm((t - 4) / 2))), anchor='lm')
    # miniatura zdjęcia
    if faza in ('wstep', 'szybko') or faza == 'wolno':
        im = Image.fromarray(zdj).resize((240, 160)); img.paste(im, (WY - 262, 165))
        d.rectangle((WY - 262 + bb[0] / 2, 165 + bb[1] / 2, WY - 262 + bb[2] / 2, 165 + bb[3] / 2), outline=(74, 240, 255, 255), width=2)
        d.text((WY - 262, 330), 'tokeny tego zdjęcia = cząsteczki', font=F_P, fill=(200, 205, 210, 255))
    if faza == 'wolno':
        n, tw = wol
        a_ = 0.35 + 0.65 * (1 - sm((u - 0.3) / 0.2))
        x0_, y0_ = 22, 175
        d.rectangle((x0_ - 8, y0_ - 34, x0_ + 8 * 76, y0_ + 4 * 52 + 6), fill=(5, 9, 12, 215), outline=(42, 51, 58, 255))
        d.text((x0_, y0_ - 30), f'paczka kroku {tw + 1}: 32 klatki z {len(set(filmy[n]))} filmów', font=F_M, fill=(235, 240, 245, 255))
        for q_ in range(32):
            m = Image.fromarray((mini[n][q_] * a_).astype(np.uint8)); img.paste(m, (x0_ + (q_ % 8) * 76, y0_ + (q_ // 8) * 52))
        # panel kroku
        x1, y1 = 22, 430
        d.rectangle((x1 - 8, y1 - 34, x1 + 700, y1 + 190), fill=(5, 9, 12, 215), outline=(42, 51, 58, 255))
        d.text((x1, y1 - 30), f'krok {tw + 1}:  |δθ| = {pl(dth[tw], 3)}   (lr {lr[tw]:.0e}, strata {pl(strata[tw])})', font=F_M, fill=(255, 175, 80, 255))
        pasek(d, x1, y1 + 4, 300, 20, dth[tw] / 0.25, (255, 160, 60, 255), 'długość kroku wag (temperatura)')
        pasek(d, x1, y1 + 34, 300, 20, krok_dl[tw] / 0.8, (120, 255, 150, 255), f'skok tokenów {pl(krok_dl[tw])} odstępu (768D, średnio)')
        d.text((x1, y1 + 70), 'z czego (energia skoku w tym kroku):', font=F_M, fill=(220, 225, 230, 255))
        pasek(d, x1, y1 + 96, 300, 18, UDZ[0, tw], (90, 160, 255, 255), f'wspólny dla całego zdjęcia {UDZ[0, tw]:.0%}')
        pasek(d, x1, y1 + 122, 300, 18, UDZ[1, tw], (120, 255, 150, 255), f'wspólny z sąsiadami (gładkie pole 3×3) {UDZ[1, tw]:.0%}')
        pasek(d, x1, y1 + 148, 300, 18, UDZ[2, tw], (255, 120, 160, 255), f'własny tokenu {UDZ[2, tw]:.0%}')
    if faza in ('szybko', 'koniec'):
        sc = int(min(NS - 1, s))
        d.text((WY - 30, 360), f'krok {sc + 1}', font=FN(28, True), fill=(235, 240, 245, 255), anchor='ra')
        # termometr
        tx, ty, th_ = WY - 80, 420, 330
        d.rounded_rectangle((tx, ty, tx + 30, ty + th_), radius=14, fill=(25, 30, 36, 230), outline=(80, 90, 100, 255))
        v = float(np.clip(uniform_filter1d(dth, 9)[sc] / 0.2, 0, 1))
        d.rounded_rectangle((tx + 5, ty + th_ - 5 - v * (th_ - 10), tx + 25, ty + th_ - 5), radius=10, fill=(255, 120, 50, 255))
        d.text((tx + 15, ty + th_ + 8), f'|δθ| {pl(dth[sc], 3)}', font=F_M, fill=(255, 175, 80, 255), anchor='mt')
        d.text((tx - 12, ty + 4), 'termometr =\ndługość kroku wag', font=F_M, fill=(255, 175, 80, 255), anchor='ra')
        # wykres: |δθ| i klatka
        x0_, y0_, sw, sh = 30, 560, 520, 200
        d.rectangle((x0_, y0_, x0_ + sw, y0_ + sh), fill=(11, 16, 20, 220), outline=(42, 51, 58, 255))
        kx = x0_ + np.arange(NS) / NS * sw
        d.line(list(zip(kx.tolist(), (y0_ + sh - uniform_filter1d(dth, 9) / 0.2 * sh).tolist())), fill=(255, 140, 60, 255), width=2)
        d.line(list(zip(kx.tolist(), (y0_ + sh - KL_S[:NS] / KL_S.max() * 0.95 * sh).tolist())), fill=(120, 255, 150, 255), width=2)
        cxp = x0_ + s / NS * sw; d.line((cxp, y0_, cxp, y0_ + sh), fill=(255, 255, 255, 230), width=2)
        d.text((x0_, y0_ - 56), 'pomarańczowy: długość kroku wag |δθ|', font=F_M, fill=(255, 150, 70, 255))
        d.text((x0_, y0_ - 32), 'zielony: klatka drgań tokenów (RMS wokół średniej ±10 kroków, w odstępach)', font=F_M, fill=(120, 255, 150, 255))
        d.text((x0_ + sw, y0_ + sh + 4), 'krok 1094 (koniec 1. epoki)', font=F_P, fill=(180, 185, 190, 255), anchor='ra')
        d.text((x0_, y0_ + sh + 4), 'krok 0', font=F_P, fill=(180, 185, 190, 255))
    if faza == 'koniec':
        a_ = sm((t - T_C) / 1.5)
        x0_, y0_ = 600, 200
        d.rectangle((x0_, y0_, x0_ + 1080, y0_ + 400), fill=(5, 9, 12, int(230 * a_)), outline=(60, 70, 80, int(255 * a_)))
        L_ = [('Co trzęsie cząsteczkami (pomiar):', (255, 255, 255)),
              (f'• skok tokenów rośnie z długością kroku wag: Spearman {pl(RHO)} krok po kroku (1. epoka)', (220, 225, 230)),
              (f'• rozgrzewka: |δθ| 0,04 → 0,17, klatka {pl(Q(5, 50))} → {pl(Q(150, 300))} odstępu; potem lekko stygnie ({pl(Q(700, 1090))})', (220, 225, 230)),
              (f'• energia skoku: wspólna dla zdjęcia {UDZ_MED[0]:.0%}, wspólna z sąsiadami {UDZ_MED[1]:.0%}, własna {UDZ_MED[2]:.0%}', (220, 225, 230)),
              ('Termometr (biegi z pokrętłami, fala 3–4):', (255, 255, 255)),
              ('• klatka rośnie jak lr^0,43;  paczka 16 / 32 / 64 / 128 przy SGD: klatka 0,39 / 0,40 / 0,38 / 0,31 — paczka nie grzeje,', (220, 225, 230)),
              ('  bo Adam i obcinanie gradientu ustalają długość kroku ≈ lr. Temperatura = długość kroku, nie lr / paczka.', (220, 225, 230)),
              ('Dlaczego to nie jest ciepło:', (255, 255, 255)),
              ('• szum jest wspólny (jeden generator dla wszystkich), kierunkowy (losowe kroki tej długości trzęsą 100–1000× słabiej)', (220, 225, 230)),
              ('  i nierównowagowy (kryształ pełznie). To temperatura efektywna układu napędzanego, jak potrząsany granulat.', (220, 225, 230))]
        for n, (l_, k_) in enumerate(L_):
            d.text((x0_ + 24, y0_ + 18 + 37 * n), l_, font=FN(20, n in (0, 4, 7)), fill=(*k_, int(255 * a_)))


def klatka(t):
    faza, s, wol, u = stan(t)
    i = int(np.clip(np.floor(s), 0, NS - 1)); v = float(s - i)
    x = X[i].astype(np.float32) * (1 - v) + X[i + 1].astype(np.float32) * v
    I = np.maximum(x, 0) / SKALA; Mk = np.maximum(x.mean(0), 0) / SKALA
    Pp = P[i] * (1 - v) + P[i + 1] * v; srodek = Pp.mean(0)
    # puls generatora i błysk laserów
    if faza == 'wolno':
        tw = wol[1]; amp = dth[tw] / 0.17
        puls = amp * (sm((u - 0.28) / 0.04) - sm((u - 0.62) / 0.06)); faza_s = np.clip((u - 0.3) / 0.3, 0, 1.2)
        zm = np.abs(X[tw + 1].astype(np.float32) - X[tw].astype(np.float32)).sum(0)
        blysk = np.clip(zm / np.percentile(zm, 95), 0, 1.5) * (sm((u - 0.58) / 0.05) - sm((u - 0.9) / 0.1))
        gen_jas = 0.6 + 2.5 * amp * np.exp(-((u - 0.3) / 0.06) ** 2)
    elif faza == 'szybko':
        amp = uniform_filter1d(dth, 5)[i] / 0.17; puls = 0.35 * amp; faza_s = (t * 1.7) % 1.0
        zm = np.abs(X[min(NS, i + 1)].astype(np.float32) - X[i].astype(np.float32)).sum(0)
        blysk = np.clip(zm / np.percentile(zm, 95), 0, 1.5) * 0.6; gen_jas = 0.6 + 1.2 * amp * (0.7 + 0.3 * np.sin(t * 25))
    else:
        amp = 0.0 if faza == 'wstep' else uniform_filter1d(dth, 5)[NS - 1] / 0.17; puls = 0.0; faza_s = 0.0; blysk = np.zeros(NK)
        gen_jas = 0.6 * sm((t - 4) / 2) if faza == 'wstep' else 0.6
    M, f, skala = macierz(t)
    mb = M.T.astype('f4').tobytes()
    buf_L.write(np.concatenate([LAS, blysk[:, None]], 1).astype('f4').tobytes())
    buf_P.write(np.concatenate([Pp, np.zeros((TOK, 1))], 1).astype('f4').tobytes())
    buf_I.write(I.astype('f4').tobytes()); buf_M.write(Mk.astype('f4').tobytes())
    wej = sm(t / 2.5); swiatl = sm((t - 4) / 3)
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0); ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mb); linie['szer'].value = 1.6
    rysuj(linie, SF, np.tile([0.25, 0.42, 0.62, 0.22 * wej], (len(SF), 1)), tryb=moderngl.LINES)
    wiazki['mvp'].write(mb); wiazki['srodek'].value = tuple(srodek); wiazki['gen'].value = tuple(GEN)
    wiazki['tryb'].value = 1; wiazki['gain'].value = 0.9 * wej; wiazki['kolor'].value = (0.25, 0.5, 1.6)
    vao_w.render(moderngl.LINES, vertices=2, instances=NK)
    wiazki['tryb'].value = 0; wiazki['gain'].value = 0.08 * wej; wiazki['kolor'].value = (0.25, 1.0, 0.45)
    vao_w.render(moderngl.LINES, vertices=2, instances=NK * TOK)
    if swiatl > 0:
        wiazki['tryb'].value = 2; wiazki['gain'].value = 0.10 * swiatl; wiazki['kolor'].value = (1.0, 0.55, 0.18)
        wiazki['faza'].value = float(faza_s); wiazki['puls'].value = float(puls)
        vao_w.render(moderngl.LINES, vertices=2, instances=NK)
    kropki['mvp'].write(mb); kropki['skala'].value = skala; kropki['pierscien'].value = 0
    kl_ = np.concatenate([np.tile([0.75, 1.0, 0.85], (NK, 1)), (wej * (0.7 + 0.0 * blysk))[:, None] * np.ones((NK, 1)) * 1.6], 1)
    rysuj(kropki, LAS, kl_, np.full(NK, 0.6))
    if blysk.max() > 0:
        rysuj(kropki, LAS, np.concatenate([np.tile([1.0, 0.75, 0.35], (NK, 1)), blysk[:, None] * (1.2 if faza == 'wolno' else 0.5)], 1), np.full(NK, 1.1))
    S = I.sum(1); jas = np.clip(S / np.percentile(np.maximum(X[0].astype(np.float32), 0).sum(1) / SKALA, 90), 0, 1.6) * 0.9 * wej
    rysuj(kropki, Pp, np.concatenate([np.tile([0.55, 1.0, 0.7], (TOK, 1)), jas[:, None]], 1), np.full(TOK, 0.38))
    rysuj(kropki, srodek[None], np.array([[0.35, 0.6, 1.0, 2.5 * wej]]), np.array([1.6]))
    if swiatl > 0:
        rysuj(kropki, GEN[None], np.array([[1.0, 0.6, 0.2, gen_jas * swiatl * 2.0]]), np.array([5.0 + 3.0 * puls]))
        rysuj(kropki, GEN[None], np.array([[1.0, 0.9, 0.7, gen_jas * swiatl]]), np.array([1.6]))
    if faza == 'wolno' and u < 0.32:                                                    # iskry paczki lecą do generatora
        q = np.clip((u - np.linspace(0, 0.1, 32)) / 0.2, 0, 1)
        ks = q[:, None] ** 1.5
        rng = np.random.default_rng(wol[1]); rozrzut = rng.normal(0, 3.0, (32, 3)) * (1 - ks)
        pos = ZRODLO + (GEN - ZRODLO) * ks + rozrzut
        rysuj(kropki, pos, np.concatenate([np.tile([1.0, 0.85, 0.5], (32, 1)), ((q > 0) & (q < 1))[:, None] * 1.4], 1), np.full(32, 1.1))
    fb_ldr.use(); ctx.disable(moderngl.BLEND); hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    kropki['pierscien'].value = 1
    rysuj(kropki, Pp[narz], np.tile([0.29, 0.94, 1.0, 0.85 * wej], (int(narz.sum()), 1)), np.full(int(narz.sum()), 0.42))
    kropki['pierscien'].value = 0
    if faza == 'wolno' and u > 0.55:                                                    # strzałki skoku × 4
        tw = wol[1]; a_ = sm((u - 0.55) / 0.08) * (1 - sm((u - 0.93) / 0.07))
        d_ = (P[tw + 1] - P[tw]) * 4.0; p0 = P[tw]; q1 = p0 + d_
        L = np.linalg.norm(d_, axis=1); dn = d_ / np.maximum(L, 1e-9)[:, None]; hl = np.minimum(0.3, 0.45 * L)[:, None]
        bok = np.cross(dn, -f); bok /= np.maximum(np.linalg.norm(bok, axis=1), 1e-9)[:, None]
        ty = q1 - dn * hl
        sg = np.concatenate([np.stack([p0, q1], 1), np.stack([q1, ty + bok * hl * 0.55], 1), np.stack([q1, ty - bok * hl * 0.55], 1)]).reshape(-1, 3)
        linie['szer'].value = 3.0
        rysuj(linie, sg, np.tile([1.0, 0.70, 0.28, 0.92 * a_], (len(sg), 1)), tryb=moderngl.LINES)
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, t, faza, s, wol, u, M)
    return img


CZASY = np.arange(int(T_D * FPS)) / FPS
if PROBA:
    CZASY = np.array([7.0, 10.5, 10.9, 11.4, 11.8, 30.9, 31.5, 50.0, 70.0, 80.0])
ff = None if PROBA else ffmpeg(HERE / 'lasery_5_drgania.mp4', FPS)
for n, t in enumerate(CZASY):
    img = klatka(float(t))
    if PROBA:
        img.save(HERE / f'lasery_5_proba_{t:04.1f}.png')
    else:
        ff.stdin.write(img.convert('RGBA').tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(CZASY)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
