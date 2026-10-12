"""Film 6: 768 laserów na sferze Fibonacciego i cząsteczki w układzie sferycznym aplikacji (nowy film, film 4 zostaje bez zmian).
Laser c = kanał c latentu (neuron fc1 ostatniego bloku, wyjście GELU) stoi w kierunku d_c spirali Fibonacciego, tej samej co w aplikacji
(app/wspolne.py: kierunki, kolory kanałów z koła barw hsv). Położenie cząsteczki jak w „Przestrzeni makrowektorów” aplikacji:

    p_j = s · Σ_c (z_jc − μ_c) d_c,   μ = średni latent całej sekwencji (806 checkpointów × 600 tokenów), s stałe,

czyli wypadkowa 768 wiązek: każdy laser ciągnie cząsteczkę w swoją stronę z siłą = natężenie kanału względem średniej. Oświetlenie jak
w filmie 4: z lasera c 600 zielonych wiązek o sile z_jc do cząsteczek i jedna niebieska o sile średniej po tokenach do środka zdjęcia
(średnia 600 położeń, w tym układzie też się porusza). Strzałki: przesunięcie węzła (średnia ±300 kroków) od węzła z kroku 0.

Dane (prawdziwe): natężenia 768D zdjęcia TRAIN 0 po 806 checkpointach (assets/.../ruch60_rp128_50tr_50val/pelne768.npy), IoU i maska
narzędzia z szescian.npz. Rzut 768D → 3D przez kierunki Fibonacciego jest stały i liniowy, ale to wybór układu (jak w aplikacji), nie pomiar.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_6_fibonacci.py [proba]"""
import sys
import time

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

from lasery_wspolne import EP, HERE, TOK, dane, ffmpeg

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
GAIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
NK = 768
FPS, WY, HY = 30, 1920, 1080
SS = 2                                                                              # nadpróbkowanie (renderujemy 4K, zmniejszamy GPU)
LATENT = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'

# ---------------- dane ----------------
sys.path.insert(0, str(HERE.parents[2] / 'app'))
from wspolne import kierunki, kolory_kanalow                                        # noqa: E402  (układ aplikacji)

D = dane(0)
kroki, iou, narz = D['kroki'].astype(np.float64), D['iou'], D['narz']
Z = np.asarray(np.load(LATENT, mmap_mode='r')[:, 0]).astype(np.float32)           # (806, 600, 768)
assert Z.shape == (len(kroki), TOK, NK)
R = 34.0
DIR = kierunki(False).astype(np.float64)                                            # (768, 3) spirala Fibonacciego, kanał c → d_c
KOL = kolory_kanalow()[:, :3]
MU = Z.mean((0, 1), dtype=np.float64)
Psur = (Z - MU) @ DIR                                                               # (806, 600, 3) wypadkowa wiązek
S_UKL = 0.62 * R / float(np.percentile(np.linalg.norm(Psur, axis=2), 99.5))         # cała chmura przez 60 epok mieści się w sferze
A3 = Psur * S_UKL
Q3 = np.stack([A3[np.abs(kroki - k) <= 300].mean(0) for k in kroki])               # węzły: średnia ±300 kroków
g3 = Q3[0]
C3 = A3.mean(1)                                                                     # (806, 3) środek zdjęcia (cel niebieskich wiązek)
SKALA = 2.0                                                                         # natężenie 2,0 = pełna wiązka (p99,9 w kroku 0 ≈ 1,8)
Zr = np.maximum(Z, 0) / SKALA
Mk = np.maximum(Z.mean(1), 0) / SKALA                                               # (806, 768) średnia po tokenach
log(f'latent {Z.shape}, natężenia > 0,1: krok 0 {(Z[0] > 0.1).mean():.0%}, koniec {(Z[-1] > 0.1).mean():.0%}')
sA, sQ, sC, sCw = (PchipInterpolator(kroki, X, axis=0) for X in (A3, Q3, C3, Q3.mean(1)))       # sCw: środek wygładzony (ślad)

# lasery na sferze Fibonacciego w kolejności kanałów (laser c stoi w kierunku d_c, tym samym, w którym ciągnie cząsteczki)
LAS = DIR * R
porz = np.random.default_rng(3).permutation(NK) / NK                                # kolejność zapalania zielonych wiązek

# cząsteczka do zbliżenia: token narzędzia najbliżej środka narzędzia na obrazie
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)       # (kolumna, wiersz)
sn = SIATKA[narz].mean(0); FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])

# ---------------- oś czasu filmu ----------------
FAZY = [(0, 'lasery'), (6, 'niebieskie'), (13, 'zielone'), (25, 'zblizenie'), (33, 'uczenie'), (73, 'koniec'), (79, None)]
T_UCZ0, T_UCZ1, T_PODZ = 35.0, 73.0, 47.0                                           # 35–47 s: kroki 0–2188, 47–73 s: do 65 640
KR_PODZ = 2 * EP


def krok_w(t):
    if t <= T_UCZ0:
        return 0.0
    if t <= T_PODZ:
        return KR_PODZ * (t - T_UCZ0) / (T_PODZ - T_UCZ0)
    return float(min(kroki[-1], KR_PODZ + (kroki[-1] - KR_PODZ) * (t - T_PODZ) / (T_UCZ1 - T_PODZ)))


sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))
KAM = [  # (czas, azymut, elewacja, odległość, cel)
    (0, -60, 14, 128, None), (6, -30, 20, 116, None), (13, 0, 28, 100, None), (22, 30, 52, 52, None),
    (27, 62, 38, 6, 'fok'), (32, 100, 30, 6, 'fok'), (36, 140, 42, 54, None), (73, 450, 42, 54, None), (79, 480, 36, 70, None)]


def kamera(t):
    for (ta, *pa), (tb, *pb) in zip(KAM[:-1], KAM[1:]):
        if t <= tb:
            u = sm((t - ta) / (tb - ta))
            break
    else:
        (ta, *pa), (tb, *pb), u = KAM[-2], KAM[-1], 1.0
    az, el, di = [pa[i] + (pb[i] - pa[i]) * u for i in range(3)]
    if 36 < t < 73:
        el += 12 * np.sin(2 * np.pi * (t - 36) / 37 * 1.5)
    poz_f = A3[0, FOK]
    ca = poz_f if pa[3] == 'fok' else np.zeros(3); cb = poz_f if pb[3] == 'fok' else np.zeros(3)
    return az, el, di, ca + (cb - ca) * u


def macierz(t):
    az, el, di, cel_ = kamera(t)
    a, e = np.radians(az), np.radians(el)
    oko = cel_ + di * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = cel_ - oko; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); u = np.cross(s, f)
    V = np.eye(4); V[0, :3], V[1, :3], V[2, :3] = s, u, -f; V[:3, 3] = -V[:3, :3] @ oko
    fov, asp, n, fa = np.radians(38), WY / HY, 0.1, 600.0
    P = np.zeros((4, 4)); P[0, 0] = 1 / np.tan(fov / 2) / asp; P[1, 1] = 1 / np.tan(fov / 2)
    P[2, 2] = (fa + n) / (n - fa); P[2, 3] = 2 * fa * n / (n - fa); P[3, 2] = -1
    return P @ V, oko, f, 1 / np.tan(fov / 2) * HY * SS / 2                          # skala piksela dla sprite'ów


# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
W4, H4 = WY * SS, HY * SS
hdr = ctx.texture((W4, H4), 4, dtype='f2'); fb_hdr = ctx.framebuffer([hdr])
ldr = ctx.texture((W4, H4), 4); fb_ldr = ctx.framebuffer([ldr])
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

wiazki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform int tryb; uniform float wl, gain, przyc, wzm; uniform int fok; uniform vec3 srodek;
layout(std430, binding=0) readonly buffer B0 { vec4 L[]; };
layout(std430, binding=1) readonly buffer B1 { vec4 P[]; };
layout(std430, binding=2) readonly buffer B2 { float I[]; };
layout(std430, binding=3) readonly buffer B3 { float M[]; };
out float v_t; out float v_i;
void main() {
    int id = gl_InstanceID; int k = tryb == 0 ? id / 600 : id; int j = id % 600;
    vec3 a = L[k].xyz; vec3 b = tryb == 0 ? P[j].xyz : srodek;
    float on = clamp((wl - L[k].w) * 25.0, 0.0, 1.0);
    float s = tryb == 0 ? I[j * 768 + k] : M[k];
    if (tryb == 0 && fok >= 0) s *= (j == fok ? wzm : przyc);
    v_i = s * on * gain; v_t = float(gl_VertexID);
    gl_Position = mvp * vec4(gl_VertexID == 0 ? a : b, 1.0);
}''', fragment_shader='''#version 430
uniform vec3 kolor; in float v_t; in float v_i; out vec4 o;
void main() { o = vec4(kolor * v_i * (0.10 + 0.90 * v_t * v_t), 1.0); }''')

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
uniform sampler2D tex; uniform float eksp; in vec2 uv; out vec4 o;
void main() { vec3 c = texture(tex, uv).rgb; c = 1.0 - exp(-c * eksp); o = vec4(pow(c, vec3(0.85)), 1.0); }''')
quad = ctx.vertex_array(ton, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4')), '2f', 'p')])

buf_L = ctx.buffer(np.concatenate([LAS, porz[:, None]], 1).astype('f4')); buf_L.bind_to_storage_buffer(0)
buf_P = ctx.buffer(reserve=TOK * 16); buf_P.bind_to_storage_buffer(1)
buf_I = ctx.buffer(reserve=TOK * NK * 4); buf_I.bind_to_storage_buffer(2)
buf_M = ctx.buffer(reserve=NK * 4); buf_M.bind_to_storage_buffer(3)
vao_w = ctx.vertex_array(wiazki, [])


def dyn_vao(prog, poz, kol, rozm=None):
    dane_ = [poz.astype('f4'), kol.astype('f4')] + ([rozm.astype('f4')[:, None]] if rozm is not None else [])
    b = ctx.buffer(np.concatenate(dane_, 1).tobytes())
    fmt = '3f 4f 1f' if rozm is not None else '3f 4f'
    return ctx.vertex_array(prog, [(b, fmt, *(['pos', 'kol', 'rozm'] if rozm is not None else ['pos', 'kol']))]), b


# siatka sfery (południki i równoleżniki)
seg = []
for lon in np.radians(np.arange(0, 180, 15)):
    t_ = np.linspace(0, 2 * np.pi, 97); p = R * np.stack([np.cos(t_) * np.cos(lon), np.cos(t_) * np.sin(lon), np.sin(t_)], 1); seg += [p[:-1], p[1:]]
for lat in np.radians(np.arange(-75, 90, 15)):
    t_ = np.linspace(0, 2 * np.pi, 97); p = R * np.stack([np.cos(t_) * np.cos(lat), np.sin(t_) * np.cos(lat), np.full_like(t_, np.sin(lat))], 1); seg += [p[:-1], p[1:]]
sf = np.stack([np.concatenate(seg[0::2]), np.concatenate(seg[1::2])], 1).reshape(-1, 3)
vao_sf, _ = dyn_vao(linie, sf, np.tile([0.25, 0.42, 0.62, 0.22], (len(sf), 1)))

# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_S, F_M = FN(34, True), FN(21), FN(17)
TEKSTY = {
    'lasery': ('768 laserów na sferze Fibonacciego = 768 kanałów latentu',
               ['Laser c stoi w kierunku d_c spirali Fibonacciego, tak jak kanał c w aplikacji (ten sam kolor z koła barw).',
                'Każdy laser to jeden neuron warstwy fc1 ostatniego bloku (wyjście GELU): jedna współrzędna latentu tokenu.']),
    'niebieskie': ('Niebieskie: z każdego lasera jedna wiązka do środka zdjęcia',
                   ['Siła = średnie natężenie kanału po 600 tokenach. Środek zdjęcia = średnia położeń wszystkich cząsteczek.',
                    'W tym układzie środek też się porusza, bo zmienia się średni latent zdjęcia.']),
    'zielone': ('Zielone: z każdego lasera 600 wiązek, po jednej do każdej cząsteczki',
                ['Siła wiązki c → j = natężenie kanału c dla tokenu j.  768 × 601 = 461 568 wiązek.',
                 'Cząsteczka stoi w wypadkowej swoich 768 wiązek: p = s · Σ (z_c − μ_c) · d_c, jak w Przestrzeni makrowektorów aplikacji.']),
    'zblizenie': ('Jedna cząsteczka z bliska: 768 wiązek ciągnie ją w 768 stron',
                  ['Pasek u dołu = 768 natężeń tej cząsteczki. Mocniejszy kanał ciągnie mocniej w stronę swojego lasera.',
                   'Sąsiedzi na obrazie nie muszą być sąsiadami tutaj: układ Fibonacciego to stały rzut 768 liczb na 3D, nie siatka obrazu.']),
    'uczenie': ('Uczenie: natężenia się zmieniają, cząsteczki wędrują',
                ['Strzałki = przesunięcie węzła (średnia ±300 kroków) od węzła z kroku 0 (szare kropki). Niebieski ślad = droga środka zdjęcia.',
                 f'1 jednostka = odstęp sąsiadów w kroku 0 w tym układzie.  Tempo: epoki 0–2 przez 12 s, epoki 2–60 przez 26 s.']),
    'koniec': ('Po 60 epokach', []),
}
iou_x = kroki / EP


def nakladka(img, faza, kr, t, kod):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    if faza == 'koniec':
        pod = KONIEC_TXT
    d.text((WY / 2, 26), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        d.text((WY / 2, 78 + 30 * n), s, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), 'Dane: natężenia 768 kanałów z 806 checkpointów biegu 35 000 (zdjęcie TRAIN 0). Położenia = stały rzut tych natężeń na kierunki Fibonacciego.',
           font=F_M, fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Ilustracja: sfera, rampa jasności wzdłuż wiązki. Zielone = wiązki do cząsteczek, niebieskie = do środka zdjęcia, '
           'błękitne pierścienie = narzędzie.', font=F_M, fill=(205, 210, 215, 255))
    if faza in ('uczenie', 'koniec'):
        d.text((WY - 30, 150), f'epoka {pl(kr / EP, 2)}   krok ' + f'{kr:,.0f}'.replace(',', ' '), font=FN(26, True), fill=(235, 240, 245, 255), anchor='ra')
        x0, y0, sw, sh = WY - 400, HY - 250, 370, 150                                    # wykres IoU
        d.rectangle((x0, y0, x0 + sw, y0 + sh), fill=(11, 16, 20, 220), outline=(42, 51, 58, 255))
        px = x0 + iou_x / iou_x[-1] * sw; py = y0 + sh - np.clip(iou, 0, 1) * sh
        d.line(list(zip(px.tolist(), py.tolist())), fill=(74, 240, 255, 255), width=2)
        d.line(list(zip(px.tolist(), (y0 + sh - PRZES_SR / PRZES_SR.max() * sh).tolist())), fill=(255, 179, 71, 255), width=2)
        cx = x0 + kr / kroki[-1] * sw; d.line((cx, y0, cx, y0 + sh), fill=(255, 255, 255, 230), width=2)
        d.text((x0, y0 - 46), 'IoU', font=F_M, fill=(74, 240, 255, 255))
        d.text((x0, y0 - 24), 'średnia długość strzałki (względna)', font=F_M, fill=(255, 179, 71, 255))
        d.text((x0 + sw, y0 + sh + 4), 'epoka 60', font=F_M, fill=(180, 185, 190, 255), anchor='ra')
    if kod is not None:                                                                 # pasek kodu cząsteczki
        x0, y0, hh = (WY - NK) // 2, HY - 150, 70
        d.rectangle((x0 - 10, y0 - 34, x0 + NK + 10, y0 + hh + 8), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
        d.text((x0, y0 - 30), f'768 natężeń cząsteczki {FOK} (wiersz {FOK // 30}, kolumna {FOK % 30}) — zielone; średnia po tokenach — niebieskie kreski',
               font=FN(15), fill=(220, 225, 230, 255))
        zi, mi = kod
        for k in range(NK):
            h = min(1.0, zi[k] / 1.25) * hh
            if h > 0.5:
                d.line((x0 + k, y0 + hh, x0 + k, y0 + hh - h), fill=(60, 255, 120, 255))
            hm = min(1.0, mi[k] / 1.25) * hh
            d.point((x0 + k, y0 + hh - hm), fill=(90, 160, 255, 255))


from scipy.spatial.distance import cdist                                              # noqa: E402
_dm = cdist(A3[0], A3[0]); np.fill_diagonal(_dm, np.inf)
A0 = float(np.median(np.sort(_dm, 1)[:, :4].mean(1)))                                 # odstęp sąsiadów w kroku 0 (ten układ)
PRZES = np.linalg.norm(Q3 - g3[None], axis=2) / A0                                     # (806, 600) długość strzałek w A0
PRZES_SR = PRZES.mean(1)
WSP = np.linalg.norm(Q3[-1].mean(0) - g3.mean(0)) / A0                                 # przesunięcie środka zdjęcia
WL = np.linalg.norm((Q3[-1] - Q3[-1].mean(0)) - (g3 - g3.mean(0)), axis=1).mean() / A0  # własne, względem środka
pa, pz = (Z[0] > 0.1).mean(), (Z[-1] > 0.1).mean()
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
KONIEC_TXT = [f'Średnie przesunięcie cząsteczki {pl(PRZES[-1].mean())} odstępu sąsiadów (narzędzie {pl(PRZES[-1][narz].mean())}, reszta {pl(PRZES[-1][~narz].mean())}).  '
              f'IoU {pl(iou[-1], 2)}.',
              f'Środek zdjęcia przesunął się o {pl(WSP)}, a cząsteczki względem środka średnio o {pl(WL)}: część ruchu jest wspólna dla całego zdjęcia.',
              f'Zielone wiązki rzadsze i mocniejsze: natężeń > 0,1 było {pa:.0%}, jest {pz:.0%}; najsilniejsza wiązka {pl(Z[0].max())} → {pl(Z[-1].max())}.']
log(KONIEC_TXT, "A0", A0, "S_UKL", S_UKL, "FOK", FOK)


def interp_kr(kr):
    i = int(np.clip(np.searchsorted(kroki, kr, side='right') - 1, 0, len(kroki) - 2)); u = float(np.clip((kr - kroki[i]) / (kroki[i + 1] - kroki[i]), 0, 1))
    return i, u


def klatka(t):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    kr = krok_w(t)
    i, u = interp_kr(kr)
    I = Zr[i] * (1 - u) + Zr[i + 1] * u; M = Mk[i] * (1 - u) + Mk[i + 1] * u
    A = sA(kr) if kr > 0 else A3[0]; Q = sQ(kr) if kr > 0 else Q3[0]; C = sC(kr) if kr > 0 else C3[0]
    mvp, oko, f, skala = macierz(t)
    mvp_b = mvp.T.astype('f4').tobytes()
    buf_P.write(np.concatenate([A, np.zeros((TOK, 1))], 1).astype('f4').tobytes())
    buf_I.write(I.astype('f4').tobytes()); buf_M.write(M.astype('f4').tobytes())
    # jasności faz
    las_on = sm(t / 4.0)
    nieb = sm((t - 6.5) / 4.0)
    ziel_wl = sm((t - 13.5) / 8.5) * 1.05
    przyc = 1.0 - 0.8 * (sm((t - 25) / 2.5) - sm((t - 32) / 2.5))
    # --- HDR: wiązki i poświaty (sumowanie addytywne) ---
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mvp_b); linie['szer'].value = 1.6
    vao_sf.render(moderngl.LINES)
    wiazki['mvp'].write(mvp_b); wiazki['srodek'].value = tuple(C.astype(float)); wiazki['przyc'].value = przyc; wiazki['wzm'].value = 1.0 + 30.0 * (1.0 - przyc) / 0.8; wiazki['fok'].value = FOK
    if nieb > 0:
        wiazki['tryb'].value = 1; wiazki['wl'].value = 2.0; wiazki['gain'].value = 0.9 * nieb; wiazki['kolor'].value = (0.25, 0.5, 1.6)
        vao_w.render(moderngl.LINES, vertices=2, instances=NK)
    if ziel_wl > 0:
        wiazki['tryb'].value = 0; wiazki['wl'].value = ziel_wl; wiazki['gain'].value = GAIN; wiazki['kolor'].value = (0.25, 1.0, 0.45)
        vao_w.render(moderngl.LINES, vertices=2, instances=NK * TOK)
    kropki['mvp'].write(mvp_b); kropki['skala'].value = skala; kropki['pierscien'].value = 0
    moc = (I.mean(0) / max(1e-6, Zr[0].mean(0).max()))                                  # moc lasera = średnie natężenie po tokenach
    kl = np.concatenate([0.35 + 0.65 * KOL, (las_on * (0.35 + 0.65 * np.clip(moc, 0, 1)))[:, None] * 2.2], 1)
    v, b = dyn_vao(kropki, LAS, kl, np.full(NK, 0.6)); v.render(moderngl.POINTS); v.release(); b.release()
    ogn = sm((t - 15) / 6)                                                              # poświata w ogniskach = suma natężeń
    if ogn > 0:
        S = I.sum(1); jas = np.clip(S / np.percentile(Zr[0].sum(1), 90), 0, 1.6) * ogn * 0.9
        kc = np.concatenate([np.tile([0.55, 1.0, 0.7], (TOK, 1)), jas[:, None]], 1)
        v, b = dyn_vao(kropki, A, kc, np.full(TOK, 0.38 * A0)); v.render(moderngl.POINTS); v.release(); b.release()
    if nieb > 0:
        v, b = dyn_vao(kropki, C[None], np.array([[0.35, 0.6, 1.0, 2.5 * nieb]]), np.array([1.6])); v.render(moderngl.POINTS); v.release(); b.release()
    # --- tonemapping do LDR, potem nakładki z przezroczystością ---
    fb_ldr.use(); ctx.disable(moderngl.BLEND)
    hdr.use(0); ton['eksp'].value = 1.0; quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    ref = sm((t - 21) / 3)
    if ref > 0:                                                                         # ogniska z kroku 0
        v, b = dyn_vao(kropki, g3, np.tile([0.62, 0.66, 0.7, 0.7 * ref], (TOK, 1)), np.full(TOK, 0.16 * A0)); v.render(moderngl.POINTS); v.release(); b.release()
    kropki['pierscien'].value = 1
    v, b = dyn_vao(kropki, A[narz], np.tile([0.29, 0.94, 1.0, 0.9 * ogn], (int(narz.sum()), 1)), np.full(int(narz.sum()), 0.42 * A0)); v.render(moderngl.POINTS); v.release(); b.release()
    if przyc < 0.99:
        v, b = dyn_vao(kropki, A[FOK:FOK + 1], np.array([[1.0, 1.0, 1.0, 0.8 * (1 - przyc) / 0.8]]), np.array([0.6 * A0])); v.render(moderngl.POINTS); v.release(); b.release()
    kropki['pierscien'].value = 0
    if kr > 0:                                                                          # ślad środka zdjęcia
        kk_ = np.linspace(0, kr, 200); sl = sCw(kk_)
        seg_ = np.stack([sl[:-1], sl[1:]], 1).reshape(-1, 3)
        linie['szer'].value = 4.0
        v, b = dyn_vao(linie, seg_, np.tile([0.35, 0.6, 1.0, 0.9], (len(seg_), 1))); v.render(moderngl.LINES); v.release(); b.release()
    if kr > 0:                                                                          # strzałki g → Q
        d_ = Q - g3; L = np.linalg.norm(d_, axis=1); ok = L > 0.03 * A0
        dn = d_ / np.maximum(L, 1e-9)[:, None]; hl = np.minimum(0.32 * A0, 0.45 * L)[:, None]
        bok = np.cross(dn, -f); bok /= np.maximum(np.linalg.norm(bok, axis=1), 1e-9)[:, None]
        tylek = Q - dn * hl
        seg = np.concatenate([np.stack([g3, Q], 1)[ok], np.stack([Q, tylek + bok * hl * 0.55], 1)[ok], np.stack([Q, tylek - bok * hl * 0.55], 1)[ok]]).reshape(-1, 3)
        kk = np.tile([1.0, 0.70, 0.28, 0.5], (len(seg), 1))
        linie['szer'].value = 1.8
        v, b = dyn_vao(linie, seg, kk); v.render(moderngl.LINES); v.release(); b.release()
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    kod = None
    if t >= 25:
        kod = (I[FOK] * SKALA, M * SKALA)
    nakladka(img, faza, kr, t, kod)
    return img


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    CZASY = np.array([10.0, 18.0, 24.0, 30.5, 45.0])
ff = None if PROBA else ffmpeg(HERE / 'lasery_6_fibonacci.mp4', FPS)
for n, t in enumerate(CZASY):
    img = klatka(float(t))
    if PROBA:
        img.save(HERE / f'lasery_6_proba_{t:04.1f}.png')
    else:
        ff.stdin.write(img.convert('RGBA').tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(CZASY)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
