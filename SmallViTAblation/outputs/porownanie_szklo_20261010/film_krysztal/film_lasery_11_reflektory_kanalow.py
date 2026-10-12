"""Film 11: 768 reflektorów (po jednym na neuron = kanał) wokół szklanej sfery, każdy z 600 diodami (po jednej na cząsteczkę = patch).
Nowy film; poprawia grupowanie z filmu 10 (tam reflektor = token, co wymagało 600 identycznych reflektorów). Zasada Z13 w ZASADY_LASER_KOLOID.md.

- Reflektor c stoi za sferą w punkcie −RR · d_c, gdzie d_c = P v̂_c (P = stały rzut 192D → 3D, trzy główne składowe kierunków v_c
  w kroku 22 974). Jego jedno ustawienie (kierunek d_c) jest wspólne dla wszystkich cząsteczek: to są wspólne wagi.
- Dioda j reflektora c świeci z jasnością z_jc (aktywacja neuronu c dla tokenu j). Wiązka c → j biegnie z reflektora do punktu sfery
  E_jc = p_j − s · d_c (|E_jc| = R), załamuje się i dochodzi do cząsteczki j w kierunku d_c: wektor siły z_jc · v_c.
  Wszystkie 600 wiązek jednego reflektora biegną w sferze równolegle.
- Położenie cząsteczki p_j = P W₂ (z_j − z̄), skala dopasowana do kroku (licznik powiększenia), jak w filmie 10.
- Uczenie: kierunki d_c się obracają, więc reflektory wędrują po sferze; jasności diod zmieniają się dla każdej cząsteczki osobno.
Dane i okienka jak w filmie 10. Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_11_reflektory_kanalow.py [proba]"""
import os
import sys
import time

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lasery_wspolne import EP, H, HERE, TOK, W, Swiatlo, dane, ffmpeg, oko, prokrustes2d, zastosuj

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
GAIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
NK = 768
FPS, WY, HY = 30, 1920, 1080
SS = 2
ASSETS = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10'
WAGI = ASSETS / '01_bieg_35000_gesty_zapis/eden/gesty/wagi'
LATENT = ASSETS / '03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
CACHE = HERE / 'reflektor_8_cache.npz'

# ---------------- dane ----------------
sys.path.insert(0, str(HERE.parents[2] / 'app'))
from wspolne import kolory_kanalow                                                  # noqa: E402

D = dane(0)
kroki_all, iou_all, narz = D['kroki'].astype(np.int64), D['iou'], D['narz']
if CACHE.exists():
    c_ = np.load(CACHE); KW, V = c_['KW'], c_['V']
else:
    import torch
    KW = np.array(sorted(int(f[1:6]) for f in os.listdir(WAGI) if f.endswith('.pt')))
    V = np.stack([torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']['bloki.5.mlp.fc2.weight']
                  .float().numpy() for k in KW])
    np.savez_compressed(CACHE, KW=KW, V=V)
NT = len(KW)
idx_lat = np.searchsorted(kroki_all, KW)
assert np.all(kroki_all[idx_lat] == KW)
Z = np.clip(np.asarray(np.load(LATENT, mmap_mode='r')[idx_lat, 0]).astype(np.float32), 0, None)   # (341, 600, 768)
iou = iou_all[idx_lat]
Vn_end = V[-1] / np.linalg.norm(V[-1], axis=0)
U_, _, _ = np.linalg.svd(Vn_end - Vn_end.mean(1, keepdims=True), full_matrices=False)
P3 = U_[:, :3].T
PV = np.einsum('ad,tdc->tca', P3, V)                                                # (341, 768, 3)
DIRS = PV / np.linalg.norm(PV, axis=2, keepdims=True)
POS_RAW = np.einsum('tjc,tca->tja', Z - Z.mean(1, keepdims=True), PV)
R = 34.0
RR = 50.0
R99 = np.percentile(np.linalg.norm(POS_RAW, axis=2), 99, axis=1)
POS = POS_RAW / R99[:, None, None] * 0.55 * R
POW = R99[-1] / R99
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
pary = np.array([(np.where(np.eye(NK, dtype=bool), 0.0, Vn[t].T @ Vn[t]).min(1) < -0.5).mean() for t in range(NT)])
KOL = kolory_kanalow()[:, :3]
SKALA = 2.0
Zr = Z / SKALA
C1 = int(np.argmax(Z[-1].var(0)))                                                   # reflektor do pokazania osobno
log(f'wagi {V.shape}, aktywacje {Z.shape}, pary {pary[0]:.0%} → {pary[-1]:.0%}, reflektor pokazowy {C1}')

G2 = D['g3'][:, :2]; Ql2 = D['Q3'][..., :2]
_, MAPY2 = prokrustes2d(Ql2, G2)
QC2 = np.stack([zastosuj(Ql2[t], MAPY2[t]) for t in range(len(Ql2))])[idx_lat]
LOK2 = np.linalg.norm(QC2 - G2, axis=-1)
LOK_MAX = float(np.percentile(LOK2, 99))
NARZ_LOK = LOK2[:, narz].mean(1) / np.maximum(LOK2[:, ~narz].mean(1), 1e-9)
OKO_F = oko()
OKO = (OKO_F * 255).astype(np.uint8)
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
porz = np.random.default_rng(3).permutation(NK) / NK                               # kolejność zapalania reflektorów
porz[C1] = 0.0

# ---------------- oś czasu ----------------
FAZY = [(0, 'sfera'), (6, 'jeden'), (16, 'wszystkie'), (26, 'zblizenie'), (34, 'uczenie'), (71, 'koniec'), (77, None)]
T_U0, T_UP, T_U1 = 36.0, 48.0, 70.0
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def indeks(t):
    if t < T_U0:
        return float(NT - 1) if t < 34 else float(NT - 1) * (1 - sm((t - 34) / 2.0))
    if t < T_UP:
        kr = 2 * EP * (t - T_U0) / (T_UP - T_U0)
    else:
        kr = min(KW[-1], 2 * EP + (KW[-1] - 2 * EP) * (t - T_UP) / (T_U1 - T_UP))
    return float(np.interp(kr, KW, np.arange(NT)))


def stan(fi):
    i = int(min(fi, NT - 2)); u = fi - i
    lerp = lambda X: X[i] * (1 - u) + X[i + 1] * u
    d = lerp(DIRS); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return lerp(POS), d, lerp(Zr), KW[i] * (1 - u) + KW[i + 1] * u, i if u < 0.5 else i + 1


KAM = [(0, -70, 16, 150, None), (6, -40, 18, 125, None), (16, -10, 24, 120, None), (24, 20, 30, 70, None),
       (28, 45, 26, 12, 'fok'), (33, 70, 22, 12, 'fok'), (37, 100, 28, 115, None), (71, 330, 28, 115, None), (77, 350, 24, 132, None)]


def macierz(t, pf, cel_c):
    for (ta, *pa), (tb, *pb) in zip(KAM[:-1], KAM[1:]):
        if t <= tb:
            u = sm((t - ta) / (tb - ta)); break
    else:
        (ta, *pa), (tb, *pb), u = KAM[-2], KAM[-1], 1.0
    az, el, di = [pa[k] + (pb[k] - pa[k]) * u for k in range(3)]
    if 6 <= t < 16:                                                                 # patrzymy z boku na wiązki reflektora C1
        b = sm((t - 6) / 2.5) * (1 - sm((t - 14) / 2.0))
        az_c = np.degrees(np.arctan2(cel_c[1], cel_c[0])) + 90.0
        az = az + (az_c - az) * b
    if 37 < t < 71:
        el += 10 * np.sin(2 * np.pi * (t - 37) / 34 * 1.5)
    ca = pf if pa[3] == 'fok' else np.zeros(3); cb = pf if pb[3] == 'fok' else np.zeros(3)
    cel = ca + (cb - ca) * u
    a, e = np.radians(az), np.radians(el)
    oko_ = cel + di * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = cel - oko_; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    Vm = np.eye(4); Vm[0, :3], Vm[1, :3], Vm[2, :3] = s, up, -f; Vm[:3, 3] = -Vm[:3, :3] @ oko_
    fov, asp, n, fa = np.radians(40), WY / HY, 0.1, 800.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n) / (n - fa); Pm[2, 3] = 2 * fa * n / (n - fa); Pm[3, 2] = -1
    return Pm @ Vm, 1 / np.tan(fov / 2) * HY * SS / 2


def rzut(mvp, p):
    q = mvp @ np.append(p, 1.0)
    if q[3] <= 0.1:
        return None
    return (q[0] / q[3] * 0.5 + 0.5) * WY, (0.5 - q[1] / q[3] * 0.5) * HY


# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
W4, H4 = WY * SS, HY * SS
hdr = ctx.texture((W4, H4), 4, dtype='f2'); fb_hdr = ctx.framebuffer([hdr])
ldr = ctx.texture((W4, H4), 4); fb_ldr = ctx.framebuffer([ldr])
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

# wiązki c → j: tryb 1 = z reflektora c do punktu sfery E_jc, tryb 0 = w sferze z E_jc do cząsteczki j (kierunek d_c)
wiazki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform int tryb, fok, fokc; uniform float gain, przyc, wzm, przycc, wzmc, wl, R, RR;
layout(std430, binding=0) readonly buffer B0 { float PORZ[]; };
layout(std430, binding=1) readonly buffer B1 { vec4 P[]; };
layout(std430, binding=2) readonly buffer B2 { vec4 DC[]; };
layout(std430, binding=3) readonly buffer B3 { float I[]; };
out float v_t; out float v_i; flat out int v_c;
void main() {
    int id = gl_InstanceID; int c = id / 600; int j = id % 600;
    vec3 p = P[j].xyz; vec3 d = DC[c].xyz;
    float b = dot(p, d); float s = b + sqrt(max(b * b - dot(p, p) + R * R, 0.0));
    vec3 e = p - s * d;
    vec3 a = tryb == 0 ? e : -RR * d; vec3 q = tryb == 0 ? p : e;
    float on = clamp((wl - PORZ[c]) * 30.0, 0.0, 1.0);
    float x = I[j * 768 + c] * on * gain;
    if (fok >= 0) x *= (j == fok ? wzm : przyc);
    if (fokc >= 0) x *= (c == fokc ? wzmc : przycc);
    v_i = x; v_t = float(gl_VertexID); v_c = c;
    gl_Position = mvp * vec4(gl_VertexID == 0 ? a : q, 1.0);
}''', fragment_shader='''#version 430
layout(std430, binding=4) readonly buffer B4 { vec4 KC[]; };
uniform int tryb; in float v_t; in float v_i; flat in int v_c; out vec4 o;
void main() { float r = tryb == 0 ? (0.25 + 0.75 * v_t * v_t) : (0.6 + 0.4 * v_t);
  o = vec4(KC[v_c].rgb * v_i * r, 1.0); }''')

kropki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform float skala;
in vec3 pos; in vec4 kol; in float rozm; out vec4 v_k;
void main() { gl_Position = mvp * vec4(pos, 1.0); gl_PointSize = clamp(rozm * skala / gl_Position.w, 1.5, 60.0); v_k = kol; }''',
                     fragment_shader='''#version 430
uniform int ksztalt; in vec4 v_k; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); float a;
  if (ksztalt == 2) { a = (max(abs(d.x), abs(d.y)) < 0.85) ? 1.0 : 0.0; }
  else { if (r2 > 1.0) discard; a = ksztalt == 1 ? smoothstep(0.55, 0.75, sqrt(r2)) * (1.0 - smoothstep(0.85, 1.0, sqrt(r2))) : exp(-r2 * 4.0); }
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

buf_O = ctx.buffer(porz.astype('f4')); buf_O.bind_to_storage_buffer(0)
buf_P = ctx.buffer(reserve=TOK * 16); buf_P.bind_to_storage_buffer(1)
buf_D = ctx.buffer(reserve=NK * 16); buf_D.bind_to_storage_buffer(2)
buf_I = ctx.buffer(reserve=TOK * NK * 4); buf_I.bind_to_storage_buffer(3)
buf_K = ctx.buffer(np.concatenate([0.25 + 0.75 * KOL, np.ones((NK, 1))], 1).astype('f4')); buf_K.bind_to_storage_buffer(4)
vao_w = ctx.vertex_array(wiazki, [])
wiazki['R'].value = R; wiazki['RR'].value = RR


def rysuj(prog, tryb, poz, kol, rozm=None):
    if len(poz) == 0:
        return
    dane_ = [poz.astype('f4'), kol.astype('f4')] + ([rozm.astype('f4')[:, None]] if rozm is not None else [])
    b = ctx.buffer(np.concatenate(dane_, 1).tobytes())
    fmt = '3f 4f 1f' if rozm is not None else '3f 4f'
    v = ctx.vertex_array(prog, [(b, fmt, *(['pos', 'kol', 'rozm'] if rozm is not None else ['pos', 'kol']))])
    v.render(tryb); v.release(); b.release()


seg = []
for lon_ in np.radians(np.arange(0, 180, 12)):
    t_ = np.linspace(0, 2 * np.pi, 121); p = R * np.stack([np.cos(t_) * np.cos(lon_), np.cos(t_) * np.sin(lon_), np.sin(t_)], 1); seg += [p[:-1], p[1:]]
for lat_ in np.radians(np.arange(-72, 90, 12)):
    t_ = np.linspace(0, 2 * np.pi, 121); p = R * np.stack([np.cos(t_) * np.cos(lat_), np.sin(t_) * np.cos(lat_), np.full_like(t_, np.sin(lat_))], 1); seg += [p[:-1], p[1:]]
SFERA = np.stack([np.concatenate(seg[0::2]), np.concatenate(seg[1::2])], 1).reshape(-1, 3)

# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_S, F_M = FN(34, True), FN(21), FN(17)
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
TEKSTY = {
    'sfera': ('768 reflektorów wokół szklanej sfery = 768 neuronów ostatniego bloku MLP',
              ['Reflektor c stoi za sferą naprzeciw swojego kierunku pchnięcia d_c (rzut kolumny fc2).  Kolor = kanał, jak w aplikacji.',
               'Każdy reflektor ma 600 diod: po jednej na każdą cząsteczkę (patch).  Jedno ustawienie reflektora = wspólne wagi dla wszystkich patchy.']),
    'jeden': ('Jeden reflektor: 600 diod, 600 wiązek, każda do swojej cząsteczki',
              ['Wiązka diody j trafia w sferę, załamuje się i dochodzi do cząsteczki j w kierunku d_c.  Po załamaniu wszystkie 600 wiązek biegną równolegle:',
               'ten neuron pcha każdą cząsteczkę w tę samą stronę, tylko z inną siłą z_jc.  Okienko z lewej = 600 diod tego reflektora na siatce obrazu.']),
    'wszystkie': ('Wszystkie 768 reflektorów: 768 × 600 = 460 800 wiązek',
                  ['Każdy reflektor oświetla wszystkie cząsteczki, każdą swoją diodą.  W środku sfery wiązki jednego koloru biegną równolegle.',
                   'Na każdą cząsteczkę działa 768 wektorów (po jednym z każdego reflektora), w 192 wymiarach; tu ich rzut do 3D.']),
    'zblizenie': ('Jedna cząsteczka z bliska: po jednej wiązce z każdego z 768 reflektorów',
                  ['Położenie cząsteczki = suma jej 768 wektorów siły (względem średniej po tokenach), w tym samym rzucie co kierunki wiązek.',
                   'Pasek u dołu = 768 natężeń tej cząsteczki: jasność jej diody w każdym reflektorze.']),
    'uczenie': ('Uczenie: reflektory obracają się wokół sfery',
                ['Kierunek d_c reflektora zmienia się raz dla wszystkich 600 cząsteczek, więc reflektor wędruje po sferze;  jasności diod zmieniają się osobno.',
                 'Okienka z lewej jak w filmie 7.  Tempo: epoki 0–2 przez 12 s, epoki 2–21 przez 22 s.']),
    'koniec': ('Po 21 epokach', []),
}
KONIEC_TXT = [f'Pary reflektorów o przeciwnych kierunkach (cos < −0,5): {pary[0]:.0%} → {pary[-1]:.0%}: reflektory ustawiły się po przeciwnych stronach sfery.',
              f'IoU {pl(float(iou[-1]), 2)}.  Szybko (dla każdej cząsteczki) zmieniają się jasności diod;  powoli (przez uczenie) obracają się reflektory.',
              'Następny krok: uwaga jako wiązki między cząsteczkami i łańcuch 6 bloków.']

import matplotlib                                                                       # noqa: E402
PLAZMA, MAGMA, VIRIDIS = matplotlib.colormaps['plasma'], matplotlib.colormaps['magma'], matplotlib.colormaps['viridis']
XL2, YL2 = W / 2 + 1.5, H / 2 + 1.5
SW2 = Swiatlo(XL2, YL2, PX=15)
OX, OY = 22, 190


def okienka(img, ii, a):
    if a <= 0.01:
        return
    qc = QC2[ii]
    lok = np.linalg.norm(qc - G2, axis=1)
    L = (SW2.rgb(SW2.studnie(qc)) * 255).astype(np.uint8)
    pan = Image.fromarray(L); dp = ImageDraw.Draw(pan)
    px = lambda P: np.stack([(P[:, 0] + XL2) * SW2.PX, (P[:, 1] + YL2) * SW2.PX], 1)
    a0, a1 = px(G2), px(qc)
    for j in np.argsort(lok):
        c = PLAZMA(min(1.0, lok[j] / LOK_MAX)); k = tuple(int(255 * x) for x in c[:3])
        dp.line((*a0[j], *a1[j]), fill=k, width=2 if lok[j] > 0.3 * LOK_MAX else 1)
    for j in np.flatnonzero(narz):
        dp.ellipse((a1[j][0] - 6, a1[j][1] - 6, a1[j][0] + 6, a1[j][1] + 6), outline=(74, 240, 255), width=2)
    pw, ph = pan.size
    ramka = Image.new('RGBA', (pw + 16, ph + 40), (11, 16, 20, 235))
    ramka.paste(pan, (8, 32))
    ImageDraw.Draw(ramka).text((8, 6), 'Rzut płaski (jak w filmie 2): studnie odpływają od adresów', font=FN(16), fill=(232, 232, 232))
    mapa = (MAGMA(np.clip(lok / LOK_MAX, 0, 1))[:, :3] * 255).reshape(H, W, 3).astype(np.uint8)
    mapa = np.asarray(Image.fromarray(mapa).resize((W * 16, H * 16), Image.NEAREST)).astype(np.float32)
    oko_m = (OKO * 0.45 + mapa * 0.55).astype(np.uint8)
    rb = Image.new('RGBA', (W * 16 + 16, H * 16 + 40), (11, 16, 20, 235))
    rb.paste(Image.fromarray(oko_m), (8, 32))
    ImageDraw.Draw(rb).text((8, 6), f'Gdzie centra odpłynęły najdalej (jasno); narzędzie/reszta ×{pl(NARZ_LOK[ii], 2)}', font=FN(16), fill=(232, 232, 232))
    for pan_, (x, y) in ((ramka, (OX, OY)), (rb, (OX, OY + ramka.size[1] + 14))):
        pan_.putalpha(Image.eval(Image.new('L', pan_.size, 255), lambda v: int(v * a)))
        img.paste(pan_, (x, y), pan_)


def panel_diod(img, z_c, a):
    """600 diod reflektora C1 na siatce obrazu 30 × 20 (na tle zdjęcia)."""
    v = np.clip(z_c / max(1e-6, np.percentile(z_c, 99)), 0, 1)
    mapa = (VIRIDIS(v)[:, :3] * 255).reshape(H, W, 3).astype(np.uint8)
    mapa = np.asarray(Image.fromarray(mapa).resize((W * 12, H * 12), Image.NEAREST)).astype(np.float32)
    tlo = np.asarray(Image.fromarray(OKO).resize((W * 12, H * 12))).astype(np.float32)
    m = (tlo * 0.3 + mapa * 0.7).astype(np.uint8)
    rb = Image.new('RGBA', (W * 12 + 16, H * 12 + 40), (11, 16, 20, 235))
    rb.paste(Image.fromarray(m), (8, 32))
    ImageDraw.Draw(rb).text((8, 6), f'600 diod reflektora {C1} (z_jc na siatce obrazu)', font=FN(16), fill=(232, 232, 232))
    rb.putalpha(Image.eval(Image.new('L', rb.size, 255), lambda q: int(q * a)))
    img.paste(rb, (OX, OY), rb)


iou_x = KW / EP


def nakladka(img, faza, t, kr, ii, I, mvp, Dc):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    if faza == 'koniec':
        pod = KONIEC_TXT
    d.text((WY / 2, 26), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        d.text((WY / 2, 78 + 30 * n), s, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), 'Dane: wagi fc2 bloku 6 z 341 checkpointów (kroki 0–22 974) i aktywacje 768 neuronów zdjęcia TRAIN 0. Reflektor c w punkcie −RR · d_c, '
           'd_c = P v̂_c;  położenia = P W₂ (z − z̄).', font=F_M, fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Ilustracja: optyka poglądowa (sfera załamuje wiązki tak, by każda dochodziła w kierunku d_c), P = 3 główne składowe; błękitne pierścienie = narzędzie.',
           font=F_M, fill=(205, 210, 215, 255))
    if 6 <= t < 16:
        panel_diod(img, I[:, C1] * SKALA, sm((t - 6.5) / 1.5) * (1 - sm((t - 15) / 1.0)))
        q = rzut(mvp, -RR * 1.08 * Dc[C1])
        if q and q[1] < HY - 90:
            d.text(q, f'reflektor {C1}', font=FN(20, True), fill=(220, 240, 255, int(255 * sm((t - 6.5) / 1.5))), anchor='mb')
    if faza in ('uczenie', 'koniec'):
        d.text((WY - 30, 150), f'epoka {pl(kr / EP, 2)}   krok ' + f'{kr:,.0f}'.replace(',', ' '), font=FN(26, True), fill=(235, 240, 245, 255), anchor='ra')
        d.text((WY - 30, 186), f'pary reflektorów o przeciwnych kierunkach: {pary[ii]:.0%}', font=FN(20, True), fill=(255, 200, 120, 255), anchor='ra')
        d.text((WY - 30, 216), f'powiększenie chmury względem końca: ×{pl(float(POW[ii]), 1)}', font=FN(18), fill=(190, 210, 230, 255), anchor='ra')
        x0, y0, sw, sh = WY - 400, HY - 250, 370, 150
        d.rectangle((x0, y0, x0 + sw, y0 + sh), fill=(11, 16, 20, 220), outline=(42, 51, 58, 255))
        px = x0 + iou_x / iou_x[-1] * sw; py = y0 + sh - np.clip(iou, 0, 1) * sh
        d.line(list(zip(px.tolist(), py.tolist())), fill=(74, 240, 255, 255), width=2)
        d.line(list(zip(px.tolist(), (y0 + sh - pary / max(pary.max(), 1e-9) * sh).tolist())), fill=(255, 179, 71, 255), width=2)
        cx = x0 + kr / KW[-1] * sw; d.line((cx, y0, cx, y0 + sh), fill=(255, 255, 255, 230), width=2)
        d.text((x0, y0 - 46), 'IoU', font=F_M, fill=(74, 240, 255, 255))
        d.text((x0, y0 - 24), 'pary przeciwnych reflektorów (względnie)', font=F_M, fill=(255, 179, 71, 255))
        d.text((x0 + sw, y0 + sh + 4), 'epoka 21', font=F_M, fill=(180, 185, 190, 255), anchor='ra')
        okienka(img, ii, sm((t - 35) / 2.0))
    if t >= 16:
        zi = I[FOK]
        x0, y0, hh = (WY - NK) // 2, HY - 150, 70
        d.rectangle((x0 - 10, y0 - 34, x0 + NK + 10, y0 + hh + 8), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
        d.text((x0, y0 - 30), f'768 natężeń cząsteczki {FOK} (wiersz {FOK // 30}, kolumna {FOK % 30}): jasność jej diody w każdym reflektorze, kolor = reflektor',
               font=FN(15), fill=(220, 225, 230, 255))
        for k in range(NK):
            h = min(1.0, zi[k] * SKALA / 1.25) * hh
            if h > 0.5:
                d.line((x0 + k, y0 + hh, x0 + k, y0 + hh - h), fill=tuple(int(255 * v) for v in (0.3 + 0.7 * KOL[k])) + (255,))


def klatka(t):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    fi = indeks(t)
    Pp, Dc, I, kr, ii = stan(fi)
    mvp, skala = macierz(t, POS[-1, FOK] if t < 36 else Pp[FOK], -RR * Dc[C1])
    mvp_b = mvp.T.astype('f4').tobytes()
    buf_P.write(np.concatenate([Pp, np.zeros((TOK, 1))], 1).astype('f4').tobytes())
    buf_D.write(np.concatenate([Dc, np.zeros((NK, 1))], 1).astype('f4').tobytes())
    buf_I.write(I.astype('f4').tobytes())
    ref_on = sm(t / 3.0)
    jeden = sm((t - 6.5) / 2.5)
    wl = 0.0 if t < 16 else sm((t - 16.5) / 7.0) * 1.05
    wiazki['mvp'].write(mvp_b)
    # wyróżnienia: reflektor C1 (6–16 s) i cząsteczka FOK (26–34 s)
    if t < 16:
        wiazki['fokc'].value = C1; wiazki['przycc'].value = 0.0; wiazki['wzmc'].value = 6.0 * jeden
    else:
        b = 1.0 - sm((t - 16) / 3.0)
        wiazki['fokc'].value = C1; wiazki['przycc'].value = 1.0; wiazki['wzmc'].value = 1.0 + 5.0 * b
    przyc = 1.0 - 0.85 * (sm((t - 26) / 2.0) - sm((t - 33) / 2.0))
    wiazki['fok'].value = FOK; wiazki['przyc'].value = przyc
    wiazki['wzm'].value = 1.0 + 30.0 * (1.0 - przyc) / 0.85
    wiazki['wl'].value = max(wl, 0.034 if t >= 6.5 else -1.0)
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mvp_b); linie['szer'].value = 1.5
    rysuj(linie, moderngl.LINES, SFERA, np.tile([0.30, 0.55, 0.75, 0.20 * ref_on], (len(SFERA), 1)))
    for tryb, g in ((1, 0.35), (0, 1.0)):
        wiazki['tryb'].value = tryb; wiazki['gain'].value = GAIN * g
        vao_w.render(moderngl.LINES, vertices=2, instances=TOK * NK)
    kropki['mvp'].write(mvp_b); kropki['skala'].value = skala
    # reflektory: kwadraty w kolorze kanału, jasność = średnie natężenie diod
    moc = I.mean(0); moc = np.clip(moc / max(1e-6, np.percentile(moc, 95)), 0.15, 1.5)
    on = np.clip((max(wl, 0.034 if t >= 6.5 else 0.0) - porz) * 30, 0, 1) if t >= 16 else (np.arange(NK) == C1) * jeden
    kropki['ksztalt'].value = 2
    rysuj(kropki, moderngl.POINTS, -RR * Dc, np.concatenate([(0.3 + 0.7 * KOL) * (0.35 + 0.9 * on * moc)[:, None], np.full((NK, 1), 0.9 * ref_on)], 1),
          np.full(NK, 1.1))
    kropki['ksztalt'].value = 0
    ogn = sm((t - 8) / 4)
    if ogn > 0:
        S = I.sum(1)
        aj = np.clip(S / np.percentile(S, 90), 0, 1.6) * ogn * (0.35 if t < 16 else 1.0)
        aj[FOK] = max(aj[FOK], ogn)
        rysuj(kropki, moderngl.POINTS, Pp, np.concatenate([np.tile([0.85, 1.0, 0.9], (TOK, 1)), aj[:, None]], 1), np.full(TOK, 0.9))
    fb_ldr.use(); ctx.disable(moderngl.BLEND)
    hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    kropki['ksztalt'].value = 1
    rysuj(kropki, moderngl.POINTS, Pp[narz], np.tile([0.29, 0.94, 1.0, 0.85 * ogn], (int(narz.sum()), 1)), np.full(int(narz.sum()), 1.6))
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, faza, t, kr, ii, I, mvp, Dc)
    return img


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    CZASY = np.array([4.0, 11.0, 21.0, 30.0, 45.0, 62.0, 74.0])
ff = None if PROBA else ffmpeg(HERE / 'lasery_11_reflektory_kanalow.mp4', FPS)
for n, t in enumerate(CZASY):
    img = klatka(float(t))
    if PROBA:
        img.save(HERE / f'lasery_11_proba_{t:04.1f}.png')
    else:
        ff.stdin.write(img.convert('RGBA').tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(CZASY)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
