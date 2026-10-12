"""Film 10: 600 reflektorów (po jednym na patch) za szklaną sferą; każdy ma 768 kanałów kierunkowych (zasada Z12 w ZASADY_LASER_KOLOID.md).
Nowy film na bazie filmu 7 (okienka z lewej: rzut płaski i mapa na oku, wykres IoU, pasek natężeń); zmieniona scena w tle.

Optyka (poglądowa, sterowana prawdziwymi wagami i aktywacjami ostatniego bloku MLP):
- reflektor j = token j (piksel obrazu owinięty wokół sfery, kolor = średni kolor patcha); 768 kanałów = 768 neuronów, jasność z_jc;
- wiązka kanału c z reflektora j trafia w sferę w punkcie E_jc, załamuje się i biegnie do cząsteczki j w kierunku d_c = P v̂_c
  (P = stały rzut 192D → 3D, trzy główne składowe kierunków v_c w kroku 22 974). Kierunek d_c jest ten sam dla wszystkich reflektorów
  (wagi wspólne), więc w środku sfery wiązki kanału c biegną równolegle (sfera telecentryczna);
- E_jc = p_j − s · d_c, gdzie s > 0 i |E_jc| = R (przecięcie sfery z prostą przez cząsteczkę o kierunku d_c);
- położenie cząsteczki p_j = S · P W₂ (z_j − z̄): rzut sumy jej 768 wektorów siły (względem średniej po tokenach), więc wiązki
  i położenia są spójne. Reflektor j świeci tylko na cząsteczkę j (MLP nie miesza tokenów).
Dane: wagi fc2 bloku 6 z 341 checkpointów (kroki 0–22 974) z reflektor_8_cache.npz i aktywacje 768 neuronów zdjęcia TRAIN 0
w tych krokach; okienka z szescian.npz jak w filmie 7.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_10_reflektory_sfera.py [proba]"""
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
P3 = U_[:, :3].T                                                                    # (3, 192)
PV = np.einsum('ad,tdc->tca', P3, V)                                                # (341, 768, 3)
DIRS = PV / np.linalg.norm(PV, axis=2, keepdims=True)                               # kierunki kanałów d_c(t)
POS_RAW = np.einsum('tjc,tca->tja', Z - Z.mean(1, keepdims=True), PV)               # (341, 600, 3) rzut Σ (z − z̄) v
R = 34.0
R99 = np.percentile(np.linalg.norm(POS_RAW, axis=2), 99, axis=1)                   # (341,) promień chmury w każdym kroku
POS = POS_RAW / R99[:, None, None] * 0.55 * R                                       # skala dopasowana do kroku (rzut widzi na starcie mało)
POW = R99[-1] / R99                                                                  # powiększenie rzutu względem końca
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
pary = np.array([(np.where(np.eye(NK, dtype=bool), 0.0, Vn[t].T @ Vn[t]).min(1) < -0.5).mean() for t in range(NT)])
KOL = kolory_kanalow()[:, :3]
SKALA = 2.0
Zr = Z / SKALA
log(f'wagi {V.shape}, aktywacje {Z.shape}, pary {pary[0]:.0%} → {pary[-1]:.0%}')

# ---- okienka z filmu 7 (rzut płaski i oko), ograniczone do kroków z wagami
G2 = D['g3'][:, :2]; Ql2 = D['Q3'][..., :2]
_, MAPY2 = prokrustes2d(Ql2, G2)
QC2 = np.stack([zastosuj(Ql2[t], MAPY2[t]) for t in range(len(Ql2))])[idx_lat]
LOK2 = np.linalg.norm(QC2 - G2, axis=-1)
LOK_MAX = float(np.percentile(LOK2, 99))
NARZ_LOK = LOK2[:, narz].mean(1) / np.maximum(LOK2[:, ~narz].mean(1), 1e-9)
OKO_F = oko()
OKO = (OKO_F * 255).astype(np.uint8)

# reflektory: obraz 30 × 20 owinięty wokół sfery (kolumna → długość, wiersz → szerokość), kolor = średni kolor patcha
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)       # (kolumna, wiersz)
RR = 50.0
lon = np.radians((SIATKA[:, 0] - 14.5) / 30 * 300.0 + 180.0)
lat = np.radians(-(SIATKA[:, 1] - 9.5) / 20 * 110.0)
REFL = RR * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
KOL_PATCH = OKO_F.reshape(H, 16, W, 16, 3).mean((1, 3)).reshape(TOK, 3)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
porz = np.random.default_rng(3).permutation(TOK) / TOK                              # kolejność zapalania reflektorów
porz[FOK] = 0.0

# ---------------- oś czasu ----------------
FAZY = [(0, 'sfera'), (6, 'jeden'), (15, 'wszystkie'), (25, 'zblizenie'), (33, 'uczenie'), (70, 'koniec'), (76, None)]
T_U0, T_UP, T_U1 = 35.0, 47.0, 69.0
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def indeks(t):
    if t < T_U0:
        return float(NT - 1) if t < 33 else float(NT - 1) * (1 - sm((t - 33) / 2.0))
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


KAM = [(0, -70, 16, 150, None), (6, -40, 18, 120, None), (15, -10, 24, 118, None), (23, 20, 30, 70, None),
       (27, 45, 26, 12, 'fok'), (32, 70, 22, 12, 'fok'), (36, 100, 28, 112, None), (70, 330, 28, 112, None), (76, 350, 24, 130, None)]


def macierz(t, pf):
    for (ta, *pa), (tb, *pb) in zip(KAM[:-1], KAM[1:]):
        if t <= tb:
            u = sm((t - ta) / (tb - ta)); break
    else:
        (ta, *pa), (tb, *pb), u = KAM[-2], KAM[-1], 1.0
    az, el, di = [pa[k] + (pb[k] - pa[k]) * u for k in range(3)]
    if 36 < t < 70:
        el += 10 * np.sin(2 * np.pi * (t - 36) / 34 * 1.5)
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

# wiązki: tryb 0 = w sferze (E_jc → p_j), tryb 1 = na zewnątrz (reflektor j → E_jc)
wiazki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform int tryb; uniform float gain, przyc, wzm, wl, R; uniform int fok;
layout(std430, binding=0) readonly buffer B0 { vec4 RF[]; };
layout(std430, binding=1) readonly buffer B1 { vec4 P[]; };
layout(std430, binding=2) readonly buffer B2 { vec4 DC[]; };
layout(std430, binding=3) readonly buffer B3 { float I[]; };
out float v_t; out float v_i; flat out int v_c;
void main() {
    int id = gl_InstanceID; int j = id / 768; int c = id % 768;
    vec3 p = P[j].xyz; vec3 d = DC[c].xyz;
    float b = dot(p, d); float s = b + sqrt(max(b * b - dot(p, p) + R * R, 0.0));
    vec3 e = p - s * d;
    vec3 a = tryb == 0 ? e : RF[j].xyz; vec3 q = tryb == 0 ? p : e;
    float on = clamp((wl - RF[j].w) * 30.0, 0.0, 1.0);
    float x = I[j * 768 + c] * on * gain;
    if (fok >= 0) x *= (j == fok ? wzm : przyc);
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

buf_RF = ctx.buffer(np.concatenate([REFL, porz[:, None]], 1).astype('f4')); buf_RF.bind_to_storage_buffer(0)
buf_P = ctx.buffer(reserve=TOK * 16); buf_P.bind_to_storage_buffer(1)
buf_D = ctx.buffer(reserve=NK * 16); buf_D.bind_to_storage_buffer(2)
buf_I = ctx.buffer(reserve=TOK * NK * 4); buf_I.bind_to_storage_buffer(3)
buf_K = ctx.buffer(np.concatenate([0.25 + 0.75 * KOL, np.ones((NK, 1))], 1).astype('f4')); buf_K.bind_to_storage_buffer(4)
vao_w = ctx.vertex_array(wiazki, [])
wiazki['R'].value = R


def rysuj(prog, tryb, poz, kol, rozm=None):
    if len(poz) == 0:
        return
    dane_ = [poz.astype('f4'), kol.astype('f4')] + ([rozm.astype('f4')[:, None]] if rozm is not None else [])
    b = ctx.buffer(np.concatenate(dane_, 1).tobytes())
    fmt = '3f 4f 1f' if rozm is not None else '3f 4f'
    v = ctx.vertex_array(prog, [(b, fmt, *(['pos', 'kol', 'rozm'] if rozm is not None else ['pos', 'kol']))])
    v.render(tryb); v.release(); b.release()


# szklana sfera: południki, równoleżniki i obrys
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
    'sfera': ('600 reflektorów za szklaną sferą = 600 patchy obrazu',
              ['Reflektor j = token j (obraz owinięty wokół sfery, kolor = kolor patcha).  Każdy reflektor ma 768 kanałów: 768 neuronów ostatniego bloku MLP.',
               'Wszystkie reflektory są identyczne (wspólne wagi): różnią się tylko tym, jak jasno świeci każdy kanał.']),
    'jeden': ('Jeden reflektor: 768 wiązek, sfera je załamuje, wszystkie trafiają w jedną cząsteczkę',
              ['Wiązka kanału c wchodzi w sferę w punkcie wyznaczonym przez kierunek d_c i biegnie do cząsteczki w tym kierunku.',
               'Jasność wiązki = z_c tej cząsteczki;  kierunek = d_c (rzut kolumny fc2).  Wiązka = wektor siły z_c · v_c.']),
    'wszystkie': ('Wszystkie 600 reflektorów: 600 × 768 = 460 800 wiązek',
                  ['Reflektor j oświetla tylko cząsteczkę j (MLP nie miesza tokenów).  W środku sfery wiązki tego samego kanału biegną równolegle:',
                   'kanał c pcha każdą cząsteczkę w tę samą stronę d_c, tylko z inną siłą z_jc.  To jest „telecentryczna” sfera.']),
    'zblizenie': ('Jedna cząsteczka z bliska: 768 wiązek z 768 kierunków',
                  ['Położenie cząsteczki = suma jej 768 wektorów siły (względem średniej po tokenach), w tym samym rzucie co kierunki wiązek.',
                   'Pasek u dołu = 768 natężeń tej cząsteczki.']),
    'uczenie': ('Uczenie: kanały obracają się we wszystkich reflektorach naraz',
                ['Kierunki d_c zmieniają się jednocześnie dla wszystkich 600 cząsteczek (te same galwa), natężenia z_jc dla każdej osobno.',
                 'Okienka z lewej jak w filmie 7.  Tempo: epoki 0–2 przez 12 s, epoki 2–21 przez 22 s.']),
    'koniec': ('Po 21 epokach', []),
}
KONIEC_TXT = [f'Pary kanałów o przeciwnych kierunkach (cos < −0,5): {pary[0]:.0%} → {pary[-1]:.0%}.   IoU {pl(float(iou[-1]), 2)}.',
              'Szybko (dla każdej cząsteczki osobno) zmieniają się jasności kanałów;  powoli (przez uczenie) obracają się kierunki,',
              'te same we wszystkich 600 reflektorach.  Następny krok: uwaga jako kontrolowany „przeciek” wiązek między cząsteczkami.']

import matplotlib                                                                       # noqa: E402
PLAZMA, MAGMA = matplotlib.colormaps['plasma'], matplotlib.colormaps['magma']
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


iou_x = KW / EP


def nakladka(img, faza, t, kr, ii, zi, mvp):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    if faza == 'koniec':
        pod = KONIEC_TXT
    d.text((WY / 2, 26), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        d.text((WY / 2, 78 + 30 * n), s, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), 'Dane: wagi fc2 bloku 6 z 341 checkpointów (kroki 0–22 974) i aktywacje 768 neuronów zdjęcia TRAIN 0. Kierunki d_c = P v̂_c, '
           'położenia = P W₂ (z − z̄); P = 3 główne składowe.', font=F_M, fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Ilustracja: optyka poglądowa (sfera telecentryczna), kolor wiązki = kanał (jak w aplikacji), błękitne pierścienie = narzędzie.',
           font=F_M, fill=(205, 210, 215, 255))
    if faza in ('sfera', 'jeden') and t > 2:
        q = rzut(mvp, REFL[FOK] * 1.06)
        if q:
            d.text(q, f'reflektor {FOK} (narzędzie)', font=FN(20, True), fill=(220, 240, 255, int(255 * sm((t - 2) / 1.5))), anchor='mb')
    if faza in ('uczenie', 'koniec'):
        d.text((WY - 30, 150), f'epoka {pl(kr / EP, 2)}   krok ' + f'{kr:,.0f}'.replace(',', ' '), font=FN(26, True), fill=(235, 240, 245, 255), anchor='ra')
        d.text((WY - 30, 186), f'pary kanałów o przeciwnych kierunkach: {pary[ii]:.0%}', font=FN(20, True), fill=(255, 200, 120, 255), anchor='ra')
        d.text((WY - 30, 216), f'powiększenie chmury względem końca: ×{pl(float(POW[ii]), 1)}', font=FN(18), fill=(190, 210, 230, 255), anchor='ra')
        x0, y0, sw, sh = WY - 400, HY - 250, 370, 150
        d.rectangle((x0, y0, x0 + sw, y0 + sh), fill=(11, 16, 20, 220), outline=(42, 51, 58, 255))
        px = x0 + iou_x / iou_x[-1] * sw; py = y0 + sh - np.clip(iou, 0, 1) * sh
        d.line(list(zip(px.tolist(), py.tolist())), fill=(74, 240, 255, 255), width=2)
        d.line(list(zip(px.tolist(), (y0 + sh - pary / max(pary.max(), 1e-9) * sh).tolist())), fill=(255, 179, 71, 255), width=2)
        cx = x0 + kr / KW[-1] * sw; d.line((cx, y0, cx, y0 + sh), fill=(255, 255, 255, 230), width=2)
        d.text((x0, y0 - 46), 'IoU', font=F_M, fill=(74, 240, 255, 255))
        d.text((x0, y0 - 24), 'pary przeciwnych kanałów (względnie)', font=F_M, fill=(255, 179, 71, 255))
        d.text((x0 + sw, y0 + sh + 4), 'epoka 21', font=F_M, fill=(180, 185, 190, 255), anchor='ra')
        okienka(img, ii, sm((t - 34) / 2.0))
    if t >= 6:
        x0, y0, hh = (WY - NK) // 2, HY - 150, 70
        d.rectangle((x0 - 10, y0 - 34, x0 + NK + 10, y0 + hh + 8), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
        d.text((x0, y0 - 30), f'768 natężeń cząsteczki {FOK} (wiersz {FOK // 30}, kolumna {FOK % 30}): jasności kanałów jej reflektora, kolor = kanał',
               font=FN(15), fill=(220, 225, 230, 255))
        for k in range(NK):
            h = min(1.0, zi[k] * SKALA / 1.25) * hh
            if h > 0.5:
                d.line((x0 + k, y0 + hh, x0 + k, y0 + hh - h), fill=tuple(int(255 * v) for v in (0.3 + 0.7 * KOL[k])) + (255,))


def klatka(t):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    fi = indeks(t)
    Pp, Dc, I, kr, ii = stan(fi)
    mvp, skala = macierz(t, POS[-1, FOK] if t < 35 else Pp[FOK])
    mvp_b = mvp.T.astype('f4').tobytes()
    buf_P.write(np.concatenate([Pp, np.zeros((TOK, 1))], 1).astype('f4').tobytes())
    buf_D.write(np.concatenate([Dc, np.zeros((NK, 1))], 1).astype('f4').tobytes())
    buf_I.write(I.astype('f4').tobytes())
    ref_on = sm(t / 3.0)
    jeden = sm((t - 6.5) / 2.5)
    wl = 0.0 if t < 15 else sm((t - 15.5) / 7.0) * 1.05                                 # zapalanie reflektorów
    przyc = 1.0 - 0.85 * (sm((t - 25) / 2.0) - sm((t - 32) / 2.0))
    if t < 15:
        przyc, wzm = 0.0, 120.0 * jeden                                                 # tylko reflektor FOK
    else:
        wzm = 1.0 + 30.0 * (1.0 - przyc) / 0.85 if przyc < 1.0 else 1.0
        wzm = max(wzm, 1.0 + 119.0 * (1.0 - sm((t - 15) / 4.0)))                        # płynne przejście z „jednego”
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mvp_b); linie['szer'].value = 1.5
    rysuj(linie, moderngl.LINES, SFERA, np.tile([0.30, 0.55, 0.75, 0.20 * ref_on], (len(SFERA), 1)))
    wiazki['mvp'].write(mvp_b); wiazki['fok'].value = FOK; wiazki['przyc'].value = przyc; wiazki['wzm'].value = wzm
    wiazki['wl'].value = max(wl, 0.034 if t >= 6.5 else -1.0)                     # reflektor FOK ma porz = 0, więc świeci od 6,5 s
    for tryb, g in ((1, 0.35), (0, 1.0)):
        wiazki['tryb'].value = tryb; wiazki['gain'].value = GAIN * g
        vao_w.render(moderngl.LINES, vertices=2, instances=TOK * NK)
    kropki['mvp'].write(mvp_b); kropki['skala'].value = skala
    # reflektory (kwadraty w kolorze patcha), jasność = suma natężeń
    S = I.sum(1); jas = np.clip(S / np.percentile(S, 90), 0.2, 1.6)
    on = np.clip((wl - porz) * 30, 0, 1); on[FOK] = max(on[FOK], jeden)
    kropki['ksztalt'].value = 2
    rysuj(kropki, moderngl.POINTS, REFL, np.concatenate([KOL_PATCH * (0.5 + 0.8 * on * jas)[:, None], np.full((TOK, 1), 0.9 * ref_on)], 1), np.full(TOK, 1.3))
    # cząsteczki: poświata = suma natężeń
    kropki['ksztalt'].value = 0
    ogn = sm((t - 8) / 4)
    if ogn > 0:
        aj = np.clip(S / np.percentile(S, 90), 0, 1.6) * ogn * np.maximum(on, 0.15 if t >= 15 else 0.0)
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
    nakladka(img, faza, t, kr, ii, I[FOK], mvp)
    return img


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    CZASY = np.array([4.0, 11.0, 20.0, 29.0, 40.0, 60.0, 73.0])
ff = None if PROBA else ffmpeg(HERE / 'lasery_10_reflektory_sfera.mp4', FPS)
for n, t in enumerate(CZASY):
    img = klatka(float(t))
    if PROBA:
        img.save(HERE / f'lasery_10_proba_{t:04.1f}.png')
    else:
        ff.stdin.write(img.convert('RGBA').tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(CZASY)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
