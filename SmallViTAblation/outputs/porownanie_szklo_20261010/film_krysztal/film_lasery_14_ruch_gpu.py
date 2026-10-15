"""Film 14: treść filmu 13 (ruch 600 tokenów TRAIN 0 rozłożony na trzy składowe + atencja, centra na zdjęciu, składowe, IoU), ale
KRÓTSZY (~1 min) i renderowany na GPU. Pierwsze kroki wolno, epoki szybko: v = min(0,8 + krok/250, 150) kroków na klatkę; przy dużej
prędkości atomy, atencja i ramka są uśredniane po kroku klatki (rozmycie ruchu zamiast migotania).
Render: moderngl (OpenGL 4.3) 1920×1080 z MSAA 4×: studnie = gaussy dodawane do tekstury float i tonowane w shaderze, cząstki i pierścienie
= sprite'y, linie = shader geometrii, mapy atencji/centrów = tekstura 30×20 + LUT kolorów; kodowanie h264_nvenc. Matplotlib rysuje TYLKO
raz statyczne tło (osie, krzywe, zdjęcia), PIL tylko 4 krótkie napisy na klatkę. Filmy 2 i 13 zostają nietknięte.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_14_ruch_gpu.py [proba]"""
import subprocess
import sys
import time

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d

from lasery_wspolne import EP, H, HERE, TOK, W, dane, oko, prokrustes2d, zastosuj

PROBA = sys.argv[1:2] == ['proba']
WY, HY = 1920, 1080
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

# ---------------- dane ----------------
D = dane(0)
kroki, narz = D['kroki'].astype(float), D['narz']; K = len(kroki); IX = np.arange(K)
g = D['g3'][:, :2]; Ql, Al = D['Q3'][..., :2], D['A3'][..., :2]
par, mapy = prokrustes2d(Ql, g)
Qc = np.stack([zastosuj(Ql[t], mapy[t]) for t in range(K)]); Ac = np.stack([zastosuj(Al[t], mapy[t]) for t in range(K)])
Gl = np.stack([(g - mapy[t][3]) @ mapy[t][1].T / mapy[t][2] + mapy[t][0] for t in range(K)])
glob = np.linalg.norm(Gl - g, axis=-1).mean(1); lok = np.linalg.norm(Qc - g, axis=-1); vib = np.linalg.norm(Ac - Qc, axis=-1)
narz_lok = lok[:, narz].mean(1) / np.maximum(lok[:, ~narz].mean(1), 1e-9)
AT = np.load(HERE / 'atencja_train0.npz')
atn, ramka, iou, GT = AT['atencja'], AT['ramka'], AT['iou'].astype(float), AT['gt']
d_iou = np.diff(iou, prepend=iou[0])
LOK99, VIB97 = float(np.percentile(lok, 99)), float(np.percentile(vib, 97))

# oś czasu: krok w każdej klatce i prędkość (kroki/klatkę)
ss, vv = [0.0] * 30, [0.0] * 30
s = 0.0
while s < kroki[-1]:
    v = min(0.8 + s / 250, 150.0); s = min(s + v, kroki[-1]); ss.append(s); vv.append(v)
ss, vv = np.array(ss + [kroki[-1]] * 90), np.array(vv + [0.0] * 90)
fr = np.arange(len(ss))
if PROBA:
    fr = fr[[0, 150, 400, 800, 1200, len(ss) - 1]]
log(f'klatek {len(ss)} ({len(ss) / 30:.0f} s)')
P = lambda X: PchipInterpolator(kroki, X, axis=0)
sAc, sQc, sGl, sQl, sAl, sPar, sAt, sRam = (P(X) for X in (Ac, Qc, Gl, Ql, Al, par, atn, ramka))
sCz = P(uniform_filter1d(Qc[:, narz].mean(1), 9, axis=0, mode='nearest'))


def rozmyte(f, kr, v):
    """Średnia z 5 chwil w obrębie kroku klatki (rozmycie ruchu przy szybkim przewijaniu)."""
    if v < 8:
        return f(kr)
    return f(np.clip(kr + v * (np.arange(5) / 4 - 0.5), 0, kroki[-1])).mean(0)


# ---------------- statyczne tło (matplotlib, raz) ----------------
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#03060a', 'text.color': '#e8e8e8'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ramka_idx = np.array([0, W - 1, TOK - 1, TOK - W, 0])


def osie(rect, tytul, xl, yl):
    ax = fig.add_axes(rect); ax.set_facecolor('#03060a'); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s_ in ax.spines.values():
        s_.set_color('#1f2b25')
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_title(tytul, fontsize=12.5, color='#e8e8e8'); return ax


ext = float(np.abs(Ql).max()) + 1.5
a1 = osie((0.01, 0.47, 0.32, 0.42), '1. RUCH GLOBALNY (układ inercjalny): cała sieć naraz', (-ext, ext), (ext * H / W + 1, -ext * H / W - 1))
a1.plot(*g[ramka_idx].T, color='#55606a', lw=1.2, ls='--')
XL, YL = W / 2 + 1.5, H / 2 + 1.5
a2 = osie((0.34, 0.47, 0.32, 0.42), '2. ODKSZTAŁCENIE LOKALNE: studnie odpływają od adresów', (-XL, XL), (YL, -YL))
a2.set_facecolor('#000000')
a3 = osie((0.67, 0.47, 0.32, 0.42), '3. DRGANIA: tokeny wokół centrów studni (kamera idzie za narzędziem)', (-3.6, 3.6), (2.5, -2.5))
a3.set_facecolor('#000000')
img = oko()


def ramka_xy(b):
    return np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]])


def okno_oka(rect, tytul):
    ax = fig.add_axes(rect); ax.set_xticks([]); ax.set_yticks([]); ax.imshow(img, extent=(0, W * 16, H * 16, 0))
    ax.plot(*ramka_xy(GT).T, color='#ffffff', lw=1.2, ls='--', zorder=5)
    ax.set_xlim(0, W * 16); ax.set_ylim(H * 16, 0); ax.set_title(tytul, fontsize=11.5, color='#e8e8e8'); return ax


a4 = okno_oka((0.01, 0.07, 0.24, 0.34), '4. Atencja bloku 6 (√): ile uwagi dostaje patch')
a4.text(0.0, -0.06, 'biała przerywana = prawdziwa ramka,  błękitna = ramka modelu', transform=a4.transAxes, fontsize=9, color='#e8e8e8', zorder=9)
a5 = okno_oka((0.26, 0.07, 0.24, 0.34), ' ')
tk = [0, 300, 1000, EP * 5, EP * 20, EP * 60]
tki = [int(np.argmin(abs(kroki - k))) for k in tk]; tkl = ['0', '300', '1k', 'e5', 'e20', 'e60']
a6 = fig.add_axes((0.545, 0.09, 0.20, 0.30)); a6.set_facecolor('#0b1014')
a6.plot(glob, color='#ffd166', lw=1.2, label='1. globalny'); a6.plot(lok.mean(1), color='#c264ff', lw=1.2, label='2. lokalny')
a6.plot(vib.mean(1), color='#ff7a45', lw=1.0, label='3. drgania')
a6.set_xticks(tki); a6.set_xticklabels(tkl); a6.set_xlim(-8, K + 8)
a6.tick_params(colors='#cfd3d8', labelsize=9); a6.grid(True, color='#22292f'); a6.set_ylabel('średnio [komórki siatki]', color='#cfd3d8')
a6.legend(fontsize=9, loc='upper left', facecolor='#161e24', edgecolor='#2a333a', labelcolor='#e8e8e8')
a6.set_title('6. Wielkość trzech składowych ruchu', fontsize=12, color='#e8e8e8')
a7 = fig.add_axes((0.785, 0.09, 0.18, 0.30)); a7.set_facecolor('#0b1014')
a7b = a7.twinx()
a7b.bar(IX, d_iou, width=1.0, color=np.where(d_iou >= 0, '#39ff7a', '#ff4d6d'), alpha=0.55)
lim = float(np.percentile(np.abs(d_iou), 99.5)) * 1.1
a7b.set_ylim(-lim, lim); a7b.tick_params(colors='#7f8a93', labelsize=8); a7b.set_ylabel('zmiana IoU od poprzedniego checkpointu', color='#7f8a93', fontsize=9)
a7.plot(iou, color='#4af0ff', lw=1.3, zorder=3)
a7.set_zorder(a7b.get_zorder() + 1); a7.patch.set_visible(False); a7b.set_facecolor('#0b1014')
a7.set_xticks(tki); a7.set_xticklabels(tkl); a7.set_ylim(0, 1); a7.set_xlim(-8, K + 8)
a7.tick_params(colors='#cfd3d8', labelsize=9); a7.grid(True, color='#22292f'); a7.set_ylabel('IoU', color='#4af0ff')
a7.set_title('7. IoU z checkpointu na checkpoint', fontsize=12, color='#e8e8e8')
fig.text(0.5, 0.92, 'Film 14: czym są przesunięcia tokenów — trzy składowe ruchu, atencja, centra na zdjęciu i IoU', ha='center', fontsize=12.5, color='#ff9a5c')
fig.canvas.draw()
TLO = np.asarray(fig.canvas.buffer_rgba()).copy()


def okno(ax):
    """Prostokąt osi w pikselach GL (x0, y0 od dołu, w, h) i przekształcenie dane → NDC."""
    x0, y0, w, h = ax.bbox.bounds
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    sx, sy = w / (xb - xa), h / (yb - ya)
    return dict(sc=(int(round(x0)), int(round(y0)), int(round(w)), int(round(h))), w=w, h=h, xa=xa, ya=ya, sx=sx, sy=sy, x0=x0, y0=y0)


def tr(o, xl=None, yl=None):
    """uniform tr (sx, ox, sy, oy): ndc = p·s + o; opcjonalnie inne granice danych (kamera panelu 3)."""
    xa, ya, sx, sy = o['xa'], o['ya'], o['sx'], o['sy']
    if xl is not None:
        xa, sx = xl[0], o['w'] / (xl[1] - xl[0]); ya, sy = yl[0], o['h'] / (yl[1] - yl[0])
    return (2 * sx / WY, 2 * (o['x0'] - xa * sx) / WY - 1, 2 * sy / HY, 2 * (o['y0'] - ya * sy) / HY - 1)


O = {k: okno(a) for k, a in zip('1234567', (a1, a2, a3, a4, a5, a6, a7))}
TYT5 = (a5.title.get_window_extent().x0 + a5.title.get_window_extent().width / 2, HY - a5.title.get_window_extent().y1)
plt.close(fig)

# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
fb_ms = ctx.framebuffer(ctx.renderbuffer((WY, HY), 4, samples=4))
fb_out = ctx.framebuffer(ctx.renderbuffer((WY, HY), 4))
tex_I = ctx.texture((WY, HY), 1, dtype='f4'); fb_I = ctx.framebuffer([tex_I])
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
PREM = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

kropki = ctx.program(vertex_shader='''#version 430
uniform vec4 tr; in vec2 pos; in vec4 kol; in float rozm; out vec4 v_k; out float v_r;
void main() { gl_Position = vec4(pos.x * tr.x + tr.y, pos.y * tr.z + tr.w, 0, 1); gl_PointSize = rozm; v_k = kol; v_r = rozm; }''',
                     fragment_shader='''#version 430
uniform int tryb; uniform float grub; uniform vec3 obw; in vec4 v_k; in float v_r; out vec4 o;
void main() {
  float R = v_r * 0.5, r = length(gl_PointCoord * 2.0 - 1.0) * R;
  if (tryb == 2) { float s = v_r / 6.0; o = vec4(v_k.a * exp(-r * r / (2.0 * s * s)), 0, 0, 1); return; }   // studnia: natężenie
  float a = clamp(R - 0.5 - r, 0.0, 1.0); vec3 c = v_k.rgb;
  if (tryb == 1) a *= clamp(r - (R - 0.5 - grub), 0.0, 1.0);                                            // pierścień
  if (tryb == 3) c = mix(c, obw, clamp(r - (R - 0.5 - grub), 0.0, 1.0));                                 // dysk z obwódką
  a *= v_k.a; o = vec4(c * a, a); }''')

linie = ctx.program(vertex_shader='''#version 430
uniform vec4 tr; in vec2 pos; in vec4 kol; out vec4 g_k;
void main() { gl_Position = vec4(pos.x * tr.x + tr.y, pos.y * tr.z + tr.w, 0, 1); g_k = kol; }''', geometry_shader='''#version 430
layout(lines) in; layout(triangle_strip, max_vertices = 4) out;
uniform vec2 vp; uniform float szer; in vec4 g_k[]; out vec4 f_k;
void main() {
  vec4 p0 = gl_in[0].gl_Position, p1 = gl_in[1].gl_Position;
  vec2 d = (p1.xy - p0.xy) * vp; d = length(d) < 1e-5 ? vec2(1.0, 0.0) : normalize(d);
  vec2 n = vec2(-d.y, d.x) * szer / vp;
  f_k = g_k[0]; gl_Position = p0 + vec4(n, 0, 0); EmitVertex(); gl_Position = p0 - vec4(n, 0, 0); EmitVertex();
  f_k = g_k[1]; gl_Position = p1 + vec4(n, 0, 0); EmitVertex(); gl_Position = p1 - vec4(n, 0, 0); EmitVertex();
  EndPrimitive(); }''', fragment_shader='''#version 430
in vec4 f_k; out vec4 o; void main() { o = vec4(f_k.rgb * f_k.a, f_k.a); }''')
linie['vp'].value = (WY / 2, HY / 2)

obraz = ctx.program(vertex_shader='''#version 430
uniform vec4 tr; in vec2 pos; in vec2 uv; out vec2 v_uv;
void main() { gl_Position = vec4(pos.x * tr.x + tr.y, pos.y * tr.z + tr.w, 0, 1); v_uv = uv; }''', fragment_shader='''#version 430
uniform sampler2D tex; uniform sampler2D lut; uniform int tryb; uniform float alfa, vmax; in vec2 v_uv; out vec4 o;
void main() {
  if (tryb == 0) { vec4 c = texture(tex, v_uv); o = vec4(c.rgb * c.a, c.a); return; }                   // obraz RGBA
  float u = clamp(texture(tex, v_uv).r / vmax, 0.0, 1.0);                                                 // mapa + LUT
  o = vec4(texture(lut, vec2(u, 0.5)).rgb * alfa, alfa); }''')

ton = ctx.program(vertex_shader='''#version 430
in vec2 pos; void main() { gl_Position = vec4(pos, 0, 1); }''', fragment_shader='''#version 430
uniform sampler2D I; out vec4 o;
void main() {
  float i = min(texelFetch(I, ivec2(gl_FragCoord.xy), 0).r, 1.4);
  vec3 L = 0.55 * i * vec3(0.22, 1.0, 0.48) + 0.25 * i * i * i * vec3(0.75, 1.0, 0.85);
  o = vec4(clamp(L, 0.0, 1.0), 0.0); }''')
pelny = ctx.vertex_array(ton, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4')), '2f', 'pos')])


def lut(nazwa):
    t = ctx.texture((256, 1), 3, (plt.get_cmap(nazwa)(np.linspace(0, 1, 256))[:, :3] * 255).astype('u1').tobytes())
    t.repeat_x = t.repeat_y = False; return t


LUT = {n: lut(n) for n in ('inferno', 'magma', 'plasma')}
PLASMA = plt.get_cmap('plasma')


def tekstura(rgba):
    t = ctx.texture((rgba.shape[1], rgba.shape[0]), 4, np.ascontiguousarray(rgba[::-1]).tobytes())
    t.repeat_x = t.repeat_y = False; return t


def prostokat(x0, y0, x1, y1, u0=0, v0=0, u1=1, v1=1):
    return np.array([[x0, y0, u0, v0], [x1, y0, u1, v0], [x0, y1, u0, v1], [x1, y1, u1, v1]], 'f4')


VB_IMG = ctx.buffer(reserve=4 * 4 * 4)
VA_IMG = ctx.vertex_array(obraz, [(VB_IMG, '2f 2f', 'pos', 'uv')])
TEX_TLO = tekstura(TLO)
TEX_MAPA = ctx.texture((W, H), 1, dtype='f4'); TEX_MAPA.filter = (moderngl.LINEAR, moderngl.LINEAR); TEX_MAPA.repeat_x = TEX_MAPA.repeat_y = False
NDC = (2 / WY, -1, 2 / HY, -1)                                       # piksele GL (od dołu) → NDC


def rysuj_obraz(tex, tr_, rect, tryb=0, lut_=None, alfa=1.0, vmax=1.0, uv=(0, 0, 1, 1)):
    VB_IMG.write(prostokat(*rect, *uv).tobytes())
    obraz['tr'].value = tr_; obraz['tryb'].value = tryb; obraz['alfa'].value = alfa; obraz['vmax'].value = vmax
    tex.use(0); obraz['tex'].value = 0
    if lut_ is not None:
        lut_.use(1); obraz['lut'].value = 1
    VA_IMG.render(moderngl.TRIANGLE_STRIP)


def rysuj_kropki(tr_, pos, kol, rozm, tryb=0, grub=1.0, obw=(1, 1, 1)):
    n = len(pos)
    kol = np.broadcast_to(np.asarray(kol, 'f4'), (n, 4)); rozm = np.broadcast_to(np.asarray(rozm, 'f4'), (n,))
    b = ctx.buffer(np.concatenate([pos.astype('f4'), kol, rozm[:, None]], 1).tobytes())
    va = ctx.vertex_array(kropki, [(b, '2f 4f 1f', 'pos', 'kol', 'rozm')])
    kropki['tr'].value = tr_; kropki['tryb'].value = tryb; kropki['grub'].value = grub; kropki['obw'].value = obw
    va.render(moderngl.POINTS); va.release(); b.release()


def rysuj_linie(tr_, seg, kol, szer):
    """seg (n,2,2): odcinki w danych; kol (4,) albo (n,4) albo (n,2,4)."""
    n = len(seg)
    if n == 0:
        return
    kol = np.asarray(kol, 'f4')
    kol = np.broadcast_to(kol if kol.ndim == 3 else (kol[:, None] if kol.ndim == 2 else kol[None, None]), (n, 2, 4))
    b = ctx.buffer(np.concatenate([seg.astype('f4'), kol], 2).tobytes())
    va = ctx.vertex_array(linie, [(b, '2f 4f', 'pos', 'kol')])
    linie['tr'].value = tr_; linie['szer'].value = szer
    va.render(moderngl.LINES); va.release(); b.release()


def lamana(p):
    return np.stack([p[:-1], p[1:]], 1)


def przerywana(p, kreska=0.35):
    out = []
    for a, b in zip(p[:-1], p[1:]):
        L = np.linalg.norm(b - a); n = max(int(L / kreska), 1)
        u = np.linspace(0, 1, n + 1)
        out += [np.stack([a + (b - a) * u[i], a + (b - a) * u[i + 1]]) for i in range(0, n, 2)]
    return np.array(out)


def strzalki(a, b, glow=0.3, kat=25):
    """Odcinki trzonu i grotu strzałek a → b (dane)."""
    d = b - a; L = np.linalg.norm(d, axis=1, keepdims=True); u = d / np.maximum(L, 1e-9)
    gl_ = np.minimum(glow, 0.35 * L); c, s_ = np.cos(np.radians(kat)), np.sin(np.radians(kat))
    l1 = b - gl_ * np.stack([u[:, 0] * c - u[:, 1] * s_, u[:, 0] * s_ + u[:, 1] * c], 1)
    l2 = b - gl_ * np.stack([u[:, 0] * c + u[:, 1] * s_, -u[:, 0] * s_ + u[:, 1] * c], 1)
    return np.concatenate([np.stack([a, b], 1), np.stack([b, l1], 1), np.stack([b, l2], 1)])


FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_P, F_M = FN(25), FN(16), FN(15)


def napis(tekst, font, kolor, x, y, wyr='l'):
    """Napis PIL w pikselach obrazu (y od góry); wyr = 'l' / 'c' / 'r'."""
    l_, t_, r_, b_ = font.getbbox(tekst); w, h = r_ - l_ + 4, b_ - t_ + 6
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); ImageDraw.Draw(im).text((2 - l_, 3 - t_), tekst, font=font, fill=kolor)
    x0 = x - (w / 2 if wyr == 'c' else w if wyr == 'r' else 0)
    t = tekstura(np.asarray(im)); rysuj_obraz(t, NDC, (x0, HY - y - h, x0 + w, HY - y)); t.release()


def scissor(o):
    ctx.scissor = o['sc']


# stałe geometryczne paneli
PX2 = O['2']['sx']; SIG = 0.26
PX3 = O['3']['w'] / 7.2
T1, T2, T4, T5, T6, T7 = (tr(O[k]) for k in '124567')
cz0 = sCz(0.0)

# ---------------- pętla klatek ----------------
ff = None
if not PROBA:
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WY}x{HY}', '-r', '30', '-i', '-',
                           '-vf', 'vflip', '-c:v', 'h264_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '19', '-b:v', '0', '-pix_fmt', 'yuv420p',
                           str(HERE / 'lasery_14_ruch_gpu.mp4')], stdin=subprocess.PIPE)
hist = []
for n, f in enumerate(fr):
    kr, v = ss[f], vv[f]; ti = float(np.interp(kr, kroki, IX)); tj = int(round(ti))
    ac = rozmyte(sAc, kr, v); qc, gl, ql, p = sQc(kr), sGl(kr), sQl(kr), sPar(kr); al = rozmyte(sAl, kr, v)
    zx, zy = sCz(kr); XL3, YL3 = (zx - 3.6, zx + 3.6), (zy + 2.5, zy - 2.5); T3 = tr(O['3'], XL3, YL3)
    # 1) studnie → tekstura float (panele 2 i 3)
    fb_I.use(); ctx.scissor = None; fb_I.clear(); ctx.blend_func = moderngl.ONE, moderngl.ONE
    scissor(O['2']); rysuj_kropki(T2, qc, (0, 0, 0, 1), 6 * SIG * PX2, tryb=2)
    scissor(O['3']); rysuj_kropki(T3, qc, (0, 0, 0, 1), 6 * SIG * PX3, tryb=2)
    # 2) obraz
    fb_ms.use(); ctx.scissor = None; fb_ms.clear(0, 0, 0, 1); ctx.blend_func = PREM
    rysuj_obraz(TEX_TLO, NDC, (0, 0, WY, HY))
    ctx.blend_func = moderngl.ONE, moderngl.ONE; tex_I.use(0); ton['I'].value = 0
    for k in '23':
        scissor(O[k]); pelny.render(moderngl.TRIANGLE_STRIP)
    ctx.blend_func = PREM
    # panel 1
    scissor(O['1'])
    rysuj_linie(T1, lamana(gl[ramka_idx]), (0.22, 1.0, 0.48, 1.0), 2.6)
    rysuj_kropki(T1, ql, (0.22, 1.0, 0.48, 0.45), 3.0); rysuj_kropki(T1, al, (1, 1, 1, 0.8), 3.0)
    rysuj_linie(T1, strzalki(g.mean(0)[None], gl.mean(0)[None], 0.4), (1.0, 0.82, 0.4, 1.0), 2.4)
    # panel 2
    scissor(O['2'])
    rysuj_kropki(T2, g, (0.33, 0.38, 0.42, 1.0), 2.5)
    dl = np.linalg.norm(qc - g, axis=1)
    kol_s = PLASMA(np.clip(dl / LOK99, 0, 1)).astype('f4'); kol_s[:, 3] = 0.8
    rysuj_linie(T2, strzalki(g, qc, 0.28), np.concatenate([kol_s] * 3), 1.0)
    rysuj_kropki(T2, qc[narz], (0.29, 0.94, 1.0, 1.0), 9.0, tryb=1, grub=1.2)
    rysuj_linie(T2, przerywana(ramka_xy((XL3[0], YL3[1], XL3[1], YL3[0])), 0.3), (1, 1, 1, 0.9), 1.2)
    # panel 3
    scissor(O['3'])
    hist.append((kr, ac)); hist = [h for h in hist[-36:] if kr - h[0] <= 200]
    if len(hist) > 1:
        Hh = np.array([h[1] for h in hist])
        seg = np.stack([Hh[:-1], Hh[1:]], 2).reshape(-1, 2, 2)
        a_ = np.repeat(np.linspace(0.04, 0.7, len(Hh) - 1), TOK)
        kk = np.zeros((len(seg), 4), 'f4'); kk[:] = (1.0, 0.82, 0.4, 1.0); kk[:, 3] = a_
        rysuj_linie(T3, seg, kk, 1.6)
    rysuj_linie(T3, np.stack([ac, qc], 1), (1, 1, 1, 0.45), 1.1)
    kol_a = PLASMA(np.clip(np.linalg.norm(ac - qc, axis=1) / VIB97, 0, 1)).astype('f4')
    rysuj_kropki(T3, ac, kol_a, 15.0, tryb=3, grub=0.8)
    rysuj_kropki(T3, ac[narz], (0.29, 0.94, 1.0, 1.0), 25.0, tryb=1, grub=1.8)
    # panel 4: atencja (√) + ramka modelu
    scissor(O['4'])
    m4 = np.maximum(rozmyte(sAt, kr, v), 0); TEX_MAPA.write(np.sqrt(m4 / max(m4.max(), 1e-12)).reshape(H, W).astype('f4').tobytes())
    rysuj_obraz(TEX_MAPA, T4, (0, 0, W * 16, H * 16), tryb=1, lut_=LUT['inferno'], alfa=0.62, vmax=1.0)
    rysuj_linie(T4, lamana(ramka_xy(rozmyte(sRam, kr, v))), (0.29, 0.94, 1.0, 1.0), 2.6)
    # panel 5: przesunięcie centrów
    scissor(O['5'])
    TEX_MAPA.write(dl.reshape(H, W).astype('f4').tobytes())
    rysuj_obraz(TEX_MAPA, T5, (0, 0, W * 16, H * 16), tryb=1, lut_=LUT['magma'], alfa=0.6, vmax=LOK99)
    # panele 6, 7: kursory
    iou_t = float(np.interp(ti, IX, iou))
    scissor(O['6']); rysuj_linie(T6, np.array([[[ti, -10], [ti, 10]]]), (1, 1, 1, 1), 1.6)
    scissor(O['7']); rysuj_linie(T7, np.array([[[ti, 0], [ti, 1]]]), (1, 1, 1, 1), 1.6)
    rysuj_kropki(T7, np.array([[ti, iou_t]]), (1, 1, 1, 1), 11.0)
    # napisy
    ctx.scissor = None
    napis(f'krok {kr:,.0f}   epoka {kr / EP:5.2f}   IoU {iou_t:.2f}'.replace(',', ' '), F_T, '#e8e8e8', WY / 2, 26, 'c')
    o1 = O['1']; napis(f'przesunięcie {np.hypot(*p[:2]):.2f} kom.   obrót {p[2]:+.1f}°   skala ×{p[3]:.3f}', F_M, '#ffd166',
                       o1['x0'] + 10, HY - o1['y0'] - 30)
    napis(f'5. Gdzie centra odpłynęły najdalej; narzędzie/reszta ×{np.interp(ti, IX, narz_lok):.2f}', F_P, '#e8e8e8', TYT5[0], TYT5[1] - 2, 'c')
    o7 = O['7']; napis(f'IoU {iou_t:.2f}   zmiana {d_iou[tj]:+.2f}', F_P, '#ffffff', o7['x0'] + o7['w'] - 8, HY - o7['y0'] - 30, 'r')
    ctx.copy_framebuffer(fb_out, fb_ms)
    if PROBA:
        a = np.frombuffer(fb_out.read(components=3), 'u1').reshape(HY, WY, 3)[::-1]
        Image.fromarray(a).save(HERE / f'lasery_14_proba_{f}.png')
    else:
        ff.stdin.write(fb_out.read(components=3))
        if n % 300 == 0:
            log(f'klatka {n}/{len(fr)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
