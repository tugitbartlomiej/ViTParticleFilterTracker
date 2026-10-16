"""Film dolina 1: dolina pewności i IoU tokenu 226 (środek ramki narzędzia) w czasie uczenia, bieg s47 (806 checkpointów).

Położenie cząsteczki to stan x̃ tokenu (192 wymiary) przed MLP bloku 6, ten sam co na sferze reflektorów w aplikacji. Z x̃ głowa daje
dokładnie pewność p i ramkę: x = x̃ + MLP₆(LN₂ x̃) (768 laserów bloku 6), o = głowa(LN x). Dolina = położenia, w których p ≥ 0,5
i IoU ramki ≥ 0,5. Głowa czyta z położenia tylko 5 kierunków (rząd jakobianu 5), więc zbiór położeń o tej samej odpowiedzi ma
192 − 5 = 187 wymiarów: to dno doliny. Rysujemy przekroje 3D przez cząsteczkę (app/dolina.py, te same liczby co okienko 12):
  lewy panel: osie sfery reflektorów (przekrój leży w rysowanym rzucie), cała chmura 600 cząsteczek, narzędzie, token 226;
  prawy panel: zbliżenie w poprzek doliny: 2 kierunki najmocniej czytane przez głowę (ściany) i 1 kierunek dna (głowa go nie czyta).
Zielona mgła = punkty siatki 40³ nad oboma progami (jaśniej = dalej nad progami), półprzezroczysta powierzchnia = brzeg doliny.
Tempo jak w filmie 14: v = min(0,8 + krok/250, 150) kroków na klatkę; w każdej klatce najbliższy checkpoint (dolina liczy się dla jego wag).
Render: moderngl 1920×1080 z MSAA 4×, napisy PIL na gotowej klatce, kodowanie h264_nvenc. Nowy film; poprzednie zostają nietknięte.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_dolina_1_pewnosc_iou.py [proba]"""
import subprocess
import sys
import time
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2] / 'app'
sys.path.insert(0, str(APP))
import dolina as DL                                                     # noqa: E402
import lasery_dane as LD                                                # noqa: E402

PROBA = sys.argv[1:2] == ['proba']
WY, HY = 1920, 1080
TOKEN, P_MIN, IOU_MIN, G, ZAS = 226, 0.5, 0.5, 40, 1.0
R_SF = LD.R_SFERY
HIST = APP / 'cache_uczenia/lasery/eafb5a973be3d490'
if not (HIST / 'dane.npz').exists():                                    # worktree bez cache: historia z głównego folderu
    HIST = Path(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/app/cache_uczenia/lasery/eafb5a973be3d490')
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

h = LD.Historia(HIST)
L = DL.Liczydlo(h)
kroki = h.kroki.astype(float); IX = np.arange(h.K)
log(f'historia {h.K} checkpointów, zgodność rzutu R² = {L.zgodnosc:.6f}')

# oś czasu jak film 14
ss = [0.0] * 30
s = 0.0
while s < kroki[-1]:
    s = min(s + min(0.8 + s / 250, 150.0), kroki[-1]); ss.append(s)
ss = np.array(ss + [kroki[-1]] * 90)
fr = np.arange(len(ss))
if PROBA:
    fr = fr[[0, 300, 700, 1000, len(ss) - 1]]
log(f'klatek {len(ss)} ({len(ss) / 30:.0f} s)')


# ---------------- macierze kamery ----------------
def persp(fov, asp, n, f):
    t = 1 / np.tan(np.radians(fov) / 2)
    return np.array([[t / asp, 0, 0, 0], [0, t, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]], 'f4')


def kamera(odl, az, el, c=(0, 0, 0)):
    a, e = np.radians(az), np.radians(el)
    oko = np.asarray(c) + odl * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = np.asarray(c) - oko; f /= np.linalg.norm(f)
    r = np.cross(f, (0, 0, 1)); r /= np.linalg.norm(r); u = np.cross(r, f)
    V = np.eye(4, dtype='f4'); V[0, :3], V[1, :3], V[2, :3] = r, u, -f
    V[:3, 3] = -V[:3, :3] @ oko
    return V


def na_ekran(mvp, P, vp):
    """Punkty sceny → piksele obrazu (x, y od góry) w oknie vp = (x0, y0 od dołu, w, h)."""
    q = np.c_[P, np.ones(len(P))] @ mvp.T
    q = q[:, :3] / q[:, 3:]
    return np.c_[vp[0] + (q[:, 0] + 1) / 2 * vp[2], HY - (vp[1] + (q[:, 1] + 1) / 2 * vp[3])]


# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
fb_ms = ctx.framebuffer(ctx.renderbuffer((WY, HY), 4, samples=4))
fb_out = ctx.framebuffer(ctx.renderbuffer((WY, HY), 4))
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
PREM = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA); DOD = (moderngl.ONE, moderngl.ONE)

pkt = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform float rozm; in vec3 p; in vec4 k; out vec4 v_k;
void main() { gl_Position = mvp * vec4(p, 1.0); gl_PointSize = rozm; v_k = k; }''', fragment_shader='''#version 430
uniform int tryb; in vec4 v_k; out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0; float r = length(d);
  if (r > 1.0) discard;
  if (tryb == 0) { o = vec4(v_k.rgb * v_k.a * exp(-r * r * 3.0), 0.0); return; }        // mgła (dodawana)
  if (tryb == 1) { float a = smoothstep(0.55, 0.7, r) * (1.0 - smoothstep(0.85, 1.0, r)) * v_k.a; o = vec4(v_k.rgb * a, a); return; }
  vec3 c = r > 0.72 ? vec3(0.0) : v_k.rgb; float a = (1.0 - smoothstep(0.9, 1.0, r)) * v_k.a; o = vec4(c * a, a); }''')

tri = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp, M; in vec3 p; in vec3 n; out vec3 v_n;
void main() { gl_Position = mvp * (M * vec4(p, 1.0)); v_n = normalize(mat3(M) * n); }''', fragment_shader='''#version 430
uniform vec4 kol; in vec3 v_n; out vec4 o;
void main() { float s = 0.35 + 0.65 * abs(dot(normalize(v_n), normalize(vec3(0.35, 0.55, 0.75)))); o = vec4(kol.rgb * s * kol.a, kol.a); }''')

lin = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; in vec3 p; in vec4 k; out vec4 g_k;
void main() { gl_Position = mvp * vec4(p, 1.0); g_k = k; }''', geometry_shader='''#version 430
layout(lines) in; layout(triangle_strip, max_vertices = 4) out;
uniform vec2 vp; uniform float szer; in vec4 g_k[]; out vec4 f_k;
void main() {
  vec4 p0 = gl_in[0].gl_Position, p1 = gl_in[1].gl_Position;
  vec2 a = p0.xy / p0.w, b = p1.xy / p1.w;
  vec2 d = (b - a) * vp; d = length(d) < 1e-5 ? vec2(1.0, 0.0) : normalize(d);
  vec2 n = vec2(-d.y, d.x) * szer / vp;
  f_k = g_k[0]; gl_Position = vec4((a + n) * p0.w, p0.zw); EmitVertex(); gl_Position = vec4((a - n) * p0.w, p0.zw); EmitVertex();
  f_k = g_k[1]; gl_Position = vec4((b + n) * p1.w, p1.zw); EmitVertex(); gl_Position = vec4((b - n) * p1.w, p1.zw); EmitVertex();
  EndPrimitive(); }''', fragment_shader='''#version 430
in vec4 f_k; out vec4 o; void main() { o = vec4(f_k.rgb * f_k.a, f_k.a); }''')


def rysuj_pkt(mvp, P, K, rozm, tryb):
    if len(P) == 0:
        return
    pkt['mvp'].write(np.ascontiguousarray(mvp.T, 'f4').tobytes()); pkt['rozm'].value = float(rozm); pkt['tryb'].value = tryb
    K = np.broadcast_to(np.asarray(K, 'f4'), (len(P), 4))
    b = ctx.buffer(np.ascontiguousarray(np.c_[P, K], 'f4').tobytes())
    va = ctx.vertex_array(pkt, [(b, '3f 4f', 'p', 'k')]); va.render(moderngl.POINTS); va.release(); b.release()


def rysuj_lin(mvp, S, K, szer, vp):
    """S (n, 2, 3) odcinki, K (n, 4) albo jeden kolor."""
    if len(S) == 0:
        return
    lin['mvp'].write(np.ascontiguousarray(mvp.T, 'f4').tobytes()); lin['szer'].value = float(szer); lin['vp'].value = (vp[2] / 2, vp[3] / 2)
    K = np.repeat(np.broadcast_to(np.asarray(K, 'f4'), (len(S), 4)), 2, 0)
    b = ctx.buffer(np.ascontiguousarray(np.c_[S.reshape(-1, 3), K], 'f4').tobytes())
    va = ctx.vertex_array(lin, [(b, '3f 4f', 'p', 'k')]); va.render(moderngl.LINES); va.release(); b.release()


def rysuj_tri(mvp, M, T, kol):
    if T is None:
        return
    tri['mvp'].write(np.ascontiguousarray(mvp.T, 'f4').tobytes()); tri['M'].write(np.ascontiguousarray(M.T, 'f4').tobytes())
    tri['kol'].value = tuple(kol)
    b = ctx.buffer(np.ascontiguousarray(np.c_[T[0], T[1]], 'f4').tobytes())
    va = ctx.vertex_array(tri, [(b, '3f 3f', 'p', 'n')]); va.render(moderngl.TRIANGLES); va.release(); b.release()


def mgla(d):
    """Punkty siatki nad oboma progami: położenia w układzie przekroju i kolory (jaśniej = dalej nad progami)."""
    q = DL.pole_jakosci(d, P_MIN, IOU_MIN).ravel()
    a = np.linspace(-d['Z'], d['Z'], d['G'], dtype='f4')
    A = np.stack(np.meshgrid(a, a, a, indexing='ij'), -1).reshape(-1, 3)
    A = A + ROZRZUT[:len(A)] * (2 * d['Z'] / (d['G'] - 1))            # stały rozrzut w obrębie oczka: bez mory regularnej siatki
    m = q >= 0; s_ = np.clip(q[m], 0, 1)[:, None]
    K = np.c_[(1 - s_) * (0.05, 0.45, 0.22) + s_ * (0.35, 1.0, 0.75), (0.2 + 0.8 * s_) * 10.0 / d['G'] * 0.9]
    return A[m], K


def szescian(Z):
    r = np.array([[x, y, z] for x in (-Z, Z) for y in (-Z, Z) for z in (-Z, Z)], 'f4')
    return np.array([[r[i], r[j]] for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(r[i] != r[j]) == 1], 'f4')


KOL_OSI = ((1.0, 0.35, 0.3), (0.35, 1.0, 0.45), (0.4, 0.6, 1.0))


def osie(Z):
    S = np.zeros((6, 2, 3), 'f4'); K = np.zeros((6, 4), 'f4')
    for k in range(3):
        S[2 * k, 1, k] = Z; S[2 * k + 1, 1, k] = -Z
        K[2 * k] = (*KOL_OSI[k], 1.0); K[2 * k + 1] = (*KOL_OSI[k], 0.35)
    return S, K


def sfera_druty(R, co=30, n=72):
    t = np.linspace(0, 2 * np.pi, n + 1); seg = []
    for lon in np.radians(np.arange(0, 180, co)):
        p = R * np.stack([np.cos(t) * np.cos(lon), np.cos(t) * np.sin(lon), np.sin(t)], 1); seg += list(zip(p[:-1], p[1:]))
    for lat in np.radians(np.arange(-90 + co, 90, co)):
        p = R * np.stack([np.cos(t) * np.cos(lat), np.sin(t) * np.cos(lat), np.full_like(t, np.sin(lat))], 1); seg += list(zip(p[:-1], p[1:]))
    return np.array(seg, 'f4')


DRUTY = sfera_druty(R_SF)
ROZRZUT = np.random.default_rng(0).uniform(-0.5, 0.5, (G ** 3, 3)).astype('f4')
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_P, F_M = FN(27, True), FN(19), FN(17)
pl = lambda x, k=2: f'{x:.{k}f}'.replace('.', ',')
VP_L, VP_P = (0, 150, 960, 850), (960, 150, 960, 850)              # okna 3D (x0, y0 od dołu, w, h)
PR = persp(32, 960 / 850, 1, 1000)

# ---------------- pętla klatek ----------------
ff = None
if not PROBA:
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WY}x{HY}', '-r', '30', '-i', '-',
                           '-c:v', 'h264_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '19', '-b:v', '0', '-pix_fmt', 'yuv420p',
                           str(HERE / 'dolina_1_pewnosc_iou.mp4')], stdin=subprocess.PIPE)
pam = (-1, None, None, None, None)
for n, f in enumerate(fr):
    kr = ss[f]; i = int(round(np.interp(kr, kroki, IX)))
    if pam[0] != i:
        dS = L.policz(i, TOKEN, 'sfera', G, ZAS); dD = L.policz(i, TOKEN, 'dno', G, ZAS)
        pam = (i, dS, dD, DL.obrys(dS, P_MIN, IOU_MIN), DL.obrys(dD, P_MIN, IOU_MIN))
    _, dS, dD, oS, oD = pam
    az = 25 + 0.12 * f
    fb_ms.use(); ctx.scissor = None; fb_ms.clear(0.012, 0.024, 0.04, 1.0)
    # --- lewy panel: osie sfery, cała chmura i dolina wokół tokenu
    ctx.viewport = VP_L
    k = 0.55 * R_SF / max(float(h.r99[i]), 1e-9)
    mvp = PR @ kamera(150, az, 20)
    ctx.blend_func = PREM
    rysuj_lin(mvp, DRUTY, (0.45, 0.6, 0.75, 0.10), 1.0, VP_L)
    Pt = np.asarray(h.pos[i], 'f4') * k
    M = np.eye(4, dtype='f4'); M[:3, :3] = k * dS['M3']; M[:3, 3] = k * dS['pos0']
    A, K = mgla(dS)
    ctx.blend_func = DOD; rysuj_pkt(mvp, A @ M[:3, :3].T + M[:3, 3], K * (1, 1, 1, 0.7), 5.0, 0)
    ctx.blend_func = PREM
    rysuj_tri(mvp, M, oS, (0.3, 0.95, 0.6, 0.08))
    rysuj_lin(mvp, szescian(dS['Z']) @ M[:3, :3].T + M[:3, 3], (0.6, 0.7, 0.8, 0.3), 1.2, VP_L)
    rysuj_pkt(mvp, Pt, (1.0, 1.0, 1.0, 0.75), 4.0, 2)
    rysuj_pkt(mvp, Pt[h.narz], (0.29, 0.94, 1.0, 1.0), 13.0, 1)
    S_, K_ = osie(dS['Z']); rysuj_lin(mvp, S_ @ M[:3, :3].T + M[:3, 3], K_, 2.2, VP_L)
    rysuj_pkt(mvp, Pt[TOKEN:TOKEN + 1], (1.0, 0.85, 0.3, 1.0), 17.0, 2)
    # --- prawy panel: zbliżenie w poprzek doliny (2 ściany + dno)
    ctx.viewport = VP_P
    sk = 10.0 / dD['Z']; Mp = np.diag([sk, sk, sk, 1.0]).astype('f4')
    mvp2 = PR @ kamera(64, az + 20, 22)
    A, K = mgla(dD)
    ctx.blend_func = DOD; rysuj_pkt(mvp2, A * sk, K, 7.0, 0)
    ctx.blend_func = PREM
    rysuj_tri(mvp2, Mp, oD, (0.3, 0.95, 0.6, 0.12))
    rysuj_lin(mvp2, szescian(10.0), (0.6, 0.7, 0.8, 0.35), 1.4, VP_P)
    S_, K_ = osie(10.0); rysuj_lin(mvp2, S_, K_, 3.0, VP_P)
    rysuj_pkt(mvp2, np.zeros((1, 3), 'f4'), (1.0, 0.85, 0.3, 1.0), 20.0, 2)
    ctx.viewport = (0, 0, WY, HY)
    ctx.copy_framebuffer(fb_out, fb_ms)
    im = Image.fromarray(np.frombuffer(fb_out.read(components=3), 'u1').reshape(HY, WY, 3)[::-1].copy())
    dr = ImageDraw.Draw(im)
    # napisy
    q = DL.pole_jakosci(dD, P_MIN, IOU_MIN)
    dr.text((WY / 2, 14), 'Dolina pewności i IoU: gdzie token 226 (środek ramki narzędzia) dawałby dobrą odpowiedź', font=F_T, fill='#f2f2f2', anchor='mt')
    dr.text((WY / 2, 56), f'krok {kr:,.0f}   epoka {pl(kr / LD.NA_EPOKE)}   pewność tokenu {pl(dS["p0"])}   IoU jego ramki {pl(dS["iou0"])}'
            .replace(',', ' ', 1) if kr >= 1000 else
            f'krok {kr:.0f}   epoka {pl(kr / LD.NA_EPOKE)}   pewność tokenu {pl(dS["p0"])}   IoU jego ramki {pl(dS["iou0"])}',
            font=F_P, fill='#ffd166', anchor='mt')
    dr.text((20, 100), 'Osie sfery reflektorów: cała chmura 600 cząsteczek', font=F_P, fill='#e8e8e8')
    dr.text((980, 100), 'Zbliżenie w poprzek doliny (192 wymiary, przekrój 3D)', font=F_P, fill='#e8e8e8')
    # podpisy osi prawego panelu przy końcach osi
    konce = na_ekran(mvp2, np.eye(3, dtype='f4') * 11.2, VP_P)
    for (x, y), t_, kol in zip(konce, ('ściana 1', 'ściana 2', 'dno'), KOL_OSI):
        dr.text((x, y), t_, font=F_M, fill=tuple(int(255 * c) for c in kol), anchor='mm')
    zs, zc = dS['zas']['dno'], dS['zas']['sciany']
    linie = [
        'Zielone: położenia stanu tokenu, w których pewność ≥ 0,5 i IoU ≥ 0,5. Jaśniej = dalej nad progami. Żółta kropka = gdzie token jest teraz.',
        f'Głowa czyta z położenia tylko {dS["rzad"]} kierunków z 192 (ściany doliny). Pozostałe {192 - dS["rzad"]} to dno: z bliska ruch wzdłuż dna '
        f'nie zmienia odpowiedzi, z daleka dno się zakrzywia.',
        f'Jakość p·IoU spada o połowę po ~{zs:.0f} wzdłuż dna i po ~{zc:.0f} w poprzek (promień chmury {dS["r99"]:.0f}).  '
        f'W prawym sześcianie {(q >= 0).mean():.0%} położeń jest w dolinie.  Lewo: pierścienie = tokeny narzędzia.',
    ]
    for j_, t_ in enumerate(linie):
        dr.text((WY / 2, HY - 128 + 38 * j_), t_, font=F_M, fill='#cfd8dc', anchor='mt')
    if PROBA:
        im.save(HERE / f'dolina_1_proba_{f}.png')
    else:
        ff.stdin.write(im.tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(fr)} (checkpoint {i})')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
