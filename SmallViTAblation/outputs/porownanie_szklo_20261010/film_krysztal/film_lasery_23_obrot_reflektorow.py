"""Film 23: strzałki obrotu przy reflektorach — gdzie w układzie laserów siedzi gradient.

Nowy film na geometrii filmu 12 (768 stałych reflektorów na zewnętrznej sferze, 600 wiązek każdy, jedno załamanie na szklanej sferze),
ale na biegu MalyViT_35000_od_zera_s47 (806 checkpointów, 60 epok), na którym są przeliczone ZASADY i dokumenty.
- Reflektor c stoi; przy nim igła = kierunek d_c = P v̂_c, w którym jego 600 wiązek pcha cząsteczki (v_c = kolumna fc2 bloku 6).
- Dwie strzałki na końcu igły = obrót d_c w czasie uczenia: kierunek = rzut zmiany v̂_c na 3D (część prostopadła do igły), długość =
  prędkość kątowa v̂_c w pełnych 192 wymiarach [stopnie na 100 kroków], skala logarytmiczna. Cienka biała: checkpointy odległe
  o ≥ 100 kroków (drganie); gruba kolorowa: obrót netto w ≥ 2000 krokach (dryf). To dokładnie to, co zrobiło uczenie (gradient
  przepuszczony przez AdamW). Pomiar (2026-10-13): lr stały po rozgrzewce, więc drganie ma prawie stałą wielkość (~3–4°/100 kroków),
  a dryf zwalnia; kąt ~ odstęp^α, α ≈ 0,6 na początku i ≈ 0,4 po 40 epokach (starzenie, jak w szkle).
- Skąd obrót: ∂L/∂v_c = Σ_j z_jc δ_j (δ_j = poprawka, którą głowa odsyła do cząsteczki j). AdamW zamienia sumę gradientów z batcha w krok
  (bezwładność = pierwszy moment, hamulec = drugi moment); weight decay (sprężyna) mnoży v_c przez liczbę < 1, czyli tylko je skraca,
  nie obraca. Obrót igły pochodzi więc wyłącznie z gradientu. Checkpointy nie zawierają stanu AdamW, dlatego film pokazuje obrót
  rzeczywisty, a nie sam gradient.
- Wiązki ze znakiem: jasność z_jc / 2, ujemna (GELU < 0) błękitna; położenie cząsteczki p_j = P W₂ (z_j − z̄) (film 12).
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_23_obrot_reflektorow.py [dane | proba]"""
import subprocess
import sys
import time

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment

from lasery_wspolne import EP, HERE, TOK, dane, oko

TRYB = sys.argv[1] if len(sys.argv) > 1 else ''
PROBA = TRYB == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
FPS, WY, HY, SS = 30, 1920, 1080, 2
NK = 768
ROOT = HERE.parents[2]
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
CACHE = HERE / 'obrot_reflektorow_23_cache.npz'
OKNO_DRGANIE, OKNO_DRYF = 100, 2000                                                 # obrót w ostatnich 100 krokach (drganie) i netto w 2000 (dryf)
sys.path.insert(0, str(ROOT / 'app'))
from wspolne import kierunki, kolory_kanalow                                        # noqa: E402


# ---------------- dane: wagi fc1/fc2 bloku 6 i natężenia z_jc zdjęcia TRAIN 0 we wszystkich checkpointach ----------------
def buduj():
    import torch
    sys.path.insert(0, str(KOD))
    import model as M
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    xin = torch.from_numpy((oko() * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float().to(dev) / 255.
    net = M.MalyViT().to(dev).eval()
    zap = {}
    net.bloki[5].mlp.act.register_forward_hook(lambda m, i, o: zap.__setitem__('z', o.detach()))
    net.bloki[5].mlp.fc2.register_forward_hook(lambda m, i, o: zap.__setitem__('y', o.detach()))
    KW = np.array(sorted(int(f.name[1:6]) for f in WAGI.glob('k*.pt')))
    NT = len(KW)
    Vn = np.zeros((NT, 192, NK), np.float32); Wn = np.zeros((NT, NK, 192), np.float32)
    DL = np.zeros((NT, 2), np.float32); Z = np.zeros((NT, TOK, NK), np.float16); BL = np.zeros(NT)
    POSR192 = np.zeros((NT, TOK, 192), np.float32)
    for n, k in enumerate(KW):
        s = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
        net.load_state_dict(s)
        with torch.no_grad():
            net(xin)
        z = zap['z'][0].float(); V = s['bloki.5.mlp.fc2.weight'].float().to(dev)
        y = zap['y'][0].float()
        BL[n] = float((z @ V.T + s['bloki.5.mlp.fc2.bias'].float().to(dev) - y).norm() / y.norm())   # kontrola: wyjście MLP = Σ_c z_jc v_c + b₂
        zc = z - z.mean(0, keepdim=True)
        POSR192[n] = (zc @ V.T).cpu().numpy()                                       # W₂ (z_j − z̄), b₂ znosi się
        Z[n] = z.cpu().numpy().astype(np.float16)
        Vc = V.cpu().numpy(); Wc = s['bloki.5.mlp.fc1.weight'].float().numpy()
        DL[n] = np.median(np.linalg.norm(Vc, axis=0)), np.median(np.linalg.norm(Wc, axis=1))
        Vn[n] = Vc / np.linalg.norm(Vc, axis=0, keepdims=True); Wn[n] = Wc / np.linalg.norm(Wc, axis=1, keepdims=True)
        if n % 100 == 0:
            log('checkpoint', n, k, f'kontrola wyjścia MLP {BL[n]:.1e}')
    # rzut 3D jak w filmie 12: 3 główne składowe kierunków v̂_c na końcu uczenia
    U_, sv, _ = np.linalg.svd(Vn[-1] - Vn[-1].mean(1, keepdims=True), full_matrices=False)
    P3 = U_[:, :3].T
    UDZ = float((sv[:3] ** 2).sum() / (sv ** 2).sum())
    PV = np.einsum('ad,tdc->tca', P3, Vn)
    DIRS = PV / np.linalg.norm(PV, axis=2, keepdims=True)                           # kierunek igły d_c (normalizacja nie zależy od |v_c|)
    POS_RAW = np.einsum('tjd,ad->tja', POSR192, P3)                                 # P W₂ (z − z̄)
    kat = lambda X, Y, os_: np.degrees(np.arccos(np.clip((X * Y).sum(os_), -1, 1)))
    wyn = {}
    for nz, okno in (('1', OKNO_DRGANIE), ('2', OKNO_DRYF)):
        # pary checkpointów odległych o ≥ okno kroków (do przodu; przy końcu uczenia wstecz)
        b = np.minimum(np.searchsorted(KW, KW + okno), NT - 1)
        a = np.minimum(np.arange(NT), np.searchsorted(KW, KW[b] - okno, side='right') - 1)
        dk = (KW[b] - KW[a]) / 100.0
        tan = DIRS[b] - DIRS[a]
        tan -= (tan * DIRS).sum(2, keepdims=True) * DIRS                            # część prostopadła do igły (obrót, nie długość)
        tan /= np.maximum(np.linalg.norm(tan, axis=2, keepdims=True), 1e-12)
        wyn['OMV' + nz] = (kat(Vn[a], Vn[b], 1) / dk[:, None]).astype(np.float32)   # (NT, 768) stopnie na 100 kroków, w 192D
        wyn['OMW' + nz] = (kat(Wn[a], Wn[b], 2) / dk[:, None]).astype(np.float32)
        wyn['TAN' + nz] = tan.astype(np.float16)
    # kąt w funkcji odstępu: wykładnik α w kąt ~ odstęp^α (1 = dryf, 0,5 = błądzenie, mniej = drganie w miejscu)
    ix = lambda k: int(np.argmin(np.abs(KW - k)))
    ALFA = np.array([[s_, np.log(np.median(kat(Vn[ix(s_)], Vn[ix(s_ + 3000)], 0)) / np.median(kat(Vn[ix(s_)], Vn[ix(s_ + 300)], 0))) / np.log(10)]
                     for s_ in (300, 5 * EP, 20 * EP, 40 * EP)])
    KATV0 = kat(Vn[:1], Vn, 1); KATW0 = kat(Wn[:1], Wn, 2)                         # obrót od kroku 0
    np.savez(CACHE, KW=KW, DIRS=DIRS.astype(np.float32), POS_RAW=POS_RAW.astype(np.float32), Z=Z, P3=P3, UDZ=UDZ,
             KATV0=KATV0.astype(np.float16), KATW0=KATW0.astype(np.float16), DL=DL, BL=BL, ALFA=ALFA, **wyn)


if not CACHE.exists() or TRYB == 'dane':
    buduj()
_c = np.load(CACHE)
KW, DIRS = _c['KW'], _c['DIRS']
TAN1, TAN2, OMV1, OMV2 = _c['TAN1'].astype(np.float32), _c['TAN2'].astype(np.float32), _c['OMV1'], _c['OMV2']
POS_RAW, Z, UDZ, KATV0, KATW0, DL, BL = _c['POS_RAW'], _c['Z'], float(_c['UDZ']), _c['KATV0'].astype(np.float32), _c['KATW0'].astype(np.float32), _c['DL'], _c['BL']
ALFA = _c['ALFA']
NT = len(KW)
log(f'dane: {NT} checkpointów, kroki {KW[0]}–{KW[-1]}; maks. błąd „wyjście MLP = Σ z v + b₂” {BL.max():.1e}; P3 niesie {UDZ:.0%} wariancji v̂_c')
MED1, MED2 = np.median(OMV1, 1), np.median(OMV2, 1)
_ix = [0, int(np.searchsorted(KW, 300)), int(np.searchsorted(KW, EP)), int(np.searchsorted(KW, 20 * EP)), NT - 1]
log('mediana prędkości v̂_c [°/100 kroków] w krokach', KW[_ix], ': drganie (100)', np.round(MED1[_ix], 2), ' dryf (2000)', np.round(MED2[_ix], 3))
log('wykładnik α (kąt ~ odstęp^α, 300 → 3000 kroków) od kroków', ALFA[:, 0].astype(int), ':', np.round(ALFA[:, 1], 2))
if TRYB == 'dane':
    sys.exit(0)

# ---------------- geometria (film 12) ----------------
R, RR = 34.0, 50.0
R99 = np.percentile(np.linalg.norm(POS_RAW, axis=2), 99, axis=1)
R99s = np.exp(uniform_filter1d(np.log(R99), 9, mode='nearest'))                    # skala chmury dopasowana do kroku, wygładzona
POS = POS_RAW / R99s[:, None, None] * 0.55 * R
FIB = kierunki(False).astype(np.float64)
_r, _cc = linear_sum_assignment(FIB @ DIRS[-1].T)                                   # miejsce najbliżej za −d_c w 60. epoce
SLOT = np.empty(NK, int); SLOT[_cc] = _r
RFIX = RR * FIB[SLOT]
KOL = kolory_kanalow()[:, :3][SLOT]
AKT = np.abs(Z[-1].astype(np.float32)).mean(0)
C1 = int(np.argmax(np.where(AKT > np.median(AKT), KATV0[-1], -1)))                  # reflektor pokazowy: najbardziej obrócony z aktywnych
LO, HI = np.log10(np.percentile(OMV2, 2)), np.log10(np.percentile(OMV2, 99.5))     # skala długości strzałki dryfu (log); drganie poza nią
LA_MAX, L_IGLA = 8.0, 5.0
GAIN = 0.035
D = dane(0)
kroki_all, iou_all, narz = D['kroki'].astype(int), D['iou'], D['narz'].astype(bool)
IOU = iou_all[np.minimum(np.searchsorted(kroki_all, KW), len(kroki_all) - 1)]
log(f'reflektor pokazowy {C1}: obrót od startu {KATV0[-1, C1]:.0f}°; skala strzałek 10^{LO:.2f} – 10^{HI:.2f} °/100 kroków')

# ---------------- oś czasu ----------------
T_JEDEN, T_UCZ, T_UP, T_U1, T_KON, T_END = 8.0, 18.0, 34.0, 56.0, 56.5, 63.0
I_WSTEP = int(np.searchsorted(KW, 300))                                             # wstęp i reflektor pokazowy: krok 300 (strzałki długie)
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def indeks(t):
    if t < T_UCZ:
        return float(I_WSTEP) * (1 - sm((t - (T_UCZ - 1.5)) / 1.5))
    if t < T_UP:
        kr = 2 * EP * ((t - T_UCZ) / (T_UP - T_UCZ)) ** 1.6                         # pierwsze kroki wolno
    else:
        kr = min(KW[-1], 2 * EP + (KW[-1] - 2 * EP) * (t - T_UP) / (T_U1 - T_UP))
    return float(np.interp(kr, KW, np.arange(NT)))


def stan(fi):
    i = int(min(fi, NT - 2)); u = fi - i
    lerp = lambda X: X[i] * (1 - u) + X[i + 1] * u
    d = lerp(DIRS); d /= np.linalg.norm(d, axis=1, keepdims=True)
    def kier(T):
        tg = lerp(T); tg -= (tg * d).sum(1, keepdims=True) * d
        return tg / np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    om = lambda O: np.exp(lerp(np.log(np.maximum(O, 1e-9))))
    zz = Z[i].astype(np.float32) * (1 - u) + Z[i + 1].astype(np.float32) * u
    return (lerp(POS), d, (kier(TAN1), om(OMV1)), (kier(TAN2), om(OMV2)), zz / 2.0, KW[i] * (1 - u) + KW[i + 1] * u,
            i if u < 0.5 else i + 1)


def kamera(t):
    """azymut, elewacja, odległość, cel; w fazie reflektora pokazowego patrzymy na niego z boku."""
    kl = [(0, -70, 18, 190), (T_JEDEN, -20, 22, 158), (T_UCZ, -20, 22, 158), (T_U1, 250, 26, 162), (T_END, 300, 24, 172)]
    for (ta, *pa), (tb, *pb) in zip(kl[:-1], kl[1:]):
        if t <= tb:
            u = sm((t - ta) / max(tb - ta, 1e-9)); break
    else:
        (ta, *pa), (tb, *pb), u = kl[-2], kl[-1], 1.0
    az, el, di = [pa[k] + (pb[k] - pa[k]) * u for k in range(3)]
    if T_UCZ < t < T_U1:
        el += 9 * np.sin(2 * np.pi * (t - T_UCZ) / (T_U1 - T_UCZ) * 1.5)
    b = sm((t - T_JEDEN) / 2.5) * (1 - sm((t - (T_UCZ - 2.5)) / 2.5))
    rc = RFIX[C1]
    az_c = np.degrees(np.arctan2(rc[1], rc[0])) + 70.0
    el_c = np.degrees(np.arcsin(rc[2] / RR)) * 0.6 + 12.0
    az, el, di = az + (az_c - az) * b, el + (el_c - el) * b, di + (42 - di) * b
    cel = (rc * 0.82) * b
    return az, el, di, cel


def macierz(t):
    az, el, di, cel = kamera(t)
    a, e = np.radians(az), np.radians(el)
    oko_ = cel + di * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = cel - oko_; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    Vm = np.eye(4); Vm[0, :3], Vm[1, :3], Vm[2, :3] = s, up, -f; Vm[:3, 3] = -Vm[:3, :3] @ oko_
    fov, asp, n, fa = np.radians(40), WY / HY, 0.1, 800.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n) / (n - fa); Pm[2, 3] = 2 * fa * n / (n - fa); Pm[3, 2] = -1
    return Pm @ Vm, 1 / np.tan(fov / 2) * HY * SS / 2, oko_


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

wiazki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform int tryb, fokc; uniform float gain, przycc, wzmc, R;
layout(std430, binding=0) readonly buffer B0 { vec4 RF[]; };
layout(std430, binding=1) readonly buffer B1 { vec4 P[]; };
layout(std430, binding=2) readonly buffer B2 { vec4 DC[]; };
layout(std430, binding=3) readonly buffer B3 { float I[]; };
out float v_t; out float v_i; flat out int v_c; flat out int v_neg;
void main() {
    int bi = gl_InstanceID; int c = bi / 600; int j = bi % 600;
    vec3 p = P[j].xyz; vec3 d = DC[c].xyz; vec3 r = RF[c].xyz;
    float bb = dot(p, d); float sd = bb + sqrt(max(bb * bb - dot(p, p) + R * R, 0.0));
    vec3 e = p - sd * d;                                                     // punkt załamania na sferze
    float z = I[j * 768 + c];                                                // ze znakiem: ujemna jasność = ten sam reflektor ciągnie w −d_c
    float x = abs(z) * gain;
    if (fokc >= 0) x *= (c == fokc ? wzmc : przycc);
    vec3 q;
    if (tryb == 1) { q = gl_VertexID == 0 ? r : e; } else { q = gl_VertexID == 0 ? e : p; }
    v_t = float(gl_VertexID); v_i = x; v_c = c; v_neg = z < 0.0 ? 1 : 0;
    gl_Position = mvp * vec4(q, 1.0);
}''', fragment_shader='''#version 430
layout(std430, binding=4) readonly buffer B4 { vec4 KC[]; };
uniform int tryb; in float v_t; in float v_i; flat in int v_c; flat in int v_neg; out vec4 o;
void main() { float r = tryb == 0 ? (0.25 + 0.75 * v_t * v_t) : (0.6 + 0.4 * v_t);
  vec3 k = v_neg == 1 ? vec3(0.30, 0.85, 1.0) : KC[v_c].rgb;
  o = vec4(k * v_i * r, 1.0); }''')

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

buf_O = ctx.buffer(np.concatenate([RFIX, np.zeros((NK, 1))], 1).astype('f4')); buf_O.bind_to_storage_buffer(0)
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


seg = []
for lon_ in np.radians(np.arange(0, 180, 12)):
    t_ = np.linspace(0, 2 * np.pi, 121); p = R * np.stack([np.cos(t_) * np.cos(lon_), np.cos(t_) * np.sin(lon_), np.sin(t_)], 1); seg += [p[:-1], p[1:]]
for lat_ in np.radians(np.arange(-72, 90, 12)):
    t_ = np.linspace(0, 2 * np.pi, 121); p = R * np.stack([np.cos(t_) * np.cos(lat_), np.sin(t_) * np.cos(lat_), np.full_like(t_, np.sin(lat_))], 1); seg += [p[:-1], p[1:]]
SFERA = np.stack([np.concatenate(seg[0::2]), np.concatenate(seg[1::2])], 1).reshape(-1, 3)

import matplotlib                                                                       # noqa: E402
PLAZMA = matplotlib.colormaps['plasma']
dlug = lambda om: np.clip((np.log10(np.maximum(om, 1e-12)) - LO) / (HI - LO), 0, 1)   # 0..1 na skali logarytmicznej


def igly(Dc, wyr):
    """igła reflektora: od reflektora L_IGLA w kierunku d_c (w tę stronę pchają jego wiązki)."""
    q = RFIX + L_IGLA * Dc
    return np.stack([RFIX, q], 1).reshape(-1, 3), np.repeat(np.concatenate([0.3 + 0.7 * KOL, (0.55 * wyr)[:, None]], 1), 2, 0)


def strzalki(Dc, Tg, om, oko_, wyr, dryf=True):
    """strzałka obrotu z końca igły: kierunek = obrót d_c (rzut 3D), długość = prędkość kątowa w 192D (log).
    dryf: gruba, kolor plasma, z grotem;  drganie: cienka biała kreska bez grotu (na skali dryfu, więc zwykle dłuższa)."""
    u = dlug(om) if dryf else np.clip((np.log10(np.maximum(om, 1e-12)) - LO) / (HI - LO), 0, 1.4)
    q = RFIX + L_IGLA * Dc
    kon = q + (0.6 + LA_MAX * u)[:, None] * Tg
    if not dryf:
        return np.stack([q, kon], 1).reshape(-1, 3), np.repeat(np.concatenate([np.full((NK, 3), 0.92), (0.45 * wyr)[:, None]], 1), 2, 0)
    wid = kon - oko_; wid /= np.linalg.norm(wid, axis=1, keepdims=True)
    nrm = np.cross(Tg, wid); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    h = (0.6 + 1.0 * u)[:, None]
    g1, g2 = kon - h * Tg + 0.55 * h * nrm, kon - h * Tg - 0.55 * h * nrm
    a = (0.35 + 0.55 * u) * wyr
    strz = np.stack([q, kon, kon, g1, kon, g2], 1).reshape(-1, 3)
    return strz, np.repeat(np.concatenate([PLAZMA(0.25 + 0.75 * u)[:, :3], a[:, None]], 1), 6, 0)


# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_S, F_M = FN(34, True), FN(21), FN(17)
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
I_E1 = int(np.searchsorted(KW, EP))
UDZ_E1 = float(np.median(KATV0[I_E1]) / max(np.median(KATV0[-1]), 1e-9))
I_E20 = int(np.searchsorted(KW, 20 * EP))
TEKSTY = {
    'wstep': ('Gdzie jest gradient: uczenie obraca reflektory',
              ['Reflektor stoi w miejscu.  Igła przy nim to kierunek d_c, w którym jego 600 wiązek pcha cząsteczki (kolumna v_c wag fc2 bloku 6).',
               f'Strzałka na końcu igły = obrót netto w {OKNO_DRYF} krokach (dryf): dokąd igła się obraca;  długość = prędkość kątowa w 192 wymiarach, skala log.']),
    'jeden': ('Skąd obrót: gradient ∂L/∂v_c = Σ_j z_jc · δ_j',
              ['δ_j = poprawka, którą głowa odsyła do cząsteczki j.  Każda wiązka reflektora niesie ją z wagą swojej jasności z_jc, więc jasne diody ciągną igłę mocniej.',
               'AdamW zamienia sumę z batcha w krok (bezwładność, hamulec).  Sprężyna (weight decay) tylko skraca v_c, nie obraca go: obrót = sam gradient.',
               f'Cienka biała kreska = obrót w {OKNO_DRGANIE} krokach (drganie): kilka razy szybszy niż dryf, bo kolejne batche ciągną w różne strony.']),
    'uczenie': ('Uczenie: 806 checkpointów, 60 epok',
                [f'lr jest stały po rozgrzewce, więc igły drgają prawie tak samo przez całe uczenie (ok. {pl(float(np.median(MED1)), 0)}° na 100 kroków).  '
                 'Dryf netto za to zwalnia:',
                 'z wiekiem igły coraz bardziej drgają w miejscu, a coraz mniej obracają się w jedną stronę.  Tempo: epoki 0–2 przez 16 s, 2–60 przez 22 s.']),
    'koniec': ('Po 60 epokach', []),
}
KONIEC_TXT = [f'Drganie (100 kroków, mediana): {pl(float(MED1[I_E1]), 1)} → {pl(float(MED1[-1]), 1)} °/100 kroków;  '
              f'dryf netto (2000 kroków): {pl(float(MED2[I_E1]), 2)} → {pl(float(MED2[-1]), 2)} °/100 kroków (epoka 1 → 60).',
              f'Kąt rośnie jak odstęp^α: α = {pl(float(ALFA[0, 1]), 2)} na początku, {pl(float(ALFA[-1, 1]), 2)} po 40 epokach '
              '(1 = dryf w jedną stronę, 0,5 = błądzenie, mniej = drganie w miejscu, jak starzejące się szkło).',
              f'Obrót igły od kroku 0 (mediana): {pl(float(np.median(KATV0[-1])), 0)}°, z tego {UDZ_E1:.0%} w pierwszej epoce;  czujniki w_c (fc1): '
              f'{pl(float(np.median(KATW0[-1])), 0)}°.  IoU {pl(float(IOU[-1]), 2)}.']

# wykres: mediana prędkości obrotu igieł (drganie i dryf, skala log) w funkcji log(krok)
WX0, WYY0, WSW, WSH = WY - 430, HY - 300, 400, 170
lx = lambda kr: WX0 + np.log10(1 + kr) / np.log10(1 + KW[-1]) * WSW
_ym = np.log10(np.concatenate([MED1, MED2]))
YL0, YL1 = np.floor(_ym.min()), np.ceil(_ym.max())
ly = lambda om: WYY0 + WSH - (np.log10(np.maximum(om, 1e-12)) - YL0) / (YL1 - YL0) * WSH


def wykres(d, kr):
    d.rectangle((WX0 - 8, WYY0 - 56, WX0 + WSW + 10, WYY0 + WSH + 26), fill=(11, 16, 20, 225), outline=(42, 51, 58, 255))
    d.text((WX0, WYY0 - 52), 'mediana prędkości obrotu igieł [°/100 kroków]', font=F_M, fill=(232, 232, 232, 255))
    d.text((WX0, WYY0 - 30), f'drganie ({OKNO_DRGANIE} kroków)', font=FN(15), fill=(235, 235, 235, 255))
    d.text((WX0 + 175, WYY0 - 30), f'dryf netto ({OKNO_DRYF} kroków)', font=FN(15), fill=(255, 170, 80, 255))
    for e in range(int(YL0), int(YL1) + 1):
        y = ly(10.0 ** e); d.line((WX0, y, WX0 + WSW, y), fill=(60, 70, 78, 255))
        d.text((WX0 - 4, y), f'10^{e}' if e not in (0, 1, -1) else ('1' if e == 0 else '10' if e == 1 else '0,1'), font=FN(13), fill=(170, 175, 180, 255), anchor='rm')
    for kk, nz in ((10, '10'), (100, '100'), (EP, 'ep 1'), (10 * EP, 'ep 10'), (KW[-1], 'ep 60')):
        x = lx(kk); d.line((x, WYY0 + WSH, x, WYY0 + WSH + 4), fill=(170, 175, 180, 255))
        d.text((x, WYY0 + WSH + 6), nz, font=FN(13), fill=(170, 175, 180, 255), anchor='mt')
    for M_, k_ in ((MED1, (235, 235, 235, 255)), (MED2, (255, 170, 80, 255))):
        d.line(list(zip(lx(KW).tolist(), ly(M_).tolist())), fill=k_, width=2)
    x = lx(kr); d.line((x, WYY0, x, WYY0 + WSH), fill=(255, 255, 255, 220), width=2)


def legenda(d):
    x0, y0 = WY - 430, 250
    d.text((x0, y0), 'długość strzałki = prędkość obrotu v̂_c (log)', font=F_M, fill=(232, 232, 232, 255))
    wart = [w_ for w_ in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10) if 10 ** LO <= w_ <= 10 ** HI]
    for n, w_ in enumerate(wart):
        u = float(dlug(w_))
        k = tuple(int(255 * v) for v in PLAZMA(0.25 + 0.75 * u)[:3]) + (255,)
        y = y0 + 30 + 24 * n
        d.line((x0, y, x0 + 12 + 150 * u, y), fill=k, width=4)
        d.text((x0 + 175, y), f'{pl(w_, 1) if w_ < 1 else int(w_)} °/100 kroków', font=FN(15), fill=(200, 205, 210, 255), anchor='lm')


def nakladka(img, faza, t, kr, ii, mvp, om1, om2):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    if faza == 'koniec':
        pod = KONIEC_TXT
    d.rectangle((0, 0, WY, 78 + 30 * len(pod) + 8), fill=(3, 6, 10, 190))           # pasy pod napisami: tekst nie ginie w scenie
    d.rectangle((0, HY - 66, WY, HY), fill=(3, 6, 10, 190))
    d.text((WY / 2, 26), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        d.text((WY / 2, 78 + 30 * n), s, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), f'Dane: bieg s47, wagi fc1/fc2 bloku 6 z {NT} checkpointów (kroki 0–{KW[-1]}), natężenia z_jc zdjęcia TRAIN 0.  Obrót z checkpointów '
           f'odległych o ≥ {OKNO_DRGANIE} i ≥ {OKNO_DRYF} kroków (rzeczywiste kroki AdamW).', font=F_M, fill=(205, 210, 215, 255))
    d.text((22, HY - 32), f'Kierunek strzałki = rzut na 3D (P = 3 główne składowe, {UDZ:.0%} wariancji v̂_c);  długość z pełnych 192 wymiarów.  '
           'Wiązki: jasność z_jc / 2, błękit = z < 0.  Optyka poglądowa jak w filmie 12.', font=F_M, fill=(205, 210, 215, 255))
    legenda(d)
    if faza in ('jeden',):
        q = rzut(mvp, RFIX[C1] * 1.06)
        if q and 0 < q[0] < WY and 0 < q[1] < HY - 90:
            d.text(q, f'reflektor {C1}: drganie {pl(float(om1[C1]), 1)}, dryf {pl(float(om2[C1]), 2)} °/100 kroków', font=FN(22, True),
                   fill=(255, 220, 160, int(255 * sm((t - T_JEDEN - 1) / 1.5))), anchor='mb')
    if faza in ('uczenie', 'koniec'):
        d.text((WY - 30, 160), f'epoka {pl(kr / EP, 2)}   krok ' + f'{kr:,.0f}'.replace(',', ' '), font=FN(26, True), fill=(235, 240, 245, 255), anchor='ra')
        d.text((WY - 30, 196), f'obrót igły od startu (mediana): {pl(float(np.median(KATV0[ii])), 0)}°', font=FN(20, True),
               fill=(255, 200, 120, 255), anchor='ra')
        wykres(d, kr)
    else:
        d.text((WY - 30, 160), f'krok {int(KW[I_WSTEP])}  (epoka {pl(KW[I_WSTEP] / EP, 2)})', font=FN(24, True), fill=(235, 240, 245, 255), anchor='ra')


def klatka(t):
    faza = 'wstep' if t < T_JEDEN else 'jeden' if t < T_UCZ else 'uczenie' if t < T_KON else 'koniec'
    fi = indeks(t)
    Pp, Dc, (Tg1, om1), (Tg2, om2), I, kr, ii = stan(fi)
    mvp, skala, oko_ = macierz(t)
    mvp_b = mvp.T.astype('f4').tobytes()
    buf_P.write(np.concatenate([Pp, np.zeros((TOK, 1))], 1).astype('f4').tobytes())
    buf_D.write(np.concatenate([Dc, np.zeros((NK, 1))], 1).astype('f4').tobytes())
    buf_I.write(I.astype('f4').tobytes())
    on = sm(t / 2.5)
    jed = sm((t - T_JEDEN) / 2.0) * (1 - sm((t - (T_UCZ - 2.0)) / 2.0))           # wyróżnienie reflektora pokazowego
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mvp_b); linie['szer'].value = 1.5
    rysuj(linie, moderngl.LINES, SFERA, np.tile([0.30, 0.55, 0.75, 0.18 * on], (len(SFERA), 1)))
    wiazki['mvp'].write(mvp_b)
    wiazki['fokc'].value = C1; wiazki['przycc'].value = 1.0 - 0.9 * jed; wiazki['wzmc'].value = 1.0 + 9.0 * jed
    for tryb, g in ((1, 0.30), (0, 1.0)):
        wiazki['tryb'].value = tryb; wiazki['gain'].value = GAIN * g * on
        vao_w.render(moderngl.LINES, vertices=2, instances=TOK * NK)
    kropki['mvp'].write(mvp_b); kropki['skala'].value = skala
    kropki['ksztalt'].value = 0
    S = np.abs(I).sum(1); aj = np.clip(S / np.percentile(S, 90), 0, 1.6) * on * (1 - 0.6 * jed)
    rysuj(kropki, moderngl.POINTS, Pp, np.concatenate([np.tile([0.85, 1.0, 0.9], (TOK, 1)), aj[:, None]], 1), np.full(TOK, 0.9))
    fb_ldr.use(); ctx.disable(moderngl.BLEND)
    hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    # po tonowaniu: reflektory, igły i strzałki (nie gubią się w świetle wiązek)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    wyr = np.full(NK, on) * (1 - 0.75 * jed); wyr[C1] = on
    kropki['ksztalt'].value = 2
    rysuj(kropki, moderngl.POINTS, RFIX, np.concatenate([(0.3 + 0.7 * KOL), (0.55 * wyr)[:, None]], 1), np.full(NK, 0.8))
    ip, ik = igly(Dc, wyr)
    dp, dk_ = strzalki(Dc, Tg1, om1, oko_, wyr, dryf=False)
    sp, sk = strzalki(Dc, Tg2, om2, oko_, wyr)
    linie['szer'].value = 2.0; rysuj(linie, moderngl.LINES, ip, ik)
    linie['szer'].value = 2.4; rysuj(linie, moderngl.LINES, sp, sk)                 # drganie tylko dla reflektora pokazowego (niżej)
    if jed > 0.01:                                                                  # reflektor pokazowy: gruba igła i strzałki
        for (a_, k_), n_, sz in (((ip, ik), 2, 6.0), ((dp, dk_), 2, 3.0), ((sp, sk), 6, 7.0)):
            linie['szer'].value = sz
            rysuj(linie, moderngl.LINES, a_.reshape(NK, n_, 3)[C1], k_.reshape(NK, n_, 4)[C1] * [1, 1, 1, jed])
    kropki['ksztalt'].value = 1
    rysuj(kropki, moderngl.POINTS, Pp[narz], np.tile([0.29, 0.94, 1.0, 0.8 * on * (1 - 0.6 * jed)], (int(narz.sum()), 1)), np.full(int(narz.sum()), 1.6))
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, faza, t, kr, ii, mvp, om1, om2)
    return img


def ffmpeg_nvenc(sciezka):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{WY}x{HY}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '21', '-b:v', '0', '-pix_fmt', 'yuv420p', str(sciezka)], stdin=subprocess.PIPE)


CZASY = np.arange(int(T_END * FPS)) / FPS
if PROBA:
    CZASY = np.array([3.0, 13.0, 21.0, 30.0, 45.0, 60.0])
ff = None if PROBA else ffmpeg_nvenc(HERE / 'lasery_23_obrot_reflektorow.mp4')
for n, t in enumerate(CZASY):
    img = klatka(float(t))
    if PROBA:
        img.save(HERE / f'lasery_23_proba_{t:04.1f}.png')
    else:
        ff.stdin.write(img.convert('RGBA').tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(CZASY)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
