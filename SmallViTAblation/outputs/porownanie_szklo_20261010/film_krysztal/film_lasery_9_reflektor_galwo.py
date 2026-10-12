"""Film 9: reflektor z 768 kanałami laserowymi (dioda → galwo X → galwo Y → soczewka o zmiennej ogniskowej → stożek → ognisko).
Nowy film (film 8 zostaje bez zmian). Zasady: ZASADY_LASER_KOLOID.md (Z2, Z5, Z11). Cząsteczki jeszcze nie rysujemy.

Kanał c = neuron c ostatniego (6.) bloku MLP:
- dioda laserowa świeci z jasnością z_c = GELU(w_c · LN(x) + b_c) wybranego tokenu (część dodatnia);
- lusterka galwanometryczne X i Y ustawiają kierunek wiązki, soczewka ustawia ogniskową, więc ognisko kanału to punkt w 3D:
      F_c = X0 + s · z_c · P v_c,
  gdzie X0 = miejsce cząsteczki (nie rysowane), v_c = kolumna c macierzy fc2 (kierunek pchnięcia), P = stały rzut 192D → 3D
  (trzy główne składowe kierunków v_c w kroku 22 974), s stałe. Odcinek X0 → F_c to wektor siły, którą ten kanał działa na cząsteczkę:
  dwa kąty galw = kierunek, ogniskowa = długość.
Fazy: obudowa; zbliżenie na jeden kanał (dioda, galwo X, galwo Y, soczewka, stożek); wszystkie kanały i chmura ognisk;
uczenie od kroku 0 do 22 974 (galwa i ogniskowe się przestawiają); podsumowanie.
Dane jak w filmie 8 (wagi fc2 bloku 6 z 341 checkpointów, aktywacje zdjęcia TRAIN 0); pamięć podręczna reflektor_8_cache.npz.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_9_reflektor_galwo.py [proba]"""
import os
import sys
import time

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lasery_wspolne import EP, HERE, TOK, dane, ffmpeg

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
NK, NY, NZ = 768, 32, 24
FPS, WY, HY = 30, 1920, 1080
SS = 2
ASSETS = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10'
WAGI = ASSETS / '01_bieg_35000_gesty_zapis/eden/gesty/wagi'
LATENT = ASSETS / '03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
CACHE = HERE / 'reflektor_8_cache.npz'

# ---------------- dane ----------------
D = dane(0)
kroki_all, narz = D['kroki'].astype(np.int64), D['narz']
if CACHE.exists():
    c_ = np.load(CACHE)
    KW, V = c_['KW'], c_['V']
else:
    import torch
    KW = np.array(sorted(int(f[1:6]) for f in os.listdir(WAGI) if f.endswith('.pt')))
    V = np.stack([torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']['bloki.5.mlp.fc2.weight']
                  .float().numpy() for k in KW])
    np.savez_compressed(CACHE, KW=KW, V=V)
NT = len(KW)
idx_lat = np.searchsorted(kroki_all, KW)
assert np.all(kroki_all[idx_lat] == KW)
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
Z = np.clip(np.asarray(np.load(LATENT, mmap_mode='r')[idx_lat, 0, FOK]).astype(np.float32), 0, None)   # (341, 768)
Vn_end = V[-1] / np.linalg.norm(V[-1], axis=0)
U_, _, _ = np.linalg.svd(Vn_end - Vn_end.mean(1, keepdims=True), full_matrices=False)
P3 = U_[:, :3].T                                                                    # (3, 192)
PV = np.einsum('ad,tdc->tca', P3, V)                                                # (341, 768, 3)
NORM = np.linalg.norm(V, axis=1)
Vn = V / NORM[:, None, :]
pary = np.array([(np.where(np.eye(NK, dtype=bool), 0.0, Vn[t].T @ Vn[t]).min(1) < -0.5).mean() for t in range(NT)])
cos0 = np.median((Vn * Vn[:1]).sum(1), axis=1)
WEK = Z[:, :, None] * PV                                                            # (341, 768, 3) rzut z_c · v_c
log(f'wagi {V.shape}, token {FOK}, pary {pary[0]:.0%} → {pary[-1]:.0%}')

# geometria
PITCH = 1.25
gy, gz = np.meshgrid((np.arange(NY) - (NY - 1) / 2) * PITCH, (np.arange(NZ) - (NZ - 1) / 2) * PITCH)
YC, ZC = gy.ravel(), -gz.ravel()
DIODA = np.stack([np.zeros(NK), YC - 0.25, ZC], 1)                                  # dioda świeci w stronę +x
G1 = DIODA + np.array([0.55, 0, 0])                                                 # galwo X (oś obrotu pionowa)
G2 = G1 + np.array([0, 0.5, 0])                                                     # galwo Y (oś obrotu pozioma)
LENS = G2 + np.array([0.7, 0, 0])                                                   # soczewka ogniskująca
R_SOCZ = 0.42
X0 = np.array([46.0, 0.0, 0.0])                                                     # miejsce cząsteczki (nie rysowane)
_dl = np.linalg.norm(WEK, axis=2)
W_REF = float(np.percentile(_dl[Z > 0.2], 90))                                     # długość ściskana: 12 · tanh(|w| / W_REF)
L_MAX = 12.0
C_ZB = int(np.argmax(np.where(_dl[-1] < 1.5 * W_REF, Z[-1], 0)))                   # kanał do zbliżenia: jasny, wektor umiarkowany
log('kanał do zbliżenia', C_ZB, 'z', Z[-1, C_ZB])
porz = np.random.default_rng(3).permutation(NK) / NK


def barwa(w):
    """kolor kierunku wektora (azymut w płaszczyźnie y–z rzutu, jasność od składowej x)."""
    h = (np.arctan2(w[..., 2], w[..., 1]) / (2 * np.pi)) % 1.0
    k = lambda n: np.clip(np.abs(((h * 6 + n) % 6) - 3) - 1, 0, 1)
    return np.stack([k(0), k(4), k(2)], -1) * 0.8 + 0.2


# ---------------- oś czasu ----------------
FAZY = [(0, 'obudowa'), (7, 'kanal'), (25, 'wszystkie'), (36, 'uczenie'), (64, 'koniec'), (71, None)]
T_U0, T_UP, T_U1 = 37.0, 47.0, 62.5
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def indeks(t):
    if t < T_U0:
        return float(NT - 1) if t < 35.0 else float(NT - 1) * (1 - sm((t - 35.0) / 2.0))
    if t < T_UP:
        kr = 2 * EP * (t - T_U0) / (T_UP - T_U0)
    else:
        kr = min(KW[-1], 2 * EP + (KW[-1] - 2 * EP) * (t - T_UP) / (T_U1 - T_UP))
    return float(np.interp(kr, KW, np.arange(NT)))


def stan(fi):
    i = int(min(fi, NT - 2)); u = fi - i
    w = WEK[i] * (1 - u) + WEK[i + 1] * u
    z = Z[i] * (1 - u) + Z[i + 1] * u
    return w, z, KW[i] * (1 - u) + KW[i + 1] * u, i if u < 0.5 else i + 1


# kamera: (czas, azymut, elewacja, odległość, cel)
PC = LENS[C_ZB]
CEL_ALL = np.array([X0[0] * 0.45, 0, 0])
KAM = [(0, -125, 15, 108, np.array([0.0, 0, 0])), (6, -60, 14, 62, np.array([1.0, 0, 0])),
       (10, -55, 22, 5.2, PC - np.array([0.3, 0.25, 0])), (18, -35, 26, 4.2, PC - np.array([0.3, 0.25, 0])),
       (21, -30, 20, 9.0, PC + np.array([3.0, 0, 0])), (25, -50, 18, 78, CEL_ALL), (36, -40, 16, 74, CEL_ALL),
       (64, 10, 22, 70, CEL_ALL), (71, 20, 26, 78, CEL_ALL)]


def macierz(t):
    for (ta, *pa), (tb, *pb) in zip(KAM[:-1], KAM[1:]):
        if t <= tb:
            u = sm((t - ta) / (tb - ta)); break
    else:
        (ta, *pa), (tb, *pb), u = KAM[-2], KAM[-1], 1.0
    az, el, di = [pa[k] + (pb[k] - pa[k]) * u for k in range(3)]
    cel = pa[3] + (pb[3] - pa[3]) * u
    a, e = np.radians(az), np.radians(el)
    oko = cel + di * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = cel - oko; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    Vm = np.eye(4); Vm[0, :3], Vm[1, :3], Vm[2, :3] = s, up, -f; Vm[:3, 3] = -Vm[:3, :3] @ oko
    fov, asp, n, fa = np.radians(40), WY / HY, 0.02, 500.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n) / (n - fa); Pm[2, 3] = 2 * fa * n / (n - fa); Pm[3, 2] = -1
    return Pm @ Vm, 1 / np.tan(fov / 2) * HY * SS / 2


def rzut(mvp, p):
    q = mvp @ np.append(p, 1.0)
    if q[3] <= 0.05:
        return None
    return (q[0] / q[3] * 0.5 + 0.5) * WY, (0.5 - q[1] / q[3] * 0.5) * HY


# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
W4, H4 = WY * SS, HY * SS
hdr = ctx.texture((W4, H4), 4, dtype='f2'); fb_hdr = ctx.framebuffer([hdr])
ldr = ctx.texture((W4, H4), 4); fb_ldr = ctx.framebuffer([ldr])
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

kropki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform float skala;
in vec3 pos; in vec4 kol; in float rozm; out vec4 v_k;
void main() { gl_Position = mvp * vec4(pos, 1.0); gl_PointSize = clamp(rozm * skala / gl_Position.w, 1.5, 40.0); v_k = kol; }''',
                     fragment_shader='''#version 430
uniform int ksztalt; in vec4 v_k; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); float a;
  if (ksztalt == 2) { a = (max(abs(d.x), abs(d.y)) < 0.8) ? 1.0 : 0.0; }
  else { if (r2 > 1.0) discard; a = ksztalt == 1 ? 1.0 - smoothstep(0.7, 1.0, sqrt(r2)) : exp(-r2 * 4.0); }
  o = vec4(v_k.rgb * a * v_k.a, a * v_k.a); }''')

linie = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; in vec3 pos; in vec4 kol; out vec4 g_k;
void main() { gl_Position = mvp * vec4(pos, 1.0); g_k = kol; }''', geometry_shader='''#version 430
layout(lines) in; layout(triangle_strip, max_vertices = 4) out;
uniform vec2 vp; uniform float szer; in vec4 g_k[]; out vec4 f_k;
void main() {
  vec4 p0 = gl_in[0].gl_Position, p1 = gl_in[1].gl_Position;
  if (p0.w < 0.02 || p1.w < 0.02) return;
  vec2 s0 = p0.xy / p0.w * vp * 0.5, s1 = p1.xy / p1.w * vp * 0.5, d = s1 - s0;
  d = length(d) < 1e-4 ? vec2(1.0, 0.0) : normalize(d);
  vec2 n = vec2(-d.y, d.x) * szer * 0.5 / (vp * 0.5);
  f_k = g_k[0]; gl_Position = p0 + vec4(n * p0.w, 0, 0); EmitVertex(); gl_Position = p0 - vec4(n * p0.w, 0, 0); EmitVertex();
  f_k = g_k[1]; gl_Position = p1 + vec4(n * p1.w, 0, 0); EmitVertex(); gl_Position = p1 - vec4(n * p1.w, 0, 0); EmitVertex();
  EndPrimitive(); }''', fragment_shader='''#version 430
in vec4 f_k; out vec4 o; void main() { o = vec4(f_k.rgb * f_k.a, f_k.a); }''')
linie['vp'].value = (W4, H4)

trojk = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; in vec3 pos; in vec4 kol; out vec4 v_k;
void main() { gl_Position = mvp * vec4(pos, 1.0); v_k = kol; }''', fragment_shader='''#version 430
in vec4 v_k; out vec4 o; void main() { o = vec4(v_k.rgb * v_k.a, v_k.a); }''')

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


def baza(d):
    """dwa wektory prostopadłe do kierunków d (N, 3)."""
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(d, e1)


def okregi(c, d, r, n=24):
    e1, e2 = baza(d)
    f = np.linspace(0, 2 * np.pi, n + 1)
    p = c[:, None] + r * (np.cos(f)[None, :, None] * e1[:, None] + np.sin(f)[None, :, None] * e2[:, None])
    return np.stack([p[:, :-1], p[:, 1:]], 2).reshape(-1, 3)


def kwadraty(c, u, w, a):
    k = np.stack([c - a * u - a * w, c + a * u - a * w, c + a * u + a * w, c - a * u + a * w], 1)
    return np.stack([k, np.roll(k, -1, 1)], 2).reshape(-1, 3), k


# obudowa
YM, ZM = NY * PITCH / 2 + 0.6, NZ * PITCH / 2 + 0.6
nar = np.array([[x, y, z] for x in (-0.6, 2.4) for y in (-YM, YM) for z in (-ZM, ZM)])
OBUD = np.array([nar[a] for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)] for a in (a, b)])

FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_S, F_M = FN(34, True), FN(21), FN(17)
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
TEKSTY = {
    'obudowa': ('Reflektor: 768 kanałów laserowych w jednej obudowie = jeden blok MLP (6. blok)',
                ['Każdy kanał to jeden neuron warstwy ukrytej.  Kanał: dioda laserowa → lusterko galwo X → lusterko galwo Y → soczewka ogniskująca.',
                 'Wiązka wychodzi z soczewki jako stożek i skupia się w ognisku.  Ognisko kanału = koniec jednego wektora siły.']),
    'kanal': ('Jeden kanał z bliska',
              ['Dioda świeci z jasnością z_c (aktywacja GELU).  Galwo X i galwo Y ustawiają kierunek wiązki,',
               'soczewka ustawia ogniskową, czyli odległość ogniska.  Kierunek i odległość razem dają wektor z_c · v_c (rzut do 3D).']),
    'wszystkie': ('Wszystkie 768 kanałów: chmura ognisk',
                  ['Ogniska wszystkich kanałów otaczają jedno miejsce (krzyżyk), w którym później stanie cząsteczka.',
                   'Ciemne kanały (z_c ≈ 0) mają ognisko w samym środku: ich wektor siły jest zerowy.']),
    'uczenie': ('Uczenie: galwa i ogniskowe się przestawiają',
                ['Każdy kanał zmienia kierunek (galwa) i długość (ogniskowa) osobno;  kanały ustawiają się w pary o przeciwnych kierunkach.',
                 'Kolor = kierunek wektora.  Ustawienie dla jednego tokenu narzędzia (cząsteczka 316), krok po kroku od 0 do 22 974.']),
    'koniec': ('Po 21 epokach', []),
}
KONIEC_TXT = [f'Pary kanałów o przeciwnych kierunkach (cos < −0,5): {pary[0]:.0%} → {pary[-1]:.0%}.   Siła kanału |v_c|: ×{pl(np.median(NORM[-1]) / np.median(NORM[0]), 2)}.',
              f'Kierunek tego samego kanału względem kroku 0: mediana cos {pl(cos0[-1], 2)} (obrót o około {np.degrees(np.arccos(cos0[-1])):.0f}°).',
              'Następny krok: cząsteczka w miejscu krzyżyka i suma 768 wektorów jako pchnięcie bloku.']


def klatka(t):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    fi = indeks(t)
    w, z, kr, ii = stan(fi)
    mvp, skala = macierz(t)
    mvp_b = mvp.T.astype('f4').tobytes()
    linie['mvp'].write(mvp_b); kropki['mvp'].write(mvp_b); kropki['skala'].value = skala; trojk['mvp'].write(mvp_b)
    wn = np.linalg.norm(w, axis=1, keepdims=True)
    F = X0 + w / np.maximum(wn, 1e-9) * L_MAX * np.tanh(wn / W_REF)               # ogniska (768, 3), długość ściśnięta
    dl = F - LENS; f_c = np.linalg.norm(dl, axis=1); d = dl / f_c[:, None]
    th_y, th_z = np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[:, 2], d[:, 0])
    kol = barwa(w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9))
    a = np.clip(z / 1.6, 0, 1.0)

    # włączanie kanałów w czasie
    if t < 7:
        jas = np.zeros(NK)
    elif t < 25:
        jas = np.zeros(NK); jas[C_ZB] = 1.0
    else:
        jas = np.clip((sm((t - 25.5) / 5.0) * 1.05 - porz) * 20, 0, 1); jas[C_ZB] = 1.0
    # animacja pierwszego kanału: wiązka, galwo X, galwo Y, ogniskowa
    sX, sY, sF = sm((t - 13.5) / 2.0), sm((t - 15.5) / 2.0), sm((t - 17.5) / 2.5)
    on_b = sm((t - 11.5) / 1.5)
    if t < 25:
        th_y = th_y.copy(); th_z = th_z.copy(); f_c = f_c.copy()
        th_y[C_ZB] *= sX; th_z[C_ZB] *= sY
        f_c[C_ZB] = 3.0 + (f_c[C_ZB] - 3.0) * sF
        dzb = np.array([np.cos(th_z[C_ZB]) * np.cos(th_y[C_ZB]), np.cos(th_z[C_ZB]) * np.sin(th_y[C_ZB]), np.sin(th_z[C_ZB])])
        d = d.copy(); d[C_ZB] = dzb / np.linalg.norm(dzb)
        F = F.copy(); F[C_ZB] = LENS[C_ZB] + f_c[C_ZB] * d[C_ZB]
    bud = sm(t / 2.5)

    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['szer'].value = 2.2
    rysuj(linie, moderngl.LINES, OBUD, np.tile([0.30, 0.42, 0.55, 0.55 * bud], (len(OBUD), 1)))
    # elementy kanałów: diody (kwadraty), lusterka galwo, soczewki
    blisko = 7 <= t < 25
    kropki['ksztalt'].value = 2
    al_d = np.full((NK, 1), 0.9 * bud)
    if blisko:
        al_d[:] = 0.2; al_d[C_ZB] = 1.0
    rysuj(kropki, moderngl.POINTS, DIODA, np.concatenate([np.tile([0.30, 0.30, 0.36], (NK, 1)) + (jas * a)[:, None] * np.array([2.4, 1.0, 0.8]),
                                                          al_d], 1), np.full(NK, 0.16))
    tilt_x, tilt_y = th_y / 2, th_z / 2                                             # lusterko obraca się o połowę kąta wiązki
    b1 = np.radians(45) + tilt_x
    u1 = np.stack([np.cos(b1), np.sin(b1), np.zeros(NK)], 1); w1 = np.tile([0, 0, 1.0], (NK, 1))
    h2 = np.tile(np.array([-1.0, 1.0, 0]) / np.sqrt(2), (NK, 1))
    w2 = np.stack([np.sin(tilt_y) / np.sqrt(2), np.sin(tilt_y) / np.sqrt(2), np.cos(tilt_y)], 1)   # e_z obrócone wokół osi h2
    lu1, k1 = kwadraty(G1, u1, w1, 0.17); lu2, k2 = kwadraty(G2, h2, w2, 0.17)
    al_el = (0.45 if t < 25 else 0.22) * bud if not blisko else 0.07 * bud
    rysuj(linie, moderngl.LINES, lu1, np.tile([0.75, 0.80, 0.9, al_el], (len(lu1), 1)))
    rysuj(linie, moderngl.LINES, lu2, np.tile([0.75, 0.80, 0.9, al_el], (len(lu2), 1)))
    so = okregi(LENS, d, R_SOCZ, 16)
    rysuj(linie, moderngl.LINES, so, np.tile([0.45, 0.75, 1.0, al_el], (len(so), 1)))
    # wiązki: dioda → G1 → G2 → soczewka, potem stożek do ogniska
    on = (jas * a) > 0.01
    if on.any():
        idx = np.flatnonzero(on); m = len(idx); ja = (jas * a)[idx]
        seg = np.concatenate([np.stack([DIODA[idx], G1[idx]], 1), np.stack([G1[idx], G2[idx]], 1), np.stack([G2[idx], LENS[idx]], 1)]).reshape(-1, 3)
        kk = np.repeat(np.tile(np.concatenate([np.tile([1.0, 0.35, 0.3], (m, 1)), (ja * 1.2)[:, None]], 1), (3, 1)), 2, 0)
        if t < 25:
            kk[:, 3] *= on_b
        linie['szer'].value = 2.0
        rysuj(linie, moderngl.LINES, seg, kk)
        # stożek: powierzchnia (trójkąty) + krawędzie
        e1, e2 = baza(d[idx]); nseg = 18 if blisko else 8
        fz = np.linspace(0, 2 * np.pi, nseg + 1)
        brzeg = LENS[idx][:, None] + R_SOCZ * (np.cos(fz)[None, :, None] * e1[:, None] + np.sin(fz)[None, :, None] * e2[:, None])
        tri = np.stack([brzeg[:, :-1], brzeg[:, 1:], np.repeat(F[idx][:, None], nseg, 1)], 2).reshape(-1, 3)
        al_s = (0.10 if blisko else 0.045) * ja * (on_b if t < 25 else 1.0) * (sF if t < 25 else 1.0) ** 0.5
        kt = np.repeat(np.concatenate([kol[idx], al_s[:, None]], 1), nseg * 3, 0)
        ctx.disable(moderngl.CULL_FACE)
        rysuj(trojk, moderngl.TRIANGLES, tri, kt)
        kr_ = np.stack([brzeg[:, ::max(1, nseg // 4)][:, :4], np.repeat(F[idx][:, None], 4, 1)], 2).reshape(-1, 3)
        kk2 = np.repeat(np.concatenate([kol[idx], (ja * (0.35 if blisko else 0.16) * (on_b if t < 25 else 1.0))[:, None]], 1), 8, 0)
        linie['szer'].value = 1.3
        rysuj(linie, moderngl.LINES, kr_, kk2)
        # ogniska
        kropki['ksztalt'].value = 0
        rysuj(kropki, moderngl.POINTS, F[idx], np.concatenate([kol[idx], (ja * 2.2 * (sF if t < 25 else 1.0))[:, None]], 1), np.full(m, 0.55))
    # wektory siły X0 → ognisko (cienkie) i krzyżyk w miejscu cząsteczki
    if t >= 25:
        aw = sm((t - 28) / 3.0)
        idx = np.flatnonzero(on)
        if len(idx):
            seg = np.stack([np.repeat(X0[None], len(idx), 0), F[idx]], 1).reshape(-1, 3)
            kk = np.repeat(np.concatenate([kol[idx], (0.25 * aw * (jas * a)[idx])[:, None]], 1), 2, 0)
            linie['szer'].value = 1.2
            rysuj(linie, moderngl.LINES, seg, kk)
        kz = np.array([X0 - [0, 1.2, 0], X0 + [0, 1.2, 0], X0 - [0, 0, 1.2], X0 + [0, 0, 1.2], X0 - [1.2, 0, 0], X0 + [1.2, 0, 0]])
        linie['szer'].value = 2.5
        rysuj(linie, moderngl.LINES, kz, np.tile([0.9, 0.95, 1.0, 0.7 * sm((t - 25) / 2)], (6, 1)))
    fb_ldr.use(); ctx.disable(moderngl.BLEND)
    hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    if blisko:                                                                      # wyraźne elementy wybranego kanału
        c = C_ZB; linie['szer'].value = 3.0
        for lu, kk_ in ((kwadraty(G1[c:c + 1], u1[c:c + 1], w1[c:c + 1], 0.17)[0], [0.85, 0.9, 1.0, 0.95]),
                        (kwadraty(G2[c:c + 1], h2[c:c + 1], w2[c:c + 1], 0.17)[0], [0.85, 0.9, 1.0, 0.95]),
                        (okregi(LENS[c:c + 1], d[c:c + 1], R_SOCZ, 40), [0.5, 0.8, 1.0, 0.95]),
                        (okregi(G1[c:c + 1] - [0, 0, 0.25], np.array([[0, 0, 1.0]]), 0.07, 16), [0.6, 0.6, 0.65, 0.9]),
                        (okregi(G2[c:c + 1] + [0.12, -0.12, 0], h2[c:c + 1], 0.07, 16), [0.6, 0.6, 0.65, 0.9])):
            rysuj(linie, moderngl.LINES, lu, np.tile(kk_, (len(lu), 1)))
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, faza, t, mvp, jas * a, kr, ii, th_y, th_z, f_c)
    return img


def nakladka(img, faza, t, mvp, za, kr, ii, th_y, th_z, f_c):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    if faza == 'koniec':
        pod = KONIEC_TXT
    d.text((WY / 2, 24), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        d.text((WY / 2, 76 + 30 * n), s, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), 'Dane: wagi fc2 bloku 6 z 341 checkpointów (kroki 0–22 974) i aktywacje 768 neuronów (zdjęcie TRAIN 0, cząsteczka 316 narzędzia).',
           font=F_M, fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Ognisko = X0 + kierunek(z_c · P v_c) · 12 · tanh(|z_c · P v_c| / w_ref);  P = 3 główne składowe kierunków v_c w kroku 22 974 (ilustracja).  Cząsteczki jeszcze nie ma.',
           font=F_M, fill=(205, 210, 215, 255))
    pod_ = []
    if 9.5 < t < 24.5:
        c = C_ZB; al = int(255 * max(0.0, min(1.0, (t - 9.5) / 1.0, (20.8 - t) / 0.6)))
        pod_ = [(DIODA[c] + [0, -0.35, 0.05], 'dioda laserowa', 'rm'), (G1[c] - [0, 0, 0.35], 'lusterko galwo X', 'mt'),
                (G2[c] + [0, 0, 0.65], 'lusterko galwo Y', 'mb'), (LENS[c] - [0, 0, 0.6], 'soczewka (zmienna ogniskowa)', 'mt')]
        if t > 18:
            pod_.append((LENS[c] + [2.2, 0, 0.35], 'stożek wiązki → ognisko', 'mb'))
        for p, s, an in pod_:
            q = rzut(mvp, np.asarray(p, float))
            if q:
                d.text(q, s, font=FN(22, True), fill=(210, 235, 255, al), anchor=an)
        x0, y0 = 30, 190
        d.rectangle((x0 - 10, y0 - 10, x0 + 430, y0 + 128), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
        d.text((x0, y0), f'kanał {c} (neuron {c} bloku 6)', font=FN(20, True), fill=(235, 240, 245, 255))
        cel = X0 - LENS[c]; f0 = np.linalg.norm(cel)
        dy, dz = np.degrees(th_y[c] - np.arctan2(cel[1], cel[0])), np.degrees(th_z[c] - np.arctan2(cel[2], cel[0]))
        d.text((x0, y0 + 30), f'odchylenie od celu: galwo X {dy:+.1f}°, galwo Y {dz:+.1f}°', font=FN(19), fill=(205, 225, 255, 255))
        d.text((x0, y0 + 58), f'ogniskowa {pl(f_c[c], 1)}  (do miejsca cząsteczki {pl(f0, 1)})', font=FN(19), fill=(205, 225, 255, 255))
        d.text((x0, y0 + 86), f'jasność diody z_c = {pl(float(Z[-1, c]), 2)}', font=FN(19), fill=(255, 170, 150, 255))
    if t >= 25:
        q = rzut(mvp, X0 + [0, 0, -1.6])
        if q and t < 40:
            d.text(q, 'tu stanie cząsteczka', font=FN(20, True), fill=(230, 240, 255, int(255 * sm((t - 26) / 2))), anchor='mt')
        q = rzut(mvp, np.array([2.4, 0, -ZM - 0.8]))
        if q and t < 40:
            d.text(q, 'obudowa: 768 kanałów (32 × 24)', font=FN(20, True), fill=(200, 230, 255, int(255 * sm((t - 26) / 2))), anchor='mt')
    if t < 7:
        q = rzut(mvp, np.array([2.4, 0, ZM + 0.8]))
        if q:
            d.text(q, 'obudowa: 768 kanałów laserowych', font=FN(22, True), fill=(200, 230, 255, int(255 * sm(t / 1.5))), anchor='mb')
    # panel matrycy diod
    cs = 8; x0, y0 = WY - 30 - NY * cs, 200
    d.rectangle((x0 - 8, y0 - 30, x0 + NY * cs + 8, y0 + NZ * cs + 8), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
    d.text((x0, y0 - 26), 'diody (jasność = z_c)', font=FN(15), fill=(220, 225, 230, 255))
    zz = np.clip(za / 0.9, 0, 1) ** 0.7
    for c in range(NK):
        r_, k_ = divmod(c, NY); v_ = int(30 + 225 * zz[c])
        d.rectangle((x0 + k_ * cs, y0 + r_ * cs, x0 + k_ * cs + cs - 2, y0 + r_ * cs + cs - 2), fill=(v_, int(v_ * 0.45), int(v_ * 0.4), 255))
    if faza in ('uczenie', 'koniec', 'wszystkie'):
        d.text((30, 190), f'krok {kr:,.0f}'.replace(',', ' ') + f'   (epoka {pl(kr / EP, 2)})', font=FN(22, True), fill=(235, 240, 245, 255))
        for n, (s, kl) in enumerate(((f'pary kanałów o przeciwnych kierunkach: {pary[ii]:.0%}', (255, 200, 120, 255)),
                                     (f'siła kanałów |v_c|: ×{pl(np.median(NORM[ii]) / np.median(NORM[0]), 2)}', (160, 220, 255, 255)),
                                     (f'kierunek kanału względem startu: cos {pl(cos0[ii], 2)}', (200, 255, 200, 255)))):
            d.text((30, 228 + 30 * n), s, font=FN(20, True), fill=kl)


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    CZASY = np.array([4.0, 11.0, 14.5, 17.0, 20.5, 23.0, 30.0, 45.0, 60.0, 67.0])
ff = None if PROBA else ffmpeg(HERE / 'lasery_9_reflektor_galwo.mp4', FPS)
for n, t in enumerate(CZASY):
    img = klatka(float(t))
    if PROBA:
        img.save(HERE / f'lasery_9_proba_{t:04.1f}.png')
    else:
        ff.stdin.write(img.convert('RGBA').tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(CZASY)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
