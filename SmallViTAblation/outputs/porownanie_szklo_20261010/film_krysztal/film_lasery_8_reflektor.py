"""Film 8: jeden reflektor matrycowy = ostatni (6.) blok MLP MałegoViT (nowy film; zasady w ZASADY_LASER_KOLOID.md, Z2, Z5, Z11).

Układ optyczny (poglądowy, ale sterowany prawdziwymi wagami i aktywacjami):
    matryca 768 diod (32 × 24)  →  mikrolusterko przy każdej diodzie  →  soczewka projekcyjna  →  płaszczyzna ogniskowa z cząsteczką.
- dioda c = neuron c warstwy ukrytej; jasność = aktywacja z_c = GELU(w_c · LN(x) + b_c) wybranego tokenu (część dodatnia);
- kąt lusterka c = θ_c = κ · P v_c, gdzie v_c = kolumna c macierzy fc2 (kierunek pchnięcia), P = stały rzut 192D → 2D
  (dwie główne składowe kierunków v_c w kroku 22 974);
- soczewka cienka: promień o kącie θ trafia w płaszczyźnie ogniskowej w punkt F·θ niezależnie od miejsca wyjścia, więc plamka c leży
  w punkcie F·θ_c, a suma Σ_c z_c · F·θ_c = F·κ·P(Σ_c z_c v_c) to rzut pchnięcia, które blok MLP daje temu tokenowi (strzałka).

Fazy: budowa reflektora; jedna cząsteczka w kroku 0; przełączanie matrycy dla różnych tokenów przy wagach z 21. epoki (bez uczenia);
uczenie od kroku 0 do 22 974 (lusterka się przestawiają, rosną, powstają pary przeciwne); podsumowanie.
Dane: fc2 bloku 5 z 341 checkpointów biegu 35 000 (assets/.../01_bieg_35000_gesty_zapis/eden/gesty/wagi) i aktywacje 768 neuronów
zdjęcia TRAIN 0 w tych samych krokach (assets/.../ruch60_rp128_50tr_50val/pelne768.npy).
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_8_reflektor.py [proba]"""
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
                  .float().numpy() for k in KW])                                    # (341, 192, 768): kolumny v_c
    np.savez_compressed(CACHE, KW=KW, V=V)
NT = len(KW)
idx_lat = np.searchsorted(kroki_all, KW)
assert np.all(kroki_all[idx_lat] == KW)
Z = np.asarray(np.load(LATENT, mmap_mode='r')[idx_lat, 0]).astype(np.float32)        # (341, 600, 768)
log(f'wagi {V.shape}, aktywacje {Z.shape}')

# rzut 192D → 2D: dwie główne składowe kierunków v_c w ostatnim checkpoincie
Vn_end = V[-1] / np.linalg.norm(V[-1], axis=0)
U_, _, _ = np.linalg.svd(Vn_end - Vn_end.mean(1, keepdims=True), full_matrices=False)
P2 = U_[:, :2].T                                                                    # (2, 192)
PV = np.einsum('ad,tdc->tca', P2, V)                                                # (341, 768, 2) rzut kierunków pchnięcia
NORM = np.linalg.norm(V, axis=1)                                                    # (341, 768) |v_c|
Vn = V / NORM[:, None, :]
pary = np.array([(np.where(np.eye(NK, dtype=bool), 0.0, Vn[t].T @ Vn[t]).min(1) < -0.5).mean() for t in range(NT)])
cos0 = np.median((Vn * Vn[:1]).sum(1), axis=1)                                      # kierunek lasera względem kroku 0
log(f'pary przeciwne: {pary[0]:.0%} → {pary[-1]:.0%}; |v| ×{np.median(NORM[-1]) / np.median(NORM[0]):.2f}; cos do startu {cos0[-1]:.2f}')

# geometria (jednostki sceny)
XM, XL, RL, FOK_ = 2.0, 16.0, 18.0, 20.0                                           # lusterka, soczewka, promień soczewki, ogniskowa
XS = XL + FOK_                                                                      # płaszczyzna ogniskowa
OFF_MAX = 7.0                                                                       # największe odsunięcie plamki
KAPPA = OFF_MAX / FOK_ / float(np.percentile(np.linalg.norm(PV, axis=2), 99.5))     # kąt = KAPPA · P v
gy, gz = np.meshgrid((np.arange(NY) - (NY - 1) / 2) * 0.75, (np.arange(NZ) - (NZ - 1) / 2) * 0.75)
DIODY = np.stack([np.zeros(NK), gy.ravel(), -gz.ravel()], 1)                        # dioda c w wierszu c // 32, kolumnie c % 32
SKALA = 2.0
Zr = np.clip(Z, 0, None) / SKALA

# wybór cząsteczek
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)       # (kolumna, wiersz)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
rng = np.random.default_rng(5)
tl = np.flatnonzero(~narz)
dal = tl[np.argsort(-np.linalg.norm(SIATKA[tl] - sn, axis=1))[:200]]
TOKENY = [FOK] + list(rng.choice(np.flatnonzero(narz & (np.arange(TOK) != FOK)), 2, replace=False)) + list(rng.choice(dal, 3, replace=False))
PCH_SKALA = 0.85 * OFF_MAX / float(np.percentile(np.linalg.norm(np.einsum('tjc,tca->tja', Zr[:, TOKENY], PV), axis=2), 99))
log('tokeny', TOKENY, 'narzędzie:', [bool(narz[j]) for j in TOKENY])

# kolor plamki = kierunek pchnięcia w rzucie (przeciwne kierunki = przeciwne barwy)
def barwa(kat):
    h = (kat / (2 * np.pi)) % 1.0
    k = lambda n: np.clip(np.abs(((h * 6 + n) % 6) - 3) - 1, 0, 1)
    return np.stack([k(0), k(4), k(2)], -1) * 0.85 + 0.15


# ---------------- oś czasu ----------------
FAZY = [(0, 'budowa'), (7, 'krok0'), (16, 'przelacz'), (29, 'uczenie'), (59, 'koniec'), (66, None)]
T_P0, T_P1 = 16.0, 29.0
T_U0, T_U1, T_UP = 30.5, 57.5, 41.0                                                 # 30,5–41 s: kroki 0–2188, 41–57,5 s: do 22 974
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def stan(t):
    """→ (indeks checkpointu jako float, token, waga przejścia do następnego tokenu, następny token)."""
    if t < T_P0:
        return 0.0, FOK, 0.0, FOK
    if t < T_P1:
        u = (t - T_P0) / (T_P1 - T_P0) * len(TOKENY)
        n = int(min(u, len(TOKENY) - 1e-6)); fr = u - n
        nast = TOKENY[(n + 1) % len(TOKENY)] if n + 1 < len(TOKENY) else FOK
        return float(NT - 1), TOKENY[n], sm((fr - 0.75) / 0.25), nast
    if t < T_U0:
        return float(NT - 1) * (1 - sm((t - T_P1) / (T_U0 - T_P1))), FOK, 0.0, FOK
    if t < T_UP:
        kr = 2 * EP * (t - T_U0) / (T_UP - T_U0)
    else:
        kr = min(KW[-1], 2 * EP + (KW[-1] - 2 * EP) * (t - T_UP) / (T_U1 - T_UP))
    return float(np.interp(kr, KW, np.arange(NT))), FOK, 0.0, FOK


def wez(fi, j, w=0.0, j2=None):
    i = int(min(fi, NT - 2)); u = fi - i
    th = (PV[i] * (1 - u) + PV[i + 1] * u) * KAPPA                                   # (768, 2) kąty lusterek
    z = Zr[i, j] * (1 - u) + Zr[i + 1, j] * u
    if w > 0:
        z = z * (1 - w) + (Zr[i, j2] * (1 - u) + Zr[i + 1, j2] * u) * w
    kr = KW[i] * (1 - u) + KW[min(i + 1, NT - 1)] * u
    return th, z, kr, i, u


KAM = [(0, -78, 22, 78), (7, -52, 16, 66), (16, -40, 12, 62), (29, -30, 10, 60), (59, -40, 18, 62), (66, -46, 22, 68)]
CEL = np.array([XS * 0.55, 0.0, 0.0])


def macierz(t):
    for (ta, *pa), (tb, *pb) in zip(KAM[:-1], KAM[1:]):
        if t <= tb:
            u = sm((t - ta) / (tb - ta)); break
    else:
        (ta, *pa), (tb, *pb), u = KAM[-2], KAM[-1], 1.0
    az, el, di = [pa[k] + (pb[k] - pa[k]) * u for k in range(3)]
    a, e = np.radians(az), np.radians(el)
    oko = CEL + di * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = CEL - oko; f /= np.linalg.norm(f); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    Vm = np.eye(4); Vm[0, :3], Vm[1, :3], Vm[2, :3] = s, up, -f; Vm[:3, 3] = -Vm[:3, :3] @ oko
    fov, asp, n, fa = np.radians(40), WY / HY, 0.1, 500.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n) / (n - fa); Pm[2, 3] = 2 * fa * n / (n - fa); Pm[3, 2] = -1
    return Pm @ Vm, 1 / np.tan(fov / 2) * HY * SS / 2


def rzut(mvp, p):
    q = mvp @ np.append(p, 1.0)
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
void main() { gl_Position = mvp * vec4(pos, 1.0); gl_PointSize = max(1.5, rozm * skala / gl_Position.w); v_k = kol; }''',
                     fragment_shader='''#version 430
uniform int ksztalt; in vec4 v_k; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d);
  float a;
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
    dane_ = [poz.astype('f4'), kol.astype('f4')] + ([rozm.astype('f4')[:, None]] if rozm is not None else [])
    b = ctx.buffer(np.concatenate(dane_, 1).tobytes())
    fmt = '3f 4f 1f' if rozm is not None else '3f 4f'
    v = ctx.vertex_array(prog, [(b, fmt, *(['pos', 'kol', 'rozm'] if rozm is not None else ['pos', 'kol']))])
    v.render(tryb); v.release(); b.release()


def okrag(x, r, n=120):
    t_ = np.linspace(0, 2 * np.pi, n + 1)
    p = np.stack([np.full_like(t_, x), r * np.cos(t_), r * np.sin(t_)], 1)
    return np.stack([p[:-1], p[1:]], 1).reshape(-1, 3)


def prostokat(x, hy, hz):
    c = np.array([[x, -hy, -hz], [x, hy, -hz], [x, hy, hz], [x, -hy, hz]])
    return np.stack([c, np.roll(c, -1, 0)], 1).reshape(-1, 3)


# stała scenografia
OBUDOWA = np.concatenate([prostokat(-0.4, 12.6, 9.6), prostokat(XM + 0.4, 12.6, 9.6)])
SOCZ = np.concatenate([okrag(XL, RL), okrag(XL - 0.9, RL * 0.93), okrag(XL + 0.9, RL * 0.93)])
siat_ = [prostokat(XS, 9.0, 9.0)]
for v_ in np.linspace(-9, 9, 7)[1:-1]:
    siat_.append(np.array([[XS, v_, -9], [XS, v_, 9], [XS, -9, v_], [XS, 9, v_]]))
EKRAN = np.concatenate(siat_)
OS_ = np.array([[-2.0, 0, 0], [XS + 4, 0, 0]])

# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_S, F_M = FN(34, True), FN(21), FN(17)
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
TEKSTY = {
    'budowa': ('Reflektor matrycowy = jeden blok MLP (ostatni, 6. blok MałegoViT)',
               ['768 diod = 768 neuronów warstwy ukrytej.  Jasność diody = aktywacja z_c (GELU) dla jednej cząsteczki.',
                'Dioda świeci na swoje lusterko: jego kąt = kierunek pchnięcia v_c (kolumna fc2).  Soczewka zamienia kąt na miejsce w ognisku.']),
    'krok0': ('Jedna cząsteczka przed uczeniem (krok 0)',
              ['Cząsteczka (token narzędzia) stoi w ognisku.  Plamka = dokąd pcha jedna wiązka;  czerwona strzałka = suma Σ z_c · v_c,',
               'czyli pchnięcie, które ten blok daje cząsteczce.  Na starcie lusterka są ustawione losowo i słabe: plamki blisko środka.']),
    'przelacz': ('Matryca przełącza się przy każdej cząsteczce, bez uczenia',
                 ['Te same lusterka (wagi z 21. epoki), kolejne cząsteczki: zmienia się tylko, które diody świecą i jak jasno (GELU).',
                  'Jak reflektor matrycowy w aucie: wzór świecenia zależy od sceny, a optyka zostaje ta sama.']),
    'uczenie': ('Uczenie: lusterka się przestawiają',
                ['Każde lusterko obraca się osobno (wspólny obrót całego reflektora prawie nie występuje), lusterka rosną i ustawiają się w pary.',
                 'Kolor plamki = kierunek pchnięcia.  Przeciwlasery lądują po przeciwnych stronach ogniska, w przeciwnych barwach.']),
    'koniec': ('Po 21 epokach', []),
}
KONIEC_TXT = [f'Pary przeciwnych lusterek (cos < −0,5): {pary[0]:.0%} → {pary[-1]:.0%}.   Siła lusterka |v_c|: ×{pl(np.median(NORM[-1]) / np.median(NORM[0]), 2)}.',
              f'Kierunek tego samego lusterka względem kroku 0: mediana cos {pl(cos0[-1], 2)}, czyli obrót o około {np.degrees(np.arccos(cos0[-1])):.0f}°.',
              'Szybko (przy każdej cząsteczce) przełącza się matryca diod;  powoli (przez uczenie) przestawiają się lusterka.']


def nakladka(img, faza, t, z, th, kr, i, j, mvp, przes):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    if faza == 'koniec':
        pod = KONIEC_TXT
    d.text((WY / 2, 24), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        d.text((WY / 2, 76 + 30 * n), s, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), 'Dane: wagi fc2 bloku 6 z 341 checkpointów (kroki 0–22 974) i aktywacje 768 neuronów zdjęcia TRAIN 0 w tych samych krokach.',
           font=F_M, fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Rzut 192D → 2D: dwie główne składowe kierunków v_c w kroku 22 974 (ilustracja). Optyka poglądowa: soczewka cienka, układ diod umowny.',
           font=F_M, fill=(205, 210, 215, 255))
    # podpisy elementów w 3D
    if t < 16:
        a = int(255 * min(1.0, t / 1.5, (16 - t) / 1.5))
        for p, s in (((0, -12.6, 10.4), 'matryca 768 diod'), ((XM, 12.6, 10.4), 'mikrolusterka (kąt = v_c)'),
                     ((XL, 0, -RL - 1.0), 'soczewka projekcyjna'), ((XS, 0, 10.2), 'płaszczyzna ogniskowa'), ((XS, 0, -1.6), 'cząsteczka')):
            x, y = rzut(mvp, np.array(p, float))
            d.text((x, y), s, font=FN(20, True), fill=(200, 230, 255, a), anchor='mt' if s.startswith('soczewka') else 'mb')
    # panel matrycy diod (widok z przodu)
    cs = 8; x0, y0 = WY - 30 - NY * cs, 200
    d.rectangle((x0 - 8, y0 - 30, x0 + NY * cs + 8, y0 + NZ * cs + 8), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
    d.text((x0, y0 - 26), 'matryca diod (jasność = z_c)', font=FN(15), fill=(220, 225, 230, 255))
    zz = np.clip(z / 0.9, 0, 1) ** 0.7
    for c in range(NK):
        r_, k_ = divmod(c, NY); v_ = int(30 + 225 * zz[c])
        d.rectangle((x0 + k_ * cs, y0 + r_ * cs, x0 + k_ * cs + cs - 2, y0 + r_ * cs + cs - 2), fill=(v_, v_, int(v_ * 0.8), 255))
    # płaszczyzna ogniskowa w 2D
    R2 = 120; cx, cy = WY - 30 - R2, y0 + NZ * cs + 60 + R2
    d.rectangle((cx - R2 - 8, cy - R2 - 30, cx + R2 + 8, cy + R2 + 8), fill=(5, 9, 12, 225), outline=(42, 51, 58, 255))
    d.text((cx - R2, cy - R2 - 26), 'płaszczyzna ogniskowa (widok z przodu)', font=FN(15), fill=(220, 225, 230, 255))
    d.line((cx - R2, cy, cx + R2, cy), fill=(40, 50, 58, 255)); d.line((cx, cy - R2, cx, cy + R2), fill=(40, 50, 58, 255))
    off = th * FOK_ / OFF_MAX * R2
    dl_ = np.linalg.norm(off, axis=1, keepdims=True)
    off = off * np.minimum(1.0, (R2 - 4) / np.maximum(dl_, 1e-9))                   # plamki poza ramką przyklejone do brzegu
    kol = (barwa(np.arctan2(th[:, 1], th[:, 0])) * 255).astype(int)
    for c in np.argsort(z):
        if z[c] > 0.02:
            r_ = 1.0 + 4.0 * min(1.0, z[c] / 0.9)
            x, y = cx + off[c, 0], cy - off[c, 1]
            d.ellipse((x - r_, y - r_, x + r_, y + r_), fill=(*kol[c], int(80 + 175 * min(1.0, z[c] / 0.6))))
    px, py = cx + przes[0] / OFF_MAX * R2, cy - przes[1] / OFF_MAX * R2
    d.line((cx, cy, px, py), fill=(255, 70, 70, 255), width=4)
    d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(230, 255, 235, 255))
    # opis cząsteczki i licznik
    d.text((30, 185), f'cząsteczka {j}: wiersz {j // 30}, kolumna {j % 30}, ' + ('narzędzie' if narz[j] else 'tło'),
           font=FN(22, True), fill=(235, 240, 245, 255))
    d.text((30, 217), f'krok {kr:,.0f}'.replace(',', ' ') + f'   (epoka {pl(kr / EP, 2)})', font=FN(20), fill=(205, 210, 215, 255))
    if faza in ('uczenie', 'koniec'):
        ii = min(i, NT - 1)
        for n, (s, kol_) in enumerate(((f'pary przeciwnych lusterek: {pary[ii]:.0%}', (255, 200, 120, 255)),
                                       (f'siła lusterek |v_c|: ×{pl(np.median(NORM[ii]) / np.median(NORM[0]), 2)}', (160, 220, 255, 255)),
                                       (f'kierunek lusterka względem startu: cos {pl(cos0[ii], 2)}', (200, 255, 200, 255)))):
            d.text((30, 257 + 30 * n), s, font=FN(20, True), fill=kol_)


def klatka(t):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    fi, j, w, j2 = stan(t)
    th, z, kr, i, u = wez(fi, j, w, j2)
    jt = j2 if w > 0.5 else j
    mvp, skala = macierz(t)
    mvp_b = mvp.T.astype('f4').tobytes()
    zap = sm((t - 7.0) / 2.5)                                                       # zapalenie diod
    bud = sm(t / 3.0)
    zz = z * zap
    S = np.concatenate([np.full((NK, 1), XS), th * FOK_], 1)                        # plamki w płaszczyźnie ogniskowej
    Mp = DIODY + np.array([XM, 0, 0])
    H = np.concatenate([np.full((NK, 1), XL), Mp[:, 1:] + th * (XL - XM)], 1)       # trafienie w soczewkę
    kol = barwa(np.arctan2(th[:, 1], th[:, 0]))
    przes = np.einsum('c,ca->a', zz, th / KAPPA) * PCH_SKALA                       # rzut Σ z_c v_c w jednostkach ekranu

    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mvp_b); kropki['mvp'].write(mvp_b); kropki['skala'].value = skala
    linie['szer'].value = 2.0
    rysuj(linie, moderngl.LINES, OBUDOWA, np.tile([0.30, 0.42, 0.55, 0.5 * bud], (len(OBUDOWA), 1)))
    rysuj(linie, moderngl.LINES, SOCZ, np.tile([0.45, 0.70, 0.95, 0.55 * bud], (len(SOCZ), 1)))
    rysuj(linie, moderngl.LINES, EKRAN, np.tile([0.30, 0.40, 0.50, 0.30 * bud], (len(EKRAN), 1)))
    rysuj(linie, moderngl.LINES, OS_, np.tile([0.35, 0.35, 0.40, 0.25 * bud], (2, 1)))
    # wiązki: dioda → lusterko → soczewka → plamka
    a = np.clip(zz, 0, 1.2)
    on = a > 0.01
    if on.any():
        g = 0.55
        seg = np.concatenate([np.stack([DIODY, Mp], 1)[on], np.stack([Mp, H], 1)[on], np.stack([H, S], 1)[on]]).reshape(-1, 3)
        kk = np.concatenate([np.concatenate([np.tile([1.0, 0.95, 0.8], (on.sum(), 1)), (a[on] * g)[:, None]], 1),
                             np.concatenate([kol[on], (a[on] * g * 0.9)[:, None]], 1),
                             np.concatenate([kol[on], (a[on] * g * 1.1)[:, None]], 1)])
        kk = np.repeat(kk, 2, 0)
        linie['szer'].value = 1.4
        rysuj(linie, moderngl.LINES, seg, kk)
    # diody (kwadraty), lusterka (kwadraciki w kolorze kierunku), plamki
    kropki['ksztalt'].value = 2
    rysuj(kropki, moderngl.POINTS, DIODY, np.concatenate([np.tile([0.35, 0.35, 0.40], (NK, 1)) + np.clip(zz, 0, 1)[:, None] * np.array([2.2, 2.0, 1.5]),
                                                            np.full((NK, 1), 0.9 * bud)], 1), np.full(NK, 0.36))
    rysuj(kropki, moderngl.POINTS, Mp, np.concatenate([kol * (0.25 + 0.75 * np.clip(np.linalg.norm(th, axis=1) / (OFF_MAX / FOK_), 0, 1))[:, None],
                                                         np.full((NK, 1), 0.8 * bud)], 1), np.full(NK, 0.26))
    kropki['ksztalt'].value = 0
    if on.any():
        rysuj(kropki, moderngl.POINTS, S[on], np.concatenate([kol[on], (a[on] * 1.6)[:, None]], 1), np.full(int(on.sum()), 0.9))
    # cząsteczka i strzałka pchnięcia
    P0 = np.array([XS, 0, 0]); P1 = P0 + np.array([0, przes[0], przes[1]])
    rysuj(kropki, moderngl.POINTS, P0[None], np.array([[0.6, 1.0, 0.7, 2.5 * bud]]), np.array([1.4]))
    fb_ldr.use(); ctx.disable(moderngl.BLEND)
    hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    L = np.linalg.norm(przes)
    if L > 0.05 and zap > 0:
        dn = (P1 - P0) / L; bok = np.cross(dn, [1.0, 0, 0]); hl = min(0.9, 0.4 * L)
        tyl = P1 - dn * hl
        seg = np.array([P0, P1, P1, tyl + bok * hl * 0.55, P1, tyl - bok * hl * 0.55])
        linie['szer'].value = 6.0
        rysuj(linie, moderngl.LINES, seg, np.tile([1.0, 0.28, 0.25, zap], (6, 1)))
    kropki['ksztalt'].value = 1
    rysuj(kropki, moderngl.POINTS, P0[None], np.array([[0.85, 1.0, 0.9, 0.9 * bud]]), np.array([0.7]))
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, faza, t, zz, th, kr, i if u < 0.5 else i + 1, jt, mvp, przes)
    return img


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    CZASY = np.array([4.0, 12.0, 19.0, 25.0, 36.0, 55.0, 62.0])
ff = None if PROBA else ffmpeg(HERE / 'lasery_8_reflektor.mp4', FPS)
for n, t in enumerate(CZASY):
    img = klatka(float(t))
    if PROBA:
        img.save(HERE / f'lasery_8_proba_{t:04.1f}.png')
    else:
        ff.stdin.write(img.convert('RGBA').tobytes())
        if n % 150 == 0:
            log(f'klatka {n}/{len(CZASY)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
