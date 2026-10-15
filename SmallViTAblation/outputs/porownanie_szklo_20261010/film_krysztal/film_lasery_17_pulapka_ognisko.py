"""Film 17: pułapka z prawdziwych danych — 768 wiązek ostatniego bloku MLP krzyżuje się w ognisku (token narzędzia 316), a wokół
stoi pozostałe 599 prawdziwych tokenów zdjęcia. Nowy film. Wagi i aktywacje z TEGO SAMEGO biegu (MalyViT_35000_od_zera_s47, 806
checkpointów do 60. epoki; przejście modelu w trybie eval odtwarza zapisany latent z R² = 1,0).

- Ognisko = x_316 (strumień rezydualny tuż przed MLP bloku 6). Wiązka c dochodzi do ogniska w kierunku d_c = P v̂_c, jasność z_c(316);
  przy ognisku strzałki sił z_c · P v_c i strzałka wypadkowa F(x_316) = Σ z_c v_c + b₂ (czerwona).
- Cząsteczka k ≠ 316 stoi w δ_k = P (x_k − x_316) i ma strzałkę zmiany po bloku P (F(x_k) − F(x_316)).
  Kolor z dokładnego rachunku w 192D po LayerNorm (bez γ, β): r_k = ‖ln(x_k + F_k) − ln(x_316 + F_316)‖ / ‖ln(x_k) − ln(x_316)‖;
  niebieski r < 1 (blok przyciąga token k do narzędzia), pomarańczowy r > 1 (odpycha). Surowe odległości rosną dla prawie wszystkich
  tokenów, bo strumień rezydualny rośnie. Pierścienie = tokeny narzędzia. Mapa r na zdjęciu oka z lewej.
- P = 3 główne składowe δ_k w ostatnim checkpoincie (ilustracja); skala dopasowana do kroku (licznik powiększenia).
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_17_pulapka_ognisko.py [proba]
"""
import subprocess
import sys
import time

import moderngl
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lasery_wspolne import EP, H, HERE, TOK, W, dane, oko

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
FPS, WY, HY, SS = 30, 1920, 1080, 2
NK = 768
ROOT = HERE.parents[2]
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
CACHE = HERE / 'pulapka_17_cache.npz'

D = dane(0)
kroki, iou, narz = D['kroki'].astype(int), D['iou'], D['narz']
SIATKA = np.stack(np.divmod(np.arange(TOK), 30)[::-1], 1).astype(np.float64)
sn = SIATKA[narz].mean(0)
FOK = int(np.flatnonzero(narz)[np.argmin(np.linalg.norm(SIATKA[narz] - sn, axis=1))])
OKO_F = oko()

if CACHE.exists():
    c_ = np.load(CACHE); X, F, Zf, V = c_['X'], c_['F'], c_['Zf'], c_['V']
else:
    sys.path.insert(0, str(KOD))
    import model as M
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_in = torch.from_numpy((OKO_F * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float().to(dev) / 255.
    net = M.MalyViT().to(dev).eval()
    zap = {}
    net.bloki[5].norm2.register_forward_hook(lambda m, i, o: zap.__setitem__('x', i[0]))
    net.bloki[5].mlp.register_forward_hook(lambda m, i, o: zap.__setitem__('f', o))
    net.bloki[5].mlp.act.register_forward_hook(lambda m, i, o: zap.__setitem__('z', o))
    X, F, Zf, V = [], [], [], []
    for n, k in enumerate(kroki):
        s = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
        net.load_state_dict(s)
        with torch.no_grad():
            net(x_in)
        X.append(zap['x'][0].float().cpu().numpy()); F.append(zap['f'][0].float().cpu().numpy())
        Zf.append(zap['z'][0, FOK].float().cpu().numpy()); V.append(s['bloki.5.mlp.fc2.weight'].float().numpy())
        if n % 100 == 0:
            log('checkpoint', n, k)
    X, F, Zf, V = (np.array(a, np.float32) for a in (X, F, Zf, V))
    np.savez_compressed(CACHE, X=X.astype(np.float16), F=F.astype(np.float16), Zf=Zf, V=V.astype(np.float16))
X, F, V = X.astype(np.float32), F.astype(np.float32), V.astype(np.float32)
NT = len(kroki)
log('dane', X.shape, F.shape, V.shape)

# kontrola: zgodność z zapisanym latentem (tryb eval, te same wagi)
_Zr = np.asarray(np.load(ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy',
                         mmap_mode='r')[[0, NT // 2, NT - 1], 0, FOK]).astype(np.float32)
for a_, b_ in zip(Zf[[0, NT // 2, NT - 1]], _Zr):
    log('R² latentu tokenu 316:', round(float(1 - ((a_ - b_) ** 2).sum() / ((b_ - b_.mean()) ** 2).sum()), 5))

# geometria względem ogniska; przyciąganie liczone tak, jak widzi to następna warstwa: po LayerNorm (bez γ, β), bo strumień
# rezydualny rośnie z bloku na blok i surowe odległości rosną dla wszystkich tokenów
DEL = X - X[:, FOK:FOK + 1]                                                       # (T, 600, 192)


def _ln(a):
    a = a - a.mean(-1, keepdims=True)
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-5)


_A0, _A1 = _ln(X), _ln(X + F)
R_ = np.linalg.norm(_A1 - _A1[:, FOK:FOK + 1], axis=2) / np.maximum(np.linalg.norm(_A0 - _A0[:, FOK:FOK + 1], axis=2), 1e-9)
del _A0, _A1
R_[:, FOK] = 1.0
U_, _, _ = np.linalg.svd((DEL[-1] - DEL[-1].mean(0)).T, full_matrices=False)
P = U_[:, :3].T                                                                   # (3, 192)
D3, DZ3 = DEL @ P.T, (F - F[:, FOK:FOK + 1]) @ P.T                                # położenia i zmiany w 3D
SKALA_T = np.percentile(np.linalg.norm(D3, axis=2), 95, axis=1)                   # (T,)
POW = SKALA_T[-1] / SKALA_T
PV = np.einsum('ad,tdc->tca', P, V)                                               # (T, 768, 3)
DIR = PV / np.maximum(np.linalg.norm(PV, axis=2, keepdims=True), 1e-9)
FW3 = F[:, FOK] @ P.T                                                             # wypadkowa przy ognisku (z b₂)
_fz = V * Zf[:, None, :]                                                           # (T, 192, 768) siły z_c v_c tokenu 316
ZNOS = np.linalg.norm(_fz.sum(2), axis=1) / np.linalg.norm(_fz, axis=1).sum(1)    # |wypadkowa| / Σ |siła|
del _fz
log('znoszenie: wypadkowa / suma sił', round(float(ZNOS[0]), 3), '→', round(float(ZNOS[-1]), 3))
ln = lambda a: np.log(np.clip(a, 1e-6, None))
STAT = dict(nar_przyc=(R_[:, narz] < 1).mean(1), tlo_przyc=(R_[:, ~narz] < 1).mean(1),
            nar_logr=np.median(ln(R_[:, narz & (np.arange(TOK) != FOK)]), 1), tlo_logr=np.median(ln(R_[:, ~narz]), 1))
log('koniec: narzędzie przyciągane', round(float(STAT['nar_przyc'][-1]), 2), 'tło przyciągane', round(float(STAT['tlo_przyc'][-1]), 2))
sys.path.insert(0, str(ROOT / 'app'))
from wspolne import kolory_kanalow                                                 # noqa: E402
KOL = kolory_kanalow()[:, :3]

# ---------------- oś czasu (pierwsze kroki wolno, epoki szybko) ----------------
FAZY = [(0, 'ognisko'), (9, 'tokeny'), (18, 'uczenie'), (58, 'koniec'), (63, None)]
T_U0 = 19.0
_kr, _t, kr_ = [], [], 0.0
while kr_ < kroki[-1]:
    _kr.append(kr_); kr_ += min(0.8 + kr_ / 250, 150)
N_U = len(_kr)
KR_U = np.array(_kr + [kroki[-1]])
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def krok_t(t):
    if t < T_U0:
        return float(kroki[-1]) if t < 18 else float(kroki[-1]) * (1 - sm((t - 18) / 1.0))
    i = int(min((t - T_U0) * FPS, N_U))
    return float(KR_U[i])


T_KON = T_U0 + N_U / FPS
log('uczenie trwa', round(N_U / FPS, 1), 's')
if T_KON + 5 > FAZY[-2][0]:
    FAZY[-2] = (T_KON + 0.5, 'koniec'); FAZY[-1] = (T_KON + 6, None)


def stan(t):
    kr = krok_t(t); fi = float(np.interp(kr, kroki, np.arange(NT))); i = int(min(fi, NT - 2)); u = fi - i
    L = lambda A: A[i] * (1 - u) + A[i + 1] * u
    s = 11.0 / L(SKALA_T)
    d = L(DIR); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return dict(kr=kr, ii=i if u < 0.5 else i + 1, D=L(D3) * s, DZ=L(DZ3) * s, R=L(R_), dir=d, z=L(Zf), Fw=L(FW3) * s,
                PV=L(PV) * s, pow=L(POW))


# ---------------- kamera ----------------
def mvp_kamery(t):
    az = np.radians(-60 + 7 * t); el = np.radians(20 + 6 * np.sin(t / 8)); di = 42.0 - 14 * (sm((t - 1) / 3) - sm((t - 8) / 2))
    oko_ = di * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    f = -oko_ / np.linalg.norm(oko_); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    Vm = np.eye(4); Vm[0, :3], Vm[1, :3], Vm[2, :3] = s, up, -f; Vm[:3, 3] = -Vm[:3, :3] @ oko_
    fov, asp, n, fa = np.radians(38), WY / HY, 0.1, 300.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n) / (n - fa); Pm[2, 3] = 2 * fa * n / (n - fa); Pm[3, 2] = -1
    return Pm @ Vm, 1 / np.tan(fov / 2) * HY * SS / 2


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
uniform int ksztalt; in vec4 v_k; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard;
  float a = ksztalt == 1 ? smoothstep(0.55, 0.75, sqrt(r2)) * (1.0 - smoothstep(0.85, 1.0, sqrt(r2))) : exp(-r2 * 3.0);
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


def strzalki(p0, p1, kol, glow=0.3):
    d = p1 - p0; L = np.linalg.norm(d, axis=1, keepdims=True); dn = d / np.maximum(L, 1e-9)
    bok = np.cross(dn, [0, 0, 1.0]); bok /= np.maximum(np.linalg.norm(bok, axis=1, keepdims=True), 1e-9)
    hl = np.minimum(glow, 0.4 * L); t_ = p1 - dn * hl
    seg = np.concatenate([np.stack([p0, p1], 1), np.stack([p1, t_ + bok * hl * 0.5], 1), np.stack([p1, t_ - bok * hl * 0.5], 1)]).reshape(-1, 3)
    return seg, np.repeat(np.tile(kol, (3, 1)), 2, 0)


def barwa_r(R):
    lr = np.clip(np.log(R) / 0.12, -1, 1)[:, None]
    return np.where(lr < 0, np.array([[0.35, 0.62, 1.0]]), np.array([[1.0, 0.55, 0.25]])), np.abs(lr)


FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
pl = lambda x, n=1: f'{x:.{n}f}'.replace('.', ',')
TEKSTY = {
    'ognisko': ('Ognisko: 768 wiązek bloku 6 krzyżuje się w tokenie narzędzia (prawdziwe dane)',
                ['Wiązka c dochodzi w kierunku d_c, jasność = aktywacja z_c tego tokenu.  Przy ognisku: siły z_c · v_c (kolor = kanał)',
                 f'i czerwona wypadkowa Σ z_c v_c w tej samej skali.  W 192D wypadkowa to tylko {ZNOS[-1]:.0%} sumy długości 768 sił: reszta się znosi (przeciwlasery).']),
    'tokeny': ('Pułapka: pozostałe 599 prawdziwych tokenów zdjęcia wokół ogniska',
               ['Strzałka = jak blok zmienia położenie tokenu względem tokenu narzędzia.  Niebieski: blok go przyciąga, pomarańczowy: odpycha —',
                'liczone w 192D po LayerNorm (tak, jak widzi to następna warstwa).  Pierścienie = tokeny narzędzia; mapa z lewej = to samo na zdjęciu.']),
    'uczenie': ('Uczenie od kroku 0 do 60. epoki: jak zmienia się pułapka',
                ['Ten sam bieg dla wag i aktywacji (R² latentu = 1,0).  Pierwsze kroki wolno, epoki szybko.',
                 'Panel z lewej: odsetek tokenów przyciąganych do narzędzia osobno dla narzędzia i dla tła.']),
    'koniec': ('Po 60 epokach', []),
}


def nakladka(img, faza, t, S):
    d = ImageDraw.Draw(img, 'RGBA')
    tyt, pod = TEKSTY[faza]
    if faza == 'koniec':
        ii = S['ii']
        pod = [f'Tokeny narzędzia przyciągane: {STAT["nar_przyc"][ii]:.0%}, tła: {STAT["tlo_przyc"][ii]:.0%}.  Mediana zmiany odległości od tokenu 316: '
               f'narzędzie ×{pl(np.exp(STAT["nar_logr"][ii]), 3)}, tło ×{pl(np.exp(STAT["tlo_logr"][ii]), 3)}.',
               'To nie jest pułapka zbierająca narzędzie w jeden punkt: ostatni blok rozsuwa tokeny narzędzia od siebie mocniej niż tło',
               '(gęsta głowa potrzebuje na każdym tokenie innych odległości do krawędzi ramki — to hipoteza do sprawdzenia).']
    d.text((WY / 2, 24), tyt, font=FN(34, True), fill=(235, 240, 245, 255), anchor='mt')
    for n, s in enumerate(pod):
        d.text((WY / 2, 76 + 30 * n), s, font=FN(21), fill=(255, 190, 120, 255), anchor='mt')
    d.text((22, HY - 58), 'Dane: MalyViT_35000_od_zera_s47, 806 checkpointów, zdjęcie TRAIN 0; x = strumień rezydualny przed MLP bloku 6, F = wyjście MLP (przejście modelu).',
           font=FN(17), fill=(205, 210, 215, 255))
    d.text((22, HY - 32), 'Rzut 192D → 3D: 3 główne składowe położeń względem narzędzia w 60. epoce (ilustracja); kolory i liczby z rachunku w 192D.',
           font=FN(17), fill=(205, 210, 215, 255))
    ii = S['ii']
    # mapa na zdjęciu
    if t >= 9:
        a = sm((t - 9.5) / 1.5)
        kol, si = barwa_r(np.clip(S['R'], 1e-6, None))
        mapa = (0.15 + 0.85 * kol * (0.25 + 0.75 * si)).reshape(H, W, 3)
        mapa[FOK // 30, FOK % 30] = [1, 1, 1]
        mapa = np.asarray(Image.fromarray((mapa * 255).astype(np.uint8)).resize((W * 16, H * 16), Image.NEAREST)).astype(np.float32)
        m = (OKO_F * 255 * 0.4 + mapa * 0.6).astype(np.uint8)
        rb = Image.new('RGBA', (W * 16 + 16, H * 16 + 40), (11, 16, 20, 235))
        rb.paste(Image.fromarray(m), (8, 32))
        ImageDraw.Draw(rb).text((8, 6), 'Na zdjęciu: niebieski = blok przyciąga do narzędzia, pomarańczowy = odpycha', font=FN(16), fill=(232, 232, 232))
        rb = rb.resize((int(rb.size[0] * 0.8), int(rb.size[1] * 0.8)))
        rb.putalpha(Image.eval(Image.new('L', rb.size, 255), lambda v: int(v * a)))
        img.paste(rb, (22, 160), rb)
        x0, y0 = 30, 160 + rb.size[1] + 20
        d.rectangle((x0 - 10, y0 - 8, x0 + 380, y0 + 150), fill=(5, 9, 12, int(225 * a)), outline=(42, 51, 58, int(255 * a)))
        for n, (s, k) in enumerate(((f'krok {S["kr"]:,.0f}'.replace(',', ' ') + f'   (epoka {pl(S["kr"] / EP, 2)})', (235, 240, 245)),
                                    (f'narzędzie przyciągane: {STAT["nar_przyc"][ii]:.0%}', (120, 175, 255)),
                                    (f'tło przyciągane: {STAT["tlo_przyc"][ii]:.0%}', (255, 170, 110)),
                                    (f'IoU: {pl(float(iou[ii]), 2)}', (74, 240, 255)),
                                    (f'powiększenie chmury: ×{pl(float(S["pow"]), 1)}', (190, 210, 230)))):
            d.text((x0, y0 + 28 * n), s, font=FN(19, True), fill=k + (int(255 * a),))


def klatka(t):
    faza = [n for (ts, n) in FAZY if ts <= t and n][-1]
    S = stan(t)
    mvp, skala = mvp_kamery(t)
    mb = mvp.T.astype('f4').tobytes()
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mb); kropki['mvp'].write(mb); kropki['skala'].value = skala
    # wiązki zbiegające się w ognisku
    z = np.clip(S['z'], 0, None); a = np.clip(z / max(np.percentile(z, 97), 1e-6), 0, 1)
    on = sm(t / 2.5) * (1 - 0.5 * sm((t - 9) / 2))
    p0 = -S['dir'] * 26.0
    seg = np.stack([p0, np.zeros_like(p0)], 1).reshape(-1, 3)
    kk = np.concatenate([0.3 + 0.7 * KOL, (a * 0.7 * on)[:, None]], 1)
    kk = np.repeat(kk, 2, 0); kk[0::2, 3] *= 0.1
    linie['szer'].value = 2.2
    rysuj(linie, moderngl.LINES, seg, kk)
    # siły poszczególnych wiązek przy ognisku (skala: 97. percentyl długości = 6 jednostek) i wypadkowa w tej samej skali
    fv = S['PV'] * z[:, None]
    S['skf'] = 6.0 / max(np.percentile(np.linalg.norm(fv, axis=1), 97), 1e-9)
    fs = sm((t - 3) / 2) * (1 - 0.7 * sm((t - 9) / 2))
    if fs > 0:
        sg, kk = strzalki(np.zeros((NK, 3)), fv * S['skf'], np.concatenate([0.3 + 0.7 * KOL, (a * 0.9 * fs)[:, None]], 1), 0.35)
        linie['szer'].value = 2.0
        rysuj(linie, moderngl.LINES, sg, kk)
    # tokeny
    tk = sm((t - 9) / 2)
    if tk > 0:
        idx = np.arange(TOK) != FOK
        kol, si = barwa_r(np.clip(S['R'], 1e-6, None))
        rysuj(kropki, moderngl.POINTS, S['D'][idx], np.concatenate([kol[idx] * (0.5 + 0.8 * si[idx]), np.full((idx.sum(), 1), 0.9 * tk)], 1), np.full(int(idx.sum()), 0.45))
        sg, kk = strzalki(S['D'][idx], S['D'][idx] + S['DZ'][idx], np.concatenate([kol[idx], (0.25 + 0.6 * si[idx]) * tk], 1), 0.35)
        linie['szer'].value = 1.5
        rysuj(linie, moderngl.LINES, sg, kk)
    rysuj(kropki, moderngl.POINTS, np.zeros((1, 3)), np.array([[0.85, 1.0, 0.9, 2.5]]), np.array([1.6]))
    fb_ldr.use(); ctx.disable(moderngl.BLEND); hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    sg, kk = strzalki(np.zeros((1, 3)), (S['PV'] * z[:, None]).sum(0)[None] * S['skf'], np.array([[1.0, 0.25, 0.22, 0.95 * sm((t - 4) / 2) * (1 - 0.5 * sm((t - 10) / 2))]]), 0.8)
    linie['szer'].value = 6.0
    rysuj(linie, moderngl.LINES, sg, kk)
    if tk > 0:
        kropki['ksztalt'].value = 1
        nz = narz & (np.arange(TOK) != FOK)
        rysuj(kropki, moderngl.POINTS, S['D'][nz], np.tile([0.29, 0.94, 1.0, 0.85 * tk], (int(nz.sum()), 1)), np.full(int(nz.sum()), 1.2))
        kropki['ksztalt'].value = 0
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, faza, t, S)
    return img


CZASY = np.arange(int(FAZY[-1][0] * FPS)) / FPS
if PROBA:
    for t in (5.0, 14.0, 22.0, 30.0, T_KON + 2):
        klatka(float(t)).save(HERE / f'lasery_17_proba_{t:04.1f}.png')
else:
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WY}x{HY}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '20', '-pix_fmt', 'yuv420p', str(HERE / 'lasery_17_pulapka_ognisko.mp4')],
                          stdin=subprocess.PIPE)
    for n, t in enumerate(CZASY):
        ff.stdin.write(klatka(float(t)).tobytes())
        if n % 300 == 0:
            log(f'klatka {n}/{len(CZASY)}')
    ff.stdin.close(); ff.wait()
log('gotowe')
