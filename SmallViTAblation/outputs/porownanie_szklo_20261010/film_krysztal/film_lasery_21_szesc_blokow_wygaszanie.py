"""Film 21: jak film 20 (sześć bloków w każdym kroku uczenia), ale każdy podkrok to „świeci, przesuwa i gaśnie”: wiązki zapalają się
na początku podkroku, cząsteczki przesuwają się w trakcie, a jasność wiązek = część przesunięcia, która jeszcze została (1 − u), więc
gasną dokładnie wtedy, gdy cząsteczka dochodzi do celu. Siły (wagi uwagi, jasności z_jc) liczone są raz, ze stanu na początku podkroku,
jak w sieci: gaśnięcie nie oznacza, że siła słabnie z odległością. Uwaga nie ciągnie cząsteczek w stronę tokenów, na które patrzą
(cos przesunięcia z kierunkiem do ważonego środka ≈ 0); przesuwa o średnią ważoną wektorów wartości. Dane i cache z filmu 20.

Opis filmu 20 (bez zmian w treści): nowy film na bazie filmu 12 (stałe reflektory na zewnętrznej sferze, szklana sfera,
jedno załamanie na powierzchni), z podkrokami bloków:
- w każdym kroku uczenia obraz przechodzi przez 6 stacji = 6 bloków enkodera; każda stacja ma 2 podkroki:
  1) uwaga: wiązki między cząsteczkami (jasność = waga uwagi A_jk, 3 głowy = 3 barwy; rysowane 2 najsilniejsze wiązki na cząsteczkę
     i głowę), cząsteczki przesuwają się o pełne wyjście uwagi;
  2) MLP: 768 reflektorów tej stacji (kolor stacji); jasność wiązki = z_jc ze znakiem (błękitne = ujemna jasność, ten sam laser
     słabo ciągnie), kierunek w sferze = P v_c, wspólny dla 600 wiązek reflektora; cząsteczki przesuwają się o wyjście MLP;
- położenie cząsteczki = P (x_j − x̄): rzut 3D jej stanu w strumieniu rezydualnym względem średniej po tokenach (nie suma wiązek
  jak w filmie 12). Rzut jest liniowy, więc przesunięcie w podkroku MLP to dokładnie Σ_c (z_jc − z̄_c) P v_c (b₂ znosi się);
- P = 3 główne składowe stanów wszystkich bloków w 60. epoce (każdy stan znormalizowany do RMS 1); skala kamery jest stała w obrębie
  bloku (oba podkroki dokładne), a między blokami kamera dobiera nową skalę, bo strumień rezydualny rośnie z głębokością;
- wagi i aktywacje z biegu MalyViT_35000_od_zera_s47 (ten sam bieg co zapisany latent); stany z przejścia modelu w trybie eval.
Fazy: wstęp; jedno przejście w 60. epoce (6 bloków × uwaga + MLP, powoli); uczenie: 24 kroki od 0 do 60. epoki, w każdym 12 podkroków;
koniec. Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_21_szesc_blokow_wygaszanie.py [dane | proba]"""
import subprocess
import sys
import time

import moderngl
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from lasery_wspolne import EP, HERE, TOK, dane, oko

TRYB = sys.argv[1] if len(sys.argv) > 1 else ''
PROBA = TRYB == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
FPS, WY, HY, SS = 30, 1920, 1080, 2
NK, NB, KTOP = 768, 6, 2
ROOT = HERE.parents[2]
WAGI = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi'
KOD = ROOT / 'assets/RuchCzasteczek_ViT_szklo_2026-10/07_kod/eden/2026-10-09_maly_vit_od_zera_35000_21epok/kod'
CACHE = HERE / 'szesc_blokow_20_cache.npz'
sys.path.insert(0, str(ROOT / 'app'))
from wspolne import kierunki                                                        # noqa: E402

D = dane(0)
kroki_all, iou_all, narz = D['kroki'].astype(int), D['iou'], D['narz'].astype(bool)
DOST = np.array(sorted(int(f.name[1:6]) for f in WAGI.glob('k*.pt')))
najbl = lambda k: int(DOST[np.argmin(np.abs(DOST - k))])
_CEL = [0, 10, 30, 60, 100, 200, 300, 500, 700, 1000, 1500, 2200, 3300, 5000, 7400, 11000, 16400, 22900, 30000, 38000, 46000,
        54000, 60000, 65640]
KR = np.array(sorted(set(najbl(k) for k in _CEL)))
NT = len(KR)


# ---------------- dane: stany przed i po każdym podkroku, latent MLP, wagi uwagi, kierunki v_c wszystkich 6 bloków ----------------
def buduj():
    sys.path.insert(0, str(KOD))
    import model as M
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    xin = torch.from_numpy((oko() * 255).round().astype(np.uint8)).permute(2, 0, 1)[None].float().to(dev) / 255.
    net = M.MalyViT().to(dev).eval()
    zap = {}
    for l, b in enumerate(net.bloki):
        b.register_forward_pre_hook(lambda m, i, l=l: zap.__setitem__(('in', l), i[0].detach()))
        b.norm2.register_forward_hook(lambda m, i, o, l=l: zap.__setitem__(('til', l), i[0].detach()))
        b.attn.qkv.register_forward_hook(lambda m, i, o, l=l: zap.__setitem__(('qkv', l), o.detach()))
        b.mlp.act.register_forward_hook(lambda m, i, o, l=l: zap.__setitem__(('z', l), o.detach()))
        b.register_forward_hook(lambda m, i, o, l=l: zap.__setitem__(('out', l), o.detach()))
    S = np.zeros((NT, NB, 3, TOK, 192), np.float32); Z = np.zeros((NT, NB, TOK, NK), np.float32)
    V = np.zeros((NT, NB, 192, NK), np.float32)
    AI = np.zeros((NT, NB, 3, TOK, KTOP), np.int16); AA = np.zeros((NT, NB, 3, TOK, KTOP), np.float32)
    BL = np.zeros((NT, NB, 2))
    for n, k in enumerate(KR):
        s = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
        net.load_state_dict(s)
        with torch.no_grad():
            net(xin)
            for l, b in enumerate(net.bloki):
                x0, x1, x2 = zap[('in', l)][0], zap[('til', l)][0], zap[('out', l)][0]
                z_ = zap[('z', l)][0].float().cpu().numpy()                             # przed kontrolą: ona nadpisuje haki
                h, hd = b.attn.num_heads, b.attn.head_dim
                qkv = zap[('qkv', l)][0].reshape(TOK, 3, h, hd).permute(1, 2, 0, 3)       # (3, głowy, N, hd)
                A = torch.softmax(qkv[0] @ qkv[1].transpose(-1, -2) * b.attn.scale, -1)  # (głowy, N, N)
                att = b.attn.proj((A @ qkv[2]).transpose(0, 1).reshape(TOK, h * hd))
                BL[n, l, 0] = float((x0 + att - x1).norm() / (x1 - x0).norm())          # kontrola: uwaga odtworzona z A
                BL[n, l, 1] = float((x1 + b.mlp(b.norm2(x1)) - x2).norm() / (x2 - x1).norm())
                A2 = A.clone(); idx = torch.arange(TOK, device=A.device); A2[:, idx, idx] = 0
                wart, ind = A2.topk(KTOP, -1)
                S[n, l] = torch.stack([x0, x1, x2]).float().cpu().numpy()
                Z[n, l] = z_
                V[n, l] = s[f'bloki.{l}.mlp.fc2.weight'].float().numpy()
                AI[n, l] = ind.cpu().numpy().astype(np.int16); AA[n, l] = wart.float().cpu().numpy()
        log('checkpoint', n, k, 'maks. błąd względny rozkładu (uwaga, MLP):', np.round(BL[n].max(0), 8))
    np.savez(CACHE, S=S, Z=Z.astype(np.float16), V=V, AI=AI, AA=AA.astype(np.float16), BL=BL, KR=KR)


if not CACHE.exists() or TRYB == 'dane':
    buduj()
_c = np.load(CACHE)
assert np.array_equal(_c['KR'], KR)
S, Z, V, AI, AA, BL = _c['S'], _c['Z'].astype(np.float32), _c['V'], _c['AI'].astype(np.int64), _c['AA'].astype(np.float32), _c['BL']
log('dane', S.shape, Z.shape, 'maks. błąd rozkładu uwaga/MLP:', BL[..., 0].max(), BL[..., 1].max())

# ---------------- rzut P, położenia, kierunki, skale ----------------
_M = []
for l in range(NB):
    for s_ in range(3):
        X_ = S[-1, l, s_] - S[-1, l, s_].mean(0)
        _M.append(X_ / np.sqrt((X_ ** 2).sum(1).mean()))
_M = np.concatenate(_M)
_U, _sv, _Vt = np.linalg.svd(_M, full_matrices=False)
P = _Vt[:3]                                                                         # (3, 192)
UDZ_P = float((_sv[:3] ** 2).sum() / (_sv ** 2).sum())
POS = np.einsum('nlsjd,ad->nlsja', S - S.mean(3, keepdims=True), P)                 # (NT, 6, 3, 600, 3)
PV = np.einsum('ad,nldc->nlca', P, V)                                               # (NT, 6, 768, 3) = P v_c
DIR = PV / np.maximum(np.linalg.norm(PV, axis=3, keepdims=True), 1e-9)
R99 = np.percentile(np.linalg.norm(POS, axis=4), 99, axis=3).max(2)                 # (NT, 6)
# kontrola dokładności: przesunięcie w podkroku MLP = rzut sumy wiązek (b₂ znosi się po odjęciu średniej)
_bl = max(float(np.abs((POS[n, l, 2] - POS[n, l, 1]) - (Z[n, l] - Z[n, l].mean(0)) @ PV[n, l]).max()
                / np.abs(POS[n, l, 2] - POS[n, l, 1]).max()) for n in (0, NT - 1) for l in range(NB))
log(f'P: 3 składowe niosą {UDZ_P:.1%} wariancji stanów; maks. błąd „przesunięcie MLP = suma wiązek” {_bl:.1e} (float16 latentu)')
log('R99 bloków (start, koniec):', np.round(R99[0], 2), np.round(R99[-1], 2))
if TRYB == 'dane':
    sys.exit(0)

R, RR = 34.0, 50.0
_RN = np.linalg.norm(POS, axis=4)                                                   # (NT, 6, 3, 600)
_R90, _RMX = np.percentile(_RN, 90, axis=3), _RN.max(axis=3)                        # (NT, 6, 3)
ZOOM = np.stack([np.minimum(0.5 * R / _R90[:, :, k:k + 2].max(2), 0.92 * R / _RMX[:, :, k:k + 2].max(2)) for k in (0, 1)], 2)
# skala kamery podkroku (NT, 6, 2): [..., 0] uwaga (stany 0→1), [..., 1] MLP (stany 1→2); w obrębie podkroku stała, więc ruch dokładny
P99 = np.array([[np.percentile(Z[n, l][Z[n, l] > 0], 99) for l in range(NB)] for n in range(NT)])      # normowanie jasności wiązek
FIB = kierunki(False).astype(np.float64)
RFIX = np.zeros((NB, NK, 3))
for l in range(NB):
    _r, _c2 = linear_sum_assignment(FIB @ DIR[-1, l].T)                             # miejsce najbliżej za −d_c (60. epoka)
    slot = np.empty(NK, int); slot[_c2] = _r
    RFIX[l] = RR * FIB[slot]
BLOK_KOL = np.array([[0.32, 0.56, 1.0], [0.28, 0.95, 0.55], [0.72, 0.52, 1.0], [1.0, 0.62, 0.25], [0.22, 0.90, 0.92], [1.0, 0.42, 0.48]])
H_KOL = np.array([[1.0, 0.45, 0.40], [1.0, 0.85, 0.35], [0.95, 0.50, 1.0]])         # 3 głowy uwagi
_rng = np.random.default_rng(5)
KC = [np.clip(BLOK_KOL[l][None] * (0.7 + 0.45 * _rng.random((NK, 1))), 0, 1) for l in range(NB)]
IOU = iou_all[np.searchsorted(kroki_all, KR)]

# ---------------- oś czasu ----------------
T_PRZ, T_PRZERWA, T_UCZ, T_KON, T_END = 5.0, 29.0, 32.0, 68.0, 73.0
DT_BLOK = (T_PRZERWA - T_PRZ) / NB                                                  # 4 s na blok w przejściu szczegółowym
DT_KROK = (T_KON - T_UCZ) / NT                                                      # 1,5 s na krok uczenia (12 podkroków)
sm = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))


def stan(t):
    """faza, checkpoint n, blok l, podkrok ('uwaga'/'mlp'/None), położenia (600, 3), skala, jasności wiązek."""
    S_ = dict(n=NT - 1, l=0, pod=None, a_att=0.0, a_mlp=0.0, ref=0.25, zoom=ZOOM[-1, 0, 0])
    if t < T_PRZ:
        S_.update(faza='wstep', p=POS[-1, 0, 0] * ZOOM[-1, 0, 0])
        S_['ref'] = 0.25 * sm(t / 2.0)
        return S_
    if t < T_PRZERWA:
        tau = t - T_PRZ; l = min(int(tau // DT_BLOK), NB - 1); w = tau - l * DT_BLOK; n = NT - 1
        S_.update(faza='przejscie', n=n, l=l)
        if w < 1.8:                                                                 # podkrok 1: uwaga — świeci, przesuwa, gaśnie
            z0 = ZOOM[n, l - 1, 1] if l > 0 else ZOOM[n, 0, 0]
            zoom = z0 + (ZOOM[n, l, 0] - z0) * sm(w / 0.35)                         # kamera dobiera skalę do podkroku
            u = sm((w - 0.45) / 1.25)                                               # postęp przesunięcia 0 → 1
            S_.update(pod='uwaga' if w >= 0.35 else None, a_att=sm((w - 0.35) / 0.2) * (1 - u),
                      p=(POS[n, l, 0] + (POS[n, l, 1] - POS[n, l, 0]) * u) * zoom, ref=0.12, zoom=zoom, u=u)
        else:                                                                       # podkrok 2: MLP — świeci, przesuwa, gaśnie
            zoom = ZOOM[n, l, 0] + (ZOOM[n, l, 1] - ZOOM[n, l, 0]) * sm((w - 1.8) / 0.3)
            u = sm((w - 2.2) / 1.6)
            fi = sm((w - 2.1) / 0.2)
            S_.update(pod='mlp', a_mlp=fi * (1 - u),
                      p=(POS[n, l, 1] + (POS[n, l, 2] - POS[n, l, 1]) * u) * zoom, ref=0.12 + 0.88 * fi * (1 - u), zoom=zoom, u=u)
        return S_
    if t < T_UCZ:
        S_.update(faza='przerwa', l=NB - 1, p=POS[-1, NB - 1, 2] * ZOOM[-1, NB - 1, 1], zoom=ZOOM[-1, NB - 1, 1], ref=0.12)
        return S_
    if t < T_KON:
        tau = t - T_UCZ; n = min(int(tau // DT_KROK), NT - 1); w = (tau - n * DT_KROK) / DT_KROK
        q = min(int(w * 2 * NB), 2 * NB - 1); l = q // 2; v = w * 2 * NB - q
        zoom = ZOOM[n, l, q % 2]
        u = sm((v - 0.05) / 0.85); fi = sm(v / 0.12); a_ = fi * (1 - u)             # świeci, przesuwa, gaśnie
        if q % 2 == 0:
            S_.update(pod='uwaga', a_att=a_, p=(POS[n, l, 0] + (POS[n, l, 1] - POS[n, l, 0]) * u) * zoom, ref=0.12, u=u)
        else:
            S_.update(pod='mlp', a_mlp=a_, p=(POS[n, l, 1] + (POS[n, l, 2] - POS[n, l, 1]) * u) * zoom, ref=0.12 + 0.88 * a_, u=u)
        S_.update(faza='uczenie', n=n, l=l, zoom=zoom, q=q)
        return S_
    S_.update(faza='koniec', l=NB - 1, p=POS[-1, NB - 1, 2] * ZOOM[-1, NB - 1, 1], zoom=ZOOM[-1, NB - 1, 1], ref=0.12)
    return S_


def macierz(t):
    az = np.radians(28 + 4.0 * t); el = np.radians(17 + 7 * np.sin(2 * np.pi * t / 45)); di = 152.0
    oko_ = di * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    f = -oko_ / np.linalg.norm(oko_); s = np.cross(f, [0, 0, 1.0]); s /= np.linalg.norm(s); up = np.cross(s, f)
    Vm = np.eye(4); Vm[0, :3], Vm[1, :3], Vm[2, :3] = s, up, -f; Vm[:3, 3] = -Vm[:3, :3] @ oko_
    fov, asp, n_, fa = np.radians(40), WY / HY, 0.1, 800.0
    Pm = np.zeros((4, 4)); Pm[0, 0] = 1 / np.tan(fov / 2) / asp; Pm[1, 1] = 1 / np.tan(fov / 2)
    Pm[2, 2] = (fa + n_) / (n_ - fa); Pm[2, 3] = 2 * fa * n_ / (n_ - fa); Pm[3, 2] = -1
    return Pm @ Vm, 1 / np.tan(fov / 2) * HY * SS / 2


# ---------------- OpenGL ----------------
ctx = moderngl.create_standalone_context(require=430)
W4, H4 = WY * SS, HY * SS
hdr = ctx.texture((W4, H4), 4, dtype='f2'); fb_hdr = ctx.framebuffer([hdr])
ldr = ctx.texture((W4, H4), 4); fb_ldr = ctx.framebuffer([ldr])
ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

# wiązki MLP c → j: tryb 1 = z reflektora c do punktu załamania E_jc na szklanej sferze, tryb 0 = z E_jc prosto do cząsteczki j
# w kierunku d_c (wspólnym dla 600 wiązek reflektora). Jasność z_jc ze znakiem: dodatnia w kolorze stacji, ujemna błękitna.
wiazki = ctx.program(vertex_shader='''#version 430
uniform mat4 mvp; uniform int tryb; uniform float gain, gneg, R, alfa;
layout(std430, binding=0) readonly buffer B0 { vec4 RF[]; };
layout(std430, binding=1) readonly buffer B1 { vec4 P[]; };
layout(std430, binding=2) readonly buffer B2 { vec4 DC[]; };
layout(std430, binding=3) readonly buffer B3 { float I[]; };
layout(std430, binding=4) readonly buffer B4 { vec4 KC[]; };
out vec3 v_k; out float v_t;
void main() {
    int id = gl_InstanceID; int c = id / 600; int j = id % 600;
    vec3 p = P[j].xyz; vec3 d = DC[c].xyz; vec3 r = RF[c].xyz;
    float bb = dot(p, d); float sd = bb + sqrt(max(bb * bb - dot(p, p) + R * R, 0.0));
    vec3 e = p - sd * d;
    float z = I[j * 768 + c] * alfa;
    v_k = z >= 0.0 ? KC[c].rgb * z * gain : vec3(0.30, 0.85, 1.0) * (-z) * gneg;
    vec3 q = tryb == 1 ? (gl_VertexID == 0 ? r : e) : (gl_VertexID == 0 ? e : p);
    v_t = float(gl_VertexID);
    gl_Position = mvp * vec4(q, 1.0);
}''', fragment_shader='''#version 430
uniform int tryb; in vec3 v_k; in float v_t; out vec4 o;
void main() { float r = tryb == 0 ? (0.25 + 0.75 * v_t * v_t) : (0.6 + 0.4 * v_t); o = vec4(v_k * r, 1.0); }''')

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

buf_O = ctx.buffer(reserve=NK * 16); buf_O.bind_to_storage_buffer(0)
buf_P = ctx.buffer(reserve=TOK * 16); buf_P.bind_to_storage_buffer(1)
buf_D = ctx.buffer(reserve=NK * 16); buf_D.bind_to_storage_buffer(2)
buf_I = ctx.buffer(reserve=TOK * NK * 4); buf_I.bind_to_storage_buffer(3)
buf_K = ctx.buffer(reserve=NK * 16); buf_K.bind_to_storage_buffer(4)
vao_w = ctx.vertex_array(wiazki, [])
wiazki['R'].value = R
GAIN, GNEG, GATT = 0.10, 0.024, 2.0          # najjaśniejsza dodatnia wiązka (z = p99 bloku) ≈ 0,10; ujemna z = −0,17 ≈ 0,004


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
    t_ = np.linspace(0, 2 * np.pi, 121); p_ = R * np.stack([np.cos(t_) * np.cos(lon_), np.cos(t_) * np.sin(lon_), np.sin(t_)], 1); seg += [p_[:-1], p_[1:]]
for lat_ in np.radians(np.arange(-72, 90, 12)):
    t_ = np.linspace(0, 2 * np.pi, 121); p_ = R * np.stack([np.cos(t_) * np.cos(lat_), np.sin(t_) * np.cos(lat_), np.full_like(t_, np.sin(lat_))], 1); seg += [p_[:-1], p_[1:]]
SFERA = np.stack([np.concatenate(seg[0::2]), np.concatenate(seg[1::2])], 1).reshape(-1, 3)

# ---------------- napisy ----------------
FN = lambda r, b=False: ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if b else 'C:/Windows/Fonts/segoeui.ttf', r)
F_T, F_S, F_M = FN(34, True), FN(21), FN(17)
pl = lambda x, k=1: f'{x:.{k}f}'.replace('.', ',')
OPIS_POD = {
    'uwaga': ['Uwaga: każda cząsteczka bierze średnią ważoną wektorów wartości innych cząsteczek (3 głowy = 3 barwy wiązek).',
              'Wiązki gasną, gdy przesunięcie się dopełnia (jasność = część, która została). Rysowane 2 najsilniejsze na cząsteczkę i głowę.'],
    'mlp': ['MLP: 768 reflektorów tej stacji; jasność wiązki = z_jc, kierunek w sferze = v_c, wspólny dla 600 wiązek reflektora.',
            'Siły liczone raz, ze stanu na początku podkroku; wiązki gasną, gdy cząsteczka dochodzi do celu. Błękitne = ujemna jasność.'],
}
TEKSTY = {
    'wstep': ('Sześć bloków w każdym kroku uczenia',
              ['W każdym kroku uczenia obraz przechodzi przez 6 bloków enkodera, czyli 6 stacji.',
               'W każdej stacji są 2 podkroki: najpierw uwaga (wiązki między cząsteczkami), potem MLP (768 reflektorów stacji).']),
    'przerwa': ('Uczenie: w każdym kroku te same 12 podkroków',
                ['24 kroki od startu do 60. epoki; w każdym obraz przechodzi 6 stacji (uwaga, potem MLP).',
                 'Między krokami AdamW przestawia wszystkie wagi: kierunki v_c, celowniki w_c i wagi uwagi.']),
    'koniec': ('Po 60 epokach',
               ['Każdy krok uczenia = jedno przejście przez 6 stacji: 6 × (uwaga + 768 laserów).',
                'Położenia to rzut 3D stanów tokenów; przesunięcie w podkroku MLP to dokładnie rzut sumy wiązek.']),
}
STOPKA = ['Dane: MalyViT_35000_od_zera_s47, zdjęcie TRAIN 0, 6 bloków (przejście modelu, tryb eval). Położenie = P (x_j − x̄), '
          f'P = 3 główne składowe stanów wszystkich bloków ({pl(100 * UDZ_P, 0)} % wariancji); skala kamery stała w obrębie podkroku.',
          'Ilustracja: szklana sfera, galwa i miejsca reflektorów (Fibonacci; w każdej stacji te same miejsca zajmuje jej własny komplet '
          '768 reflektorów). Dokładne: z_jc, v_c, A_jk, stany x_j.']


def pasek(d, S_, t):
    """6 stacji u dołu ekranu; aktywna stacja i podkrok wyróżnione."""
    x0, y0, dx, r = WY / 2 - 5 * 72 / 2 - 40, HY - 150, 72, 24
    d.text((x0 - 50, y0), 'stacje:', font=F_M, fill=(200, 205, 210, 255), anchor='rm')
    for l in range(NB):
        cx = x0 + l * dx
        kol = tuple(int(255 * v) for v in BLOK_KOL[l])
        akt = S_['faza'] in ('przejscie', 'uczenie') and S_['l'] == l
        d.ellipse((cx - r, y0 - r, cx + r, y0 + r), fill=kol + ((255,) if akt else (60,)), outline=kol + (255,), width=3 if akt else 1)
        d.text((cx, y0), str(l + 1), font=FN(20, True), fill=(10, 12, 16, 255) if akt else (230, 230, 230, 255), anchor='mm')
        if l < NB - 1:
            d.line((cx + r + 4, y0, cx + dx - r - 4, y0), fill=(150, 155, 160, 255), width=2)
        if akt and S_['pod']:
            d.text((cx, y0 + r + 14), 'uwaga' if S_['pod'] == 'uwaga' else 'MLP', font=FN(18, True),
                   fill=(255, 140, 120, 255) if S_['pod'] == 'uwaga' else kol + (255,), anchor='mt')
    d.text((x0 + 5 * dx + r + 26, y0), '→ głowa', font=F_M, fill=(200, 205, 210, 255), anchor='lm')


def nakladka(img, S_, t):
    d = ImageDraw.Draw(img, 'RGBA')
    f = S_['faza']
    if f in TEKSTY:
        tyt, pod = TEKSTY[f]
    elif f == 'przejscie':
        tyt = f'Jedno przejście obrazu w 60. epoce · blok {S_["l"] + 1} · ' + ('uwaga' if S_['pod'] == 'uwaga' else 'MLP' if S_['pod'] else 'zmiana stacji')
        pod = OPIS_POD[S_['pod']] if S_['pod'] else ['Cząsteczki przechodzą do następnej stacji: wyjście bloku jest wejściem następnego.',
                                                    'Kamera dobiera skalę do stacji, bo strumień rezydualny rośnie z głębokością (licznik po prawej).']
    else:
        tyt = f'Uczenie: krok {S_["n"] + 1} z {NT}, podkrok {S_["q"] + 1} z 12 · blok {S_["l"] + 1} · ' + ('uwaga' if S_['pod'] == 'uwaga' else 'MLP')
        pod = ['W każdym kroku uczenia obraz przechodzi wszystkie 6 stacji; między krokami AdamW przestawia wagi.',
               'W każdym podkroku wiązki świecą, przesuwają cząsteczki i gasną, gdy przesunięcie się dopełni.']
    d.text((WY / 2, 26), tyt, font=F_T, fill=(235, 240, 245, 255), anchor='mt')
    for k_, s_ in enumerate(pod):
        d.text((WY / 2, 78 + 30 * k_), s_, font=F_S, fill=(255, 190, 120, 255), anchor='mt')
    if f in ('przejscie', 'uczenie', 'przerwa', 'koniec'):
        kr = int(KR[S_['n']])
        d.text((WY - 30, 160), f'epoka {pl(kr / EP, 2)}   krok ' + f'{kr:,.0f}'.replace(',', ' '), font=FN(26, True), fill=(235, 240, 245, 255), anchor='ra')
        d.text((WY - 30, 196), f'powiększenie kamery ×{pl(S_["zoom"] / ZOOM[-1, NB - 1, 1], 1)} (wyjście bloku 6 w 60. epoce = ×1)', font=FN(18), fill=(190, 210, 230, 255), anchor='ra')
    if f == 'uczenie':
        x0, y0, sw, sh = WY - 400, HY - 300, 370, 120
        d.rectangle((x0, y0, x0 + sw, y0 + sh), fill=(11, 16, 20, 220), outline=(42, 51, 58, 255))
        xs = np.log1p(KR / 50.0) / np.log1p(KR[-1] / 50.0)
        d.line(list(zip((x0 + xs * sw).tolist(), (y0 + sh - np.clip(IOU, 0, 1) * sh).tolist())), fill=(74, 240, 255, 255), width=2)
        cx = x0 + xs[S_['n']] * sw; d.line((cx, y0, cx, y0 + sh), fill=(255, 255, 255, 230), width=2)
        d.text((x0, y0 - 24), 'IoU wykrycia narzędzia (oś kroków logarytmiczna)', font=F_M, fill=(74, 240, 255, 255))
    pasek(d, S_, t)
    for k_, s_ in enumerate(STOPKA):
        d.text((22, HY - 58 + 26 * k_), s_, font=F_M, fill=(205, 210, 215, 255))


def klatka(t):
    S_ = stan(t)
    n, l = S_['n'], S_['l']
    p = S_['p']
    mvp, skala = macierz(t)
    mvp_b = mvp.T.astype('f4').tobytes()
    fb_hdr.use(); fb_hdr.clear(0.008, 0.012, 0.02, 1.0)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    linie['mvp'].write(mvp_b); linie['szer'].value = 1.5
    rysuj(linie, moderngl.LINES, SFERA, np.tile([0.30, 0.55, 0.75, 0.18], (len(SFERA), 1)))
    # uwaga: wiązki k → j między cząsteczkami
    if S_['a_att'] > 0.01:
        j_ = np.repeat(np.arange(TOK), KTOP)
        pk, pj, kol = [], [], []
        for h in range(3):
            k_ = AI[n, l, h].reshape(-1); a_ = AA[n, l, h].reshape(-1)
            pk.append(p[k_]); pj.append(p[j_])
            kol.append(np.concatenate([H_KOL[h][None] * np.clip(a_[:, None] * GATT * S_['a_att'], 0, 1.5),
                                       np.ones((len(a_), 1))], 1))
        pk, pj, kol = np.concatenate(pk), np.concatenate(pj), np.concatenate(kol)
        linie['szer'].value = 2.2
        rysuj(linie, moderngl.LINES, np.stack([pk, pj], 1).reshape(-1, 3), np.repeat(kol, 2, 0))
    # MLP: 768 reflektorów stacji l
    buf_O.write(np.concatenate([RFIX[l], np.zeros((NK, 1))], 1).astype('f4').tobytes())
    buf_K.write(np.concatenate([KC[l], np.ones((NK, 1))], 1).astype('f4').tobytes())
    if S_['a_mlp'] > 0.01:
        buf_P.write(np.concatenate([p, np.zeros((TOK, 1))], 1).astype('f4').tobytes())
        buf_D.write(np.concatenate([DIR[n, l], np.zeros((NK, 1))], 1).astype('f4').tobytes())
        buf_I.write(Z[n, l].astype('f4').tobytes())
        wiazki['mvp'].write(mvp_b); wiazki['alfa'].value = S_['a_mlp']
        for tryb, g in ((1, 0.35), (0, 1.0)):
            wiazki['tryb'].value = tryb; wiazki['gain'].value = GAIN * g / P99[n, l]; wiazki['gneg'].value = GNEG * g
            vao_w.render(moderngl.LINES, vertices=2, instances=TOK * NK)
    kropki['mvp'].write(mvp_b); kropki['skala'].value = skala
    moc = np.abs(Z[n, l]).mean(0); moc = np.clip(moc / max(1e-6, np.percentile(moc, 95)), 0.15, 1.5)
    kropki['ksztalt'].value = 2
    rysuj(kropki, moderngl.POINTS, RFIX[l], np.concatenate([KC[l] * (0.35 + 0.9 * S_['ref'] * moc)[:, None], np.full((NK, 1), 0.85)], 1),
          np.full(NK, 1.1))
    fb_ldr.use(); ctx.disable(moderngl.BLEND)
    hdr.use(0); quad.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    kropki['ksztalt'].value = 0                                                     # cząsteczki na wierzchu, po mapowaniu tonów
    rysuj(kropki, moderngl.POINTS, p, np.concatenate([np.tile([0.92, 1.0, 0.94], (TOK, 1)), np.full((TOK, 1), 0.95)], 1), np.full(TOK, 0.75))
    kropki['ksztalt'].value = 1
    rysuj(kropki, moderngl.POINTS, p[narz], np.tile([0.29, 0.94, 1.0, 0.85], (int(narz.sum()), 1)), np.full(int(narz.sum()), 1.6))
    ldr.build_mipmaps()
    raw = np.frombuffer(ldr.read(level=1), np.uint8).reshape(HY, WY, 4)[::-1]
    img = Image.fromarray(raw[..., :3].copy())
    nakladka(img, S_, t)
    return img


CZASY = np.arange(int(T_END * FPS)) / FPS
if PROBA:
    for t in [5.6, 6.2, 6.7, 7.3, 8.0, 8.8, 18.1, 46.0, 66.2]:
        klatka(t).save(HERE / f'lasery_21_proba_{t:04.1f}.png')
    log('próby zapisane')
else:
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WY}x{HY}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '20', '-pix_fmt', 'yuv420p', str(HERE / 'lasery_21_szesc_blokow_wygaszanie.mp4')],
                          stdin=subprocess.PIPE)
    for k_, t in enumerate(CZASY):
        ff.stdin.write(np.asarray(klatka(float(t))).tobytes())
        if k_ % 150 == 0:
            log(f'klatka {k_}/{len(CZASY)}')
    ff.stdin.close(); ff.wait()
    log('gotowe: lasery_21_szesc_blokow_wygaszanie.mp4')
