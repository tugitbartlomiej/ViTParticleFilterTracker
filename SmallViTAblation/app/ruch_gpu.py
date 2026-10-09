"""Ruch całego zbioru liczony i rysowany na GPU (OpenGL 4.3: compute shadery GLSL 4.30 + shadery rysowania).

Zapisane stany osi czasu (float32 [cząsteczka, (x, y, z, norma 768D)]) trafiają do jednego bufora SSBO karty
(„magazyn”, S slotów po jednym stanie). Gdy cała oś mieści się w budżecie VRAM, na kartę trafiają wszystkie stany;
w przeciwnym razie magazyn działa jak pierścień: zostają stan początkowy, bieżąca para i kilka następnych, a stany
wczytuje w tle pula wątków (odczyt pliku i weryfikacja SHA256 w `ruch_zbioru.Timeline.frame`).

Kernele compute (wszystko, co zależy od cząsteczek, liczy karta):
- `stat` (przy wysłaniu stanu k): dla każdego obrazu granice osi PCA (położenie i przesunięcie od pierwszego kroku),
  maksimum normy 768D, |Δ| w PCA i prędkości względem poprzedniego zapisu -> bufor PIB[k, obraz];
- `heat` (przy wysłaniu stanu k): mapa 20 × 30, średnia normy 768D każdego patcha po wszystkich obrazach;
- `kin` (w klatce, gdy zmienią się parametry): położenie każdej cząsteczki widoku, przesunięcie od pierwszego kroku
  i wielkość koloru. W zapisanym stanie (s = 0 albo s = 1) wynik jest bitowo równy zapisowi; pomiędzy zapisami
  to jawny odcinek liniowy (stan niezapisany, oznaczany w UI);
- `bb` + `ramka`: granice osi i skale kolorów z PIB (wszystkie zapisane kroki) albo z bieżącego stanu -> środek i
  skala sceny czytane wprost przez shader wierzchołków (bez odczytu na CPU);
- `hist` + `wybor`: dokładna mediana i maksimum normy 768D w widoku (histogram bitów, bez sortowania);
- `pick`: najbliższa kliknięciu cząsteczka w pikselach ekranu (dwa przebiegi atomowe).
Rysowanie: punkty i ślady z buforów (`gl_VertexID`/`gl_InstanceID` + SSBO, bez atrybutów wierzchołków i VAO).

CPU przekazuje tylko parametry klatki (para stanów, postęp, tryby) i wysyła nowe stany. Odczyt z karty jest tylko dla
statystyk wyświetlanych w panelu (600 liczb mapy + kilka liczb mediany, raz na zmianę stanu) i dla kliknięcia.
"""
import ctypes
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt6.QtCore import pyqtSignal
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

NONE = 0xFFFFFFFF
BINS = 65536
R_HIST = 3 * BINS                       # początek wyników w buforze HIST
MAX_BLOK = 2147483647                   # typowe GL_MAX_SHADER_STORAGE_BLOCK_SIZE (NVIDIA); sprawdzane przy starcie
KOLORY = ('norma', 'predkosc', 'delta')
KADRY = ('zbior', 'widoczne', 'biezacy')

_NAGLOWEK = """#version 430
//Q16DEF
layout(std430, binding = 0) buffer BS { uint SW[]; };         // magazyn stanów: slot * N + cząsteczka (słowa)
layout(std430, binding = 10) buffer BQ { vec4 QP[]; };        // [2 * sloty] q16: (lo.xyz, norm_max), (span.xyz, 0)
layout(std430, binding = 1) buffer BCUR { vec4 CUR[]; };      // [N] bieżąca klatka: położenie, wielkość koloru
layout(std430, binding = 2) buffer BFR { vec4 FR[]; };        // [8] środek+skala, granice, skale kolorów
layout(std430, binding = 3) buffer BPIB { vec4 PIB[]; };      // [T * obrazy * 4] statystyki obrazu w kroku
layout(std430, binding = 4) buffer BKN { uint KNOWN[]; };     // [T] 1 = statystyki kroku policzone
layout(std430, binding = 5) buffer BHEAT { float HEAT[]; };   // [T * tokeny] średnia norma patcha po obrazach
layout(std430, binding = 6) buffer BHIST { uint HIST[]; };    // 3 x 65536 histogramów + wyniki
layout(std430, binding = 7) buffer BPICK { uint PICK[]; };    // [4] wybór punktu
layout(std430, binding = 8) buffer BSLOT { uint SLOT[]; };    // [T] slot magazynu kroku albo NONE
layout(std430, binding = 9) buffer BBB { uint BB[]; };        // [8] granice bieżącego stanu (kody ord)
const uint NONE = 0xffffffffu;
const uint BINS = 65536u;
const uint R = 3u * 65536u;
uniform uint u_N;
uint ord(float f) { uint u = floatBitsToUint(f); return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u); }
float unord(uint u) { return (u & 0x80000000u) != 0u ? uintBitsToFloat(u & 0x7fffffffu) : uintBitsToFloat(~u); }
// Stan cząsteczki i w slocie sl: f32 = 4 słowa bitowo z pliku; q16 = 2 słowa po dwie wartości uint16,
// x = lo + (q / 65535) * span, norma = (q / 65535) * norm_max (ten sam wzór co ruch_zbioru.odkwantyzuj).
vec4 stan(uint sl, uint i) {
#if Q16
    uint b = (sl * u_N + i) * 2u;
    vec2 xy = unpackUnorm2x16(SW[b]);
    vec2 zn = unpackUnorm2x16(SW[b + 1u]);
    vec4 lo = QP[2u * sl], sp = QP[2u * sl + 1u];
    return vec4(lo.xyz + vec3(xy, zn.x) * sp.xyz, zn.y * lo.w);
#else
    uint b = (sl * u_N + i) * 4u;
    return uintBitsToFloat(uvec4(SW[b], SW[b + 1u], SW[b + 2u], SW[b + 3u]));
#endif
}
"""

CS_KIN = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_first, u_count, u_sa, u_sb, u_s0, u_ruch;
uniform int u_kolor;
uniform float u_s, u_dt;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= u_count) return;
    uint i = u_first + g;
    vec4 a = stan(u_sa, i);
    vec4 b = stan(u_sb, i);
    vec3 p0 = stan(u_s0, i).xyz;
    vec3 x; float n;
    if (u_s <= 0.0) { x = a.xyz; n = a.w; }                  // zapisany stan a: bitowo z pliku
    else if (u_s >= 1.0) { x = b.xyz; n = b.w; }             // zapisany stan b
    else { x = a.xyz + (b.xyz - a.xyz) * u_s; n = a.w + (b.w - a.w) * u_s; }   // stan niezapisany
    vec3 d = x - p0;
    float c = n;
    if (u_kolor == 1) c = u_dt > 0.0 ? length(b.xyz - a.xyz) / u_dt : 0.0;
    else if (u_kolor == 2) c = length(d);
    CUR[i] = vec4(u_ruch == 1u ? d : x, c);
}
"""

CS_STAT = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_k, u_slot, u_s0, u_prev, u_tpi, u_img;
uniform float u_dt;
shared vec3 s_lo[256], s_hi[256], s_mlo[256], s_mhi[256], s_mx[256];
void main() {
    uint im = gl_WorkGroupID.x, l = gl_LocalInvocationID.x;
    vec3 lo = vec3(3.0e38), hi = vec3(-3.0e38), mlo = vec3(3.0e38), mhi = vec3(-3.0e38), mx = vec3(0.0);
    for (uint t = l; t < u_tpi; t += 256u) {
        uint i = im * u_tpi + t;
        vec4 p = stan(u_slot, i);
        vec3 d = p.xyz - stan(u_s0, i).xyz;
        lo = min(lo, p.xyz); hi = max(hi, p.xyz); mlo = min(mlo, d); mhi = max(mhi, d);
        float v = 0.0;
        if (u_prev != NONE && u_dt > 0.0) v = length(p.xyz - stan(u_prev, i).xyz) / u_dt;
        mx = max(mx, vec3(p.w, length(d), v));
    }
    s_lo[l] = lo; s_hi[l] = hi; s_mlo[l] = mlo; s_mhi[l] = mhi; s_mx[l] = mx;
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) {
            s_lo[l] = min(s_lo[l], s_lo[l + w]); s_hi[l] = max(s_hi[l], s_hi[l + w]);
            s_mlo[l] = min(s_mlo[l], s_mlo[l + w]); s_mhi[l] = max(s_mhi[l], s_mhi[l + w]);
            s_mx[l] = max(s_mx[l], s_mx[l + w]);
        }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) {
        uint b = (u_k * u_img + im) * 4u;
        PIB[b] = vec4(s_lo[0], s_mx[0].x);          // granice położeń, maks. normy 768D
        PIB[b + 1u] = vec4(s_hi[0], s_mx[0].y);     // ..., maks. |Δ| w PCA od pierwszego kroku
        PIB[b + 2u] = vec4(s_mlo[0], s_mx[0].z);    // granice przesunięć, maks. prędkości względem poprzedniego zapisu
        PIB[b + 3u] = vec4(s_mhi[0], 0.0);
        if (im == 0u) KNOWN[u_k] = 1u;
    }
}
"""

CS_HEAT = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_k, u_slot, u_tpi, u_img;
shared float s_sum[256];
void main() {
    uint t = gl_WorkGroupID.x, l = gl_LocalInvocationID.x;
    float acc = 0.0;
    for (uint im = l; im < u_img; im += 256u) acc += stan(u_slot, im * u_tpi + t).w;
    s_sum[l] = acc;
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) s_sum[l] += s_sum[l + w];
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) HEAT[u_k * u_tpi + t] = s_sum[0] / float(u_img);
}
"""

CS_BB = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_first, u_count;
shared vec3 s_lo[256], s_hi[256];
void main() {
    uint g = gl_GlobalInvocationID.x, l = gl_LocalInvocationID.x;
    vec3 p = g < u_count ? CUR[u_first + g].xyz : vec3(0.0);
    s_lo[l] = g < u_count ? p : vec3(3.0e38);
    s_hi[l] = g < u_count ? p : vec3(-3.0e38);
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) { s_lo[l] = min(s_lo[l], s_lo[l + w]); s_hi[l] = max(s_hi[l], s_hi[l + w]); }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) {
        atomicMin(BB[0], ord(s_lo[0].x)); atomicMin(BB[1], ord(s_lo[0].y)); atomicMin(BB[2], ord(s_lo[0].z));
        atomicMax(BB[3], ord(s_hi[0].x)); atomicMax(BB[4], ord(s_hi[0].y)); atomicMax(BB[5], ord(s_hi[0].z));
    }
}
"""

CS_RAMKA = _NAGLOWEK + """
layout(local_size_x = 1024) in;
uniform uint u_T, u_img, u_img0, u_img1, u_ruch, u_tryb, u_manifest;
uniform vec3 u_mlo, u_mhi;
uniform float u_normscale;
shared vec3 s_lo[1024], s_hi[1024], s_mx[1024];
void main() {
    uint l = gl_LocalInvocationID.x;
    vec3 lo = vec3(3.0e38), hi = vec3(-3.0e38), mx = vec3(0.0);
    uint total = u_T * u_img;
    for (uint e = l; e < total; e += 1024u) {
        uint k = e / u_img, im = e - k * u_img;
        if (KNOWN[k] == 0u) continue;
        uint b = e * 4u;
        vec4 q0 = PIB[b], q1 = PIB[b + 1u], q2 = PIB[b + 2u], q3 = PIB[b + 3u];
        mx = max(mx, vec3(q0.w, q2.w, q1.w));             // skale kolorów zawsze z całego zbioru
        if (u_tryb == 0u || (im >= u_img0 && im < u_img1)) {
            if (u_ruch == 1u) { lo = min(lo, q2.xyz); hi = max(hi, q3.xyz); }
            else { lo = min(lo, q0.xyz); hi = max(hi, q1.xyz); }
        }
    }
    s_lo[l] = lo; s_hi[l] = hi; s_mx[l] = mx;
    memoryBarrierShared(); barrier();
    for (uint w = 512u; w > 0u; w >>= 1u) {
        if (l < w) { s_lo[l] = min(s_lo[l], s_lo[l + w]); s_hi[l] = max(s_hi[l], s_hi[l + w]); s_mx[l] = max(s_mx[l], s_mx[l + w]); }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) {
        lo = s_lo[0]; hi = s_hi[0];
        if (u_tryb == 2u) {
            lo = vec3(unord(BB[0]), unord(BB[1]), unord(BB[2]));
            hi = vec3(unord(BB[3]), unord(BB[4]), unord(BB[5]));
        }
        if (u_manifest == 1u && u_tryb == 0u) { lo = min(lo, u_mlo); hi = max(hi, u_mhi); }
        if (any(greaterThan(lo, hi))) { lo = vec3(-1.0); hi = vec3(1.0); }   // jeszcze brak danych
        vec3 e = hi - lo;
        FR[0] = vec4((lo + hi) * 0.5, 2.0 / max(max(max(e.x, e.y), e.z), 0.001));
        FR[1] = vec4(lo, 0.0);
        FR[2] = vec4(hi, 0.0);
        FR[3] = vec4(max(s_mx[0].x, max(u_normscale, 1e-6)), max(s_mx[0].y, 1e-6), max(s_mx[0].z, 1e-6), 0.0);
    }
}
"""

CS_HIST = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_slot, u_first, u_count, u_faza;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= u_count) return;
    uint bits = floatBitsToUint(stan(u_slot, u_first + g).w);     // norma >= 0: porządek bitów = porządek liczb
    if (u_faza == 0u) { atomicAdd(HIST[bits >> 16u], 1u); atomicMax(HIST[R + 2u], bits); }
    else {
        uint p = bits >> 16u, q = bits & 0xffffu;
        if (p == HIST[R]) atomicAdd(HIST[BINS + q], 1u);
        if (p == HIST[R + 1u]) atomicAdd(HIST[2u * BINS + q], 1u);
    }
}
"""

CS_WYBOR = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_faza, u_klo, u_khi;
shared uint s_a[256], s_b[256];
// k-ty element (od 0) histogramu o początku `off`: (kosz, pozycja w koszu)
uvec2 wybierz(uint off, uint k, bool pierwszy) {
    uint acc = 0u, c = 0u;
    for (c = 0u; c < 256u; c++) {
        uint v = pierwszy ? s_a[c] : s_b[c];
        if (acc + v > k) break;
        acc += v;
    }
    c = min(c, 255u);
    uint bin = c * 256u;
    for (uint j = 0u; j < 256u; j++) {
        uint v = HIST[off + c * 256u + j];
        if (acc + v > k) { bin = c * 256u + j; break; }
        acc += v;
    }
    return uvec2(bin, k - acc);
}
void main() {
    uint l = gl_LocalInvocationID.x;
    uint offa = u_faza == 0u ? 0u : BINS, offb = u_faza == 0u ? 0u : 2u * BINS;
    uint a = 0u, b = 0u;
    for (uint j = 0u; j < 256u; j++) { a += HIST[offa + l * 256u + j]; b += HIST[offb + l * 256u + j]; }
    s_a[l] = a; s_b[l] = b;
    memoryBarrierShared(); barrier();
    if (l == 0u) {
        if (u_faza == 0u) {
            uvec2 x = wybierz(0u, u_klo, true), y = wybierz(0u, u_khi, true);
            HIST[R] = x.x; HIST[R + 3u] = x.y; HIST[R + 1u] = y.x; HIST[R + 4u] = y.y;
        } else {
            uvec2 x = wybierz(BINS, HIST[R + 3u], true), y = wybierz(2u * BINS, HIST[R + 4u], false);
            HIST[R + 5u] = (HIST[R] << 16u) | x.x;
            HIST[R + 6u] = (HIST[R + 1u] << 16u) | y.x;
        }
    }
}
"""

CS_PICK = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform mat4 u_mvp;
uniform vec2 u_klik, u_wh;
uniform uint u_first, u_count, u_faza;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= u_count) return;
    uint i = u_first + g;
    vec4 f = FR[0];
    vec4 c = u_mvp * vec4((CUR[i].xyz - f.xyz) * f.w, 1.0);
    if (c.w <= 1e-9) return;                                       // za kamerą
    vec2 n = c.xy / c.w;
    vec2 d = vec2((n.x + 1.0) * 0.5 * u_wh.x, (1.0 - n.y) * 0.5 * u_wh.y) - u_klik;
    uint d2 = floatBitsToUint(dot(d, d));
    if (u_faza == 0u) atomicMin(PICK[0], d2);
    else if (d2 == PICK[0]) atomicMin(PICK[1], i);                 // przy remisie najmniejszy indeks
}
"""

VS_PUNKTY = _NAGLOWEK + """
uniform mat4 u_mvp;
uniform uint u_first;
uniform int u_kolor, u_wybrany;
uniform float u_size;
out float v_t;
flat out int v_ring;
void main() {
    uint i = u_wybrany >= 0 ? uint(u_wybrany) : u_first + uint(gl_VertexID);
    vec4 c = CUR[i];
    vec4 f = FR[0];
    gl_Position = u_mvp * vec4((c.xyz - f.xyz) * f.w, 1.0);
    gl_PointSize = u_wybrany >= 0 ? max(u_size * 5.0, 12.0) : u_size;
    v_t = clamp(c.w / max(FR[3][u_kolor], 1e-6), 0.0, 1.0);
    v_ring = u_wybrany >= 0 ? 1 : 0;
}
"""

FS_PUNKTY = """#version 430
in float v_t;
flat in int v_ring;
uniform float u_opacity;
out vec4 kolor;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = dot(p, p);
    if (r > 1.0) discard;
    if (v_ring == 1) { if (r < 0.5) discard; kolor = vec4(1.0, 1.0, 1.0, 0.95); return; }
    kolor = vec4(mix(vec3(0.23, 0.72, 0.68), vec3(1.0, 0.91, 0.68), v_t), u_opacity);
}
"""

VS_SLAD = _NAGLOWEK + """
uniform mat4 u_mvp;
uniform uint u_first, u_ia, u_K, u_ruch, u_s0;
uniform int u_wybrany;
uniform float u_alfa;
out vec4 v_kol;
void main() {
    uint i = u_wybrany >= 0 ? uint(u_wybrany) : u_first + uint(gl_InstanceID);
    int v = gl_VertexID;
    uint e = uint(v / 2 + v % 2);                                   // punkt śladu 0..K, K = bieżąca klatka
    vec3 p = CUR[i].xyz;
    if (e < u_K) {
        int idx = max(int(u_ia) - int(u_K) + 1 + int(e), 0);
        uint sl = SLOT[idx];
        if (sl != NONE) {
            p = stan(sl, i).xyz;
            if (u_ruch == 1u) p -= stan(u_s0, i).xyz;
        }
    }
    vec4 f = FR[0];
    gl_Position = u_mvp * vec4((p - f.xyz) * f.w, 1.0);
    v_kol = vec4(0.82, 0.88, 1.0, u_alfa * (0.1 + 0.9 * float(e) / float(max(u_K, 1u))));
}
"""

FS_KOLOR = """#version 430
in vec4 v_kol;
out vec4 kolor;
void main() { kolor = v_kol; }
"""

BUFORY = ('S', 'CUR', 'FR', 'PIB', 'KNOWN', 'HEAT', 'HIST', 'PICK', 'SLOT', 'BB', 'QP')


def _ord(x):
    """Kod porządkowy float32 -> uint32 (jak ord() w shaderach)."""
    u = np.asarray(x, np.float32).view(np.uint32)
    return np.where(u & 0x80000000, ~u, u | 0x80000000).astype(np.uint32)


class ChmuraRuchuGPU(GLGraphicsItem):
    """Element sceny pyqtgraph: magazyn stanów na karcie, kernele compute i rysowanie chmury z SSBO.

    Część CPU (bez kontekstu GL): ustaw_os, ustaw_klatke, pompuj, gotowy, zadaj_statystyki, zadaj_wybor.
    Część GL (w paint): wysyłanie stanów, kernele, odczyt wyników, rysowanie, przywrócenie stanu GL.
    """
    statystyki_gotowe = pyqtSignal(object)       # dict: klucz, mapa, mediana, maksimum (z karty)
    wybrano = pyqtSignal(int)                    # indeks cząsteczki albo -1

    def __init__(self, watki=3):
        super().__init__()
        self.setGLOptions({GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True,
                           'glBlendFunc': (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)})
        self._executor = ThreadPoolExecutor(max_workers=watki, thread_name_prefix='ruch-gpu-stany')
        self._prog = None
        self._buf = {}
        self._rozm = {}
        self._uni = {}
        self.renderer = ''
        self.blad = ''                           # błąd GL (kompilacja, alokacja, rysowanie)
        self.blad_odczytu = ''                   # błąd pliku stanu (rozmiar, SHA256, wartości)
        self.max_blok = MAX_BLOK
        self.vram_wolna = None                   # bajty wolnej pamięci karty (NVX), gdy sterownik ją podaje
        self.budzet = 2 << 30
        self.na_klatke = 2                       # ile dodatkowych stanów wysłać w jednym rysowaniu
        self._zeruj_os()
        # parametry klatki (ustawiane przez ustaw_klatke)
        self.a = self.b = None
        self.s = 0.0
        self.ruch = True
        self.kolor = 0
        self.kadr = 0
        self.first, self.count = 0, 0
        self.rozmiar, self.krycie = 1.5, .2
        self.slad = 0
        self.wybrany = None
        self.normscale = 1.0
        self.granice_manifest = None
        self._klucz_kin = None
        self._ramka_brudna = True
        self._cur_ok = False
        self._rys_count = self._rys_first = self.draw_count = 0
        self.ostatni_wybor = None
        self._stat_klucz = None
        self._stat_zrobione = None
        self._wybor = None
        self.fps = 0.0
        self._klatki, self._zegar = 0, time.perf_counter()
        self.czas_gpu_ms = 0.0
        self.wyslane_bajty = 0

    # ------------------------------------------------------------------ CPU: oś czasu i strumień stanów
    def _zeruj_os(self):
        self.timeline = None
        self.steps = []
        self._idx = {}
        self.N = self.T = self.images = self.tpi = 0
        self.frame_bytes = 0
        self.q16 = getattr(self, 'q16', False)
        self.n_slotow = 0
        self._sloty = OrderedDict()              # krok -> slot (kolejność = ostatnie użycie)
        self._wolne = []
        self._gotowe = OrderedDict()             # krok -> np.ndarray czekające na wysłanie
        self._w_toku = {}
        self._generacja = getattr(self, '_generacja', 0) + 1
        self._uklad = None                       # układ buforów na karcie (zmiana = ponowna alokacja)
        self._slot_map_brudna = True
        self._znane = set()

    def ustaw_os(self, timeline, gotowe=None, budzet=None):
        """Nowa oś czasu. `gotowe`: już wczytane stany {krok: [N, 4]} (pierwsza para z wątku wczytywania)."""
        self._zeruj_os()
        self.timeline = timeline
        self.steps = list(timeline.steps)
        self._idx = {s: i for i, s in enumerate(self.steps)}
        self.T, self.N = len(self.steps), int(timeline.count)
        self.images = int(timeline.image_count) or 1
        if self.N % self.images:
            self.images = 1
        self.tpi = self.N // self.images
        self.q16 = getattr(timeline, 'frame_format', 'f32') == 'q16'
        self.frame_bytes = self.N * (8 if self.q16 else 16)     # q16: połowa pamięci karty na stan
        self.normscale = float(timeline.metadata.get('norm_scale', 1.0) or 1.0)
        if budzet is not None:
            self.budzet = int(budzet)
        self._planuj_sloty()
        for krok, arr in (gotowe or {}).items():
            if krok in self._idx:
                self._gotowe[krok] = self._sprawdz(arr)
        self.a = self.b = self.steps[0] if self.steps else None
        self._klucz_kin, self._cur_ok, self._ramka_brudna = None, False, True
        self._stat_klucz = self._stat_zrobione = None
        self.wybrany = None
        self.blad = self.blad_odczytu = ''
        self.update()

    def ustaw_granice_manifestu(self, natural, movement):
        """Granice osi zapisane przez producenta (cały zbiór, wszystkie kroki) - łączone z granicami z karty."""
        self.granice_manifest = (None if natural is None else np.asarray(natural, np.float32),
                                 None if movement is None else np.asarray(movement, np.float32))
        self._ramka_brudna = True
        self.update()

    def ustaw_budzet(self, bajty):
        bajty = int(bajty)
        if bajty != self.budzet:
            self.budzet = bajty
            if self.timeline is not None:
                self._planuj_sloty()
                self._sloty.clear(); self._wolne = []
                self._uklad = None
                self._klucz_kin, self._cur_ok = None, False
                self.update()

    def _planuj_sloty(self):
        if not self.T:
            self.n_slotow = 0
            return
        limit = min(self.budzet, self.max_blok)
        if self.vram_wolna is not None:
            limit = min(limit, int(self.vram_wolna * .8) - (256 << 20))
        self.n_slotow = int(max(3, min(self.T, limit // max(self.frame_bytes, 1))))

    @property
    def cala_os(self):
        """Czy wszystkie zapisane stany mieszczą się na karcie naraz."""
        return self.n_slotow >= self.T

    def _sprawdz(self, arr):
        arr = np.ascontiguousarray(arr, dtype='<u2' if self.q16 else np.float32)
        if arr.shape != (self.N, 4):
            raise ValueError(f'Niezgodny układ stanu: {arr.shape}, oczekiwano ({self.N}, 4).')
        return arr

    @staticmethod
    def _czytaj(timeline, krok, generacja, n):
        """Wątek tła: odczyt stanu z weryfikacją rozmiaru, SHA256 i wartości. Timeline.read czyta plik raz
        (skrót i sprawdzenie z tych samych bajtów); atrapy osi czasu w testach mają tylko frame()."""
        if getattr(timeline, 'frame_format', 'f32') == 'q16':     # słowa uint16 prosto z pliku na kartę
            arr = timeline.read_raw(krok)
            if arr.shape != (n, 4):
                raise ValueError(f'Niezgodny układ stanu {krok}: {arr.shape}, oczekiwano ({n}, 4).')
            return generacja, np.ascontiguousarray(arr, dtype='<u2')
        czytaj = getattr(timeline, 'read', None)
        arr = czytaj(krok) if czytaj is not None else timeline.frame(krok)
        out = np.ascontiguousarray(arr, dtype=np.float32)
        if out is arr and not out.flags.writeable:
            out = out.copy()
        mm = getattr(arr, '_mmap', None)
        if mm is not None:
            out = np.array(out, copy=True)
            try:
                mm.close()
            except (BufferError, ValueError):
                pass
        if out.shape != (n, 4):
            raise ValueError(f'Niezgodny układ stanu {krok}: {out.shape}, oczekiwano ({n}, 4).')
        return generacja, out

    def _plan(self):
        """Kolejność stanów potrzebnych na karcie: pierwszy krok, bieżąca para, następne (cała oś, gdy się mieści)."""
        if not self.T:
            return []
        ia = self._idx.get(self.a, 0)
        ib = self._idx.get(self.b, ia)
        kolej = [0, ia, ib]
        zasieg = self.T if self.cala_os else max(0, min(self.n_slotow - 3, 8))
        kolej += [(ib + 1 + k) % self.T for k in range(zasieg)]
        if self.cala_os:
            kolej += list(range(self.T))
        wynik, byly = [], set()
        for i in kolej:
            if i not in byly:
                byly.add(i); wynik.append(self.steps[i])
        return wynik if self.cala_os else wynik[:self.n_slotow]

    def gotowy(self, krok):
        return krok in self._sloty or krok in self._gotowe

    def pompuj(self, limit=6):
        """Zbiera stany wczytane w tle i zleca następne. Woła GUI (np. co klatkę). Zwraca True, gdy coś trwa."""
        if self.timeline is None:
            return False
        for krok, fut in list(self._w_toku.items()):
            if fut.done():
                del self._w_toku[krok]
                try:
                    gen, arr = fut.result()
                except Exception as e:                       # zły plik: pokaż błąd, nie wysyłaj na kartę
                    self.blad_odczytu = f'stan {krok}: {type(e).__name__}: {e}'
                    continue
                if gen == self._generacja and krok not in self._sloty:
                    self._gotowe[krok] = arr
        for krok in self._plan():
            if len(self._w_toku) + len(self._gotowe) >= limit:
                break
            if not self.gotowy(krok) and krok not in self._w_toku:
                self._w_toku[krok] = self._executor.submit(self._czytaj, self.timeline, krok, self._generacja, self.N)
        if self._gotowe:
            self.update()
        return bool(self._w_toku or self._gotowe)

    def wczytaj_teraz(self, kroki):
        """Synchronicznie (testy, skrypty): wczytaj podane stany do kolejki wysyłki."""
        for krok in kroki:
            if not self.gotowy(krok):
                self._gotowe[krok] = self._czytaj(self.timeline, krok, self._generacja, self.N)[1]
        self.update()

    def zajety(self):
        return bool(self._w_toku or self._gotowe or self._wybor is not None
                    or (self._stat_klucz is not None and self._stat_klucz != self._stat_zrobione))

    def na_karcie(self):
        return len(self._sloty)

    def pamiec_karty(self):
        """Bajty zajęte na karcie przez bufory tej chmury."""
        return int(sum(self._rozm.values()))

    # ------------------------------------------------------------------ CPU: parametry klatki
    def ustaw_klatke(self, a, b, s, ruch, kolor, kadr, first, count, rozmiar, krycie, slad=0):
        """Tylko liczby: para zapisanych stanów, postęp s (0 = stan a, 1 = stan b), tryby. Bez danych cząsteczek."""
        nowe = (a, b, float(s), bool(ruch), int(kolor), int(kadr), int(first), int(count))
        stare = (self.a, self.b, self.s, self.ruch, self.kolor, self.kadr, self.first, self.count)
        if (bool(ruch), int(kadr), int(first), int(count)) != (self.ruch, self.kadr, self.first, self.count):
            self._ramka_brudna = True
        widok = (float(rozmiar), float(krycie), int(slad)) != (self.rozmiar, self.krycie, self.slad)
        self.a, self.b, self.s, self.ruch, self.kolor, self.kadr, self.first, self.count = nowe
        self.rozmiar, self.krycie, self.slad = float(rozmiar), float(krycie), int(slad)
        if nowe != stare or widok:
            self.update()
            return True
        return False

    def ustaw_wybrany(self, indeks):
        self.wybrany = None if indeks is None or indeks < 0 else int(indeks)
        self.update()

    def zadaj_statystyki(self, krok, first, count):
        """Mapa 20 × 30 i dokładna mediana normy 768D w widoku dla zapisanego kroku (liczone na karcie)."""
        klucz = (krok, int(first), int(count), self._generacja)
        if klucz != self._stat_klucz:
            self._stat_klucz = klucz
            self.update()

    def zadaj_wybor(self, x, y, w, h):
        """Kliknięcie w pikselach widżetu (x, y) przy rozmiarze (w, h); wynik przez sygnał `wybrano`."""
        self._wybor = (float(x), float(y), float(w), float(h))
        self.update()

    # ------------------------------------------------------------------ GL
    def _program(self, vs=None, fs=None, cs=None):
        wariant = lambda src: src.replace('//Q16DEF', f'#define Q16 {int(self.q16)}')
        if cs is not None:
            return compileProgram(compileShader(wariant(cs), GL.GL_COMPUTE_SHADER))
        return compileProgram(compileShader(wariant(vs), GL.GL_VERTEX_SHADER), compileShader(fs, GL.GL_FRAGMENT_SHADER),
                              validate=False)

    def _kompiluj(self):
        """Programy w wariancie formatu bieżącej osi czasu (f32 albo q16); ponownie tylko przy zmianie formatu."""
        if self._prog:
            for prog in self._prog.values():
                GL.glDeleteProgram(prog)
        self._prog = dict(kin=self._program(cs=CS_KIN), stat=self._program(cs=CS_STAT), heat=self._program(cs=CS_HEAT),
                          bb=self._program(cs=CS_BB), ramka=self._program(cs=CS_RAMKA), hist=self._program(cs=CS_HIST),
                          wybor=self._program(cs=CS_WYBOR), pick=self._program(cs=CS_PICK),
                          punkty=self._program(VS_PUNKTY, FS_PUNKTY), slad=self._program(VS_SLAD, FS_KOLOR))
        self._uni = {}
        self._prog_q16 = self.q16

    def _init(self):
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
        try:
            self.max_blok = int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE))
        except Exception:
            self.max_blok = MAX_BLOK
        if self.max_blok <= 0:
            self.max_blok = MAX_BLOK
        self._odswiez_vram()
        self._kompiluj()
        ids = np.atleast_1d(GL.glGenBuffers(len(BUFORY) + 1))
        self._buf = dict(zip(BUFORY + ('RB',), (int(i) for i in ids)))     # RB: mały bufor odczytu
        self._uni = {}
        self._planuj_sloty()

    def _odswiez_vram(self):
        try:
            kb = int(np.atleast_1d(GL.glGetIntegerv(0x9049))[0])      # GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
            self.vram_wolna = kb * 1024 + self.pamiec_karty()
        except Exception:
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass
            self.vram_wolna = None

    def _u(self, prog, nazwa):
        k = (prog, nazwa)
        if k not in self._uni:
            self._uni[k] = GL.glGetUniformLocation(self._prog[prog], nazwa)
        return self._uni[k]

    def _uniformy(self, prog, **wartosci):
        GL.glUseProgram(self._prog[prog])
        for nazwa, v in wartosci.items():
            loc = self._u(prog, nazwa)
            if loc < 0:
                continue
            if isinstance(v, np.ndarray) and v.shape == (4, 4):
                GL.glUniformMatrix4fv(loc, 1, False, v)
            elif isinstance(v, (tuple, list, np.ndarray)):
                v = [float(x) for x in v]
                (GL.glUniform2f if len(v) == 2 else GL.glUniform3f)(loc, *v)
            elif isinstance(v, float):
                GL.glUniform1f(loc, v)
            elif isinstance(v, bool):
                GL.glUniform1ui(loc, int(v))
            elif nazwa in ('u_kolor', 'u_wybrany'):
                GL.glUniform1i(loc, int(v))
            else:
                GL.glUniform1ui(loc, int(v) & 0xFFFFFFFF)

    def _alokuj(self, nazwa, bajty, dane=None, uzycie=GL.GL_DYNAMIC_DRAW):
        while GL.glGetError() != GL.GL_NO_ERROR:          # błędy innych elementów nie są błędem alokacji
            pass
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[nazwa])
        bajty = max(int(bajty), 16)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, bajty, dane, uzycie)
        if GL.glGetError() == GL.GL_OUT_OF_MEMORY:
            raise MemoryError(f'Za mało pamięci karty na bufor {nazwa} ({bajty / 1e6:.0f} MB).')
        self._rozm[nazwa] = bajty

    def _przygotuj(self):
        """Alokacja buforów dla bieżącej osi czasu i budżetu (tylko przy zmianie układu)."""
        if self._uklad is not None and self._uklad[:5] == (self._generacja, self.N, self.T, self.images, self.tpi)                 and self._uklad[5] == self.n_slotow:
            return
        self._odswiez_vram()
        self._planuj_sloty()
        while True:
            try:
                self._alokuj('S', self.n_slotow * self.frame_bytes)
                break
            except MemoryError:
                if self.n_slotow <= 3:
                    raise
                self.n_slotow = max(3, self.n_slotow // 2)
        self._alokuj('CUR', self.N * 16, np.zeros((self.N, 4), np.float32))
        self._alokuj('QP', self.n_slotow * 32, np.zeros((self.n_slotow * 2, 4), np.float32))
        if getattr(self, '_prog_q16', None) != self.q16:
            self._kompiluj()
        self._alokuj('FR', 8 * 16, np.zeros((8, 4), np.float32))
        self._alokuj('PIB', self.T * self.images * 64)
        self._alokuj('KNOWN', self.T * 4, np.zeros(self.T, np.uint32))
        self._alokuj('HEAT', self.T * self.tpi * 4)
        self._alokuj('HIST', (R_HIST + 8) * 4)
        self._alokuj('PICK', 16)
        self._alokuj('SLOT', self.T * 4, np.full(self.T, NONE, np.uint32))
        self._alokuj('BB', 32)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._sloty.clear()
        self._wolne = list(range(self.n_slotow))
        self._znane = set()
        self._uklad = (self._generacja, self.N, self.T, self.images, self.tpi, self.n_slotow)
        self._slot_map_brudna = True
        self._klucz_kin, self._cur_ok, self._ramka_brudna = None, False, True

    def _wiaz(self):
        for i, n in enumerate(BUFORY):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, self._buf[n])

    def _chronione(self):
        ia = self._idx.get(self.a, 0)
        ib = self._idx.get(self.b, ia)
        ochrona = {self.steps[0], self.steps[ia], self.steps[ib]}
        if not self.cala_os:
            for k in range(1, max(0, min(self.n_slotow - 3, 8)) + 1):
                ochrona.add(self.steps[(ib + k) % self.T])
        return ochrona, ia

    def _slot_dla(self, krok):
        if self._wolne:
            return self._wolne.pop(0)
        ochrona, ia = self._chronione()
        kandydaci = [k for k in self._sloty if k not in ochrona]
        if not kandydaci:
            return None

        def odleglosc(k):                         # najpierw stany daleko za bieżącym, potem daleko przed
            i = self._idx[k]
            return (ia - i) + self.T if i < ia else i - ia
        stary = max(kandydaci, key=odleglosc)
        slot = self._sloty.pop(stary)
        return slot

    def _wyslij_stany(self):
        """Najpierw potrzebne teraz (pierwszy krok, para), potem do `na_klatke` kolejnych."""
        if not self._gotowe:
            return
        pilne = [s for s in dict.fromkeys((self.steps[0], self.a, self.b)) if s in self._gotowe]
        reszta = [s for s in self._gotowe if s not in pilne]
        if self.steps[0] not in self._sloty and self.steps[0] not in self._gotowe:
            reszta = []                             # statystyki potrzebują pierwszego kroku na karcie
        for krok in pilne + reszta[:self.na_klatke]:
            slot = self._slot_dla(krok)
            if slot is None:
                break
            arr = self._gotowe.pop(krok)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['S'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, slot * self.frame_bytes, self.frame_bytes, arr)
            self.wyslane_bajty += self.frame_bytes
            if self.q16:
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['QP'])
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, slot * 32, 32, self.timeline.quant(krok))
            self._sloty[krok] = slot
            self._slot_map_brudna = True
            if krok == self.a or krok == self.b:
                self._klucz_kin = None
            self._statystyki_kroku(krok, slot)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    def _statystyki_kroku(self, krok, slot):
        k = self._idx[krok]
        s0 = self._sloty.get(self.steps[0])
        if s0 is None:
            return
        prev = self._sloty.get(self.steps[k - 1]) if k > 0 else None
        dt = float(krok - self.steps[k - 1]) if k > 0 else 0.0
        self._wiaz()
        self._uniformy('stat', u_N=self.N, u_k=k, u_slot=slot, u_s0=s0, u_prev=NONE if prev is None else prev,
                       u_tpi=self.tpi, u_img=self.images, u_dt=dt)
        GL.glDispatchCompute(self.images, 1, 1)
        self._uniformy('heat', u_N=self.N, u_k=k, u_slot=slot, u_tpi=self.tpi, u_img=self.images)
        GL.glDispatchCompute(self.tpi, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        self._znane.add(krok)
        self._ramka_brudna = True
        if krok == self.steps[0]:                 # przesunięcia pozostałych kroków liczone względem pierwszego
            for inny, sl in list(self._sloty.items()):
                if inny != krok:
                    self._statystyki_kroku(inny, sl)

    def _odswiez_mape_slotow(self):
        if not self._slot_map_brudna:
            return
        m = np.full(self.T, NONE, np.uint32)
        for krok, slot in self._sloty.items():
            m[self._idx[krok]] = slot
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['SLOT'])
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, m.nbytes, m)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._slot_map_brudna = False

    def _kinematyka(self):
        sa, sb, s0 = self._sloty.get(self.a), self._sloty.get(self.b), self._sloty.get(self.steps[0])
        if sa is None or sb is None or s0 is None:
            return False
        first, count = self.first, min(self.count, self.N - self.first)
        klucz = (sa, sb, s0, self.a, self.b, self.s, self.ruch, self.kolor, first, count)
        if klucz == self._klucz_kin and self._cur_ok:
            return False
        for k in (self.a, self.b):
            self._sloty.move_to_end(k)
        self._uniformy('kin', u_N=self.N, u_first=first, u_count=count, u_sa=sa, u_sb=sb, u_s0=s0,
                       u_ruch=self.ruch, u_kolor=self.kolor, u_s=float(self.s),
                       u_dt=float(self.b - self.a) if self.b != self.a else 0.0)
        GL.glDispatchCompute((count + 255) // 256, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        self._klucz_kin, self._cur_ok, self._rys_count = klucz, True, count
        self._rys_first = first
        return True

    def _ramka(self, kin_nowa):
        if self.kadr == 2:
            if not kin_nowa and not self._ramka_brudna:
                return
            bb = np.array([NONE] * 3 + [0] * 3 + [0, 0], np.uint32)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['BB'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, bb.nbytes, bb)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            self._uniformy('bb', u_N=self.N, u_first=self._rys_first, u_count=self._rys_count)
            GL.glDispatchCompute((self._rys_count + 255) // 256, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        elif not self._ramka_brudna:
            return
        img0 = self.first // max(self.tpi, 1)
        img1 = max(img0 + 1, -(-(self.first + self.count) // max(self.tpi, 1)))
        nat, mov = self.granice_manifest or (None, None)
        g = mov if self.ruch else nat
        self._uniformy('ramka', u_N=self.N, u_T=self.T, u_img=self.images, u_img0=img0, u_img1=img1,
                       u_ruch=self.ruch, u_tryb=self.kadr, u_manifest=g is not None,
                       u_mlo=tuple(g[:, 0]) if g is not None else (0.0, 0.0, 0.0),
                       u_mhi=tuple(g[:, 1]) if g is not None else (0.0, 0.0, 0.0),
                       u_normscale=float(self.normscale))
        GL.glDispatchCompute(1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        self._ramka_brudna = False

    def _czytaj_bufor(self, nazwa, dtype, n, offset=0):
        """Odczyt fragmentu bufora przez kopię na karcie do małego bufora RB. Bezpośredni glGetBufferSubData z
        bufora roboczego sprawia, że sterownik przenosi CAŁY bufor do pamięci systemowej (dla 1,5 GB magazynu:
        ~0,5 s, a potem kernele i atomiki czytają przez PCIe), więc bufory robocze nigdy nie są czytane wprost."""
        out = np.empty(n, dtype)
        if out.nbytes == 0:
            return out
        if self._rozm.get('RB', 0) < out.nbytes:
            self._alokuj('RB', max(out.nbytes, 1 << 16), None, GL.GL_STREAM_READ)
        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, self._buf[nazwa])
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, self._buf['RB'])
        GL.glCopyBufferSubData(GL.GL_COPY_READ_BUFFER, GL.GL_COPY_WRITE_BUFFER, int(offset), 0, out.nbytes)
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, self._buf['RB'])
        GL.glGetBufferSubData(GL.GL_COPY_READ_BUFFER, 0, out.nbytes, out.ctypes.data_as(ctypes.c_void_p))
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)
        return out

    def _statystyki_widoku(self):
        """Mapa i dokładna mediana normy 768D dla zadanego kroku i widoku; odczyt kilku liczb z karty."""
        if self._stat_klucz is None or self._stat_klucz == self._stat_zrobione:
            return
        krok, first, count, gen = self._stat_klucz
        slot = self._sloty.get(krok)
        if gen != self._generacja or slot is None or krok not in self._znane:
            return
        count = min(count, self.N - first)
        if count <= 0:
            self._stat_zrobione = self._stat_klucz
            return
        k = self._idx[krok]
        if count == self.N:
            mapa = self._czytaj_bufor('HEAT', np.float32, self.tpi, k * self.tpi * 4)
        else:
            if self.q16:
                q = self._czytaj_bufor('S', np.uint16, count * 4, (slot * self.N + first) * 8).reshape(-1, 4)
                mapa = (q[:, 3].astype(np.float32) / np.float32(65535)) * self.timeline.quant(krok)[3]
            else:
                mapa = self._czytaj_bufor('S', np.float32, count * 4, (slot * self.N + first) * 16).reshape(-1, 4)[:, 3]
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['HIST'])
        GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, None)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        grupy = (count + 255) // 256
        for faza in (0, 1):
            self._uniformy('hist', u_N=self.N, u_slot=slot, u_first=first, u_count=count, u_faza=faza)
            GL.glDispatchCompute(grupy, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            self._uniformy('wybor', u_N=self.N, u_faza=faza, u_klo=(count - 1) // 2, u_khi=count // 2)
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        wyn = self._czytaj_bufor('HIST', np.uint32, 8, R_HIST * 4)
        f = wyn.view(np.float32)
        self._stat_zrobione = self._stat_klucz
        self.statystyki_gotowe.emit({'klucz': self._stat_klucz, 'krok': krok, 'first': first, 'count': count,
                                     'mapa': mapa, 'mediana_dol': float(f[5]), 'mediana_gora': float(f[6]),
                                     'mediana': (float(f[5]) + float(f[6])) / 2, 'maksimum': float(f[2]),
                                     'bity': (int(wyn[5]), int(wyn[6]), int(wyn[2]))})

    def _wybierz(self, mvp):
        if self._wybor is None or not self._cur_ok:
            return
        x, y, w, h = self._wybor
        self._wybor = None
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['PICK'])
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, 16, np.full(4, NONE, np.uint32))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        for faza in (0, 1):
            self._uniformy('pick', u_N=self.N, u_mvp=mvp, u_klik=(x, y), u_wh=(w, h), u_first=self._rys_first,
                           u_count=self._rys_count, u_faza=faza)
            GL.glDispatchCompute((self._rys_count + 255) // 256, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        wyn = self._czytaj_bufor('PICK', np.uint32, 4)
        d2 = float(wyn[:1].view(np.float32)[0]) if wyn[0] != NONE else np.inf
        self.ostatni_wybor = (int(wyn[1]) if wyn[1] != NONE else -1, d2)
        self.wybrano.emit(int(wyn[1]) if wyn[1] != NONE and d2 <= 144.0 else -1)

    def oblicz(self, mvp=None):
        """Wszystkie kernele klatki (bez rysowania). Wołać przy aktywnym kontekście widoku."""
        self._przygotuj()
        self._wiaz()
        self._wyslij_stany()
        self._odswiez_mape_slotow()
        self._wiaz()
        kin = self._kinematyka()
        if self._cur_ok:
            self._ramka(kin)
        self._statystyki_widoku()
        if mvp is not None:
            self._wybierz(mvp)

    def paint(self):
        if self.timeline is None or not self.T:
            return
        try:
            if self._prog is None:
                self._init()
            self.setupGLState()
            mvp = np.asarray(self.mvpMatrix().data(), np.float32).reshape(4, 4)
            self.oblicz(mvp)
            if self._cur_ok:
                self._rysuj(mvp)
            self._licz_fps()
            self.blad = ''
        except Exception as e:                                   # błąd GL nie może zatrzymać całej aplikacji
            self.blad = f'{type(e).__name__}: {e}'
        finally:                                                 # stan GL jak przed rysowaniem: inne elementy i QPainter
            GL.glUseProgram(0)
            GL.glLineWidth(1.0)
            GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            for i in range(len(BUFORY)):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, 0)

    def _rysuj(self, mvp):
        s0 = self._sloty.get(self.steps[0], 0)
        ia = self._idx.get(self.a, 0)
        if self.slad > 0:
            if self._rys_count <= 6000:
                n, wyb = self._rys_count, -1
            else:
                n, wyb = (1, self.wybrany) if self.wybrany is not None else (0, -1)
            if n:
                self._uniformy('slad', u_N=self.N, u_mvp=mvp, u_first=self._rys_first, u_ia=ia, u_K=self.slad,
                               u_ruch=self.ruch, u_s0=s0, u_wybrany=wyb, u_alfa=min(1.0, self.krycie * 2.5))
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2 * self.slad, n)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_POINT_SPRITE)
        self._uniformy('punkty', u_N=self.N, u_mvp=mvp, u_first=self._rys_first, u_kolor=self.kolor, u_wybrany=-1,
                       u_size=float(self.rozmiar), u_opacity=float(self.krycie))
        GL.glDrawArrays(GL.GL_POINTS, 0, self._rys_count)
        if self.wybrany is not None and self.wybrany < self.N:
            GL.glUniform1i(self._u('punkty', 'u_wybrany'), self.wybrany)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)
        self.draw_count = self._rys_count

    def _licz_fps(self):
        self._klatki += 1
        now = time.perf_counter()
        if now - self._zegar >= 1.0:
            self.fps = self._klatki / (now - self._zegar)
            self._zegar, self._klatki = now, 0

    # ------------------------------------------------------------------ odczyt (testy i diagnostyka)
    def odczytaj(self, nazwa, n, dtype=np.float32, offset=0, kolumny=4):
        """n wierszy bufora `nazwa` (CUR, FR, PIB, HEAT, ...). Wołać przy aktywnym kontekście widoku."""
        out = self._czytaj_bufor(nazwa, dtype, n * kolumny, offset)
        return out.reshape(n, kolumny) if kolumny > 1 else out

    def slot(self, krok):
        return self._sloty.get(krok)

    def release(self):
        """Zwalnia bufory i programy. Wołać przy aktywnym kontekście widoku."""
        if self._buf:
            GL.glDeleteBuffers(len(self._buf), list(self._buf.values()))
        if self._prog:
            for p in self._prog.values():
                GL.glDeleteProgram(p)
        self._prog, self._buf, self._rozm, self._uni = None, {}, {}, {}
        self._uklad = None
        self._sloty.clear()

    def zamknij(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
