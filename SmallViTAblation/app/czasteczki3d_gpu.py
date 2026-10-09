"""Cząsteczki 3D nad obrazem liczone i rysowane na GPU (OpenGL 4.3: compute shadery GLSL + SSBO), natywnie pod Windows.

Na kartę trafiają tylko dane, które się zmieniły: latent PRZED i PO [600, 768] przy nowej klatce, wybór kanałów albo
baza PCA przy zmianie opcji, mapa atencji [600], łuki (źródło, cel, waga) i zdjęcie 480 x 320. Resztę liczą shadery:

- kernel `wartosci`: wartości wektorów każdej cząsteczki PRZED i PO (kanały albo wyniki PCA (z − μ) W);
- kernele `histogram` i `wybor`: 99,5. centyl |wartości| (skala długości) dokładnie jak np.percentile (wybór
  rangi po 32 bitach klucza w dwóch przebiegach histogramu, potem interpolacja liniowa jak w NumPy);
- kernel `mieszanka` (jedna grupa robocza): kolor patcha = kolory wektorów ważone siłą albo PCA jako RGB albo
  kosinus do wybranego patcha, z centylami (sortowanie bitoniczne 600 wartości w pamięci współdzielonej);
- kernel `podloze`: tekstura płaszczyzny obrazu (jasność zdjęcia, mapa atencji inferno, mieszanka kolorów
  kwadratami albo gładko jak scipy.ndimage.zoom rzędu 1);
- shadery wierzchołków: mikrowektory i końce (środek + długość · kierunek kanału, długość przycięta), szare
  wektory PRZED, okręgi zera, łuki atencji z grubością według wagi (5 klas) i groty; bez list wierzchołków z CPU
  (gl_VertexID + SSBO).

Przejście PRZED → PO (suwak, animacja) to jedna liczba t w uniformie: geometria interpoluje się w shaderze.
Obrót kamery tylko rysuje (bez żadnego przeliczenia). Wymaga OpenGL 4.3 (NVIDIA/AMD/Intel pod Windows).
"""
import ctypes

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

NT = 600
KOLUMNY, WIERSZE, KOMORKA = 30, 20, 16
SZER, WYS = KOLUMNY * KOMORKA, WIERSZE * KOMORKA
KANALY = 768
BINY = 65536                                    # histogram 16 bitów klucza
SEG_LUKU = 24                                    # odcinków na łuk atencji (jak dotąd)
SEG_ZERA = 32                                    # odcinków na okrąg zera

BUFORY = ('LAT', 'KAN', 'PCA', 'VAL', 'KIER', 'KOL', 'VIS', 'MAPA', 'LUK', 'LUT', 'HIST', 'ST', 'MIES', 'RGBP')

_SSBO = """
layout(std430, binding = 0) buffer BLAT { float LAT[]; };      // [2 * 600 * 768] latent PRZED, potem PO
layout(std430, binding = 1) buffer BKAN { uint KAN[]; };       // [C] kanały, z których liczone są wektory
layout(std430, binding = 2) buffer BPCA { float PCA[]; };      // mu [C], potem W [C, K] wierszami
layout(std430, binding = 3) buffer BVAL { float VAL[]; };      // [2 * 600 * K] wartości wektorów PRZED, PO
layout(std430, binding = 4) buffer BKIER { vec4 KIER[]; };     // [K] kierunki wektorów
layout(std430, binding = 5) buffer BKOL { vec4 KOL[]; };       // [K] kolory wektorów
layout(std430, binding = 6) buffer BVIS { uint VIS[]; };       // rysowane cząsteczki (tokeny)
layout(std430, binding = 7) buffer BMAPA { float MAPA[]; };    // [600] mapa atencji 0..1
layout(std430, binding = 8) buffer BLUK { vec4 LUK[]; };       // [E] łuki: źródło, cel, waga
layout(std430, binding = 9) buffer BLUT { vec4 LUT[]; };       // [3 * 256] inferno, cool, RdBu_r (matplotlib)
layout(std430, binding = 10) buffer BHIST { uint HIST[]; };    // [3 * 65536] histogramy wyboru rangi
layout(std430, binding = 11) buffer BST { uint ST[]; };        // wyniki wyboru: kosze, rangi, klucze, m
layout(std430, binding = 12) buffer BMIES { vec4 MIES[]; };    // [600] kolor mieszanki (rgb) i siła
layout(std430, binding = 13) buffer BRGBP { float RGBP[]; };   // PCA RGB: mu [768], W [768, 3], lo [3], hi [3]
const uint NT = 600u;
uniform uint u_K, u_C;
uniform int u_pca, u_wtryb, u_mapa;
uniform float u_t, u_h;
float wart(uint s, uint t, uint j) { return VAL[(s * NT + t) * u_K + j]; }
float biezaca(uint t, uint j) {                // wartość wektora j tokenu t w tym, co pokazuje widok
    float a = wart(0u, t, j);
    if (u_wtryb == 0) return a;
    float b = wart(1u, t, j);
    if (u_wtryb == 2) return b - a;
    if (u_wtryb == 4) return a + u_t * (b - a);
    return b;
}
float lat_biezacy(uint t, uint c) {            // pełny latent (768) tokenu t w tym, co pokazuje widok
    float a = LAT[t * 768u + c];
    if (u_wtryb == 0) return a;
    float b = LAT[(NT + t) * 768u + c];
    if (u_wtryb == 2) return b - a;
    if (u_wtryb == 4) return a + u_t * (b - a);
    return b;
}
vec3 srodek(uint t) {                          // środek patcha na obrazie, podniesiony o atencję
    float z = (u_h > 0.0 && u_mapa == 1) ? u_h * MAPA[t] : 0.0;
    return vec3(float(t % 30u) * 16.0 + 8.0, 320.0 - (float(t / 30u) * 16.0 + 8.0), z);
}
int indeks_lut(float x) { return clamp(int(x * 256.0), 0, 255); }   // jak matplotlib dla wartości 0..1
"""

CS_WARTOSCI = "#version 430\n" + _SSBO + """
layout(local_size_x = 256) in;
uniform uint u_S;                              // 1: tylko PRZED, 2: PRZED i PO
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_S * NT * u_K) return;
    uint j = i % u_K, st = i / u_K;            // st = s * 600 + t
    uint baza = st * 768u;
    if (u_pca == 0) { VAL[i] = LAT[baza + KAN[j]]; return; }
    float acc = 0.0;
    for (uint c = 0u; c < u_C; c++) acc += (LAT[baza + KAN[c]] - PCA[c]) * PCA[u_C + c * u_K + j];
    VAL[i] = acc;
}
"""

CS_HISTOGRAM = "#version 430\n" + _SSBO + """
layout(local_size_x = 256) in;
uniform uint u_N, u_faza;
uniform int u_zakres;                          // 0: |PRZED|, 1: |PRZED| i |PO|, 2: |PO − PRZED|
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_N) return;
    float v = u_zakres == 2 ? VAL[NT * u_K + i] - VAL[i] : VAL[i];
    uint k = floatBitsToUint(abs(v));          // dla liczb >= 0 porządek kluczy = porządek wartości
    uint hi = k >> 16u;
    if (u_faza == 0u) { atomicAdd(HIST[hi], 1u); return; }
    if (hi == ST[0]) atomicAdd(HIST[65536u + (k & 65535u)], 1u);
    if (hi == ST[2]) atomicAdd(HIST[131072u + (k & 65535u)], 1u);
}
"""

CS_WYBOR = "#version 430\n" + _SSBO + """
layout(local_size_x = 1024) in;
uniform uint u_faza, u_ra, u_rb;
uniform float u_gamma;
shared uint pre[1024];
shared uint w_kosz, w_ranga;
void szukaj(uint seg, uint r) {                // kosz histogramu, w którym leży element o randze r (od 0)
    uint l = gl_LocalInvocationID.x, baza = seg * 65536u + l * 64u, s = 0u;
    for (uint b = 0u; b < 64u; b++) s += HIST[baza + b];
    pre[l] = s;
    barrier();
    for (uint o = 1u; o < 1024u; o <<= 1u) {
        uint v = l >= o ? pre[l - o] : 0u;
        barrier();
        pre[l] += v;
        barrier();
    }
    uint inc = pre[l], c = inc - s;
    if (r >= c && r < inc) {
        for (uint b = 0u; b < 64u; b++) {
            uint h = HIST[baza + b];
            if (r < c + h) { w_kosz = l * 64u + b; w_ranga = r - c; break; }
            c += h;
        }
    }
    barrier();
}
void main() {
    uint l = gl_LocalInvocationID.x;
    if (u_faza == 0u) {
        szukaj(0u, u_ra);
        if (l == 0u) { ST[0] = w_kosz; ST[1] = w_ranga; }
        barrier();
        szukaj(0u, u_rb);
        if (l == 0u) { ST[2] = w_kosz; ST[3] = w_ranga; }
    } else {
        uint ra = ST[1], rb = ST[3];
        barrier();
        szukaj(1u, ra);
        uint ka = (ST[0] << 16u) | w_kosz;
        barrier();
        szukaj(2u, rb);
        if (l == 0u) {
            uint kb = (ST[2] << 16u) | w_kosz;
            float a = uintBitsToFloat(ka), b = uintBitsToFloat(kb), d = b - a;
            float m = u_gamma >= 0.5 ? b - d * (1.0 - u_gamma) : a + d * u_gamma;     // jak numpy _lerp
            ST[4] = floatBitsToUint(max(m, 1e-6));
            ST[5] = ka; ST[6] = kb;
        }
    }
}
"""

CS_MIESZANKA = "#version 430\n" + _SSBO + """
layout(local_size_x = 1024) in;
uniform int u_waga, u_jasn, u_tok;
uniform float u_nasyc;
uniform uint u_k98, u_k99;
uniform float u_g98, u_g99;
shared float klucz[1024];
shared float sr[768];
void sortuj() {                                // sortowanie bitoniczne 1024 kluczy rosnąco
    uint l = gl_LocalInvocationID.x;
    for (uint k = 2u; k <= 1024u; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            uint p = l ^ j;
            if (p > l) {
                float a = klucz[l], b = klucz[p];
                bool rosnaco = (l & k) == 0u;
                if ((a > b) == rosnaco) { klucz[l] = b; klucz[p] = a; }
            }
            barrier();
        }
    }
}
float centyl(uint k, float g) {                // interpolacja jak numpy _lerp
    float a = klucz[k], b = klucz[k + 1u], d = b - a;
    return g >= 0.5 ? b - d * (1.0 - g) : a + d * g;
}
void main() {
    uint l = gl_LocalInvocationID.x;
    bool jest = l < NT;
    if (u_waga == 4) {                         // 3 główne składowe pełnego latentu jako RGB
        if (!jest) return;
        vec3 S = vec3(0.0);
        for (uint c = 0u; c < 768u; c++) {
            float x = lat_biezacy(l, c) - RGBP[c];
            S += x * vec3(RGBP[768u + c * 3u], RGBP[768u + c * 3u + 1u], RGBP[768u + c * 3u + 2u]);
        }
        vec3 lo = vec3(RGBP[3072u], RGBP[3073u], RGBP[3074u]), hi = vec3(RGBP[3075u], RGBP[3076u], RGBP[3077u]);
        MIES[l] = vec4(clamp((S - lo) / max(hi - lo, vec3(1e-9)), 0.0, 1.0), 0.0);
        return;
    }
    if (u_waga == 5) {                         // kosinus latentu (po odjęciu średniej klatki) z wybranym patchem
        if (u_tok < 0) { if (jest) MIES[l] = vec4(vec3(0.35), 0.0); return; }
        if (l < 768u) {
            float s = 0.0;
            for (uint t = 0u; t < NT; t++) s += lat_biezacy(t, l);
            sr[l] = s / float(NT);
        }
        barrier();
        if (!jest) return;
        uint tk = uint(u_tok);
        float d = 0.0, n = 0.0, nt = 0.0;
        for (uint c = 0u; c < 768u; c++) {
            float x = lat_biezacy(l, c) - sr[c], y = lat_biezacy(tk, c) - sr[c];
            d += x * y; n += x * x; nt += y * y;
        }
        float cs = d / max(sqrt(n) * sqrt(nt), 1e-12);
        MIES[l] = vec4(LUT[512 + indeks_lut((clamp(cs, -1.0, 1.0) + 1.0) / 2.0)].rgb, 0.0);
        return;
    }
    vec3 C = vec3(0.0); float sila = 0.0, g = 0.0;
    if (jest) {
        for (uint j = 0u; j < u_K; j++) {
            float x = biezaca(l, j);
            float w = u_waga == 0 ? abs(x) : u_waga == 1 ? x * x : u_waga == 2 ? (x * x) * (x * x) : max(x, 0.0);
            sila += w;
            C += w * KOL[j].rgb;
        }
        C /= max(sila, 1e-12);
        g = (C.r + C.g + C.b) / 3.0;
        vec3 o = abs(C - g);
        klucz[l] = max(o.r, max(o.g, o.b));
    } else klucz[l] = 3.0e38;
    barrier();
    sortuj();
    float ref = max(centyl(u_k98, u_g98), 1e-6);
    barrier();
    vec3 W = clamp(g + (C - g) * (0.45 / ref) * u_nasyc, 0.0, 1.0);
    if (u_jasn == 1) {
        klucz[l] = jest ? sila : 3.0e38;
        barrier();
        sortuj();
        float p = max(centyl(u_k99, u_g99), 1e-12);
        W *= clamp(sila / p, 0.0, 1.0);
    }
    if (jest) MIES[l] = vec4(W, sila);
}
"""

CS_PODLOZE = "#version 430\n" + _SSBO + """
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) writeonly uniform image2D u_wyj;
uniform sampler2D u_rgb;
uniform int u_obraz, u_atencja, u_mies, u_gladko;
uniform float u_jasnosc, u_kr_at, u_kr_mies, u_alfa;
void main() {
    ivec2 g = ivec2(gl_GlobalInvocationID.xy);
    if (g.x >= 480 || g.y >= 320) return;
    int r = 319 - g.y, x = g.x;                // wiersz zdjęcia; tekstura ma y w górę (jak dotąd)
    uint p = uint((r / 16) * 30 + x / 16);
    vec3 img = vec3(0.0);
    if (u_obraz == 1) img = texelFetch(u_rgb, ivec2(x, r), 0).rgb * u_jasnosc;
    if (u_atencja == 1) img = img * (1.0 - u_kr_at) + LUT[indeks_lut(MAPA[p])].rgb * u_kr_at;
    if (u_mies == 1) {
        vec3 M;
        if (u_gladko == 1) {                   // scipy.ndimage.zoom(order=1): we = wy · (n_we − 1) / (n_wy − 1)
            float rr = float(r) * 19.0 / 319.0, cc = float(x) * 29.0 / 479.0;
            int r0 = min(int(rr), 18), c0 = min(int(cc), 28);
            float fr = rr - float(r0), fc = cc - float(c0);
            uint a = uint(r0 * 30 + c0);
            M = (1.0 - fr) * ((1.0 - fc) * MIES[a].rgb + fc * MIES[a + 1u].rgb)
                + fr * ((1.0 - fc) * MIES[a + 30u].rgb + fc * MIES[a + 31u].rgb);
        } else M = MIES[p].rgb;
        img = img * (1.0 - u_kr_mies) + M * u_kr_mies;
    }
    imageStore(u_wyj, g, vec4(floor(clamp(img * 255.0, 0.0, 255.0)) / 255.0, u_alfa));
}
"""

VS_CZASTECZKI = "#version 430\n" + _SSBO + """
uniform mat4 u_mvp;
uniform int u_warstwa, u_zrodlo, u_klasa;     // warstwa: 0 wektory, 1 końce, 2 okręgi zera, 3 łuki, 4 groty
                                               // u_klasa < 0: łuki wszystkich klas (odczyt w testach)
uniform vec4 u_kol;                            // alfa >= 0: jeden kolor dla wszystkich
uniform float u_rozm, u_r0, u_proc, u_maks;
out vec4 v_kol;
out vec3 v_poz;                                // położenie w układzie sceny (odczyt transform feedback w testach)
out float v_rozm;
vec3 koniec(uint tok, uint j, vec3 c) {
    float m = uintBitsToFloat(ST[4]);
    float skala = 8.0 > u_r0 ? (8.0 - u_r0) / m * u_proc : 8.0 / m;
    float v = u_zrodlo == 0 ? wart(0u, tok, j) : biezaca(tok, j);
    float dl = max(u_r0 + skala * v, 0.0);
    if (u_maks > 0.0) dl = min(dl, u_maks);
    return c + dl * KIER[j].xyz;
}
void main() {
    int v = gl_VertexID;
    vec3 p; vec4 kol; float rozm = u_rozm;
    bool poza = false;
    if (u_warstwa == 0) {                      // odcinek środek -> koniec wektora j cząsteczki
        uint para = uint(v / 2), tok = VIS[para / u_K], j = para % u_K;
        vec3 c = srodek(tok);
        p = v % 2 == 0 ? c : koniec(tok, j, c);
        kol = KOL[j];
    } else if (u_warstwa == 1) {               // punkt na końcu wektora
        uint tok = VIS[uint(v) / u_K], j = uint(v) % u_K;
        p = koniec(tok, j, srodek(tok));
        kol = KOL[j];
    } else if (u_warstwa == 2) {               // okrąg zera wokół środka
        uint tok = VIS[uint(v) / 64u];
        int i = (v % 64) / 2 + v % 2;
        float th = 6.283185307179586 * float(i) / 32.0;
        p = srodek(tok) + u_r0 * vec3(cos(th), sin(th), 0.0);
        kol = vec4(0.6, 0.6, 0.6, 0.7);
    } else if (u_warstwa == 3) {               // łuk źródło -> cel, w klasie grubości u_klasa
        vec4 e = LUK[v / 48];
        float w = e.z;
        poza = u_klasa >= 0 && min(int(clamp(w, 0.0, 1.0) * 5.0), 4) != u_klasa;
        int i = (v % 48) / 2 + v % 2;
        float u = float(i) / 24.0;
        vec3 a = srodek(uint(e.x)), b = srodek(uint(e.y));
        float h = 0.35 * length(b - a) + 8.0;
        p = a * (1.0 - u) + b * u;
        p.z += 4.0 * h * u * (1.0 - u);
        kol = vec4(LUT[256 + indeks_lut(w)].rgb, (0.15 + 0.85 * u) * (0.35 + 0.65 * w));
    } else {                                   // grot łuku: punkt nad celem
        vec4 e = LUK[v];
        p = srodek(uint(e.y)) + vec3(0.0, 0.0, 0.5);
        kol = vec4(LUT[256 + indeks_lut(e.z)].rgb, 1.0);
        rozm = 4.0 + 8.0 * e.z;
    }
    if (u_kol.a >= 0.0) kol = u_kol;
    gl_Position = poza ? vec4(2.0, 2.0, 2.0, 1.0) : u_mvp * vec4(p, 1.0);   // inna klasa grubości: poza ekranem
    gl_PointSize = rozm;
    v_kol = kol;
    v_poz = p;
    v_rozm = rozm;
}
"""

FS_KOLOR = """#version 430
in vec4 v_kol;
out vec4 kolor;
void main() { kolor = v_kol; }
"""

FS_PUNKT = """#version 430
in vec4 v_kol;
out vec4 kolor;
void main() {                                  // okrągły punkt jak GLScatterPlotItem
    vec2 q = (gl_PointCoord - 0.5) * 2.0;
    if (dot(q, q) > 1.0) discard;
    kolor = v_kol;
}
"""

VS_PLASZCZYZNA = """#version 430
uniform mat4 u_mvp;
out vec2 v_uv;
void main() {                                  // pas trójkątów (0,0) (480,0) (0,320) (480,320), jak GLImageItem
    vec2 uv = vec2(float(gl_VertexID % 2), float(gl_VertexID / 2));
    v_uv = uv;
    gl_Position = u_mvp * vec4(uv * vec2(480.0, 320.0), 0.0, 1.0);
}
"""

FS_PLASZCZYZNA = """#version 430
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 kolor;
void main() { kolor = texture(u_tex, v_uv); }
"""


def indeks_centyla(n, q):
    """(k, gamma) jak w np.percentile(method='linear'): wartość = x[k] + gamma · (x[k + 1] − x[k])."""
    q = np.float64(q) / 100
    vi = np.float64(n) * q + (1 + q * (1 - 1 - 1)) - 1          # numpy: _compute_virtual_index(n, q, 1, 1)
    k = int(np.floor(vi))
    k = min(max(k, 0), n - 1)
    return k, float(vi - k)


def _lut():
    import matplotlib
    L = [matplotlib.colormaps[n](np.arange(256)).astype(np.float32) for n in ('inferno', 'cool', 'RdBu_r')]
    return np.ascontiguousarray(np.concatenate(L))


class CzasteczkiGPU(GLGraphicsItem):
    """Element sceny pyqtgraph: liczy (compute) i rysuje mikrowektory, okręgi zera, łuki i groty atencji oraz
    przygotowuje teksturę płaszczyzny obrazu (rysuje ją PlaszczyznaGPU, później, przez przezroczystość)."""

    def __init__(self):
        super().__init__()
        self.setGLOptions('translucent')
        self._prog = None
        self._buf, self._rozm, self._uni = {}, {}, {}
        self._dane = {'LUT': _lut(), 'ST': np.zeros(8, np.uint32)}   # bufory czekające na wysłanie
        self._lat = None                         # (przed, po) czekające na wysłanie
        self._rgb_dane, self._rgb_obiekt = None, None
        self._tex_rgb = self._tex_pl = None
        self._klucze = {}
        self.blad, self.renderer = '', ''
        self.po_skali = None                     # funkcja(m) wołana po policzeniu 99,5. centyla na karcie
        self.po_bledzie = None                   # funkcja(tekst) wołana po błędzie OpenGL
        self.m = None
        # stan danych
        self.S, self.K, self.C, self.pca = 0, 0, 0, False
        self.n_vis, self.n_luk = 0, 0
        self.mapa_ok = False
        # parametry rysowania (uniformy)
        self.p = dict(wtryb=0, t=1.0, r0=4.0, proc=2.0, maks=0.0, h=0.0, styl=0, gr=1.0, szare=False, zera=False,
                      czastki=True, grubosc_luk=6.0, waga=0, jasn=False, nasyc=1.0, tok=-1, mies=False, gladko=False,
                      obraz=False, jasnosc=.7, atencja=False, kr_at=.5, kr_mies=.6, alfa=0)
        # co trzeba przeliczyć przy najbliższym rysowaniu
        self._brudne = {'wartosci', 'skala', 'mieszanka', 'podloze'}

    # ------------------------------------------------------------ dane z CPU (bez kontekstu GL)
    def _wyslij_pozniej(self, nazwa, a):
        self._dane[nazwa] = np.ascontiguousarray(a)

    def ustaw_latent(self, przed, po):
        """Latent PRZED [600, 768] i PO (albo None); wysyłany na kartę przy najbliższym rysowaniu."""
        self._lat = (np.ascontiguousarray(przed, np.float32), None if po is None else np.ascontiguousarray(po, np.float32))
        self.S = 1 if po is None else 2
        self._brudne |= {'wartosci', 'skala', 'mieszanka', 'podloze'}
        self.update()

    def ustaw_obraz(self, rgb):
        """Zdjęcie [320, 480, 3] uint8 (wejście ViT); ten sam obiekt nie jest wysyłany ponownie."""
        if rgb is not self._rgb_obiekt:
            self._rgb_obiekt = rgb
            self._rgb_dane = None if rgb is None else np.ascontiguousarray(np.asarray(rgb)[:WYS, :SZER, :3], np.uint8)
            self._brudne.add('podloze')
            self.update()

    def ustaw_wektory(self, klucz, kanaly, pca, kierunki, kolory):
        """Przestrzeń wektorów cząsteczki: kanały [C]; pca = (mu [C], W [C, K]) albo None (wtedy K = C);
        kierunki [K, 3], kolory [K, 4]. Klucz: te same wektory = nic do wysłania."""
        if self._klucze.get('wektory') == klucz:
            return
        self._klucze['wektory'] = klucz
        kan = np.asarray(kanaly, np.uint32)
        self.C = len(kan)
        self._wyslij_pozniej('KAN', kan)
        if pca is None:
            self.pca, self.K = False, len(kan)
        else:
            mu, W = pca
            self.pca, self.K = True, W.shape[1]
            self._wyslij_pozniej('PCA', np.concatenate([np.asarray(mu, np.float32).ravel(), np.asarray(W, np.float32).ravel()]))
        D = np.zeros((self.K, 4), np.float32); D[:, :3] = kierunki
        self._wyslij_pozniej('KIER', D)
        self._wyslij_pozniej('KOL', np.asarray(kolory, np.float32))
        self._brudne |= {'wartosci', 'skala', 'mieszanka', 'podloze'}
        self.update()

    def ustaw_widoczne(self, tokeny):
        tok = np.asarray(tokeny, np.uint32)
        if self._klucze.get('vis') is not None and np.array_equal(self._klucze['vis'], tok):
            return
        self._klucze['vis'] = tok
        self.n_vis = len(tok)
        if len(tok):
            self._wyslij_pozniej('VIS', tok)
        self.update()

    def ustaw_mape(self, A):
        """Mapa atencji [20, 30] w 0..1 albo None (podnoszenie środków i kolor płaszczyzny)."""
        if A is None:
            zmiana = self.mapa_ok
            self.mapa_ok = False
        else:
            a = np.asarray(A, np.float32).reshape(-1)
            zmiana = not (self.mapa_ok and np.array_equal(self._klucze.get('mapa'), a))
            if zmiana:
                self._klucze['mapa'] = a
                self._wyslij_pozniej('MAPA', a)
            self.mapa_ok = True
        if zmiana:
            self._brudne.add('podloze')
            self.update()

    def ustaw_luki(self, strzalki):
        """(źródła, cele, wagi 0..1) albo None."""
        if strzalki is None or not len(strzalki[0]):
            self.n_luk = 0
        else:
            src, dst, w = strzalki
            E = np.zeros((len(src), 4), np.float32); E[:, 0] = src; E[:, 1] = dst; E[:, 2] = w
            if not (self.n_luk == len(E) and np.array_equal(self._klucze.get('luki'), E)):
                self._klucze['luki'] = E
                self._wyslij_pozniej('LUK', E)
            self.n_luk = len(E)
        self.update()

    def ustaw_baze_rgb(self, baza):
        """Baza PCA RGB (mu [768], W [768, 3], lo [3], hi [3]) do koloru „PCA jako RGB”."""
        if baza is None or self._klucze.get('rgbp') is baza:
            return
        self._klucze['rgbp'] = baza
        mu, W, lo, hi = baza
        self._wyslij_pozniej('RGBP', np.concatenate([np.asarray(x, np.float32).ravel() for x in (mu, W, lo, hi)]))
        self._brudne |= {'mieszanka', 'podloze'}
        self.update()

    # parametry: które przeliczenia unieważnia zmiana danego parametru
    _ZALEZNOSCI = {'wtryb': ('skala', 'mieszanka', 'podloze'), 't': ('mieszanka', 'podloze'),
                   'waga': ('mieszanka', 'podloze'), 'jasn': ('mieszanka', 'podloze'), 'nasyc': ('mieszanka', 'podloze'),
                   'tok': ('mieszanka', 'podloze'), 'mies': ('mieszanka', 'podloze'), 'gladko': ('podloze',),
                   'obraz': ('podloze',), 'jasnosc': ('podloze',), 'atencja': ('podloze',), 'kr_at': ('podloze',),
                   'kr_mies': ('podloze',), 'alfa': ('podloze',)}

    def ustaw(self, **p):
        """Parametry widoku (liczby i przełączniki). Przelicza na karcie tylko to, co od nich zależy."""
        for k, v in p.items():
            if self.p.get(k) != v:
                self.p[k] = v
                self._brudne.update(self._ZALEZNOSCI.get(k, ()))
        self.update()

    # ------------------------------------------------------------ GL
    def _init(self):
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
        cs = lambda z: compileProgram(compileShader(z, GL.GL_COMPUTE_SHADER))
        vf = lambda v, f: compileProgram(compileShader(v, GL.GL_VERTEX_SHADER), compileShader(f, GL.GL_FRAGMENT_SHADER),
                                         validate=False)
        self._prog = dict(wartosci=cs(CS_WARTOSCI), histogram=cs(CS_HISTOGRAM), wybor=cs(CS_WYBOR),
                          mieszanka=cs(CS_MIESZANKA), podloze=cs(CS_PODLOZE),
                          linie=vf(VS_CZASTECZKI, FS_KOLOR), punkty=vf(VS_CZASTECZKI, FS_PUNKT),
                          plaszczyzna=vf(VS_PLASZCZYZNA, FS_PLASZCZYZNA))
        ids = GL.glGenBuffers(len(BUFORY))
        self._buf = dict(zip(BUFORY, (int(i) for i in np.atleast_1d(ids))))
        for n in BUFORY:                                          # każdy bufor istnieje od początku
            if n not in self._dane:
                self._dane[n] = np.zeros(4, np.float32)
        self._dane['HIST'] = np.zeros(3 * BINY, np.uint32)
        self._dane['MIES'] = np.zeros((NT, 4), np.float32)
        self._dane['MAPA'] = self._dane['MAPA'] if len(self._dane['MAPA']) == NT else np.zeros(NT, np.float32)
        self._tex_pl = int(GL.glGenTextures(1))                   # płaszczyzna obrazu 480 x 320 RGBA8
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_pl)
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA8, SZER, WYS)
        self._parametry_tekstury()
        self._tex_rgb = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_rgb)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, SZER, WYS, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        self._parametry_tekstury()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    @staticmethod
    def _parametry_tekstury():
        for par, w in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST),
                       (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, par, w)

    def _u(self, prog, nazwa):
        k = (prog, nazwa)
        if k not in self._uni:
            self._uni[k] = GL.glGetUniformLocation(self._prog[prog], nazwa)
        return self._uni[k]

    def _uni_ustaw(self, prog, **u):
        for k, v in u.items():
            loc = self._u(prog, k)
            if loc < 0:
                continue
            if isinstance(v, (bool, int, np.integer)) and k in ('u_K', 'u_C', 'u_S', 'u_N', 'u_faza', 'u_ra', 'u_rb', 'u_k98', 'u_k99'):
                GL.glUniform1ui(loc, int(v))
            elif isinstance(v, (bool, int, np.integer)):
                GL.glUniform1i(loc, int(v))
            elif isinstance(v, tuple):
                GL.glUniform4f(loc, *map(float, v))
            else:
                GL.glUniform1f(loc, float(v))

    def _wspolne(self, prog):
        p = self.p
        self._uni_ustaw(prog, u_K=self.K, u_C=self.C, u_pca=int(self.pca), u_wtryb=int(p['wtryb']) if self.S == 2 else 0,
                        u_mapa=int(self.mapa_ok),
                        u_t=float(p['t']), u_h=float(p['h']))

    def _wyslij(self):
        if self._lat is not None:                                 # latent: dwa kawałki bez sklejania na CPU
            przed, po = self._lat
            n = przed.nbytes * 2
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['LAT'])
            if self._rozm.get('LAT', 0) != n:
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n, None, GL.GL_DYNAMIC_DRAW)
                self._rozm['LAT'] = n
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, przed.nbytes, przed)
            if po is not None:
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, przed.nbytes, po.nbytes, po)
            self._lat = None
        potrzeba = 2 * NT * max(self.K, 1) * 4                   # bufor wartości PRZED i PO dla bieżącego K
        czeka = self._dane['VAL'].nbytes if 'VAL' in self._dane else 0
        if max(self._rozm.get('VAL', 0), czeka) < potrzeba:          # także gdy dane przyszły przed 1. klatką
            self._dane['VAL'] = np.zeros(potrzeba // 4, np.float32)
        for n, a in list(self._dane.items()):
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[n])
            if self._rozm.get(n, -1) >= a.nbytes > 0:
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, a.nbytes, a)
            else:
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, max(a.nbytes, 16), a if a.nbytes else None, GL.GL_DYNAMIC_DRAW)
                self._rozm[n] = max(a.nbytes, 16)
        self._dane.clear()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        if self._rgb_dane is not None:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_rgb)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, SZER, WYS, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self._rgb_dane)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            self._rgb_dane = None

    def _dispatch(self, prog, n, grupa=256):
        GL.glDispatchCompute(max(1, (int(n) + grupa - 1) // grupa), 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

    def oblicz(self):
        """Kernele compute (tylko te, które unieważniła zmiana danych albo opcji). Przy aktywnym kontekście."""
        for i, n in enumerate(BUFORY):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, self._buf[n])
        p, b = self.p, self._brudne
        gotowe = self.S > 0 and self.K > 0
        if gotowe and 'wartosci' in b:
            GL.glUseProgram(self._prog['wartosci']); self._wspolne('wartosci')
            self._uni_ustaw('wartosci', u_S=self.S)
            self._dispatch('wartosci', self.S * NT * self.K)
        if gotowe and 'skala' in b and p['czastki']:
            self._skala()
            b.discard('skala')
        if gotowe and 'mieszanka' in b and p['mies']:
            GL.glUseProgram(self._prog['mieszanka']); self._wspolne('mieszanka')
            k98, g98 = indeks_centyla(NT, 98); k99, g99 = indeks_centyla(NT, 99)
            self._uni_ustaw('mieszanka', u_waga=int(p['waga']), u_jasn=int(p['jasn']), u_tok=int(p['tok']),
                            u_nasyc=float(p['nasyc']), u_k98=k98, u_g98=g98, u_k99=k99, u_g99=g99)
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            b.discard('mieszanka')
        if 'podloze' in b:
            GL.glUseProgram(self._prog['podloze']); self._wspolne('podloze')
            GL.glActiveTexture(GL.GL_TEXTURE1); GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_rgb)
            GL.glBindImageTexture(0, self._tex_pl, 0, GL.GL_FALSE, 0, GL.GL_WRITE_ONLY, GL.GL_RGBA8)
            self._uni_ustaw('podloze', u_rgb=1, u_obraz=int(p['obraz'] and self._rgb_obiekt is not None),
                            u_atencja=int(p['atencja'] and self.mapa_ok), u_mies=int(p['mies'] and gotowe),
                            u_gladko=int(p['gladko']), u_jasnosc=float(p['jasnosc']), u_kr_at=float(p['kr_at']),
                            u_kr_mies=float(p['kr_mies']), u_alfa=float(p['alfa']) / 255)
            GL.glDispatchCompute((SZER + 15) // 16, (WYS + 15) // 16, 1)
            GL.glMemoryBarrier(GL.GL_TEXTURE_FETCH_BARRIER_BIT | GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            GL.glBindImageTexture(0, 0, 0, GL.GL_FALSE, 0, GL.GL_WRITE_ONLY, GL.GL_RGBA8)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0); GL.glActiveTexture(GL.GL_TEXTURE0)
            b.discard('podloze')
        if gotowe:
            b.discard('wartosci')

    def _zakres(self):
        """(tryb zakresu, liczba wartości): z czego liczyć 99,5. centyl (jak dotąd na CPU)."""
        w = self.p['wtryb'] if self.S == 2 else 0
        if w == 2:
            return 2, NT * self.K
        if w == 0:
            return 0, NT * self.K
        return 1, 2 * NT * self.K

    def _skala(self):
        """99,5. centyl |wartości| na karcie: wybór rangi po 16 starszych, potem 16 młodszych bitach klucza."""
        zakres, N = self._zakres()
        k, g = indeks_centyla(N, 99.5)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['HIST'])
        GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, None)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        for faza in (0, 1):
            GL.glUseProgram(self._prog['histogram']); self._wspolne('histogram')
            self._uni_ustaw('histogram', u_N=N, u_faza=faza, u_zakres=zakres)
            self._dispatch('histogram', N)
            GL.glUseProgram(self._prog['wybor'])
            self._uni_ustaw('wybor', u_faza=faza, u_ra=k, u_rb=min(k + 1, N - 1), u_gamma=g)
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        st = self.odczytaj('ST', 8, np.uint32)                    # jedna liczba do opisu (połowa patcha = m)
        self.m = float(st[4:5].view(np.float32)[0])
        if self.po_skali is not None:
            self.po_skali(self.m)

    def paint(self):
        try:
            if self._prog is None:
                self._init()
            self._wyslij()
            self.oblicz()
            if self.S == 0 or self.K == 0:
                return
            self.setupGLState()
            self._rysuj(np.asarray(self.mvpMatrix().data(), np.float32))
            self.blad = ''
        except Exception as e:                                    # błąd GL nie może zatrzymać całej aplikacji
            self.blad = f'{type(e).__name__}: {e}'
            if self.po_bledzie is not None:
                self.po_bledzie(self.blad)
        finally:
            przywroc_stan()

    def _uniformy_warstw(self, prog, mvp):
        p = self.p
        GL.glUseProgram(self._prog[prog]); self._wspolne(prog)
        GL.glUniformMatrix4fv(self._u(prog, 'u_mvp'), 1, False, mvp)
        self._uni_ustaw(prog, u_r0=float(p['r0']), u_proc=float(p['proc']), u_maks=float(p['maks']), u_zrodlo=-1,
                        u_kol=(0, 0, 0, -1), u_klasa=0, u_rozm=float(p['gr']) + 1.5)

    def _rysuj(self, mvp):
        p = self.p
        r0 = float(p['r0'])
        for prog in ('linie', 'punkty'):
            self._uniformy_warstw(prog, mvp)
        GL.glUseProgram(self._prog['linie'])
        cz = p['czastki'] and self.n_vis
        if cz and p['zera'] and r0 > 0:                           # kolejność warstw jak dotąd
            self._linie(2, 2 * SEG_ZERA * self.n_vis, 1.0)
        if cz and p['szare']:
            self._uni_ustaw('linie', u_zrodlo=0, u_kol=(.55, .55, .55, .5))
            self._linie(0, 2 * self.K * self.n_vis, p['gr'])
            self._uni_ustaw('linie', u_zrodlo=-1, u_kol=(0, 0, 0, -1))
        if cz and p['styl'] in (0, 2):
            self._linie(0, 2 * self.K * self.n_vis, p['gr'])
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_POINT_SPRITE)
        if cz and p['styl'] in (1, 2):
            GL.glUseProgram(self._prog['punkty'])
            self._uni_ustaw('punkty', u_warstwa=1)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.K * self.n_vis)
        if self.n_luk:
            GL.glUseProgram(self._prog['linie'])
            for k, g in enumerate(np.linspace(1.0, max(1.0, float(p['grubosc_luk'])), 5)):
                self._uni_ustaw('linie', u_klasa=k)
                self._linie(3, 2 * SEG_LUKU * self.n_luk, g)
            GL.glUseProgram(self._prog['punkty'])
            self._uni_ustaw('punkty', u_warstwa=4)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.n_luk)

    def _linie(self, warstwa, n, szer):
        GL.glUniform1i(self._u('linie', 'u_warstwa'), warstwa)
        GL.glLineWidth(float(szer))
        GL.glDrawArrays(GL.GL_LINES, 0, int(n))

    # ------------------------------------------------------------ odczyt (testy i diagnostyka)
    def odczytaj(self, nazwa, n, dtype=np.float32, kolumny=1):
        """n elementów (po `kolumny` liczb) z bufora `nazwa`. Wołać przy aktywnym kontekście widoku."""
        out = np.empty((n, kolumny) if kolumny > 1 else n, dtype)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[nazwa])
        GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, out.ctypes.data_as(ctypes.c_void_p))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        return out

    def _program_tf(self):
        """Shader wierzchołków cząsteczek z przechwytywaniem wyjść (transform feedback): v_poz, v_kol, v_rozm."""
        if 'tf' not in self._prog:
            prog = GL.glCreateProgram()
            for zr, typ in ((VS_CZASTECZKI, GL.GL_VERTEX_SHADER), (FS_KOLOR, GL.GL_FRAGMENT_SHADER)):
                GL.glAttachShader(prog, compileShader(zr, typ))
            nazwy = [b'v_poz', b'v_kol', b'v_rozm']
            tab = (ctypes.c_char_p * len(nazwy))(*nazwy)
            GL.glTransformFeedbackVaryings(prog, len(nazwy), ctypes.cast(tab, ctypes.POINTER(ctypes.POINTER(GL.GLchar))),
                                           GL.GL_INTERLEAVED_ATTRIBS)
            GL.glLinkProgram(prog)
            if GL.glGetProgramiv(prog, GL.GL_LINK_STATUS) != GL.GL_TRUE:
                raise RuntimeError(GL.glGetProgramInfoLog(prog))
            self._prog['tf'] = prog
        return 'tf'

    def geometria(self, warstwa, zrodlo=-1):
        """Wierzchołki warstwy (0 wektory, 1 końce, 2 okręgi zera, 3 łuki, 4 groty) policzone tym samym shaderem
        co rysowanie, przechwycone na karcie: (położenia [n, 3], kolory [n, 4], rozmiary [n]). zrodlo=0: wektory
        PRZED na szaro (tryb nałożony). Do testów; wołać przy aktywnym kontekście, po narysowaniu klatki."""
        n = {0: 2 * self.K * self.n_vis, 1: self.K * self.n_vis, 2: 2 * SEG_ZERA * self.n_vis,
             3: 2 * SEG_LUKU * self.n_luk, 4: self.n_luk}[warstwa]
        out = np.zeros((max(n, 1), 8), np.float32)
        if n == 0:
            return out[:0, :3], out[:0, 3:7], out[:0, 7]
        prog = self._program_tf()
        for i, nazwa in enumerate(BUFORY):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, self._buf[nazwa])
        self._uniformy_warstw(prog, np.eye(4, dtype=np.float32))
        self._uni_ustaw(prog, u_warstwa=warstwa, u_klasa=-1, u_zrodlo=zrodlo,
                        u_kol=(.55, .55, .55, .5) if zrodlo == 0 else (0, 0, 0, -1))
        tf = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, tf)
        GL.glBufferData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, out.nbytes, None, GL.GL_STATIC_READ)
        GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, tf)
        GL.glEnable(GL.GL_RASTERIZER_DISCARD)
        GL.glBeginTransformFeedback(GL.GL_POINTS)
        GL.glDrawArrays(GL.GL_POINTS, 0, n)
        GL.glEndTransformFeedback()
        GL.glDisable(GL.GL_RASTERIZER_DISCARD)
        GL.glGetBufferSubData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, out.nbytes, out.ctypes.data_as(ctypes.c_void_p))
        GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0)
        GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0)
        GL.glDeleteBuffers(1, [tf])
        przywroc_stan()
        return out[:, :3], out[:, 3:7], out[:, 7]

    def odczytaj_podloze(self):
        """Tekstura płaszczyzny [320, 480, 4] uint8, wiersz 0 = dół obrazu (jak dane GLImageItem)."""
        out = np.empty((WYS, SZER, 4), np.uint8)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_pl)
        stare = GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, out.ctypes.data_as(ctypes.c_void_p))
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, int(stare))
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return out


class PlaszczyznaGPU(GLGraphicsItem):
    """Płaszczyzna obrazu z = 0 z teksturą policzoną przez CzasteczkiGPU: rysowana po liniach (wyższa wartość
    głębi), z testem głębi, bez zapisu do bufora głębi, jak ObrazPrzezroczysty (GLImageItem, filtr NEAREST)."""

    def __init__(self, zrodlo):
        super().__init__()
        self.zrodlo = zrodlo
        self.setGLOptions('translucent')
        self.setDepthValue(10)

    def paint(self):
        z = self.zrodlo
        if z._prog is None or z._tex_pl is None or z.p['alfa'] <= 0:
            return
        try:
            self.setupGLState()
            GL.glDepthMask(GL.GL_FALSE)
            GL.glUseProgram(z._prog['plaszczyzna'])
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(z._prog['plaszczyzna'], 'u_mvp'), 1, False,
                                  np.asarray(self.mvpMatrix().data(), np.float32))
            GL.glUniform1i(GL.glGetUniformLocation(z._prog['plaszczyzna'], 'u_tex'), 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, z._tex_pl)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        except Exception as e:
            z.blad = f'{type(e).__name__}: {e}'
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            przywroc_stan()


def przywroc_stan():
    """Stan GL jak przed rysowaniem (inne elementy pyqtgraph i napisy QPainter)."""
    GL.glUseProgram(0)
    GL.glLineWidth(1.0)
    GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
    for jednostka in (GL.GL_TEXTURE1, GL.GL_TEXTURE0):
        GL.glActiveTexture(jednostka)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    for i in range(len(BUFORY)):
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, 0)
