"""Przestrzeń makrowektorów liczona i rysowana na GPU (OpenGL 4.3: compute shadery + GLSL).

Położenia wszystkich kroków P [T, 600, 3] trafiają na kartę raz, po przeliczeniu sekwencji. Każda klatka animacji
to tylko kilka liczb (krok docelowy, postęp przelotu); resztę liczą shadery:

- kernel `przelot` (compute): położenie każdej cząstki w tej chwili, CUR = OD + (P[t] − OD) · u,
  z tym samym łagodnym postępem u = x²(3 − 2x) co dotąd na CPU;
- kernel `mikro` (compute, jedna grupa robocza): mikrowektory wybranej cząstki w czasie przelotu, reszta do
  lecącej cząstki rozłożona po równo, suma (wypadkowa) i łańcuch (sumy prefiksowe w ustalonej kolejności);
- shadery wierzchołków czytają te bufory: cząstki, ślady z P (bez listy odcinków na CPU), zdjęcie rozpięte na
  siatce cząstek (interpolacja dwuliniowa w shaderze), linie siatki, brzeg obrazu, strzałki atencji z grotami,
  mikrowektory, łańcuch, wypadkowa i wybrana cząstka.

CPU przygotowuje dane tylko przy zmianie kroku albo opcji (kolory 600 cząstek, lista widocznych, siły wybranej
cząstki), a w czasie przelotu nie liczy żadnych położeń. Wymaga OpenGL 4.3 (NVIDIA/AMD/Intel pod Windows).
"""
import ctypes

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

NT = 600
WIERSZE, KOLUMNY = 20, 30

_SSBO = """
layout(std430, binding = 0) readonly buffer BP { vec4 P[]; };        // [T * 600] położenia wszystkich kroków
layout(std430, binding = 1) buffer BCUR { vec4 CUR[]; };              // [600] położenia w tej chwili
layout(std430, binding = 2) readonly buffer BOD { vec4 OD[]; };       // [600] położenia na starcie przelotu
layout(std430, binding = 3) readonly buffer BKOL { vec4 KOL[]; };     // [600] kolory cząstek (RGBA)
layout(std430, binding = 4) readonly buffer BATR { vec4 ATR[]; };     // [600] rozmiar px, maska narzędzia
layout(std430, binding = 5) readonly buffer BVIS { uint VIS[]; };     // widoczne cząstki
layout(std430, binding = 6) readonly buffer BFV { vec4 FV[]; };       // [2K] mikrowektory: start, cel
layout(std430, binding = 7) buffer BMV { vec4 MV[]; };                // [K] bieżące, [K..2K] łańcuch (K + 1 punktów)
layout(std430, binding = 8) readonly buffer BKK { vec4 KK[]; };       // [K] kolory kanałów
layout(std430, binding = 9) readonly buffer BARR { vec4 ARR[]; };     // [E] strzałki atencji: źródło, cel, waga
layout(std430, binding = 10) readonly buffer BSEG { vec4 SEG[]; };    // brzeg: (token a, token b), kolor
layout(std430, binding = 11) readonly buffer BORD { uint ORD[]; };    // [K] kolejność łańcucha
const uint NT = 600u;
"""

CS_PRZELOT = "#version 430\n" + _SSBO + """
layout(local_size_x = 64) in;
uniform uint u_t;
uniform float u_s;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= NT) return;
    vec4 b = P[u_t * NT + i];
    if (u_s >= 1.0) { CUR[i] = vec4(b.xyz, 1.0); return; }
    vec3 a = OD[i].xyz;
    CUR[i] = vec4(a + (b.xyz - a) * u_s, 1.0);
}
"""

CS_MIKRO = "#version 430\n" + _SSBO + """
layout(local_size_x = 256) in;
uniform uint u_K, u_tok;
uniform float u_s;
uniform int u_reszta;
shared vec3 suma[256];
void main() {
    uint l = gl_LocalInvocationID.x;
    vec3 acc = vec3(0.0);
    for (uint k = l; k < u_K; k += 256u) {
        vec3 v0 = FV[k].xyz, v1 = FV[u_K + k].xyz;
        vec3 v = (u_s >= 1.0) ? v1 : v0 + (v1 - v0) * u_s;
        MV[k] = vec4(v, 0.0);
        acc += v;
    }
    suma[l] = acc;
    barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) suma[l] += suma[l + w];
        barrier();
    }
    if (u_reszta == 1) {                       // w czasie przelotu suma trafia dokładnie w lecącą cząstkę
        vec3 r = (CUR[u_tok].xyz - suma[0]) / float(u_K);
        for (uint k = l; k < u_K; k += 256u) MV[k].xyz += r;
    }
    memoryBarrierBuffer();
    barrier();
    if (l == 0u) {                             // łańcuch: wektory koniec do początku, od najsilniejszego
        vec3 c = vec3(0.0);
        MV[u_K] = vec4(c, 0.0);
        for (uint j = 0u; j < u_K; j++) {
            c += MV[ORD[j]].xyz;
            MV[u_K + 1u + j] = vec4(c, 0.0);
        }
    }
}
"""

_KOLOR_WEZLA = """
uniform sampler2D u_rgb;
uniform int u_kolor;                           // 0 zdjęcie, 1 kolory cząstek, 2 zdjęcie z narzędziem na czerwono
vec3 kolor_w(float rr, float cc) {
    int r0 = min(int(floor(rr)), 18), c0 = min(int(floor(cc)), 28);
    float fr = rr - float(r0), fc = cc - float(c0);
    uint a = uint(r0 * 30 + c0);
    if (u_kolor == 1) {
        vec3 g = mix(KOL[a].rgb, KOL[a + 1u].rgb, fc), d = mix(KOL[a + 30u].rgb, KOL[a + 31u].rgb, fc);
        return mix(g, d, fr);
    }
    int y = clamp(int(floor(8.0 + rr * 16.0 + 0.5)), 0, 319), x = clamp(int(floor(8.0 + cc * 16.0 + 0.5)), 0, 479);
    vec3 k = texelFetch(u_rgb, ivec2(x, y), 0).rgb;
    if (u_kolor == 2) {
        float m = mix(mix(ATR[a].y, ATR[a + 1u].y, fc), mix(ATR[a + 30u].y, ATR[a + 31u].y, fc), fr);
        k = k * (1.0 - 0.7 * m) + vec3(1.0, 0.15, 0.1) * 0.7 * m;
    }
    return k;
}
vec3 polozenie_w(float rr, float cc) {
    int r0 = min(int(floor(rr)), 18), c0 = min(int(floor(cc)), 28);
    float fr = rr - float(r0), fc = cc - float(c0);
    uint a = uint(r0 * 30 + c0);
    vec3 g = mix(CUR[a].xyz, CUR[a + 1u].xyz, fc), d = mix(CUR[a + 30u].xyz, CUR[a + 31u].xyz, fc);
    return mix(g, d, fr);
}
"""

VS_LINIE = "#version 430\n" + _SSBO + _KOLOR_WEZLA + """
uniform mat4 u_mvp;
uniform int u_tryb, u_L, u_klasa;
uniform uint u_t0, u_K;
uniform float u_mnoz, u_alfa;
out vec4 v_kol;
void main() {
    int v = gl_VertexID;
    vec3 p = vec3(0.0);
    vec4 kol = vec4(1.0);
    if (u_tryb == 0) {                          // ślady: odcinki między zapisanymi krokami t0 .. t
        uint tok = VIS[gl_InstanceID];
        int odc = v / 2;
        p = P[(u_t0 + uint(odc + v % 2)) * NT + tok].xyz;
        float a = u_L > 1 ? 0.08 + 0.52 * float(odc) / float(u_L - 1) : 0.08;
        kol = vec4(KOL[tok].rgb, a);
    } else if (u_tryb == 1) {                   // linie wierszy i kolumn siatki patchy
        int s = v / 2, r, c;
        if (s < 580) { r = s / 29; c = s % 29 + v % 2; }
        else { s -= 580; r = s / 30 + v % 2; c = s % 30; }
        p = CUR[uint(r * 30 + c)].xyz;
        kol = vec4(clamp(kolor_w(float(r), float(c)) * 1.3, 0.0, 1.0), u_alfa);
    } else if (u_tryb == 2) {                   // brzeg obrazu
        vec4 sg = SEG[(v / 2) * 2];
        p = CUR[uint(v % 2 == 0 ? sg.x : sg.y)].xyz;
        kol = SEG[(v / 2) * 2 + 1];
    } else if (u_tryb == 3) {                   // strzałki atencji źródło -> cel, w klasie grubości u_klasa
        vec4 s = ARR[v / 2];
        float w = clamp(s.z, 0.0, 1.0);
        if (min(int(w * 5.0), 4) != u_klasa) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_kol = vec4(0.0); return; }
        p = CUR[uint(v % 2 == 0 ? s.x : s.y)].xyz;
        kol = vec4(w, 1.0 - w, 1.0, 0.25 + 0.75 * w);
    } else if (u_tryb == 4) {                   // mikrowektory ze wspólnego środka
        uint k = uint(v / 2);
        p = v % 2 == 0 ? vec3(0.0) : MV[k].xyz * u_mnoz;
        kol = vec4(KK[k].rgb, 1.0);
    } else if (u_tryb == 5) {                   // łańcuch
        p = MV[u_K + uint(v)].xyz;
        kol = KK[ORD[v == 0 ? 0 : v - 1]];
    } else {                                    // wypadkowa = suma mikrowektorów
        p = v == 0 ? vec3(0.0) : MV[2u * u_K].xyz;
        kol = vec4(1.0, 1.0, 1.0, 0.95);
    }
    gl_Position = u_mvp * vec4(p, 1.0);
    v_kol = kol;
}
"""

VS_PUNKTY = "#version 430\n" + _SSBO + """
uniform mat4 u_mvp;
uniform int u_tryb;
uniform uint u_tok;
uniform vec4 u_kol;
uniform float u_rozm;
out vec4 v_kol;
void main() {
    vec3 p; vec4 c; float s;
    if (u_tryb == 0) { uint i = VIS[gl_VertexID]; p = CUR[i].xyz; c = KOL[i]; s = ATR[i].x; }
    else if (u_tryb == 1) { p = CUR[u_tok].xyz; c = u_kol; s = u_rozm; }
    else { vec4 a = ARR[gl_VertexID]; float w = clamp(a.z, 0.0, 1.0); p = CUR[uint(a.y)].xyz;
           c = vec4(w, 1.0 - w, 1.0, 0.25 + 0.75 * w); s = 7.0; }
    gl_Position = u_mvp * vec4(p, 1.0);
    gl_PointSize = s;
    v_kol = c;
}
"""

FS_PUNKTY = """#version 430
in vec4 v_kol;
out vec4 kolor;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    if (dot(q, q) > 1.0) discard;
    kolor = v_kol;
}
"""

FS_KOLOR = """#version 430
in vec4 v_kol;
out vec4 kolor;
void main() { kolor = v_kol; }
"""

VS_ZDJECIE = "#version 430\n" + _SSBO + _KOLOR_WEZLA + """
uniform mat4 u_mvp;
uniform int u_C, u_n;
uniform float u_alfa;
out vec4 v_kol;
void main() {
    float rr = float(gl_VertexID / u_C) / float(u_n), cc = float(gl_VertexID % u_C) / float(u_n);
    gl_Position = u_mvp * vec4(polozenie_w(rr, cc), 1.0);
    v_kol = vec4(kolor_w(rr, cc), u_alfa);
}
"""

BUFORY = ('P', 'CUR', 'OD', 'KOL', 'ATR', 'VIS', 'FV', 'MV', 'KK', 'ARR', 'SEG', 'ORD')


def _brzeg():
    """Odcinki brzegu siatki 20 x 30: góra niebieska, lewa żółta, dół i prawa szare (jak dotąd)."""
    G = np.arange(NT).reshape(WIERSZE, KOLUMNY)
    wynik = []
    for L, c in ((G[0], (.3, .6, 1, .9)), (G[:, 0], (1, .85, .2, .9)), (G[-1], (.6, .6, .6, .5)), (G[:, -1], (.6, .6, .6, .5))):
        for a, b in zip(L[:-1], L[1:]):
            wynik += [(a, b, 0, 0), c]
    return np.asarray(wynik, np.float32)


class MakroGPU(GLGraphicsItem):
    """Jeden element sceny pyqtgraph, który rysuje wszystkie ruchome warstwy przestrzeni makro z buforów GPU."""

    def __init__(self):
        super().__init__()
        self.setGLOptions('translucent')
        self._prog: dict | None = None
        self._buf: dict = {}
        self._rozm = {}
        self._dane: dict = {'SEG': _brzeg()}                 # bufory czekające na wysłanie przy najbliższym rysowaniu
        self._wiersze = []
        self._tekstura = None
        self._rgb = None
        self._indeksy = None
        self._siatka = None
        self.T = 0
        self.blad = ''
        self.renderer = ''
        # stan kroku (ustawiany przez ustaw_krok) i przelotu (ustaw_przelot)
        self.t, self.s, self.reszta = 0, 1.0, False
        self.n_wid, self.wybrany, self.rozm_wybr = 0, None, 16.0
        self.slad_L, self.slad_t0, self.pokaz_slady = 0, 0, False
        self.K, self.tryb_mikro, self.mnoz_mikro = 0, 3, 1.0
        self.n_strz, self.grubosc_max = 0, 6.0
        self.zdj_tryb, self.zdj_kolor, self.zdj_alfa, self.zdj_alfa_lin, self.zdj_n = 0, 0, .45, .7, 4
        self.pokaz_brzeg, self.pokaz_czastki = False, True

    # ------------------------------------------------------------ dane z CPU (bez kontekstu GL)
    def ustaw_P(self, P):
        """Położenia wszystkich kroków [T, 600, 3]; wysyłane na kartę raz."""
        P = np.asarray(P, np.float32)
        self.T = len(P)
        X = np.zeros((self.T * NT, 4), np.float32)
        X[:, :3] = P.reshape(-1, 3)
        self._dane['P'] = X
        self._wiersze = []
        self.update()

    def ustaw_wiersz(self, t, Pt):
        """Jeden krok P[t] [600, 3] (Grand Tour zmienia bieżący krok)."""
        X = np.zeros((NT, 4), np.float32); X[:, :3] = Pt
        self._wiersze.append((int(t), X))
        self.update()

    def ustaw_obraz(self, rgb):
        """Zdjęcie [320, 480, 3] uint8 do siatki zdjęcia; ten sam obiekt nie jest wysyłany ponownie."""
        if rgb is not self._rgb:
            self._rgb = rgb
            self._dane['RGB'] = None if rgb is None else np.ascontiguousarray(np.asarray(rgb)[:320, :480, :3], np.uint8)

    def ustaw_krok(self, t, od, kolory, rozmiary, maska, widoczne, wybrany=None, rozm_wybr=16.0,
                   slad=(0, 0, False), mikro=None, tryb_mikro=3, mnoz_mikro=1.0, kolory_kan=None,
                   strzalki=None, grubosc_max=6.0, zdjecie=(0, 0, .45, .7, 4), brzeg=False, czastki=True):
        """Wszystko, co stałe w czasie jednego przelotu (zmienia się tylko przy nowym kroku albo opcji).
        od [600, 3]: położenia na starcie przelotu; kolory [600, 4]; rozmiary [600]; maska [600] (0/1);
        widoczne: tokeny; slad = (L, t0, widoczny); mikro = (v_start [K, 3], v_cel [K, 3], kolejność [K]);
        strzalki = (src, dst, w); zdjecie = (tryb 0-3, kolor 0-2, krycie, krycie linii, gęstość)."""
        self.t = int(t)
        X = np.zeros((NT, 4), np.float32); X[:, :3] = od; self._dane['OD'] = X
        self._dane['KOL'] = np.ascontiguousarray(kolory, np.float32)
        A = np.zeros((NT, 4), np.float32); A[:, 0] = rozmiary; A[:, 1] = maska; self._dane['ATR'] = A
        wid = np.asarray(widoczne, np.uint32)
        self.n_wid = len(wid); self._dane['VIS'] = wid
        self.pokaz_czastki = bool(czastki)
        self.wybrany, self.rozm_wybr = (None if wybrany is None else int(wybrany)), float(rozm_wybr)
        self.slad_L, self.slad_t0, self.pokaz_slady = int(slad[0]), int(slad[1]), bool(slad[2])
        self.tryb_mikro, self.mnoz_mikro = int(tryb_mikro), float(mnoz_mikro)
        if mikro is not None and self.wybrany is not None:
            v0, v1, kolej = mikro
            self.K = len(v1)
            F = np.zeros((2 * self.K, 4), np.float32); F[:self.K, :3] = v0; F[self.K:, :3] = v1
            self._dane['FV'] = F
            self._dane['ORD'] = np.asarray(kolej, np.uint32)
            self._dane['KK'] = np.ascontiguousarray(kolory_kan, np.float32)
            if self._rozm.get('MV', 0) < (2 * self.K + 1) * 16:
                self._dane['MV'] = np.zeros((2 * self.K + 1, 4), np.float32)
        else:
            self.tryb_mikro = 3
        if strzalki is not None and len(strzalki[0]):
            src, dst, w = strzalki
            S = np.zeros((len(src), 4), np.float32); S[:, 0] = src; S[:, 1] = dst; S[:, 2] = w
            self._dane['ARR'] = S; self.n_strz = len(S)
        else:
            self.n_strz = 0
        self.grubosc_max = float(grubosc_max)
        self.zdj_tryb, self.zdj_kolor, self.zdj_alfa, self.zdj_alfa_lin, self.zdj_n = zdjecie
        self.pokaz_brzeg = bool(brzeg)
        self.update()

    def ustaw_przelot(self, s, reszta):
        """Postęp przelotu (już wygładzony) i czy mikrowektory mają dostać resztę do lecącej cząstki."""
        self.s, self.reszta = float(s), bool(reszta)
        self.update()

    # ------------------------------------------------------------ GL
    def _program(self, vs, fs=None, cs=None):
        if cs is not None:
            return compileProgram(compileShader(cs, GL.GL_COMPUTE_SHADER))
        return compileProgram(compileShader(vs, GL.GL_VERTEX_SHADER), compileShader(fs, GL.GL_FRAGMENT_SHADER),
                              validate=False)

    def _init(self):
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
        self._prog = dict(przelot=self._program(None, cs=CS_PRZELOT), mikro=self._program(None, cs=CS_MIKRO),
                          linie=self._program(VS_LINIE, FS_KOLOR), punkty=self._program(VS_PUNKTY, FS_PUNKTY),
                          zdjecie=self._program(VS_ZDJECIE, FS_KOLOR))
        self._uni = {}
        ids = GL.glGenBuffers(len(BUFORY))
        self._buf = dict(zip(BUFORY, (int(i) for i in np.atleast_1d(ids))))
        for n in BUFORY:                                  # każdy bufor istnieje od początku (min. 16 B)
            if n not in self._dane:
                self._dane[n] = np.zeros((1, 4), np.float32)
        self._dane['CUR'] = np.zeros((NT, 4), np.float32)
        self._dane['MV'] = self._dane.get('MV') if self._dane.get('MV') is not None and len(self._dane['MV']) > 1 \
            else np.zeros((2 * 768 + 1, 4), np.float32)

    def _u(self, prog, nazwa):
        k = (prog, nazwa)
        if k not in self._uni:
            self._uni[k] = GL.glGetUniformLocation(self._prog[prog], nazwa)
        return self._uni[k]

    def _wyslij(self):
        for n, a in list(self._dane.items()):
            if n == 'RGB':
                self._wyslij_obraz(a)
            else:
                a = np.ascontiguousarray(a)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[n])
                if self._rozm.get(n, -1) >= a.nbytes and a.nbytes > 0:
                    GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, a.nbytes, a)
                else:
                    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, max(a.nbytes, 16), a if a.nbytes else None, GL.GL_DYNAMIC_DRAW)
                    self._rozm[n] = max(a.nbytes, 16)
        self._dane.clear()
        if self._wiersze:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['P'])
            for t, X in self._wiersze:
                if 0 <= t < self.T:
                    GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, t * NT * 16, X.nbytes, X)
            self._wiersze = []
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    def _wyslij_obraz(self, rgb):
        if rgb is None:
            return
        if self._tekstura is None:
            self._tekstura = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tekstura)
        stare = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, rgb.shape[1], rgb.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, int(stare))           # QPainter (napisy) zakłada swoją wartość
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _indeksy_siatki(self, R, C):
        if self._siatka == (R, C):
            return
        i = np.arange(R * C).reshape(R, C)
        q = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1).reshape(-1, 4)
        tw = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.uint32)
        if self._indeksy is None:
            self._indeksy = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._indeksy)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, tw.nbytes, tw, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        self._siatka, self._n_ind = (R, C), tw.size

    def oblicz(self):
        """Kernele compute: położenia w tej chwili i mikrowektory. Wołać przy aktywnym kontekście."""
        for i, n in enumerate(BUFORY):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, self._buf[n])
        GL.glUseProgram(self._prog['przelot'])
        GL.glUniform1ui(self._u('przelot', 'u_t'), min(self.t, max(self.T - 1, 0)))
        GL.glUniform1f(self._u('przelot', 'u_s'), self.s)
        GL.glDispatchCompute((NT + 63) // 64, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        if self.wybrany is not None and self.tryb_mikro != 3 and self.K:
            GL.glUseProgram(self._prog['mikro'])
            GL.glUniform1ui(self._u('mikro', 'u_K'), self.K)
            GL.glUniform1ui(self._u('mikro', 'u_tok'), self.wybrany)
            GL.glUniform1f(self._u('mikro', 'u_s'), self.s)
            GL.glUniform1i(self._u('mikro', 'u_reszta'), int(self.reszta))
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)

    def paint(self):
        if self.T == 0 and 'P' not in self._dane:
            return
        try:
            if self._prog is None:
                self._init()
            self._wyslij()
            if self.T == 0:
                return
            self.setupGLState()
            self.oblicz()
            mvp = np.asarray(self.mvpMatrix().data(), np.float32)
            self._rysuj(mvp)
            self.blad = ''
        except Exception as e:                                   # błąd GL nie może zatrzymać całej aplikacji
            self.blad = f'{type(e).__name__}: {e}'
        finally:                                                 # stan GL jak przed rysowaniem: inne elementy i QPainter
            GL.glUseProgram(0)
            GL.glLineWidth(1.0)
            GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            for i in range(len(BUFORY)):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, 0)

    def _linie(self, tryb, n, instancje=1, szer=1.0, prymityw=GL.GL_LINES, **uni):
        p = self._prog['linie']
        GL.glUniform1i(self._u('linie', 'u_tryb'), tryb)
        for k, v in uni.items():
            (GL.glUniform1ui if k in ('u_t0', 'u_K') else GL.glUniform1i if k in ('u_L', 'u_klasa', 'u_kolor')
             else GL.glUniform1f)(self._u('linie', k), v)
        GL.glLineWidth(float(szer))
        GL.glDrawArraysInstanced(prymityw, 0, int(n), int(instancje))

    def _rysuj(self, mvp):
        # ta sama kolejność warstw co dotąd: ślady, łańcuch, mikrowektory, wypadkowa, strzałki, groty, cząstki,
        # wybrana cząstka, zdjęcie, linie zdjęcia, brzeg
        GL.glUseProgram(self._prog['linie'])
        GL.glUniformMatrix4fv(self._u('linie', 'u_mvp'), 1, False, mvp)
        GL.glUniform1ui(self._u('linie', 'u_K'), self.K)
        if self.pokaz_slady and self.pokaz_czastki and self.slad_L > 0 and self.n_wid:
            self._linie(0, 2 * self.slad_L, self.n_wid, 1.0, u_t0=self.slad_t0, u_L=self.slad_L)
        mik = self.wybrany is not None and self.K
        if mik and self.tryb_mikro in (1, 2):
            self._linie(5, self.K + 1, 1, 2.0, prymityw=GL.GL_LINE_STRIP)
        if mik and self.tryb_mikro in (0, 2):
            self._linie(4, 2 * self.K, 1, 1.5, u_mnoz=self.mnoz_mikro)
            self._linie(6, 2, 1, 4.0)
        if self.n_strz:
            for k, g in enumerate(np.linspace(1.0, max(1.0, self.grubosc_max), 5)):
                self._linie(3, 2 * self.n_strz, 1, g, u_klasa=k)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_POINT_SPRITE)
        GL.glUseProgram(self._prog['punkty'])
        GL.glUniformMatrix4fv(self._u('punkty', 'u_mvp'), 1, False, mvp)
        if self.n_strz:
            GL.glUniform1i(self._u('punkty', 'u_tryb'), 2); GL.glDrawArrays(GL.GL_POINTS, 0, self.n_strz)
        if self.pokaz_czastki and self.n_wid:
            GL.glUniform1i(self._u('punkty', 'u_tryb'), 0); GL.glDrawArrays(GL.GL_POINTS, 0, self.n_wid)
        if self.wybrany is not None:
            GL.glUniform1i(self._u('punkty', 'u_tryb'), 1)
            GL.glUniform1ui(self._u('punkty', 'u_tok'), self.wybrany)
            GL.glUniform4f(self._u('punkty', 'u_kol'), 1, 1, 1, .9)
            GL.glUniform1f(self._u('punkty', 'u_rozm'), self.rozm_wybr)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)
        if self.zdj_tryb:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tekstura or 0)
        if self.zdj_tryb in (1, 3):
            n = int(self.zdj_n); R, C = 19 * n + 1, 29 * n + 1
            self._indeksy_siatki(R, C)
            GL.glUseProgram(self._prog['zdjecie'])
            GL.glUniformMatrix4fv(self._u('zdjecie', 'u_mvp'), 1, False, mvp)
            GL.glUniform1i(self._u('zdjecie', 'u_C'), C); GL.glUniform1i(self._u('zdjecie', 'u_n'), n)
            GL.glUniform1i(self._u('zdjecie', 'u_kolor'), int(self.zdj_kolor))
            GL.glUniform1i(self._u('zdjecie', 'u_rgb'), 0)
            GL.glUniform1f(self._u('zdjecie', 'u_alfa'), float(self.zdj_alfa))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._indeksy)
            GL.glDrawElements(GL.GL_TRIANGLES, self._n_ind, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glUseProgram(self._prog['linie'])
        if self.zdj_tryb in (2, 3):
            GL.glUniform1i(self._u('linie', 'u_rgb'), 0)
            self._linie(1, 2 * (WIERSZE * (KOLUMNY - 1) + (WIERSZE - 1) * KOLUMNY), 1, 1.0,
                        u_kolor=int(self.zdj_kolor), u_alfa=float(self.zdj_alfa_lin))
        if self.zdj_tryb:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        if self.pokaz_brzeg:
            self._linie(2, 2 * (2 * (KOLUMNY - 1) + 2 * (WIERSZE - 1)), 1, 2.5)
        GL.glLineWidth(1.0)

    # ------------------------------------------------------------ odczyt (testy i diagnostyka)
    def odczytaj(self, nazwa, n):
        """n wektorów vec4 z bufora `nazwa` (CUR, MV, ...) -> [n, 4]. Wołać przy aktywnym kontekście widoku."""
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[nazwa])
        out = np.empty((n, 4), np.float32)
        GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, out.ctypes.data_as(ctypes.c_void_p))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        return out
