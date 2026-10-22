"""Elementy OpenGL 4.3 zakładki „Lasery i koloidy” (pyqtgraph GLGraphicsItem + shadery GLSL 4.30 z buforami SSBO).

Cała historia (wszystkie checkpointy: położenia, kierunki, mapy) trafia na kartę RAZ, po wczytaniu. W klatce CPU podaje tylko
ułamkowy indeks czasu u_t i opcje; interpolację między checkpointami, kolory, rozmiary, punkty załamania na sferze i mapy na zdjęciu
liczą shadery. Jedyne dane wysyłane w trakcie odtwarzania to natężenia z_jc (600 × 768) pary sąsiednich checkpointów dla wiązek,
wymieniane, gdy oś czasu przejdzie do następnej pary.

- `RuchPlaski`: panele 1–3 filmu 2/16 (ruch globalny, studnie i odkształcenie lokalne, drgania wokół centrów ze śladami);
- `SferaReflektorow`: film 12 (768 stałych reflektorów, 600 wiązek każdy, jedno załamanie na sferze, cząsteczki p_j = P W₂ (z − z̄));
- `MapyNaZdjeciu`: kafelki zdjęcia z mapą 30 × 20 (atencja 6 bloków, centra, drgania, natężenie reflektora) i ramkami.
"""
import ctypes

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

TOK, NK, H, W = 600, 768, 20, 30
LUTY = ('plasma', 'magma', 'inferno', 'viridis')
NM = 9                                 # mapy: 0–5 atencja bloków, 6 średnia atencja, 7 centra (lok), 8 drgania (vib); 9 = reflektor (osobno)

_CZAS = """
uniform float u_t;
uniform int u_K;
void para(out uint a, out uint b, out float f) {
    float tt = clamp(u_t, 0.0, float(u_K - 1));
    a = uint(floor(tt)); b = min(a + 1u, uint(u_K - 1)); f = tt - float(a);
}
"""
_LUT = """
uniform sampler2D u_lut;
vec3 lut(int k, float v) { return texture(u_lut, vec2(0.002 + 0.996 * clamp(v, 0.0, 1.0), (float(k) + 0.5) / 4.0)).rgb; }
"""
_KSZTALT = """
uniform int u_ksztalt;              // 0 poświata studni, 1 pierścień, 2 dysk z jasną obwódką, 3 miękka kropka, 4 kwadrat
vec4 ksztalt(vec4 k) {
    vec2 d = gl_PointCoord * 2.0 - 1.0; float r = length(d);
    if (u_ksztalt == 4) return (max(abs(d.x), abs(d.y)) < 0.85) ? k : vec4(0.0);
    if (r > 1.0) discard;
    if (u_ksztalt == 0) {           // jak tonowanie filmu: 0,55·I zielony + 0,25·I³ białozielony (I = gauss)
        float I = k.a * exp(-r * r * 4.5);
        return vec4(0.55 * I * vec3(0.22, 1.0, 0.48) + 0.25 * I * I * I * vec3(0.75, 1.0, 0.85), 1.0);
    }
    if (u_ksztalt == 1) { float a = smoothstep(0.6, 0.78, r) * (1.0 - smoothstep(0.86, 1.0, r)); return vec4(k.rgb, k.a * a); }
    if (u_ksztalt == 2) { vec3 c = r > 0.72 ? vec3(1.0) : k.rgb; return vec4(c, k.a * (1.0 - smoothstep(0.9, 1.0, r))); }
    return vec4(k.rgb, k.a * exp(-r * r * 3.0));
}
"""


def _program(vs, fs):
    return compileProgram(compileShader(vs, GL.GL_VERTEX_SHADER), compileShader(fs, GL.GL_FRAGMENT_SHADER), validate=False)


def lut_rgb():
    import matplotlib
    return np.stack([(matplotlib.colormaps[n](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8) for n in LUTY])  # (4, 256, 3)


class _ElementGPU(GLGraphicsItem):
    """Wspólne: bufory SSBO o stałych numerach wiązań (BUFORY), tekstury (LUT, zdjęcie), kompilacja przy pierwszym rysowaniu,
    przywracanie stanu GL po rysowaniu (inne elementy pyqtgraph i napisy QPainter)."""
    BUFORY = ()

    def __init__(self):
        super().__init__()
        self._prog = None
        self._buf = {}
        self._rozm = {}
        self._dane = {}
        self._tex = {}
        self._tex_dane = {}
        self._uni = {}
        self.K = 0
        self.t = 0.0
        self.blad = ''
        self._tex_dane['lut'] = lut_rgb().reshape(4, 256, 3)

    def ustaw_bufor(self, nazwa, a):
        self._dane[nazwa] = np.ascontiguousarray(a)
        self.update()

    def ustaw_teksture(self, nazwa, rgb):
        self._tex_dane[nazwa] = None if rgb is None else np.ascontiguousarray(rgb, np.uint8)
        self.update()

    def ustaw_czas(self, t):
        self.t = float(t)
        self.update()

    def _programy(self):
        raise NotImplementedError

    def _u(self, prog, nazwa):
        k = (prog, nazwa)
        if k not in self._uni:
            self._uni[k] = GL.glGetUniformLocation(self._prog[prog], nazwa)
        return self._uni[k]

    def _ui(self, prog, **kw):
        for k, v in kw.items():
            loc = self._u(prog, k)
            if loc < 0:
                continue
            if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 4:
                GL.glUniform4f(loc, *[float(x) for x in v])
            elif isinstance(v, (tuple, list, np.ndarray)) and len(v) == 16:
                GL.glUniformMatrix4fv(loc, 1, False, np.asarray(v, np.float32))
            elif isinstance(v, (bool, int, np.integer)):
                GL.glUniform1i(loc, int(v))
            else:
                GL.glUniform1f(loc, float(v))

    def _wyslij(self):
        for n, a in list(self._dane.items()):
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[n])
            if self._rozm.get(n, -1) >= a.nbytes and a.nbytes > 0:
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, a.nbytes, a)
            else:
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, max(a.nbytes, 16), a if a.nbytes else None, GL.GL_DYNAMIC_DRAW)
                self._rozm[n] = max(a.nbytes, 16)
        self._dane.clear()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        for n, a in list(self._tex_dane.items()):
            if a is None:
                continue
            if n not in self._tex:
                self._tex[n] = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex[n])
            stare = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, a.shape[1], a.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, a)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, int(stare))
            filtr = GL.GL_LINEAR
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filtr)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filtr)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._tex_dane.clear()

    def paint(self):
        if self.K < 2:
            return
        try:
            if self._prog is None:
                self._prog = self._programy()
                ids = np.atleast_1d(GL.glGenBuffers(len(self.BUFORY)))
                self._buf = {n: int(i) for n, i in zip(self.BUFORY, ids)}
                for n in self.BUFORY:
                    self._dane.setdefault(n, np.zeros((1, 4), np.float32))
            self._wyslij()
            for i, n in enumerate(self.BUFORY):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, self._buf[n])
            GL.glActiveTexture(GL.GL_TEXTURE1); GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex.get('lut', 0))
            GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex.get('rgb', 0))
            GL.glDisable(GL.GL_DEPTH_TEST); GL.glEnable(GL.GL_BLEND)
            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE); GL.glEnable(GL.GL_POINT_SPRITE)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            self._rysuj(np.asarray(self.mvpMatrix().data(), np.float32))
            self.blad = ''
        except Exception as e:                                          # błąd GL nie zatrzymuje aplikacji
            self.blad = f'{type(e).__name__}: {e}'
        finally:
            GL.glUseProgram(0)
            GL.glLineWidth(1.0)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glActiveTexture(GL.GL_TEXTURE1); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            for i in range(len(self.BUFORY)):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, 0)

    @staticmethod
    def _addytywnie(tak):
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE) if tak else GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)


# ====================================================================== panele 1–3: rzut płaski
_SSBO_2D = """
layout(std430, binding = 0) readonly buffer BG { vec4 G[]; };     // [600] węzeł idealny (x, y), maska narzędzia, 0
layout(std430, binding = 1) readonly buffer BA { vec4 A[]; };     // [K*600] węzeł po Prokrustesie (xy), atom po Prokrustesie (zw)
layout(std430, binding = 2) readonly buffer BB { vec4 B[]; };     // [K*600] węzeł lab (xy), atom lab (zw)
layout(std430, binding = 3) readonly buffer BC { vec4 C[]; };     // [K*600] siatka w układzie inercjalnym (xy), lok, vib
"""
VS_2D = "#version 430\n" + _SSBO_2D + _CZAS + _LUT + """
uniform mat4 u_mvp;
uniform int u_tryb, u_zbior, u_kolor, u_L, u_tok;
uniform float u_rozm, u_lok99, u_vib97, u_jas, u_strz;
uniform vec4 u_kol, u_box;
out vec4 v_k;
vec4 iA(uint i) { uint a, b; float f; para(a, b, f); return mix(A[a * 600u + i], A[b * 600u + i], f); }
vec4 iB(uint i) { uint a, b; float f; para(a, b, f); return mix(B[a * 600u + i], B[b * 600u + i], f); }
vec4 iC(uint i) { uint a, b; float f; para(a, b, f); return mix(C[a * 600u + i], C[b * 600u + i], f); }
vec4 scena(vec2 p) { return u_mvp * vec4(p.x, -p.y, 0.0, 1.0); }   // y obrazu rośnie w dół
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
void main() {
    int v = gl_VertexID;
    v_k = u_kol;
    if (u_tryb == 0) {                       // punkty: 0 węzły (studnie), 1 atomy, 2 węzły lab, 3 atomy lab, 4 siatka idealna
        uint i = uint(v);
        vec2 p = u_zbior == 0 ? iA(i).xy : u_zbior == 1 ? iA(i).zw : u_zbior == 2 ? iB(i).xy : u_zbior == 3 ? iB(i).zw : G[i].xy;
        if (u_kolor == 1) v_k = vec4(lut(0, iC(i).w / u_vib97), u_kol.a);
        if (u_kolor == 2 && G[i].z < 0.5) { gl_Position = POZA; return; }                 // tylko narzędzie
        if (u_kolor == 3 && int(i) != u_tok) { gl_Position = POZA; return; }              // tylko wybrany
        if (u_zbior == 0 && u_kolor == 0) v_k.a = u_jas;
        gl_Position = scena(p); gl_PointSize = u_rozm;
    } else if (u_tryb == 1) {                // strzałki adres → centrum, kolor = długość
        uint i = uint(v / 2);
        vec4 c = iC(i);
        vec2 g = G[i].xy, q = iA(i).xy;
        vec2 p = (v % 2 == 0) ? g : g + (q - g) * u_strz;
        v_k = vec4(lut(0, c.z / u_lok99), (v % 2 == 0) ? 0.35 : 0.95);
        gl_Position = scena(p);
    } else if (u_tryb == 2) {                // szprychy atom → centrum
        uint i = uint(v / 2); vec4 a = iA(i);
        gl_Position = scena((v % 2 == 0) ? a.zw : a.xy);
    } else if (u_tryb == 3) {                // ślady atomów: L odcinków z zapisanych checkpointów, koniec w bieżącej chwili
        uint i = uint(gl_InstanceID);
        uint a, b; float f; para(a, b, f);
        int k = v / 2 + v % 2;               // punkt 0..L
        int ti = int(a) - u_L + k;
        vec2 p = (k == u_L) ? iA(i).zw : A[uint(max(ti, 0)) * 600u + i].zw;
        if (ti < 0) { gl_Position = POZA; return; }
        float al = 0.05 + 0.65 * float(k) / float(max(u_L, 1));
        v_k = vec4(1.0, 0.82, 0.4, al * u_kol.a);
        gl_Position = scena(p);
    } else if (u_tryb == 4) {                // ramka siatki w układzie inercjalnym (rogi tokenów 0, 29, 599, 570)
        uint r[5] = uint[5](0u, 29u, 599u, 570u, 0u);
        gl_Position = scena(iC(r[v]).xy);
    } else if (u_tryb == 5) {                // ramka siatki idealnej
        uint r[5] = uint[5](0u, 29u, 599u, 570u, 0u);
        gl_Position = scena(G[r[v]].xy);
    } else if (u_tryb == 6) {                // prostokąt u_box = (x0, y0, x1, y1) w komórkach
        vec2 r[5] = vec2[5](u_box.xy, vec2(u_box.z, u_box.y), u_box.zw, vec2(u_box.x, u_box.w), u_box.xy);
        gl_Position = scena(r[v]);
    } else if (u_tryb == 8) {                // podkroki: wypadkowe przesunięcie tokenu w podkroku (stan a → a+1), skala u_strz
        uint i = uint(v / 2); uint a, b; float f; para(a, b, f);
        vec2 p0 = A[a * 600u + i].xy, p1 = A[b * 600u + i].xy;
        v_k = vec4(u_kol.rgb, (v % 2 == 0) ? 0.15 : u_kol.a);
        gl_Position = scena((v % 2 == 0) ? p0 : p0 + (p1 - p0) * u_strz);
    } else if (u_tryb == 7) {                // strzałka ruchu globalnego: środek siatki idealnej → środek siatki inercjalnej
        vec2 c = 0.25 * (iC(0u).xy + iC(29u).xy + iC(599u).xy + iC(570u).xy);
        gl_Position = scena(v == 0 ? vec2(0.0) : c);
    }
}
"""
FS_2D = "#version 430\n" + _KSZTALT + """
uniform int u_punkty;
in vec4 v_k; out vec4 o;
void main() { o = u_punkty == 1 ? ksztalt(v_k) : v_k; }
"""


class RuchPlaski(_ElementGPU):
    """Rzut płaski 600 tokenów (x, y) w trzech trybach panelu: 'globalny', 'lokalny', 'drgania'."""
    BUFORY = ('G', 'A', 'B', 'C')

    def __init__(self, tryb='lokalny'):
        super().__init__()
        self.tryb = tryb
        self.opcje = dict(rozm=4.0, pxc=20.0, jas=1.0, sig=0.26, strz=1.0, L=24, tok=-1, box=None, narz=True)
        self.lok99, self.vib97 = 1.0, 1.0

    def ustaw_historie(self, h):
        K = h.K
        G = np.zeros((TOK, 4), np.float32); G[:, :2] = h.g; G[:, 2] = h.narz
        A = np.concatenate([h.qc, h.ac], 2).reshape(K * TOK, 4)
        B = np.concatenate([h.wezly[..., :2], h.atomy[..., :2]], 2).reshape(K * TOK, 4)
        C = np.concatenate([h.glb, h.lok[..., None], h.vib[..., None]], 2).reshape(K * TOK, 4)
        for n, a in zip(self.BUFORY, (G, A, B, C)):
            self.ustaw_bufor(n, a.astype(np.float32))
        self.K = K
        self.lok99, self.vib97 = max(h.LOK99, 1e-6), max(h.VIB97, 1e-6)

    def _programy(self):
        return dict(p=_program(VS_2D, FS_2D))

    def _pkt(self, zbior, kolor, rozm, ksztalt, kol, add=False):
        self._ui('p', u_tryb=0, u_zbior=zbior, u_kolor=kolor, u_rozm=float(rozm), u_ksztalt=ksztalt, u_kol=kol, u_punkty=1)
        self._addytywnie(add)
        GL.glDrawArrays(GL.GL_POINTS, 0, TOK)

    def _lin(self, tryb, n, szer, kol=(1, 1, 1, 1), instancje=1, prym=GL.GL_LINES):
        self._ui('p', u_tryb=tryb, u_kol=kol, u_punkty=0)
        self._addytywnie(False)
        GL.glLineWidth(float(szer))
        GL.glDrawArraysInstanced(prym, 0, n, instancje)

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['p'])
        GL.glActiveTexture(GL.GL_TEXTURE1)
        self._ui('p', u_mvp=mvp, u_t=self.t, u_K=self.K, u_lut=1, u_lok99=self.lok99, u_vib97=self.vib97, u_jas=float(o['jas']),
                 u_strz=float(o['strz']), u_tok=int(o['tok']), u_L=int(o['L']))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        pxc = float(o['pxc'])
        studnia = max(2.0, 6.0 * o['sig'] * pxc)                     # średnica poświaty = 6 σ
        tok = max(1.5, o['rozm'])
        if self.tryb == 'globalny':
            self._lin(5, 5, 1.2, (0.33, 0.38, 0.42, 0.9), prym=GL.GL_LINE_STRIP)
            self._lin(4, 5, 2.2, (0.22, 1.0, 0.48, 1.0), prym=GL.GL_LINE_STRIP)
            self._pkt(2, 0, tok, 3, (0.22, 1.0, 0.48, 0.5))
            self._pkt(3, 0, tok * 0.8, 3, (1.0, 1.0, 1.0, 0.85))
            self._lin(7, 2, 2.5, (1.0, 0.82, 0.4, 1.0))
        elif self.tryb == 'lokalny':
            self._pkt(0, 0, studnia, 0, (0, 0, 0, 1), add=True)
            self._pkt(4, 0, max(1.5, tok * 0.6), 3, (0.33, 0.38, 0.42, 0.9))
            self._lin(1, 2 * TOK, 1.4)
            if o['narz']:
                self._pkt(0, 2, max(6.0, pxc * 0.9), 1, (0.29, 0.94, 1.0, 1.0))
            if o['box'] is not None:
                self._ui('p', u_box=o['box'])
                self._lin(6, 5, 1.0, (1, 1, 1, 0.8), prym=GL.GL_LINE_STRIP)
        else:                                                           # drgania
            self._pkt(0, 0, studnia, 0, (0, 0, 0, 1), add=True)
            if o['L'] > 0:
                self._lin(3, 2 * int(o['L']), 1.3, (1, 1, 1, 1), instancje=TOK)
            self._lin(2, 2 * TOK, 0.8, (1, 1, 1, 0.45))
            self._pkt(1, 1, max(4.0, pxc * 0.32), 2, (1, 1, 1, 1))
            if o['narz']:
                self._pkt(1, 2, max(8.0, pxc * 0.6), 1, (0.29, 0.94, 1.0, 1.0))
        if o.get('pod_kol') is not None and self.tryb != 'globalny':   # podkroki: wypadkowe przesunięcia (uwaga / MLP)
            self._ui('p', u_strz=float(o.get('pod_skala', 1.0)))
            self._lin(8, 2 * TOK, 1.6, (*o['pod_kol'], 0.95))
            self._ui('p', u_strz=float(o['strz']))
        if o['tok'] >= 0:
            self._pkt(1 if self.tryb == 'drgania' else 0 if self.tryb == 'lokalny' else 3, 3, max(14.0, pxc * 0.8), 1,
                      (1.0, 0.85, 0.2, 1.0))


# ====================================================================== sfera reflektorów (film 12)
_SSBO_S = """
layout(std430, binding = 0) readonly buffer BR { vec4 RF[]; };    // [768] stały reflektor (miejsce na sferze zewnętrznej)
layout(std430, binding = 1) readonly buffer BK { vec4 KC[]; };    // [768] kolor reflektora
layout(std430, binding = 2) readonly buffer BP { vec4 P[]; };     // [K*600] cząsteczka p_j (skala), w = narzędzie
layout(std430, binding = 3) readonly buffer BD { vec4 D[]; };     // [K*768] kierunek wiązki w sferze d_c
layout(std430, binding = 4) readonly buffer BI { float I[]; };    // [2*600*768] natężenia z_jc pary checkpointów (a, a+1)
layout(std430, binding = 5) readonly buffer BS { vec4 SF[]; };    // siatka sfery (pary punktów)
layout(std430, binding = 6) readonly buffer BT { float AT[]; };   // [600*600] podkroki: atencja bloku (średnia głów), wiersz = kto słucha
layout(std430, binding = 7) readonly buffer BC { vec4 C[]; };     // [K*600] pułapka koloidu: średnie położenie z ±okna kroków (ta sama skala)
layout(std430, binding = 8) readonly buffer BKP { vec4 KP[]; };   // [600] kolor ścieżki koloidu
layout(std430, binding = 9) readonly buffer BKR { float KR[]; };  // [K] krok uczenia checkpointu (zanik historii)
layout(std430, binding = 10) readonly buffer BOD { vec4 OD[]; };  // [K*768] obrót igły, dryf netto (≥ 2000 kroków): xyz kierunek, w °/100 kroków
layout(std430, binding = 11) readonly buffer BOJ { vec4 OJ[]; };  // [K*768] obrót igły, drganie (≥ 100 kroków)
layout(std430, binding = 12) readonly buffer BW { float WY[]; };  // wybór: [768] maska reflektorów, [600] maska cząsteczek, listy indeksów
bool wyb_r(uint c) { return WY[c] > 0.5; }
bool wyb_t(uint j) { return WY[768u + j] > 0.5; }
uint lista_r(uint k) { return uint(WY[1368u + k]); }               // k-ty wybrany reflektor
uint lista_t(uint k) { return uint(WY[2136u + k]); }               // k-ta wybrana cząsteczka
"""
VS_S = "#version 430\n" + _SSBO_S + _CZAS + _LUT + """
uniform mat4 u_mvp;
uniform int u_tryb, u_wiazki, u_seg, u_tok, u_refl, u_blokD, u_cel, u_filtr, u_kol_sciezki;
uniform float u_zanik, u_dnorm, u_skala;   // u_skala: mnożnik rozmiaru chmury koloidów i pułapek w sferze            // u_dnorm: odpływ od adresu dający pełny kolor (99. percentyl)
uniform float u_R, u_gain, u_prog, u_rozm, u_pu;
uniform float u_neg;                      // wzmocnienie ujemnej jasności (|z| ≤ 0,17); 1 = prawdziwa proporcja
uniform float u_zgas;                     // płynne przejście podkroków: 0 pełna jasność wiązek / linii uwagi, 1 zgaszone
uniform float u_olo, u_ohi, u_ola, u_oigla;   // strzałki obrotu: skala log10 prędkości (dół, góra), długość maks., długość igły
uniform vec4 u_kol;
out vec4 v_k; out float v_t;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
vec3 pj(uint j) { uint a, b; float f; para(a, b, f); return u_skala * mix(P[a * 600u + j].xyz, P[b * 600u + j].xyz, f); }
vec3 dc(uint c) {                         // podkroki: kierunki bloku u_blokD (bez interpolacji między blokami)
    if (u_blokD >= 0) return normalize(D[uint(u_blokD) * 768u + c].xyz);
    uint a, b; float f; para(a, b, f); return normalize(mix(D[a * 768u + c].xyz, D[b * 768u + c].xyz, f));
}
float nat(uint j, uint c) { return mix(I[j * 768u + c], I[460800u + j * 768u + c], u_pu); }
vec3 w_chwili(uint j, float tt) {          // koloid (u_cel 0) albo pułapka (1) w ułamkowej chwili tt
    float t2 = clamp(tt, 0.0, float(u_K - 1)); uint a = uint(floor(t2)); uint b = min(a + 1u, uint(u_K - 1)); float f = t2 - float(a);
    return u_skala * (u_cel == 0 ? mix(P[a * 600u + j].xyz, P[b * 600u + j].xyz, f) : mix(C[a * 600u + j].xyz, C[b * 600u + j].xyz, f));
}
bool odrzuc(uint j) { return (u_filtr == 1 && P[j].w < 0.5) || (u_filtr == 2 && !wyb_t(j)); }
vec3 kolor_drogi(uint j, vec3 p, float tt) {  // u_kol_sciezki: 0 jednolity, 1 kolor cząsteczki, 2 odpływ od adresu (plasma)
    if (u_kol_sciezki == 1) return KP[j].rgb;
    if (u_kol_sciezki == 2) return lut(0, length(p - w_chwili(j, 0.0)) / max(u_dnorm * u_skala, 1e-9));
    return u_kol.rgb;
}
void main() {
    int v = gl_VertexID;
    v_k = u_kol; v_t = 0.0;
    if (u_tryb == 0) {                        // wiązki: odcinek reflektor → punkt załamania (u_seg 0) albo punkt → cząsteczka (1)
        int id = gl_InstanceID; uint j, c;
        if (u_wiazki == 2) { j = lista_t(uint(id / 768)); c = uint(id % 768); }        // do wybranych cząsteczek
        else if (u_wiazki == 3) { c = lista_r(uint(id / 600)); j = uint(id % 600); }   // z wybranych reflektorów
        else { c = uint(id / 600); j = uint(id % 600); }
        float zj = nat(j, c);                  // ze znakiem: ujemna jasność = ten sam laser słabo ciągnie (−v_c)
        float x = abs(zj) * u_gain * (zj < 0.0 ? u_neg : 1.0);
        if (x < u_prog) { gl_Position = POZA; return; }
        vec3 p = pj(j), d = dc(c);
        float bb = dot(p, d); float s = bb + sqrt(max(bb * bb - dot(p, p) + u_R * u_R, 0.0));
        vec3 e = p - s * d;
        vec3 q = (u_seg == 0) ? (v == 0 ? RF[c].xyz : e) : (v == 0 ? e : p);
        v_t = float(v);
        float r = u_seg == 0 ? (0.6 + 0.4 * v_t) : (0.25 + 0.75 * v_t * v_t);
        v_k = vec4((zj < 0.0 ? vec3(0.30, 0.85, 1.0) : 0.25 + 0.75 * KC[c].rgb) * min(x, 4.0) * r * (1.0 - u_zgas), 1.0);
        gl_Position = u_mvp * vec4(q, 1.0);
    } else if (u_tryb == 1) {                 // cząsteczki (0 wszystkie, 1 narzędzie, 2 wybrana)
        uint j = uint(v);
        if (u_wiazki == 1 && P[j].w < 0.5) { gl_Position = POZA; return; }
        if (u_wiazki == 2 && !wyb_t(j)) { gl_Position = POZA; return; }
        gl_Position = u_mvp * vec4(pj(j), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 2) {                 // reflektory (0 wszystkie, 2 wybrany)
        uint c = uint(v);
        if (u_wiazki == 2 && !wyb_r(c)) { gl_Position = POZA; return; }
        v_k = vec4(0.25 + 0.75 * KC[c].rgb, u_kol.a);
        gl_Position = u_mvp * vec4(RF[c].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 3) {                 // siatka szklanej sfery
        gl_Position = u_mvp * vec4(SF[v].xyz, 1.0);
    } else if (u_tryb == 4) {                 // kierunek wybranych reflektorów (instancja = reflektor z listy): średnica sfery wzdłuż d_c
        uint c = lista_r(uint(gl_InstanceID)); vec3 d = dc(c);
        gl_Position = u_mvp * vec4((v == 0 ? -u_R : u_R) * d, 1.0);
        v_k = vec4(0.25 + 0.75 * KC[c].rgb, 0.9);
    } else if (u_tryb == 5) {                 // podkrok uwagi: siły między cząsteczkami jako linie i – j (jasność = atencja)
        uint i = uint(gl_InstanceID / 600), j = uint(gl_InstanceID % 600);
        if (u_wiazki == 2) { i = lista_t(uint(gl_InstanceID / 600)); j = uint(gl_InstanceID % 600); }
        float a = AT[i * 600u + j] * u_gain;
        if (i == j || a < u_prog) { gl_Position = POZA; return; }
        v_k = vec4(vec3(0.35, 0.8, 1.0) * min(a, 4.0) * (1.0 - u_zgas), 1.0);
        gl_Position = u_mvp * vec4(pj(v == 0 ? i : j), 1.0);
    } else if (u_tryb == 8) {                 // pułapki koloidów (poświata studni), u_filtr jak wektory
        uint j = uint(v); if (odrzuc(j)) { gl_Position = POZA; return; }
        gl_Position = u_mvp * vec4(w_chwili(j, u_t), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 9) {                 // wektory: u_seg 1 = z poprzedniego checkpointu, 2 = od kroku 0 (adres startu); skala u_gain
        uint j = uint(gl_InstanceID); if (odrzuc(j)) { gl_Position = POZA; return; }
        vec3 p1 = w_chwili(j, u_t), p0 = w_chwili(j, u_seg == 2 ? 0.0 : u_t - 1.0);
        v_k = vec4(kolor_drogi(j, p1, u_t), u_kol.a * (v == 0 ? 0.2 : 1.0));
        gl_Position = u_mvp * vec4(v == 0 ? p0 : p0 + (p1 - p0) * u_gain, 1.0);
    } else if (u_tryb == 10) {                // cała historia: droga od checkpointu 0 do teraz (instancja = cząsteczka, wierzchołek = checkpoint)
        uint j = uint(gl_InstanceID); if (odrzuc(j)) { gl_Position = POZA; return; }
        float tt = min(float(v), u_t);
        float al;
        if (u_zanik > 0.0) {                   // zanik: odcinek sprzed u_zanik kroków prawie znika (exp(−3·wiek/u_zanik))
            float k1 = clamp(u_t, 0.0, float(u_K - 1)); uint a1 = uint(floor(k1));
            float teraz = mix(KR[a1], KR[min(a1 + 1u, uint(u_K - 1))], k1 - float(a1));
            uint iv = uint(min(float(v), float(u_K - 1)));
            float wiek = max(teraz - KR[iv], 0.0);
            al = exp(-3.0 * wiek / u_zanik);
            if (al < 0.01) { gl_Position = POZA; return; }
        } else {
            al = 0.15 + 0.85 * float(v) / max(u_t, 1.0);
        }
        vec3 pv = w_chwili(j, tt);
        v_k = vec4(kolor_drogi(j, pv, tt), u_kol.a * al);
        gl_Position = u_mvp * vec4(pv, 1.0);
    } else if (u_tryb == 11) {                // igła reflektora (u_seg 0) i strzałka obrotu: 1 dryf netto (z grotem), 2 drganie
        uint c = uint(gl_InstanceID);
        if (u_wiazki == 2 && !wyb_r(c)) { gl_Position = POZA; return; }
        vec3 r = RF[c].xyz, d = dc(c), q = r + u_oigla * d;
        if (u_seg == 0) {
            v_k = vec4(0.25 + 0.75 * KC[c].rgb, u_kol.a);
            gl_Position = u_mvp * vec4(v == 0 ? r : q, 1.0); return;
        }
        uint a, b; float f; para(a, b, f);
        vec4 oa = u_seg == 1 ? OD[a * 768u + c] : OJ[a * 768u + c], ob = u_seg == 1 ? OD[b * 768u + c] : OJ[b * 768u + c];
        vec3 T = mix(oa.xyz, ob.xyz, f); T -= dot(T, d) * d;
        if (length(T) < 1e-6) { gl_Position = POZA; return; }
        T = normalize(T);
        float w = exp(mix(log(max(oa.w, 1e-6)), log(max(ob.w, 1e-6)), f));          // prędkość kątowa, interpolacja w skali log
        float u = clamp((log(w) / log(10.0) - u_olo) / max(u_ohi - u_olo, 1e-6), 0.0, u_seg == 2 ? 1.4 : 1.0);
        vec3 e = q + (0.6 + u_ola * u) * T;
        vec3 n = normalize(cross(T, d)); float h = (0.6 + 1.0 * u) * u_ola / 8.0;
        vec3 p = v == 0 ? q : (v == 3 ? e - h * T + 0.55 * h * n : (v == 5 ? e - h * T - 0.55 * h * n : e));
        v_k = vec4(u_seg == 1 ? lut(0, 0.25 + 0.75 * u) : vec3(0.92), u_kol.a * (0.35 + 0.55 * min(u, 1.0)));
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 6 || u_tryb == 7) {  // wypadkowy wektor podkroku: stan a → a + (stan a+1 − stan a) · skala; 7 = grot
        uint j = uint(u_tryb == 6 ? gl_InstanceID : v);
        if (u_wiazki == 2 && !wyb_t(j)) { gl_Position = POZA; return; }
        uint a, b; float f; para(a, b, f);
        vec3 p0 = u_skala * P[a * 600u + j].xyz, p1 = u_skala * P[b * 600u + j].xyz;
        vec3 q = (u_tryb == 6 && v == 0) ? p0 : p0 + (p1 - p0) * u_gain;
        v_k = vec4(u_kol.rgb, u_kol.a * ((u_tryb == 6 && v == 0) ? 0.12 : 1.0));
        gl_Position = u_mvp * vec4(q, 1.0); gl_PointSize = u_rozm;
    }
}
"""
FS_S = "#version 430\n" + _KSZTALT + """
uniform int u_punkty;
in vec4 v_k; in float v_t; out vec4 o;
void main() { o = u_punkty == 1 ? ksztalt(v_k) : v_k; }
"""


VS_TON = """#version 430
out vec2 v_uv;
void main() { vec2 p = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0; v_uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }
"""
FS_TON = """#version 430
uniform sampler2D u_hdr; uniform float u_ekspozycja;
in vec2 v_uv; out vec4 o;
void main() { vec3 c = texture(u_hdr, v_uv).rgb * u_ekspozycja; c = 1.0 - exp(-c); o = vec4(pow(c, vec3(0.85)), 1.0); }
"""


def siatka_sfery(R, co=12, n=96):
    seg = []
    t = np.linspace(0, 2 * np.pi, n + 1)
    for lon in np.radians(np.arange(0, 180, co)):
        p = R * np.stack([np.cos(t) * np.cos(lon), np.cos(t) * np.sin(lon), np.sin(t)], 1); seg += [p[:-1], p[1:]]
    for lat in np.radians(np.arange(-90 + co, 90, co)):
        p = R * np.stack([np.cos(t) * np.cos(lat), np.sin(t) * np.cos(lat), np.full_like(t, np.sin(lat))], 1); seg += [p[:-1], p[1:]]
    a, b = np.concatenate(seg[0::2]), np.concatenate(seg[1::2])
    X = np.zeros((2 * len(a), 4), np.float32); X[0::2, :3] = a; X[1::2, :3] = b
    return X


class SferaReflektorow(_ElementGPU):
    BUFORY = ('RF', 'KC', 'P', 'D', 'I', 'SF', 'AT', 'C', 'KP', 'KR', 'OD', 'OJ', 'WY')

    def __init__(self, R=34.0):
        super().__init__()
        self.R = float(R)
        self.pod = None                          # podkroki: słownik lasery_dane.podkroki albo None (tryb osi uczenia)
        self.blok = -1
        self.opcje = dict(wiazki=1, gain=0.05, prog=0.02, tok=0, refl=0, rozm=5.0, sfera=0.10, reflektory=True, narz=True,
                          widok=0, wek_kol=0, wek_pul=0, filtr=1, skala_wek=1.0)
        self.para = (-1, -1)
        self.u_pary = 0.0
        self._sf = siatka_sfery(self.R)
        self.n_sf = len(self._sf)
        self.ustaw_bufor('SF', self._sf)
        self.ustaw_wybor([0], [0])

    def ustaw_wybor(self, refl, tok):
        """Wybrane reflektory i cząsteczki (listy indeksów, pierwszy = główny): maski i listy dla wiązek, punktów i wektorów."""
        refl = [int(c) for c in refl if 0 <= c < NK] or [0]; tok = [int(j) for j in tok if 0 <= j < TOK] or [0]
        WY = np.zeros(1368 + NK + TOK, np.float32)
        WY[refl] = 1.0; WY[NK + np.asarray(tok)] = 1.0
        WY[1368:1368 + len(refl)] = refl; WY[1368 + NK:1368 + NK + len(tok)] = tok
        self.n_refl, self.n_tok = len(refl), len(tok)
        self.ustaw_bufor('WY', WY)

    def ustaw_historie(self, h, skala_stala=False):
        self.pod = None; self.przejscie = False; self.blok = -1
        K = h.K
        RF = np.zeros((NK, 4), np.float32); RF[:, :3] = h.rfix
        KC = np.ones((NK, 4), np.float32); KC[:, :3] = h.kol
        self.ustaw_bufor('RF', RF); self.ustaw_bufor('KC', KC)
        self.ustaw_skale(h, skala_stala)
        Dd = np.zeros((K, NK, 4), np.float32); Dd[..., :3] = h.dirs
        self.ustaw_bufor('D', Dd.reshape(-1, 4))
        self.obr_skala = None                                       # strzałki obrotu igieł (film 23): skala log10 z dryfu netto
        if getattr(h, 'obr_dryf', None) is not None:
            self.ustaw_bufor('OD', np.ascontiguousarray(h.obr_dryf.reshape(-1, 4), np.float32))
            self.ustaw_bufor('OJ', np.ascontiguousarray(h.obr_drg.reshape(-1, 4), np.float32))
            w = h.obr_dryf[..., 3]
            self.obr_skala = (float(np.log10(max(np.percentile(w, 2), 1e-6))), float(np.log10(max(np.percentile(w, 99.5), 1e-6))))
        self.K = K
        self.para = (-1, -1)

    def ustaw_skale(self, h, stala):
        r = (h.r99[-1] * np.ones(h.K) if stala else h.r99).astype(np.float32)
        P = np.zeros((h.K, TOK, 4), np.float32)
        P[..., :3] = h.pos / np.maximum(r, 1e-9)[:, None, None] * 0.55 * self.R
        P[..., 3] = h.narz
        self.ustaw_bufor('P', P.reshape(-1, 4))
        self.dnorm_k = float(np.percentile(np.linalg.norm(P[..., :3] - P[:1, :, :3], axis=2), 99))
        # pułapka = średnie położenie koloidu z ±okna kroków (jak centra studni w filmie 2), w tych samych współrzędnych sfery
        kr = np.asarray(h.kroki, np.int64); okno = int(getattr(h, 'okno', 300))
        cs = np.cumsum(P[..., :3].astype(np.float64), 0)
        lo = np.searchsorted(kr, kr - okno, 'left'); hi = np.searchsorted(kr, kr + okno, 'right')
        C = P.copy()
        for t in range(h.K):
            C[t, :, :3] = (cs[hi[t] - 1] - (cs[lo[t] - 1] if lo[t] > 0 else 0)) / (hi[t] - lo[t])
        self._C_surowe = C
        self.ustaw_bufor('KR', kr.astype(np.float32))
        self._wyslij_pulapki()

    def ustaw_wygladzanie(self, tryb, okno, alfa):
        """Wygładzanie pułapek wzdłuż checkpointów (podkroków): 0 brak, 1 średnia ważona krocząca (wagi 1..okno, tylko przeszłość),
        2 EMA (alfa na checkpoint), 3 najpierw średnia krocząca, potem EMA. Zmienia pułapki we wszystkich widokach sfery."""
        self._wyg = (int(tryb), int(okno), float(alfa))
        self._wyslij_pulapki()

    def _wyslij_pulapki(self):
        C = getattr(self, '_C_surowe', None)
        if C is None:
            return
        tryb, okno, alfa = getattr(self, '_wyg', (0, 10, 0.2))
        X = C[..., :3].astype(np.float64)
        if tryb in (1, 3) and okno > 1:                              # WMA: Σ w_k x_{t−k}, w = okno, okno−1, …, 1
            w = np.arange(okno, 0, -1, dtype=np.float64)
            Y = np.zeros_like(X); S = np.zeros(len(X))
            for k, wk in enumerate(w):
                Y[k:] += wk * X[:len(X) - k]; S[k:] += wk
            X = Y / S[:, None, None]
        if tryb in (2, 3):                                           # EMA: y_t = α x_t + (1 − α) y_{t−1}
            Y = X.copy()
            for t in range(1, len(X)):
                Y[t] = alfa * X[t] + (1 - alfa) * Y[t - 1]
            X = Y
        W = C.copy(); W[..., :3] = X
        self.ustaw_bufor('C', np.ascontiguousarray(W.reshape(-1, 4), np.float32))
        self.dnorm_p = float(np.percentile(np.linalg.norm(X - X[:1], axis=2), 99))

    @staticmethod
    def _tryb_koloru(o):
        k = int(o.get('kol_sciezki', 0))
        return 0 if k == 0 else 2 if k == 4 else 1

    def ustaw_kolory_sciezek(self, K):
        self.ustaw_bufor('KP', np.ascontiguousarray(K, np.float32))

    def ustaw_natezenia(self, h, t):
        """Natężenia z_jc dla pary checkpointów wokół t (wysyłane tylko przy zmianie pary)."""
        a = int(np.clip(np.floor(t), 0, h.K - 1)); b = min(a + 1, h.K - 1)
        self.u_pary = float(t - a)
        if (a, b) != self.para:
            I = np.empty((2, TOK, NK), np.float32)
            I[0] = np.asarray(h.Z[a], np.float32) / 2.0                        # z / 2 jak film 12, ze znakiem: ujemna jasność
            I[1] = np.asarray(h.Z[b], np.float32) / 2.0                        # (GELU ≥ −0,17) rysowana na błękitno w shaderze
            self.ustaw_bufor('I', I.reshape(-1))
            self.para = (a, b)
        self.update()

    def pozycje_podkrokow(self, d):
        """13 stanów (układ '_n') w skali sfery: JEDNA skala na checkpoint (99. percentyl po wszystkich stanach do 0,55 R),
        jak w ustaw_podkroki; osobna skala każdego stanu wyglądałaby jak ruch bez sił."""
        pos = d['pos_n']
        r = float(np.percentile(np.linalg.norm(pos, axis=2), 99))
        return (pos / max(r, 1e-9) * 0.55 * self.R).astype(np.float32)

    def ustaw_przejscie(self, Pa, Pb, narz):
        """Płynny powrót do wejścia sieci: 2 stany (wyjście poprzedniego checkpointu → wejście następnego), czas 0..1, bez wiązek."""
        P = np.zeros((2, TOK, 4), np.float32); P[0, :, :3] = Pa; P[1, :, :3] = Pb; P[..., 3] = narz
        self.ustaw_bufor('P', P.reshape(-1, 4)); self._C_surowe = P; self._wyslij_pulapki()
        self.pod = None; self.przejscie = True; self.K = 2; self.t = 0.0
        self.update()

    def ustaw_podkroki(self, d, narz):
        """Przejście przez sieć jednego checkpointu: 13 stanów strumienia rezydualnego (układ znormowany '_n') jako czas cząsteczek,
        w podkroku MLP bloku b reflektory i wiązki tego bloku, w podkroku uwagi linie atencji między cząsteczkami."""
        pos = d['pos_n']
        r = float(np.percentile(np.linalg.norm(pos, axis=2), 99))                  # jedna skala na checkpoint: ruch tylko od sił
        P = np.zeros((13, TOK, 4), np.float32)
        P[..., :3] = pos / max(r, 1e-9) * 0.55 * self.R
        P[..., 3] = narz
        Dd = np.zeros((6, NK, 4), np.float32); Dd[..., :3] = d['dirs_n']
        self.ustaw_bufor('P', P.reshape(-1, 4)); self.ustaw_bufor('D', Dd.reshape(-1, 4))
        self.dnorm_k = float(np.percentile(np.linalg.norm(P[..., :3] - P[:1, :, :3], axis=2), 99))
        self._C_surowe = P                                          # w podkrokach pułapka = stan (brak okna czasu)
        self.ustaw_bufor('KR', np.arange(13, dtype=np.float32))
        self._wyslij_pulapki()
        self.pod = d; self.przejscie = False; self.blok = -1; self.K = 13; self.u_pary = 0.0; self.para = (-1, -1)
        self.blok_a = -1; self._rf_klucz = None; self.m_pod = 0.0
        self.ustaw_podkrok(self.t)

    def ustaw_podkrok(self, s):
        """s = ułamkowy podkrok 0..12; przy zmianie bloku wysyła natężenia (blok MLP) i atencję (blok uwagi), reflektory w chwili s.
        opcja 'plynnie' (szerokość przenikania w podkrokach, 0 = osobne kroki): wiązki MLP i linie uwagi przenikają się na granicy
        podkroków (`przenikanie`). Reflektory stoją: zbiór miejsc i kolorów jest ten sam w każdym bloku (`reflektory_przejscia`)."""
        self.t = float(np.clip(s, 0, 12))
        f = float(self.opcje.get('plynnie', 0.0))
        self.m_pod, bm, ba = przenikanie(self.t, f)
        if self.pod is not None:
            if bm != self.blok:
                z = np.asarray(self.pod['z'][bm], np.float32) / 2.0                 # ze znakiem: ujemna jasność na błękitno
                self.ustaw_bufor('I', np.concatenate([z.reshape(-1), z.reshape(-1)]).astype(np.float32))
                self.blok = bm
            if ba != getattr(self, 'blok_a', -1):
                self.ustaw_bufor('AT', self.pod['ah'][ba].mean(0).astype(np.float32).reshape(-1))
                self.blok_a = ba
            kier, kol, kl = reflektory_przejscia(self.pod, self.t, f)
            if kl != getattr(self, '_rf_klucz', None):
                RF = np.zeros((NK, 4), np.float32); RF[:, :3] = 50.0 * kier
                KC = np.ones((NK, 4), np.float32); KC[:, :3] = kol
                self.ustaw_bufor('RF', RF); self.ustaw_bufor('KC', KC)
                self._rf_klucz = kl
        self.update()

    def _programy(self):
        return dict(s=_program(VS_S, FS_S), ton=_program(VS_TON, FS_TON))

    def _hdr(self, w, h):
        """Bufor ramki RGBA16F na światło wiązek (sumowane bez nasycenia, potem tonowane jak w filmie 12)."""
        if getattr(self, '_fbo_rozm', None) != (w, h):
            if getattr(self, '_fbo', None) is None:
                self._fbo = int(GL.glGenFramebuffers(1)); self._fbo_tex = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            stary = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._fbo_tex, 0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(stary))
            self._fbo_rozm = (w, h)

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['s'])
        pod = self.pod is not None
        wm = float(getattr(self, 'm_pod', 0.0)) if pod else 1.0     # waga wiązek MLP (podkroki: 1 MLP, 0 uwaga, pomiędzy = przenikanie)
        uwaga = pod and wm < 0.5                     # podkrok uwagi: zamiast wiązek laserów linie sił między cząsteczkami
        self._ui('s', u_mvp=mvp, u_t=self.t, u_K=self.K, u_R=self.R, u_gain=float(o['gain']), u_prog=float(o['prog']),
                 u_neg=float(o.get('neg', 1.0)),             # 1 = prawdziwa proporcja do dodatnich (zgodnie z ViT)
                 u_tok=int(o['tok']), u_refl=int(o['refl']), u_pu=0.0 if pod else self.u_pary, u_blokD=self.blok if pod else -1,
                 u_skala=float(o.get('skala_chmury', 1.0)))
        if o['sfera'] > 0:
            self._ui('s', u_tryb=3, u_kol=(0.45, 0.6, 0.75, float(o['sfera'])), u_punkty=0)
            self._addytywnie(False); GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, self.n_sf)
        tryb_w = 0 if getattr(self, 'przejscie', False) else int(o['wiazki'])
        tryb_a = int(o.get('sily', 1)) if pod and wm < 0.999 else 0
        if pod and wm < 1e-3:
            tryb_w = 0
        if tryb_w or tryb_a:
            n = 600 * 768 if tryb_w == 1 else 768 * self.n_tok if tryb_w == 2 else 600 * self.n_refl
            vp = [int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT)]
            stary = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
            self._hdr(vp[2], vp[3])
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glViewport(0, 0, vp[2], vp[3])
            GL.glClearColor(0, 0, 0, 0); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            self._addytywnie(True); GL.glLineWidth(float(o.get('grub_w', 1.0)))
            if tryb_w:
                for seg in (0, 1):
                    self._ui('s', u_tryb=0, u_wiazki=tryb_w, u_seg=seg, u_punkty=0, u_zgas=1.0 - wm)
                    GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, n)
            if tryb_a:                                 # 1: kogo słucha wybrana cząsteczka, 2: wszystkie pary
                ga = float(o.get('gain_a', 1.0)) * 12.0 * (20.0 if tryb_a == 1 else 1.0)
                self._ui('s', u_tryb=5, u_wiazki=2 if tryb_a == 1 else 1, u_gain=ga, u_prog=float(o.get('prog_a', 0.02)), u_punkty=0,
                         u_zgas=wm)
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, 600 * self.n_tok if tryb_a == 1 else 360000)
            self._ui('s', u_zgas=0.0)
            GL.glLineWidth(1.0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, stary)
            GL.glViewport(*vp)
            GL.glUseProgram(self._prog['ton'])
            GL.glActiveTexture(GL.GL_TEXTURE2); GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
            self._ui('ton', u_hdr=2, u_ekspozycja=float(o.get('ekspozycja', 1.0)))
            self._addytywnie(True)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0); GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glUseProgram(self._prog['s'])
        if tryb_w == 3 and o.get('kierunek', True):
            self._ui('s', u_tryb=4, u_punkty=0); self._addytywnie(False); GL.glLineWidth(2.0)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, self.n_refl)
        if o['reflektory']:
            self._ui('s', u_tryb=2, u_wiazki=0, u_rozm=6.0, u_ksztalt=4, u_kol=(1, 1, 1, 0.25 + 0.6 * wm), u_punkty=1)
            self._addytywnie(False); GL.glDrawArrays(GL.GL_POINTS, 0, NK)
            if tryb_w == 3:
                self._ui('s', u_wiazki=2, u_rozm=16.0); GL.glDrawArrays(GL.GL_POINTS, 0, NK)
        obr = int(o.get('obrot', 0))                                 # strzałki obrotu igieł: 0 brak, 1 dryf, 2 drganie, 3 oba
        if obr and not pod and not getattr(self, 'przejscie', False) and getattr(self, 'obr_skala', None):
            dl = float(o.get('dl_obr', 1.0))
            self._ui('s', u_tryb=11, u_olo=self.obr_skala[0], u_ohi=self.obr_skala[1], u_ola=8.0 * dl, u_oigla=5.0 * dl, u_lut=1, u_punkty=0)
            self._addytywnie(False)
            for wyb, grub in ((0, 1.0), (2, 2.4)):                  # wszystkie reflektory, potem wybrany (grubo)
                if wyb == 2 and not o.get('obr_wybrany', True):
                    continue
                self._ui('s', u_wiazki=wyb, u_seg=0, u_kol=(1, 1, 1, 0.55 if wyb == 0 else 1.0)); GL.glLineWidth(1.4 * grub)
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, NK)
                if obr in (2, 3):
                    self._ui('s', u_seg=2, u_kol=(1, 1, 1, 0.8 if wyb == 0 else 1.0)); GL.glLineWidth(1.2 * grub)
                    GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, NK)
                if obr in (1, 3):
                    self._ui('s', u_seg=1, u_kol=(1, 1, 1, 1.0)); GL.glLineWidth(2.2 * grub)
                    GL.glDrawArraysInstanced(GL.GL_LINES, 0, 6, NK)
            GL.glLineWidth(1.0)
        widok = int(o.get('widok', 0))                               # 0 koloidy, 1 pułapki, 2 oba
        filtr = int(o.get('filtr', 1)); sk = float(o.get('skala_wek', 1.0))
        KOL_K, KOL_P = (1.0, 0.82, 0.4), (0.3, 1.0, 0.5)
        for cel, wek in ((1, int(o.get('wek_pul', 0))), (0, int(o.get('wek_kol', 0)))):
            if wek == 3:                                                  # cała historia: droga od checkpointu 0 do teraz
                self._ui('s', u_tryb=10, u_cel=cel, u_filtr=filtr, u_kol=(*(KOL_P if cel else KOL_K), 0.8), u_punkty=0,
                         u_kol_sciezki=self._tryb_koloru(o), u_zanik=float(o.get('zanik', 0.0)), u_lut=1,
                         u_dnorm=getattr(self, 'dnorm_p' if cel else 'dnorm_k', 1.0))
                self._addytywnie(False); GL.glLineWidth(1.4 if filtr else 1.0)
                GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, int(np.floor(self.t)) + 2, TOK)
        if widok in (1, 2):                                               # pułapki: zielona poświata studni
            self._ui('s', u_tryb=8, u_cel=1, u_filtr=0, u_rozm=float(o.get('rozm_pul', 13.0)), u_ksztalt=0,
                     u_kol=(0, 0, 0, float(o.get('jas_pul', 0.55))), u_punkty=1)
            self._addytywnie(True); GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
        if widok in (0, 2):
            self._ui('s', u_tryb=1, u_wiazki=0, u_rozm=float(o['rozm']), u_ksztalt=3, u_kol=(1.0, 1.0, 1.0, 0.9), u_punkty=1)
            self._addytywnie(True); GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
        self._addytywnie(False)
        for cel, wek in ((1, int(o.get('wek_pul', 0))), (0, int(o.get('wek_kol', 0)))):
            if wek in (1, 2):                                             # z kroku na krok albo od kroku 0
                self._ui('s', u_tryb=9, u_cel=cel, u_filtr=filtr, u_seg=wek, u_gain=sk, u_kol=(*(KOL_P if cel else KOL_K), 1.0), u_punkty=0,
                         u_kol_sciezki=self._tryb_koloru(o), u_lut=1, u_dnorm=getattr(self, 'dnorm_p' if cel else 'dnorm_k', 1.0))
                GL.glLineWidth(2.5 if filtr == 2 else 1.6)
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
        GL.glLineWidth(1.0)
        if o['narz']:
            if int(o.get('widok', 0)) == 1:                              # same pułapki: pierścienie na pułapkach narzędzia
                self._ui('s', u_tryb=8, u_cel=1, u_filtr=1, u_rozm=float(o['rozm']) * 3.0, u_ksztalt=1, u_kol=(0.29, 0.94, 1.0, 1.0))
            else:
                self._ui('s', u_tryb=1, u_wiazki=1, u_rozm=float(o['rozm']) * 3.0, u_ksztalt=1, u_kol=(0.29, 0.94, 1.0, 1.0))
            GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
        if tryb_w == 2:
            self._ui('s', u_tryb=1, u_wiazki=2, u_rozm=float(o['rozm']) * 4.0, u_ksztalt=1, u_kol=(1.0, 0.85, 0.2, 1.0))
            GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
        if pod and self.t < 12.0 and o.get('wypadkowe', True):   # wypadkowe wektory podkroku: uwaga niebieskie, MLP pomarańczowe
            kol = (0.35, 0.8, 1.0) if uwaga else (1.0, 0.62, 0.25)
            sk = float(o.get('skala_wyp', 1.0))
            self._addytywnie(False)
            self._ui('s', u_tryb=6, u_wiazki=0, u_gain=sk, u_kol=(*kol, 0.55), u_punkty=0); GL.glLineWidth(1.2)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
            self._ui('s', u_wiazki=2, u_kol=(*kol, 1.0)); GL.glLineWidth(3.5)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
            self._ui('s', u_tryb=7, u_wiazki=2, u_rozm=11.0, u_ksztalt=3, u_kol=(*kol, 1.0), u_punkty=1)
            GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
            GL.glLineWidth(1.0)


# ====================================================================== pułapka optyczna: ogniskowa wiązek (okienko 11 i sfera)
_SSBO_O = """
layout(std430, binding = 0) readonly buffer BG { vec4 OG[]; };   // [2·2·N] para checkpointów (a, b): N × (rgb, dominacja), N × (pole f xyz, |f|) surowe
layout(std430, binding = 1) readonly buffer BT { vec4 OT[]; };   // [K·4] ekran: koloid, pułapka (w = 1 zero pola), środek siatki (w = pół boku), skok
layout(std430, binding = 2) readonly buffer BL { vec4 OL[]; };   // [2·n] znaczniki: położenie (ekran), kolor
layout(std430, binding = 3) readonly buffer BW { vec4 OW[]; };   // [2·n] wiązki i wkłady wybranej cząsteczki: położenie, kolor
layout(std430, binding = 4) readonly buffer BB { vec4 OB[]; };   // [2·n] światło ogniska: kierunek wiązki + jasność, kolor
layout(std430, binding = 7) readonly buffer BQ { vec4 OP[]; };   // [2·600] cząsteczki w tle: położenie (ekran), kolor przyciemniony
layout(std430, binding = 6) readonly buffer BF { vec4 OF[]; };   // [2·n] obrys ogniska: trójkąty, potem krawędzie (położenie, kolor)
layout(std430, binding = 5) readonly buffer BR { vec4 OR[]; };   // [2·n] skala sfery (bez skali chmury): siatka szkła, wiązki od reflektorów, reflektory
"""
VS_O = "#version 430\n" + _SSBO_O + """
uniform mat4 u_mvp;
uniform int u_tryb, u_G, u_N, u_ka, u_kb, u_pods, u_wzgl, u_slad, u_wsp, u_sl;
uniform float u_ff, u_fc, u_skala, u_ostr, u_tlo, u_jas, u_rozm, u_strz;
uniform float u_w0, u_rozb, u_Lb, u_px; uniform int u_M; uniform vec4 u_c;
uniform vec4 u_sA, u_sB;        // x: skala surowe → ekran, y: mediana |f| na siatce (surowe)
uniform vec4 u_skA, u_skB;      // skok koloidu (surowy), do pola względem skoku
uniform vec4 u_ref;             // pułapka teraz (ślad w układzie pułapki)
uniform float u_faza; uniform int u_plyn;   // płynące strzałki pola: faza = czas · tempo
out vec4 v_k;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
vec3 wezel(int inst, int i) {
    vec4 c = OT[(inst == 0 ? u_ka : u_kb) * 4 + 2];
    ivec3 q = ivec3(i % u_G, (i / u_G) % u_G, i / (u_G * u_G));
    return c.xyz + ((vec3(q) / float(u_G - 1)) * 2.0 - 1.0) * c.w;
}
bool poza_kula(int inst, int i) {           // pole tylko w kuli komórki tokenu (promień = połowa boku siatki)
    vec4 c = OT[(inst == 0 ? u_ka : u_kb) * 4 + 2];
    return length(wezel(inst, i) - c.xyz) > c.w * 1.0001;
}
vec3 slad(int k, int sl) {                 // punkt śladu (sl 0 koloid, 1 pułapka); ostatni w ułamku chwili
    if (k != u_kb) return OT[k * 4 + sl].xyz;
    return mix(OT[u_ka * 4 + sl].xyz, OT[u_kb * 4 + sl].xyz, sl == 0 ? u_fc : u_ff);
}
void main() {
    int v = gl_VertexID;
    v_k = vec4(0.0);
    if (u_tryb == 0) {                      // mgła pola: instancja 0 = checkpoint a, 1 = b (przenikanie u_ff)
        int inst = gl_InstanceID, i = v;
        vec4 kol = OG[(inst * 2) * u_N + i], pol = OG[(inst * 2 + 1) * u_N + i];
        vec4 s = inst == 0 ? u_sA : u_sB;
        float sig = max(u_ostr * s.y, 1e-30);
        float g = exp(-0.5 * pol.w * pol.w / (sig * sig));          // ognisko: tam, gdzie pchnięcia wiązek się znoszą
        float a = (inst == 0 ? 1.0 - u_ff : u_ff) * u_jas * (u_tlo + (1.0 - u_tlo) * g);
        if (a < 1e-4 || poza_kula(inst, i)) { gl_Position = POZA; return; }
        v_k = vec4(kol.rgb * (0.45 + 0.55 * kol.a) * a, 1.0);
        gl_Position = u_mvp * vec4(u_skala * wezel(inst, i), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 1) {               // strzałki pola co u_pods węzłów (instancja = węzeł + N·(0 a, 1 b)), 6 wierzchołków: trzon i grot
        int inst = gl_InstanceID / u_N, i = gl_InstanceID % u_N;
        ivec3 q = ivec3(i % u_G, (i / u_G) % u_G, i / (u_G * u_G));
        float w = inst == 0 ? 1.0 - u_ff : u_ff;
        if (w < 0.02 || q.x % u_pods != 0 || q.y % u_pods != 0 || q.z % u_pods != 0 || poza_kula(inst, i)) { gl_Position = POZA; return; }
        vec4 kol = OG[(inst * 2) * u_N + i], pol = OG[(inst * 2 + 1) * u_N + i];
        vec4 s = inst == 0 ? u_sA : u_sB;
        vec3 f = pol.xyz - (u_wzgl == 1 ? (inst == 0 ? u_skA.xyz : u_skB.xyz) : vec3(0.0));
        vec3 p = u_skala * wezel(inst, i), e = p + u_skala * s.x * u_strz * f, d = e - p;
        float L = length(d);
        if (L < 1e-6) { gl_Position = POZA; return; }
        float plyn = 1.0;
        if (u_plyn == 1) {                    // strzałka przesuwa się wzdłuż pola (faza węzła losowa), pojawia się i gaśnie
            float ph = fract(u_faza + fract(sin(float(i) * 12.9898) * 43758.5453));
            vec3 sh = (ph - 0.5) * d; p += sh; e += sh; plyn = sin(3.14159265 * ph);
        }
        vec4 kc = OT[(inst == 0 ? u_ka : u_kb) * 4 + 2];
        if (length(p - u_skala * kc.xyz) > u_skala * kc.w || length(e - u_skala * kc.xyz) > u_skala * kc.w) { gl_Position = POZA; return; }
        vec3 t = d / L, n = normalize(cross(t, abs(t.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)));
        vec3 r = v == 0 ? p : (v == 3 ? e - 0.3 * L * t + 0.15 * L * n : (v == 5 ? e - 0.3 * L * t - 0.15 * L * n : e));
        v_k = vec4(mix(vec3(1.0), kol.rgb, 0.7), w * u_jas * plyn * (v == 0 ? 0.35 : 0.95));
        gl_Position = u_mvp * vec4(r, 1.0);
    } else if (u_tryb == 2) {               // znaczniki (linie albo punkty z bufora OL)
        v_k = OL[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OL[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 7) {               // światło ogniska: wiązka gaussowska i (instancja) z przewężeniem w0 w cząsteczce
        vec4 b = OB[2 * gl_InstanceID], kl = OB[2 * gl_InstanceID + 1];
        float zR = u_w0 / max(u_rozb, 1e-4);                                   // długość Rayleigha z kąta rozbieżności
        float a = log(u_Lb / zR + sqrt(u_Lb * u_Lb / (zR * zR) + 1.0));         // asinh(Lb/zR): próbki gęściej przy ognisku
        float t = float(v) / float(max(u_M - 1, 1)) * 2.0 - 1.0;
        float sx = zR * (exp(a * t) - exp(-a * t)) * 0.5;
        float w = u_w0 * sqrt(1.0 + sx * sx / (zR * zR));
        float I = b.w * u_jas * (u_w0 / w) * (u_w0 / w);
        if (I < 1e-5) { gl_Position = POZA; return; }
        v_k = vec4(kl.rgb * I, 1.0);
        gl_Position = u_mvp * vec4(u_skala * (u_c.xyz - sx * b.xyz), 1.0);
        gl_PointSize = clamp(2.0 * w * u_skala * u_px, 1.5, 256.0);
    } else if (u_tryb == 10) {              // pozostałe cząsteczki w tle (przyciemnione)
        v_k = OP[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OP[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 9) {               // obrys ogniska: powierzchnia równej jasności światła wiązek (bufor OF)
        v_k = OF[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OF[2 * v].xyz, 1.0);
    } else if (u_tryb == 8) {               // reflektory na sferze i ich wiązki do cząsteczki (współrzędne sfery, bez u_skala)
        v_k = OR[2 * v + 1];
        gl_Position = u_mvp * vec4(OR[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 6) {               // 768 wiązek zbiegających się w wybranej cząsteczce i ich wkłady (bufor OW)
        v_k = OW[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OW[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 3) {               // ślad koloidu (u_sl 0) albo pułapki (1) przez checkpointy (wierzchołek = checkpoint)
        int k = v;
        vec3 p = slad(k, u_sl);
        if (u_wsp == 1) p = p - slad(k, 1) + u_ref.xyz;                 // układ pułapki: pułapka stoi w miejscu
        float wiek = float(u_kb - k) / max(float(u_slad), 1.0);
        vec3 c = u_sl == 0 ? vec3(1.0, 0.82, 0.4) : vec3(0.85, 0.6, 1.0);
        v_k = vec4(c, 0.12 + 0.88 * (1.0 - wiek));
        gl_Position = u_mvp * vec4(u_skala * p, 1.0); gl_PointSize = u_rozm;
    }
}
"""
FS_O = "#version 430\n" + """
uniform int u_punkty;                   // 0 linie, 1 mgła (miękka plama, kolor przemnożony), 2 pełna kropka
in vec4 v_k; out vec4 o;
void main() {
    if (u_punkty == 0) { o = v_k; return; }
    vec2 d = gl_PointCoord * 2.0 - 1.0; float r = length(d);
    if (r > 1.0) discard;
    if (u_punkty == 1) { o = vec4(v_k.rgb * exp(-r * r * 3.0), 1.0); return; }
    o = vec4(v_k.rgb, v_k.a * (1.0 - smoothstep(0.7, 1.0, r)));
}
"""

KOL_SCIAGA, KOL_WYPYCHA, KOL_OBROT = (0.3, 0.85, 1.0), (1.0, 0.35, 0.25), (0.82, 0.45, 1.0)
KOL_SKOK, KOL_PULAPKA, KOL_RUCH = (1.0, 0.62, 0.25), (1.0, 1.0, 1.0), (0.35, 1.0, 0.5)


def przenikanie(s, f):
    """Przejście uwaga ↔ MLP między podkrokami: (m, bm, ba) = waga MLP 0..1 (uwaga ma 1 − m), blok MLP i blok uwagi do
    narysowania. f = szerokość przenikania w podkrokach: 0 = osobne kroki (skokowo, jak dawniej), 1 = przenikanie przez cały
    podkrok. Zmiana bloku MLP wypada w połowie podkroku uwagi, a bloku uwagi w połowie podkroku MLP (tam ich waga jest 0)."""
    s = float(np.clip(s, 0, 12)); k = int(np.clip(np.floor(s), 0, 11)); u = s - k; b = k // 2
    m = float(k % 2)
    f = float(np.clip(f, 0, 1))
    if f <= 0:
        return m, b, b
    n = int(round(s))
    if 1 <= n <= 11 and abs(s - n) < f / 2:
        t = (s - n + f / 2) / f; t = t * t * (3 - 2 * t)
        lewo = float((n - 1) % 2); m = lewo * (1 - t) + (1 - lewo) * t
    if k % 2:
        return m, b, (b if u < 0.5 else min(b + 1, 5))
    return m, (max(b - 1, 0) if u < 0.5 else b), b


def reflektory_przejscia(d, s, f):
    """Reflektory STOJĄ: 768 stałych miejsc na zewnętrznej sferze (spirala Fibonacciego), kolor należy do miejsca. Zmienia się tylko,
    który neuron bloku świeci z którego miejsca (slot[b], przypisanie węgierskie), czyli pod jakim kątem jego wiązka wchodzi
    w szklaną sferę (punkt załamania e, dalej wzdłuż kierunku pchnięcia d_c do cząsteczki). Zwraca kierunki (768, 3) i kolory
    (768, 3) miejsc w kolejności neuronów bloku MLP rysowanego w chwili s (`przenikanie`) oraz klucz zmiany."""
    from wspolne import kierunki, kolory_kanalow
    _, bm, _ = przenikanie(s, f)
    sl = d['slot'][bm]
    return kierunki(False)[sl], kolory_kanalow()[:, :3][sl], (id(d), bm)


class PoleOgniskowej(_ElementGPU):
    """Pułapka optyczna z laserów wokół wybranego koloidu (ogniskowa.policz): mgła pola w kolorach wiązek (jasna tam, gdzie
    pchnięcia się znoszą), strzałki pola albo kształtu pułapki (pole względem skoku koloidu), osie siodła i wir w koloidzie
    i w pułapce, skok koloidu w bloku 6, ślady koloidu i pułapki przez checkpointy. Te same współrzędne co SferaReflektorow."""
    BUFORY = ('OG', 'OT', 'OL', 'OW', 'OB', 'OR', 'OF', 'OP')

    def __init__(self):
        super().__init__()
        self.d = None
        self.opcje = dict(mgla=True, jas=0.25, ostr=0.35, tlo=0.05, rozm=22.0, strzalki=2, dl_strz=0.8, pods=3, osie=True, skok=True,
                          slad=1, n_slad=30, wsp=0, skala_chmury=1.0, kropki=True, pudlo=True,
                          wiazki=True, jasnosc=0, gain_w=0.6, dl_wiazki=1.0, wklady=1, dl_wkladow=1.0)
        self.kol = None; self.n_wiazek = self.n_ow = self.n_ob = 0; self.tryb_pod = False; self.Lb = 1.0; self.c_ogn = np.zeros(3)
        self.opcje.update(swiatlo=True, jas_sw=0.35, szer_sw=0.02, rozb_sw=0.15, px=10.0, linie_wiazek=True, osie_ogniska=True,
                          refl=False, R=34.0, sfera=0.10, jas_refl=1.0, obrys=False, poziom_obr=0.5, jas_obr=0.35, pow_obr=20.0)
        self.n_of_tr = self.n_of_kr = 0
        self.n_op = 0
        self.RF = None; self.n_or_lin = self.n_or_wiaz = self.n_or_pkt = 0; self._siatka = None
        self.para = (-1, -1)
        self.ka = self.kb = 0
        self.ff = self.fc = 0.0
        self.n_ol = 0; self.n_pkt = 0

    def ustaw_dane(self, d, skala_k):
        """d: wynik ogniskowa.policz; skala_k (K,): surowe jednostki rzutu → ekran sfery (0,55 R / r99)."""
        self.d = d
        s = np.asarray(skala_k, np.float32)[:, None]
        K = len(d['kroki'])
        OT = np.zeros((K, 4, 4), np.float32)
        OT[:, 0, :3] = d['srodek'] * s
        OT[:, 1, :3] = d['ogn'] * s; OT[:, 1, 3] = d['zb']
        OT[:, 2, :3] = d['srodek_siatki'] * s; OT[:, 2, 3] = d['L'] * s[:, 0]
        OT[:, 3, :3] = d['skok'] * s; OT[:, 3, 3] = np.linalg.norm(d['skok'], axis=1) * s[:, 0]
        self.OT, self.skala_k = OT, s[:, 0]
        self.ustaw_bufor('OT', OT.reshape(-1, 4))
        self.G = int(d['G']); self.N = self.G ** 3
        self.K = K; self.para = (-1, -1)
        self.update()

    def czysc(self):
        self.d = None; self.K = 0; self.update()

    def ustaw_chwile(self, t, kolejnosc=False):
        """t = ułamkowy indeks checkpointu. kolejnosc: najpierw lasery (pole, pułapka, skok), potem koloid."""
        if self.d is None:
            return
        K = self.K
        t = float(np.clip(t, 0, K - 1)); a = int(np.floor(t)); b = min(a + 1, K - 1); f = t - a
        self.ka, self.kb = a, b
        self.ff, self.fc = (min(2 * f, 1.0), max(2 * f - 1, 0.0)) if kolejnosc else (f, f)
        if (a, b) != self.para:
            OG = np.empty((2, 2, self.N, 4), np.float32)
            for i, k in enumerate((a, b)):
                OG[i, 0] = self.d['kolor'][k]; OG[i, 1] = self.d['pole'][k]
            self.ustaw_bufor('OG', OG.reshape(-1, 4))
            self.para = (a, b)
        if not self.opcje.get('tylko_pole', False):                            # podkroki: sama mgła i strzałki (bez osi i wiązek)
            self._znaczniki()
            self._wiazki()
        self.update()

    def ustaw_kolory(self, kol):
        """Kolory 768 reflektorów (jak na sferze), do wiązek zbiegających się w cząsteczce."""
        self.kol = np.asarray(kol, np.float32)[:, :3]

    def ustaw_tlo(self, P, kol=None, jas=0.35, bez=None):
        """Pozostałe cząsteczki w tle (ekran sfery, bez skali chmury): P (600, 3), kol (600, 3) kolory cząsteczek, jas = przyciemnienie,
        bez = indeks cząsteczki pominiętej (ta w ognisku). P = None wyłącza tło."""
        if P is None:
            self.n_op = 0; self.update(); return
        P = np.asarray(P, np.float32); n = len(P)
        kol = np.full((n, 3), 0.8, np.float32) if kol is None else np.asarray(kol, np.float32)[:, :3]
        B = np.zeros((n, 2, 4), np.float32); B[:, 0, :3] = P
        B[:, 1, :3] = kol * float(jas); B[:, 1, 3] = 1.0
        if bez is not None and 0 <= int(bez) < n:
            B = np.delete(B, int(bez), axis=0)
        self.n_op = len(B)
        self.ustaw_bufor('OP', B.reshape(-1, 4))

    def ustaw_reflektory(self, RF):
        """Miejsca 768 reflektorów bloku 6 na zewnętrznej sferze (h.rfix, współrzędne sfery), do widoku w skali sfery."""
        self.RF = None if RF is None else np.asarray(RF, np.float64)[:, :3]

    def _wiazki(self):
        """Oś uczenia: 768 wiązek bloku 6 zbiegających się w wybranej cząsteczce (dane ogniskowa.policz: P3 v_c, z_jc, z̄_c)."""
        o, d = self.opcje, self.d
        if self.tryb_pod:
            return
        if not o.get('wiazki', True) or d is None or 'pv' not in d or self.kol is None:
            self._ognisko_czysc()
            return
        a, b, ff = self.ka, self.kb, self.ff
        k = self.k_teraz(); s = float(self.skala_k[k])
        pv = ((1 - ff) * d['pv'][a] + ff * d['pv'][b]) * s                            # (768, 3) ekran
        zj = (1 - ff) * d['zj'][a].astype(np.float32) + ff * d['zj'][b].astype(np.float32)
        zb = (1 - ff) * d['zbar'][a].astype(np.float32) + ff * d['zbar'][b].astype(np.float32)
        wk = (zj - zb)[:, None] * pv
        if int(o.get('jasnosc', 0)) == 0:                                      # wkład do skoku: wiązka wchodzi z kierunku, w który pcha
            kier, kol, jas = wk, 0.25 + 0.75 * self.kol, np.linalg.norm(wk, axis=1)
        else:                                                                  # jak na sferze: kierunek v_c, błękit = ujemna dioda
            kier, jas = pv, np.abs(zj) * np.linalg.norm(pv, axis=1)
            kol = np.where((zj < 0)[:, None], np.array([0.30, 0.85, 1.0], np.float32), 0.25 + 0.75 * self.kol)
        self.ustaw_ognisko(self.koloid(), kier, wk, jas, kol, ruch=None, zrodla=self.RF,
                           zapalone=float((zj > 0.05).mean()), ujemne=float((zj < -0.05).mean()))

    def _ognisko_czysc(self):
        self.n_wiazek = self.n_ow = self.n_ob = 0
        self.n_or_lin = self.n_or_wiaz = self.n_or_pkt = 0
        self.n_of_tr = self.n_of_kr = 0
        self.info_wiazek = None
        self.ustaw_bufor('OW', np.zeros((1, 4), np.float32)); self.ustaw_bufor('OB', np.zeros((1, 4), np.float32))

    def ustaw_ognisko(self, c, kier, wk, jas, kol, ruch=None, zrodla=None, prosto=False, Lb_staly=None, refl_tlo=None,
                      jas_wzgl=False, **info):
        """Ognisko JEDNEJ cząsteczki (ekran sfery): c środek; kier (n, 3) kierunki wiązek (dowolnej długości), wk (n, 3) wkłady
        do skoku, jas (n,) ≥ 0 jasności, kol (n, 3) kolory; ruch = prawdziwe przesunięcie (kontrola, biała strzałka).
        Wiązka i wchodzi w cząsteczkę z kierunku −kier_i; światło ogniska rysuje shader (wiązki gaussowskie, przewężenie w c);
        wkłady jako strzałki albo łańcuch głowa–ogon (koniec = Σ wk = skok, pomarańczowy).
        zrodla (n, 3): skąd świecą wiązki w widoku w skali sfery (opcja refl): reflektory na zewnętrznej sferze (prosto=False,
        wiązka reflektor → szklana sfera → cząsteczka, jak film 12) albo inne cząsteczki w ich współrzędnych (prosto=True, uwaga);
        prosto może być tablicą (n,) – płynne przejście podkroków: wiązki MLP i wkłady uwagi naraz.
        jas_wzgl: jasności już względne 0..1 (bez normowania do najjaśniejszej; przenikanie dwóch zestawów wiązek)."""
        o = self.opcje
        c = np.asarray(c, np.float64); kier = np.asarray(kier, np.float64); wk = np.asarray(wk, np.float64)
        jas = np.asarray(jas, np.float64); kol = np.asarray(kol, np.float32)[:, :3]; n = len(kier)
        kn = kier / np.maximum(np.linalg.norm(kier, axis=1), 1e-12)[:, None]
        skok = wk.sum(0); Ls = float(np.linalg.norm(skok))
        dl_w = np.linalg.norm(wk, axis=1); suma = float(dl_w.sum())
        Lb = float(o.get('dl_wiazki', 1.0)) * max(1.5 * Ls, 0.35 * suma / max(np.sqrt(n), 1.0), 1e-6)
        if Lb_staly is not None:                                                         # podkroki: jedna długość na całe przejście
            Lb = float(o.get('dl_wiazki', 1.0)) * max(float(Lb_staly), 1e-6)
        przytnij = bool(o.get('przytnij', False)) and not self.tryb_pod and self.d is not None and int(self.d['G']) > 1
        if przytnij:                                                                     # kadr na komórkę tokenu: wiązki i wkłady w sześcianie
            Lb = min(Lb, self.pol_boku())
        self.Lb = Lb
        jn = (jas if jas_wzgl else jas / max(float(jas.max()), 1e-12)) * float(o.get('gain_w', 1.0))
        czesci = []
        refl = bool(o.get('refl', False)) and zrodla is not None and len(zrodla) == n
        self._reflektory(c, kn, jn, kol, zrodla if refl else None, prosto, refl_tlo if refl else None)
        if o.get('linie_wiazek', True) and not refl:                                   # wiązki: źródło (ciemne) → cząsteczka
            W = np.zeros((n, 2, 2, 4), np.float32)
            W[:, 0, 0, :3] = c - Lb * kn; W[:, 1, 0, :3] = c
            W[:, 0, 1, :3] = kol * jn[:, None] * 0.08; W[:, 1, 1, :3] = kol * jn[:, None] * 0.7; W[:, :, 1, 3] = 1.0
            czesci.append(W)
        n_w = n * 2 if czesci else 0
        tryb = int(o.get('wklady', 1))
        if tryb:
            dw = float(o.get('dl_wkladow', 1.0)); kk = kol
            if tryb == 1:                                                                # strzałki od cząsteczki
                sk_w = dw * (Ls / max(float(np.median(dl_w)), 1e-12) * 0.15)
                if przytnij:                                                             # najdłuższy wkład najwyżej do ścianki sześcianu
                    sk_w = min(sk_w, Lb / max(float(dl_w.max()), 1e-30))
                P0 = np.repeat(c[None], n, 0); P1 = c + wk * sk_w
            else:                                                                        # łańcuch głowa–ogon, od najdłuższego
                o_ = np.argsort(-dl_w); cum = np.cumsum(wk[o_], 0) * dw
                P0 = c + np.vstack([np.zeros((1, 3)), cum[:-1]]); P1 = c + cum; kk = kol[o_]
            Wk = np.zeros((n, 2, 2, 4), np.float32)
            Wk[:, 0, 0, :3] = P0; Wk[:, 1, 0, :3] = P1
            Wk[:, :, 1, :3] = kk[:, None, :]; Wk[:, 0, 1, 3] = 0.35; Wk[:, 1, 1, 3] = 0.9
            czesci.append(Wk)
        extra = []

        def strz(p_, q_, kol_, a_):
            dd = q_ - p_; L_ = float(np.linalg.norm(dd))
            if L_ < 1e-12:
                return
            t_ = dd / L_; n_ = np.cross(t_, [0, 0, 1.0] if abs(t_[2]) < 0.9 else [1.0, 0, 0]); n_ /= np.linalg.norm(n_)
            g = 0.2 * L_
            for pp, qq in ((p_, q_), (q_, q_ - g * t_ + 0.5 * g * n_), (q_, q_ - g * t_ - 0.5 * g * n_)):
                extra.append((pp, qq, kol_, a_))

        strz(c, c + skok, (1.0, 0.62, 0.25), 1.0)                                        # skok = Σ wkładów (pomarańczowy)
        if ruch is not None:
            strz(c, c + np.asarray(ruch, np.float64), (1.0, 1.0, 1.0), 0.7)             # prawdziwy ruch (biały, kontrola)
        # kształt ogniska: tensor kierunków T = Σ jas d dᵀ / Σ jas; ognisko wydłużone wzdłuż kierunku o największej wartości
        T = (jas[:, None, None] * kn[:, :, None] * kn[:, None, :]).sum(0) / max(float(jas.sum()), 1e-30)
        lam, wek = np.linalg.eigh(T); lam, wek = lam[::-1], wek[:, ::-1]
        dl_osi = 1.0 / np.sqrt(np.maximum(1.0 - lam, 1e-6)); dl_osi = dl_osi / dl_osi.max() * 0.25 * Lb
        if o.get('osie_ogniska', True):
            for e in range(3):
                extra.append((c - dl_osi[e] * wek[:, e], c + dl_osi[e] * wek[:, e], (1.0, 1.0, 1.0), 0.55))
        if extra:
            E = np.zeros((len(extra), 2, 2, 4), np.float32)
            for i, (pp, qq, kl, a_) in enumerate(extra):
                E[i, 0, 0, :3] = pp; E[i, 1, 0, :3] = qq; E[i, :, 1, :3] = kl; E[i, :, 1, 3] = a_
            czesci.append(E)
        W = np.concatenate(czesci) if czesci else np.zeros((0, 2, 2, 4), np.float32)
        self.n_wiazek, self.n_ow = n_w, len(W) * 2
        pkt = np.array([[*c, 0.0], [1.0, 0.82, 0.4, 1.0]], np.float32)                 # sama cząsteczka (kropka) na końcu bufora
        self.ustaw_bufor('OW', np.concatenate([W.reshape(-1, 4), pkt]))
        B = np.zeros((n, 2, 4), np.float32)                                             # światło ogniska: kierunek + jasność, kolor
        B[:, 0, :3] = kn; B[:, 0, 3] = jn; B[:, 1, :3] = kol
        self.ustaw_bufor('OB', B.reshape(-1, 4)); self.n_ob = n; self.c_ogn = c
        obr = self._obrys(c, kn, jn, Lb) if o.get('obrys', False) else None
        top = np.argsort(-dl_w)[:3]
        self.info_wiazek = dict(info, n=n, suma=suma, wypadkowa=Ls, top=top, udzial=dl_w[top] / max(suma, 1e-12),
                                ksztalt=lam, wydluzenie=float(np.sqrt((1 - lam[2]) / max(1 - lam[0], 1e-6))), obrys=obr,
                                blad=(float(np.linalg.norm(skok - np.asarray(ruch)) / max(float(np.linalg.norm(ruch)), 1e-30))
                                      if ruch is not None else None))
        self.update()

    def _obrys(self, c, kn, jn, Lb):
        """Obrys 3D ogniska: powierzchnia, na której suma światła wiązek gaussowskich (te same co w rysowanym świetle:
        przewężenie w0, długość Rayleigha zR = w0 / rozbieżność, profil exp(−3ρ²/w²), jasność ∝ (w0/w)²) spada do ułamka
        `poziom_obr` jasności w cząsteczce (0,5 = połowa szczytu, jak FWHM objętości ogniska w optyce). Kształt zależy od danych
        (kierunki i jasności wiązek), rozmiar od wybranego przewężenia. Zwraca osie obrysu (długości / w0, kierunki) albo None."""
        o = self.opcje
        w0 = float(o.get('szer_sw', 0.02)) * Lb; zR = w0 / max(float(o.get('rozb_sw', 0.15)), 1e-4)
        szczyt = float(jn.sum()); poz = float(o.get('poziom_obr', 0.5)) * szczyt
        self.n_of_tr = self.n_of_kr = 0
        if szczyt <= 0 or w0 <= 0:
            return None
        try:
            import torch
            from skimage.measure import marching_cubes
        except ImportError:
            return None
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = jn > 1e-4 * float(jn.max())
        K = torch.as_tensor(kn[m], dtype=torch.float32, device=dev); J = torch.as_tensor(jn[m], dtype=torch.float32, device=dev)

        def jasnosc(X):                                                         # suma wiązek w punktach X (względem cząsteczki)
            out = []
            for i in range(0, len(X), 16384):
                x = torch.as_tensor(X[i:i + 16384], dtype=torch.float32, device=dev)
                s = x @ K.T; r2 = (x * x).sum(1, keepdim=True)
                w2 = w0 * w0 * (1 + s * s / (zR * zR)); rho = (r2 - s * s).clamp_min(0)
                out.append(((J * (w0 * w0 / w2) * torch.exp(-3 * rho / w2)) * (s.abs() <= Lb)).sum(1).cpu().numpy())
            return np.concatenate(out)

        G, h = 33, 2.0 * w0                                                      # pudło rośnie, aż obrys się w nim zmieści
        while True:
            ax = np.linspace(-h, h, G)
            X = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
            V = jasnosc(X).reshape(G, G, G)
            brzeg = max(V[0].max(), V[-1].max(), V[:, 0].max(), V[:, -1].max(), V[:, :, 0].max(), V[:, :, -1].max())
            if brzeg < poz or h >= Lb:
                break
            h *= 1.6
        if V.max() <= poz:
            return None
        v, f, nrm, _ = marching_cubes(V, poz, spacing=(2 * h / (G - 1),) * 3)
        v = v - h
        L = np.array([0.35, 0.5, 0.8]); L /= np.linalg.norm(L)                  # cieniowanie: światło z ustalonego kierunku
        sz = 0.35 + 0.65 * np.abs(nrm @ L)
        jo = float(o.get('jas_obr', 0.35))
        pw = float(o.get('pow_obr', 20.0))                                       # lupa: obrys powiększony wokół cząsteczki
        if o.get('przytnij', False) and not self.tryb_pod and self.d is not None and int(self.d['G']) > 1:
            pw = self.pol_boku() / max(float(np.linalg.norm(v, axis=1).max()), 1e-30)   # kadr komórki: obrys wypełnia kulę komórki
        Tr = np.zeros((len(f) * 3, 2, 4), np.float32)
        Tr[:, 0, :3] = c + pw * v[f.reshape(-1)]
        Tr[:, 1, :3] = (sz[f.reshape(-1)][:, None] * np.array([1.0, 0.9, 0.62]))
        Tr[:, 1, 3] = 0.5 * jo                                                   # powierzchnia słabo, warstwice wyraźnie
        vc = v - v.mean(0)                                                       # osie obrysu: główne kierunki wierzchołków
        lam, wek = np.linalg.eigh(vc.T @ vc / len(vc))
        rz = vc @ wek; dl = rz.max(0) - rz.min(0)
        o_ = np.argsort(-dl)
        odc = []                                                                 # warstwice: przekroje płaszczyznami ⟂ osi 1 i 2
        for e, kol_ in ((wek[:, o_[0]], (1.0, 0.82, 0.4)), (wek[:, o_[1]], (0.85, 0.9, 1.0))):
            dd = v @ e
            for l in np.linspace(dd.min(), dd.max(), 7)[1:-1]:
                s = dd[f] - l                                                    # (F, 3) znak wierzchołków względem płaszczyzny
                pk = []
                for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                    sa, sb = s[:, a_], s[:, b_]
                    t_ = np.where(sa * sb < 0, sa / np.where(sa - sb == 0, 1, sa - sb), np.nan)
                    pk.append(v[f[:, a_]] + t_[:, None] * (v[f[:, b_]] - v[f[:, a_]]))
                pk = np.stack(pk, 1)                                             # (F, 3 krawędzie, 3)
                ok = ~np.isnan(pk[..., 0])
                m2 = ok.sum(1) == 2
                if m2.any():
                    P2 = pk[m2][ok[m2]].reshape(-1, 2, 3)
                    odc.append((P2, kol_))
        if odc:
            Kr = np.concatenate([np.concatenate([P.reshape(-1, 1, 3), np.zeros((P.size // 3, 1, 3))], 1) for P, _ in odc])
            Kr = np.concatenate([Kr, np.zeros((len(Kr), 2, 1))], 2).astype(np.float32)
            Kr[:, 0, :3] = c + pw * Kr[:, 0, :3]
            Kr[:, 1] = np.concatenate([np.tile([*k_, 0.55], (P.size // 3, 1)) for P, k_ in odc])
        else:
            Kr = np.zeros((0, 2, 4), np.float32)
        self.n_of_tr, self.n_of_kr = len(Tr), len(Kr)
        self.ustaw_bufor('OF', np.concatenate([Tr.reshape(-1, 4), Kr.reshape(-1, 4)]))
        return dict(osie=dl[o_] / w0, kier=wek[:, o_], poziom=float(o.get('poziom_obr', 0.5)), w0=w0, pow=pw)

    def _reflektory(self, c, kn, jn, kol, zrodla, prosto, refl_tlo=None):
        """Widok w skali sfery (film 12): szklana sfera R, reflektory na zewnętrznej sferze i ich wiązki do cząsteczki.
        Wiązka c idzie od reflektora do punktu załamania e na szklanej sferze, a dalej wzdłuż kierunku kn_c do cząsteczki
        (e = p − s·kn, |e| = R); w uwadze źródłami są inne cząsteczki (linia prosta). Współrzędne bez skali chmury."""
        o = self.opcje
        if zrodla is None:
            self.n_or_lin = self.n_or_wiaz = self.n_or_pkt = 0
            return
        R = float(o.get('R', 34.0)); sk = float(o.get('skala_chmury', 1.0))
        p = np.asarray(c, np.float64) * sk; n = len(kn)
        if self._siatka is None or self._siatka[0] != R:
            self._siatka = (R, siatka_sfery(R)[:, :3])
        S = self._siatka[1]
        G = np.zeros((len(S), 2, 4), np.float32); G[:, 0, :3] = S; G[:, 1] = (0.45, 0.6, 0.75, float(o.get('sfera', 0.10)))
        jr = (jn * float(o.get('jas_refl', 1.0)))[:, None] * kol
        pr = np.broadcast_to(np.asarray(prosto, bool), (n,))                            # tablica: płynne przejście MLP ↔ uwaga
        Zs = np.asarray(zrodla, np.float64)
        Wy = []
        ip = np.flatnonzero(pr)
        if len(ip):                                                                      # uwaga: od innych cząsteczek prosto
            W = np.zeros((len(ip), 2, 2, 4), np.float32)
            W[:, 0, 0, :3] = Zs[ip] * sk; W[:, 1, 0, :3] = p
            W[:, 0, 1, :3] = 0.15 * jr[ip]; W[:, 1, 1, :3] = 0.8 * jr[ip]; W[..., 1, 3] = 1.0
            Wy.append(W)
        ir = np.flatnonzero(~pr)
        if len(ir):                                                                      # reflektor → szkło → cząsteczka
            kr, jrr = kn[ir], jr[ir]
            b = kr @ p; s = b + np.sqrt(np.maximum(b * b - p @ p + R * R, 0.0))
            e = p - s[:, None] * kr
            W = np.zeros((len(ir), 2, 2, 4), np.float32)
            W[:, 0, 0, :3] = Zs[ir]; W[:, 1, 0, :3] = e
            W[:, 0, 1, :3] = 0.25 * jrr; W[:, 1, 1, :3] = 0.4 * jrr
            W2 = np.zeros((len(ir), 2, 2, 4), np.float32)
            W2[:, 0, 0, :3] = e; W2[:, 1, 0, :3] = p
            W2[:, 0, 1, :3] = 0.4 * jrr; W2[:, 1, 1, :3] = 0.9 * jrr
            W = np.concatenate([W, W2]); W[..., 1, 3] = 1.0
            Wy.append(W)
        W = np.concatenate(Wy) if Wy else np.zeros((0, 2, 2, 4), np.float32)
        P = np.zeros((len(ir), 2, 4), np.float32); P[:, 0, :3] = Zs[ir]                 # punkty tylko reflektorów: cząsteczki-źródła
        P[:, 1, :3] = 0.25 + 0.75 * np.asarray(kol, np.float32)[ir]; P[:, 1, 3] = 0.85   # są w tle (przyciemnione)
        if refl_tlo is not None:                                                         # uwaga: reflektory bloku też świecą (bez wiązek)
            RF_, KR_ = refl_tlo
            P2 = np.zeros((len(RF_), 2, 4), np.float32); P2[:, 0, :3] = RF_
            P2[:, 1, :3] = 0.25 + 0.75 * np.asarray(KR_, np.float32)[:, :3]; P2[:, 1, 3] = 0.85
            P = np.concatenate([P, P2]); n = len(P)
        self.n_or_lin, self.n_or_wiaz, self.n_or_pkt = len(G), len(W) * 2, n
        self.ustaw_bufor('OR', np.concatenate([G.reshape(-1, 4), W.reshape(-1, 4), P.reshape(-1, 4)]))

    # ---------------------------------------------------------------- położenia w tej chwili (ekran, bez skali chmury)
    def koloid(self):
        return (1 - self.fc) * self.OT[self.ka, 0, :3] + self.fc * self.OT[self.kb, 0, :3]

    def pulapka(self):
        return (1 - self.ff) * self.OT[self.ka, 1, :3] + self.ff * self.OT[self.kb, 1, :3]

    def srodek_pola(self):
        return (1 - self.ff) * self.OT[self.ka, 2, :3] + self.ff * self.OT[self.kb, 2, :3]

    def pol_boku(self):
        return float((1 - self.ff) * self.OT[self.ka, 2, 3] + self.ff * self.OT[self.kb, 2, 3])

    def k_teraz(self):
        return self.kb if self.ff >= 0.5 else self.ka

    def _znaczniki(self):
        """Linie znaczników na CPU (kilkadziesiąt odcinków): skok, pułapka, osie siodła i wir, kierunek do pułapki, pudło siatki."""
        o, d = self.opcje, self.d
        k = self.k_teraz(); s = float(self.skala_k[k])
        L = self.pol_boku(); c = self.koloid(); T = self.pulapka()
        lin = []

        def odc(p, q, kol, a=1.0):
            lin.append((p, (*kol, a))); lin.append((q, (*kol, a)))

        def strzalka(p, q, kol, a=1.0, grot=0.28):
            p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
            dd = q - p; Ln = float(np.linalg.norm(dd))
            if Ln < 1e-9:
                return
            t_ = dd / Ln; n_ = np.cross(t_, [0, 0, 1.0] if abs(t_[2]) < 0.9 else [1.0, 0, 0]); n_ /= np.linalg.norm(n_)
            odc(p, q, kol, a)
            odc(q, q - grot * Ln * t_ + 0.5 * grot * Ln * n_, kol, a); odc(q, q - grot * Ln * t_ - 0.5 * grot * Ln * n_, kol, a)

        def ksztalt(srodek, lam, wek, r):
            lam = np.asarray(lam); wek = np.asarray(wek)
            m = max(float(np.abs(lam).max()), 1e-12); uzyte = set()
            for i in range(3):
                if i in uzyte:
                    continue
                l = lam[i]
                if abs(l.imag) > 1e-6 * max(abs(l), 1e-30):                    # para zespolona: wir w płaszczyźnie (Re v, Im v)
                    uzyte.update(j for j in range(i + 1, 3) if abs(lam[j] - np.conj(l)) < 1e-4 * max(abs(l), 1e-30))
                    vr, vi = wek[:, i].real, wek[:, i].imag
                    th = np.linspace(0, 2 * np.pi, 41)
                    P = np.outer(np.cos(th), vr) - np.outer(np.sin(th), vi)       # x(t) = Re(e^{λt} v): obrót od vr ku −vi
                    P *= r * (0.45 + 0.55 * min(abs(l) / m, 1)) / max(np.linalg.norm(P, axis=1).max(), 1e-12)
                    sg = 1.0 if l.imag > 0 else -1.0
                    for a_ in range(40):
                        odc(srodek + P[a_], srodek + P[a_ + 1], KOL_OBROT, 0.9)
                    for a_ in (0, 13, 27):                                       # groty wskazują kierunek obrotu
                        strzalka(srodek + P[a_], srodek + P[a_] + sg * (P[a_ + 1] - P[a_]) * 3.0, KOL_OBROT, 1.0, 0.6)
                else:
                    v = wek[:, i].real; v = v / max(np.linalg.norm(v), 1e-12)
                    dl = r * (0.35 + 0.65 * min(abs(l.real) / m, 1))
                    kol = KOL_SCIAGA if l.real < 0 else KOL_WYPYCHA
                    for zn in (1, -1):                                           # ściąga: groty do środka, wypycha: na zewnątrz
                        if l.real < 0:
                            strzalka(srodek + zn * dl * v, srodek + zn * 0.25 * dl * v, kol, 0.95)
                        else:
                            strzalka(srodek + zn * 0.25 * dl * v, srodek + zn * dl * v, kol, 0.95)
                uzyte.add(i)

        if o['pudlo']:                                                           # kula komórki tokenu (tu liczone jest pole)
            C = self.srodek_pola(); th = np.linspace(0, 2 * np.pi, 49)
            E = np.eye(3)
            okregi = [(E[0], E[1], 0.0), (E[0], E[2], 0.0), (E[1], E[2], 0.0)]
            okregi += [(E[0], E[1], z_) for z_ in (-0.6, 0.6)]
            for e1, e2, z_ in okregi:
                r_ = L * np.sqrt(1 - z_ * z_); e3 = np.cross(e1, e2)
                P_ = C + z_ * L * e3 + r_ * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
                for a_ in range(48):
                    odc(P_[a_], P_[a_ + 1], (0.5, 0.6, 0.7), 0.35 if z_ == 0 else 0.2)
        zb = bool(d['zb'][k])
        if o['osie']:
            ksztalt(c, d['lam0'][k], d['wek0'][k], 0.35 * L)                     # kształt pola w koloidzie
            if zb:
                ksztalt(T, d['lam'][k], d['wek'][k], 0.35 * L)                   # kształt pułapki
        if zb:                                                                   # pułapka: trzy okręgi (kulka)
            th = np.linspace(0, 2 * np.pi, 33); r = 0.07 * L
            for e1, e2 in (((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1))):
                Pk = T + r * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
                for a_ in range(32):
                    odc(Pk[a_], Pk[a_ + 1], KOL_PULAPKA, 1.0)
            odc(c, T, (1, 1, 1), 0.25)
        else:                                                                    # kierunek do pułapki (oszacowanie liniowe), do brzegu pudła
            q = d['ogn_lin'][k] * s
            dd = q - c; Ln = float(np.linalg.norm(dd))
            if Ln > 1e-9:
                strzalka(c, c + dd / Ln * min(Ln, 0.95 * L), (0.6, 0.75, 0.85), 0.55, 0.15)
        if o['skok'] and not o.get('wiazki', True):                             # skok koloidu w bloku 6 (gdy wiązki go nie rysują)
            sk = (1 - self.ff) * self.OT[self.ka, 3, :3] + self.ff * self.OT[self.kb, 3, :3]
            strzalka(c, c + sk, KOL_SKOK, 1.0, 0.2)
        self.info_ruch = None; self.n_ol_ruch = 0
        if self.kb > self.ka:                                                  # wypadkowy ruch koloidu w kroku ka → ka + 1
            p0, p1 = self.OT[self.ka, 0, :3].astype(np.float64), self.OT[self.kb, 0, :3].astype(np.float64)
            dr = p1 - p0; Lr = float(np.linalg.norm(dr))
            skk = self.OT[self.ka, 3, :3].astype(np.float64); Lk = float(np.linalg.norm(skk))
            self.info_ruch = dict(dl=Lr / max(float(self.skala_k[self.ka]), 1e-30),
                                  cos=float(dr @ skk / max(Lr * Lk, 1e-30)), stosunek=Lr / max(Lk, 1e-30))
            if o.get('ruch_nast', True) and Lr > 0:
                q = dr
                if o.get('przytnij', False) and self.G > 1:                     # w kadrze komórki: kierunek prawdziwy, długość do ścianki
                    q = dr / Lr * min(Lr, 0.95 * L)
                n_przed = len(lin)
                strzalka(p0, p0 + q, KOL_RUCH, 1.0, 0.25)
                self.n_ol_ruch = len(lin) - n_przed                             # na końcu bufora: rysowana grubo, na wierzchu
        n = len(lin)
        OL = np.zeros((max(n, 1), 2, 4), np.float32)
        for i, (p, kol) in enumerate(lin):
            OL[i, 0, :3] = p; OL[i, 1] = kol
        self.n_ol = n
        self.ustaw_bufor('OL', OL.reshape(-1, 4))

    def _programy(self):
        return dict(o=_program(VS_O, FS_O))

    def ustaw_tryb_pod(self, on):
        """on: ognisko z podkroków przejścia przez sieć (dane podaje zakładka przez ustaw_ognisko); off: oś uczenia (blok 6)."""
        self.tryb_pod = bool(on)
        if on:
            self.K = max(self.K, 2)
        elif self.d is None:
            self.K = 0
        self.update()

    def _rysuj(self, mvp):
        o = self.opcje
        v = self.view()                                                          # piksele na jednostkę sceny (rozmiar plamek światła)
        if v is not None and v.height() > 0:
            o['px'] = v.height() * v.devicePixelRatioF() / (2 * float(v.opts['distance']) * np.tan(np.radians(float(v.opts['fov'])) / 2))
        GL.glUseProgram(self._prog['o'])
        self._ui('o', u_mvp=mvp, u_skala=float(o['skala_chmury']))
        if o.get('tylko_pole', False):                                           # podkroki: sama mgła i strzałki pola w komórce
            if self.d is not None:
                self._rysuj_pole(o)
            return
        uczenie = (not self.tryb_pod) and self.d is not None
        if o.get('tlo_czastki', True) and self.n_op:                             # pozostałe cząsteczki w tle, pod wszystkim
            self._addytywnie(False)
            self._ui('o', u_tryb=10, u_punkty=2, u_rozm=float(o.get('rozm_tla', 5.0)))
            GL.glDrawArrays(GL.GL_POINTS, 0, self.n_op)
        if uczenie:
            self._rysuj_pole(o)
        if o.get('refl', False) and self.n_or_pkt:                               # skala sfery: szkło, wiązki od reflektorów, reflektory
            self._ui('o', u_tryb=8, u_punkty=0)
            if o.get('sfera', 0.10) > 0:
                self._addytywnie(False); GL.glLineWidth(1.0)
                GL.glDrawArrays(GL.GL_LINES, 0, self.n_or_lin)
            self._addytywnie(True); GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, self.n_or_lin, self.n_or_wiaz)
            self._addytywnie(False)
            self._ui('o', u_punkty=2, u_rozm=6.0)
            GL.glDrawArrays(GL.GL_POINTS, self.n_or_lin + self.n_or_wiaz, self.n_or_pkt)
        if o.get('swiatlo', True) and self.n_ob:                                 # światło ogniska: wiązki gaussowskie
            w0 = float(o.get('szer_sw', 0.02)) * self.Lb; rozb = float(o.get('rozb_sw', 0.15))
            zR = w0 / max(rozb, 1e-4); a = float(np.arcsinh(self.Lb / max(zR, 1e-12)))
            M = int(np.clip(np.ceil(2 * a / (rozb * 0.6)), 16, 160))
            self._addytywnie(True)
            self._ui('o', u_tryb=7, u_punkty=1, u_w0=w0, u_rozb=rozb, u_Lb=self.Lb, u_px=float(o.get('px', 10.0)), u_M=M,
                     u_c=(*self.c_ogn, 0), u_jas=float(o.get('jas_sw', 1.0)))
            GL.glDrawArraysInstanced(GL.GL_POINTS, 0, M, self.n_ob)
        if o.get('obrys', False) and self.n_of_tr:                               # obrys ogniska: półprzezroczysta powierzchnia i siatka
            self._ui('o', u_tryb=9, u_punkty=0)
            self._addytywnie(False)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.n_of_tr)
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, self.n_of_tr, self.n_of_kr)
        if self.n_ow:                                                            # wiązki (addytywnie), potem wkłady i strzałki
            self._ui('o', u_tryb=6, u_punkty=0)
            if self.n_wiazek:
                self._addytywnie(True); GL.glLineWidth(1.0)
                GL.glDrawArrays(GL.GL_LINES, 0, self.n_wiazek)
            if self.n_ow > self.n_wiazek:
                self._addytywnie(False); GL.glLineWidth(1.4)
                GL.glDrawArrays(GL.GL_LINES, self.n_wiazek, self.n_ow - self.n_wiazek)
        if uczenie:
            nr = int(getattr(self, 'n_ol_ruch', 0))
            if self.n_ol > nr:
                self._addytywnie(False); GL.glLineWidth(2.0)
                self._ui('o', u_tryb=2, u_punkty=0)
                GL.glDrawArrays(GL.GL_LINES, 0, self.n_ol - nr)
            if nr:                                                               # nadchodzący ruch koloidu: grubo, na wierzchu
                self._addytywnie(False); GL.glLineWidth(4.5)
                self._ui('o', u_tryb=2, u_punkty=0)
                GL.glDrawArrays(GL.GL_LINES, self.n_ol - nr, nr)
                GL.glLineWidth(1.0)
            k = self.k_teraz()
            self._addytywnie(False)                                              # koloid teraz i pułapka (gdy jest zero pola)
            self._ui('o', u_tryb=3, u_wsp=int(o['wsp']), u_slad=1, u_ref=(*self.pulapka(), 0))
            for cel, rozm in ((1, 9.0), (0, 13.0)):
                if cel == 1 and not bool(self.d['zb'][k]) and not o.get('pulapka_zast', False):
                    continue                                                     # zastępcza (najsłabsze pole) tylko na życzenie
                self._ui('o', u_sl=cel, u_punkty=2, u_rozm=rozm)
                GL.glDrawArrays(GL.GL_POINTS, self.kb, 1)
        elif self.n_ob:                                                          # podkroki: kropka cząsteczki z bufora OW
            self._addytywnie(False)
            self._ui('o', u_tryb=6, u_punkty=2, u_rozm=13.0)
            GL.glDrawArrays(GL.GL_POINTS, self.n_ow, 1)
        GL.glLineWidth(1.0)

    def _rysuj_pole(self, o):
        """Oś uczenia: lokalne pole na siatce (mgła, strzałki) i ślady koloidu oraz pułapki przez checkpointy."""
        k = self.k_teraz()
        if self.G > 1:
            sA = (float(self.skala_k[self.ka]), float(self.d['fmed'][self.ka]), 0, 0)
            sB = (float(self.skala_k[self.kb]), float(self.d['fmed'][self.kb]), 0, 0)
            odst = 2.0 * float(self.OT[k, 2, 3]) / max(self.G - 1, 1) * max(int(o['pods']), 1)
            wzgl = int(o['strzalki']) == 2
            typ = (np.median(np.linalg.norm(self.d['pole'][k][:, :3].astype(np.float32) - (self.d['skok'][k] if wzgl else 0), axis=1))
                   * float(self.skala_k[k]))
            self._ui('o', u_G=self.G, u_N=self.N, u_ka=self.ka, u_kb=self.kb, u_ff=self.ff, u_fc=self.fc, u_sA=sA, u_sB=sB,
                     u_skA=(*self.d['skok'][self.ka], 0), u_skB=(*self.d['skok'][self.kb], 0))
            if o['mgla']:
                self._addytywnie(True)
                self._ui('o', u_tryb=0, u_punkty=1, u_ostr=float(o['ostr']), u_tlo=float(o['tlo']), u_jas=float(o['jas']),
                         u_rozm=float(o['rozm']))
                GL.glDrawArraysInstanced(GL.GL_POINTS, 0, self.N, 2)
            if int(o['strzalki']):
                self._addytywnie(False); GL.glLineWidth(1.2)
                self._ui('o', u_tryb=1, u_punkty=0, u_wzgl=int(wzgl), u_pods=max(int(o['pods']), 1), u_jas=1.0,
                         u_plyn=int(bool(o.get('plyn', True))), u_faza=float(o.get('faza', 0.0)),
                         u_strz=float(o['dl_strz']) * odst / max(typ, 1e-12))
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 6, 2 * self.N)
        else:
            self._ui('o', u_ka=self.ka, u_kb=self.kb, u_ff=self.ff, u_fc=self.fc)
        sl = int(o['slad'])
        if sl:
            k0 = max(0, self.kb - int(o['n_slad']))
            self._addytywnie(False)
            self._ui('o', u_tryb=3, u_wsp=int(o['wsp']), u_slad=int(o['n_slad']), u_ref=(*self.pulapka(), 0))
            for cel in ((0, 1) if sl == 2 else (0,)):
                self._ui('o', u_sl=cel, u_punkty=0); GL.glLineWidth(1.6 if cel == 0 else 1.2)
                GL.glDrawArrays(GL.GL_LINE_STRIP, k0, self.kb - k0 + 1)
                if o['kropki']:                                                  # skoki: położenia w kolejnych checkpointach
                    self._ui('o', u_punkty=2, u_rozm=5.0 if cel == 0 else 4.0)
                    GL.glDrawArrays(GL.GL_POINTS, k0, self.kb - k0 + 1)


# ====================================================================== mapy na zdjęciu
_SSBO_M = """
layout(std430, binding = 0) readonly buffer BM { float M[]; };     // [K*NM*600] mapy
layout(std430, binding = 1) readonly buffer BX { float MX[]; };    // [K*NM] maksimum mapy w checkpoincie
layout(std430, binding = 2) readonly buffer BR { vec4 RAM[]; };    // [K] ramka modelu (px wejścia)
layout(std430, binding = 3) readonly buffer BF { float RE[]; };    // [2*600] łączne natężenie wybranych reflektorów (para checkpointów)
"""
VS_M = "#version 430\n" + _SSBO_M + _CZAS + """
uniform mat4 u_mvp;
uniform int u_tryb;
uniform vec2 u_o;                       // lewy górny róg kafelka w scenie (komórki)
uniform vec4 u_box;
out vec2 v_c;
vec4 scena(vec2 c) { return u_mvp * vec4(u_o.x + c.x, u_o.y - c.y, 0.0, 1.0); }
void main() {
    int v = gl_VertexID;
    if (u_tryb == 0) {                  // kafelek: 2 trójkąty 30 × 20 komórek
        vec2 r[6] = vec2[6](vec2(0, 0), vec2(30, 0), vec2(30, 20), vec2(0, 0), vec2(30, 20), vec2(0, 20));
        v_c = r[v]; gl_Position = scena(r[v]);
    } else {                            // ramka: 1 = u_box (px), 2 = ramka modelu w chwili t (px)
        vec4 b = u_box;
        if (u_tryb == 2) { uint a, bb; float f; para(a, bb, f); b = mix(RAM[a], RAM[bb], f); }
        b /= 16.0;
        vec2 r[5] = vec2[5](b.xy, vec2(b.z, b.y), b.zw, vec2(b.x, b.w), b.xy);
        v_c = r[v]; gl_Position = scena(r[v]);
    }
}
"""
FS_M = "#version 430\n" + _SSBO_M + _CZAS + _LUT + """
uniform sampler2D u_rgb;
uniform int u_tryb, u_mapa, u_NM, u_norm, u_lutk;
uniform float u_vmax, u_gamma, u_alfa, u_pu;
uniform vec4 u_kol;
in vec2 v_c; out vec4 o;
float wart(uint a, int j) {
    if (u_mapa == 9) return mix(RE[j], RE[600 + j], u_pu);
    return M[(a * uint(u_NM) + uint(u_mapa)) * 600u + uint(j)];
}
float probka(uint a, vec2 c) {          // dwuliniowo po środkach komórek
    vec2 q = clamp(c - 0.5, vec2(0.0), vec2(29.0, 19.0));
    int x0 = int(floor(q.x)), y0 = int(floor(q.y)); int x1 = min(x0 + 1, 29), y1 = min(y0 + 1, 19);
    vec2 f = q - vec2(x0, y0);
    float g = mix(wart(a, y0 * 30 + x0), wart(a, y0 * 30 + x1), f.x), d = mix(wart(a, y1 * 30 + x0), wart(a, y1 * 30 + x1), f.x);
    return mix(g, d, f.y);
}
void main() {
    if (u_tryb != 0) { o = u_kol; return; }
    vec3 foto = texture(u_rgb, v_c / vec2(30.0, 20.0)).rgb;
    uint a, b; float f; para(a, b, f);
    float v = mix(probka(a, v_c), probka(b, v_c), f);
    float m;
    if (u_norm == 0 && u_mapa < 9) m = mix(MX[a * uint(u_NM) + uint(u_mapa)], MX[b * uint(u_NM) + uint(u_mapa)], f);
    else m = u_vmax;
    v = pow(clamp(v / max(m, 1e-12), 0.0, 1.0), u_gamma);
    o = vec4(mix(foto, lut(u_lutk, v), u_alfa), 1.0);
}
"""


class MapyNaZdjeciu(_ElementGPU):
    """Kafelki: lista słowników (mapa, x, y, norma, vmax, gamma, lut, alfa); każdy rysuje zdjęcie z mapą i ramki."""
    BUFORY = ('M', 'MX', 'RAM', 'RE')

    def __init__(self):
        super().__init__()
        self.kafelki = []
        self.gt = None
        self.ramki = dict(model=True, gt=True)
        self.u_pary = 0.0
        self.para = (-1, -1, -1)

    def ustaw_historie(self, h):
        K = h.K
        Mp = np.zeros((K, NM, TOK), np.float32)
        Mp[:, :6] = h.atn; Mp[:, 6] = h.atn.mean(1); Mp[:, 7] = h.lok; Mp[:, 8] = h.vib
        self.ustaw_bufor('M', Mp.reshape(-1))
        self.ustaw_bufor('MX', Mp.max(2).reshape(-1))
        R = np.asarray(h.ramka, np.float32)
        self.ustaw_bufor('RAM', R)
        self.ustaw_teksture('rgb', h.rgb)
        self.gt = h.gt
        self.K = K
        self.para = (-1, -1, -1)

    @staticmethod
    def oswietlenie(Z, c):
        """Łączne oświetlenie cząsteczek z reflektorów c (lista): Σ_c max(z_jc, 0); ujemna jasność nie świeci."""
        return np.asarray(Z[:, list(c)], np.float32).clip(0, None).sum(1)

    def ustaw_reflektor(self, h, t, c):
        c = tuple(int(x) for x in np.atleast_1d(c))
        a = int(np.clip(np.floor(t), 0, h.K - 1)); b = min(a + 1, h.K - 1)
        self.u_pary = float(t - a)
        if (a, b, c) != self.para:
            RE = np.stack([self.oswietlenie(h.Z[a], c), self.oswietlenie(h.Z[b], c)])
            self.ustaw_bufor('RE', RE.reshape(-1))
            self.para = (a, b, c)
        self.update()

    def _programy(self):
        return dict(m=_program(VS_M, FS_M))

    def _rysuj(self, mvp):
        GL.glUseProgram(self._prog['m'])
        self._ui('m', u_mvp=mvp, u_t=self.t, u_K=self.K, u_NM=NM, u_rgb=0, u_lut=1, u_pu=self.u_pary)
        self._addytywnie(False)
        for k in self.kafelki:
            GL.glUniform2f(self._u('m', 'u_o'), float(k['x']), float(k['y']))
            self._ui('m', u_tryb=0, u_mapa=int(k['mapa']), u_norm=int(k.get('norma', 0)), u_vmax=float(k.get('vmax', 1.0)),
                     u_gamma=float(k.get('gamma', 1.0)), u_lutk=int(k.get('lut', 2)), u_alfa=float(k.get('alfa', 0.6)))
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            if self.ramki.get('gt') and self.gt is not None:
                self._ui('m', u_tryb=1, u_box=tuple(self.gt), u_kol=(1.0, 1.0, 1.0, 0.85)); GL.glLineWidth(1.2)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, 5)
            if self.ramki.get('model'):
                self._ui('m', u_tryb=2, u_kol=(0.29, 0.94, 1.0, 1.0)); GL.glLineWidth(1.8)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, 5)


# ====================================================================== przejście przez sieć: podkroki w głębokości
_SSBO_P = """
layout(std430, binding = 0) readonly buffer BX { vec4 X[]; };     // [13*600] stan strumienia rezydualnego (rzut 3D)
layout(std430, binding = 1) readonly buffer BK { vec4 KOL[]; };   // [600] kolor cząsteczki
layout(std430, binding = 2) readonly buffer BA { float AH[]; };   // [6*3*600*600] atencja głów (wiersz i = kto słucha, kolumna j = kogo)
layout(std430, binding = 3) readonly buffer BU { vec4 UH[]; };    // [6*3*600] W_o v^h_j w rzucie (wkład cząsteczki j w głowie h)
layout(std430, binding = 4) readonly buffer BZ { float Z[]; };    // [6*600*768] natężenia laserów z_jc
layout(std430, binding = 5) readonly buffer BV { vec4 PV[]; };    // [6*768] kierunek lasera v_c w rzucie
layout(std430, binding = 6) readonly buffer BC { vec4 KC[]; };    // [768] kolor lasera
layout(std430, binding = 7) readonly buffer BF { float AF[]; };   // [6*3*600*600] atencja do sił f_ij: surowa albo A_ij − ā_j (układy '_c', '_n')
"""
VS_P = "#version 430\n" + _SSBO_P + """
uniform mat4 u_mvp;
uniform float u_s, u_gain, u_prog, u_skala_sil, u_rozm;
uniform int u_tryb, u_tok, u_g;
uniform int u_blok1;                       // płynne przejście: blok + 1 do narysowania (0 = blok bieżącego podkroku)
uniform float u_zgas;                      // płynne przejście: 0 pełne siły / lasery, 1 zgaszone (krótsze i ciemniejsze)
uniform vec4 u_kol;
out vec4 v_k;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
vec3 xs(uint s, uint i) { return X[min(s, 12u) * 600u + i].xyz; }
vec3 teraz(uint i) { float s = clamp(u_s, 0.0, 12.0); uint a = uint(floor(s)); return mix(xs(a, i), xs(a + 1u, i), s - float(a)); }
float att(uint b, uint i, uint j) { uint o = b * 1080000u + i * 600u + j; return (AH[o] + AH[o + 360000u] + AH[o + 720000u]) / 3.0; }
vec3 sila(uint b, uint i, uint j) {
    vec3 f = vec3(0.0);
    for (uint h = 0u; h < 3u; h++) f += AF[(b * 3u + h) * 360000u + i * 600u + j] * UH[(b * 3u + h) * 600u + j].xyz;
    return f;
}
void main() {
    int v = gl_VertexID; v_k = u_kol;
    uint k = uint(clamp(floor(u_s), 0.0, 11.0));
    uint b = u_blok1 > 0 ? uint(u_blok1 - 1) : k / 2u;
    float wz = 1.0 - u_zgas;
    if (u_tryb == 0) {                         // cząsteczki w bieżącej chwili podkroku (u_g = 1: tylko wybrana)
        uint i = uint(v); if (u_g == 1 && int(i) != u_tok) { gl_Position = POZA; return; }
        v_k = u_g == 1 ? u_kol : vec4(KOL[i].rgb, u_kol.a);
        gl_Position = u_mvp * vec4(teraz(i), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 1) {                  // tory w głębokości: stany 0 .. bieżący (instancja = cząsteczka)
        uint i = uint(gl_InstanceID);
        if (u_g == 1 && int(i) != u_tok) { gl_Position = POZA; return; }
        int s0 = int(floor(clamp(u_s, 0.0, 12.0)));
        vec3 p = v <= s0 ? xs(uint(v), i) : teraz(i);
        v_k = u_g == 1 ? u_kol : vec4(KOL[i].rgb, u_kol.a);
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 2) {                  // linie uwagi i – j (jasność = średnia atencja głów); u_g = 1: tylko i = wybrana
        uint i = uint(gl_InstanceID / 600), j = uint(gl_InstanceID % 600);
        if (u_g == 1) { i = uint(u_tok); j = uint(gl_InstanceID); }
        float a = att(b, i, j) * u_gain;
        if (i == j || a < u_prog) { gl_Position = POZA; return; }
        v_k = vec4(u_kol.rgb * min(a, 3.0) * wz, 1.0);
        gl_Position = u_mvp * vec4(teraz(v == 0 ? i : j), 1.0);
    } else if (u_tryb == 3) {                  // wektory sił f_ij na wybraną cząsteczkę i (instancja = j)
        uint i = uint(u_tok), j = uint(gl_InstanceID);
        float a = att(b, i, j) * u_gain;
        if (i == j || a < u_prog) { gl_Position = POZA; return; }
        vec3 p = teraz(i);
        v_k = vec4(KOL[j].rgb, 0.9 * wz);
        gl_Position = u_mvp * vec4(v == 0 ? p : p + sila(b, i, j) * u_skala_sil * wz, 1.0);
    } else if (u_tryb == 4) {                  // wektory laserów z_jc v_c na wybraną cząsteczkę (instancja = c)
        uint j = uint(u_tok), c = uint(gl_InstanceID);
        float z = Z[(b * 600u + j) * 768u + c];  // ze znakiem: wektor z_jc v_c, ujemny (błękitny) ciągnie w stronę −v_c
        if (abs(z) * u_gain < u_prog) { gl_Position = POZA; return; }
        v_k = z < 0.0 ? vec4(0.30, 0.85, 1.0, 0.85 * wz) : vec4(0.25 + 0.75 * KC[c].rgb, 0.85 * wz);
        vec3 p = teraz(j);
        gl_Position = u_mvp * vec4(v == 0 ? p : p + z * PV[b * 768u + c].xyz * u_skala_sil * wz, 1.0);
    } else if (u_tryb == 5) {                  // przesunięcie każdej cząsteczki w całym podkroku (stan k → k+1)
        uint i = uint(v / 2);
        v_k = vec4(KOL[i].rgb, v % 2 == 0 ? 0.1 : 0.7);
        gl_Position = u_mvp * vec4(v % 2 == 0 ? xs(k, i) : xs(k + 1u, i), 1.0);
    }
}
"""
FS_P = "#version 430\n" + _KSZTALT + """
uniform int u_punkty;
in vec4 v_k; out vec4 o;
void main() { o = u_punkty == 1 ? ksztalt(v_k) : v_k; }
"""


class PrzejscieGPU(_ElementGPU):
    """Podkroki jednego przejścia w przód: 13 stanów strumienia rezydualnego 600 tokenów (rzut 3D), siły uwagi f_ij między cząsteczkami
    i wektory 768 laserów MLP. CPU podaje tylko u_s (ułamkowy podkrok 0..12) i opcje."""
    BUFORY = ('X', 'KOL', 'AH', 'UH', 'Z', 'PV', 'KC', 'AF')

    def __init__(self):
        super().__init__()
        self.s = 0.0
        self.opcje = dict(tok=0, sily=1, gain=1.0, prog=0.02, skala_sil=1.0, lasery=True, gain_l=1.0, prog_l=0.05, skala_l=1.0,
                          tory=1, rozm=5.0, przes=True)
        from wspolne import kolory_kanalow
        self.ustaw_bufor('KC', kolory_kanalow().astype(np.float32))
        self._ostatnie = None
        self._ostatnie_sr = None

    def ustaw_podkroki(self, d, kolory, uklad=''):
        """uklad: '' = surowy strumień rezydualny, '_c' = względem środka chmury w każdym stanie, '_n' = '_c' w jednej skali
        na checkpoint (nie LayerNorm). W '_c'/'_n' siły są wycentrowane jak położenia (A_ij − ā_j, z_jc − z̄_c, bez biasów),
        więc suma rysowanych wektorów = rysowany ruch cząsteczki (dokładnie, jak w ViT)."""
        X = np.zeros((13 * TOK, 4), np.float32); X[:, :3] = d['pos' + uklad].reshape(-1, 3)
        U = np.zeros((6 * 3 * TOK, 4), np.float32); U[:, :3] = d['uh' + uklad].reshape(-1, 3)
        PV = np.zeros((6 * NK, 4), np.float32); PV[:, :3] = d['pv' + uklad].reshape(-1, 3)
        dane = [('X', X), ('KOL', kolory), ('UH', U), ('PV', PV)]
        srodek = uklad != '' and 'ah_c' in d                         # stary cache (WERSJA 1) nie ma ah_c / z_c
        if self._ostatnie is not d:                                  # linie uwagi: zawsze surowa atencja (kogo słucha)
            dane += [('AH', d['ah'].reshape(-1))]
        if self._ostatnie_sr != (id(d), srodek):
            dane += [('AF', (d['ah_c'] if srodek else d['ah']).reshape(-1)), ('Z', (d['z_c'] if srodek else d['z']).reshape(-1))]
        self._ostatnie = d
        self._ostatnie_sr = (id(d), srodek)
        for n, a in dane:
            self.ustaw_bufor(n, np.ascontiguousarray(a, np.float32))
        self.K = 13

    def ustaw_podkrok(self, s):
        self.s = float(s); self.update()

    def _programy(self):
        return dict(p=_program(VS_P, FS_P))

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['p'])
        self._ui('p', u_mvp=mvp, u_s=self.s, u_tok=int(o['tok']))
        k = int(np.clip(np.floor(self.s), 0, 11)); w_ruchu = self.s < 12.0
        wm, bm, ba = przenikanie(self.s, float(o.get('plynnie', 0.0)))   # waga laserów MLP; uwaga ma 1 − wm
        uwaga = w_ruchu and wm < 0.999
        self._ui('p', u_blok1=ba + 1, u_zgas=wm)
        if o['tory'] == 2:                            # tory w głębokości wszystkich cząsteczek (słabo)
            self._ui('p', u_tryb=1, u_g=0, u_kol=(1, 1, 1, 0.10), u_punkty=0); self._addytywnie(False); GL.glLineWidth(1.0)
            GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, 13, TOK)
        if o['tory'] >= 1:                            # tor wybranej cząsteczki przez wszystkie podkroki
            self._ui('p', u_tryb=1, u_g=1, u_kol=(1.0, 0.85, 0.2, 0.9), u_punkty=0); self._addytywnie(False); GL.glLineWidth(2.0)
            GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, 13, TOK)
        if w_ruchu and o['przes']:
            self._ui('p', u_tryb=5, u_punkty=0); self._addytywnie(False); GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, 2 * TOK)
        if uwaga and o['sily'] == 2:                  # wszystkie pary: linie atencji
            self._ui('p', u_tryb=2, u_g=0, u_gain=float(o['gain']), u_prog=float(o['prog']), u_kol=(0.35, 0.8, 1.0, 1.0), u_punkty=0)
            self._addytywnie(True); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK * TOK)
        if uwaga and o['sily'] >= 1:                  # wybrana cząsteczka: kogo słucha (linie) i siły f_ij (wektory)
            self._ui('p', u_tryb=2, u_g=1, u_gain=float(o['gain']) * 20, u_prog=float(o['prog']), u_kol=(0.35, 0.8, 1.0, 1.0), u_punkty=0)
            self._addytywnie(True); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
            self._ui('p', u_tryb=3, u_skala_sil=float(o['skala_sil']), u_gain=float(o['gain']) * 20)
            self._addytywnie(False); GL.glLineWidth(1.8); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
        if w_ruchu and wm > 1e-3 and o['lasery']:
            self._ui('p', u_tryb=4, u_gain=float(o['gain_l']), u_prog=float(o['prog_l']), u_skala_sil=float(o['skala_l']), u_punkty=0,
                     u_blok1=bm + 1, u_zgas=1.0 - wm)
            self._addytywnie(False); GL.glLineWidth(1.5); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, NK)
        self._ui('p', u_blok1=0, u_zgas=0.0)
        GL.glLineWidth(1.0)
        self._ui('p', u_tryb=0, u_g=0, u_rozm=float(o['rozm']), u_ksztalt=3, u_kol=(1, 1, 1, 1.0), u_punkty=1)
        self._addytywnie(False); GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
        self._ui('p', u_g=1, u_rozm=float(o['rozm']) * 4, u_ksztalt=1, u_kol=(1.0, 0.85, 0.2, 1.0))
        GL.glDrawArrays(GL.GL_POINTS, 0, TOK)


# ====================================================================== dolina pewności i IoU wybranej cząsteczki (okienko 12 i sfera)
_SSBO_D = """
layout(std430, binding = 0) readonly buffer BV { vec4 DV[]; };   // [2·G³] punkt przekroju (a, b, c) i pewność p; (IoU, 0, 0, 0)
layout(std430, binding = 1) readonly buffer BT { vec4 DT[]; };   // [2·3n] obrys doliny: wierzchołek (a, b, c), normalna
layout(std430, binding = 2) readonly buffer BL { vec4 DL[]; };   // [2·m] linie: osie przekroju i krawędzie sześcianu (położenie, kolor)
"""
VS_D = "#version 430\n" + _SSBO_D + """
uniform mat4 u_mvp, u_M;                 // u_M: układ przekroju (192 wymiary, baza E) → scena
uniform int u_tryb;
uniform float u_pmin, u_imin, u_tlo, u_jas, u_rozm;
uniform vec4 u_kol;
out vec4 v_k; out vec3 v_n;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
void main() {
    int v = gl_VertexID;
    v_n = vec3(0.0, 0.0, 1.0);
    if (u_tryb == 0) {                   // mgła doliny: punkty siatki, gdzie pewność i IoU są nad progami
        vec4 a = DV[2 * v]; float p = a.w, q = DV[2 * v + 1].x;
        bool op = p >= u_pmin, oi = q >= u_imin;
        vec3 c; float al;
        if (op && oi) {
            float s = clamp(min((p - u_pmin) / max(1.0 - u_pmin, 1e-3), (q - u_imin) / max(1.0 - u_imin, 1e-3)), 0.0, 1.0);
            c = mix(vec3(0.05, 0.45, 0.22), vec3(0.35, 1.0, 0.75), s); al = 0.2 + 0.8 * s;
        } else if (u_tlo > 0.0 && (op || oi)) {   // tylko jedno z dwóch: pomarańczowe = sama pewność, niebieskie = samo IoU
            c = op ? vec3(1.0, 0.55, 0.15) : vec3(0.25, 0.55, 1.0); al = u_tlo;
        } else { gl_Position = POZA; return; }
        v_k = vec4(c * al * u_jas, 1.0);
        gl_Position = u_mvp * (u_M * vec4(a.xyz, 1.0)); gl_PointSize = u_rozm;
    } else if (u_tryb == 1) {            // obrys doliny: trójkąty powierzchni p = p_min albo IoU = IoU_min
        vec4 a = DT[2 * v];
        v_n = normalize(mat3(u_M) * DT[2 * v + 1].xyz);
        v_k = u_kol;
        gl_Position = u_mvp * (u_M * vec4(a.xyz, 1.0));
    } else if (u_tryb == 2) {            // osie i sześcian
        v_k = DL[2 * v + 1];
        gl_Position = u_mvp * (u_M * vec4(DL[2 * v].xyz, 1.0));
    } else {                             // cząsteczka w środku przekroju
        v_k = u_kol;
        gl_Position = u_mvp * (u_M * vec4(0.0, 0.0, 0.0, 1.0)); gl_PointSize = u_rozm;
    }
}
"""
FS_D = "#version 430\n" + """
uniform int u_punkty;                    // 0 linie, 1 mgła (miękka plama), 2 powierzchnia, 3 kropka z obwódką
in vec4 v_k; in vec3 v_n; out vec4 o;
void main() {
    if (u_punkty == 0) { o = v_k; return; }
    if (u_punkty == 2) {
        float sw = 0.35 + 0.65 * abs(dot(normalize(v_n), normalize(vec3(0.35, 0.55, 0.75))));
        o = vec4(v_k.rgb * sw, v_k.a); return;
    }
    vec2 d = gl_PointCoord * 2.0 - 1.0; float r = length(d);
    if (r > 1.0) discard;
    if (u_punkty == 1) { o = vec4(v_k.rgb * exp(-r * r * 3.0), 1.0); return; }
    o = r > 0.7 ? vec4(0.0, 0.0, 0.0, 1.0) : v_k;
}
"""


class DolinaGPU(_ElementGPU):
    """Przekrój 3D doliny pewności i IoU wokół wybranej cząsteczki (dolina.py): mgła punktów siatki G³ nad progami, obrys doliny
    (marching cubes), osie przekroju i sześcian. Progi p_min i IoU_min są w shaderze (mgła) i w obrysie (liczony od nowa na CPU).
    Przekształcenie M (4 × 4): układ przekroju → scena (okienko 12: skala; sfera: p_j + M3 a w skali sfery)."""
    BUFORY = ('DV', 'DT', 'DL')
    KOL_OSI = ((1.0, 0.35, 0.3), (0.35, 1.0, 0.45), (0.4, 0.6, 1.0))

    def __init__(self):
        super().__init__()
        self.opcje = dict(p_min=0.5, iou_min=0.5, mgla=True, obrys=True, tlo=0.0, jas=1.0, rozm=5.0, siatka=True,
                          kol_obrysu=(0.3, 0.95, 0.6, 0.10), rozm_cz=14.0)
        self.M = np.eye(4, dtype=np.float32)
        self.n = self.n_tr = self.n_l = 0

    def czysc(self):
        self.K = 0
        self.update()

    def ustaw_dane(self, d):
        G, Z = int(d['G']), float(d['Z'])
        a = np.linspace(-Z, Z, G, dtype=np.float32)
        A, B, C = np.meshgrid(a, a, a, indexing='ij')
        DV = np.zeros((G ** 3, 2, 4), np.float32)
        DV[:, 0, 0], DV[:, 0, 1], DV[:, 0, 2] = A.ravel(), B.ravel(), C.ravel()
        DV[:, 0, 3] = d['p'].ravel(); DV[:, 1, 0] = d['iou'].ravel()
        self.ustaw_bufor('DV', DV.reshape(-1, 4)); self.n = G ** 3
        L = []
        for k, kol in enumerate(self.KOL_OSI):                     # osie: od cząsteczki do brzegu sześcianu (+), słabiej w stronę −
            e = np.zeros(3, np.float32); e[k] = Z
            L += [(np.zeros(3), (*kol, 1.0)), (e, (*kol, 1.0)), (np.zeros(3), (*kol, 0.35)), (-e, (*kol, 0.35))]
        r = np.array([[x, y, z] for x in (-Z, Z) for y in (-Z, Z) for z in (-Z, Z)], np.float32)
        for i in range(8):
            for j in range(i + 1, 8):
                if np.count_nonzero(r[i] != r[j]) == 1:
                    L += [(r[i], (0.6, 0.7, 0.8, 0.25)), (r[j], (0.6, 0.7, 0.8, 0.25))]
        DL = np.zeros((len(L), 2, 4), np.float32)
        for i, (p_, k_) in enumerate(L):
            DL[i, 0, :3] = p_; DL[i, 1] = k_
        self.ustaw_bufor('DL', DL.reshape(-1, 4)); self.n_l = len(L)
        self.K = 2
        self.update()

    def ustaw_obrys(self, tr):
        """tr = (wierzchołki (3n, 3), normalne (3n, 3)) albo None."""
        if tr is None:
            self.n_tr = 0
        else:
            T = np.zeros((len(tr[0]), 2, 4), np.float32); T[:, 0, :3] = tr[0]; T[:, 1, :3] = tr[1]
            self.ustaw_bufor('DT', T.reshape(-1, 4)); self.n_tr = len(tr[0])
        self.update()

    def ustaw_przeksztalcenie(self, M):
        self.M = np.asarray(M, np.float32)
        self.update()

    def _programy(self):
        return dict(d=_program(VS_D, FS_D))

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['d'])
        self._ui('d', u_mvp=mvp, u_M=np.ascontiguousarray(self.M.T).reshape(-1), u_pmin=float(o['p_min']), u_imin=float(o['iou_min']),
                 u_tlo=float(o['tlo']), u_jas=float(o['jas']), u_rozm=float(o['rozm']))
        if o['mgla'] and self.n:                                     # mgła sumuje się wzdłuż promienia: jasność ~ 1/G
            G = max(round(self.n ** (1 / 3)), 1)
            self._ui('d', u_tryb=0, u_punkty=1, u_jas=float(o['jas']) * 10.0 / G); self._addytywnie(True)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.n)
        if o['obrys'] and self.n_tr:
            self._ui('d', u_tryb=1, u_punkty=2, u_kol=tuple(o['kol_obrysu'])); self._addytywnie(False)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.n_tr)
        if o['siatka'] and self.n_l:
            self._ui('d', u_tryb=2, u_punkty=0); self._addytywnie(False); GL.glLineWidth(2.4)
            GL.glDrawArrays(GL.GL_LINES, 0, self.n_l); GL.glLineWidth(1.0)
        self._ui('d', u_tryb=3, u_punkty=3, u_rozm=float(o['rozm_cz']), u_kol=(1.0, 0.85, 0.3, 1.0)); self._addytywnie(False)
        GL.glDrawArrays(GL.GL_POINTS, 0, 1)


# ====================================================================== rysunek techniczny reflektora (okienko 13)
_SSBO_R = """
layout(std430, binding = 0) readonly buffer BL { vec4 RL[]; };   // [2·n] wierzchołki odcinków: położenie, kolor
layout(std430, binding = 1) readonly buffer BP { vec4 RP[]; };   // [2·m] punkty: położenie i rozmiar (w), kolor
"""
VS_R = "#version 430\n" + _SSBO_R + """
uniform mat4 u_mvp; uniform int u_tryb, u_start;
out vec4 v_k;
void main() {
    int v = gl_VertexID + u_start;
    if (u_tryb == 0) { v_k = RL[2 * v + 1]; gl_Position = u_mvp * vec4(RL[2 * v].xyz, 1.0); }
    else { v_k = RP[2 * v + 1]; gl_Position = u_mvp * vec4(RP[2 * v].xyz, 1.0); gl_PointSize = RP[2 * v].w; }
}
"""
FS_R = "#version 430\n" + """
uniform int u_punkty; in vec4 v_k; out vec4 o;
void main() {
    if (u_punkty == 0) { o = v_k; return; }
    vec2 d = gl_PointCoord * 2.0 - 1.0; float r = length(d);
    if (r > 1.0) discard;
    o = vec4(r > 0.72 ? vec3(0.0) : v_k.rgb, v_k.a * (1.0 - smoothstep(0.9, 1.0, r)));
}
"""


class SchematGPU(_ElementGPU):
    """Odcinki i punkty rysunku technicznego (reflektor_rysunek.Schemat), grupowane według grubości linii."""
    BUFORY = ('RL', 'RP')

    def __init__(self):
        super().__init__()
        self.grupy = []
        self.n_p = 0

    def ustaw_schemat(self, S):
        if not S.seg:
            self.K = 0; self.update(); return
        szer = np.array([s[3] for s in S.seg]); porz = np.argsort(szer, kind='stable')
        L = np.zeros((len(S.seg), 2, 2, 4), np.float32)
        for k, i in enumerate(porz):
            a, b, kol, _ = S.seg[i]
            L[k, 0, 0, :3], L[k, 1, 0, :3] = a, b
            L[k, 0, 1] = L[k, 1, 1] = kol
        self.ustaw_bufor('RL', L.reshape(-1, 4))
        self.grupy = []
        sz = np.round(szer[porz], 1)
        for w in np.unique(sz):
            idx = np.flatnonzero(sz == w)
            self.grupy.append((int(idx[0]) * 2, int(len(idx)) * 2, float(w)))
        P = np.zeros((max(len(S.pkt), 1), 2, 4), np.float32)
        for k, (p, kol, r) in enumerate(S.pkt):
            P[k, 0, :3] = p; P[k, 0, 3] = r; P[k, 1] = kol
        self.ustaw_bufor('RP', P.reshape(-1, 4)); self.n_p = len(S.pkt)
        self.K = 2
        self.update()

    def czysc(self):
        self.K = 0
        self.update()

    def _programy(self):
        return dict(r=_program(VS_R, FS_R))

    def _rysuj(self, mvp):
        GL.glUseProgram(self._prog['r'])
        self._ui('r', u_mvp=mvp)
        self._addytywnie(False)
        for start, n, w in self.grupy:
            self._ui('r', u_tryb=0, u_start=start, u_punkty=0); GL.glLineWidth(w)
            GL.glDrawArrays(GL.GL_LINES, 0, n)
        GL.glLineWidth(1.0)
        if self.n_p:
            self._ui('r', u_tryb=1, u_start=0, u_punkty=1)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.n_p)
