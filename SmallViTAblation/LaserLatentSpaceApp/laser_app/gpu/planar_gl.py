"""Okienka 1–3: rzut płaski ruchu (globalny, studnie i strzałki od adresów, drgania) w shaderach (PlanarMotion)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _CZAS, _GPUElement, _KSZTALT, _LUT, _program, TOK

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


class PlanarMotion(_GPUElement):
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
