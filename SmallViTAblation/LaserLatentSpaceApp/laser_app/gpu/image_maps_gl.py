"""Okienka 4–5: mapy na zdjęciu (centra, drgania, atencja, natężenie reflektora) (ImageMaps)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _CZAS, _GPUElement, _LUT, _program, NM, TOK

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


class ImageMaps(_GPUElement):
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
