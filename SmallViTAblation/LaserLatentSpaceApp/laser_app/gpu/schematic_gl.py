"""Okienko 13: rysunek techniczny reflektora (odcinki, punkty) (SchematicGPU)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _GPUElement, _program

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


class SchematicGPU(_GPUElement):
    """Odcinki i punkty rysunku technicznego (reflector_drawing.Schemat), grupowane według grubości linii."""
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
