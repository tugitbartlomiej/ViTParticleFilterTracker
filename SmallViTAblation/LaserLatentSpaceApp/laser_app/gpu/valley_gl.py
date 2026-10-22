"""Okienko 12: dolina pewności i IoU wokół cząsteczki – mgła siatki i obrys (ValleyGPU)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _GPUElement, _program

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


class ValleyGPU(_GPUElement):
    """Przekrój 3D doliny pewności i IoU wokół wybranej cząsteczki (valley.py): mgła punktów siatki G³ nad progami, obrys doliny
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
