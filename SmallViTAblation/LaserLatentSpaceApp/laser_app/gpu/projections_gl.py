"""Element OpenGL 4.3 okienek rzutów R1–R7 (projections_view.py): cząsteczki, ślady wędrówki i strzałki liczone w shaderach.

Cała historia położeń i sił w bazie widoku (K × 600 × vec4) trafia na kartę RAZ, przy zmianie bazy, środka, skali albo wygładzania.
W klatce CPU podaje tylko ułamkowy indeks czasu u_t (i kierunek kamery): interpolację między checkpointami, ślady (instancja =
cząsteczka, wierzchołki = kolejne checkpointy, z osobnego bufora wygładzonych położeń PS), zanik śladów, groty strzałek zwrócone do
kamery i krycie strzałki według zachowanej części siły liczy karta. Elementy lokalne (wiązki wybranej cząsteczki, pole w ramie
Schura, kierunki własne) to lista strzałek, odcinków i kropek wysyłana tylko przy zmianie checkpointu albo cząsteczki; mogą być
zakotwiczone w cząsteczce (płyną z nią w czasie).
"""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _CZAS, _GPUElement, _KSZTALT, _program, TOK

_SSBO = """
layout(std430, binding = 0) readonly buffer BP { vec4 P[]; };   // [K*600] położenie w widoku (xyz)
layout(std430, binding = 1) readonly buffer BF { vec4 F[]; };   // [K*600] rzut siły netto (xyz, skala widoku), w = zachowana część długości
layout(std430, binding = 2) readonly buffer BM { vec4 M[]; };   // [600] x: 0 tło, 1 narzędzie, 2 ukryta, 3 tor zerowy; y wybrana; z ślad; w strzałka
layout(std430, binding = 3) readonly buffer BS { vec4 S[]; };   // [N] lokalne: początek (xyz), w: 0 odcinek, 1 strzałka z grotem, 2 kropka
layout(std430, binding = 4) readonly buffer BV { vec4 SV[]; };  // [N] wektor (xyz), w: 0 cienka, 1 gruba
layout(std430, binding = 5) readonly buffer BK { vec4 SK[]; };  // [N] kolor RGBA
layout(std430, binding = 6) readonly buffer BPS { vec4 PS[]; }; // [K*600] położenie wygładzone (ślady)
"""
VS = "#version 430\n" + _SSBO + _CZAS + """
uniform mat4 u_mvp;
uniform int u_tryb, u_L, u_tlo, u_klasa, u_rho, u_kotwica;
uniform float u_rozm, u_strz, u_grot, u_osie;
uniform vec3 u_kam;
uniform vec4 u_kol;
out vec4 v_k;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
vec3 pj(uint j) { uint a, b; float f; para(a, b, f); return mix(P[a * 600u + j].xyz, P[b * 600u + j].xyz, f); }
vec3 psj(uint j) { uint a, b; float f; para(a, b, f); return mix(PS[a * 600u + j].xyz, PS[b * 600u + j].xyz, f); }
vec4 fj(uint j) { uint a, b; float f; para(a, b, f); return mix(F[a * 600u + j], F[b * 600u + j], f); }
// strzałka jako 3 odcinki (6 wierzchołków): trzon 0–1, groty 1–2 i 1–3, zwrócone do kamery
int wierzch(int v) { int s = v / 2; return v % 2 == 0 ? (s == 0 ? 0 : 1) : (s == 0 ? 1 : s + 1); }
vec3 strzalka(vec3 p0, vec3 d, int w) {
    vec3 e = p0 + d;
    if (w < 2) return w == 0 ? p0 : e;
    float L = length(d);
    if (L < 1e-9) return e;
    vec3 u = d / L, n = cross(u, u_kam);
    if (length(n) < 1e-6) n = cross(u, vec3(0.0, 0.0, 1.0));
    n = normalize(n);
    float h = min(u_grot, 0.3 * L);
    return e - h * u + (w == 2 ? 0.45 : -0.45) * h * n;
}
vec3 kolor(uint j) {
    if (M[j].y > 1.5) return vec3(1.0, 1.0, 1.0);                 // pierwsza wybrana
    if (M[j].y > 0.5) return vec3(1.0, 0.82, 0.4);
    if (M[j].x > 2.5) return vec3(0.75, 0.78, 0.82);
    return M[j].x > 0.5 ? vec3(1.0, 0.36, 0.31) : vec3(0.56, 0.64, 0.72);
}
bool ukryta(uint j) { return M[j].x > 1.5 && M[j].x < 2.5; }
void main() {
    int v = gl_VertexID;
    v_k = u_kol;
    if (u_tryb == 0) {                       // cząsteczki; u_klasa 1 = tylko wybrane (duże), 0 = reszta
        uint j = uint(v);
        bool wyb = M[j].y > 0.5;
        if (ukryta(j) || (u_klasa == 1 && !wyb) || (u_klasa == 0 && u_tlo == 0 && M[j].x < 0.5 && !wyb)) { gl_Position = POZA; return; }
        float al = u_klasa == 1 ? 1.0 : (M[j].x > 0.5 ? 0.95 : 0.5);
        v_k = vec4(kolor(j), al * u_kol.a);
        gl_Position = u_mvp * vec4(pj(j), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 1) {                // ślady: instancja = cząsteczka, punkty: checkpointy a−L+1 … a i bieżąca chwila
        uint j = uint(gl_InstanceID);
        if (M[j].z < 0.5 || ukryta(j)) { gl_Position = POZA; return; }
        uint a, b; float f; para(a, b, f);
        int k = v / 2 + v % 2;               // punkt 0..L (L = bieżąca chwila)
        int ti = int(a) - u_L + 1 + k;
        if (ti < 0) { gl_Position = POZA; return; }
        vec3 p = (k == u_L) ? psj(j) : PS[uint(ti) * 600u + j].xyz;
        float al = (0.04 + 0.5 * float(k) / float(max(u_L, 1))) * (M[j].y > 0.5 ? 1.7 : 1.0);   // wybrane jaśniej
        v_k = vec4(kolor(j), min(al, 1.0) * u_kol.a);
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 2) {                // strzałki sił netto: instancja = cząsteczka, krycie = zachowana część długości
        uint j = uint(gl_InstanceID);
        if (M[j].w < 0.5 || ukryta(j)) { gl_Position = POZA; return; }
        vec4 fv = fj(j);
        v_k = vec4(1.0, 0.62, 0.18, (u_rho == 1 ? max(fv.w, 0.12) : 1.0) * u_kol.a);
        gl_Position = u_mvp * vec4(strzalka(pj(j), u_strz * fv.xyz, wierzch(v)), 1.0);
    } else if (u_tryb == 3) {                // lokalne strzałki i odcinki (instancja = element), u_klasa = grubość
        uint i = uint(gl_InstanceID);
        int w = wierzch(v);
        if (S[i].w > 1.5 || int(SV[i].w + 0.5) != u_klasa || (w >= 2 && S[i].w < 0.5)) { gl_Position = POZA; return; }
        vec3 o = u_kotwica >= 0 ? pj(uint(u_kotwica)) : vec3(0.0);
        v_k = SK[i];
        gl_Position = u_mvp * vec4(strzalka(o + S[i].xyz, SV[i].xyz, w), 1.0);
    } else if (u_tryb == 4) {                // lokalne kropki
        uint i = uint(v);
        if (S[i].w < 1.5) { gl_Position = POZA; return; }
        vec3 o = u_kotwica >= 0 ? pj(uint(u_kotwica)) : vec3(0.0);
        v_k = SK[i]; gl_Position = u_mvp * vec4(o + S[i].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 5) {                // osie widoku: 3 odcinki ±u_osie
        int o = v / 2;
        vec3 e = vec3(float(o == 0), float(o == 1), float(o == 2));
        v_k = vec4(o == 0 ? vec3(1.0, 0.45, 0.45) : (o == 1 ? vec3(0.45, 1.0, 0.55) : vec3(0.5, 0.65, 1.0)), 0.5);
        gl_Position = u_mvp * vec4((v % 2 == 0 ? -u_osie : u_osie) * e, 1.0);
    }
}
"""
FS = "#version 430\n" + _KSZTALT + """
uniform int u_punkty;
in vec4 v_k; out vec4 o;
void main() { o = u_punkty == 1 ? ksztalt(v_k) : v_k; }
"""


class ProjectionGPU(_GPUElement):
    """Jeden widok rzutu: położenia i siły całej historii w buforach, maska cząsteczek, elementy lokalne."""
    BUFORY = ('P', 'F', 'M', 'S', 'SV', 'SK', 'PS')

    def __init__(self):
        super().__init__()
        self.opcje = dict(czastki=True, tlo=True, rozm=5.0, L=0, od0=False, strzalki=False, strz=1.0, rho=True, grot=0.8,
                          osie=12.0, kotwica=-1)
        self.N = 0
        self.ustaw_polozenia(np.zeros((2, TOK, 3), np.float32))
        self.ustaw_maske(np.zeros((TOK, 4), np.float32))

    def ustaw_polozenia(self, P, F=None, rho=None, PS=None):
        """P, F: (K, 600, 3) w jednostkach widoku; rho: (K, 600) zachowana część długości siły (krycie strzałki);
        PS: (K, 600, 3) położenia do śladów (np. wygładzone), domyślnie P."""
        P = np.asarray(P, np.float32)
        K = P.shape[0]
        if K < 2:                                                      # jeden checkpoint: powtórzony (shader interpoluje parę)
            P = np.repeat(P, 2, 0); K = 2
            F = None if F is None else np.repeat(np.asarray(F), 2, 0)
            rho = None if rho is None else np.repeat(np.asarray(rho), 2, 0)
            PS = None if PS is None else np.repeat(np.asarray(PS), 2, 0)
        P4 = np.zeros((K, TOK, 4), np.float32); P4[..., :3] = P
        F4 = np.zeros((K, TOK, 4), np.float32)
        if F is not None:
            F4[..., :3] = F
            F4[..., 3] = 1.0 if rho is None else rho
        self.ustaw_bufor('P', P4.reshape(-1, 4)); self.ustaw_bufor('F', F4.reshape(-1, 4))
        if PS is None:
            self.ustaw_bufor('PS', P4.reshape(-1, 4))
        else:
            S4 = np.zeros((K, TOK, 4), np.float32); S4[..., :3] = PS
            self.ustaw_bufor('PS', S4.reshape(-1, 4))
        self.K = K

    def ustaw_maske(self, M):
        self.ustaw_bufor('M', np.asarray(M, np.float32).reshape(TOK, 4))

    def ustaw_lokalne(self, S=None, SV=None, SK=None):
        """Strzałki/odcinki/kropki lokalne: S (N, 4) początek + rodzaj, SV (N, 4) wektor + grubość, SK (N, 4) kolor."""
        if S is None or len(S) == 0:
            self.N = 0
            return
        self.ustaw_bufor('S', np.asarray(S, np.float32)); self.ustaw_bufor('SV', np.asarray(SV, np.float32))
        self.ustaw_bufor('SK', np.asarray(SK, np.float32))
        self.N = len(S)

    def _programy(self):
        return dict(p=_program(VS, FS))

    def _kamera(self):
        v = self.view()
        if v is None:
            return (0.0, 0.0, 1.0)
        c, s = v.cameraPosition(), v.opts['center']
        d = np.array([c.x() - s.x(), c.y() - s.y(), c.z() - s.z()], np.float64)
        n = np.linalg.norm(d)
        return tuple(d / n) if n > 1e-12 else (0.0, 0.0, 1.0)

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['p'])
        self._ui('p', u_mvp=mvp, u_t=self.t, u_K=self.K, u_strz=float(o['strz']), u_grot=float(o['grot']), u_osie=float(o['osie']),
                 u_tlo=int(bool(o['tlo'])), u_rho=int(bool(o['rho'])), u_kotwica=int(o['kotwica']), u_kol=(1, 1, 1, 1))
        GL.glUniform3f(self._u('p', 'u_kam'), *self._kamera())
        a = int(np.clip(np.floor(self.t), 0, self.K - 1))
        L = a + 1 if o['od0'] else min(int(o['L']), a + 1)               # punkty śladu: checkpointy a−L+1 … a i bieżąca chwila
        self._addytywnie(False)
        if o['osie'] > 0:
            self._ui('p', u_tryb=5, u_punkty=0); GL.glLineWidth(1.0); GL.glDrawArrays(GL.GL_LINES, 0, 6)
        if o['czastki'] and L > 0:
            self._ui('p', u_tryb=1, u_L=L, u_punkty=0); GL.glLineWidth(1.3)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2 * L, TOK)
        if self.N:
            for klasa, szer in ((0, 1.4), (1, 3.0)):
                self._ui('p', u_tryb=3, u_klasa=klasa, u_punkty=0); GL.glLineWidth(szer)
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 6, self.N)
        if o['czastki'] and o['strzalki']:
            self._ui('p', u_tryb=2, u_punkty=0); GL.glLineWidth(1.8)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 6, TOK)
        if o['czastki']:
            rozm = float(o['rozm'])
            self._ui('p', u_tryb=0, u_klasa=0, u_rozm=rozm, u_ksztalt=3, u_punkty=1); GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
            self._ui('p', u_tryb=0, u_klasa=1, u_rozm=max(9.0, 2.4 * rozm), u_ksztalt=2, u_punkty=1)
            GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
        if self.N:
            self._ui('p', u_tryb=4, u_rozm=max(4.0, float(o['rozm'])), u_ksztalt=3, u_punkty=1)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.N)
