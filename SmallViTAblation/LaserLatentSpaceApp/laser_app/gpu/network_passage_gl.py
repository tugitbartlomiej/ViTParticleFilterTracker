"""Okienko P: przejście sygnału przez 6 bloków – siły uwagi, lasery MLP, tory (NetworkPassageGPU)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _GPUElement, _KSZTALT, _program, NK, przenikanie, TOK

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


class NetworkPassageGPU(_GPUElement):
    """Podkroki jednego przejścia w przód: 13 stanów strumienia rezydualnego 600 tokenów (rzut 3D), siły uwagi f_ij między cząsteczkami
    i wektory 768 laserów MLP. CPU podaje tylko u_s (ułamkowy podkrok 0..12) i opcje."""
    BUFORY = ('X', 'KOL', 'AH', 'UH', 'Z', 'PV', 'KC', 'AF')

    def __init__(self):
        super().__init__()
        self.s = 0.0
        self.opcje = dict(tok=0, sily=1, gain=1.0, prog=0.02, skala_sil=1.0, lasery=True, gain_l=1.0, prog_l=0.05, skala_l=1.0,
                          tory=1, rozm=5.0, przes=True)
        from laser_app.compute.common import kolory_kanalow
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
