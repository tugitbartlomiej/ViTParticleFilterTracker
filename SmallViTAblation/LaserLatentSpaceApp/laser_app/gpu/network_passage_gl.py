"""Okienko P: przejście sygnału przez 6 bloków – siły uwagi, lasery MLP, tory (NetworkPassageGPU)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _GPUElement, _KSZTALT, NK, _program, przenikanie, TOK

_SSBO_P = """
layout(std430, binding = 0) readonly buffer BX { vec4 X[]; };     // [13*600] stan strumienia rezydualnego (rzut 3D)
layout(std430, binding = 1) readonly buffer BK { vec4 KOL[]; };   // [600] kolor cząsteczki
layout(std430, binding = 2) readonly buffer BA { float AH[]; };   // [6*3*600*600] atencja głów (wiersz i = kto słucha, kolumna j = kogo)
layout(std430, binding = 3) readonly buffer BU { vec4 UH[]; };    // [6*3*600] W_o v^h_j w rzucie (wkład cząsteczki j w głowie h)
layout(std430, binding = 4) readonly buffer BZ { float Z[]; };    // [6*600*768] natężenia laserów z_jc
layout(std430, binding = 5) readonly buffer BV { vec4 PV[]; };    // [6*768] kierunek lasera v_c w rzucie
layout(std430, binding = 6) readonly buffer BC { vec4 KC[]; };    // [768] kolor lasera
layout(std430, binding = 7) readonly buffer BF { float AF[]; };   // [6*3*600*600] atencja do sił f_ij: surowa albo A_ij − ā_j (układy '_c', '_n')
layout(std430, binding = 8) readonly buffer BTR { vec4 TR[]; };   // [u_off*600] stany poprzednich przejść (tory przez kolejne checkpointy)
"""
VS_P = "#version 430\n" + _SSBO_P + """
uniform mat4 u_mvp;
uniform float u_s, u_gain, u_prog, u_skala_sil, u_rozm;
uniform int u_tryb, u_tok, u_g;
uniform int u_glowa;                       // uwaga: 0 = średnia 3 głów, 1..3 = jedna głowa
uniform int u_znak;                        // lasery: 0 = wszystkie, 1 = tylko dodatnie z > 0, -1 = tylko ujemne
uniform float u_alfa;                      // krycie laserów
uniform int u_blok1;                       // płynne przejście: blok + 1 do narysowania (0 = blok bieżącego podkroku)
uniform float u_zgas;                      // płynne przejście: 0 pełne siły / lasery, 1 zgaszone (krótsze i ciemniejsze)
uniform int u_off;                         // stanów poprzednich przejść przed bieżącymi 13 (tory nie zaczynają się od nowa)
uniform vec4 u_kol;
out vec4 v_k;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
vec3 xs(uint s, uint i) { return X[min(s, 12u) * 600u + i].xyz; }
vec3 teraz(uint i) { float s = clamp(u_s, 0.0, 12.0); uint a = uint(floor(s)); return mix(xs(a, i), xs(a + 1u, i), s - float(a)); }
float att(uint b, uint i, uint j) {
    uint o = b * 1080000u + i * 600u + j;
    if (u_glowa > 0) return AH[o + uint(u_glowa - 1) * 360000u];
    return (AH[o] + AH[o + 360000u] + AH[o + 720000u]) / 3.0;
}
vec3 sila(uint b, uint i, uint j) {
    vec3 f = vec3(0.0);
    for (uint h = 0u; h < 3u; h++) if (u_glowa == 0 || int(h) == u_glowa - 1) f += AF[(b * 3u + h) * 360000u + i * 600u + j] * UH[(b * 3u + h) * 600u + j].xyz;
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
        int s0 = int(floor(clamp(u_s, 0.0, 12.0))), vs = v - u_off;
        vec3 p = vs < 0 ? TR[uint(v) * 600u + i].xyz : vs <= s0 ? xs(uint(vs), i) : teraz(i);
        float al = u_off > 0 && vs < 0 ? 0.25 + 0.75 * float(v) / float(u_off) : 1.0;     // starsze przejścia słabiej
        v_k = u_g == 1 ? vec4(u_kol.rgb, u_kol.a * al) : vec4(KOL[i].rgb, u_kol.a * al);
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
        if (abs(z) * u_gain < u_prog || (u_znak == 1 && z <= 0.0) || (u_znak == -1 && z >= 0.0)) { gl_Position = POZA; return; }
        v_k = z < 0.0 ? vec4(0.30, 0.85, 1.0, u_alfa * wz) : vec4(0.25 + 0.75 * KC[c].rgb, u_alfa * wz);
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
    BUFORY = ('X', 'KOL', 'AH', 'UH', 'Z', 'PV', 'KC', 'AF', 'TR')
    # tory przez kolejne checkpointy (jak ścieżki sfery): dolacz_slad = True tuż przed wymianą przy odtwarzaniu z synchronizacją
    dolacz_slad = False
    slad_max = 50
    n_slad = 0

    def __init__(self):
        super().__init__()
        self.s = 0.0
        # widoczność: wiazki = wszystkie wiązki i linie naraz (bramka nad pojedynczymi przełącznikami), linie / wektory = linie uwagi
        # i wektory sił f_ij wybranej cząsteczki, przes = przesunięcia w podkroku, czastki / wybrana = punkty; top_l > 0 = tylko tyle
        # najsilniejszych laserów (|z|), znak_l: 0 wszystkie, 1 dodatnie, -1 ujemne; glowa: 0 = średnia głów, 1..3 = jedna głowa
        self.opcje = dict(tok=0, sily=1, gain=1.0, prog=0.02, skala_sil=1.0, lasery=True, gain_l=1.0, prog_l=0.05, skala_l=1.0,
                          tory=1, rozm=5.0, przes=True, wiazki=True, linie=True, wektory=True, czastki=True, wybrana=True,
                          glowa=0, znak_l=0, top_l=0, alfa_l=0.85, grub=1.0)
        from laser_app.compute.common import kolory_kanalow
        self.ustaw_bufor('KC', kolory_kanalow().astype(np.float32))
        self._ostatnie = None
        self._ostatnie_sr = None

    def ustaw_podkroki(self, d, kolory, uklad=''):
        """uklad: '' = surowy strumień rezydualny, '_c' = względem środka chmury w każdym stanie, '_n' = '_c' w jednej skali
        na checkpoint (nie LayerNorm). W '_c'/'_n' siły są wycentrowane jak położenia (A_ij − ā_j, z_jc − z̄_c, bez biasów),
        więc suma rysowanych wektorów = rysowany ruch cząsteczki (dokładnie, jak w ViT)."""
        X = np.zeros((13 * TOK, 4), np.float32); X[:, :3] = d['pos' + uklad].reshape(-1, 3)
        klucz = (d.get('krok'), uklad)
        if klucz != getattr(self, '_slad_klucz', None):
            stare = getattr(self, '_X', None)
            if self.dolacz_slad and self.slad_max > 0 and stare is not None and getattr(self, '_slad_klucz', (None, None))[1] == uklad:
                self._slad = (getattr(self, '_slad', []) + [stare])[-self.slad_max:]
            else:
                self._slad = []
            self._slad_klucz = klucz; self._X = X
            self.n_slad = 13 * len(self._slad)
            self.ustaw_bufor('TR', np.concatenate(self._slad, 0) if self._slad else np.zeros((1, 4), np.float32))
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
        self._z = d['z_c'] if srodek else d['z']                     # do wyboru najsilniejszych laserów (top_l) na CPU
        for n, a in dane:
            self.ustaw_bufor(n, np.ascontiguousarray(a, np.float32))
        self.K = 13

    def ustaw_podkrok(self, s):
        self.s = float(s); self.update()

    def _prog_laserow(self, b):
        """Próg |z| laserów bloku b dla wybranej cząsteczki: opcja prog_l albo, przy top_l > 0, |z| top_l-tego najsilniejszego
        lasera (po filtrze znaku), żeby świeciło najwyżej top_l wiązek."""
        o = self.opcje; prog = float(o['prog_l']); n = int(o.get('top_l', 0))
        z = getattr(self, '_z', None)
        if n <= 0 or z is None:
            return prog
        z = np.asarray(z[b, int(o['tok'])], np.float64)
        znak = int(o.get('znak_l', 0))
        a = np.abs(z[z > 0] if znak == 1 else z[z < 0] if znak == -1 else z) * float(o['gain_l'])
        if a.size <= n:
            return prog
        return max(prog, float(np.partition(a, a.size - n)[a.size - n]) * (1 - 1e-6))

    def _programy(self):
        return dict(p=_program(VS_P, FS_P))

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['p'])
        self._ui('p', u_mvp=mvp, u_s=self.s, u_tok=int(o['tok']), u_off=int(self.n_slad))
        k = int(np.clip(np.floor(self.s), 0, 11)); w_ruchu = self.s < 12.0
        wm, bm, ba = przenikanie(self.s, float(o.get('plynnie', 0.0)))   # waga laserów MLP; uwaga ma 1 − wm
        uwaga = w_ruchu and wm < 0.999
        self._ui('p', u_blok1=ba + 1, u_zgas=wm)
        g = float(o.get('grub', 1.0)); wiazki = bool(o.get('wiazki', True))
        self._ui('p', u_glowa=int(o.get('glowa', 0)))
        if wiazki and o['tory'] == 2:                 # tory w głębokości wszystkich cząsteczek (słabo)
            self._ui('p', u_tryb=1, u_g=0, u_kol=(1, 1, 1, 0.10), u_punkty=0); self._addytywnie(False); GL.glLineWidth(g)
            GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, 13 + self.n_slad, TOK)
        if wiazki and o['tory'] >= 1:                 # tor wybranej cząsteczki przez wszystkie podkroki
            self._ui('p', u_tryb=1, u_g=1, u_kol=(1.0, 0.85, 0.2, 0.9), u_punkty=0); self._addytywnie(False); GL.glLineWidth(2.0 * g)
            GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, 13 + self.n_slad, TOK)
        if wiazki and w_ruchu and o['przes']:
            self._ui('p', u_tryb=5, u_punkty=0); self._addytywnie(False); GL.glLineWidth(g)
            GL.glDrawArrays(GL.GL_LINES, 0, 2 * TOK)
        uwaga = uwaga and wiazki
        if uwaga and o['sily'] == 2 and o.get('linie', True):        # wszystkie pary: linie atencji
            self._ui('p', u_tryb=2, u_g=0, u_gain=float(o['gain']), u_prog=float(o['prog']), u_kol=(0.35, 0.8, 1.0, 1.0), u_punkty=0)
            self._addytywnie(True); GL.glLineWidth(g); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK * TOK)
        if uwaga and o['sily'] >= 1 and o.get('linie', True):        # wybrana cząsteczka: kogo słucha (linie)
            self._ui('p', u_tryb=2, u_g=1, u_gain=float(o['gain']) * 20, u_prog=float(o['prog']), u_kol=(0.35, 0.8, 1.0, 1.0), u_punkty=0)
            self._addytywnie(True); GL.glLineWidth(g); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
        if uwaga and o['sily'] >= 1 and o.get('wektory', True):      # siły f_ij na wybraną cząsteczkę (wektory)
            self._ui('p', u_tryb=3, u_skala_sil=float(o['skala_sil']), u_gain=float(o['gain']) * 20, u_prog=float(o['prog']), u_punkty=0)
            self._addytywnie(False); GL.glLineWidth(1.8 * g); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
        if wiazki and w_ruchu and wm > 1e-3 and o['lasery']:
            self._ui('p', u_tryb=4, u_gain=float(o['gain_l']), u_prog=self._prog_laserow(bm), u_skala_sil=float(o['skala_l']), u_punkty=0,
                     u_blok1=bm + 1, u_zgas=1.0 - wm, u_znak=int(o.get('znak_l', 0)), u_alfa=float(o.get('alfa_l', 0.85)))
            self._addytywnie(False); GL.glLineWidth(1.5 * g); GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, NK)
        self._ui('p', u_blok1=0, u_zgas=0.0)
        GL.glLineWidth(1.0)
        if o.get('czastki', True):
            self._ui('p', u_tryb=0, u_g=0, u_rozm=float(o['rozm']), u_ksztalt=3, u_kol=(1, 1, 1, 1.0), u_punkty=1)
            self._addytywnie(False); GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
        if o.get('wybrana', True):
            self._ui('p', u_tryb=0, u_g=1, u_rozm=float(o['rozm']) * 4, u_ksztalt=1, u_kol=(1.0, 0.85, 0.2, 1.0), u_punkty=1)
            self._addytywnie(False); GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
