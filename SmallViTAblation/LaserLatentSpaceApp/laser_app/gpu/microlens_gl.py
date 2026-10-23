"""Okienko M: scena mikrosoczewek (MicrolensScene) rysowana w jednym widoku razem ze sferą reflektorów.

Alegoria (rozmowa 2026-10-23): zdjęcie na dole NAŚWIETLA półfabrykaty 600 mikrosoczewek przez filtry łodygi (4 konwolucje);
kształt mikrosoczewki = położenie koloidu w 192 wymiarach (w sferze: formują je lasery MLP i gwiazda uwagi, 12 podkroków);
na górze jednostajna lampa prześwietla matrycę gotowych soczewek: każda przepuszcza światło TYLKO na własną komórkę ekranu
(głowa jest na token), a jej kształt ustala pięć rzeczy: transmisję (ciepło = pewność) i prostokąt plamki (l, t, r, b).
Ekran = mapa ciepła 20 × 30 z ramką najjaśniejszej komórki; kalka = cel z etykiety (Gauss wokół środka narzędzia, ramka GT).

Geometria sceny (jednostki sfery, R = 34, reflektory na promieniu 50): komórka siatki = 2 × 2, siatka 60 × 40 w płaszczyźnie XY.
Z: zdjęcie −76, płyty łodygi −68…−50, sfera w środku, matryca soczewek +60, ekran +78.
Bufory: G [13·600·5] głowa na każdym z 13 stanów (sigmoid ciepła, exp l t r b w krokach siatki); P [13·600] koloidy (skala sfery);
KT [600] kolor łaty (w = narzędzie); WY [600] wybrane; CEL [600] cel ciepła; RAM [14] ramki: 13 stanów (szczyt ciepła) + GT (px).
"""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _GPUElement, H, _KSZTALT, _LUT, _program, TOK, W

KROK_PX = 16.0                      # piksele wejścia na komórkę siatki
Z_OBRAZ, Z_SOCZ, Z_EKRAN = -76.0, 60.0, 78.0
Z_PLYTY = (-68.0, -62.0, -56.0, -50.0)                      # 4 konwolucje łodygi (nad zdjęciem, pod sferą)

_SSBO_M = """
layout(std430, binding = 0) readonly buffer BG { float G[]; };    // [13*600*5] głowa: sigmoid ciepła, l, t, r, b (komórki)
layout(std430, binding = 1) readonly buffer BP { vec4 P[]; };     // [13*600] koloid (skala sfery)
layout(std430, binding = 2) readonly buffer BK { vec4 KT[]; };    // [600] kolor łaty, w = narzędzie
layout(std430, binding = 3) readonly buffer BW { float WY[]; };   // [600] wybrane cząsteczki
layout(std430, binding = 4) readonly buffer BC { float CEL[]; };  // [600] cel ciepła (kalka z etykiety)
layout(std430, binding = 5) readonly buffer BR { vec4 RAM[]; };   // [14] ramka xyxy px: 13 stanów (szczyt), 13 = GT
"""
VS_M = "#version 430\n" + _SSBO_M + _LUT + """
uniform mat4 u_mvp;
uniform int u_tryb, u_seg, u_filtr, u_ekran, u_sonda, u_v;
uniform float u_s, u_z, u_gain, u_prog, u_gamma, u_skala, u_rozm, u_krycie;
uniform vec4 u_kol;
out vec4 v_k; out vec2 v_uv;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
const float ZO = %(zo).1f, ZS = %(zs).1f, ZE = %(ze).1f;
vec2 kom(uint j) { float c = float(j %% 30u), r = float(j / 30u); return vec2(-30.0 + 2.0 * (c + 0.5), 20.0 - 2.0 * (r + 0.5)); }
float g(uint s, uint j, uint k) { return G[(min(s, 12u) * 600u + j) * 5u + k]; }
float s_glowy() { return u_sonda == 1 ? 12.0 : clamp(u_s, 0.0, 12.0); }
float cieplo(uint j) { float s = s_glowy(); uint a = uint(floor(s)); return mix(g(a, j, 0u), g(a + 1u, j, 0u), s - float(a)); }
vec3 poz(uint j) { float s = clamp(u_s, 0.0, 12.0); uint a = uint(floor(s)); uint b = min(a + 1u, 12u);
                   return u_skala * mix(P[a * 600u + j].xyz, P[b * 600u + j].xyz, s - float(a)); }
bool odrzuc(uint j) { return u_filtr == 0 || (u_filtr == 1 && KT[j].w < 0.5) || (u_filtr == 2 && WY[j] < 0.5); }
vec2 px_scena(vec2 px) { return vec2(px.x / 8.0 - 30.0, 20.0 - px.y / 8.0); }
void main() {
    int v = gl_VertexID; uint j = uint(gl_InstanceID);
    v_k = u_kol; v_uv = vec2(0.0);
    vec2 r6[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    if (u_tryb == 0) {                        // płaszczyzna z teksturą (zdjęcie, płyty łodygi): cała siatka 60 × 40 na wysokości u_z
        vec2 q = r6[v] * vec2(30.0, 20.0); v_uv = vec2(r6[v].x * 0.5 + 0.5, 0.5 - r6[v].y * 0.5);
        gl_Position = u_mvp * vec4(q, u_z, 1.0);
    } else if (u_tryb == 1) {                 // obrys płaszczyzny (4 wierzchołki, pętla)
        vec2 r4[4] = vec2[4](vec2(-30, -20), vec2(30, -20), vec2(30, 20), vec2(-30, 20));
        gl_Position = u_mvp * vec4(r4[v], u_z, 1.0);
    } else if (u_tryb == 2) {                 // naświetlenie: komórka zdjęcia → koloid (kształt soczewki bierze się ze zdjęcia)
        if (odrzuc(j)) { gl_Position = POZA; return; }
        vec3 p = v == 0 ? vec3(kom(j), ZO + 0.2) : poz(j);
        v_k = vec4(KT[j].rgb * u_gain, u_krycie * (v == 0 ? 0.9 : 0.35));
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 3) {                 // koloid → własna mikrosoczewka w matrycy (soczewka przyjmuje kształt koloidu)
        if (odrzuc(j)) { gl_Position = POZA; return; }
        vec3 p = v == 0 ? poz(j) : vec3(kom(j), ZS - 0.2);
        float c = pow(clamp(cieplo(j), 0.0, 1.0), u_gamma);
        v_k = vec4(mix(KT[j].rgb, vec3(1.0, 0.95, 0.8), c) * u_gain, u_krycie * (v == 0 ? 0.3 : 0.9));
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 4) {                 // matryca mikrosoczewek: kółko w komórce, jasność = transmisja (ciepło)
        float c = pow(clamp(cieplo(j), 0.0, 1.0), u_gamma);
        vec2 q = kom(j) + r6[v] * u_rozm; v_uv = r6[v];
        v_k = vec4(mix(vec3(0.16, 0.22, 0.30), vec3(1.0, 0.95, 0.8), c), 0.55 + 0.45 * c);
        gl_Position = u_mvp * vec4(q, ZS, 1.0);
    } else if (u_tryb == 5) {                 // wiązki lampy: soczewka → własna komórka ekranu, jasność = ciepło (addytywnie)
        float c = cieplo(j);
        if (c < u_prog) { gl_Position = POZA; return; }
        vec3 p = vec3(kom(j), v == 0 ? ZS + 0.3 : ZE - 0.3);
        v_k = vec4(vec3(1.0, 0.92, 0.7) * pow(c, u_gamma) * u_gain, 1.0);
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 6) {                 // ekran: komórka = cień soczewki (ciepło), różnica cień − kalka, albo sama kalka
        float c = cieplo(j), cel = CEL[j]; vec3 kol;
        if (u_ekran == 1) { float d = clamp(c - cel, -1.0, 1.0); kol = d > 0.0 ? mix(vec3(0.5), vec3(1.0, 0.25, 0.15), d) : mix(vec3(0.5), vec3(0.2, 0.45, 1.0), -d); }
        else if (u_ekran == 2) kol = vec3(0.1) + 0.9 * vec3(pow(cel, u_gamma));
        else kol = lut(2, pow(c, u_gamma));
        vec2 q = kom(j) + r6[v] * 0.98; v_uv = r6[v];
        v_k = vec4(kol, 1.0);
        gl_Position = u_mvp * vec4(q, ZE, 1.0);
    } else if (u_tryb == 7) {                 // ramki na ekranie: u_seg 0 = ramka z głowy (szczyt ciepła w chwili s), 1 = GT
        vec4 b;
        if (u_seg == 1) b = RAM[13];
        else { float s = s_glowy(); uint a = uint(floor(s)); b = mix(RAM[a], RAM[min(a + 1u, 12u)], s - float(a)); }
        vec2 r5[5] = vec2[5](b.xy, vec2(b.z, b.y), b.zw, vec2(b.x, b.w), b.xy);
        gl_Position = u_mvp * vec4(px_scena(r5[v]), ZE + 0.4, 1.0);
    } else if (u_tryb == 8) {                 // zaznaczenie wybranych komórek (obrys) na wysokości u_z
        if (WY[j] < 0.5) { gl_Position = POZA; return; }
        vec2 r5[5] = vec2[5](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1), vec2(-1, -1));
        gl_Position = u_mvp * vec4(kom(j) + r5[v] * 1.05, u_z, 1.0);
    } else if (u_tryb == 9) {                 // lampa (poświata) pod matrycą albo nad zdjęciem: punkt
        gl_Position = u_mvp * vec4(0.0, 0.0, u_z, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 10) {                // promienie lampy: cztery narożniki od lampy do matrycy (światło jednostajne)
        vec2 r4[4] = vec2[4](vec2(-30, -20), vec2(30, -20), vec2(30, 20), vec2(-30, 20));
        vec3 p = v == 0 ? vec3(0.0, 0.0, u_z) : vec3(r4[gl_InstanceID], ZS - 0.5);
        v_k = vec4(u_kol.rgb, u_kol.a * (v == 0 ? 0.9 : 0.15));
        gl_Position = u_mvp * vec4(p, 1.0);
    }
}
""" % dict(zo=Z_OBRAZ, zs=Z_SOCZ, ze=Z_EKRAN)
FS_M = "#version 430\n" + _KSZTALT + """
uniform sampler2D u_rgb;
uniform int u_punkty, u_tex, u_kolo;
uniform float u_alfa;
in vec4 v_k; in vec2 v_uv; out vec4 o;
void main() {
    if (u_tex == 1) { o = vec4(texture(u_rgb, v_uv).rgb * v_k.rgb, u_alfa); return; }
    if (u_punkty == 1) { o = ksztalt(v_k); return; }
    if (u_kolo == 1) { float r = length(v_uv); if (r > 1.0) discard; o = vec4(v_k.rgb, v_k.a * (1.0 - smoothstep(0.85, 1.0, r))); return; }
    o = v_k;
}
"""


def cel_ciepla(gt, h=H, w=W):
    """Cel mapy ciepła jak `model.cele`: Gauss wokół środka ramki GT (px wejścia), 1 w komórce środka. (600,) float32; zera bez GT."""
    if gt is None:
        return np.zeros(h * w, np.float32)
    x0, y0, x1, y1 = [float(v) for v in gt]
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing='ij')
    cx, cy = (x0 + x1) / 2 / KROK_PX - .5, (y0 + y1) / 2 / KROK_PX - .5
    sx, sy = max(.7, (x1 - x0) / KROK_PX / 6), max(.7, (y1 - y0) / KROK_PX / 6)
    g = np.exp(-((xs - cx) ** 2 / (2 * sx ** 2) + (ys - cy) ** 2 / (2 * sy ** 2)))
    ix, iy = min(max(int(round(cx)), 0), w - 1), min(max(int(round(cy)), 0), h - 1)
    g[iy, ix] = 1.0
    return g.reshape(-1).astype(np.float32)


def glowa_na_komorki(glowa):
    """Logity głowy (13, 600, 5) → (13, 600, 5): sigmoid ciepła i exp(l, t, r, b) w komórkach siatki (jak `model.dekoduj`)."""
    G = np.asarray(glowa, np.float64).reshape(13, TOK, 5).copy()
    G[..., 0] = 1.0 / (1.0 + np.exp(-G[..., 0]))
    G[..., 1:] = np.exp(np.clip(G[..., 1:], -20, 20))
    return G.astype(np.float32)


def ramki_szczytu(G, w=W):
    """Ramka xyxy (px wejścia) z komórki o największym cieple w każdym z 13 stanów (G po `glowa_na_komorki`) i jej indeks."""
    R = np.zeros((13, 4), np.float32); J = np.zeros(13, np.int64)
    for s in range(13):
        j = int(np.argmax(G[s, :, 0])); J[s] = j
        cx, cy = (j % w + .5) * KROK_PX, (j // w + .5) * KROK_PX
        l, t, r, b = G[s, j, 1:] * KROK_PX
        R[s] = (cx - l, cy - t, cx + r, cy + b)
    return R, J


def kolory_lat(rgb, h=H, w=W):
    """Średni kolor każdej łaty zdjęcia (320 × 480 → 20 × 30), (600, 3) w 0..1."""
    a = np.asarray(rgb, np.float32) / 255.0
    return a.reshape(h, a.shape[0] // h, w, a.shape[1] // w, 3).mean((1, 3)).reshape(-1, 3)


class MicrolensScene(_GPUElement):
    """Zdjęcie i płyty łodygi pod sferą, matryca mikrosoczewek i ekran nad nią; koloidy w sferze rysuje osobny ReflectorSphere
    tego samego widoku (ten element dostaje tylko ich położenia P do promieni)."""
    BUFORY = ('G', 'P', 'KT', 'WY', 'CEL', 'RAM')

    def __init__(self):
        super().__init__()
        self.opcje = dict(obraz=True, plyty=True, krycie_plyt=0.55, promienie=1, jas_prom=1.0, krycie_prom=0.6, linki=1,
                          soczewki=True, lampa=True, jas_lampy=1.0, prog=0.02, ekran=0, sonda=0, gamma=1.0, ramki=True,
                          skala=1.0, kalka_obrys=True)
        self.G = None; self.J = None; self.gt = None; self.n_plyt = 0
        self.ustaw_wybor([0])

    def czysc(self):
        self.K = 0; self.G = None; self.update()

    def ustaw_wybor(self, tok):
        WY = np.zeros(TOK, np.float32); WY[[int(j) for j in tok if 0 <= int(j) < TOK]] = 1.0
        self.ustaw_bufor('WY', WY)

    def ustaw_zdjecie(self, rgb, gt, narz):
        """Zdjęcie (tekstura), kolory łat, cel ciepła i ramka GT; narz = maska tokenów narzędzia (do filtra promieni)."""
        self.ustaw_teksture('rgb', rgb)
        KT = np.ones((TOK, 4), np.float32); KT[:, :3] = kolory_lat(rgb); KT[:, 3] = np.asarray(narz, np.float32)
        self.ustaw_bufor('KT', KT)
        self.gt = None if gt is None else [float(v) for v in gt]
        self.ustaw_bufor('CEL', cel_ciepla(self.gt))

    def ustaw_przejscie(self, d, P):
        """Przejście checkpointu (history.podkroki z kluczem 'glowa') i koloidy P (13, 600, 3) w skali sfery (jak ReflectorSphere)."""
        if 'glowa' not in d:
            self.czysc(); return
        self.G = glowa_na_komorki(d['glowa'])
        self.ustaw_bufor('G', self.G.reshape(-1))
        Pq = np.zeros((13, TOK, 4), np.float32); Pq[..., :3] = np.asarray(P, np.float32)[:13, :, :3]
        self.ustaw_bufor('P', Pq.reshape(-1, 4))
        R, self.J = ramki_szczytu(self.G)
        RAM = np.zeros((14, 4), np.float32); RAM[:13] = R
        if self.gt is not None:
            RAM[13] = self.gt
        self.ustaw_bufor('RAM', RAM)
        lod = d.get('lodyga') or []
        self.n_plyt = 0
        for i, e in enumerate(lod[:4]):
            self.ustaw_teksture(f'plyta{i}', e); self.n_plyt = i + 1
        self.K = 13
        self.update()

    def opis(self, s, iou):
        """Tekst pod widokiem dla ułamkowego podkroku s (iou(ramka, gt) z zewnątrz)."""
        if self.G is None:
            return 'Mikrosoczewki: czekam na przejście przez sieć bieżącego checkpointu.'
        s = int(np.clip(round(12.0 if self.opcje.get('sonda') else s), 0, 12))
        c = self.G[s, :, 0]; j = int(self.J[s])
        opis_s = 'stan końcowy (to czyta sieć)' if s == 12 else f'sonda: głowa na stanie po podkroku {s} (sieć czyta tylko stan 12)'
        txt = (f'{opis_s};  cień: maks. ciepło {c[j]:.2f} w komórce ({j // W}, {j % W}), tło (mediana) {np.median(c):.3f};  '
               f'jasnych komórek (> 0,5): {int((c > 0.5).sum())}')
        if self.gt is not None and iou is not None:
            txt += f';  IoU ramki szczytu z GT {iou(self.ramka(s), self.gt):.2f}'
        return txt

    def ramka(self, s):
        return ramki_szczytu(self.G[s:s + 1].repeat(13, 0))[0][0] if self.G is not None else None

    def _programy(self):
        return dict(m=_program(VS_M, FS_M))

    def _tekstura(self, nazwa):
        GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex.get(nazwa, 0))

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['m'])
        self._ui('m', u_mvp=mvp, u_s=self.t, u_skala=float(o.get('skala', 1.0)), u_sonda=int(o.get('sonda', 0)), u_rgb=0, u_lut=1,
                 u_gamma=float(o.get('gamma', 1.0)), u_punkty=0, u_tex=0, u_kolo=0)
        self._addytywnie(False)
        # --- zdjęcie i płyty łodygi (od dołu)
        if o.get('obraz', True):
            self._tekstura('rgb')
            self._ui('m', u_tryb=0, u_z=Z_OBRAZ, u_tex=1, u_alfa=1.0, u_kol=(1, 1, 1, 1)); GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            self._ui('m', u_tryb=1, u_z=Z_OBRAZ, u_tex=0, u_kol=(0.8, 0.85, 0.9, 0.9)); GL.glLineWidth(1.2)
            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 4)
            if o.get('plyty', True):
                al = float(o.get('krycie_plyt', 0.55))
                for i in range(self.n_plyt):
                    self._tekstura(f'plyta{i}')
                    self._ui('m', u_tryb=0, u_z=Z_PLYTY[i], u_tex=1, u_alfa=al, u_kol=(1, 1, 1, 1)); GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
                    self._ui('m', u_tryb=1, u_z=Z_PLYTY[i], u_tex=0, u_kol=(0.55, 0.7, 0.85, 0.7)); GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 4)
            self._tekstura('rgb')
        # --- promienie: naświetlenie (zdjęcie → koloid) i linki (koloid → soczewka)
        filtr = int(o.get('promienie', 1))
        if filtr:
            self._addytywnie(True); GL.glLineWidth(1.0)
            self._ui('m', u_tryb=2, u_filtr=filtr, u_gain=float(o.get('jas_prom', 1.0)), u_krycie=float(o.get('krycie_prom', 0.6)))
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
            self._addytywnie(False)
        linki = int(o.get('linki', 1))
        if linki:
            self._addytywnie(True); GL.glLineWidth(1.0)
            self._ui('m', u_tryb=3, u_filtr=linki, u_gain=float(o.get('jas_prom', 1.0)), u_krycie=float(o.get('krycie_prom', 0.6)))
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
            self._addytywnie(False)
        # --- matryca mikrosoczewek
        if o.get('soczewki', True):
            self._ui('m', u_tryb=1, u_z=Z_SOCZ, u_kol=(0.5, 0.65, 0.8, 0.8)); GL.glLineWidth(1.2); GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 4)
            self._ui('m', u_tryb=4, u_rozm=0.85, u_kolo=1); GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, 6, TOK)
            self._ui('m', u_kolo=0)
            self._ui('m', u_tryb=8, u_z=Z_SOCZ + 0.2, u_kol=(1.0, 0.85, 0.2, 1.0)); GL.glLineWidth(2.0)
            GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, 5, TOK)
        # --- lampa i jej wiązki przez soczewki na ekran
        if o.get('lampa', True):
            self._addytywnie(True)
            self._ui('m', u_tryb=9, u_z=Z_SOCZ - 12.0, u_rozm=42.0, u_ksztalt=0, u_kol=(1, 1, 1, 0.9), u_punkty=1)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)
            self._ui('m', u_punkty=0, u_tryb=10, u_z=Z_SOCZ - 12.0, u_kol=(1.0, 0.95, 0.8, 0.5)); GL.glLineWidth(1.0)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, 4)
            self._ui('m', u_tryb=5, u_prog=float(o.get('prog', 0.02)), u_gain=float(o.get('jas_lampy', 1.0))); GL.glLineWidth(1.6)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
            self._addytywnie(False)
        # --- ekran: cień, kalka, ramki
        self._ui('m', u_tryb=6, u_ekran=int(o.get('ekran', 0))); GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, 6, TOK)
        self._ui('m', u_tryb=1, u_z=Z_EKRAN, u_kol=(0.85, 0.9, 1.0, 0.9)); GL.glLineWidth(1.4); GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 4)
        if o.get('ramki', True):
            if self.gt is not None:
                self._ui('m', u_tryb=7, u_seg=1, u_kol=(1.0, 1.0, 1.0, 0.9)); GL.glLineWidth(1.4); GL.glDrawArrays(GL.GL_LINE_STRIP, 0, 5)
            self._ui('m', u_tryb=7, u_seg=0, u_kol=(0.29, 0.94, 1.0, 1.0)); GL.glLineWidth(2.2); GL.glDrawArrays(GL.GL_LINE_STRIP, 0, 5)
        self._ui('m', u_tryb=8, u_z=Z_EKRAN + 0.3, u_kol=(1.0, 0.85, 0.2, 1.0)); GL.glLineWidth(2.0)
        GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, 5, TOK)
        GL.glLineWidth(1.0)
