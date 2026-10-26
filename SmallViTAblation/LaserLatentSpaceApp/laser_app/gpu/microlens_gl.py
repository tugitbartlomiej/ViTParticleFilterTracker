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
layout(std430, binding = 6) readonly buffer BF { vec4 FL[]; };    // [600] pole laserów bloku na koloid: siła (xy, względna), |F|
layout(std430, binding = 7) readonly buffer BL { vec4 LT[]; };    // [600*K*2] najsilniejsze lasery: (kierunek xy, |z| wzgl., znak), kolor
layout(std430, binding = 8) readonly buffer BJ { vec4 JK[]; };    // [600*3] pole wokół koloidu z jakobianu (względne):
                                                                  //   [S00 S01 S02 ω0], [S11 S12 S22 ω1], [ω2 |f| κ 0]
                                                                  //   κ = kontrakcja uwagi bloku / 95. percentyl |κ| (−1.5…1.5)
"""
VS_M = "#version 430\n" + _SSBO_M + _LUT + """
uniform mat4 u_mvp;
uniform int u_tryb, u_seg, u_filtr, u_ekran, u_sonda, u_v;
uniform float u_s, u_z, u_gain, u_prog, u_gamma, u_skala, u_rozm, u_krycie;
uniform float u_czas, u_tempo, u_lep, u_rozm_pkt;   // super szkło: czas animacji [s], tempo krążenia, lepkość, wielkość punktu
uniform sampler2D u_rgb;                  // zdjęcie (kolor promieni skupianych przez koloidy)
uniform int u_nb;                         // promieni w pęku na koloid
uniform float u_ap;
uniform float u_wzm;
uniform int u_zal;                        // 1: wiązki wychodzące z koloidów łamią się na powierzchni szklanej sfery
uniform float u_n;                        // współczynnik załamania szkła sfery                      // wzmocnienie wiązek wychodzących z koloidów (ogniskowanie)                       // apertura koloidu: promień pęku [jednostki sfery]
uniform int u_pole, u_nk;                 // pole laserów formuje soczewki: 1 = kształt soczewek, u_nk laserów na soczewkę
uniform float u_def, u_dl;                // siła odkształcenia soczewki, długość wiązek laserów
uniform int u_ma_jk, u_gest;              // pole z jakobianu dostępne; 1 = maź: rysowanie gęstości
uniform float u_maz_r, u_maz_def, u_maz_wir;   // maź: promień, odkształcenie polem (siodło), tempo wiru
uniform vec2 u_vp;                        // rozmiar widoku [px] (wielkość drobin mazi w świecie)
uniform float u_sobr, u_R;               // u_R: promień szklanej sfery (wejście wiązek z oryginalnej łaty)
uniform int u_jas_zdj;                    // 1: jasność promieni ze zdjęcia (ilustracja), 0: tylko kolor łaty, jasność stała
uniform int u_skup_z;                     // skupianie mazi: 0 = kontrakcja uwagi κ (JK[.].z; opcja skup_z 0/1), 1 = ciepło głowy (odczyt)
// u_sobr: skala podstawy kolumny                     // skala podstawy kolumny: zdjęcie, płyty, matryca, ekran (1 = siatka 60 × 40)
uniform vec4 u_kol;
out vec4 v_k; out vec2 v_uv;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
const float ZO = %(zo).1f, ZS = %(zs).1f, ZE = %(ze).1f, ZP = %(zp).1f;
vec2 kom(uint j) { float c = float(j %% 30u), r = float(j / 30u); return vec2(-30.0 + 2.0 * (c + 0.5), 20.0 - 2.0 * (r + 0.5)); }
float g(uint s, uint j, uint k) { return G[(min(s, 12u) * 600u + j) * 5u + k]; }
float s_glowy() { return u_sonda == 1 ? 12.0 : clamp(u_s, 0.0, 12.0); }
float cieplo(uint j) { float s = s_glowy(); uint a = uint(floor(s)); return mix(g(a, j, 0u), g(a + 1u, j, 0u), s - float(a)); }
vec3 poz(uint j) { float s = clamp(u_s, 0.0, 12.0); uint a = uint(floor(s)); uint b = min(a + 1u, 12u);
                   return u_skala * mix(P[a * 600u + j].xyz, P[b * 600u + j].xyz, s - float(a)); }
bool odrzuc(uint j) { return u_filtr == 0 || (u_filtr == 1 && KT[j].w < 0.5) || (u_filtr == 2 && WY[j] < 0.5); }
vec3 jas_laty(vec3 k) {                    // kolor promienia z łaty: u_jas_zdj 0 = barwa łaty o stałej luminancji 0,5 (sieć nie przenosi
    if (u_jas_zdj == 1) return k;          // jasności łat: łata wchodzi do konwolucji, nie do szkła), 1 = jak na zdjęciu (ilustracja)
    float l = dot(k, vec3(0.2126, 0.7152, 0.0722));
    return l > 1e-3 ? min(k * (0.5 / l), vec3(2.0)) : vec3(0.5);
}
float dL(vec3 q, vec3 T, vec3 u1, vec3 u2, float f) {   // pochodna drogi optycznej po kącie na okręgu wielkim (płaszczyzna padania)
    vec3 e = u_R * (cos(f) * u1 + sin(f) * u2), de = u_R * (-sin(f) * u1 + cos(f) * u2);
    return dot(u_n * normalize(e - q) + normalize(e - T), de);
}
vec3 wyjscie_ze_szkla(vec3 q, vec3 T) {     // punkt wyjścia ze szklanej sfery (promień u_R, współczynnik u_n) drogi q → T: prawo Snella,
    if (u_zal == 0 || dot(q, q) >= u_R * u_R || dot(T, T) <= u_R * u_R) return q;   // zero dL/dφ najbliższe przecięcia prostej q → T
    vec3 u1 = length(q) > 1e-6 * u_R ? normalize(q) : normalize(T);
    vec3 w = T - dot(T, u1) * u1; vec3 u2 = length(w) > 1e-9 ? normalize(w) : vec3(0.0, 0.0, 1.0);
    float fT = atan(dot(T, u2), dot(T, u1));
    vec3 d = normalize(T - q); float b = dot(q, d), c = dot(q, q) - u_R * u_R;
    vec3 el = q + (-b + sqrt(max(b * b - c, 0.0))) * d; float fl = atan(dot(el, u2), dot(el, u1));
    float a0 = -0.5, a1 = fT + 0.5, fp = a0, hp = dL(q, T, u1, u2, a0), lo = 0.0, hi = 0.0, hlo = 0.0; bool jest = false;
    for (int k = 1; k <= 48; k++) {
        float f = a0 + (a1 - a0) * float(k) / 48.0, hf = dL(q, T, u1, u2, f);
        if (hp * hf <= 0.0 && (!jest || abs(0.5 * (fp + f) - fl) < abs(0.5 * (lo + hi) - fl))) { lo = fp; hi = f; hlo = hp; jest = true; }
        hp = hf; fp = f;
    }
    if (!jest) return el;
    for (int i = 0; i < 22; i++) {
        float m = 0.5 * (lo + hi), hm = dL(q, T, u1, u2, m);
        if (hlo * hm <= 0.0) hi = m; else { lo = m; hlo = hm; }
    }
    float f = 0.5 * (lo + hi);
    return u_R * (cos(f) * u1 + sin(f) * u2);
}
float dLw(vec3 q, vec3 u1, float f) {     // pochodna drogi optycznej fali płaskiej z dołu: e_z + n·|q − e|, e na okręgu (u1, z)
    vec3 ez = vec3(0.0, 0.0, 1.0), e = u_R * (cos(f) * u1 + sin(f) * ez), de = u_R * (-sin(f) * u1 + cos(f) * ez);
    return dot(ez + u_n * normalize(e - q), de);
}
vec3 wejscie_do_szkla(vec3 q) {             // punkt wejścia pionowego promienia w dolną półkulę, z którego po załamaniu (Snell) promień
    if (u_zal == 0 || dot(q, q) >= u_R * u_R) return vec3(q.xy, -1e4);              // dochodzi do q; poza sferą / bez załamania:
    vec2 r2 = q.xy; float rq = length(r2);                                             // z = −1e4 = „pionowo prosto do q”
    vec3 u1 = rq > 1e-6 ? vec3(r2 / rq, 0.0) : vec3(1.0, 0.0, 0.0);
    float fl = atan(-sqrt(max(u_R * u_R - rq * rq, 0.0)), rq);                         // pionowa przez q trafia tu w półkulę
    float a0 = -3.1316, a1 = -0.01, fp = a0, hp = dLw(q, u1, a0), lo = 0.0, hi = 0.0, hlo = 0.0; bool jest = false;
    for (int k = 1; k <= 48; k++) {
        float f = a0 + (a1 - a0) * float(k) / 48.0, hf = dLw(q, u1, f);
        if (hp * hf <= 0.0 && (!jest || abs(0.5 * (fp + f) - fl) < abs(0.5 * (lo + hi) - fl))) { lo = fp; hi = f; hlo = hp; jest = true; }
        hp = hf; fp = f;
    }
    if (!jest) return vec3(q.xy, -1e4);
    for (int i = 0; i < 22; i++) {
        float m = 0.5 * (lo + hi), hm = dLw(q, u1, m);
        if (hlo * hm <= 0.0) hi = m; else { lo = m; hlo = hm; }
    }
    float f = 0.5 * (lo + hi);
    return u_R * (cos(f) * u1 + sin(f) * vec3(0.0, 0.0, 1.0));
}
vec2 komS(uint j) { return kom(j) * u_sobr; }
mat3 S_pola(uint j) { vec4 a = JK[3u * j], b = JK[3u * j + 1u]; return mat3(a.x, a.y, a.z, a.y, b.x, b.y, a.z, b.y, b.z); }
vec3 om_pola(uint j) { return vec3(JK[3u * j].w, JK[3u * j + 1u].w, JK[3u * j + 2u].x); }
vec3 obroc(vec3 v, vec3 k, float a) { return v * cos(a) + cross(k, v) * sin(a) + k * dot(k, v) * (1.0 - cos(a)); }   // Rodrigues
float px_swiat(vec3 p, float r) {                 // promień kuli r wokół p w pikselach (największy z trzech osi świata)
    vec4 c0 = u_mvp * vec4(p, 1.0); vec2 n0 = c0.xy / c0.w; float m = 0.0;
    for (int i = 0; i < 3; i++) { vec3 d = vec3(0.0); d[i] = r; vec4 c = u_mvp * vec4(p + d, 1.0); m = max(m, length((c.xy / c.w - n0) * 0.5 * u_vp)); }
    return m;
}                 // komórka w skali podstawy kolumny
float hasz(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return float(x) / 4294967295.0; }
vec2 px_scena(vec2 px) { return vec2(px.x / 8.0 - 30.0, 20.0 - px.y / 8.0); }
vec2 srodek_ramki(uint j) {                 // środek ramki przewidzianej przez głowę tokenu j [px wejścia] (l, t, r, b od środka komórki)
    float s = s_glowy(); uint a = uint(floor(s)); uint b = min(a + 1u, 12u); float f = s - float(a);
    float l = mix(g(a, j, 1u), g(b, j, 1u), f), t = mix(g(a, j, 2u), g(b, j, 2u), f);
    float r = mix(g(a, j, 3u), g(b, j, 3u), f), bo = mix(g(a, j, 4u), g(b, j, 4u), f);
    vec2 c = (vec2(float(j %% 30u), float(j / 30u)) + 0.5) * 16.0;
    return c + 8.0 * vec2(r - l, bo - t);
}
void main() {
    int v = gl_VertexID; uint j = uint(gl_InstanceID);
    v_k = u_kol; v_uv = vec2(0.0);
    vec2 r6[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    if (u_tryb == 0) {                        // płaszczyzna z teksturą (zdjęcie, płyty łodygi): cała siatka 60 × 40 na wysokości u_z
        vec2 q = r6[v] * vec2(30.0, 20.0) * u_sobr; v_uv = vec2(r6[v].x * 0.5 + 0.5, 0.5 - r6[v].y * 0.5);
        gl_Position = u_mvp * vec4(q, u_z, 1.0);
    } else if (u_tryb == 1) {                 // obrys płaszczyzny (4 wierzchołki, pętla)
        vec2 r4[4] = vec2[4](vec2(-30, -20), vec2(30, -20), vec2(30, 20), vec2(-30, 20));
        gl_Position = u_mvp * vec4(r4[v] * u_sobr, u_z, 1.0);
    } else if (u_tryb == 2 && u_seg == 4) {   // ogniskowanie: stałe pole równoległe ze zdjęcia (pionowo, kąt nie zależy od koloidu);
        uint jj = uint(gl_InstanceID) / uint(u_nb), bb = uint(gl_InstanceID) %% uint(u_nb);   // koloid stoi w polu, zbiera pęk spod
        if (odrzuc(jj)) { gl_Position = POZA; return; }                                      // siebie i wypuszcza go pod własnym kątem
        vec3 k = poz(jj);                                                                    // w środek ramki przewidzianej przez głowę
        float fi = 6.2832 * (float(bb) + 0.5) / float(u_nb) + 0.7 * float(jj);
        vec2 xy = k.xy + (u_nb > 1 ? u_ap * vec2(cos(fi), sin(fi)) : vec2(0.0));
        vec2 cel = px_scena(srodek_ramki(jj)) * u_sobr;                                       // ognisko na ekranie (px → scena)
        vec3 q = vec3(xy, k.z), T = vec3(cel, ZE - 0.3), e = wyjscie_ze_szkla(q, T);         // załamanie na wyjściu ze sfery
        vec3 w = wejscie_do_szkla(q); bool cien = w.z < -1e3 && u_zal == 1 && dot(q, q) < u_R * u_R;   // cień kuli: fala tu nie dociera
        if (w.z < -1e3) w = q;
        if (cien && v < 4) { gl_Position = POZA; return; }
        vec2 xz = w.xy;                                                                       // pionowy promień zdjęcia nad punktem wejścia
        vec3 p = v == 0 ? vec3(xz, ZO + 0.2) : (v == 1 || v == 2) ? w : (v == 3 || v == 4) ? q : (v == 5 || v == 6) ? e : T;
        vec2 uv = vec2(xz.x / (60.0 * u_sobr) + 0.5, 0.5 - xz.y / (40.0 * u_sobr));
        bool poza = any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)));
        vec3 kol = poza ? vec3(0.0) : jas_laty(textureLod(u_rgb, uv, 0.0).rgb);              // kolor łaty (jasność tylko z opcją)
        float c = pow(clamp(cieplo(jj), 0.0, 1.0), u_gamma);                                  // transmisja koloidu = ciepło
        float sk = u_gain * u_krycie * 3.0 / float(u_nb);
        // wejście przygaszone (stałe pole), wyjście ze WSZYSTKICH koloidów: ciemne przygaszone, jasne (ciepło) rozjaśnione
        v_k = vec4((v < 4 ? kol * sk * 0.35 : mix(kol, vec3(1.0, 0.95, 0.8), c) * sk * u_wzm * (0.05 + 0.95 * c)) * (cien ? 0.15 : 1.0), 1.0);
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 13) {                // szklane kulki koloidów: punkt w miejscu koloidu, zabarwienie łatą, ciepło w v_uv.x
        j = uint(v);                          // rysowane bez instancji: koloid = numer wierzchołka
        if (odrzuc(j)) { gl_Position = POZA; return; }
        float c = pow(clamp(cieplo(j), 0.0, 1.0), u_gamma);
        v_k = vec4(mix(KT[j].rgb, vec3(1.0, 0.95, 0.8), 0.35 + 0.5 * c), 1.0); v_uv = vec2(c, 0.0);
        gl_Position = u_mvp * vec4(poz(j), 1.0); gl_PointSize = u_rozm_pkt * (0.85 + 0.3 * c);
    } else if (u_tryb == 14) {                // wiązki laserów formujące soczewkę: wchodzą z kierunku −v_c i pchają wzdłuż v_c
        uint jj = uint(gl_InstanceID) / uint(u_nk);
        if (odrzuc(jj)) { gl_Position = POZA; return; }
        vec4 L0 = LT[2 * gl_InstanceID], L1 = LT[2 * gl_InstanceID + 1];
        if (L0.z < u_prog) { gl_Position = POZA; return; }
        float r = u_rozm * u_sobr; vec2 c = komS(jj), d = L0.xy;
        vec2 q = v == 0 ? c - d * (r + u_dl * L0.z * r * 3.0) : c - d * r * 0.15;
        vec3 kol = L0.w < 0.0 ? vec3(0.30, 0.85, 1.0) : 0.25 + 0.75 * L1.rgb;      // ujemna jasność na błękitno (jak na sferze)
        v_k = vec4(kol * (0.25 + 0.75 * L0.z) * u_gain * (v == 0 ? 0.25 : 1.0), 1.0);
        gl_Position = u_mvp * vec4(q, ZS + 0.15, 1.0);
    } else if (u_tryb == 16) {                // maź (kropla w ścinaniu): drobiny w elipsoidzie pola wokół koloidu (S rozciąga i ściska, ω obraca),
        uint jj = uint(gl_InstanceID);        // rozciągnięte ruchem w podkroku (uwaga albo MLP); gęstość sumowana w buforze, potem powierzchnia
        if (odrzuc(jj)) { gl_Position = POZA; return; }
        uint kk = uint(v), q0 = jj * 977u + kk * 13u;
        float h1 = hasz(q0 + 1u), h2 = hasz(q0 + 2u), h3 = hasz(q0 + 3u), h4 = hasz(q0 + 4u);
        vec3 sb = normalize(vec3(h1, h2, h3) * 2.0 - 1.0 + 1e-4) * pow(h4, 0.333);   // równomiernie w kuli
        vec3 o = sb * u_maz_r;
        if (u_ma_jk == 1) {
            mat3 S = S_pola(jj); vec3 om = om_pola(jj);
            o += u_maz_def * (S * o);                                           // siodło: rozciąga wzdłuż λ > 0, ściska wzdłuż λ < 0
            float w = length(om);
            if (w > 1e-6) o = obroc(o, om / w, u_czas * u_maz_wir * w * (0.6 + 0.8 * h2));   // wir: obrót wokół osi ω
        }
        vec3 c = poz(jj);
        float s = clamp(u_s, 0.0, 12.0); uint a = uint(floor(s)); uint b = min(a + 1u, 12u);
        vec3 d = u_skala * (P[b * 600u + jj].xyz - P[a * 600u + jj].xyz);   // ruch w podkroku (uwaga albo lasery)
        float dl = length(d); vec3 e = dl > 1e-6 ? d / dl : vec3(0.0, 0.0, 1.0);
        o += e * dot(o, e) * u_lep * min(dl / max(u_maz_r, 1e-6), 3.0);
        o -= d * u_lep * 0.5 * h1;
        float ka = u_ma_jk == 1 ? JK[3u * jj + 2u].z : 0.0;                 // κ: kontrakcja uwagi tokenu (znormalizowana)
        float cc = u_skup_z == 0 ? clamp(ka, 0.0, 1.5) : pow(clamp(cieplo(jj), 0.0, 1.0), u_gamma);   // świeci tylko dodatnia κ
        float rk = u_skup_z == 0 ? 1.0 - 0.3 * clamp(ka, -1.0, 0.0) : 1.0;  // oddalanie od sąsiadów: drobina szersza (maź rozmyta)
        v_k = vec4(mix(KT[jj].rgb, vec3(1.0, 0.95, 0.8), min(cc, 1.0)), 0.45 + 0.55 * cc); v_uv = vec2(cc, 0.0);   // v_uv.x: skupianie
        vec3 pw = c + o;
        gl_Position = u_mvp * vec4(pw, 1.0); gl_PointSize = clamp(2.0 * px_swiat(pw, u_maz_r * 0.55 * rk), 2.0, 160.0);
    } else if (u_tryb == 12) {                // stałe pole równoległe: pionowe promienie z każdej łaty zdjęcia, niezależne od koloidów;
        vec2 xy = komS(j);                    // kończą się na powierzchni szklanej sfery (dalej biegnie tylko światło przez koloidy)
        vec2 uv = vec2(xy.x / (60.0 * u_sobr) + 0.5, 0.5 - xy.y / (40.0 * u_sobr));
        v_k = vec4(textureLod(u_rgb, uv, 0.0).rgb * u_gain, 1.0);
        float r2 = u_R * u_R - dot(xy, xy);
        float zk = r2 > 0.0 ? -sqrt(r2) : 0.0;                       // wejście w dolną półkulę; poza obrysem sfery: do równika
        gl_Position = u_mvp * vec4(xy, v == 0 ? ZO + 0.2 : max(zk, ZP + 0.3), 1.0);
    } else if (u_tryb == 2 && u_seg == 3) {   // skupianie: pęk u_nb równoległych promieni ze zdjęcia pod koloidem (apertura u_ap)
        uint jj = uint(gl_InstanceID) / uint(u_nb), bb = uint(gl_InstanceID) %% uint(u_nb);   // biegnie pionowo przez płyty do
        if (odrzuc(jj)) { gl_Position = POZA; return; }                                      // płaszczyzny koloidu, a koloid
        vec3 k = poz(jj);                                                                    // skupia go w ognisku: swojej soczewce
        float fi = 6.2832 * (float(bb) + 0.5) / float(u_nb) + 0.7 * float(jj);
        vec2 xy = k.xy + (u_nb > 1 ? u_ap * vec2(cos(fi), sin(fi)) : vec2(0.0));
        vec3 q = vec3(xy, k.z), T = vec3(komS(jj), ZS - 0.2), e = wyjscie_ze_szkla(q, T);
        vec3 w = wejscie_do_szkla(q); bool cien = w.z < -1e3 && u_zal == 1 && dot(q, q) < u_R * u_R;
        if (w.z < -1e3) w = q; vec2 xz = w.xy;
        if (cien && v < 4) { gl_Position = POZA; return; }
        vec3 p = v == 0 ? vec3(xz, ZO + 0.2) : (v == 1 || v == 2) ? w : (v == 3 || v == 4) ? q : (v == 5 || v == 6) ? e : T;
        vec2 uv = vec2(xz.x / (60.0 * u_sobr) + 0.5, 0.5 - xz.y / (40.0 * u_sobr));         // światło z tego miejsca zdjęcia
        bool poza = any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)));
        vec3 kol = poza ? vec3(0.0) : jas_laty(textureLod(u_rgb, uv, 0.0).rgb);
        v_k = vec4(kol * u_gain * u_krycie * (v < 4 ? 0.9 : 0.7) * 3.0 / float(u_nb) * (cien ? 0.15 : 1.0), 1.0);   // addytywnie
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 2) {                 // naświetlenie: wiązka równoległa (pionowo) przez zdjęcie i płyty łodygi,
        if (odrzuc(j)) { gl_Position = POZA; return; }   // nad ostatnią płytą biegnie do koloidu, a na koloidzie się załamuje (link)
        vec3 k = poz(j);
        // u_seg 2: z własnej łaty oryginalnego zdjęcia pionowo (równolegle) przez płyty aż do wejścia w szklaną sferę (promień u_R),
        //         tam załamanie do koloidu (jak wiązki reflektorów); najpóźniej tuż pod koloidem, żeby wiązka nie szła w dół;
        // u_seg 1: z punktu zdjęcia pod koloidem pionowo prosto na koloid (zdjęcie zmniejszone pod chmurę)
        vec2 xy = u_seg == 1 ? k.xy : komS(j);
        float zw = ZP + 0.3;
        vec3 wy = vec3(xy, zw);
        if (u_seg == 2) {                     // jedna wiązka: pionowo z oryginalnego zdjęcia, załamanie przy wejściu w szkło prosto do koloidu
            vec3 w = wejscie_do_szkla(k);
            if (w.z < -1e3 && u_zal == 1 && dot(k, k) < u_R * u_R) { gl_Position = POZA; return; }   // cień kuli: fala tu nie dociera
            if (w.z < -1e3) w = vec3(k.xy, max(ZP + 0.3, min(-sqrt(max(u_R * u_R - dot(k.xy, k.xy), 0.0)), k.z - 1.5)));
            xy = w.xy; wy = w;
        }
        vec3 p = v == 0 ? vec3(xy, ZO + 0.2) : (v == 1 || v == 2) ? wy : k;
        vec2 uvj = vec2(xy.x / (60.0 * u_sobr) + 0.5, 0.5 - xy.y / (40.0 * u_sobr));
        vec3 kj = jas_laty(u_seg == 2 ? textureLod(u_rgb, clamp(uvj, 0.0, 1.0), 0.0).rgb : KT[j].rgb);   // jedna wiązka: kolor zdjęcia nad wejściem
        v_k = vec4(kj * u_gain * u_krycie * (v < 2 ? 0.9 : (v == 2 ? 0.7 : 0.45)), 1.0);   // addytywnie: krycie w jasności
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 3) {                 // koloid → własna mikrosoczewka w matrycy (soczewka przyjmuje kształt koloidu)
        if (odrzuc(j)) { gl_Position = POZA; return; }
        vec3 k0 = poz(j), T = vec3(komS(j), ZS - 0.2), e = wyjscie_ze_szkla(k0, T);         // załamanie na wyjściu ze sfery
        vec3 p = v == 0 ? k0 : (v == 1 || v == 2) ? e : T;
        float c = pow(clamp(cieplo(j), 0.0, 1.0), u_gamma);
        v_k = vec4(mix(KT[j].rgb, vec3(1.0, 0.95, 0.8), c) * u_gain * u_krycie * (v == 0 ? 0.3 : 0.9), 1.0);
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 4) {                 // matryca mikrosoczewek: kółko w komórce, jasność = transmisja (ciepło)
        float c = pow(clamp(cieplo(j), 0.0, 1.0), u_gamma);
        vec2 loc = r6[v] * u_rozm * u_sobr; v_uv = r6[v];
        if (u_pole == 1 && u_ma_jk == 1) {    // pole laserów z jakobianu: osie soczewki = osie rozciągania/ściskania (siodło),
            mat3 S = S_pola(j); float a = S[0][0], b = S[0][1], dd = S[1][1];   // obrót soczewki = wir pola (ω_z), animowany
            float sr = 0.5 * (a + dd), rr = sqrt(0.25 * (a - dd) * (a - dd) + b * b);
            float th = 0.5 * atan(2.0 * b, a - dd) + u_czas * u_maz_wir * om_pola(j).z;
            vec2 e1 = vec2(cos(th), sin(th)), e2 = vec2(-e1.y, e1.x);
            float r1 = clamp(1.0 + u_def * (sr + rr), 0.35, 2.2), r2 = clamp(1.0 + u_def * (sr - rr), 0.35, 2.2);
            loc = e1 * (loc.x * r1) + e2 * (loc.y * r2);
        } else if (u_pole == 1) {             // pole laserów: soczewka wydłużona i obrócona wzdłuż wypadkowej siły laserów na koloid
            vec2 F = FL[j].xy * u_def; float m = min(length(F), 1.5);
            vec2 e1 = m > 1e-6 ? normalize(F) : vec2(1.0, 0.0), e2 = vec2(-e1.y, e1.x);
            loc = e1 * (loc.x * (1.0 + m)) + e2 * (loc.y / (1.0 + 0.6 * m));
        }
        vec2 q = komS(j) + loc;
        v_k = vec4(mix(vec3(0.16, 0.22, 0.30), vec3(1.0, 0.95, 0.8), c), 0.55 + 0.45 * c);
        gl_Position = u_mvp * vec4(q, ZS, 1.0);
    } else if (u_tryb == 5) {                 // wiązki lampy: soczewka → własna komórka ekranu, jasność = ciepło (addytywnie)
        float c = cieplo(j);
        if (c < u_prog) { gl_Position = POZA; return; }
        vec3 p = vec3(komS(j), v == 0 ? ZS + 0.3 : ZE - 0.3);
        v_k = vec4(vec3(1.0, 0.92, 0.7) * pow(c, u_gamma) * u_gain, 1.0);
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 6) {                 // ekran: komórka = cień soczewki (ciepło), różnica cień − kalka, albo sama kalka
        float c = cieplo(j), cel = CEL[j]; vec3 kol;
        if (u_ekran == 1) { float d = clamp(c - cel, -1.0, 1.0); kol = d > 0.0 ? mix(vec3(0.5), vec3(1.0, 0.25, 0.15), d) : mix(vec3(0.5), vec3(0.2, 0.45, 1.0), -d); }
        else if (u_ekran == 2) kol = vec3(0.1) + 0.9 * vec3(pow(cel, u_gamma));
        else kol = lut(2, pow(c, u_gamma));
        vec2 q = komS(j) + r6[v] * 0.98 * u_sobr; v_uv = r6[v];
        v_k = vec4(kol, 1.0);
        gl_Position = u_mvp * vec4(q, ZE, 1.0);
    } else if (u_tryb == 7) {                 // ramki na ekranie: u_seg 0 = ramka z głowy (szczyt ciepła w chwili s), 1 = GT
        vec4 b;
        if (u_seg == 1) b = RAM[13];
        else { float s = s_glowy(); uint a = uint(floor(s)); b = mix(RAM[a], RAM[min(a + 1u, 12u)], s - float(a)); }
        vec2 r5[5] = vec2[5](b.xy, vec2(b.z, b.y), b.zw, vec2(b.x, b.w), b.xy);
        gl_Position = u_mvp * vec4(px_scena(r5[v]) * u_sobr, ZE + 0.4, 1.0);
    } else if (u_tryb == 8) {                 // zaznaczenie wybranych komórek (obrys) na wysokości u_z
        if (WY[j] < 0.5) { gl_Position = POZA; return; }
        vec2 r5[5] = vec2[5](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1), vec2(-1, -1));
        gl_Position = u_mvp * vec4(komS(j) + r5[v] * 1.05 * u_sobr, u_z, 1.0);
    } else if (u_tryb == 9) {                 // lampa (poświata) pod matrycą albo nad zdjęciem: punkt
        gl_Position = u_mvp * vec4(0.0, 0.0, u_z, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 10) {                // promienie lampy: cztery narożniki od lampy do matrycy (światło jednostajne)
        vec2 r4[4] = vec2[4](vec2(-30, -20), vec2(30, -20), vec2(30, 20), vec2(-30, 20));
        vec3 p = v == 0 ? vec3(0.0, 0.0, u_z) : vec3(r4[gl_InstanceID] * u_sobr, ZS - 0.5);
        v_k = vec4(u_kol.rgb, u_kol.a * (v == 0 ? 0.9 : 0.15));
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 11) {                // super szkło: drobne cząstki krążące wokół koloidu, ciągnięte za jego ruchem (maź)
        if (odrzuc(j)) { gl_Position = POZA; return; }
        uint kk = uint(v), q = j * 131u + kk * 7u;
        float h1 = hasz(q + 1u), h2 = hasz(q + 2u), h3 = hasz(q + 3u), h4 = hasz(q + 4u);
        vec3 c = poz(j);
        float s = clamp(u_s, 0.0, 12.0); uint a = uint(floor(s)); uint b = min(a + 1u, 12u);
        vec3 d = u_skala * (P[b * 600u + j].xyz - P[a * 600u + j].xyz);   // przesunięcie w bieżącym podkroku (lasery albo uwaga)
        float dl = length(d); vec3 e = dl > 1e-6 ? d / dl : vec3(0.0, 0.0, 1.0);
        vec3 t1 = normalize(abs(e.z) < 0.9 ? cross(e, vec3(0, 0, 1)) : cross(e, vec3(1, 0, 0))), t2 = cross(e, t1);
        float r = u_rozm * (0.35 + 0.65 * sqrt(h1));                          // orbita: losowa płaszczyzna, promień z hasza
        float ka = 6.2832 * h2, kb = 3.1416 * (h3 - 0.5);
        vec3 n = normalize(cos(kb) * (cos(ka) * t1 + sin(ka) * t2) + sin(kb) * e);
        vec3 u1 = normalize(cross(n, abs(n.z) < 0.9 ? vec3(0, 0, 1) : vec3(1, 0, 0))), u2 = cross(n, u1);
        float th = 6.2832 * h4 + u_czas * u_tempo * (0.6 + 0.8 * h3) / (0.5 + r / max(u_rozm, 1e-6));   // dalej = wolniej (lepkość)
        vec3 o = r * (cos(th) * u1 + sin(th) * u2);
        o += e * dot(o, e) * u_lep * min(dl / max(u_rozm, 1e-6), 3.0);       // maź rozciągnięta wzdłuż ruchu
        o -= d * u_lep * (0.25 + 0.75 * h1) * (0.5 + 0.5 * sin(th));         // ogon: cząstki zostają za koloidem
        float cc = pow(clamp(cieplo(j), 0.0, 1.0), u_gamma);
        v_k = vec4(mix(KT[j].rgb, vec3(1.0, 0.95, 0.8), cc) * u_gain, u_krycie);
        gl_Position = u_mvp * vec4(c + o, 1.0); gl_PointSize = u_rozm_pkt * (0.6 + 0.8 * h2);
    }
}
""" % dict(zo=Z_OBRAZ, zs=Z_SOCZ, ze=Z_EKRAN, zp=Z_PLYTY[-1])
FS_M = "#version 430\n" + _KSZTALT + """
uniform sampler2D u_rgb;
uniform int u_punkty, u_tex, u_kolo, u_szklo, u_gest;
uniform float u_alfa;
in vec4 v_k; in vec2 v_uv;
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 o_sk;         // tylko bufor mazi: Σ w · skupianie drobiny (inne bufory nie mają tego wyjścia)
void main() {
    o_sk = vec4(0.0);
    if (u_tex == 1) { o = vec4(texture(u_rgb, v_uv).rgb * v_k.rgb, u_alfa); return; }
    if (u_gest == 1) {                      // maź: gęstość drobiny (Gauss) z kolorem, sumowana addytywnie w buforze mazi
        vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d);
        if (r2 > 1.0) discard;
        float w = exp(-r2 * 2.5) * v_k.a; o = vec4(v_k.rgb * w, w); o_sk = vec4(v_uv.x * w, 0.0, 0.0, 0.0); return;
    }
    if (u_szklo == 1) {                     // szklana kulka: przezroczysty środek, brzeg Fresnela, połysk, ognisko światła po drugiej stronie
        vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d);
        if (r2 > 1.0) discard;
        vec3 n = vec3(d.x, -d.y, sqrt(1.0 - r2));
        float fr = pow(1.0 - n.z, 2.5);                                         // Fresnel: odbicie rośnie ku brzegowi
        float po = pow(max(dot(n, normalize(vec3(-0.45, 0.55, 0.7))), 0.0), 48.0);   // połysk od światła z góry
        float og = exp(-dot(d - vec2(0.18, 0.28), d - vec2(0.18, 0.28)) * 22.0) * (0.25 + 0.75 * v_uv.x);   // skupione światło (ognisko)
        float kr = 1.0 - smoothstep(0.92, 1.0, sqrt(r2));
        float obw = smoothstep(0.80, 0.93, sqrt(r2)) * (1.0 - smoothstep(0.93, 1.0, sqrt(r2)));   // cienka jasna obwódka szkła
        vec3 kol = v_k.rgb * (0.22 + 1.1 * fr) + vec3(0.85, 0.92, 1.0) * (0.55 * obw + po) + v_k.rgb * og * 1.6;
        o = vec4(kol, kr * clamp(0.30 + 0.70 * fr + 0.6 * obw + po + 0.6 * og, 0.0, 1.0) * u_alfa); return;
    }
    if (u_punkty == 1) { o = ksztalt(v_k); return; }
    if (u_kolo == 1) { float r = length(v_uv); if (r > 1.0) discard; o = vec4(v_k.rgb, v_k.a * (1.0 - smoothstep(0.85, 1.0, r))); return; }
    o = v_k;
}
"""

VS_PE = """#version 430
out vec2 v_uv;
void main() { vec2 p = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0; v_uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }
"""
FS_ROZMYCIE = """#version 430
uniform sampler2D u_src; uniform vec2 u_kier; uniform float u_prog_jas;
in vec2 v_uv; out vec4 o;
void main() {                                   // rozmycie Gaussa 9 próbek wzdłuż u_kier (texel); u_prog_jas > 0: tylko jasne miejsca
    const float w[5] = float[5](0.227, 0.195, 0.122, 0.054, 0.016);
    vec3 c = texture(u_src, v_uv).rgb * w[0];
    for (int i = 1; i < 5; i++) { c += texture(u_src, v_uv + u_kier * float(i)).rgb * w[i]; c += texture(u_src, v_uv - u_kier * float(i)).rgb * w[i]; }
    if (u_prog_jas > 0.0) c = max(c - vec3(u_prog_jas), vec3(0.0));
    o = vec4(c, 1.0);
}
"""
FS_SWIATLO = """#version 430
uniform sampler2D u_hdr, u_blum; uniform float u_eksp, u_sila;
in vec2 v_uv; out vec4 o;
void main() {                                   // światło: suma bez nasycenia + poświata, mapowanie tonalne (1 − e^−x), lekka gamma
    vec3 c = texture(u_hdr, v_uv).rgb * u_eksp + texture(u_blum, v_uv).rgb * u_sila;
    c = 1.0 - exp(-c); o = vec4(pow(c, vec3(0.85)), 1.0);
}
"""


FS_MAZ = """#version 430
uniform sampler2D u_dens, u_hdr, u_skt; uniform vec2 u_texel; uniform int u_ma_hdr, u_skup_z;
uniform float u_prog_m, u_krycie_m, u_skup, u_refr;
in vec2 v_uv; out vec4 o;
void main() {                                   // powierzchnia mazi z gęstości drobin: krawędź (gradient), ugięte i skupione światło wiązek
                                                // u_skup_z 0: skupianie × średnie clamp(κ, 0, 1.5) drobin w pikselu (kontrakcja uwagi)
    vec4 D = texture(u_dens, v_uv); float A = D.a;
    if (A < u_prog_m * 0.4) discard;
    float al = smoothstep(u_prog_m * 0.4, u_prog_m * 1.6, A);
    vec2 dx = vec2(u_texel.x * 2.0, 0.0), dy = vec2(0.0, u_texel.y * 2.0);
    vec2 g = vec2(texture(u_dens, v_uv + dx).a - texture(u_dens, v_uv - dx).a, texture(u_dens, v_uv + dy).a - texture(u_dens, v_uv - dy).a) / max(A, 1e-4);
    float kr = clamp(length(g) * 1.5, 0.0, 1.0);                                   // krawędź mazi (jak Fresnel)
    vec3 kol = D.rgb / max(A, 1e-6);
    vec3 L = u_ma_hdr == 1 ? texture(u_hdr, v_uv - g * u_refr * u_texel * 40.0).rgb : vec3(0.0);   // światło ugięte przez maź
    float kf = u_skup_z == 0 ? texture(u_skt, v_uv).r / max(A, 1e-4) : 1.0;       // kontrakcja: średnie skupianie drobin w pikselu
    vec3 sw = 1.0 - exp(-L * u_skup * (0.5 + min(A, 3.0)) * kf);                  // skupione: gęściej = jaśniej
    vec3 cialo = kol * (0.10 + 0.55 * kr) + vec3(0.85, 0.92, 1.0) * pow(kr, 3.0) * 0.6;
    float a = al * u_krycie_m * (0.30 + 0.55 * kr);
    o = vec4(cialo * a + sw * al, a);            // mieszanie ONE, ONE_MINUS_SRC_ALPHA (premultiplied)
}
"""


def wyjscie_ze_szkla(q, T, R, n, probki=48, bisekcje=22):
    """Ta sama procedura co `wyjscie_ze_szkla` w shaderze (wzorzec do testów): punkt e na sferze o promieniu R, w którym droga
    q (w szkle, współczynnik n) → e → T (w powietrzu) spełnia prawo Snella n·sin θ_szkło = sin θ_powietrze, czyli punkt
    stacjonarny drogi optycznej L = n·|e − q| + |T − e| (zasada Fermata; przy wyjściu ze sfery bywa to siodło, nie minimum).
    e leży na okręgu wielkim w płaszczyźnie (środek, q, T) = płaszczyzna padania; szukamy zera dL/dφ na łuku między q a T:
    próbkowanie, wybór zmiany znaku najbliższej przecięcia prostej q → T, bisekcja. q wewnątrz, T na zewnątrz; inaczej q."""
    q, T = np.asarray(q, np.float64), np.asarray(T, np.float64)
    if q @ q >= R * R or T @ T <= R * R:
        return q
    u1 = q / np.linalg.norm(q) if np.linalg.norm(q) > 1e-6 * R else T / np.linalg.norm(T)
    w = T - (T @ u1) * u1
    u2 = w / np.linalg.norm(w) if np.linalg.norm(w) > 1e-9 else np.array([0.0, 0.0, 1.0])
    fT = np.arctan2(T @ u2, T @ u1)
    d = (T - q) / np.linalg.norm(T - q); b = q @ d; c = q @ q - R * R
    el = q + (-b + np.sqrt(max(b * b - c, 0.0))) * d; fl = np.arctan2(el @ u2, el @ u1)

    def h(f):
        e = R * (np.cos(f) * u1 + np.sin(f) * u2); de = R * (-np.sin(f) * u1 + np.cos(f) * u2)
        return (n * (e - q) / np.linalg.norm(e - q) + (e - T) / np.linalg.norm(e - T)) @ de
    a0, a1 = -0.5, fT + 0.5
    best, hp, fp = None, h(a0), a0
    for k in range(1, probki + 1):
        f = a0 + (a1 - a0) * k / probki; hf = h(f)
        if hp * hf <= 0 and (best is None or abs(0.5 * (fp + f) - fl) < abs(0.5 * (best[0] + best[1]) - fl)):
            best = (fp, f, hp)
        hp, fp = hf, f
    if best is None:
        return el
    lo, hi, hlo = best
    for _ in range(bisekcje):
        m = 0.5 * (lo + hi); hm = h(m)
        if hlo * hm <= 0:
            hi = m
        else:
            lo, hlo = m, hm
    f = 0.5 * (lo + hi)
    return R * (np.cos(f) * u1 + np.sin(f) * u2)



def wejscie_do_szkla(q, R, n, probki=48, bisekcje=22):
    """Ta sama procedura co `wejscie_do_szkla` w shaderze (wzorzec do testów): punkt e na dolnej półkuli sfery o promieniu R, w którym
    pionowy promień (fala płaska z dołu, kierunek +z) łamie się (n·sin θ_szkło = sin θ_powietrze) i dochodzi do q w szkle.
    Punkt stacjonarny drogi optycznej e_z + n·|q − e| na okręgu wielkim w płaszczyźnie (oś z, q). q poza sferą: None."""
    q = np.asarray(q, np.float64)
    if q @ q >= R * R:
        return None
    rq = float(np.hypot(q[0], q[1])); u1 = np.array([q[0] / rq, q[1] / rq, 0.0]) if rq > 1e-6 else np.array([1.0, 0.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0]); fl = np.arctan2(-np.sqrt(max(R * R - rq * rq, 0.0)), rq)

    def h(f):
        e = R * (np.cos(f) * u1 + np.sin(f) * ez); de = R * (-np.sin(f) * u1 + np.cos(f) * ez)
        return (ez + n * (e - q) / np.linalg.norm(e - q)) @ de
    a0, a1 = -3.1316, -0.01
    best, hp, fp = None, h(a0), a0
    for k in range(1, probki + 1):
        f = a0 + (a1 - a0) * k / probki; hf = h(f)
        if hp * hf <= 0 and (best is None or abs(0.5 * (fp + f) - fl) < abs(0.5 * (best[0] + best[1]) - fl)):
            best = (fp, f, hp)
        hp, fp = hf, f
    if best is None:
        return None
    lo, hi, hlo = best
    for _ in range(bisekcje):
        m = 0.5 * (lo + hi); hm = h(m)
        if hlo * hm <= 0:
            hi = m
        else:
            lo, hlo = m, hm
    f = 0.5 * (lo + hi)
    return R * (np.cos(f) * u1 + np.sin(f) * ez)


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
    BUFORY = ('G', 'P', 'KT', 'WY', 'CEL', 'RAM', 'FL', 'LT', 'JK')

    def __init__(self):
        super().__init__()
        self.opcje = dict(obraz=True, plyty=True, krycie_plyt=0.55, promienie=3, rownolegle=0, pek_n=6, pek_ap=1.0, wzm_wyj=6.0, pole=True, zalamanie=True, n_szkla=1.5, wyglad=1, mz_r=2.5, mz_def=0.8, mz_wir=1.0, mz_n=48, mz_prog=0.6, mz_kr=0.7, mz_skup=1.5, mz_refr=1.0, skup_z=0, jas_zdj=False, pole_las=True, pole_def=0.6, pole_wiazki=True, pole_dla=3, pole_prog=0.05, pole_dl=1.0, pole_jas=1.0, swiatlo=True, ekspozycja=1.0, blum=1.0, blum_prog=0.15, blum_szer=2.0, grub_sw=1.6, szklo=True, szklo_dla=3, szklo_r=22.0, szklo_krycie=0.9, jas_pola=0.12, jas_prom=0.7, krycie_prom=0.25, linki=0,
                          soczewki=True, lampa=True, jas_lampy=1.0, prog=0.02, ekran=0, sonda=0, gamma=1.0, ramki=True,
                          skala=1.0, kalka_obrys=True)
        self.G = None; self.J = None; self.gt = None; self.n_plyt = 0
        self.czas = 0.0                          # czas animacji super szkła [s]
        self.n_las = 0                           # laserów na soczewkę w buforze LT (0 = brak pola)
        self.ma_jk = False                       # pole wokół koloidów z jakobianu (bufor JK)
        self.prz = False                         # powrót między checkpointami: 2 stany (G, P), bez sondy
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

    def ustaw_pole_laserow(self, F, idx, zt, kier, kol, JK=None):
        """Pole laserów bloku MLP na koloidy: F (600, 3) wypadkowa siła Σ_c (z_jc − z̄_c) v_c w rzucie sceny, idx/zt (600, K)
        najsilniejsze lasery tokenu i ich jasności, kier (768, 3) kierunki pchania v_c, kol (768, 3) kolory reflektorów;
        JK (600, 3, 4) pole z jakobianu (MicrolensMixin.pole_z_jakobianu: ścinanie S, wirowość ω, |F|, kontrakcja uwagi κ)
        formujące maź i soczewki."""
        F = np.asarray(F, np.float64); m = float(np.percentile(np.linalg.norm(F[:, :2], axis=1), 99)) or 1.0
        FL = np.zeros((TOK, 4), np.float32); FL[:, :2] = F[:, :2] / m; FL[:, 3] = np.linalg.norm(F[:, :2], axis=1) / m
        self.ustaw_bufor('FL', FL)
        K = int(idx.shape[1]); zt = np.asarray(zt, np.float64); zm = float(np.percentile(np.abs(zt), 99)) or 1.0
        d2 = np.asarray(kier, np.float64)[idx][..., :2]; d2 /= np.maximum(np.linalg.norm(d2, axis=2, keepdims=True), 1e-9)
        LT = np.zeros((TOK, K, 2, 4), np.float32)
        LT[..., 0, :2] = d2; LT[..., 0, 2] = np.clip(np.abs(zt) / zm, 0, 1.5); LT[..., 0, 3] = np.sign(zt)
        LT[..., 1, :3] = np.asarray(kol, np.float32)[idx]; LT[..., 1, 3] = 1.0
        self.ustaw_bufor('LT', LT.reshape(-1, 4)); self.n_las = K
        if JK is not None:
            self.ustaw_bufor('JK', np.ascontiguousarray(JK, np.float32).reshape(-1, 4)); self.ma_jk = True
        self.update()

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
        lod = d.get('lodyga')
        if lod is not None:                      # brak klucza: płyty zostają (mieszanie checkpointów co klatkę)
            self.n_plyt = 0
            for i, e in enumerate((lod or [])[:4]):
                self.ustaw_teksture(f'plyta{i}', e); self.n_plyt = i + 1
        # zdjęcie zmniejszone tak, żeby leżało pod chmurą koloidów (99. percentyl |x|, |y| po 13 stanach, proporcje 3 : 2)
        Pa = np.abs(np.asarray(P, np.float64)[:13, :, :2])
        self.s_obr = float(np.clip(max(np.percentile(Pa[..., 0], 99) / 30.0, np.percentile(Pa[..., 1], 99) / 20.0) * 1.08, 0.05, 2.0))
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
        return dict(m=_program(VS_M, FS_M), roz=_program(VS_PE, FS_ROZMYCIE), sw=_program(VS_PE, FS_SWIATLO), maz=_program(VS_PE, FS_MAZ))

    # ------------------------------------------------------------------ wiązki jak światło: bufor HDR, poświata, mapowanie tonalne
    @staticmethod
    def _tekstura_fbo(w, h):
        t = int(GL.glGenTextures(1)); GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        for par, war in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                         (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, par, war)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        f = int(GL.glGenFramebuffers(1)); stary = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, f)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, stary)
        return f, t

    @staticmethod
    def _tekstura_fbo_dodatkowa(f, w, h):
        """Druga tekstura RGBA16F jako COLOR_ATTACHMENT1 istniejącego FBO f (wyjście location 1 shadera: skupianie mazi); oba wyjścia
        w glDrawBuffers tego FBO (stan FBO, ustawiany raz). Zwraca (f, tekstura)."""
        t = int(GL.glGenTextures(1)); GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        for par, war in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                         (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, par, war)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        stary = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, f)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, t, 0)
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, stary)
        return f, t

    def _bufory_swiatla(self, w, h):
        if getattr(self, '_sw_rozm', None) == (w, h):
            return
        for f, t in getattr(self, '_sw_fbo', []):
            GL.glDeleteFramebuffers(1, [f]); GL.glDeleteTextures([t])
        w2, h2 = max(w // 2, 1), max(h // 2, 1)
        self._sw_fbo = [self._tekstura_fbo(w, h), self._tekstura_fbo(w2, h2), self._tekstura_fbo(w2, h2)]   # HDR, poświata ×2
        self._sw_rozm = (w, h); self._sw_rozm2 = (w2, h2)

    def _rysuj_maz(self, o, vp):
        """Maź: drobiny (gęstość z kolorem) do bufora mazi, potem powierzchnia: krawędź z gradientu gęstości, światło wiązek z bufora
        HDR ugięte wzdłuż gradientu i skupione tam, gdzie maź gęstsza; skup_z 0: skupianie × kontrakcja uwagi κ drobin (drugie wyjście
        FBO, średnia ważona gęstością), 1: jaśniej na koloidach o dużym cieple (odczyt głowy, jak dawniej)."""
        w, h = vp[2], vp[3]
        if getattr(self, '_maz_rozm', None) != (w, h):
            if getattr(self, '_maz_fbo', None):
                GL.glDeleteFramebuffers(1, [self._maz_fbo[0]]); GL.glDeleteTextures([self._maz_fbo[1], self._maz_fbo[2]])
            f, t = self._tekstura_fbo(w, h); _, t_sk = self._tekstura_fbo_dodatkowa(f, w, h)
            self._maz_fbo = (f, t, t_sk); self._maz_rozm = (w, h)
        stary = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._maz_fbo[0]); GL.glViewport(0, 0, w, h)
        GL.glClearColor(0, 0, 0, 0); GL.glClear(GL.GL_COLOR_BUFFER_BIT)   # czyści oba wyjścia: gęstość i skupianie
        GL.glUseProgram(self._prog['m']); self._addytywnie(True)
        skup_z = int(o.get('skup_z', 0))
        self._ui('m', u_tryb=16, u_gest=1, u_punkty=0, u_szklo=0, u_filtr=int(o.get('szklo_dla', 3)), u_lep=float(o.get('maz_lep', 1.0)),
                 u_skup_z=int(skup_z >= 2))                                   # 0/1 = κ (względem chmury / bezwzględna), 2 = ciepło
        GL.glDrawArraysInstanced(GL.GL_POINTS, 0, int(o.get('mz_n', 48)), TOK)
        self._ui('m', u_gest=0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, stary); GL.glViewport(*vp)
        GL.glUseProgram(self._prog['maz']); GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glActiveTexture(GL.GL_TEXTURE2); GL.glBindTexture(GL.GL_TEXTURE_2D, self._maz_fbo[1])
        ma_hdr = getattr(self, '_sw_fbo', None) is not None
        GL.glActiveTexture(GL.GL_TEXTURE4); GL.glBindTexture(GL.GL_TEXTURE_2D, self._maz_fbo[2])
        GL.glActiveTexture(GL.GL_TEXTURE3); GL.glBindTexture(GL.GL_TEXTURE_2D, self._sw_fbo[0][1] if ma_hdr else 0)
        self._ui('maz', u_dens=2, u_hdr=3, u_skt=4, u_skup_z=skup_z, u_ma_hdr=int(ma_hdr), u_prog_m=float(o.get('mz_prog', 0.6)),
                 u_krycie_m=float(o.get('mz_kr', 0.7)), u_skup=float(o.get('mz_skup', 1.5)), u_refr=float(o.get('mz_refr', 1.0)))
        GL.glUniform2f(self._u('maz', 'u_texel'), 1.0 / max(w, 1), 1.0 / max(h, 1))
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0); GL.glActiveTexture(GL.GL_TEXTURE4); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE2); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0); self._tekstura('rgb')
        GL.glUseProgram(self._prog['m']); self._addytywnie(False)

    def _swiatlo_start(self):
        """Wiązki rysowane od teraz trafiają do bufora HDR (sumowane bez nasycenia). Zwraca stan do przywrócenia."""
        vp = [int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT)]
        stary = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        self._bufory_swiatla(vp[2], vp[3])
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._sw_fbo[0][0]); GL.glViewport(0, 0, vp[2], vp[3])
        GL.glClearColor(0, 0, 0, 0); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        return vp, stary

    def _swiatlo_koniec(self, stan, o):
        """Poświata (jasne miejsca rozmyte na połowie rozdzielczości, dwa przebiegi Gaussa) i złożenie z mapowaniem tonalnym."""
        vp, stary = stan
        (_, t_hdr), (f1, t1), (f2, t2) = self._sw_fbo
        w2, h2 = self._sw_rozm2
        sila = float(o.get('blum', 1.0)); szer = float(o.get('blum_szer', 2.0))
        GL.glDisable(GL.GL_BLEND)
        GL.glUseProgram(self._prog['roz'])
        GL.glViewport(0, 0, w2, h2)
        for zrodlo, cel, kier, prog in ((t_hdr, f1, (szer / w2, 0.0), float(o.get('blum_prog', 0.15))), (t1, f2, (0.0, szer / h2), 0.0),
                                        (t2, f1, (szer * 2 / w2, 0.0), 0.0), (t1, f2, (0.0, szer * 2 / h2), 0.0)):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, cel)
            GL.glActiveTexture(GL.GL_TEXTURE2); GL.glBindTexture(GL.GL_TEXTURE_2D, zrodlo)
            self._ui('roz', u_src=2, u_prog_jas=prog); GL.glUniform2f(self._u('roz', 'u_kier'), *kier)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, stary); GL.glViewport(*vp)
        GL.glEnable(GL.GL_BLEND); self._addytywnie(True)
        GL.glUseProgram(self._prog['sw'])
        GL.glActiveTexture(GL.GL_TEXTURE2); GL.glBindTexture(GL.GL_TEXTURE_2D, t_hdr)
        GL.glActiveTexture(GL.GL_TEXTURE3); GL.glBindTexture(GL.GL_TEXTURE_2D, t2)
        self._ui('sw', u_hdr=2, u_blum=3, u_eksp=float(o.get('ekspozycja', 1.0)), u_sila=sila)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0); GL.glActiveTexture(GL.GL_TEXTURE2); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0); self._tekstura('rgb')
        GL.glUseProgram(self._prog['m']); self._addytywnie(False)

    def _tekstura(self, nazwa):
        GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex.get(nazwa, 0))

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['m'])
        self._ui('m', u_mvp=mvp, u_s=self.t, u_skala=float(o.get('skala', 1.0)), u_sonda=0 if self.prz else int(o.get('sonda', 0)), u_rgb=0, u_lut=1,
                 u_gamma=float(o.get('gamma', 1.0)), u_punkty=0, u_tex=0, u_kolo=0)
        self._addytywnie(False)
        tryb_w = int(o.get('rownolegle', 0))       # 0 ogniskowanie na ekranie, 1 skupianie w soczewce, 2 z oryginalnej łaty, 3 pod koloidem
        R = float(getattr(self, 'R', 34.0))
        sobr = min(1.0, 0.95 * R / float(np.hypot(30.0, 20.0))) if tryb_w in (0, 1, 2) else \
            float(getattr(self, 's_obr', 1.0)) * float(o.get('skala', 1.0))
        self._ui('m', u_R=R)
        self._ui('m', u_sobr=sobr)
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
        maz_on = bool(o.get('szklo', True)) and int(o.get('wyglad', 1)) == 1
        swiatlo = bool(o.get('swiatlo', True)) or maz_on              # maź skupia światło wiązek: potrzebny bufor HDR
        vp0 = [int(x) for x in GL.glGetIntegerv(GL.GL_VIEWPORT)]
        GL.glUniform2f(self._u('m', 'u_vp'), float(vp0[2]), float(vp0[3]))
        self._ui('m', u_ma_jk=int(self.ma_jk), u_czas=float(getattr(self, 'czas', 0.0)), u_maz_wir=float(o.get('mz_wir', 1.0)),
                 u_maz_r=float(o.get('mz_r', 2.5)), u_maz_def=float(o.get('mz_def', 0.8)), u_gest=0)
        stan_sw = self._swiatlo_start() if swiatlo else None        # wiązki jak światło: od tąd do złożenia trafiają do HDR
        if swiatlo:
            GL.glUseProgram(self._prog['m']); GL.glLineWidth(float(o.get('grub_sw', 1.6)))
        # --- stałe pole równoległe ze zdjęcia (kąt i położenie nie zależą od koloidów)
        if o.get('pole', True) and o.get('obraz', True):
            self._addytywnie(True); GL.glLineWidth(1.0)
            self._ui('m', u_tryb=12, u_z=Z_SOCZ - 0.5, u_gain=float(o.get('jas_pola', 0.12)))
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK)
            self._addytywnie(False)
        # --- promienie: naświetlenie (zdjęcie → koloid) i linki (koloid → soczewka)
        filtr = int(o.get('promienie', 1))
        if filtr:
            self._addytywnie(True); GL.glLineWidth(1.0)
            self._ui('m', u_tryb=2, u_seg=(4, 3, 2, 1)[min(tryb_w, 3)], u_sobr=sobr, u_nb=int(o.get('pek_n', 6)), u_ap=float(o.get('pek_ap', 1.0)),
                     u_wzm=float(o.get('wzm_wyj', 6.0)), u_filtr=filtr, u_gain=float(o.get('jas_prom', 1.0)), u_krycie=float(o.get('krycie_prom', 0.6)),
                     u_jas_zdj=int(bool(o.get('jas_zdj', False))))                 # jasność łat tylko jako ilustracja (domyślnie wyłączona)
            self._ui('m', u_zal=int(bool(o.get('zalamanie', True))), u_n=float(o.get('n_szkla', 1.5)))
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 8 if tryb_w in (0, 1) else 4, TOK * (int(o.get('pek_n', 6)) if tryb_w in (0, 1) else 1))   # zdjęcie → wejście → koloid → wyjście → cel
            self._addytywnie(False)
        linki = int(o.get('linki', 1))
        if linki:
            self._addytywnie(True); GL.glLineWidth(1.0)
            self._ui('m', u_tryb=3, u_filtr=linki, u_gain=float(o.get('jas_prom', 1.0)), u_krycie=float(o.get('krycie_prom', 0.6)),
                     u_zal=int(bool(o.get('zalamanie', True))), u_n=float(o.get('n_szkla', 1.5)))
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 4, TOK)               # koloid → wyjście ze sfery → soczewka
            self._addytywnie(False)
        if stan_sw is not None:
            self._swiatlo_koniec(stan_sw, o)
        if maz_on:
            self._rysuj_maz(o, vp0)
        # --- super szkło: cząstki krążące wokół koloidów (maź ciągnięta za ruchem w podkroku)
        if o.get('maz', False):
            self._addytywnie(True)
            szklo = bool(o.get('szklo', True))
            self._addytywnie(not szklo)
            self._ui('m', u_tryb=11, u_filtr=int(o.get('maz_filtr', 3)), u_czas=float(getattr(self, 'czas', 0.0)),
                     u_tempo=float(o.get('maz_tempo', 1.0)), u_lep=float(o.get('maz_lep', 1.0)), u_rozm=float(o.get('maz_r', 1.6)),
                     u_rozm_pkt=float(o.get('maz_pkt', 3.0)) * (2.0 if szklo else 1.0), u_gain=float(o.get('maz_jas', 0.6)) * (1.6 if szklo else 1.0),
                     u_krycie=1.0, u_punkty=1, u_ksztalt=3, u_szklo=int(szklo), u_alfa=0.8)
            GL.glDrawArraysInstanced(GL.GL_POINTS, 0, int(o.get('maz_n', 24)), TOK)
            self._ui('m', u_punkty=0, u_szklo=0)
            self._addytywnie(False)
        # --- koloidy jako szklane kulki (nad kropkami sfery)
        if o.get('szklo', True) and int(o.get('wyglad', 1)) == 0:
            self._addytywnie(False)
            self._ui('m', u_tryb=13, u_filtr=int(o.get('szklo_dla', 3)), u_rozm_pkt=float(o.get('szklo_r', 22.0)), u_szklo=1, u_punkty=1,
                     u_alfa=float(o.get('szklo_krycie', 0.9)))
            GL.glDrawArrays(GL.GL_POINTS, 0, TOK)
            self._ui('m', u_szklo=0, u_punkty=0)
        # --- matryca mikrosoczewek
        if o.get('soczewki', True):
            self._ui('m', u_tryb=1, u_z=Z_SOCZ, u_kol=(0.5, 0.65, 0.8, 0.8)); GL.glLineWidth(1.2); GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 4)
            pole = bool(o.get('pole_las', True)) and self.n_las > 0
            self._ui('m', u_tryb=4, u_rozm=0.85, u_kolo=1, u_pole=int(pole), u_def=float(o.get('pole_def', 0.6)))
            GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, 6, TOK)
            self._ui('m', u_kolo=0, u_pole=0)
            if pole and o.get('pole_wiazki', True):           # wiązki laserów formujące soczewki (addytywnie, w świetle też)
                self._addytywnie(True); GL.glLineWidth(1.4)
                self._ui('m', u_tryb=14, u_nk=self.n_las, u_rozm=0.85, u_filtr=int(o.get('pole_dla', 3)), u_prog=float(o.get('pole_prog', 0.05)),
                         u_dl=float(o.get('pole_dl', 1.0)), u_gain=float(o.get('pole_jas', 1.0)))
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, TOK * self.n_las)
                self._addytywnie(False)
            self._ui('m', u_tryb=8, u_z=Z_SOCZ + 0.2, u_kol=(1.0, 0.85, 0.2, 1.0)); GL.glLineWidth(2.0)
            GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, 5, TOK)
        # --- lampa i jej wiązki przez soczewki na ekran
        if o.get('lampa', True):
            self._addytywnie(True)
            if False:                                  # lampa zbędna: światłem jest samo zdjęcie
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
