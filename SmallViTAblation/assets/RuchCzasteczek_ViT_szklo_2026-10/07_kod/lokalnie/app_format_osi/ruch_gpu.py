"""Ruch całego zbioru liczony i rysowany na GPU (OpenGL 4.3: compute shadery GLSL 4.30 + shadery rysowania).

Zapisane stany osi czasu (float32 [cząsteczka, (x, y, z, norma 768D)]) trafiają do jednego bufora SSBO karty
(„magazyn”, S slotów po jednym stanie). Gdy cała oś mieści się w budżecie VRAM, na kartę trafiają wszystkie stany;
w przeciwnym razie magazyn działa jak pierścień: zostają stan początkowy, bieżąca para i kilka następnych, a stany
wczytuje w tle pula wątków (odczyt pliku i weryfikacja SHA256 w `ruch_zbioru.Timeline.frame`).

Kernele compute (wszystko, co zależy od cząsteczek, liczy karta):
- `stat` (przy wysłaniu stanu k): dla każdego obrazu granice osi PCA (położenie i przesunięcie od pierwszego kroku),
  maksimum normy 768D, |Δ| w PCA i prędkości względem poprzedniego zapisu -> bufor PIB[k, obraz];
- `heat` (przy wysłaniu stanu k): mapa 20 × 30, średnia normy 768D każdego patcha po wszystkich obrazach;
- `kin` (w klatce, gdy zmienią się parametry): położenie każdej cząsteczki widoku, przesunięcie od pierwszego kroku
  i wielkość koloru. W zapisanym stanie (s = 0 albo s = 1) wynik jest bitowo równy zapisowi; pomiędzy zapisami
  to jawny odcinek liniowy (stan niezapisany, oznaczany w UI);
- `bb` + `ramka`: granice osi i skale kolorów z PIB (wszystkie zapisane kroki) albo z bieżącego stanu -> środek i
  skala sceny czytane wprost przez shader wierzchołków (bez odczytu na CPU);
- `hist` + `wybor`: dokładna mediana i maksimum normy 768D w widoku (histogram bitów, bez sortowania);
- `pick`: najbliższa kliknięciu cząsteczka w pikselach ekranu (dwa przebiegi atomowe).
Rysowanie: punkty i ślady z buforów (`gl_VertexID`/`gl_InstanceID` + SSBO, bez atrybutów wierzchołków i VAO).

CPU przekazuje tylko parametry klatki (para stanów, postęp, tryby) i wysyła nowe stany. Odczyt z karty jest tylko dla
statystyk wyświetlanych w panelu (600 liczb mapy + kilka liczb mediany, raz na zmianę stanu) i dla kliknięcia.
"""
import ctypes
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt6.QtCore import pyqtSignal
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

NONE = 0xFFFFFFFF
BINS = 65536
R_HIST = 3 * BINS                       # początek wyników w buforze HIST
MAX_BLOK = 2147483647                   # typowe GL_MAX_SHADER_STORAGE_BLOCK_SIZE (NVIDIA); sprawdzane przy starcie
KOLORY = ('norma', 'predkosc', 'delta')
KADRY = ('zbior', 'widoczne', 'biezacy')

_NAGLOWEK = """#version 430
//Q16DEF
layout(std430, binding = 0) buffer BS { uint SW[]; };         // magazyn stanów: slot * N + cząsteczka (słowa)
layout(std430, binding = 10) buffer BQ { vec4 QP[]; };        // [2 * sloty] q16: (lo.xyz, norm_max), (span.xyz, 0)
layout(std430, binding = 11) buffer BMASK { uint MASK[]; };   // [N] bit 0: patch w ramce GT; bity 16-31: atencja 0..1
layout(std430, binding = 12) buffer BSEL { uint SEL[]; };     // [600] 1 = zaznaczony patch
layout(std430, binding = 13) buffer BCONF { float CONF[]; };  // [T * obrazy] pewność ViT obrazu w zapisanym kroku
layout(std430, binding = 14) buffer BPMD { vec4 PMD[]; };     // [2 * T * tokeny] średnia patcha po obrazach, maks. |odchylenie|
layout(std430, binding = 15) buffer BKG { float KG[]; };      // Koopman: [0..2047] średnie, osie, wagi; [2048..] sumy częściowe
layout(std430, binding = 1) buffer BCUR { vec4 CUR[]; };      // [N] bieżąca klatka: położenie, wielkość koloru
layout(std430, binding = 2) buffer BFR { vec4 FR[]; };        // [8] środek+skala, granice, skale kolorów
layout(std430, binding = 3) buffer BPIB { vec4 PIB[]; };      // [T * obrazy * 4] statystyki obrazu w kroku
layout(std430, binding = 4) buffer BKN { uint KNOWN[]; };     // [T] 1 = statystyki kroku policzone
layout(std430, binding = 5) buffer BHEAT { float HEAT[]; };   // [T * tokeny] średnia norma patcha po obrazach
layout(std430, binding = 6) buffer BHIST { uint HIST[]; };    // 3 x 65536 histogramów + wyniki
layout(std430, binding = 7) buffer BPICK { uint PICK[]; };    // [4] wybór punktu
layout(std430, binding = 8) buffer BSLOT { uint SLOT[]; };    // [T] slot magazynu kroku albo NONE
layout(std430, binding = 9) buffer BBB { uint BB[]; };        // [8] granice bieżącego stanu (kody ord)
const uint NONE = 0xffffffffu;
const uint BINS = 65536u;
const uint R = 3u * 65536u;
uniform uint u_N;
uint ord(float f) { uint u = floatBitsToUint(f); return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u); }
float unord(uint u) { return (u & 0x80000000u) != 0u ? uintBitsToFloat(u & 0x7fffffffu) : uintBitsToFloat(~u); }
// Stan cząsteczki i w slocie sl: f32 = 4 słowa bitowo z pliku; q16 = 2 słowa po dwie wartości uint16,
// x = lo + (q / 65535) * span, norma = (q / 65535) * norm_max (ten sam wzór co ruch_zbioru.odkwantyzuj).
vec4 stan(uint sl, uint i) {
#if Q16
    uint b = (sl * u_N + i) * 2u;
    vec2 xy = unpackUnorm2x16(SW[b]);
    vec2 zn = unpackUnorm2x16(SW[b + 1u]);
    vec4 lo = QP[2u * sl], sp = QP[2u * sl + 1u];
    return vec4(lo.xyz + vec3(xy, zn.x) * sp.xyz, zn.y * lo.w);
#else
    uint b = (sl * u_N + i) * 4u;
    return uintBitsToFloat(uvec4(SW[b], SW[b + 1u], SW[b + 2u], SW[b + 3u]));
#endif
}
// Wygładzanie ruchu w czasie (u_wyg): 0 brak (stan bitowo z pliku), 1 średnia ruchoma wyśrodkowana z u_wyg_n
// zapisów (bez opóźnienia), 2 EMA przyczynowa od początku osi: wagi α(1-α)^j, pierwszy zapis (1-α)^k, obcięta do
// u_wyg_n zapisów wstecz (waga < 1e-4) i znormalizowana. Brakujące na karcie stany są pomijane (normalizacja wag).
uniform uint u_wyg, u_wyg_n, u_T_w;
uniform float u_wyg_a;
ivec2 okno_wyg(uint k) {
    int lo = u_wyg == 1u ? int(k) - int(u_wyg_n / 2u) : int(k) - int(u_wyg_n) + 1;
    int hi = u_wyg == 1u ? int(k) + int(u_wyg_n / 2u) : int(k);
    return ivec2(max(lo, 0), min(hi, int(max(u_T_w, 1u)) - 1));
}
float waga_wyg(uint k, int q) {
    if (u_wyg == 1u) return 1.0;
    int j = int(k) - q;
    return q == 0 ? pow(1.0 - u_wyg_a, float(j)) : u_wyg_a * pow(1.0 - u_wyg_a, float(j));
}
vec4 stan_g(uint k, uint i) {
    if (u_wyg == 0u) return stan(SLOT[k], i);
    ivec2 o = okno_wyg(k);
    vec4 acc = vec4(0.0);
    float ws = 0.0;
    for (int q = o.x; q <= o.y; q++) {
        uint sl = SLOT[uint(q)];
        if (sl == NONE) continue;
        float w = waga_wyg(k, q);
        acc += w * stan(sl, i); ws += w;
    }
    return ws > 0.0 ? acc / ws : stan(SLOT[k], i);
}
// Układy widoku (u_uklad): 0 PCA 3D, 1 siatka zdjęcia (X, Y = kolumna i wiersz patcha) + cecha na Z,
// 2 przestrzeń fazowa (X = cecha, Y = jej prędkość, Z = norma zmiany 768D).
// Cecha (u_cecha): 0-2 składowe PCA (rzut wszystkich 768 cech na osie PCA), 3 norma zmiany w 768D,
// 4 odległość od środka w PCA. Przy u_ruch = 1 cechy PCA liczone względem pierwszego kroku.
uniform uint u_uklad, u_cecha, u_mozaika, u_kafle, u_tpi_w;
float cecha(vec4 st, vec4 st0, uint ruch) {
    if (u_cecha < 3u) return st[u_cecha] - (ruch == 1u ? st0[u_cecha] : 0.0);
    if (u_cecha == 3u) return st.w;
    return length(st.xyz - (ruch == 1u ? st0.xyz : vec3(0.0)));
}
// Wygląd: barwy i filtr widocznych cząsteczek (te same reguły w punktach, śladach, wyborze i zdjęciu).
uniform uint u_co, u_tylko_narz, u_tylko_zazn, u_jest_zazn, u_ka, u_img_w, u_jest_conf;
uniform uint u_att;                         // heatmapa atencji: bit 0 na cząsteczkach, bit 1 na zdjęciu
uniform float u_att_wzm, u_att_alfa;
float att(uint i) { return float(MASK[i] >> 16u) / 65535.0; }
bool narz(uint i) { return (MASK[i] & 1u) != 0u; }
// Obszar oka patcha z segmentacji (bity 1-2 MASK): 0 tło, 1 rogówka, 2 źrenica, 3 narzędzie. Bez segmentacji:
// narzędzie wg ramki GT (bit 0) albo tło.
uniform uint u_jest_obszary, u_obszar;
uint obszar(uint i) { return u_jest_obszary == 1u ? ((MASK[i] >> 1u) & 3u) : (narz(i) ? 3u : 0u); }
// Właściwości ruchu cząsteczki (fizyka), KG[u_wl0 + 12 i + c]: 0 temperatura (energia drgań własnych na zapis, po
// odjęciu pola kolektywnego), 1 wykładnik MSD α, 2 VACF (cos kolejnych przesunięć), 3 prostoliniowość (netto/droga),
// 4 tempo powrotu κ (Ornstein-Uhlenbeck), 5 ruch w rozgrzewce, 6 zmiana 768D na końcu okna, 7 reżim
// (0 balistyczny, 1 dyfuzyjny, 2 uwięziony, -1 brak danych), 8 okres dominujący ruchu (zapisy, z widma przyrostów),
// 9 udział energii ruchu w paśmie okresów [u_pas_lo, u_pas_hi], 10 energia kinetyczna ½⟨|Δx|²⟩ na zapis,
// 11 energia potencjalna ½ κ ⟨|x − x̄|²⟩; wirtualna 12 = energia całkowita (10 + 11).
uniform uint u_wl0, u_wl1, u_wl_ok, u_wx, u_wy, u_wz, u_en, u_zak_c;   // u_wl1: okno wokół zapisu b (interpolacja)
uniform vec3 u_wl_lo, u_wl_hi;
uniform vec4 u_wl_kol;                      // zakres koloru: log temperatury (x, y), α (z, w)
uniform vec4 u_en_kol;                      // zakres koloru energii: log (x, y)
uniform vec4 u_zak[3];                      // zakresy koloru 17: (od, do, włączony, 0)
uniform vec3 u_zkol[3];                     // ich kolory
float wl_z(uint baza, uint i, uint c) { return KG[baza + 12u * i + c]; }
float wl(uint i, uint c) { return wl_z(u_wl0, i, c); }
float wl_vz(uint baza, uint i, uint c) { return c == 12u ? wl_z(baza, i, 10u) + wl_z(baza, i, 11u) : wl_z(baza, i, c); }
float wl_v(uint i, uint c) { return wl_vz(u_wl0, i, c); }
bool wl_log(uint c) { return c == 0u || c == 5u || c == 6u || c == 10u || c == 11u || c == 12u; }
float wl_osz(uint baza, uint i, uint c) {  // temperatura, ruch rozgrzewki, zmiana 768D i energie w skali log
    float v = wl_vz(baza, i, c);
    return wl_log(c) ? log(max(v, 1e-12)) : v;
}
float wl_os(uint i, uint c) { return wl_osz(u_wl0, i, c); }
vec3 inferno(float t) {                     // przybliżenie wielomianowe palety inferno (jak mapy atencji w Uczenie ViT)
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0002, 0.0016, -0.0194), c1 = vec3(0.1065, 0.5640, 3.9327), c2 = vec3(11.6025, -3.9729, -15.9424),
         c3 = vec3(-41.7040, 17.4364, 44.3541), c4 = vec3(77.1629, -33.4023, -81.8073), c5 = vec3(-71.3194, 32.6261, 73.2095),
         c6 = vec3(25.1311, -12.2426, -23.0703);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
}
vec3 hsv(float h, float s, float v) {
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}
vec3 ramp(float t) { return mix(vec3(0.23, 0.72, 0.68), vec3(1.0, 0.91, 0.68), t); }
vec3 viridis(float t) {                     // przybliżenie wielomianowe palety viridis
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.2777, 0.0054, 0.3341), c1 = vec3(0.1050, 1.4046, 1.3846), c2 = vec3(-0.3309, 0.2148, 0.0951),
         c3 = vec3(-4.6342, -5.7991, -19.3324), c4 = vec3(6.2283, 14.1799, 56.6906), c5 = vec3(4.7764, -13.7451, -65.3530),
         c6 = vec3(-5.4355, 4.6459, 26.3124);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
}
uint patch_i(uint i) { return i % max(u_tpi_w, 1u); }
// Klastry Koopmana (kolor 7), mieszanina Gaussów z klastry_koopmana: u_kl[0] = (mediana osi, liczba klastrów),
// u_kl[1] = (skala osi, wymiar 2|3), u_kl[2].x = próg outliera w σ; klaster c: u_kl[3 + 2c] = (μ, log π - Σ log σ),
// u_kl[4 + 2c] = (σ, 1 = klaster | -1 = za mały). Przydział: max log π - Σ log σ - d²/2, d = |(z - μ) / σ|.
uniform vec4 u_kl[24];
vec3 barwa_klastra(uint i) {
    int n = int(u_kl[0].w);
    if (n <= 0 || u_uklad != 3u) return vec3(0.45);
    vec3 z = (CUR[i].xyz - u_kl[0].xyz) / u_kl[1].xyz;
    if (u_kl[1].w < 2.5) z.z = 0.0;
    float best = -3.0e38, dbest = 0.0; int c = 0;
    for (int q = 0; q < n; q++) {
        vec3 e = (z - u_kl[3 + 2 * q].xyz) / u_kl[4 + 2 * q].xyz;
        float d2 = dot(e, e), sc = u_kl[3 + 2 * q].w - 0.5 * d2;
        if (sc > best) { best = sc; dbest = d2; c = q; }
    }
    if (u_kl[4 + 2 * c].w < 0.0 || sqrt(dbest) > u_kl[2].x) return vec3(0.33);     // outlier: szary
    return hsv(fract(float(c) * 0.6180339 + 0.02), 0.8, 1.0);
}
// kolor: 0-2 wielkość z kinematyki (norma 768D, prędkość PCA, |Δ| PCA), 3 położenie patcha w siatce, 4 narzędzie
// i tło wg ramki GT, 5 pewność ViT obrazu w kroku, 6 numer obrazu, 7 klaster Koopmana (osie pod klastry),
// 8 temperatura, 9 reżim ruchu, 10 wykładnik MSD α, 11 obszar oka (narzędzie, źrenica, rogówka, tło),
// 12 przyspieszenie chwilowe |v1 - v0| w PCA (inferno, pierwiastek z ułamka 2 × maks. prędkości), 13 prędkość chwilowa
// (różnica centralna), 14 prędkość chwilowa względem średniej prędkości tej cząsteczki (log2, niebieski-biały-czerwony)
vec3 barwa_trybu(uint i, float w, int kolor) {
    uint t = patch_i(i), im = i / max(u_tpi_w, 1u);
    if (kolor == 7) return barwa_klastra(i);
    if (kolor == 12) return inferno(clamp(sqrt(w / max(2.0 * FR[3][1], 1e-6)), 0.0, 1.0));   // skala: 2 × maks. prędkość
    if (kolor == 13) return inferno(clamp(sqrt(w / max(FR[3][1], 1e-6)), 0.0, 1.0));         // skala: maks. prędkość
    if (kolor == 14) {                             // log2(v / średnia): niebieski wolniej, biały typowo, czerwony zryw
        if (w <= 0.0) return vec3(0.45);
        float t = clamp(log2(w) / 2.0, -1.0, 1.0);
        return t < 0.0 ? mix(vec3(0.95), vec3(0.15, 0.45, 1.0), -t) : mix(vec3(0.95), vec3(1.0, 0.15, 0.1), t);
    }
    if (kolor == 8) return u_wl_ok == 1u ? inferno(clamp((log(max(wl(i, 0u), 1e-12)) - u_wl_kol.x)
                                                     / max(u_wl_kol.y - u_wl_kol.x, 1e-6), 0.0, 1.0)) : vec3(0.45);
    if (kolor == 9) {
        float r = u_wl_ok == 1u ? wl(i, 7u) : -1.0;
        return r < -0.5 ? vec3(0.45) : r < 0.5 ? vec3(1.0, 0.72, 0.15) : r < 1.5 ? vec3(0.3, 0.75, 1.0) : vec3(0.8, 0.3, 0.95);
    }
    if (kolor == 10) return u_wl_ok == 1u ? viridis(clamp((wl(i, 1u) - u_wl_kol.z) / max(u_wl_kol.w - u_wl_kol.z, 1e-6), 0.0, 1.0))
                                          : vec3(0.45);
    if (kolor == 15) {                             // energia: kinetyczna / potencjalna / całkowita (log, inferno)
        if (u_wl_ok == 0u) return vec3(0.45);
        float v = u_en == 0u ? wl(i, 10u) : u_en == 1u ? wl(i, 11u) : wl(i, 10u) + wl(i, 11u);
        return inferno(clamp((log(max(v, 1e-12)) - u_en_kol.x) / max(u_en_kol.y - u_en_kol.x, 1e-6), 0.0, 1.0));
    }
    if (kolor == 16) return u_wl_ok == 1u ? inferno(clamp(wl(i, 9u), 0.0, 1.0)) : vec3(0.45);   // udział energii w paśmie
    if (kolor == 17) {                             // zakresy wybranej właściwości: pierwszy pasujący zakres
        if (u_wl_ok == 0u) return vec3(0.45);
        float v = wl_v(i, u_zak_c);
        for (int q = 0; q < 3; q++)
            if (u_zak[q].z > 0.5 && v >= u_zak[q].x && v <= u_zak[q].y) return u_zkol[q];
        return vec3(0.3);
    }
    if (kolor == 11) {
        uint o = obszar(i);
        return o == 3u ? vec3(1.0, 0.25, 0.15) : o == 2u ? vec3(0.25, 0.55, 1.0) : o == 1u ? vec3(0.3, 0.85, 0.4) : vec3(0.5, 0.52, 0.56);
    }
    if (kolor <= 2) return ramp(clamp(w / max(FR[3][kolor], 1e-6), 0.0, 1.0));
    if (kolor == 3) return hsv(float(t % 30u) / 30.0, 0.85, 0.45 + 0.55 * float(t / 30u) / 19.0);
    if (kolor == 4) return narz(i) ? vec3(1.0, 0.2, 0.15) : vec3(0.55, 0.55, 0.6);
    if (kolor == 5) return u_jest_conf == 1u ? viridis(CONF[u_ka * max(u_img_w, 1u) + im]) : vec3(0.5);
    return hsv(fract(float(im) * 0.6180339), 0.7, 0.95);
}
// Moc koloru u_moc: 0 = sama jasność (odcienie szarości), 1 = barwa trybu bez zmian, > 1 = mocniej nasycona.
uniform float u_moc;
vec3 barwa(uint i, float w, int kolor) {
    vec3 c = barwa_trybu(i, w, kolor);
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    return clamp(mix(vec3(l), c, u_moc), 0.0, 1.0);
}
bool widoczna(uint i) {
    uint t = patch_i(i);
    if (u_co > 1u && ((t % 30u) % u_co != 0u || (t / 30u) % u_co != 0u)) return false;
    if (u_tylko_narz == 1u && !narz(i)) return false;
    if (u_obszar > 0u && obszar(i) != u_obszar - 1u) return false;
    if (u_tylko_zazn == 1u && u_jest_zazn == 1u && SEL[t] == 0u) return false;
    return true;
}
// Cecha bez położenia patcha (u_bez_poz = 1): od cechy odejmowana jest ta sama cecha średniej tego patcha po
// wszystkich obrazach w tym samym kroku (m; m0 = średnia w pierwszym kroku). Usuwa to, co w latencie wynika z samego
// miejsca na obrazie (kodowanie pozycji, typowy układ sceny) - w siatce zdjęcia pokazują je już osie X i Y.
uniform uint u_bez_poz;
vec4 srednia_patcha(uint k, uint t) {
    if (u_wyg == 0u) return PMD[2u * (k * max(u_tpi_w, 1u) + t)];
    ivec2 o = okno_wyg(k);
    vec4 acc = vec4(0.0);
    float ws = 0.0;
    for (int q = o.x; q <= o.y; q++) {
        if (KNOWN[uint(q)] == 0u) continue;
        float w = waga_wyg(k, q);
        acc += w * PMD[2u * (uint(q) * max(u_tpi_w, 1u) + t)]; ws += w;
    }
    return ws > 0.0 ? acc / ws : PMD[2u * (k * max(u_tpi_w, 1u) + t)];
}
float cecha_bp(vec4 st, vec4 st0, uint ruch, vec4 m, vec4 m0) {
    if (u_bez_poz == 0u) return cecha(st, st0, ruch);
    if (u_cecha < 3u) return (st[u_cecha] - (ruch == 1u ? st0[u_cecha] : 0.0)) - (m[u_cecha] - (ruch == 1u ? m0[u_cecha] : 0.0));
    if (u_cecha == 3u) return st.w - m.w;
    return length(st.xyz - m.xyz - (ruch == 1u ? st0.xyz - m0.xyz : vec3(0.0)));
}
// Układ Koopmana (u_uklad = 3): stan kroku h_k = [s_k, s_k-1, ..., s_k-d+1], s = (PC1, PC2, PC3, norma 768D)
// standaryzowane średnią i odchyleniem z całej osi. Funkcje własne Koopmana ψ_j(h) = w_j · h (lewe wektory własne).
// Każda oś to zbiór składowych (Re albo Im ψ wybranych modów), łączonych sumą (rzut liniowy) albo modułem √Σ².
// KG[8 + 3a ..]: oś a = (pierwszy zestaw, liczba zestawów, łączenie 0 suma / 1 moduł); zestaw s: KG[24 + 16 s ..].
uniform uint u_kd, u_kz;
vec4 kw(uint zestaw, uint j) { uint o = 24u + zestaw * 16u + j * 4u; return vec4(KG[o], KG[o + 1u], KG[o + 2u], KG[o + 3u]); }
vec4 h_opozn(uint i, uint k, uint j) {
    uint kk = uint(max(int(k) - int(j), 0));
    vec4 mu = vec4(KG[0], KG[1], KG[2], KG[3]), isd = vec4(KG[4], KG[5], KG[6], KG[7]);
    return (j < u_kd && SLOT[kk] != NONE) ? (stan_g(kk, i) - mu) * isd : vec4(0.0);
}
float os_koop(uint a, vec4 h0, vec4 h1, vec4 h2, vec4 h3) {
    uint z0 = uint(KG[8u + 3u * a]), n = uint(KG[9u + 3u * a]);
    bool modul = KG[10u + 3u * a] > 0.5;
    float acc = 0.0;
    for (uint z = z0; z < z0 + n; z++) {
        float v = dot(kw(z, 0u), h0) + dot(kw(z, 1u), h1) + dot(kw(z, 2u), h2) + dot(kw(z, 3u), h3);
        acc += modul ? v * v : v;
    }
    return modul ? sqrt(acc) : acc;
}
vec3 koop(uint i, uint k) {
    vec4 h0 = h_opozn(i, k, 0u), h1 = h_opozn(i, k, 1u), h2 = h_opozn(i, k, 2u), h3 = h_opozn(i, k, 3u);
    return vec3(os_koop(0u, h0, h1, h2, h3), os_koop(1u, h0, h1, h2, h3), os_koop(2u, h0, h1, h2, h3));
}
vec2 siatka(uint i) {                       // patch na zdjęciu: (kolumna, -wiersz); w mozaice kafel obrazu
    uint im = i / max(u_tpi_w, 1u), t = i - im * max(u_tpi_w, 1u);
    vec2 p = vec2(float(t % 30u), -float(t / 30u));
    if (u_mozaika == 1u) p += vec2(float(im % max(u_kafle, 1u)) * 31.0, -float(im / max(u_kafle, 1u)) * 21.0);
    return p;
}
"""

CS_KIN = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_first, u_count, u_sa, u_sb, u_s0, u_ruch, u_sv0, u_sv1, u_kb, u_kv0, u_kv1, u_kc, u_czasy0;
uniform int u_kolor;
uniform float u_s, u_dt, u_dtv, u_dta0, u_dta1;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= u_count) return;
    uint i = u_first + g;
    vec4 a = stan_g(u_ka, i);                      // bez wygładzania: stan(u_sa, i), bitowo z pliku
    vec4 b = stan_g(u_kb, i);
    vec3 p0 = stan_g(0u, i).xyz;
    vec3 x; float n;
    if (u_s <= 0.0) { x = a.xyz; n = a.w; }                  // zapisany stan a: bitowo z pliku
    else if (u_s >= 1.0) { x = b.xyz; n = b.w; }             // zapisany stan b
    else { x = a.xyz + (b.xyz - a.xyz) * u_s; n = a.w + (b.w - a.w) * u_s; }   // stan niezapisany
    vec3 d = x - p0;
    float c = n;
    if (u_kolor == 1) c = u_dt > 0.0 ? length(b.xyz - a.xyz) / u_dt : 0.0;
    else if (u_kolor == 2) c = length(d);
    else if (u_kolor == 12) {                     // przyspieszenie chwilowe: |v1 - v0| wokół zapisu u_kc (na krok treningu)
        c = 0.0;
        if (u_dta0 > 0.0 && u_dta1 > 0.0 && u_kc >= 1u && u_kc + 1u < u_T_w
            && SLOT[u_kc - 1u] != NONE && SLOT[u_kc] != NONE && SLOT[u_kc + 1u] != NONE) {
            vec3 xm = stan_g(u_kc - 1u, i).xyz, x0 = stan_g(u_kc, i).xyz, xp = stan_g(u_kc + 1u, i).xyz;
            c = length((xp - x0) / u_dta1 - (x0 - xm) / u_dta0);
        }
    } else if (u_kolor == 13 || u_kolor == 14) {  // prędkość chwilowa (różnica centralna) i względna do średniej cząsteczki
        c = 0.0;
        if (u_dta0 + u_dta1 > 0.0 && u_kc >= 1u && u_kc + 1u < u_T_w && SLOT[u_kc - 1u] != NONE && SLOT[u_kc + 1u] != NONE) {
            float v = length(stan_g(u_kc + 1u, i).xyz - stan_g(u_kc - 1u, i).xyz) / (u_dta0 + u_dta1);
            if (u_kolor == 13) c = v;
            else {                                 // średnia prędkość tej cząsteczki po wszystkich odcinkach na karcie
                float sv = 0.0, nv = 0.0;
                for (uint k = 1u; k < u_T_w; k++) {
                    if (SLOT[k] == NONE || SLOT[k - 1u] == NONE) continue;
                    float dt = KG[u_czasy0 + 4u * k + 3u] - KG[u_czasy0 + 4u * (k - 1u) + 3u];
                    if (dt > 0.0) { sv += length(stan(SLOT[k], i).xyz - stan(SLOT[k - 1u], i).xyz) / dt; nv += 1.0; }
                }
                c = (nv > 0.0 && sv > 0.0) ? v / (sv / nv) : 0.0;
            }
        }
    }
    vec3 w = u_ruch == 1u ? d : x;
    if (u_uklad == 3u) {
        vec3 ka = koop(i, u_ka), kb = koop(i, u_kb);
        w = u_s <= 0.0 ? ka : (u_s >= 1.0 ? kb : ka + (kb - ka) * u_s);
    } else if (u_uklad == 4u) {                    // fizyka ruchu: osie z właściwości cząsteczki (okno zapisów)
        vec3 pa = vec3(wl_osz(u_wl0, i, u_wx), wl_osz(u_wl0, i, u_wy), wl_osz(u_wl0, i, u_wz));
        vec3 pb = vec3(wl_osz(u_wl1, i, u_wx), wl_osz(u_wl1, i, u_wy), wl_osz(u_wl1, i, u_wz));
        w = u_wl_ok == 1u ? mix(pa, pb, clamp(u_s, 0.0, 1.0)) : vec3(0.0);   // okna wokół zapisów a i b: płynnie
    } else if (u_uklad != 0u) {
        vec4 st0 = stan_g(0u, i);
        uint t = patch_i(i);
        vec4 m0 = srednia_patcha(0u, t), ma = srednia_patcha(u_ka, t), mb = srednia_patcha(u_kb, t);
        vec4 m = u_s <= 0.0 ? ma : (u_s >= 1.0 ? mb : ma + (mb - ma) * u_s);
        float f = cecha_bp(vec4(x, n), st0, u_ruch, m, m0);
        if (u_uklad == 1u) w = vec3(siatka(i), f);
        else {                                     // prędkość cechy na odcinku (u_sv0 -> u_sv1)
            float v = 0.0;
            if (u_sv0 != NONE && u_sv1 != NONE && u_dtv > 0.0)
                v = (cecha_bp(stan_g(u_kv1, i), st0, u_ruch, srednia_patcha(u_kv1, t), m0)
                     - cecha_bp(stan_g(u_kv0, i), st0, u_ruch, srednia_patcha(u_kv0, t), m0)) / u_dtv;
            w = vec3(f, v, n);
        }
    }
    CUR[i] = vec4(w, c);
}
"""

CS_STAT = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_k, u_slot, u_s0, u_prev, u_tpi, u_img;
uniform float u_dt;
shared vec3 s_lo[256], s_hi[256], s_mlo[256], s_mhi[256], s_mx[256];
shared float s_vn[256];
void main() {
    uint im = gl_WorkGroupID.x, l = gl_LocalInvocationID.x;
    vec3 lo = vec3(3.0e38), hi = vec3(-3.0e38), mlo = vec3(3.0e38), mhi = vec3(-3.0e38), mx = vec3(0.0);
    float vn = 0.0;
    for (uint t = l; t < u_tpi; t += 256u) {
        uint i = im * u_tpi + t;
        vec4 p = stan(u_slot, i);
        vec3 d = p.xyz - stan(u_s0, i).xyz;
        lo = min(lo, p.xyz); hi = max(hi, p.xyz); mlo = min(mlo, d); mhi = max(mhi, d);
        float v = 0.0;
        if (u_prev != NONE && u_dt > 0.0) {
            vec4 q = stan(u_prev, i);
            v = length(p.xyz - q.xyz) / u_dt;
            vn = max(vn, abs(p.w - q.w) / u_dt);
        }
        mx = max(mx, vec3(p.w, length(d), v));
    }
    s_lo[l] = lo; s_hi[l] = hi; s_mlo[l] = mlo; s_mhi[l] = mhi; s_mx[l] = mx; s_vn[l] = vn;
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) {
            s_lo[l] = min(s_lo[l], s_lo[l + w]); s_hi[l] = max(s_hi[l], s_hi[l + w]);
            s_mlo[l] = min(s_mlo[l], s_mlo[l + w]); s_mhi[l] = max(s_mhi[l], s_mhi[l + w]);
            s_mx[l] = max(s_mx[l], s_mx[l + w]); s_vn[l] = max(s_vn[l], s_vn[l + w]);
        }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) {
        uint b = (u_k * u_img + im) * 4u;
        PIB[b] = vec4(s_lo[0], s_mx[0].x);          // granice położeń, maks. normy 768D
        PIB[b + 1u] = vec4(s_hi[0], s_mx[0].y);     // ..., maks. |Δ| w PCA od pierwszego kroku
        PIB[b + 2u] = vec4(s_mlo[0], s_mx[0].z);    // granice przesunięć, maks. prędkości względem poprzedniego zapisu
        PIB[b + 3u] = vec4(s_mhi[0], s_vn[0]);     // ..., maks. prędkości normy 768D
        if (im == 0u) KNOWN[u_k] = 1u;
    }
}
"""

CS_HEAT = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_k, u_slot, u_tpi, u_img;
shared vec4 s_v[256];
void main() {
    uint t = gl_WorkGroupID.x, l = gl_LocalInvocationID.x;
    vec4 acc = vec4(0.0);
    for (uint im = l; im < u_img; im += 256u) acc += stan(u_slot, im * u_tpi + t);
    s_v[l] = acc;
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) s_v[l] += s_v[l + w];
        memoryBarrierShared(); barrier();
    }
    vec4 m = s_v[0] / float(u_img);
    memoryBarrierShared(); barrier();
    if (l == 0u) { HEAT[u_k * u_tpi + t] = m.w; PMD[2u * (u_k * u_tpi + t)] = m; }
    vec4 dev = vec4(0.0);                          // największe |odchylenie| od średniej patcha (zakres osi)
    for (uint im = l; im < u_img; im += 256u) dev = max(dev, abs(stan(u_slot, im * u_tpi + t) - m));
    s_v[l] = dev;
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) s_v[l] = max(s_v[l], s_v[l + w]);
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) PMD[2u * (u_k * u_tpi + t) + 1u] = s_v[0];
}
"""

CS_BB = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_first, u_count;
shared vec3 s_lo[256], s_hi[256];
void main() {
    uint g = gl_GlobalInvocationID.x, l = gl_LocalInvocationID.x;
    vec3 p = g < u_count ? CUR[u_first + g].xyz : vec3(0.0);
    s_lo[l] = g < u_count ? p : vec3(3.0e38);
    s_hi[l] = g < u_count ? p : vec3(-3.0e38);
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) { s_lo[l] = min(s_lo[l], s_lo[l + w]); s_hi[l] = max(s_hi[l], s_hi[l + w]); }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) {
        atomicMin(BB[0], ord(s_lo[0].x)); atomicMin(BB[1], ord(s_lo[0].y)); atomicMin(BB[2], ord(s_lo[0].z));
        atomicMax(BB[3], ord(s_hi[0].x)); atomicMax(BB[4], ord(s_hi[0].y)); atomicMax(BB[5], ord(s_hi[0].z));
    }
}
"""

CS_RAMKA = _NAGLOWEK + """
layout(local_size_x = 512) in;          // 512: pamięć współdzielona < 48 KB
uniform uint u_T, u_img, u_img0, u_img1, u_ruch, u_tryb, u_manifest;      // u_tpi_w, u_bez_poz: nagłówek
uniform vec3 u_mlo, u_mhi;
uniform vec2 u_xylo, u_xyhi;
uniform float u_normscale;
shared vec3 s_lo[512], s_hi[512], s_mx[512];
shared float s_vn[512];
shared vec4 s_pd[512];
void main() {
    uint l = gl_LocalInvocationID.x;
    vec3 lo = vec3(3.0e38), hi = vec3(-3.0e38), mx = vec3(0.0);
    float vn = 0.0;
    vec4 pd = vec4(0.0);
    if (u_bez_poz == 1u)
        for (uint e = l; e < u_T * max(u_tpi_w, 1u); e += 512u)
            if (KNOWN[e / max(u_tpi_w, 1u)] != 0u) pd = max(pd, PMD[2u * e + 1u]);
    uint total = u_T * u_img;
    for (uint e = l; e < total; e += 512u) {
        uint k = e / u_img, im = e - k * u_img;
        if (KNOWN[k] == 0u) continue;
        uint b = e * 4u;
        vec4 q0 = PIB[b], q1 = PIB[b + 1u], q2 = PIB[b + 2u], q3 = PIB[b + 3u];
        mx = max(mx, vec3(q0.w, q2.w, q1.w));             // skale kolorów zawsze z całego zbioru
        vn = max(vn, q3.w);
        if (u_tryb == 0u || (im >= u_img0 && im < u_img1)) {
            if (u_ruch == 1u) { lo = min(lo, q2.xyz); hi = max(hi, q3.xyz); }
            else { lo = min(lo, q0.xyz); hi = max(hi, q1.xyz); }
        }
    }
    s_lo[l] = lo; s_hi[l] = hi; s_mx[l] = mx; s_vn[l] = vn; s_pd[l] = pd;
    memoryBarrierShared(); barrier();
    for (uint w = 256u; w > 0u; w >>= 1u) {
        if (l < w) { s_lo[l] = min(s_lo[l], s_lo[l + w]); s_hi[l] = max(s_hi[l], s_hi[l + w]); s_mx[l] = max(s_mx[l], s_mx[l + w]);
                     s_vn[l] = max(s_vn[l], s_vn[l + w]); s_pd[l] = max(s_pd[l], s_pd[l + w]); }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) {
        lo = s_lo[0]; hi = s_hi[0];
        if (u_tryb == 2u) {
            lo = vec3(unord(BB[0]), unord(BB[1]), unord(BB[2]));
            hi = vec3(unord(BB[3]), unord(BB[4]), unord(BB[5]));
        }
        if (u_manifest == 1u && u_tryb == 0u && u_uklad == 0u) { lo = min(lo, u_mlo); hi = max(hi, u_mhi); }
        if (any(greaterThan(lo, hi))) { lo = vec3(-1.0); hi = vec3(1.0); }   // jeszcze brak danych
        if (u_uklad == 4u && u_tryb != 2u) {             // fizyka: zakres osi z percentyli 1-99 (CPU)
            lo = u_wl_lo; hi = u_wl_hi;
        } else if (u_uklad == 3u && u_tryb != 2u) {      // Koopman: zakres po wszystkich krokach (kernel kzakres)
            lo = vec3(unord(HIST[R + 16u]), unord(HIST[R + 17u]), unord(HIST[R + 18u]));
            hi = vec3(unord(HIST[R + 19u]), unord(HIST[R + 20u]), unord(HIST[R + 21u]));
            if (any(greaterThan(lo, hi))) { lo = vec3(-1.0); hi = vec3(1.0); }
        } else if (u_uklad != 0u && u_tryb != 2u) {     // granice osi układu z granic PCA i maksimów kroków
            vec2 zc = u_cecha < 3u ? vec2(lo[u_cecha], hi[u_cecha])
                    : u_cecha == 3u ? vec2(0.0, max(s_mx[0].x, 1e-6))
                    : vec2(0.0, u_ruch == 1u ? s_mx[0].z : length(max(abs(lo), abs(hi))));
            float vmax = max(u_cecha == 3u ? s_vn[0] : s_mx[0].y, 1e-6);
            if (u_bez_poz == 1u) {                 // cecha bez położenia: wokół 0, do maks. odchylenia od średniej patcha
                float k = u_ruch == 1u ? 2.0 : 1.0;
                float d = u_cecha < 3u ? s_pd[0][u_cecha] * k : u_cecha == 3u ? s_pd[0].w : length(s_pd[0].xyz) * k;
                zc = u_cecha == 4u ? vec2(0.0, max(d, 1e-6)) : vec2(-max(d, 1e-6), max(d, 1e-6));
                vmax *= 2.0;
            }
            if (u_uklad == 1u) { lo = vec3(u_xylo, zc.x); hi = vec3(u_xyhi, zc.y); }
            else { lo = vec3(zc.x, -vmax, 0.0); hi = vec3(zc.y, vmax, max(s_mx[0].x, 1e-6)); }
        }
        vec3 e = max(hi - lo, vec3(1e-6));
        FR[0] = vec4((lo + hi) * 0.5, 2.0 / max(max(max(e.x, e.y), e.z), 0.001));
        // skale osi: PCA jednakowa (zachowuje kształt chmury); siatka: zdjęcie w proporcjach, Z o wysokości 1;
        // przestrzeń fazowa: każda oś do tej samej długości
        FR[4] = u_uklad == 0u ? vec4(vec3(FR[0].w), 0.0)
              : u_uklad == 1u ? vec4(vec2(2.0 / max(max(e.x, e.y), 1e-6)), 1.0 / e.z, 0.0)
              : vec4(2.0 / e, 0.0);                       // przestrzeń fazowa i Koopman: każda oś do tej samej długości
        FR[1] = vec4(lo, 0.0);
        FR[2] = vec4(hi, 0.0);
        FR[3] = vec4(max(s_mx[0].x, max(u_normscale, 1e-6)), max(s_mx[0].y, 1e-6), max(s_mx[0].z, 1e-6), 0.0);
    }
}
"""

CS_KSTAT = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_T;
shared vec4 s1[256], s2[256];
void main() {                                      // sumy częściowe s i s^2 po krokach na karcie (średnia, odchylenie)
    uint i = gl_GlobalInvocationID.x, l = gl_LocalInvocationID.x;
    vec4 a = vec4(0.0), b = vec4(0.0);
    if (i < u_N)
        for (uint k = 0u; k < u_T; k++) {
            uint sl = SLOT[k];
            if (sl == NONE) continue;
            vec4 v = stan(sl, i);
            a += v; b += v * v;
        }
    s1[l] = a; s2[l] = b;
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) { s1[l] += s1[l + w]; s2[l] += s2[l + w]; }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u) {
        uint o = 2048u + gl_WorkGroupID.x * 8u;
        for (uint c = 0u; c < 4u; c++) { KG[o + c] = s1[0][c]; KG[o + 4u + c] = s2[0][c]; }
    }
}
"""

CS_KGRAM = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_T;
shared vec4 sh[256 * 5];                           // [cząsteczka * (d + 1) + j] = s_(k+1-j), standaryzowane
float hk(uint p, uint idx) { return sh[p * (u_kd + 1u) + idx / 4u + 1u][idx % 4u]; }    // h_k
float hk1(uint p, uint idx) { return sh[p * (u_kd + 1u) + idx / 4u][idx % 4u]; }        // h_(k+1)
float wyraz(uint e, uint D) {                     // e < D*D: Cxx[r][c] = sum h_k[r] h_k[c]; dalej Cyx = sum h_(k+1)[r] h_k[c]
    uint m = e / (D * D), rc = e % (D * D), r = rc / D, c = rc % D;
    float acc = 0.0;
    for (uint p = 0u; p < 256u; p++) acc += hk(p, c) * (m == 0u ? hk(p, r) : hk1(p, r));
    return acc;
}
void main() {                                      // macierze Grama Hankel-DMD; sumy częściowe na grupę
    uint l = gl_LocalInvocationID.x, g = gl_WorkGroupID.x, i = g * 256u + l;
    uint D = 4u * u_kd, E = 2u * D * D;
    vec4 mu = vec4(KG[0], KG[1], KG[2], KG[3]), isd = vec4(KG[4], KG[5], KG[6], KG[7]);
    float a0 = 0.0, a1 = 0.0;
    for (uint k = u_kd - 1u; k + 1u < u_T; k++) {
        bool ok = true;
        for (uint j = 0u; j <= u_kd; j++) if (SLOT[k + 1u - j] == NONE) ok = false;
        if (!ok) continue;                         // ten sam warunek dla całej grupy (zależy tylko od k)
        for (uint j = 0u; j <= u_kd; j++)
            sh[l * (u_kd + 1u) + j] = i < u_N ? (stan(SLOT[k + 1u - j], i) - mu) * isd : vec4(0.0);
        memoryBarrierShared(); barrier();
        if (l < E) a0 += wyraz(l, D);
        if (l + 256u < E) a1 += wyraz(l + 256u, D);
        memoryBarrierShared(); barrier();
    }
    uint o = 2048u + g * 512u;
    if (l < E) KG[o + l] = a0;
    if (l + 256u < E) KG[o + l + 256u] = a1;
}
"""

CS_KZAKRES = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_T;
shared vec3 s_lo[256], s_hi[256];
void main() {                                      // zakres współrzędnych Koopmana po wszystkich krokach (stałe osie)
    uint i = gl_GlobalInvocationID.x, l = gl_LocalInvocationID.x;
    vec3 lo = vec3(3.0e38), hi = vec3(-3.0e38);
    if (i < u_N)
        for (uint k = 0u; k < u_T; k++) {
            if (SLOT[k] == NONE) continue;
            vec3 c = koop(i, k);
            lo = min(lo, c); hi = max(hi, c);
        }
    s_lo[l] = lo; s_hi[l] = hi;
    memoryBarrierShared(); barrier();
    for (uint w = 128u; w > 0u; w >>= 1u) {
        if (l < w) { s_lo[l] = min(s_lo[l], s_lo[l + w]); s_hi[l] = max(s_hi[l], s_hi[l + w]); }
        memoryBarrierShared(); barrier();
    }
    if (l == 0u && s_lo[0].x <= s_hi[0].x) {
        atomicMin(HIST[R + 16u], ord(s_lo[0].x)); atomicMin(HIST[R + 17u], ord(s_lo[0].y)); atomicMin(HIST[R + 18u], ord(s_lo[0].z));
        atomicMax(HIST[R + 19u], ord(s_hi[0].x)); atomicMax(HIST[R + 20u], ord(s_hi[0].y)); atomicMax(HIST[R + 21u], ord(s_hi[0].z));
    }
}
"""

CS_KPROBKA = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_M, u_first, u_count, u_kk, u_off;
void main() {                                      // próbka stanów Hankela h_k (do doboru osi pod klastry na CPU)
    uint g = gl_GlobalInvocationID.x;
    if (g >= u_M) return;
    uint i = u_first + min(uint((float(g) + 0.5) * float(u_count) / float(u_M)), u_count - 1u);
    uint o = 2048u + (u_off + g) * KPROBKA_W;
    vec4 h[4];
    h[0] = h_opozn(i, u_kk, 0u); h[1] = h_opozn(i, u_kk, 1u); h[2] = h_opozn(i, u_kk, 2u); h[3] = h_opozn(i, u_kk, 3u);
    for (uint j = 0u; j < 4u; j++)
        for (uint c = 0u; c < 4u; c++) KG[o + 4u * j + c] = h[j][c];
    KG[o + 16u] = (widoczna(i) && SLOT[u_kk] != NONE) ? 1.0 : 0.0;
    KG[o + 17u] = float(i);
}
""".replace('KPROBKA_W', '20u')

CS_RUCHK = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_first, u_count, u_k0, u_K;
shared float sh[256 * 24];
void main() {                                      // statystyki populacji w krokach u_k0..u_k0+u_K-1, osobno dla 4 obszarów:
    uint l = gl_LocalInvocationID.x, g = gl_WorkGroupID.x, gi = gl_GlobalInvocationID.x;   // Σd(3), Σ|d|², n, Σcos
    bool ok = gi < u_count;
    uint i = u_first + gi, c = ok ? obszar(i) : 0u;
    for (uint j = 0u; j < u_K; j++) {
        uint k = u_k0 + j;                         // przyrost k: x_k - x_(k-1)
        for (uint q = 0u; q < 24u; q++) sh[l * 24u + q] = 0.0;
        if (ok && k >= 1u && SLOT[k] != NONE && SLOT[k - 1u] != NONE) {
            vec3 xk = stan(SLOT[k], i).xyz, xp = stan(SLOT[k - 1u], i).xyz;
            vec3 d = xk - xp;
            float cs = 0.0;
            if (k >= 2u && SLOT[k - 2u] != NONE) {
                vec3 dp = xp - stan(SLOT[k - 2u], i).xyz;
                float a = length(d) * length(dp);
                cs = a > 1e-20 ? dot(d, dp) / a : 0.0;
            }
            uint o = l * 24u + c * 6u;
            sh[o] = d.x; sh[o + 1u] = d.y; sh[o + 2u] = d.z; sh[o + 3u] = dot(d, d); sh[o + 4u] = 1.0; sh[o + 5u] = cs;
        }
        memoryBarrierShared(); barrier();
        for (uint w = 128u; w > 0u; w >>= 1u) {
            if (l < w) for (uint q = 0u; q < 24u; q++) sh[l * 24u + q] += sh[(l + w) * 24u + q];
            memoryBarrierShared(); barrier();
        }
        if (l == 0u) for (uint q = 0u; q < 24u; q++) KG[2048u + (g * u_K + j) * 24u + q] = sh[q];
        memoryBarrierShared(); barrier();
    }
}
"""

CS_WLASC = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_first, u_count, u_ka0, u_kb0, u_kwarm, u_tau, u_pole0, u_wl_out;
uniform float u_prog_a, u_prog_b, u_pas_lo, u_pas_hi;
vec3 pole(uint k) { return vec3(KG[u_pole0 + 4u * k], KG[u_pole0 + 4u * k + 1u], KG[u_pole0 + 4u * k + 2u]); }
void main() {                                      // właściwości ruchu cząsteczki w oknie zapisów [u_ka0, u_kb0]
    uint gi = gl_GlobalInvocationID.x;
    if (gi >= u_count) return;
    uint i = u_first + gi;
    float e = 0.0, droga = 0.0, cs = 0.0, nc = 0.0, n = 0.0, m1 = 0.0, mt = 0.0, nt = 0.0, nx = 0.0;
    vec3 sx = vec3(0.0), xs = vec3(0.0), xe = vec3(0.0), xp = vec3(0.0), dp = vec3(0.0);
    bool hp = false, hd = false;
    for (uint k = u_ka0; k <= u_kb0; k++) {
        if (SLOT[k] == NONE) { hp = false; hd = false; continue; }
        vec3 x = stan(SLOT[k], i).xyz;
        if (nx == 0.0) xs = x;
        xe = x; sx += x; nx += 1.0;
        if (hp) {
            vec3 d = x - xp, dw = d - pole(k);
            e += dot(dw, dw); droga += length(d); m1 += dot(d, d); n += 1.0;
            if (hd) { float a = length(d) * length(dp); if (a > 1e-20) { cs += dot(d, dp) / a; nc += 1.0; } }
            dp = d; hd = true;
        }
        if (k + u_tau <= u_kb0 && SLOT[k + u_tau] != NONE) {
            vec3 y = stan(SLOT[k + u_tau], i).xyz - x;
            mt += dot(y, y); nt += 1.0;
        }
        xp = x; hp = true;
    }
    vec3 xm = sx / max(nx, 1.0);
    float num = 0.0, den = 0.0;
    hp = false;
    for (uint k = u_ka0; k <= u_kb0; k++) {
        if (SLOT[k] == NONE) { hp = false; continue; }
        vec3 x = stan(SLOT[k], i).xyz;
        if (hp) { vec3 c = xp - xm; num += dot(c, x - xp); den += dot(c, c); }
        xp = x; hp = true;
    }
    float temp = n > 0.0 ? e / n : 0.0, msd1 = n > 0.0 ? m1 / n : 0.0, msdt = nt > 0.0 ? mt / nt : 0.0;
    float alfa = (msd1 > 1e-20 && msdt > 1e-20 && u_tau > 1u) ? log(msdt / msd1) / log(float(u_tau)) : 0.0;
    float vacf = nc > 0.0 ? cs / nc : 0.0;
    float prosto = droga > 1e-20 ? length(xe - xs) / droga : 0.0;
    float kappa = den > 1e-20 ? -num / den : 0.0;
    float rozg = (SLOT[0] != NONE && SLOT[u_kwarm] != NONE) ? length(stan(SLOT[u_kwarm], i).xyz - stan(SLOT[0], i).xyz) : 0.0;
    float n768 = SLOT[u_kb0] != NONE ? stan(SLOT[u_kb0], i).w : 0.0;
    float rez = n < 2.0 ? -1.0 : (vacf > u_prog_b && prosto > u_prog_b) ? 0.0 : (alfa >= u_prog_a ? 1.0 : 2.0);
    // widmo przyrostów: najwyżej 128 kolejnych przyrostów ze środka okna, DFT dla f = 1..m/2 (okres m/f zapisów)
    uint m0 = u_kb0 - u_ka0, m = min(m0, 128u), j0 = u_ka0 + (m0 - m) / 2u;
    vec3 dd[128];
    for (uint j = 0u; j < m; j++) {
        uint k = j0 + j;
        dd[j] = (SLOT[k] != NONE && SLOT[k + 1u] != NONE) ? stan(SLOT[k + 1u], i).xyz - stan(SLOT[k], i).xyz : vec3(0.0);
    }
    float pmax = -1.0, okres_dom = 0.0, ptot = 0.0, ppas = 0.0;
    for (uint f = 1u; 2u * f <= m; f++) {
        float th = 6.283185307 * float(f) / float(m);
        vec2 w = vec2(cos(th), -sin(th)), z = vec2(1.0, 0.0);
        vec3 re = vec3(0.0), im = vec3(0.0);
        for (uint j = 0u; j < m; j++) {
            re += dd[j] * z.x; im += dd[j] * z.y;
            z = vec2(z.x * w.x - z.y * w.y, z.x * w.y + z.y * w.x);
        }
        float pw = dot(re, re) + dot(im, im);
        if (2u * f != m) pw *= 2.0;                // częstotliwość dodatnia i ujemna (poza Nyquistem)
        float okres = float(m) / float(f);
        ptot += pw;
        if (okres >= u_pas_lo && okres <= u_pas_hi) ppas += pw;
        if (pw > pmax) { pmax = pw; okres_dom = okres; }
    }
    float ekin = 0.5 * msd1, epot = 0.5 * max(kappa, 0.0) * (n > 0.0 ? den / n : 0.0);
    uint o = u_wl_out + 12u * i;
    KG[o] = temp; KG[o + 1u] = alfa; KG[o + 2u] = vacf; KG[o + 3u] = prosto;
    KG[o + 4u] = kappa; KG[o + 5u] = rozg; KG[o + 6u] = n768; KG[o + 7u] = rez;
    KG[o + 8u] = okres_dom; KG[o + 9u] = ptot > 0.0 ? ppas / ptot : 0.0; KG[o + 10u] = ekin; KG[o + 11u] = epot;
}
"""

CS_HIST = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_slot, u_first, u_count, u_faza;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= u_count) return;
    uint bits = floatBitsToUint(stan(u_slot, u_first + g).w);     // norma >= 0: porządek bitów = porządek liczb
    if (u_faza == 0u) { atomicAdd(HIST[bits >> 16u], 1u); atomicMax(HIST[R + 2u], bits); }
    else {
        uint p = bits >> 16u, q = bits & 0xffffu;
        if (p == HIST[R]) atomicAdd(HIST[BINS + q], 1u);
        if (p == HIST[R + 1u]) atomicAdd(HIST[2u * BINS + q], 1u);
    }
}
"""

CS_WYBOR = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform uint u_faza, u_klo, u_khi;
shared uint s_a[256], s_b[256];
// k-ty element (od 0) histogramu o początku `off`: (kosz, pozycja w koszu)
uvec2 wybierz(uint off, uint k, bool pierwszy) {
    uint acc = 0u, c = 0u;
    for (c = 0u; c < 256u; c++) {
        uint v = pierwszy ? s_a[c] : s_b[c];
        if (acc + v > k) break;
        acc += v;
    }
    c = min(c, 255u);
    uint bin = c * 256u;
    for (uint j = 0u; j < 256u; j++) {
        uint v = HIST[off + c * 256u + j];
        if (acc + v > k) { bin = c * 256u + j; break; }
        acc += v;
    }
    return uvec2(bin, k - acc);
}
void main() {
    uint l = gl_LocalInvocationID.x;
    uint offa = u_faza == 0u ? 0u : BINS, offb = u_faza == 0u ? 0u : 2u * BINS;
    uint a = 0u, b = 0u;
    for (uint j = 0u; j < 256u; j++) { a += HIST[offa + l * 256u + j]; b += HIST[offb + l * 256u + j]; }
    s_a[l] = a; s_b[l] = b;
    memoryBarrierShared(); barrier();
    if (l == 0u) {
        if (u_faza == 0u) {
            uvec2 x = wybierz(0u, u_klo, true), y = wybierz(0u, u_khi, true);
            HIST[R] = x.x; HIST[R + 3u] = x.y; HIST[R + 1u] = y.x; HIST[R + 4u] = y.y;
        } else {
            uvec2 x = wybierz(BINS, HIST[R + 3u], true), y = wybierz(2u * BINS, HIST[R + 4u], false);
            HIST[R + 5u] = (HIST[R] << 16u) | x.x;
            HIST[R + 6u] = (HIST[R + 1u] << 16u) | y.x;
        }
    }
}
"""

CS_PICK = _NAGLOWEK + """
layout(local_size_x = 256) in;
uniform mat4 u_mvp;
uniform vec2 u_klik, u_wh;
uniform uint u_first, u_count, u_faza;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= u_count) return;
    uint i = u_first + g;
    if (!widoczna(i)) return;
    vec4 f = FR[0];
    vec4 c = u_mvp * vec4((CUR[i].xyz - f.xyz) * FR[4].xyz, 1.0);
    if (c.w <= 1e-9) return;                                       // za kamerą
    vec2 n = c.xy / c.w;
    vec2 d = vec2((n.x + 1.0) * 0.5 * u_wh.x, (1.0 - n.y) * 0.5 * u_wh.y) - u_klik;
    uint d2 = floatBitsToUint(dot(d, d));
    if (u_faza == 0u) atomicMin(PICK[0], d2);
    else if (d2 == PICK[0]) atomicMin(PICK[1], i);                 // przy remisie najmniejszy indeks
}
"""

VS_PUNKTY = _NAGLOWEK + """
uniform mat4 u_mvp;
uniform uint u_first;
uniform int u_kolor, u_wybrany;
uniform float u_size, u_opacity;
out vec4 v_rgba;
flat out int v_ring;
void main() {
    uint i = u_wybrany >= 0 ? uint(u_wybrany) : u_first + uint(gl_VertexID);
    if (u_wybrany < 0 && !widoczna(i)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
                                         v_rgba = vec4(0.0); v_ring = 0; return; }
    vec4 c = CUR[i];
    vec4 f = FR[0];
    gl_Position = u_mvp * vec4((c.xyz - f.xyz) * FR[4].xyz, 1.0);
    float a = u_opacity, size = u_size;
    if (u_jest_zazn == 1u) {                       // zaznaczone patche jasne i większe, reszta blada
        if (SEL[patch_i(i)] != 0u) { a = max(a, 0.6); size *= 1.5; } else a *= 0.12;
    }
    vec3 rgb = barwa(i, c.w, u_kolor);
    if ((u_att & 1u) != 0u) { float m = att(i); rgb = inferno(m); size *= 0.4 + u_att_wzm * m; }   // atencja
    gl_PointSize = u_wybrany >= 0 ? max(u_size * 5.0, 12.0) : size;
    v_rgba = vec4(rgb, a);
    v_ring = u_wybrany >= 0 ? 1 : 0;
}
"""

FS_PUNKTY = """#version 430
in vec4 v_rgba;
flat in int v_ring;
out vec4 kolor;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = dot(p, p);
    if (r > 1.0) discard;
    if (v_ring == 1) { if (r < 0.5) discard; kolor = vec4(1.0, 1.0, 1.0, 0.95); return; }
    kolor = v_rgba;
}
"""

VS_SLAD = _NAGLOWEK + """
uniform mat4 u_mvp;
uniform uint u_first, u_ia, u_K, u_ruch, u_s0;
uniform int u_wybrany;
uniform float u_alfa;
out vec4 v_kol;
void main() {
    uint i = u_wybrany >= 0 ? uint(u_wybrany) : u_first + uint(gl_InstanceID);
    if (u_wybrany < 0 && !widoczna(i)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_kol = vec4(0.0); return; }
    int v = gl_VertexID;
    uint e = uint(v / 2 + v % 2);                                   // punkt śladu 0..K, K = bieżąca klatka
    vec3 p = CUR[i].xyz;
    if (e < u_K) {
        int idx = max(int(u_ia) - int(u_K) + 1 + int(e), 0);
        uint sl = SLOT[idx];
        if (sl != NONE) {
            vec4 st = stan_g(uint(idx), i);
            if (u_uklad == 3u) p = koop(i, uint(idx));
            else if (u_uklad == 1u) p = vec3(siatka(i), cecha_bp(st, stan_g(0u, i), u_ruch,
                                                          srednia_patcha(uint(idx), patch_i(i)), srednia_patcha(0u, patch_i(i))));
            else {
                p = st.xyz;
                if (u_ruch == 1u) p -= stan_g(0u, i).xyz;
            }
        }
    }
    vec4 f = FR[0];
    gl_Position = u_mvp * vec4((p - f.xyz) * FR[4].xyz, 1.0);
    v_kol = vec4(0.82, 0.88, 1.0, u_alfa * (0.1 + 0.9 * float(e) / float(max(u_K, 1u))));
}
"""

FS_KOLOR = """#version 430
in vec4 v_kol;
out vec4 kolor;
void main() { kolor = v_kol; }
"""

_ZDJ_WSPOLNE = """
uniform mat4 u_mvp;
uniform uint u_first;
uniform int u_zkolor, u_kolor;
uniform sampler2D u_rgb;
vec3 pozycja_w(float rr, float cc) {           // dwuliniowo między położeniami cząsteczek siatki 20 x 30
    int r0 = min(int(floor(rr)), 18), c0 = min(int(floor(cc)), 28);
    float fr = rr - float(r0), fc = cc - float(c0);
    uint a = u_first + uint(r0 * 30 + c0);
    vec3 g = mix(CUR[a].xyz, CUR[a + 1u].xyz, fc), d = mix(CUR[a + 30u].xyz, CUR[a + 31u].xyz, fc);
    vec4 f = FR[0];
    return (mix(g, d, fr) - f.xyz) * FR[4].xyz;
}
vec3 kolor_w(float rr, float cc) {             // 0 zdjęcie, 1 barwy cząsteczek, 2 zdjęcie z narzędziem na czerwono
    int r0 = min(int(floor(rr)), 18), c0 = min(int(floor(cc)), 28);
    float fr = rr - float(r0), fc = cc - float(c0);
    uint a = u_first + uint(r0 * 30 + c0);
    if (u_zkolor == 1) {
        vec3 g = mix(barwa(a, CUR[a].w, u_kolor), barwa(a + 1u, CUR[a + 1u].w, u_kolor), fc);
        vec3 d = mix(barwa(a + 30u, CUR[a + 30u].w, u_kolor), barwa(a + 31u, CUR[a + 31u].w, u_kolor), fc);
        return mix(g, d, fr);
    }
    int y = clamp(int(floor(8.0 + rr * 16.0 + 0.5)), 0, 319), x = clamp(int(floor(8.0 + cc * 16.0 + 0.5)), 0, 479);
    vec3 k = texelFetch(u_rgb, ivec2(x, y), 0).rgb;
    if (u_zkolor == 2) {
        float m = mix(mix(float(MASK[a] & 1u), float(MASK[a + 1u] & 1u), fc), mix(float(MASK[a + 30u] & 1u), float(MASK[a + 31u] & 1u), fc), fr);
        k = k * (1.0 - 0.7 * m) + vec3(1.0, 0.15, 0.1) * 0.7 * m;
    }
    if ((u_att & 2u) != 0u) {                      // heatmapa atencji na zdjęciu (dwuliniowo między patchami)
        float m = mix(mix(att(a), att(a + 1u), fc), mix(att(a + 30u), att(a + 31u), fc), fr);
        k = mix(k, inferno(m), u_att_alfa * (0.15 + 0.85 * m));   // słaba atencja: zostaje zdjęcie
    }
    return k;
}
"""

VS_ZDJ = _NAGLOWEK + _ZDJ_WSPOLNE + """
uniform int u_C, u_n;
uniform float u_alfa;
out vec4 v_kol;
void main() {
    float rr = float(gl_VertexID / u_C) / float(u_n), cc = float(gl_VertexID % u_C) / float(u_n);
    gl_Position = u_mvp * vec4(pozycja_w(rr, cc), 1.0);
    v_kol = vec4(kolor_w(rr, cc), u_alfa);
}
"""

VS_ZLIN = _NAGLOWEK + _ZDJ_WSPOLNE + """
uniform int u_rodzaj;
uniform float u_alfa;
out vec4 v_kol;
void main() {
    int v = gl_VertexID, s = v / 2, r, c;
    if (u_rodzaj == 0) {                             // linie wierszy i kolumn patchy
        if (s < 580) { r = s / 29; c = s % 29 + v % 2; }
        else { s -= 580; r = s / 30 + v % 2; c = s % 30; }
        gl_Position = u_mvp * vec4(pozycja_w(float(r), float(c)), 1.0);
        v_kol = vec4(clamp(kolor_w(float(r), float(c)) * 1.3, 0.0, 1.0), u_alfa);
        return;
    }
    // brzeg obrazu: góra niebieska, lewa żółta, dół i prawa szare (jak w przestrzeni makrowektorów)
    vec4 kol;
    if (s < 29) { r = 0; c = s + v % 2; kol = vec4(0.3, 0.6, 1.0, 0.9); }
    else if (s < 48) { s -= 29; r = s + v % 2; c = 0; kol = vec4(1.0, 0.85, 0.2, 0.9); }
    else if (s < 77) { s -= 48; r = 19; c = s + v % 2; kol = vec4(0.6, 0.6, 0.6, 0.5); }
    else { s -= 77; r = s + v % 2; c = 29; kol = vec4(0.6, 0.6, 0.6, 0.5); }
    gl_Position = u_mvp * vec4(pozycja_w(float(r), float(c)), 1.0);
    v_kol = kol;
}
"""

BUFORY = ('S', 'CUR', 'FR', 'PIB', 'KNOWN', 'HEAT', 'HIST', 'PICK', 'SLOT', 'BB', 'QP', 'MASK', 'SEL', 'CONF', 'PMD', 'KG')


KG_NAGLOWEK = 2048                       # floaty nagłówka Koopmana (średnie, opisy osi, do 127 zestawów wag)
KG_ZESTAWY = (KG_NAGLOWEK - 24) // 16
KPROBKA_W = 20                           # floaty na cząsteczkę próbki Koopmana: h (16), widoczna, indeks, 2 wolne
KPROBKA_MAX = 16384                      # największa próbka (miejsce w części roboczej bufora KG)
KL_ZERO = np.zeros((24, 4), np.float32)  # brak modelu klastrów (u_kl)


def osie_z_wyboru(wyb, czesc=0):
    """Przypisanie osi z automatycznego wyboru modów: X, Y z wybierz_mody, Z = Re, Im albo moduł modu dynamicznego."""
    jz = int(wyb['z'])
    z = ([(jz, 're')], 0) if czesc == 0 else ([(jz, 'im')], 0) if czesc == 1 else ([(jz, 're'), (jz, 'im')], 1)
    return dict(x=dict(skladowe=[(int(wyb['x'][1]), wyb['x'][0])], kombinacja=0),
                y=dict(skladowe=[(int(wyb['y'][1]), wyb['y'][0])], kombinacja=0),
                z=dict(skladowe=z[0], kombinacja=z[1]))


def wybierz_mody(vals, xy=0, z=0, tol=1e-7, pomin=()):
    """Wybór modów Koopmana do osi. vals: wartości własne (zespolone).
    xy: 0 = zespolony niezmiennik najbliższy λ = 1 (X = Re, Y = Im), 1 = dwa rzeczywiste najbliższe 1,
        2 = najbliższy +1 (X) i najbliższy -1 (Y);
    z: mod najbardziej dynamiczny spośród pozostałych: 0 = największe |λ - 1|, 1 = największy |arg λ| (oscylacja),
       2 = największe |ln |λ|| (wzrost albo gaśnięcie).
    Zwraca dict(x=(część, j), y=(część, j), z=j, uzyte=set) - część 're' albo 'im'."""
    vals = np.asarray(vals, complex)
    n = len(vals)
    pomin = set(int(j) for j in pomin)
    if len(pomin) >= n:
        pomin = set()
    rzecz = lambda j: abs(vals[j].imag) <= tol * max(1.0, abs(vals[j]))
    sprzez = lambda j: {k for k in range(n) if abs(vals[k] - np.conj(vals[j])) <= 1e-9 * max(1.0, abs(vals[j]))}
    blisko = lambda cel: sorted((j for j in range(n) if j not in pomin), key=lambda j: (abs(vals[j] - cel), -vals[j].imag))
    wynik = None
    if xy == 0:
        zesp = [j for j in blisko(1.0) if vals[j].imag > tol]
        if zesp:
            j = zesp[0]
            wynik = dict(x=('re', j), y=('im', j), uzyte={j} | sprzez(j))
    if xy == 2:
        j1 = blisko(1.0)[0]
        poza = {j1} | sprzez(j1)
        j2 = next((j for j in blisko(-1.0) if j not in poza), j1)
        wynik = dict(x=('re', j1), y=('re', j2), uzyte=poza | {j2} | sprzez(j2))
    if wynik is None:                                  # xy == 1 albo brak zespolonego: dwa najbliższe 1
        kolej, uzyte, osie = blisko(1.0), set(), []
        for j in kolej:
            if j in uzyte:
                continue
            if rzecz(j):
                osie.append(('re', j)); uzyte |= {j}
            elif not osie or len(osie) == 1:
                osie += [('re', j), ('im', j)] if not osie else [('re', j)]
                uzyte |= {j} | sprzez(j)
            if len(osie) >= 2:
                break
        osie += [osie[-1]] * (2 - len(osie))
        wynik = dict(x=osie[0], y=osie[1], uzyte=uzyte)
    reszta = [j for j in range(n) if j not in wynik['uzyte'] and j not in pomin] or              [j for j in range(n) if j not in pomin] or list(range(n))
    miara = {0: lambda j: abs(vals[j] - 1.0), 1: lambda j: abs(np.angle(vals[j])),
             2: lambda j: abs(np.log(max(abs(vals[j]), 1e-300)))}[int(z)]
    wynik['z'] = max(reszta, key=lambda j: (miara(j), vals[j].imag))
    return wynik


def _ord(x):
    """Kod porządkowy float32 -> uint32 (jak ord() w shaderach)."""
    u = np.asarray(x, np.float32).view(np.uint32)
    return np.where(u & 0x80000000, ~u, u | 0x80000000).astype(np.uint32)


class ChmuraRuchuGPU(GLGraphicsItem):
    """Element sceny pyqtgraph: magazyn stanów na karcie, kernele compute i rysowanie chmury z SSBO.

    Część CPU (bez kontekstu GL): ustaw_os, ustaw_klatke, pompuj, gotowy, zadaj_statystyki, zadaj_wybor.
    Część GL (w paint): wysyłanie stanów, kernele, odczyt wyników, rysowanie, przywrócenie stanu GL.
    """
    statystyki_gotowe = pyqtSignal(object)       # dict: klucz, mapa, mediana, maksimum (z karty)
    koopman_gotowy = pyqtSignal(object)          # dict: widmo, wybrane mody, opis (po dopasowaniu Koopmana)
    probka_koopmana = pyqtSignal(object)
    fizyka_gotowa = pyqtSignal(object)           # dict: właściwości ruchu cząsteczek widoku i serie populacji w krokach         # dict: stany Hankela próbki cząsteczek i rozkład własny (dobór osi)
    wybrano = pyqtSignal(int)                    # indeks cząsteczki albo -1

    def __init__(self, watki=3):
        super().__init__()
        self.setGLOptions({GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True,
                           'glBlendFunc': (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)})
        self._executor = ThreadPoolExecutor(max_workers=watki, thread_name_prefix='ruch-gpu-stany')
        self._prog = None
        self._buf = {}
        self._rozm = {}
        self._uni = {}
        self.renderer = ''
        self.blad = ''                           # błąd GL (kompilacja, alokacja, rysowanie)
        self.blad_odczytu = ''                   # błąd pliku stanu (rozmiar, SHA256, wartości)
        self.max_blok = MAX_BLOK
        self.vram_wolna = None                   # bajty wolnej pamięci karty (NVX), gdy sterownik ją podaje
        self.budzet = 2 << 30
        self.na_klatke = 2                       # ile dodatkowych stanów wysłać w jednym rysowaniu
        self._zeruj_os()
        # parametry klatki (ustawiane przez ustaw_klatke)
        self.a = self.b = None
        self.s = 0.0
        self.ruch = True
        self.kolor = 0
        self.kadr = 0
        self.first, self.count = 0, 0
        self.rozmiar, self.krycie = 1.5, .2
        self.slad = 0
        self.uklad, self.cecha, self.mozaika = 0, 0, False   # układ widoku (patrz ustaw_uklad)
        self.co, self.tylko_narz, self.tylko_zazn = 1, False, False
        self.bez_poz = False                         # cecha bez położenia patcha (siatka i przestrzeń fazowa)
        self.koop = dict(d=3, xy=0, z=0, czesc=0, osie=None)   # Koopman: opóźnienia, wybór automatyczny albo osie
        self.wyg = (0, 5, .3)                         # wygładzanie ruchu: tryb (0 brak, 1 średnia, 2 EMA), okno, α
        self._koop_A = None                          # (klucz, A, wartości, lewe wektory, Cxx, liczba par, średnia, odch.)
        self._koop_klucz_wag = None
        self._kprobka = None                         # zadana próbka stanów Hankela: (m, cały ruch)
        self._kl, self._kl_osie = KL_ZERO, None             # model klastrów Koopmana (kolor 7) i osie, dla których działa
        self.moc_koloru = 1.0                                # nasycenie barw punktów (0 szarość, 1 bez zmian)
        self._obszary = None                                 # klasa obszaru oka każdej cząsteczki (0..3) albo None
        self.obszar_filtr = 0                                # 0 wszystkie, 1..4 tylko obszar (klasa + 1)
        self.fiz = dict(okno=0, rozgrzewka=60, tau=4, prog_a=.7, prog_b=.3, osie=(0, 6, 1), pas_lo=2.0, pas_hi=4.0,
                        energia=0, zak_c=8, zak=((2.0, 3.0, 1.0, 0.0), (3.0, 6.0, 1.0, 0.0), (6.0, 1e9, 1.0, 0.0)),
                        zkol=((1.0, .25, .15), (1.0, .85, .2), (.3, .6, 1.0)),
                        lo=(-1.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0), kol=(-5.0, 0.0, 0.0, 1.0), en_kol=(-8.0, 0.0, 0.0, 0.0))
        self._fiz_pole_klucz = self._fiz_klucz = None
        self._fiz_seria = None
        self._fiz_odczyt = False
        self.fiz_info = ''
        self.koop_info = None
        self.zaznaczone = np.zeros(600, np.uint32)
        self._maska = self._pewnosci = None          # [N] uint32, [T, obrazy] float32 (wysyłane przy rysowaniu)
        self._atencja = None                         # [N] uint16: atencja 0..1 (górne bity MASK)
        self.att = (0, 2.5, .6)                      # heatmapa: bity (1 cząsteczki, 2 zdjęcie), powiększenie, krycie
        self._pomocnicze_brudne = True
        self.zdjecie = (0, 0, .45, .7, 4)            # tryb, kolor, krycie, krycie linii, gęstość
        self.brzeg = False
        self._rgb, self._rgb_brudne, self._tekstura, self._indeksy, self._siatka = None, False, None, None, None
        self.wybrany = None
        self.normscale = 1.0
        self.granice_manifest = None
        self._klucz_kin = None
        self._ramka_brudna = True
        self._cur_ok = False
        self._rys_count = self._rys_first = self.draw_count = 0
        self.ostatni_wybor = None
        self._stat_klucz = None
        self._stat_zrobione = None
        self._wybor = None
        self.fps = 0.0
        self._klatki, self._zegar = 0, time.perf_counter()
        self.czas_gpu_ms = 0.0
        self.wyslane_bajty = 0

    # ------------------------------------------------------------------ CPU: oś czasu i strumień stanów
    def _zeruj_os(self):
        self.timeline = None
        self.steps = []
        self._idx = {}
        self.N = self.T = self.images = self.tpi = 0
        self.frame_bytes = 0
        self.q16 = getattr(self, 'q16', False)
        self.n_slotow = 0
        self._sloty = OrderedDict()              # krok -> slot (kolejność = ostatnie użycie)
        self._wolne = []
        self._gotowe = OrderedDict()             # krok -> np.ndarray czekające na wysłanie
        self._w_toku = {}
        self._generacja = getattr(self, '_generacja', 0) + 1
        self._uklad = None                       # układ buforów na karcie (zmiana = ponowna alokacja)
        self._slot_map_brudna = True
        self._znane = set()

    def ustaw_os(self, timeline, gotowe=None, budzet=None):
        """Nowa oś czasu. `gotowe`: już wczytane stany {krok: [N, 4]} (pierwsza para z wątku wczytywania)."""
        self._zeruj_os()
        self.timeline = timeline
        self.steps = list(timeline.steps)
        self._idx = {s: i for i, s in enumerate(self.steps)}
        self.T, self.N = len(self.steps), int(timeline.count)
        self.images = int(timeline.image_count) or 1
        if self.N % self.images:
            self.images = 1
        self.tpi = self.N // self.images
        self.q16 = getattr(timeline, 'frame_format', 'f32') == 'q16'
        self.frame_bytes = self.N * (8 if self.q16 else 16)     # q16: połowa pamięci karty na stan
        self.normscale = float(timeline.metadata.get('norm_scale', 1.0) or 1.0)
        if budzet is not None:
            self.budzet = int(budzet)
        self._planuj_sloty()
        for krok, arr in (gotowe or {}).items():
            if krok in self._idx:
                self._gotowe[krok] = self._sprawdz(arr)
        self.a = self.b = self.steps[0] if self.steps else None
        self._klucz_kin, self._cur_ok, self._ramka_brudna = None, False, True
        self._stat_klucz = self._stat_zrobione = None
        self.wybrany = None
        self.blad = self.blad_odczytu = ''
        self.update()

    def ustaw_granice_manifestu(self, natural, movement):
        """Granice osi zapisane przez producenta (cały zbiór, wszystkie kroki) - łączone z granicami z karty."""
        self.granice_manifest = (None if natural is None else np.asarray(natural, np.float32),
                                 None if movement is None else np.asarray(movement, np.float32))
        self._ramka_brudna = True
        self.update()

    def ustaw_budzet(self, bajty):
        bajty = int(bajty)
        if bajty != self.budzet:
            self.budzet = bajty
            if self.timeline is not None:
                self._planuj_sloty()
                self._sloty.clear(); self._wolne = []
                self._uklad = None
                self._klucz_kin, self._cur_ok = None, False
                self.update()

    def _planuj_sloty(self):
        if not self.T:
            self.n_slotow = 0
            return
        limit = min(self.budzet, self.max_blok)
        if self.vram_wolna is not None:
            limit = min(limit, int(self.vram_wolna * .8) - (256 << 20))
        self.n_slotow = int(max(3, min(self.T, limit // max(self.frame_bytes, 1))))

    @property
    def cala_os(self):
        """Czy wszystkie zapisane stany mieszczą się na karcie naraz."""
        return self.n_slotow >= self.T

    def _sprawdz(self, arr):
        arr = np.ascontiguousarray(arr, dtype='<u2' if self.q16 else np.float32)
        if arr.shape != (self.N, 4):
            raise ValueError(f'Niezgodny układ stanu: {arr.shape}, oczekiwano ({self.N}, 4).')
        return arr

    @staticmethod
    def _czytaj(timeline, krok, generacja, n):
        """Wątek tła: odczyt stanu z weryfikacją rozmiaru, SHA256 i wartości. Timeline.read czyta plik raz
        (skrót i sprawdzenie z tych samych bajtów); atrapy osi czasu w testach mają tylko frame()."""
        if getattr(timeline, 'frame_format', 'f32') == 'q16':     # słowa uint16 prosto z pliku na kartę
            arr = timeline.read_raw(krok)
            if arr.shape != (n, 4):
                raise ValueError(f'Niezgodny układ stanu {krok}: {arr.shape}, oczekiwano ({n}, 4).')
            return generacja, np.ascontiguousarray(arr, dtype='<u2')
        czytaj = getattr(timeline, 'read', None)
        arr = czytaj(krok) if czytaj is not None else timeline.frame(krok)
        out = np.ascontiguousarray(arr, dtype=np.float32)
        if out is arr and not out.flags.writeable:
            out = out.copy()
        mm = getattr(arr, '_mmap', None)
        if mm is not None:
            out = np.array(out, copy=True)
            try:
                mm.close()
            except (BufferError, ValueError):
                pass
        if out.shape != (n, 4):
            raise ValueError(f'Niezgodny układ stanu {krok}: {out.shape}, oczekiwano ({n}, 4).')
        return generacja, out

    def _plan(self):
        """Kolejność stanów potrzebnych na karcie: pierwszy krok, bieżąca para, następne (cała oś, gdy się mieści)."""
        if not self.T:
            return []
        ia = self._idx.get(self.a, 0)
        ib = self._idx.get(self.b, ia)
        kolej = [0, ia, ib]
        zasieg = self.T if self.cala_os else max(0, min(self.n_slotow - 3, 8))
        kolej += [(ib + 1 + k) % self.T for k in range(zasieg)]
        if self.cala_os:
            kolej += list(range(self.T))
        wynik, byly = [], set()
        for i in kolej:
            if i not in byly:
                byly.add(i); wynik.append(self.steps[i])
        return wynik if self.cala_os else wynik[:self.n_slotow]

    def gotowy(self, krok):
        return krok in self._sloty or krok in self._gotowe

    def pompuj(self, limit=6):
        """Zbiera stany wczytane w tle i zleca następne. Woła GUI (np. co klatkę). Zwraca True, gdy coś trwa."""
        if self.timeline is None:
            return False
        for krok, fut in list(self._w_toku.items()):
            if fut.done():
                del self._w_toku[krok]
                try:
                    gen, arr = fut.result()
                except Exception as e:                       # zły plik: pokaż błąd, nie wysyłaj na kartę
                    self.blad_odczytu = f'stan {krok}: {type(e).__name__}: {e}'
                    continue
                if gen == self._generacja and krok not in self._sloty:
                    self._gotowe[krok] = arr
        for krok in self._plan():
            if len(self._w_toku) + len(self._gotowe) >= limit:
                break
            if not self.gotowy(krok) and krok not in self._w_toku:
                self._w_toku[krok] = self._executor.submit(self._czytaj, self.timeline, krok, self._generacja, self.N)
        if self._gotowe:
            self.update()
        return bool(self._w_toku or self._gotowe)

    def wczytaj_teraz(self, kroki):
        """Synchronicznie (testy, skrypty): wczytaj podane stany do kolejki wysyłki."""
        for krok in kroki:
            if not self.gotowy(krok):
                self._gotowe[krok] = self._czytaj(self.timeline, krok, self._generacja, self.N)[1]
        self.update()

    def zajety(self):
        return bool(self._w_toku or self._gotowe or self._wybor is not None or self._kprobka is not None
                    or (self._stat_klucz is not None and self._stat_klucz != self._stat_zrobione))

    def na_karcie(self):
        return len(self._sloty)

    def pamiec_karty(self):
        """Bajty zajęte na karcie przez bufory tej chmury."""
        return int(sum(self._rozm.values()))

    # ------------------------------------------------------------------ CPU: parametry klatki
    def ustaw_klatke(self, a, b, s, ruch, kolor, kadr, first, count, rozmiar, krycie, slad=0):
        """Tylko liczby: para zapisanych stanów, postęp s (0 = stan a, 1 = stan b), tryby. Bez danych cząsteczek."""
        nowe = (a, b, float(s), bool(ruch), int(kolor), int(kadr), int(first), int(count))
        stare = (self.a, self.b, self.s, self.ruch, self.kolor, self.kadr, self.first, self.count)
        if (bool(ruch), int(kadr), int(first), int(count)) != (self.ruch, self.kadr, self.first, self.count):
            self._ramka_brudna = True
        widok = (float(rozmiar), float(krycie), int(slad)) != (self.rozmiar, self.krycie, self.slad)
        self.a, self.b, self.s, self.ruch, self.kolor, self.kadr, self.first, self.count = nowe
        self.rozmiar, self.krycie, self.slad = float(rozmiar), float(krycie), int(slad)
        if nowe != stare or widok:
            self.update()
            return True
        return False

    def ustaw_wygladzanie(self, tryb=0, okno=5, alfa=.3):
        """Wygładzanie ruchu cząsteczek w czasie (liczy shader z zapisów na karcie): 0 brak, 1 średnia ruchoma
        wyśrodkowana z `okno` zapisów, 2 EMA o współczynniku `alfa` (przyczynowa, od początku osi)."""
        nowe = (int(tryb), max(1, int(okno)), float(min(max(alfa, 1e-3), 1.0)))
        if nowe != self.wyg:
            self.wyg = nowe
            self._koop_klucz_wag = None if self.uklad == 3 else self._koop_klucz_wag   # zakres osi Koopmana od nowa
            self._ramka_brudna, self._klucz_kin = True, None
            self.update()

    def _dlugosc_wyg(self):
        tryb, okno, alfa = self.wyg
        if tryb == 1:
            return okno | 1                                      # okno nieparzyste: wyśrodkowane
        if tryb == 2:
            return 1 if alfa >= 1 else int(min(self.T or 1, np.ceil(np.log(1e-4) / np.log(1 - alfa)) + 1))
        return 1

    def ustaw_koopmana(self, d=3, xy=0, z=0, czesc=0, osie=None):
        """Układ Koopmana: d opóźnień zanurzenia (1-4). Osie: `osie` = {'x'|'y'|'z': {'skladowe': [(nr modu, 're'|'im')],
        'kombinacja': 0 suma | 1 moduł}} z numerami modów z koop_info['mody'][..]['j']; None = wybór automatyczny
        (niezmienniki na X, Y według xy, mod dynamiczny na Z według z, jego część czesc: 0 Re, 1 Im, 2 moduł)."""
        if osie is not None:
            osie = {a: dict(skladowe=[(int(j), str(c)) for j, c in osie[a]['skladowe']],
                            kombinacja=int(osie[a].get('kombinacja', 0))) for a in ('x', 'y', 'z')}
        nowe = dict(d=max(1, min(4, int(d))), xy=int(xy), z=int(z), czesc=int(czesc), osie=osie)
        if nowe != self.koop:
            if nowe['d'] != self.koop['d']:
                self._koop_A = None
            self.koop = nowe
            self._koop_klucz_wag = None
            self._ramka_brudna, self._klucz_kin = True, None
            self.update()

    def zadaj_probke_koopmana(self, m=4096, caly_ruch=False):
        """Próbka m cząsteczek widoku (równomiernie po zakresie first..first+count): stany Hankela h z karty w bieżącym
        zapisie albo (caly_ruch) w 8 zapisach rozłożonych po osi czasu. Wynik przez sygnał `probka_koopmana`."""
        self._kprobka = (max(16, min(int(m), KPROBKA_MAX)), bool(caly_ruch))
        self.update()

    def _probka_koopmana(self):
        if self._kprobka is None:
            return
        if self.uklad != 3:
            self._kprobka = None
            self.probka_koopmana.emit(dict(blad='Dobór osi pod klastry działa w układzie „Koopman”.'))
            return
        if self._koop_A is None or self._koop_klucz_wag is None:
            return                                              # Koopman jeszcze się nie dopasował: próbka później
        m, caly = self._kprobka
        self._kprobka = None
        d = self._koop_d
        biezacy = self._idx.get(self.a, 0)
        if caly:                                                # 8 zapisów rozłożonych po osi i bieżący
            kroki = sorted(set(int(round(x)) for x in np.linspace(d - 1, self.T - 1, min(8, self.T - d + 1))) | {biezacy})
        else:
            kroki = [biezacy]
        first, count = self.first, min(self.count, self.N - self.first)
        mk = max(1, min(m // len(kroki), count))
        for n, kk in enumerate(kroki):
            self._uniformy('kprobka', u_N=self.N, u_M=mk, u_first=first, u_count=count, u_kk=kk, u_off=n * mk,
                           **dict(self._wspolne(), u_kd=d))
            GL.glDispatchCompute((mk + 255) // 256, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        raw = self._czytaj_bufor('KG', np.float32, len(kroki) * mk * KPROBKA_W, KG_NAGLOWEK * 4).reshape(-1, KPROBKA_W)
        ok = raw[:, 16] > .5
        _, A, vals, W, Cxx, npar, mu, sd = self._koop_A
        war, prog, zdeg = self._koop_war
        grupy = np.repeat(np.arange(len(kroki)), mk)[ok]
        self.probka_koopmana.emit(dict(H=raw[ok, :4 * d].astype(np.float64), czastki=raw[ok, 17].astype(np.int64),
                                       grupy=grupy, biezacy=kroki.index(biezacy),
                                       kroki=[self.steps[k] for k in kroki], vals=vals, W=W, war=war, prog=prog,
                                       zdeg=sorted(zdeg), d=d, osie=(self.koop_info or {}).get('osie')))

    # ------------------------------------------------------------------ obszary oka i fizyka ruchu
    def ustaw_obszary(self, klasy=None):
        """Klasa obszaru oka każdej cząsteczki [N] (0 tło, 1 rogówka, 2 źrenica, 3 narzędzie) albo None."""
        self._obszary = None if klasy is None else np.ascontiguousarray(klasy, np.uint8).reshape(-1)
        self._pomocnicze_brudne = True
        self._fiz_pole_klucz = self._fiz_klucz = None
        self.update()

    def _wyslij_czasy(self):
        """Kroki treningu zapisów do KG[pole0 + 4k + 3] (kolor 14); pole kolektywne fizyki zostaje w xyz."""
        if getattr(self, '_czasy_ok', None) == self._generacja or not hasattr(self, '_pole0'):
            return
        pk = getattr(self, '_pole_np', None)
        if pk is None or len(pk) != self.T:
            pk = np.zeros((self.T, 4), np.float32)
        pk[:, 3] = np.asarray(self.steps, np.float32)
        self._pole_np = pk
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['KG'])
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, self._pole0 * 4, pk.nbytes, pk)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        self._czasy_ok = self._generacja

    def ustaw_filtr_obszaru(self, obszar):
        """Widoczne tylko cząsteczki obszaru: 0 wszystkie, 1 tło, 2 rogówka, 3 źrenica, 4 narzędzie."""
        if int(obszar) != self.obszar_filtr:
            self.obszar_filtr = int(obszar)
            self.update()

    def ustaw_kolory_fizyki(self, energia=None, zak_c=None, zak=None, zkol=None, en_kol=None, pas=None):
        """Kolor 15 (energia 0 kinetyczna, 1 potencjalna, 2 całkowita; zakres log en_kol), kolor 17 (właściwość zak_c,
        do 3 zakresów zak = [(od, do, włączony, 0)] i kolorów zkol), pasmo okresów (od, do) w zapisach dla koloru 16."""
        nowe = dict(self.fiz)
        for k, v in (('energia', energia), ('zak_c', zak_c), ('en_kol', en_kol)):
            if v is not None:
                nowe[k] = tuple(float(x) for x in v) if k == 'en_kol' else int(v)
        if zak is not None:
            nowe['zak'] = tuple(tuple(float(x) for x in z) for z in zak)
        if zkol is not None:
            nowe['zkol'] = tuple(tuple(float(x) for x in z) for z in zkol)
        if pas is not None:
            nowe['pas_lo'], nowe['pas_hi'] = float(pas[0]), float(pas[1])
        if nowe != self.fiz:
            self.fiz = nowe
            self.update()

    def ustaw_fizyke(self, okno=None, rozgrzewka=None, tau=None, prog_a=None, prog_b=None, osie=None):
        """Parametry właściwości ruchu: okno (zapisów w każdą stronę bieżącego kroku; 0 = cała oś), koniec rozgrzewki
        (krok treningu), τ dla wykładnika MSD (zapisy), progi reżimów (α dyfuzji, VACF i prostoliniowość balistyki),
        osie układu 4 (numery właściwości 0..6)."""
        nowe = dict(self.fiz)
        for k, v in (('okno', okno), ('rozgrzewka', rozgrzewka), ('tau', tau), ('prog_a', prog_a), ('prog_b', prog_b)):
            if v is not None:
                nowe[k] = v
        if osie is not None:
            nowe['osie'] = tuple(int(o) for o in osie)
        if nowe != self.fiz:
            self.fiz = nowe
            self._ramka_brudna, self._klucz_kin = True, None
            self.update()

    def ustaw_zakresy_fizyki(self, lo, hi, kol):
        """Zakres osi układu 4 (po przekształceniu log) i kolorów (log temperatury, α), liczone przez GUI z odczytu."""
        self.fiz.update(lo=tuple(float(v) for v in lo), hi=tuple(float(v) for v in hi), kol=tuple(float(v) for v in kol))
        self._ramka_brudna, self._klucz_kin = True, None
        self.update()

    def zadaj_odczyt_fizyki(self):
        """Odczytaj właściwości ruchu z karty po następnym przeliczeniu (wynik przez sygnał fizyka_gotowa)."""
        self._fiz_odczyt = True
        self.update()

    def _fiz_ok(self):
        return int(self._fiz_klucz is not None and self._fiz_klucz[0][1:3] == (self.first, min(self.count, self.N - self.first)))

    def _fizyka(self):
        """Statystyki populacji w każdym kroku (4 obszary) i właściwości ruchu cząsteczek widoku, liczone na karcie."""
        potrzebna = self.uklad == 4 or self.kolor in (8, 9, 10, 15, 16, 17) or self._fiz_odczyt
        if not potrzebna or self._prog is None or 'wlasc' not in self._prog:
            return
        if self.na_karcie() < self.T or self.T < 3:
            self.fiz_info = f'Fizyka ruchu czeka na stany na karcie ({self.na_karcie()}/{self.T}).'
            return
        first, count = self.first, min(self.count, self.N - self.first)
        G = (count + 255) // 256
        kp = (self._generacja, first, count, None if self._obszary is None else hash(self._obszary.tobytes()))
        if self._fiz_pole_klucz != kp:
            self._odswiez_mape_slotow(); self._wiaz()
            K = 512 // 24
            suma = np.zeros((self.T, 24))
            for k0 in range(0, self.T, K):
                kk = min(K, self.T - k0)
                self._uniformy('ruchk', u_N=self.N, u_first=first, u_count=count, u_k0=k0, u_K=kk, **self._wspolne())
                GL.glDispatchCompute(G, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                cz = self._czytaj_bufor('KG', np.float32, G * kk * 24, KG_NAGLOWEK * 4).reshape(G, kk, 24)
                suma[k0:k0 + kk] = cz.astype(np.float64).sum(0)
            S = suma.reshape(self.T, 4, 6)
            n = S[:, :, 4]
            nc = n.sum(1)
            pole = S[:, :, :3].sum(1) / np.maximum(nc, 1)[:, None]                  # średni przyrost populacji
            pole[0] = 0
            pk = np.zeros((self.T, 4), np.float32); pk[:, :3] = pole; pk[:, 3] = np.asarray(self.steps, np.float32)
            self._pole_np, self._czasy_ok = pk, self._generacja
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['KG'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, self._pole0 * 4, pk.nbytes, pk)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
            with np.errstate(invalid='ignore', divide='ignore'):
                temp = (S[:, :, 3] - 2 * (pole[:, None, :] * S[:, :, :3]).sum(2) + n * (pole ** 2).sum(1)[:, None]) / n
                cosk = S[:, :, 5] / n
            temp[0] = np.nan; cosk[:2] = np.nan
            calk = S[:, :, 3].sum(1) / np.maximum(nc, 1)
            self._fiz_seria = dict(kroki=list(self.steps), temperatura=temp, cos=cosk, n=n[1],
                                   pole=(pole ** 2).sum(1), energia=calk,
                                   kolektywnosc=np.where(calk > 0, (pole ** 2).sum(1) / np.maximum(calk, 1e-30), np.nan))
            self._fiz_pole_klucz, self._fiz_klucz = kp, None
        import bisect
        ka, kb = self._idx.get(self.a, 0), self._idx.get(self.b, 0)
        W = int(self.fiz['okno'])

        def okno(kc):
            return (0, self.T - 1) if W <= 0 else (max(0, kc - W), min(self.T - 1, kc + W))
        k0, k1 = okno(ka)
        kB = okno(kb)
        kw = min(bisect.bisect_left(self.steps, int(self.fiz['rozgrzewka'])), self.T - 1)
        tau = max(2, min(int(self.fiz['tau']), max(k1 - k0, 2)))
        klucz = (kp, (k0, k1), kB, kw, tau, float(self.fiz['prog_a']), float(self.fiz['prog_b']),
                 float(self.fiz['pas_lo']), float(self.fiz['pas_hi']))
        if klucz != self._fiz_klucz:
            self._odswiez_mape_slotow(); self._wiaz()
            okna = [((k0, k1), self._wl0)] + ([(kB, self._wl1)] if kB != (k0, k1) else [])
            for (o0, o1), baza in okna:          # okno wokół a (kolory, statystyki) i wokół b (płynny ruch układu 4)
                t_ = max(2, min(int(self.fiz['tau']), max(o1 - o0, 2)))
                self._uniformy('wlasc', u_N=self.N, u_first=first, u_count=count, u_ka0=o0, u_kb0=o1, u_kwarm=kw,
                               u_tau=t_, u_pole0=self._pole0, u_wl_out=baza, u_prog_a=float(self.fiz['prog_a']),
                               u_prog_b=float(self.fiz['prog_b']), u_pas_lo=float(self.fiz['pas_lo']),
                               u_pas_hi=float(self.fiz['pas_hi']), **self._wspolne())
                GL.glDispatchCompute(G, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            self._wl1_uzyte = okna[-1][1]
            self._fiz_klucz = klucz
            self._ramka_brudna, self._klucz_kin = True, None
            self.fiz_info = ''
        if self._fiz_odczyt:
            self._fiz_odczyt = False
            wl = self._czytaj_bufor('KG', np.float32, count * 12, (self._wl0 + 12 * first) * 4).reshape(count, 12)
            self.fizyka_gotowa.emit(dict(wl=wl, first=first, count=count, okno=(self.steps[k0], self.steps[k1]),
                                         rozgrzewka=self.steps[kw], tau=tau, seria=self._fiz_seria,
                                         obszary=None if self._obszary is None else self._obszary[first:first + count]))

    def ustaw_moc_koloru(self, moc):
        """Moc kolorowania punktów: 0 = odcienie szarości, 1 = barwa trybu koloru, do 3 = mocniej nasycona."""
        moc = float(min(max(moc, 0.0), 3.0))
        if moc != self.moc_koloru:
            self.moc_koloru = moc
            self.update()

    def ustaw_klastry(self, model=None):
        """Model klastrów z klastry_koopmana.dobierz (mieszanina: μ, σ, log π, próg; standaryzacja osi) do koloru 7. Działa tylko,
        dopóki osie Koopmana są tymi, dla których go policzono; None = brak."""
        kl = np.zeros((24, 4), np.float32)
        if model is not None:
            n = min(len(model['srodki']), 10)
            kl[0, :3], kl[0, 3] = model['mediana'], n
            kl[1, :3], kl[1, 3] = model['skala'], model['wym']
            kl[2, 0] = model['prog']
            sig = np.asarray(model['sigma'], np.float64)[:n]
            kl[3:3 + 2 * n:2, :3] = np.asarray(model['srodki'])[:n]
            kl[3:3 + 2 * n:2, 3] = np.asarray(model['logpi'])[:n] - np.log(sig).sum(1)
            kl[4:4 + 2 * n:2, :3] = sig
            kl[4:4 + 2 * n:2, 3] = np.where(np.asarray(model['duzy'])[:n], 1.0, -1.0)
            osie = {a: dict(skladowe=[(int(j), str(c)) for j, c in model['osie'][a]['skladowe']],
                            kombinacja=int(model['osie'][a].get('kombinacja', 0))) for a in ('x', 'y', 'z')}
        self._kl, self._kl_osie = kl, (osie if model is not None else None)
        self.update()

    def przelicz_koopmana(self):
        self._koop_A, self._koop_klucz_wag = None, None
        self._ramka_brudna, self._klucz_kin = True, None
        self.update()

    def _koopman(self):
        """Dopasowanie operatora Koopmana (Hankel-DMD) na karcie i wagi funkcji własnych dla osi. Tylko gdy wszystkie
        zapisane stany są na karcie. Zwraca True, gdy wagi są gotowe."""
        if self.uklad != 3:
            return True
        if self.na_karcie() < self.T or self.T < 3:
            self.koop_info = {'opis': f'Koopman czeka na stany na karcie ({self.na_karcie()}/{self.T}).'}
            return False
        d = min(self.koop['d'], self.T - 2)
        klucz = (self._generacja, d)
        G = (self.N + 255) // 256
        if self._koop_A is None or self._koop_A[0] != klucz:
            self._odswiez_mape_slotow(); self._wiaz()
            self._uniformy('kstat', u_N=self.N, u_T=self.T, **self._wspolne())
            GL.glDispatchCompute(G, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            cz = self._czytaj_bufor('KG', np.float32, G * 8, KG_NAGLOWEK * 4).reshape(G, 8).astype(np.float64).sum(0)
            n = float(self.N * self.T)
            mu = cz[:4] / n
            sd = np.sqrt(np.maximum(cz[4:] / n - mu * mu, 1e-12))
            glowa = np.zeros(8, np.float32); glowa[:4] = mu; glowa[4:] = 1.0 / sd
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['KG'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, glowa.nbytes, glowa)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            self._uniformy('kgram', u_N=self.N, u_T=self.T, **dict(self._wspolne(), u_kd=d))
            GL.glDispatchCompute(G, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            D = 4 * d
            P = self._czytaj_bufor('KG', np.float32, G * 512, KG_NAGLOWEK * 4).reshape(G, 512)[:, :2 * D * D]
            P = P.astype(np.float64).sum(0)
            Cxx, Cyx = P[:D * D].reshape(D, D), P[D * D:].reshape(D, D)
            ridge = 1e-8 * max(np.trace(Cxx) / D, 1e-30)
            A = Cyx @ np.linalg.inv(Cxx + ridge * np.eye(D))
            vals, W = np.linalg.eig(A.T)                       # kolumny: lewe wektory własne (w^T A = λ w^T)
            npar = float(self.N * (self.T - d))
            self._koop_A = (klucz, A, vals, W, Cxx, npar, mu, sd)
            self._koop_klucz_wag = None
        osie_zadane = self.koop.get('osie')
        klucz_wag = (klucz, self.koop['xy'], self.koop['z'], self.koop['czesc'], repr(osie_zadane))
        if self._koop_klucz_wag != klucz_wag:
            _, A, vals, W, Cxx, npar, mu, sd = self._koop_A
            # wariancja funkcji własnych na danych: mody o wariancji ~0 to artefakty zanurzenia (np. więcej opóźnień niż
            # potrzeba) - nie trafiają do wyboru ani tabeli, a ich skala ma dolną granicę (bez dzielenia przez zero)
            war = np.array([max(float(np.real(np.conj(W[:, j]) @ Cxx @ W[:, j])) / npar, 0.0) for j in range(len(vals))])
            prog = 1e-8 * max(float(np.trace(Cxx)) / npar / len(vals), 1e-30)
            zdeg = set(int(j) for j in np.flatnonzero(war < prog))
            self._koop_war = (war, prog, zdeg)
            wyb = wybierz_mody(vals, self.koop['xy'], self.koop['z'], pomin=zdeg)
            osie = osie_zadane
            if osie is None or any(j >= len(vals) for a in osie.values() for j, _ in a['skladowe']):
                osie = osie_z_wyboru(wyb, self.koop['czesc'])
            skala = lambda j: 1.0 / np.sqrt(max(war[j], prog))
            czesc = lambda j, c: (np.real if c == 're' else np.imag)(W[:, j] * skala(j))
            naglowek = np.zeros(KG_NAGLOWEK - 8, np.float32)       # KG[8..]: opisy osi i zestawy wag
            zestawy = []
            for a, nazwa in enumerate(('x', 'y', 'z')):
                wektory = [czesc(j, c) for j, c in osie[nazwa]['skladowe']] or [np.zeros(4 * d)]
                if osie[nazwa]['kombinacja'] == 0:
                    wektory = [np.sum(wektory, axis=0)]             # suma = jeden rzut liniowy
                naglowek[3 * a:3 * a + 3] = (len(zestawy), len(wektory), osie[nazwa]['kombinacja'])
                zestawy += wektory
            if len(zestawy) > KG_ZESTAWY:
                raise ValueError(f'za dużo składowych modów na osiach ({len(zestawy)} > {KG_ZESTAWY})')
            for zs, w in enumerate(zestawy):
                naglowek[16 + 16 * zs:16 + 16 * zs + 4 * d] = np.asarray(w, np.float32)   # KG[24 + 16 zs]
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['KG'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 8 * 4, naglowek.nbytes, naglowek)
            init = np.array([NONE] * 3 + [0] * 3, np.uint32)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['HIST'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, (R_HIST + 16) * 4, init.nbytes, init)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
            self._uniformy('kzakres', u_N=self.N, u_T=self.T, **dict(self._wspolne(), u_kd=d))
            GL.glDispatchCompute(G, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            self._koop_klucz_wag = klucz_wag
            self._koop_d = d
            krok = float(np.mean(np.diff(self.steps))) if len(self.steps) > 1 else 1.0
            mody = []
            for j in np.argsort(np.abs(vals - 1.0)):
                if int(j) in zdeg:
                    continue
                lam = complex(vals[j]); kat = float(np.degrees(np.angle(lam)))
                rola = ', '.join(f'{os_.upper()} {c}' for os_ in ('x', 'y', 'z') for jj, c in osie[os_]['skladowe']
                                 if jj == j)
                mody.append(dict(j=int(j), re=lam.real, im=lam.imag, modul=abs(lam), kat=kat,
                                 okres=(360.0 / abs(kat) * krok) if abs(kat) > 1e-9 else None, rola=rola))
            self.koop_info = dict(d=d, mody=mody, x=wyb['x'], y=wyb['y'], z=int(wyb['z']), krok=krok,
                                  pary=int(npar), opis='', osie=osie, automatyczne=osie_zadane is None,
                                  zdegenerowane=len(zdeg))
            self.koopman_gotowy.emit(self.koop_info)
            self._ramka_brudna, self._klucz_kin = True, None
        return True

    def ustaw_bez_polozenia(self, wl):
        """Odejmij od cechy średnią tego patcha po obrazach w tym samym kroku (usuwa położenie patcha na obrazie)."""
        if bool(wl) != self.bez_poz:
            self.bez_poz = bool(wl)
            self._ramka_brudna, self._klucz_kin = True, None
            self.update()

    def ustaw_uklad(self, uklad, cecha=0, mozaika=False):
        """Układ widoku: 0 PCA 3D, 1 siatka zdjęcia + cecha na Z, 2 przestrzeń fazowa (cecha, jej prędkość,
        norma 768D). Cecha: 0-2 składowe PCA, 3 norma zmiany 768D, 4 odległość od środka PCA."""
        nowe = (int(uklad), int(cecha), bool(mozaika))
        if nowe != (self.uklad, self.cecha, self.mozaika):
            self.uklad, self.cecha, self.mozaika = nowe
            self._ramka_brudna, self._klucz_kin = True, None
            self.update()

    def ustaw_wyglad(self, co=1, tylko_narz=False, tylko_zazn=False):
        """Filtr widocznych cząsteczek: co n-ty patch w wierszach i kolumnach, tylko patche w ramce GT, tylko
        zaznaczone patche."""
        nowe = (max(1, int(co)), bool(tylko_narz), bool(tylko_zazn))
        if nowe != (self.co, self.tylko_narz, self.tylko_zazn):
            self.co, self.tylko_narz, self.tylko_zazn = nowe
            self.update()

    def ustaw_zaznaczenie(self, patche):
        """Zaznaczone patche (numery 0..599, wspólne dla wszystkich obrazów); pusty = brak zaznaczenia."""
        z = np.zeros(600, np.uint32)
        z[np.asarray(sorted(patche), int)] = 1 if len(patche) else 0
        if not np.array_equal(z, self.zaznaczone):
            self.zaznaczone = z
            self._pomocnicze_brudne = True
            self.update()

    def ustaw_dane_pomocnicze(self, maska=None, pewnosci=None):
        """maska [N] (1 = patch w ramce GT obrazu), pewnosci [T, obrazy] (pewność ViT obrazu w zapisanym kroku)."""
        self._maska = None if maska is None else np.ascontiguousarray(maska, np.uint32)
        self._pewnosci = None if pewnosci is None else np.ascontiguousarray(pewnosci, np.float32)
        self._pomocnicze_brudne = True
        self.update()

    def ustaw_zdjecie(self, rgb=None, tryb=0, kolor=0, krycie=.45, gestosc=4, brzeg=False):
        """Zdjęcie [320, 480, 3] uint8 rozpięte na cząsteczkach widocznego obrazu (tylko widok jednego obrazu):
        tryb 0 brak, 1 powierzchnia, 2 siatka linii, 3 oba; kolor 0 zdjęcie, 1 barwy cząsteczek, 2 narzędzie na czerwono."""
        if rgb is not self._rgb:
            self._rgb, self._rgb_brudne = rgb, rgb is not None
        al = min(1., krycie * 1.6) if tryb == 2 else min(1., krycie + .25)
        self.zdjecie = (int(tryb), int(kolor), float(krycie), float(al), max(1, int(gestosc)))
        self.brzeg = bool(brzeg)
        self.update()

    def ustaw_atencje(self, wartosci=None, first=0, tryb=None, wzm=None, krycie=None):
        """Heatmapa atencji: wartosci [n] w 0..1 dla cząsteczek first..first+n-1 (None = bez zmiany danych);
        tryb: bity 1 (kolor inferno i rozmiar cząsteczek) i 2 (na zdjęciu); wzm: rozmiar = rozmiar · (0,4 + wzm · mapa)."""
        if wartosci is not None and self.N:
            if self._atencja is None or len(self._atencja) != self.N:
                self._atencja = np.zeros(self.N, np.uint32)
            v = np.clip(np.asarray(wartosci, np.float32), 0, 1)
            self._atencja[first:first + len(v)] = np.rint(v * 65535).astype(np.uint32)
            self._pomocnicze_brudne = True
        nowe = (self.att[0] if tryb is None else int(tryb), self.att[1] if wzm is None else float(wzm),
                self.att[2] if krycie is None else float(krycie))
        if nowe != self.att or wartosci is not None:
            self.att = nowe
            self.update()

    def wyczysc_atencje(self):
        self._atencja = None
        self._pomocnicze_brudne = True
        self.update()

    def _wyslij_pomocnicze(self):
        if not self._pomocnicze_brudne:
            return
        maska = self._maska if self._maska is not None and self._maska.size == self.N else np.zeros(self.N, np.uint32)
        maska = maska & 1
        if self._obszary is not None and self._obszary.size == self.N:
            maska = maska | (self._obszary.astype(np.uint32) << 1)
        if self._atencja is not None and self._atencja.size == self.N:
            maska = maska | (self._atencja << 16)
        for nazwa, dane, n in (('MASK', maska.astype(np.uint32), self.N), ('SEL', self.zaznaczone, 600),
                               ('CONF', self._pewnosci, self.T * self.images)):
            arr = dane if dane is not None and dane.size == n else np.zeros(max(n, 1), np.uint32 if nazwa != 'CONF' else np.float32)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[nazwa])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, arr.nbytes, arr)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._pomocnicze_brudne = False

    def _wyslij_obraz(self):
        if not self._rgb_brudne or self._rgb is None:
            return
        rgb = np.ascontiguousarray(np.asarray(self._rgb)[:320, :480, :3], np.uint8)
        if self._tekstura is None:
            self._tekstura = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tekstura)
        stare = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, rgb.shape[1], rgb.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, int(stare))           # QPainter (napisy) zakłada swoją wartość
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._rgb_brudne = False

    def _indeksy_siatki(self, R, C):
        if self._siatka == (R, C):
            return
        i = np.arange(R * C).reshape(R, C)
        q = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1).reshape(-1, 4)
        tw = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.uint32)
        if self._indeksy is None:
            self._indeksy = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._indeksy)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, tw.nbytes, tw, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        self._siatka, self._n_ind = (R, C), tw.size

    def kafle(self):
        """Ile kafli obrazów w wierszu mozaiki (kafel 31 × 21 patchy; całość w przybliżeniu kwadratowa)."""
        return max(1, int(np.ceil(np.sqrt(max(self.images, 1) * 21 / 31))))

    def granice_siatki(self, img0, img1):
        """Granice X, Y siatki zdjęcia dla obrazów img0..img1-1 (w mozaice: ich kafle)."""
        wiersze = max(self.tpi // 30, 1) if self.tpi else 20
        if not self.mozaika:
            return (0.0, -(wiersze - 1.0)), (29.0, 0.0)
        C = self.kafle()
        im = np.arange(img0, max(img1, img0 + 1))
        x0, y0 = (im % C) * 31.0, -(im // C) * 21.0
        return (float(x0.min()), float((y0 - (wiersze - 1)).min())), (float(x0.max() + 29), float(y0.max()))

    def _wspolne(self):
        return dict(u_uklad=self.uklad, u_cecha=self.cecha, u_mozaika=self.mozaika, u_kafle=self.kafle(),
                    u_tpi_w=self.tpi or 600, u_co=self.co, u_tylko_narz=self.tylko_narz, u_tylko_zazn=self.tylko_zazn,
                    u_jest_zazn=bool(self.zaznaczone.any()), u_ka=self._idx.get(self.a, 0), u_img_w=self.images,
                    u_jest_conf=self._pewnosci is not None, u_bez_poz=self.bez_poz,
                    u_kd=getattr(self, '_koop_d', self.koop['d']), u_kz=self.koop['czesc'],
                    u_wyg=self.wyg[0], u_wyg_n=self._dlugosc_wyg(), u_wyg_a=float(self.wyg[2]), u_T_w=self.T,
                    u_att=self.att[0], u_att_wzm=float(self.att[1]), u_att_alfa=float(self.att[2]),
                    u_moc=float(self.moc_koloru),
                    u_jest_obszary=self._obszary is not None, u_obszar=self.obszar_filtr,
                    u_wl0=getattr(self, '_wl0', 0), u_wl1=getattr(self, '_wl1_uzyte', getattr(self, '_wl0', 0)),
                    u_wl_ok=self._fiz_ok(),
                    u_wx=self.fiz['osie'][0], u_wy=self.fiz['osie'][1], u_wz=self.fiz['osie'][2],
                    u_wl_lo=tuple(self.fiz['lo']), u_wl_hi=tuple(self.fiz['hi']), u_wl_kol=tuple(self.fiz['kol']),
                    u_en=int(self.fiz['energia']), u_en_kol=tuple(self.fiz['en_kol']), u_zak_c=int(self.fiz['zak_c']),
                    u_zak=np.asarray(self.fiz['zak'], np.float32).reshape(3, 4),
                    u_zkol=np.asarray(self.fiz['zkol'], np.float32).reshape(3, 3),
                    u_kl=self._kl if self._kl_osie is not None and self._kl_osie == self.koop.get('osie') else KL_ZERO)

    def ustaw_wybrany(self, indeks):
        self.wybrany = None if indeks is None or indeks < 0 else int(indeks)
        self.update()

    def zadaj_statystyki(self, krok, first, count):
        """Mapa 20 × 30 i dokładna mediana normy 768D w widoku dla zapisanego kroku (liczone na karcie)."""
        klucz = (krok, int(first), int(count), self._generacja)
        if klucz != self._stat_klucz:
            self._stat_klucz = klucz
            self.update()

    def zadaj_wybor(self, x, y, w, h):
        """Kliknięcie w pikselach widżetu (x, y) przy rozmiarze (w, h); wynik przez sygnał `wybrano`."""
        self._wybor = (float(x), float(y), float(w), float(h))
        self.update()

    # ------------------------------------------------------------------ GL
    def _program(self, vs=None, fs=None, cs=None):
        wariant = lambda src: src.replace('//Q16DEF', f'#define Q16 {int(self.q16)}')
        if cs is not None:
            return compileProgram(compileShader(wariant(cs), GL.GL_COMPUTE_SHADER))
        return compileProgram(compileShader(wariant(vs), GL.GL_VERTEX_SHADER), compileShader(fs, GL.GL_FRAGMENT_SHADER),
                              validate=False)

    def _kompiluj(self):
        """Programy w wariancie formatu bieżącej osi czasu (f32 albo q16); ponownie tylko przy zmianie formatu."""
        if self._prog:
            for prog in self._prog.values():
                GL.glDeleteProgram(prog)
        self._prog = dict(kin=self._program(cs=CS_KIN), stat=self._program(cs=CS_STAT), heat=self._program(cs=CS_HEAT),
                          bb=self._program(cs=CS_BB), ramka=self._program(cs=CS_RAMKA), hist=self._program(cs=CS_HIST),
                          wybor=self._program(cs=CS_WYBOR), pick=self._program(cs=CS_PICK),
                          punkty=self._program(VS_PUNKTY, FS_PUNKTY), slad=self._program(VS_SLAD, FS_KOLOR),
                          zdj=self._program(VS_ZDJ, FS_KOLOR), zlin=self._program(VS_ZLIN, FS_KOLOR),
                          kstat=self._program(cs=CS_KSTAT), kgram=self._program(cs=CS_KGRAM),
                          kzakres=self._program(cs=CS_KZAKRES), kprobka=self._program(cs=CS_KPROBKA),
                          ruchk=self._program(cs=CS_RUCHK), wlasc=self._program(cs=CS_WLASC))
        self._uni = {}
        self._prog_q16 = self.q16

    def _init(self):
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
        try:
            self.max_blok = int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE))
        except Exception:
            self.max_blok = MAX_BLOK
        if self.max_blok <= 0:
            self.max_blok = MAX_BLOK
        self._odswiez_vram()
        self._kompiluj()
        ids = np.atleast_1d(GL.glGenBuffers(len(BUFORY) + 1))
        self._buf = dict(zip(BUFORY + ('RB',), (int(i) for i in ids)))     # RB: mały bufor odczytu
        self._uni = {}
        self._planuj_sloty()

    def _odswiez_vram(self):
        try:
            kb = int(np.atleast_1d(GL.glGetIntegerv(0x9049))[0])      # GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
            self.vram_wolna = kb * 1024 + self.pamiec_karty()
        except Exception:
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass
            self.vram_wolna = None

    def _u(self, prog, nazwa):
        k = (prog, nazwa)
        if k not in self._uni:
            self._uni[k] = GL.glGetUniformLocation(self._prog[prog], nazwa)
        return self._uni[k]

    def _uniformy(self, prog, **wartosci):
        GL.glUseProgram(self._prog[prog])
        for nazwa, v in wartosci.items():
            loc = self._u(prog, nazwa)
            if loc < 0:
                continue
            if isinstance(v, np.ndarray) and v.shape == (4, 4):
                GL.glUniformMatrix4fv(loc, 1, False, v)
            elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 4:          # tablica vec4 (u_kl, u_zak)
                GL.glUniform4fv(loc, len(v), np.ascontiguousarray(v, np.float32))
            elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:          # tablica vec3 (u_zkol)
                GL.glUniform3fv(loc, len(v), np.ascontiguousarray(v, np.float32))
            elif isinstance(v, (tuple, list, np.ndarray)):
                v = [float(x) for x in v]
                {2: GL.glUniform2f, 3: GL.glUniform3f, 4: GL.glUniform4f}[len(v)](loc, *v)
            elif isinstance(v, float):
                GL.glUniform1f(loc, v)
            elif isinstance(v, bool):
                GL.glUniform1ui(loc, int(v))
            elif nazwa in ('u_kolor', 'u_wybrany', 'u_zkolor', 'u_C', 'u_n', 'u_rodzaj', 'u_rgb'):   # GLSL int
                GL.glUniform1i(loc, int(v))
            else:
                GL.glUniform1ui(loc, int(v) & 0xFFFFFFFF)

    def _alokuj(self, nazwa, bajty, dane=None, uzycie=GL.GL_DYNAMIC_DRAW):
        while GL.glGetError() != GL.GL_NO_ERROR:          # błędy innych elementów nie są błędem alokacji
            pass
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[nazwa])
        bajty = max(int(bajty), 16)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, bajty, dane, uzycie)
        if GL.glGetError() == GL.GL_OUT_OF_MEMORY:
            raise MemoryError(f'Za mało pamięci karty na bufor {nazwa} ({bajty / 1e6:.0f} MB).')
        self._rozm[nazwa] = bajty

    def _przygotuj(self):
        """Alokacja buforów dla bieżącej osi czasu i budżetu (tylko przy zmianie układu)."""
        if self._uklad is not None and self._uklad[:5] == (self._generacja, self.N, self.T, self.images, self.tpi)                 and self._uklad[5] == self.n_slotow:
            return
        self._odswiez_vram()
        self._planuj_sloty()
        while True:
            try:
                self._alokuj('S', self.n_slotow * self.frame_bytes)
                break
            except MemoryError:
                if self.n_slotow <= 3:
                    raise
                self.n_slotow = max(3, self.n_slotow // 2)
        self._alokuj('CUR', self.N * 16, np.zeros((self.N, 4), np.float32))
        self._alokuj('QP', self.n_slotow * 32, np.zeros((self.n_slotow * 2, 4), np.float32))
        self._alokuj('MASK', self.N * 4)
        self._alokuj('SEL', 600 * 4)
        self._alokuj('CONF', max(self.T * self.images, 1) * 4)
        self._alokuj('PMD', max(self.T * self.tpi, 1) * 32)
        self._wl0 = KG_NAGLOWEK + max(((self.N + 255) // 256) * 512, KPROBKA_MAX * KPROBKA_W)
        self._wl1 = self._wl0 + 12 * self.N                        # drugie okno (wokół zapisu b) do płynnej animacji
        self._pole0 = self._wl1 + 12 * self.N                      # właściwości ruchu (12 na cząsteczkę), pole kolektywne
        n_kg = self._pole0 + 4 * max(self.T, 1)
        self._alokuj('KG', n_kg * 4, np.zeros(n_kg, np.float32))
        self._fiz_pole_klucz = self._fiz_klucz = None
        self._czasy_ok = None                                        # bufor wyzerowany: czasy zapisów od nowa
        self._koop_A = None                                          # nowy układ buforów = nowe dopasowanie
        self._pomocnicze_brudne = True
        if getattr(self, '_prog_q16', None) != self.q16:
            self._kompiluj()
        self._alokuj('FR', 8 * 16, np.zeros((8, 4), np.float32))
        self._alokuj('PIB', self.T * self.images * 64)
        self._alokuj('KNOWN', self.T * 4, np.zeros(self.T, np.uint32))
        self._alokuj('HEAT', self.T * self.tpi * 4)
        self._alokuj('HIST', (R_HIST + 32) * 4)                # + R + 16..21: zakres osi Koopmana
        self._alokuj('PICK', 16)
        self._alokuj('SLOT', self.T * 4, np.full(self.T, NONE, np.uint32))
        self._alokuj('BB', 32)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._sloty.clear()
        self._wolne = list(range(self.n_slotow))
        self._znane = set()
        self._uklad = (self._generacja, self.N, self.T, self.images, self.tpi, self.n_slotow)
        self._slot_map_brudna = True
        self._klucz_kin, self._cur_ok, self._ramka_brudna = None, False, True

    def _wiaz(self):
        for i, n in enumerate(BUFORY):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, self._buf[n])

    def _chronione(self):
        ia = self._idx.get(self.a, 0)
        ib = self._idx.get(self.b, ia)
        ochrona = {self.steps[0], self.steps[ia], self.steps[ib]}
        if not self.cala_os:
            for k in range(1, max(0, min(self.n_slotow - 3, 8)) + 1):
                ochrona.add(self.steps[(ib + k) % self.T])
        return ochrona, ia

    def _slot_dla(self, krok):
        if self._wolne:
            return self._wolne.pop(0)
        ochrona, ia = self._chronione()
        kandydaci = [k for k in self._sloty if k not in ochrona]
        if not kandydaci:
            return None

        def odleglosc(k):                         # najpierw stany daleko za bieżącym, potem daleko przed
            i = self._idx[k]
            return (ia - i) + self.T if i < ia else i - ia
        stary = max(kandydaci, key=odleglosc)
        slot = self._sloty.pop(stary)
        return slot

    def _wyslij_stany(self):
        """Najpierw potrzebne teraz (pierwszy krok, para), potem do `na_klatke` kolejnych."""
        if not self._gotowe:
            return
        pilne = [s for s in dict.fromkeys((self.steps[0], self.a, self.b)) if s in self._gotowe]
        reszta = [s for s in self._gotowe if s not in pilne]
        if self.steps[0] not in self._sloty and self.steps[0] not in self._gotowe:
            reszta = []                             # statystyki potrzebują pierwszego kroku na karcie
        for krok in pilne + reszta[:self.na_klatke]:
            slot = self._slot_dla(krok)
            if slot is None:
                break
            arr = self._gotowe.pop(krok)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['S'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, slot * self.frame_bytes, self.frame_bytes, arr)
            self.wyslane_bajty += self.frame_bytes
            if self.q16:
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['QP'])
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, slot * 32, 32, self.timeline.quant(krok))
            self._sloty[krok] = slot
            self._slot_map_brudna = True
            if krok == self.a or krok == self.b:
                self._klucz_kin = None
            self._statystyki_kroku(krok, slot)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    def _statystyki_kroku(self, krok, slot):
        k = self._idx[krok]
        s0 = self._sloty.get(self.steps[0])
        if s0 is None:
            return
        prev = self._sloty.get(self.steps[k - 1]) if k > 0 else None
        dt = float(krok - self.steps[k - 1]) if k > 0 else 0.0
        self._wiaz()
        self._uniformy('stat', u_N=self.N, u_k=k, u_slot=slot, u_s0=s0, u_prev=NONE if prev is None else prev,
                       u_tpi=self.tpi, u_img=self.images, u_dt=dt)
        GL.glDispatchCompute(self.images, 1, 1)
        self._uniformy('heat', u_N=self.N, u_k=k, u_slot=slot, u_tpi=self.tpi, u_img=self.images)
        GL.glDispatchCompute(self.tpi, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        self._znane.add(krok)
        self._ramka_brudna = True
        if krok == self.steps[0]:                 # przesunięcia pozostałych kroków liczone względem pierwszego
            for inny, sl in list(self._sloty.items()):
                if inny != krok:
                    self._statystyki_kroku(inny, sl)

    def _odswiez_mape_slotow(self):
        if not self._slot_map_brudna:
            return
        m = np.full(self.T, NONE, np.uint32)
        for krok, slot in self._sloty.items():
            m[self._idx[krok]] = slot
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['SLOT'])
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, m.nbytes, m)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._slot_map_brudna = False

    def _kinematyka(self):
        sa, sb, s0 = self._sloty.get(self.a), self._sloty.get(self.b), self._sloty.get(self.steps[0])
        if sa is None or sb is None or s0 is None:
            return False
        first, count = self.first, min(self.count, self.N - self.first)
        klucz = (sa, sb, s0, self.a, self.b, self.s, self.ruch, self.kolor, first, count,
                 self.uklad, self.cecha, self.mozaika, self.bez_poz, self._koop_klucz_wag, self.koop['czesc'], self.wyg,
                 (self._fiz_klucz, self.fiz['osie']) if self.uklad == 4 else None)
        # prędkość cechy w przestrzeni fazowej: odcinek a -> b, a w ostatnim zapisie odcinek poprzedni -> a
        if self.a != self.b:
            sv0, sv1, dtv = sa, sb, float(self.b - self.a)
            kv0, kv1 = self._idx.get(self.a, 0), self._idx.get(self.b, 0)
        else:
            ia = self._idx.get(self.a, 0)
            prev = self.steps[ia - 1] if ia > 0 else None
            sv0 = self._sloty.get(prev) if prev is not None else None
            sv1, dtv = sa, float(self.a - prev) if prev is not None else 0.0
            kv0, kv1 = max(ia - 1, 0), ia
        if klucz == self._klucz_kin and self._cur_ok:
            return False
        for k in (self.a, self.b):
            self._sloty.move_to_end(k)
        # przyspieszenie chwilowe: zapis najbliższy bieżącej chwili i odstępy (kroki treningu) do sąsiednich zapisów
        kc = self._idx.get(self.a, 0) if self.s < .5 else self._idx.get(self.b, 0)
        dta0 = float(self.steps[kc] - self.steps[kc - 1]) if kc >= 1 else 0.0
        dta1 = float(self.steps[kc + 1] - self.steps[kc]) if kc + 1 < self.T else 0.0
        if self.kolor == 14:
            self._wyslij_czasy()
        if self.kolor in (12, 13, 14):                        # sąsiednie zapisy muszą być na karcie
            for k in (kc - 1, kc + 1):
                if 0 <= k < self.T and self.steps[k] in self._sloty:
                    self._sloty.move_to_end(self.steps[k])
        self._uniformy('kin', u_N=self.N, u_first=first, u_count=count, u_sa=sa, u_sb=sb, u_s0=s0,
                       u_ruch=self.ruch, u_kolor=self.kolor, u_s=float(self.s),
                       u_dt=float(self.b - self.a) if self.b != self.a else 0.0,
                       u_sv0=NONE if sv0 is None else sv0, u_sv1=NONE if sv1 is None else sv1, u_dtv=dtv,
                       u_kb=self._idx.get(self.b, 0), u_kv0=kv0, u_kv1=kv1,
                       u_kc=kc, u_dta0=dta0, u_dta1=dta1, u_czasy0=getattr(self, '_pole0', 0),
                       **self._wspolne())
        GL.glDispatchCompute((count + 255) // 256, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        self._klucz_kin, self._cur_ok, self._rys_count = klucz, True, count
        self._rys_first = first
        return True

    def _ramka(self, kin_nowa):
        if self.kadr == 2:
            if not kin_nowa and not self._ramka_brudna:
                return
            bb = np.array([NONE] * 3 + [0] * 3 + [0, 0], np.uint32)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['BB'])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, bb.nbytes, bb)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            self._uniformy('bb', u_N=self.N, u_first=self._rys_first, u_count=self._rys_count)
            GL.glDispatchCompute((self._rys_count + 255) // 256, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        elif not self._ramka_brudna:
            return
        img0 = self.first // max(self.tpi, 1)
        img1 = max(img0 + 1, -(-(self.first + self.count) // max(self.tpi, 1)))
        nat, mov = self.granice_manifest or (None, None)
        g = mov if self.ruch else nat
        xylo, xyhi = self.granice_siatki(img0, img1)
        self._uniformy('ramka', u_N=self.N, u_T=self.T, u_img=self.images, u_img0=img0, u_img1=img1,
                       u_ruch=self.ruch, u_tryb=self.kadr, u_manifest=g is not None,
                       u_mlo=tuple(g[:, 0]) if g is not None else (0.0, 0.0, 0.0),
                       u_mhi=tuple(g[:, 1]) if g is not None else (0.0, 0.0, 0.0),
                       u_normscale=float(self.normscale), u_xylo=xylo, u_xyhi=xyhi, **self._wspolne())
        GL.glDispatchCompute(1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        self._ramka_brudna = False

    def _czytaj_bufor(self, nazwa, dtype, n, offset=0):
        """Odczyt fragmentu bufora przez kopię na karcie do małego bufora RB. Bezpośredni glGetBufferSubData z
        bufora roboczego sprawia, że sterownik przenosi CAŁY bufor do pamięci systemowej (dla 1,5 GB magazynu:
        ~0,5 s, a potem kernele i atomiki czytają przez PCIe), więc bufory robocze nigdy nie są czytane wprost."""
        out = np.empty(n, dtype)
        if out.nbytes == 0:
            return out
        if self._rozm.get('RB', 0) < out.nbytes:
            self._alokuj('RB', max(out.nbytes, 1 << 16), None, GL.GL_STREAM_READ)
        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, self._buf[nazwa])
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, self._buf['RB'])
        GL.glCopyBufferSubData(GL.GL_COPY_READ_BUFFER, GL.GL_COPY_WRITE_BUFFER, int(offset), 0, out.nbytes)
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, self._buf['RB'])
        GL.glGetBufferSubData(GL.GL_COPY_READ_BUFFER, 0, out.nbytes, out.ctypes.data_as(ctypes.c_void_p))
        GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
        GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)
        return out

    def _statystyki_widoku(self):
        """Mapa i dokładna mediana normy 768D dla zadanego kroku i widoku; odczyt kilku liczb z karty."""
        if self._stat_klucz is None or self._stat_klucz == self._stat_zrobione:
            return
        krok, first, count, gen = self._stat_klucz
        slot = self._sloty.get(krok)
        if gen != self._generacja or slot is None or krok not in self._znane:
            return
        count = min(count, self.N - first)
        if count <= 0:
            self._stat_zrobione = self._stat_klucz
            return
        k = self._idx[krok]
        if count == self.N:
            mapa = self._czytaj_bufor('HEAT', np.float32, self.tpi, k * self.tpi * 4)
        else:
            if self.q16:
                q = self._czytaj_bufor('S', np.uint16, count * 4, (slot * self.N + first) * 8).reshape(-1, 4)
                mapa = (q[:, 3].astype(np.float32) / np.float32(65535)) * self.timeline.quant(krok)[3]
            else:
                mapa = self._czytaj_bufor('S', np.float32, count * 4, (slot * self.N + first) * 16).reshape(-1, 4)[:, 3]
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['HIST'])
        GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, None)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        grupy = (count + 255) // 256
        for faza in (0, 1):
            self._uniformy('hist', u_N=self.N, u_slot=slot, u_first=first, u_count=count, u_faza=faza)
            GL.glDispatchCompute(grupy, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            self._uniformy('wybor', u_N=self.N, u_faza=faza, u_klo=(count - 1) // 2, u_khi=count // 2)
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        wyn = self._czytaj_bufor('HIST', np.uint32, 8, R_HIST * 4)
        f = wyn.view(np.float32)
        self._stat_zrobione = self._stat_klucz
        self.statystyki_gotowe.emit({'klucz': self._stat_klucz, 'krok': krok, 'first': first, 'count': count,
                                     'mapa': mapa, 'mediana_dol': float(f[5]), 'mediana_gora': float(f[6]),
                                     'mediana': (float(f[5]) + float(f[6])) / 2, 'maksimum': float(f[2]),
                                     'bity': (int(wyn[5]), int(wyn[6]), int(wyn[2]))})

    def _wybierz(self, mvp):
        if self._wybor is None or not self._cur_ok:
            return
        x, y, w, h = self._wybor
        self._wybor = None
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf['PICK'])
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, 16, np.full(4, NONE, np.uint32))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        for faza in (0, 1):
            self._uniformy('pick', **self._wspolne(), u_N=self.N, u_mvp=mvp, u_klik=(x, y), u_wh=(w, h), u_first=self._rys_first,
                           u_count=self._rys_count, u_faza=faza)
            GL.glDispatchCompute((self._rys_count + 255) // 256, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        wyn = self._czytaj_bufor('PICK', np.uint32, 4)
        d2 = float(wyn[:1].view(np.float32)[0]) if wyn[0] != NONE else np.inf
        self.ostatni_wybor = (int(wyn[1]) if wyn[1] != NONE else -1, d2)
        self.wybrano.emit(int(wyn[1]) if wyn[1] != NONE and d2 <= 144.0 else -1)

    def oblicz(self, mvp=None):
        """Wszystkie kernele klatki (bez rysowania). Wołać przy aktywnym kontekście widoku."""
        self._przygotuj()
        self._wiaz()
        self._wyslij_stany()
        self._odswiez_mape_slotow()
        self._wiaz()
        self._koopman()
        self._wiaz()
        self._probka_koopmana()
        self._wyslij_pomocnicze()                      # MASK (obszary oka) przed statystykami fizyki
        self._wiaz()
        self._fizyka()
        self._wiaz()
        kin = self._kinematyka()
        if self._cur_ok:
            self._ramka(kin)
        self._statystyki_widoku()
        if mvp is not None:
            self._wybierz(mvp)

    def paint(self):
        if self.timeline is None or not self.T:
            return
        try:
            if self._prog is None:
                self._init()
            self.setupGLState()
            mvp = np.asarray(self.mvpMatrix().data(), np.float32).reshape(4, 4)
            self.oblicz(mvp)
            if self._cur_ok:
                self._rysuj(mvp)
            self._licz_fps()
            self.blad = ''
        except Exception as e:                                   # błąd GL nie może zatrzymać całej aplikacji
            self.blad = f'{type(e).__name__}: {e}'
        finally:                                                 # stan GL jak przed rysowaniem: inne elementy i QPainter
            GL.glUseProgram(0)
            GL.glLineWidth(1.0)
            GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            for i in range(len(BUFORY)):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, 0)

    def _rysuj(self, mvp):
        s0 = self._sloty.get(self.steps[0], 0)
        ia = self._idx.get(self.a, 0)
        if self.slad > 0 and self.uklad != 2:          # w przestrzeni fazowej ślad położeń nie ma sensu
            if self._rys_count <= 6000:
                n, wyb = self._rys_count, -1
            else:
                n, wyb = (1, self.wybrany) if self.wybrany is not None else (0, -1)
            if n:
                self._uniformy('slad', u_N=self.N, u_mvp=mvp, u_first=self._rys_first, u_ia=ia, u_K=self.slad,
                               u_ruch=self.ruch, u_s0=s0, u_wybrany=wyb, u_alfa=min(1.0, self.krycie * 2.5),
                               **self._wspolne())
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2 * self.slad, n)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_POINT_SPRITE)
        self._wyslij_pomocnicze()
        self._wyslij_obraz()
        self._uniformy('punkty', u_N=self.N, u_mvp=mvp, u_first=self._rys_first, u_kolor=self.kolor, u_wybrany=-1,
                       u_size=float(self.rozmiar), u_opacity=float(self.krycie), **self._wspolne())
        GL.glDrawArrays(GL.GL_POINTS, 0, self._rys_count)
        if self.wybrany is not None and self.wybrany < self.N:
            GL.glUniform1i(self._u('punkty', 'u_wybrany'), self.wybrany)
            GL.glDrawArrays(GL.GL_POINTS, 0, 1)
        self.draw_count = self._rys_count
        self._rysuj_zdjecie(mvp)

    def _rysuj_zdjecie(self, mvp):
        """Zdjęcie obrazu rozpięte na jego 600 cząsteczkach (powierzchnia, linie siatki, brzeg) - tylko widok
        jednego obrazu z siatką 20 x 30; położenia i barwy liczy shader z CUR."""
        tryb, kolor, krycie, krycie_lin, n = self.zdjecie
        if self._rys_count != 600 or self.tpi != 600 or not (tryb or self.brzeg):
            return
        if kolor != 1 and self._tekstura is None and (self.att[0] & 2):
            kolor = 1                                              # bez zdjęcia heatmapa na barwach cząsteczek
        if kolor != 1 and self._tekstura is None:
            tryb = 0
        wspolne = dict(u_N=self.N, u_mvp=mvp, u_first=self._rys_first, u_zkolor=kolor, u_kolor=self.kolor, u_rgb=0,
                       **self._wspolne())
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tekstura or 0)
        if tryb in (1, 3):
            R, C = 19 * n + 1, 29 * n + 1
            self._indeksy_siatki(R, C)
            self._uniformy('zdj', u_C=C, u_n=n, u_alfa=float(krycie), **wspolne)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._indeksy)
            GL.glDrawElements(GL.GL_TRIANGLES, self._n_ind, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        if tryb in (2, 3):
            self._uniformy('zlin', u_rodzaj=0, u_alfa=float(krycie_lin), **wspolne)
            GL.glDrawArrays(GL.GL_LINES, 0, 2 * (20 * 29 + 19 * 30))
        if self.brzeg:
            self._uniformy('zlin', u_rodzaj=1, u_alfa=1.0, **wspolne)
            GL.glLineWidth(2.5)
            GL.glDrawArrays(GL.GL_LINES, 0, 2 * (2 * 29 + 2 * 19))
            GL.glLineWidth(1.0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _licz_fps(self):
        self._klatki += 1
        now = time.perf_counter()
        if now - self._zegar >= 1.0:
            self.fps = self._klatki / (now - self._zegar)
            self._zegar, self._klatki = now, 0

    # ------------------------------------------------------------------ odczyt (testy i diagnostyka)
    def odczytaj(self, nazwa, n, dtype=np.float32, offset=0, kolumny=4):
        """n wierszy bufora `nazwa` (CUR, FR, PIB, HEAT, ...). Wołać przy aktywnym kontekście widoku."""
        out = self._czytaj_bufor(nazwa, dtype, n * kolumny, offset)
        return out.reshape(n, kolumny) if kolumny > 1 else out

    def slot(self, krok):
        return self._sloty.get(krok)

    def release(self):
        """Zwalnia bufory i programy. Wołać przy aktywnym kontekście widoku."""
        if self._buf:
            GL.glDeleteBuffers(len(self._buf), list(self._buf.values()))
        if self._prog:
            for p in self._prog.values():
                GL.glDeleteProgram(p)
        if self._tekstura is not None:
            GL.glDeleteTextures(1, [self._tekstura])
        if self._indeksy is not None:
            GL.glDeleteBuffers(1, [self._indeksy])
        self._tekstura = self._indeksy = self._siatka = None
        self._rgb_brudne = self._rgb is not None
        self._prog, self._buf, self._rozm, self._uni = None, {}, {}, {}
        self._uklad = None
        self._sloty.clear()

    def zamknij(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
