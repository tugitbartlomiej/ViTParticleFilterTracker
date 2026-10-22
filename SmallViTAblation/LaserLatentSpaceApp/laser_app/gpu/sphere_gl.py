"""Sfera reflektorów: 768 reflektorów, wiązki do 600 koloidów (albo kilku zdjęć), koloidy i ich ślady (ReflectorSphere)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import (_CZAS, _GPUElement, _KSZTALT, LIMIT_ZH, _LUT, NK, _program, przenikanie,
                                reflektory_przejscia, siatka_sfery, TOK, W)

_SSBO_S = """
layout(std430, binding = 0) readonly buffer BR { vec4 RF[]; };    // [768] stały reflektor (miejsce na sferze zewnętrznej)
layout(std430, binding = 1) readonly buffer BK { vec4 KC[]; };    // [768] kolor reflektora
layout(std430, binding = 2) readonly buffer BP { vec4 P[]; };     // [K*600] cząsteczka p_j (skala), w = narzędzie
layout(std430, binding = 3) readonly buffer BD { vec4 D[]; };     // [K*768] kierunek wiązki w sferze d_c
layout(std430, binding = 4) readonly buffer BI { float I[]; };    // [2*600*768] natężenia z_jc pary checkpointów (a, a+1)
layout(std430, binding = 5) readonly buffer BS { vec4 SF[]; };    // siatka sfery (pary punktów)
layout(std430, binding = 6) readonly buffer BT { float AT[]; };   // [600*600] podkroki: atencja bloku (średnia głów), wiersz = kto słucha
layout(std430, binding = 7) readonly buffer BC { vec4 C[]; };     // [K*600] pułapka koloidu: średnie położenie z ±okna kroków (ta sama skala)
layout(std430, binding = 8) readonly buffer BKP { vec4 KP[]; };   // [600] kolor ścieżki koloidu
layout(std430, binding = 9) readonly buffer BKR { float KR[]; };  // [K] krok uczenia checkpointu (zanik historii)
layout(std430, binding = 10) readonly buffer BOD { vec4 OD[]; };  // [K*768] obrót igły, dryf netto (≥ 2000 kroków): xyz kierunek, w °/100 kroków
layout(std430, binding = 11) readonly buffer BOJ { vec4 OJ[]; };  // [K*768] obrót igły, drganie (≥ 100 kroków)
layout(std430, binding = 12) readonly buffer BW { float WY[]; };  // wybór: [768] maska reflektorów, [600] maska cząsteczek, listy indeksów
layout(std430, binding = 13) readonly buffer BZH { uint ZH[]; };  // [K*T*768] całe natężenia z_jc historii, float16 parami (unpackHalf2x16)
bool wyb_r(uint c) { return WY[c] > 0.5; }
uniform int u_T;                          // cząsteczek na sferze: 600 × liczba zdjęć zestawu (podkroki: 600)
bool wyb_t(uint j) { return WY[768u + j] > 0.5; }
uint lista_r(uint k) { return uint(WY[768u + uint(u_T) + k]); }   // k-ty wybrany reflektor
uint lista_t(uint k) { return uint(WY[1536u + uint(u_T) + k]); }  // k-ta wybrana cząsteczka
"""
VS_S = "#version 430\n" + _SSBO_S + _CZAS + _LUT + """
uniform mat4 u_mvp;
uniform int u_tryb, u_wiazki, u_seg, u_tok, u_refl, u_blokD, u_cel, u_filtr, u_kol_sciezki;
uniform float u_zanik, u_dnorm, u_skala;   // u_skala: mnożnik rozmiaru chmury koloidów i pułapek w sferze            // u_dnorm: odpływ od adresu dający pełny kolor (99. percentyl)
uniform float u_R, u_gain, u_prog, u_rozm, u_pu;
uniform float u_neg;                      // wzmocnienie ujemnej jasności (|z| ≤ 0,17); 1 = prawdziwa proporcja
uniform float u_zgas;                     // płynne przejście podkroków: 0 pełna jasność wiązek / linii uwagi, 1 zgaszone
uniform float u_olo, u_ohi, u_ola, u_oigla;   // strzałki obrotu: skala log10 prędkości (dół, góra), długość maks., długość igły
uniform vec4 u_kol;
out vec4 v_k; out float v_t;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
vec3 pj(uint j) { uint a, b; float f; para(a, b, f); return u_skala * mix(P[a * uint(u_T) + j].xyz, P[b * uint(u_T) + j].xyz, f); }
vec3 dc(uint c) {                         // podkroki: kierunki bloku u_blokD (bez interpolacji między blokami)
    if (u_blokD >= 0) return normalize(D[uint(u_blokD) * 768u + c].xyz);
    uint a, b; float f; para(a, b, f); return normalize(mix(D[a * 768u + c].xyz, D[b * 768u + c].xyz, f));
}
uniform int u_zcale, u_ka, u_kb;          // u_zcale = 1: natężenia z ZH (cała historia na GPU), para checkpointów (u_ka, u_kb)
float zh(uint k, uint j, uint c) { uint i = (k * uint(u_T) + j) * 768u + c; vec2 h = unpackHalf2x16(ZH[i >> 1u]); return (i & 1u) == 0u ? h.x : h.y; }
float nat(uint j, uint c) {
    if (u_zcale == 1) return 0.5 * mix(zh(uint(u_ka), j, c), zh(uint(u_kb), j, c), u_pu);   // z / 2 jak film 12
    return mix(I[j * 768u + c], I[uint(u_T) * 768u + j * 768u + c], u_pu);
}
vec3 w_chwili(uint j, float tt) {          // koloid (u_cel 0) albo pułapka (1) w ułamkowej chwili tt
    float t2 = clamp(tt, 0.0, float(u_K - 1)); uint a = uint(floor(t2)); uint b = min(a + 1u, uint(u_K - 1)); float f = t2 - float(a);
    return u_skala * (u_cel == 0 ? mix(P[a * uint(u_T) + j].xyz, P[b * uint(u_T) + j].xyz, f) : mix(C[a * uint(u_T) + j].xyz, C[b * uint(u_T) + j].xyz, f));
}
bool odrzuc(uint j) { return (u_filtr == 1 && P[j].w < 0.5) || (u_filtr == 2 && !wyb_t(j)); }
vec3 kolor_drogi(uint j, vec3 p, float tt) {  // u_kol_sciezki: 0 jednolity, 1 kolor cząsteczki, 2 odpływ od adresu (plasma)
    if (u_kol_sciezki == 1) return KP[j].rgb;
    if (u_kol_sciezki == 2) return lut(0, length(p - w_chwili(j, 0.0)) / max(u_dnorm * u_skala, 1e-9));
    return u_kol.rgb;
}
void main() {
    int v = gl_VertexID;
    v_k = u_kol; v_t = 0.0;
    if (u_tryb == 0) {                        // wiązki: odcinek reflektor → punkt załamania (u_seg 0) albo punkt → cząsteczka (1)
        int id = gl_InstanceID; uint j, c;
        if (u_wiazki == 2) { j = lista_t(uint(id / 768)); c = uint(id % 768); }        // do wybranych cząsteczek
        else if (u_wiazki == 3) { c = lista_r(uint(id / u_T)); j = uint(id % u_T); }   // z wybranych reflektorów
        else { c = uint(id / u_T); j = uint(id % u_T); }
        float zj = nat(j, c);                  // ze znakiem: ujemna jasność = ten sam laser słabo ciągnie (−v_c)
        float x = abs(zj) * u_gain * (zj < 0.0 ? u_neg : 1.0);
        if (x < u_prog) { gl_Position = POZA; return; }
        vec3 p = pj(j), d = dc(c);
        float bb = dot(p, d); float s = bb + sqrt(max(bb * bb - dot(p, p) + u_R * u_R, 0.0));
        vec3 e = p - s * d;
        vec3 q = (u_seg == 0) ? (v == 0 ? RF[c].xyz : e) : (v == 0 ? e : p);
        v_t = float(v);
        float r = u_seg == 0 ? (0.6 + 0.4 * v_t) : (0.25 + 0.75 * v_t * v_t);
        v_k = vec4((zj < 0.0 ? vec3(0.30, 0.85, 1.0) : 0.25 + 0.75 * KC[c].rgb) * min(x, 4.0) * r * (1.0 - u_zgas), 1.0);
        gl_Position = u_mvp * vec4(q, 1.0);
    } else if (u_tryb == 1) {                 // cząsteczki (0 wszystkie, 1 narzędzie, 2 wybrana)
        uint j = uint(v);
        if (u_wiazki == 1 && P[j].w < 0.5) { gl_Position = POZA; return; }
        if (u_wiazki == 2 && !wyb_t(j)) { gl_Position = POZA; return; }
        gl_Position = u_mvp * vec4(pj(j), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 2) {                 // reflektory (0 wszystkie, 2 wybrany)
        uint c = uint(v);
        if (u_wiazki == 2 && !wyb_r(c)) { gl_Position = POZA; return; }
        v_k = vec4(0.25 + 0.75 * KC[c].rgb, u_kol.a);
        gl_Position = u_mvp * vec4(RF[c].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 3) {                 // siatka szklanej sfery
        gl_Position = u_mvp * vec4(SF[v].xyz, 1.0);
    } else if (u_tryb == 4) {                 // kierunek wybranych reflektorów (instancja = reflektor z listy): średnica sfery wzdłuż d_c
        uint c = lista_r(uint(gl_InstanceID)); vec3 d = dc(c);
        gl_Position = u_mvp * vec4((v == 0 ? -u_R : u_R) * d, 1.0);
        v_k = vec4(0.25 + 0.75 * KC[c].rgb, 0.9);
    } else if (u_tryb == 5) {                 // podkrok uwagi: siły między cząsteczkami jako linie i – j (jasność = atencja)
        uint i = uint(gl_InstanceID / u_T), j = uint(gl_InstanceID % u_T);
        if (u_wiazki == 2) { i = lista_t(uint(gl_InstanceID / u_T)); j = uint(gl_InstanceID % u_T); }
        float a = AT[i * uint(u_T) + j] * u_gain;
        if (i == j || a < u_prog) { gl_Position = POZA; return; }
        v_k = vec4(vec3(0.35, 0.8, 1.0) * min(a, 4.0) * (1.0 - u_zgas), 1.0);
        gl_Position = u_mvp * vec4(pj(v == 0 ? i : j), 1.0);
    } else if (u_tryb == 8) {                 // pułapki koloidów (poświata studni), u_filtr jak wektory
        uint j = uint(v); if (odrzuc(j)) { gl_Position = POZA; return; }
        gl_Position = u_mvp * vec4(w_chwili(j, u_t), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 9) {                 // wektory: u_seg 1 = z poprzedniego checkpointu, 2 = od kroku 0 (adres startu); skala u_gain
        uint j = uint(gl_InstanceID); if (odrzuc(j)) { gl_Position = POZA; return; }
        vec3 p1 = w_chwili(j, u_t), p0 = w_chwili(j, u_seg == 2 ? 0.0 : u_t - 1.0);
        v_k = vec4(kolor_drogi(j, p1, u_t), u_kol.a * (v == 0 ? 0.2 : 1.0));
        gl_Position = u_mvp * vec4(v == 0 ? p0 : p0 + (p1 - p0) * u_gain, 1.0);
    } else if (u_tryb == 10) {                // cała historia: droga od checkpointu 0 do teraz (instancja = cząsteczka, wierzchołek = checkpoint)
        uint j = uint(gl_InstanceID); if (odrzuc(j)) { gl_Position = POZA; return; }
        float tt = min(float(v), u_t);
        float al;
        if (u_zanik > 0.0) {                   // zanik: odcinek sprzed u_zanik kroków prawie znika (exp(−3·wiek/u_zanik))
            float k1 = clamp(u_t, 0.0, float(u_K - 1)); uint a1 = uint(floor(k1));
            float teraz = mix(KR[a1], KR[min(a1 + 1u, uint(u_K - 1))], k1 - float(a1));
            uint iv = uint(min(float(v), float(u_K - 1)));
            float wiek = max(teraz - KR[iv], 0.0);
            al = exp(-3.0 * wiek / u_zanik);
            if (al < 0.01) { gl_Position = POZA; return; }
        } else {
            al = 0.15 + 0.85 * float(v) / max(u_t, 1.0);
        }
        vec3 pv = w_chwili(j, tt);
        v_k = vec4(kolor_drogi(j, pv, tt), u_kol.a * al);
        gl_Position = u_mvp * vec4(pv, 1.0);
    } else if (u_tryb == 11) {                // igła reflektora (u_seg 0) i strzałka obrotu: 1 dryf netto (z grotem), 2 drganie
        uint c = uint(gl_InstanceID);
        if (u_wiazki == 2 && !wyb_r(c)) { gl_Position = POZA; return; }
        vec3 r = RF[c].xyz, d = dc(c), q = r + u_oigla * d;
        if (u_seg == 0) {
            v_k = vec4(0.25 + 0.75 * KC[c].rgb, u_kol.a);
            gl_Position = u_mvp * vec4(v == 0 ? r : q, 1.0); return;
        }
        uint a, b; float f; para(a, b, f);
        vec4 oa = u_seg == 1 ? OD[a * 768u + c] : OJ[a * 768u + c], ob = u_seg == 1 ? OD[b * 768u + c] : OJ[b * 768u + c];
        vec3 T = mix(oa.xyz, ob.xyz, f); T -= dot(T, d) * d;
        if (length(T) < 1e-6) { gl_Position = POZA; return; }
        T = normalize(T);
        float w = exp(mix(log(max(oa.w, 1e-6)), log(max(ob.w, 1e-6)), f));          // prędkość kątowa, interpolacja w skali log
        float u = clamp((log(w) / log(10.0) - u_olo) / max(u_ohi - u_olo, 1e-6), 0.0, u_seg == 2 ? 1.4 : 1.0);
        vec3 e = q + (0.6 + u_ola * u) * T;
        vec3 n = normalize(cross(T, d)); float h = (0.6 + 1.0 * u) * u_ola / 8.0;
        vec3 p = v == 0 ? q : (v == 3 ? e - h * T + 0.55 * h * n : (v == 5 ? e - h * T - 0.55 * h * n : e));
        v_k = vec4(u_seg == 1 ? lut(0, 0.25 + 0.75 * u) : vec3(0.92), u_kol.a * (0.35 + 0.55 * min(u, 1.0)));
        gl_Position = u_mvp * vec4(p, 1.0);
    } else if (u_tryb == 6 || u_tryb == 7) {  // wypadkowy wektor podkroku: stan a → a + (stan a+1 − stan a) · skala; 7 = grot
        uint j = uint(u_tryb == 6 ? gl_InstanceID : v);
        if (u_wiazki == 2 && !wyb_t(j)) { gl_Position = POZA; return; }
        uint a, b; float f; para(a, b, f);
        vec3 p0 = u_skala * P[a * uint(u_T) + j].xyz, p1 = u_skala * P[b * uint(u_T) + j].xyz;
        vec3 q = (u_tryb == 6 && v == 0) ? p0 : p0 + (p1 - p0) * u_gain;
        v_k = vec4(u_kol.rgb, u_kol.a * ((u_tryb == 6 && v == 0) ? 0.12 : 1.0));
        gl_Position = u_mvp * vec4(q, 1.0); gl_PointSize = u_rozm;
    }
}
"""
FS_S = "#version 430\n" + _KSZTALT + """
uniform int u_punkty;
in vec4 v_k; in float v_t; out vec4 o;
void main() { o = u_punkty == 1 ? ksztalt(v_k) : v_k; }
"""


VS_TON = """#version 430
out vec2 v_uv;
void main() { vec2 p = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0; v_uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }
"""
FS_TON = """#version 430
uniform sampler2D u_hdr; uniform float u_ekspozycja;
in vec2 v_uv; out vec4 o;
void main() { vec3 c = texture(u_hdr, v_uv).rgb * u_ekspozycja; c = 1.0 - exp(-c); o = vec4(pow(c, vec3(0.85)), 1.0); }
"""


class ReflectorSphere(_GPUElement):
    BUFORY = ('RF', 'KC', 'P', 'D', 'I', 'SF', 'AT', 'C', 'KP', 'KR', 'OD', 'OJ', 'WY', 'ZH')

    def __init__(self, R=34.0):
        super().__init__()
        self.R = float(R)
        self.pod = None                          # podkroki: słownik history.podkroki albo None (tryb osi uczenia)
        self.blok = -1
        self.opcje = dict(wiazki=1, gain=0.05, prog=0.02, tok=0, refl=0, rozm=5.0, sfera=0.10, reflektory=True, narz=True,
                          widok=0, wek_kol=0, wek_pul=0, filtr=1, skala_wek=1.0)
        self.para = (-1, -1)
        self.u_pary = 0.0
        self.T = TOK                              # cząsteczek na sferze (zestaw zdjęć: 600 × n)
        self._sf = siatka_sfery(self.R)
        self.n_sf = len(self._sf)
        self.ustaw_bufor('SF', self._sf)
        self.ustaw_wybor([0], [0])

    def ustaw_wybor(self, refl, tok):
        """Wybrane reflektory i cząsteczki (listy indeksów, pierwszy = główny): maski i listy dla wiązek, punktów i wektorów."""
        refl = [int(c) for c in refl if 0 <= c < NK] or [0]; T = self.T; tok = [int(j) for j in tok if 0 <= j < T] or [0]
        WY = np.zeros(2 * NK + 2 * T, np.float32)
        WY[refl] = 1.0; WY[NK + np.asarray(tok)] = 1.0                    # maski: [768 reflektorów][T cząsteczek]
        WY[NK + T:NK + T + len(refl)] = refl; WY[2 * NK + T:2 * NK + T + len(tok)] = tok   # listy indeksów
        self.n_refl, self.n_tok = len(refl), len(tok)
        self.ustaw_bufor('WY', WY)

    def ustaw_historie(self, h, skala_stala=False):
        self.pod = None; self.przejscie = False; self.blok = -1
        K = h.K
        RF = np.zeros((NK, 4), np.float32); RF[:, :3] = h.rfix
        KC = np.ones((NK, 4), np.float32); KC[:, :3] = h.kol
        self.ustaw_bufor('RF', RF); self.ustaw_bufor('KC', KC)
        self.ustaw_skale(h, skala_stala)
        Dd = np.zeros((K, NK, 4), np.float32); Dd[..., :3] = h.dirs
        self.ustaw_bufor('D', Dd.reshape(-1, 4))
        self.obr_skala = None                                       # strzałki obrotu igieł (film 23): skala log10 z dryfu netto
        if getattr(h, 'obr_dryf', None) is not None:
            self.ustaw_bufor('OD', np.ascontiguousarray(h.obr_dryf.reshape(-1, 4), np.float32))
            self.ustaw_bufor('OJ', np.ascontiguousarray(h.obr_drg.reshape(-1, 4), np.float32))
            w = h.obr_dryf[..., 3]
            self.obr_skala = (float(np.log10(max(np.percentile(w, 2), 1e-6))), float(np.log10(max(np.percentile(w, 99.5), 1e-6))))
        self.K = K
        self.para = (-1, -1); self.zh_para = (0, min(1, K - 1))
        nb = int(np.prod(h.Z.shape)) * 2
        if nb > LIMIT_ZH:                       # zestaw wielu zdjęć: całość nie mieści się rozsądnie na karcie, zostaje para checkpointów
            st = getattr(self, '_duze', {}).pop('ZH', None)
            if st is not None:
                st['stop'] = True
            self._zh_zrodlo = None
        elif getattr(self, '_zh_zrodlo', None) is not h.Z:      # to samo Z (np. zmiana aktywnego zdjęcia zestawu): bez ponownej wysyłki
            self.wyslij_porcjami('ZH', h.Z, max(1, 16 * TOK // int(h.Z.shape[1])))   # całe natężenia na kartę (float16), w tle porcjami
            self._zh_zrodlo = h.Z

    def ustaw_skale(self, h, stala):
        r = (h.r99[-1] * np.ones(h.K) if stala else h.r99).astype(np.float32)
        self.T = T = int(h.pos.shape[1])                              # 600 × zdjęcia zestawu
        P = np.zeros((h.K, T, 4), np.float32)
        P[..., :3] = h.pos / np.maximum(r, 1e-9)[:, None, None] * 0.55 * self.R
        P[..., 3] = h.narz
        self.ustaw_bufor('P', P.reshape(-1, 4))
        self.dnorm_k = float(np.percentile(np.linalg.norm(P[..., :3] - P[:1, :, :3], axis=2), 99))
        # pułapka = średnie położenie koloidu z ±okna kroków (jak centra studni w filmie 2), w tych samych współrzędnych sfery
        kr = np.asarray(h.kroki, np.int64); okno = int(getattr(h, 'okno', 300))
        cs = np.cumsum(P[..., :3].astype(np.float64), 0)
        lo = np.searchsorted(kr, kr - okno, 'left'); hi = np.searchsorted(kr, kr + okno, 'right')
        C = P.copy()
        for t in range(h.K):
            C[t, :, :3] = (cs[hi[t] - 1] - (cs[lo[t] - 1] if lo[t] > 0 else 0)) / (hi[t] - lo[t])
        self._C_surowe = C
        self.ustaw_bufor('KR', kr.astype(np.float32))
        self._wyslij_pulapki()

    def ustaw_wygladzanie(self, tryb, okno, alfa):
        """Wygładzanie pułapek wzdłuż checkpointów (podkroków): 0 brak, 1 średnia ważona krocząca (wagi 1..okno, tylko przeszłość),
        2 EMA (alfa na checkpoint), 3 najpierw średnia krocząca, potem EMA. Zmienia pułapki we wszystkich widokach sfery."""
        self._wyg = (int(tryb), int(okno), float(alfa))
        self._wyslij_pulapki()

    def _wyslij_pulapki(self):
        C = getattr(self, '_C_surowe', None)
        if C is None:
            return
        tryb, okno, alfa = getattr(self, '_wyg', (0, 10, 0.2))
        X = C[..., :3].astype(np.float64)
        if tryb in (1, 3) and okno > 1:                              # WMA: Σ w_k x_{t−k}, w = okno, okno−1, …, 1
            w = np.arange(okno, 0, -1, dtype=np.float64)
            Y = np.zeros_like(X); S = np.zeros(len(X))
            for k, wk in enumerate(w):
                Y[k:] += wk * X[:len(X) - k]; S[k:] += wk
            X = Y / S[:, None, None]
        if tryb in (2, 3):                                           # EMA: y_t = α x_t + (1 − α) y_{t−1}
            Y = X.copy()
            for t in range(1, len(X)):
                Y[t] = alfa * X[t] + (1 - alfa) * Y[t - 1]
            X = Y
        W = C.copy(); W[..., :3] = X
        self.ustaw_bufor('C', np.ascontiguousarray(W.reshape(-1, 4), np.float32))
        self.dnorm_p = float(np.percentile(np.linalg.norm(X - X[:1], axis=2), 99))

    @staticmethod
    def _tryb_koloru(o):
        k = int(o.get('kol_sciezki', 0))
        return 0 if k == 0 else 2 if k == 4 else 1

    def ustaw_kolory_sciezek(self, K):
        self.ustaw_bufor('KP', np.ascontiguousarray(K, np.float32))

    def ustaw_natezenia(self, h, t):
        """Natężenia z_jc dla pary checkpointów wokół t (wysyłane tylko przy zmianie pary)."""
        a = int(np.clip(np.floor(t), 0, h.K - 1)); b = min(a + 1, h.K - 1)
        self.u_pary = float(t - a)
        self.zh_para = (a, b)
        if self.duzy_gotowy('ZH'):                                              # cała historia już na karcie: tylko numery checkpointów
            self.update(); return
        if (a, b) != self.para:
            I = np.empty((2, self.T, NK), np.float32)
            I[0] = np.asarray(h.Z[a], np.float32) / 2.0                        # z / 2 jak film 12, ze znakiem: ujemna jasność
            I[1] = np.asarray(h.Z[b], np.float32) / 2.0                        # (GELU ≥ −0,17) rysowana na błękitno w shaderze
            self.ustaw_bufor('I', I.reshape(-1))
            self.para = (a, b)
        self.update()

    def pozycje_podkrokow(self, d):
        """13 stanów (układ '_n') w skali sfery: JEDNA skala na checkpoint (99. percentyl po wszystkich stanach do 0,55 R),
        jak w ustaw_podkroki; osobna skala każdego stanu wyglądałaby jak ruch bez sił."""
        pos = d['pos_n']
        r = float(np.percentile(np.linalg.norm(pos, axis=2), 99))
        return (pos / max(r, 1e-9) * 0.55 * self.R).astype(np.float32)

    def ustaw_przejscie(self, Pa, Pb, narz):
        """Płynny powrót do wejścia sieci: 2 stany (wyjście poprzedniego checkpointu → wejście następnego), czas 0..1, bez wiązek."""
        self.T = TOK; P = np.zeros((2, TOK, 4), np.float32); P[0, :, :3] = Pa; P[1, :, :3] = Pb; P[..., 3] = narz
        self.ustaw_bufor('P', P.reshape(-1, 4)); self._C_surowe = P; self._wyslij_pulapki()
        self.pod = None; self.przejscie = True; self.K = 2; self.t = 0.0
        self.update()

    def ustaw_podkroki(self, d, narz):
        """Przejście przez sieć jednego checkpointu: 13 stanów strumienia rezydualnego (układ znormowany '_n') jako czas cząsteczek,
        w podkroku MLP bloku b reflektory i wiązki tego bloku, w podkroku uwagi linie atencji między cząsteczkami."""
        pos = d['pos_n']
        r = float(np.percentile(np.linalg.norm(pos, axis=2), 99))                  # jedna skala na checkpoint: ruch tylko od sił
        self.T = TOK; P = np.zeros((13, TOK, 4), np.float32)
        P[..., :3] = pos / max(r, 1e-9) * 0.55 * self.R
        P[..., 3] = narz
        Dd = np.zeros((6, NK, 4), np.float32); Dd[..., :3] = d['dirs_n']
        self.ustaw_bufor('P', P.reshape(-1, 4)); self.ustaw_bufor('D', Dd.reshape(-1, 4))
        self.dnorm_k = float(np.percentile(np.linalg.norm(P[..., :3] - P[:1, :, :3], axis=2), 99))
        self._C_surowe = P                                          # w podkrokach pułapka = stan (brak okna czasu)
        self.ustaw_bufor('KR', np.arange(13, dtype=np.float32))
        self._wyslij_pulapki()
        self.pod = d; self.przejscie = False; self.blok = -1; self.K = 13; self.u_pary = 0.0; self.para = (-1, -1)
        self.blok_a = -1; self._rf_klucz = None; self.m_pod = 0.0
        self.ustaw_podkrok(self.t)

    def ustaw_podkrok(self, s):
        """s = ułamkowy podkrok 0..12; przy zmianie bloku wysyła natężenia (blok MLP) i atencję (blok uwagi), reflektory w chwili s.
        opcja 'plynnie' (szerokość przenikania w podkrokach, 0 = osobne kroki): wiązki MLP i linie uwagi przenikają się na granicy
        podkroków (`przenikanie`). Reflektory stoją: zbiór miejsc i kolorów jest ten sam w każdym bloku (`reflektory_przejscia`)."""
        self.t = float(np.clip(s, 0, 12))
        f = float(self.opcje.get('plynnie', 0.0))
        self.m_pod, bm, ba = przenikanie(self.t, f)
        if self.pod is not None:
            if bm != self.blok:
                z = np.asarray(self.pod['z'][bm], np.float32) / 2.0                 # ze znakiem: ujemna jasność na błękitno
                self.ustaw_bufor('I', np.concatenate([z.reshape(-1), z.reshape(-1)]).astype(np.float32))
                self.blok = bm
            if ba != getattr(self, 'blok_a', -1):
                self.ustaw_bufor('AT', self.pod['ah'][ba].mean(0).astype(np.float32).reshape(-1))
                self.blok_a = ba
            kier, kol, kl = reflektory_przejscia(self.pod, self.t, f)
            if kl != getattr(self, '_rf_klucz', None):
                RF = np.zeros((NK, 4), np.float32); RF[:, :3] = 50.0 * kier
                KC = np.ones((NK, 4), np.float32); KC[:, :3] = kol
                self.ustaw_bufor('RF', RF); self.ustaw_bufor('KC', KC)
                self._rf_klucz = kl
        self.update()

    def _programy(self):
        return dict(s=_program(VS_S, FS_S), ton=_program(VS_TON, FS_TON))

    def _hdr(self, w, h):
        """Bufor ramki RGBA16F na światło wiązek (sumowane bez nasycenia, potem tonowane jak w filmie 12)."""
        if getattr(self, '_fbo_rozm', None) != (w, h):
            if getattr(self, '_fbo', None) is None:
                self._fbo = int(GL.glGenFramebuffers(1)); self._fbo_tex = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            stary = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._fbo_tex, 0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(stary))
            self._fbo_rozm = (w, h)

    def _rysuj(self, mvp):
        o = self.opcje
        GL.glUseProgram(self._prog['s'])
        pod = self.pod is not None
        wm = float(getattr(self, 'm_pod', 0.0)) if pod else 1.0     # waga wiązek MLP (podkroki: 1 MLP, 0 uwaga, pomiędzy = przenikanie)
        uwaga = pod and wm < 0.5                     # podkrok uwagi: zamiast wiązek laserów linie sił między cząsteczkami
        self._ui('s', u_mvp=mvp, u_t=self.t, u_K=self.K, u_R=self.R, u_gain=float(o['gain']), u_prog=float(o['prog']),
                 u_neg=float(o.get('neg', 1.0)),             # 1 = prawdziwa proporcja do dodatnich (zgodnie z ViT)
                 u_tok=int(o['tok']), u_refl=int(o['refl']), u_pu=0.0 if pod else self.u_pary, u_blokD=self.blok if pod else -1,
                 u_skala=float(o.get('skala_chmury', 1.0)), u_T=int(self.T), u_zcale=int(not pod and self.duzy_gotowy('ZH')),
                 u_ka=int(getattr(self, 'zh_para', (0, 0))[0]), u_kb=int(getattr(self, 'zh_para', (0, 0))[1]))
        if o['sfera'] > 0:
            self._ui('s', u_tryb=3, u_kol=(0.45, 0.6, 0.75, float(o['sfera'])), u_punkty=0)
            self._addytywnie(False); GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, self.n_sf)
        tryb_w = 0 if getattr(self, 'przejscie', False) else int(o['wiazki'])
        tryb_a = int(o.get('sily', 1)) if pod and wm < 0.999 else 0
        if pod and wm < 1e-3:
            tryb_w = 0
        if tryb_w or tryb_a:
            n = self.T * 768 if tryb_w == 1 else 768 * self.n_tok if tryb_w == 2 else self.T * self.n_refl
            vp = [int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT)]
            stary = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
            self._hdr(vp[2], vp[3])
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glViewport(0, 0, vp[2], vp[3])
            GL.glClearColor(0, 0, 0, 0); GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            self._addytywnie(True); GL.glLineWidth(float(o.get('grub_w', 1.0)))
            if tryb_w:
                for seg in (0, 1):
                    self._ui('s', u_tryb=0, u_wiazki=tryb_w, u_seg=seg, u_punkty=0, u_zgas=1.0 - wm)
                    GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, n)
            if tryb_a:                                 # 1: kogo słucha wybrana cząsteczka, 2: wszystkie pary
                ga = float(o.get('gain_a', 1.0)) * 12.0 * (20.0 if tryb_a == 1 else 1.0)
                self._ui('s', u_tryb=5, u_wiazki=2 if tryb_a == 1 else 1, u_gain=ga, u_prog=float(o.get('prog_a', 0.02)), u_punkty=0,
                         u_zgas=wm)
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, self.T * self.n_tok if tryb_a == 1 else self.T * self.T)
            self._ui('s', u_zgas=0.0)
            GL.glLineWidth(1.0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, stary)
            GL.glViewport(*vp)
            GL.glUseProgram(self._prog['ton'])
            GL.glActiveTexture(GL.GL_TEXTURE2); GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
            self._ui('ton', u_hdr=2, u_ekspozycja=float(o.get('ekspozycja', 1.0)))
            self._addytywnie(True)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0); GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glUseProgram(self._prog['s'])
        if tryb_w == 3 and o.get('kierunek', True):
            self._ui('s', u_tryb=4, u_punkty=0); self._addytywnie(False); GL.glLineWidth(2.0)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, self.n_refl)
        if o['reflektory']:
            self._ui('s', u_tryb=2, u_wiazki=0, u_rozm=6.0, u_ksztalt=4, u_kol=(1, 1, 1, 0.25 + 0.6 * wm), u_punkty=1)
            self._addytywnie(False); GL.glDrawArrays(GL.GL_POINTS, 0, NK)
            if tryb_w == 3:
                self._ui('s', u_wiazki=2, u_rozm=16.0); GL.glDrawArrays(GL.GL_POINTS, 0, NK)
        obr = int(o.get('obrot', 0))                                 # strzałki obrotu igieł: 0 brak, 1 dryf, 2 drganie, 3 oba
        if obr and not pod and not getattr(self, 'przejscie', False) and getattr(self, 'obr_skala', None):
            dl = float(o.get('dl_obr', 1.0))
            self._ui('s', u_tryb=11, u_olo=self.obr_skala[0], u_ohi=self.obr_skala[1], u_ola=8.0 * dl, u_oigla=5.0 * dl, u_lut=1, u_punkty=0)
            self._addytywnie(False)
            for wyb, grub in ((0, 1.0), (2, 2.4)):                  # wszystkie reflektory, potem wybrany (grubo)
                if wyb == 2 and not o.get('obr_wybrany', True):
                    continue
                self._ui('s', u_wiazki=wyb, u_seg=0, u_kol=(1, 1, 1, 0.55 if wyb == 0 else 1.0)); GL.glLineWidth(1.4 * grub)
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, NK)
                if obr in (2, 3):
                    self._ui('s', u_seg=2, u_kol=(1, 1, 1, 0.8 if wyb == 0 else 1.0)); GL.glLineWidth(1.2 * grub)
                    GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, NK)
                if obr in (1, 3):
                    self._ui('s', u_seg=1, u_kol=(1, 1, 1, 1.0)); GL.glLineWidth(2.2 * grub)
                    GL.glDrawArraysInstanced(GL.GL_LINES, 0, 6, NK)
            GL.glLineWidth(1.0)
        widok = int(o.get('widok', 0))                               # 0 koloidy, 1 pułapki, 2 oba
        filtr = int(o.get('filtr', 1)); sk = float(o.get('skala_wek', 1.0))
        KOL_K, KOL_P = (1.0, 0.82, 0.4), (0.3, 1.0, 0.5)
        for cel, wek in ((1, int(o.get('wek_pul', 0))), (0, int(o.get('wek_kol', 0)))):
            if wek == 3:                                                  # cała historia: droga od checkpointu 0 do teraz
                self._ui('s', u_tryb=10, u_cel=cel, u_filtr=filtr, u_kol=(*(KOL_P if cel else KOL_K), 0.8), u_punkty=0,
                         u_kol_sciezki=self._tryb_koloru(o), u_zanik=float(o.get('zanik', 0.0)), u_lut=1,
                         u_dnorm=getattr(self, 'dnorm_p' if cel else 'dnorm_k', 1.0))
                self._addytywnie(False); GL.glLineWidth(1.4 if filtr else 1.0)
                GL.glDrawArraysInstanced(GL.GL_LINE_STRIP, 0, int(np.floor(self.t)) + 2, self.T)
        if widok in (1, 2):                                               # pułapki: zielona poświata studni
            self._ui('s', u_tryb=8, u_cel=1, u_filtr=0, u_rozm=float(o.get('rozm_pul', 13.0)), u_ksztalt=0,
                     u_kol=(0, 0, 0, float(o.get('jas_pul', 0.55))), u_punkty=1)
            self._addytywnie(True); GL.glDrawArrays(GL.GL_POINTS, 0, self.T)
        if widok in (0, 2):
            self._ui('s', u_tryb=1, u_wiazki=0, u_rozm=float(o['rozm']), u_ksztalt=3, u_kol=(1.0, 1.0, 1.0, 0.9), u_punkty=1)
            self._addytywnie(True); GL.glDrawArrays(GL.GL_POINTS, 0, self.T)
        self._addytywnie(False)
        for cel, wek in ((1, int(o.get('wek_pul', 0))), (0, int(o.get('wek_kol', 0)))):
            if wek in (1, 2):                                             # z kroku na krok albo od kroku 0
                self._ui('s', u_tryb=9, u_cel=cel, u_filtr=filtr, u_seg=wek, u_gain=sk, u_kol=(*(KOL_P if cel else KOL_K), 1.0), u_punkty=0,
                         u_kol_sciezki=self._tryb_koloru(o), u_lut=1, u_dnorm=getattr(self, 'dnorm_p' if cel else 'dnorm_k', 1.0))
                GL.glLineWidth(2.5 if filtr == 2 else 1.6)
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, self.T)
        GL.glLineWidth(1.0)
        if o['narz']:
            if int(o.get('widok', 0)) == 1:                              # same pułapki: pierścienie na pułapkach narzędzia
                self._ui('s', u_tryb=8, u_cel=1, u_filtr=1, u_rozm=float(o['rozm']) * 3.0, u_ksztalt=1, u_kol=(0.29, 0.94, 1.0, 1.0))
            else:
                self._ui('s', u_tryb=1, u_wiazki=1, u_rozm=float(o['rozm']) * 3.0, u_ksztalt=1, u_kol=(0.29, 0.94, 1.0, 1.0))
            GL.glDrawArrays(GL.GL_POINTS, 0, self.T)
        if tryb_w == 2:
            self._ui('s', u_tryb=1, u_wiazki=2, u_rozm=float(o['rozm']) * 4.0, u_ksztalt=1, u_kol=(1.0, 0.85, 0.2, 1.0))
            GL.glDrawArrays(GL.GL_POINTS, 0, self.T)
        if pod and self.t < 12.0 and o.get('wypadkowe', True):   # wypadkowe wektory podkroku: uwaga niebieskie, MLP pomarańczowe
            kol = (0.35, 0.8, 1.0) if uwaga else (1.0, 0.62, 0.25)
            sk = float(o.get('skala_wyp', 1.0))
            self._addytywnie(False)
            self._ui('s', u_tryb=6, u_wiazki=0, u_gain=sk, u_kol=(*kol, 0.55), u_punkty=0); GL.glLineWidth(1.2)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, self.T)
            self._ui('s', u_wiazki=2, u_kol=(*kol, 1.0)); GL.glLineWidth(3.5)
            GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, self.T)
            self._ui('s', u_tryb=7, u_wiazki=2, u_rozm=11.0, u_ksztalt=3, u_kol=(*kol, 1.0), u_punkty=1)
            GL.glDrawArrays(GL.GL_POINTS, 0, self.T)
            GL.glLineWidth(1.0)
