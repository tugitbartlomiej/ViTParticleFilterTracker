"""Okienko 11: pułapka optyczna wokół koloidu – mgła i strzałki pola, wiązki zbiegające się w cząsteczce, światło ogniska, obrys 3D (liczony w tle), reflektory (FocalField)."""
import numpy as np
from OpenGL import GL

from laser_app.gpu.base import _GPUElement, _program, siatka_sfery, W

_SSBO_O = """
layout(std430, binding = 0) readonly buffer BG { vec4 OG[]; };   // [2·2·N] para checkpointów (a, b): N × (rgb, dominacja), N × (pole f xyz, |f|) surowe
layout(std430, binding = 1) readonly buffer BT { vec4 OT[]; };   // [K·4] ekran: koloid, pułapka (w = 1 zero pola), środek siatki (w = pół boku), skok
layout(std430, binding = 2) readonly buffer BL { vec4 OL[]; };   // [2·n] znaczniki: położenie (ekran), kolor
layout(std430, binding = 3) readonly buffer BW { vec4 OW[]; };   // [2·n] wiązki i wkłady wybranej cząsteczki: położenie, kolor
layout(std430, binding = 4) readonly buffer BB { vec4 OB[]; };   // [2·n] światło ogniska: kierunek wiązki + jasność, kolor
layout(std430, binding = 7) readonly buffer BQ { vec4 OP[]; };   // [2·600] cząsteczki w tle: położenie (ekran), kolor przyciemniony
layout(std430, binding = 6) readonly buffer BF { vec4 OF[]; };   // [2·n] obrys ogniska: trójkąty, potem krawędzie (położenie, kolor)
layout(std430, binding = 5) readonly buffer BR { vec4 OR[]; };   // [2·n] skala sfery (bez skali chmury): siatka szkła, wiązki od reflektorów, reflektory
"""
VS_O = "#version 430\n" + _SSBO_O + """
uniform mat4 u_mvp;
uniform int u_tryb, u_G, u_N, u_ka, u_kb, u_pods, u_wzgl, u_slad, u_wsp, u_sl;
uniform float u_ff, u_fc, u_skala, u_ostr, u_tlo, u_jas, u_rozm, u_strz;
uniform float u_w0, u_rozb, u_Lb, u_px; uniform int u_M; uniform vec4 u_c;
uniform vec4 u_sA, u_sB;        // x: skala surowe → ekran, y: mediana |f| na siatce (surowe)
uniform vec4 u_skA, u_skB;      // skok koloidu (surowy), do pola względem skoku
uniform vec4 u_ref;             // pułapka teraz (ślad w układzie pułapki)
uniform float u_faza; uniform int u_plyn;   // płynące strzałki pola: faza = czas · tempo
out vec4 v_k;
const vec4 POZA = vec4(2.0, 2.0, 2.0, 1.0);
vec3 wezel(int inst, int i) {
    vec4 c = OT[(inst == 0 ? u_ka : u_kb) * 4 + 2];
    ivec3 q = ivec3(i % u_G, (i / u_G) % u_G, i / (u_G * u_G));
    return c.xyz + ((vec3(q) / float(u_G - 1)) * 2.0 - 1.0) * c.w;
}
bool poza_kula(int inst, int i) {           // pole tylko w kuli komórki tokenu (promień = połowa boku siatki)
    vec4 c = OT[(inst == 0 ? u_ka : u_kb) * 4 + 2];
    return length(wezel(inst, i) - c.xyz) > c.w * 1.0001;
}
vec3 slad(int k, int sl) {                 // punkt śladu (sl 0 koloid, 1 pułapka); ostatni w ułamku chwili
    if (k != u_kb) return OT[k * 4 + sl].xyz;
    return mix(OT[u_ka * 4 + sl].xyz, OT[u_kb * 4 + sl].xyz, sl == 0 ? u_fc : u_ff);
}
void main() {
    int v = gl_VertexID;
    v_k = vec4(0.0);
    if (u_tryb == 0) {                      // mgła pola: instancja 0 = checkpoint a, 1 = b (przenikanie u_ff)
        int inst = gl_InstanceID, i = v;
        vec4 kol = OG[(inst * 2) * u_N + i], pol = OG[(inst * 2 + 1) * u_N + i];
        vec4 s = inst == 0 ? u_sA : u_sB;
        float sig = max(u_ostr * s.y, 1e-30);
        float g = exp(-0.5 * pol.w * pol.w / (sig * sig));          // ognisko: tam, gdzie pchnięcia wiązek się znoszą
        float a = (inst == 0 ? 1.0 - u_ff : u_ff) * u_jas * (u_tlo + (1.0 - u_tlo) * g);
        if (a < 1e-4 || poza_kula(inst, i)) { gl_Position = POZA; return; }
        v_k = vec4(kol.rgb * (0.45 + 0.55 * kol.a) * a, 1.0);
        gl_Position = u_mvp * vec4(u_skala * wezel(inst, i), 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 1) {               // strzałki pola co u_pods węzłów (instancja = węzeł + N·(0 a, 1 b)), 6 wierzchołków: trzon i grot
        int inst = gl_InstanceID / u_N, i = gl_InstanceID % u_N;
        ivec3 q = ivec3(i % u_G, (i / u_G) % u_G, i / (u_G * u_G));
        float w = inst == 0 ? 1.0 - u_ff : u_ff;
        if (w < 0.02 || q.x % u_pods != 0 || q.y % u_pods != 0 || q.z % u_pods != 0 || poza_kula(inst, i)) { gl_Position = POZA; return; }
        vec4 kol = OG[(inst * 2) * u_N + i], pol = OG[(inst * 2 + 1) * u_N + i];
        vec4 s = inst == 0 ? u_sA : u_sB;
        vec3 f = pol.xyz - (u_wzgl == 1 ? (inst == 0 ? u_skA.xyz : u_skB.xyz) : vec3(0.0));
        vec3 p = u_skala * wezel(inst, i), e = p + u_skala * s.x * u_strz * f, d = e - p;
        float L = length(d);
        if (L < 1e-6) { gl_Position = POZA; return; }
        float plyn = 1.0;
        if (u_plyn == 1) {                    // strzałka przesuwa się wzdłuż pola (faza węzła losowa), pojawia się i gaśnie
            float ph = fract(u_faza + fract(sin(float(i) * 12.9898) * 43758.5453));
            vec3 sh = (ph - 0.5) * d; p += sh; e += sh; plyn = sin(3.14159265 * ph);
        }
        vec4 kc = OT[(inst == 0 ? u_ka : u_kb) * 4 + 2];
        if (length(p - u_skala * kc.xyz) > u_skala * kc.w || length(e - u_skala * kc.xyz) > u_skala * kc.w) { gl_Position = POZA; return; }
        vec3 t = d / L, n = normalize(cross(t, abs(t.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)));
        vec3 r = v == 0 ? p : (v == 3 ? e - 0.3 * L * t + 0.15 * L * n : (v == 5 ? e - 0.3 * L * t - 0.15 * L * n : e));
        v_k = vec4(mix(vec3(1.0), kol.rgb, 0.7), w * u_jas * plyn * (v == 0 ? 0.35 : 0.95));
        gl_Position = u_mvp * vec4(r, 1.0);
    } else if (u_tryb == 2) {               // znaczniki (linie albo punkty z bufora OL)
        v_k = OL[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OL[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 7) {               // światło ogniska: wiązka gaussowska i (instancja) z przewężeniem w0 w cząsteczce
        vec4 b = OB[2 * gl_InstanceID], kl = OB[2 * gl_InstanceID + 1];
        float zR = u_w0 / max(u_rozb, 1e-4);                                   // długość Rayleigha z kąta rozbieżności
        float a = log(u_Lb / zR + sqrt(u_Lb * u_Lb / (zR * zR) + 1.0));         // asinh(Lb/zR): próbki gęściej przy ognisku
        float t = float(v) / float(max(u_M - 1, 1)) * 2.0 - 1.0;
        float sx = zR * (exp(a * t) - exp(-a * t)) * 0.5;
        float w = u_w0 * sqrt(1.0 + sx * sx / (zR * zR));
        float I = b.w * u_jas * (u_w0 / w) * (u_w0 / w);
        if (I < 1e-5) { gl_Position = POZA; return; }
        v_k = vec4(kl.rgb * I, 1.0);
        gl_Position = u_mvp * vec4(u_skala * (u_c.xyz - sx * b.xyz), 1.0);
        gl_PointSize = clamp(2.0 * w * u_skala * u_px, 1.5, 256.0);
    } else if (u_tryb == 10) {              // pozostałe cząsteczki w tle (przyciemnione)
        v_k = OP[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OP[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 9) {               // obrys ogniska: powierzchnia równej jasności światła wiązek (bufor OF)
        v_k = OF[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OF[2 * v].xyz, 1.0);
    } else if (u_tryb == 8) {               // reflektory na sferze i ich wiązki do cząsteczki (współrzędne sfery, bez u_skala)
        v_k = OR[2 * v + 1];
        gl_Position = u_mvp * vec4(OR[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 6) {               // 768 wiązek zbiegających się w wybranej cząsteczce i ich wkłady (bufor OW)
        v_k = OW[2 * v + 1];
        gl_Position = u_mvp * vec4(u_skala * OW[2 * v].xyz, 1.0); gl_PointSize = u_rozm;
    } else if (u_tryb == 3) {               // ślad koloidu (u_sl 0) albo pułapki (1) przez checkpointy (wierzchołek = checkpoint)
        int k = v;
        vec3 p = slad(k, u_sl);
        if (u_wsp == 1) p = p - slad(k, 1) + u_ref.xyz;                 // układ pułapki: pułapka stoi w miejscu
        float wiek = float(u_kb - k) / max(float(u_slad), 1.0);
        vec3 c = u_sl == 0 ? vec3(1.0, 0.82, 0.4) : vec3(0.85, 0.6, 1.0);
        v_k = vec4(c, 0.12 + 0.88 * (1.0 - wiek));
        gl_Position = u_mvp * vec4(u_skala * p, 1.0); gl_PointSize = u_rozm;
    }
}
"""
FS_O = "#version 430\n" + """
uniform int u_punkty;                   // 0 linie, 1 mgła (miękka plama, kolor przemnożony), 2 pełna kropka
in vec4 v_k; out vec4 o;
void main() {
    if (u_punkty == 0) { o = v_k; return; }
    vec2 d = gl_PointCoord * 2.0 - 1.0; float r = length(d);
    if (r > 1.0) discard;
    if (u_punkty == 1) { o = vec4(v_k.rgb * exp(-r * r * 3.0), 1.0); return; }
    o = vec4(v_k.rgb, v_k.a * (1.0 - smoothstep(0.7, 1.0, r)));
}
"""

KOL_SCIAGA, KOL_WYPYCHA, KOL_OBROT = (0.3, 0.85, 1.0), (1.0, 0.35, 0.25), (0.82, 0.45, 1.0)
KOL_SKOK, KOL_PULAPKA, KOL_RUCH = (1.0, 0.62, 0.25), (1.0, 1.0, 1.0), (0.35, 1.0, 0.5)


def siatka_obrysu(kn, jn, Lb, w0, zR, poz, poziom_wzgl=0.5):
    """Obrys ogniska w układzie cząsteczki (bez położenia i lupy): objętość światła wiązek gaussowskich na siatce 33³ liczona na
    GPU (PyTorch), izopowierzchnia na poziomie `poz` (common.izopowierzchnia), cieniowanie, osie i warstwice (przekroje ⟂ dwóch
    najdłuższych osi). Zwraca dict(v, f, sz, kr, osie, kier, w0, poziom) albo None (brak powierzchni)."""
    import torch
    from laser_app.compute.common import izopowierzchnia
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = jn > 1e-4 * float(jn.max())
    K = torch.as_tensor(kn[m], dtype=torch.float32, device=dev); J = torch.as_tensor(jn[m], dtype=torch.float32, device=dev)

    def jasnosc(X):                                                             # suma wiązek w punktach X (względem cząsteczki)
        x = torch.as_tensor(X, dtype=torch.float32, device=dev)
        out = []
        for i in range(0, len(x), 16384):
            xi = x[i:i + 16384]
            s = xi @ K.T; r2 = (xi * xi).sum(1, keepdim=True)
            w2 = w0 * w0 * (1 + s * s / (zR * zR)); rho = (r2 - s * s).clamp_min(0)
            out.append(((J * (w0 * w0 / w2) * torch.exp(-3 * rho / w2)) * (s.abs() <= Lb)).sum(1))
        return torch.cat(out).cpu().numpy()                                     # jedna synchronizacja z kartą na objętość

    G, h = 33, 2.0 * w0                                                          # pudło rośnie, aż obrys się w nim zmieści
    while True:
        ax = np.linspace(-h, h, G)
        X = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
        V = jasnosc(X).reshape(G, G, G)
        brzeg = max(V[0].max(), V[-1].max(), V[:, 0].max(), V[:, -1].max(), V[:, :, 0].max(), V[:, :, -1].max())
        if brzeg < poz or h >= Lb:
            break
        h *= 1.6
    if V.max() <= poz:
        return None
    try:
        v, f, nrm = izopowierzchnia(V, poz, 2 * h / (G - 1))
    except (ValueError, RuntimeError):
        return None
    v = v - h
    L = np.array([0.35, 0.5, 0.8]); L /= np.linalg.norm(L)                      # cieniowanie: światło z ustalonego kierunku
    sz = 0.35 + 0.65 * np.abs(nrm @ L)
    vc = v - v.mean(0)                                                           # osie obrysu: główne kierunki wierzchołków
    lam, wek = np.linalg.eigh(vc.T @ vc / len(vc))
    rz = vc @ wek; dl = rz.max(0) - rz.min(0)
    o_ = np.argsort(-dl)
    odc = []                                                                     # warstwice: przekroje płaszczyznami ⟂ osi 1 i 2
    for e, kol_ in ((wek[:, o_[0]], (1.0, 0.82, 0.4)), (wek[:, o_[1]], (0.85, 0.9, 1.0))):
        dd = v @ e
        for l in np.linspace(dd.min(), dd.max(), 7)[1:-1]:
            s = dd[f] - l                                                        # (F, 3) znak wierzchołków względem płaszczyzny
            pk = []
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                sa, sb = s[:, a_], s[:, b_]
                t_ = np.where(sa * sb < 0, sa / np.where(sa - sb == 0, 1, sa - sb), np.nan)
                pk.append(v[f[:, a_]] + t_[:, None] * (v[f[:, b_]] - v[f[:, a_]]))
            pk = np.stack(pk, 1)                                                 # (F, 3 krawędzie, 3)
            ok = ~np.isnan(pk[..., 0])
            m2 = ok.sum(1) == 2
            if m2.any():
                odc.append((pk[m2][ok[m2]].reshape(-1, 2, 3), kol_))
    if odc:
        Kr = np.concatenate([np.concatenate([P.reshape(-1, 1, 3), np.zeros((P.size // 3, 1, 3))], 1) for P, _ in odc])
        Kr = np.concatenate([Kr, np.zeros((len(Kr), 2, 1))], 2).astype(np.float32)
        Kr[:, 1] = np.concatenate([np.tile([*k_, 0.55], (P.size // 3, 1)) for P, k_ in odc])
    else:
        Kr = np.zeros((0, 2, 4), np.float32)
    return dict(v=v, f=np.asarray(f), sz=sz, kr=Kr, osie=dl[o_] / w0, kier=wek[:, o_], w0=w0, poziom=float(poziom_wzgl))


try:
    from PyQt6.QtCore import QObject as _QObject, pyqtSignal as _pyqtSignal

    class _OutlineWorker(_QObject):
        """Wątek liczący obrys ogniska (siatka_obrysu) dla OSTATNIEGO zlecenia; wynik sygnałem `gotowe` w wątku okna. Dzięki temu
        odtwarzanie z obrysem nie czeka na objętość i izopowierzchnię (wcześniej kilkanaście–kilkadziesiąt ms w każdej klatce)."""
        gotowe = _pyqtSignal(object)

        def __init__(self):
            super().__init__()
            import threading
            self._zl = None
            self._ev = threading.Event()
            threading.Thread(target=self._petla, daemon=True, name='obrys_ogniska').start()

        def zlec(self, args):
            self._zl = args
            self._ev.set()

        def _petla(self):
            while True:
                self._ev.wait(); self._ev.clear()
                args, self._zl = self._zl, None
                if args is None:
                    continue
                try:
                    kn, jn, Lb, w0, zR, poz = args
                    w = siatka_obrysu(kn, jn, Lb, w0, zR, poz, poz / max(float(jn.sum()), 1e-30))
                except Exception:                                                   # noqa: BLE001 (np. brak torch): bez obrysu
                    import traceback
                    traceback.print_exc()
                    w = None
                self.gotowe.emit(w)
except ImportError:                                                               # bez Qt (testy obliczeń): tylko siatka_obrysu
    pass


class FocalField(_GPUElement):
    """Pułapka optyczna z laserów wokół wybranego koloidu (focal_trap.policz): mgła pola w kolorach wiązek (jasna tam, gdzie
    pchnięcia się znoszą), strzałki pola albo kształtu pułapki (pole względem skoku koloidu), osie siodła i wir w koloidzie
    i w pułapce, skok koloidu w bloku 6, ślady koloidu i pułapki przez checkpointy. Te same współrzędne co ReflectorSphere."""
    BUFORY = ('OG', 'OT', 'OL', 'OW', 'OB', 'OR', 'OF', 'OP')

    def __init__(self):
        super().__init__()
        self.d = None
        self.opcje = dict(mgla=True, jas=0.25, ostr=0.35, tlo=0.05, rozm=22.0, strzalki=2, dl_strz=0.8, pods=3, osie=True, skok=True,
                          slad=1, n_slad=30, wsp=0, skala_chmury=1.0, kropki=True, pudlo=True,
                          wiazki=True, jasnosc=0, gain_w=0.6, dl_wiazki=1.0, wklady=1, dl_wkladow=1.0)
        self.kol = None; self.n_wiazek = self.n_ow = self.n_ob = 0; self.tryb_pod = False; self.Lb = 1.0; self.c_ogn = np.zeros(3)
        self.opcje.update(swiatlo=True, jas_sw=0.35, szer_sw=0.02, rozb_sw=0.15, px=10.0, linie_wiazek=True, osie_ogniska=True,
                          refl=False, R=34.0, sfera=0.10, jas_refl=1.0, obrys=False, poziom_obr=0.5, jas_obr=0.35, pow_obr=20.0)
        self.n_of_tr = self.n_of_kr = 0
        self.n_op = 0
        self.RF = None; self.n_or_lin = self.n_or_wiaz = self.n_or_pkt = 0; self._siatka = None
        self.para = (-1, -1)
        self.ka = self.kb = 0
        self.ff = self.fc = 0.0
        self.n_ol = 0; self.n_pkt = 0

    def ustaw_dane(self, d, skala_k):
        """d: wynik focal_trap.policz; skala_k (K,): surowe jednostki rzutu → ekran sfery (0,55 R / r99)."""
        self.d = d
        s = np.asarray(skala_k, np.float32)[:, None]
        K = len(d['kroki'])
        OT = np.zeros((K, 4, 4), np.float32)
        OT[:, 0, :3] = d['srodek'] * s
        OT[:, 1, :3] = d['ogn'] * s; OT[:, 1, 3] = d['zb']
        OT[:, 2, :3] = d['srodek_siatki'] * s; OT[:, 2, 3] = d['L'] * s[:, 0]
        OT[:, 3, :3] = d['skok'] * s; OT[:, 3, 3] = np.linalg.norm(d['skok'], axis=1) * s[:, 0]
        self.OT, self.skala_k = OT, s[:, 0]
        self.ustaw_bufor('OT', OT.reshape(-1, 4))
        self.G = int(d['G']); self.N = self.G ** 3
        self.K = K; self.para = (-1, -1)
        self.update()

    def czysc(self):
        self.d = None; self.K = 0; self.update()

    def ustaw_chwile(self, t, kolejnosc=False):
        """t = ułamkowy indeks checkpointu. kolejnosc: najpierw lasery (pole, pułapka, skok), potem koloid."""
        if self.d is None:
            return
        K = self.K
        t = float(np.clip(t, 0, K - 1)); a = int(np.floor(t)); b = min(a + 1, K - 1); f = t - a
        self.ka, self.kb = a, b
        self.ff, self.fc = (min(2 * f, 1.0), max(2 * f - 1, 0.0)) if kolejnosc else (f, f)
        if (a, b) != self.para:
            OG = np.empty((2, 2, self.N, 4), np.float32)
            for i, k in enumerate((a, b)):
                OG[i, 0] = self.d['kolor'][k]; OG[i, 1] = self.d['pole'][k]
            self.ustaw_bufor('OG', OG.reshape(-1, 4))
            self.para = (a, b)
        if not self.opcje.get('tylko_pole', False):                            # podkroki: sama mgła i strzałki (bez osi i wiązek)
            self._znaczniki()
            self._wiazki()
        self.update()

    def ustaw_kolory(self, kol):
        """Kolory 768 reflektorów (jak na sferze), do wiązek zbiegających się w cząsteczce."""
        self.kol = np.asarray(kol, np.float32)[:, :3]

    def ustaw_tlo(self, P, kol=None, jas=0.35, bez=None):
        """Pozostałe cząsteczki w tle (ekran sfery, bez skali chmury): P (600, 3), kol (600, 3) kolory cząsteczek, jas = przyciemnienie,
        bez = indeks cząsteczki pominiętej (ta w ognisku). P = None wyłącza tło."""
        if P is None:
            self.n_op = 0; self.update(); return
        P = np.asarray(P, np.float32); n = len(P)
        kol = np.full((n, 3), 0.8, np.float32) if kol is None else np.asarray(kol, np.float32)[:, :3]
        B = np.zeros((n, 2, 4), np.float32); B[:, 0, :3] = P
        B[:, 1, :3] = kol * float(jas); B[:, 1, 3] = 1.0
        if bez is not None and 0 <= int(bez) < n:
            B = np.delete(B, int(bez), axis=0)
        self.n_op = len(B)
        self.ustaw_bufor('OP', B.reshape(-1, 4))

    def ustaw_reflektory(self, RF):
        """Miejsca 768 reflektorów bloku 6 na zewnętrznej sferze (h.rfix, współrzędne sfery), do widoku w skali sfery."""
        self.RF = None if RF is None else np.asarray(RF, np.float64)[:, :3]

    def _wiazki(self):
        """Oś uczenia: 768 wiązek bloku 6 zbiegających się w wybranej cząsteczce (dane focal_trap.policz: P3 v_c, z_jc, z̄_c)."""
        o, d = self.opcje, self.d
        if self.tryb_pod:
            return
        widac_cos = o.get('wiazki', True) or o.get('obrys', False) or o.get('swiatlo', True) or int(o.get('wklady', 1))
        if not widac_cos or d is None or 'pv' not in d or self.kol is None:  # obrys, światło i wkłady działają też bez linii wiązek
            self._ognisko_czysc()
            return
        a, b, ff = self.ka, self.kb, self.ff
        k = self.k_teraz(); s = float(self.skala_k[k])
        pv = ((1 - ff) * d['pv'][a] + ff * d['pv'][b]) * s                            # (768, 3) ekran
        zj = (1 - ff) * d['zj'][a].astype(np.float32) + ff * d['zj'][b].astype(np.float32)
        zb = (1 - ff) * d['zbar'][a].astype(np.float32) + ff * d['zbar'][b].astype(np.float32)
        wk = (zj - zb)[:, None] * pv
        if int(o.get('jasnosc', 0)) == 0:                                      # wkład do skoku: wiązka wchodzi z kierunku, w który pcha
            kier, kol, jas = wk, 0.25 + 0.75 * self.kol, np.linalg.norm(wk, axis=1)
        else:                                                                  # jak na sferze: kierunek v_c, błękit = ujemna dioda
            kier, jas = pv, np.abs(zj) * np.linalg.norm(pv, axis=1)
            kol = np.where((zj < 0)[:, None], np.array([0.30, 0.85, 1.0], np.float32), 0.25 + 0.75 * self.kol)
        self.ustaw_ognisko(self.koloid(), kier, wk, jas, kol, ruch=None, zrodla=self.RF,
                           zapalone=float((zj > 0.05).mean()), ujemne=float((zj < -0.05).mean()))

    def _ognisko_czysc(self):
        self.n_wiazek = self.n_ow = self.n_ob = 0
        self.n_or_lin = self.n_or_wiaz = self.n_or_pkt = 0
        self.n_of_tr = self.n_of_kr = 0
        self.info_wiazek = None
        self.ustaw_bufor('OW', np.zeros((1, 4), np.float32)); self.ustaw_bufor('OB', np.zeros((1, 4), np.float32))

    def ustaw_ognisko(self, c, kier, wk, jas, kol, ruch=None, zrodla=None, prosto=False, Lb_staly=None, refl_tlo=None,
                      jas_wzgl=False, **info):
        """Ognisko JEDNEJ cząsteczki (ekran sfery): c środek; kier (n, 3) kierunki wiązek (dowolnej długości), wk (n, 3) wkłady
        do skoku, jas (n,) ≥ 0 jasności, kol (n, 3) kolory; ruch = prawdziwe przesunięcie (kontrola, biała strzałka).
        Wiązka i wchodzi w cząsteczkę z kierunku −kier_i; światło ogniska rysuje shader (wiązki gaussowskie, przewężenie w c);
        wkłady jako strzałki albo łańcuch głowa–ogon (koniec = Σ wk = skok, pomarańczowy).
        zrodla (n, 3): skąd świecą wiązki w widoku w skali sfery (opcja refl): reflektory na zewnętrznej sferze (prosto=False,
        wiązka reflektor → szklana sfera → cząsteczka, jak film 12) albo inne cząsteczki w ich współrzędnych (prosto=True, uwaga);
        prosto może być tablicą (n,) – płynne przejście podkroków: wiązki MLP i wkłady uwagi naraz.
        jas_wzgl: jasności już względne 0..1 (bez normowania do najjaśniejszej; przenikanie dwóch zestawów wiązek)."""
        o = self.opcje
        c = np.asarray(c, np.float64); kier = np.asarray(kier, np.float64); wk = np.asarray(wk, np.float64)
        jas = np.asarray(jas, np.float64); kol = np.asarray(kol, np.float32)[:, :3]; n = len(kier)
        kn = kier / np.maximum(np.linalg.norm(kier, axis=1), 1e-12)[:, None]
        skok = wk.sum(0); Ls = float(np.linalg.norm(skok))
        dl_w = np.linalg.norm(wk, axis=1); suma = float(dl_w.sum())
        Lb = float(o.get('dl_wiazki', 1.0)) * max(1.5 * Ls, 0.35 * suma / max(np.sqrt(n), 1.0), 1e-6)
        if Lb_staly is not None:                                                         # podkroki: jedna długość na całe przejście
            Lb = float(o.get('dl_wiazki', 1.0)) * max(float(Lb_staly), 1e-6)
        przytnij = bool(o.get('przytnij', False)) and not self.tryb_pod and self.d is not None and int(self.d['G']) > 1
        if przytnij:                                                                     # kadr na komórkę tokenu: wiązki i wkłady w sześcianie
            Lb = min(Lb, self.pol_boku())
        self.Lb = Lb
        jn = (jas if jas_wzgl else jas / max(float(jas.max()), 1e-12)) * float(o.get('gain_w', 1.0))
        czesci = []
        wiazki = bool(o.get('wiazki', True))                                             # „768 wiązek…”: same linie wiązek
        refl = wiazki and bool(o.get('refl', False)) and zrodla is not None and len(zrodla) == n
        self._reflektory(c, kn, jn, kol, zrodla if refl else None, prosto, refl_tlo if refl else None)
        if wiazki and o.get('linie_wiazek', True) and not refl:                         # wiązki: źródło (ciemne) → cząsteczka
            W = np.zeros((n, 2, 2, 4), np.float32)
            W[:, 0, 0, :3] = c - Lb * kn; W[:, 1, 0, :3] = c
            W[:, 0, 1, :3] = kol * jn[:, None] * 0.08; W[:, 1, 1, :3] = kol * jn[:, None] * 0.7; W[:, :, 1, 3] = 1.0
            czesci.append(W)
        n_w = n * 2 if czesci else 0
        tryb = int(o.get('wklady', 1))
        if tryb:
            dw = float(o.get('dl_wkladow', 1.0)); kk = kol
            if tryb == 1:                                                                # strzałki od cząsteczki
                sk_w = dw * (Ls / max(float(np.median(dl_w)), 1e-12) * 0.15)
                if przytnij:                                                             # najdłuższy wkład najwyżej do ścianki sześcianu
                    sk_w = min(sk_w, Lb / max(float(dl_w.max()), 1e-30))
                P0 = np.repeat(c[None], n, 0); P1 = c + wk * sk_w
            else:                                                                        # łańcuch głowa–ogon, od najdłuższego
                o_ = np.argsort(-dl_w); cum = np.cumsum(wk[o_], 0) * dw
                P0 = c + np.vstack([np.zeros((1, 3)), cum[:-1]]); P1 = c + cum; kk = kol[o_]
            Wk = np.zeros((n, 2, 2, 4), np.float32)
            Wk[:, 0, 0, :3] = P0; Wk[:, 1, 0, :3] = P1
            Wk[:, :, 1, :3] = kk[:, None, :]; Wk[:, 0, 1, 3] = 0.35; Wk[:, 1, 1, 3] = 0.9
            czesci.append(Wk)
        extra = []

        def strz(p_, q_, kol_, a_):
            dd = q_ - p_; L_ = float(np.linalg.norm(dd))
            if L_ < 1e-12:
                return
            t_ = dd / L_; n_ = np.cross(t_, [0, 0, 1.0] if abs(t_[2]) < 0.9 else [1.0, 0, 0]); n_ /= np.linalg.norm(n_)
            g = 0.2 * L_
            for pp, qq in ((p_, q_), (q_, q_ - g * t_ + 0.5 * g * n_), (q_, q_ - g * t_ - 0.5 * g * n_)):
                extra.append((pp, qq, kol_, a_))

        strz(c, c + skok, (1.0, 0.62, 0.25), 1.0)                                        # skok = Σ wkładów (pomarańczowy)
        if ruch is not None:
            strz(c, c + np.asarray(ruch, np.float64), (1.0, 1.0, 1.0), 0.7)             # prawdziwy ruch (biały, kontrola)
        # kształt ogniska: tensor kierunków T = Σ jas d dᵀ / Σ jas; ognisko wydłużone wzdłuż kierunku o największej wartości
        T = (jas[:, None, None] * kn[:, :, None] * kn[:, None, :]).sum(0) / max(float(jas.sum()), 1e-30)
        lam, wek = np.linalg.eigh(T); lam, wek = lam[::-1], wek[:, ::-1]
        dl_osi = 1.0 / np.sqrt(np.maximum(1.0 - lam, 1e-6)); dl_osi = dl_osi / dl_osi.max() * 0.25 * Lb
        if o.get('osie_ogniska', True):
            for e in range(3):
                extra.append((c - dl_osi[e] * wek[:, e], c + dl_osi[e] * wek[:, e], (1.0, 1.0, 1.0), 0.55))
        if extra:
            E = np.zeros((len(extra), 2, 2, 4), np.float32)
            for i, (pp, qq, kl, a_) in enumerate(extra):
                E[i, 0, 0, :3] = pp; E[i, 1, 0, :3] = qq; E[i, :, 1, :3] = kl; E[i, :, 1, 3] = a_
            czesci.append(E)
        W = np.concatenate(czesci) if czesci else np.zeros((0, 2, 2, 4), np.float32)
        self.n_wiazek, self.n_ow = n_w, len(W) * 2
        pkt = np.array([[*c, 0.0], [1.0, 0.82, 0.4, 1.0]], np.float32)                 # sama cząsteczka (kropka) na końcu bufora
        self.ustaw_bufor('OW', np.concatenate([W.reshape(-1, 4), pkt]))
        B = np.zeros((n, 2, 4), np.float32)                                             # światło ogniska: kierunek + jasność, kolor
        B[:, 0, :3] = kn; B[:, 0, 3] = jn; B[:, 1, :3] = kol
        self.ustaw_bufor('OB', B.reshape(-1, 4)); self.n_ob = n; self.c_ogn = c
        if o.get('obrys', False):
            obr = self._obrys(c, kn, jn, Lb)
        else:
            obr = self._obrys_wynik = None
        top = np.argsort(-dl_w)[:3]
        self.info_wiazek = dict(info, n=n, suma=suma, wypadkowa=Ls, top=top, udzial=dl_w[top] / max(suma, 1e-12),
                                ksztalt=lam, wydluzenie=float(np.sqrt((1 - lam[2]) / max(1 - lam[0], 1e-6))), obrys=obr,
                                blad=(float(np.linalg.norm(skok - np.asarray(ruch)) / max(float(np.linalg.norm(ruch)), 1e-30))
                                      if ruch is not None else None))
        self.update()

    def _obrys(self, c, kn, jn, Lb):
        """Obrys 3D ogniska: powierzchnia, na której suma światła wiązek gaussowskich (te same co w rysowanym świetle:
        przewężenie w0, długość Rayleigha zR = w0 / rozbieżność, profil exp(−3ρ²/w²), jasność ∝ (w0/w)²) spada do ułamka
        `poziom_obr` jasności w cząsteczce (0,5 = połowa szczytu, jak FWHM objętości ogniska w optyce). Kształt zależy od danych
        (kierunki i jasności wiązek), rozmiar od wybranego przewężenia. Zwraca osie obrysu (długości / w0, kierunki) albo None.
        Liczenie (objętość światła na GPU przez PyTorch + izopowierzchnia) idzie w wątku w tle (`_OutlineWorker`, zawsze ostatnie
        zlecenie), więc klatka nie czeka; rysowany jest ostatni gotowy obrys przesunięty do bieżącego położenia cząsteczki."""
        o = self.opcje
        w0 = float(o.get('szer_sw', 0.02)) * Lb; zR = w0 / max(float(o.get('rozb_sw', 0.15)), 1e-4)
        szczyt = float(jn.sum())
        if szczyt <= 0 or w0 <= 0:
            self.n_of_tr = self.n_of_kr = 0
            return None
        if getattr(self, '_licz_obrys', None) is None:
            self._licz_obrys = _OutlineWorker()
            self._licz_obrys.gotowe.connect(self._obrys_gotowy)
        self._licz_obrys.zlec((np.array(kn, np.float32), np.array(jn, np.float32), float(Lb), w0, zR,
                               float(o.get('poziom_obr', 0.5)) * szczyt))
        w = getattr(self, '_obrys_wynik', None)
        return None if w is None else self._obrys_wstaw(c, w)

    def _obrys_gotowy(self, w):
        """Wątek policzył obrys: zapamiętujemy (w układzie cząsteczki) i wstawiamy go w bieżące położenie."""
        if not self.opcje.get('obrys', False):
            return
        self._obrys_wynik = w
        info = self._obrys_wstaw(getattr(self, 'c_ogn', np.zeros(3)), w)
        if isinstance(getattr(self, 'info_wiazek', None), dict):
            self.info_wiazek['obrys'] = info
        self.update()

    def _obrys_wstaw(self, c, w):
        """Bufor OF: trójkąty obrysu i warstwice w punkcie c (ekran), z lupą i kryciem z opcji. w = None: brak obrysu."""
        o = self.opcje
        if w is None:
            self.n_of_tr = self.n_of_kr = 0
            return None
        v, f, sz = w['v'], w['f'], w['sz']
        c = np.asarray(c, np.float64)
        jo = float(o.get('jas_obr', 0.35))
        pw = float(o.get('pow_obr', 20.0))                                       # lupa: obrys powiększony wokół cząsteczki
        if o.get('przytnij', False) and not self.tryb_pod and self.d is not None and int(self.d['G']) > 1:
            pw = self.pol_boku() / max(float(np.linalg.norm(v, axis=1).max()), 1e-30)   # kadr komórki: obrys wypełnia kulę komórki
        Tr = np.zeros((len(f) * 3, 2, 4), np.float32)
        Tr[:, 0, :3] = c + pw * v[f.reshape(-1)]
        Tr[:, 1, :3] = (sz[f.reshape(-1)][:, None] * np.array([1.0, 0.9, 0.62]))
        Tr[:, 1, 3] = 0.5 * jo                                                   # powierzchnia słabo, warstwice wyraźnie
        Kr = w['kr'].copy()
        Kr[:, 0, :3] = c + pw * Kr[:, 0, :3]
        self.n_of_tr, self.n_of_kr = len(Tr), len(Kr)
        self.ustaw_bufor('OF', np.concatenate([Tr.reshape(-1, 4), Kr.reshape(-1, 4)]))
        return dict(osie=w['osie'], kier=w['kier'], poziom=w['poziom'], w0=w['w0'], pow=pw)

    def _reflektory(self, c, kn, jn, kol, zrodla, prosto, refl_tlo=None):
        """Widok w skali sfery (film 12): szklana sfera R, reflektory na zewnętrznej sferze i ich wiązki do cząsteczki.
        Wiązka c idzie od reflektora do punktu załamania e na szklanej sferze, a dalej wzdłuż kierunku kn_c do cząsteczki
        (e = p − s·kn, |e| = R); w uwadze źródłami są inne cząsteczki (linia prosta). Współrzędne bez skali chmury."""
        o = self.opcje
        if zrodla is None:
            self.n_or_lin = self.n_or_wiaz = self.n_or_pkt = 0
            return
        R = float(o.get('R', 34.0)); sk = float(o.get('skala_chmury', 1.0))
        p = np.asarray(c, np.float64) * sk; n = len(kn)
        if self._siatka is None or self._siatka[0] != R:
            self._siatka = (R, siatka_sfery(R)[:, :3])
        S = self._siatka[1]
        G = np.zeros((len(S), 2, 4), np.float32); G[:, 0, :3] = S; G[:, 1] = (0.45, 0.6, 0.75, float(o.get('sfera', 0.10)))
        jr = (jn * float(o.get('jas_refl', 1.0)))[:, None] * kol
        pr = np.broadcast_to(np.asarray(prosto, bool), (n,))                            # tablica: płynne przejście MLP ↔ uwaga
        Zs = np.asarray(zrodla, np.float64)
        Wy = []
        ip = np.flatnonzero(pr)
        if len(ip):                                                                      # uwaga: od innych cząsteczek prosto
            W = np.zeros((len(ip), 2, 2, 4), np.float32)
            W[:, 0, 0, :3] = Zs[ip] * sk; W[:, 1, 0, :3] = p
            W[:, 0, 1, :3] = 0.15 * jr[ip]; W[:, 1, 1, :3] = 0.8 * jr[ip]; W[..., 1, 3] = 1.0
            Wy.append(W)
        ir = np.flatnonzero(~pr)
        if len(ir):                                                                      # reflektor → szkło → cząsteczka
            kr, jrr = kn[ir], jr[ir]
            b = kr @ p; s = b + np.sqrt(np.maximum(b * b - p @ p + R * R, 0.0))
            e = p - s[:, None] * kr
            W = np.zeros((len(ir), 2, 2, 4), np.float32)
            W[:, 0, 0, :3] = Zs[ir]; W[:, 1, 0, :3] = e
            W[:, 0, 1, :3] = 0.25 * jrr; W[:, 1, 1, :3] = 0.4 * jrr
            W2 = np.zeros((len(ir), 2, 2, 4), np.float32)
            W2[:, 0, 0, :3] = e; W2[:, 1, 0, :3] = p
            W2[:, 0, 1, :3] = 0.4 * jrr; W2[:, 1, 1, :3] = 0.9 * jrr
            W = np.concatenate([W, W2]); W[..., 1, 3] = 1.0
            Wy.append(W)
        W = np.concatenate(Wy) if Wy else np.zeros((0, 2, 2, 4), np.float32)
        P = np.zeros((len(ir), 2, 4), np.float32); P[:, 0, :3] = Zs[ir]                 # punkty tylko reflektorów: cząsteczki-źródła
        P[:, 1, :3] = 0.25 + 0.75 * np.asarray(kol, np.float32)[ir]; P[:, 1, 3] = 0.85   # są w tle (przyciemnione)
        if refl_tlo is not None:                                                         # uwaga: reflektory bloku też świecą (bez wiązek)
            RF_, KR_ = refl_tlo
            P2 = np.zeros((len(RF_), 2, 4), np.float32); P2[:, 0, :3] = RF_
            P2[:, 1, :3] = 0.25 + 0.75 * np.asarray(KR_, np.float32)[:, :3]; P2[:, 1, 3] = 0.85
            P = np.concatenate([P, P2]); n = len(P)
        self.n_or_lin, self.n_or_wiaz, self.n_or_pkt = len(G), len(W) * 2, n
        self.ustaw_bufor('OR', np.concatenate([G.reshape(-1, 4), W.reshape(-1, 4), P.reshape(-1, 4)]))

    # ---------------------------------------------------------------- położenia w tej chwili (ekran, bez skali chmury)
    def koloid(self):
        return (1 - self.fc) * self.OT[self.ka, 0, :3] + self.fc * self.OT[self.kb, 0, :3]

    def pulapka(self):
        return (1 - self.ff) * self.OT[self.ka, 1, :3] + self.ff * self.OT[self.kb, 1, :3]

    def srodek_pola(self):
        return (1 - self.ff) * self.OT[self.ka, 2, :3] + self.ff * self.OT[self.kb, 2, :3]

    def pol_boku(self):
        return float((1 - self.ff) * self.OT[self.ka, 2, 3] + self.ff * self.OT[self.kb, 2, 3])

    def k_teraz(self):
        return self.kb if self.ff >= 0.5 else self.ka

    def _znaczniki(self):
        """Linie znaczników na CPU (kilkadziesiąt odcinków): skok, pułapka, osie siodła i wir, kierunek do pułapki, pudło siatki."""
        o, d = self.opcje, self.d
        k = self.k_teraz(); s = float(self.skala_k[k])
        L = self.pol_boku(); c = self.koloid(); T = self.pulapka()
        lin = []

        def odc(p, q, kol, a=1.0):
            lin.append((p, (*kol, a))); lin.append((q, (*kol, a)))

        def strzalka(p, q, kol, a=1.0, grot=0.28):
            p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
            dd = q - p; Ln = float(np.linalg.norm(dd))
            if Ln < 1e-9:
                return
            t_ = dd / Ln; n_ = np.cross(t_, [0, 0, 1.0] if abs(t_[2]) < 0.9 else [1.0, 0, 0]); n_ /= np.linalg.norm(n_)
            odc(p, q, kol, a)
            odc(q, q - grot * Ln * t_ + 0.5 * grot * Ln * n_, kol, a); odc(q, q - grot * Ln * t_ - 0.5 * grot * Ln * n_, kol, a)

        def ksztalt(srodek, lam, wek, r):
            lam = np.asarray(lam); wek = np.asarray(wek)
            m = max(float(np.abs(lam).max()), 1e-12); uzyte = set()
            for i in range(3):
                if i in uzyte:
                    continue
                l = lam[i]
                if abs(l.imag) > 1e-6 * max(abs(l), 1e-30):                    # para zespolona: wir w płaszczyźnie (Re v, Im v)
                    uzyte.update(j for j in range(i + 1, 3) if abs(lam[j] - np.conj(l)) < 1e-4 * max(abs(l), 1e-30))
                    vr, vi = wek[:, i].real, wek[:, i].imag
                    th = np.linspace(0, 2 * np.pi, 41)
                    P = np.outer(np.cos(th), vr) - np.outer(np.sin(th), vi)       # x(t) = Re(e^{λt} v): obrót od vr ku −vi
                    P *= r * (0.45 + 0.55 * min(abs(l) / m, 1)) / max(np.linalg.norm(P, axis=1).max(), 1e-12)
                    sg = 1.0 if l.imag > 0 else -1.0
                    for a_ in range(40):
                        odc(srodek + P[a_], srodek + P[a_ + 1], KOL_OBROT, 0.9)
                    for a_ in (0, 13, 27):                                       # groty wskazują kierunek obrotu
                        strzalka(srodek + P[a_], srodek + P[a_] + sg * (P[a_ + 1] - P[a_]) * 3.0, KOL_OBROT, 1.0, 0.6)
                else:
                    v = wek[:, i].real; v = v / max(np.linalg.norm(v), 1e-12)
                    dl = r * (0.35 + 0.65 * min(abs(l.real) / m, 1))
                    kol = KOL_SCIAGA if l.real < 0 else KOL_WYPYCHA
                    for zn in (1, -1):                                           # ściąga: groty do środka, wypycha: na zewnątrz
                        if l.real < 0:
                            strzalka(srodek + zn * dl * v, srodek + zn * 0.25 * dl * v, kol, 0.95)
                        else:
                            strzalka(srodek + zn * 0.25 * dl * v, srodek + zn * dl * v, kol, 0.95)
                uzyte.add(i)

        if o['pudlo']:                                                           # kula komórki tokenu (tu liczone jest pole)
            C = self.srodek_pola(); th = np.linspace(0, 2 * np.pi, 49)
            E = np.eye(3)
            okregi = [(E[0], E[1], 0.0), (E[0], E[2], 0.0), (E[1], E[2], 0.0)]
            okregi += [(E[0], E[1], z_) for z_ in (-0.6, 0.6)]
            for e1, e2, z_ in okregi:
                r_ = L * np.sqrt(1 - z_ * z_); e3 = np.cross(e1, e2)
                P_ = C + z_ * L * e3 + r_ * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
                for a_ in range(48):
                    odc(P_[a_], P_[a_ + 1], (0.5, 0.6, 0.7), 0.35 if z_ == 0 else 0.2)
        zb = bool(d['zb'][k])
        if o['osie']:
            ksztalt(c, d['lam0'][k], d['wek0'][k], 0.35 * L)                     # kształt pola w koloidzie
            if zb:
                ksztalt(T, d['lam'][k], d['wek'][k], 0.35 * L)                   # kształt pułapki
        if zb:                                                                   # pułapka: trzy okręgi (kulka)
            th = np.linspace(0, 2 * np.pi, 33); r = 0.07 * L
            for e1, e2 in (((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1))):
                Pk = T + r * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
                for a_ in range(32):
                    odc(Pk[a_], Pk[a_ + 1], KOL_PULAPKA, 1.0)
            odc(c, T, (1, 1, 1), 0.25)
        else:                                                                    # kierunek do pułapki (oszacowanie liniowe), do brzegu pudła
            q = d['ogn_lin'][k] * s
            dd = q - c; Ln = float(np.linalg.norm(dd))
            if Ln > 1e-9:
                strzalka(c, c + dd / Ln * min(Ln, 0.95 * L), (0.6, 0.75, 0.85), 0.55, 0.15)
        if o['skok'] and not o.get('wiazki', True):                             # skok koloidu w bloku 6 (gdy wiązki go nie rysują)
            sk = (1 - self.ff) * self.OT[self.ka, 3, :3] + self.ff * self.OT[self.kb, 3, :3]
            strzalka(c, c + sk, KOL_SKOK, 1.0, 0.2)
        self.info_ruch = None; self.n_ol_ruch = 0
        if self.kb > self.ka:                                                  # wypadkowy ruch koloidu w kroku ka → ka + 1
            p0, p1 = self.OT[self.ka, 0, :3].astype(np.float64), self.OT[self.kb, 0, :3].astype(np.float64)
            dr = p1 - p0; Lr = float(np.linalg.norm(dr))
            skk = self.OT[self.ka, 3, :3].astype(np.float64); Lk = float(np.linalg.norm(skk))
            self.info_ruch = dict(dl=Lr / max(float(self.skala_k[self.ka]), 1e-30),
                                  cos=float(dr @ skk / max(Lr * Lk, 1e-30)), stosunek=Lr / max(Lk, 1e-30))
            if o.get('ruch_nast', True) and Lr > 0:
                q = dr
                if o.get('przytnij', False) and self.G > 1:                     # w kadrze komórki: kierunek prawdziwy, długość do ścianki
                    q = dr / Lr * min(Lr, 0.95 * L)
                n_przed = len(lin)
                strzalka(p0, p0 + q, KOL_RUCH, 1.0, 0.25)
                self.n_ol_ruch = len(lin) - n_przed                             # na końcu bufora: rysowana grubo, na wierzchu
        n = len(lin)
        OL = np.zeros((max(n, 1), 2, 4), np.float32)
        for i, (p, kol) in enumerate(lin):
            OL[i, 0, :3] = p; OL[i, 1] = kol
        self.n_ol = n
        self.ustaw_bufor('OL', OL.reshape(-1, 4))

    def _programy(self):
        return dict(o=_program(VS_O, FS_O))

    def ustaw_tryb_pod(self, on):
        """on: ognisko z podkroków przejścia przez sieć (dane podaje zakładka przez ustaw_ognisko); off: oś uczenia (blok 6)."""
        self.tryb_pod = bool(on)
        if on:
            self.K = max(self.K, 2)
        elif self.d is None:
            self.K = 0
        self.update()

    def _rysuj(self, mvp):
        o = self.opcje
        v = self.view()                                                          # piksele na jednostkę sceny (rozmiar plamek światła)
        if v is not None and v.height() > 0:
            o['px'] = v.height() * v.devicePixelRatioF() / (2 * float(v.opts['distance']) * np.tan(np.radians(float(v.opts['fov'])) / 2))
        GL.glUseProgram(self._prog['o'])
        self._ui('o', u_mvp=mvp, u_skala=float(o['skala_chmury']))
        if o.get('tylko_pole', False):                                           # podkroki: sama mgła i strzałki pola w komórce
            if self.d is not None:
                self._rysuj_pole(o)
            return
        uczenie = (not self.tryb_pod) and self.d is not None
        if o.get('tlo_czastki', True) and self.n_op:                             # pozostałe cząsteczki w tle, pod wszystkim
            self._addytywnie(False)
            self._ui('o', u_tryb=10, u_punkty=2, u_rozm=float(o.get('rozm_tla', 5.0)))
            GL.glDrawArrays(GL.GL_POINTS, 0, self.n_op)
        if uczenie:
            self._rysuj_pole(o)
        if o.get('refl', False) and self.n_or_pkt:                               # skala sfery: szkło, wiązki od reflektorów, reflektory
            self._ui('o', u_tryb=8, u_punkty=0)
            if o.get('sfera', 0.10) > 0:
                self._addytywnie(False); GL.glLineWidth(1.0)
                GL.glDrawArrays(GL.GL_LINES, 0, self.n_or_lin)
            self._addytywnie(True); GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, self.n_or_lin, self.n_or_wiaz)
            self._addytywnie(False)
            self._ui('o', u_punkty=2, u_rozm=6.0)
            GL.glDrawArrays(GL.GL_POINTS, self.n_or_lin + self.n_or_wiaz, self.n_or_pkt)
        if o.get('swiatlo', True) and self.n_ob:                                 # światło ogniska: wiązki gaussowskie
            w0 = float(o.get('szer_sw', 0.02)) * self.Lb; rozb = float(o.get('rozb_sw', 0.15))
            zR = w0 / max(rozb, 1e-4); a = float(np.arcsinh(self.Lb / max(zR, 1e-12)))
            M = int(np.clip(np.ceil(2 * a / (rozb * 0.6)), 16, 160))
            self._addytywnie(True)
            self._ui('o', u_tryb=7, u_punkty=1, u_w0=w0, u_rozb=rozb, u_Lb=self.Lb, u_px=float(o.get('px', 10.0)), u_M=M,
                     u_c=(*self.c_ogn, 0), u_jas=float(o.get('jas_sw', 1.0)))
            GL.glDrawArraysInstanced(GL.GL_POINTS, 0, M, self.n_ob)
        if o.get('obrys', False) and self.n_of_tr:                               # obrys ogniska: półprzezroczysta powierzchnia i siatka
            self._ui('o', u_tryb=9, u_punkty=0)
            self._addytywnie(False)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.n_of_tr)
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, self.n_of_tr, self.n_of_kr)
        if self.n_ow:                                                            # wiązki (addytywnie), potem wkłady i strzałki
            self._ui('o', u_tryb=6, u_punkty=0)
            if self.n_wiazek:
                self._addytywnie(True); GL.glLineWidth(1.0)
                GL.glDrawArrays(GL.GL_LINES, 0, self.n_wiazek)
            if self.n_ow > self.n_wiazek:
                self._addytywnie(False); GL.glLineWidth(1.4)
                GL.glDrawArrays(GL.GL_LINES, self.n_wiazek, self.n_ow - self.n_wiazek)
        if uczenie:
            nr = int(getattr(self, 'n_ol_ruch', 0))
            if self.n_ol > nr:
                self._addytywnie(False); GL.glLineWidth(2.0)
                self._ui('o', u_tryb=2, u_punkty=0)
                GL.glDrawArrays(GL.GL_LINES, 0, self.n_ol - nr)
            if nr:                                                               # nadchodzący ruch koloidu: grubo, na wierzchu
                self._addytywnie(False); GL.glLineWidth(4.5)
                self._ui('o', u_tryb=2, u_punkty=0)
                GL.glDrawArrays(GL.GL_LINES, self.n_ol - nr, nr)
                GL.glLineWidth(1.0)
            k = self.k_teraz()
            self._addytywnie(False)                                              # koloid teraz i pułapka (gdy jest zero pola)
            self._ui('o', u_tryb=3, u_wsp=int(o['wsp']), u_slad=1, u_ref=(*self.pulapka(), 0))
            for cel, rozm in ((1, 9.0), (0, 13.0)):
                if cel == 1 and not bool(self.d['zb'][k]) and not o.get('pulapka_zast', False):
                    continue                                                     # zastępcza (najsłabsze pole) tylko na życzenie
                self._ui('o', u_sl=cel, u_punkty=2, u_rozm=rozm)
                GL.glDrawArrays(GL.GL_POINTS, self.kb, 1)
        elif self.n_ob:                                                          # podkroki: kropka cząsteczki z bufora OW
            self._addytywnie(False)
            self._ui('o', u_tryb=6, u_punkty=2, u_rozm=13.0)
            GL.glDrawArrays(GL.GL_POINTS, self.n_ow, 1)
        GL.glLineWidth(1.0)

    def _rysuj_pole(self, o):
        """Oś uczenia: lokalne pole na siatce (mgła, strzałki) i ślady koloidu oraz pułapki przez checkpointy."""
        k = self.k_teraz()
        if self.G > 1:
            sA = (float(self.skala_k[self.ka]), float(self.d['fmed'][self.ka]), 0, 0)
            sB = (float(self.skala_k[self.kb]), float(self.d['fmed'][self.kb]), 0, 0)
            odst = 2.0 * float(self.OT[k, 2, 3]) / max(self.G - 1, 1) * max(int(o['pods']), 1)
            wzgl = int(o['strzalki']) == 2
            typ = (np.median(np.linalg.norm(self.d['pole'][k][:, :3].astype(np.float32) - (self.d['skok'][k] if wzgl else 0), axis=1))
                   * float(self.skala_k[k]))
            self._ui('o', u_G=self.G, u_N=self.N, u_ka=self.ka, u_kb=self.kb, u_ff=self.ff, u_fc=self.fc, u_sA=sA, u_sB=sB,
                     u_skA=(*self.d['skok'][self.ka], 0), u_skB=(*self.d['skok'][self.kb], 0))
            if o['mgla']:
                self._addytywnie(True)
                self._ui('o', u_tryb=0, u_punkty=1, u_ostr=float(o['ostr']), u_tlo=float(o['tlo']), u_jas=float(o['jas']),
                         u_rozm=float(o['rozm']))
                GL.glDrawArraysInstanced(GL.GL_POINTS, 0, self.N, 2)
            if int(o['strzalki']):
                self._addytywnie(False); GL.glLineWidth(1.2)
                self._ui('o', u_tryb=1, u_punkty=0, u_wzgl=int(wzgl), u_pods=max(int(o['pods']), 1), u_jas=1.0,
                         u_plyn=int(bool(o.get('plyn', True))), u_faza=float(o.get('faza', 0.0)),
                         u_strz=float(o['dl_strz']) * odst / max(typ, 1e-12))
                GL.glDrawArraysInstanced(GL.GL_LINES, 0, 6, 2 * self.N)
        else:
            self._ui('o', u_ka=self.ka, u_kb=self.kb, u_ff=self.ff, u_fc=self.fc)
        sl = int(o['slad'])
        if sl:
            k0 = max(0, self.kb - int(o['n_slad']))
            self._addytywnie(False)
            self._ui('o', u_tryb=3, u_wsp=int(o['wsp']), u_slad=int(o['n_slad']), u_ref=(*self.pulapka(), 0))
            for cel in ((0, 1) if sl == 2 else (0,)):
                self._ui('o', u_sl=cel, u_punkty=0); GL.glLineWidth(1.6 if cel == 0 else 1.2)
                GL.glDrawArrays(GL.GL_LINE_STRIP, k0, self.kb - k0 + 1)
                if o['kropki']:                                                  # skoki: położenia w kolejnych checkpointach
                    self._ui('o', u_punkty=2, u_rozm=5.0 if cel == 0 else 4.0)
                    GL.glDrawArrays(GL.GL_POINTS, k0, self.kb - k0 + 1)
