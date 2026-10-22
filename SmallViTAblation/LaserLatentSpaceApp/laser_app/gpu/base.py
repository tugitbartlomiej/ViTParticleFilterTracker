"""Wspólna część elementów OpenGL 4.3: stałe siatki tokenów i kanałów, kompilacja programów shaderów, palety (LUT) i klasa bazowa _GPUElement (bufory SSBO, wysyłka dużych tablic porcjami w tle)."""
import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

TOK, NK, H, W = 600, 768, 20, 30
LIMIT_ZH = 3 * 2 ** 30          # najwięcej bajtów natężeń Z trzymanych w całości na karcie (zestaw > ~4 zdjęć: para checkpointów)
LUTY = ('plasma', 'magma', 'inferno', 'viridis')
NM = 9                                 # mapy: 0–5 atencja bloków, 6 średnia atencja, 7 centra (lok), 8 drgania (vib); 9 = reflektor (osobno)

_CZAS = """
uniform float u_t;
uniform int u_K;
void para(out uint a, out uint b, out float f) {
    float tt = clamp(u_t, 0.0, float(u_K - 1));
    a = uint(floor(tt)); b = min(a + 1u, uint(u_K - 1)); f = tt - float(a);
}
"""
_LUT = """
uniform sampler2D u_lut;
vec3 lut(int k, float v) { return texture(u_lut, vec2(0.002 + 0.996 * clamp(v, 0.0, 1.0), (float(k) + 0.5) / 4.0)).rgb; }
"""
_KSZTALT = """
uniform int u_ksztalt;              // 0 poświata studni, 1 pierścień, 2 dysk z jasną obwódką, 3 miękka kropka, 4 kwadrat
vec4 ksztalt(vec4 k) {
    vec2 d = gl_PointCoord * 2.0 - 1.0; float r = length(d);
    if (u_ksztalt == 4) return (max(abs(d.x), abs(d.y)) < 0.85) ? k : vec4(0.0);
    if (r > 1.0) discard;
    if (u_ksztalt == 0) {           // jak tonowanie filmu: 0,55·I zielony + 0,25·I³ białozielony (I = gauss)
        float I = k.a * exp(-r * r * 4.5);
        return vec4(0.55 * I * vec3(0.22, 1.0, 0.48) + 0.25 * I * I * I * vec3(0.75, 1.0, 0.85), 1.0);
    }
    if (u_ksztalt == 1) { float a = smoothstep(0.6, 0.78, r) * (1.0 - smoothstep(0.86, 1.0, r)); return vec4(k.rgb, k.a * a); }
    if (u_ksztalt == 2) { vec3 c = r > 0.72 ? vec3(1.0) : k.rgb; return vec4(c, k.a * (1.0 - smoothstep(0.9, 1.0, r))); }
    return vec4(k.rgb, k.a * exp(-r * r * 3.0));
}
"""


def _program(vs, fs):
    return compileProgram(compileShader(vs, GL.GL_VERTEX_SHADER), compileShader(fs, GL.GL_FRAGMENT_SHADER), validate=False)


def lut_rgb():
    import matplotlib
    return np.stack([(matplotlib.colormaps[n](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8) for n in LUTY])  # (4, 256, 3)


class _GPUElement(GLGraphicsItem):
    """Wspólne: bufory SSBO o stałych numerach wiązań (BUFORY), tekstury (LUT, zdjęcie), kompilacja przy pierwszym rysowaniu,
    przywracanie stanu GL po rysowaniu (inne elementy pyqtgraph i napisy QPainter)."""
    BUFORY = ()

    def __init__(self):
        super().__init__()
        self._prog = None
        self._buf = {}
        self._rozm = {}
        self._dane = {}
        self._tex = {}
        self._tex_dane = {}
        self._uni = {}
        self.K = 0
        self.t = 0.0
        self.blad = ''
        self._tex_dane['lut'] = lut_rgb().reshape(4, 256, 3)

    def ustaw_bufor(self, nazwa, a):
        self._dane[nazwa] = np.ascontiguousarray(a)
        self.update()

    # ------------------------------------------------------------------ duże bufory wysyłane porcjami (np. całe Z historii na GPU)
    def wyslij_porcjami(self, nazwa, zrodlo, wierszy=16):
        """Bufor `nazwa` = całe `zrodlo` (tablica albo mmap, pierwszy wymiar = wiersze) na GPU bez zatrzymania okna: wątek czyta
        porcje po `wierszy` z dysku, a każde rysowanie wysyła gotowe porcje (glBufferSubData, do ~8 ms na klatkę) i prosi o
        następną klatkę, aż całość jest na karcie (`duzy_gotowy`). Nowe wywołanie przerywa poprzednie wysyłanie tego bufora."""
        import queue
        import threading
        stary = getattr(self, '_duze', {}).get(nazwa)
        if stary is not None:
            stary['stop'] = True
        n = int(zrodlo.shape[0]); nb = int(np.prod(zrodlo.shape) * zrodlo.dtype.itemsize)
        st = dict(nbytes=nb, kolejka=queue.Queue(maxsize=4), wyslane=0, przydzielony=False, gotowy=False, stop=False)

        def czytaj():
            for a in range(0, n, wierszy):
                if st['stop']:
                    return
                kawalek = np.ascontiguousarray(zrodlo[a:a + wierszy]).reshape(-1).view(np.uint8)
                while not st['stop']:
                    try:
                        st['kolejka'].put(kawalek, timeout=0.2); break
                    except queue.Full:
                        pass
        self.__dict__.setdefault('_duze', {})[nazwa] = st
        threading.Thread(target=czytaj, daemon=True, name=f'wysylka_{nazwa}').start()
        self.update()

    def duzy_gotowy(self, nazwa):
        st = getattr(self, '_duze', {}).get(nazwa)
        return bool(st and st['gotowy'])

    def _wyslij_duze(self):
        import queue
        import time
        t0 = time.perf_counter()
        for nazwa, st in list(getattr(self, '_duze', {}).items()):
            if st['gotowy'] or st['stop']:
                continue
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[nazwa])
            if not st['przydzielony']:
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, st['nbytes'], None, GL.GL_STATIC_DRAW)
                self._rozm[nazwa] = st['nbytes']; st['przydzielony'] = True
            while time.perf_counter() - t0 < 0.008:
                try:
                    k = st['kolejka'].get_nowait()
                except queue.Empty:
                    break
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, st['wyslane'], k.nbytes, k)
                st['wyslane'] += k.nbytes
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            if st['wyslane'] >= st['nbytes']:
                st['gotowy'] = True
            else:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(16, self.update)                        # następna porcja w następnej klatce

    def ustaw_teksture(self, nazwa, rgb):
        self._tex_dane[nazwa] = None if rgb is None else np.ascontiguousarray(rgb, np.uint8)
        self.update()

    def ustaw_czas(self, t):
        self.t = float(t)
        self.update()

    def _programy(self):
        raise NotImplementedError

    def _u(self, prog, nazwa):
        k = (prog, nazwa)
        if k not in self._uni:
            self._uni[k] = GL.glGetUniformLocation(self._prog[prog], nazwa)
        return self._uni[k]

    def _ui(self, prog, **kw):
        for k, v in kw.items():
            loc = self._u(prog, k)
            if loc < 0:
                continue
            if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 4:
                GL.glUniform4f(loc, *[float(x) for x in v])
            elif isinstance(v, (tuple, list, np.ndarray)) and len(v) == 16:
                GL.glUniformMatrix4fv(loc, 1, False, np.asarray(v, np.float32))
            elif isinstance(v, (bool, int, np.integer)):
                GL.glUniform1i(loc, int(v))
            else:
                GL.glUniform1f(loc, float(v))

    def _wyslij(self):
        for n, a in list(self._dane.items()):
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buf[n])
            if self._rozm.get(n, -1) >= a.nbytes and a.nbytes > 0:
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, a.nbytes, a)
            else:
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, max(a.nbytes, 16), a if a.nbytes else None, GL.GL_DYNAMIC_DRAW)
                self._rozm[n] = max(a.nbytes, 16)
        self._dane.clear()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        for n, a in list(self._tex_dane.items()):
            if a is None:
                continue
            if n not in self._tex:
                self._tex[n] = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex[n])
            stare = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, a.shape[1], a.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, a)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, int(stare))
            filtr = GL.GL_LINEAR
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filtr)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filtr)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._tex_dane.clear()

    def paint(self):
        if self.K < 2:
            return
        try:
            if self._prog is None:
                self._prog = self._programy()
                ids = np.atleast_1d(GL.glGenBuffers(len(self.BUFORY)))
                self._buf = {n: int(i) for n, i in zip(self.BUFORY, ids)}
                for n in self.BUFORY:
                    self._dane.setdefault(n, np.zeros((1, 4), np.float32))
            self._wyslij()
            if getattr(self, '_duze', None):
                self._wyslij_duze()
            for i, n in enumerate(self.BUFORY):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, self._buf[n])
            GL.glActiveTexture(GL.GL_TEXTURE1); GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex.get('lut', 0))
            GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex.get('rgb', 0))
            GL.glDisable(GL.GL_DEPTH_TEST); GL.glEnable(GL.GL_BLEND)
            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE); GL.glEnable(GL.GL_POINT_SPRITE)
            GL.glEnable(GL.GL_LINE_SMOOTH)
            self._rysuj(np.asarray(self.mvpMatrix().data(), np.float32))
            self.blad = ''
        except Exception as e:                                          # błąd GL nie zatrzymuje aplikacji
            self.blad = f'{type(e).__name__}: {e}'
        finally:
            GL.glUseProgram(0)
            GL.glLineWidth(1.0)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glActiveTexture(GL.GL_TEXTURE1); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            for i in range(len(self.BUFORY)):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, i, 0)

    @staticmethod
    def _addytywnie(tak):
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE) if tak else GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)


def siatka_sfery(R, co=12, n=96):
    seg = []
    t = np.linspace(0, 2 * np.pi, n + 1)
    for lon in np.radians(np.arange(0, 180, co)):
        p = R * np.stack([np.cos(t) * np.cos(lon), np.cos(t) * np.sin(lon), np.sin(t)], 1); seg += [p[:-1], p[1:]]
    for lat in np.radians(np.arange(-90 + co, 90, co)):
        p = R * np.stack([np.cos(t) * np.cos(lat), np.sin(t) * np.cos(lat), np.full_like(t, np.sin(lat))], 1); seg += [p[:-1], p[1:]]
    a, b = np.concatenate(seg[0::2]), np.concatenate(seg[1::2])
    X = np.zeros((2 * len(a), 4), np.float32); X[0::2, :3] = a; X[1::2, :3] = b
    return X


def przenikanie(s, f):
    """Przejście uwaga ↔ MLP między podkrokami: (m, bm, ba) = waga MLP 0..1 (uwaga ma 1 − m), blok MLP i blok uwagi do
    narysowania. f = szerokość przenikania w podkrokach: 0 = osobne kroki (skokowo, jak dawniej), 1 = przenikanie przez cały
    podkrok. Zmiana bloku MLP wypada w połowie podkroku uwagi, a bloku uwagi w połowie podkroku MLP (tam ich waga jest 0)."""
    s = float(np.clip(s, 0, 12)); k = int(np.clip(np.floor(s), 0, 11)); u = s - k; b = k // 2
    m = float(k % 2)
    f = float(np.clip(f, 0, 1))
    if f <= 0:
        return m, b, b
    n = int(round(s))
    if 1 <= n <= 11 and abs(s - n) < f / 2:
        t = (s - n + f / 2) / f; t = t * t * (3 - 2 * t)
        lewo = float((n - 1) % 2); m = lewo * (1 - t) + (1 - lewo) * t
    if k % 2:
        return m, b, (b if u < 0.5 else min(b + 1, 5))
    return m, (max(b - 1, 0) if u < 0.5 else b), b


def reflektory_przejscia(d, s, f):
    """Reflektory STOJĄ: 768 stałych miejsc na zewnętrznej sferze (spirala Fibonacciego), kolor należy do miejsca. Zmienia się tylko,
    który neuron bloku świeci z którego miejsca (slot[b], przypisanie węgierskie), czyli pod jakim kątem jego wiązka wchodzi
    w szklaną sferę (punkt załamania e, dalej wzdłuż kierunku pchnięcia d_c do cząsteczki). Zwraca kierunki (768, 3) i kolory
    (768, 3) miejsc w kolejności neuronów bloku MLP rysowanego w chwili s (`przenikanie`) oraz klucz zmiany."""
    from laser_app.compute.common import kierunki, kolory_kanalow
    _, bm, _ = przenikanie(s, f)
    sl = d['slot'][bm]
    return kierunki(False)[sl], kolory_kanalow()[:, :3][sl], (id(d), bm)
