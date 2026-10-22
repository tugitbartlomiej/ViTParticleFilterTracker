"""Czas rysowania każdego widocznego widoku OpenGL (repaint + glFinish) i stan wysyłki całego Z na GPU."""
import os
import sys
import time

APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
import numpy as np  # noqa: E402
from PyQt6.QtCore import QCoreApplication, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app  # noqa: E402
from OpenGL import GL  # noqa: E402

o = app.Okno(); o.show()
z = o.zakladka_laserow


def nazwa(v):
    for k, d in z.doki.items():
        if d.isAncestorOf(v):
            return k
    return '?'


def mierz():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', os.environ.get('HIST', '2cca2743e1c74b14')))
    t0 = time.perf_counter()
    while not z.es.duzy_gotowy('ZH') and time.perf_counter() - t0 < 60:
        z.vs.update(); a.processEvents(); time.sleep(0.005)
    st = z.es._duze['ZH']
    print(f'ZH na GPU: {z.es.duzy_gotowy("ZH")} po {time.perf_counter() - t0:.1f} s ({st["wyslane"] / 2**20:.0f} MB)', flush=True)
    if z.es.duzy_gotowy('ZH'):                                       # odczyt z karty = Z z dysku?
        z.vs.makeCurrent()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, z.es._buf['ZH'])
        h = z.h; bl = []
        for k in (0, h.K // 2, h.K - 1):
            off = k * 600 * 768 * 2
            b = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, off, 600 * 768 * 2)
            bl.append(bool(np.array_equal(np.frombuffer(bytes(b), np.float16).reshape(600, 768), np.asarray(h.Z[k]))))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0); z.vs.doneCurrent()
        print('ZH zgodne z Z (k = 0, K/2, K−1):', bl, flush=True)
    from widok_gl import WidokGL
    for nazwa_w, ust in (('domyślnie', {}), ('bez wiązek', {'wiazki': 0}), ('wiązki wycięte progiem', {'prog': 99.0}),
                         ('sfera bez wiązek i reflektorów', {'wiazki': 0, 'reflektory': False, 'sfera': 0.0})):
        stare = dict(z.es.opcje); z.es.opcje.update(ust)
        cz = []
        for i in range(30):
            z.es.ustaw_czas(z.h.K * 0.4 + i * 0.37); z.es.ustaw_natezenia(z.h, z.h.K * 0.4 + i * 0.37)
            c0 = time.perf_counter(); z.vs.repaint(); z.vs.makeCurrent(); GL.glFinish(); z.vs.doneCurrent(); cz.append(time.perf_counter() - c0)
        z.es.opcje.clear(); z.es.opcje.update(stare)
        print(f'  sfera [{nazwa_w}]: mediana {np.median(cz) * 1e3:.1f} ms', flush=True)
    print('  rozmiar sfery px:', z.vs.width(), z.vs.height(), 'dpr', z.vs.devicePixelRatio(), flush=True)
    el = {}
    for it in z.vs.items:                                               # czas paint() każdego elementu sceny sfery (z glFinish)
        stara = it.paint

        def owin(stara=stara, it=it):
            GL.glFinish(); c0 = time.perf_counter(); stara(); GL.glFinish()
            el.setdefault(type(it).__name__ + f'#{id(it) % 1000}', []).append(time.perf_counter() - c0)
        it.paint = owin
    for i in range(30):
        z.ustaw_czas(z.h.K * 0.45 + i * 0.37); z.vs.repaint(); a.processEvents()
    for k, c in el.items():
        print(f'    element {k}: mediana {np.median(c) * 1e3:.1f} ms, maks {np.max(c) * 1e3:.1f} ms', flush=True)
    import cProfile, pstats, io
    pr = cProfile.Profile()
    for i in range(20):
        z.ustaw_czas(z.h.K * 0.5 + i * 0.37); pr.enable(); z.vs.repaint(); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(12); print(s.getvalue()[:2500], flush=True)
    czasy = {}
    for i in range(40):
        z.ustaw_czas(z.h.K * 0.4 + i * 0.37)
        for v in list(WidokGL._instancje):
            if not (v.isVisible() and not v.visibleRegion().isEmpty()):
                continue
            c0 = time.perf_counter(); v.repaint(); v.makeCurrent(); GL.glFinish(); v.doneCurrent()
            czasy.setdefault(nazwa(v), []).append(time.perf_counter() - c0)
        a.processEvents()
    for k, c in sorted(czasy.items(), key=lambda x: -np.median(x[1])):
        print(f'  okienko {k:>4}: mediana {np.median(c) * 1e3:6.1f} ms, p90 {np.percentile(c, 90) * 1e3:6.1f} ms', flush=True)
    a.quit()


QTimer.singleShot(1500, mierz)
a.exec()
