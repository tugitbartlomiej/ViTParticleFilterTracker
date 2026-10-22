"""Czy po włączeniu podkroków (F3) okienka nachodzą na siebie: geometrie doków w oknie."""
import os, sys, time
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt, QTimer, QRect, QPoint
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app
o = app.Okno(); o.resize(1280, 860); o.show()
z = o.zakladka_laserow
def p(n=20):
    for _ in range(n):
        a.processEvents(); time.sleep(0.02)
def geo(w):
    tl = w.mapTo(o, QPoint(0, 0)); return QRect(tl, w.size())
def raport(nap):
    ds = {'panel': z.dok_panel, 'podkroki': z.dok_podkroki}
    ds.update({k: d for k, d in z.doki.items() if z.widac(k)})
    g = {k: geo(d) for k, d in ds.items() if d.isVisible() and not d.isFloating()}
    nach = [(x, y, g[x].intersected(g[y])) for x in g for y in g if x < y and g[x].intersects(g[y]) and (g[x].intersected(g[y]).width() > 2 and g[x].intersected(g[y]).height() > 2)]
    print(nap, 'okno', o.size().width(), o.size().height(), 'podkroki', g.get('podkroki'), 'min', z.dok_podkroki.minimumSizeHint(), 'panel', g.get('panel'))
    print('   nachodzą:', [(x, y, (r.width(), r.height())) for x, y, r in nach], flush=True)
def start():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')); p()
    raport('start')
    z.ch_pod.setChecked(True); p(20); raport('podkroki wł.')
    o.resize(1100, 650); p(20); raport('okno niższe 700')
    o.grab().save(r'C:\Users\bartl\AppData\Local\Temp\claude\W--Studia-PhD-projekt-VIT-ViTParticleFilterTracker\7a3f5b79-c883-4cdf-9559-5f8815a97330\scratchpad\nachodzenie.png')
    a.quit()
QTimer.singleShot(1500, start)
a.exec()
