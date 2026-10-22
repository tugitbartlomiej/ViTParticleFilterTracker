"""Błąd zwijania grup w okienku ustawień sfery: które widżety zwiniętych wierszy są nadal widoczne i kto je pokazuje."""
import os, sys, time
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
import laser_app  # noqa
from PyQt6.QtCore import QCoreApplication, Qt, QTimer, QPoint
from PyQt6.QtWidgets import QApplication, QFormLayout
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
from laser_app.main_window import MainWindow
from laser_app.ui.settings_panel import SettingsPopup
from laser_app.ui.parameter_groups import GroupHeader
o = MainWindow(); o.resize(1500, 900); o.show(); z = o.zakladka_laserow
def p(n=10):
    for _ in range(n): a.processEvents(); time.sleep(0.02)
def start():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')); p()
    z.menu_okienka('s', QPoint(300, 200)); p()
    pp = SettingsPopup.otwarte[-1]
    pp.zwin_wszystko(True); p(10)
    pp.grab().save(os.path.join(OUT, 'zwiniete_sfera.png'))
    sek = z._sek['sfera']
    for g in sek.findChildren(GroupHeader):
        f = g.formularz
        for rr in g._wiersze():
            for rola in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole, QFormLayout.ItemRole.SpanningRole):
                el = f.itemAt(rr, rola)
                if el is None: continue
                w = el.widget()
                ws = [w] if w is not None else ([el.layout().itemAt(i).widget() for i in range(el.layout().count())] if el.layout() else [])
                for w_ in ws:
                    if w_ is not None and w_.isVisible():
                        print('WIDOCZNY w zwiniętym wierszu', g.tytul_grupy, rr, type(w_).__name__, getattr(w_, 'text', lambda: '')()[:30], 'rowVisible', f.isRowVisible(rr), flush=True)
    a.quit()
QTimer.singleShot(1500, start); a.exec()
