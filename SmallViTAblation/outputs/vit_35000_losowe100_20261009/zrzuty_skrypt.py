"""Edytor osi Koopmana na prawdziwym pomiarze: tabela, sortowanie, filtry, szybkie przypisanie."""
import os
import sys
import time

APP = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/app'
OUT = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP); sys.path.insert(0, APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
q = QApplication(sys.argv)
from zakladka_ruchu import ZakladkaRuchu  # noqa: E402

z = ZakladkaRuchu(None); z.resize(1500, 950); z.show(); q.processEvents()
z.edit_timeline.setText(r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd')
def czekaj(warunek, limit):
    t = time.perf_counter()
    while not warunek() and time.perf_counter() - t < limit:
        q.processEvents(); time.sleep(.01)
z._load_timeline(); czekaj(lambda: z.timeline is not None, 60)
z.cloud.wczytaj_teraz(z.timeline.steps); czekaj(lambda: z.cloud.na_karcie() == z.cloud.T, 60)
print('stany', z.cloud.T, 'cząsteczek', z.cloud.N, flush=True)
z.combo_scope.setCurrentIndex(0); z.chk_smooth.setChecked(True)
z.combo_wyg.setCurrentIndex(z.combo_wyg.findData(2)); z.spin_wyg_alfa.setValue(.4)
z.spin_opacity.setValue(.35); z.spin_size.setValue(2.0); z.spin_trail.setValue(4)
z.combo_color.setCurrentIndex(z.combo_color.findData(4)); z.spin_moc.setValue(1.3)
z.btn_panel.setChecked(True)
z.view.setCameraPosition(distance=3.6, elevation=22, azimuth=40)
for uklad, nazwa in ((0, 'pca'), (1, 'siatka')):
    z.combo_uklad.setCurrentIndex(uklad); czekaj(lambda: False, 1)
    for idx in (0, 1, 5, 21):
        z._seek_index(idx); czekaj(lambda: False, 1.2)
        z.view.grabFramebuffer().save(os.path.join(OUT, f'traj100_{nazwa}_e{idx:02d}.png'))
    # stan niezapisany: połowa drogi epoka 10 -> 11 (interpolacja na karcie)
    z._seek_index(10); a, b = z.timeline.steps[10], z.timeline.steps[11]
    z._visual_step = (a + b) / 2; z._render(); czekaj(lambda: False, 1.2)
    z.view.grabFramebuffer().save(os.path.join(OUT, f'traj100_{nazwa}_e10_5.png'))
print('błąd GPU:', repr(z.cloud.blad), flush=True)
print(z.frame_label.text(), flush=True)
os._exit(0)
