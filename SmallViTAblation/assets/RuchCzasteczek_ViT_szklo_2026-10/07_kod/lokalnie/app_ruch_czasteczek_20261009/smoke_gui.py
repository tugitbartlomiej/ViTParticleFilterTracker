"""Sprawdza rzeczywistą nową zakładkę i shader na lokalnym GPU, bez uruchamiania obliczeń 1500 obrazów."""
import json
import os
import sys
import time
import faulthandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/app_ruch_czasteczek_20261009'
OUT.mkdir(parents=True, exist_ok=True)
faulthandler.dump_traceback_later(45, repeat=True)
print('Import bibliotek GUI…', flush=True)
os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
os.environ['EDYTOR_BEZ_AUTOSTARTU'] = '1'
sys.path.insert(0, str(ROOT / 'app'))

from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
qapp = QApplication([])
import app
print('Tworzenie pełnego okna…', flush=True)
from styl import STYL

# Obliczenia starego SD nie są częścią tego testu. Nie zmieniamy aplikacji ani jej modelu.
app.WatekModelu.start = lambda self: None
window = app.Okno()
print('Pełne okno utworzone.', flush=True)
qapp.setStyleSheet(STYL)
window.resize(1600, 1050)
tab = window.zakladka_ruchu
window.zakladki_glowne.setCurrentWidget(tab)
window.show()
tab.btn_load_checkpoint.click()
started = time.perf_counter()
stage = -1
proof = {'status': 'running', 'tabs': [window.zakladki_glowne.tabText(i) for i in range(window.zakladki_glowne.count())]}


def finish(error=None):
    proof['status'] = 'fail' if error is not None else 'pass'
    proof['error'] = error
    (OUT / 'SMOKE_GUI.json').write_text(json.dumps(proof, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False), flush=True)
    timer.stop()
    window.close()
    qapp.exit(1 if error is not None else 0)


def check():
    global stage
    try:
        if time.perf_counter() - started > 90:
            finish('timeout: ' + tab.status.text()); return
        if tab.cloud.error:
            finish(tab.cloud.error); return
        if stage == -1 and tab._active_checkpoint and tab.watek is None:
            proof['loaded_checkpoint'] = tab._active_checkpoint
            tab.btn_history.click(); stage = 0
        elif stage == 0 and tab.timeline and tab.cloud.draw_count == 900000 and tab.watek is None:
            tab.opcje.zakladki.setCurrentIndex(3)
            tab.spin_tempo.setValue(30)
            tab._toggle_play()
            stage = 1
        elif stage == 1 and 30 < tab._visual_step < 200 and .01 < tab.cloud.blend < .99:
            tab.btn_play.click()
            proof['pause_step'] = tab._visual_step
            proof['pause_blend'] = tab.cloud.blend
            proof['renderer'] = tab.cloud.renderer
            proof['count'] = tab.cloud.draw_count
            proof['cpu_frame_cache_bytes'] = sum(a.nbytes for a in tab._cache.values())
            window.grab().save(str(OUT / 'ZAKLADKA_RUCHU.png'))
            tab.chk_smooth.setChecked(False)
            assert tab.cloud.blend == 0
            tab.chk_smooth.setChecked(True)
            assert tab._visual_step == proof['pause_step']
            tab.spin_tempo.setValue(500)
            tab._toggle_play()
            stage = 2
        elif stage == 2 and tab._visual_step == 500 and not tab._playing:
            proof['endpoint'] = tab._visual_step
            proof['count_endpoint'] = tab.cloud.draw_count
            proof['status_text'] = tab.status.text()
            proof['gpu_status'] = tab.gpu_status.text()
            proof['metadata_checkpoint_sha256'] = tab.timeline.checkpoint_sha256
            assert tab._render()
            assert tab.cloud.blend == 0
            assert tab._pair == (500, 500)
            tab.view.makeCurrent()
            from OpenGL import GL
            proof['gl_error'] = int(GL.glGetError())
            tab.view.doneCurrent()
            assert proof['gl_error'] == 0
            tab.combo_scope.setCurrentIndex(1); tab.spin_image.setValue(17)
            assert tab.cloud.first == 17 * 600 and tab.cloud.count == 600
            proof['single_image_identity'] = {'first': tab.cloud.first, 'count': tab.cloud.count}
            tab.combo_scope.setCurrentIndex(0)
            tab.opcje.zakladki.setCurrentIndex(0)
            window.grab().save(str(OUT / 'CHECKPOINT_I_DANE.png'))
            proof['warm_start'] = time.perf_counter(); stage = 3
        elif stage == 3 and time.perf_counter() - proof['warm_start'] > 3:
            proof['warm_native_fps'] = tab.cloud.fps
            tab.opcje.zakladki.setCurrentIndex(3)
            window.grab().save(str(OUT / 'ZAKLADKA_RUCHU.png'))
            tab.spin_images.setValue(1); tab.spin_steps.setValue(2); tab.spin_save.setValue(1)
            tab.spin_conf.setValue(.999); tab.spin_iou.setValue(.999)
            tab.btn_compute.click(); stage = 4
        elif stage == 4 and tab.watek is None:
            result = Path(tab.edit_timeline.text())
            native = json.loads((result / 'manifest.json').read_text(encoding='utf-8'))
            assert native['status'] == 'complete'
            assert native['checkpoint_sha256'] == proof['loaded_checkpoint']['sha256']
            assert native['weights_unchanged']
            proof['actual_checkpoint_two_step'] = {'path': str(result), 'status': native['status'],
                                                   'steps': native['steps'], 'weights_unchanged': native['weights_unchanged']}
            tab.btn_open.click(); stage = 5
        elif stage == 5 and tab.watek is None and tab.timeline.count == 600:
            proof['fresh_new_run_frame'] = list(tab.timeline.frame(2).shape)
            tab.opcje.zakladki.setCurrentIndex(1)
            window.grab().save(str(OUT / 'NOWY_POMIAR.png'))
            finish()
    except Exception as error:
        import traceback
        traceback.print_exc()
        finish(f'{type(error).__name__}: {error}')


timer = QTimer()
timer.timeout.connect(check)
timer.start(100)
sys.exit(qapp.exec())
