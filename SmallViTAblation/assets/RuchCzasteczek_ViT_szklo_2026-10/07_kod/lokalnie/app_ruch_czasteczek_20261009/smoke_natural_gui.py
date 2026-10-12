"""Native GUI test of saved-training checkpoint history and independent training logs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/app_ruch_czasteczek_20261009'
OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--timeline')
args = parser.parse_args()
os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
os.environ['EDYTOR_BEZ_AUTOSTARTU'] = '1'
sys.path.insert(0, str(ROOT / 'app'))
from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
qapp = QApplication([])
import app
from styl import STYL
app.WatekModelu.start = lambda self: None
window = app.Okno()
qapp.setStyleSheet(STYL)
window.resize(1600, 1100)
tab = window.zakladka_ruchu
window.zakladki_glowne.setCurrentWidget(tab)
window.show()
tab.opcje.zakladki.setCurrentIndex(2)
tab.btn_scan_natural.click()
started = time.monotonic()
proof = {'status': 'running'}
stage = 0


def finish(error=None):
    proof.update(status='pass' if error is None else 'fail', error=error)
    (OUT / 'SMOKE_NATURAL_GUI.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False), flush=True)
    timer.stop(); window.close(); qapp.exit(0 if error is None else 1)


def check():
    global stage
    try:
        if time.monotonic() - started > 180:
            raise TimeoutError(tab.status.text())
        if tab.cloud.error:
            raise RuntimeError(tab.cloud.error)
        if stage == 0 and tab.watek is None:
            assert '101 checkpointów' in tab.status.text(), tab.status.text()
            proof['discovered_history'] = tab.status.text()
            tab.btn_load_logs.click(); stage = 1
        elif stage == 1 and tab.watek is None:
            assert tab._training_logs['series']['loss']['steps'] == list(range(50, 1001, 50))
            assert tab._training_logs['series']['map50']['steps'] == list(range(0, 1001, 100))
            proof['loss_records'] = 20; proof['map_records'] = 11
            if args.timeline:
                tab.edit_timeline.setText(args.timeline); tab.btn_open.click(); stage = 3
            else:
                tab.spin_natural_images.setValue(4); tab.spin_natural_end.setValue(10)
                tab.btn_compute_natural.click(); stage = 2
        elif stage == 2 and tab.watek is None:
            path = Path(tab.edit_timeline.text())
            meta = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
            assert meta['status'] == 'complete', tab.status.text()
            assert meta['inference_only'] and meta['weights_unchanged']
            proof['computed_path'] = str(path)
            tab.btn_open.click(); stage = 3
        elif stage == 3 and tab.watek is None and tab.timeline:
            assert tab.timeline.metadata['time_axis'] == 'training_checkpoint'
            assert 'bez podmiany aktywacji' in tab.metrics.text()
            proof['steps'] = tab.timeline.steps
            proof['point_count'] = tab.timeline.count
            tab._seek_index(len(tab.timeline.steps) - 1); stage = 4
        elif stage == 4 and tab.cloud.draw_count == tab.timeline.count and tab._pair == (tab.timeline.steps[-1], tab.timeline.steps[-1]):
            assert tab.cloud.renderer and tab._visual_step == tab.timeline.steps[-1]
            proof['renderer'] = tab.cloud.renderer
            proof['last_measured_step'] = tab._visual_step
            proof['metrics'] = tab.timeline.stats(tab.timeline.steps[-1])
            proof['loaded_log_map50'] = tab._training_logs['series']['map50']['values'][-1]
            window.grab().save(str(OUT / 'UCZENIE_NATURALNE.png'))
            finish()
    except Exception as error:
        import traceback
        traceback.print_exc()
        finish(f'{type(error).__name__}: {error}')


timer = QTimer()
timer.timeout.connect(check)
timer.start(100)
sys.exit(qapp.exec())
