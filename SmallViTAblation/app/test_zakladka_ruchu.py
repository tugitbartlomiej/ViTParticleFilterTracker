"""Testy sterowania: pauza i wybór stanu nie mogą zmieniać rzeczywistych danych.

Bez okna (bez kontekstu GL): sprawdzają, jakie parametry zakładka przekazuje karcie (ruch_gpu.ChmuraRuchuGPU),
opisy w dymkach, oznaczenie stanów niezapisanych i brak wczytywania czegokolwiek przy tworzeniu zakładki.
Zgodność obliczeń karty z numpy sprawdza test_ruch_gpu.py.
"""
import os
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QSlider)

os.environ.setdefault('EDYTOR_BEZ_USTAWIEN', '1')
os.environ.setdefault('EDYTOR_BEZ_AUTOSTARTU', '1')

from zakladka_ruchu import ZakladkaRuchu
import ustawienia


class PrzykladowyTor:
    steps = [0, 5, 10]
    image_count = 2
    count = 1200
    metadata = {'norm_scale': 2.0}
    checkpoint_sha256 = 'abc'
    image_names = ['pierwszy.png', 'drugi.png']

    def frame(self, step):
        a = np.zeros((1200, 4), np.float32)
        a[:, 0] = np.arange(1200) / 1200 + step
        a[:, 3] = step
        return a

    def stats(self, step):
        return {'confidence': .5 + step / 100, 'iou': .4 + step / 100, 'both90': 0}

    def bounds(self, movement):
        return np.array([[-1, 12], [-1, 1], [-1, 1]], np.float32)

    def curves(self):
        return {'steps': self.steps, 'confidence': [.5, .55, .6], 'iou': [.4, .45, .5]}


class SterowanieTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tab = ZakladkaRuchu()
        self.tab.timer.stop()
        self.tor = PrzykladowyTor()
        self.tab._loaded((self.tor, self.tor.frame(0), self.tor.frame(5)))

    def tearDown(self):
        self.assertTrue(self.tab.shutdown())
        self.tab.deleteLater()
        self.app.processEvents()

    def test_pause_preserves_fraction_and_data(self):
        original = self.tor.frame(5)
        self.tab._visual_step = 2.5
        self.tab._render()
        self.tab._playing = True
        self.tab._pause()
        self.assertEqual(self.tab.cloud.s, .5)
        self.assertEqual((self.tab.cloud.a, self.tab.cloud.b), (0, 5))
        self.assertEqual(self.tab._visual_step, 2.5)
        self.assertFalse(self.tab._playing)
        np.testing.assert_array_equal(self.tab.cloud._gotowe[5], original)

    def test_disable_interpolation_keeps_measured_state(self):
        self.tab._visual_step = 3
        self.tab._render()
        self.assertAlmostEqual(self.tab.cloud.s, .6)
        self.assertIn('NIEZAPISANY', self.tab.frame_label.text())
        self.tab.chk_smooth.setChecked(False)
        self.assertEqual(self.tab.cloud.s, 0)
        self.assertIn('stanu 0', self.tab.metrics.text())
        self.assertNotIn('NIEZAPISANY', self.tab.frame_label.text())
        self.assertIn('Zapisany stan 0', self.tab.frame_label.text())

    def test_only_parameters_change_per_frame(self):
        """Klatka w obrębie pary nie wysyła danych cząsteczek: tylko liczby dla karty."""
        self.tab._visual_step = 1.0; self.tab._render()
        queued = dict(self.tab.cloud._gotowe)
        for step in (1.5, 2.0, 4.9):
            self.tab._visual_step = step; self.tab._render()
            self.assertAlmostEqual(self.tab.cloud.s, step / 5)
        self.assertEqual(set(self.tab.cloud._gotowe), set(queued))
        self.assertTrue(all(self.tab.cloud._gotowe[k] is queued[k] for k in queued))

    def test_unloaded_state_is_not_skipped(self):
        """Odtwarzanie zatrzymuje się na zapisie, którego jeszcze nie ma na karcie, i przechodzi dalej po wczytaniu."""
        self.tab._visual_step = 4.0
        self.tab._playing = True
        self.tab._advance(1.0)                          # 50 kroków/s: do 10 byłoby 6 kroków
        self.assertEqual(self.tab._visual_step, 5.0)
        self.assertFalse(self.tab._render())
        self.assertIn('Wczytywanie', self.tab.frame_label.text())
        self.tab.cloud.wczytaj_teraz([10])
        self.tab._advance(.05)
        self.assertAlmostEqual(self.tab._visual_step, 7.5)
        self.assertTrue(self.tab._render())
        self.tab._advance(1.0)                          # koniec osi czasu: pauza na ostatnim zapisie
        self.assertEqual(self.tab._visual_step, 10.0)
        self.assertFalse(self.tab._playing)

    def test_no_loading_or_timer_at_construction(self):
        with patch('zakladka_ruchu.R.Timeline') as timeline, patch.object(ZakladkaRuchu, '_load_timeline') as load,                 patch.object(ZakladkaRuchu, '_latest_natural') as latest:
            tab = ZakladkaRuchu()
            self.app.processEvents()
            timeline.assert_not_called(); load.assert_not_called(); latest.assert_not_called()
            self.assertIsNone(tab.timeline)
            self.assertIsNone(tab.cloud.timeline)
            self.assertFalse(tab.timer.isActive())
            self.assertTrue(tab.shutdown())
            tab.deleteLater()

    def test_restoring_settings_loads_nothing(self):
        with patch('zakladka_ruchu.R.Timeline') as timeline, patch.object(ZakladkaRuchu, '_job') as job:
            tab = ZakladkaRuchu()
            saved = ustawienia.stan(self.tab)
            saved['edit_timeline'] = 'F:/jakis/zapis'
            ustawienia.ustaw(tab, saved)
            self.app.processEvents()
            timeline.assert_not_called(); job.assert_not_called()
            self.assertIsNone(tab.timeline)
            self.assertFalse(tab.timer.isActive())
            self.assertTrue(tab.shutdown())
            tab.deleteLater()

    def test_every_control_has_specific_tooltip(self):
        tab = self.tab
        controls = [w for t in (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit, QSlider)
                    for w in tab.findChildren(t)
                    if type(w).__module__.startswith('PyQt6') and w.window() is tab and not w.parent().inherits('QTabBar')
                    and w.objectName() not in ('qt_spinbox_lineedit',) and not isinstance(w.parent(), QAbstractSpinBox)
                    and not isinstance(w.parent(), QComboBox)]
        own = {id(w) for name, w in vars(tab).items() if isinstance(w, (QAbstractButton, QAbstractSpinBox, QComboBox,
                                                                       QLineEdit, QSlider))}
        own.add(id(tab.opcje.btn_opcje))
        checked = [w for w in controls if id(w) in own]
        missing = [w.text() if hasattr(w, 'text') and callable(w.text) else type(w).__name__
                   for w in checked if len(w.toolTip().strip()) < 15]
        self.assertEqual(missing, [])
        self.assertGreaterEqual(len(checked), 50)
        self.assertEqual(len(checked), len(own))

    def test_scope_exact_token_identity(self):
        self.tab.combo_scope.setCurrentIndex(1)
        self.tab.spin_image.setValue(1)
        self.assertEqual(self.tab.cloud.first, 600)
        self.assertEqual(self.tab.cloud.count, 600)
        self.assertEqual(self.tab.image_name.text(), 'drugi.png')

    def test_seek_and_configuration_roundtrip(self):
        self.tab._seek_index(1)
        self.assertEqual(self.tab._visual_step, 5)
        self.tab.spin_tempo.setValue(130)
        self.tab.edit_checkpoint.setText('C:/inne/wagi.pt')
        saved = ustawienia.stan(self.tab)
        self.tab.spin_tempo.setValue(20)
        self.tab.edit_checkpoint.setText('zmienione')
        ustawienia.ustaw(self.tab, saved)
        self.assertEqual(self.tab.spin_tempo.value(), 130)
        self.assertEqual(self.tab.edit_checkpoint.text(), 'C:/inne/wagi.pt')
        self.assertNotIn('suwak', saved)

    def test_checkpoint_mismatch_visible(self):
        self.tab._active_checkpoint = {'sha256': 'inny'}
        self.tab._provenance()
        self.assertIn('innego checkpointu', self.tab.status.text())

    def test_late_finished_cannot_clear_new_worker(self):
        old, active = Mock(), Mock()
        self.tab.watek = active
        with patch.object(self.tab, 'sender', return_value=old):
            self.tab._job_finished()
            self.tab._job_done('old result')
        self.assertIs(self.tab.watek, active)
        old.deleteLater.assert_called_once()
        old.on_done.assert_not_called()
        self.tab.watek = None

    def test_natural_history_labels_and_logs_are_distinct(self):
        self.tor.metadata = {'norm_scale': 2.0, 'time_axis': 'training_checkpoint',
                             'training_logs': {'series': {'loss': {'steps': [0, 10], 'values': [3.0, 1.0]},
                                                          'map50': {'steps': [0, 10], 'values': [.1, .2]}},
                                               'files': [{'path': 'LOG.json'}]}}
        self.tab._loaded((self.tor, self.tor.frame(0), self.tor.frame(5)))
        self.assertIn('Naturalny', self.tab.info.text())
        self.assertIn('bez podmiany', self.tab.metrics.text())
        self.assertIn('checkpointy 0–10', self.tab.status.text())
        x, y = self.tab.log_line.getData()
        np.testing.assert_array_equal(x, [0, 10])
        np.testing.assert_array_equal(y, [3, 1])
        self.tab.combo_log_metric.setCurrentIndex(1)
        np.testing.assert_array_equal(self.tab.log_line.getData()[1], [.1, .2])
        self.tor.metadata = {'norm_scale': 2.0}
        self.tab._loaded((self.tor, self.tor.frame(0), self.tor.frame(5)))
        self.assertTrue(self.tab.log_box.isHidden())

    def test_natural_configuration_roundtrip_and_actual_steps(self):
        self.tab.edit_checkpoint_dir.setText('F:/historia')
        self.tab.edit_training_logs.setText('F:/historia/LOG.json')
        self.tab.spin_natural_start.setValue(100)
        self.tab.spin_natural_stride.setValue(20)
        self.tab.spin_natural_batch.setValue(4)
        config = self.tab._natural_config()
        self.assertEqual(config.start_step, 100)
        self.assertEqual(config.checkpoint_stride, 20)
        self.assertEqual(config.batch_size, 4)
        saved = ustawienia.stan(self.tab)
        self.tab.edit_checkpoint_dir.setText('inne')
        self.tab.spin_natural_stride.setValue(10)
        ustawienia.ustaw(self.tab, saved)
        self.assertEqual(self.tab.edit_checkpoint_dir.text(), 'F:/historia')
        self.assertEqual(self.tab.spin_natural_stride.value(), 20)

    def test_natural_open_prefers_complete_but_can_show_partial(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            complete, partial = root / 'complete', root / 'partial'
            for path, status in ((complete, 'complete'), (partial, 'running')):
                path.mkdir()
                (path / 'manifest.json').write_text(json.dumps({'status': status,
                    'time_axis': 'training_checkpoint', 'image_count': 2, 'steps': [0, 10]}), encoding='utf-8')
            self.tab.spin_natural_images.setValue(2)
            with patch('zakladka_ruchu.N.OUTPUT_ROOT', root), patch.object(self.tab, '_load_timeline') as load:
                self.tab._latest_natural()
                self.assertEqual(self.tab.edit_timeline.text(), str(complete))
                (complete / 'manifest.json').unlink()
                self.tab._latest_natural()
                self.assertEqual(self.tab.edit_timeline.text(), str(partial))
                self.assertEqual(load.call_count, 2)

    def test_background_measurement_is_not_duplicated(self):
        with patch.object(self.tab, '_existing_natural_measurement', return_value=True), \
                patch.object(self.tab, '_job') as job:
            self.tab._compute_natural()
            job.assert_not_called()
            self.assertIn('już pracuje', self.tab.status.text())


if __name__ == '__main__':
    unittest.main()
