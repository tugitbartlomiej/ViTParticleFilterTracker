"""Mapy atencji liczone wsadowo (atencja_zbioru.py) == mapy panelu OpcjeAtencji z Uczenie ViT (jeden obraz).

Potrzebuje prawdziwego checkpointu gęstej historii i zdjęcia z manifestu; bez nich test jest pomijany.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POMIAR = Path(__file__).resolve().parents[2] / 'outputs/ruch_czasteczek_gui/20261009T080332_natural_0051d7971abb_q16'


@pytest.mark.skipif(not (POMIAR / 'manifest.json').is_file(), reason='brak lokalnego pomiaru naturalnego')
def test_mapy_wsadowe_zgodne_z_panelem():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)          # referencja: inaczej Qt usuwa aplikację
    import torch
    import atencja_zbioru as AZ
    import detekcja as D
    import ruch_zbioru as R
    from opcje_atencji import OpcjeAtencji
    m = json.loads((POMIAR / 'manifest.json').read_text(encoding='utf-8'))
    ck, img = m['checkpoints'][50]['path'], m['resolved_image_paths'][440]
    if not (Path(ck).is_file() and Path(img).is_file()):
        pytest.skip('brak checkpointu albo zdjęcia')
    rgb = R._private_module('dane').decode_resize(Path(img).read_bytes())
    det = D.Detektor(ck)
    op = OpcjeAtencji(); op.ustaw_zrodlo(lambda b: det.atencja(rgb, b))
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().to(det.dev) / 255
    for mapa in range(6):
        for blok, rollout in ((6, 0), (3, 1)):
            for glowa in (0, 1, 3):
                for skala in (0, 1):
                    op.combo_mapa.setCurrentIndex(mapa); op.spin_blok.setValue(blok); op.combo_bloki.setCurrentIndex(rollout)
                    op.combo_glowa.setCurrentIndex(glowa); op.combo_skala.setCurrentIndex(skala); op._cache = {}
                    tok = 255 if mapa in (1, 2) else None
                    ref, _ = op.mapa(tok)
                    v = AZ.mapy(det.net, x, AZ.ustawienia(op), tok)
                    np.testing.assert_allclose(v[0], ref.reshape(-1), atol=1e-4)
    assert app is not None
