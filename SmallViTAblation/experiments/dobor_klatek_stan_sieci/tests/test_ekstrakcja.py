"""CPU checks for label-free detector-state extraction and cache provenance."""
import importlib
import json

import numpy as np
import pytest


def module():
    return importlib.import_module(
        'SmallViTAblation.experiments.dobor_klatek_stan_sieci.ekstrakcja')


def test_extracts_predicted_region_and_keeps_tiny_box_local():
    ex = module()
    fmap = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    np.testing.assert_allclose(ex.pool_predicted_roi(fmap, [[.25, .25, .5, .5]]), [[2.5]])
    # No patch centre lies in this box: use its nearest patch, never the entire scene.
    np.testing.assert_allclose(ex.pool_predicted_roi(fmap, [[.98, .98, .001, .001]]), [[15]])
    with pytest.raises(ValueError, match='positive'):
        ex.pool_predicted_roi(fmap, [[.5, .5, 0, .2]])


def test_prediction_selection_preserves_box_and_score_alignment():
    ex = module()
    boxes = np.array([[[.2, .3, .1, .1], [.7, .6, .2, .3]]])
    logits = np.array([[[-3., -4.], [2., -5.]]])
    score, predicted = ex.best_prediction(logits, boxes)
    np.testing.assert_allclose(predicted, boxes[:, 1])
    np.testing.assert_allclose(score, [1 / (1 + np.exp(-2))])
    with pytest.raises(ValueError, match='finite'):
        ex.best_prediction(logits * np.nan, boxes)


def test_manifest_ignores_gt_and_val_but_missing_train_is_fatal(tmp_path):
    ex = module()
    (tmp_path / 'a.jpg').write_bytes(b'image')
    manifest = tmp_path / 'pool.jsonl'
    rows = [dict(frame_id='val', split='val', path='absent.jpg', annotations='unread'),
            dict(frame_id='train', split='train', path='a.jpg', video_id='v',
                 group_id='g', frame_index=5, annotations={'bbox': 'poison'})]
    manifest.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    chosen = ex.load_train_rows(manifest, tmp_path)
    assert len(chosen) == 1 and chosen[0]['frame_id'] == 'train'
    assert 'annotations' not in chosen[0]
    (tmp_path / 'a.jpg').unlink()
    with pytest.raises(FileNotFoundError, match='a.jpg'):
        ex.load_train_rows(manifest, tmp_path)


def test_cache_reuse_checks_provenance_ids_and_shapes(tmp_path):
    ex = module()
    target = tmp_path / 'features.npz'
    meta = dict(schema_version=1, checkpoint_sha256='abc', manifest_sha256='def')
    arrays = dict(frame_ids=np.array(['f']), features=np.ones((1, 256), np.float32),
                  inputs=np.ones((1, 6), np.float32), pred_box=np.ones((1, 4), np.float32),
                  observed=np.array([True]), frame_index=np.array([5.]))
    ex.save_atomic(target, arrays, meta)
    assert ex.reuse_compatible(target, meta, ['f'])
    with pytest.raises(ValueError, match='incompatible'):
        ex.reuse_compatible(target, dict(meta, checkpoint_sha256='other'), ['f'])
    with pytest.raises(ValueError, match='frame_ids'):
        ex.reuse_compatible(target, meta, ['other'])
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        ex.save_atomic(target, arrays, dict(meta, checkpoint_sha256='other'))
    assert target.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp*'))
