"""CPU evidence for weak patch probes and checkpoint drift measurements."""
import importlib

import numpy as np
import pytest


def module():
    return importlib.import_module(
        'SmallViTAblation.experiments.dobor_klatek_stan_sieci.diagnostyka_checkpointow')


def test_patch_metrics_reward_separation_and_handle_ties():
    ex = module()
    y = np.array([0, 1, 0, 1])
    tied = ex.binary_metrics(y, np.zeros(4))
    good = ex.binary_metrics(y, np.array([-3, 3, -2, 2]))
    assert tied['auroc'] == pytest.approx(.5)
    assert tied['average_precision'] == pytest.approx(.5)
    assert good['auroc'] == pytest.approx(1.)
    assert good['average_precision'] == pytest.approx(1.)
    assert good['balanced_bce'] < tied['balanced_bce']
    assert ex.binary_metrics(np.ones(4), np.zeros(4))['auroc'] is None


def test_linear_cka_is_rotation_invariant_but_not_semantic_accuracy():
    ex = module()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 7))
    rotation, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    assert ex.linear_cka(x, x @ rotation + 9.) == pytest.approx(1., abs=1e-12)
    assert ex.linear_cka(x, rng.normal(size=x.shape)) < .15
    assert ex.linear_cka(np.ones_like(x), x) is None


def test_validation_values_and_labels_cannot_change_fitted_readout():
    ex = module()
    rng = np.random.default_rng(3)
    tokens = rng.normal(size=(7, 32, 4))
    labels = tokens[:, :, 0] > 0
    train = np.array([True] * 5 + [False] * 2)
    first = ex.fit_patch_readout(tokens, labels, train, seed=42)
    tokens[~train] = 100000.
    labels[~train] = ~labels[~train]
    second = ex.fit_patch_readout(tokens, labels, train, seed=42)
    for key in ('mean', 'scale', 'weights'):
        np.testing.assert_array_equal(first[key], second[key])
    assert set(first['fit_image_indices']) <= set(np.flatnonzero(train))
    assert np.max(first['mean']) < 2


def test_probe_is_fixed_group_stratified_and_rejects_cross_split_groups():
    ex = module()
    train = [dict(frame_id=f't{i}', split='train', group_id=f'train{i}', video_id=f'tv{i}', frame_index=i)
             for i in range(5)]
    val = [dict(frame_id=f'v{g}_{i}', split='val', group_id=f'val{g}', video_id=f'vv{g}', frame_index=i)
           for g in range(3) for i in range(20)]
    chosen = ex.select_probe(train + val, [r['frame_id'] for r in train], val_count=12, seed=42)
    assert chosen == ex.select_probe(train + val, [r['frame_id'] for r in train], val_count=12, seed=42)
    groups = [r['group_id'] for r in chosen if r['split'] == 'val']
    assert [groups.count(f'val{g}') for g in range(3)] == [4, 4, 4]
    assert [r['frame_id'] for r in chosen[:5]] == [r['frame_id'] for r in train]
    val[0]['group_id'] = 'train0'
    with pytest.raises(ValueError, match='group'):
        ex.select_probe(train + val, [r['frame_id'] for r in train], val_count=12, seed=42)
