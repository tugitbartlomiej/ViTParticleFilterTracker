"""Bounded backend verification; all generated fixtures/results are temporary.

Run with F:/Python/python.exe -B -m pytest -p no:cacheprovider
app/test_ruch_zbioru.py. No dataset-wide optimization, Eden job or SD loading.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ruch_zbioru as backend


def fixture_timeline(tmp_path, historical=False):
    files = {}
    for step in [0, 3, 7]:
        array = np.array([[step, 1, 2, step], [step+1, 2, 3, step]], dtype='<f4')
        raw = array.tobytes()
        name = f'step_{step:03d}.bin'
        (tmp_path / name).write_bytes(raw)
        files[str(step)] = {'name': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    manifest = {'schema_version': 1 if historical else 2, 'status': 'complete',
                'point_count': 2, 'image_count': 2, 'tokens_per_image': 1,
                'image_names': ['first.jpg', 'second.jpg'],
                'images': [{'index': 0, 'image_name': 'first.jpg'}, {'index': 1, 'image_name': 'second.jpg'}],
                'steps': [0, 3, 7], 'files': files,
                'bounds': {'natural': [[0, 8], [1, 2], [2, 3]], 'movement': [[0, 7], [0, 0], [0, 0]]},
                'stats': {str(s): {'confidence': .8, 'iou': .7, 'both90': 0} for s in [0, 3, 7]},
                'curves': {'confidence': [.5]*8, 'iou': [.6]*8}}
    backend._json(tmp_path / 'manifest.json', manifest)
    return manifest


def test_native_and_historical_dynamic_dimensions(tmp_path):
    for historical in (False, True):
        fixture_timeline(tmp_path, historical)
        t = backend.Timeline(tmp_path)
        assert t.steps == [0, 3, 7] and t.count == 2 and t.image_count == 2
        assert t.image_names == ['first.jpg', 'second.jpg']
        assert t.frame(3).dtype == np.float32 and t.frame(3).flags.c_contiguous
        np.testing.assert_array_equal(t.frame(3)[:, 0], [3, 4])
        assert t.bounds(True).shape == (3, 2)
        assert t.stats(3)['confidence'] == .8
        assert t.curves()['steps'] == list(range(8))
        with pytest.raises(KeyError):
            t.frame(1)


def test_natural_subset_preserves_nonzero_training_step(tmp_path):
    m = fixture_timeline(tmp_path)
    m['time_axis'] = 'training_checkpoint'
    m['steps'] = [3, 7]
    del m['files']['0']
    backend._json(tmp_path / 'manifest.json', m)
    t = backend.Timeline(tmp_path)
    assert t.steps == [3, 7]
    np.testing.assert_array_equal(t.frame(3)[:, 0], [3, 4])
    m['time_axis'] = 'latent_optimization'
    backend._json(tmp_path / 'manifest.json', m)
    with pytest.raises(ValueError):
        backend.Timeline(tmp_path)


@pytest.mark.parametrize('mutation', ['missing', 'short', 'corrupt', 'nonfinite', 'negative'])
def test_first_read_rejects_bad_binary(tmp_path, mutation):
    m = fixture_timeline(tmp_path)
    t = backend.Timeline(tmp_path)
    p = tmp_path / m['files']['3']['name']
    if mutation == 'missing':
        p.unlink()
    elif mutation == 'short':
        p.write_bytes(b'bad')
    elif mutation == 'corrupt':
        p.write_bytes(b'x' * 32)
    else:
        a = np.fromfile(p, dtype='<f4').reshape(2, 4)
        a[0, 3] = np.nan if mutation == 'nonfinite' else -1
        p.write_bytes(a.tobytes())
        m['files']['3']['sha256'] = backend._sha(p)
        backend._json(tmp_path / 'manifest.json', m)
        t = backend.Timeline(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        t.frame(3)
    with pytest.raises((ValueError, FileNotFoundError)):
        t.read(3)                       # one-pass GPU upload path applies the same checks


def test_one_pass_read_matches_memmap_and_fills_cache(tmp_path):
    fixture_timeline(tmp_path)
    t = backend.Timeline(tmp_path)
    for step in t.steps:
        values = t.read(step)
        assert values.dtype == np.float32 and values.flags.c_contiguous and values.shape == (2, 4)
        frame = t.frame(step)
        assert np.array_equal(values, frame) and values.tobytes() == frame.tobytes()
        frame._mmap.close()
    assert set(t._verified) == set(t.steps)


def test_cached_hash_is_invalidated_when_file_changes(tmp_path):
    m = fixture_timeline(tmp_path)
    t = backend.Timeline(tmp_path)
    a = t.frame(3)
    a._mmap.close()
    (tmp_path / m['files']['3']['name']).write_bytes(b'x'*32)
    with pytest.raises(ValueError, match='SHA256'):
        t.frame(3)


@pytest.mark.parametrize('key', ['confidence', 'iou', 'both90'])
def test_missing_metrics_fail_at_construction(tmp_path, key):
    m = fixture_timeline(tmp_path)
    del m['stats']['3'][key]
    backend._json(tmp_path/'manifest.json', m)
    with pytest.raises(ValueError, match=key):
        backend.Timeline(tmp_path)


@pytest.mark.parametrize('mutation', ['order', 'duplicate_step', 'missing_entry', 'dimensions', 'images', 'escape'])
def test_manifest_contract_rejects_inconsistency(tmp_path, mutation):
    m = fixture_timeline(tmp_path)
    if mutation == 'order': m['steps'] = [0, 7, 3]
    if mutation == 'duplicate_step': m['steps'] = [0, 3, 3, 7]
    if mutation == 'missing_entry': del m['files']['3']
    if mutation == 'dimensions': m['point_count'] = 3
    if mutation == 'images': m['images'].reverse()
    if mutation == 'escape': m['files']['3']['name'] = '../escape.bin'
    backend._json(tmp_path / 'manifest.json', m)
    with pytest.raises(ValueError):
        backend.Timeline(tmp_path)


@pytest.mark.parametrize('key,value', [('steps', 0), ('save_every', 0), ('max_images', -1), ('lr', 0),
    ('lr', float('nan')), ('regularization', -1), ('target_iou', 1.1), ('target_confidence', -.1),
    ('seed', -1), ('device', 'cuda:-1'), ('optimizer', 'bogus'), ('constraint', 'bogus'), ('steps', True)])
def test_config_rejects_invalid_bounds(key, value):
    c = backend.Config()
    setattr(c, key, value)
    with pytest.raises(ValueError): c.validate()


def test_config_defaults_are_real_assets():
    assert backend.Config().validate()
    assert backend.DEFAULT_CHECKPOINT.is_file() and backend.DEFAULT_MANIFEST.is_file()
    assert backend.DEFAULT_TIMELINE.is_dir()


def test_resource_estimates_include_final_noninterval_frame():
    c = backend.Config(steps=7, save_every=3)
    estimate = backend._estimates(c, 2)
    assert estimate['frame_count'] == 4  # 0,3,6,7
    assert estimate['disk_bytes'] > 4*2*600*16
    assert not estimate['full_latent_states_persisted']


def test_module_gui_entry_point_forwards_contract(monkeypatch):
    config, progress, cancel = object(), object(), object()
    def fake(self, c, progress, cancel):
        assert c is config and progress is expected_progress and cancel is expected_cancel
        return Path('result')
    expected_progress, expected_cancel = progress, cancel
    monkeypatch.setattr(backend.Engine, 'run', fake)
    assert backend.run(config, progress=progress, cancel=cancel) == Path('result')


@pytest.fixture(scope='module')
def detector_fixture(tmp_path_factory):
    import cv2
    import torch
    # Limit fixture execution time without changing process thread settings in production.
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp('detector')
    net = backend._private_module('model_snapshot').MalyViT().eval()
    torch.save({'net': net.state_dict(), 'krok': 123}, root / 'checkpoint.pt')
    cv2.imwrite(str(root / 'one.png'), np.full((320, 480, 3), 127, np.uint8))
    coco = {'images': [{'id': 8, 'file_name': str(root/'one.png'), 'width': 480, 'height': 320}],
            'annotations': [{'id': 91, 'image_id': 8, 'category_id': 1, 'bbox': [120, 100, 80, 64]}]}
    backend._json(root / 'coco.json', coco)
    yield root, net
    torch.set_num_threads(old)


def test_fast_tail_matches_full_intervention_and_weights_are_frozen(detector_fixture):
    import torch
    root, template = detector_fixture
    core = backend._private_module('pelne_latenty')
    net = backend._private_module('model_snapshot').MalyViT().eval().requires_grad_(False)
    net.load_state_dict(template.state_dict())
    x = torch.full((1, 3, 320, 480), .5)
    original, z, q = core.capture_baseline(net, x)
    before = core.state_fingerprint(net)
    changed = (z + .01).detach().requires_grad_(True)
    tail = core.fast_tail(net, z, q, changed)
    hook = net.bloki[-1].mlp.act.register_forward_hook(lambda m, a, o: changed)
    try:
        with torch.no_grad(): full, _ = net(x)
    finally: hook.remove()
    torch.testing.assert_close(tail, full, rtol=1e-5, atol=2e-6)
    tail.sum().backward()
    assert changed.grad is not None
    assert all(not p.requires_grad and p.grad is None for p in net.parameters())
    assert core.state_fingerprint(net) == before


def test_engine_one_image_two_steps_and_partial_cancel(tmp_path, monkeypatch, detector_fixture):
    root, _ = detector_fixture
    monkeypatch.setattr(backend, 'OUTPUT_ROOT', tmp_path)
    c = backend.Config(checkpoint=str(root/'checkpoint.pt'), manifest=str(root/'coco.json'),
                       steps=2, save_every=5, max_images=1, device='cpu', target_confidence=1., target_iou=1.)
    events = []
    checkpoint_sha = backend._sha(c.checkpoint)
    out = backend.Engine().run(c, progress=events.append)
    t = backend.Timeline(out)
    assert t.steps == [0, 2] and t.count == 600 and t.image_count == 1
    assert t.metadata['status'] == 'complete' and t.metadata['weights_unchanged']
    assert t.metadata['freeze']['all_parameters_frozen'] and t.metadata['checkpoint_step'] == 123
    assert t.checkpoint_sha256 == checkpoint_sha == backend._sha(c.checkpoint)
    assert t.frame(2).shape == (600, 4) and np.max(t.frame(2)[:, 3]) > 0
    assert np.all(t.frame(0)[:, 3] == 0)
    metrics = np.load(out/'metrics.npy')
    np.testing.assert_array_equal(metrics[0, :, 4], [0, 1, 2])
    assert t.curves()['steps'] == [0, 1, 2]
    assert len(t.curves()['confidence']) == 3
    assert all('message' in e and 0 <= e['fraction'] <= 1 for e in events)
    assert [e['fraction'] for e in events] == sorted(e['fraction'] for e in events)
    assert events[-1]['fraction'] == 1.
    assert not list(out.glob('*.bin.part'))
    assert not any('latent' in p.name or p.suffix in ('.jpg', '.png', '.pt') for p in out.iterdir())
    # Cancel after natural dataset frame; preserve measured state and metadata.
    cancelled = False
    def report(e):
        nonlocal cancelled
        if e['phase'] == 'baseline': cancelled = True
    partial = backend.Engine().run(c, progress=report, cancel=lambda: cancelled)
    tp = backend.Timeline(partial)
    assert tp.metadata['status'] == 'cancelled' and tp.steps == [0]
    assert tp.curves()['steps'] == [0]
    assert np.load(partial/'metrics.npy')[0, 0, 4] == 0


def test_engine_success_holds_states_and_seed_repeats_pca(tmp_path, monkeypatch, detector_fixture):
    root, _ = detector_fixture
    monkeypatch.setattr(backend, 'OUTPUT_ROOT', tmp_path)
    c = backend.Config(checkpoint=str(root/'checkpoint.pt'), manifest=str(root/'coco.json'),
                       steps=2, save_every=1, max_images=1, device='cpu', target_confidence=0., target_iou=0.)
    a, b = backend.Engine().run(c), backend.Engine().run(c)
    ta, tb = backend.Timeline(a), backend.Timeline(b)
    assert a != b and ta.metadata['pca_basis_sha256'] == tb.metadata['pca_basis_sha256']
    np.testing.assert_array_equal(ta.frame(0), ta.frame(1))
    np.testing.assert_array_equal(ta.frame(0), ta.frame(2))
    assert ta.stats(2)['active'] == 0 and ta.stats(2)['real_step'] == 0
    assert ta.stats(2)['success_count'] == 1 and ta.stats(2)['both90'] == 0
    assert ta.curves()['success_count'] == [1, 1, 1]
    assert np.all(np.load(a/'metrics.npy')[0, :, 4:6] == 0)


def test_huge_budget_rejected_before_tensor_or_frame_allocation(tmp_path, monkeypatch, detector_fixture):
    root, _ = detector_fixture
    monkeypatch.setattr(backend, 'OUTPUT_ROOT', tmp_path)
    from types import SimpleNamespace
    monkeypatch.setattr(backend.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    c = backend.Config(checkpoint=str(root/'checkpoint.pt'), manifest=str(root/'coco.json'),
                       steps=10**12, save_every=1, max_images=1, device='cpu')
    assert backend._estimates(c, 1)['frame_count'] == 10**12+1
    with pytest.raises(ValueError, match='insufficient disk'):
        backend.run(c)
    output = next(tmp_path.iterdir())
    assert {p.name for p in output.iterdir()} == {'config.json', 'manifest.json'}
    assert json.loads((output/'manifest.json').read_text())['status'] == 'failed'


def test_dataset_rejects_hash_or_annotation_mismatch(tmp_path, detector_fixture):
    root, _ = detector_fixture
    data = backend._private_module('dane')
    rows = data.build_manifest(root/'coco.json', lambda n: Path(n).read_bytes())
    rows[0]['image_sha256'] = '0'*64
    backend._json(tmp_path/'rows.json', rows)
    c = backend.Config(manifest=str(tmp_path/'rows.json'), image_root=str(root), max_images=1)
    with pytest.raises(ValueError, match='SHA256'):
        backend._dataset(c, lambda: False, lambda e: None)
    rows[0]['image_sha256'] = backend._sha(root/'one.png')
    rows[0]['bbox_xyxy_480'][0] += 1
    backend._json(tmp_path/'rows.json', rows)
    with pytest.raises(ValueError, match='resize'):
        backend._dataset(c, lambda: False, lambda e: None)
