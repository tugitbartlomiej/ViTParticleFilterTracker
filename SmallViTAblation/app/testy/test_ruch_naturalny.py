"""Bounded verification, with all outputs in pytest temporary directories.

F:/Python/python.exe -B -m pytest -p no:cacheprovider app/test_ruch_naturalny.py
Optional real fp32 full-forward smoke: set RUCH_NATURALNY_REAL_SMOKE=1.
No dataset-wide runs, image copies, optimizer or historical-result edits.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ruch_naturalny as natural
import ruch_zbioru as shared


def save_checkpoint(path, step, value=0.):
    torch.save({'net': {'weight': torch.full((6,), value), 'counter': torch.tensor(4)}, 'krok': step}, path)


def metadata(out):
    return json.loads((out / 'manifest.json').read_text(encoding='utf-8'))


def frame(out, step):
    m = metadata(out)
    return np.fromfile(out / m['files'][str(step)]['name'], dtype='<f4').reshape(-1, 4)


def test_discover_numeric_order_payload_and_real_sparse_steps(tmp_path):
    for step in (100, 0, 10, 30): save_checkpoint(tmp_path / f'k{step}.pt', step)
    (tmp_path / 'old_checkpoint.pt').write_bytes(b'not an eligible checkpoint')
    records = natural.discover(tmp_path, end_step=100)
    assert [r['step'] for r in records] == [0, 10, 30, 100]
    for r in records:
        assert set(r) == {'path', 'step', 'sha256', 'size_bytes'}
        assert r['sha256'] == shared._sha(r['path'])
        assert r['size_bytes'] == Path(r['path']).stat().st_size
    assert [r['step'] for r in natural.discover(tmp_path, 10, 100, 30)] == [10, 100]


@pytest.mark.parametrize('failure', ['wrong_step', 'duplicate', 'no_krok', 'legacy', 'missing_start', 'nonfinite'])
def test_discovery_rejects_ambiguous_or_invalid_checkpoints(tmp_path, failure):
    p = tmp_path / 'k0000.pt'
    save_checkpoint(p, 0)
    if failure == 'wrong_step': save_checkpoint(p, 7)
    if failure == 'duplicate': save_checkpoint(tmp_path / 'k0.pt', 0)
    if failure == 'no_krok': torch.save({'net': {'weight': torch.ones(1)}}, p)
    if failure == 'legacy': torch.save({'weight': torch.ones(1)}, p)
    if failure == 'missing_start': p.rename(tmp_path / 'k0010.pt')
    if failure == 'nonfinite': save_checkpoint(p, 0, float('nan'))
    with pytest.raises(ValueError): natural.discover(tmp_path)


@pytest.mark.parametrize('args', [(-1, 10, 1), (0, -1, 1), (10, 0, 1), (0, 10, 0), (True, 10, 1)])
def test_discovery_range_validation(tmp_path, args):
    with pytest.raises(ValueError): natural.discover(tmp_path, *args)


def test_logs_real_steps_and_file_hashes(tmp_path):
    shared._json(tmp_path / 'LOG.json', [{'krok': 100, 'strata': 2., 'heat': .3}, {'krok': 50, 'strata': 3.}])
    shared._json(tmp_path / 'OCENY.json', [{'krok': 0, 'mAP50': .01}, {'krok': 100, 'mAP50': .4}])
    result = natural.load_logs(tmp_path)
    assert result['series'] == {'loss': {'steps': [50, 100], 'values': [3., 2.]},
                                'map50': {'steps': [0, 100], 'values': [.01, .4]}}
    assert result['warnings'] == []
    assert len(result['files']) == 2
    for f in result['files']: assert f['sha256'] == shared._sha(f['path'])


def test_logs_no_epoch_renumbering_unknown_fields_explicitly_warned(tmp_path):
    p = tmp_path / 'history.jsonl'
    p.write_text('{"epoch": 8, "loss": 1}\n{"step": 700, "loss": 2, "unknown": 4}\n', encoding='utf-8')
    result = natural.load_logs(p)
    assert result['series']['loss'] == {'steps': [700], 'values': [2.]}
    assert result['series']['map50']['steps'] == []
    assert any('no explicit training step' in w for w in result['warnings'])
    assert any('unknown' in w for w in result['warnings'])
    assert natural.load_logs('')['warnings']
    p = tmp_path / 'unknown.json'
    shared._json(p, {'epochs': [1, 2], 'loss': [4, 3]})
    assert 'unknown log format' in natural.load_logs(p)['warnings'][0]


@pytest.mark.parametrize('rows', [[{'krok': 10, 'strata': float('nan')}],
    [{'krok': 10, 'strata': 1, 'unknown': float('inf')}],
    [{'krok': 1.5, 'strata': 1}], [{'krok': True, 'strata': 1}],
    [{'krok': 2, 'step': 3, 'strata': 1}], [{'krok': 1, 'strata': True}],
    [{'krok': 1, 'mAP50': 2}], [{'krok': 1, 'strata': 1}, {'krok': 1, 'strata': 2}]])
def test_logs_reject_nonfinite_or_ambiguous_data(tmp_path, rows):
    p = tmp_path / 'LOG.json'
    p.write_text(json.dumps(rows), encoding='utf-8')
    with pytest.raises(ValueError): natural.load_logs(p)


class TinyNet(torch.nn.Module):
    instances = []

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(6))
        self.register_buffer('counter', torch.tensor(4))
        self.dropout = torch.nn.Dropout(.5)
        self.instances.append(self)


@pytest.fixture
def small(tmp_path, monkeypatch):
    import cv2
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    wagi = inputs / 'wagi'
    wagi.mkdir()
    for step, value in [(0, 0.), (10, .1), (20, .2), (30, .3)]:
        save_checkpoint(wagi / f'k{step:04d}.pt', step, value)
    images = []
    annotations = []
    # Deliberately unsorted names: checkpoint loops must preserve manifest order.
    for i, name in enumerate(['z.png', 'a.png', 'm.png']):
        p = inputs / name
        cv2.imwrite(str(p), np.full((60, 80, 3), 40+i*30, np.uint8))
        images.append({'id': i+9, 'file_name': str(p), 'width': 80, 'height': 60})
        annotations.append({'id': i+19, 'image_id': i+9, 'category_id': 1, 'bbox': [5+i, 9, 20, 30]})
    shared._json(inputs / 'coco.json', {'images': images, 'annotations': annotations})
    monkeypatch.setattr(natural, 'OUTPUT_ROOT', tmp_path / 'outputs')
    monkeypatch.setattr(natural, 'TOKENS', 4)
    monkeypatch.setattr(natural, 'DIM', 6)
    original = shared._private_module
    monkeypatch.setattr(shared, '_private_module', lambda name: SimpleNamespace(MalyViT=TinyNet)
                        if name == 'model_snapshot' else original(name))
    TinyNet.instances = []
    batches = []

    def infer(net, rows, paths, dev, core):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == 'highest'
        assert all(not p.requires_grad and p.grad is None for p in net.parameters())
        assert all(not m.training for m in net.modules())
        batches.append([r['image_name'] for r in rows])
        for row, path in zip(rows, paths):
            if shared._sha(path) != row['image_sha256']: raise ValueError('image changed during run')
        z = []
        metrics = []
        for row in rows:
            i = row['index']
            initial = np.random.default_rng(100+i).normal(size=(4, 6)).astype(np.float32)
            # Motion outside the 3D PCA subspace, so projected norm is insufficient.
            z.append(initial + net.weight.detach().cpu().numpy() * np.arange(1, 7, dtype=np.float32))
            metrics.append([.5+i*.02+float(net.weight[0]), .4+i*.03])
        return np.asarray(z), np.asarray(metrics)

    monkeypatch.setattr(natural, '_infer', infer)
    config = natural.NaturalConfig(checkpoint_dir=str(wagi), manifest=str(inputs / 'coco.json'),
                                   end_step=30, batch_size=2, device='cpu', frame_format='f32',
                                   reference_checkpoint=str(wagi / 'k0030.pt'))
    return SimpleNamespace(config=config, wagi=wagi, inputs=inputs, batches=batches, infer=infer)


def test_run_fixed_basis_exact_norm_order_metrics_and_timeline(small):
    before = {p.name: shared._sha(p) for p in small.wagi.iterdir()}
    rng = torch.get_rng_state().clone()
    events = []
    out = natural.run(small.config, progress=events.append)
    m = metadata(out)
    assert m['status'] == 'complete' and m['steps'] == [0, 10, 20, 30]
    assert m['time_axis'] == 'training_checkpoint' and m['baseline_step'] == 0
    assert m['checkpoint_sha256'] == before['k0030.pt']
    assert m['image_names'] == ['z.png', 'a.png', 'm.png']
    assert m['reference_verification']['matches']
    assert not list(out.glob('*.tmp')) and not list(out.glob('*.part'))
    assert not any(p.suffix in ('.pt', '.png', '.jpg') for p in out.iterdir())
    assert before == {p.name: shared._sha(p) for p in small.wagi.iterdir()}
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a['weights_unchanged'] and a['all_parameters_frozen'] and a['eval'] for a in m['checkpoint_audits'])
    assert all(p.grad is None for net in TinyNet.instances for p in net.parameters())
    assert small.batches == [['z.png', 'a.png'], ['m.png']]*4  # baseline never inferred twice
    assert [e['fraction'] for e in events] == sorted(e['fraction'] for e in events)
    assert events[-1]['fraction'] == 1.
    assert np.all(frame(out, 0)[:, 3] == 0)
    pca = np.load(out / 'initial_pca.npz')
    for step in m['steps']:
        value = float(torch.load(small.wagi / f'k{step:04d}.pt', weights_only=True)['net']['weight'][0])
        expected = []
        base = []
        for i in range(3):
            initial = np.random.default_rng(100+i).normal(size=(4, 6)).astype(np.float32)
            base.append(initial)
            expected.append(initial + value*np.arange(1, 7, dtype=np.float32))
        expected, base = np.concatenate(expected), np.concatenate(base)
        actual = frame(out, step)
        np.testing.assert_allclose(actual[:, :3], (expected-pca['mean']) @ pca['components'], atol=2e-7)
        np.testing.assert_allclose(actual[:, 3], np.linalg.norm(expected.astype(np.float64)-base, axis=1), atol=2e-7)
        assert m['stats'][str(step)]['confidence'] == pytest.approx(.52+value)
        assert m['stats'][str(step)]['iou'] == pytest.approx(.43)
        assert m['stats'][str(step)]['both90'] == 0
    assert np.any(frame(out, 30)[:, 3] > np.linalg.norm(frame(out, 30)[:, :3]-frame(out, 0)[:, :3], axis=1)+.001)
    # Real shared reader already supports step-0 v2, main adapts nonzero training axis.
    timeline = shared.Timeline(out)
    assert timeline.steps == [0, 10, 20, 30] and timeline.curves()['steps'] == timeline.steps
    binary = timeline.frame(10)
    assert binary.shape == (12, 4)
    binary._mmap.close()
    telemetry = np.load(out / 'metrics.npy')
    np.testing.assert_array_equal(telemetry[0, :, 3], [0, 10, 20, 30])


def test_subset_nonzero_start_reference_outside_selection(small):
    small.config.start_step, small.config.end_step = 10, 20
    out = natural.run(small.config)
    m = metadata(out)
    assert m['steps'] == [10, 20] and m['baseline_step'] == 10
    assert m['reference_verification']['step'] == 30 and m['reference_verification']['matches']
    assert m['checkpoint_sha256'] == shared._sha(small.wagi / 'k0020.pt')
    assert np.all(frame(out, 10)[:, 3] == 0)
    assert np.all(frame(out, 20)[:, 3] > 0)
    assert m['curves']['steps'] == [10, 20]
    # Reference can also be an earlier saved step within the full directory.
    small.config.reference_checkpoint = str(small.wagi / 'k0010.pt')
    assert metadata(natural.run(small.config))['reference_verification']['matches']


def test_atomic_publication_fsync_uses_writable_descriptor(small, monkeypatch):
    original_fsync = os.fsync
    synced = []
    def writable_fsync(fd):
        # Zero-byte write checks access without changing bytes. In particular,
        # a read-only frame handle fails here on Windows and POSIX.
        assert os.write(fd, b'') == 0
        synced.append(fd)
        original_fsync(fd)
    monkeypatch.setattr(natural.os, 'fsync', writable_fsync)
    out = natural.run(small.config)
    assert synced and metadata(out)['status'] == 'complete'
    assert all((out / r['name']).stat().st_size == 12*16 for r in metadata(out)['files'].values())


def test_reference_mismatch_fails_before_inference(small):
    bad = small.inputs / 'reference.pt'
    save_checkpoint(bad, 30, .9)
    small.config.reference_checkpoint = str(bad)
    with pytest.raises(ValueError, match='reference SHA256'):
        natural.run(small.config)
    out = next(Path(natural.OUTPUT_ROOT).iterdir())
    m = metadata(out)
    assert m['status'] == 'failed' and m['steps'] == [] and m['checkpoint_sha256'] is None
    assert not small.batches and not list(out.glob('*.tmp'))


@pytest.mark.parametrize('stage', ['before', 'baseline', 'frame', 'measure'])
def test_cancel_preserves_only_complete_frames_and_removes_baseline(small, stage):
    cancelled = stage == 'before'
    def report(event):
        nonlocal cancelled
        if event['phase'] == stage: cancelled = True
    out = natural.run(small.config, progress=report, cancel=lambda: cancelled)
    m = metadata(out)
    assert m['status'] == 'cancelled' and not list(out.glob('*.tmp'))
    assert m['steps'] == ([0] if stage in ('frame', 'measure') else [])
    assert m['curves']['steps'] == m['steps']
    assert m['checkpoint_sha256'] == (shared._sha(small.wagi / 'k0000.pt') if m['steps'] else None)
    for record in m['files'].values(): assert (out / record['name']).stat().st_size == 12*16
    if stage == 'measure':
        assert m['current_frame_completed_images'] == 2
        assert (out / 'step_0010.bin.part').stat().st_size == 8*16
        telemetry = np.load(out / 'metrics.npy')
        assert np.isfinite(telemetry[:2, 1]).all() and np.isnan(telemetry[2, 1]).all()
        assert m['observed_checkpoints_per_image'] == [2, 2, 1]


@pytest.mark.parametrize('source', ['future_checkpoint', 'past_checkpoint', 'image'])
def test_change_detection_preserves_finished_frames(small, source):
    def report(e):
        if e['phase'] != 'frame' or e['step'] != 0: return
        if source == 'image': (small.inputs / 'z.png').write_bytes(b'changed')
        else:
            step = 10 if source == 'future_checkpoint' else 0
            save_checkpoint(small.wagi / f'k{step:04d}.pt', step, .9)
    with pytest.raises(ValueError, match='changed'): natural.run(small.config, progress=report)
    out = next(Path(natural.OUTPUT_ROOT).iterdir())
    m = metadata(out)
    assert m['status'] == 'failed' and m['steps'][0] == 0
    assert not list(out.glob('*.tmp'))


@pytest.mark.parametrize('kind', ['buffer', 'parameter', 'training', 'gradient'])
def test_training_immutability_checked_before_publication(small, monkeypatch, kind):
    def bad(net, rows, paths, dev, core):
        z, metrics = small.infer(net, rows, paths, dev, core)
        if kind == 'buffer': net.counter.add_(1)
        if kind == 'parameter': net.weight.add_(1)
        if kind == 'training': net.train()
        if kind == 'gradient': net.weight.grad = torch.ones_like(net.weight)
        return z, metrics
    monkeypatch.setattr(natural, '_infer', bad)
    with pytest.raises((RuntimeError, AssertionError)): natural.run(small.config)
    out = next(Path(natural.OUTPUT_ROOT).iterdir())
    assert metadata(out)['status'] == 'failed' and metadata(out)['steps'] == []
    assert not list(out.glob('*.tmp'))


def test_resource_precheck_baseline_allowance_and_batch_sensitive(small, monkeypatch):
    estimate = natural._estimates(1500, 101, 4)
    assert estimate['baseline_temporary_bytes'] == 1500*4*6*4  # mocked dimensions
    assert estimate['disk_bytes'] > estimate['baseline_temporary_bytes']
    assert estimate['cuda_bytes'] < 940 << 20
    assert natural._estimates(1500, 101, 16)['cuda_bytes'] > estimate['cuda_bytes']
    monkeypatch.setattr(natural.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match='temporary baseline'): natural.run(small.config)
    out = next(Path(natural.OUTPUT_ROOT).iterdir())
    assert not small.batches and not list(out.glob('*.tmp'))


def test_real_dimensions_disk_budget_and_small_cuda_batch():
    estimate = natural._estimates(1500, 101, 4)
    assert estimate['baseline_temporary_bytes'] == 2_764_800_000
    assert estimate['disk_bytes'] > 2_764_800_000 + 101*1500*600*16
    assert estimate['cuda_bytes'] == 256 << 20
    assert natural._estimates(1500, 101, 16)['cuda_bytes'] == 640 << 20


@pytest.mark.parametrize('kind', ['success', 'cancel', 'error'])
def test_precision_flags_restored_on_every_exit(small, monkeypatch, kind):
    original = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32)
    stop = False
    def report(e):
        nonlocal stop
        if e['phase'] == 'frame':
            if kind == 'error': raise RuntimeError('callback error')
            if kind == 'cancel': stop = True
    try:
        if kind == 'error':
            with pytest.raises(RuntimeError, match='callback error'): natural.run(small.config, progress=report)
        else: natural.run(small.config, progress=report, cancel=lambda: stop)
        assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32) == before
    finally:
        torch.set_float32_matmul_precision(original[0])
        torch.backends.cuda.matmul.allow_tf32 = original[1]
        torch.backends.cudnn.allow_tf32 = original[2]


def test_manifest_order_and_gt_resize_validation(small):
    rows, _ = shared._dataset(small.config, lambda: False, lambda e: None)
    assert rows[0]['bbox_xyxy_480'] == [30., 48., 150., 208.]
    rows[0]['index'] = 1
    p = small.inputs / 'bad.json'
    shared._json(p, rows)
    small.config.manifest = str(p)
    with pytest.raises(ValueError, match='order'): natural.run(small.config)


@pytest.mark.parametrize('key,value', [('batch_size', 0), ('batch_size', True), ('max_images', -1),
    ('start_step', -1), ('end_step', -1), ('checkpoint_stride', 0), ('seed', -1), ('device', 'cuda:-1')])
def test_config_bounds(small, key, value):
    setattr(small.config, key, value)
    with pytest.raises(ValueError): small.config.validate()


@pytest.mark.skipif(os.environ.get('RUCH_NATURALNY_REAL_SMOKE') != '1', reason='explicit real one-image two-checkpoint smoke')
def test_real_one_image_two_checkpoints_fp32_and_gt(tmp_path, monkeypatch):
    # Real saved history, actual shared input manifest, no synthetic model weights.
    monkeypatch.setattr(natural, 'OUTPUT_ROOT', tmp_path / 'real')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        c = natural.NaturalConfig(end_step=10, max_images=1, batch_size=1,
                                   device=os.environ.get('RUCH_NATURALNY_SMOKE_DEVICE', 'cpu'))
        before = shared._sha(natural.DEFAULT_CHECKPOINT)
        out = natural.run(c)
        m = metadata(out)
        assert m['status'] == 'complete' and m['steps'] == [0, 10]
        assert m['reference_verification']['matches']
        assert frame(out, 0).shape == (600, 4) and np.all(frame(out, 0)[:, 3] == 0)
        assert np.any(frame(out, 10)[:, 3] > 0)
        assert all(a['weights_unchanged'] for a in m['checkpoint_audits'])
        assert shared._sha(natural.DEFAULT_CHECKPOINT) == before
        rows, paths = shared._dataset(c, lambda: False, lambda e: None)
        core = shared._private_module('pelne_latenty')
        net = shared._private_module('model_snapshot').MalyViT().float().eval().requires_grad_(False)
        payload, _ = natural._payload(natural.DEFAULT_CHECKPOINT_DIR / 'k0010.pt', expected_step=10)
        net.load_state_dict(payload['net'])
        dev = shared._device('cpu')
        with natural._fp32(dev):
            output, z, _, _ = shared._baseline(net, rows[0], paths[0], dev, core)
            expected = core.top1_metrics(output, torch.tensor(rows[0]['bbox_xyxy_480']))
        assert m['stats']['10']['confidence'] == pytest.approx(expected['confidence'], abs=1e-5)
        assert m['stats']['10']['iou'] == pytest.approx(expected['iou'], abs=1e-5)
        assert not list(out.glob('*.tmp'))
    finally:
        torch.set_num_threads(old_threads)


# ---------------------------------------------------------------- zapis 16-bitowy i uzupełnianie brakujących kroków
def test_q16_half_size_same_stats_and_values_within_half_step(small):
    """Format q16: połowa bajtów, statystyki z float32 identyczne, wartości w granicy pół kroku kwantyzacji."""
    from dataclasses import replace
    f32 = natural.run(small.config)
    q16 = natural.run(replace(small.config, frame_format='q16'))
    a, b = metadata(f32), metadata(q16)
    assert b['status'] == 'complete' and b['frame_format'] == 'q16' and b['steps'] == a['steps']
    assert a['stats'] == b['stats'] and a['curves'] == b['curves']
    t = shared.Timeline(q16)
    assert t.frame_format == 'q16'
    for step in a['steps']:
        item = b['files'][str(step)]
        assert item['format'] == 'q16' and (q16 / item['name']).stat().st_size == a['point_count'] * 8
        exact, approx = frame(f32, step), t.read(step)
        span = np.array(item['quant']['span'] + [item['quant']['norm_max']], np.float32)
        assert (np.abs(exact - approx) <= span / 65535 * .5 + 1e-6).all()
        assert np.array_equal(t.read_raw(step), np.fromfile(q16 / item['name'], '<u2').reshape(-1, 4))
    assert np.all(t.read(a['steps'][0])[:, 3] == 0)          # norma kroku bazowego dokładnie 0


@pytest.mark.parametrize('fmt', ['f32', 'q16'])
def test_resume_missing_steps_equals_uninterrupted(small, fmt):
    """Przerwany pomiar + uzupelnij() == pomiar bez przerwy, w obu formatach (bitowo pliki i metryki)."""
    from dataclasses import replace
    config = replace(small.config, frame_format=fmt)
    pelny = natural.run(config)
    klatki = []
    przerwany = natural.run(config, progress=lambda e: klatki.append(1) if e.get('phase') == 'frame' else None,
                            cancel=lambda: len(klatki) >= 2)
    assert metadata(przerwany)['status'] == 'cancelled' and metadata(przerwany)['steps'] == [0, 10]
    assert any(d == przerwany for d, *_ in natural.niekompletne(natural.OUTPUT_ROOT))
    natural.uzupelnij(przerwany)
    a, b = metadata(pelny), metadata(przerwany)
    assert b['status'] == 'complete' and b['steps'] == a['steps']
    assert b['resumes'][-1]['baseline_bitwise_reproduced'] and b['resumes'][-1]['status'] == 'complete'
    for step in a['steps']:
        assert (pelny / a['files'][str(step)]['name']).read_bytes() == (przerwany / b['files'][str(step)]['name']).read_bytes()
    assert a['stats'] == b['stats'] and a['curves'] == b['curves'] and a['bounds'] == b['bounds']
    assert np.array_equal(np.load(pelny / 'metrics.npy'), np.load(przerwany / 'metrics.npy'), equal_nan=True)
    assert not list(przerwany.glob('*.part')) and not list(przerwany.glob('*.tmp'))


def test_convert_f32_timeline_to_q16_copy(small):
    """Konwersja istniejącego pomiaru: kopia q16 obok, oryginał nietknięty, wartości w granicy pół kroku."""
    f32 = natural.run(small.config)
    przed = {p.name: shared._sha(p) for p in f32.iterdir()}
    kopia = shared.konwertuj_q16(f32)
    assert kopia.name == f32.name + '_q16' and {p.name: shared._sha(p) for p in f32.iterdir()} == przed
    a, b = shared.Timeline(f32), shared.Timeline(kopia)
    assert b.frame_format == 'q16' and b.steps == a.steps
    assert b.metadata['stats'] == a.metadata['stats'] and (kopia / 'metrics.npy').is_file()
    for step in a.steps:
        q = b.metadata['files'][str(step)]['quant']
        tol = np.array(q['span'] + [q['norm_max']], np.float32) / 65535 * .5 + 1e-6
        assert (np.abs(a.read(step) - b.read(step)) <= tol).all()
