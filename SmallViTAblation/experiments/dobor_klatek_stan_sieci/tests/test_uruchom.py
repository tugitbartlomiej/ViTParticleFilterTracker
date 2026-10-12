"""Immutable pilot plans and honest one-seed summaries, without GPU training."""
import importlib
import json
from pathlib import Path

import pytest
import yaml


def module():
    return importlib.import_module('SmallViTAblation.experiments.dobor_klatek_stan_sieci.uruchom')


def fixture(tmp_path):
    project = tmp_path / 'SmallViTAblation'
    configs = project / 'configs'
    configs.mkdir(parents=True)
    base = configs / 'protocol.yaml'
    base.write_text(yaml.safe_dump(dict(training=dict(epochs=30, lr=.0001),
                                       model=dict(init='coco_pretrained'))))
    (configs / 'profiles.yaml').write_text(yaml.safe_dump(dict(version=1, profiles={
        'nano_static_pilot': dict(epochs=30, models=['rf-detr-nano'], budgets=[500,1500,2500], custom='keep'),
        'other': dict(epochs=50)}, untouched=dict(flag=True))))
    variants = tmp_path / 'variants'
    variants.mkdir()
    common = [f'l0_{i}' for i in range(500)]
    for strategy in ('random', 'readout_loss', 'koopman_langevin', 'placebo', 'random_large'):
        budget = 3000 if strategy == 'random_large' else 1500
        content = dict(strategy=strategy, seed=42, budget=budget,
                       frame_ids=common + [f'{strategy}_{i}' for i in range(budget - 500)])
        (variants / f'pilot_{strategy}.json').write_text(json.dumps(content))
    (variants / 'MANIFEST.json').write_text(json.dumps(dict(status='complete')))
    return project, variants, tmp_path / 'pilot', base


def test_build_creates_five_seed_matched_tasks_without_initializing_queue(tmp_path):
    ex = module()
    project, variants, out, base = fixture(tmp_path)
    result = ex.build(project, variants, out, base, seed=42, epochs=10)
    tasks = [json.loads(line) for line in (out / 'queue.jsonl').read_text().splitlines()]
    assert len(tasks) == 5
    assert {t['train_seed'] for t in tasks} == {42}
    assert {t['strategy']: t['budget'] for t in tasks} == dict(
        random=1500, readout_loss=1500, koopman_langevin=1500, placebo=1500, random_large=3000)
    assert all(t['run_id'].endswith('_s42') for t in tasks)
    assert all('trening.py' in t['argv'][1] for t in tasks)
    assert not (out / 'queue').exists()
    assert result['tasks'] == 5
    assert result['checkpoint_interval_epochs'] == 2
    copied = yaml.safe_load((out / 'config/profiles.yaml').read_text())
    assert copied['profiles']['nano_static_pilot']['epochs'] == 10
    assert copied['profiles']['nano_static_pilot']['custom'] == 'keep'
    assert copied['profiles']['nano_static_pilot']['budgets'] == [500,1500,2500,3000]
    assert yaml.safe_load((base.parent / 'profiles.yaml').read_text())['profiles']['nano_static_pilot']['budgets'] == [500,1500,2500]
    assert copied['profiles']['other']['epochs'] == 50
    assert copied['untouched'] == {'flag': True}
    assert yaml.safe_load(base.read_text())['training']['epochs'] == 30
    with pytest.raises(FileExistsError):
        ex.build(project, variants, out, base, seed=42, epochs=10)


@pytest.mark.parametrize('problem', ['duplicate', 'different_l0', 'budget'])
def test_invalid_selection_never_creates_a_plan(tmp_path, problem):
    ex = module()
    project, variants, out, base = fixture(tmp_path)
    path = variants / 'pilot_placebo.json'
    variant = json.loads(path.read_text())
    if problem == 'duplicate':
        variant['frame_ids'][-1] = variant['frame_ids'][0]
    elif problem == 'different_l0':
        variant['frame_ids'][0] = 'other_l0'
    else:
        variant['budget'] = 1000
    path.write_text(json.dumps(variant))
    with pytest.raises(ValueError):
        ex.build(project, variants, out, base, seed=42, epochs=10)
    assert not out.exists()


def test_report_requires_complete_runs_and_compares_last_epoch_only(tmp_path):
    ex = module()
    project, variants, out, base = fixture(tmp_path)
    ex.build(project, variants, out, base, seed=42, epochs=10)
    with pytest.raises(FileNotFoundError):
        ex.report(out)
    values = dict(random=.6, readout_loss=.61, koopman_langevin=.615, placebo=.612, random_large=.62)
    for strategy, value in values.items():
        run = out / 'results' / f'pilot_{strategy}_s42'
        run.mkdir(parents=True)
        payload = dict(status='passed', evaluation_set='val', test_evaluated=False,
                       training_seed=42, selection_seed=42, epochs_completed=10,
                       checkpoints={'last_epoch': {'metrics': {
                           'mAP_0.5:0.95': value, 'n_images': 12043}}})
        (run / 'validation_report.json').write_text(json.dumps(payload))
        (run / 'epoch_val_map.json').write_text(json.dumps(dict(per_epoch_map_50_95_val={
            str(i): (i+1)/100 for i in range(10)})))
    result = ex.report(out)
    assert result['deltas']['koopman_langevin_minus_readout_loss'] == pytest.approx(.005)
    assert result['deltas']['koopman_langevin_minus_placebo'] == pytest.approx(.003)
    assert result['statistical_significance_tested'] is False
    assert result['checkpoint_rule'] == 'last_epoch'
    assert result['practical_gate_met'] is True
    assert result['deltas']['koopman_langevin_minus_random_large'] == pytest.approx(-.005)
    assert len(result['training_curves']) == 5
    # End-of-epoch exposure proxy; comparison uses only the overlap [3000,15000].
    assert result['exposure_comparison']['common_range'] == [3000, 15000]
    assert result['exposure_comparison']['normalized_area']['random'] == pytest.approx(.06)
    assert result['exposure_comparison']['normalized_area']['random_large'] == pytest.approx(.03)
