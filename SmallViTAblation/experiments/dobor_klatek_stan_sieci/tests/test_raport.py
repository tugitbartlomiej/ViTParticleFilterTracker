"""Synthetic fixtures test reporting safeguards, never experimental conclusions."""
import copy
import importlib
import json

import pytest


def module():
    return importlib.import_module('SmallViTAblation.experiments.dobor_klatek_stan_sieci.raport')


def fixtures():
    values = dict(random=.60, readout_loss=.61, koopman_langevin=.615,
                  placebo=.612, random_large=.62)
    budgets = {a: 3000 if a == 'random_large' else 1500 for a in values}
    deltas = {f'koopman_langevin_minus_{a}': values['koopman_langevin'] - values[a]
              for a in values if a != 'koopman_langevin'}
    curves = {a: [dict(epoch_index=i, completed_epochs=i + 1, ap=(i + 1) / 100,
                       image_presentations_proxy=(i + 1) * budgets[a]) for i in range(10)]
              for a in values}
    pilot = dict(status='complete', checkpoint_rule='last_epoch', metric='mAP_0.5:0.95',
                 evaluation_set='val', test_evaluated=False, epochs=10, training_seed=42,
                 selection_seed=42, statistical_significance_tested=False,
                 rows=[dict(strategy=a, run_id=f'{a}_s42', ap=v, n_images=12043,
                            budget=budgets[a], best_epoch_ap_training_log=.99) for a, v in values.items()],
                 deltas=deltas, training_curves=curves,
                 practical_threshold_vs_readout_loss=.004, practical_gate_met=True,
                 exposure_comparison=dict(status='computed', common_range=[3000, 15000],
                                          normalized_area={a: .03 if a == 'random_large' else .06 for a in values}))
    selection = dict(status='complete', test_used=False, labels_used_for_ranking=True,
                     feature_meta=dict(feature='last_raw_vit_patch', spatial_pooling=False,
                                       labels_used=False, elapsed_seconds=12.5), elapsed_seconds=7.25,
                     selection=dict(missing_temporal_windows=10, positive_gain=20, arms={}))
    for a in values:
        added = budgets[a] - 500
        selection['selection']['arms'][a] = dict(selected=budgets[a], metadata=dict(
            seed=42, selected_count=added, budget=added, selected_by_score=0,
            selected_by_random=added, random_fill_count=0 if a.startswith('random') else added,
            random_fill_reason=None if a.startswith('random') else 'insufficient_positive_scores_under_spacing',
            requested_min_gap_frames=5, final_min_gap_frames=5, spacing_relaxed=False,
            relaxations=[], excluded_initial_count=500), overlap_added={b: 1. if b == a else .2 for b in values})
    # This overlap is directional, not a Jaccard index.
    selection['selection']['arms']['random']['overlap_added']['random_large'] = 1.
    selection['selection']['arms']['random_large']['overlap_added']['random'] = .4
    diagnostics = dict(status='complete', readout_fit_uses_val=False, used_for_ranking=False,
                       extraction_uses_labels=False, cka_only_measures_drift=True,
                       langevin_fitted_on_training_clock=False, statistical_significance_tested=False,
                       probe=dict(train_frames=500, val_frames=192), rows=[])
    for a in values:
        for epoch in (2, 4, 6, 8, 10):
            diagnostics['rows'].append(dict(arm=a, run_id=f'{a}_s42', completed_epochs=epoch,
                                           metrics=dict(auroc=.5 + epoch / 100,
                                                        average_precision=.4, balanced_bce=.6),
                                           cka_to_previous_checkpoint=None if epoch == 2 else .9))
    return pilot, selection, diagnostics


def test_exposure_area_compares_only_common_nominal_presentations():
    pilot, selection, _ = fixtures()
    checked = module().validate_inputs(pilot, selection)
    assert checked['exposure']['common_range'] == [3000, 15000]
    assert checked['exposure']['normalized_area']['random'] == pytest.approx(.06)
    assert checked['exposure']['normalized_area']['random_large'] == pytest.approx(.03)
    assert checked['deltas']['koopman_langevin_minus_readout_loss'] == pytest.approx(.005)
    assert checked['gate'] is True


@pytest.mark.parametrize('corruption', ['duplicate_arm', 'wrong_delta', 'string_gate',
                                       'wrong_exposure', 'wrong_area', 'test_evaluated'])
def test_contradictory_reports_are_rejected(corruption):
    pilot, selection, _ = fixtures()
    if corruption == 'duplicate_arm':
        pilot['rows'][-1] = copy.deepcopy(pilot['rows'][0])
    elif corruption == 'wrong_delta':
        pilot['deltas']['koopman_langevin_minus_placebo'] = .5
    elif corruption == 'string_gate':
        pilot['practical_gate_met'] = 'false'
    elif corruption == 'wrong_exposure':
        pilot['training_curves']['random_large'][0]['image_presentations_proxy'] = 1500
    elif corruption == 'wrong_area':
        pilot['exposure_comparison']['normalized_area']['random_large'] = .06
    else:
        pilot['test_evaluated'] = True
    with pytest.raises(ValueError):
        module().validate_inputs(pilot, selection)


def test_validation_leakage_or_mismatched_run_invalidates_patch_comparison():
    pilot, selection, diagnostics = fixtures()
    diagnostics['readout_fit_uses_val'] = True
    with pytest.raises(ValueError):
        module().validate_inputs(pilot, selection, diagnostics)
    diagnostics['readout_fit_uses_val'] = False
    diagnostics['rows'][0]['run_id'] = 'different_run'
    with pytest.raises(ValueError):
        module().validate_inputs(pilot, selection, diagnostics)


def test_artifact_uses_final_ap_and_marks_unknown_costs_and_weak_labels(tmp_path):
    paths = []
    for name, value in zip(('pilot', 'selection', 'diagnostics'), fixtures()):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        paths.append(path)
    out = tmp_path / 'synthetic_report'
    module().build(paths[0], paths[1], out, diagnostics_path=paths[2])
    prose = (out / 'WYNIKI_PILOT.md').read_text(encoding='utf-8')
    assert '| Koopman + Langevin | 1500 | 0.6150 |' in prose
    assert '0.9900' not in prose  # a best epoch must not become the primary endpoint
    assert 'brak pomiaru' in prose
    assert 'nie jest segmentacją' in prose
    assert 'statyczny' in prose and 'jednym seedzie' in prose
    assert 'mianownikiem' in prose and 'Jaccarda' in prose
    assert '12.50 s' in prose and '7.25 s' in prose
    assert 'README.md' in prose and 'LITERATURA.md' in prose
    for name in ('WYKRES_UCZENIA.png', 'WYKRES_PATCHY.png'):
        assert (out / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    with pytest.raises(FileExistsError):
        module().build(paths[0], paths[1], out)


def test_invalid_data_does_not_create_output(tmp_path):
    pilot, selection, _ = fixtures()
    pilot['practical_gate_met'] = False
    a, b = tmp_path / 'p.json', tmp_path / 's.json'
    a.write_text(json.dumps(pilot), encoding='utf-8')
    b.write_text(json.dumps(selection), encoding='utf-8')
    out = tmp_path / 'report'
    with pytest.raises(ValueError):
        module().build(a, b, out)
    assert not out.exists()
