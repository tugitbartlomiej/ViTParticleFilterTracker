import json
import numpy as np


def test_budget_is_total_not_additional_and_large_random_is_nested(tmp_path):
    from SmallViTAblation.experiments.dobor_klatek_stan_sieci.wybierz import build_variants
    rows = [dict(frame_id=str(i), group_id='g', video_id='v', frame_index=i*5, split='train')
            for i in range(60)]
    e_obs = np.linspace(.01, .5, 60)
    result = build_variants(rows, ['0', '1'], e_obs, e_obs*.5, tmp_path,
                            budget=12, large_budget=24)
    assert len(result['arms']) == 5
    arms = {name: json.loads((tmp_path / info['file']).read_text()) for name, info in result['arms'].items()}
    for name, arm in arms.items():
        expected = 24 if name == 'random_large' else 12
        assert len(arm['frame_ids']) == len(set(arm['frame_ids'])) == expected
        assert arm['frame_ids'][:2] == ['0', '1']
    assert set(arms['random']['frame_ids']) <= set(arms['random_large']['frame_ids'])
    assert not arms['random_large']['labels_used_for_ranking']


def test_placebo_preserves_missing_history_availability():
    from SmallViTAblation.experiments.dobor_klatek_stan_sieci.selekcja import placebo_scores
    rows = [dict(frame_id=str(i), group_id='g', video_id='v', frame_index=i*5, split='train') for i in range(12)]
    observed = np.ones(12)
    dynamic = np.r_[np.ones(6), np.zeros(6)]
    eligible = np.r_[np.zeros(6, dtype=bool), np.ones(6, dtype=bool)]
    scores, meta = placebo_scores(rows, observed, dynamic, seed=42, eligible=eligible)
    assert np.all(scores[:6] == 0) and np.all(scores[6:] == 1)
    assert meta['excluded_missing_history_count'] == 6
