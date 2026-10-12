import numpy as np
import pytest


def test_windows_do_not_bridge_irregular_gaps_or_videos():
    from SmallViTAblation.experiments.dobor_klatek_stan_sieci.dane import temporal_windows
    rows = [dict(frame_id=str(i), video_id='a', group_id='g', split='train', frame_index=t)
            for i, t in enumerate([0, 5, 10, 30, 35, 40])]
    rows += [dict(frame_id='b', video_id='b', group_id='g', split='train', frame_index=45)]
    assert temporal_windows(rows, context=2, step=5).tolist() == [[0, 1, 2], [3, 4, 5]]


def test_windows_reject_evaluation_rows_and_duplicate_ids():
    from SmallViTAblation.experiments.dobor_klatek_stan_sieci.dane import temporal_windows
    row = dict(frame_id='a', video_id='v', group_id='g', split='val', frame_index=0)
    with pytest.raises(ValueError, match='train'):
        temporal_windows([row], context=2, step=5)
    row['split'] = 'train'
    with pytest.raises(ValueError, match='duplicate'):
        temporal_windows([row, row], context=2, step=5)


def test_boxes_normalized_without_confusing_corner_with_center():
    from SmallViTAblation.experiments.dobor_klatek_stan_sieci.dane import target_box
    row = dict(native_width=100, native_height=200, annotations=[dict(bbox=[20, 40, 40, 80])])
    np.testing.assert_allclose(target_box(row), [.4, .4, .4, .4])
    with pytest.raises(ValueError):
        target_box(dict(row, annotations=[]))
