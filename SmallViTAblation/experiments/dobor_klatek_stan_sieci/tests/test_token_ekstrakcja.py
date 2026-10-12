import numpy as np
import pytest
from SmallViTAblation.experiments.dobor_klatek_stan_sieci.token_ekstrakcja import project_patches


def test_each_patch_is_retained_and_only_channels_are_projected():
    image = np.arange(24, dtype=float).reshape(1, 2, 3, 4)
    patches = project_patches(image, np.eye(2))
    assert patches.shape == (1, 12, 2)
    np.testing.assert_array_equal(patches[0, 6], image[0, :, 1, 2])
    changed = image.copy()
    changed[0, :, 1, 2] += 100
    diff = project_patches(changed, np.eye(2))-patches
    assert np.flatnonzero(np.abs(diff).sum(-1)).tolist() == [6]


def test_wrong_channel_layout_rejected():
    with pytest.raises(ValueError):
        project_patches(np.zeros((1, 3, 4, 2)), np.eye(2))
