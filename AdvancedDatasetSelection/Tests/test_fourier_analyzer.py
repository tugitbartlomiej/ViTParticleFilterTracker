import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer


def test_compute_frequency_features_returns_expected_shape_and_energy_split():
    analyzer = FourierAnalyzer()
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

    features = analyzer.compute_frequency_features(image)

    assert features.shape == (5,)
    assert np.isclose(np.sum(features[:3]), 1.0, atol=1e-5)
    assert 0.0 <= features[3] <= 1.0
    assert 0.0 <= features[4] <= 1.0


def test_filter_redundant_removes_duplicate_frequency_signatures():
    analyzer = FourierAnalyzer(similarity_threshold=0.98)

    base_image = np.zeros((16, 16, 3), dtype=np.uint8)
    vertical_stripes = base_image.copy()
    vertical_stripes[:, ::2] = 255

    features_a = analyzer.compute_frequency_features(base_image)
    features_b = analyzer.compute_frequency_features(base_image)  # identical to features_a
    features_c = analyzer.compute_frequency_features(vertical_stripes)

    features = np.vstack([features_a, features_b, features_c])

    selected_indices = analyzer.filter_redundant(features, threshold=0.97)

    assert len(selected_indices) == 2
    assert set(selected_indices) == {0, 2}
