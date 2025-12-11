import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from AdvancedDatasetSelection.selection_methods.combined_selector import CombinedSelector
from AdvancedDatasetSelection.selection_methods.k_center_greedy import KCenterGreedy
from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer


class _DummyDINO:
    def compute_features_batch(self, image_paths, show_progress=True):  # pragma: no cover - simple stub
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(image_paths), 4)), image_paths


class _DummySAM:
    def compute_complexity_without_sam(self, image):  # pragma: no cover - simple stub
        return {"complexity_score": float(np.mean(image) / 255.0)}


class _DummyEL2N:
    pass


class _DummySelector(CombinedSelector):
    def __init__(self):
        super().__init__(
            fourier_analyzer=FourierAnalyzer(similarity_threshold=0.9),
            dino_extractor=_DummyDINO(),
            sam_extractor=_DummySAM(),
            el2n_scorer=_DummyEL2N(),
            k_center=KCenterGreedy(distance_metric="euclidean"),
        )

    def extract_all_features(self, image_paths, labels=None, use_cache=True, show_progress=True):  # pragma: no cover - controlled in tests
        rng = np.random.default_rng(2)
        self.fourier_features = rng.normal(size=(len(image_paths), 5))
        self.dino_features = rng.normal(size=(len(image_paths), 4))
        self.sam_scores = rng.random(len(image_paths))
        self.el2n_scores = np.linspace(0.0, 1.0, len(image_paths))
        self.valid_paths = list(image_paths)
        return {
            "fourier_features": self.fourier_features,
            "dino_features": self.dino_features,
            "sam_scores": self.sam_scores,
            "el2n_scores": self.el2n_scores,
            "valid_paths": self.valid_paths,
        }


def test_stepwise_selection_with_manual_features():
    np.random.seed(0)
    selector = CombinedSelector(
        fourier_analyzer=FourierAnalyzer(similarity_threshold=0.95),
        dino_extractor=_DummyDINO(),
        sam_extractor=_DummySAM(),
        el2n_scorer=_DummyEL2N(),
        k_center=KCenterGreedy(distance_metric="euclidean"),
    )

    selector.fourier_features = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=float,
    )
    selector.valid_paths = [f"img_{i}.jpg" for i in range(6)]
    selector.dino_features = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2],
            [0.6, 0.5, 0.4, 0.3],
            [0.9, 1.0, 1.1, 1.2],
            [1.0, 1.1, 1.2, 1.3],
        ]
    )
    selector.sam_scores = np.linspace(0.0, 1.0, 6)
    selector.el2n_scores = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.6])

    step1_indices, step1_paths = selector.step1_fourier_prefilter(threshold=0.9)

    assert step1_indices == [0, 2, 4]
    assert step1_paths == ["img_0.jpg", "img_2.jpg", "img_4.jpg"]

    combined = selector.step2_combine_features(step1_indices)
    assert combined.shape == (3, selector.dino_features.shape[1] + 1)

    np.random.seed(1)
    step3_indices = selector.step3_k_center_select(
        combined_features=combined,
        original_indices=step1_indices,
        target_size=2,
        oversampling=1.0,
    )

    assert len(step3_indices) == 2
    assert set(step3_indices).issubset(set(step1_indices))

    final_indices = selector.step4_el2n_rank(step3_indices, target_size=1, keep_hard=True)
    assert len(final_indices) == 1
    assert final_indices[0] == max(step3_indices, key=lambda idx: selector.el2n_scores[idx])


def test_select_optimal_subset_with_dummy_extractors():
    selector = _DummySelector()
    image_paths = [f"image_{i}.jpg" for i in range(10)]

    selected_indices, selected_paths, stats = selector.select_optimal_subset(
        image_paths=image_paths,
        target_size=3,
        use_cache=False,
    )

    assert len(selected_indices) == 3
    assert len(selected_paths) == 3
    assert stats["original_count"] == len(image_paths)
    assert stats["final_count"] == 3
    assert set(selected_paths).issubset(set(image_paths))
