"""Prepared for Eden: these tests must not be executed locally in this turn."""

import numpy as np
import pytest
from PIL import Image

from SmallViTAblation.experiments.dobor_cech_adaptacyjny.cechy import (
    extract_features,
    spatial_descriptor,
)


def test_measures_bbox_and_roi_without_inventing_optional_annotations():
    row = {"width": 100, "height": 80, "annotations": [{"bbox": [10, 20, 40, 20]}]}
    values = extract_features(row, Image.new("RGB", (100, 80), (255, 255, 255)))
    assert values["bbox_area_fraction"] == pytest.approx(0.1)
    assert values["bbox_aspect_ratio"] == pytest.approx(2.0)
    assert values["bbox_center_x"] == pytest.approx(0.3)
    assert values["bbox_center_y"] == pytest.approx(0.375)
    assert values["bbox_border_distance"] == pytest.approx(0.1)
    assert values["roi_brightness"] == pytest.approx(1.0)
    assert values["roi_glare_fraction"] == pytest.approx(1.0)
    assert values["orientation_deg"] is None
    assert values["occlusion_fraction"] is None
    assert values["tip_visible"] is None


def test_explicit_annotations_preserve_zero_and_do_not_change_measured_geometry():
    row = {"width": 8, "height": 8, "bbox": [0, 0, 4, 4]}
    extra = {"orientation_deg": 0, "occlusion_fraction": 0, "tip_visible": False}
    values = extract_features(row, Image.new("L", (8, 8), 0), extra)
    assert [values[key] for key in extra] == [0.0, 0.0, 0.0]
    assert values["bbox_area_fraction"] == 0.25


def test_missing_bbox_does_not_look_like_a_zero_sized_visible_instrument():
    row = {"width": 8, "height": 8, "annotations": []}
    values = extract_features(row, Image.new("RGB", (8, 8)))
    assert values["bbox_area_fraction"] is None
    assert values["roi_brightness"] is None
    assert np.isnan(spatial_descriptor(np.ones((4, 2)), row, (2, 2))).all()


def test_ambiguous_targets_are_rejected_in_both_feature_views():
    row = {"width": 8, "height": 8, "annotations": [{"bbox": [0, 0, 2, 2]}, {"bbox": [4, 4, 2, 2]}]}
    with pytest.raises(ValueError, match="multiple"):
        extract_features(row, Image.new("RGB", (8, 8)))
    with pytest.raises(ValueError, match="multiple"):
        spatial_descriptor(np.ones((4, 2)), row, (2, 2))


def test_spatial_moments_distinguish_identical_mean_at_different_locations():
    tokens = np.array([[2.0], [2.0], [9.0], [9.0]])
    left = {"width": 8, "height": 8, "bbox": [0, 0, 4, 4]}
    right = {"width": 8, "height": 8, "bbox": [4, 0, 4, 4]}
    np.testing.assert_allclose(spatial_descriptor(tokens, left, (2, 2)), [2, -1, -1])
    np.testing.assert_allclose(spatial_descriptor(tokens, right, (2, 2)), [2, 1, -1])


def test_small_bbox_uses_nearest_patch_without_reading_outside_token_grid():
    row = {"width": 8, "height": 8, "bbox": [0, 0, 0.25, 0.25]}
    np.testing.assert_allclose(spatial_descriptor(np.array([[3.0], [7.0], [8.0], [9.0]]), row, (2, 2)), [3, -1.5, -1.5])
    with pytest.raises(ValueError, match="grid"):
        spatial_descriptor(np.ones((3, 1)), row, (2, 2))


def test_blur_proxy_increases_when_roi_is_uniform_and_dimensions_are_checked():
    row = {"width": 8, "height": 8, "bbox": [0, 0, 8, 8]}
    checker = Image.fromarray(((np.indices((8, 8)).sum(axis=0) % 2) * 255).astype("uint8"))
    uniform = extract_features(row, Image.new("L", (8, 8), 128))
    sharp = extract_features(row, checker)
    assert uniform["roi_blur"] > sharp["roi_blur"]
    with pytest.raises(ValueError, match="dimension"):
        extract_features(row, Image.new("L", (16, 16)))


@pytest.mark.parametrize("extra", [{"occlusion_fraction": 2}, {"tip_visible": 0.3}, {"orientation_deg": float("inf")}])
def test_invalid_explicit_annotations_are_not_silently_clamped(extra):
    with pytest.raises(ValueError):
        extract_features({"width": 8, "height": 8, "bbox": [0, 0, 8, 8]}, Image.new("L", (8, 8)), extra)
