"""Candidate retrieval contracts; ready for the Eden test job."""

import numpy as np
import pytest

from SmallViTAblation.experiments.dobor_cech_adaptacyjny.selekcja import retrieve_candidates


def dark_need():
    return {"need_id": "dark", "priority": 1.0, "conditions": [{"feature": "brightness", "feature_index": 0, "lower": None, "upper": 0.3, "lower_inclusive": False, "upper_inclusive": True}]}


def test_retrieves_required_traits_and_does_not_match_missing_values():
    values = np.array([[0.1], [np.nan], [0.2], [0.8], [0.9]])
    indices, reasons = retrieve_candidates(values, ["brightness"], [dark_need()], ["a", "b", "c", "d", "e"], limit=2, exploration_fraction=0, seed=42)
    assert set(indices.tolist()) == {0, 2}
    assert all(reason["selection_reason"] == "feature_match" for reason in reasons)
    assert all(reason["need_ids"] == ["dark"] for reason in reasons)


def test_exact_budget_has_explicit_exploration_and_labeled_random_shortfall():
    values = np.array([[0.1], [np.nan], [0.8], [0.9], [1.0]])
    indices, reasons = retrieve_candidates(values, ["brightness"], [dark_need()], list("abcde"), limit=4, exploration_fraction=0.25, seed=17)
    assert len(indices) == len(set(indices.tolist())) == 4
    assert [reason["selection_reason"] for reason in reasons] == ["feature_match", "exploration", "random_fallback", "random_fallback"]
    assert reasons[0]["frame_id"] == "a"
    again, again_reasons = retrieve_candidates(values, ["brightness"], [dark_need()], list("abcde"), limit=4, exploration_fraction=0.25, seed=17)
    np.testing.assert_array_equal(indices, again)
    assert reasons == again_reasons


def test_no_needs_abstains_from_trait_claims_and_still_returns_budget():
    indices, reasons = retrieve_candidates(np.full((5, 1), np.nan), ["brightness"], [], list("abcde"), limit=3, exploration_fraction=0, seed=42)
    assert len(indices) == 3
    assert all(reason["selection_reason"] == "random_fallback" for reason in reasons)
    assert all(reason["need_ids"] == [] for reason in reasons)


def test_two_trait_query_requires_both_traits_to_be_measured_and_match():
    need = dark_need()
    need["conditions"].append({"feature": "size", "feature_index": 1, "lower": 0.5, "upper": None, "lower_inclusive": False, "upper_inclusive": False})
    values = np.array([[0.1, 0.8], [0.1, np.nan], [0.1, 0.2], [0.8, 0.8]])
    indices, _ = retrieve_candidates(values, ["brightness", "size"], [need], list("abcd"), limit=1, exploration_fraction=0, seed=1)
    assert indices.tolist() == [0]


def test_duplicate_frame_ids_and_impossible_budget_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        retrieve_candidates(np.zeros((2, 1)), ["brightness"], [], ["a", "a"], limit=2, exploration_fraction=0, seed=42)
    with pytest.raises(ValueError, match="limit"):
        retrieve_candidates(np.zeros((2, 1)), ["brightness"], [], ["a", "b"], limit=3, exploration_fraction=0, seed=42)
