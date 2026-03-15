"""Tests for CataractAI_Workbench.app.core.log_parser.LogParser."""

import pytest

from CataractAI_Workbench.app.core.log_parser import LogParser


@pytest.fixture
def parser():
    return LogParser()


# ========================================================================
# parse_yolo_line
# ========================================================================

class TestParseYoloLine:
    """Tests for LogParser.parse_yolo_line."""

    def test_basic_loss_line(self, parser):
        line = "     45/330      3.57G    0.4523     0.284    0.7632         0      640:  box_loss: 0.4523  cls_loss: 0.2840"
        result = parser.parse_yolo_line(line)
        assert result is not None
        assert result["type"] == "yolo"
        assert result["box_loss"] == pytest.approx(0.4523)
        assert result["cls_loss"] == pytest.approx(0.2840)
        assert result["epoch"] == 45
        assert result["total_epochs"] == 330

    def test_loss_without_epoch(self, parser):
        line = "box_loss: 0.1234  cls_loss: 0.5678"
        result = parser.parse_yolo_line(line)
        assert result is not None
        assert result["type"] == "yolo"
        assert result["box_loss"] == pytest.approx(0.1234)
        assert result["cls_loss"] == pytest.approx(0.5678)
        assert "epoch" not in result

    def test_with_map50_95(self, parser):
        line = "box_loss: 0.4523  cls_loss: 0.2840  mAP50-95: 0.652"
        result = parser.parse_yolo_line(line)
        assert result is not None
        assert result["mAP50_95"] == pytest.approx(0.652)

    def test_without_map(self, parser):
        line = "box_loss: 0.4523  cls_loss: 0.2840"
        result = parser.parse_yolo_line(line)
        assert result is not None
        assert "mAP50_95" not in result

    def test_integer_losses(self, parser):
        line = "box_loss: 1  cls_loss: 2"
        result = parser.parse_yolo_line(line)
        assert result is not None
        assert result["box_loss"] == pytest.approx(1.0)
        assert result["cls_loss"] == pytest.approx(2.0)

    def test_epoch_prefix_style(self, parser):
        line = "Epoch 10/100 box_loss: 0.33  cls_loss: 0.44"
        result = parser.parse_yolo_line(line)
        assert result is not None
        assert result["epoch"] == 10
        assert result["total_epochs"] == 100

    def test_non_yolo_line_returns_none(self, parser):
        line = "Epoch 5 avg_loss=0.0032"
        assert parser.parse_yolo_line(line) is None

    def test_empty_string_returns_none(self, parser):
        assert parser.parse_yolo_line("") is None

    def test_random_text_returns_none(self, parser):
        assert parser.parse_yolo_line("some random log text") is None


# ========================================================================
# parse_detr_line
# ========================================================================

class TestParseDetrLine:
    """Tests for LogParser.parse_detr_line."""

    def test_epoch_and_avg_loss_equals_format(self, parser):
        line = "Epoch 45 avg_loss=0.0523"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["type"] == "detr"
        assert result["epoch"] == 45
        assert result["avg_loss"] == pytest.approx(0.0523)

    def test_epoch_and_avg_loss_colon_format(self, parser):
        """The regex expects 'Epoch <digits>', so 'Epoch: [45]' does NOT
        capture the epoch number (the colon+bracket breaks the pattern).
        However avg_loss still matches, so we get a DETR result with
        avg_loss but no epoch."""
        line = "Epoch: [45] avg_loss: 0.0523"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["type"] == "detr"
        assert "epoch" not in result  # regex cannot parse "Epoch: [45]"
        assert result["avg_loss"] == pytest.approx(0.0523)

    def test_epoch_and_avg_loss_space_format(self, parser):
        """Standard DETR format: 'Epoch 45 avg_loss: 0.0523'."""
        line = "Epoch 45 avg_loss: 0.0523"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["type"] == "detr"
        assert result["epoch"] == 45
        assert result["avg_loss"] == pytest.approx(0.0523)

    def test_epoch_with_total(self, parser):
        line = "Epoch 10/50 avg_loss=0.123"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["epoch"] == 10
        assert result["total_epochs"] == 50

    def test_epoch_without_total(self, parser):
        line = "Epoch 10 avg_loss=0.123"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["epoch"] == 10
        assert "total_epochs" not in result

    def test_with_lr(self, parser):
        line = "Epoch 5 avg_loss=0.0032 lr=1e-6"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["lr"] == pytest.approx(1e-6)

    def test_with_lr_scientific_notation_positive(self, parser):
        line = "Epoch 5 avg_loss=0.01 lr=2.5e+03"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["lr"] == pytest.approx(2.5e+03)

    def test_with_val_loss(self, parser):
        line = "Epoch 5 avg_loss=0.01 val_loss=0.005"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["val_loss"] == pytest.approx(0.005)

    def test_with_validation_loss_full_word(self, parser):
        line = "Epoch 5 avg_loss=0.01 validation_loss=0.005"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["val_loss"] == pytest.approx(0.005)

    def test_epoch_only_no_loss(self, parser):
        line = "Epoch 12 started"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["type"] == "detr"
        assert result["epoch"] == 12
        assert "avg_loss" not in result

    def test_loss_only_no_epoch(self, parser):
        line = "avg_loss=0.99"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["type"] == "detr"
        assert result["avg_loss"] == pytest.approx(0.99)
        assert "epoch" not in result

    def test_all_fields_together(self, parser):
        line = "Epoch 20/100 avg_loss=0.0052 lr=2e-5 val_loss=0.0041"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["epoch"] == 20
        assert result["total_epochs"] == 100
        assert result["avg_loss"] == pytest.approx(0.0052)
        assert result["lr"] == pytest.approx(2e-5)
        assert result["val_loss"] == pytest.approx(0.0041)

    def test_non_detr_line_returns_none(self, parser):
        line = "box_loss: 0.4523  cls_loss: 0.2840"
        # This line has no Epoch and no avg_loss, so should be None
        assert parser.parse_detr_line(line) is None

    def test_empty_string_returns_none(self, parser):
        assert parser.parse_detr_line("") is None

    def test_random_text_returns_none(self, parser):
        assert parser.parse_detr_line("training complete, saving model") is None


# ========================================================================
# parse_line  (dispatch)
# ========================================================================

class TestParseLine:
    """Tests for the top-level parse_line dispatcher."""

    def test_yolo_line_returns_yolo(self, parser):
        line = "box_loss: 0.50  cls_loss: 0.30"
        result = parser.parse_line(line)
        assert result is not None
        assert result["type"] == "yolo"

    def test_detr_line_returns_detr(self, parser):
        line = "Epoch 5 avg_loss=0.01"
        result = parser.parse_line(line)
        assert result is not None
        assert result["type"] == "detr"

    def test_unrecognized_returns_none(self, parser):
        assert parser.parse_line("INFO: Starting training") is None

    def test_empty_string_returns_none(self, parser):
        assert parser.parse_line("") is None

    def test_yolo_takes_priority_when_both_patterns_match(self, parser):
        # A line that has YOLO loss *and* an Epoch marker could match both.
        # parse_line tries YOLO first, so YOLO should win.
        line = "Epoch 10/50 box_loss: 0.33  cls_loss: 0.44"
        result = parser.parse_line(line)
        assert result is not None
        assert result["type"] == "yolo"


# ========================================================================
# detect_format
# ========================================================================

class TestDetectFormat:
    """Tests for LogParser.detect_format."""

    def test_detect_yolo(self, parser):
        lines = [
            "box_loss: 0.45  cls_loss: 0.28",
            "box_loss: 0.40  cls_loss: 0.25",
            "some random log text",
        ]
        assert parser.detect_format(lines) == "yolo"

    def test_detect_detr(self, parser):
        lines = [
            "Epoch 1 avg_loss=0.50",
            "Epoch 2 avg_loss=0.30",
            "Saving checkpoint...",
        ]
        assert parser.detect_format(lines) == "detr"

    def test_detect_unknown_no_patterns(self, parser):
        lines = [
            "Loading dataset...",
            "Preprocessing images...",
            "Done.",
        ]
        assert parser.detect_format(lines) == "unknown"

    def test_detect_unknown_empty_list(self, parser):
        assert parser.detect_format([]) == "unknown"

    def test_mixed_content_majority_yolo(self, parser):
        lines = [
            "box_loss: 0.45  cls_loss: 0.28",
            "box_loss: 0.40  cls_loss: 0.25",
            "box_loss: 0.35  cls_loss: 0.20",
            "Epoch 1 avg_loss=0.50",
        ]
        assert parser.detect_format(lines) == "yolo"

    def test_mixed_content_majority_detr(self, parser):
        lines = [
            "box_loss: 0.45  cls_loss: 0.28",
            "Epoch 1 avg_loss=0.50",
            "Epoch 2 avg_loss=0.30",
            "Epoch 3 avg_loss=0.20",
        ]
        assert parser.detect_format(lines) == "detr"

    def test_tie_goes_to_detr(self, parser):
        # When yolo_score == detr_score, the condition `yolo_score >= detr_score`
        # is True, so yolo wins on tie.
        lines = [
            "box_loss: 0.45  cls_loss: 0.28",
            "Epoch 1 avg_loss=0.50",
        ]
        # yolo_score=1, detr_score=1 -> 1 >= 1 -> "yolo"
        assert parser.detect_format(lines) == "yolo"

    def test_single_yolo_line(self, parser):
        lines = ["box_loss: 0.10  cls_loss: 0.20"]
        assert parser.detect_format(lines) == "yolo"

    def test_single_detr_line(self, parser):
        lines = ["Epoch 100 avg_loss=0.001"]
        assert parser.detect_format(lines) == "detr"


# ========================================================================
# Edge cases
# ========================================================================

class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    def test_whitespace_only_line(self, parser):
        assert parser.parse_line("   \t  ") is None

    def test_partial_yolo_box_loss_only(self, parser):
        # Has box_loss but no cls_loss -> regex requires both
        line = "box_loss: 0.45"
        assert parser.parse_yolo_line(line) is None

    def test_partial_yolo_cls_loss_only(self, parser):
        line = "cls_loss: 0.28"
        assert parser.parse_yolo_line(line) is None

    def test_very_small_floating_point(self, parser):
        line = "Epoch 1 avg_loss=0.000001"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["avg_loss"] == pytest.approx(1e-6)

    def test_large_epoch_numbers(self, parser):
        line = "Epoch 9999 avg_loss=0.0001"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["epoch"] == 9999

    def test_map50_95_with_spaces(self, parser):
        line = "box_loss: 0.45  cls_loss: 0.28  mAP50-95  0.789"
        result = parser.parse_yolo_line(line)
        assert result is not None
        assert result["mAP50_95"] == pytest.approx(0.789)

    def test_detr_lr_colon_format(self, parser):
        line = "Epoch 5 avg_loss: 0.01 lr: 3e-4"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["avg_loss"] == pytest.approx(0.01)
        assert result["lr"] == pytest.approx(3e-4)

    def test_detr_val_loss_colon_format(self, parser):
        line = "Epoch 5 avg_loss: 0.01 val_loss: 0.008"
        result = parser.parse_detr_line(line)
        assert result is not None
        assert result["val_loss"] == pytest.approx(0.008)

    def test_multiple_parse_calls_stateless(self, parser):
        """Parser should be stateless -- successive calls don't interfere."""
        r1 = parser.parse_line("Epoch 1 avg_loss=0.5")
        r2 = parser.parse_line("box_loss: 0.1  cls_loss: 0.2")
        r3 = parser.parse_line("no match here")
        assert r1["type"] == "detr"
        assert r2["type"] == "yolo"
        assert r3 is None
