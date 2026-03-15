"""Log parser for YOLO and DETR training output lines."""

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------

# YOLO ultralytics-style training output
_YOLO_LOSS_RE = re.compile(
    r"box_loss:\s*(?P<box_loss>\d+\.?\d*)\s+"
    r"cls_loss:\s*(?P<cls_loss>\d+\.?\d*)"
)
_YOLO_MAP_RE = re.compile(
    r"mAP50-95[:\s]+(?P<map50_95>\d+\.?\d*)"
)
_YOLO_EPOCH_RE = re.compile(
    r"(?:Epoch\s+)?(?P<epoch>\d+)/(?P<total>\d+)"
)

# DETR-style training output
_DETR_EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)(?:/(?P<total>\d+))?"
)
_DETR_LOSS_RE = re.compile(
    r"avg_loss[=:]\s*(?P<avg_loss>\d+\.?\d*)"
)
_DETR_LR_RE = re.compile(
    r"lr[=:]\s*(?P<lr>\d+\.?\d*(?:e[+-]?\d+)?)"
)
_DETR_VAL_LOSS_RE = re.compile(
    r"val(?:idation)?_loss[=:]\s*(?P<val_loss>\d+\.?\d*)"
)

# Generic progress
_GENERIC_PROGRESS_RE = re.compile(
    r"(?P<current>\d+)/(?P<total>\d+)\s+(?:frames|images|samples|steps)"
)


class LogParser:
    """Stateless parser that extracts structured data from training logs.

    Usage::

        parser = LogParser()
        info = parser.parse_line("Epoch 5 avg_loss=0.0032 lr=1e-6")
        # info == {"type": "detr", "epoch": 5, "avg_loss": 0.0032, "lr": 1e-06}
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_line(self, line: str) -> Optional[dict]:
        """Try to extract training metrics from *line*.

        Returns a dict with ``type`` key set to ``"yolo"`` or ``"detr"``
        and parsed metric values, or ``None`` if the line is not
        recognized.
        """
        result = self.parse_yolo_line(line)
        if result is not None:
            return result
        result = self.parse_detr_line(line)
        if result is not None:
            return result
        return None

    def parse_yolo_line(self, line: str) -> Optional[dict]:
        """Parse a YOLO-style training log line.

        Expected fields: ``box_loss``, ``cls_loss``, ``mAP50_95``,
        ``epoch``, ``total_epochs``.
        """
        loss_match = _YOLO_LOSS_RE.search(line)
        if loss_match is None:
            return None

        info: dict = {"type": "yolo"}
        info["box_loss"] = float(loss_match.group("box_loss"))
        info["cls_loss"] = float(loss_match.group("cls_loss"))

        map_match = _YOLO_MAP_RE.search(line)
        if map_match:
            info["mAP50_95"] = float(map_match.group("map50_95"))

        epoch_match = _YOLO_EPOCH_RE.search(line)
        if epoch_match:
            info["epoch"] = int(epoch_match.group("epoch"))
            info["total_epochs"] = int(epoch_match.group("total"))

        return info

    def parse_detr_line(self, line: str) -> Optional[dict]:
        """Parse a DETR-style training log line.

        Expected fields: ``epoch``, ``avg_loss``, ``lr``, ``val_loss``.
        """
        epoch_match = _DETR_EPOCH_RE.search(line)
        loss_match = _DETR_LOSS_RE.search(line)

        if epoch_match is None and loss_match is None:
            return None

        info: dict = {"type": "detr"}

        if epoch_match:
            info["epoch"] = int(epoch_match.group("epoch"))
            if epoch_match.group("total"):
                info["total_epochs"] = int(epoch_match.group("total"))

        if loss_match:
            info["avg_loss"] = float(loss_match.group("avg_loss"))

        lr_match = _DETR_LR_RE.search(line)
        if lr_match:
            info["lr"] = float(lr_match.group("lr"))

        val_match = _DETR_VAL_LOSS_RE.search(line)
        if val_match:
            info["val_loss"] = float(val_match.group("val_loss"))

        return info

    def detect_format(self, lines: List[str]) -> str:
        """Auto-detect the training format from a batch of log lines.

        Returns ``"yolo"``, ``"detr"``, or ``"unknown"``.
        """
        yolo_score = 0
        detr_score = 0

        for line in lines:
            if _YOLO_LOSS_RE.search(line):
                yolo_score += 1
            if _DETR_EPOCH_RE.search(line) or _DETR_LOSS_RE.search(line):
                detr_score += 1

        if yolo_score == 0 and detr_score == 0:
            return "unknown"
        return "yolo" if yolo_score >= detr_score else "detr"
