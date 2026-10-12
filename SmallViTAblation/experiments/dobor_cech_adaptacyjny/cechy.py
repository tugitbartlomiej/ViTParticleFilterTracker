"""Measured image traits and a separate, weak-GTbbox diagnostic descriptor.

Rows contain width/height and either a COCO ``bbox`` (absolute xywh) or
``annotations`` already filtered to the target instrument. Multiple target boxes
are rejected; no annotation-order-dependent choice is made. Missing target boxes
produce missing features, never zeros. Pixels outside the image are clipped.

The descriptor here is only for feature probes. It MUST NOT replace the raw
single-patch particles fitted by Koopman--GLE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


FEATURE_NAMES = (
    "bbox_area_fraction",
    "bbox_aspect_ratio",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_border_distance",
    "roi_brightness",
    "roi_blur",
    "roi_glare_fraction",
    "orientation_deg",
    "occlusion_fraction",
    "tip_visible",
)
OPTIONAL_FEATURE_NAMES = ("orientation_deg", "occlusion_fraction", "tip_visible")


def _dimensions(row: dict, image_size: tuple[int, int] | None = None) -> tuple[int, int]:
    width = row.get("width", image_size[0] if image_size else None)
    height = row.get("height", image_size[1] if image_size else None)
    if width is None or height is None:
        raise ValueError("Image dimensions width and height are required")
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be finite and positive")
    if int(width) != width or int(height) != height:
        raise ValueError("Image dimensions must be integers")
    if image_size is not None and (width, height) != image_size:
        raise ValueError("Row dimensions differ from the decoded image dimensions")
    return int(width), int(height)


def _bbox(row: dict, width: int, height: int) -> tuple[float, float, float, float] | None:
    if "annotations" in row:
        annotations = row["annotations"]
        if not isinstance(annotations, (list, tuple)):
            raise ValueError("annotations must be a list of target annotations")
        if len(annotations) > 1:
            raise ValueError("Ambiguous multiple target annotations: exactly one target is supported")
        source = annotations[0].get("bbox") if annotations else None
        if row.get("bbox") is not None:
            if source is None or not np.array_equal(np.asarray(source), np.asarray(row["bbox"])):
                raise ValueError("Conflicting bbox and annotations fields")
    else:
        source = row.get("bbox")
    if source is None:
        return None
    box = np.asarray(source, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
        raise ValueError("A bbox must contain finite xywh with positive width and height")
    x0, y0 = max(0.0, float(box[0])), max(0.0, float(box[1]))
    x1, y1 = min(float(width), float(box[0] + box[2])), min(float(height), float(box[1] + box[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The target bbox does not intersect the image")
    return x0, y0, x1, y1


def _explicit_annotations(row: dict, extra: dict | None) -> dict[str, float | None]:
    # Only explicitly named annotations are accepted, never aspect-ratio heuristics.
    source = {key: row.get(key) for key in OPTIONAL_FEATURE_NAMES}
    nested = row.get("features") or {}
    source.update({key: nested[key] for key in OPTIONAL_FEATURE_NAMES if key in nested})
    if extra is not None:
        source.update({key: extra[key] for key in OPTIONAL_FEATURE_NAMES if key in extra})
    result: dict[str, float | None] = {}
    for key, value in source.items():
        if value is None:
            result[key] = None
            continue
        value = float(value)
        if np.isnan(value):
            result[key] = None
            continue
        if not np.isfinite(value):
            raise ValueError(f"Explicit {key} must be finite or missing")
        if key == "occlusion_fraction" and not 0 <= value <= 1:
            raise ValueError("occlusion_fraction must be in [0, 1]")
        if key == "tip_visible" and value not in (0.0, 1.0):
            raise ValueError("tip_visible must be a boolean or 0/1")
        result[key] = value
    return result


def extract_features(row: dict, image: Image.Image, extra: dict | None = None) -> dict[str, float | None]:
    """Return measured traits and explicit optional annotations in FEATURE_NAMES.

    Geometry uses the clipped target bbox. Border distance is the smallest
    normalized gap between any box edge and its image edge. ROI brightness is
    mean luminance / 255. Glare is the fraction of ROI luminance >= 0.95 and is
    only a brightness proxy. ``roi_blur = 1 / (1 + 100 * variance(Laplacian))``;
    larger values mean less high-frequency structure, not a clinical blur label.
    ROIs smaller than three pixels on either axis have missing blur.
    """
    width, height = _dimensions(row, image.size)
    box = _bbox(row, width, height)
    result: dict[str, float | None] = dict.fromkeys(FEATURE_NAMES)
    result.update(_explicit_annotations(row, extra))
    if box is None:
        return result
    x0, y0, x1, y1 = box
    result.update(
        bbox_area_fraction=(x1 - x0) * (y1 - y0) / (width * height),
        bbox_aspect_ratio=(x1 - x0) / (y1 - y0),
        bbox_center_x=(x0 + x1) / (2 * width),
        bbox_center_y=(y0 + y1) / (2 * height),
        bbox_border_distance=min(x0 / width, y0 / height, (width - x1) / width, (height - y1) / height),
    )
    crop = image.crop((int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1)))).convert("L")
    gray = np.asarray(crop, dtype=np.float64) / 255.0
    result["roi_brightness"] = float(gray.mean())
    result["roi_glare_fraction"] = float(np.mean(gray >= 0.95))
    if min(gray.shape) >= 3:
        laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
        result["roi_blur"] = float(1.0 / (1.0 + 100.0 * np.var(laplacian)))
    return result


def spatial_descriptor(tokens: np.ndarray, row: dict, grid_shape: tuple[int, int]) -> np.ndarray:
    """Concatenate selected-token mean and first x/y moments (shape ``3*C``).

    Tokens are row-major, one raw patch per row. Select patch centers inside the
    target bbox; for a tiny box containing no center use its closest patch center.
    Positions are global image coordinates normalized to [-1, 1], preserving
    location even for a single selected patch. Missing boxes return all NaNs.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[1] == 0 or not np.isfinite(tokens).all():
        raise ValueError("tokens must be a finite [patches, channels] matrix")
    if len(grid_shape) != 2 or any(int(v) != v or v <= 0 for v in grid_shape):
        raise ValueError("grid_shape must contain positive integer height and width")
    gh, gw = map(int, grid_shape)
    if gh * gw != len(tokens):
        raise ValueError("Token count does not match grid_shape")
    width, height = _dimensions(row)
    box = _bbox(row, width, height)
    if box is None:
        return np.full(3 * tokens.shape[1], np.nan, dtype=np.float64)
    yy, xx = np.meshgrid((np.arange(gh) + 0.5) / gh, (np.arange(gw) + 0.5) / gw, indexing="ij")
    xx, yy = xx.ravel(), yy.ravel()
    x0, y0, x1, y1 = box
    mask = (xx >= x0 / width) & (xx < x1 / width) & (yy >= y0 / height) & (yy < y1 / height)
    if not mask.any():
        # Distance in pixels avoids bias toward one axis on non-square images.
        distance = (xx * width - (x0 + x1) / 2) ** 2 + (yy * height - (y0 + y1) / 2) ** 2
        mask[int(np.argmin(distance))] = True
    selected = tokens[mask]
    return np.concatenate((selected.mean(axis=0), (selected * (2 * xx[mask, None] - 1)).mean(axis=0), (selected * (2 * yy[mask, None] - 1)).mean(axis=0)))
