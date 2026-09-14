"""Wspólne funkcje ewaluacji (Agent 3, pipeline/eval/).

Metryki COCO przez pycocotools, dopasowanie TP/FP/FN, IO plików predykcji.
Wszystkie progi pochodzą z configs/protocol.yaml — brak zaszytych hiperparametrów.

Konwencja plików w katalogu wyników biegu:
  predictions_<set>.json        — predykcje COCO results (zapisane ZAWSZE, przed metrykami)
  evaluated_images_<set>.json   — lista obrazów faktycznie ocenionych + ścieżka anotacji
  metrics_<set>.json            — pojedynczy blok wyników (README.md sekcja 4.5)
  metrics.json / metrics__<rule>.json — złożone wyniki biegu
  operating_point.json          — próg max-F1 wybrany na walidacji + krzywe F1
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

BLOCK_FIELDS = [
    "mAP_0.5", "mAP_0.5:0.95", "mAP_0.75", "AR@100",
    "TP", "FP", "FN", "precision", "recall", "f1",
    "n_images", "n_annotations",
]


class PycocotoolsMissingError(RuntimeError):
    pass


def ensure_utf8_io() -> None:
    """Konsola Windows bywa cp1250 — wymuszamy UTF-8, żeby komunikaty nie padały."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def require_pycocotools():
    """Leniwy import pycocotools; przy braku czytelny błąd z instrukcją instalacji."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        return COCO, COCOeval
    except ImportError as exc:  # pragma: no cover
        raise PycocotoolsMissingError(
            "Brak pakietu pycocotools — metryki mAP wymagają go.\n"
            "Instalacja:  py -3.11 -m pip install pycocotools\n"
            "(zależności wypisane w pipeline/eval/requirements.txt)"
        ) from exc


# ---------------------------------------------------------------------------
# protocol.yaml
# ---------------------------------------------------------------------------

def load_protocol(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f)
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: protocol.yaml nie jest mapą")
    ev = protocol.get("evaluation", {})
    for key in ("map_conf_threshold", "iou_threshold"):
        if key not in ev:
            raise KeyError(f"{path}: brak wymaganego pola evaluation.{key}")
    return protocol


def protocol_field(protocol: dict, *path, default=...):
    """Pole z protocol.yaml po ścieżce; KeyError gdy brak i nie podano default."""
    node = protocol
    for key in path:
        if not isinstance(node, dict) or key not in node:
            if default is not ...:
                return default
            raise KeyError("protocol.yaml: brak pola " + ".".join(map(str, path)))
        node = node[key]
    return node


def project_root(protocol_path: str | Path) -> Path:
    """Katalog SmallViTAblation (rodzic configs/)."""
    return Path(protocol_path).resolve().parent.parent


def repo_root(protocol_path: str | Path) -> Path:
    """Korzeń repozytorium (rodzic SmallViTAblation)."""
    return project_root(protocol_path).parent


def resolve_protocol_relative(protocol_path: str | Path, value: str) -> Path:
    """Ścieżka z protocol.yaml względem korzenia repo; akceptuje też absolutne."""
    p = Path(value)
    if p.is_absolute():
        return p
    candidate = repo_root(protocol_path) / p
    if candidate.parent.exists():
        return candidate
    return (Path.cwd() / p).resolve()


# ---------------------------------------------------------------------------
# JSON / predykcje COCO
# ---------------------------------------------------------------------------

def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
        f.write("\n")
    return path


def load_predictions(path: str | Path) -> list[dict]:
    dt = load_json(path)
    if not isinstance(dt, list):
        raise ValueError(f"{path}: plik predykcji COCO musi być listą rekordów")
    for i, d in enumerate(dt):
        for key in ("image_id", "category_id", "bbox", "score"):
            if key not in d:
                raise ValueError(f"{path}: rekord #{i} bez pola '{key}'")
        if len(d["bbox"]) != 4:
            raise ValueError(f"{path}: rekord #{i} — bbox musi mieć 4 elementy [x,y,w,h]")
    return dt


def save_predictions(dt: list[dict], path: str | Path) -> Path:
    clean = [
        {
            "image_id": int(d["image_id"]),
            "category_id": int(d["category_id"]),
            "bbox": [float(v) for v in d["bbox"]],
            "score": float(d["score"]),
        }
        for d in dt
    ]
    return save_json(clean, path)


# ---------------------------------------------------------------------------
# Ground truth: przycinanie do obrazów faktycznie ocenionych
# ---------------------------------------------------------------------------

def load_gt_dataset(annotations_path: str | Path) -> dict:
    ds = load_json(annotations_path)
    for key in ("images", "annotations", "categories"):
        if key not in ds:
            raise ValueError(f"{annotations_path}: anotacje COCO bez klucza '{key}'")
    return ds


def filter_gt_to_images(gt: dict, image_ids: set[int]) -> dict:
    """GT przycięty do podanego zbioru image_id (obrazy faktycznie ocenione)."""
    unknown = image_ids - {int(im["id"]) for im in gt["images"]}
    if unknown:
        raise ValueError(f"Nieznane image_id w zbiorze ocenianym: {sorted(unknown)}")
    images = [im for im in gt["images"] if int(im["id"]) in image_ids]
    keep = {int(im["id"]) for im in images}
    annotations = [a for a in gt["annotations"] if int(a["image_id"]) in keep]
    return {**gt, "images": images, "annotations": annotations}


def evaluated_image_ids_from_dir(gt: dict, images_dir: str | Path) -> set[int]:
    """Identyfikatory obrazów z anotacji, których pliki ISTNIEJĄ w images_dir."""
    images_dir = Path(images_dir).resolve()
    candidates = [path.resolve() for path in images_dir.rglob("*")
                  if path.suffix.lower() in IMAGE_EXTENSIONS]
    by_relative = {path.relative_to(images_dir).as_posix().lower(): path
                   for path in candidates}
    by_basename: dict[str, list[Path]] = {}
    for path in candidates:
        by_basename.setdefault(path.name.lower(), []).append(path)
    ids = set()
    for im in gt["images"]:
        declared = Path(str(im["file_name"]))
        if declared.is_absolute() and declared.is_file():
            ids.add(int(im["id"]))
            continue
        relative = declared.as_posix().lstrip("./").lower()
        if relative in by_relative:
            ids.add(int(im["id"]))
            continue
        matches = by_basename.get(declared.name.lower(), [])
        if len(matches) > 1:
            raise ValueError(
                f"Niejednoznaczna nazwa COCO {declared.name!r}; zachowaj ścieżkę względną")
        if matches:
            ids.add(int(im["id"]))
    return ids


def build_gt_index(gt: dict) -> dict:
    """{image_id: {'boxes': (N,4) xywh float, 'cats': (N,) int}} dla dopasowania TP/FP."""
    index: dict[int, dict] = {}
    for im in gt["images"]:
        index[int(im["id"])] = {"boxes": np.zeros((0, 4)), "cats": np.zeros((0,), dtype=int)}
    for ann in gt["annotations"]:
        img_id = int(ann["image_id"])
        entry = index.setdefault(img_id, {"boxes": np.zeros((0, 4)), "cats": np.zeros((0,), dtype=int)})
        entry["boxes"] = np.vstack([entry["boxes"], np.asarray(ann["bbox"], dtype=float).reshape(1, 4)])
        entry["cats"] = np.append(entry["cats"], int(ann["category_id"]))
    return index


def remap_dt_categories_single_class(gt: dict, dt: list[dict]) -> tuple[list[dict], bool]:
    """Dla jednej klasy mapuje wyłącznie indeks modelu 0 na identyfikator GT.

    RF-DETR po finetuningu na 1 klasę zwraca class_id=0, a anotacje COCO mają
    zwykle id=1 — bez remapu wszystko byłoby FP. Zachowanie sterowane polem
    eval_agent3.evaluation.single_class_remap w protocol.yaml (tu: wywołujący
    decyduje, funkcja tylko wykonuje).
    """
    cats = {int(c["id"]) for c in gt["categories"]}
    if len(cats) != 1:
        return dt, False
    target = next(iter(cats))
    unknown = {int(d["category_id"]) for d in dt} - {0, target}
    if unknown:
        raise ValueError(f"Nieoczekiwane category_id: {sorted(unknown)}; "
                         f"single_class_remap dopuszcza tylko 0 lub {target}")
    remapped = [dict(d, category_id=target) for d in dt]
    return remapped, any(int(d["category_id"]) != target for d in dt)


# ---------------------------------------------------------------------------
# mAP przez pycocotools — pełna krzywa PR
# ---------------------------------------------------------------------------

def compute_map_metrics(gt: dict, dt: list[dict]) -> dict:
    """mAP@0.5, mAP@[0.5:0.95], mAP@0.75, AR@100 z pełnej krzywej PR (pycocotools).

    gt — przycięty dataset COCO; dt — lista rekordów COCO results.
    Wartości -1 (niezdefiniowane) zwracane jako 0.0.
    """
    COCO, COCOeval = require_pycocotools()
    gt_ids = {int(im["id"]) for im in gt["images"]}
    dt_ids = {int(d["image_id"]) for d in dt}
    if dt_ids - gt_ids:
        raise ValueError(f"Predykcje zawierają image_id spoza ocenianego GT: {sorted(dt_ids - gt_ids)}")
    zeros = {"mAP_0.5": 0.0, "mAP_0.5:0.95": 0.0, "mAP_0.75": 0.0, "AR@100": 0.0}
    if not gt["images"] or not gt["annotations"] or not dt:
        return zeros

    coco_gt = COCO()
    coco_gt.dataset = json.loads(json.dumps(gt))  # COCOeval mutuje dataset
    coco_gt.createIndex()

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            coco_dt = coco_gt.loadRes(dt)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.imgIds = sorted(gt_ids)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
    except Exception as exc:  # pragma: no cover — defensywnie
        raise RuntimeError(f"pycocotools COCOeval nie zadziałał: {exc}\n{buf.getvalue()}") from exc

    def clean(x) -> float:
        return 0.0 if x is None or float(x) < 0 else float(x)

    return {
        "mAP_0.5": clean(s[1]),
        "mAP_0.5:0.95": clean(s[0]),
        "mAP_0.75": clean(s[2]),
        "AR@100": clean(s[8]),
    }


def load_validation_operating_point(path: str | Path) -> tuple[float, str]:
    """Fail closed when threshold provenance is absent or names a test set."""
    op = load_json(path)
    if op.get("threshold_selected_on") != "val":
        raise ValueError("Próg wymaga pochodzenia z walidacji: threshold_selected_on='val'")
    threshold = float(op["operating_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Próg walidacyjny musi być skończony i należeć do [0, 1]")
    return threshold, "val"


# ---------------------------------------------------------------------------
# TP/FP/FN — greedy matching przy progu operacyjnym
# ---------------------------------------------------------------------------

def _iou_one_vs_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros(0)
    ax1, ay1, aw, ah = box
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1 = boxes[:, 0]
    by1 = boxes[:, 1]
    bx2 = bx1 + boxes[:, 2]
    by2 = by1 + boxes[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = aw * ah + boxes[:, 2] * boxes[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def _match_sorted_predictions(entry: dict, preds: list[dict], iou_thr: float) -> np.ndarray:
    """Mark TP once in stable score order; bound the temporary IoU matrix.

    Greedy assignment of a detection cannot change when lower-score detections
    are appended. Equal scores keep their input order, as in match_detections.
    """
    flags = np.zeros(len(preds), dtype=bool)
    boxes, cats = entry["boxes"], entry["cats"]
    if not preds or not boxes.shape[0]:
        return flags
    matched = np.zeros(boxes.shape[0], dtype=bool)
    remaining = boxes.shape[0]
    # Avoid a predictions x GT matrix for an entire crowded image.
    chunk_size = max(1, min(256, 65536 // boxes.shape[0]))
    for start in range(0, len(preds), chunk_size):
        chunk = preds[start:start + chunk_size]
        pb = np.asarray([p["bbox"] for p in chunk], dtype=float)
        pred_cats = np.asarray([int(p["category_id"]) for p in chunk])
        ix1 = np.maximum(pb[:, None, 0], boxes[None, :, 0])
        iy1 = np.maximum(pb[:, None, 1], boxes[None, :, 1])
        ix2 = np.minimum((pb[:, 0] + pb[:, 2])[:, None], boxes[:, 0] + boxes[:, 2])
        iy2 = np.minimum((pb[:, 1] + pb[:, 3])[:, None], boxes[:, 1] + boxes[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = (pb[:, 2] * pb[:, 3])[:, None] + boxes[:, 2] * boxes[:, 3] - inter
        ious = np.zeros_like(inter)
        np.divide(inter, union, out=ious, where=union > 0)
        same_cat = pred_cats[:, None] == cats[None, :]
        for offset, row in enumerate(ious):
            candidates = np.where(same_cat[offset] & ~matched, row, -1.0)
            best = int(np.argmax(candidates))
            if candidates[best] >= iou_thr:
                flags[start + offset] = True
                if not matched[best]:
                    remaining -= 1
                matched[best] = True
                # All later predictions are FP once every GT is claimed.
                if not remaining and iou_thr >= 0:
                    return flags
    return flags


def match_detections(gt_index: dict, dt: list[dict], conf_thr: float, iou_thr: float) -> dict:
    """Greedy matching per obraz (predykcje malejąco po score), kategoria musi się zgadzać.

    Zwraca: tp, fp, fn, tp_scores (ufności trafień), n_gt.
    """
    per_image: dict[int, list[dict]] = {}
    for d in dt:
        if float(d["score"]) >= conf_thr:
            per_image.setdefault(int(d["image_id"]), []).append(d)

    tp = fp = 0
    tp_scores: list[float] = []
    n_gt = 0
    for img_id, entry in gt_index.items():
        boxes = entry["boxes"]
        n_gt += int(boxes.shape[0])
        preds = per_image.get(img_id, [])
        if not preds:
            continue
        preds = sorted(preds, key=lambda d: -float(d["score"]))
        flags = _match_sorted_predictions(entry, preds, iou_thr)
        n_tp = int(flags.sum())
        tp += n_tp
        fp += len(preds) - n_tp
        tp_scores.extend(float(p["score"]) for p, hit in zip(preds, flags) if hit)
    fn = n_gt - tp
    return {"tp": tp, "fp": fp, "fn": fn, "tp_scores": tp_scores, "n_gt": n_gt}


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def counting_metrics(gt_index: dict, dt: list[dict], conf_thr: float, iou_thr: float) -> dict:
    m = match_detections(gt_index, dt, conf_thr, iou_thr)
    out = {"TP": int(m["tp"]), "FP": int(m["fp"]), "FN": int(m["fn"])}
    out.update(_prf(m["tp"], m["fp"], m["fn"]))
    out["n_annotations"] = int(m["n_gt"])
    return out


def threshold_candidates(scores: np.ndarray, max_points: int, extra: list[float] | None = None) -> np.ndarray:
    """Progi do krzywej F1: unikalne score'y predykcji (<= max_points) + skrajne + extra."""
    uniq = np.unique(np.asarray(scores, dtype=float))
    if uniq.size > max_points:
        idx = np.linspace(0, uniq.size - 1, max_points).round().astype(int)
        uniq = np.unique(uniq[idx])
    cands = np.concatenate([[0.0], uniq, [1.0]])
    if extra:
        cands = np.concatenate([cands, np.asarray(extra, dtype=float)])
    return np.unique(np.clip(cands, 0.0, 1.0))


def f1_curve(gt_index: dict, dt: list[dict], thresholds: np.ndarray, iou_thr: float) -> dict:
    """Exact F1 curve from one greedy assignment, independent of threshold count."""
    scores = np.asarray([float(d["score"]) for d in dt], dtype=float)
    order = np.argsort(-scores, kind="stable")
    scores_sorted = scores[order] if scores.size else scores
    negative_scores_sorted = -scores_sorted
    per_image: dict[int, list[int]] = {}
    eligible = np.zeros(len(dt), dtype=bool)
    hits = np.zeros(len(dt), dtype=bool)
    for rank, index in enumerate(order):
        img_id = int(dt[index]["image_id"])
        # The previous implementation matched each prefix with conf_thr=0.0
        # and ignored detections outside gt_index. Preserve both behaviours.
        if img_id in gt_index and scores_sorted[rank] >= 0.0:
            per_image.setdefault(img_id, []).append(rank)
            eligible[rank] = True
    for img_id, ranks in per_image.items():
        preds = [dt[order[rank]] for rank in ranks]
        hits[ranks] = _match_sorted_predictions(gt_index[img_id], preds, iou_thr)
    tp_prefix = np.concatenate(([0], np.cumsum(hits, dtype=np.int64)))
    total_prefix = np.concatenate(([0], np.cumsum(eligible, dtype=np.int64)))
    n_gt = sum(int(entry["boxes"].shape[0]) for entry in gt_index.values())
    cols = {"threshold": [], "precision": [], "recall": [], "f1": [], "TP": [], "FP": [], "FN": []}
    for thr in sorted(thresholds):
        k = int(np.searchsorted(negative_scores_sorted, -thr, side="right")) if scores_sorted.size else 0
        tp = int(tp_prefix[k])
        fp = int(total_prefix[k]) - tp
        fn = n_gt - tp
        prf = _prf(tp, fp, fn)
        cols["threshold"].append(float(thr))
        cols["TP"].append(tp)
        cols["FP"].append(fp)
        cols["FN"].append(fn)
        cols["precision"].append(prf["precision"])
        cols["recall"].append(prf["recall"])
        cols["f1"].append(prf["f1"])
    return cols


def best_f1_threshold(curve: dict) -> tuple[float, float]:
    """Próg maksymalizujący F1 (remisy: niższy próg). Zwraca (threshold, f1)."""
    if not curve["threshold"]:
        return 0.0, 0.0
    f1s = np.asarray(curve["f1"], dtype=float)
    best_i = int(np.argmax(f1s))
    return float(curve["threshold"][best_i]), float(f1s[best_i])


# ---------------------------------------------------------------------------
# Pacjenci
# ---------------------------------------------------------------------------

def count_patients(file_names: list[str], pool_manifest_path: str | Path | None = None):
    """Liczba pacjentów dla ocenionych obrazów (kontrakt: pacjent obok obrazów).

    Z pool_manifest.jsonl — po polu patient_id; inaczej heurystyka: katalog
    nadrzędny pliku (konwencja '<dataset>/<patient>/<frame>.jpg'). Brak danych → None.
    """
    if pool_manifest_path is not None:
        patients = set()
        names = {Path(str(n)).name.lower() for n in file_names}
        with open(pool_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if Path(str(rec.get("path", ""))).name.lower() in names:
                    pid = rec.get("patient_id")
                    if pid is not None:
                        patients.add(pid)
        return len(patients) if patients else None
    parents = set()
    for n in file_names:
        parent = Path(str(n)).parent.name
        if parent not in ("", "."):
            parents.add(parent)
    return len(parents) if parents else None
