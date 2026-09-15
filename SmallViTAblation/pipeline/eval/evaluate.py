#!/usr/bin/env python3
"""evaluate.py — inferencja detektora + metryki COCO (Agent 3, pipeline/eval/).

Zasady protokołu (agents/README.md §5, AGENT_3_EWALUACJA.md):
  1. Inferencja STANDARDOWA — zwykłe dekodowanie detektora, wszystkie predykcje
     powyżej progu. ŻADNEGO wybierania pojedynczego zapytania modelu.
  2. Próg ufności generowania predykcji = evaluation.map_conf_threshold (0.001).
  3. mAP przez pycocotools z pełnej krzywej PR: mAP@0.5, mAP@[0.5:0.95],
     mAP@0.75, AR@100.
  4. Predykcje ZAWSZE najpierw lądują na dysku (predictions_<set>.json);
     kolejne metryki liczone są z pliku, bez ponownej inferencji.
  5. Ground truth przycinany do obrazów faktycznie ocenionych.

Tryby:
  A) inferencja:   --checkpoint CKPT --annotations A --images-dir D --set val
  B) z pliku:      --predictions-file F --annotations A [--images-dir D] --set val
  C) złożenie:     --assemble-only (składa metrics.json z gotowych bloków w --output-dir)

Wyjście (w --output-dir, domyślnie <SmallViTAblation>/results/<run_id>):
  predictions_<set>.json, evaluated_images_<set>.json, metrics_<set>.json,
  metrics.json (format agents/README.md §4.5)

Przykład:
  py -3.11 pipeline/eval/evaluate.py --protocol configs/protocol.yaml \
      --run-id M-DESF__s42__b5000 --checkpoint-rule last_epoch --set val \
      --checkpoint results/.../last.pth --annotations data/val.json --images-dir data/images
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.eval import common  # noqa: E402

# Mapowanie protocol.yaml model.name -> klasa biblioteki rfdetr (import leniwy).
RFDETR_CLASS_BY_NAME = {
    "rf-detr-nano": "RFDETRNano",
    "rf-detr-small": "RFDETRSmall",
    "rf-detr-base": "RFDETRBase",
    "rf-detr-large": "RFDETRLarge",
    "rf-detr": "RFDETR",
}


# ---------------------------------------------------------------------------
# Inferencja (leniwy import rfdetr — tylko w tym miejscu)
# ---------------------------------------------------------------------------

def load_detector(checkpoint: Path, model_name: str, resolution: int | None, device: str | None):
    try:
        import rfdetr  # leniwy import: ciężka zależność tylko przy inferencji
    except ImportError as exc:
        raise RuntimeError(
            "Brak pakietu rfdetr (potrzebny TYLKO do inferencji z checkpointu).\n"
            "Instalacja:  py -3.11 -m pip install rfdetr\n"
            "Albo użyj trybu --predictions-file z gotowymi predykcjami COCO."
        ) from exc
    cls_name = RFDETR_CLASS_BY_NAME.get(model_name)
    if cls_name is None:
        raise ValueError(
            f"Nieznany model.name='{model_name}' w protocol.yaml; "
            f"oczekiwano jednej z: {sorted(RFDETR_CLASS_BY_NAME)}"
        )
    cls = getattr(rfdetr, cls_name)
    kwargs = {"pretrain_weights": str(checkpoint)}
    if resolution is not None:
        kwargs["resolution"] = int(resolution)
    if device:
        kwargs["device"] = device
    return cls(**kwargs)


def run_inference(model, image_items: list[tuple[Path, int]],
                  conf_thr: float) -> list[dict]:
    """Inferencja standardowa: model.predict + zwykłe dekodowanie, WSZYSTKIE
    predykcje powyżej conf_thr (= evaluation.map_conf_threshold)."""
    dt: list[dict] = []
    for path, img_id in image_items:
        det = model.predict(str(path), threshold=conf_thr)
        xyxy = getattr(det, "xyxy", [])
        conf = getattr(det, "confidence", [])
        cls_ids = getattr(det, "class_id", [])
        names = getattr(det, "data", {}).get("class_name")
        for index, ((x1, y1, x2, y2), score, cls_id) in enumerate(zip(xyxy, conf, cls_ids)):
            # RF-DETR 1.10 may return its no-object slot above a low threshold.
            # It labels that slot explicitly; it is not a foreground detection.
            if names is not None and names[index] == "__background__":
                continue
            x1, y1, x2, y2 = (float(v) for v in (x1, y1, x2, y2))
            dt.append({
                "image_id": img_id,
                "category_id": int(cls_id),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })
    return dt


# ---------------------------------------------------------------------------
# Predykcje + obrazy ocenione (sidecar)
# ---------------------------------------------------------------------------

def _sidecar_path(output_dir: Path, set_name: str) -> Path:
    return output_dir / f"evaluated_images_{set_name}.json"


def _predictions_path(output_dir: Path, set_name: str) -> Path:
    return output_dir / f"predictions_{set_name}.json"


def prepare_predictions(output_dir: Path, protocol: dict, set_name: str,
                        annotations: Path, *, checkpoint: Path | None = None,
                        images_dir: Path | None = None,
                        predictions_file: Path | None = None,
                        evaluated_images_file: Path | None = None,
                        device: str | None = None, model_name: str | None = None,
                        resolution: int | None = None,
                        max_images: int | None = None) -> dict:
    """Tryb A: inferencja -> predictions_<set>.json; tryb B: kopiowanie istniejącego pliku.

    Zawsze zapisuje na dysk predykcje ORAZ sidecar z listą obrazów faktycznie
    ocenionych (potrzebną do przycięcia GT — wymaganie twarde 5).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    conf_thr = float(common.protocol_field(protocol, "evaluation", "map_conf_threshold"))
    gt = common.load_gt_dataset(annotations)
    pred_path = _predictions_path(output_dir, set_name)

    if checkpoint is not None:
        if images_dir is None:
            raise ValueError("Tryb inferencji (--checkpoint) wymaga --images-dir")
        model_name = model_name or common.protocol_field(protocol, "model", "name")
        if resolution is None:
            resolution = common.protocol_field(protocol, "model", "resolution", default=None)
        root = Path(images_dir).resolve()
        candidates = [p.resolve() for p in root.rglob("*")
                      if p.suffix.lower() in common.IMAGE_EXTENSIONS]
        by_relative = {p.relative_to(root).as_posix().lower(): p for p in candidates}
        by_basename = {}
        for path in candidates:
            by_basename.setdefault(path.name.lower(), []).append(path)

        def resolve_image(file_name: str) -> Path | None:
            declared = Path(file_name)
            if declared.is_absolute() and declared.is_file():
                return declared.resolve()
            relative = declared.as_posix().lstrip("./").lower()
            if relative in by_relative:
                return by_relative[relative]
            matches = by_basename.get(declared.name.lower(), [])
            if len(matches) > 1:
                raise ValueError(
                    f"ambiguous COCO basename {declared.name!r}; preserve relative paths")
            return matches[0] if matches else None

        resolved = [(im, resolve_image(str(im["file_name"]))) for im in gt["images"]]
        images = sorted(((im, path) for im, path in resolved if path is not None),
                        key=lambda item: int(item[0]["id"]))
        if max_images is not None:
            images = images[:max_images]
        if not images:
            raise RuntimeError(
                f"Żaden obraz z anotacji {annotations} nie istnieje w {images_dir}"
            )
        image_items = [(path, int(im["id"])) for im, path in images]
        model = load_detector(checkpoint, model_name, resolution, device)
        dt = run_inference(model, image_items, conf_thr)
        evaluated_ids = {int(im["id"]) for im, _ in images}
        file_names = [str(im["file_name"]) for im, _ in images]
        mode = "inference"
    else:
        if predictions_file is None:
            raise ValueError("Wymagany --checkpoint albo --predictions-file")
        dt = common.load_predictions(predictions_file)
        if evaluated_images_file is not None:
            data = common.load_json(evaluated_images_file)
            evaluated_ids = {int(i) for i in data["image_ids"]}
        elif images_dir is not None:
            evaluated_ids = common.evaluated_image_ids_from_dir(gt, images_dir)
        else:
            # Import without an explicit subset means the complete annotation set.
            # Missing detections must never be interpreted as missing evaluation.
            evaluated_ids = {int(im["id"]) for im in gt["images"]}
        id2name = {int(im["id"]): str(im["file_name"]) for im in gt["images"]}
        file_names = [id2name[i] for i in sorted(evaluated_ids) if i in id2name]
        mode = "predictions_file"

    common.filter_gt_to_images(gt, evaluated_ids)  # reject unknown sidecar IDs
    unknown_predictions = {int(d["image_id"]) for d in dt} - evaluated_ids
    if unknown_predictions:
        raise ValueError(f"Predykcje zawierają image_id spoza ocenianego zbioru: "
                         f"{sorted(unknown_predictions)}")
    # Validate the evaluation contract before replacing any existing result.
    if predictions_file is None or Path(predictions_file).resolve() != pred_path.resolve():
        common.save_predictions(dt, pred_path)
    sidecar = {
        "set": set_name,
        "mode": mode,
        "annotations_file": str(annotations.resolve()),
        "images_dir": str(images_dir.resolve()) if images_dir else None,
        "predictions_file": str(pred_path.resolve()),
        "conf_threshold_for_predictions": conf_thr,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_images": len(evaluated_ids),
        "image_ids": sorted(int(i) for i in evaluated_ids),
        "file_names": file_names,
    }
    common.save_json(sidecar, _sidecar_path(output_dir, set_name))
    return sidecar


# ---------------------------------------------------------------------------
# Blok metryk (README §4.5) liczony Z PLIKU predykcji
# ---------------------------------------------------------------------------

@contextmanager
def _metric_phase(output_dir: Path, set_name: str, phase: str, timings: dict):
    """Expose CPU metric progress even when subprocess stdout is redirected."""
    progress = output_dir / f"evaluation_progress_{set_name}.json"
    common.save_json({"status": "running", "phase": phase,
                      "completed_phase_seconds": dict(timings),
                      "updated_at": datetime.now(timezone.utc).isoformat()}, progress)
    print(f"[eval:{set_name}] {phase}: start", flush=True)
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        common.save_json({"status": "failed", "phase": phase,
                          "error": str(exc), "completed_phase_seconds": dict(timings),
                          "updated_at": datetime.now(timezone.utc).isoformat()}, progress)
        raise
    else:
        timings[phase] = time.perf_counter() - started
        common.save_json({"status": "running", "phase": phase + ":completed",
                          "completed_phase_seconds": dict(timings),
                          "updated_at": datetime.now(timezone.utc).isoformat()}, progress)
        print(f"[eval:{set_name}] {phase}: {timings[phase]:.3f}s", flush=True)


def compute_block(output_dir: Path, protocol: dict, set_name: str, *,
                  operating_threshold: float | None = None,
                  threshold_source: str | None = None,
                  pool_manifest: Path | None = None) -> dict:
    sidecar_path = _sidecar_path(output_dir, set_name)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"{sidecar_path} nie istnieje — najpierw uruchom evaluate.py "
            f"(tryb inferencji lub --predictions-file) dla zbioru '{set_name}'."
        )
    started = time.perf_counter()
    timings = {}
    with _metric_phase(output_dir, set_name, "load_predictions", timings):
        sidecar = common.load_json(sidecar_path)
        pred_path = Path(sidecar["predictions_file"])
        gt = common.load_gt_dataset(sidecar["annotations_file"])
        evaluated_ids = {int(i) for i in sidecar["image_ids"]}
        gt_trim = common.filter_gt_to_images(gt, evaluated_ids)
        dt = common.load_predictions(pred_path)
        dt = [d for d in dt if int(d["image_id"]) in evaluated_ids]

    single_remap = bool(common.protocol_field(
        protocol, "eval_agent3", "evaluation", "single_class_remap", default=True))
    remapped = False
    if single_remap:
        dt, remapped = common.remap_dt_categories_single_class(gt_trim, dt)

    print(f"[eval:{set_name}] images={len(evaluated_ids)}, predictions={len(dt)}", flush=True)
    with _metric_phase(output_dir, set_name, "coco_map", timings):
        map_metrics = common.compute_map_metrics(gt_trim, dt)

    iou_thr = float(common.protocol_field(protocol, "evaluation", "iou_threshold"))
    with _metric_phase(output_dir, set_name, "operating_threshold", timings):
        if operating_threshold is None:
            operating_threshold, threshold_source = resolve_operating_threshold(
                output_dir, protocol, gt_trim, dt, iou_thr, set_name)

    gt_index = common.build_gt_index(gt_trim)
    with _metric_phase(output_dir, set_name, "counting_metrics", timings):
        counts = common.counting_metrics(gt_index, dt, float(operating_threshold), iou_thr)

    block = {
        "mAP_0.5": map_metrics["mAP_0.5"],
        "mAP_0.5:0.95": map_metrics["mAP_0.5:0.95"],
        "mAP_0.75": map_metrics["mAP_0.75"],
        "AR@100": map_metrics["AR@100"],
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "precision": counts["precision"],
        "recall": counts["recall"],
        "f1": counts["f1"],
        "n_images": len(evaluated_ids),
        "n_annotations": counts["n_annotations"],
    }
    n_patients = common.count_patients(sidecar.get("file_names", []), pool_manifest)

    meta = {
        "set": set_name,
        "block": block,
        "n_patients": n_patients,
        "operating_threshold": float(operating_threshold),
        "threshold_selected_on": threshold_source,
        "iou_threshold": iou_thr,
        "conf_threshold_for_predictions": sidecar.get("conf_threshold_for_predictions"),
        "single_class_remap_applied": remapped,
        "predictions_file": str(pred_path),
        "annotations_file": sidecar["annotations_file"],
        "gt_trimmed_to_evaluated_images": True,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "metric_phase_seconds": dict(timings),
    }
    common.save_json(meta, output_dir / f"metrics_{set_name}.json")
    common.save_json({"status": "completed", "phase": "metrics_written",
                      "completed_phase_seconds": dict(timings),
                      "total_metric_seconds": time.perf_counter() - started,
                      "updated_at": datetime.now(timezone.utc).isoformat()},
                     output_dir / f"evaluation_progress_{set_name}.json")
    return meta


def resolve_operating_threshold(output_dir: Path, protocol: dict, gt_trim: dict,
                                dt: list[dict], iou_thr: float, set_name: str
                                ) -> tuple[float, str]:
    """Użyj zapisanego progu z val; bez niego można dopasować próg tylko na val."""
    op_file = output_dir / "operating_point.json"
    if op_file.exists():
        return common.load_validation_operating_point(op_file)
    if set_name != "val":
        raise ValueError("Brak progu z walidacji. Uruchom operating_point.py na val "
                         "przed liczeniem metryk testu; nie wolno stroić progu na teście.")
    max_points = int(common.protocol_field(
        protocol, "eval_agent3", "operating_point", "f1_curve_max_points", default=200))
    scores = [float(d["score"]) for d in dt]
    thresholds = common.threshold_candidates(
        np.asarray(scores) if scores else np.asarray([]), max_points)
    curve = common.f1_curve(common.build_gt_index(gt_trim), dt, thresholds, iou_thr)
    thr, f1 = common.best_f1_threshold(curve)
    return thr, "val"


# ---------------------------------------------------------------------------
# Składanie metrics.json (README §4.5)
# ---------------------------------------------------------------------------

def assemble(output_dir: Path, protocol: dict, run_id: str, checkpoint_rule: str,
             metrics_name: str = "metrics.json", *,
             operating_threshold: float | None = None,
             threshold_selected_on: str | None = None,
             pool_manifest: Path | None = None) -> Path:
    op_file = output_dir / "operating_point.json"
    if operating_threshold is None and op_file.exists():
        operating_threshold, threshold_selected_on = common.load_validation_operating_point(op_file)
    if operating_threshold is None:
        raise ValueError(
            "Brak progu operacyjnego: uruchom najpierw operating_point.py "
            "(wybór max-F1 na walidacji) albo podaj --operating-threshold."
        )
    threshold_selected_on = threshold_selected_on or "explicit"

    iou_thr = float(common.protocol_field(protocol, "evaluation", "iou_threshold"))
    results: dict[str, dict] = {}
    patients: dict[str, int | None] = {}
    for block_file in sorted(output_dir.glob("metrics_*.json")):
        if block_file.name.startswith("metrics__"):
            continue  # named aggregates are outputs, never input blocks
        set_name = block_file.stem[len("metrics_"):]
        meta = common.load_json(block_file)
        if "results" in meta and "block" not in meta:
            continue  # aggregates may also use a custom --metrics-name
        block = dict(meta["block"])
        if abs(float(meta["operating_threshold"]) - float(operating_threshold)) > 1e-12:
            # przeliczenie TP/FP/FN z ZAPISANYCH predykcji — bez ponownej inferencji
            sidecar = common.load_json(_sidecar_path(output_dir, set_name))
            gt = common.load_gt_dataset(sidecar["annotations_file"])
            gt_trim = common.filter_gt_to_images(
                gt, {int(i) for i in sidecar["image_ids"]})
            dt = common.load_predictions(sidecar["predictions_file"])
            dt = [d for d in dt if int(d["image_id"]) in {int(i) for i in sidecar["image_ids"]}]
            if bool(common.protocol_field(
                    protocol, "eval_agent3", "evaluation", "single_class_remap", default=True)):
                dt, _ = common.remap_dt_categories_single_class(gt_trim, dt)
            counts = common.counting_metrics(
                common.build_gt_index(gt_trim), dt, float(operating_threshold), iou_thr)
            block.update({
                "TP": counts["TP"], "FP": counts["FP"], "FN": counts["FN"],
                "precision": counts["precision"], "recall": counts["recall"],
                "f1": counts["f1"], "n_annotations": counts["n_annotations"],
            })
        results[set_name] = block
        patients[set_name] = meta.get("n_patients")

    if not results:
        raise RuntimeError(f"{output_dir}: brak bloków metrics_<set>.json do złożenia")

    metrics = {
        "run_id": run_id,
        "checkpoint_rule": checkpoint_rule,
        "results": results,
        "operating_threshold": float(operating_threshold),
        "threshold_selected_on": threshold_selected_on,
        "n_patients": patients,
    }
    return common.save_json(metrics, output_dir / metrics_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inferencja + metryki COCO (mAP z pełnej krzywej PR, pycocotools).")
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--run-id", default=None, help="run_id biegu (kontrakt §4.5)")
    p.add_argument("--checkpoint-rule", choices=["last_epoch", "best_val"], default=None)
    p.add_argument("--set", dest="set_name", default=None,
                   help="nazwa zbioru: val / test / external_full / external_clean")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="tryb A: checkpoint RF-DETR do inferencji")
    p.add_argument("--predictions-file", type=Path, default=None,
                   help="tryb B: istniejący plik predykcji COCO results (bez modelu)")
    p.add_argument("--evaluated-images", type=Path, default=None,
                   help="tryb B: sidecar z listą obrazów ocenionych (jeśli był)")
    p.add_argument("--annotations", type=Path, default=None, help="anotacje COCO zbioru")
    p.add_argument("--images-dir", type=Path, default=None, help="katalog obrazów")
    p.add_argument("--device", default=None, help="np. cuda / cpu (inferencja)")
    p.add_argument("--model-name", default=None,
                   help="nadpisanie model.name z protocol.yaml (inferencja)")
    p.add_argument("--resolution", type=int, default=None,
                   help="nadpisanie model.resolution z protocol.yaml (inferencja)")
    p.add_argument("--max-images", type=int, default=None,
                   help="limit obrazów (wyłącznie testy dymne)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="domyślnie <SmallViTAblation>/results/<run-id>")
    p.add_argument("--operating-threshold", type=float, default=None,
                   help="jawny próg operacyjny TP/FP/F1 (normalnie: z operating_point.json)")
    p.add_argument("--metrics-name", default="metrics.json",
                   help="nazwa składanego pliku wyników (np. metrics__last_epoch.json)")
    p.add_argument("--assemble-only", action="store_true",
                   help="tylko złożenie metrics.json z istniejących bloków")
    p.add_argument("--pool-manifest", type=Path, default=None,
                   help="pool_manifest.jsonl do policzenia pacjentów")
    return p


def main(argv: list[str] | None = None) -> int:
    common.ensure_utf8_io()
    args = build_parser().parse_args(argv)
    protocol = common.load_protocol(args.protocol)

    output_dir = args.output_dir
    if output_dir is None:
        if not args.run_id:
            print("BŁĄD: podaj --run-id albo --output-dir", file=sys.stderr)
            return 2
        output_dir = common.project_root(args.protocol) / "results" / args.run_id
    output_dir = Path(output_dir)

    threshold_source = "cli" if args.operating_threshold is not None else None

    if args.assemble_only:
        if not (args.run_id and args.checkpoint_rule):
            print("BŁĄD: --assemble-only wymaga --run-id i --checkpoint-rule", file=sys.stderr)
            return 2
        path = assemble(output_dir, protocol, args.run_id, args.checkpoint_rule,
                        args.metrics_name, operating_threshold=args.operating_threshold,
                        threshold_selected_on=None if args.operating_threshold is None else "cli",
                        pool_manifest=args.pool_manifest)
        print(f"Złożono {path}")
        return 0

    if not args.set_name:
        print("BŁĄD: podaj --set (val/test/external_full/external_clean)", file=sys.stderr)
        return 2
    if args.annotations is None:
        print("BŁĄD: --annotations jest wymagane (GT do metryk)", file=sys.stderr)
        return 2
    if (args.checkpoint is None) == (args.predictions_file is None):
        print("BŁĄD: podaj dokładnie jeden z: --checkpoint | --predictions-file", file=sys.stderr)
        return 2

    warnings.filterwarnings("ignore", category=UserWarning)

    prepare_predictions(
        output_dir, protocol, args.set_name, args.annotations,
        checkpoint=args.checkpoint, images_dir=args.images_dir,
        predictions_file=args.predictions_file,
        evaluated_images_file=args.evaluated_images,
        device=args.device, model_name=args.model_name,
        resolution=args.resolution, max_images=args.max_images,
    )
    meta = compute_block(output_dir, protocol, args.set_name,
                         operating_threshold=args.operating_threshold,
                         threshold_source=threshold_source,
                         pool_manifest=args.pool_manifest)
    b = meta["block"]
    print(f"[{args.set_name}] mAP@0.5={b['mAP_0.5']:.4f}  mAP@[.5:.95]={b['mAP_0.5:0.95']:.4f}  "
          f"mAP@0.75={b['mAP_0.75']:.4f}  AR@100={b['AR@100']:.4f}")
    print(f"[{args.set_name}] próg={meta['operating_threshold']:.4f} ({meta['threshold_selected_on']})  "
          f"TP={b['TP']} FP={b['FP']} FN={b['FN']}  P={b['precision']:.4f} "
          f"R={b['recall']:.4f} F1={b['f1']:.4f}  "
          f"n_images={b['n_images']} n_annotations={b['n_annotations']}")

    if args.run_id and args.checkpoint_rule:
        try:
            path = assemble(output_dir, protocol, args.run_id, args.checkpoint_rule,
                            args.metrics_name,
                            # compute_block may have selected the operating point on val.
                            # Pass that resolved provenance instead of the raw CLI option.
                            operating_threshold=meta["operating_threshold"],
                            threshold_selected_on=meta["threshold_selected_on"],
                            pool_manifest=args.pool_manifest)
            print(f"Złożono {path}")
        except ValueError as exc:
            print(f"UWAGA: metrics.json nie złożony ({exc})", file=sys.stderr)
    else:
        print("UWAGA: bez --run-id/--checkpoint-rule metrics.json nie został złożony; "
              "zapisano blok metrics_" + args.set_name + ".json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
