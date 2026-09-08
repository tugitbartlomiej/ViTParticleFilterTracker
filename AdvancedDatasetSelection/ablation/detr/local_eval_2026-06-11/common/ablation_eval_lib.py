"""
Biblioteka lokalnej ewaluacji ablacji DETR V1-V7 (RTX 3070, bez Edena).

Logika wspolna dla wszystkich wariantow; skrypty w folderach v1..v7 sa tylko
cienkimi nakladkami wywolujacymi funkcje stad:

  run_smoke(variant_id)     - test poprawnosci: 50 obrazow external (split test)
                              + 50 obrazow same-dist (split test)
  run_external(variant_id)  - pelny external benchmark: JEDNA inferencja na 2333
                              obrazach, potem metryki FULL / CLEAN (minus 48
                              leaked) / per-split (debug i porownania z LEGACY)
  run_samedist(variant_id)  - pelny same-distribution: split val + test

Solidnosc ladowania checkpointow:
  - obsluga obu formatow .pth (dict z 'model_state_dict' ORAZ surowy state_dict),
  - num_queries i num_labels INFEROWANE z ksztaltow wag (nie zakladamy 100),
  - typ backbone (timm vs natywny) wykrywany z nazw kluczy,
  - budowa modelu offline (use_pretrained_backbone=False) z fallbackiem na
    cache HuggingFace.
"""

import io
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import paths_config as cfg

# Konsola Windows: wymus UTF-8 (znana pulapka projektu)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# =============================================================================
# Ladowanie modelu
# =============================================================================

def _extract_state_dict(ckpt_path: Path) -> Tuple[dict, str]:
    """Wyciaga state_dict z pliku .pth niezaleznie od formatu zapisu.

    Zwraca (state_dict, opis_zrodla_epoki).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    epoch_info = "?"

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                epoch_info = str(ckpt.get("epoch", "?"))
                sd = ckpt[key]
                break
        else:
            # Surowy state_dict (same tensory) - format best_model.pth z Edena
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd = ckpt
            else:
                raise ValueError(
                    f"Nieznany format checkpointa {ckpt_path.name}: "
                    f"klucze={list(ckpt.keys())[:10]}"
                )
    else:
        raise ValueError(f"Nieobslugiwany typ checkpointa: {type(ckpt)}")

    # DDP zapisuje z prefiksem 'module.'
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd, epoch_info


# Transformers zmienial nazewnictwo wag DETR miedzy wersjami 4.x (tak zapisano
# checkpointy na Edenie) a 5.x (tak buduje lokalna transformers 5.1.0):
#   backbone:  'model.backbone.conv_encoder.model.*' -> 'model.backbone.model.*'
#   attention: '.self_attn/.encoder_attn .out_proj.' -> '.o_proj.'
#   FFN:       '.fc1./.fc2.'                          -> '.mlp.fc1./.mlp.fc2.'
_RENAME_RULES = (
    ("model.backbone.conv_encoder.model.", "model.backbone.model."),
    (".self_attn.out_proj.", ".self_attn.o_proj."),
    (".encoder_attn.out_proj.", ".encoder_attn.o_proj."),
    (".fc1.", ".mlp.fc1."),
    (".fc2.", ".mlp.fc2."),
)


def _remap_to_model_naming(sd: dict, model_keys) -> dict:
    """Tlumaczy nazwy wag checkpointa na nazewnictwo zbudowanego modelu.

    Regula jest stosowana tylko wtedy, gdy przetlumaczony klucz FAKTYCZNIE
    istnieje w modelu (dziala w obu kierunkach: stary ckpt -> nowy model
    i odwrotnie). Klucze niepasujace zostaja bez zmian (wyjda w unexpected).
    """
    model_keys = set(model_keys)
    out, remapped = {}, 0
    for k, v in sd.items():
        if k in model_keys:
            out[k] = v
            continue
        cand = k
        for old, new in _RENAME_RULES:  # kierunek: stary ckpt -> nowy model
            cand = cand.replace(old, new)
        if cand in model_keys:
            out[cand] = v
            remapped += 1
            continue
        cand = k
        for old, new in _RENAME_RULES:  # kierunek odwrotny
            cand = cand.replace(new, old)
        if cand in model_keys:
            out[cand] = v
            remapped += 1
        else:
            out[k] = v
    if remapped:
        print(f"[remap] przemianowano {remapped} kluczy "
              f"(zmiana nazewnictwa transformers 4.x <-> 5.x)")
    return out


def load_model(variant_id: str, device: torch.device):
    """Buduje DetrForObjectDetection zgodny z wagami i laduje checkpoint."""
    from transformers import DetrConfig, DetrForObjectDetection

    ckpt_path = cfg.CHECKPOINTS[variant_id]
    print(f"[{variant_id}] Ladowanie: {ckpt_path}")
    sd, epoch_info = _extract_state_dict(ckpt_path)

    # Inferencja architektury z ksztaltow wag
    nq = sd["model.query_position_embeddings.weight"].shape[0]
    num_labels = sd["class_labels_classifier.weight"].shape[0] - 1  # +1 = "no object"
    # timm-resnet ma 'conv1.weight'; natywny HF-resnet ma 'embedder...convolution.weight'
    uses_timm = any(".backbone." in k and k.endswith("conv1.weight") for k in sd)

    print(f"[{variant_id}] epoka={epoch_info}, num_queries={nq}, "
          f"num_labels={num_labels}, timm_backbone={uses_timm}")

    try:
        config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    except Exception:
        config = DetrConfig()  # domyslne = DETR-R50
    config.num_queries = int(nq)
    config.num_labels = int(num_labels)
    config.use_timm_backbone = bool(uses_timm)
    config.use_pretrained_backbone = False  # wagi i tak nadpisujemy

    model = DetrForObjectDetection(config)
    sd = _remap_to_model_naming(sd, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[{variant_id}] UWAGA missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[{variant_id}] UWAGA unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if len(missing) > 10:
        raise RuntimeError(
            f"Za duzo brakujacych wag ({len(missing)}) - architektura niezgodna "
            f"z checkpointem. Przerwano, zeby nie liczyc bzdur."
        )

    model.to(device).eval()
    return model, {"epoch": epoch_info, "num_queries": int(nq),
                   "num_labels": int(num_labels), "timm": bool(uses_timm),
                   "checkpoint": str(ckpt_path)}


def load_processor():
    from transformers import DetrImageProcessor
    try:
        return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    except Exception:
        print("[WARN] Brak cache HF - uzywam domyslnego DetrImageProcessor "
              "(parametry zgodne z DETR-R50).")
        return DetrImageProcessor()


# =============================================================================
# Dane: wczytywanie i scalanie COCO
# =============================================================================

def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_leaked() -> Set[str]:
    with open(cfg.LEAKED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = data["images"] if isinstance(data, dict) else data
    return {n.lower() for n in names}


def _ensure_ann_fields(ann: dict) -> dict:
    """COCOeval wymaga area i iscrowd - uzupelnia jesli brak."""
    if "area" not in ann or not ann.get("area"):
        ann["area"] = float(ann["bbox"][2]) * float(ann["bbox"][3])
    ann.setdefault("iscrowd", 0)
    return ann


def merge_external() -> Tuple[dict, Dict[int, Path]]:
    """Scala train+valid+test external benchmarku w JEDEN zbior (FULL).

    Renumeruje id obrazow i anotacji (w splitach Roboflow id zaczynaja sie
    od 0 i koliduja). Zwraca (coco_dict, mapa: image_id -> katalog obrazu).
    """
    merged = {"info": {"description": "External benchmark FULL (train+valid+test merged)"},
              "licenses": [], "categories": None, "images": [], "annotations": []}
    dir_by_img_id: Dict[int, Path] = {}
    next_img_id, next_ann_id = 1, 1

    for split in cfg.EXTERNAL_SPLITS:
        split_dir = cfg.EXTERNAL_DIR / split
        coco = load_coco(split_dir / "_annotations.coco.json")
        if merged["categories"] is None:
            merged["categories"] = coco["categories"]
        id_map = {}
        for img in sorted(coco["images"], key=lambda i: i["id"]):
            new_img = dict(img)
            id_map[img["id"]] = next_img_id
            new_img["id"] = next_img_id
            new_img["source_split"] = split  # pole dodatkowe (pycocotools ignoruje)
            merged["images"].append(new_img)
            dir_by_img_id[next_img_id] = split_dir
            next_img_id += 1
        for ann in coco["annotations"]:
            new_ann = _ensure_ann_fields(dict(ann))
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[ann["image_id"]]
            merged["annotations"].append(new_ann)
            next_ann_id += 1

    print(f"[external] FULL: {len(merged['images'])} obrazow, "
          f"{len(merged['annotations'])} anotacji")
    return merged, dir_by_img_id


def filter_coco(coco: dict, keep_img_ids: Set[int], description: str) -> dict:
    """Podzbior COCO ograniczony do wskazanych obrazow."""
    out = {"info": {"description": description},
           "licenses": coco.get("licenses", []),
           "categories": coco["categories"],
           "images": [i for i in coco["images"] if i["id"] in keep_img_ids],
           "annotations": [a for a in coco["annotations"] if a["image_id"] in keep_img_ids]}
    return out


def gt_category_id(coco: dict) -> int:
    """Id kategorii do przypisania predykcjom: najczestsza kategoria w GT.

    (Eksporty Roboflow miewaja pusta superkategorie z id=0 - cats[0] bywa zla.)
    """
    counts = Counter(a["category_id"] for a in coco["annotations"])
    if counts:
        return counts.most_common(1)[0][0]
    return int(coco["categories"][0]["id"])


# =============================================================================
# Inferencja
# =============================================================================

class CocoInferenceDataset(Dataset):
    """Dataset inferencyjny: obrazy z COCO dict + katalog(i) zrodlowe."""

    def __init__(self, coco: dict, images_dir, processor,
                 dir_by_img_id: Optional[Dict[int, Path]] = None,
                 limit: Optional[int] = None):
        self.processor = processor
        self.items = []
        self.missing = []
        for img in sorted(coco["images"], key=lambda i: i["id"]):
            base = dir_by_img_id[img["id"]] if dir_by_img_id else Path(images_dir)
            fp = base / img["file_name"]
            if fp.exists():
                self.items.append((fp, int(img["id"]),
                                   int(img.get("height", 0)), int(img.get("width", 0))))
            else:
                self.missing.append(img["file_name"])
        if self.missing:
            print(f"[WARN] brak {len(self.missing)}/{len(coco['images'])} plikow "
                  f"obrazow na dysku (np. {self.missing[:3]}) - GT zostanie "
                  f"ograniczone do obrazow istniejacych")
        if limit is not None:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, img_id, h, w = self.items[idx]
        img = Image.open(fp).convert("RGB")
        if not h or not w:
            w, h = img.size
        enc = self.processor(images=img, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"].squeeze(0),
                "image_id": img_id, "orig_size": (h, w)}

    def kept_image_ids(self) -> Set[int]:
        return {it[1] for it in self.items}


def _collate_pad(batch):
    """Pad do max H,W w batchu + pixel_mask (rozmiary moga sie roznic)."""
    max_h = max(b["pixel_values"].shape[1] for b in batch)
    max_w = max(b["pixel_values"].shape[2] for b in batch)
    pv = torch.zeros(len(batch), 3, max_h, max_w)
    pm = torch.zeros(len(batch), max_h, max_w, dtype=torch.long)
    for i, b in enumerate(batch):
        c, h, w = b["pixel_values"].shape
        pv[i, :, :h, :w] = b["pixel_values"]
        pm[i, :h, :w] = 1
    return {"pixel_values": pv, "pixel_mask": pm,
            "image_ids": [b["image_id"] for b in batch],
            "orig_sizes": [b["orig_size"] for b in batch]}


@torch.no_grad()
def run_inference(model, dataset, processor, device,
                  batch_size: int, category_id: int,
                  num_workers: int = 2) -> List[dict]:
    """Inferencja na calym dataset -> predykcje w formacie COCO results."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_pad,
                        num_workers=num_workers, shuffle=False)
    predictions = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        pv = batch["pixel_values"].to(device)
        pm = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pv, pixel_mask=pm)
        target_sizes = torch.tensor(batch["orig_sizes"], device=device)
        post = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes)
        for p, img_id in zip(post, batch["image_ids"]):
            boxes = p["boxes"].cpu().numpy()
            scores = p["scores"].cpu().numpy()
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                predictions.append({
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1),
                             float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                    "score": float(s),
                })
        if (bi + 1) % 25 == 0:
            done = (bi + 1) * batch_size
            rate = done / max(1e-6, time.time() - t0)
            print(f"  ... {min(done, len(dataset))}/{len(dataset)} obrazow "
                  f"({rate:.1f} img/s)")
    print(f"  Inferencja: {len(dataset)} obrazow w {time.time()-t0:.0f}s, "
          f"{len(predictions)} predykcji")
    return predictions


@torch.no_grad()
def run_inference_q81(model, dataset, processor, device,
                      batch_size: int, category_id: int,
                      num_workers: int = 2,
                      conf_threshold: float = cfg.CONF_THRESHOLD,
                      query_id: int = cfg.DETR_QUERY_ID) -> List[dict]:
    """Protokol Q81 - replika 1:1 `run_detr_q81_inference` z
    benchmark_yolo_vs_detr_q81_all_epoch.py (zrodlo headline 85.74%/2.04%):
    softmax(logits), prawdopodobienstwo klasy 0 w slocie `query_id`,
    JEDNA predykcja na obraz, dodawana tylko gdy score >= conf_threshold (0.3);
    ramka: znormalizowane cxcywh * oryginalny rozmiar obrazu -> xywh.
    (processor nieuzywany - pozostawiony dla zgodnosci sygnatury)
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_pad,
                        num_workers=num_workers, shuffle=False)
    predictions = []
    t0 = time.time()
    n_done = 0
    for bi, batch in enumerate(loader):
        pv = batch["pixel_values"].to(device)
        pm = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pv, pixel_mask=pm)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # [B, Q, C+1]
        q_scores = probs[:, query_id, 0]
        q_boxes = outputs.pred_boxes[:, query_id]  # znormalizowane cxcywh
        for i, img_id in enumerate(batch["image_ids"]):
            n_done += 1
            score = float(q_scores[i])
            if score < conf_threshold:
                continue
            h, w = batch["orig_sizes"][i]
            cx, cy, bw, bh = (float(v) for v in q_boxes[i].cpu().numpy())
            predictions.append({
                "image_id": int(img_id),
                "category_id": int(category_id),
                "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                "score": score,
            })
        if (bi + 1) % 50 == 0:
            rate = n_done / max(1e-6, time.time() - t0)
            print(f"  ... {n_done}/{len(dataset)} obrazow ({rate:.1f} img/s)")
    print(f"  Inferencja Q81: {len(dataset)} obrazow w {time.time()-t0:.0f}s, "
          f"{len(predictions)} predykcji (slot {query_id}, prog {conf_threshold})")
    return predictions


# =============================================================================
# Metryki
# =============================================================================

def _iou_xywh(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(1e-8, union)


def detection_counts(preds: List[dict], gt: dict,
                     conf: float = cfg.CONF_THRESHOLD,
                     iou_thr: float = cfg.IOU_THRESHOLD) -> dict:
    """TP/FP/FN + precision/recall/F1 przy progu pewnosci (jak w benchmarkach)."""
    preds_f = [p for p in preds if p["score"] >= conf]
    gt_by_img: Dict[int, list] = {}
    for ann in gt["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)
    preds_by_img: Dict[int, list] = {}
    for p in preds_f:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    tp = fp = fn = 0
    for img in gt["images"]:
        img_id = img["id"]
        gts = gt_by_img.get(img_id, [])
        prs = sorted(preds_by_img.get(img_id, []), key=lambda x: -x["score"])
        matched = set()
        for pr in prs:
            best_iou, best_gt = 0.0, -1
            for gi, g in enumerate(gts):
                i = _iou_xywh(pr["bbox"], g["bbox"])
                if i > best_iou:
                    best_iou, best_gt = i, gi
            if best_iou >= iou_thr and best_gt not in matched:
                tp += 1
                matched.add(best_gt)
            else:
                fp += 1
        fn += len(gts) - len(matched)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec, "F1": f1,
            "conf_threshold": conf, "iou_threshold": iou_thr}


def coco_map(preds: List[dict], gt: dict, workdir: Path, tag: str) -> dict:
    """COCO mAP. GT i predykcje zapisywane do workdir (reprodukowalnosc)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path = workdir / f"gt_{tag}.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f)

    coco_gt = COCO(str(gt_path))
    gt_img_ids = set(coco_gt.getImgIds())
    preds_f = [p for p in preds if p["image_id"] in gt_img_ids]
    if not preds_f:
        return {"mAP_0.5:0.95": 0.0, "mAP_0.5": 0.0, "mAP_0.75": 0.0, "AR@100": 0.0}

    coco_dt = coco_gt.loadRes(preds_f)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP_0.5:0.95": float(ev.stats[0]), "mAP_0.5": float(ev.stats[1]),
            "mAP_0.75": float(ev.stats[2]), "AR@100": float(ev.stats[8])}


def evaluate_on(preds: List[dict], gt: dict, workdir: Path, tag: str) -> dict:
    m = coco_map(preds, gt, workdir, tag)
    d = detection_counts(preds, gt)
    n_img = len(gt["images"])
    n_ann = len(gt["annotations"])
    print(f"  [{tag}] N={n_img} img / {n_ann} ann | "
          f"mAP@0.5={m['mAP_0.5']*100:.2f}% | mAP@[0.5:0.95]={m['mAP_0.5:0.95']*100:.2f}% | "
          f"F1={d['F1']*100:.1f}% (TP={d['TP']})")
    return {"n_images": n_img, "n_annotations": n_ann, **m, **d}


# =============================================================================
# Raport
# =============================================================================

def write_report(out_dir: Path, variant_id: str, model_info: dict,
                 results: Dict[str, dict], title: str):
    lines = [f"# {title}", "",
             f"- Wariant: **{variant_id}** ({cfg.VARIANT_LABELS[variant_id]})",
             f"- Checkpoint: `{model_info['checkpoint']}` (epoka {model_info['epoch']})",
             f"- num_queries={model_info['num_queries']}, "
             f"num_labels={model_info['num_labels']}, timm={model_info['timm']}",
             f"- Progi: conf={cfg.CONF_THRESHOLD}, IoU={cfg.IOU_THRESHOLD}", "",
             "| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in results.items():
        lines.append(
            f"| {tag} | {r['n_images']} | {r['mAP_0.5']*100:.2f}% "
            f"| {r['mAP_0.5:0.95']*100:.2f}% | {r['F1']*100:.1f}% "
            f"| {r['Precision']*100:.1f}% | {r['Recall']*100:.1f}% "
            f"| {r['TP']} | {r['FP']} | {r['FN']} |")
    report = "\n".join(lines) + "\n"
    with open(out_dir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"variant": variant_id, "label": cfg.VARIANT_LABELS[variant_id],
                   "model_info": model_info, "results": results}, f, indent=2)
    print(f"\nZapisano: {out_dir / 'REPORT.md'}")
    print(report)


def _prepare(variant_id: str, run_name: str, need_samedist: bool = True):
    """Wspolny setup: walidacja sciezek, katalog wynikow, model, procesor."""
    if not cfg.check_paths(variant_id, need_samedist=need_samedist):
        sys.exit(f"Brakuje plikow dla {variant_id} - przerwano.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA niedostepne - inferencja na CPU bedzie BARDZO wolna.")
    out_dir = cfg.RESULTS_DIR / variant_id / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, model_info = load_model(variant_id, device)
    processor = load_processor()
    return device, out_dir, model, model_info, processor


def _release(model):
    """Zwalnia model i pamiec GPU - istotne przy sekwencyjnym przechodzeniu
    przez wiele wariantow w jednym procesie (orkiestrator smoke testow)."""
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Punkty wejscia (wywolywane przez skrypty per wariant)
# =============================================================================

def run_smoke(variant_id: str, limit: int = cfg.SMOKE_LIMIT,
              batch_size: int = cfg.DEFAULT_BATCH_SIZE, num_workers: int = 2):
    """Smoke test: external/test (50 obrazow) + same-dist/test (pierwsze 50).

    Cel: sprawdzic, ze model sie laduje, inferencja dziala i mAP > 0
    na OBU datasetach, zanim spalimy godziny GPU na pelne runy.
    """
    print(f"=== SMOKE TEST {variant_id} (limit {limit} obrazow per dataset) ===")
    device, out_dir, model, model_info, processor = _prepare(variant_id, "smoke")
    results = {}

    # --- Dataset 1: external, split test (dokladnie 50 obrazow) ---
    ext_dir = cfg.EXTERNAL_DIR / "test"
    ext_gt = load_coco(ext_dir / "_annotations.coco.json")
    for a in ext_gt["annotations"]:
        _ensure_ann_fields(a)
    cat_id = gt_category_id(ext_gt)
    ds = CocoInferenceDataset(ext_gt, ext_dir, processor, limit=limit)
    print(f"\n[1/2] External (test split): {len(ds)} obrazow")
    preds = run_inference(model, ds, processor, device, batch_size, cat_id, num_workers)
    gt_lim = filter_coco(ext_gt, ds.kept_image_ids(), "smoke external test")
    with open(out_dir / "predictions_external_test.json", "w", encoding="utf-8") as f:
        json.dump(preds, f)
    results["external_test_smoke"] = evaluate_on(preds, gt_lim, out_dir, "external_test_smoke")

    # --- Dataset 2: same-dist, split test (pierwsze `limit` obrazow) ---
    sd_gt = load_coco(cfg.SAMEDIST_ANN_DIR / variant_id / "annotations_test.json")
    for a in sd_gt["annotations"]:
        _ensure_ann_fields(a)
    cat_id = gt_category_id(sd_gt)
    ds = CocoInferenceDataset(sd_gt, cfg.SAMEDIST_IMAGES_DIR, processor, limit=limit)
    print(f"\n[2/2] Same-dist (test split, limit {limit}): {len(ds)} obrazow")
    preds = run_inference(model, ds, processor, device, batch_size, cat_id, num_workers)
    gt_lim = filter_coco(sd_gt, ds.kept_image_ids(), "smoke same-dist test")
    with open(out_dir / "predictions_samedist_test.json", "w", encoding="utf-8") as f:
        json.dump(preds, f)
    results["samedist_test_smoke"] = evaluate_on(preds, gt_lim, out_dir, "samedist_test_smoke")

    write_report(out_dir, variant_id, model_info, results,
                 f"Smoke test {variant_id} (po {limit} obrazow z obu datasetow)")
    _smoke_verdict(results)
    _release(model)


def _smoke_verdict(results: Dict[str, dict]):
    problems = [tag for tag, r in results.items() if r["mAP_0.5"] <= 0.0]
    if problems:
        print(f"\n[SMOKE: PODEJRZANE] mAP@0.5 = 0 dla: {problems}. "
              f"Sprawdz ladowanie modelu / kategorie / sciezki obrazow "
              f"PRZED pelnym runem.")
    else:
        print("\n[SMOKE: OK] Obie ewaluacje zwracaja niezerowe mAP - "
              "mozna uruchamiac pelne runy.")


def run_external(variant_id: str, batch_size: int = cfg.DEFAULT_BATCH_SIZE,
                 num_workers: int = 2, protocol: str = "all"):
    """Pelny external benchmark: inferencja RAZ na FULL (2333 obrazow),
    metryki: FULL, CLEAN (minus 48 leaked) oraz per-split (debug/porownania).

    protocol:
      "all" - standardowa ewaluacja DETR (wszystkie 100 queries, threshold 0.0)
      "q81" - protokol artykulu (1 predykcja/obraz ze slotu 81, prog 0.3)
    """
    run_name = "external" if protocol == "all" else f"external_{protocol}"
    print(f"=== EXTERNAL BENCHMARK {variant_id} (protokol: {protocol.upper()}; "
          f"FULL + CLEAN + per-split) ===")
    device, out_dir, model, model_info, processor = _prepare(
        variant_id, run_name, need_samedist=False)
    model_info["protocol"] = protocol

    full_gt, dir_by_img_id = merge_external()
    cat_id = gt_category_id(full_gt)
    ds = CocoInferenceDataset(full_gt, None, processor, dir_by_img_id=dir_by_img_id)
    if ds.missing:
        full_gt = filter_coco(full_gt, ds.kept_image_ids(),
                              full_gt["info"]["description"] + " (istniejace pliki)")
    infer = run_inference if protocol == "all" else run_inference_q81
    preds = infer(model, ds, processor, device, batch_size, cat_id, num_workers)
    with open(out_dir / "predictions_full.json", "w", encoding="utf-8") as f:
        json.dump(preds, f)

    results = {}
    results["FULL"] = evaluate_on(preds, full_gt, out_dir, "FULL")

    # CLEAN: bez 48 obrazow, ktore przeciekly do treningu 20k
    leaked = load_leaked()
    clean_ids = {i["id"] for i in full_gt["images"]
                 if i["file_name"].lower() not in leaked}
    n_excluded = len(full_gt["images"]) - len(clean_ids)
    print(f"[CLEAN] wykluczono {n_excluded} leaked obrazow (oczekiwane: 48)")
    clean_gt = filter_coco(full_gt, clean_ids, "External CLEAN (minus leaked)")
    results["CLEAN"] = evaluate_on(preds, clean_gt, out_dir, "CLEAN")

    # Per-split (do debugowania i porownan z historycznymi benchmarkami)
    for split in cfg.EXTERNAL_SPLITS:
        ids = {i["id"] for i in full_gt["images"] if i.get("source_split") == split}
        split_gt = filter_coco(full_gt, ids, f"External split {split}")
        results[f"split_{split}"] = evaluate_on(preds, split_gt, out_dir, f"split_{split}")

    write_report(out_dir, variant_id, model_info, results,
                 f"External benchmark {variant_id} - cross-dataset "
                 f"(protokol {protocol.upper()})")
    _release(model)


def run_samedist(variant_id: str, splits: Tuple[str, ...] = ("val", "test"),
                 batch_size: int = cfg.DEFAULT_BATCH_SIZE, num_workers: int = 2,
                 limit: Optional[int] = None, protocol: str = "all"):
    """Pelny same-distribution: ewaluacja na splitach val i test
    (video-level split, identyczny dla wszystkich wariantow)."""
    run_name = "samedist" if protocol == "all" else f"samedist_{protocol}"
    print(f"=== SAME-DIST {variant_id} (protokol: {protocol.upper()}; "
          f"splity: {', '.join(splits)}) ===")
    device, out_dir, model, model_info, processor = _prepare(variant_id, run_name)
    model_info["protocol"] = protocol
    infer = run_inference if protocol == "all" else run_inference_q81
    results = {}
    for split in splits:
        gt = load_coco(cfg.SAMEDIST_ANN_DIR / variant_id / f"annotations_{split}.json")
        for a in gt["annotations"]:
            _ensure_ann_fields(a)
        cat_id = gt_category_id(gt)
        ds = CocoInferenceDataset(gt, cfg.SAMEDIST_IMAGES_DIR, processor, limit=limit)
        print(f"\n[{split}] {len(ds)} obrazow")
        preds = infer(model, ds, processor, device, batch_size, cat_id, num_workers)
        with open(out_dir / f"predictions_{split}.json", "w", encoding="utf-8") as f:
            json.dump(preds, f)
        # GT ZAWSZE ograniczone do obrazow faktycznie zewaluowanych - obrazy
        # nieobecne na dysku nie moga zanizac metryk jako "wieczne FN"
        gt_used = filter_coco(gt, ds.kept_image_ids(), f"same-dist {split}")
        results[split] = evaluate_on(preds, gt_used, out_dir, split)
        results[split]["missing_images"] = len(ds.missing)

    write_report(out_dir, variant_id, model_info, results,
                 f"Same-distribution {variant_id} (video-level split)")
    _release(model)
