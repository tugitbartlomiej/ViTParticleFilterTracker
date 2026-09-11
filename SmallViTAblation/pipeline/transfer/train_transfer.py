#!/usr/bin/env python3
"""Training adapters for fast probes and architecture transfer.

Trains one run on YOLOv8n, RT-DETRv2-S, D-FINE-S or the retained current-YOLO
control. Everything that must
byc identyczne (rozdzielczosc, zbior, budzet epok, podzial, ziarno, sciezka
ewaluacji) czytane jest z protocol.yaml — zero zaszytych wartosci.

CLI (analogiczne do Agenta 2):
    py -3.11 pipeline/transfer/train_transfer.py \
        --run-id M-DESF__s42__b20000__d-fine-s \
        --variant-file variants/M-DESF__s42__b20000.json \
        --protocol configs/protocol.yaml \
        --data-root /raid/data \
        --out results \
        --arch d-fine-s

Dodaj --dry-run aby wykonac PELNA walidacje (wejscia, konfiguracja, iso-resource,
sha256, budowa manifestu in-memory, zapis run_manifest.json) BEZ treningu.

Importy bibliotek modeli (ultralytics, d-fine) sa LAZY — wewnatrz funkcji.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.transfer import transfer_common as tc  # noqa: E402

ARCH_CHOICES = ["yolov8n", "rtdetrv2-s", "d-fine-s", "yolo-current"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def apply_profile(protocol: dict, profile_path: Path, profile_name: str | None,
                  arch: str) -> dict:
    """Apply only declared compute limits; scientific inputs stay unchanged."""
    if not profile_name:
        return protocol
    profiles = tc.load_yaml(profile_path).get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"unknown experiment profile {profile_name!r}")
    profile = profiles[profile_name]
    allowed = profile.get("models", [])
    if allowed and arch not in allowed:
        raise ValueError(f"profile {profile_name!r} does not allow model {arch!r}")
    import copy
    resolved = copy.deepcopy(protocol)
    resolved["training"]["epochs"] = int(profile["epochs"])
    resolutions = profile.get("resolution_by_model", {})
    if arch in resolutions:
        resolved["model"]["resolution"] = int(resolutions[arch])
    precisions = profile.get("precision_by_model", {})
    if arch in precisions:
        resolved["training"]["precision"] = str(precisions[arch])
    resolved["active_profile"] = profile_name
    return resolved


def seed_everything(seed: int) -> None:
    """Ziarno we wszystkich bibliotekach (README 2.5): torch, numpy, random."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # lazy
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# walidacja wejść (używana też przez --dry-run)
# --------------------------------------------------------------------------- #
def validate_inputs(args, protocol, iso, variant, pool_by_id, data_root):
    """Pelna walidacja. Zwraca (raport_dict, warnings). Rzuca ValueError na bledy."""
    tc.assert_no_early_stopping(iso)
    warnings = tc.validate_variant(variant, pool_by_id)

    frame_ids = variant["frame_ids"]
    unknown_labels = [fid for fid in frame_ids
                      if pool_by_id[fid].get("label_status") == "unknown"]
    if unknown_labels:
        raise ValueError(
            f"{len(unknown_labels)} selected frames still have label_status=unknown "
            f"(example: {unknown_labels[:3]}). Annotate or verify negatives before training.")
    no_path = [fid for fid in frame_ids if not pool_by_id[fid].get("path")]
    if no_path:
        raise ValueError(
            f"{len(no_path)} frames are archive-backed and have no materialized path "
            f"(example: {no_path[:3]}). Export/materialize the selected subset first.")
    missing_imgs = [fid for fid in frame_ids
                    if not (data_root / pool_by_id[fid]["path"]).exists()]
    if missing_imgs:
        raise ValueError(
            f"{len(missing_imgs)} obrazow treningowych nie istnieje pod data-root "
            f"(przyklad: {missing_imgs[:3]})")

    # walidacja obecnosci zbioru walidacyjnego (podzial danych — ten sam plik)
    val_rows = [r for r in pool_by_id.values() if r.get("split") == "val"]
    if not val_rows:
        warnings.append("pool_manifest: brak klatek split=val (ewaluacja niemozliwa)")
    unknown_val = [r["frame_id"] for r in val_rows if r.get("label_status") == "unknown"]
    if unknown_val:
        raise ValueError(f"{len(unknown_val)} validation frames have unverified labels")
    missing_val = [r["frame_id"] for r in val_rows
                   if not r.get("path") or not (data_root / r["path"]).exists()]
    if missing_val:
        raise ValueError(f"{len(missing_val)} obrazow walidacyjnych nie istnieje")

    seed = int(variant["seed"])
    report = {
        "run_id": args.run_id,
        "arch": args.arch,
        "model_id": protocol.get("transfer", {}).get(
            "architectures", {}).get(args.arch, {}).get("model_id"),
        "n_train_frames": len(frame_ids),
        "n_val_frames": len(val_rows),
        "resolution": iso["resolution"],
        "epochs": iso["epochs"],
        "seed": seed,
        "batch_size": iso["batch_size"],
        "best_val_criterion": iso["best_val_criterion"],
        "map_conf_threshold": iso["map_conf_threshold"],
        "eval_iou_threshold": iso["eval_iou_threshold"],
        "protocol_sha256": tc.sha256_file(args.protocol),
        "variant_file_sha256": tc.sha256_file(args.variant_file),
    }
    return report, warnings


# --------------------------------------------------------------------------- #
# budowa zbiorow COCO (train z pliku wariantu, val/test z podzialu)
# --------------------------------------------------------------------------- #
def build_split_coco(pool_by_id, data_root, frame_ids=None, split=None, read_sizes=False):
    reader = tc.image_size_from_file if read_sizes else None
    if frame_ids is not None:
        rows = [pool_by_id[fid] for fid in frame_ids]
    else:
        rows = [r for r in pool_by_id.values() if r.get("split") == split]
    return tc.build_coco(rows, data_root, image_size_reader=reader)


def coco_to_yolo_dataset(train_coco, val_coco, data_root, ds_root, resolution):
    """Materializuje dataset YOLO (obrazy jako linki, etykiety txt) + data.yaml.

    Nie KOPIUJE obrazow treningowych wielokrotnie — uzywa hardlinkow (fallback:
    symlink/kopia). Etykiety: klasa 0 (instrument), format cx cy w h (norm).
    """
    ds_root = Path(ds_root)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ds_root / sub).mkdir(parents=True, exist_ok=True)

    def _link(src, dst):
        if dst.exists() or dst.is_symlink():
            return
        try:
            os.link(src, dst)
        except OSError:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    def emit(coco, split):
        anns_by_img = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            src = (data_root / im["file_name"]).resolve()
            source_path = Path(im["file_name"])
            stem = source_path.stem
            suffix = source_path.suffix.lower() or ".jpg"
            unique_name = f"{int(im['id']):08d}_{stem}{suffix}"
            dst_img = ds_root / "images" / split / unique_name
            _link(src, dst_img)
            W, H = im["width"] or resolution, im["height"] or resolution
            lines = []
            for a in anns_by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                cx = (x + w / 2.0) / W
                cy = (y + h / 2.0) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
            (ds_root / "labels" / split / f"{Path(unique_name).stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8")

    emit(train_coco, "train")
    emit(val_coco, "val")
    data_yaml = ds_root / "data.yaml"
    data_yaml.write_text(
        f"path: {ds_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n  0: {tc.CATEGORY_NAME}\n",
        encoding="utf-8")
    return data_yaml


# --------------------------------------------------------------------------- #
# LAZY trenery architektur
# --------------------------------------------------------------------------- #
def train_yolo(args, protocol, iso, variant, seed, out_run, data_root, train_coco, val_coco):
    """Train a pinned Ultralytics detector. Import stays lazy."""
    from ultralytics import YOLO  # lazy

    arch_cfg = protocol["transfer"]["architectures"][args.arch]
    model_id = arch_cfg["model_id"]
    ds_root = out_run / "yolo_ds"
    data_yaml = coco_to_yolo_dataset(train_coco, val_coco, data_root, ds_root, iso["resolution"])

    model = YOLO(model_id)  # wagi COCO — start zawsze ten sam (README 5.3)
    tcfg = protocol["training"]
    optimizer = str(tcfg.get("optimizer", "auto"))
    optimizer = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD"}.get(
        optimizer.lower(), optimizer)
    model.train(
        data=str(data_yaml),
        epochs=iso["epochs"],                 # bez early stopping
        patience=0,                           # 0 = wylacza early stopping w ultralytics
        imgsz=iso["resolution"],              # ISO-RESOURCE: ta sama rozdzielczosc
        batch=iso["batch_size"],
        seed=seed, deterministic=True,
        optimizer=optimizer,
        lr0=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        warmup_epochs=int(tcfg.get("warmup_epochs", 0)),
        # The curated pool already contains the declared offline variants.
        # Keep every online augmentation at its identity value.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.0, bgr=0.0,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        multi_scale=False, close_mosaic=0,
        amp=str(iso.get("precision", "")).lower() in {"fp16", "bf16"},
        project=str(out_run), name="train", exist_ok=True,
        device=args.device, verbose=True,
    )
    weights = out_run / "train" / "weights"
    shutil.copy2(weights / "last.pt", out_run / "last.pth")
    shutil.copy2(weights / "best.pt", out_run / "best.pth")
    return model


def train_dfine(args, protocol, iso, variant, seed, out_run, data_root, train_coco, val_coco):
    """Trenuje D-FINE-S. D-FINE nie jest pakietem pip z czystym API — trenuje sie
    przez train.py ze zklonowanego repo, konfiguracja YAML wskazujaca nasz COCO.

    Repozytorium: --d-fine-repo albo zmienna srodowiskowa DFINE_REPO.
    Import/uruchomienie LAZY. Patrz README transfer (zal. integracji)."""
    import subprocess  # lazy

    arch_cfg = protocol["transfer"]["architectures"]["d-fine-s"]
    repo = Path(args.d_fine_repo or os.environ.get("DFINE_REPO", "")).resolve()
    if not (repo / "train.py").exists():
        raise FileNotFoundError(
            f"Nie znaleziono repo D-FINE (train.py) w {repo}. Ustaw --d-fine-repo "
            "lub DFINE_REPO (git clone https://github.com/Peterande/D-FINE).")

    ann_dir = out_run / "dfine_ann"
    ann_dir.mkdir(parents=True, exist_ok=True)
    train_json = ann_dir / "train.json"
    val_json = ann_dir / "val.json"
    tc.write_json(train_json, train_coco)
    tc.write_json(val_json, val_coco)

    # Konfiguracja zbioru i treningu D-FINE (nadpisuje domyslny COCO 640x640
    # na ISO-RESOURCE resolution z protocol.yaml).
    cfg = out_run / "dfine_custom.yml"
    cfg.write_text(_dfine_config_yaml(arch_cfg, iso, seed, train_json, val_json, out_run),
                   encoding="utf-8")

    cmd = [sys.executable, str(repo / "train.py"), "-c", str(cfg), "--seed", str(seed)]
    if args.eval_only:
        cmd.append("--eval-only")
    subprocess.run(cmd, cwd=str(repo), check=True)
    # D-FINE zapisuje checkpointy w output_dir; kopiujemy last/best.
    _collect_dfine_checkpoints(out_run)


def materialize_rtdetr_dataset(train_coco, val_coco, data_root, ds_root):
    """Link images once and rewrite COCO file names for the official loader."""
    ds_root = Path(ds_root)
    ann_dir = ds_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    def emit(coco, split):
        image_dir = ds_root / split
        image_dir.mkdir(parents=True, exist_ok=True)
        copied = {**coco, "images": [dict(image) for image in coco["images"]]}
        for image in copied["images"]:
            source = (data_root / image["file_name"]).resolve()
            suffix = source.suffix.lower() or ".jpg"
            name = f"{int(image['id']):08d}_{source.stem}{suffix}"
            target = image_dir / name
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    try:
                        os.symlink(source, target)
                    except OSError:
                        shutil.copy2(source, target)
            image["file_name"] = name
        target_json = ann_dir / f"{split}.json"
        tc.write_json(target_json, copied)
        return image_dir, target_json

    return emit(train_coco, "train"), emit(val_coco, "val")


def _rtdetr_config(base_config: Path, train_paths, val_paths, out_run: Path,
                   iso: dict, protocol: dict, n_train_images: int) -> str:
    """Override every inherited data transform that can alter an image.

    The official base recipe enables photometric distortion, zoom-out, IoU
    crop, horizontal flip and multiscale collation.  Merely setting its stop
    epoch leaves these active, so this adapter replaces the complete ops list.
    """
    train_images, train_json = train_paths
    val_images, val_json = val_paths
    resolution = int(iso["resolution"])
    training = protocol["training"]
    steps_per_epoch = max(1, (int(n_train_images) + int(iso["batch_size"]) - 1)
                          // int(iso["batch_size"]))
    warmup_steps = int(training.get("warmup_epochs", 0)) * steps_per_epoch
    use_amp = str(iso.get("precision", "")).lower() in {"bf16", "fp16"}
    return (
        "# Generated adapter for the official lyuwenyu/RT-DETR repository.\n"
        f"__include__: ['{base_config.as_posix()}']\n"
        f"output_dir: '{(out_run / 'rtdetrv2_out').as_posix()}'\n"
        "num_classes: 1\n"
        "remap_mscoco_category: False\n"
        f"epoches: {int(iso['epochs'])}\n"
        f"use_amp: {str(use_amp)}\n"
        "optimizer:\n"
        "  type: AdamW\n"
        "  params:\n"
        "    -\n"
        "      params: '^(?=.*backbone)(?!.*(?:norm|bn)).*$'\n"
        f"      lr: {float(training['lr_backbone'])}\n"
        "    -\n"
        "      params: '^(?=.*(?:norm|bn)).*$'\n"
        "      weight_decay: 0.0\n"
        f"  lr: {float(training['lr'])}\n"
        "  betas: [0.9, 0.999]\n"
        f"  weight_decay: {float(training['weight_decay'])}\n"
        "lr_warmup_scheduler:\n"
        "  type: LinearWarmup\n"
        f"  warmup_duration: {warmup_steps}\n"
        "train_dataloader:\n"
        f"  total_batch_size: {int(iso['batch_size'])}\n"
        "  dataset:\n"
        f"    img_folder: '{train_images.as_posix()}'\n"
        f"    ann_file: '{train_json.as_posix()}'\n"
        "    transforms:\n"
        "      ops:\n"
        f"        - {{type: Resize, size: [{resolution}, {resolution}]}}\n"
        "        - {type: SanitizeBoundingBoxes, min_size: 1}\n"
        "        - {type: ConvertPILImage, dtype: 'float32', scale: True}\n"
        "        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}\n"
        "      policy: ~\n"
        "  collate_fn:\n"
        "    scales: ~\n"
        "val_dataloader:\n"
        f"  total_batch_size: {int(iso['batch_size'])}\n"
        "  dataset:\n"
        f"    img_folder: '{val_images.as_posix()}'\n"
        f"    ann_file: '{val_json.as_posix()}'\n"
        "    transforms:\n"
        "      ops:\n"
        f"        - {{type: Resize, size: [{resolution}, {resolution}]}}\n"
        "        - {type: ConvertPILImage, dtype: 'float32', scale: True}\n"
    )


def train_rtdetrv2(args, protocol, iso, seed, out_run, data_root,
                   train_coco, val_coco):
    """Run the official PyTorch RT-DETRv2-S/R18 training entry point."""
    repo_arg = args.rtdetr_repo or os.environ.get("RTDETR_REPO", "")
    if not repo_arg:
        raise FileNotFoundError("Set --rtdetr-repo or RTDETR_REPO to the official checkout")
    repo = Path(repo_arg).resolve()
    source = repo / "rtdetrv2_pytorch"
    train_script = source / "tools" / "train.py"
    arch_cfg = protocol["transfer"]["architectures"]["rtdetrv2-s"]
    base_config = source / arch_cfg["config"]
    if not train_script.is_file() or not base_config.is_file():
        raise FileNotFoundError(
            f"Expected official rtdetrv2_pytorch checkout under {source}")
    tuning = args.rtdetr_weights or os.environ.get("RTDETRV2_S_WEIGHTS", "")
    if not tuning:
        raise ValueError("COCO initialization requires --rtdetr-weights or RTDETRV2_S_WEIGHTS")
    ds = materialize_rtdetr_dataset(
        train_coco, val_coco, data_root, out_run / "rtdetrv2_ds")
    cfg = out_run / "rtdetrv2_custom.yml"
    cfg.write_text(_rtdetr_config(base_config, ds[0], ds[1], out_run, iso,
                                  protocol, len(train_coco["images"])),
                   encoding="utf-8")
    cmd = [sys.executable, str(train_script), "-c", str(cfg), "-t", str(tuning),
           "--seed", str(seed)]
    if str(iso.get("precision", "")).lower() in {"bf16", "fp16"}:
        cmd.append("--use-amp")
    subprocess.run(cmd, cwd=str(source), check=True)
    _collect_rtdetr_checkpoints(out_run)


def _collect_rtdetr_checkpoints(out_run: Path) -> None:
    output = out_run / "rtdetrv2_out"
    last = output / "last.pth"
    best = output / "best.pth"
    missing = [path.name for path in (last, best) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"official RT-DETRv2 output misses required checkpoint(s) {missing} in {output}")
    shutil.copy2(last, out_run / "last.pth")
    shutil.copy2(best, out_run / "best.pth")


def _dfine_config_yaml(arch_cfg, iso, seed, train_json, val_json, out_run):
    model_id = arch_cfg["model_id"]
    res = iso["resolution"]
    return (
        "# Auto-generowane przez Agent 4 (train_transfer.py).\n"
        f"__include__: ['{model_id}.yml']\n"
        f"output_dir: '{out_run.as_posix()}/dfine_out'\n"
        "train_dataloader:\n"
        f"  total_batch_size: {iso['batch_size']}\n"
        "  dataset:\n"
        "    img_folder: ''\n"
        f"    ann_file: '{train_json.as_posix()}'\n"
        "  collate_fn:\n"
        f"    base_size: {res}\n"
        f"    target_size: {res}\n"
        "val_dataloader:\n"
        "  dataset:\n"
        "    img_folder: ''\n"
        f"    ann_file: '{val_json.as_posix()}'\n"
        "  collate_fn:\n"
        f"    base_size: {res}\n"
        f"    target_size: {res}\n"
        "train_cfg:\n"
        f"  epochs: {iso['epochs']}\n"
        "  flat_epoch: true\n"
        f"  seed: {seed}\n"
    )


def _collect_dfine_checkpoints(out_run):
    ddir = out_run / "dfine_out"
    best = sorted(ddir.glob("*best*.pth")) if ddir.exists() else []
    last = sorted(ddir.glob("checkpoint*latest*.pth")) if ddir.exists() else []
    if best:
        shutil.copy2(best[-1], out_run / "best.pth")
    if last:
        shutil.copy2(last[-1], out_run / "last.pth")


# --------------------------------------------------------------------------- #
# ewaluacja → surowe predykcje + metrics.json (README 4.5)
# --------------------------------------------------------------------------- #
def save_raw_predictions_yolo(model, data_yaml, out_run, iso, split_name):
    """Zapisuje surowe predykcje (conf 0.001) do dalszych analiz bez inferencji."""
    res = model.val(data=str(data_yaml), imgsz=iso["resolution"], conf=iso["map_conf_threshold"],
                    iou=iso["nms_iou_threshold"], save_json=True, split="val", verbose=False)
    pred_path = getattr(res, "save_dir", None)
    return pred_path


def compute_and_write_metrics(args, protocol, iso, out_run, gt_val, preds_val,
                              gt_test=None, preds_test=None):
    """Liczy metrics.json (walidacja wybiera prog; test liczony tym progiem)."""
    from SmallViTAblation.pipeline.transfer import metrics as M  # lazy
    gts_val = M.parse_gt(gt_val)[0]
    preds_val_arr = M.parse_preds(preds_val)
    thr, _ = M.select_operating_threshold(preds_val_arr, gts_val, iso["eval_iou_threshold"])

    results = {"val": M.compute_block(gt_val, preds_val, thr, iso["eval_iou_threshold"])}
    if gt_test is not None and preds_test is not None:
        results["test"] = M.compute_block(gt_test, preds_test, thr, iso["eval_iou_threshold"])
    # external_full / external_clean dopisze Agent 3 (external_once.py, zero-look).
    metrics = M.assemble_metrics(
        run_id=args.run_id, checkpoint_rule="best_val",
        operating_threshold=thr, results=results)
    tc.write_json(out_run / "metrics.json", metrics)
    return metrics


# --------------------------------------------------------------------------- #
def build_argparser():
    ap = argparse.ArgumentParser(description="Trening przeniesienia (Agent 4), iso-resource.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant-file", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", choices=ARCH_CHOICES, required=True)
    ap.add_argument("--train-seed", type=int, default=None,
                    help="paired training seed; selection seed remains in variant metadata")
    ap.add_argument("--dry-run", action="store_true",
                    help="pelna walidacja + zapis run_manifest.json, bez treningu")
    ap.add_argument("--pool-manifest", type=Path, default=None,
                    help="domyslnie <data-root>/pool_manifest.jsonl")
    ap.add_argument("--device", default=None, help="urzadzenie treningu (np. 0, cpu)")
    ap.add_argument("--d-fine-repo", default=None, help="sciezka do zklonowanego repo D-FINE")
    ap.add_argument("--rtdetr-repo", default=None,
                    help="path to the official lyuwenyu/RT-DETR checkout")
    ap.add_argument("--rtdetr-weights", default=None,
                    help="COCO-pretrained RT-DETRv2-S checkpoint for -t")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="katalog git (domyslnie: 3 poziomy w gore od tego pliku)")
    ap.add_argument("--eval-only", action="store_true", help="D-FINE: tylko ewaluacja")
    ap.add_argument("--profile", default=None,
                    help="execution profile from configs/profiles.yaml")
    ap.add_argument("--profiles-config", type=Path, default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root).resolve()
    pool_path = args.pool_manifest or (data_root / "pool_manifest.jsonl")

    protocol = tc.load_protocol(args.protocol)
    profiles_path = args.profiles_config or (Path(args.protocol).parent / "profiles.yaml")
    protocol = apply_profile(protocol, profiles_path, args.profile, args.arch)
    iso = tc.resolve_iso_resource(protocol)
    variant = tc.load_variant(args.variant_file)
    pool_by_id = tc.load_pool_manifest(pool_path)

    started_at = now_iso()
    seed = int(args.train_seed if args.train_seed is not None else variant["seed"])

    report, warnings = validate_inputs(args, protocol, iso, variant, pool_by_id, data_root)
    report["seed"] = seed
    for w in warnings:
        print(f"[warn] {w}")

    out_run = Path(args.out) / args.run_id
    out_run.mkdir(parents=True, exist_ok=True)

    # --- manifest (budowany in-memory zawsze; zapisywany tez w --dry-run) ---
    epochs_completed = 0 if args.dry_run else iso["epochs"]
    manifest = tc.build_run_manifest(
        run_id=args.run_id, arch=args.arch,
        protocol_path=args.protocol, variant_path=args.variant_file,
        iso=iso, protocol=protocol, seed=seed,
        started_at=started_at,
        finished_at=now_iso() if args.dry_run else None,
        epochs_completed=epochs_completed,
        repo_root=repo_root, dry_run=args.dry_run)
    manifest["experiment_profile"] = args.profile
    if args.profile:
        manifest["profiles_config"] = str(Path(profiles_path).resolve())
        manifest["profiles_sha256"] = tc.sha256_file(profiles_path)
    manifest["experiment"] = {
        "strategy": variant.get("strategy", variant.get("variant", variant.get("variant_id"))),
        "budget": variant.get("budget"),
        "selection_seed": variant.get("seed"),
        "train_seed": seed,
        "round": variant.get("round"),
    }

    print("=== locked recipe (iso-resource within each architecture) ===")
    for k in ("resolution", "precision", "epochs", "seed", "batch_size",
              "best_val_criterion"):
        print(f"  {k}: {report.get(k, iso.get(k))}")
    print(f"  variant_file: {args.variant_file.name} "
          f"(sha256 {report['variant_file_sha256'][:12]}...)")

    if args.dry_run:
        tc.write_json(out_run / "run_manifest.json", manifest)
        print(f"[dry-run] Zapisano {out_run / 'run_manifest.json'} (bez treningu)")
        print(f"[dry-run] arch={args.arch} model_id={report['model_id']} "
              f"n_train={report['n_train_frames']} n_val={report['n_val_frames']}")
        print(f"[dry-run] nms: {manifest['nms']['applies']} "
              f"iou={manifest['nms']['iou_threshold']} conf={manifest['nms']['conf_threshold']}")
        return 0

    # --- prawdziwy trening ---
    seed_everything(seed)
    train_coco = build_split_coco(pool_by_id, data_root, frame_ids=variant["frame_ids"],
                                  read_sizes=True)
    val_coco = build_split_coco(pool_by_id, data_root, split="val", read_sizes=True)
    tc.write_json(out_run / "gt_train.json", train_coco)
    tc.write_json(out_run / "gt_val.json", val_coco)

    if args.arch in {"yolov8n", "yolo-current"}:
        model = train_yolo(args, protocol, iso, variant, seed, out_run, data_root,
                           train_coco, val_coco)
        # surowe predykcje (conf 0.001) + metrics.json
        import json as _json
        data_yaml = out_run / "yolo_ds" / "data.yaml"
        pred_dir = save_raw_predictions_yolo(model, data_yaml, out_run, iso, "val")
        preds_val = _load_ultralytics_json(pred_dir, val_coco)
        tc.write_json(out_run / "preds_val_raw.json", preds_val)
        compute_and_write_metrics(args, protocol, iso, out_run, val_coco, preds_val)
    elif args.arch == "d-fine-s":
        train_dfine(args, protocol, iso, variant, seed, out_run, data_root,
                    train_coco, val_coco)
    else:
        train_rtdetrv2(args, protocol, iso, seed, out_run, data_root,
                      train_coco, val_coco)

    manifest["finished_at"] = now_iso()
    manifest["epochs_completed"] = iso["epochs"]
    manifest["library_versions"] = tc.library_versions(
        ["ultralytics", "numpy", "pycocotools", "pillow", "torch"])
    tc.write_json(out_run / "run_manifest.json", manifest)
    print(f"Zapisano run_manifest.json, last.pth, best.pth w {out_run}")
    return 0


def _load_ultralytics_json(pred_dir, coco):
    """Czyta predictions.json (COCO) ultralytics; fallback: puste."""
    import json as _json
    if pred_dir is None:
        return []
    cand = Path(pred_dir) / "predictions.json"
    if cand.exists():
        return _json.loads(cand.read_text(encoding="utf-8"))
    return []


if __name__ == "__main__":
    sys.exit(main())
