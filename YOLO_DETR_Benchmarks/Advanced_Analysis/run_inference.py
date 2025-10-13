import yaml
import os
import json
import time
import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from ultralytics import YOLO
import inspect
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_image_files(images_dir):
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # Sortowanie numeryczne, jeśli pliki mają nazwy typu frame_1.jpg, frame_10.jpg
    try:
        image_files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
    except ValueError:
        pass # Standardowe sortowanie alfabetyczne
    return image_files

def build_coco_image_id_map(annotations_path):
    """
    Build a mapping from image file basename -> COCO image id using ground truth JSON.
    If annotations are unavailable or unreadable, return None.
    """
    try:
        if not annotations_path or not os.path.exists(annotations_path):
            return None
        coco = COCO(annotations_path)
        mapping = {}
        for img in coco.dataset.get('images', []):
            fname = os.path.basename(img.get('file_name', ''))
            if fname:
                mapping[fname] = img['id']
        # Fallback: if mapping is empty, return None
        return mapping or None
    except Exception:
        return None

def get_category_id_mapper(config, model_type):
    """
    Optional label map from model label index -> COCO category id.
    If not provided, defaults to:
      - YOLO: idx + 1 (common case when categories start at 1)
      - DETR: identity (assumes labels already match COCO ids)
    """
    label_map = (config.get('label_map', {}) or {}).get(model_type)
    if isinstance(label_map, dict):
        # Ensure keys are ints (they may come as strings from YAML)
        return {int(k): int(v) for k, v in label_map.items()}
    return None

def _allowlist_ultralytics_pickle_classes():
    """Allowlist common Ultralytics and torch.nn classes for PyTorch 2.6 safe pickle.
    Has no effect on older PyTorch versions. Best‑effort and safe to ignore failures.
    """
    try:
        from torch.serialization import add_safe_globals  # PyTorch >= 2.6
    except Exception:
        return

    to_add = []

    # Add comprehensive torch.nn module classes
    try:
        import torch.nn as nn
        # Layer modules
        torch_nn_classes = [
            # Convolution layers
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            # Pooling layers
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
            # Padding layers
            nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReplicationPad1d,
            nn.ReplicationPad2d, nn.ReplicationPad3d, nn.ZeroPad2d,
            # Normalization layers
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
            # Activation functions
            nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU,
            nn.Sigmoid, nn.Tanh, nn.Softmax, nn.LogSoftmax, nn.Hardswish, nn.SiLU, nn.Mish,
            # Linear layers
            nn.Linear, nn.Bilinear,
            # Dropout layers
            nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
            # Recurrent layers
            nn.RNN, nn.LSTM, nn.GRU,
            # Transformer layers
            nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder,
            nn.TransformerEncoderLayer, nn.TransformerDecoderLayer,
            nn.MultiheadAttention,
            # Container modules
            nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict,
            # Embedding layers
            nn.Embedding, nn.EmbeddingBag,
            # Upsampling
            nn.Upsample, nn.UpsamplingBilinear2d, nn.UpsamplingNearest2d,
            # Other
            nn.Flatten, nn.Unflatten, nn.Identity,
        ]
        to_add.extend(torch_nn_classes)

        # Add from torch.nn.modules submodules
        for submod in [
            'conv', 'linear', 'pooling', 'batchnorm', 'dropout', 'activation',
            'normalization', 'padding', 'sparse', 'distance', 'loss', 'container',
            'transformer', 'rnn', 'pixelshuffle', 'upsampling', 'fold',
        ]:
            try:
                mod = __import__(f'torch.nn.modules.{submod}', fromlist=['*'])
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if hasattr(obj, '__module__') and 'torch.nn' in obj.__module__:
                        to_add.append(obj)
            except Exception:
                pass

    except Exception as e:
        print(f"Warning: Could not add torch.nn classes: {e}")

    # Ultralytics classes commonly present in YOLOv8 checkpoints
    try:
        from ultralytics.nn import tasks as utasks
        for name, obj in inspect.getmembers(utasks, inspect.isclass):
            to_add.append(obj)
    except Exception:
        pass

    # Directly import known critical classes that may not be in module namespace
    critical_classes = []
    for class_path in [
        'ultralytics.utils.loss.DFLoss',
        'ultralytics.utils.loss.BboxLoss',
        'ultralytics.utils.loss.v8DetectionLoss',
        'ultralytics.utils.tal.TaskAlignedAssigner',
        'ultralytics.utils.IterableSimpleNamespace',
    ]:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            critical_classes.append(cls)
            print(f"  Directly imported: {class_path}")
        except Exception as e:
            print(f"  Failed to import {class_path}: {e}")
    to_add.extend(critical_classes)

    # Ultralytics utils classes (IterableSimpleNamespace, etc.)
    for modname in [
        'ultralytics.utils',
        'ultralytics.utils.tal',      # Task Aligned Assigner
        'ultralytics.utils.metrics',  # Metrics classes
        'ultralytics.utils.ops',      # Operations
        'ultralytics.utils.loss',     # Loss functions
        'ultralytics.utils.checks',   # Check utilities
        'ultralytics.utils.torch_utils',  # Torch utilities
        'ultralytics.nn.modules',
        'ultralytics.nn.modules.conv',
        'ultralytics.nn.modules.block',
        'ultralytics.nn.modules.head',
        'ultralytics.nn.modules.common',
    ]:
        try:
            mod = __import__(modname, fromlist=['*'])
            # Get all classes from the module (more aggressive)
            for name in dir(mod):
                if not name.startswith('_'):  # Skip private attributes
                    try:
                        obj = getattr(mod, name)
                        if inspect.isclass(obj):
                            to_add.append(obj)
                    except Exception:
                        pass
        except Exception:
            continue

    # Filter Nones and deduplicate
    to_add = [c for c in set(to_add) if c is not None]

    try:
        from torch.serialization import add_safe_globals
        add_safe_globals(to_add)
        print(f"Added {len(to_add)} classes to PyTorch safe globals for weights loading")
    except Exception as e:
        print(f"Warning: Could not add safe globals: {e}")


def run_yolo_inference(config):
    print("Running YOLOv8 inference...")
    # Resolve YOLO weights; prefer epoch100 if available
    model_path = resolve_path(
        config['models']['yolo']['path'],
        candidates=[
            'models/YOLO/epoch100.pt',
            'models/YOLO/best.pt',
            'models/YOLO/yolo_inference_model_final/yolo_inference_model.pt',
        ],
        expect_dir=False,
    )
    images_dir = resolve_path(
        config['dataset']['images_dir'],
        expect_dir=True,
    )
    output_dir = config['output']['directory']
    conf_threshold = config['inference_params']['confidence_threshold']

    # PyTorch 2.6+ changed default to weights_only=True which breaks loading older
    # Ultralytics checkpoints with removed/renamed classes (e.g., DFLoss).
    # Create dummy DFLoss class for backward compatibility with older checkpoints.
    import ultralytics.utils.loss as loss_module
    if not hasattr(loss_module, 'DFLoss'):
        class DFLoss(torch.nn.Module):
            """Dummy DFLoss class for backward compatibility with older ultralytics checkpoints."""
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, *args, **kwargs):
                return args[0] if args else torch.tensor(0.0)
        loss_module.DFLoss = DFLoss
        print(f"Created dummy DFLoss class in ultralytics.utils.loss for backward compatibility")

    # Monkey-patch torch_safe_load to use weights_only=False for our trusted checkpoint.
    import ultralytics.nn.tasks as tasks_module
    def patched_torch_safe_load(file, *args, **kwargs):
        """Load checkpoint with weights_only=False for compatibility with older checkpoints."""
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        return ckpt, file  # Return both checkpoint and file path as expected by ultralytics
    tasks_module.torch_safe_load = patched_torch_safe_load
    print(f"Patched ultralytics.nn.tasks.torch_safe_load to use weights_only=False")

    print(f"Loading YOLO weights from: {model_path}")
    model = YOLO(model_path)
    image_files = get_image_files(images_dir)

    coco_results = []
    # Prefer COCO image id mapping if GT is available; fallback to enumeration
    gt_map = build_coco_image_id_map(config['dataset'].get('annotations_path'))
    image_id_map = gt_map if gt_map else {name: i for i, name in enumerate(image_files)}
    # Optional category mapper
    cat_map = get_category_id_mapper(config, 'yolo')

    # Performance measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for image_name in tqdm(image_files, desc="YOLO Inference"):
        image_path = os.path.join(images_dir, image_name)
        results = model(image_path, conf=conf_threshold, verbose=False)
        
        image_id = image_id_map.get(image_name, image_id_map.get(os.path.basename(image_name)))
        if image_id is None:
            # Fallback to enumeration index if not found
            image_id = image_files.index(image_name)

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                score = box.conf[0].item()
                cls_idx = int(box.cls[0].item())
                # Map class index to COCO category id
                if cat_map is not None:
                    category_id = int(cat_map.get(cls_idx, cls_idx))
                else:
                    category_id = cls_idx + 1  # common default when COCO ids start at 1

                coco_results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "score": score,
                })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'yolo_predictions.json')
    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=4)
    print(f"YOLO predictions saved to {output_path}")

    # Save performance metrics
    total_time = max(time.time() - start_time, 1e-9)
    fps = len(image_files) / total_time
    vram_mb = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_mb = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    perf = {"frames": len(image_files), "total_seconds": total_time, "fps": fps, "vram_mb": vram_mb}
    perf_path = os.path.join(output_dir, 'yolo_performance.json')
    with open(perf_path, 'w') as f:
        json.dump(perf, f, indent=4)
    print(f"YOLO performance saved to {perf_path}")

def load_detr_from_checkpoint(checkpoint_path, device):
    """
    Load DETR model from original training checkpoint to preserve Query81 specialization.
    Query 81 achieved 96.3% hit rate in original checkpoint vs 0% in converted model.
    """
    print(f"Loading DETR from original checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model with appropriate configuration
    # Use num_labels=1 for single class (surgical_tool)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,  # SURGICAL TOOL CLASS ONLY
        ignore_mismatched_sizes=True
    )

    # Load weights with STRICT validation to preserve Query81
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    model.to(device)
    model.eval()
    print("Successfully loaded DETR with Query81 specialization preserved")
    return model

def run_detr_inference(config):
    print("Running DETR inference...")

    # Try to load from checkpoint first (preserves Query81), fallback to HF model
    checkpoint_candidates = [
        'models/DETR/checkpoint_epoch_100.pth',
        '../BackgroundFinetuned/Models/DETR/checkpoint_epoch_100.pth',
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        try:
            p = resolve_path(candidate, must_exist=True, expect_dir=False)
            checkpoint_path = p
            break
        except FileNotFoundError:
            continue

    if not checkpoint_path:
        # Fallback to HF converted model (will lose Query81 specialization)
        print("WARNING: Original checkpoint not found, using converted model (Query81 specialization will be lost)")
        model_path = resolve_path(
            config['models']['detr']['path'],
            candidates=[
                'DETR/detr_inference_model_final',
            ],
            expect_dir=True,
        )

    images_dir = resolve_path(
        config['dataset']['images_dir'],
        expect_dir=True,
    )
    output_dir = config['output']['directory']
    conf_threshold = config['inference_params']['confidence_threshold']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from checkpoint or HF
    if checkpoint_path:
        model = load_detr_from_checkpoint(checkpoint_path, device)
        # Create processor from base model
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    else:
        processor = DetrImageProcessor.from_pretrained(model_path)
        model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    
    image_files = get_image_files(images_dir)
    coco_results = []
    # Prefer COCO image id mapping if GT is available; fallback to enumeration
    gt_map = build_coco_image_id_map(config['dataset'].get('annotations_path'))
    image_id_map = gt_map if gt_map else {name: i for i, name in enumerate(image_files)}
    # Optional category mapper
    cat_map = get_category_id_mapper(config, 'detr')

    # Performance measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for image_name in tqdm(image_files, desc="DETR Inference"):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]
        
        image_id = image_id_map.get(image_name, image_id_map.get(os.path.basename(image_name)))
        if image_id is None:
            image_id = image_files.index(image_name)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            cls_idx = int(label.item())
            if cat_map is not None:
                category_id = int(cat_map.get(cls_idx, cls_idx))
            else:
                category_id = cls_idx  # assume DETR already aligns to COCO ids

            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": round(score.item(), 3),
            })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detr_predictions.json')
    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=4)
    print(f"DETR predictions saved to {output_path}")

    # Save performance metrics
    total_time = max(time.time() - start_time, 1e-9)
    fps = len(image_files) / total_time
    vram_mb = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_mb = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    perf = {"frames": len(image_files), "total_seconds": total_time, "fps": fps, "vram_mb": vram_mb}
    perf_path = os.path.join(output_dir, 'detr_performance.json')
    with open(perf_path, 'w') as f:
        json.dump(perf, f, indent=4)
    print(f"DETR performance saved to {perf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for YOLO or DETR models.")
    parser.add_argument('--model_type', type=str, required=True, choices=['yolo', 'detr'], help='Type of model to run inference for.')
    args = parser.parse_args()

    config = load_config()
    
    if args.model_type == 'yolo':
        run_yolo_inference(config)
    elif args.model_type == 'detr':
        run_detr_inference(config)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repo root

def resolve_path(preferred: str, candidates: list[str] | None = None, must_exist: bool = True, expect_dir: bool = False) -> str:
    """
    Resolve a path robustly:
    - Expand env vars and ~
    - Try as-is (abs or rel to CWD)
    - Try relative to repo root
    - Try fallback candidates relative to repo root
    Prints the chosen path for clarity.
    """
    paths_tried = []
    def _ok(p: Path) -> bool:
        return p.is_dir() if expect_dir else p.is_file()

    # 1) Expand
    p = Path(os.path.expandvars(os.path.expanduser(preferred)))
    # 2) As-is
    if _ok(p):
        print(f"Resolved path: {p}")
        return str(p)
    paths_tried.append(p)
    # 3) Relative to repo root
    pr = ROOT / p
    if _ok(pr):
        print(f"Resolved path (repo-relative): {pr}")
        return str(pr)
    paths_tried.append(pr)
    # 4) Candidates
    for c in (candidates or []):
        cpath = ROOT / c
        if _ok(cpath):
            print(f"Resolved path (fallback): {cpath}")
            return str(cpath)
        paths_tried.append(cpath)

    if must_exist:
        print("ERROR: Could not resolve path. Tried:")
        for t in paths_tried:
            print(f" - {t}")
        raise FileNotFoundError(f"Path not found: {preferred}")
    else:
        print(f"Using non-existing path: {preferred}")
        return str(p)
