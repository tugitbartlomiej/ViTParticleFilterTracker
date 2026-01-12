"""
YOLO Checkpoint Analysis Script with Visualizations
Analyzes training progression from saved YOLOv8 checkpoints.

Usage:
    python analyze_yolo_checkpoints.py [--checkpoint-dir PATH] [--output-dir PATH]
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import argparse
import matplotlib.pyplot as plt
import re

# Default paths
DEFAULT_CHECKPOINT_DIR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\exp\weights")
DEFAULT_OUTPUT_DIR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\yolo")


DTYPES_FLOAT = (torch.float16, torch.float32, torch.bfloat16)


def load_yolo_checkpoint(path):
    """Load YOLO checkpoint and return its contents."""
    print(f"Loading: {path.name}")
    try:
        # YOLO checkpoints contain custom classes, need weights_only=False
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        return ckpt
    except Exception as e:
        print(f"  Warning: Could not load {path.name}: {e}")
        return None


def _get_state_dict_from_ultralytics_model(obj):
    """Best-effort extraction of a torch state_dict from various Ultralytics objects."""
    if obj is None:
        return None

    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict()
        except Exception:
            return None

    # Some Ultralytics wrappers keep the actual nn.Module under .model
    inner = getattr(obj, "model", None)
    if inner is not None and hasattr(inner, "state_dict"):
        try:
            return inner.state_dict()
        except Exception:
            return None

    return None


def _summarize_weight_norms(state_dict):
    """Compute simple weight norm stats from a state dict."""
    norms = []
    for _, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if tensor.dtype not in DTYPES_FLOAT:
            continue
        # float() for stable norm on fp16/bf16
        norms.append(float(torch.norm(tensor.float()).item()))

    if not norms:
        return None

    return {
        "weight_norm_mean": float(np.mean(norms)),
        "weight_norm_max": float(np.max(norms)),
        "weight_norm_std": float(np.std(norms)),
        "num_weight_tensors": int(len(norms)),
    }


def analyze_yolo_model(ckpt):
    """Analyze YOLO model state."""
    info = {
        'has_model': False,
        'has_ema': False,
        'has_optimizer': False,
        'epoch': None,
        'best_fitness': None,
        'train_args': {},
    }

    if ckpt is None:
        return info

    # Check for model
    model_obj = ckpt.get('model', None)
    ema_obj = ckpt.get('ema', None)

    info['has_model'] = model_obj is not None
    info['has_ema'] = ema_obj is not None

    # Weight statistics: prefer 'model' if present, otherwise fall back to EMA
    weight_source = None
    state_dict = None

    if model_obj is not None:
        state_dict = _get_state_dict_from_ultralytics_model(model_obj)
        weight_source = "model" if state_dict is not None else None

    if state_dict is None and ema_obj is not None:
        state_dict = _get_state_dict_from_ultralytics_model(ema_obj)
        weight_source = "ema" if state_dict is not None else None

    if state_dict is not None:
        info["weight_source"] = weight_source
        norms_summary = _summarize_weight_norms(state_dict)
        if norms_summary:
            info.update(norms_summary)

    # Check for optimizer
    if 'optimizer' in ckpt:
        info['has_optimizer'] = True
        opt_state = ckpt['optimizer']

        if isinstance(opt_state, dict) and 'param_groups' in opt_state:
            info['learning_rates'] = [g.get('lr', 0) for g in opt_state['param_groups']]
            info['weight_decays'] = [g.get('weight_decay', 0) for g in opt_state['param_groups']]
            info['momentums'] = [g.get('momentum', 0) for g in opt_state['param_groups']]

    # Get epoch
    info['epoch'] = ckpt.get('epoch', None)

    # Get best fitness
    info['best_fitness'] = ckpt.get('best_fitness', None)

    # Get training args
    if 'train_args' in ckpt:
        train_args = ckpt['train_args']
        if hasattr(train_args, '__dict__'):
            info['train_args'] = {k: v for k, v in train_args.__dict__.items()
                                 if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
        elif isinstance(train_args, dict):
            info['train_args'] = {k: v for k, v in train_args.items()
                                 if isinstance(v, (int, float, str, bool))}

    return info


def create_visualizations(results, output_dir):
    """Create matplotlib visualizations for YOLO training analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    def _normalize(vals):
        if not vals:
            return [], None, None
        vmin = min(vals)
        vmax = max(vals)
        if vmax == vmin:
            return [0.0 for _ in vals], vmin, vmax
        return [(v - vmin) / (vmax - vmin) for v in vals], vmin, vmax

    epochs = [r['epoch'] for r in results if r['epoch'] is not None]

    if not epochs:
        print("No epoch data available for visualization")
        return

    # ==========================================
    # Figure 1: Weight Norm Evolution
    # ==========================================
    weight_norms = [r.get('weight_norm_mean', 0) for r in results if r['epoch'] is not None]

    if weight_norms and any(w > 0 for w in weight_norms):
        fig, ax = plt.subplots(figsize=(12, 6))
        weight_x_norm, _, _ = _normalize(epochs)
        weight_y_norm, _, _ = _normalize(weight_norms)
        ax.plot(weight_x_norm, weight_y_norm, 'b-o', linewidth=2, markersize=8, label='Mean Weight Norm')
        ax.set_xlabel('Normalized Epoch', fontsize=12)
        ax.set_ylabel('Normalized Mean Weight Norm', fontsize=12)
        ax.set_title('YOLOv8 Weight Norm Evolution During Training', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(epochs) > 1:
            z = np.polyfit(weight_x_norm, weight_y_norm, 1)
            p = np.poly1d(z)
            ax.plot(weight_x_norm, p(weight_x_norm), 'r--', alpha=0.5, label='Trend')

        plt.tight_layout()
        plt.savefig(output_dir / 'yolo_weight_norm.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'yolo_weight_norm.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: yolo_weight_norm.png/pdf")

    # ==========================================
    # Figure 2: Learning Rate Schedule
    # ==========================================
    lrs = [r.get('learning_rates', [0])[0] if r.get('learning_rates') else 0 for r in results if r['epoch'] is not None]

    if lrs and any(lr > 0 for lr in lrs):
        fig, ax = plt.subplots(figsize=(12, 6))
        lr_x_norm, _, _ = _normalize(epochs)
        lr_y_norm, _, _ = _normalize(lrs)
        ax.plot(lr_x_norm, lr_y_norm, 'g-s', linewidth=2, markersize=8)
        ax.set_xlabel('Normalized Epoch', fontsize=12)
        ax.set_ylabel('Normalized Learning Rate', fontsize=12)
        ax.set_title('YOLOv8 Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'yolo_learning_rate.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'yolo_learning_rate.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: yolo_learning_rate.png/pdf")

    # ==========================================
    # Figure 3: Best Fitness Evolution
    # ==========================================
    fitness = [r.get('best_fitness', 0) or 0 for r in results if r['epoch'] is not None]

    if fitness and any(f > 0 for f in fitness):
        fig, ax = plt.subplots(figsize=(12, 6))
        fit_x_norm, _, _ = _normalize(epochs)
        fit_y_norm, _, _ = _normalize(fitness)
        ax.plot(fit_x_norm, fit_y_norm, 'm-^', linewidth=2, markersize=8)
        ax.set_xlabel('Normalized Epoch', fontsize=12)
        ax.set_ylabel('Normalized Best Fitness', fontsize=12)
        ax.set_title('YOLOv8 Best Fitness Over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate max
        if max(fitness) > 0:
            max_idx = np.argmax(fitness)
            ax.annotate(f'Best: {fitness[max_idx]:.4f}',
                       xy=(fit_x_norm[max_idx], fit_y_norm[max_idx]),
                       xytext=(min(fit_x_norm[max_idx] + 0.05, 0.95), min(fit_y_norm[max_idx] + 0.05, 0.95)),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=10, color='green')

        plt.tight_layout()
        plt.savefig(output_dir / 'yolo_fitness.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'yolo_fitness.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: yolo_fitness.png/pdf")

    # ==========================================
    # Figure 4: Training Summary
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Weight norm
    ax = axes[0, 0]
    if weight_norms and any(w > 0 for w in weight_norms):
        ax.plot(weight_x_norm, weight_y_norm, 'b-o', linewidth=2, markersize=6)
        ax.set_title('Weight Norm Evolution', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No weight norm data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Normalized Epoch')
    ax.set_ylabel('Normalized Mean Weight Norm')
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[0, 1]
    if lrs and any(lr > 0 for lr in lrs):
        ax.plot(lr_x_norm, lr_y_norm, 'g-s', linewidth=2, markersize=6)
        ax.set_title('Learning Rate Schedule', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Normalized Epoch')
    ax.set_ylabel('Normalized Learning Rate')
    ax.grid(True, alpha=0.3)

    # Fitness
    ax = axes[1, 0]
    if fitness and any(f > 0 for f in fitness):
        ax.plot(fit_x_norm, fit_y_norm, 'm-^', linewidth=2, markersize=6)
        ax.set_title('Best Fitness', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No fitness data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Normalized Epoch')
    ax.set_ylabel('Normalized Fitness')
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')

    # Get train args from first result with them
    train_args = {}
    for r in results:
        if r.get('train_args'):
            train_args = r['train_args']
            break

    num_params = results[0].get('num_params', 0) if results else 0

    summary_text = f"""
    YOLOv8 Training Summary
    ═══════════════════════════════════

    Checkpoints Analyzed: {len(results)}
    Epochs: {min(epochs) if epochs else 'N/A'} → {max(epochs) if epochs else 'N/A'}

    Model:
      • Parameters: {num_params:,}
      • Has EMA: {any(r.get('has_ema') for r in results)}

    Training Config:
      • Batch Size: {train_args.get('batch', 'N/A')}
      • Image Size: {train_args.get('imgsz', 'N/A')}
      • Epochs Target: {train_args.get('epochs', 'N/A')}

    Weight Norm:
      • Start: {f'{weight_norms[0]:.4f}' if weight_norms and weight_norms[0] > 0 else 'N/A'}
      • End: {f'{weight_norms[-1]:.4f}' if weight_norms and weight_norms[-1] > 0 else 'N/A'}
      • Change: {f'{((weight_norms[-1]-weight_norms[0])/weight_norms[0]*100):+.1f}%' if weight_norms and weight_norms[0] > 0 else 'N/A'}

    Best Fitness: {f'{max(fitness):.4f}' if fitness and max(fitness) > 0 else 'N/A'}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('YOLOv8 Training Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_training_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'yolo_training_summary.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: yolo_training_summary.png/pdf")

    print(f"\n[OK] All visualizations saved to: {output_dir}")


def analyze_checkpoints(ckpt_dir, output_dir):
    """Analyze all YOLO checkpoints in directory."""
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir)

    # Find all checkpoints (epoch*.pt pattern)
    epoch_files = []
    for f in ckpt_dir.glob("epoch*.pt"):
        match = re.search(r'epoch(\d+)\.pt', f.name)
        if match:
            epoch_files.append((int(match.group(1)), f))

    epoch_files.sort(key=lambda x: x[0])

    # Also check for best.pt
    best_file = ckpt_dir / "best.pt"
    emergency_file = ckpt_dir / "emergency_checkpoint.pt"

    print(f"\n{'='*60}")
    print(f"  YOLO CHECKPOINT ANALYSIS")
    print(f"  Found {len(epoch_files)} epoch checkpoints in:")
    print(f"  {ckpt_dir}")
    print(f"{'='*60}\n")

    results = []

    for epoch_num, ckpt_path in epoch_files:
        ckpt = load_yolo_checkpoint(ckpt_path)
        if ckpt is None:
            continue

        info = analyze_yolo_model(ckpt)
        info['epoch'] = epoch_num
        info['checkpoint_file'] = ckpt_path.name
        results.append(info)

        del ckpt

    # Analyze best.pt if exists
    if best_file.exists():
        print(f"\nAnalyzing best.pt...")
        ckpt = load_yolo_checkpoint(best_file)
        if ckpt:
            best_info = analyze_yolo_model(ckpt)
            best_info['checkpoint_file'] = 'best.pt'
            print(f"  Best model epoch: {best_info.get('epoch', 'N/A')}")
            print(f"  Best fitness: {best_info.get('best_fitness', 'N/A')}")
            del ckpt

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  TRAINING PROGRESSION SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Epoch':<8} {'Weight Norm':<14} {'LR':<12} {'Fitness':<12} {'Model':<6} {'EMA':<5} {'Opt':<5}")
    print("-" * 75)

    for r in results:
        epoch_str = str(r.get('epoch', 'N/A'))
        wn_str = f"{r.get('weight_norm_mean', 0):.4f}" if r.get('weight_norm_mean') else "N/A"
        lr_str = f"{r.get('learning_rates', [0])[0]:.2e}" if r.get('learning_rates') else "N/A"
        fit_str = f"{r.get('best_fitness', 0):.4f}" if r.get('best_fitness') else "N/A"
        model_str = "✓" if r.get('has_model') else "✗"
        ema_str = "✓" if r.get('has_ema') else "✗"
        opt_str = "✓" if r.get('has_optimizer') else "✗"

        print(f"{epoch_str:<8} {wn_str:<14} {lr_str:<12} {fit_str:<12} {model_str:<6} {ema_str:<5} {opt_str:<5}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "yolo_checkpoint_analysis.json"

    # Make results JSON serializable
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
        serializable_results.append(sr)

    with open(output_file, 'w') as f:
        json.dump({
            'summary': serializable_results,
            'checkpoint_dir': str(ckpt_dir),
            'num_checkpoints': len(results),
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")

    # Create visualizations
    if results:
        print("\n" + "="*60)
        print("  GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        create_visualizations(results, output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze YOLO training checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                       help='Directory containing YOLO checkpoints')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Directory for output files and visualizations')

    args = parser.parse_args()

    analyze_checkpoints(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
