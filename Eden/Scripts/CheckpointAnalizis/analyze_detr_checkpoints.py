"""
DETR Checkpoint Analysis Script with Visualizations
Analyzes training progression from saved DETR checkpoints.

Usage:
    python analyze_detr_checkpoints.py [--checkpoint-dir PATH] [--output-dir PATH]
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Default paths
DEFAULT_CHECKPOINT_DIR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR")
DEFAULT_OUTPUT_DIR = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr")


def load_checkpoint(path):
    """Load checkpoint and return its contents."""
    print(f"Loading: {path.name}")
    return torch.load(path, map_location='cpu')


def analyze_model_weights(state_dict):
    """Analyze model weights statistics."""
    stats = {}
    for name, param in state_dict.items():
        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            param_np = param.float().numpy()
            stats[name] = {
                'mean': float(np.mean(param_np)),
                'std': float(np.std(param_np)),
                'min': float(np.min(param_np)),
                'max': float(np.max(param_np)),
                'norm': float(np.linalg.norm(param_np)),
                'has_nan': bool(np.isnan(param_np).any()),
                'has_inf': bool(np.isinf(param_np).any()),
                'shape': list(param.shape),
                'numel': param.numel()
            }
    return stats


def analyze_optimizer_state(optimizer_state):
    """Extract learning rates and Adam moments from optimizer."""
    info = {
        'param_groups': [],
        'adam_moments': {}
    }

    if 'param_groups' in optimizer_state:
        for i, group in enumerate(optimizer_state['param_groups']):
            group_info = {
                'group_idx': i,
                'lr': group.get('lr', None),
                'weight_decay': group.get('weight_decay', None),
                'betas': group.get('betas', None),
                'eps': group.get('eps', None),
            }
            info['param_groups'].append(group_info)

    if 'state' in optimizer_state:
        moment_stats = {'exp_avg_norms': [], 'exp_avg_sq_norms': []}
        for param_id, state in optimizer_state['state'].items():
            if 'exp_avg' in state:
                moment_stats['exp_avg_norms'].append(
                    float(torch.norm(state['exp_avg']).item())
                )
            if 'exp_avg_sq' in state:
                moment_stats['exp_avg_sq_norms'].append(
                    float(torch.norm(state['exp_avg_sq']).item())
                )

        if moment_stats['exp_avg_norms']:
            info['adam_moments'] = {
                'exp_avg_mean_norm': np.mean(moment_stats['exp_avg_norms']),
                'exp_avg_max_norm': np.max(moment_stats['exp_avg_norms']),
                'exp_avg_sq_mean_norm': np.mean(moment_stats['exp_avg_sq_norms']),
                'exp_avg_sq_max_norm': np.max(moment_stats['exp_avg_sq_norms']),
            }

    return info


def create_visualizations(results, layer_evolution, output_dir):
    """Create matplotlib visualizations for training analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    def _normalize(vals):
        if not vals:
            return [], None, None
        vmin = min(vals)
        vmax = max(vals)
        if vmax == vmin:
            return [0.0 for _ in vals], vmin, vmax
        return [(v - vmin) / (vmax - vmin) for v in vals], vmin, vmax

    epochs = [r['epoch'] for r in results]
    losses = [r['loss'] for r in results if r['loss'] is not None]
    loss_epochs = [r['epoch'] for r in results if r['loss'] is not None]

    # ==========================================
    # Figure 1: Training Loss Over Epochs
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_x_norm, _, _ = _normalize(loss_epochs)
    loss_y_norm, _, _ = _normalize(losses)
    ax.plot(loss_x_norm, loss_y_norm, 'b-o', linewidth=2, markersize=8, label='Training Loss')
    ax.set_xlabel('Normalized Epoch', fontsize=12)
    ax.set_ylabel('Normalized Loss', fontsize=12)
    ax.set_title('DETR Training Loss Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate min loss
    min_loss_idx = np.argmin(losses)
    min_x = loss_x_norm[min_loss_idx]
    min_y = loss_y_norm[min_loss_idx]
    ax.annotate(f'Min: {losses[min_loss_idx]:.4f}',
                xy=(min_x, min_y),
                xytext=(min(min_x + 0.05, 0.95), min(min_y + 0.05, 0.95)),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig(output_dir / 'detr_training_loss.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'detr_training_loss.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: detr_training_loss.png/pdf")

    # ==========================================
    # Figure 2: Weight Norm Evolution
    # ==========================================
    weight_norms = [r.get('weight_norm_mean', 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    weight_x_norm, _, _ = _normalize(epochs)
    weight_y_norm, _, _ = _normalize(weight_norms)
    ax.plot(weight_x_norm, weight_y_norm, 'r-s', linewidth=2, markersize=8)
    ax.set_xlabel('Normalized Epoch', fontsize=12)
    ax.set_ylabel('Normalized Mean Weight Norm', fontsize=12)
    ax.set_title('DETR Weight Norm Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(weight_x_norm, weight_y_norm, 1)
    p = np.poly1d(z)
    ax.plot(weight_x_norm, p(weight_x_norm), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'detr_weight_norm.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'detr_weight_norm.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: detr_weight_norm.png/pdf")

    # ==========================================
    # Figure 3: Key Layer Evolution
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layer_colors = {
        'class_labels_classifier.weight': '#2ecc71',
        'bbox_predictor.layers.2.weight': '#3498db',
        'model.encoder.layers.0.self_attn.out_proj.weight': '#e74c3c',
        'model.decoder.layers.0.self_attn.out_proj.weight': '#9b59b6',
    }

    layer_names_short = {
        'class_labels_classifier.weight': 'Classification Head',
        'bbox_predictor.layers.2.weight': 'BBox Predictor',
        'model.encoder.layers.0.self_attn.out_proj.weight': 'Encoder Self-Attn',
        'model.decoder.layers.0.self_attn.out_proj.weight': 'Decoder Self-Attn',
    }

    for idx, (layer_name, evolution) in enumerate(list(layer_evolution.items())[:4]):
        ax = axes[idx // 2, idx % 2]
        layer_epochs = [e['epoch'] for e in evolution]
        layer_norms = [e['norm'] for e in evolution]

        color = layer_colors.get(layer_name, 'blue')
        short_name = layer_names_short.get(layer_name, layer_name.split('.')[-1])

        layer_x_norm, _, _ = _normalize(layer_epochs)
        layer_y_norm, _, _ = _normalize(layer_norms)
        ax.plot(layer_x_norm, layer_y_norm, '-o', color=color, linewidth=2, markersize=8)
        ax.set_xlabel('Normalized Epoch', fontsize=10)
        ax.set_ylabel('Normalized Weight Norm', fontsize=10)
        ax.set_title(short_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Calculate and show trend
        if len(layer_norms) > 1:
            change = (layer_norms[-1] - layer_norms[0]) / layer_norms[0] * 100
            ax.text(0.02, 0.98, f'Change: {change:+.1f}%', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   color='green' if change > 0 else 'red')

    plt.suptitle('DETR Key Layer Weight Evolution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'detr_layer_evolution.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'detr_layer_evolution.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: detr_layer_evolution.png/pdf")

    # ==========================================
    # Figure 4: Adam Moments Analysis
    # ==========================================
    adam_m = [r.get('adam_moments', {}).get('exp_avg_mean_norm', 0) for r in results]
    adam_v = [r.get('adam_moments', {}).get('exp_avg_sq_mean_norm', 0) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    adam_x_norm, _, _ = _normalize(epochs)
    adam_m_norm, _, _ = _normalize(adam_m)
    adam_v_norm, _, _ = _normalize(adam_v)

    ax1.plot(adam_x_norm, adam_m_norm, 'g-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Normalized Epoch', fontsize=12)
    ax1.set_ylabel('Normalized exp_avg Norm (m)', fontsize=12)
    ax1.set_title('Adam First Moment (Gradient Moving Average)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(adam_x_norm, adam_v_norm, 'purple', marker='o', linewidth=2, markersize=8)
    ax2.set_xlabel('Normalized Epoch', fontsize=12)
    ax2.set_ylabel('Normalized exp_avg_sq Norm (v)', fontsize=12)
    ax2.set_title('Adam Second Moment (Squared Gradient MA)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DETR Adam Optimizer State Evolution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'detr_adam_moments.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'detr_adam_moments.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: detr_adam_moments.png/pdf")

    # ==========================================
    # Figure 5: Combined Training Summary
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(loss_x_norm, loss_y_norm, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Normalized Epoch')
    ax.set_ylabel('Normalized Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Weight Norm
    ax = axes[0, 1]
    ax.plot(weight_x_norm, weight_y_norm, 'r-s', linewidth=2, markersize=8)
    ax.set_xlabel('Normalized Epoch')
    ax.set_ylabel('Normalized Mean Weight Norm')
    ax.set_title('Weight Norm', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Learning Rate (bar chart)
    ax = axes[1, 0]
    lrs_main = [r.get('learning_rates', [0])[0] for r in results]
    lrs_backbone = [r.get('learning_rates', [0, 0])[1] if len(r.get('learning_rates', [])) > 1 else 0 for r in results]

    lrs_all = lrs_main + lrs_backbone
    lrs_norm, _, _ = _normalize(lrs_all)
    if lrs_norm:
        lrs_main_norm = lrs_norm[:len(lrs_main)]
        lrs_backbone_norm = lrs_norm[len(lrs_main):]
    else:
        lrs_main_norm = []
        lrs_backbone_norm = []

    x_norm, _, _ = _normalize(epochs)
    if len(epochs) > 1:
        step = 1.0 / (len(epochs) - 1)
        width = step * 0.4
    else:
        width = 0.1

    ax.bar([x - width / 2 for x in x_norm], lrs_main_norm, width, label='Main LR', color='#3498db')
    ax.bar([x + width / 2 for x in x_norm], lrs_backbone_norm, width, label='Backbone LR', color='#2ecc71')
    ax.set_xlabel('Normalized Epoch')
    ax.set_ylabel('Normalized Learning Rate')
    ax.set_title('Learning Rates', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Model Summary (text)
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    DETR Training Summary
    ═══════════════════════════════════

    Checkpoints Analyzed: {len(results)}
    Epochs: {epochs[0]} → {epochs[-1]}

    Loss:
      • Start: {losses[0]:.4f}
      • End: {losses[-1]:.4f}
      • Min: {min(losses):.4f} (epoch {loss_epochs[np.argmin(losses)]})
      • Change: {(losses[-1]-losses[0])/losses[0]*100:+.1f}%

    Weight Norm:
      • Start: {weight_norms[0]:.4f}
      • End: {weight_norms[-1]:.4f}
      • Change: {(weight_norms[-1]-weight_norms[0])/weight_norms[0]*100:+.1f}%

    Total Parameters: {results[0].get('total_params', 0):,}

    Status: {'✓ No NaN/Inf detected' if not any(r.get('nan_layers') or r.get('inf_layers') for r in results) else '⚠ Issues detected'}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('DETR Training Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'detr_training_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'detr_training_summary.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: detr_training_summary.png/pdf")

    print(f"\n[OK] All visualizations saved to: {output_dir}")


def compare_checkpoints(ckpt_dir, output_dir, min_epoch=None, max_epoch=None, epochs=None):
    """Compare checkpoints in directory with optional epoch filtering."""
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir)

    # Find all checkpoints
    all_files = []
    for p in ckpt_dir.glob("checkpoint_epoch_*.pth"):
        try:
            ep = int(p.stem.split('_')[-1])
        except Exception:
            continue
        all_files.append((ep, p))

    if epochs:
        epoch_set = set(epochs)
        all_files = [pair for pair in all_files if pair[0] in epoch_set]
    else:
        if min_epoch is not None:
            all_files = [pair for pair in all_files if pair[0] >= min_epoch]
        if max_epoch is not None:
            all_files = [pair for pair in all_files if pair[0] <= max_epoch]

    ckpt_files = [p for _, p in sorted(all_files, key=lambda x: x[0])]

    if not ckpt_files:
        print("No checkpoints found!")
        return

    print(f"\n{'='*60}")
    print(f"  DETR CHECKPOINT ANALYSIS")
    print(f"  Found {len(ckpt_files)} checkpoints in:")
    print(f"  {ckpt_dir}")
    print(f"{'='*60}\n")

    results = []
    layer_evolution = defaultdict(list)

    # Key layers to track
    key_layers = [
        'class_labels_classifier.weight',
        'bbox_predictor.layers.2.weight',
        'model.encoder.layers.0.self_attn.out_proj.weight',
        'model.decoder.layers.0.self_attn.out_proj.weight',
        'model.backbone.conv_encoder.model.layer4.2.conv3.weight',
    ]

    for ckpt_path in ckpt_files:
        epoch = int(ckpt_path.stem.split('_')[-1])
        try:
            ckpt = load_checkpoint(ckpt_path)
        except Exception as exc:
            print(f"  Warning: skipping {ckpt_path.name}: {exc}")
            continue

        epoch_result = {
            'epoch': epoch,
            'loss': ckpt.get('loss', None),
            'has_model': 'model_state_dict' in ckpt,
            'has_optimizer': 'optimizer_state_dict' in ckpt,
            'has_scaler': 'scaler_state_dict' in ckpt,
        }

        if 'model_state_dict' in ckpt:
            weight_stats = analyze_model_weights(ckpt['model_state_dict'])

            nan_layers = [k for k, v in weight_stats.items() if v['has_nan']]
            inf_layers = [k for k, v in weight_stats.items() if v['has_inf']]

            epoch_result['nan_layers'] = nan_layers
            epoch_result['inf_layers'] = inf_layers
            epoch_result['total_params'] = sum(v['numel'] for v in weight_stats.values())

            for layer_name in key_layers:
                if layer_name in weight_stats:
                    layer_evolution[layer_name].append({
                        'epoch': epoch,
                        'norm': weight_stats[layer_name]['norm'],
                        'mean': weight_stats[layer_name]['mean'],
                        'std': weight_stats[layer_name]['std'],
                    })

            all_norms = [v['norm'] for v in weight_stats.values()]
            epoch_result['weight_norm_mean'] = float(np.mean(all_norms))
            epoch_result['weight_norm_max'] = float(np.max(all_norms))

        if 'optimizer_state_dict' in ckpt:
            opt_info = analyze_optimizer_state(ckpt['optimizer_state_dict'])
            epoch_result['learning_rates'] = [g['lr'] for g in opt_info['param_groups']]
            epoch_result['adam_moments'] = opt_info.get('adam_moments', {})

        results.append(epoch_result)
        del ckpt

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TRAINING PROGRESSION SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Epoch':<8} {'Loss':<12} {'LR (main)':<12} {'LR (backbone)':<14} {'Weight Norm':<12} {'Issues'}")
    print("-" * 80)

    for r in results:
        loss_str = f"{r['loss']:.6f}" if r['loss'] else "N/A"
        lr_main = r.get('learning_rates', [None])[0]
        lr_backbone = r.get('learning_rates', [None, None])[1] if len(r.get('learning_rates', [])) > 1 else None
        lr_main_str = f"{lr_main:.2e}" if lr_main else "N/A"
        lr_back_str = f"{lr_backbone:.2e}" if lr_backbone else "N/A"
        weight_norm = f"{r.get('weight_norm_mean', 0):.4f}"

        issues = []
        if r.get('nan_layers'):
            issues.append(f"NaN({len(r['nan_layers'])})")
        if r.get('inf_layers'):
            issues.append(f"Inf({len(r['inf_layers'])})")
        issues_str = ", ".join(issues) if issues else "OK"

        print(f"{r['epoch']:<8} {loss_str:<12} {lr_main_str:<12} {lr_back_str:<14} {weight_norm:<12} {issues_str}")

    # Save results to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "detr_checkpoint_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': results,
            'layer_evolution': {k: v for k, v in layer_evolution.items()}
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")

    # Create visualizations
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    create_visualizations(results, layer_evolution, output_dir)

    return results, layer_evolution


def main():
    parser = argparse.ArgumentParser(description='Analyze DETR training checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                       help='Directory containing DETR checkpoints')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Directory for output files and visualizations')
    parser.add_argument('--min-epoch', type=int, default=None, help='Minimum epoch to include')
    parser.add_argument('--max-epoch', type=int, default=None, help='Maximum epoch to include')
    parser.add_argument('--epochs', type=str, default="", help='Comma-separated epoch list to include')

    args = parser.parse_args()

    epochs = None
    if args.epochs:
        try:
            epochs = [int(x.strip()) for x in args.epochs.split(",") if x.strip()]
        except Exception:
            epochs = None

    compare_checkpoints(args.checkpoint_dir, args.output_dir, args.min_epoch, args.max_epoch, epochs)


if __name__ == "__main__":
    main()
