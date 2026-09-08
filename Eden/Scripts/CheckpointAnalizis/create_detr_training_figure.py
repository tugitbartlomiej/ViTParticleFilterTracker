"""
Create publication-quality DETR training dynamics figure (3 panels).
Loads ALL available checkpoints for dense epoch coverage.

Panels: Training Loss | Weight Norm | Learning Rates

Usage:
    py -3.11 create_detr_training_figure.py
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CHECKPOINT_DIR = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden"
    r"\Checkpoints\DETR\DETR_Training_Sessions"
    r"\2025-10-31_tooltip_single_class\DETR_Checkpoints"
)
OUTPUT_DIR = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden"
    r"\Scripts\CheckpointAnalizis\output\detr_figure5"
)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_data(ckpt_dir):
    """Load all checkpoints and extract training metrics."""
    files = []
    for p in ckpt_dir.glob("checkpoint_epoch_*.pth"):
        try:
            ep = int(p.stem.split("_")[-1])
            files.append((ep, p))
        except ValueError:
            continue

    files.sort(key=lambda x: x[0])
    print(f"Found {len(files)} checkpoints: {[e for e, _ in files]}")

    data = []
    for epoch, path in files:
        print(f"  Loading epoch {epoch}...")
        ckpt = load_checkpoint(path)
        entry = {"epoch": epoch, "loss": ckpt.get("loss")}

        # Learning rates from optimizer
        opt = ckpt.get("optimizer_state_dict", {})
        if "param_groups" in opt:
            groups = opt["param_groups"]
            entry["lr_main"] = groups[0].get("lr", 0) if len(groups) > 0 else 0
            entry["lr_backbone"] = groups[1].get("lr", 0) if len(groups) > 1 else 0

        # Weight norms from model
        msd = ckpt.get("model_state_dict", {})
        norms = []
        for name, tensor in msd.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
                norms.append(float(torch.norm(tensor.float()).item()))
        if norms:
            entry["weight_norm_mean"] = float(np.mean(norms))

        data.append(entry)
        del ckpt

    return data


def create_figure(data, output_dir):
    """Create publication-quality 3-panel training dynamics figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [d["epoch"] for d in data]
    losses = [d["loss"] for d in data]
    weight_norms = [d["weight_norm_mean"] for d in data]
    lr_main = [d.get("lr_main", 0) for d in data]
    lr_backbone = [d.get("lr_backbone", 0) for d in data]

    # --- Style setup ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })

    fig, axes = plt.subplots(3, 1, figsize=(7, 10))
    fig.subplots_adjust(hspace=0.38)

    # Color palette
    c_loss = "#1f77b4"
    c_loss_ann = "#6baed6"  # lighter blue for annotations
    c_norm = "#d62728"
    c_norm_ann = "#fc9272"  # lighter red for annotations
    c_lr_main = "#2ca02c"
    c_lr_bb = "#ff7f0e"

    # Evenly-spaced x positions to avoid crowding in early epochs
    x_pos = np.arange(len(epochs))
    tick_labels = [str(e) for e in epochs]

    # === Panel 1: Training Loss ===
    ax = axes[0]
    ax.plot(x_pos, losses, color=c_loss, marker="o", markersize=5,
            linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3)
    # Min loss annotation
    min_idx = int(np.argmin(losses))
    ax.scatter(x_pos[min_idx], losses[min_idx], s=100, color=c_loss,
               edgecolors="black", linewidths=1.2, zorder=4, marker="*")
    ax.annotate(
        f"Min: {losses[min_idx]:.4f}\n(epoch {epochs[min_idx]})",
        xy=(x_pos[min_idx], losses[min_idx]),
        xytext=(x_pos[min_idx] - 5, losses[min_idx] + 0.05),
        fontsize=8, color=c_loss_ann,
        arrowprops=dict(arrowstyle="-|>", color=c_loss_ann, lw=1.2,
                        linestyle="--"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=c_loss_ann, alpha=0.9),
    )
    loss_pad = (max(losses) - min(losses)) * 0.12
    ax.set_ylim(min(losses) - loss_pad, max(losses) + loss_pad * 2)
    ax.fill_between(x_pos, ax.get_ylim()[0], losses, alpha=0.06, color=c_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("(a) Training Loss", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels)

    # === Panel 2: Weight Norm ===
    ax = axes[1]
    ax.plot(x_pos, weight_norms, color=c_norm, marker="s", markersize=5,
            linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3)
    # Trend line
    z = np.polyfit(x_pos, weight_norms, 1)
    trend_y = np.polyval(z, x_pos)
    ax.plot(x_pos, trend_y, "--", color=c_norm, alpha=0.4, linewidth=1.2,
            label="Linear trend")
    wn_pad = (max(weight_norms) - min(weight_norms)) * 0.15
    ax.set_ylim(min(weight_norms) - wn_pad, max(weight_norms) + wn_pad * 1.5)
    ax.fill_between(x_pos, ax.get_ylim()[0], weight_norms,
                    alpha=0.06, color=c_norm)
    # Start/end value annotations
    ax.annotate(f"{weight_norms[0]:.2f}", xy=(x_pos[0], weight_norms[0]),
                xytext=(x_pos[0] + 1.5, weight_norms[0] - wn_pad * 0.7),
                fontsize=8, color=c_norm_ann,
                arrowprops=dict(arrowstyle="-|>", color=c_norm_ann, lw=1.2,
                                linestyle="--"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=c_norm_ann, alpha=0.9))
    ax.annotate(f"{weight_norms[-1]:.2f}", xy=(x_pos[-1], weight_norms[-1]),
                xytext=(x_pos[-1] - 4, weight_norms[-1] + wn_pad * 0.5),
                fontsize=8, color=c_norm_ann,
                arrowprops=dict(arrowstyle="-|>", color=c_norm_ann, lw=1.2,
                                linestyle="--"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=c_norm_ann, alpha=0.9))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Weight Norm")
    ax.set_title("(b) Weight Norm", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels)

    # === Panel 3: Learning Rates ===
    ax = axes[2]
    bar_width = 0.35
    ax.bar(x_pos - bar_width / 2, lr_main, bar_width,
           color=c_lr_main, alpha=0.85, label=f"Main LR = {lr_main[0]:.0e}",
           edgecolor="white", linewidth=0.5)
    ax.bar(x_pos + bar_width / 2, lr_backbone, bar_width,
           color=c_lr_bb, alpha=0.85, label=f"Backbone LR = {lr_backbone[0]:.0e}",
           edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(c) Learning Rates", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels)

    fig.suptitle("DETR Training Dynamics (Epochs 20\u2013170)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    # Save
    for fmt in ("png", "pdf", "svg"):
        out_path = output_dir / f"detr_training_dynamics.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    data = extract_data(CHECKPOINT_DIR)
    print(f"\nExtracted data for {len(data)} epochs")
    for d in data:
        print(f"  E{d['epoch']:3d}  loss={d['loss']:.6f}  "
              f"wn={d['weight_norm_mean']:.4f}  "
              f"lr={d.get('lr_main', 0):.2e}/{d.get('lr_backbone', 0):.2e}")
    create_figure(data, OUTPUT_DIR)
    print("\nDone!")
