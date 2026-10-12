from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[2] / "outputs" / "vit_k8000_podmiana_8z256_dino_m0011_eden_20261008" / "results"
control = json.loads((root / "control_256_original" / "RESULT.json").read_text(encoding="utf-8"))
combined = json.loads((root / "combined_8_replacements" / "RESULT.json").read_text(encoding="utf-8"))
comparison = json.loads((root / "COMPARISON.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
for result, label, color in [(control, "kontrola: 256 oryginalnych", "#546e7a"), (combined, "248 + 8 DINO", "#e76f51")]:
    trajectory = result["probe_trajectory"]
    steps = [x["step"] for x in trajectory]
    axes[0].plot(steps, [x["best_iou"] for x in trajectory], label=label, color=color, lw=2)
    axes[1].plot(steps, [x["top_confidence"] for x in trajectory], label=label, color=color, lw=2)
    axes[2].plot(steps, [x["best_confidence_times_iou"] for x in trajectory], label=label, color=color, lw=2)
for ax, title in zip(axes, ["Best IoU sondy m0011", "Top confidence", "Confidence x IoU"]):
    ax.set_title(title)
    ax.set_xlabel("aktualizacja")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("wartość")
axes[0].legend(loc="lower right")
fig.suptitle("Wpływ wspólnej podmiany 8 z 256 obrazów")
fig.savefig(root / "TRAJEKTORIE.png", dpi=180)
plt.close(fig)


def peak(result: dict, key: str) -> dict:
    return max(result["probe_trajectory"], key=lambda x: x[key])


effect = comparison["effect_vs_control"]
ctrl_peak = peak(control, "best_confidence_times_iou")
comb_peak = peak(combined, "best_confidence_times_iou")
lines = [
    "# Analiza podmiany 8 z 256 obrazów",
    "",
    "Kontrola i wariant zostały wykonane kolejno na tym samym GPU od identycznego checkpointa K8000. Wariant usunął osiem oryginalnych obrazów i wstawił osiem unikalnych obrazów DINO.",
    "",
    "## Wynik po 500 aktualizacjach",
    "",
    f"- Kontrola: IoU {control['probe_after']['best_iou']:.4f}, confidence {control['probe_after']['top_confidence']:.4f}, guard AP50 {control['guard_ap50_after']:.4f}.",
    f"- Osiem podmian: IoU {combined['probe_after']['best_iou']:.4f}, confidence {combined['probe_after']['top_confidence']:.4f}, guard AP50 {combined['guard_ap50_after']:.4f}.",
    f"- Efekt ponad kontrolę: IoU {effect['best_iou_difference_in_differences']:+.4f}, confidence {effect['top_confidence_difference_in_differences']:+.4f}, guard AP50 {effect['guard_ap50_difference_in_differences']:+.4f}.",
    "",
    "## Trajektoria",
    "",
    f"- Najwyższe confidence x IoU kontroli: {ctrl_peak['best_confidence_times_iou']:.4f} w kroku {ctrl_peak['step']}.",
    f"- Najwyższe confidence x IoU wariantu: {comb_peak['best_confidence_times_iou']:.4f} w kroku {comb_peak['step']}.",
    "",
    "Osiem podmian przesunęło ramkę nieco bliżej etykiety sondy, lecz obniżyło jej pewność i pogorszyło AP50 na odłożonych filmach. W tej konfiguracji zestaw nie jest lepszy jako ogólna pula treningowa. Wynik nadal pochodzi z jednego seeda i jednego wyboru ośmiu usuwanych obrazów.",
]
(root / "ANALIZA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
