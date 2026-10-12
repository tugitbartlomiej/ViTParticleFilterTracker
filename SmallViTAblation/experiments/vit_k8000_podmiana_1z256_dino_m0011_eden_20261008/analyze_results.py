from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "vit_k8000_podmiana_1z256_dino_m0011_eden_20261008"
RESULTS = OUTPUT / "results"
POOL = json.loads((OUTPUT / "input_pack" / "pool.json").read_text(encoding="utf-8"))


def load(name: str) -> dict:
    return json.loads((RESULTS / name / "RESULT.json").read_text(encoding="utf-8"))


control = load("control")
ctrl_iou_gain = control["probe_after"]["best_iou"] - control["probe_before"]["best_iou"]
ctrl_conf_gain = control["probe_after"]["top_confidence"] - control["probe_before"]["top_confidence"]
ctrl_guard_gain = control["guard_ap50_delta"]

rows = []
by_image = {}
for idx, image_id in enumerate(POOL["interventions"], 1):
    result = load(f"intervention_{idx:02d}")
    before, after = result["probe_before"], result["probe_after"]
    row = {
        "variant": f"intervention_{idx:02d}",
        "image_id": image_id,
        "iou_before": before["best_iou"],
        "iou_after": after["best_iou"],
        "iou_gain": after["best_iou"] - before["best_iou"],
        "iou_effect_vs_control": (after["best_iou"] - before["best_iou"]) - ctrl_iou_gain,
        "confidence_before": before["top_confidence"],
        "confidence_after": after["top_confidence"],
        "confidence_gain": after["top_confidence"] - before["top_confidence"],
        "confidence_effect_vs_control": (after["top_confidence"] - before["top_confidence"]) - ctrl_conf_gain,
        "guard_ap50_before": result["guard_ap50_before"],
        "guard_ap50_after": result["guard_ap50_after"],
        "guard_ap50_gain": result["guard_ap50_delta"],
        "guard_ap50_effect_vs_control": result["guard_ap50_delta"] - ctrl_guard_gain,
    }
    rows.append(row)
    by_image[image_id] = row

with (RESULTS / "EFEKT_PODMIANY.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

mapped = []
for item in POOL["step_mapping"]:
    row = dict(by_image[item["image_id"]])
    row.update(query=item["query"], dino_cosine=item["cosine"])
    mapped.append(row)
with (RESULTS / "EFEKT_11_KROKOW.csv").open("w", encoding="utf-8-sig", newline="") as f:
    fields = ["query", "image_id", "dino_cosine"] + list(rows[0])
    fields = list(dict.fromkeys(fields))
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(mapped)

labels = [r["variant"].replace("intervention_", "I") for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
for ax, key, title in [
    (axes[0], "iou_effect_vs_control", "Efekt podmiany: best IoU"),
    (axes[1], "confidence_effect_vs_control", "Efekt podmiany: confidence"),
    (axes[2], "guard_ap50_effect_vs_control", "Efekt podmiany: guard AP50"),
]:
    vals = [r[key] for r in rows]
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="#263238", lw=1)
    ax.set_title(title)
    ax.set_xlabel("wariant naturalnego obrazu")
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Różnica przyrostów względem kontroli (difference-in-differences)")
fig.savefig(RESULTS / "EFEKT_PODMIANY.png", dpi=180)
plt.close(fig)

best_guard = max(rows, key=lambda r: r["guard_ap50_effect_vs_control"])
best_iou = max(rows, key=lambda r: r["iou_effect_vs_control"])
worst = min(rows, key=lambda r: r["confidence_effect_vs_control"])

lines = [
    "# Analiza efektu podmiany jednego obrazu",
    "",
    "Każda gałąź zaczęła od identycznego checkpointa K8000 i wykonała 500 aktualizacji na 255 wspólnych obrazach oraz jednym zmienianym obrazie. Efekt obrazu liczony jest jako różnica przyrostu danej gałęzi i przyrostu kontroli. To usuwa dużą poprawę wynikającą z samego dalszego treningu.",
    "",
    f"Kontrola: best IoU {control['probe_before']['best_iou']:.4f} -> {control['probe_after']['best_iou']:.4f}; confidence {control['probe_before']['top_confidence']:.4f} -> {control['probe_after']['top_confidence']:.4f}; guard AP50 {control['guard_ap50_before']:.4f} -> {control['guard_ap50_after']:.4f}.",
    "",
    "| wariant | naturalny obraz | efekt IoU | efekt confidence | efekt guard AP50 |",
    "|---|---|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['variant']} | {r['image_id']} | {r['iou_effect_vs_control']:+.4f} | "
        f"{r['confidence_effect_vs_control']:+.4f} | {r['guard_ap50_effect_vs_control']:+.4f} |"
    )
lines += [
    "",
    f"Najlepszy efekt na guard AP50 dał {best_guard['variant']} ({best_guard['image_id']}): {best_guard['guard_ap50_effect_vs_control']:+.4f} ponad kontrolę; jednocześnie efekt IoU wyniósł {best_guard['iou_effect_vs_control']:+.4f}.",
    f"Najlepszy efekt na sondzie IoU dał {best_iou['variant']} ({best_iou['image_id']}): {best_iou['iou_effect_vs_control']:+.4f}, ale guard AP50 zmienił się o {best_iou['guard_ap50_effect_vs_control']:+.4f} względem kontroli.",
    f"Najbardziej szkodliwy był {worst['variant']} ({worst['image_id']}): efekt confidence {worst['confidence_effect_vs_control']:+.4f} i IoU {worst['iou_effect_vs_control']:+.4f}.",
    "",
    "To jest eksperyment z jednym seedem i jedną podmianą na 256 obrazów. Małe efekty rzędu kilku tysięcznych są nierozstrzygające; należy powtórzyć najbardziej obiecujące i kontrolne warianty na kilku seedach. Różne GPU dawały drobne różnice już w kroku zero przy BF16, dlatego raport opiera się na zmianie wewnątrz każdej gałęzi, a nie tylko na wartości końcowej.",
]
(RESULTS / "ANALIZA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({"best_guard": best_guard, "best_iou": best_iou, "worst": worst}, ensure_ascii=False, indent=2))
