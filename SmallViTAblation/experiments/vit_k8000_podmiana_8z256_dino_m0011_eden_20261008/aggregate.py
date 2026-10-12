from __future__ import annotations

import json
import sys
from pathlib import Path


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


root = Path(sys.argv[1])
control = json.loads((root / "control_256_original" / "RESULT.json").read_text(encoding="utf-8"))
combined = json.loads((root / "combined_8_replacements" / "RESULT.json").read_text(encoding="utf-8"))


def gain(result: dict, key: str) -> float:
    return result["probe_after"][key] - result["probe_before"][key]


summary = {
    "state": "complete",
    "design": "248 shared original images + 8 DINO replacements versus 256 original images",
    "control": {
        "probe_before": control["probe_before"],
        "probe_after": control["probe_after"],
        "guard_ap50_before": control["guard_ap50_before"],
        "guard_ap50_after": control["guard_ap50_after"],
        "guard_ap50_gain": control["guard_ap50_delta"],
    },
    "combined_8": {
        "removed_ids": combined["variant"]["removed_ids"],
        "inserted_ids": combined["variant"]["inserted_ids"],
        "probe_before": combined["probe_before"],
        "probe_after": combined["probe_after"],
        "guard_ap50_before": combined["guard_ap50_before"],
        "guard_ap50_after": combined["guard_ap50_after"],
        "guard_ap50_gain": combined["guard_ap50_delta"],
    },
    "effect_vs_control": {
        "best_iou_difference_in_differences": gain(combined, "best_iou") - gain(control, "best_iou"),
        "top_confidence_difference_in_differences": gain(combined, "top_confidence") - gain(control, "top_confidence"),
        "top_confidence_iou_difference_in_differences": gain(combined, "top_confidence_iou") - gain(control, "top_confidence_iou"),
        "guard_ap50_difference_in_differences": combined["guard_ap50_delta"] - control["guard_ap50_delta"],
    },
    "checkpoints": {
        "control": {"path": control["checkpoint"], "sha256": control["checkpoint_sha256"]},
        "combined_8": {"path": combined["checkpoint"], "sha256": combined["checkpoint_sha256"]},
    },
}
atomic_json(root / "COMPARISON.json", summary)

e = summary["effect_vs_control"]
lines = [
    "# Wynik podmiany 8 z 256 obrazów",
    "",
    "Obie gałęzie zaczęły od tego samego checkpointa K8000 i wykonały 500 aktualizacji. Kontrola zawierała 256 oryginalnych obrazów; wariant zawierał 248 tych samych obrazów i osiem unikalnych obrazów DINO.",
    "",
    f"- Efekt best IoU ponad kontrolę: {e['best_iou_difference_in_differences']:+.6f}",
    f"- Efekt top confidence ponad kontrolę: {e['top_confidence_difference_in_differences']:+.6f}",
    f"- Efekt IoU ramki top-confidence ponad kontrolę: {e['top_confidence_iou_difference_in_differences']:+.6f}",
    f"- Efekt guard AP50 ponad kontrolę: {e['guard_ap50_difference_in_differences']:+.6f}",
    "",
    "Wynik jest pojedynczą próbą z jednym seedem. Do potwierdzenia wpływu zestawu potrzebne są powtórzenia na kilku seedach i innych ośmiu usuwanych obrazach kontrolnych.",
]
(root / "WYNIK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
