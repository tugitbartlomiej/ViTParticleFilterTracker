from __future__ import annotations

import json
import sys
from pathlib import Path


def main(root: Path, data: Path) -> None:
    pool = json.loads((data / "pool.json").read_text(encoding="utf-8"))
    results = []
    for path in sorted(root.glob("*/RESULT.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if len(results) != 9:
        raise RuntimeError(f"Oczekiwano 9 wyników, znaleziono {len(results)}")
    control = next(r for r in results if r["variant"]["name"] == "control")
    by_id = {r["variant"]["slot_id"]: r for r in results if r["variant"]["name"] != "control"}
    rows = []
    for mapping in pool["step_mapping"]:
        r = by_id[mapping["image_id"]]
        before, after = r["probe_before"], r["probe_after"]
        c_after = control["probe_after"]
        rows.append({
            **mapping,
            "variant": r["variant"]["name"],
            "best_iou_before": before["best_iou"],
            "best_iou_after": after["best_iou"],
            "best_iou_delta": after["best_iou"] - before["best_iou"],
            "best_iou_delta_vs_control": after["best_iou"] - c_after["best_iou"],
            "top_confidence_before": before["top_confidence"],
            "top_confidence_after": after["top_confidence"],
            "top_confidence_delta": after["top_confidence"] - before["top_confidence"],
            "top_confidence_delta_vs_control": after["top_confidence"] - c_after["top_confidence"],
            "best_iou_confidence_after": after["best_iou_confidence"],
            "guard_ap50_after": r["guard_ap50_after"],
            "guard_ap50_delta": r["guard_ap50_delta"],
            "guard_ap50_delta_vs_control": r["guard_ap50_after"] - control["guard_ap50_after"],
        })
    summary = {
        "state": "complete",
        "variants_computed": len(results),
        "interpolation_steps_mapped": len(rows),
        "control": control,
        "rows": rows,
        "best_by_iou_vs_control": max(rows, key=lambda x: x["best_iou_delta_vs_control"]),
        "best_by_confidence_vs_control": max(rows, key=lambda x: x["top_confidence_delta_vs_control"]),
        "results": results,
    }
    (root / "COMPARISON.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Wyniki podmiany 1 z 256 obrazów", "", f"Kontrola po 500 krokach: best IoU {control['probe_after']['best_iou']:.4f}, top confidence {control['probe_after']['top_confidence']:.4f}, guard AP50 {control['guard_ap50_after']:.4f}.", "", "| krok obrazu SD | naturalny obraz | cosine DINO | best IoU po | różnica IoU vs kontrola | confidence po | różnica confidence vs kontrola | guard AP50 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['query']} | {row['image_id']} | {row['cosine']:.4f} | {row['best_iou_after']:.4f} | {row['best_iou_delta_vs_control']:+.4f} | {row['top_confidence_after']:.4f} | {row['top_confidence_delta_vs_control']:+.4f} | {row['guard_ap50_after']:.4f} |")
    (root / "WYNIKI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_iou": summary["best_by_iou_vs_control"], "best_confidence": summary["best_by_confidence_vs_control"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
