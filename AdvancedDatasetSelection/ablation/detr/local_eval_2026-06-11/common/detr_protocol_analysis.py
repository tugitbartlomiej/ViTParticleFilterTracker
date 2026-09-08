"""
Analiza protokolu DETR na external benchmarku — czy Q81 zaniza DETR-a?

Kontekst: pod kuracja YOLO (34.17% FULL) > DETR (26.04% FULL, protokol Q81).
Zanim zaakceptujemy wniosek, wykluczamy artefakty protokolu. Z zapisanych
predykcji ALL-QUERIES (v1, threshold 0.0 — dokladnie 100 predykcji/obraz,
w kolejnosci slotow) liczymy BEZ GPU:

  1. per-slot mAP@0.5 FULL dla KAZDEGO ze 100 slotow (czy slot 81 jest najlepszy?)
  2. protokol top-1-per-obraz (najpewniejsza predykcja sposrod 100)
  3. slot 81 bez progu 0.3 oraz z progiem (sanity vs 26.04%)
  4. ORACLE: odsetek anotacji, ktore trafia KTORYKOLWIEK slot (IoU>=0.5,
     ignorujac score) — gorna granica DETR-a pod dowolnym protokolem wyboru
  5. all-queries (sanity vs 21.40%)

Uruchomienie:  py -3.11 detr_protocol_analysis.py [--variant v1]
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_config as cfg
from ablation_eval_lib import merge_external, gt_category_id, coco_map, _iou_xywh

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NUM_QUERIES = 100


def main():
    ap = argparse.ArgumentParser(description="Analiza protokolu DETR (per-slot/top-1/oracle)")
    ap.add_argument("--variant", default="v1")
    args = ap.parse_args()

    preds_path = cfg.RESULTS_DIR / args.variant / "external" / "predictions_full.json"
    if not preds_path.exists():
        sys.exit(f"Brak {preds_path} — najpierw uruchom run_eval_external.py (protokol all).")

    out_dir = cfg.RESULTS_DIR / args.variant / "protocol_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, _ = merge_external()
    cat_id = gt_category_id(merged)

    with open(preds_path, encoding="utf-8") as f:
        all_preds = json.load(f)
    print(f"Wczytano {len(all_preds)} predykcji ({args.variant}, all-queries)")

    # Grupowanie po obrazie Z ZACHOWANIEM kolejnosci insercji = kolejnosci slotow
    by_img = defaultdict(list)
    for p in all_preds:
        by_img[p["image_id"]].append(p)
    bad = {i: len(v) for i, v in by_img.items() if len(v) != NUM_QUERIES}
    if bad:
        sys.exit(f"Obrazy bez dokladnie {NUM_QUERIES} predykcji: {list(bad.items())[:5]} "
                 f"— kolejnosc slotow niewiarygodna, przerwano.")
    print(f"OK: {len(by_img)} obrazow x {NUM_QUERIES} slotow (kolejnosc zachowana)")

    results = {}

    # --- sanity: all-queries (oczekiwane ~21.40% dla v1) ---
    results["all_queries"] = coco_map(all_preds, merged, out_dir, "all_queries")

    # --- 1. per-slot mAP ---
    print("\nLiczenie per-slot mAP (100 ewaluacji)...")
    slot_maps = []
    for s in range(NUM_QUERIES):
        sp = [by_img[i][s] for i in by_img]
        m = coco_map(sp, merged, out_dir, f"slot_{s:03d}")
        slot_maps.append((s, m["mAP_0.5"]))
        if (s + 1) % 20 == 0:
            print(f"  ... {s+1}/{NUM_QUERIES}")
    slot_maps.sort(key=lambda x: -x[1])
    results["best_slots_top10"] = [(s, round(v * 100, 2)) for s, v in slot_maps[:10]]
    slot81 = next(v for s, v in slot_maps if s == 81)
    results["slot81_unthresholded_mAP50"] = slot81
    rank81 = next(i for i, (s, _) in enumerate(slot_maps, 1) if s == 81)
    results["slot81_rank"] = rank81

    # --- 2. top-1 per obraz (najwyzszy score sposrod 100) ---
    top1 = [max(by_img[i], key=lambda p: p["score"]) for i in by_img]
    results["top1_per_image"] = coco_map(top1, merged, out_dir, "top1")

    # --- 3. slot 81 z progiem 0.3 (sanity vs 26.04) ---
    s81_t = [by_img[i][81] for i in by_img if by_img[i][81]["score"] >= 0.3]
    results["slot81_thresholded_mAP50"] = coco_map(s81_t, merged, out_dir, "s81_t03")["mAP_0.5"]

    # --- 4. ORACLE: czy ktorykolwiek slot trafia anotacje (IoU>=0.5) ---
    gt_by_img = defaultdict(list)
    for a in merged["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    hit, total = 0, 0
    hit81 = 0
    for img_id, anns in gt_by_img.items():
        preds_i = by_img.get(img_id, [])
        for a in anns:
            total += 1
            if any(_iou_xywh(p["bbox"], a["bbox"]) >= 0.5 for p in preds_i):
                hit += 1
            if preds_i and _iou_xywh(preds_i[81]["bbox"], a["bbox"]) >= 0.5:
                hit81 += 1
    results["oracle_any_slot_recall"] = hit / total
    results["oracle_slot81_recall"] = hit81 / total

    # --- raport ---
    print("\n=== ANALIZA PROTOKOLU DETR (", args.variant, ", FULL 2333/2346 ) ===")
    print(f"all-queries mAP@0.5:            {results['all_queries']['mAP_0.5']*100:.2f}%  (sanity: 21.40%)")
    print(f"slot 81 z progiem 0.3:          {results['slot81_thresholded_mAP50']*100:.2f}%  (sanity: 26.04%)")
    print(f"slot 81 BEZ progu:              {slot81*100:.2f}%")
    print(f"top-1 per obraz (po score):     {results['top1_per_image']['mAP_0.5']*100:.2f}%")
    print(f"NAJLEPSZY slot: #{slot_maps[0][0]} = {slot_maps[0][1]*100:.2f}%  "
          f"(slot 81: ranga {rank81}/100)")
    print(f"top-10 slotow: {results['best_slots_top10']}")
    print(f"ORACLE (ktorykolwiek slot, IoU>=0.5): recall {hit/total*100:.1f}% "
          f"({hit}/{total} anotacji)")
    print(f"ORACLE slot 81:                       recall {hit81/total*100:.1f}%")
    print(f"\nPorownanie: YOLO 20k ep270 FULL = 34.17% (arch.) / 34.26% (swieza)")

    with open(out_dir / "protocol_analysis.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2, default=float)
    print(f"Zapisano: {out_dir / 'protocol_analysis.json'}")


if __name__ == "__main__":
    main()
