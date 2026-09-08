"""
Wizualizacja 100 queries DETR (V1 curated 20k, ep270) na external benchmark.

Dla wybranych obrazow rysuje:
  - GT bbox (zielony, gruby)
  - chmura 100 predykcji query (czerwone, alpha = score -> widac ktore pewne)
  - box-ORACLE = slot o max IoU z GT (pomaranczowy) + jego slot idx, IoU, score
  - slot 81 (niebieski przerywany) = realny protokol Q81

Plus podsumowanie ilosciowe: jak pewne sa boxy trafiajace w oracle (czy
przezylyby prog conf=0.3) oraz ktore sloty wygrywaja.

Bez GPU - wszystko z zapisanych predykcji (100 boxow/obraz).
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

BASE = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\ablation\detr\local_eval_2026-06-11\results\v1\external")
DATASET = Path(r"E:\cataract_surgery_Instruments_detection.v1i.coco")
OUT = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\ablation\detr\local_eval_2026-06-11\results\v1\oracle_viz")
OUT.mkdir(parents=True, exist_ok=True)

CONF = 0.3          # prog realnego protokolu
IOU_HIT = 0.5
SLOT_Q81 = 81
N_ORACLE_ONLY = 7   # przyklady: oracle trafil, Q81 spudlowal (najciekawsze)
N_Q81_HIT = 4       # przyklady: Q81 tez trafil


def iou(a, b):
    # a,b = [x,y,w,h]
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


print("Wczytuje GT i predykcje...")
gt = json.load(open(BASE / "gt_FULL.json"))
preds = json.load(open(BASE / "predictions_full.json"))

img_info = {im["id"]: im for im in gt["images"]}
gt_boxes = defaultdict(list)
for a in gt["annotations"]:
    gt_boxes[a["image_id"]].append(a["bbox"])

# predykcje: 100 na obraz, w kolejnosci = slot 0..99
pred_by_img = defaultdict(list)
for p in preds:
    pred_by_img[p["image_id"]].append((p["bbox"], p["score"]))

# === ANALIZA: dla kazdego GT-boxa znajdz najlepszy slot (oracle) i slot81 ===
records = []  # (image_id, gt_idx, best_slot, best_iou, best_score, q81_iou, q81_score)
oracle_hit_scores = []
oracle_hit_winning_slot = []
oracle_hit_below_conf = 0
oracle_hits = 0
for img_id, boxes in pred_by_img.items():
    if img_id not in gt_boxes:
        continue
    # boxes[k] = slot k
    for gi, g in enumerate(gt_boxes[img_id]):
        best_iou, best_slot = 0.0, -1
        for k, (pb, sc) in enumerate(boxes):
            i = iou(g, pb)
            if i > best_iou:
                best_iou, best_slot = i, k
        q81_box, q81_score = boxes[SLOT_Q81]
        q81_iou = iou(g, q81_box)
        records.append((img_id, gi, best_slot, best_iou,
                        boxes[best_slot][1] if best_slot >= 0 else 0.0,
                        q81_iou, q81_score))
        if best_iou >= IOU_HIT:
            oracle_hits += 1
            oracle_hit_scores.append(boxes[best_slot][1])
            oracle_hit_winning_slot.append(best_slot)
            if boxes[best_slot][1] < CONF:
                oracle_hit_below_conf += 1

n_gt = len(records)
print("\n================= PODSUMOWANIE ILOSCIOWE =================")
print(f"Liczba GT-narzedzi (FULL): {n_gt}")
print(f"Oracle-trafienia (jakikolwiek slot, IoU>={IOU_HIT}): {oracle_hits} ({100*oracle_hits/n_gt:.1f}%)")
if oracle_hit_scores:
    ss = sorted(oracle_hit_scores)
    med = ss[len(ss)//2]
    print(f"Score zwycieskiego (oracle) boxa: min {min(ss):.3f}  mediana {med:.3f}  max {max(ss):.3f}")
    print(f"Oracle-trafienia z score < {CONF} (zostalyby ODRZUCONE progiem): "
          f"{oracle_hit_below_conf} ({100*oracle_hit_below_conf/oracle_hits:.1f}% trafien)")
slot_counter = Counter(oracle_hit_winning_slot)
print(f"Najczestsze zwycieskie sloty (oracle): {slot_counter.most_common(8)}")
print(f"Liczba ROZNYCH slotow wygrywajacych: {len(slot_counter)} / 100")
print("=========================================================\n")

# === WYBOR PRZYKLADOW ===
oracle_only = [r for r in records if r[3] >= IOU_HIT and r[5] < IOU_HIT]   # oracle hit, q81 miss
q81_hit = [r for r in records if r[5] >= IOU_HIT]                          # q81 tez trafil
oracle_only.sort(key=lambda r: r[3], reverse=True)   # najlepsze IoU najpierw
q81_hit.sort(key=lambda r: r[5], reverse=True)
chosen = oracle_only[:N_ORACLE_ONLY] + q81_hit[:N_Q81_HIT]
print(f"Rysuje {len(chosen)} przykladow ({min(N_ORACLE_ONLY,len(oracle_only))} oracle-only + {min(N_Q81_HIT,len(q81_hit))} q81-hit)...")


def draw(rec, tag, idx):
    img_id, gi, best_slot, best_iou, best_score, q81_iou, q81_score = rec
    info = img_info[img_id]
    split = info.get("source_split", "train")
    fpath = DATASET / split / info["file_name"]
    if not fpath.exists():
        print(f"  [skip] brak pliku: {fpath}")
        return
    im = Image.open(fpath).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(im)

    boxes = pred_by_img[img_id]
    # chmura 100 query: alpha = pewnosc
    for k, (pb, sc) in enumerate(boxes):
        alpha = max(0.03, min(0.95, sc))
        ax.add_patch(Rectangle((pb[0], pb[1]), pb[2], pb[3], fill=False,
                               edgecolor="red", lw=1.0, alpha=alpha))
    # GT
    g = gt_boxes[img_id][gi]
    ax.add_patch(Rectangle((g[0], g[1]), g[2], g[3], fill=False,
                           edgecolor="lime", lw=3.0, label="GT"))
    # oracle best
    ob = boxes[best_slot][0]
    ax.add_patch(Rectangle((ob[0], ob[1]), ob[2], ob[3], fill=False,
                           edgecolor="orange", lw=2.5,
                           label=f"oracle slot {best_slot}"))
    # slot 81
    qb = boxes[SLOT_Q81][0]
    ax.add_patch(Rectangle((qb[0], qb[1]), qb[2], qb[3], fill=False,
                           edgecolor="deepskyblue", lw=2.0, linestyle="--",
                           label="slot 81 (Q81)"))
    ax.set_title(
        f"img {img_id} [{split}]  |  ORACLE slot {best_slot}: IoU={best_iou:.2f} score={best_score:.3f}\n"
        f"Q81: IoU={q81_iou:.2f} score={q81_score:.3f}  |  prog conf={CONF}",
        fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    fname = OUT / f"{idx:02d}_{tag}_img{img_id}_{split}_oracleIoU{best_iou:.2f}.png"
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano: {fname.name}")


for i, rec in enumerate(chosen):
    tag = "oracleONLY" if rec[5] < IOU_HIT else "q81HIT"
    draw(rec, tag, i)

# === maly raport tekstowy ===
rep = OUT / "_SUMMARY.txt"
with open(rep, "w", encoding="utf-8") as f:
    f.write("Oracle / 100-queries analiza (DETR V1 curated 20k, ep270, external FULL)\n")
    f.write(f"GT-narzedzi: {n_gt}\n")
    f.write(f"Oracle-trafienia (IoU>={IOU_HIT}): {oracle_hits} ({100*oracle_hits/n_gt:.1f}%)\n")
    if oracle_hit_scores:
        ss = sorted(oracle_hit_scores)
        f.write(f"Score oracle-boxa: min {min(ss):.3f} mediana {ss[len(ss)//2]:.3f} max {max(ss):.3f}\n")
        f.write(f"Oracle-trafienia z score<{CONF} (odrzucone progiem): {oracle_hit_below_conf} "
                f"({100*oracle_hit_below_conf/oracle_hits:.1f}%)\n")
    f.write(f"Roznych zwycieskich slotow: {len(slot_counter)}/100\n")
    f.write(f"Top sloty: {slot_counter.most_common(10)}\n")
print(f"\nRaport: {rep}")
print(f"Folder z przykladami: {OUT}")
