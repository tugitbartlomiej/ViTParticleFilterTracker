"""
Centralna konfiguracja sciezek dla LOKALNEJ ewaluacji ablacji DETR V1-V7.

Wszystkie sciezki zweryfikowane 2026-06-11 (istnienie plikow potwierdzone).
Zadnych operacji na klastrze Eden - wylacznie zasoby lokalne (dyski F: i E:).

Zrodla modeli:
  V1      - produkcyjny checkpoint ep270 (headline artykulu: 85.74% mAP@0.5)
  V2-V5   - jedyne istniejace kopie (na Edenie usuniete w cleanupie 2026-04-20)
  V6-V7   - swieze best_modele pobrane z Edena 2026-06-11 (NIE uzywac starych
            kopii z ablation_best\\v6, v7 - sa sprzed udanych rerunow!)
"""

from pathlib import Path

ROOT_VIT = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker")
ABLATIONS_DIR = ROOT_VIT / "Eden" / "Checkpoints" / "Ablations"
FRESH_DIR = ABLATIONS_DIR / "Fresh_from_Eden_2026-06-11_v6_v7_completed_runs"

# --- Checkpointy per wariant -------------------------------------------------
CHECKPOINTS = {
    # v0 = model NIEKUROWANY: ep170 (koniec treningu bazowego na 91k, start fine-tune).
    # W artykule cytowany jako 2.04% mAP@0.5 (Q81, valid) - punkt odniesienia curation gain.
    "v0": ROOT_VIT / "Eden" / "Checkpoints" / "DETR" / "DETR_Training_Sessions"
          / "2025-12-13_20kDataset_small_LR" / "ckpt_20k_finetune"
          / "checkpoint_epoch_170.pth",
    "v1": ROOT_VIT / "Eden" / "Checkpoints" / "DETR" / "DETR_Training_Sessions"
          / "2025-12-13_20kDataset_small_LR" / "ckpt_20k_finetune"
          / "checkpoint_epoch_270.pth",
    "v2": ABLATIONS_DIR / "ablation_best" / "v2" / "best_model.pth",
    "v3": ABLATIONS_DIR / "ablation_best" / "v3" / "best_model.pth",
    "v4": ABLATIONS_DIR / "ablation_best" / "v4" / "best_model.pth",
    "v5": ABLATIONS_DIR / "ablation_best" / "v5" / "best_model.pth",
    "v6": FRESH_DIR / "v6" / "best_model.pth",
    "v7": FRESH_DIR / "v7" / "best_model.pth",
}

# Etykiety wariantow wg faktycznie wytrenowanych konfiguracji (sesja 2026-05-24).
# UWAGA: PLAN.md z marca ma inna definicje V4 (NO_CLUSTERING) - przed wpisaniem
# do artykulu potwierdzic etykiety z logow treningu (Ablations\logs\).
VARIANT_LABELS = {
    "v0": "UNCURATED baseline (ep170, przed fine-tune)",
    "v1": "FULL pipeline (baseline, ep270)",
    "v2": "NO_FOURIER",
    "v3": "NO_EL2N",
    "v4": "QUALITY_ONLY",
    "v5": "RANDOM",
    "v6": "DIVERSITY_ONLY (DINO+K-Center)",
    "v7": "EL2N_ONLY (random+EL2N rerank)",
}

# --- Dataset 1: external benchmark (cross-dataset) ---------------------------
# Publiczny zbior operacji zaciemy z innego osrodka. Foldery train/valid/test
# to tylko opakowanie autorow - W ARTYKULE traktowane jako JEDEN niedzielony
# zbior testowy (scalamy do FULL=2333; CLEAN=FULL minus 48 leaked).
EXTERNAL_DIR = Path(r"E:\cataract_surgery_Instruments_detection.v1i.coco")
EXTERNAL_SPLITS = ("train", "valid", "test")  # 2083 + 200 + 50 = 2333 obrazow

# Lista 48 obrazow, ktore przeciekly do zbioru treningowego 20k
LEAKED_JSON = ROOT_VIT / "YOLO_DETR_Benchmarks" / "scripts" / "leaked_valid_images.json"

# --- Dataset 2: same-distribution (wlasny zbior, video-level split) ----------
# Anotacje per wariant: <SAMEDIST_ANN_DIR>/<vid>/annotations_{val,test}.json
# (identyczny video-level split we wszystkich wariantach, seed=42)
SAMEDIST_ANN_DIR = ABLATIONS_DIR / "ablation"
SAMEDIST_IMAGES_DIR = (ROOT_VIT / "Eden" / "Datasets" / "datasets_20250606.tar"
                       / "datasets_20250606" / "datasets"
                       / "DETR_augmented_dataset_20250218" / "images")  # 90,141 jpg

# --- Wyniki -------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# --- Parametry ewaluacji (spojne z wczesniejszymi skryptami projektu) ---------
CONF_THRESHOLD = 0.3   # prog pewnosci dla TP/FP/FN; w protokole Q81 takze prog predykcji
IOU_THRESHOLD = 0.5
DETR_QUERY_ID = 81     # protokol Q81: jedyny slot DETR uzywany w benchmarkach artykulu
DEFAULT_BATCH_SIZE = 4  # RTX 3070 8 GB
SMOKE_LIMIT = 50


def check_paths(variant_id: str, need_samedist: bool = True) -> bool:
    """Sprawdza istnienie wszystkich plikow potrzebnych dla wariantu.

    Zwraca True gdy komplet; wypisuje brakujace sciezki.
    need_samedist=False dla runow czysto external (np. v0 nie ma anotacji ablacyjnych).
    """
    ok = True
    targets = {
        f"checkpoint {variant_id}": CHECKPOINTS[variant_id],
        "external train json": EXTERNAL_DIR / "train" / "_annotations.coco.json",
        "external valid json": EXTERNAL_DIR / "valid" / "_annotations.coco.json",
        "external test json": EXTERNAL_DIR / "test" / "_annotations.coco.json",
        "leaked json": LEAKED_JSON,
        "same-dist images dir": SAMEDIST_IMAGES_DIR,
    }
    if need_samedist:
        targets["same-dist val json"] = SAMEDIST_ANN_DIR / variant_id / "annotations_val.json"
        targets["same-dist test json"] = SAMEDIST_ANN_DIR / variant_id / "annotations_test.json"
    for name, path in targets.items():
        if not path.exists():
            print(f"[BRAK] {name}: {path}")
            ok = False
    if ok:
        print(f"[OK] Wszystkie sciezki dla {variant_id} istnieja.")
    return ok
