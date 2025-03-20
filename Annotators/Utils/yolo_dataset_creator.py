import datetime
import os
import random
import shutil

# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\DeepSortYolo\ProcessedVideos"  # Main directory
ANNOTATED_KEYWORD = "annotated"
RAW_KEYWORD = "raw"

# Train/Val split ratio
TRAIN_SPLIT = 0.8

# YOLO classes
NAMES = ["class0"]  # example classes
NC = len(NAMES)

# Create output folder name with today's date
today_str = datetime.datetime.now().strftime("%Y%m%d")
OUTPUT_DATASET_DIR = os.path.join(ROOT_DIR, f"yolo_dataset_{today_str}")

# ============================================================
# END OF CONFIG
# ============================================================

def find_subdir_by_keyword(base_path: str, keyword: str) -> str:
    """
    Finds a subdirectory in base_path whose name contains the given keyword.
    Returns full path or raises FileNotFoundError if none is found.
    """
    keyword_lower = keyword.lower()
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path) and keyword_lower in entry.lower():
            return full_path
    raise FileNotFoundError(f"No subdirectory containing '{keyword}' found in: {base_path}")


def collect_pairs(annotated_dir: str, raw_dir: str):
    """
    Returns list of (image_path, label_path) pairs matching by filename (without extension).
    """
    txt_files = [f for f in os.listdir(annotated_dir) if f.lower().endswith(".txt")]
    txt_basenames = set(os.path.splitext(f)[0] for f in txt_files)

    img_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".jpg")]
    img_basenames = set(os.path.splitext(f)[0] for f in img_files)

    common_basenames = txt_basenames.intersection(img_basenames)
    pairs = []
    for base in common_basenames:
        img_path = os.path.join(raw_dir, base + ".jpg")
        txt_path = os.path.join(annotated_dir, base + ".txt")
        pairs.append((img_path, txt_path))

    return pairs


def prepare_yolo_structure(output_dir: str) -> None:
    """
    Creates the folder structure for YOLO: images/train, images/val, labels/train, labels/val.
    """
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


def copy_files_with_progress(pairs, output_dir: str, train_split: float) -> None:
    """
    Randomly splits pairs into train and val, copies files, and shows progress in %.
    """
    random.shuffle(pairs)
    train_size = int(len(pairs) * train_split)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    all_pairs = [("train", p) for p in train_pairs] + [("val", p) for p in val_pairs]
    total = len(all_pairs)

    for i, (subset, (img_path, txt_path)) in enumerate(all_pairs, start=1):
        img_name = os.path.basename(img_path)
        txt_name = os.path.basename(txt_path)

        shutil.copy2(img_path, os.path.join(output_dir, "images", subset, img_name))
        shutil.copy2(txt_path, os.path.join(output_dir, "labels", subset, txt_name))

        progress = (i / total) * 100
        print(f"Progress: {i}/{total} ({progress:.2f}%)")


def create_data_yaml(output_dir: str, nc: int, names: list) -> None:
    """
    Creates data.yaml with train/val paths, number of classes, and class names.
    """
    data_yaml_path = os.path.join(output_dir, "data.yaml")

    train_path = os.path.join(output_dir, "images", "train")
    val_path = os.path.join(output_dir, "images", "val")

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {train_path}\n")
        f.write(f"val: {val_path}\n")
        f.write(f"nc: {nc}\n")
        f.write("names: [")
        f.write(", ".join(f"'{name}'" for name in names))
        f.write("]\n")


def main():
    # Locate annotated and raw directories
    annotated_dir = find_subdir_by_keyword(ROOT_DIR, ANNOTATED_KEYWORD)
    raw_dir = find_subdir_by_keyword(ROOT_DIR, RAW_KEYWORD)

    # Collect matching (image, label) pairs
    pairs = collect_pairs(annotated_dir, raw_dir)
    if not pairs:
        print("No matching pairs found.")
        return
    print(f"Found {len(pairs)} matching pairs.")

    # Prepare YOLO folder structure
    prepare_yolo_structure(OUTPUT_DATASET_DIR)

    # Copy files with progress
    copy_files_with_progress(pairs, OUTPUT_DATASET_DIR, TRAIN_SPLIT)

    # Create data.yaml
    create_data_yaml(OUTPUT_DATASET_DIR, NC, NAMES)

    print(f"Dataset created in: {OUTPUT_DATASET_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
