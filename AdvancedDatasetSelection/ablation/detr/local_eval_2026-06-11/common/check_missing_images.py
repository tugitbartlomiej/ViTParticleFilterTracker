"""Narzedzie diagnostyczne: ile obrazow z anotacji same-dist brakuje na dysku.

Uruchomienie:  py -3.11 check_missing_images.py [v1 ... v7]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_config as cfg

variants = sys.argv[1:] or ["v1"]
disk = {p.name for p in cfg.SAMEDIST_IMAGES_DIR.glob("*.jpg")}
print(f"Plikow jpg na dysku: {len(disk)}")

for vid in variants:
    for split in ("val", "test"):
        ann = cfg.SAMEDIST_ANN_DIR / vid / f"annotations_{split}.json"
        with open(ann, "r", encoding="utf-8") as f:
            imgs = json.load(f)["images"]
        missing = [i["file_name"] for i in imgs if i["file_name"] not in disk]
        print(f"{vid}/{split}: {len(missing)}/{len(imgs)} brakuje"
              + (f" (np. {missing[:3]})" if missing else ""))
