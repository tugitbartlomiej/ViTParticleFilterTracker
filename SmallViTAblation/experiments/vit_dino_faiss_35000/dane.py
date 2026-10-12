"""Streaming oznaczonego TRAIN z tego samego źródła i polityki GT co indeks.
Nie zapisuje danych, nie ładuje modelu, nie inicjalizuje CUDA i nie miesza splitów.
"""
import hashlib
import json
import csv
from pathlib import Path
import worker as canonical

def train_archives():
    groups=json.loads(canonical.SPLIT.read_text(encoding='utf-8'))['groups']
    return [p for p in sorted(canonical.ARCHIVES.glob('*.zip')) if groups.get(p.stem)=='train']

def iter_labeled_train(on_defect=None):
    """Yield (validated_record, PIL_RGB), sorted film then frame; validate before yield.

    Jedno dekodowanie na klatkę. Ścieżki i ID deduplikowane. Każdy obraz musi mieć
    dodatnią ramkę narzędzia; same cornea/pupil nie wystarczają. Błędy dekodowania
    i GT poza obrazem są zgłaszane przez opcjonalny callback on_defect(record,error).
    Nie zakładaj, że wszystkie 283577 kandydatów przejdą pełne dekodowanie.
    """
    csv.field_size_limit(32*1024*1024)
    seen_paths=set(); seen_ids=set()
    for archive in train_archives():
        records,stat=canonical.inventory_one(archive)
        for r in records:
            key=(r['archive'],r['member'])
            if key in seen_paths or r['id'] in seen_ids: continue
            seen_paths.add(key); seen_ids.add(r['id'])
            try:
                rgb,raw=canonical.read_image(r)
                w,h=rgb.size
                if any(b['bbox_xywh'][0]+b['bbox_xywh'][2]>w+1e-4 or b['bbox_xywh'][1]+b['bbox_xywh'][3]>h+1e-4 for b in r['gt']):
                    raise ValueError('bbox outside decoded image')
                record=dict(r,width=w,height=h,image_sha256=hashlib.sha256(raw).hexdigest(),pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            except Exception as e:
                if on_defect is not None: on_defect(r,str(e))
                continue
            yield record,rgb

# Exact shared DINO transform: RGB shorter256 BICUBIC -> crop224 -> ImageNet.
DINO_PREPROCESS=canonical.PRE
