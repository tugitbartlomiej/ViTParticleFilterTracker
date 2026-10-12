"""Pobiera oficjalne wagi COCO do osobnego eksperymentu; bez treningu."""
import argparse
from pathlib import Path
from dane import sha256, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    from rfdetr.assets.model_weights import download_pretrain_weights
    root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    target=root/'rf-detr-nano.pth'
    download_pretrain_weights(str(target))
    write_json(root/'MANIFEST.json',dict(file=str(target),bytes=target.stat().st_size,sha256=sha256(target),source='RF-DETR official COCO pretrained weights'))
    print(target,flush=True)
if __name__=='__main__': main()
