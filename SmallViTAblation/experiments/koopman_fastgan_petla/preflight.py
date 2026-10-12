"""Odczyt wersji, wejsc i kontraktow API; nie uruchamia trenowania ani GPU."""
import argparse
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out'); args=ap.parse_args()
    report={}
    for name in ['torch','rfdetr','pytorch-lightning','numpy','faiss-cpu','transformers','pycocotools']:
        try: report[name]=md.version(name)
        except md.PackageNotFoundError: report[name]=None
    import rfdetr
    report['rfdetr_path']=rfdetr.__file__
    from rfdetr import RFDETRNano
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    report['model_config']=RFDETRNanoConfig().model_dump(mode='json')
    report['train_fields']=sorted(TrainConfig.model_fields)
    print(json.dumps(report,indent=2))
    if args.out: Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__': main()
