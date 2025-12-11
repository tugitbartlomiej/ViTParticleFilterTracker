import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to builder_config.yaml')
    args = ap.parse_args()

    try:
        import yaml  # type: ignore
    except Exception:
        print("Install pyyaml: pip install pyyaml")
        sys.exit(1)

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    videos_dir = cfg['source_videos_dir']
    out_dir = Path(cfg['output_dir'])
    out_images = out_dir / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract & select frames
    sampling = cfg.get('sampling', {})
    run([
        sys.executable, str(Path(__file__).with_name('extract_and_select_frames.py')),
        '--videos', videos_dir,
        '--out', str(out_images),
        '--target-fps', str(sampling.get('target_fps', 2.0)),
        '--blur-th', str(sampling.get('blur_threshold', 80)),
        '--phash-th', str(sampling.get('phash_threshold', 10)),
        '--max-per-video', str(sampling.get('max_frames_per_video', 1500) or 0),
        '--manifest', str(out_dir / 'frames_manifest.json'),
    ])

    # 2) Optional: YOLO pseudo-labels
    yolo_cfg = cfg.get('yolo_pseudo_labels', {})
    preds_json = out_dir / 'pseudo_annotations.json'
    if yolo_cfg.get('enabled', False):
        run([
            sys.executable, str(Path(__file__).with_name('run_yolo_pseudo_labels.py')),
            '--images', str(out_images),
            '--weights', yolo_cfg['weights_path'],
            '--out', str(preds_json),
            '--category-id', str(int(cfg['category']['id'])),
            '--category-name', str(cfg['category']['name']),
            '--conf', str(yolo_cfg.get('conf_threshold', 0.25)),
        ])
    else:
        # If user provides an existing COCO predictions JSON
        if not preds_json.exists():
            print("YOLO pseudo-labels disabled and no preds JSON present. Create one before splitting.")
            sys.exit(1)

    # 3) Build grouped splits & COCO
    splits = cfg.get('splits', {'train': 0.7, 'val': 0.15, 'test': 0.15})
    negs = cfg.get('negatives', {'include': True, 'max_ratio': 0.3})
    run([
        sys.executable, str(Path(__file__).with_name('split_and_build_coco.py')),
        '--images', str(out_images),
        '--preds', str(preds_json),
        '--out', str(out_dir),
        '--category-id', str(int(cfg['category']['id'])),
        '--category-name', str(cfg['category']['name']),
        '--train', str(splits.get('train', 0.7)),
        '--val', str(splits.get('val', 0.15)),
        '--test', str(splits.get('test', 0.15)),
        '--include-negatives' if negs.get('include', True) else '',
        '--neg-ratio', str(negs.get('max_ratio', 0.3)),
    ])

    print("Dataset build complete.")


if __name__ == '__main__':
    main()

