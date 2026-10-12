"""Evaluate saved weights on fixed images; never resumes or trains the network."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'app'))
import ruch_naturalny as N
import ruch_zbioru as R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=1500)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = N.NaturalConfig(checkpoint_dir=str(N.DEFAULT_CHECKPOINT_DIR), manifest=str(R.DEFAULT_MANIFEST),
                          logs=str(N.DEFAULT_LOGS), max_images=args.images, end_step=args.end,
                          checkpoint_stride=args.stride, batch_size=args.batch, device=args.device)
    out = ROOT / 'outputs/app_ruch_czasteczek_20261009'
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'POMIAR_NATURALNY_STATUS.json'
    start = time.monotonic()
    last_save = -10.
    def progress(event):
        nonlocal last_save
        elapsed = time.monotonic() - start
        if elapsed - last_save >= 3 or event.get('phase') in ('frame', 'finished'):
            payload = dict(event, elapsed_seconds=elapsed)
            R._json(status_path, payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            last_save = elapsed
    result = N.run(cfg, progress=progress)
    timeline = R.Timeline(result)
    for step in timeline.steps:
        assert timeline.frame(step).shape == (args.images * 600, 4)
    metadata = timeline.metadata
    assert metadata['status'] == 'complete' and metadata['inference_only'] and metadata['weights_unchanged']
    assert not (result / 'baseline.float32.tmp').exists()
    summary = {'status': 'verified', 'path': str(result), 'image_count': timeline.image_count,
               'point_count': timeline.count, 'steps': timeline.steps,
               'checkpoint_sha256': timeline.checkpoint_sha256, 'time_axis': metadata['time_axis'],
               'inference_only': True, 'weights_unchanged': True,
               'elapsed_seconds': time.monotonic() - start,
               'initial_metrics': timeline.stats(timeline.steps[0]),
               'final_metrics': timeline.stats(timeline.steps[-1])}
    R._json(out / f'NATURALNY_WYNIK_{args.images}.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('initial_metrics', 'final_metrics')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
