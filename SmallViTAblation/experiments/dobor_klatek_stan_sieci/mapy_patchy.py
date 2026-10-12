"""Eksport trzech z góry wybranych klatek VAL i map odczytu patchy; bez pobierania obrazów.

Export: --diagnostics-root DIAGNOSTYKA_ROOT --out NOWY.json
Render: --render-json NOWY.json --dataset-root Dataset/Cataract-LMM --outdir NOWY_KATALOG
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np
if __package__:
    from .diagnostyka_checkpointow import read_jsonl, readout_logits, sha256
else:
    from diagnostyka_checkpointow import read_jsonl, readout_logits, sha256

WARNING = ('Słaby odczyt bbox uczony na L0; sigmoid nie jest skalibrowanym prawdopodobieństwem. '
           'Mapa nie jest uwagą modelu, segmentacją ani dowodem tożsamości narzędzia. '
           'Pierwsze trzy ID walidacji wybrano alfabetycznie, bez oceny wyników.')
COMPARISONS = [('random', 2), ('random', 10), ('koopman_langevin', 2), ('koopman_langevin', 10)]


def export_maps(diagnostics_root, out):
    root, out = Path(diagnostics_root), Path(out)
    if out.exists():
        raise FileExistsError(out)
    report_path = root / 'DIAGNOSTYKA.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('completed diagnostics required')
    frames = sorted((r for r in read_jsonl(root / 'probe_metadata.jsonl') if r.get('split') == 'val'),
                    key=lambda r: str(r['frame_id']))[:3]
    if len(frames) != 3 or len({r['frame_id'] for r in frames}) != 3:
        raise ValueError('three unique validation frames required')
    selected = [str(r['frame_id']) for r in frames]
    maps, grid = [], None
    for arm, epoch in COMPARISONS:
        matches = [r for r in report['rows'] if r['arm'] == arm and r['completed_epochs'] == epoch]
        if len(matches) != 1:
            raise ValueError(f'one diagnostic required for {arm}, epoch {epoch}')
        result = matches[0]
        cache = root / 'patches' / arm / f'epoch_{epoch:03d}'
        meta = json.loads((cache / 'meta.json').read_text(encoding='utf-8'))
        if meta.get('spatial_pooling') or meta['checkpoint_sha256'] != result['checkpoint_sha256']:
            raise ValueError('patch provenance mismatch')
        current_grid = list(map(int, meta['grid_shape']))
        if grid is not None and current_grid != grid:
            raise ValueError('grid changed between checkpoints')
        grid = current_grid
        rows = read_jsonl(cache / 'rows.jsonl')
        ids = [str(r['frame_id']) for r in rows]
        if len(set(ids)) != len(ids) or not set(selected) <= set(ids):
            raise ValueError('invalid frame alignment')
        position = {fid: i for i, fid in enumerate(ids)}
        readout_path = cache / 'readout_l0.npz'
        if sha256(readout_path) != result['readout']['sha256']:
            raise ValueError('readout changed after diagnostics')
        with np.load(readout_path, allow_pickle=False) as data:
            model = {key: data[key] for key in ('mean', 'scale', 'weights')}
            if list(map(str, data['frame_ids'])) != ids:
                raise ValueError('readout/cache frame ordering differs')
        tokens = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
        if tokens.shape[:2] != (len(rows), int(np.prod(grid))):
            raise ValueError('token shape does not match grid and rows')
        logits = readout_logits(model, tokens[[position[fid] for fid in selected]])
        probabilities = 1. / (1. + np.exp(-np.clip(logits, -60., 60.)))
        maps.append(dict(arm=arm, completed_epochs=epoch, checkpoint_sha256=result['checkpoint_sha256'],
                         readout_sha256=result['readout']['sha256'],
                         probabilities=np.round(probabilities.reshape(3, *grid), 4).tolist()))
    payload = dict(schema_version=1, selection='first_three_sorted_val_frame_ids', frames=frames,
                   grid_shape=grid, maps=maps, warning=WARNING, color_scale=[0., 1.],
                   diagnostics_sha256=sha256(report_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return payload


def _original_image(dataset_root, row):
    from PIL import Image
    archive = Path(dataset_root) / '3_Object_Tracking/annotations' / f"{row['video_id']}.zip"
    with zipfile.ZipFile(archive) as source:
        names = [name for name in source.namelist() if PurePosixPath(name).name == 'annotation.csv']
        if len(names) != 1:
            raise ValueError('exactly one annotation.csv required')
        for entry in csv.DictReader(io.StringIO(source.read(names[0]).decode('utf-8-sig'))):
            if int(entry['frame_number']) == int(row['frame_index']):
                path = PurePosixPath(names[0]).parent / PurePosixPath(entry['filename'])
                with Image.open(io.BytesIO(source.read(str(path)))) as image:
                    return image.convert('RGB')
    raise ValueError(f"original image absent: {row['frame_id']}")


def render_maps(render_json, dataset_root, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    payload = json.loads(Path(render_json).read_text(encoding='utf-8'))
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(3, 5, figsize=(20, 10), layout='constrained')
    titles = ['Obraz i ramka GT', 'Losowy: epoka 2', 'Losowy: epoka 10', 'K+L: epoka 2', 'K+L: epoka 10']
    for index, row in enumerate(payload['frames']):
        image = _original_image(dataset_root, row)
        width, height = image.size
        for column, axis in enumerate(axes[index]):
            axis.imshow(image)
            if column:
                heat = np.asarray(payload['maps'][column-1]['probabilities'][index])
                colored = axis.imshow(heat, extent=(0, width, height, 0), vmin=0, vmax=1,
                                      cmap='magma', alpha=.55, interpolation='nearest')
            for annotation in row['annotations']:
                x, y, w, h = annotation['bbox']
                sx, sy = width / row['native_width'], height / row['native_height']
                axis.add_patch(Rectangle((x*sx, y*sy), w*sx, h*sy, fill=False, edgecolor='#00ffff', linewidth=1.2))
            axis.set_title(titles[column] + (f"\n{row['video_id']}, klatka {row['frame_index']}" if column == 0 else ''), fontsize=10)
            axis.set_axis_off()
        image.close()
    figure.colorbar(colored, ax=axes[:, 1:], fraction=.016, label='Sigmoid odczytu L0 (nieskalibrowany)')
    figure.suptitle('Pojedyncze patche ViT: słaby odczyt obszaru ramki, nie uwaga ani segmentacja', fontsize=13)
    target = output / 'MAPY_PATCHY.png'
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostics-root', 'out', 'render-json', 'dataset-root', 'outdir'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.render_json and args.dataset_root and args.outdir:
        print(render_maps(args.render_json, args.dataset_root, args.outdir))
    elif args.diagnostics_root and args.out and not args.render_json:
        export_maps(args.diagnostics_root, args.out)
        print(args.out)
    else:
        parser.error('provide export pair or render-json, dataset-root and new outdir')


if __name__ == '__main__':
    main()
