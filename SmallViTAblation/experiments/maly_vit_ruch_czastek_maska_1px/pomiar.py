"""Inwentaryzacja lokalnych wag i fp32 eval wszystkich tokenow, bez treningu.

python -B pomiar.py [--inventory-only] [--batch-size 2]
Wybrany jeden wspolny przebieg seed47: proba_s47/wspolny, kroki 60 i 120.
Nie laczymy proby z osobna rozgrzewka do 1000 ani z ramionami po rozgalezieniu.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import cv2
import numpy as np
import torch
import timm
from dane import HERE, ROOT, OUT, MANIFEST, prepare, save_json, sha256_file

SELECTED = ROOT / 'experiments/maly_vit_petla_sd_rownanie/wyniki/proba_s47/wspolny'
MODEL = ROOT / 'experiments/maly_vit_od_zera/model.py'


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def inventory():
    records = []
    # Nie podazamy za dowiazaniami; tylko lokalny katalog SmallViTAblation.
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv')
                   and not os.path.islink(Path(base) / d)]
        for name in files:
            p = Path(base) / name
            if p.suffix.lower() not in ('.pt', '.pth', '.ckpt', '.safetensors'):
                continue
            rel = p.relative_to(ROOT).as_posix()
            rec = dict(path=str(p.resolve()), relative_path=rel, bytes=p.stat().st_size,
                       selected=p.parent == SELECTED and p.name in ('k01.pt', 'k02.pt'))
            if 'maly_vit' in rel:
                rec['sha256'] = sha256_file(p)
                if 's47' in rel or 'seed47' in rel:
                    try:
                        ck = torch.load(p, map_location='cpu', weights_only=True)
                        st = ck.get('net', ck)
                        rec['step'] = ck.get('krok')
                        rec['architecture_matches'] = isinstance(st, dict) and 'lodyga.0.0.weight' in st
                        if rec['architecture_matches']:
                            rec['state_sha256'] = state_hash(st)
                        rec['reason'] = ('selected_same_common_run' if rec['selected'] else
                                         'not_selected_other_run_or_branch_or_single_checkpoint')
                    except Exception as exc:
                        rec['inspection_error'] = repr(exc)
            records.append(rec)
    records.sort(key=lambda r: r['relative_path'])
    selected = sorted([r for r in records if r['selected']], key=lambda r: r['step'])
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), scan_root=str(ROOT),
                  symlinks_followed=False, checkpoint_records=records,
                  selected_run=str(SELECTED.parent), selected_steps=[r['step'] for r in selected],
                  limitation='Brak gestych wag co 10 aktualizacji. Pelna rozgrzewka seed47 ma tylko krok 1000. '
                             'Wybrano dwa istniejace stany wspolnej krotkiej proby seed47 (60,120). '
                             'Oddzielnych przebiegow, ramion, srednich LATENT.npz i odtworzonej inicjalizacji nie laczono.',
                  max_updates=1000, training_performed=False)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / 'inventory.json', report)
    return selected, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory-only', action='store_true')
    ap.add_argument('--batch-size', type=int, default=2)
    args = ap.parse_args()
    assert 1 <= args.batch_size <= 12
    checkpoints, inv = inventory()
    print('Selected checkpoints:', [(r['step'], r['path']) for r in checkpoints], flush=True)
    if args.inventory_only:
        return
    if len(checkpoints) < 2 or len({r.get('state_sha256') for r in checkpoints}) < 2:
        raise RuntimeError('Brak dwoch roznych checkpointow jednego przebiegu; zapisano inventory.')
    assert [r['step'] for r in checkpoints] == [60, 120]
    assert torch.cuda.is_available(), 'Ten pomiar wymaga lokalnego CUDA.'
    # Caly forward fp32; brak autocast i TF32.
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    X, images = prepare()
    spec = importlib.util.spec_from_file_location('maly_vit_model_readonly', MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.MalyViT().float().cuda().eval()
    net.zapisuj_latent(True)
    shape = (len(checkpoints), len(images), 600, 768)
    dest = OUT / 'latent.npy'
    if dest.exists():
        raise FileExistsError(f'Nie nadpisuje istniejacego pomiaru: {dest}')
    latent = np.lib.format.open_memmap(dest, mode='w+', dtype=np.float32, shape=shape)
    metadata = dict(status='running', steps=[r['step'] for r in checkpoints], images=images,
                    checkpoints=checkpoints, shape=list(shape), dtype='float32', storage_dtype='float32',
                    inference_dtype='float32', autocast=False, tf32=False, eval=True, no_grad=True,
                    batch_size=args.batch_size, seed=47, run=str(SELECTED.parent),
                    latent_layer='bloki[-1].mlp.act (GELU)', token_order='row-major: token=y*30+x',
                    grid_hw=[20, 30], input_size_wh=[480, 320], patch_size=16,
                    image_transform='Full RGB frame, cv2.INTER_AREA original_size_wh -> 480x320; no crop/augmentation. '
                                    'MalyViT applies its own ImageNet normalization.',
                    mask_transform='Original cystotome_seg rasterized with rounded vertices and cv2.fillPoly. '
                                   'Float32 INTER_AREA to 480x320, binary mask = area>0. '
                                   'coverage = mean of this binary mask over each 16x16 input cell; '
                                   'area_coverage separately preserves pre-binarization fractional area.',
                    mask_definitions=dict(tool='coverage>0 (>=1 input pixel)', tool25='coverage>=0.25',
                                          background='zero pixels per 16x16 cell after 17x17 dilation of input binary mask; constant-zero border'),
                    dtype_change='Requested f16 replaced by f32 because only 2 checkpoints; no f16 storage quantization.',
                    limitations=[inv['limitation'], 'T=2 gives one displacement interval; insufficient for dynamics or forecasting.',
                                 'tool25 controls threshold on the same conservative mask; not an exact reproduction of old mask resampling.'],
                    source_manifest=dict(path=str(MANIFEST), sha256=sha256_file(MANIFEST)),
                    code=[dict(path=str(p), sha256=sha256_file(p)) for p in (MODEL, HERE/'dane.py', HERE/'pomiar.py')],
                    device=torch.cuda.get_device_name(0), versions=dict(python=platform.python_version(),
                    torch=torch.__version__, timm=timm.__version__, numpy=np.__version__, opencv=cv2.__version__),
                    parameters=sum(p.numel() for p in net.parameters()), training_performed=False,
                    files=dict(latent='latent.npy', masks='maski.npz', input_images='obrazy.npy', inventory='inventory.json'))
    save_json(OUT / 'metadata.json', metadata)
    checks = []
    started = time.perf_counter()
    with torch.no_grad():
        for t, record in enumerate(checkpoints):
            assert sha256_file(record['path']) == record['sha256'], 'Checkpoint changed during measurement'
            ck = torch.load(record['path'], map_location='cpu', weights_only=True)
            net.load_state_dict(ck['net'], strict=True)
            net.eval()
            for s in range(0, len(X), args.batch_size):
                batch = torch.from_numpy(X[s:s+args.batch_size].copy()).permute(0, 3, 1, 2).cuda().float() / 255
                _, z = net(batch)
                assert z.dtype == torch.float32 and tuple(z.shape[1:]) == (600, 768)
                arr = z.cpu().numpy()
                assert np.isfinite(arr).all()
                latent[t, s:s+len(arr)] = arr
                if s == 0:
                    _, z_repeat = net(batch)
                    repeat_error = float(np.max(np.abs(z_repeat.cpu().numpy() - arr)))
                    checks.append(dict(step=record['step'], repeated_forward_max_abs_error=repeat_error))
            latent.flush()
            print(f"Measured step {record['step']}: {shape[1]}x600x768 fp32", flush=True)
    net.zapisuj_latent(False)
    del latent
    stored = np.load(dest, mmap_mode='r')
    assert stored.shape == shape and stored.dtype == np.float32 and np.isfinite(stored).all()
    assert not np.array_equal(stored[0], stored[1]), 'Identical latents at distinct checkpoints'
    metadata.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-started, validation=checks,
                    latent_sha256=sha256_file(dest), latent_bytes=dest.stat().st_size,
                    artifacts=[dict(file=p.name, bytes=p.stat().st_size, sha256=sha256_file(p))
                               for p in (OUT/'maski.npz', OUT/'obrazy.npy', OUT/'images.json', OUT/'inventory.json')])
    save_json(OUT / 'metadata.json', metadata)
    print(f'Complete: {dest}, {shape}, float32, {dest.stat().st_size} bytes', flush=True)


if __name__ == '__main__':
    main()
