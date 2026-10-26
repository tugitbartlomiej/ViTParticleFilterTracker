"""Trening MałegoViT od zera w jednym z 6 wariantów ablacji, tym samym przepisem co kampania kompensacji (train_replicate.py:
35 000 klatek TRAIN, 65 640 kroków, paczka 32, AdamW 5e-4, wd 0,05, rozgrzewka 300, bf16, odbicie 0,5 i jasność/kontrast ±20 %,
ustawienia deterministyczne) i z tą samą siatką 12 checkpointów: 0, 300, 1000, 3000, 6000, 10 000, 16 000, 24 000, 33 000, 44 000,
55 000, 65 640. Pełny stan do wznowienia co epokę (1094 kroki) w restart.pt.

Warianty (--wariant):
  base            bez zmian (kontrola na tym samym sprzęcie i kodzie co ablacje);
  parallel        blok x + attn(LN₁x) + mlp(LN₂x) zamiast kolejno (uwaga i MLP widzą to samo wejście);
  frozen_mlp      MLP i LN₂ każdego bloku zamrożone na inicjalizacji; uczy się łodyga, uwaga, LN₁, norma i głowa;
  frozen_attn     uwaga i LN₁ każdego bloku zamrożone na inicjalizacji; uczy się reszta;
  frozen_both     uwaga, LN₁, MLP i LN₂ każdego bloku zamrożone na inicjalizacji (kontrola: uczy się tylko łodyga, norma końcowa
                  i głowa; bloki są losowym, stałym mieszaniem przez resztkę) — dodane w eksperymencie kompensacja_lodyga_i_cifar;
  no_wd           weight decay 0 dla wszystkich parametrów;
  shuffled_labels ramki GT przetasowane między klatkami (stała permutacja z ziarna seed + 1): zero kontrolne bez sygnału.
Wagi zapisywane w formacie {'net', 'krok', 'seed', 'wariant'}; state_dict ma te same klucze co base (warianty nie zmieniają modułów),
więc pomiar_mosty_eden.py wczytuje je bez zmian, a wariant 'parallel' przekazuje mu, skąd MLP bierze wejście.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import time
import types
import zipfile
from pathlib import Path

import numpy as np

SIATKA = [0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640]
WARIANTY = ('base', 'parallel', 'frozen_mlp', 'frozen_attn', 'no_wd', 'shuffled_labels', 'frozen_both')
EPOKA = 1094


def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _forward_rownolegly(self, x):
    """timm Block bez sekwencji: oba podbloki liczone z wejścia bloku (ls i drop_path są Identity w tym modelu)."""
    return x + self.drop_path1(self.ls1(self.attn(self.norm1(x)))) + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


def zastosuj_wariant(net, wariant):
    """Zwraca listę parametrów uczonych (reszta zamrożona) i opis."""
    if wariant == 'parallel':
        for blk in net.bloki:
            blk.forward = types.MethodType(_forward_rownolegly, blk)
    if wariant in ('frozen_mlp', 'frozen_attn', 'frozen_both'):
        for blk in net.bloki:
            mod = {'frozen_mlp': [blk.mlp, blk.norm2], 'frozen_attn': [blk.attn, blk.norm1],
                   'frozen_both': [blk.attn, blk.norm1, blk.mlp, blk.norm2]}[wariant]
            for m in mod:
                for p in m.parameters():
                    p.requires_grad_(False)
    uczone = [p for p in net.parameters() if p.requires_grad]
    return uczone, dict(wariant=wariant, parametry_uczone=int(sum(p.numel() for p in uczone)),
                        parametry_wszystkie=int(sum(p.numel() for p in net.parameters())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--archives', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--wariant', required=True, choices=WARIANTY)
    p.add_argument('--steps', type=int, default=65640); p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='64 klatek, 4 kroki: sprawdzenie kodu, nie replikacja')
    a = p.parse_args()
    if a.smoke:
        a.steps = 4
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import cv2
    import torch
    import model as M
    torch.set_num_threads(4); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('Trening wymaga CUDA')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    identity = dict(seed=a.seed, wariant=a.wariant, steps=a.steps, manifest_sha256=file_sha(a.manifest), model_sha256=file_sha(Path(M.__file__)),
                    batch=32, lr=.0005, weight_decay=0.0 if a.wariant == 'no_wd' else .05, warmup=300, amp='bfloat16', clip=1.,
                    technical_smoke=a.smoke, checkpoint_grid=[0, 2, 4] if a.smoke else SIATKA,
                    source='train_replicate.py kampanii kompensacja_funkcjonalna_dgx2_20261017 + warianty ablacji')
    meta = out / 'TRAINING.json'
    if meta.exists():
        if not a.resume or json.loads(meta.read_text()) != identity:
            raise ValueError('Istnieje inny bieg w tym katalogu; wznowienie wymaga zgodnej tożsamości')
    else:
        if a.resume:
            raise ValueError('Nie ma czego wznawiać')
        meta.write_text(json.dumps(identity, indent=1), encoding='utf-8')
    rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding='utf-8').splitlines() if l.strip()]
    split = json.loads((Path(a.manifest).parent / 'split_manifest.json').read_text(encoding='utf-8'))['groups']
    if len(rows) != 35000 or any(r['split'].lower() != 'train' or split.get(r['film']) != 'train' for r in rows):
        raise ValueError('Oczekiwano 35 000 wierszy TRAIN zgodnych z podziałem filmów')
    if a.smoke:
        rows = rows[:64]
    X = np.empty((len(rows), 320, 480, 3), dtype=np.uint8); boxes = []; archive = None; z = None
    for i, r in enumerate(rows):
        name = Path(r['archive'].replace('\\', '/')).name
        if archive != name:
            if z:
                z.close()
            z = zipfile.ZipFile(Path(a.archives) / name); archive = name
        raw = z.read(r['member'])
        if hashlib.sha256(raw).hexdigest() != r['image_sha256']:
            raise ValueError('Suma klatki treningowej niezgodna z manifestem')
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        if (w, h) != (r['width'], r['height']):
            raise ValueError('Wymiary klatki niezgodne z manifestem')
        X[i] = cv2.cvtColor(cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        b = np.array([g['bbox_xywh'] for g in r['gt']], dtype=np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]; b *= np.array([480 / w, 320 / h, 480 / w, 320 / h]); boxes.append(b)
        if i % 1000 == 0:
            print(f'wczytano {i}/{len(rows)}', flush=True)
    if z:
        z.close()
    if a.wariant == 'shuffled_labels':                                   # etykiety z innej klatki: stała permutacja (ziarno seed + 1)
        perm = np.random.default_rng(a.seed + 1).permutation(len(rows))
        boxes = [boxes[j] for j in perm]
        (out / 'PERMUTACJA_ETYKIET.json').write_text(json.dumps(dict(ziarno=a.seed + 1, pierwsze_16=perm[:16].tolist(),
                                                                   sha256=hashlib.sha256(perm.tobytes()).hexdigest())), encoding='utf-8')
    targets = [M.cele(boxes, 20, 30)]
    flipped = [np.stack([480 - b[:, 2], b[:, 1], 480 - b[:, 0], b[:, 3]], 1) if len(b) else b for b in boxes]
    targets.append(M.cele(flipped, 20, 30))
    net = M.MalyViT().cuda()
    uczone, opis_w = zastosuj_wariant(net, a.wariant)
    (out / 'WARIANT.json').write_text(json.dumps(opis_w, indent=1), encoding='utf-8')
    wd = identity['weight_decay']
    dec = [v for v in uczone if v.ndim >= 2]; nd = [v for v in uczone if v.ndim < 2]
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=wd), dict(params=nd, weight_decay=0)], lr=.0005, betas=(.9, .999))
    rng = np.random.default_rng(a.seed); queue = []; step = 0; stop = [False]

    def checkpoint(full=False):
        directory = out / 'weights'; directory.mkdir(exist_ok=True)
        target = (out / 'restart.pt') if full else directory / f'k{step:05d}.pt'
        tmp = target.with_suffix('.tmp')
        st = dict(net=net.state_dict(), krok=step, seed=a.seed, wariant=a.wariant, training_manifest_sha256=identity['manifest_sha256'])
        if full:
            st.update(opt=opt.state_dict(), queue=queue, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state(), identity=identity)
        torch.save(st, tmp); os.replace(tmp, target)

    if a.resume:
        st = torch.load(out / 'restart.pt', map_location='cpu', weights_only=False)      # własny plik wznowienia (obiekty numpy)
        if st['identity'] != identity:
            raise ValueError('Tożsamość wznowienia niezgodna')
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); step = st['krok']; queue = st['queue']
        rng.bit_generator.state = st['rng']; torch.set_rng_state(st['torch_rng']); torch.cuda.set_rng_state_all(st['cuda_rng'])
        random.setstate(st['python_rng']); np.random.set_state(st['numpy_rng'])
        print(f'wznowienie od kroku {step} (wagi, AdamW, kolejka permutacji, RNG numpy/torch/cuda/python)', flush=True)
    else:
        checkpoint()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stop.__setitem__(0, True))
    net.train(); start = time.perf_counter()
    with (out / 'loss.jsonl').open('a', encoding='utf-8') as log:
        while step < a.steps and not stop[0]:
            if len(queue) < 32:
                queue += rng.permutation(len(rows)).tolist()
            ix = np.array(queue[:32]); queue = queue[32:]; flip = rng.random(32) < .5
            x = torch.from_numpy(X[ix]).cuda().permute(0, 3, 1, 2).float() / 255
            fm = torch.from_numpy(flip).cuda(); x = torch.where(fm[:, None, None, None], x.flip(3), x)
            ac = torch.from_numpy(rng.uniform(.8, 1.2, (32, 2)).astype(np.float32)).cuda(); mean = x.mean((1, 2, 3), keepdim=True)
            x = ((x - mean) * ac[:, 0, None, None, None] + mean * ac[:, 1, None, None, None]).clamp(0, 1)
            ts = []
            for k in range(3):
                t0 = targets[0][k][ix]; t1 = targets[1][k][ix]
                mask = torch.from_numpy(flip).reshape((32,) + (1,) * (t0.ndim - 1)); ts.append(torch.where(mask, t1, t0).cuda())
            for g in opt.param_groups:
                g['lr'] = .0005 * min(1, (step + 1) / 300)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                raw, _ = net(x)
            loss, parts = M.strata(raw, *ts)
            if not torch.isfinite(loss):
                raise FloatingPointError('Strata nieskończona')
            loss.backward(); torch.nn.utils.clip_grad_norm_(uczone, 1.); opt.step(); step += 1
            if step % 50 == 0:
                log.write(json.dumps(dict(step=step, loss=float(loss), **parts)) + '\n'); log.flush()
            if step in identity['checkpoint_grid'] or step == a.steps:
                checkpoint()
            if step % EPOKA == 0:
                checkpoint(full=True); print(f'{a.wariant} s{a.seed} krok={step} {time.perf_counter() - start:.0f} s', flush=True)
        checkpoint(full=True)
        if stop[0]:
            raise SystemExit(143)
    (out / 'COMPLETE.json').write_text(json.dumps(dict(step=step, seed=a.seed, wariant=a.wariant, status='complete',
                                                      seconds=time.perf_counter() - start, torch=torch.__version__, cuda=torch.version.cuda,
                                                      gpu=torch.cuda.get_device_name(0)), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
