#!/usr/bin/env python3
"""Trening RF-DETR z sonda tozsamosci po KAZDYM kroku uczenia (2026-09-23).

Opakowuje `pipeline/train/train_rfdetr.py`: podmienia `rfdetr.training.build_trainer` tak, zeby do
trenera Lightning doszedl jeden dodatkowy callback. Przepis uczenia (optymalizator, harmonogram,
augmentacje, wczesne zatrzymanie) zostaje bez zmian.

Sonda: stala paczka klatek z filmow walidacyjnych, kazda w kilku wersjach (jak w `tozsamosc_obrot.py`):
  orig, rot_-30, rot_+30, dl_1.4 (wydluzenie), tlo_a (narzedzie na tle innej operacji),
  nast (ta sama scena 5 klatek pozniej, ruch prawdziwy).
Po kazdym kroku optymalizatora siec w trybie ewaluacji (bez gradientu, wagi surowe, nie EMA)
liczy dla sondy: wyjscie dekodera najlepszego zapytania, srednia tokenow backbone w przewidzianej
ramce, ramke i pewnosc. Zapisujemy tez identyfikatory klatek uczacych z tego kroku, zeby potem
przypisac zmiane latentu konkretnym klatkom.

    python -P sonda_treningu.py --sonda-data <images/ + annotations_val.json> --sonda-out <dir> \
        -- --run-id ... --variant-file ... --protocol ... --data-root ... --out ... [reszta train_rfdetr]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU))
sys.path.insert(0, str(TU.parents[1] / 'pipeline' / 'train'))

from tozsamosc_obrot import przeksztalc  # noqa: E402

WARIANTY = ['orig', 'rot_-30', 'rot_+30', 'dl_1.4', 'tlo_a', 'nast']


def zbuduj_sonde(data: Path, ile: int, seed: int):
    from PIL import Image, ImageFilter
    coco = json.loads((data / 'annotations_val.json').read_text(encoding='utf-8'))
    box_of = {}
    for a in coco['annotations']:
        box_of.setdefault(a['image_id'], a['bbox'])
    po_id = {im['frame_id']: im for im in coco['images']}
    po_filmie: dict[str, list] = {}
    for im in coco['images']:
        b = box_of.get(im['id'])
        if b and b[2] >= 16 and b[3] >= 8:
            po_filmie.setdefault(im['video_id'], []).append(im)
    rng = np.random.default_rng(seed)
    filmy = sorted(po_filmie)
    klatki = []
    na_film = max(1, int(np.ceil(ile / len(filmy))))
    for f in filmy:
        lista = sorted(po_filmie[f], key=lambda r: r['frame_id'])
        for i in np.linspace(0, len(lista) - 1, na_film + 2)[1:-1].round().astype(int):
            im = lista[i]
            fid = im['frame_id'].rsplit('/', 1)
            nast = po_id.get(f"{fid[0]}/{int(fid[1]) + 5:08d}")
            if nast is not None and nast['id'] in box_of:
                klatki.append((im, nast))
    klatki = [klatki[i] for i in sorted(rng.choice(len(klatki), min(ile, len(klatki)), replace=False))]
    obrazy, ramki, ids = [], [], []
    for im, nast in klatki:
        src = Image.open(data / im['file_name']).convert('RGB')
        inne = [f for f in filmy if f != im['video_id']]
        dr = po_filmie[inne[int(rng.integers(len(inne)))]]
        dr = dr[int(rng.integers(len(dr)))]
        d_ = Image.open(data / dr['file_name']).convert('RGB').resize(src.size)
        bx, by, bw, bh = box_of[dr['id']]
        ob = (int(max(0, bx - .3 * bw)), int(max(0, by - .3 * bh)), int(min(d_.width, bx + 1.3 * bw)),
              int(min(d_.height, by + 1.3 * bh)))
        d_.paste(d_.crop(ob).filter(ImageFilter.GaussianBlur(25)), ob[:2])
        for w in WARIANTY:
            if w == 'nast':
                obrazy.append(Image.open(data / nast['file_name']).convert('RGB')); ramki.append(box_of[nast['id']])
            else:
                o, b = przeksztalc(src, box_of[im['id']], w, [d_, d_], rng)
                obrazy.append(o); ramki.append(b)
        ids.append(im['frame_id'])
    return obrazy, np.array(ramki, np.float32), ids


def zrob_callback(obrazy, ramki_gt, ids, out: Path, co_ile: int):
    import torch
    import torch.nn.functional as F
    from pytorch_lightning.callbacks import Callback

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    surowe = torch.stack([torch.from_numpy(np.asarray(o, dtype=np.uint8)).permute(2, 0, 1) for o in obrazy])
    rozmiary = np.array([[o.width, o.height] for o in obrazy], np.float32)

    class Sonda(Callback):
        def __init__(self):
            self.cap = {}
            self.zap, self.tok, self.ramka, self.pew, self.kroki, self.epoki, self.batch_ids, self.straty = \
                [], [], [], [], [], [], [], []
            self.t0 = time.perf_counter()
            self.wejscie = None

        def _net(self, pl_module):
            return getattr(pl_module, 'model', pl_module)

        def on_fit_start(self, trainer, pl_module):
            net = self._net(pl_module)
            backbone = next(m for _, m in net.named_modules() if type(m).__name__ == 'WindowedDinov2WithRegistersBackbone')
            self.aktywna = False                              # hooki zapisuja tylko w czasie pomiaru sondy
            def hook_fm(module, inputs, output):
                if self.aktywna:
                    maps = output.feature_maps if hasattr(output, 'feature_maps') else output[0]
                    self.cap['fm'] = maps[-1]
            def hook_hs(module, inputs):
                if self.aktywna:
                    self.cap['hs'] = inputs[0]
            backbone.register_forward_hook(hook_fm)
            net.class_embed.register_forward_pre_hook(hook_hs)
            res = None
            for obj in (pl_module, net):
                mc = getattr(obj, 'model_config', None)
                if mc is not None and getattr(mc, 'resolution', None):
                    res = int(mc.resolution)
            self.res = res or 384
            x = surowe.float() / 255.
            x = F.interpolate(x, size=(self.res, self.res), mode='bilinear', align_corners=False)
            self.wejscie = ((x - mean) / std)
            print(f'[sonda] klatek sondy {len(ids)}, obrazow {len(obrazy)}, rozdzielczosc {self.res}', flush=True)
            self._pomiar(trainer, pl_module, None, None)          # stan przed pierwszym krokiem

        def _pomiar(self, trainer, pl_module, batch, outputs):
            net = self._net(pl_module)
            byl_trening = net.training
            net.eval()
            dev = next(net.parameters()).device
            zap, tok, rp, pw = [], [], [], []
            self.aktywna = True
            with torch.no_grad():
                for s in range(0, len(self.wejscie), 48):
                    xb = self.wejscie[s:s + 48].to(dev)
                    out = net(xb)
                    logits, coords = out['pred_logits'], out['pred_boxes']
                    hs = self.cap['hs']
                    if hs.dim() == 4:
                        hs = hs[-1]
                    nq = min(hs.shape[1], logits.shape[1])
                    score, best = logits[:, :nq].float().sigmoid().max(-1).values.max(-1)
                    ar = torch.arange(len(xb), device=dev)
                    zap.append(hs[ar, best].float().cpu().numpy())
                    c = coords[ar, best].float().clamp(0, 1).cpu().numpy()
                    rp.append(c); pw.append(score.float().cpu().numpy())
                    fm = self.cap['fm'].float()
                    b, k, h, w = fm.shape
                    z = fm.permute(0, 2, 3, 1).reshape(b, h * w, k).cpu().numpy()
                    ys, xs = (np.arange(h) + .5) / h, (np.arange(w) + .5) / w
                    ins = (((xs[None, None, :] - c[:, 0, None, None]) ** 2 <= (c[:, 2, None, None] / 2) ** 2) &
                           ((ys[None, :, None] - c[:, 1, None, None]) ** 2 <= (c[:, 3, None, None] / 2) ** 2)).reshape(b, -1)
                    tok.append(np.stack([z[j][ins[j]].mean(0) if ins[j].any() else z[j].mean(0) for j in range(b)]))
            self.aktywna = False
            if byl_trening:
                net.train()
            c = np.concatenate(rp)
            self.zap.append(np.concatenate(zap).astype(np.float16))
            self.tok.append(np.concatenate(tok).astype(np.float16))
            self.ramka.append(np.stack([(c[:, 0] - c[:, 2] / 2) * rozmiary[:, 0], (c[:, 1] - c[:, 3] / 2) * rozmiary[:, 1],
                                        c[:, 2] * rozmiary[:, 0], c[:, 3] * rozmiary[:, 1]], 1).astype(np.float32))
            self.pew.append(np.concatenate(pw).astype(np.float16))
            self.kroki.append(int(trainer.global_step)); self.epoki.append(int(trainer.current_epoch))
            ids_b = []
            if batch is not None:
                try:
                    ids_b = [int(t['image_id']) for t in batch[1]]
                except Exception:
                    ids_b = []
            self.batch_ids.append(ids_b)
            strata = None
            if isinstance(outputs, dict) and 'loss' in outputs:
                strata = float(outputs['loss'])
            elif hasattr(outputs, 'item'):
                strata = float(outputs.item())
            self.straty.append(strata)
            if len(self.kroki) % 100 == 0:
                self.zapisz()
                print(f'[sonda] krok {trainer.global_step}, pomiarow {len(self.kroki)}, '
                      f'{time.perf_counter() - self.t0:.0f}s', flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step % co_ile == 0:
                self._pomiar(trainer, pl_module, batch, outputs)

        def zapisz(self):
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / 'SONDA.npz', zapytanie=np.stack(self.zap), tokeny_ramki=np.stack(self.tok),
                                ramka=np.stack(self.ramka), pewnosc=np.stack(self.pew), kroki=np.array(self.kroki),
                                epoki=np.array(self.epoki), ramka_gt=ramki_gt)
            (out / 'SONDA_META.json').write_text(json.dumps(dict(
                warianty=WARIANTY, frame_ids=ids, batch_image_ids=self.batch_ids, straty=self.straty,
                rozdzielczosc=self.res), ensure_ascii=False), encoding='utf-8')

        def on_fit_end(self, trainer, pl_module):
            self.zapisz()
            print(f'[sonda] koniec: {len(self.kroki)} pomiarow w {out}', flush=True)

    return Sonda()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv:
        sys.exit('uzycie: sonda_treningu.py [opcje sondy] -- [argumenty train_rfdetr]')
    k = argv.index('--')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sonda-data', type=Path, required=True)
    ap.add_argument('--sonda-out', type=Path, required=True)
    ap.add_argument('--sonda-klatki', type=int, default=32)
    ap.add_argument('--co-ile-krokow', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv[:k])

    obrazy, ramki, ids = zbuduj_sonde(args.sonda_data, args.sonda_klatki, args.seed)
    callback = zrob_callback(obrazy, ramki, ids, args.sonda_out, args.co_ile_krokow)

    import rfdetr.training as T
    oryginal = T.build_trainer
    def z_sonda(*a, **kw):
        trainer = oryginal(*a, **kw)
        if kw.get('include_training_callbacks', True):
            trainer.callbacks.append(callback)
            print('[sonda] callback dopiety do trenera', flush=True)
        return trainer
    T.build_trainer = z_sonda

    import train_rfdetr
    return train_rfdetr.main(argv[k + 1:])


if __name__ == '__main__':
    raise SystemExit(main())
