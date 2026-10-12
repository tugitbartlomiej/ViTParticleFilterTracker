"""RF-DETR 1.10.1 adapter. Runtime dependencies are imported only on Eden.

Training and selection use identical fixed-square, ImageNet-normalized images.
Selection is eval-mode, group-DETR=1, gradients restricted to final head layers.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import random
import time
import numpy as np


@contextmanager
def diagnostic_transaction(module):
    """Diagnostics cannot change RNG, mixed train/eval modes, or .grad buffers."""
    import torch
    modes = [(m, m.training) for m in module.modules()]
    flags = [(p, p.requires_grad) for p in module.parameters()]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            module.eval()
            yield
    finally:
        for m, mode in modes:
            m.training = mode
        for p, flag in flags:
            p.requires_grad_(flag)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


@contextmanager
def virtual_parameters(parameters, gradient, learning_rate):
    """A virtual SGD step only; never touches optimizer or parameter .grad."""
    import torch
    originals = [p.detach().clone() for p in parameters]
    gradient = np.asarray(gradient)
    if gradient.shape != (sum(p.numel() for p in parameters),) or not np.isfinite(gradient).all():
        raise ValueError('invalid virtual gradient')
    try:
        offset = 0
        with torch.no_grad():
            for p, old in zip(parameters, originals):
                n = p.numel()
                g = torch.as_tensor(gradient[offset:offset+n].copy(), device=p.device, dtype=p.dtype).reshape_as(p)
                p.copy_(old - learning_rate*g)
                offset += n
        yield
    finally:
        with torch.no_grad():
            for p, old in zip(parameters, originals):
                p.copy_(old)


def image_tensor(row, resolution):
    import torch
    from PIL import Image
    with Image.open(row['image_path']) as image:
        image = image.convert('RGB').resize((resolution, resolution), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).copy()/255.
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
    std = tensor.new_tensor([.229, .224, .225])[:, None, None]
    return (tensor-mean)/std, float(pixels.mean())


def target_tensor(row, image_id, resolution):
    import torch
    w, h = row['native_width'], row['native_height']
    boxes = []
    for a in row['annotations']:
        x, y, bw, bh = a['bbox']
        if not 0 <= x < x+bw <= w or not 0 <= y < y+bh <= h:
            raise ValueError(f'invalid bbox: {row["frame_id"]}')
        boxes.append([(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h])
    return dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                labels=torch.zeros(len(boxes), dtype=torch.int64),
                image_id=torch.tensor([image_id]), orig_size=torch.tensor([h, w]),
                size=torch.tensor([resolution, resolution]),
                area=torch.tensor([a['bbox'][2]*a['bbox'][3] for a in row['annotations']]),
                iscrowd=torch.zeros(len(boxes), dtype=torch.int64))


class FrameDataset:
    """An immutable full FIT dataset. The sampler chooses the epoch's rows."""
    def __init__(self, rows, resolution):
        self.rows, self.resolution = rows, resolution

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return image_tensor(row, self.resolution)[0], target_tensor(row, index+1, self.resolution)


class DetectorBackend:
    def __init__(self, module, config):
        self.module, self.net, self.criterion, self.config = module, module.model, module.criterion, config
        self.device = next(self.net.parameters()).device
        if not hasattr(self.net.class_embed, 'weight') or not hasattr(self.net.bbox_embed, 'layers'):
            raise RuntimeError('unsupported RF-DETR head layout')
        self.parameters = list(self.net.class_embed.parameters()) + list(self.net.bbox_embed.layers[-1].parameters())
        wanted = {id(p) for p in self.parameters}
        self.parameter_names = [n for n, p in self.net.named_parameters() if id(p) in wanted]
        if len(wanted) != len(self.parameters) or self.net.class_embed.out_features != self.module.model_config.num_classes+1:
            raise RuntimeError('expected unique final head parameters and a one-class model')

    def _samples(self, tensors):
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
        return nested_tensor_from_tensor_list([t.to(self.device) for t in tensors])

    def loss(self, row):
        import torch
        image, _ = image_tensor(row, self.config['resolution'])
        target = {k: v.to(self.device) for k, v in target_tensor(row, 1, self.config['resolution']).items()}
        outputs = self.net(self._samples([image]))
        parts = self.criterion(outputs, [target])
        loss = torch.stack([v*self.criterion.weight_dict[k] for k, v in parts.items() if k in self.criterion.weight_dict]).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite detector loss: {row["frame_id"]}')
        return loss

    def losses(self, rows):
        import torch
        with torch.no_grad():
            return np.array([float(self.loss(row)) for row in rows], dtype=np.float64)

    def gradient(self, row):
        import torch
        values = torch.autograd.grad(self.loss(row), self.parameters, allow_unused=False)
        flat = torch.cat([v.detach().flatten().float() for v in values]).cpu().numpy()
        if not np.isfinite(flat).all():
            raise FloatingPointError('nonfinite head gradient')
        return flat

    def select(self, candidate_rows, reference_rows, budget, seed, out):
        """Fixed candidate gradients, refreshed D gradients after virtual steps."""
        from .glister import select_gradients
        from .dane import write_json
        wanted = {id(p) for p in self.parameters}
        for p in self.net.parameters():
            p.requires_grad_(id(p) in wanted)
        gradients = np.stack([self.gradient(row) for row in candidate_rows])
        groups = [str(r['group_id']) for r in reference_rows]
        weights = np.array([1/groups.count(g) for g in groups], dtype=np.float64)
        weights /= weights.sum()
        refresh = []

        def reference(selected):
            update = gradients[list(selected)].mean(axis=0) if selected else np.zeros(gradients.shape[1])
            with virtual_parameters(self.parameters, update, self.config['virtual_lr']):
                result = sum(w*self.gradient(row) for w, row in zip(weights, reference_rows))
            refresh.append(dict(selected_before=len(selected), reference_norm=float(np.linalg.norm(result))))
            return result

        chosen = select_gradients(gradients, reference, budget, rounds=self.config['glister_rounds'], seed=seed)
        write_json(Path(out)/'gradient_selection.json', dict(parameter_names=self.parameter_names,
                   gradient_parameters=gradients.shape[1], candidates=len(candidate_rows), budget=budget,
                   approximation='head-only chunked GLISTER Taylor; virtual mean SGD; real optimizer AdamW',
                   eval_mode=True, reference_weighting='equal operation mass', refreshes=refresh))
        return chosen

    def extract(self, rows, out):
        """Current live weights, no new detector; one feature vector per patch."""
        import torch
        from .dynamika import LivePatchRecorder
        from .dane import write_json, sha256
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        inputs = np.empty((len(rows), 6), dtype=np.float32)
        tokens = None
        start = time.monotonic()
        with LivePatchRecorder(self.net, dim=self.config['token_dim'], seed=self.config['projection_seed']) as recorder:
            with torch.no_grad():
                for first in range(0, len(rows), self.config['batch_size']):
                    batch = rows[first:first+self.config['batch_size']]
                    images, brightness = zip(*(image_tensor(r, self.config['resolution']) for r in batch))
                    output = self.net(self._samples(images))
                    values = recorder.pop()
                    if tokens is None:
                        tokens = np.lib.format.open_memmap(out/'tokens.npy', mode='w+', dtype='float16',
                                                           shape=(len(rows), *values.shape[1:]))
                    if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
                        raise FloatingPointError('invalid patch features')
                    tokens[first:first+len(batch)] = values
                    confidence, queries = output['pred_logits'][..., 0].sigmoid().max(dim=1)
                    boxes = output['pred_boxes'][torch.arange(len(batch), device=self.device), queries]
                    inputs[first:first+len(batch)] = np.column_stack((confidence.cpu().numpy(), boxes.cpu().numpy(), brightness))
                    if first % (self.config['batch_size']*100) == 0:
                        print(f'[latent] {first+len(batch)}/{len(rows)}', flush=True)
            grid_shape = recorder.grid_shape
            np.save(out/'projection.npy', recorder.projection)
        tokens.flush()
        np.save(out/'inputs.npy', inputs)
        write_json(out/'meta.json', dict(grid_shape=list(grid_shape), rows=len(rows), particle='one spatial patch',
                   temporal_correspondence='Eulerian fixed grid; not tracked material identity',
                   projected_channels=self.config['token_dim'], projection_sha256=sha256(out/'projection.npy'),
                   elapsed_seconds=time.monotonic()-start))
        return tokens, inputs, grid_shape


def build_training(protocol_dir, run_dir, initial_weights, config, sampler, fit_rows):
    """One Trainer.fit call preserves optimizer and scheduler across selections."""
    import torch
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
    from rfdetr.datasets import build_dataset
    from rfdetr.training.module_data import _namespace_from_configs
    from torch.utils.data import DataLoader
    from pytorch_lightning import seed_everything
    seed_everything(config['seed'], workers=True)
    mc = RFDETRNanoConfig(num_classes=1, pretrain_weights=str(initial_weights),
                          resolution=config['resolution'], amp=True, compile=False)
    tc = TrainConfig(dataset_dir=str(Path(protocol_dir)/'coco'), dataset_file='roboflow',
                     output_dir=str(run_dir), epochs=config['epochs'], batch_size=config['batch_size'],
                     grad_accum_steps=1, lr=config['lr'], lr_encoder=config['lr_encoder'],
                     weight_decay=config['weight_decay'], warmup_epochs=0., seed=config['seed'],
                     use_ema=False, eval_base_model=True, run_test=False, early_stopping=False,
                     aug_config={}, scale_jitter=False, multi_scale=False, augmentation_backend='cpu',
                     pack_targets=False, num_workers=config['num_workers'], amp_dtype='bf16',
                     devices=1, num_nodes=1, accelerator='gpu', eval_interval=1,
                     eval_max_dets=100, best_model_metric='map', compute_val_loss=True,
                     num_sanity_val_steps=0, progress_bar=None, tensorboard=False, wandb=False)

    class AdaptiveDataModule(RFDETRDataModule):
        def setup(self, stage):
            if stage not in ('fit', 'validate'):
                raise ValueError('test is intentionally unavailable in this experiment')
            if self._dataset_train is None:
                self._dataset_train = FrameDataset(fit_rows, config['resolution'])
            if self._dataset_val is None:
                self._dataset_val = build_dataset('val', _namespace_from_configs(mc, tc), mc.resolution)

        def train_dataloader(self):
            return DataLoader(self._dataset_train, batch_size=config['batch_size'], sampler=sampler,
                              drop_last=False, num_workers=config['num_workers'], pin_memory=True,
                              persistent_workers=config['num_workers'] > 0, collate_fn=self._collate_fn)

    module = RFDETRModelModule(mc, tc)
    if module.model.class_embed.out_features != mc.num_classes+1:
        raise RuntimeError('COCO loader changed one-class head; refusing to run')
    datamodule = AdaptiveDataModule(mc, tc)
    trainer = build_trainer(tc, mc, accelerator='gpu', devices=1, inference_mode=False,
                            use_distributed_sampler=False)
    return module, datamodule, trainer
