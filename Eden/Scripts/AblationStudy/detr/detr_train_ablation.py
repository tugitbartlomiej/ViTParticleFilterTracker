"""
DETR Training Script — Ablation Study Variant

Based on detr_train_optimized.py with one key change:
  - Accepts --annotations_val_path for pre-split (video-level) datasets
  - When provided, skips random_split() and loads train/val from separate JSONs
  - Fully backward-compatible: omitting --annotations_val_path uses random_split()
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

torch._dynamo.config.suppress_errors = True


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not running in distributed mode")
        return None, None, None
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class SurgicalToolDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, augment=False):
        if dist.get_rank() == 0:
            print(f"Loading dataset from: {annotations_file}")
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augment = augment
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)]) if augment else None

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        self.categories = coco_data['categories']
        self.image_id_to_image = {img['id']: img for img in coco_data['images']}
        self.image_id_to_anns = {}

        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            ann['bbox'] = [float(x) for x in ann['bbox']]
            self.image_id_to_anns[img_id].append(ann)

        self.image_ids = []
        all_ids = set(self.image_id_to_image.keys())
        for img_id in self.image_id_to_anns:
            if img_id in all_ids:
                self.image_ids.append(img_id)
        ann_ids = set(self.image_ids)
        for img_id in all_ids:
            if img_id not in ann_ids:
                self.image_ids.append(img_id)
                self.image_id_to_anns[img_id] = []
        self.image_ids = sorted(set(self.image_ids))

        if dist.get_rank() == 0:
            print(f"  {len(self.image_ids)} images loaded")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_image[image_id]
        annotations = self.image_id_to_anns.get(image_id, [])

        formatted = [{'bbox': a['bbox'], 'category_id': a['category_id'],
                       'area': a.get('area', a['bbox'][2] * a['bbox'][3]),
                       'iscrowd': a.get('iscrowd', 0)} for a in annotations]

        path = str(self.images_dir / image_info['file_name'])
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return None

        if self.augment and self.transform:
            image = self.transform(image)

        try:
            enc = self.processor(images=image, annotations={'image_id': image_id, 'annotations': formatted},
                                 return_tensors="pt")
        except Exception:
            return None

        labels = enc["labels"][0] if enc["labels"] else {
            'class_labels': torch.tensor([], dtype=torch.int64),
            'boxes': torch.tensor([], dtype=torch.float32)
        }
        return {"pixel_values": enc["pixel_values"].squeeze(0),
                "pixel_mask": enc["pixel_mask"].squeeze(0), "labels": labels}


def collate_fn_padded(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    try:
        bs = len(batch)
        mh = max(x["pixel_values"].shape[-2] for x in batch)
        mw = max(x["pixel_values"].shape[-1] for x in batch)
        ch = batch[0]["pixel_values"].shape[0]
        pv = torch.zeros((bs, ch, mh, mw), dtype=batch[0]["pixel_values"].dtype)
        pm = torch.zeros((bs, mh, mw), dtype=batch[0]["pixel_mask"].dtype)
        labels = []
        for i, item in enumerate(batch):
            h, w = item["pixel_values"].shape[-2], item["pixel_values"].shape[-1]
            pv[i, :, :h, :w] = item["pixel_values"]
            pm[i, :h, :w] = item["pixel_mask"]
            labels.append(item["labels"])
        return {"pixel_values": pv, "pixel_mask": pm, "labels": labels}
    except Exception:
        return None


def find_latest_checkpoint(checkpoint_dir):
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    final = d / "final_checkpoint.pth"
    if final.exists():
        return final
    files = sorted(d.glob("checkpoint_epoch_*.pth"), key=lambda x: int(x.stem.split('_')[-1]))
    return files[-1] if files else None


def load_checkpoint(path, model, optimizer, scaler, scheduler, device):
    path = Path(path)
    if not path.is_file():
        return model, optimizer, scaler, scheduler, 0
    if dist.get_rank() == 0:
        print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location={'cuda:0': f'cuda:{dist.get_rank()}'})
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd)
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt.get('epoch', 0)
    if dist.get_rank() == 0:
        print(f"Resumed from epoch {epoch}")
    return model, optimizer, scaler, scheduler, epoch


def train(args):
    rank, world_size, local_rank = setup_ddp()
    if rank == 0:
        print(f"Ablation training | GPUs: {world_size} | Args: {args}")

    device = torch.device(f'cuda:{local_rank}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        for d in [args.output_dir, args.checkpoint_dir, args.best_model_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "final_model").mkdir(parents=True, exist_ok=True)
        log_dir = Path(args.output_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    dist.barrier()

    with open(args.annotations_path, 'r') as f:
        coco = json.load(f)
    cats = coco.get('categories', [])
    id2label = {c['id']: c['name'] for c in cats}
    label2id = {v: k for k, v in id2label.items()}
    processor = DetrImageProcessor.from_pretrained(args.model_checkpoint)

    # === KEY CHANGE: video-level pre-split support ===
    if args.annotations_val_path:
        if rank == 0:
            print(f"PRE-SPLIT mode: train={args.annotations_path}, val={args.annotations_val_path}")
        train_dataset = SurgicalToolDataset(args.images_dir, args.annotations_path, processor, args.augment)
        val_dataset = SurgicalToolDataset(args.images_dir, args.annotations_val_path, processor, False)
    else:
        if rank == 0:
            print("RANDOM split mode (legacy)")
        full = SurgicalToolDataset(args.images_dir, args.annotations_path, processor, args.augment)
        if args.train_val_split < 1.0:
            ts = int(args.train_val_split * len(full))
            train_dataset, val_dataset = random_split(full, [ts, len(full) - ts])
        else:
            train_dataset, val_dataset = full, None

    if rank == 0:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=True) if val_dataset else None

    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     collate_fn=collate_fn_padded, pin_memory=True, drop_last=True,
                     persistent_workers=args.num_workers > 0,
                     prefetch_factor=2 if args.num_workers > 0 else None)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, **dl_kwargs)
    val_dl = DataLoader(val_dataset, sampler=val_sampler, **dl_kwargs) if val_dataset else None

    config = DetrConfig.from_pretrained(args.model_checkpoint, num_labels=len(id2label),
                                         id2label=id2label, label2id=label2id)
    config.num_queries = args.num_queries
    model = DetrForObjectDetection.from_pretrained(args.model_checkpoint, config=config,
                                                    ignore_mismatched_sizes=True).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay,
                                   fused=torch.cuda.is_available())
    scaler = GradScaler(enabled=args.use_amp)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma,
                                                                patience=5, min_lr=args.lr_min)

    start_epoch = 0
    if args.resume_training:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest:
            model.module, optimizer, scaler, scheduler, start_epoch = load_checkpoint(
                latest, model.module, optimizer, scaler, scheduler, device)
            optimizer.param_groups[0]['lr'] = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.lr_backbone
            remaining = args.epochs - start_epoch
            if scheduler and args.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=args.lr_min)

    best_val_loss = float('inf')
    patience_counter = 0
    accum = args.gradient_accumulation_steps
    epoch = start_epoch

    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_sampler.set_epoch(epoch)
            model.train()
            total_loss, n_batches = 0.0, 0
            pbar = tqdm(train_dl, desc=f"E{epoch+1}") if rank == 0 else train_dl
            optimizer.zero_grad()

            for i, batch in enumerate(pbar):
                skip = torch.tensor([1 if batch is None else 0], device=device, dtype=torch.int32)
                dist.all_reduce(skip, op=dist.ReduceOp.MAX)
                if skip.item():
                    continue

                bd = {'pixel_values': batch['pixel_values'].to(device, non_blocking=True),
                      'pixel_mask': batch['pixel_mask'].to(device, non_blocking=True),
                      'labels': [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                  for k, v in ld.items()} for ld in batch['labels']]}

                with autocast(enabled=args.use_amp):
                    out = model(**bd)
                    loss = out.loss / accum
                scaler.scale(loss).backward()

                if (i + 1) % accum == 0 or (i + 1) == len(train_dl):
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                bl = loss.item() * accum
                total_loss += bl
                n_batches += 1
                if rank == 0:
                    pbar.set_postfix(loss=f"{bl:.4f}")

            dist.barrier()
            lt = torch.tensor(total_loss, device=device)
            ct = torch.tensor(n_batches, device=device)
            dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            dist.all_reduce(ct, op=dist.ReduceOp.SUM)
            avg_train = lt.item() / max(ct.item(), 1)

            if rank == 0:
                print(f"E{epoch+1} train={avg_train:.4f} ({time.time()-t0:.0f}s)")
                writer.add_scalar("Loss/train", avg_train, epoch)

            if val_dl:
                if val_sampler:
                    val_sampler.set_epoch(epoch)
                model.eval()
                vl, vb = 0.0, 0
                with torch.no_grad():
                    for bv in (tqdm(val_dl, desc=f"Val E{epoch+1}") if rank == 0 else val_dl):
                        skip = torch.tensor([1 if bv is None else 0], device=device, dtype=torch.int32)
                        dist.all_reduce(skip, op=dist.ReduceOp.MAX)
                        if skip.item():
                            continue
                        bd = {'pixel_values': bv['pixel_values'].to(device, non_blocking=True),
                              'pixel_mask': bv['pixel_mask'].to(device, non_blocking=True),
                              'labels': [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                          for k, v in ld.items()} for ld in bv['labels']]}
                        with autocast(enabled=args.use_amp):
                            o = model(**bd)
                        vl += o.loss.item()
                        vb += 1

                dist.barrier()
                vlt = torch.tensor(vl, device=device)
                vct = torch.tensor(vb, device=device)
                dist.all_reduce(vlt, op=dist.ReduceOp.SUM)
                dist.all_reduce(vct, op=dist.ReduceOp.SUM)
                avg_val = vlt.item() / max(vct.item(), 1)

                if rank == 0:
                    print(f"E{epoch+1} val={avg_val:.4f}")
                    writer.add_scalar("Loss/val", avg_val, epoch)

                dist.barrier()

                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val)
                    else:
                        scheduler.step()
                    if rank == 0:
                        writer.add_scalar("LR/main", optimizer.param_groups[0]['lr'], epoch)

                if rank == 0:
                    if (epoch + 1) % args.save_interval == 0:
                        ckpt = {'epoch': epoch + 1, 'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scaler_state_dict': scaler.state_dict(), 'loss': avg_val}
                        if scheduler:
                            ckpt['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(ckpt, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth")

                    if avg_val < best_val_loss:
                        best_val_loss = avg_val
                        patience_counter = 0
                        best_path = os.path.join(args.best_model_dir, 'best_model.pth')
                        torch.save(model.module.state_dict(), best_path)
                        print(f"  Best model saved (val={avg_val:.4f}) -> {best_path}")
                    else:
                        patience_counter += 1

                dist.barrier()
                stop = (torch.tensor([1 if patience_counter >= args.patience else 0], device=device)
                        if rank == 0 else torch.tensor([0], device=device))
                dist.broadcast(stop, src=0)
                if stop.item():
                    if rank == 0:
                        print("Early stopping.")
                    break

            dist.barrier()

    except KeyboardInterrupt:
        pass
    finally:
        try:
            dist.barrier() if dist.is_initialized() else None
        except Exception:
            pass
        if rank == 0:
            fm = Path(args.output_dir) / "final_model"
            fm.mkdir(parents=True, exist_ok=True)
            torch.save(model.module.state_dict(), str(fm / "final_model.pth"))
            ckpt = {'epoch': epoch + 1, 'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict()}
            if scheduler:
                ckpt['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(ckpt, Path(args.checkpoint_dir) / "final_checkpoint.pth")
            writer.close()
            print("Done.")
        try:
            dist.barrier() if dist.is_initialized() else None
        except Exception:
            pass
        cleanup_ddp()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DETR Ablation Training")
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--annotations_path", type=str, required=True)
    p.add_argument("--annotations_val_path", type=str, default=None,
                    help="Separate val annotations (video-level pre-split)")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--best_model_dir", type=str, default="./best_model")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--model_checkpoint", type=str, default="facebook/detr-resnet-50")
    p.add_argument("--num_queries", type=int, default=100)
    p.add_argument("--train_val_split", type=float, default=0.9)
    p.add_argument("--augment", action='store_true')
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.1)
    p.add_argument("--use_amp", action='store_true', default=False)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                    choices=["none", "cosine", "step", "plateau"])
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--lr_step_size", type=int, default=30)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--resume_training", action='store_true')
    train(p.parse_args())
