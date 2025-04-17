#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset


# Import DETR components


# Argumenty wiersza poleceń
def get_args_parser():
    parser = argparse.ArgumentParser('Ustawienia dla treningu DETR', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to frozen weights. Useful for fine-tuning")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Nazwa backbone do użycia")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--output_dir', default='output',
                        help='Ścieżka gdzie zapisać checkpointy')
    parser.add_argument('--device', default='cuda',
                        help='Urządzenie do użycia')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Wznów z checkpointa')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_classes', default=91, type=int)  # 80 classes + background

    return parser


# Definicja modelu DETR
class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        hidden_dim = transformer.d_model

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Input projection from backbone to transformer
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, samples):
        features = self.backbone(samples)
        src = features[-1]
        mask = torch.zeros_like(src[:, 0]).to(torch.bool)

        pos = torch.zeros_like(src[:, :hidden_dim])
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out


# Pomocnicze klasy i funkcje
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Transformer
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # (h*w, bs, c)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (h*w, bs, c)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (num_queries, bs, c)
        mask = mask.flatten(1)  # (bs, h*w)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


# BackboneBase do używania różnych backbones z torchvision
class BackboneBase(nn.Module):
    def __init__(self, backbone, train_backbone, num_channels, return_interm_layers):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}

        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


# ResNet backbone z torchvision
class ResNetBackbone(BackboneBase):
    def __init__(self, name, train_backbone, return_interm_layers, dilation):
        backbone = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# Klasa dla danych COCO
class CocoDetectionDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Wczytaj obraz
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_folder, path)).convert('RGB')

        # Przygotuj bounding boxes i etykiety
        boxes = []
        labels = []

        for ann in annotations:
            if 'bbox' in ann:
                # COCO format is [x, y, width, height]
                # Convert to [x_min, y_min, x_max, y_max]
                x, y, w, h = ann['bbox']
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])

        # Konwersja do tensorów
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.as_tensor([img.height, img.width]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


# Przekształcenia danych
class DetectionTransforms:
    def __init__(self, image_size=(800, 1333), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # Przekształcenia obrazu
        image = T.functional.resize(image, self.image_size)
        image = T.functional.to_tensor(image)
        image = T.functional.normalize(image, mean=self.mean, std=self.std)

        # Dostosuj bounding boxes do nowego rozmiaru
        h, w = image.shape[-2:]
        orig_h, orig_w = target['orig_size']

        scale_w = w / orig_w
        scale_h = h / orig_h

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h
            target['boxes'] = boxes

        return image, target


# Funkcja do trenowania modelu
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    metric_logger = {}
    metric_logger['loss'] = 0.0
    metric_logger['class_error'] = 0.0

    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Redukcja metryk
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = losses.item()

        metric_logger['loss'] += losses_reduced
        if 'class_error' in loss_dict_reduced:
            metric_logger['class_error'] += loss_dict_reduced['class_error']

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if i % print_freq == 0:
            print(f"{header} Batch: [{i}/{len(data_loader)}] Loss: {losses_reduced:.6f}")

    # Średnia z epoki
    metric_logger['loss'] /= len(data_loader)
    metric_logger['class_error'] /= len(data_loader)

    return metric_logger


# Funkcja do ewaluacji modelu
@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = {}
    metric_logger['loss'] = 0.0
    metric_logger['class_error'] = 0.0

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # Redukcja metryk
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}

        metric_logger['loss'] += loss_dict_reduced['loss']
        if 'class_error' in loss_dict_reduced:
            metric_logger['class_error'] += loss_dict_reduced['class_error']

    # Średnia z całości
    metric_logger['loss'] /= len(data_loader)
    metric_logger['class_error'] /= len(data_loader)

    return metric_logger


# Funkcja do zapisu checkpointa
def save_checkpoint(model, optimizer, lr_scheduler, epoch, output_dir):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
    }

    torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_{epoch:04}.pth'))
    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.pth'))


# Główna funkcja trenująca
def main(args):
    # Ustaw ziarno losowości dla powtarzalności
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # Utwórz katalog wyjściowy
    os.makedirs(args.output_dir, exist_ok=True)

    # Przygotuj model
    backbone = ResNetBackbone(args.backbone, True, False, False)
    transformer = Transformer()
    model = DETR(backbone, transformer, args.num_classes, num_queries=100)
    model.to(device)

    # Przygotuj criterion (loss function)
    class DETRLoss(nn.Module):
        def __init__(self, num_classes, matcher, weight_dict, eos_coef):
            super().__init__()
            self.num_classes = num_classes
            self.matcher = matcher
            self.weight_dict = weight_dict
            self.eos_coef = eos_coef

            empty_weight = torch.ones(num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

        def loss_labels(self, outputs, targets, indices, num_boxes):
            pred_logits = outputs['pred_logits']

            idx = indices[0][0]
            target_classes_o = targets[0]['labels'][idx]
            target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=pred_logits.device)
            target_classes[0, indices[0][0]] = target_classes_o

            loss_ce = nn.functional.cross_entropy(pred_logits.transpose(1, 2), target_classes,
                                                  self.empty_weight)
            losses = {'loss_ce': loss_ce}

            return losses

        def loss_boxes(self, outputs, targets, indices, num_boxes):
            pred_boxes = outputs['pred_boxes']

            idx = indices[0][0]
            target_boxes = targets[0]['boxes'][idx]

            loss_bbox = nn.functional.l1_loss(pred_boxes[0, idx], target_boxes, reduction='none')

            losses = {}
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(pred_boxes[0, idx]),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou'] = loss_giou.sum() / num_boxes

            return losses

        def forward(self, outputs, targets):
            # Uproszczona wersja dla przykładu
            # W rzeczywistości należy zaimplementować matcher i pełną funkcję straty

            # Przykładowe indeksy jako zamiennik matchera
            indices = [([0, 1, 2], [0, 1, 2])]
            num_boxes = sum(len(t['labels']) for t in targets)

            losses = {}
            losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

            # Oblicz całkowitą stratę jako sumę ważoną
            loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
            losses['loss'] = loss

            return losses

    # Uproszczona wersja matchera
    class HungarianMatcher:
        def __init__(self, cost_class, cost_bbox, cost_giou):
            self.cost_class = cost_class
            self.cost_bbox = cost_bbox
            self.cost_giou = cost_giou

        @torch.no_grad()
        def __call__(self, outputs, targets):
            # Uproszczona implementacja
            return [([0, 1, 2], [0, 1, 2])]

    # Box operations for loss
    class box_ops:
        @staticmethod
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(-1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=-1)

        @staticmethod
        def generalized_box_iou(boxes1, boxes2):
            # Simplified implementation
            return torch.eye(len(boxes1))

    # Inicjalizacja matchera i criterion
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    criterion = DETRLoss(args.num_classes, matcher, weight_dict, args.eos_coef)
    criterion.to(device)

    # Przygotuj optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Przygotuj dataloaders
    transforms = DetectionTransforms()

    train_dataset = CocoDetectionDataset(
        img_folder=os.path.join(args.coco_path, 'train2017'),
        ann_file=os.path.join(args.coco_path, 'annotations', 'instances_train2017.json'),
        transforms=transforms
    )

    val_dataset = CocoDetectionDataset(
        img_folder=os.path.join(args.coco_path, 'val2017'),
        ann_file=os.path.join(args.coco_path, 'annotations', 'instances_val2017.json'),
        transforms=transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=lambda x: x
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=lambda x: x
    )

    # Wznów trening z checkpointa, jeśli podano
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    # Rozpocznij trening
    print("Rozpoczynam trening...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Rozpoczynam epokę {epoch}")
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        print(f"Epoka {epoch}: Strata treningowa {train_stats['loss']:.6f}")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            # Ewaluacja
            eval_stats = evaluate(model, criterion, val_loader, device)
            print(f"Epoka {epoch}: Strata walidacyjna {eval_stats['loss']:.6f}")

            # Zapisz checkpoint
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args.output_dir)

    # Zapisz ostateczny model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))
    print("Trening zakończony!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)