#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.detr import DETR
from models.backbone import build_backbone
from models.transformer import build_transformer
from data.dataset import CocoDetectionDataset
from data.transforms import get_transforms
from data.utils import collate_fn
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Set DETR parameters', add_help=False)
    
    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        help='Name of the backbone to use')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of object classes (włącznie z tłem)')
    parser.add_argument('--num_queries', default=100, type=int,
                        help='Number of object queries')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Size of the transformer hidden dimension')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads in transformer')
    parser.add_argument('--num_encoder_layers', default=6, type=int,
                        help='Number of encoder layers in transformer')
    parser.add_argument('--num_decoder_layers', default=6, type=int,
                        help='Number of decoder layers in transformer')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate in transformer')
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help='Size of the feedforward layers in transformer')
    parser.add_argument('--pre_norm', action='store_true',
                        help='Use pre-normalization in transformer')
    
    # Dataset parameters
    parser.add_argument('--data_path', 
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset', 
                        type=str, help='Path to dataset')
    parser.add_argument('--ann_file', 
                        default='augmented_coco_14655-14660_20250417_011402.json', 
                        type=str, help='Annotation file')
    parser.add_argument('--img_folder', default='images', 
                        type=str, help='Image folder')
    parser.add_argument('--fixed_size', action='store_true',
                        help='Use fixed size for images (recommended)')
    
    # Training parameters
    parser.add_argument('--epochs', default=5, type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--lr_drop', default=20, type=int,
                        help='Epoch at which to decrease lr')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Clip max norm')
    
    # Output parameters
    parser.add_argument('--output_dir', default='outputs', 
                        type=str, help='Path to save outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    
    return parser


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    transform_train, transform_val = get_transforms(fixed_size=args.fixed_size)
    
    # Poprawka do ścieżek - użycie os.path.join zamiast prostego łączenia ścieżek
    img_folder_path = os.path.join(args.data_path, args.img_folder)
    ann_file_path = os.path.join(args.data_path, args.ann_file)
    
    # Sprawdzamy, czy plik anotacji istnieje
    if not os.path.exists(ann_file_path):
        print(f"UWAGA: Plik anotacji {ann_file_path} nie istnieje!")
        print("Szukam dostępnych plików JSON w katalogu z danymi...")
        json_files = [f for f in os.listdir(args.data_path) if f.endswith('.json')]
        if json_files:
            print(f"Znaleziono następujące pliki JSON: {json_files}")
            args.ann_file = json_files[0]  # Użyj pierwszego znalezionego pliku JSON
            ann_file_path = os.path.join(args.data_path, args.ann_file)
            print(f"Używam pliku: {args.ann_file}")
        else:
            print("Nie znaleziono żadnych plików JSON w katalogu z danymi.")
            return
    
    dataset_train = CocoDetectionDataset(
        img_folder=img_folder_path,
        ann_file=ann_file_path,
        transforms=transform_train
    )
    
    # For now, use the same dataset for validation
    dataset_val = CocoDetectionDataset(
        img_folder=os.path.join(args.data_path, args.img_folder),
        ann_file=os.path.join(args.data_path, args.ann_file),
        transforms=transform_val
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset_train))
    val_size = len(dataset_train) - train_size
    
    # Upewnij się, że oba zbiory mają co najmniej jeden element
    if train_size == 0:
        train_size = 1
        val_size = len(dataset_train) - 1
    
    if val_size == 0 and len(dataset_train) > 1:
        val_size = 1
        train_size = len(dataset_train) - 1
    
    # Jeśli mamy mniej niż 2 obrazy, użyj tego samego zbioru dla treningu i walidacji
    if len(dataset_train) < 2:
        dataset_train, dataset_val = dataset_train, dataset_train
    else:
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, val_size])
    
    # Create data loaders
    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2
    )
    
    data_loader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    
    # Build model
    print("Building model...")
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim
    )
    
    # Move model to the right device
    model.to(device)
    
    # Set up optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr * 0.1,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model=model, 
            optimizer=optimizer,
            data_loader=data_loader_train,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        eval_stats = evaluate(
            model=model,
            data_loader=data_loader_val,
            device=device
        )
        
        # Log metrics
        for k, v in {**train_stats, **eval_stats}.items():
            writer.add_scalar(k, v, epoch)
        
        # Save model checkpoint
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print(f"Saved checkpoint for epoch {epoch}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model': model.state_dict(),
        'args': args,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Ustaw fixed_size domyślnie na True - rozwiąże problem różnych rozmiarów obrazów
    if not hasattr(args, 'fixed_size'):
        args.fixed_size = True
    
    main(args)