"""
MIXED GENTLE TRAINING FOR DETR - BACKGROUND AS NO-OBJECT
=====================================
Proper background training: 70% tooltip + 30% background in mixed batches
Learning rate: 1e-6 (gentle fine-tuning to preserve tooltip knowledge)
Max epochs: 3 (prevent overfitting)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
import json
import os
from pathlib import Path
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.optim import AdamW
from tqdm import tqdm
import random


class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, prefix=""):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.prefix = prefix
        
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        print(f"{prefix} Dataset: {len(self.image_ids)} images, {len(self.annotations)} annotations")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = self.images_dir / image_info['file_name']
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get annotations for this image
        image_annotations = self.annotations_by_image.get(image_id, [])
        
        # Prepare annotations for DETR processor
        annotations = []
        if image_annotations:
            # Tooltip images - have tool annotations
            for ann in image_annotations:
                annotations.append({
                    'image_id': image_id,
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    'iscrowd': ann.get('iscrowd', 0)
                })
        
        # Create proper COCO format target
        target = {
            'image_id': image_id,
            'annotations': annotations
        }
        
        # Process with DETR processor
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'pixel_mask': encoding['pixel_mask'].squeeze(),
            'labels': encoding['labels'][0] if 'labels' in encoding else None
        }


class MixedDataset(Dataset):
    """Mixed dataset that samples 70% tooltip + 30% background"""
    def __init__(self, tooltip_dataset, background_dataset, tooltip_ratio=0.7):
        self.tooltip_dataset = tooltip_dataset
        self.background_dataset = background_dataset
        self.tooltip_ratio = tooltip_ratio
        
        # Calculate effective length based on the larger dataset
        tooltip_len = len(tooltip_dataset)
        background_len = len(background_dataset)
        
        # Make sure we can sample both datasets adequately
        self.length = max(tooltip_len, int(background_len / (1 - tooltip_ratio)))
        
        print(f"Mixed Dataset: {self.length} samples (70% tooltip, 30% background)")
        print(f"Tooltip pool: {tooltip_len}, Background pool: {background_len}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Decide whether to sample from tooltip or background
        if random.random() < self.tooltip_ratio:
            # Sample from tooltip dataset
            tooltip_idx = idx % len(self.tooltip_dataset)
            return self.tooltip_dataset[tooltip_idx]
        else:
            # Sample from background dataset
            bg_idx = idx % len(self.background_dataset)
            return self.background_dataset[bg_idx]


def collate_fn(batch):
    """Custom collate function for DETR training"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
    
    labels = []
    for item in batch:
        if item['labels'] is not None:
            labels.append(item['labels'])
        else:
            # Create empty labels for background images
            labels.append({
                'class_labels': torch.tensor([], dtype=torch.long),
                'boxes': torch.tensor([], dtype=torch.float32).reshape(0, 4)
            })
    
    return {
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'labels': labels
    }


def gentle_fine_tune(args):
    """Gentle fine-tuning with mixed dataset"""
    print("MIXED GENTLE TRAINING - BACKGROUND AS NO-OBJECT")
    print(f"Device: {args.device}")
    
    # Load processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Create datasets
    print("\nLoading datasets...")
    tooltip_dataset = COCODataset(
        args.tooltip_images_dir, 
        args.tooltip_annotations_path, 
        processor, 
        "TOOLTIP"
    )
    
    background_dataset = COCODataset(
        args.background_images_dir, 
        args.background_annotations_path, 
        processor, 
        "BACKGROUND"
    )
    
    # Create mixed dataset
    mixed_dataset = MixedDataset(tooltip_dataset, background_dataset)
    
    # Split train/val (90/10)
    train_size = int(0.9 * len(mixed_dataset))
    val_size = len(mixed_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mixed_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load pretrained tooltip model
    print(f"\nLoading tooltip model: {args.checkpoint_path}")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=1, ignore_mismatched_sizes=True)
    
    # Load tooltip checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        if 'model_state_dict' in checkpoint['model_state_dict']:
            state_dict = checkpoint['model_state_dict']['model_state_dict']
        else:
            state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    
    # GENTLE fine-tuning optimizer - VERY low learning rate to preserve tooltip knowledge
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"\nTraining setup:")
    print(f"Learning rate: {args.lr} (backbone: {args.lr_backbone})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # Training
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Training E{epoch+1}"):
            pixel_values = batch['pixel_values'].to(args.device)
            pixel_mask = batch['pixel_mask'].to(args.device)
            labels = batch['labels']
            
            # Move labels to device
            for i in range(len(labels)):
                labels[i]['class_labels'] = labels[i]['class_labels'].to(args.device)
                labels[i]['boxes'] = labels[i]['boxes'].to(args.device)
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation E{epoch+1}"):
                pixel_values = batch['pixel_values'].to(args.device)
                pixel_mask = batch['pixel_mask'].to(args.device)
                labels = batch['labels']
                
                # Move labels to device
                for i in range(len(labels)):
                    labels[i]['class_labels'] = labels[i]['class_labels'].to(args.device)
                    labels[i]['boxes'] = labels[i]['boxes'].to(args.device)
                
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Validation improved: {best_val_loss:.4f}")
            
            # Save checkpoint
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'mixed_gentle_model.pth'))
            print(f"Best model saved!")
        
        model.train()
    
    print("\nMixed gentle training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: {args.output_dir}/mixed_gentle_model.pth")


def main():
    parser = argparse.ArgumentParser(description="DETR Mixed Gentle Training")
    
    # Dataset paths
    parser.add_argument("--tooltip_images_dir", type=str, 
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\train",
                       help="Path to tooltip images")
    parser.add_argument("--tooltip_annotations_path", type=str,
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\coco_annotations_from_yolo_dataset_20250218.json",
                       help="Path to tooltip annotations")
    parser.add_argument("--background_images_dir", type=str,
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\Background\train",
                       help="Path to background images")
    parser.add_argument("--background_annotations_path", type=str,
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\Background\annotations\train_annotations.json",
                       help="Path to background annotations")
    
    # Model paths
    parser.add_argument("--checkpoint_path", type=str,
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\DETR\detr_inference_model.pth",
                       help="Path to tooltip model checkpoint")
    parser.add_argument("--output_dir", type=str,
                       default="./mixed_gentle_output",
                       help="Output directory")
    
    # Training parameters - GENTLE SETTINGS
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (GENTLE)")
    parser.add_argument("--lr_backbone", type=float, default=1e-7, help="Backbone learning rate (GENTLE)")  
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3, help="Max epochs (GENTLE)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    gentle_fine_tune(args)


if __name__ == "__main__":
    main()