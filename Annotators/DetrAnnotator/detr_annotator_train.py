import argparse
import glob
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor


class SurgicalToolDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, processor, image_size=(800, 800), augment=False):
        print("Initializing dataset...")
        self.images_dir = images_dir
        self.processor = processor
        self.image_size = image_size
        self.augment = augment

        # More aggressive augmentations for better generalization
        if self.augment:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),  # Increased rotation range
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
        else:
            self.augmentations = None

        # Load annotations from JSON file
        try:
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
            print(f"Successfully loaded annotations from {annotations_file}")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            raise

        # Set category_id to 0 for all annotations (single-class model)
        for ann in self.annotations['annotations']:
            ann['category_id'] = 0

        # Verify which images exist and build mappings
        self.valid_images = []
        self.id_to_filename = {}
        self.id_to_annotations = {}

        print(f"Searching for images in: {images_dir}")
        print(f"Total images in annotations: {len(self.annotations['images'])}")

        try:
            existing_files = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
            print(f"Files found in directory: {len(existing_files)}")
        except Exception as e:
            print(f"Error accessing directory {images_dir}: {e}")
            existing_files = set()

        valid_count = 0
        for img in self.annotations['images']:
            image_filename = img['file_name']
            if image_filename in existing_files:
                self.valid_images.append(img)
                self.id_to_filename[img['id']] = image_filename
                valid_count += 1
                if valid_count % 1000 == 0:
                    print(f"Validated {valid_count} images so far...")

        valid_annotations = 0
        # Filter to ensure only images with a single annotation are included
        for ann in self.annotations['annotations']:
            if ann['image_id'] in self.id_to_filename:
                if ann['image_id'] not in self.id_to_annotations:
                    self.id_to_annotations[ann['image_id']] = []
                self.id_to_annotations[ann['image_id']].append(ann)
                valid_annotations += 1

        # Verify that each image has exactly one tool
        imgs_with_multiple_annotations = sum(1 for anns in self.id_to_annotations.values() if len(anns) > 1)
        print(f"Images with multiple annotations: {imgs_with_multiple_annotations}")

        print(f"Loaded {len(self.valid_images)} valid images out of {len(self.annotations['images'])} in annotations")
        print(f"Dataset has {valid_annotations} valid annotations out of {len(self.annotations['annotations'])} total")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        image_info = self.valid_images[idx]
        image_id = image_info['id']
        image_filename = self.id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            raise e

        annotations = self.id_to_annotations.get(image_id, [])

        # Apply augmentations
        if self.augmentations:
            image = self.augmentations(image)

        # Ensure annotations use the full bbox for the tool
        for ann in annotations:
            # Ensure minimal bbox size to avoid tiny detections
            width = max(ann['bbox'][2], 20)  # Minimum width of 20 pixels
            height = max(ann['bbox'][3], 20)  # Minimum height of 20 pixels
            ann['bbox'][2] = width
            ann['bbox'][3] = height

        # Prepare annotations in COCO format
        coco_annotations = {'image_id': image_id, 'annotations': annotations}
        try:
            encoding = self.processor(
                images=image,
                annotations=[coco_annotations],
                return_tensors="pt",
                size={'shortest_edge': self.image_size[0], 'longest_edge': self.image_size[1]}
            )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            raise e

        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": target}


def collate_fn(batch):
    try:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        return {"pixel_values": pixel_values, "labels": labels}
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise e


def check_data_integrity(dataset: object) -> bool:
    total_images = len(dataset.annotations['images'])
    valid_images = len(dataset.valid_images)
    missing_files = total_images - valid_images

    print(f"Total images in annotations: {total_images}")
    print(f"Valid images found: {valid_images}")
    print(f"Missing image files: {missing_files}")

    images_without_annotations = sum(1 for img in dataset.valid_images if img['id'] not in dataset.id_to_annotations)
    print(f"Images without annotations: {images_without_annotations}")

    # Count the number of objects per image
    objects_per_image = {}
    for img_id, anns in dataset.id_to_annotations.items():
        objects_per_image[img_id] = len(anns)

    if len(objects_per_image) > 0:
        avg_objects = sum(objects_per_image.values()) / len(objects_per_image)
        print(f"Average objects per image: {avg_objects:.2f}")

    if valid_images == 0:
        print("CRITICAL ERROR: No valid images found!")
        return False
    elif valid_images - images_without_annotations == 0:
        print("CRITICAL ERROR: No images with annotations found!")
        return False
    else:
        print("Dataset is ready for training with available data.")
        return True


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    else:
        print("GPU not available, running on CPU")


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_epoch(model, data_loader, optimizer, device, epoch, writer=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased from 0.1 to 1.0
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        if writer is not None:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), global_step)
            if hasattr(outputs, 'loss_dict'):
                for loss_name, loss_value in outputs.loss_dict.items():
                    writer.add_scalar(f"Loss/{loss_name}", loss_value.item(), global_step)

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

    print("GPU memory usage after epoch:")
    print_gpu_memory()
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"[CHECKPOINT] Saved to: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"[CHECKPOINT] ERROR saving checkpoint: {e}")
        return None


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = checkpoint_files[-1]
    print(f"[CHECKPOINT] Found latest checkpoint: {latest}")
    return latest


def load_checkpoint(checkpoint_path, model, optimizer, device):
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Checkpoint {checkpoint_path} does not exist")
        return None, 0

    print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[CHECKPOINT] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return model, start_epoch
    except Exception as e:
        print(f"[CHECKPOINT] Error loading checkpoint: {e}")
        return None, 0


def parse_args():
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description='DETR Surgical Tool Detection - Training')

    # Data paths
    parser.add_argument('--train_images_dir', type=str,
                        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/images",
                        help='Directory containing training images')
    parser.add_argument('--train_annotations_file', type=str,
                        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_coco_450-14660_20250417_133558.json",
                        help='Path to COCO annotations JSON file')
    parser.add_argument('--checkpoint_dir', type=str,
                        default="./checkpoints",
                        help='Directory to save checkpoints')
    parser.add_argument('--best_model_dir', type=str,
                        default="./detr_tool_tracking_model_best",
                        help='Directory to save the best model')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=70,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--image_size', type=int, nargs=2, default=[800, 800],
                        help='Image size for training (height, width)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.8,
                        help='Factor for learning rate scheduler')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (0-1)')

    # Model parameters
    parser.add_argument('--num_queries', type=int, default=2,
                        help='Number of queries for DETR model')
    parser.add_argument('--bbox_cost', type=float, default=5,
                        help='Bbox cost weight')
    parser.add_argument('--class_cost', type=float, default=1,
                        help='Class cost weight')
    parser.add_argument('--giou_cost', type=float, default=4,
                        help='GIoU cost weight')
    parser.add_argument('--giou_loss_coefficient', type=float, default=4,
                        help='GIoU loss coefficient')
    parser.add_argument('--bbox_loss_coefficient', type=float, default=5,
                        help='Bbox loss coefficient')
    parser.add_argument('--eos_coefficient', type=float, default=0.1,
                        help='EOS coefficient')

    # Other settings
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--use_best_model', action='store_true',
                        help='Use saved best model if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    return args


def main():
    print("=" * 80)
    print("DETR SURGICAL TOOL DETECTION - TRAINING START")
    print("=" * 80)

    # Parse arguments
    args = parse_args()

    # Set paths from arguments
    train_images_dir = args.train_images_dir
    train_annotations_file = args.train_annotations_file
    checkpoint_dir = Path(args.checkpoint_dir)
    best_model_dir = Path(args.best_model_dir)

    print(f"[PATHS] Train Images Directory: {train_images_dir}")
    print(f"[PATHS] Train Annotations File: {train_annotations_file}")
    print(f"[PATHS] Checkpoint Directory: {checkpoint_dir.absolute()}")
    print(f"[PATHS] Best Model Directory: {best_model_dir.absolute()}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training settings from arguments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = tuple(args.image_size)

    # Learning rate scheduler parameters
    warmup_steps = args.warmup_steps
    lr_scheduler_patience = args.lr_scheduler_patience
    lr_scheduler_factor = args.lr_scheduler_factor

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir='./runs/detr_training')

    print("=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)

    # Model parameters from arguments
    num_queries = args.num_queries
    custom_config = {
        "num_queries": num_queries,
        "bbox_cost": args.bbox_cost,
        "class_cost": args.class_cost,
        "giou_cost": args.giou_cost,
        "giou_loss_coefficient": args.giou_loss_coefficient,
        "bbox_loss_coefficient": args.bbox_loss_coefficient,
        "eos_coefficient": args.eos_coefficient,
    }

    # Try to load saved model if it exists and if requested
    if best_model_dir.exists() and best_model_dir.is_dir() and args.use_best_model:
        print(f"[MODEL] Found saved model in {best_model_dir}")
        try:
            print(f"[MODEL] Attempting to load model from {best_model_dir}...")
            model = DetrForObjectDetection.from_pretrained(str(best_model_dir))
            processor = DetrImageProcessor.from_pretrained(str(best_model_dir))
            for key, value in custom_config.items():
                if hasattr(model.config, key):
                    print(f"[MODEL] Updating parameter {key} from {getattr(model.config, key)} to {value}")
                    setattr(model.config, key, value)
            print("[MODEL] SUCCESS! Loaded model and processor from saved best model.")
        except Exception as e:
            print(f"[MODEL] ERROR loading model: {e}")
            print("[MODEL] Loading default pre-trained model...")
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=1,
                num_queries=num_queries,
                ignore_mismatched_sizes=True
            )
            for key, value in custom_config.items():
                if hasattr(model.config, key):
                    print(f"[MODEL] Setting parameter {key} to {value}")
                    setattr(model.config, key, value)
            model.config.id2label = {0: "surgical_tool"}
            model.config.label2id = {"surgical_tool": 0}
            model.config.num_labels = 1
            num_channels = model.class_labels_classifier.in_features
            model.class_labels_classifier = torch.nn.Linear(num_channels, model.config.num_labels + 1)
            processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                size={'shortest_edge': image_size[0], 'longest_edge': image_size[1]}
            )
    else:
        print(f"[MODEL] No previously saved model found in {best_model_dir} or not requested")
        print("[MODEL] Loading default pre-trained model...")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,
            num_queries=num_queries,
            ignore_mismatched_sizes=True
        )
        for key, value in custom_config.items():
            if hasattr(model.config, key):
                print(f"[MODEL] Setting parameter {key} to {value}")
                setattr(model.config, key, value)
        model.config.id2label = {0: "surgical_tool"}
        model.config.label2id = {"surgical_tool": 0}
        model.config.num_labels = 1
        num_channels = model.class_labels_classifier.in_features
        model.class_labels_classifier = torch.nn.Linear(num_channels, model.config.num_labels + 1)
        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50",
            size={'shortest_edge': image_size[0], 'longest_edge': image_size[1]}
        )

    print("=" * 80)
    print("DATASET LOADING")
    print("=" * 80)

    # Create dataset with augmentation
    full_dataset = SurgicalToolDataset(
        images_dir=train_images_dir,
        annotations_file=train_annotations_file,
        processor=processor,
        image_size=image_size,
        augment=True  # Enable augmentations for training
    )

    if not check_data_integrity(full_dataset):
        print("Aborting training due to data issues in the dataset.")
        return

    # Create validation split
    dataset_size = len(full_dataset)
    val_split = args.val_split
    train_size = int((1 - val_split) * dataset_size)
    val_size = dataset_size - train_size

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Add learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=True
    )

    model.to(device)

    print("=" * 80)
    print("CHECKPOINT CHECKING")
    print("=" * 80)

    start_epoch = 0
    if args.resume_training:
        latest_checkpoint = find_latest_checkpoint(str(checkpoint_dir))
        if latest_checkpoint:
            loaded_model, start_epoch = load_checkpoint(latest_checkpoint, model, optimizer, device)
            if loaded_model is not None:
                model = loaded_model
                print(f"Resuming training from epoch {start_epoch}")
            else:
                start_epoch = 0
                print("Starting training from beginning (checkpoint loading failed)")
        else:
            print("[CHECKPOINT] No checkpoints found, starting training from scratch.")
    else:
        print("[CHECKPOINT] Not resuming from checkpoint, starting training from scratch.")

    print("=" * 80)
    print(f"TRAINING STARTING FROM EPOCH {start_epoch + 1}")
    print("=" * 80)

    # Track best model for early stopping
    best_val_loss = float('inf')
    patience = 10  # Wait for 10 epochs without improvement before stopping
    patience_counter = 0
    
    # Initialize epoch variable in case training loop is skipped
    epoch = start_epoch - 1
    
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}...")
            avg_train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
            print(f"Epoch {epoch + 1} finished. Training Loss: {avg_train_loss:.4f}")

            # Validate after each epoch
            avg_val_loss = evaluate_model(model, val_loader, device)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Log validation loss
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)

            # Update learning rate based on validation loss
            lr_scheduler.step(avg_val_loss)

            # Early stopping and model saving logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save best model
                print(f"[MODEL] New best validation loss: {best_val_loss:.4f}, saving model...")
                os.makedirs(str(best_model_dir), exist_ok=True)
                model.save_pretrained(str(best_model_dir / f"detr_tool_tracking_model_best_epoch_{epoch + 1}"))
                processor.save_pretrained(str(best_model_dir / f"detr_tool_tracking_model_best_epoch_{epoch + 1}"))
            else:
                patience_counter += 1
                print(
                    f"[EARLY STOPPING] No improvement for {patience_counter} epochs. Best val loss: {best_val_loss:.4f}")

                if patience_counter >= patience:
                    print(f"[EARLY STOPPING] Stopping training after {patience} epochs without improvement.")
                    break

            # Save checkpoint every 4 epochs
            if (epoch + 1) % 4 == 0:
                save_checkpoint(model, optimizer, epoch, str(checkpoint_dir))

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model and checkpoint...")
        save_checkpoint(model, optimizer, epoch, str(checkpoint_dir))
        model.save_pretrained("./detr_tool_tracking_model_interrupted")
        processor.save_pretrained("./detr_tool_tracking_model_interrupted")
        print("Interrupted model saved to: ./detr_tool_tracking_model_interrupted")

    print("=" * 80)
    print("TRAINING FINISHED - SAVING FINAL MODEL")
    print("=" * 80)

    # Save final checkpoint with proper epoch value
    save_checkpoint(model, optimizer, epoch, str(checkpoint_dir))
    model.save_pretrained("./detr_tool_tracking_model_final")
    processor.save_pretrained("./detr_tool_tracking_model_final")
    print("Model and processor saved successfully.")
    print("Final model saved to: ./detr_tool_tracking_model_final")

    writer.close()


if __name__ == "__main__":
    main()
