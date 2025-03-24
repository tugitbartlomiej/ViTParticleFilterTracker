import glob
import json
import os

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset, random_split
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

        # Augmentation pipeline
        if self.augment:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
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

        # Sort valid images by filename to ensure consistent ordering
        self.valid_images.sort(key=lambda x: x['file_name'])

        valid_annotations = 0
        for ann in self.annotations['annotations']:
            if ann['image_id'] in self.id_to_filename:
                if ann['image_id'] not in self.id_to_annotations:
                    self.id_to_annotations[ann['image_id']] = []
                self.id_to_annotations[ann['image_id']].append(ann)
                valid_annotations += 1

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

        if self.augmentations:
            image = self.augmentations(image)

        # Prepare annotations in COCO format
        coco_annotations = {
            'image_id': image_id,
            'annotations': annotations
        }

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

        return {"pixel_values": pixel_values, "labels": target, "image_path": image_path}


def collate_fn(batch):
    try:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        return {"pixel_values": pixel_values, "labels": labels, "image_paths": image_paths}
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise e


def check_data_integrity(dataset):
    """Verify dataset integrity and return whether training can proceed."""
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


def select_image_range(dataset, start_idx, end_idx):
    """
    Select a subset of images from the dataset based on index range.

    Args:
        dataset: Full dataset
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)

    Returns:
        Subset of the dataset
    """
    # Determine start and end indices
    start = max(0, min(start_idx, len(dataset) - 1))
    end = min(end_idx, len(dataset))

    if start >= end:
        raise ValueError(f"Invalid range: start_idx ({start}) must be less than end_idx ({end})")

    # Create range of indices
    selected_indices = list(range(start, end))

    # Check if selected images have annotations
    valid_indices = []
    for idx in selected_indices:
        image_info = dataset.valid_images[idx]
        image_id = image_info['id']
        if image_id in dataset.id_to_annotations and len(dataset.id_to_annotations[image_id]) > 0:
            valid_indices.append(idx)

    print(f"Selected {len(valid_indices)} images with annotations from range {start} to {end}")

    # Create subset with selected indices
    return Subset(dataset, valid_indices)


def save_images_with_annotations(dataset, indices, output_dir, max_images=100):
    """
    Save images with drawn bounding boxes for visual inspection.

    Args:
        dataset: Dataset containing images and annotations
        indices: List of image indices to process
        output_dir: Folder to save resulting images
        max_images: Maximum number of images to process
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images with annotations to {output_dir}...")

    # Limit the number of images to process
    indices = indices[:min(len(indices), max_images)]

    for idx in tqdm(indices, desc="Generating visualizations"):
        # Get image and annotations
        image_info = dataset.valid_images[idx]
        image_id = image_info['id']
        image_filename = dataset.id_to_filename[image_id]
        image_path = os.path.join(dataset.images_dir, image_filename)

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Get annotations
        annotations = dataset.id_to_annotations.get(image_id, [])

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)

        for ann in annotations:
            # COCO format bbox is [x, y, width, height]
            # Need to convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Add category label and annotation ID
            label = f"ID: {ann['id']} Cat: {ann['category_id']}"
            draw.text((x1, max(0, y1 - 15)), label, fill="red")

        # Add image information
        draw.text((10, 10), f"Image ID: {image_id}, Filename: {image_filename}", fill="blue")
        draw.text((10, 30), f"Annotations: {len(annotations)}", fill="blue")

        # Save image with annotations
        output_filename = f"{idx}_{image_filename}"
        output_path = os.path.join(output_dir, output_filename)
        image.save(output_path)

    print(f"Saved {len(indices)} images with annotations in {output_dir}")


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    else:
        print("GPU not available, running on CPU")


def train_epoch(model, data_loader, optimizer, device, epoch, writer=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        # TensorBoard logging
        if writer is not None:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), global_step)

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

    # GPU memory usage after epoch
    print("GPU memory usage after epoch:")
    print_gpu_memory()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate_epoch(model, data_loader, device, epoch, writer=None):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Validation Epoch {epoch + 1}", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix({"val_loss": loss.item()})

            # TensorBoard logging
            if writer is not None:
                global_step = epoch * len(data_loader) + batch_idx
                writer.add_scalar("Loss/validation", loss.item(), global_step)

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_dir):
    """Save training state checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"[CHECKPOINT] Saved to: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"[CHECKPOINT] ERROR saving checkpoint: {e}")
        return None


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = checkpoint_files[-1]
    print(f"[CHECKPOINT] Found latest checkpoint: {latest}")
    return latest


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training state from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Checkpoint {checkpoint_path} does not exist")
        return None, 0, float('inf')

    print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"[CHECKPOINT] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return model, start_epoch, best_val_loss
    except Exception as e:
        print(f"[CHECKPOINT] Error loading checkpoint: {e}")
        return None, 0, float('inf')


def main():
    print("=" * 80)
    print("DETR SURGICAL TOOL DETECTION - STARTING TRAINING")
    print("=" * 80)

    # ========== HARDCODED PARAMETERS ==========
    # Paths for data and output
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train"
    annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Detr/coco_annotations_from_yolo_dataset_20250218.json"
    output_dir = "./training_ranged/output"
    checkpoint_dir = "./training_ranged/checkpoints"
    best_model_dir = "./training_ranged/best_model"
    visualization_dir = "./training_ranged/visualizations"

    # Image selection range - MODIFY THESE TO SELECT DIFFERENT IMAGES
    start_idx = 400  # Start from image 750
    end_idx = 1000  # End at image 800 (exclusive)

    # Training parameters
    num_epochs = 80
    learning_rate = 1e-5
    batch_size = 4
    image_size = (800, 800)
    checkpoint_interval = 4  # Save checkpoint every 4 epochs
    validation_loss_threshold = 0.1  # Early stopping threshold
    # =========================================

    print(f"Training on images from {start_idx} to {end_idx}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Setup paths
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    best_model_dir = os.path.abspath(best_model_dir)

    # Display paths for better diagnostics
    print(f"[PATHS] Images directory: {os.path.abspath(images_dir)}")
    print(f"[PATHS] Annotations file: {os.path.abspath(annotations_file)}")
    print(f"[PATHS] Output directory: {os.path.abspath(output_dir)}")
    print(f"[PATHS] Checkpoint directory: {checkpoint_dir}")
    print(f"[PATHS] Best model directory: {best_model_dir}")

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")
    print(f"[TRAINING] Number of epochs: {num_epochs}")
    print(f"[TRAINING] Batch size: {batch_size}")
    print(f"[TRAINING] Learning rate: {learning_rate}")
    print(f"[TRAINING] Image size: {image_size}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    print("=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)

    # Model configuration parameters
    custom_config = {
        "num_queries": 5,  # Reduced from 100 since we have max 2 tools per image
        "bbox_cost": 2,  # Change from default 5
        "class_cost": 2,  # Increase from default 1
        "giou_cost": 2,  # Standard IoU cost
        "giou_loss_coefficient": 2,  # Standard IoU loss coefficient
        "bbox_loss_coefficient": 2,  # Reduced from default 5
        "eos_coefficient": 0.8,  # Significantly increased from 0.1 to penalize empty predictions
    }

    # First, check if a saved model exists
    if os.path.exists(best_model_dir) and os.listdir(best_model_dir):
        print(f"[MODEL] Found saved model in {best_model_dir}")
        try:
            print(f"[MODEL] Attempting to load model from {best_model_dir}...")
            # Load saved model and processor
            model = DetrForObjectDetection.from_pretrained(best_model_dir)
            processor = DetrImageProcessor.from_pretrained(best_model_dir)

            # Update configuration of loaded model
            for key, value in custom_config.items():
                if hasattr(model.config, key):
                    print(f"[MODEL] Updating parameter {key} from {getattr(model.config, key)} to {value}")
                    setattr(model.config, key, value)

            print("[MODEL] SUCCESS! Successfully loaded model and processor.")
        except Exception as e:
            print(f"[MODEL] ERROR loading model: {e}")
            print("[MODEL] Loading default pre-trained model...")
            # Load default model in case of error
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=1,
                num_queries=custom_config["num_queries"],
                ignore_mismatched_sizes=True
            )

            # Apply configuration changes
            for key, value in custom_config.items():
                if hasattr(model.config, key):
                    print(f"[MODEL] Setting parameter {key} to {value}")
                    setattr(model.config, key, value)

            # Configure model for one class
            model.config.id2label = {0: "surgical_tool"}
            model.config.label2id = {"surgical_tool": 0}
            model.config.num_labels = 1

            # Initialize classification layer
            num_channels = model.class_labels_classifier.in_features
            model.class_labels_classifier = torch.nn.Linear(num_channels, model.config.num_labels + 1)

            # Initialize processor
            processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                size={'shortest_edge': image_size[0], 'longest_edge': image_size[1]}
            )
    else:
        print(f"[MODEL] No previously saved model found in {best_model_dir}")
        print("[MODEL] Loading default pre-trained model...")
        # Load pre-trained DETR model and processor
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,
            num_queries=custom_config["num_queries"],
            ignore_mismatched_sizes=True
        )

        # Apply configuration changes
        for key, value in custom_config.items():
            if hasattr(model.config, key):
                print(f"[MODEL] Setting parameter {key} to {value}")
                setattr(model.config, key, value)

        # Configure model for one class
        model.config.id2label = {0: "surgical_tool"}
        model.config.label2id = {"surgical_tool": 0}
        model.config.num_labels = 1

        # Initialize classification layer
        num_channels = model.class_labels_classifier.in_features
        model.class_labels_classifier = torch.nn.Linear(num_channels, model.config.num_labels + 1)

        # Initialize processor
        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50",
            size={'shortest_edge': image_size[0], 'longest_edge': image_size[1]}
        )

    print("=" * 80)
    print("DATASET LOADING")
    print("=" * 80)

    # Load full dataset
    full_dataset = SurgicalToolDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        processor=processor,
        image_size=image_size,
        augment=False  # Don't apply augmentations to the full dataset
    )

    if not check_data_integrity(full_dataset):
        print("Aborting training due to issues with dataset.")
        return

    # Save visualization of annotations for a subset of images
    visualization_samples_dir = os.path.join(output_dir, "annotation_samples")
    os.makedirs(visualization_samples_dir, exist_ok=True)

    # Select a few images for visualization
    selected_indices = list(range(start_idx, min(start_idx + 20, end_idx)))
    save_images_with_annotations(
        dataset=full_dataset,
        indices=selected_indices,
        output_dir=visualization_samples_dir,
        max_images=20
    )

    # Select range of images for training
    print(f"[DATASET] Selecting images from index {start_idx} to {end_idx}")
    selected_dataset = select_image_range(full_dataset, start_idx, end_idx)
    print(f"[DATASET] Selected {len(selected_dataset)} images for training")

    # Split dataset into training and validation sets
    dataset_size = len(selected_dataset)
    val_split = 0.1  # 10% for validation
    train_size = int((1 - val_split) * dataset_size)
    val_size = dataset_size - train_size

    # Set seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(selected_dataset, [train_size, val_size])

    print(f"[DATASET] Training set size: {len(train_dataset)}")
    print(f"[DATASET] Validation set size: {len(val_dataset)}")

    # Prepare data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    # Move model to device
    model.to(device)

    print("=" * 80)
    print("CHECKPOINT CHECKING")
    print("=" * 80)

    # Check for existing checkpoints and load the latest one
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        loaded_model, start_epoch, best_val_loss = load_checkpoint(
            latest_checkpoint, model, optimizer, scheduler, device
        )
        if loaded_model is not None:
            model = loaded_model
            print(f"Starting training from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_val_loss = float('inf')
            print("Starting training from beginning (checkpoint loading failed)")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        print("[CHECKPOINT] No checkpoints found, starting training from scratch.")

    print("=" * 80)
    print(f"TRAINING STARTING FROM EPOCH {start_epoch + 1}")
    print("=" * 80)

    # Training loop
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}...")
            avg_train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)

            # Perform validation
            avg_val_loss = validate_epoch(model, val_loader, device, epoch, writer)

            # Adjust learning rate based on validation loss
            scheduler.step(avg_val_loss)

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                model.save_pretrained(best_model_dir)
                processor.save_pretrained(best_model_dir)
                print(f"Best model saved to: {best_model_dir}")

            # Save checkpoint at regular intervals
            if (epoch + 1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_dir)

            print(
                f"Epoch {epoch + 1} finished. Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Early stopping if validation loss reaches threshold
            if avg_val_loss <= validation_loss_threshold:
                print(f"Validation loss has reached threshold of {validation_loss_threshold}. Stopping training.")
                # Save final checkpoint before stopping
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_dir)
                break

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model and checkpoint...")
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_dir)
        model.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_interrupted"))
        processor.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_interrupted"))
        print(f"Interrupted model saved to: {os.path.join(output_dir, 'detr_tool_tracking_model_interrupted')}")

    # Save final model
    print("=" * 80)
    print("TRAINING FINISHED - SAVING FINAL MODEL")
    print("=" * 80)

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_dir)
    model.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_final"))
    processor.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_final"))
    print("Model and processor saved successfully.")
    print(f"Final model saved to: {os.path.join(output_dir, 'detr_tool_tracking_model_final')}")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()