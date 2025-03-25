import glob
import json
import os
import pickle
import random
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor


# Define worker initialization function OUTSIDE of main()
# This is critical for Windows multiprocessing to work
def worker_init_function(worker_id):
    """Initialize worker with reproducible seed"""
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)
    random.seed(42 + worker_id)


class SurgicalToolDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, processor, image_size=(800, 800),
                 augment=False, augment_strength='mild', cache_dir=None, max_cache_size=1000):
        """
        Initialize dataset with optimized loading and caching and gentle augmentations.

        Args:
            images_dir: Directory containing images
            annotations_file: Path to COCO format annotations
            processor: DetrImageProcessor for preprocessing
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            augment_strength: Controls augmentation intensity ('mild', 'medium', or 'strong')
            cache_dir: Directory to store cache files (None = use annotations directory)
            max_cache_size: Maximum number of processed images to keep in memory
        """
        print("Initializing dataset...")
        self.images_dir = images_dir
        self.processor = processor
        self.image_size = image_size
        self.augment = augment

        # Initialize cache
        self.processor_cache = {}
        self.max_cache_size = max_cache_size

        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.dirname(annotations_file)
        os.makedirs(cache_dir, exist_ok=True)

        # Set augmentation parameters based on strength
        if augment_strength == 'mild':
            # Very gentle augmentations appropriate for medical imagery
            rotation_degrees = 5
            translate = (0.05, 0.05)
            scale = (0.95, 1.05)
            brightness = 0.1
            contrast = 0.1
            saturation = 0.05
            hue = 0.02
            blur_sigma = (0.1, 1.0)
            flip_prob = 0.3
        elif augment_strength == 'medium':
            # Moderate augmentations
            rotation_degrees = 10
            translate = (0.1, 0.1)
            scale = (0.9, 1.1)
            brightness = 0.2
            contrast = 0.2
            saturation = 0.1
            hue = 0.05
            blur_sigma = (0.1, 1.5)
            flip_prob = 0.5
        elif augment_strength == 'strong':
            # Stronger augmentations, but still controlled
            rotation_degrees = 15
            translate = (0.15, 0.15)
            scale = (0.85, 1.15)
            brightness = 0.3
            contrast = 0.3
            saturation = 0.15
            hue = 0.07
            blur_sigma = (0.1, 2.0)
            flip_prob = 0.5
        else:
            raise ValueError(f"Unknown augmentation strength: {augment_strength}")

        print(f"Using {augment_strength} augmentations with rotation {rotation_degrees}°")

        # Augmentation pipeline with gentle transforms suitable for medical imagery
        if self.augment:
            # Create a list of transforms that will be applied with probability
            transform_list = []

            # Add geometric transforms
            transform_list.append(
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=rotation_degrees,
                        translate=translate,
                        scale=scale,
                        fill=0,  # Fill with black
                    ),
                ], p=0.7)  # Apply geometric transforms with 70% probability
            )

            # Add horizontal flip if appropriate
            transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

            # Add color transforms
            transform_list.append(
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue
                    ),
                ], p=0.5)  # Apply color transforms with 50% probability
            )

            # Add slight blur occasionally
            transform_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=blur_sigma)
                ], p=0.2)  # Apply blur with only 20% probability
            )

            # Create the final composition of transforms
            self.augmentations = transforms.Compose(transform_list)
        else:
            self.augmentations = None

        # Create a unique cache file name based on dataset parameters
        cache_name = f"dataset_cache_{os.path.basename(annotations_file).split('.')[0]}"
        if augment:
            cache_name += "_augmented"
        self.cache_file = os.path.join(cache_dir, f"{cache_name}.pkl")

        # Try to load from cache first
        if os.path.exists(self.cache_file):
            print(f"Loading dataset from cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.valid_images = cache_data['valid_images']
                    self.id_to_filename = cache_data['id_to_filename']
                    self.id_to_annotations = cache_data['id_to_annotations']
                    print(f"Successfully loaded {len(self.valid_images)} images from cache")
                    return
            except Exception as e:
                print(f"Error loading from cache: {e}. Rebuilding dataset...")

        # If cache loading failed, build dataset from scratch
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
        # This is done more efficiently using set operations
        self.valid_images = []
        self.id_to_filename = {}
        self.id_to_annotations = {}

        print(f"Searching for images in: {images_dir}")
        print(f"Total images in annotations: {len(self.annotations['images'])}")

        # Get all existing files as a set for O(1) lookup
        try:
            existing_files = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
            print(f"Files found in directory: {len(existing_files)}")
        except Exception as e:
            print(f"Error accessing directory {images_dir}: {e}")
            existing_files = set()

        # Process images and check if they exist
        valid_count = 0
        for img in self.annotations['images']:
            image_filename = img['file_name']
            if image_filename in existing_files:
                self.valid_images.append(img)
                self.id_to_filename[img['id']] = image_filename
                valid_count += 1
                if valid_count % 1000 == 0:
                    print(f"Validated {valid_count} images so far...")

        # Sort valid images by filename for consistent ordering
        self.valid_images.sort(key=lambda x: x['file_name'])

        # Build annotation mapping more efficiently
        valid_annotations = 0
        for ann in self.annotations['annotations']:
            if ann['image_id'] in self.id_to_filename:
                if ann['image_id'] not in self.id_to_annotations:
                    self.id_to_annotations[ann['image_id']] = []
                self.id_to_annotations[ann['image_id']].append(ann)
                valid_annotations += 1

        print(f"Loaded {len(self.valid_images)} valid images out of {len(self.annotations['images'])} in annotations")
        print(f"Dataset has {valid_annotations} valid annotations out of {len(self.annotations['annotations'])} total")

        # Save to cache for future use
        try:
            cache_data = {
                'valid_images': self.valid_images,
                'id_to_filename': self.id_to_filename,
                'id_to_annotations': self.id_to_annotations
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Dataset cache saved to {self.cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save dataset cache: {e}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        """Get dataset item with optimized processing and caching"""
        image_info = self.valid_images[idx]
        image_id = image_info['id']
        image_filename = self.id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, image_filename)

        # Cache key combines image path and augmentation state
        cache_key = f"{image_path}_{self.augment}"

        # Try to retrieve from cache first
        if cache_key in self.processor_cache:
            return self.processor_cache[cache_key]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            raise e

        annotations = self.id_to_annotations.get(image_id, [])

        # Apply augmentations
        if self.augmentations and self.augment:
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

        result = {
            "pixel_values": pixel_values,
            "labels": target,
            "image_path": image_path
        }

        # Add to cache, managing cache size
        if len(self.processor_cache) >= self.max_cache_size:
            # Remove a random item when cache is full
            keys = list(self.processor_cache.keys())
            del self.processor_cache[keys[0]]

        self.processor_cache[cache_key] = result
        return result


def collate_fn(batch):
    """Optimized collate function with error handling"""
    try:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        return {"pixel_values": pixel_values, "labels": labels, "image_paths": image_paths}
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Print problematic batch items to help debugging
        for i, item in enumerate(batch):
            if "pixel_values" not in item or "labels" not in item:
                print(f"Item {i} is missing key fields: {item.keys()}")
        raise e


def check_data_integrity(dataset):
    """Verify dataset integrity and return whether training can proceed."""
    total_images = len(dataset.annotations['images']) if hasattr(dataset, 'annotations') else "unknown"
    valid_images = len(dataset.valid_images)

    if hasattr(dataset, 'annotations'):
        missing_files = len(dataset.annotations['images']) - valid_images
    else:
        missing_files = "unknown"

    print(f"Total images in annotations: {total_images}")
    print(f"Valid images found: {valid_images}")
    print(f"Missing image files: {missing_files}")

    # Count images without annotations
    images_without_annotations = sum(1 for img in dataset.valid_images if img['id'] not in dataset.id_to_annotations)
    print(f"Images without annotations: {images_without_annotations}")

    # Count the number of objects per image
    objects_per_image = {}
    for img_id, anns in dataset.id_to_annotations.items():
        objects_per_image[img_id] = len(anns)

    if len(objects_per_image) > 0:
        avg_objects = sum(objects_per_image.values()) / len(objects_per_image)
        print(f"Average objects per image: {avg_objects:.2f}")

        # More detailed statistics
        objects_counts = list(objects_per_image.values())
        print(f"Min objects per image: {min(objects_counts)}")
        print(f"Max objects per image: {max(objects_counts)}")

        # Count frequency of object counts
        from collections import Counter
        count_freq = Counter(objects_counts)
        print("Object count distribution:")
        for count, freq in sorted(count_freq.items()):
            print(f"  {count} object(s): {freq} images ({freq / len(objects_per_image) * 100:.1f}%)")

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
    """Print detailed GPU memory usage statistics."""
    if not torch.cuda.is_available():
        print("GPU not available, running on CPU")
        return

    print("\n===== GPU Memory Usage =====")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    # More detailed stats if available
    if hasattr(torch.cuda, 'memory_stats'):
        stats = torch.cuda.memory_stats()
        print(f"Active Allocations: {stats['active_bytes.all.current'] / 1024 ** 2:.2f} MB")
        print(f"Peak Allocated: {stats.get('allocated_bytes.all.peak', 0) / 1024 ** 2:.2f} MB")

    # Print per-device stats if multiple GPUs
    if torch.cuda.device_count() > 1:
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")


def select_image_range(dataset, start_idx, end_idx):
    """
    Select a subset of images from the dataset based on index range with improved error checking.

    Args:
        dataset: Full dataset
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)

    Returns:
        Subset of the dataset
    """
    # Validate inputs
    if not isinstance(start_idx, int) or not isinstance(end_idx, int):
        raise TypeError(f"Indices must be integers: start_idx={start_idx}, end_idx={end_idx}")

    # Determine start and end indices with bounds checking
    start = max(0, min(start_idx, len(dataset) - 1))
    end = min(end_idx, len(dataset))

    if start >= end:
        raise ValueError(f"Invalid range: start_idx ({start}) must be less than end_idx ({end})")

    # Create range of indices
    selected_indices = list(range(start, end))

    # Check if selected images have annotations and filter invalid ones
    valid_indices = []
    for idx in selected_indices:
        image_info = dataset.valid_images[idx]
        image_id = image_info['id']
        if image_id in dataset.id_to_annotations and len(dataset.id_to_annotations[image_id]) > 0:
            valid_indices.append(idx)

    valid_percentage = len(valid_indices) / len(selected_indices) * 100 if selected_indices else 0
    print(
        f"Selected {len(valid_indices)} images with annotations from range {start} to {end} ({valid_percentage:.1f}% valid)")

    # Create subset with selected indices
    return Subset(dataset, valid_indices)


def save_images_with_annotations(dataset, indices, output_dir, max_images=20):
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


def train_epoch(model, data_loader, optimizer, device, epoch, scaler,
                writer=None, log_every=10, grad_accum_steps=1, empty_cache_freq=20):
    """
    Train model for one epoch using mixed precision and gradient accumulation.

    Args:
        model: The DETR model
        data_loader: Training data loader
        optimizer: Model optimizer
        device: Device to train on (cuda/cpu)
        epoch: Current epoch number
        scaler: GradScaler for mixed precision training
        writer: TensorBoard writer
        log_every: Log every N batches
        grad_accum_steps: Number of gradient accumulation steps
        empty_cache_freq: Empty CUDA cache every N batches
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

    # Track time for benchmarking using CUDA events if available
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    # Track batch times for reporting
    batch_times = []

    optimizer.zero_grad()  # Zero gradients at start of epoch

    for batch_idx, batch in enumerate(progress_bar):
        batch_start = time.time()

        # Move data to device
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch["labels"]]

        # Forward pass with mixed precision
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=scaler.is_enabled()):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / grad_accum_steps  # Normalize loss for gradient accumulation

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights after gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Calculate full loss for metrics
        total_loss += outputs.loss.item() * grad_accum_steps  # Un-normalize for reporting

        # Calculate batch processing time
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)

        # Update progress bar
        avg_batch_time = sum(batch_times[-50:]) / min(len(batch_times), 50)
        progress_bar.set_postfix({
            "loss": outputs.loss.item() * grad_accum_steps,
            "batch_time": f"{avg_batch_time:.3f}s",
            "imgs/sec": f"{batch['pixel_values'].size(0) / avg_batch_time:.1f}"
        })

        # Periodic memory cleanup for CUDA
        if torch.cuda.is_available() and batch_idx % empty_cache_freq == 0:
            torch.cuda.empty_cache()

        # TensorBoard logging
        if writer is not None and batch_idx % log_every == 0:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Loss/train", outputs.loss.item() * grad_accum_steps, global_step)
            writer.add_scalar("Performance/images_per_second", batch['pixel_values'].size(0) / avg_batch_time,
                              global_step)

            # Log additional loss components if available
            if hasattr(outputs, 'loss_dict'):
                for loss_name, loss_value in outputs.loss_dict.items():
                    writer.add_scalar(f"Loss/{loss_name}", loss_value.item(), global_step)

        if batch_idx % log_every == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(data_loader)}], "
                  f"Loss: {outputs.loss.item() * grad_accum_steps:.4f}, "
                  f"Rate: {batch['pixel_values'].size(0) / avg_batch_time:.1f} img/s")

    # Make sure to update any remaining gradients
    if len(data_loader) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Report epoch statistics using CUDA timing if available
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        epoch_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    else:
        epoch_time = time.time() - start_time

    avg_loss = total_loss / len(data_loader)
    images_per_sec = len(data_loader.dataset) / epoch_time

    print(
        f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - Avg loss: {avg_loss:.4f} - Rate: {images_per_sec:.1f} img/s")

    # GPU memory usage after epoch
    print("GPU memory usage after epoch:")
    print_gpu_memory()

    return avg_loss


def validate_epoch(model, data_loader, device, epoch, writer=None):
    """
    Validate model on validation set with additional metrics.

    Args:
        model: The DETR model
        data_loader: Validation data loader
        device: Device to validate on (cuda/cpu)
        epoch: Current epoch number
        writer: TensorBoard writer
    """
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Validation Epoch {epoch + 1}", leave=False)

    # Track additional metrics for validation
    all_confidences = []
    all_ious = []
    boxes_per_image = []
    zero_box_images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch["labels"]]

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix({"val_loss": loss.item()})

            # TensorBoard logging
            if writer is not None and batch_idx % 10 == 0:
                global_step = epoch * len(data_loader) + batch_idx
                writer.add_scalar("Loss/validation", loss.item(), global_step)

            # Calculate additional metrics from predictions
            logits = outputs.logits
            pred_boxes = outputs.pred_boxes

            # Iterate through batch
            for i in range(len(labels)):
                # Get prediction confidences
                probs = logits[i].softmax(-1)
                scores = probs[:, :-1].max(-1).values
                keep = scores > 0.5  # Threshold for confident predictions

                # Track confident predictions per image
                confident_boxes = keep.sum().item()
                boxes_per_image.append(confident_boxes)
                if confident_boxes == 0:
                    zero_box_images += 1

                # Record confidence scores
                all_confidences.extend(scores[keep].cpu().numpy().tolist())

                # TODO: Calculate IoU between predictions and ground truth
                # This requires more detailed processing of the DETR outputs
                # Placeholder for now

    avg_loss = total_loss / len(data_loader)

    # Summarize validation metrics
    avg_boxes = sum(boxes_per_image) / len(boxes_per_image) if boxes_per_image else 0
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    print(f"Validation Metrics - Loss: {avg_loss:.4f}, Avg boxes/image: {avg_boxes:.2f}, "
          f"Avg confidence: {avg_confidence:.4f}, Empty predictions: {zero_box_images}/{len(boxes_per_image)}")

    # Log detailed metrics to TensorBoard
    if writer is not None:
        writer.add_scalar("Metrics/validation_loss", avg_loss, epoch)
        writer.add_scalar("Metrics/avg_boxes_per_image", avg_boxes, epoch)
        writer.add_scalar("Metrics/avg_confidence", avg_confidence, epoch)
        writer.add_scalar("Metrics/empty_prediction_rate", zero_box_images / len(boxes_per_image), epoch)

        # Add confidence score histogram
        if all_confidences:
            writer.add_histogram("Validation/confidence_scores", np.array(all_confidences), epoch)

    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, scaler, checkpoint_dir):
    """Save training state checkpoint with mixed precision scaler."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load training state from checkpoint including mixed precision scaler."""
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Checkpoint {checkpoint_path} does not exist")
        return None, 0, float('inf')

    print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint[
            'scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"[CHECKPOINT] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return model, start_epoch, best_val_loss
    except Exception as e:
        print(f"[CHECKPOINT] Error loading checkpoint: {e}")
        return None, 0, float('inf')


def main():
    print("=" * 80)
    print("DETR SURGICAL TOOL DETECTION - STARTING TRAINING (OPTIMIZED VERSION)")
    print("=" * 80)

    # ========== CONFIGURABLE PARAMETERS ==========
    # Paths for data and output
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train"
    annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Detr/coco_annotations_from_yolo_dataset_20250218.json"
    output_dir = "./training_ranged/output"
    checkpoint_dir = "./training_ranged/checkpoints"
    best_model_dir = "./training_ranged/best_model"
    visualization_dir = "./training_ranged/visualizations"
    cache_dir = "./training_ranged/cache"  # New: cache directory

    # Image selection range
    start_idx = 992
    end_idx = 1000

    # Training parameters
    num_epochs = 80
    learning_rate = 5e-5  # Increased from 1e-5
    batch_size = 8  # Increased from 4
    gradient_accumulation_steps = 2  # New: each effective batch = batch_size * gradient_accumulation_steps
    image_size = (640, 640)  # Reduced from (800, 800) for faster training
    early_stopping_patience = 10  # New: stop training after N epochs without improvement

    # Augmentation settings
    apply_augmentations = True  # Whether to use data augmentation
    augmentation_strength = 'mild'  # Options: 'mild', 'medium', 'strong'

    # Memory optimization
    empty_cache_freq = 20  # Empty CUDA cache every N batches

    # Checkpointing and validation
    checkpoint_interval = 4  # Save checkpoint every N epochs
    validation_interval = 2  # Only validate every N epochs to save time
    validation_loss_threshold = 0.1  # Early stopping threshold

    # Performance optimization
    mixed_precision = True  # Enable mixed precision training
    torch_compile = hasattr(torch, 'compile')  # Use torch.compile if available (PyTorch 2.0+)
    pin_memory = True
    num_workers = 0  # Use 0 to avoid multiprocessing issues on Windows
    prefetch_factor = 2
    # =========================================

    print(f"Training on images from {start_idx} to {end_idx}")
    print(f"Using mixed precision: {mixed_precision}")
    print(f"Using torch.compile: {torch_compile}")
    print(f"Image size: {image_size}")
    print(
        f"Gradient accumulation steps: {gradient_accumulation_steps} (effective batch size: {batch_size * gradient_accumulation_steps})")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Setup paths
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    best_model_dir = os.path.abspath(best_model_dir)

    # Display paths for better diagnostics
    print(f"[PATHS] Images directory: {os.path.abspath(images_dir)}")
    print(f"[PATHS] Annotations file: {os.path.abspath(annotations_file)}")
    print(f"[PATHS] Output directory: {os.path.abspath(output_dir)}")
    print(f"[PATHS] Checkpoint directory: {checkpoint_dir}")
    print(f"[PATHS] Best model directory: {best_model_dir}")
    print(f"[PATHS] Cache directory: {cache_dir}")

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")
    print(f"[TRAINING] Number of epochs: {num_epochs}")
    print(f"[TRAINING] Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"[TRAINING] Learning rate: {learning_rate}")
    print(f"[TRAINING] Image size: {image_size}")

    # Initialize mixed precision scaler with proper device type
    scaler = GradScaler(enabled=mixed_precision)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    print("=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)

    # DETR model configuration optimized for single-class detection
    custom_config = {
        "num_queries": 4,  # Reduced from 5 to 4 - one tool per image
        "bbox_cost": 5,  # Increased from 2 to 5 - more weight on bbox accuracy
        "class_cost": 1,  # Reduced from 2 to 1 - less important for single class
        "giou_cost": 4,  # Increased from 2 to 4 - better localization
        "giou_loss_coefficient": 4,  # Increased from 2 to 4
        "bbox_loss_coefficient": 5,  # Increased from 2 to 5
        "eos_coefficient": 0.1,  # Decreased from 0.8 to 0.1 for better recall
        "encoder_layers": 3,  # Reduced from 4 for faster training
        "decoder_layers": 3,  # Reduced from 4 for faster training
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

    # Apply torch.compile if available (safely with Triton checking)
    if torch_compile:
        try:
            # First check if Triton is available
            import importlib.util
            triton_available = importlib.util.find_spec("triton") is not None

            if triton_available:
                print("[MODEL] Triton library found, applying torch.compile()...")
                model = torch.compile(model)
                print("[MODEL] Successfully applied torch.compile.")
            else:
                print("[MODEL] Triton library not found, skipping torch.compile()")
                print("[MODEL] To use torch.compile(), install with: pip install triton")
                torch_compile = False
        except Exception as e:
            print(f"[MODEL] Error during torch.compile(): {e}")
            print("[MODEL] Continuing without compilation")
            torch_compile = False

    print("=" * 80)
    print("DATASET LOADING")
    print("=" * 80)

    # Load full dataset with caching
    full_dataset = SurgicalToolDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        processor=processor,
        image_size=image_size,
        augment=False,  # Don't apply augmentations to the full dataset for initial loading
        augment_strength=augmentation_strength,
        cache_dir=cache_dir
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

    # Apply augmentation only to training dataset
    # First create the splits
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(selected_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create a new augmented dataset for training
    train_dataset = SurgicalToolDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        processor=processor,
        image_size=image_size,
        augment=apply_augmentations,  # Apply augmentations only to training set
        augment_strength=augmentation_strength,
        cache_dir=cache_dir
    )

    # Create a specialized subset using only training indices
    train_dataset = Subset(train_dataset, train_indices.indices)

    # Create a non-augmented dataset for validation
    val_dataset = Subset(selected_dataset, val_indices.indices)

    print(f"[AUGMENTATION] Applied {augmentation_strength} augmentations to training set: {apply_augmentations}")
    print(f"[DATASET] Training set size: {len(train_dataset)}")
    print(f"[DATASET] Validation set size: {len(val_dataset)}")

    # Create data loaders - use num_workers=0 for Windows to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs

    # Set up optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999),  # Default betas
        eps=1e-8  # Default epsilon
    )

    # Use CosineAnnealingWarmRestarts for faster convergence
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # Reset after 20 epochs
        T_mult=1,  # Keep same cycle length
        eta_min=learning_rate / 100  # Min LR
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
            latest_checkpoint, model, optimizer, scheduler, scaler, device
        )
        if loaded_model is not None:
            model = loaded_model
            print(f"Starting training from epoch {start_epoch}")
            # Initialize epoch in case the loop doesn't run
            epoch = start_epoch - 1
        else:
            start_epoch = 0
            best_val_loss = float('inf')
            epoch = -1  # Initialize epoch
            print("Starting training from beginning (checkpoint loading failed)")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        epoch = -1  # Initialize epoch
        print("[CHECKPOINT] No checkpoints found, starting training from scratch.")

    # Check if training is already complete
    if start_epoch >= num_epochs:
        print(f"Training already complete (start_epoch {start_epoch} >= num_epochs {num_epochs})")
        print("Use a larger num_epochs value to continue training.")
        # Use the last completed epoch number
        epoch = start_epoch - 1
    else:
        print("=" * 80)
        print(f"TRAINING STARTING FROM EPOCH {start_epoch + 1}")
        print("=" * 80)

        # Early stopping tracker
        patience_counter = 0

        # Training loop
        try:
            for current_epoch in range(start_epoch, num_epochs):
                # Store current epoch in outer scope
                epoch = current_epoch

                epoch_start_time = time.time()
                print(f"Starting epoch {epoch + 1}/{num_epochs}...")

                # Training phase
                avg_train_loss = train_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    scaler=scaler,
                    writer=writer,
                    log_every=10,
                    grad_accum_steps=gradient_accumulation_steps,
                    empty_cache_freq=empty_cache_freq
                )

                # Update scheduler
                if isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step()

                # Perform validation only every validation_interval epochs to save time
                # But always validate on the first and last epochs
                if (epoch + 1) % validation_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
                    avg_val_loss = validate_epoch(model, val_loader, device, epoch, writer)
                    print(f"Validation Loss: {avg_val_loss:.4f}")

                    # Save best model based on validation loss
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                        model.save_pretrained(best_model_dir)
                        processor.save_pretrained(best_model_dir)
                        print(f"Best model saved to: {best_model_dir}")
                    else:
                        # Increment patience counter if no improvement
                        patience_counter += 1
                        print(f"No improvement over best validation loss: {best_val_loss:.4f}. "
                              f"Patience: {patience_counter}/{early_stopping_patience}")
                else:
                    print(
                        f"Skipping validation for epoch {epoch + 1} (will validate every {validation_interval} epochs)")

                # Save checkpoint at regular intervals
                if (epoch + 1) % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, scaler, checkpoint_dir)

                # Report overall epoch stats
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. "
                      f"Training Loss: {avg_train_loss:.4f}")

                # Early stopping based on patience
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                    break

                # Early stopping if validation loss reaches threshold
                if best_val_loss <= validation_loss_threshold:
                    print(f"Validation loss has reached threshold of {validation_loss_threshold}. Stopping training.")
                    break

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving current model and checkpoint...")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, scaler, checkpoint_dir)

            # Save interrupted model
            interrupted_model_dir = os.path.join(output_dir, "detr_tool_tracking_model_interrupted")
            os.makedirs(interrupted_model_dir, exist_ok=True)
            model.save_pretrained(interrupted_model_dir)
            processor.save_pretrained(interrupted_model_dir)
            print(f"Interrupted model saved to: {interrupted_model_dir}")

    # Save final model
    print("=" * 80)
    print("TRAINING FINISHED - SAVING FINAL MODEL")
    print("=" * 80)

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, scaler, checkpoint_dir)

    # Save final model
    final_model_dir = os.path.join(output_dir, "detr_tool_tracking_model_final")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    print("Model and processor saved successfully.")
    print(f"Final model saved to: {final_model_dir}")

    # Save training configuration
    config = {
        "training_range": {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "actual_images": len(selected_dataset)
        },
        "training_params": {
            "num_epochs": num_epochs,
            "completed_epochs": epoch + 1,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "mixed_precision": mixed_precision,
            "image_size": image_size
        },
        "augmentation": {
            "enabled": apply_augmentations,
            "strength": augmentation_strength
        },
        "model_config": custom_config,
        "best_validation_loss": float(best_val_loss),
        "early_stopped": patience_counter >= early_stopping_patience if 'patience_counter' in locals() else False
    }

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Close TensorBoard writer
    writer.close()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    # Enable multiprocessing support for Windows
    import multiprocessing

    multiprocessing.freeze_support()
    main()