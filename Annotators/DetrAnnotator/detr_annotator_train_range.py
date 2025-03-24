import glob
import json
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig


# Configuration class to centralize all parameters
class Config:
    # Data paths
    TRAIN_IMAGES_DIR = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train"
    TRAIN_ANNOTATIONS_FILE = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Detr/coco_annotations_from_yolo_dataset_20250218.json"
    OUTPUT_DIR = "./training_ranged/output_ranged"
    CHECKPOINT_DIR = "./training_ranged/checkpoints_ranged"
    BEST_MODEL_DIR = "./training_ranged/best_model"
    TEST_OUTPUT_DIR = "./training_ranged/model_test_on_training"

    # Data selection
    START_IDX = 0
    END_IDX = 950

    # Training parameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = (640, 640)
    NUM_QUERIES = 10
    USE_PRETRAINED = True

    # Loss coefficients
    BBOX_COST = 5
    CLASS_COST = 2
    GIOU_COST = 4
    GIOU_LOSS_COEFFICIENT = 4
    BBOX_LOSS_COEFFICIENT = 5
    EOS_COEFFICIENT = 0.5

    # Early stopping parameters
    PATIENCE = 5

    # Evaluation parameters
    IOU_THRESHOLD = 0.5
    TEST_SAMPLES_PER_EPOCH = 20
    TEST_CONFIDENCE_THRESHOLD = 0.2
    VAL_SPLIT = 0.15

    # Bounding box correction - CRITICAL FIX FOR ANNOTATION OFFSET
    BBOX_OFFSET_X = -10  # Adjust based on observed offset (negative = move left)
    BBOX_OFFSET_Y = -10  # Adjust based on observed offset (negative = move up)


class SurgicalToolDataset(torch.utils.data.Dataset):
    """Dataset for surgical tool detection with DETR."""

    def __init__(self, images_dir, annotations_file, processor, image_size=(800, 800),
                 augment=False, curriculum_difficulty=None,
                 bbox_offset_x=0, bbox_offset_y=0):
        print("Initializing dataset...")
        self.images_dir = images_dir
        self.processor = processor
        self.image_size = image_size
        self.augment = augment
        self.curriculum_difficulty = curriculum_difficulty
        self.bbox_offset_x = bbox_offset_x
        self.bbox_offset_y = bbox_offset_y

        if self.augment:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
        else:
            self.augmentations = None

        # Load annotations from JSON file with UTF-8 encoding to fix character issues
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"Successfully loaded annotations from {annotations_file}")
        except UnicodeDecodeError:
            # If UTF-8 fails, try with Latin-1 which rarely fails
            try:
                with open(annotations_file, 'r', encoding='latin-1') as f:
                    self.annotations = json.load(f)
                print(f"Successfully loaded annotations using latin-1 encoding")
            except Exception as e:
                print(f"Error loading annotations: {e}")
                raise

        # Set category_id to 0 for all annotations (single-class model)
        for ann in self.annotations['annotations']:
            ann['category_id'] = 0

            # Apply annotation offset correction
            if self.bbox_offset_x != 0 or self.bbox_offset_y != 0:
                ann['bbox'][0] += self.bbox_offset_x
                ann['bbox'][1] += self.bbox_offset_y
                ann['bbox'][0] = max(0, ann['bbox'][0])
                ann['bbox'][1] = max(0, ann['bbox'][1])

        # Build dataset mappings
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

        # Sort valid images by filename for consistent ordering
        self.valid_images.sort(key=lambda x: x['file_name'])

        # Apply curriculum filtering if specified
        if self.curriculum_difficulty:
            self._apply_curriculum_filtering()

        # Create annotation mapping
        valid_annotations = 0
        for ann in self.annotations['annotations']:
            if ann['image_id'] in self.id_to_filename:
                if ann['image_id'] not in self.id_to_annotations:
                    self.id_to_annotations[ann['image_id']] = []
                self.id_to_annotations[ann['image_id']].append(ann)
                valid_annotations += 1

        # Calculate statistics about the dataset
        self.calculate_dataset_statistics()

    def calculate_dataset_statistics(self):
        """Calculate and display dataset statistics for better understanding."""
        imgs_with_annotations = sum(1 for img_id in self.id_to_filename if img_id in self.id_to_annotations)
        imgs_with_multiple_annotations = sum(1 for anns in self.id_to_annotations.values() if len(anns) > 1)

        print(f"Loaded {len(self.valid_images)} valid images out of {len(self.annotations['images'])} in annotations")
        print(f"Dataset has {len(self.id_to_annotations)} images with annotations")
        print(f"Images with multiple annotations: {imgs_with_multiple_annotations}")
        print(
            f"Dataset has {sum(len(anns) for anns in self.id_to_annotations.values())} valid annotations out of {len(self.annotations['annotations'])} total")

        # Calculate average annotations per image
        if self.id_to_annotations:
            avg_anns_per_img = sum(len(anns) for anns in self.id_to_annotations.values()) / len(self.id_to_annotations)
            print(f"Average annotations per image: {avg_anns_per_img:.2f}")

    def _apply_curriculum_filtering(self):
        """Filter images based on curriculum difficulty level."""
        print(f"Applying curriculum filtering for difficulty level: {self.curriculum_difficulty}")

        # Simple curriculum strategy based on image index
        total_images = len(self.valid_images)

        if self.curriculum_difficulty == 'easy':
            # Use first 1/3 of images (assumed easier)
            self.valid_images = self.valid_images[:total_images // 3]
        elif self.curriculum_difficulty == 'medium':
            # Use middle 1/3 of images
            start_idx = total_images // 3
            end_idx = 2 * total_images // 3
            self.valid_images = self.valid_images[start_idx:end_idx]
        elif self.curriculum_difficulty == 'hard':
            # Use last 1/3 of images (assumed harder)
            self.valid_images = self.valid_images[2 * total_images // 3:]
        else:
            # If invalid difficulty level, keep all images
            pass

        print(f"After curriculum filtering: {len(self.valid_images)} images")

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
            # FIXED: Using proper size parameter format to avoid max_size deprecation warning
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
        return {"pixel_values": pixel_values, "labels": target,
                "image_path": image_path, "image_id": image_id}


def collate_fn(batch):
    """Custom collate function to handle variable-sized annotations."""
    try:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_paths": image_paths,
            "image_ids": image_ids
        }
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise e


def check_data_integrity(dataset):
    """Verify dataset integrity."""
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

        # Display object count distribution
        counts = Counter(objects_per_image.values())
        print("Objects per image distribution:")
        for count, num_images in sorted(counts.items()):
            print(f"  {count} object(s): {num_images} images ({num_images / len(objects_per_image) * 100:.1f}%)")

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
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        # Print per-device memory
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
    else:
        print("GPU not available, running on CPU")


def train_epoch(model, data_loader, optimizer, device, epoch, writer=None):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    cls_loss = 0
    bbox_loss = 0
    giou_loss = 0

    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Extract individual loss components if available
        loss_dict = getattr(outputs, 'loss_dict', {})
        if loss_dict:
            cls_loss += loss_dict.get('loss_ce', torch.tensor(0.0)).item()
            bbox_loss += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
            giou_loss += loss_dict.get('loss_giou', torch.tensor(0.0)).item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        # TensorBoard logging
        if writer is not None:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), global_step)

            # Log individual loss components
            if hasattr(outputs, 'loss_dict'):
                for loss_name, loss_value in outputs.loss_dict.items():
                    writer.add_scalar(f"Loss/{loss_name}", loss_value.item(), global_step)

        # Print detailed loss periodically
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

    # Log average losses for the epoch
    if writer is not None and len(data_loader) > 0:
        writer.add_scalar("Loss/train_epoch", total_loss / len(data_loader), epoch)
        if loss_dict:
            writer.add_scalar("Loss/cls_epoch", cls_loss / len(data_loader), epoch)
            writer.add_scalar("Loss/bbox_epoch", bbox_loss / len(data_loader), epoch)
            writer.add_scalar("Loss/giou_epoch", giou_loss / len(data_loader), epoch)

    print("GPU memory usage after epoch:")
    print_gpu_memory()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate_model(model, data_loader, device, processor, epoch=None, writer=None,
                   visualize_predictions=False, output_dir=None, iou_threshold=0.5,
                   confidence_threshold=0.5):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    # Metrics tracking
    all_targets = []  # True labels (0 or 1 for presence)
    all_predictions = []  # Predicted labels (0 or 1 for presence)

    # More detailed metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # IoU tracking
    all_ious = []

    # For visualization
    if visualize_predictions and output_dir:
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            image_paths = batch["image_paths"]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            # Visualize predictions (every 5 batches)
            if visualize_predictions and output_dir and batch_idx % 5 == 0:
                # Get predictions
                pred_logits = outputs.logits
                pred_boxes = outputs.pred_boxes

                # Process batch
                for i in range(len(pixel_values)):
                    # Get image
                    image_path = image_paths[i]
                    image = Image.open(image_path).convert("RGB")

                    # Get predictions for this image
                    scores = pred_logits[i].softmax(-1)[..., 0]
                    boxes = pred_boxes[i]

                    # Keep only predictions with high enough scores
                    keep = scores > 0.1  # Lower threshold to see more predictions (0.1 instead of 0.5)
                    boxes = boxes[keep]
                    scores = scores[keep]

                    # Convert to xyxy format
                    h, w = image.size
                    boxes_xyxy = []
                    for box in boxes:
                        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
                        cx, cy, bw, bh = box.cpu().numpy()
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        boxes_xyxy.append([x1, y1, x2, y2])

                    # Draw predictions
                    draw = ImageDraw.Draw(image)
                    for box, score in zip(boxes_xyxy, scores):
                        x1, y1, x2, y2 = box
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        draw.text((x1, y1), f"Score: {score.item():.2f}", fill="red")

                    # Save image
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_{filename}")
                    image.save(output_path)

            # Log to TensorBoard
            if writer is not None and epoch is not None:
                global_step = epoch * len(data_loader) + batch_idx
                writer.add_scalar("Loss/validation", loss.item(), global_step)

    # Calculate average metrics
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0

    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate average IoU
    avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0

    # Log metrics to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar("Metrics/Precision", precision, epoch)
        writer.add_scalar("Metrics/Recall", recall, epoch)
        writer.add_scalar("Metrics/F1", f1_score, epoch)
        writer.add_scalar("Metrics/AvgIoU", avg_iou, epoch)

    # Print detailed metrics
    print(f"Validation Metrics:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}")

    return avg_loss, precision, recall, f1_score


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_dir):
    """Save training state checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None, device=None):
    """
    Load training state from checkpoint with robust error handling.

    Modified to handle parameter size mismatches by preserving original model.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Checkpoint {checkpoint_path} does not exist")
        return model, 0, {}  # Return original model, not None

    print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device if device else 'cpu')

        # Try to load model weights
        try:
            # CRITICAL FIX: Handle parameter size mismatches
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']

            # Filter out incompatible layers
            compatible_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}

            # Print statistics about loaded parameters
            print(f"[CHECKPOINT] Loading {len(compatible_dict)}/{len(model_dict)} compatible layers")

            # Only update if we have compatible layers
            if compatible_dict:
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print("[CHECKPOINT] Loaded compatible parameters")
            else:
                print("[CHECKPOINT] No compatible parameters found, using fresh model")
                # Return original model with no parameter updates
                return model, 0, {}

        except RuntimeError as e:
            print(f"[CHECKPOINT] Error loading model state: {e}")
            print("[CHECKPOINT] Continuing with original model")
            # Return original model instead of None
            return model, 0, {}

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Get saved metrics
        metrics = checkpoint.get('metrics', {})

        # Get epoch number
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch

        print(f"[CHECKPOINT] Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"[CHECKPOINT] Best metrics so far: {metrics}")

        return model, start_epoch, metrics

    except Exception as e:
        print(f"[CHECKPOINT] Error loading checkpoint: {e}")
        # Return original model instead of None
        return model, 0, {}


def select_image_range(dataset, start_idx, end_idx):
    """Select a subset of images from the dataset based on index range."""
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
    print(f"Percentage of images with annotations: {len(valid_indices) / len(selected_indices) * 100:.2f}%")

    # Create subset with selected indices
    return Subset(dataset, valid_indices)


def save_images_with_annotations(dataset, indices, output_dir, max_images=100):
    """Save images with drawn bounding boxes for visual verification."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images with annotations to {output_dir}...")

    # Limit number of images to process
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

        # Draw bounding boxes on image
        draw = ImageDraw.Draw(image)

        for ann in annotations:
            # COCO format is [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
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





def model_on_training_data_testing(model, processor, dataset, full_dataset, epoch, device,
                                   num_samples=10, confidence_threshold=0.1,
                                   output_dir="./model_test_on_training", phase="validation"):
    """
    Tests model on dataset samples and saves visualizations with bounding boxes.

    Args:
        model: DETR model for testing
        processor: Image processor for the model
        dataset: Dataset object to test on
        full_dataset: Complete dataset (for accessing original annotations)
        epoch: Current epoch number
        device: Device to run inference on (cuda or cpu)
        num_samples: Number of samples to test
        confidence_threshold: Detection confidence threshold
        output_dir: Directory for saving visualization results
        phase: Testing phase name (e.g., "train", "validation", "test")
    """
    # Create phase and epoch specific directory
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Select random samples from dataset
    dataset_indices = list(range(len(dataset)))
    indices = np.random.choice(dataset_indices, size=min(num_samples, len(dataset)), replace=False)

    print(f"Testing model after epoch {epoch} on {len(indices)} {phase} samples...")

    # Initialize results collection
    results = []

    with torch.no_grad():  # Disable gradient calculation for faster inference
        for i, idx in enumerate(tqdm(indices, desc=f"Testing {phase} data (epoch {epoch})")):
            try:
                # Get correct index in original dataset
                if hasattr(dataset, 'indices'):  # If dataset is a Subset
                    actual_idx = dataset.indices[idx]
                else:
                    actual_idx = idx

                # Get image information
                image_info = full_dataset.valid_images[actual_idx]
                image_id = image_info['id']
                image_filename = full_dataset.id_to_filename[image_id]
                image_path = os.path.join(full_dataset.images_dir, image_filename)

                # Get ground truth annotations
                original_annotations = full_dataset.id_to_annotations.get(image_id, [])

                # Load original image
                original_image = Image.open(image_path).convert("RGB")

                # Run model inference
                inputs = processor(
                    images=original_image,
                    return_tensors="pt",
                    size={'shortest_edge': 640, 'longest_edge': 640}  # Use consistent size format
                ).to(device)

                outputs = model(**inputs)

                # Process model predictions
                target_sizes = torch.tensor([original_image.size[::-1]]).to(device)
                results_processed = processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=confidence_threshold
                )[0]

                # Create visualization
                result_image = original_image.copy()
                draw = ImageDraw.Draw(result_image)

                # Try to use a better font if available
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()

                # Draw ground truth boxes (green)
                for ann in original_annotations:
                    # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]
                    x, y, w, h = ann['bbox']
                    x1, y1, x2, y2 = x, y, x + w, y + h

                    # Draw ground truth box
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    draw.text((x1, max(0, y1 - 20)), "Ground Truth", fill="green", font=font)

                # Draw predicted boxes (red)
                for score, label, box in zip(results_processed["scores"],
                                             results_processed["labels"],
                                             results_processed["boxes"]):
                    score_val = score.item()
                    label_val = label.item()
                    x1, y1, x2, y2 = box.cpu().numpy()

                    # Draw prediction box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    # Add label with confidence score
                    label_text = f"{model.config.id2label[label_val]}: {score_val:.2f}"
                    draw.text((x1, max(0, y1 - 20)), label_text, fill="red", font=font)

                # Add image information
                draw.text((10, 10), f"Epoch: {epoch} ({phase})", fill="blue", font=font)
                draw.text((10, 30), f"Image ID: {image_id}", fill="blue", font=font)

                # Create descriptive filename that sorts well
                base_filename = os.path.basename(image_path)
                result_filename = f"{phase}_{i:02d}_{base_filename}"
                result_path = os.path.join(epoch_dir, result_filename)

                # Save visualization
                result_image.save(result_path)

                # Record metrics for this sample
                pred_count = len(results_processed["scores"])
                max_score = float(results_processed["scores"].max()) if pred_count > 0 else 0

                results.append({
                    "epoch": epoch,
                    "phase": phase,
                    "sample_idx": i,
                    "dataset_idx": actual_idx,
                    "image_id": image_id,
                    "image_path": str(image_path),  # Convert Path to string for JSON serialization
                    "ground_truth_count": len(original_annotations),
                    "prediction_count": pred_count,
                    "max_confidence": max_score
                })

            except Exception as e:
                print(f"Error processing sample {i} (dataset index {idx}): {str(e)}")
                continue

    # Save metrics summary for this epoch and phase
    metrics_path = os.path.join(epoch_dir, f"{phase}_metrics.json")
    try:
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving metrics: {str(e)}")

    # Print summary statistics
    print(f"\nResults summary for epoch {epoch} ({phase} data):")
    if results:
        avg_score = np.mean([r['max_confidence'] for r in results])
        no_detect = sum(1 for r in results if r['prediction_count'] == 0)
        correct_detect = sum(1 for r in results
                             if r['ground_truth_count'] == r['prediction_count'] and r['prediction_count'] > 0)

        print(f"Average max confidence: {avg_score:.4f}")
        print(f"Samples with no detections: {no_detect}/{len(results)} ({no_detect / len(results) * 100:.1f}%)")
        print(f"Samples with correct detection count: {correct_detect}/{len(results)} "
              f"({correct_detect / len(results) * 100:.1f}%)")
    else:
        print("No valid results to display")

    # Return model to training mode
    model.train()

    # Return results dictionary with metrics that can be used externally
    return {
        "metrics": {
            "avg_confidence": float(avg_score) if results else 0,
            "no_detection_count": no_detect if results else 0,
            "no_detection_percent": float(no_detect / len(results) * 100) if results else 0,
            "correct_detection_count": correct_detect if results else 0,
            "correct_detection_percent": float(correct_detect / len(results) * 100) if results else 0,
            "total_samples": len(results)
        }
    }

def train_model_with_curriculum(config):
    """Train model using curriculum learning strategy."""
    # Set up directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(config.TEST_OUTPUT_DIR, exist_ok=True)

    visualization_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT_DIR, 'tensorboard'))

    # Define curriculum stages
    curriculum_stages = [
        {"name": "easy", "epochs": 5},
        {"name": "medium", "epochs": 5},
        {"name": "hard", "epochs": 5},
        {"name": "full", "epochs": 5}
    ]

    # DETR model configuration
    model_config = {
        "num_queries": config.NUM_QUERIES,
        "bbox_cost": config.BBOX_COST,
        "class_cost": config.CLASS_COST,
        "giou_cost": config.GIOU_COST,
        "giou_loss_coefficient": config.GIOU_LOSS_COEFFICIENT,
        "bbox_loss_coefficient": config.BBOX_LOSS_COEFFICIENT,
        "eos_coefficient": config.EOS_COEFFICIENT,
    }

    # Initialize or load model
    print("\nInitializing model...")
    if os.path.exists(config.BEST_MODEL_DIR) and os.listdir(config.BEST_MODEL_DIR):
        print(f"Loading model from {config.BEST_MODEL_DIR}")

        # Load DetrConfig and update with our parameters
        try:
            detr_config = DetrConfig.from_pretrained(config.BEST_MODEL_DIR)

            # Update configuration with our settings
            for key, value in model_config.items():
                if hasattr(detr_config, key):
                    print(f"Updating parameter {key}: {getattr(detr_config, key)} -> {value}")
                    setattr(detr_config, key, value)

            # Load model and processor
            model = DetrForObjectDetection.from_pretrained(
                config.BEST_MODEL_DIR,
                config=detr_config,
                ignore_mismatched_sizes=True  # Allow parameter size mismatches
            )
            processor = DetrImageProcessor.from_pretrained(config.BEST_MODEL_DIR)
        except Exception as e:
            print(f"Error loading saved model: {e}")
            model = None
    else:
        model = None

    # If model loading failed or no saved model exists, initialize a new one
    if model is None:
        print("Initializing new model...")
        if config.USE_PRETRAINED:
            # Load pretrained DETR model
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=1,
                num_queries=config.NUM_QUERIES,
                ignore_mismatched_sizes=True
            )

            # Update model configuration
            for key, value in model_config.items():
                if hasattr(model.config, key):
                    print(f"Setting parameter {key} to {value}")
                    setattr(model.config, key, value)

            # Set label mappings
            model.config.id2label = {0: "surgical_tool"}
            model.config.label2id = {"surgical_tool": 0}
            model.config.num_labels = 1

            # Initialize processor
            processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                # FIXED: Using proper size parameter format to avoid max_size deprecation warning
                size={'shortest_edge': config.IMAGE_SIZE[0], 'longest_edge': config.IMAGE_SIZE[1]}
            )
        else:
            # Initialize model from scratch
            detr_config = DetrConfig(num_labels=1, **model_config)
            model = DetrForObjectDetection(detr_config)

            # Set label mappings
            model.config.id2label = {0: "surgical_tool"}
            model.config.label2id = {"surgical_tool": 0}
            model.config.num_labels = 1

            # Initialize processor
            processor = DetrImageProcessor(
                size={'shortest_edge': config.IMAGE_SIZE[0], 'longest_edge': config.IMAGE_SIZE[1]}
            )

    # Load full dataset once (for validation and final evaluation)
    print("\nLoading full dataset...")
    full_dataset = SurgicalToolDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        annotations_file=config.TRAIN_ANNOTATIONS_FILE,
        processor=processor,
        image_size=config.IMAGE_SIZE,
        augment=False,  # No augmentation for full dataset
        # CRITICAL FIX: Apply annotation offset correction
        bbox_offset_x=config.BBOX_OFFSET_X,
        bbox_offset_y=config.BBOX_OFFSET_Y
    )

    # Verify dataset integrity
    if not check_data_integrity(full_dataset):
        print("Dataset has critical issues. Aborting training.")
        return

    # Visualize sample annotations
    vis_samples_dir = os.path.join(config.OUTPUT_DIR, "sample_annotations")
    sample_indices = list(range(0, 100, 10))  # Select 10 evenly spaced samples
    save_images_with_annotations(full_dataset, sample_indices, vis_samples_dir)

    # Select data range for training
    print(f"\nSelecting image range {config.START_IDX} to {config.END_IDX}...")
    selected_dataset = select_image_range(full_dataset, config.START_IDX, config.END_IDX)

    # Initial train/val split from selected dataset
    dataset_size = len(selected_dataset)
    val_size = int(config.VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    # Set random seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(selected_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=3,
        verbose=True
    )

    # Move model to device
    model.to(device)

    # Set up tracking variables
    best_val_metrics = {
        'loss': float('inf'),
        'precision': 0,
        'recall': 0,
        'f1_score': 0
    }
    patience_counter = 0
    start_epoch = 0
    total_epochs = sum(stage['epochs'] for stage in curriculum_stages)

    # Check for existing checkpoint
    latest_checkpoint = find_latest_checkpoint(config.CHECKPOINT_DIR)
    if latest_checkpoint:
        model, start_epoch, saved_metrics = load_checkpoint(
            latest_checkpoint, model, optimizer, scheduler, device
        )

        if saved_metrics:
            best_val_metrics.update(saved_metrics)
            print(f"Resuming from epoch {start_epoch} with metrics: {best_val_metrics}")

    # Curriculum learning loop
    epoch_offset = 0
    global_epoch = start_epoch

    # Determine which curriculum stage to start from based on resumed epoch
    current_stage_idx = 0
    remaining_epochs = global_epoch
    while current_stage_idx < len(curriculum_stages) and remaining_epochs >= curriculum_stages[current_stage_idx][
        'epochs']:
        remaining_epochs -= curriculum_stages[current_stage_idx]['epochs']
        epoch_offset += curriculum_stages[current_stage_idx]['epochs']
        current_stage_idx += 1

    # Skip completed stages
    curriculum_stages = curriculum_stages[current_stage_idx:]
    stage_start_epoch = global_epoch - epoch_offset if global_epoch > epoch_offset else 0

    print(f"\nStarting curriculum training from stage {current_stage_idx}")

    try:
        for stage_idx, stage in enumerate(curriculum_stages):
            stage_name = stage['name']
            stage_epochs = stage['epochs']

            print(f"\n{'=' * 80}")
            print(
                f"CURRICULUM STAGE {stage_idx + current_stage_idx}: {stage_name.upper()} (EPOCHS {epoch_offset}-{epoch_offset + stage_epochs - 1})")
            print(f"{'=' * 80}")

            # Create curriculum dataset for this stage
            if stage_name == 'full':
                # For the final stage, use all selected data
                curr_train_dataset = SurgicalToolDataset(
                    images_dir=config.TRAIN_IMAGES_DIR,
                    annotations_file=config.TRAIN_ANNOTATIONS_FILE,
                    processor=processor,
                    image_size=config.IMAGE_SIZE,
                    augment=True,  # Enable augmentation for training
                    # CRITICAL FIX: Apply annotation offset correction
                    bbox_offset_x=config.BBOX_OFFSET_X,
                    bbox_offset_y=config.BBOX_OFFSET_Y
                )
                curr_train_dataset = select_image_range(curr_train_dataset, config.START_IDX, config.END_IDX)

                # Exclude validation samples
                train_indices = [i for i in range(len(curr_train_dataset))
                                 if
                                 i not in [val_dataset.indices[j] - config.START_IDX for j in range(len(val_dataset))]]
                curr_train_dataset = Subset(curr_train_dataset, train_indices)
            else:
                # For earlier stages, filter by difficulty
                curr_train_dataset = SurgicalToolDataset(
                    images_dir=config.TRAIN_IMAGES_DIR,
                    annotations_file=config.TRAIN_ANNOTATIONS_FILE,
                    processor=processor,
                    image_size=config.IMAGE_SIZE,
                    augment=True,  # Enable augmentation for training
                    curriculum_difficulty=stage_name,  # Apply curriculum filtering
                    # CRITICAL FIX: Apply annotation offset correction
                    bbox_offset_x=config.BBOX_OFFSET_X,
                    bbox_offset_y=config.BBOX_OFFSET_Y
                )
                curr_train_dataset = select_image_range(curr_train_dataset, config.START_IDX, config.END_IDX)

            print(f"Stage '{stage_name}' dataset size: {len(curr_train_dataset)}")

            # Create data loader for this stage
            train_loader = DataLoader(
                curr_train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )

            # Training loop for this stage
            for stage_epoch in range(stage_start_epoch, stage_epochs):
                global_epoch = epoch_offset + stage_epoch
                print(f"\nStarting epoch {global_epoch + 1}/{total_epochs}...")

                # Train for one epoch
                train_loss = train_epoch(model, train_loader, optimizer, device, global_epoch, writer)
                print(f"Epoch {global_epoch + 1} finished. Training Loss: {train_loss:.4f}")

                # Evaluate model
                val_loss, precision, recall, f1_score = evaluate_model(
                    model, val_loader, device, processor, global_epoch, writer,
                    visualize_predictions=False,  # Disable detailed visualization for now
                    output_dir=visualization_dir,
                    iou_threshold=config.IOU_THRESHOLD,
                    confidence_threshold=config.TEST_CONFIDENCE_THRESHOLD
                )
                print(f"Validation Loss: {val_loss:.4f}, F1: {f1_score:.4f}")

                # Test model on training data sample
                test_results = model_on_training_data_testing(
                    model=model,
                    processor=processor,
                    dataset=curr_train_dataset,
                    full_dataset=full_dataset,
                    epoch=global_epoch,
                    device=device,
                    num_samples=config.TEST_SAMPLES_PER_EPOCH,
                    confidence_threshold=config.TEST_CONFIDENCE_THRESHOLD,
                    output_dir=os.path.join(config.TEST_OUTPUT_DIR, f"epoch_{global_epoch}"),
                    phase="train"
                )

                # Update learning rate based on validation loss
                scheduler.step(val_loss)

                # Update best metrics and save model if improved
                metrics_improved = False

                if val_loss < best_val_metrics['loss']:
                    improvement = best_val_metrics['loss'] - val_loss
                    best_val_metrics['loss'] = val_loss
                    metrics_improved = True
                    print(f"New best validation loss: {val_loss:.4f} (improved by {improvement:.4f})")

                if f1_score > best_val_metrics['f1_score']:
                    improvement = f1_score - best_val_metrics['f1_score']
                    best_val_metrics['f1_score'] = f1_score
                    best_val_metrics['precision'] = precision
                    best_val_metrics['recall'] = recall
                    metrics_improved = True
                    print(f"New best F1 score: {f1_score:.4f} (improved by {improvement:.4f})")

                # Save checkpoint
                save_checkpoint(model, optimizer, scheduler, global_epoch, best_val_metrics, config.CHECKPOINT_DIR)

                # Save best model if metrics improved
                if metrics_improved:
                    patience_counter = 0
                    model.save_pretrained(os.path.join(config.BEST_MODEL_DIR, f"epoch_{global_epoch + 1}"))
                    processor.save_pretrained(os.path.join(config.BEST_MODEL_DIR, f"epoch_{global_epoch + 1}"))

                    # Save best overall model
                    model.save_pretrained(config.BEST_MODEL_DIR)
                    processor.save_pretrained(config.BEST_MODEL_DIR)
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs. Best metrics: {best_val_metrics}")

                    # Early stopping check
                    if patience_counter >= config.PATIENCE:
                        print(f"Early stopping after {patience_counter} epochs without improvement.")
                        break

            # Reset stage start epoch for next stage
            stage_start_epoch = 0
            # Update epoch offset for next stage
            epoch_offset += stage_epochs

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        save_checkpoint(model, optimizer, scheduler, global_epoch, best_val_metrics, config.CHECKPOINT_DIR)
        model.save_pretrained(os.path.join(config.OUTPUT_DIR, "interrupted_model"))
        processor.save_pretrained(os.path.join(config.OUTPUT_DIR, "interrupted_model"))

    # Save final model
    print("\nTraining complete. Saving final model...")
    save_checkpoint(model, optimizer, scheduler, global_epoch, best_val_metrics, config.CHECKPOINT_DIR)
    model.save_pretrained(os.path.join(config.OUTPUT_DIR, "final_model"))
    processor.save_pretrained(os.path.join(config.OUTPUT_DIR, "final_model"))

    # Final evaluation on test set
    print("\nPerforming final evaluation on test set...")

    # Get test dataset (10% of selected data not in training or validation)
    test_indices = [i for i in range(config.START_IDX, config.END_IDX)
                    if i not in train_dataset.indices and i not in val_dataset.indices]
    if len(test_indices) > 100:
        test_indices = test_indices[:100]  # Limit to 100 samples

    test_dataset = Subset(full_dataset, test_indices)

    # Evaluate on test dataset
    test_results = model_on_training_data_testing(
        model=model,
        processor=processor,
        dataset=test_dataset,
        full_dataset=full_dataset,
        epoch=global_epoch,
        device=device,
        num_samples=len(test_dataset),
        confidence_threshold=config.TEST_CONFIDENCE_THRESHOLD,
        output_dir=os.path.join(config.OUTPUT_DIR, "final_test_results"),
        phase="test"
    )

    # Save test results
    with open(os.path.join(config.OUTPUT_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(test_results["metrics"], f, indent=2)

    # Print final summary
    print("\nTraining and evaluation complete!")
    print(f"Best validation metrics: {best_val_metrics}")
    print(f"Final test metrics: {test_results['metrics']}")
    print(f"Model saved to: {config.BEST_MODEL_DIR}")
    print(f"Final model saved to: {os.path.join(config.OUTPUT_DIR, 'final_model')}")

    # Close TensorBoard writer
    writer.close()

    return model, processor, best_val_metrics


def main():
    """Main function to run training."""
    print("=" * 80)
    print("DETR SURGICAL TOOL DETECTION - ENHANCED TRAINING")
    print("=" * 80)

    # Load configuration
    config = Config()
    print("Configuration:")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Run training with curriculum learning
    model, processor, metrics = train_model_with_curriculum(config)

    return model, processor, metrics


if __name__ == "__main__":
    main()