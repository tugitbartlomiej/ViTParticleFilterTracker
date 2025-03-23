import glob
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig


# Configuration class to centralize all parameters
class Config:
    # Data paths
    TRAIN_IMAGES_DIR = "/mnt/evafs/faculty/home/bpiotrowski/datasets/yolo_dataset_20250218/images/train"
    TRAIN_ANNOTATIONS_FILE = "/mnt/evafs/faculty/home/bpiotrowski/datasets/yolo_dataset_20250218/coco_annotations_from_yolo_dataset_20250218.json"
    OUTPUT_DIR = "./training_ranged/output_ranged"
    CHECKPOINT_DIR = "./training_ranged/checkpoints_ranged"
    BEST_MODEL_DIR = "./training_ranged/best_model"
    TEST_OUTPUT_DIR = "./training_ranged/model_test_on_training"

    # Data selection - IMPORTANT: Recommended to use 2000-5000 images for proper training
    START_IDX = 0  # Start from beginning of dataset
    END_IDX = 2000  # Use first 2000 images (increased from 50)

    # Training parameters
    NUM_EPOCHS = 50  # Increased from 10 to 20 for better training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = (640, 640)
    NUM_QUERIES = 25  # Increased from 10 to 25 for more flexibility
    USE_PRETRAINED = True

    # Loss coefficients
    BBOX_COST = 5  # Increased from 4 to 5
    CLASS_COST = 2  # Kept at 2
    GIOU_COST = 4  # Increased from 3 to 4
    GIOU_LOSS_COEFFICIENT = 4  # Kept at 4
    BBOX_LOSS_COEFFICIENT = 5  # Kept at 5
    EOS_COEFFICIENT = 0.5  # Kept at 0.5

    # Focal loss parameters
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0  # Added gamma parameter

    # Early stopping parameters
    PATIENCE = 5

    # Evaluation parameters
    IOU_THRESHOLD = 0.5  # IoU threshold for true positive
    TEST_SAMPLES_PER_EPOCH = 20  # Increased from 10 to 20
    TEST_CONFIDENCE_THRESHOLD = 0.3  # Reduced from 0.5 to catch more detections
    VAL_SPLIT = 0.15  # Increased from 0.1 to 0.15

    # Visualization parameters
    VISUALIZE_EVERY_N_EPOCHS = 1


class SurgicalToolDataset(torch.utils.data.Dataset):
    """
    Dataset for surgical tool detection using DETR.

    Loads images and annotations in COCO format, performs validation checks,
    and applies optional data augmentation.

    Args:
        images_dir (str): Directory containing the images
        annotations_file (str): Path to COCO format annotations JSON file
        processor (DetrImageProcessor): Image processor for DETR model
        image_size (tuple): Target image size (height, width)
        augment (bool): Whether to apply data augmentation
        curriculum_difficulty (str): Difficulty level for curriculum learning ('easy', 'medium', 'hard')
    """

    def __init__(self, images_dir, annotations_file, processor, image_size=(800, 800),
                 augment=False, curriculum_difficulty=None):
        print("Initializing dataset...")
        self.images_dir = images_dir
        self.processor = processor
        self.image_size = image_size
        self.augment = augment
        self.curriculum_difficulty = curriculum_difficulty

        # Enhanced augmentations for better generalization
        if self.augment:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),  # Added vertical flip
                transforms.RandomRotation(30),  # Increased from 15 to 30 degrees
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Increased intensity
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),  # More aggressive scaling
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # Added sharpness adjustment
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

        # Sort valid images by filename for reproducibility
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
        """
        Filter images based on curriculum difficulty level.
        This implements a simple curriculum learning strategy.
        """
        print(f"Applying curriculum filtering for difficulty level: {self.curriculum_difficulty}")

        # This is a simplified approach - in a real implementation, you would
        # calculate difficulty based on various factors like image complexity,
        # tool visibility, number of tools, etc.

        # Simple curriculum strategy based on image index (as a proxy for difficulty)
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
            # Fixed deprecated max_size warning by using size parameter properly
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
    """
    Custom collate function to handle variable-sized annotations.

    Args:
        batch: A list of samples from the dataset

    Returns:
        A dictionary with batched data
    """
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
        # Print problematic items for debugging
        for i, item in enumerate(batch):
            print(f"Item {i} shapes: {item['pixel_values'].shape}")
        raise e


def check_data_integrity(dataset):
    """
    Verify dataset integrity and return whether training can proceed.

    Args:
        dataset: The dataset to verify

    Returns:
        bool: True if data is valid for training, False otherwise
    """
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

        # Display object count distribution for better understanding
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


def train_epoch(model, data_loader, optimizer, device, epoch, writer=None, config=None):
    """
    Train model for one epoch.

    Args:
        model: The model to train
        data_loader: DataLoader for training data
        optimizer: The optimizer
        device: Device to train on (cuda/cpu)
        epoch: Current epoch number
        writer: TensorBoard writer
        config: Configuration parameters

    Returns:
        float: Average training loss for the epoch
    """
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
        # Gradient clipping to prevent exploding gradients
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
    """
    Evaluate model on validation set with improved metrics.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation data
        device: Device to evaluate on (cuda/cpu)
        processor: Image processor for post-processing
        epoch: Current epoch number
        writer: TensorBoard writer
        visualize_predictions: Whether to save visualization images
        output_dir: Directory to save visualizations
        iou_threshold: IoU threshold for true positive detection
        confidence_threshold: Confidence threshold for predictions

    Returns:
        tuple: (avg_loss, precision, recall, f1_score)
    """
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

            # Calculate metrics
            pred_logits = outputs.logits
            pred_boxes = outputs.pred_boxes

            # Process each image in batch
            for i in range(len(pixel_values)):
                # Get ground truth boxes for this image
                gt_boxes = []
                gt_classes = []

                if 'boxes' in labels[i]:
                    gt_boxes = labels[i]['boxes'].cpu().numpy()
                    gt_classes = labels[i].get('class_labels', torch.zeros(len(gt_boxes))).cpu().numpy()

                # Track presence of objects (for overall detection metrics)
                has_gt_object = len(gt_boxes) > 0
                all_targets.append(1 if has_gt_object else 0)

                # Get predictions for this image
                scores = pred_logits[i].softmax(-1)[..., 0]
                boxes = pred_boxes[i]

                # Keep only predictions with high enough scores
                keep = scores > confidence_threshold
                boxes = boxes[keep]
                scores = scores[keep]

                # Track if model predicted any object
                has_pred_object = len(scores) > 0 and scores.max() > confidence_threshold
                all_predictions.append(1 if has_pred_object else 0)

                # Calculate IoU for box matching (simplified for this example)
                if len(gt_boxes) > 0 and len(boxes) > 0:
                    # Convert predicted boxes to xyxy format for IoU calculation
                    image = Image.open(image_paths[i]).convert("RGB")
                    h, w = image.size

                    pred_boxes_xyxy = []
                    for box in boxes:
                        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
                        cx, cy, bw, bh = box.cpu().numpy()
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        pred_boxes_xyxy.append([x1, y1, x2, y2])

                    # Convert ground truth boxes if needed (adapt this based on your label format)
                    gt_boxes_xyxy = []
                    for box in gt_boxes:
                        # If ground truth is already in xyxy format, use as is
                        # Otherwise convert from your format to xyxy
                        gt_boxes_xyxy.append(box)  # Adjust this based on your format

                    # Calculate IoU for each prediction against each ground truth box
                    max_ious = []
                    for pred_box in pred_boxes_xyxy:
                        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes_xyxy]
                        max_iou = max(ious) if ious else 0
                        max_ious.append(max_iou)
                        all_ious.append(max_iou)

                    # Count true positives (predictions with IoU > threshold)
                    tp_detections = sum(1 for iou in max_ious if iou >= iou_threshold)
                    true_positives += tp_detections
                    false_positives += len(pred_boxes_xyxy) - tp_detections
                    false_negatives += max(0, len(gt_boxes_xyxy) - tp_detections)
                else:
                    # If no ground truth boxes but predictions exist, all are false positives
                    if len(boxes) > 0:
                        false_positives += len(boxes)
                    # If ground truth boxes but no predictions, all are false negatives
                    if len(gt_boxes) > 0:
                        false_negatives += len(gt_boxes)

                # Visualize predictions (every few batches or if specifically requested)
                if visualize_predictions and (batch_idx % 2 == 0 or batch_idx < 3):
                    image = Image.open(image_paths[i]).convert("RGB")

                    # Create visualization with improved formatting
                    result_image = visualize_predictions_on_image(
                        image=image,
                        pred_boxes=boxes,
                        pred_scores=scores,
                        gt_boxes=gt_boxes,
                        iou_threshold=iou_threshold
                    )

                    # Save image
                    filename = os.path.basename(image_paths[i])
                    output_path = os.path.join(epoch_output_dir, f"val_batch_{batch_idx}_img_{i}_{filename}")
                    result_image.save(output_path)

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

    # Calculate confusion matrix for overall detection (not per-box)
    cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1])

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

    # Plot and save confusion matrix
    if epoch is not None and writer is not None:
        fig, ax = plt.figure(figsize=(8, 8)), plt.subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                    xticklabels=['No Object', 'Object'],
                    yticklabels=['No Object', 'Object'])
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title(f'Detection Confusion Matrix (Epoch {epoch + 1})')

        # Save the figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))
            plt.close()

    return avg_loss, precision, recall, f1_score


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1: First box in format [x1, y1, x2, y2]
        box2: Second box in format [x1, y1, x2, y2]

    Returns:
        float: IoU score
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou


def visualize_predictions_on_image(image, pred_boxes, pred_scores, gt_boxes=None, iou_threshold=0.5):
    """
    Create visualization with predictions and ground truth boxes.

    Args:
        image: PIL Image
        pred_boxes: Predicted boxes
        pred_scores: Prediction scores
        gt_boxes: Ground truth boxes (optional)
        iou_threshold: IoU threshold for true positives

    Returns:
        PIL.Image: Image with visualizations
    """
    # Create a copy of the image
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    # Try to load a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    # Draw ground truth boxes first (in green)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            # Convert box format if needed
            if len(box) == 4:  # Assuming [x1, y1, x2, y2] or [x, y, w, h]
                if box[2] < 1 and box[3] < 1:  # Normalized coordinates
                    # Convert from normalized [x, y, w, h] to pixel coordinates
                    w, h = image.size
                    x1, y1 = box[0] * w, box[1] * h
                    x2, y2 = x1 + box[2] * w, y1 + box[3] * h
                else:
                    # Assuming [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                # Draw ground truth box
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                draw.text((x1, max(0, y1 - 15)), "Ground Truth", fill="green", font=font)

    # Draw predicted boxes (in different colors based on status)
    h, w = image.size
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        cx, cy, bw, bh = box.cpu().numpy()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Calculate IoU with ground truth to determine color
        color = "red"  # Default: prediction
        label = f"Pred: {score.item():.2f}"

        if gt_boxes is not None and len(gt_boxes) > 0:
            # Convert predicted box to the same format as ground truth for IoU calculation
            pred_box = [x1, y1, x2, y2]

            # Calculate IoUs with all ground truth boxes
            ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
            max_iou = max(ious) if ious else 0

            # Color code based on IoU
            if max_iou >= iou_threshold:
                color = "blue"  # True positive
                label = f"TP: {score.item():.2f}, IoU: {max_iou:.2f}"
            else:
                color = "red"  # False positive
                label = f"FP: {score.item():.2f}, IoU: {max_iou:.2f}"

        # Draw box and label
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw filled background for text
        text_bbox = draw.textbbox((x1, max(0, y1 - 15)), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((x1, max(0, y1 - 15)), label, fill="white", font=font)

    # Add information about the number of detections
    info_text = f"GT: {len(gt_boxes) if gt_boxes is not None else 0}, Pred: {len(pred_boxes)}"
    draw.text((10, 10), info_text, fill="yellow", font=font)

    return result_image


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_dir):
    """
    Save training state checkpoint with comprehensive metrics.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler (if any)
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        checkpoint_dir: Directory to save checkpoint

    Returns:
        str: Path to saved checkpoint
    """
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
    """
    Find the most recent checkpoint in directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        str: Path to latest checkpoint, or None if none found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None

    # Sort by epoch number (extracted from filename)
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = checkpoint_files[-1]
    print(f"[CHECKPOINT] Found latest checkpoint: {latest}")
    return latest


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None, device=None):
    """Load training state from checkpoint with error handling."""
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Checkpoint {checkpoint_path} does not exist")
        return model, 0, {}  # Return original model, not None

    print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device if device else 'cpu')

        # Try to load model weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
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
    print(f"Percentage of images with annotations: {len(valid_indices) / len(selected_indices) * 100:.2f}%")

    # Create subset with selected indices
    return Subset(dataset, valid_indices)


def save_images_with_annotations(dataset, indices, output_dir, max_images=100):
    """
    Save images with drawn bounding boxes for visual verification.

    Args:
        dataset: Dataset containing images and annotations
        indices: List of image indices to process
        output_dir: Folder to save output images
        max_images: Maximum number of images to process
    """
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


def model_testing_on_dataset(model, processor, dataset, device,
                             num_samples=50, confidence_threshold=0.1,
                             output_dir="./model_test_results", phase="test"):
    """
    Test model on a dataset with detailed metrics and visualization.

    Args:
        model: Model to test
        processor: Image processor
        dataset: Dataset to test on
        device: Device to run on
        num_samples: Number of samples to test
        confidence_threshold: Confidence threshold for detections
        output_dir: Directory to save visualizations
        phase: Testing phase name (e.g., "test", "train")

    Returns:
        dict: Dictionary of test results and metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Ensure model is in evaluation mode
    model.eval()

    # Select random samples from dataset
    dataset_indices = list(range(len(dataset)))
    if num_samples < len(dataset):
        indices = np.random.choice(dataset_indices, size=num_samples, replace=False)
    else:
        indices = dataset_indices
        print(f"Testing on all {len(indices)} samples in dataset")

    print(f"Testing model on {len(indices)} samples...")

    # Metrics
    results = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_ious = []

    with torch.no_grad():  # Disable gradient calculation
        for i, idx in enumerate(tqdm(indices, desc=f"{phase.capitalize()} Evaluation")):
            # Get image info based on dataset type
            if hasattr(dataset, 'indices'):  # If dataset is a Subset
                actual_idx = dataset.indices[idx]
                base_dataset = dataset.dataset
                image_info = base_dataset.valid_images[actual_idx]
                image_id = image_info['id']
                image_filename = base_dataset.id_to_filename[image_id]
                image_path = os.path.join(base_dataset.images_dir, image_filename)
                original_annotations = base_dataset.id_to_annotations.get(image_id, [])
            else:
                image_info = dataset.valid_images[idx]
                image_id = image_info['id']
                image_filename = dataset.id_to_filename[image_id]
                image_path = os.path.join(dataset.images_dir, image_filename)
                original_annotations = dataset.id_to_annotations.get(image_id, [])

            # Load image
            original_image = Image.open(image_path).convert("RGB")
            width, height = original_image.size

            # Prepare image for model
            inputs = processor(images=original_image, return_tensors="pt").to(device)

            # Run model
            outputs = model(**inputs)

            # Process predictions
            target_sizes = torch.tensor([original_image.size[::-1]]).to(device)
            results_processed = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=confidence_threshold
            )[0]

            # Get predictions
            pred_boxes = results_processed["boxes"].cpu().numpy()
            pred_scores = results_processed["scores"].cpu().numpy()
            pred_labels = results_processed["labels"].cpu().numpy()

            # Get ground truth boxes
            gt_boxes = []
            for ann in original_annotations:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
            gt_boxes = np.array(gt_boxes) if gt_boxes else np.array([])

            # Calculate metrics
            # For each ground truth box, check if there is a matching prediction
            matches = []
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                for gt_idx, gt_box in enumerate(gt_boxes):
                    best_iou = 0
                    best_match = -1

                    for pred_idx, pred_box in enumerate(pred_boxes):
                        if pred_idx in [m[1] for m in matches]:  # Skip already matched predictions
                            continue

                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred_idx

                    # If we found a match with IoU > threshold
                    if best_iou >= 0.5:
                        matches.append((gt_idx, best_match, best_iou))
                        all_ious.append(best_iou)

            # Count TP, FP, FN
            true_positives += len(matches)
            false_positives += len(pred_boxes) - len(matches)
            false_negatives += len(gt_boxes) - len(matches)

            # Create visualization
            result_image = visualize_predictions_on_image(
                image=original_image,
                pred_boxes=torch.from_numpy(pred_boxes) if len(pred_boxes) > 0 else torch.zeros((0, 4)),
                pred_scores=torch.from_numpy(pred_scores) if len(pred_scores) > 0 else torch.zeros(0),
                gt_boxes=gt_boxes if len(gt_boxes) > 0 else None,
                iou_threshold=0.5
            )

            # Save visualization
            result_path = os.path.join(output_dir, f"{phase}_{i:03d}_{os.path.basename(image_path)}")
            result_image.save(result_path)

            # Record results
            results.append({
                "idx": idx if not hasattr(dataset, 'indices') else actual_idx,
                "image_path": image_path,
                "gt_boxes_count": len(gt_boxes),
                "pred_boxes_count": len(pred_boxes),
                "matches": len(matches),
                "max_score": float(np.max(pred_scores)) if len(pred_scores) > 0 else 0,
                "iou": float(np.mean([m[2] for m in matches])) if matches else 0
            })

    # Calculate overall metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0

    # Print summary
    print(f"\nModel {phase} results summary:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")

    # Create visualization of detection success by confidence score
    plot_detection_metrics(results, output_dir, phase)

    # Return aggregated results
    return {
        "results": results,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_iou": avg_iou,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    }


def plot_detection_metrics(results, output_dir, phase="test"):
    """
    Create visualizations of detection metrics.

    Args:
        results: List of test results
        output_dir: Directory to save plots
        phase: Testing phase name
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    scores = [r["max_score"] for r in results]
    correct = [r["matches"] == r["gt_boxes_count"] and r["matches"] == r["pred_boxes_count"] for r in results]
    ious = [r["iou"] for r in results if r["iou"] > 0]

    # Plot 1: Detection success by confidence score
    plt.figure(figsize=(10, 6))
    plt.scatter(scores, correct, alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Detection Success by Confidence Score')
    plt.xlabel('Max Confidence Score')
    plt.ylabel('Correct Detection (1=Yes, 0=No)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{phase}_detection_by_confidence.png"))
    plt.close()

    # Plot 2: Distribution of IoUs
    if ious:
        plt.figure(figsize=(10, 6))
        plt.hist(ious, bins=20, alpha=0.7)
        plt.axvline(x=0.5, color='r', linestyle='--', label='IoU Threshold (0.5)')
        plt.title('Distribution of IoU Values for Matched Detections')
        plt.xlabel('IoU Value')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{phase}_iou_distribution.png"))
        plt.close()

    # Plot 3: Box count distribution
    gt_counts = [r["gt_boxes_count"] for r in results]
    pred_counts = [r["pred_boxes_count"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.hist([gt_counts, pred_counts], bins=max(max(gt_counts, default=0), max(pred_counts, default=0)) + 1,
             label=['Ground Truth', 'Predictions'], alpha=0.7)
    plt.title('Distribution of Box Counts')
    plt.xlabel('Number of Boxes')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{phase}_box_count_distribution.png"))
    plt.close()


def analyze_test_progress(test_results_file):
    """
    Analyze model progress based on saved test results.

    Args:
        test_results_file: Path to JSON file with test results
    """
    if not os.path.exists(test_results_file):
        print(f"File {test_results_file} does not exist")
        return

    with open(test_results_file, "r") as f:
        results = json.load(f)

    # Group results by epoch
    epochs = {}
    for r in results:
        epoch = r["epoch"]
        if epoch not in epochs:
            epochs[epoch] = []
        epochs[epoch].append(r)

    # Analyze results for each epoch
    print("\nTraining progress analysis from test data:")
    print("-" * 80)
    print(f"{'Epoch':^6} | {'Avg. max score':^15} | {'No detection':^15} | {'Correct count':^15}")
    print("-" * 80)

    epoch_metrics = []

    for epoch in sorted(epochs.keys()):
        epoch_results = epochs[epoch]
        avg_score = np.mean([r['max_score'] for r in epoch_results])
        no_detect = sum(1 for r in epoch_results if r['pred_boxes'] == 0)
        correct_detect = sum(1 for r in epoch_results if r['gt_boxes'] == r['pred_boxes'] and r['pred_boxes'] > 0)

        print(
            f"{epoch:^6} | {avg_score:^15.4f} | {no_detect:^5}/{len(epoch_results):^5} {no_detect / len(epoch_results) * 100:^3.1f}% | {correct_detect:^5}/{len(epoch_results):^5} {correct_detect / len(epoch_results) * 100:^3.1f}%")

        epoch_metrics.append({
            'epoch': epoch,
            'avg_score': avg_score,
            'no_detect_rate': no_detect / len(epoch_results) * 100,
            'correct_rate': correct_detect / len(epoch_results) * 100
        })

    # Plot metrics over epochs
    if epoch_metrics:
        epochs = [m['epoch'] for m in epoch_metrics]
        avg_scores = [m['avg_score'] for m in epoch_metrics]
        no_detect_rates = [m['no_detect_rate'] for m in epoch_metrics]
        correct_rates = [m['correct_rate'] for m in epoch_metrics]

        plt.figure(figsize=(12, 8))

        # Plot average score
        plt.subplot(3, 1, 1)
        plt.plot(epochs, avg_scores, 'o-', color='blue')
        plt.title('Average Max Confidence Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)

        # Plot no detection rate
        plt.subplot(3, 1, 2)
        plt.plot(epochs, no_detect_rates, 'o-', color='red')
        plt.title('No Detection Rate (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage')
        plt.grid(True, alpha=0.3)

        # Plot correct detection rate
        plt.subplot(3, 1, 3)
        plt.plot(epochs, correct_rates, 'o-', color='green')
        plt.title('Correct Detection Rate (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_dir = os.path.dirname(test_results_file)
        plt.savefig(os.path.join(plot_dir, 'training_progress_metrics.png'))
        plt.close()


def train_model_with_curriculum(config):
    """
    Train model using curriculum learning strategy.

    Args:
        config: Configuration settings
    """
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
        "use_focal_loss": config.USE_FOCAL_LOSS,
        "focal_alpha": config.FOCAL_ALPHA,
        "focal_gamma": config.FOCAL_GAMMA,
    }

    # Initialize or load model
    print("\nInitializing model...")
    if os.path.exists(config.BEST_MODEL_DIR) and os.listdir(config.BEST_MODEL_DIR):
        print(f"Loading model from {config.BEST_MODEL_DIR}")

        # Load DetrConfig and update with our parameters
        detr_config = DetrConfig.from_pretrained(config.BEST_MODEL_DIR)
        for key, value in model_config.items():
            if hasattr(detr_config, key):
                print(f"Updating parameter {key}: {getattr(detr_config, key)} -> {value}")
                setattr(detr_config, key, value)

        # Load model and processor
        model = DetrForObjectDetection.from_pretrained(
            config.BEST_MODEL_DIR,
            config=detr_config,
            ignore_mismatched_sizes=True
        )
        processor = DetrImageProcessor.from_pretrained(config.BEST_MODEL_DIR)
    else:
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
        augment=False  # No augmentation for full dataset
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
                    augment=True  # Enable augmentation for training
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
                    curriculum_difficulty=stage_name  # Apply curriculum filtering
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
                train_loss = train_epoch(model, train_loader, optimizer, device, global_epoch, writer, config)
                print(f"Epoch {global_epoch + 1} finished. Training Loss: {train_loss:.4f}")

                # Evaluate model
                val_loss, precision, recall, f1_score = evaluate_model(
                    model, val_loader, device, processor, global_epoch, writer,
                    visualize_predictions=(global_epoch % config.VISUALIZE_EVERY_N_EPOCHS == 0),
                    output_dir=visualization_dir,
                    iou_threshold=config.IOU_THRESHOLD,
                    confidence_threshold=config.TEST_CONFIDENCE_THRESHOLD
                )
                print(f"Validation Loss: {val_loss:.4f}, F1: {f1_score:.4f}")

                # Test model on training data sample
                test_results = model_testing_on_dataset(
                    model=model,
                    processor=processor,
                    dataset=curr_train_dataset,
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
    test_results = model_testing_on_dataset(
        model=model,
        processor=processor,
        dataset=test_dataset,
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

    # Display configuration
    print("Configuration:")
    for attr, value in vars(config).items():
        print(f"  {attr}: {value}")

    # Run training with curriculum learning
    model, processor, metrics = train_model_with_curriculum(config)

    return model, processor, metrics


if __name__ == "__main__":
    main()