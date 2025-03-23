import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig


class SurgicalToolDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, processor, image_size=(800, 800), augment=False):
        """
        Initialize the surgical tool dataset.

        Args:
            images_dir: Directory containing images
            annotations_file: Path to COCO format annotations file
            processor: DETR image processor
            image_size: Target image size for model
            augment: Whether to apply data augmentation
        """
        print("Initializing dataset...")
        self.images_dir = images_dir
        self.processor = processor
        self.image_size = image_size
        self.augment = augment

        # Strong augmentations for better generalization
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

        # Verify that images have annotations
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
        return {"pixel_values": pixel_values, "labels": target,
                "image_path": image_path}  # Include image path for debugging


def collate_fn(batch):
    """Collate function to handle variable-sized annotations."""
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


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    else:
        print("GPU not available, running on CPU")


def train_epoch(model, data_loader, optimizer, device, epoch, writer=None):
    """Train model for one epoch."""
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def evaluate_model(model, data_loader, device, epoch=None, writer=None, visualize_predictions=False, output_dir=None):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    # For visualization
    if visualize_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir):
    """Save training state checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load training state from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Checkpoint {checkpoint_path} does not exist")
        return None, 0, float('inf')

    print(f"[CHECKPOINT] Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[CHECKPOINT] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return model, start_epoch, best_val_loss
    except Exception as e:
        print(f"[CHECKPOINT] Error loading checkpoint: {e}")
        return None, 0, float('inf')


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
    Zapisuje obrazy z narysowanymi bounding boxami do celów kontroli wizualnej.

    Args:
        dataset: Dataset zawierający obrazy i adnotacje
        indices: Lista indeksów obrazów do przetworzenia
        output_dir: Folder do zapisania wynikowych obrazów
        max_images: Maksymalna liczba obrazów do przetworzenia
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Zapisywanie obrazów z adnotacjami do {output_dir}...")

    # Limit liczby obrazów do przetworzenia
    indices = indices[:min(len(indices), max_images)]

    for idx in tqdm(indices, desc="Generowanie wizualizacji"):
        # Pobierz obraz i adnotacje
        image_info = dataset.valid_images[idx]
        image_id = image_info['id']
        image_filename = dataset.id_to_filename[image_id]
        image_path = os.path.join(dataset.images_dir, image_filename)

        # Wczytaj obraz
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
        except Exception as e:
            print(f"Błąd otwierania obrazu {image_path}: {e}")
            continue

        # Pobierz adnotacje
        annotations = dataset.id_to_annotations.get(image_id, [])

        # Narysuj bounding boxy na obrazie
        draw = ImageDraw.Draw(image)

        for ann in annotations:
            # Format bbox w COCO to [x, y, width, height]
            # Musimy przekonwertować na [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Narysuj prostokąt
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Dodaj etykietę kategorii i ID adnotacji
            label = f"ID: {ann['id']} Cat: {ann['category_id']}"
            draw.text((x1, max(0, y1 - 15)), label, fill="red")

        # Dodaj informacje o obrazie
        draw.text((10, 10), f"Image ID: {image_id}, Filename: {image_filename}", fill="blue")
        draw.text((10, 30), f"Annotations: {len(annotations)}", fill="blue")

        # Zapisz obraz z adnotacjami
        output_filename = f"{idx}_{image_filename}"
        output_path = os.path.join(output_dir, output_filename)
        image.save(output_path)

    print(f"Zapisano {len(indices)} obrazów z adnotacjami w {output_dir}")


def model_on_training_data_testing(model, processor, dataset, full_dataset, epoch,
                                num_samples=10, confidence_threshold=0.1,
                                output_dir="./model_test_on_training"):
    """
    Testuje model na próbkach z danych treningowych po każdej epoce.

    Args:
        model: Model DETR do testowania
        processor: Procesor obrazów dla modelu
        dataset: Obiekt dataset zawierający dane treningowe
        full_dataset: Pełny dataset (do dostępu do oryginalnych adnotacji)
        epoch: Numer epoki
        num_samples: Liczba próbek do testowania
        confidence_threshold: Próg pewności detekcji
        output_dir: Katalog do zapisu wyników wizualizacji
    """
    # Utwórz katalog dla danej epoki
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Upewnij się, że model jest w trybie ewaluacji
    model.eval()

    # Pobierz urządzenie, na którym jest model
    device = next(model.parameters()).device

    # Wybierz losowe indeksy z datasetu
    dataset_indices = list(range(len(dataset)))
    indices = np.random.choice(dataset_indices, size=min(num_samples, len(dataset)), replace=False)

    print(f"Testowanie modelu po epoce {epoch} na {len(indices)} próbkach treningowych...")

    results = []

    with torch.no_grad():  # Wyłącz obliczanie gradientów dla przyspieszenia
        for i, idx in enumerate(tqdm(indices, desc=f"Test po epoce {epoch}")):
            # Pobierz indeks w oryginalnym datasecie
            if hasattr(dataset, 'indices'):  # Jeśli to jest podzbiór (Subset)
                actual_idx = dataset.indices[idx]
            else:
                actual_idx = idx

            # Pobierz informacje o obrazie
            image_info = full_dataset.valid_images[actual_idx]
            image_id = image_info['id']
            image_filename = full_dataset.id_to_filename[image_id]
            image_path = os.path.join(full_dataset.images_dir, image_filename)

            # Pobierz oryginalne adnotacje
            original_annotations = full_dataset.id_to_annotations.get(image_id, [])

            # Wczytaj oryginalny obraz
            original_image = Image.open(image_path).convert("RGB")
            width, height = original_image.size

            # Przygotuj obraz dla modelu
            inputs = processor(images=original_image, return_tensors="pt").to(device)

            # Uruchom model
            outputs = model(**inputs)

            # Przetwórz predykcje
            target_sizes = torch.tensor([original_image.size[::-1]]).to(device)
            results_processed = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=confidence_threshold
            )[0]

            # Stwórz wizualizację
            result_image = original_image.copy()
            draw = ImageDraw.Draw(result_image)

            # Narysuj rzeczywiste bounding boxy (zielone)
            for ann in original_annotations:
                # Format COCO: [x, y, width, height]
                x, y, w, h = ann['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Narysuj rzeczywisty bbox
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, y1 - 15), "PRAWDZIWY", fill="green")

            # Narysuj przewidziane bounding boxy (czerwone)
            for score, label, box in zip(results_processed["scores"],
                                         results_processed["labels"],
                                         results_processed["boxes"]):
                score_val = score.item()
                label_val = label.item()
                x1, y1, x2, y2 = box.cpu().numpy()

                # Narysuj przewidziany bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 15),
                          f"{model.config.id2label[label_val]}: {score_val:.2f}",
                          fill="red")

            # Dodaj informacje o epoce
            draw.text((10, 10), f"Epoka: {epoch}", fill="blue")

            # Zapisz wynik
            result_path = os.path.join(epoch_dir, f"test_{i:02d}_{os.path.basename(image_path)}")
            result_image.save(result_path)

            # Zapisz informacje o detekcji do analizy
            pred_count = len(results_processed["scores"])
            max_score = float(results_processed["scores"].max()) if pred_count > 0 else 0

            results.append({
                "epoch": epoch,
                "idx": actual_idx,
                "image_path": image_path,
                "gt_boxes": len(original_annotations),
                "pred_boxes": pred_count,
                "max_score": max_score
            })

    # Wydrukuj podsumowanie
    print("\nPodsumowanie wyników testu po epoce", epoch)
    if results:
        avg_score = np.mean([r['max_score'] for r in results])
        no_detect = sum(1 for r in results if r['pred_boxes'] == 0)
        correct_detect = sum(1 for r in results if r['gt_boxes'] == r['pred_boxes'] and r['pred_boxes'] > 0)

        print(f"Średni najwyższy wynik detekcji: {avg_score:.4f}")
        print(f"Obrazy bez detekcji: {no_detect} / {len(results)} ({no_detect / len(results) * 100:.1f}%)")
        print(
            f"Obrazy z poprawną liczbą detekcji: {correct_detect} / {len(results)} ({correct_detect / len(results) * 100:.1f}%)")
    else:
        print("Brak wyników do wyświetlenia")

    # Przywróć model do trybu treningu
    model.train()

    return results


def analyze_test_progress(test_results_file):
    """
    Analizuje postęp treningu na podstawie zapisanych wyników testów.

    Args:
        test_results_file: Ścieżka do pliku JSON z wynikami testów
    """
    if not os.path.exists(test_results_file):
        print(f"Plik {test_results_file} nie istnieje")
        return

    with open(test_results_file, "r") as f:
        results = json.load(f)

    # Grupowanie wyników według epok
    epochs = {}
    for r in results:
        epoch = r["epoch"]
        if epoch not in epochs:
            epochs[epoch] = []
        epochs[epoch].append(r)

    # Analiza wyników dla każdej epoki
    print("\nPostęp treningu - analiza testów na danych treningowych:")
    print("-" * 80)
    print(f"{'Epoka':^6} | {'Śr. max score':^15} | {'Brak detekcji':^15} | {'Poprawna liczba':^15}")
    print("-" * 80)

    for epoch in sorted(epochs.keys()):
        epoch_results = epochs[epoch]
        avg_score = np.mean([r['max_score'] for r in epoch_results])
        no_detect = sum(1 for r in epoch_results if r['pred_boxes'] == 0)
        correct_detect = sum(1 for r in epoch_results if r['gt_boxes'] == r['pred_boxes'] and r['pred_boxes'] > 0)

        print(
            f"{epoch:^6} | {avg_score:^15.4f} | {no_detect:^5}/{len(epoch_results):^5} {no_detect / len(epoch_results) * 100:^3.1f}% | {correct_detect:^5}/{len(epoch_results):^5} {correct_detect / len(epoch_results) * 100:^3.1f}%")


def main():
    """Main training function with hardcoded parameters and model testing after each epoch."""

    # ========== HARDCODED PARAMETERS ==========
    # Paths for data and output
    train_images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train"
    train_annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Detr/coco_annotations_from_yolo_dataset_20250218.json"
    output_dir = "./training_ranged/output_ranged"
    checkpoint_dir = "./training_ranged/checkpoints_ranged"
    best_model_dir = "./training_ranged/best_model"
    test_output_dir = "./training_ranged/model_test_on_training"  # Nowy katalog na wyniki testów

    # Image selection range - MODIFY THESE TO SELECT DIFFERENT IMAGES
    start_idx = 0  # Start from image 0
    end_idx = 800  # End at image 800 (exclusive)

    # Training parameters
    num_epochs = 10
    batch_size = 8
    learning_rate = 5e-5
    image_size = (800, 800)
    num_queries = 2  # DETR parameter for number of objects to detect
    use_pretrained = True  # Whether to use pretrained model as starting point

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement before stopping

    # Test parameters
    test_samples_per_epoch = 10
    test_confidence_threshold = 0.1

    # =========================================

    print("=" * 80)
    print("DETR SURGICAL TOOL DETECTION - TRAINING START")
    print("=" * 80)
    print(f"Training on images from {start_idx} to {end_idx}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)  # Katalog na wyniki testów
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    # Setup paths
    checkpoint_dir = Path(checkpoint_dir)
    best_model_dir = Path(best_model_dir)

    print(f"[PATHS] Train Images Directory: {train_images_dir}")
    print(f"[PATHS] Train Annotations File: {train_annotations_file}")
    print(f"[PATHS] Output Directory: {output_dir}")
    print(f"[PATHS] Checkpoint Directory: {checkpoint_dir.absolute()}")
    print(f"[PATHS] Best Model Directory: {best_model_dir.absolute()}")
    print(f"[PATHS] Test Output Directory: {test_output_dir}")

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")
    print(f"[TRAINING] Number of epochs: {num_epochs}")
    print(f"[TRAINING] Batch size: {batch_size}")
    print(f"[TRAINING] Learning rate: {learning_rate}")
    print(f"[TRAINING] Image size: {image_size}")
    print(f"[TRAINING] Using pretrained model: {use_pretrained}")
    print(f"[TESTING] Samples per epoch: {test_samples_per_epoch}")
    print(f"[TESTING] Confidence threshold: {test_confidence_threshold}")

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    print("=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)

    # DETR model configuration
    custom_config = {
        "num_queries": num_queries,
        "bbox_cost": 5,
        "class_cost": 1,
        "giou_cost": 4,
        "giou_loss_coefficient": 4,
        "bbox_loss_coefficient": 5,
        "eos_coefficient": 0.1,
    }

    # Check for existing model
    if os.path.exists(best_model_dir) and len(os.listdir(best_model_dir)) > 0:
        print(f"[MODEL] Found saved model in {best_model_dir}")
        try:
            print(f"[MODEL] Attempting to load model from {best_model_dir}...")
            # Load saved model and processor
            config = DetrConfig.from_pretrained(str(best_model_dir))

            # Update configuration with our settings
            for key, value in custom_config.items():
                if hasattr(config, key):
                    print(f"[MODEL] Updating parameter {key} from {getattr(config, key)} to {value}")
                    setattr(config, key, value)

            model = DetrForObjectDetection.from_pretrained(
                str(best_model_dir),
                config=config,
                ignore_mismatched_sizes=True
            )
            processor = DetrImageProcessor.from_pretrained(str(best_model_dir))
            print("[MODEL] SUCCESS! Loaded model and processor from saved best model.")
        except Exception as e:
            print(f"[MODEL] ERROR loading model: {e}")
            print("[MODEL] Loading default pre-trained model...")
            # Use pretrained model from HuggingFace
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=1,
                num_queries=num_queries,
                ignore_mismatched_sizes=True
            ) if use_pretrained else None

            # If not using pretrained, initialize model from scratch
            if model is None:
                config = DetrConfig(num_labels=1, **custom_config)
                model = DetrForObjectDetection(config)

            # Apply custom configuration
            for key, value in custom_config.items():
                if hasattr(model.config, key):
                    print(f"[MODEL] Setting parameter {key} to {value}")
                    setattr(model.config, key, value)

            model.config.id2label = {0: "surgical_tool"}
            model.config.label2id = {"surgical_tool": 0}
            model.config.num_labels = 1

            # Initialize processor
            processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                size={'shortest_edge': image_size[0], 'longest_edge': image_size[1]}
            )
    else:
        print(f"[MODEL] No previously saved model found in {best_model_dir}")
        print("[MODEL] Loading default pre-trained model...")
        # Use pretrained model from HuggingFace
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,
            num_queries=num_queries,
            ignore_mismatched_sizes=True
        ) if use_pretrained else None

        # If not using pretrained, initialize model from scratch
        if model is None:
            config = DetrConfig(num_labels=1, **custom_config)
            model = DetrForObjectDetection(config)

        # Apply custom configuration
        for key, value in custom_config.items():
            if hasattr(model.config, key):
                print(f"[MODEL] Setting parameter {key} to {value}")
                setattr(model.config, key, value)

        model.config.id2label = {0: "surgical_tool"}
        model.config.label2id = {"surgical_tool": 0}
        model.config.num_labels = 1

        # Initialize processor
        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50",
            size={'shortest_edge': image_size[0], 'longest_edge': image_size[1]}
        )

    print("=" * 80)
    print("DATASET LOADING")
    print("=" * 80)

    # Create full dataset
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

    # Create data loaders
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

    # Setup optimizer and move model to device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=3,
        verbose=True
    )

    model.to(device)

    print("=" * 80)
    print("CHECKPOINT CHECKING")
    print("=" * 80)

    # Check for existing checkpoints
    latest_checkpoint = find_latest_checkpoint(str(checkpoint_dir))
    if latest_checkpoint:
        loaded_model, start_epoch, best_val_loss = load_checkpoint(latest_checkpoint, model, optimizer, device)
        if loaded_model is not None:
            model = loaded_model
            print(f"[CHECKPOINT] Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0
            best_val_loss = float('inf')
            print("[CHECKPOINT] Starting training from beginning (checkpoint loading failed)")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        print("[CHECKPOINT] No checkpoints found, starting training from scratch.")

    print("=" * 80)
    print(f"TRAINING STARTING FROM EPOCH {start_epoch + 1}")
    print("=" * 80)

    # Track best model for early stopping
    patience_counter = 0

    # Initialize epoch variable before the loop to ensure it's always defined
    epoch = start_epoch - 1

    # Lista do przechowywania wyników testów z każdej epoki
    all_test_results = []

    # Check if we need to do any training at all
    if start_epoch >= num_epochs:
        print(f"[TRAINING] All {num_epochs} epochs already completed. No further training needed.")
    else:
        try:
            for epoch in range(start_epoch, num_epochs):
                print(f"Starting epoch {epoch + 1}/{num_epochs}...")

                # Train for one epoch
                avg_train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
                print(f"Epoch {epoch + 1} finished. Training Loss: {avg_train_loss:.4f}")

                # Validate model
                avg_val_loss = evaluate_model(
                    model,
                    val_loader,
                    device,
                    epoch,
                    writer,
                    visualize_predictions=(epoch % 2 == 0),  # Visualize predictions every 2 epochs
                    output_dir=visualization_dir
                )
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Testuj model na danych treningowych po każdej epoce
                print("=" * 60)
                print(f"TESTOWANIE MODELU PO EPOCE {epoch + 1}")
                print("=" * 60)

                epoch_test_results = model_on_training_data_testing(
                    model=model,
                    processor=processor,
                    dataset=train_dataset,
                    full_dataset=full_dataset,
                    epoch=epoch,
                    num_samples=test_samples_per_epoch,  # Liczba próbek do testowania
                    confidence_threshold=test_confidence_threshold,  # Próg pewności detekcji
                    output_dir=test_output_dir
                )
                all_test_results.extend(epoch_test_results)

                # Zapisz wyniki testów do pliku JSON
                with open(os.path.join(test_output_dir, f"test_results_up_to_epoch_{epoch}.json"), "w") as f:
                    # Konwertuj ścieżki na stringi, ponieważ obiekty Path nie są serializowalne w JSON
                    json_results = []
                    for r in all_test_results:
                        r_copy = r.copy()
                        r_copy["image_path"] = str(r_copy["image_path"])
                        json_results.append(r_copy)
                    json.dump(json_results, f, indent=4)

                # Update learning rate based on validation loss
                lr_scheduler.step(avg_val_loss)

                # Early stopping and model saving logic
                if avg_val_loss < best_val_loss:
                    improvement = best_val_loss - avg_val_loss
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    # Save best model
                    print(
                        f"[MODEL] New best validation loss: {best_val_loss:.4f} (improved by {improvement:.4f}), saving model...")
                    os.makedirs(str(best_model_dir), exist_ok=True)
                    model.save_pretrained(str(best_model_dir / f"detr_tool_model_epoch_{epoch + 1}"))
                    processor.save_pretrained(str(best_model_dir / f"detr_tool_model_epoch_{epoch + 1}"))

                    # Also save the best overall model
                    model.save_pretrained(str(best_model_dir))
                    processor.save_pretrained(str(best_model_dir))
                else:
                    patience_counter += 1
                    print(
                        f"[EARLY STOPPING] No improvement for {patience_counter} epochs. Best val loss: {best_val_loss:.4f}")

                    if patience_counter >= patience:
                        print(f"[EARLY STOPPING] Stopping training after {patience} epochs without improvement.")
                        break

                # Save checkpoint every epoch
                save_checkpoint(model, optimizer, epoch, best_val_loss, str(checkpoint_dir))

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving current model and checkpoint...")
            save_checkpoint(model, optimizer, epoch, best_val_loss, str(checkpoint_dir))
            model.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_interrupted_ranged"))
            processor.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_interrupted_ranged"))
            print(
                f"Interrupted model saved to: {os.path.join(output_dir, 'detr_tool_tracking_model_interrupted_ranged')}")

    print("=" * 80)
    print("TRAINING FINISHED - SAVING FINAL MODEL")
    print("=" * 80)

    # Save final checkpoint and model
    save_checkpoint(model, optimizer, epoch, best_val_loss, str(checkpoint_dir))
    model.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_final"))
    processor.save_pretrained(os.path.join(output_dir, "detr_tool_tracking_model_final"))
    print(f"Final model saved to: {os.path.join(output_dir, 'detr_tool_tracking_model_final')}")

    # Analyze test progress
    if all_test_results:
        print("=" * 80)
        print("ANALIZA POSTĘPU TRENINGU")
        print("=" * 80)
        analyze_test_progress(os.path.join(test_output_dir, f"test_results_up_to_epoch_{epoch}.json"))

    # Close TensorBoard writer
    writer.close()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_model_dir}")
    print(f"Final model saved to: {os.path.join(output_dir, 'detr_tool_tracking_model_final')}")
    print(f"Test results saved to: {test_output_dir}")


if __name__ == "__main__":
    main()