import torch
from torch.utils.data import DataLoader, random_split, Subset
from transformers import DetrForObjectDetection, DetrImageProcessor
import os
import json
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

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
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # Ustawienie category_id na 0 dla wszystkich anotacji
        for ann in self.annotations['annotations']:
            ann['category_id'] = 0

        # Filter out images that don't exist and create mappings
        self.valid_images = []
        self.id_to_filename = {}
        self.id_to_annotations = {}

        for img in self.annotations['images']:
            image_path = os.path.join(self.images_dir, img['file_name'])
            if os.path.exists(image_path):
                self.valid_images.append(img)
                self.id_to_filename[img['id']] = img['file_name']

        for ann in self.annotations['annotations']:
            if ann['image_id'] in self.id_to_filename:
                if ann['image_id'] not in self.id_to_annotations:
                    self.id_to_annotations[ann['image_id']] = []
                self.id_to_annotations[ann['image_id']].append(ann)

        print(f"Loaded {len(self.valid_images)} valid images from {annotations_file}")

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

        return {"pixel_values": pixel_values, "labels": target}


def collate_fn(batch):
    try:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        return {"pixel_values": pixel_values, "labels": labels}
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise e


def check_data_integrity(dataset):
    total_images = len(dataset.annotations['images'])
    valid_images = len(dataset.valid_images)
    missing_files = total_images - valid_images

    print(f"Total images in annotations: {total_images}")
    print(f"Valid images found: {valid_images}")
    print(f"Missing image files: {missing_files}")

    images_without_annotations = sum(1 for img in dataset.valid_images if img['id'] not in dataset.id_to_annotations)
    print(f"Images without annotations: {images_without_annotations}")

    if missing_files > 0:
        print("/nFirst 10 missing files:")
        for img in dataset.annotations['images'][:10]:
            if img['id'] not in dataset.id_to_filename:
                print(f" - {img['file_name']}")

    if images_without_annotations > 0:
        print("/nFirst 10 images without annotations:")
        for img in dataset.valid_images[:10]:
            if img['id'] not in dataset.id_to_annotations:
                print(f" - {img['file_name']}")

    print("/nDataset is ready for training with available data.")


def print_gpu_memory():
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


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


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def main():
    print("Starting training...")

    # Paths
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/images"
    annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_annotations_20241115_175519.json"

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 50
    learning_rate = 1e-5
    batch_size = 4
    image_size = (800, 800)

    # Early stopping and thresholds
    validation_loss_threshold = 0.1  # Desired validation loss threshold

    # TensorBoard writer
    writer = SummaryWriter(log_dir='./runs/detr_training')

    # Load pre-trained DETR model and processor
    print("Loading pre-trained DETR model and processor...")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,  # Ustawiamy num_labels na 1 dla jednej klasy
        ignore_mismatched_sizes=True
    )

    # Konfiguracja modelu dla jednej klasy
    model.config.id2label = {0: "surgical_tool"}
    model.config.label2id = {"surgical_tool": 0}
    model.config.num_labels = 1

    # Inicjalizacja warstwy klasyfikacyjnej
    num_channels = model.class_labels_classifier.in_features
    model.class_labels_classifier = torch.nn.Linear(num_channels, model.config.num_labels + 1)

    # Updated processor initialization
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={'shortest_edge': image_size[0], 'longest_edge': image_size[1]}
    )

    # Load dataset
    print("Loading dataset...")
    dataset = SurgicalToolDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        processor=processor,
        image_size=image_size,
        augment=True  # Augmentation can be enabled or disabled
    )

    # Check data integrity
    check_data_integrity(dataset)

    # Split dataset
    np.random.seed(42)  # For reproducibility
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    print(f"Dataset split into {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples.")

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

    test_loader = DataLoader(
        test_dataset,
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

    # Training loop
    best_val_loss = float('inf')

    try:
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}...")
            avg_train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)

            # Perform validation
            avg_val_loss = validate_epoch(model, val_loader, device, epoch, writer)

            # Adjust the learning rate based on validation loss
            scheduler.step(avg_val_loss)

            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                model.save_pretrained("./detr_tool_tracking_model_best")
                processor.save_pretrained("./detr_tool_tracking_model_best")
                print("Best model saved to: ./detr_tool_tracking_model_best")

            print(f"Epoch {epoch + 1} finished. Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Stop if validation loss is below threshold
            if avg_val_loss <= validation_loss_threshold:
                print(f"Validation loss has reached the threshold of {validation_loss_threshold}. Stopping training.")
                break

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving the current model...")
        model.save_pretrained("./detr_tool_tracking_model_interrupted")
        processor.save_pretrained("./detr_tool_tracking_model_interrupted")
        print("Interrupted model saved to: ./detr_tool_tracking_model_interrupted")

    # Save final model
    print("Saving final model and processor...")
    model.save_pretrained("./detr_tool_tracking_model_final")
    processor.save_pretrained("./detr_tool_tracking_model_final")
    print("Model and processor saved successfully.")
    print("Final model saved to: ./detr_tool_tracking_model_final")

    # Optional: Evaluate on test set
    print("Evaluating on test set...")
    test_loss = validate_epoch(model, test_loader, device, epoch, writer)
    print(f"Test Loss: {test_loss:.4f}")

    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
