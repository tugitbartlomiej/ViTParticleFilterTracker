import torch
from torch.utils.data import DataLoader, random_split
from transformers import DetrForObjectDetection, DetrImageProcessor
import os
import json
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


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

        # Map all category_ids to 0
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
        print("\nFirst 10 missing files:")
        for img in dataset.annotations['images'][:10]:
            if img['id'] not in dataset.id_to_filename:
                print(f" - {img['file_name']}")

    if images_without_annotations > 0:
        print("\nFirst 10 images without annotations:")
        for img in dataset.valid_images[:10]:
            if img['id'] not in dataset.id_to_annotations:
                print(f" - {img['file_name']}")

    print("\nDataset is ready for training with available data.")


def train_epoch(model, data_loader, optimizer, device, epoch, scheduler, writer=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader) + batch_idx)

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    scheduler.step(avg_loss)
    return avg_loss


def main():
    print("Starting training...")

    # Paths
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/output_frames/raw_images/76_100"
    annotations_file = (
        "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/output"
        "/coco_annotations_75_100.json"
    )

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print("Training will be performed on GPU.")
    else:
        print("Training will be performed on CPU.")

    num_epochs = 100
    learning_rate = 1e-5
    batch_size = 8
    image_size = (800, 800)

    # TensorBoard writer
    writer = SummaryWriter(log_dir='./runs/detr_training')

    # Load pre-trained DETR model and processor
    print("Loading pre-trained DETR model and processor...")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,  # Setting one class
        ignore_mismatched_sizes=True
    )

    # Configure the model for one class
    model.config.id2label = {0: "surgical_tool"}
    model.config.label2id = {"surgical_tool": 0}
    model.config.num_labels = 1

    # Initialize weights for the classifier
    num_channels = model.class_labels_classifier.in_features
    model.class_labels_classifier = torch.nn.Linear(num_channels, model.config.num_labels + 1)

    # Adjust the criterion's empty_weight
    if hasattr(model, 'criterion') and hasattr(model.criterion, 'empty_weight'):
        model.criterion.empty_weight = torch.ones(model.config.num_labels + 1)
        model.criterion.empty_weight[0] = 0.1  # Lower weight for the background
        model.criterion.empty_weight = model.criterion.empty_weight.to(device)

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
        augment=True
    )

    # Check data integrity
    check_data_integrity(dataset)

    # Split dataset
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Prepare data loaders with num_workers=0 for debugging
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to debug DataLoader issues
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

    # Prepare parameters for optimization with different learning rates
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": learning_rate / 10,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": learning_rate,
        },
    ]

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5
    )

    # Move model to device
    model.to(device)

    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, scheduler, writer)

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss: {best_loss:.4f}. Saving model...")
            model.save_pretrained("./detr_tool_tracking_model_best")
            processor.save_pretrained("./detr_tool_tracking_model_best")

        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

        # Early stopping check
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print("Learning rate too small. Stopping training...")
            break

    # Save final model
    print("Saving final model and processor...")
    model.save_pretrained("./detr_tool_tracking_model_final")
    processor.save_pretrained("./detr_tool_tracking_model_final")
    print("Model and processor saved successfully.")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
