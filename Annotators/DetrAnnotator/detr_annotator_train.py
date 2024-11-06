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

        image = Image.open(image_path).convert("RGB")
        annotations = self.id_to_annotations.get(image_id, [])

        if self.augmentations:
            image = self.augmentations(image)

        coco_annotations = {
            'image_id': image_id,
            'annotations': annotations
        }

        encoding = self.processor(
            images=image,
            annotations=[coco_annotations],
            return_tensors="pt",
            size=self.image_size
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": target}


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def train_epoch(model, data_loader, optimizer, device, epoch, scheduler, writer=None):
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
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader) + batch_idx)

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

    scheduler.step(total_loss / len(data_loader))
    return total_loss / len(data_loader)


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


def main():
    print("Starting training...")

    # Paths
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/Raw_Images"
    annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/annotations.json"

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print("Training will be performed on GPU.")
    else:
        print("Training will be performed on CPU.")

    num_epochs = 15
    learning_rate = 1e-5  # Updated learning rate
    batch_size = 8  # Increased batch size
    image_size = (800, 800)

    # TensorBoard writer
    writer = SummaryWriter(log_dir='./runs/detr_training')

    # Load pre-trained DETR model and image processor
    print("Loading pre-trained DETR model and processor...")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load dataset
    print("Loading dataset...")
    dataset = SurgicalToolDataset(images_dir, annotations_file, processor, image_size=image_size, augment=True)

    # Split dataset into train and validation sets
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Check data integrity
    check_data_integrity(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, scheduler, writer)

        # Optionally: Evaluate on validation set (not implemented, but can be added)

        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # Save model and processor
    print("Saving model and processor...")
    model.save_pretrained("./detr_tool_tracking_model")
    processor.save_pretrained("./detr_tool_tracking_model")
    print("Model and processor saved successfully.")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
