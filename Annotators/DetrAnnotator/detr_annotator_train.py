import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
import os
import json
from PIL import Image
from tqdm.auto import tqdm


# Dataset class for training
class SurgicalToolDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, processor, image_size=(800, 800)):
        print("Initializing dataset...")
        self.images_dir = images_dir
        self.processor = processor
        self.image_size = image_size

        # Load annotations from JSON file
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        print(f"Loaded {len(self.annotations['images'])} images from {annotations_file}")

        # Map image filenames to their IDs
        self.filename_to_id = {ann['file_name']: ann['id'] for ann in self.annotations['images']}
        self.id_to_annotations = self._map_id_to_annotations()

    def _map_id_to_annotations(self):
        # Create a mapping from image ID to annotations
        id_to_ann = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in id_to_ann:
                id_to_ann[image_id] = []
            id_to_ann[image_id].append({
                'bbox': ann['bbox'],
                'category_id': ann['category_id'],
                'area': ann['area']
            })
        return id_to_ann

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        # Load image and corresponding annotations
        image_info = self.annotations['images'][idx]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.images_dir, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_filename} not found in {self.images_dir}")

        image = Image.open(image_path).convert("RGB")
        image_id = image_info['id']
        annotations = self.id_to_annotations.get(image_id, [])

        coco_annotations = {
            'image_id': image_id,
            'annotations': annotations
        }

        # Process the image and annotations for DETR
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


def train_epoch(model, data_loader, optimizer, device, epoch):
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

        # Log every few batches
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def main():
    print("Starting training...")

    # Paths
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/Raw_Images"
    annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/annotations.json"

    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10
    learning_rate = 1e-5
    batch_size = 4
    image_size = (800, 800)

    # Load pre-trained DETR model and image processor
    print("Loading pre-trained DETR model and processor...")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load dataset
    print("Loading dataset...")
    dataset = SurgicalToolDataset(images_dir, annotations_file, processor, image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        avg_loss = train_epoch(model, data_loader, optimizer, device, epoch)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # Save model and processor
    print("Saving model and processor...")
    model.save_pretrained("./detr_tool_tracking")
    processor.save_pretrained("./detr_tool_tracking")
    print("Model and processor saved successfully.")


if __name__ == "__main__":
    main()
