import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import json
import os
from tqdm.auto import tqdm


class SurgicalToolDataset(Dataset):
    def __init__(self, images_dir, annotations_file):
        self.images_dir = images_dir

        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            self.coco = json.load(f)

        # Create a mapping from filename to image_id
        self.filename_to_id = {img['file_name']: img['id'] for img in self.coco['images']}

        # Create a mapping from image_id to annotations
        self.image_id_to_annotations = {}
        for ann in self.coco['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # List of image filenames
        self.image_filenames = [img['file_name'] for img in self.coco['images']]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_id = self.filename_to_id[filename]
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert('RGB')

        annotations = self.image_id_to_annotations.get(image_id, [])

        # Ensure each annotation has 'area' and 'iscrowd'
        for ann in annotations:
            if 'area' not in ann:
                bbox = ann['bbox']
                ann['area'] = bbox[2] * bbox[3]  # width * height
            if 'iscrowd' not in ann:
                ann['iscrowd'] = 0  # Assume not crowded

        target = {
            'image_id': image_id,
            'annotations': annotations
        }

        return image, target


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    images_dir = 'E:/wspolne_labelowanie/valid/'
    annotations_file = 'E:/wspolne_labelowanie/valid/_annotations.coco.json'

    # Load model and processor
    model = DetrForObjectDetection.from_pretrained('./surgical_tool_detector')
    processor = DetrImageProcessor.from_pretrained('./surgical_tool_detector')

    # Prepare dataset and dataloader
    val_dataset = SurgicalToolDataset(images_dir, annotations_file)

    def collate_fn(batch):
        images, targets = list(zip(*batch))
        encoding = processor(images=list(images), annotations=list(targets), return_tensors="pt")
        pixel_values = encoding['pixel_values']
        labels = encoding['labels']
        return {'pixel_values': pixel_values, 'labels': labels}

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Evaluate model
    model.to(device)
    avg_loss = evaluate_model(model, val_dataloader, device)

    print(f'Validation Loss: {avg_loss:.4f}')
    print('mAP calculation is not implemented in this script.')


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            pixel_values = batch['pixel_values'].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


if __name__ == '__main__':
    main()
