# utils/tool_tip_dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
import cv2
import logging

class ToolTipDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform

        # Ustawienie logowania
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Ładowanie adnotacji
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Zakładamy, że adnotacje zawierają listę obiektów z kluczem 'image_id'
        self.image_annotations = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)

        # Ładowanie listy plików obrazów
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()  # Upewnij się, że są posortowane, jeśli to konieczne

        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_name)
        if image is None:
            raise ValueError(f"Failed to load image {img_name}")

        # Zakładamy, że adnotacje są powiązane z obrazem poprzez nazwę pliku bez rozszerzenia
        image_id = os.path.splitext(self.image_files[idx])[0]  # Przykład
        anns = self.image_annotations.get(image_id, [])

        # Przetwarzanie adnotacji
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            self.logger.info(f"Image {img_name} has no annotations.")

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if boxes.numel() > 0:
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            target['area'] = torch.tensor([0.0])

        target['iscrowd'] = torch.zeros(len(boxes), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        else:
            # Konwersja do Tensor i normalizacja
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target
