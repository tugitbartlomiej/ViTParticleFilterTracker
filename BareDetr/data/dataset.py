#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torchvision
from pycocotools.coco import COCO
from PIL import Image

class CocoDetectionDataset(torch.utils.data.Dataset):
    """
    Dataset dla detekcji obiektów w formacie COCO.
    
    Args:
        img_folder (str): ścieżka do folderu z obrazami
        ann_file (str): ścieżka do pliku z adnotacjami COCO
        transforms (callable, optional): transformacje do zastosowania na obrazach
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms
        
        # Wczytaj adnotacje COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Filtrujemy obrazy bez adnotacji
        ids_with_ann = set(_ann["image_id"] for _ann in self.coco.anns.values())
        self.ids = [img_id for img_id in self.ids if img_id in ids_with_ann]
        
        # Przypisujemy etykiety dla klas
        self.coco_to_target = {c: i for i, c in enumerate(sorted(self.coco.getCatIds()))}
        
        print(f"Loaded {len(self.ids)} images from COCO dataset")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Zwraca obraz i jego adnotacje.
        
        Args:
            idx (int): indeks obrazu
            
        Returns:
            tuple: (obraz, adnotacje)
        """
        # Pobierz informacje o obrazie
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info["file_name"])
        
        # Wczytaj obraz
        img = Image.open(img_path).convert("RGB")
        
        # Pobierz adnotacje
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Konwertuj adnotacje do formatu, którego oczekuje DETR
        boxes = []
        labels = []
        
        for ann in anns:
            if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
                continue
                
            # COCO bbox format: [x, y, width, height]
            # DETR bbox format: [x_center, y_center, width, height] normalized to [0, 1]
            x, y, w, h = ann["bbox"]
            x_center = x + w / 2
            y_center = y + h / 2
            
            # Normalizacja do [0, 1]
            x_center /= img_info["width"]
            y_center /= img_info["height"]
            w /= img_info["width"]
            h /= img_info["height"]
            
            boxes.append([x_center, y_center, w, h])
            labels.append(self.coco_to_target[ann["category_id"]])
        
        # Konwertuj na tensory
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Tworzenie docelowego słownika z adnotacjami
        target = {
            "image_id": torch.tensor([img_id]),
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.as_tensor([img_info["height"], img_info["width"]]),
            "size": torch.as_tensor([img_info["height"], img_info["width"]])
        }
        
        # Zastosuj transformacje jeśli są dostępne
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target