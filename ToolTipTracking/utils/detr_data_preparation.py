import os
import json
import torch
from PIL import Image
from torchvision.transforms import functional as F
from pycocotools.coco import COCO

class DETRDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, coco_annotation_file, transform=None):
        self.img_folder = img_folder
        self.coco = COCO(coco_annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in coco_annotation:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.tensor([])
        target['iscrowd'] = torch.zeros(len(boxes), dtype=torch.int64)

        if self.transform:
            img, target = self.transform(img, target)
        else:
            img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.ids)
