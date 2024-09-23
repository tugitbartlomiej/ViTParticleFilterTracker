# models/tool_tip_tracker.py

import torch
import torch.nn as nn
from torchvision.models.detection import detr_resnet50


class DETRTracker(nn.Module):
    def __init__(self, num_classes):
        super(DETRTracker, self).__init__()
        # Inicjalizacja modelu DETR z pretrenowanymi wagami
        self.model = detr_resnet50(weights='DEFAULT', num_classes=num_classes)

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): Lista obrazów, każdy o kształcie [C, H, W].
            targets (list[Dict], optional): Lista targetów, jeden dla każdego obrazu, zawierających:
                - boxes (Tensor[N, 4]): prawdziwe ramki w formacie [xmin, ymin, xmax, ymax]
                - labels (Tensor[N]): etykiety klas dla każdej prawdziwej ramki
        Returns:
            Jeśli `targets` nie jest `None`, zwraca słownik strat.
            Jeśli `targets` jest `None`, zwraca wykrycia.
        """
        return self.model(images, targets)
