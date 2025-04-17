#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms.functional as F


def collate_fn(batch):
    """
    Funkcja do łączenia próbek w batch podczas ładowania danych.
    Dostosowuje rozmiary obrazów do jednolitego rozmiaru w ramach batch'a.
    
    Args:
        batch (list): lista elementów (image, target)
        
    Returns:
        tuple: (images, targets)
    """
    # Rozdziel obrazy i cele na osobne listy
    images = []
    targets = []
    
    # Znajdź maksymalny rozmiar w batch'u
    max_height = 0
    max_width = 0
    
    for img, tgt in batch:
        _, h, w = img.shape
        max_height = max(max_height, h)
        max_width = max(max_width, w)
    
    # Dostosuj wszystkie obrazy do jednolitego rozmiaru
    for img, tgt in batch:
        # Dostosuj obraz do maksymalnego rozmiaru przez padding
        _, h, w = img.shape
        pad_bottom = max_height - h
        pad_right = max_width - w
        
        # Zastosuj padding
        if pad_bottom > 0 or pad_right > 0:
            img = F.pad(img, (0, 0, pad_right, pad_bottom))
            
        images.append(img)
        targets.append(tgt)
        
    # Łącz obrazy w tensor
    images = torch.stack(images, dim=0)
    
    return images, targets