#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    """
    Łączy wiele transformacji w jedną.
    
    Args:
        transforms (list): lista transformacji
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Normalize:
    """
    Normalizuje tensor obrazu.
    
    Args:
        mean (list): średnia dla normalizacji
        std (list): odchylenie standardowe dla normalizacji
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    """
    Konwertuje obraz PIL do tensora.
    """
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    """
    Losowe odbicie obrazu w poziomie.
    
    Args:
        prob (float): prawdopodobieństwo odbicia
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Odbij obraz
            image = F.hflip(image)
            
            # Odbij boxy
            if "boxes" in target:
                boxes = target["boxes"]
                # Odbicie współrzędnej x_center: x_new = 1 - x_old
                boxes[:, 0] = 1 - boxes[:, 0]
                target["boxes"] = boxes
                
        return image, target


class RandomResize:
    """
    Losowa zmiana rozmiaru obrazu.
    
    Args:
        min_size (int): minimalny rozmiar
        max_size (int): maksymalny rozmiar
    """
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # Wybierz losowy rozmiar
        size = random.randint(self.min_size, self.max_size)
        
        # Zmień rozmiar obrazu
        image = F.resize(image, size)
        
        # Zaktualizuj informacje o rozmiarze
        h, w = image.size
        target["size"] = torch.tensor([h, w])
        
        return image, target


class FixedResize:
    """
    Zmiana rozmiaru obrazu do określonego, stałego rozmiaru.
    
    Args:
        height (int): wysokość docelowa
        width (int): szerokość docelowa
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width
        
    def __call__(self, image, target):
        # Zmień rozmiar obrazu do stałego rozmiaru
        image = F.resize(image, (self.height, self.width))
        
        # Zaktualizuj informacje o rozmiarze
        target["size"] = torch.tensor([self.height, self.width])
        
        return image, target


def get_transforms(fixed_size=False):
    """
    Zwraca transformacje dla treningu i walidacji.
    
    Args:
        fixed_size (bool): czy używać stałego rozmiaru obrazów
        
    Returns:
        tuple: (transform_train, transform_val)
    """
    # Normalizacja
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if fixed_size:
        # Używamy stałego rozmiaru dla wszystkich obrazów
        height, width = 640, 640  # Standardowy rozmiar dla detekcji obiektów
        
        transform_train = Compose([
            RandomHorizontalFlip(),
            FixedResize(height=height, width=width),
            ToTensor(),
            normalize,
        ])
        
        transform_val = Compose([
            FixedResize(height=height, width=width),
            ToTensor(),
            normalize,
        ])
    else:
        # Losowe rozmiary dla treningu
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        transform_train = Compose([
            RandomHorizontalFlip(),
            RandomResize(min_size=scales[0], max_size=scales[-1]),
            ToTensor(),
            normalize,
        ])
        
        # Transformacje dla walidacji (bez augmentacji)
        transform_val = Compose([
            ToTensor(),
            normalize,
        ])
    
    return transform_train, transform_val