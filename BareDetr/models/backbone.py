#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict

from models.position_encoding import PositionEmbeddingSine


class FrozenBatchNorm2d(nn.Module):
    """
    Zamrożona warstwa BatchNorm2d, która zawsze używa średnich i wariancji treningowych.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # Operacja bez gradientów
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """
    Klasa bazowa dla sieci backbone.
    
    Args:
        backbone (nn.Module): sieć backbone
        train_backbone (bool): czy trenować backbone
        num_channels (int): liczba kanałów wyjściowych
        return_interm_layers (bool): czy zwracać warstwy pośrednie
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        
        # Określamy które warstwy wyjściowe mają być zwracane
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
            
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        
        # Czy trenować parametry backbone
        if not train_backbone:
            for name, parameter in backbone.named_parameters():
                parameter.requires_grad_(False)
        
        # Inicjalizacja kodowania pozycji
        self.position_embedding = PositionEmbeddingSine(num_channels // 2, normalize=True)

    def forward(self, tensor_list):
        """
        Forward pass dla backbone.
        
        Args:
            tensor_list (NestedTensor): batched images
            
        Returns:
            tuple: (cechy, pozycje)
        """
        xs = self.body(tensor_list)
        
        # Konwersja z Dict[str, Tensor] na listę
        out = []
        pos = []
        
        for name, x in xs.items():
            out.append(x)
            # Kodowanie pozycji dla każdej mapy cech
            pos.append(self.position_embedding(x))
            
        return out, pos


class Backbone(BackboneBase):
    """
    Implementacja ResNet jako backbone dla DETR.
    
    Args:
        name (str): wersja ResNet, np. 'resnet50'
        train_backbone (bool): czy trenować backbone
        return_interm_layers (bool): czy zwracać warstwy pośrednie
        dilation (bool): czy używać dilation
    """
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        # Ustawiamy domyślnie bez dilacji
        replace_stride_with_dilation = None
        
        # Jeśli dilation jest True, ustawiamy dilację w ostatnich dwóch warstwach
        if dilation:
            replace_stride_with_dilation = [False, False, True]
            
        # Liczba kanałów dla każdej wersji ResNet
        backbone_channels = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
        }
        
        # Tworzymy model ResNet
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=replace_stride_with_dilation,
            pretrained=True,
            norm_layer=FrozenBatchNorm2d
        )
        
        assert name in backbone_channels, f"Backbone {name} not supported"
        num_channels = backbone_channels[name]
        
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def build_backbone(args):
    """
    Buduje backbone na podstawie argumentów.
    
    Args:
        args: argumenty z parsera
        
    Returns:
        Backbone: model backbone
    """
    train_backbone = True
    return_interm_layers = False
    dilation = False
    
    backbone = Backbone(
        args.backbone,
        train_backbone,
        return_interm_layers,
        dilation
    )
    
    return backbone