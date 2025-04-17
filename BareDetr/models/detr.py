#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class DETR(nn.Module):
    """
    DETR model implementation.
    
    Args:
        backbone (nn.Module): backbone model (e.g., ResNet)
        transformer (nn.Module): transformer model
        num_classes (int): number of object classes
        num_queries (int): number of object queries
        hidden_dim (int): hidden dimension size
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Define projection layers - dodajemy projekcję z backbone channels do hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
        # Define output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object" class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 for (x, y, w, h)
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
    def forward(self, samples):
        """
        Forward pass through DETR.
        
        Args:
            samples (NestedTensor): batched images and their padding masks
            
        Returns:
            dict: contains 'pred_logits' and 'pred_boxes'
        """
        # Extract features from backbone
        features, pos = self.backbone(samples)
        
        # Get the last feature map (highest resolution)
        src = features[-1]
        
        # Project features to transformer dimensions
        src_proj = self.input_proj(src)
        
        # Get position encodings - dodajemy projekcję dla pozycji, jeśli wymiary się nie zgadzają
        pos_embed = pos[-1]
        if pos_embed.shape[1] != self.hidden_dim:
            # Dodajemy projekcję pozycji do odpowiedniego wymiaru
            pos_embed = pos_embed[:, :self.hidden_dim, :, :]
        
        # Flatten spatial dimensions
        bs, c, h, w = src_proj.shape
        src_flatten = src_proj.flatten(2).permute(0, 2, 1)  # [batch_size, h*w, hidden_dim]
        pos_flatten = pos_embed.flatten(2).permute(0, 2, 1)  # [batch_size, h*w, hidden_dim]
        
        # Get query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [batch_size, num_queries, hidden_dim]
        
        # Pass through transformer
        # The core component where encoder and attention mechanisms can be modified
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(src_flatten, tgt, query_embed, pos_flatten)
        
        # Output predictions
        outputs_class = self.class_embed(hs)  # [batch_size, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [batch_size, num_queries, 4]
        
        out = {
            'pred_logits': outputs_class[-1],  # Last decoder layer
            'pred_boxes': outputs_coord[-1],   # Last decoder layer
            'aux_outputs': self._set_aux_loss(outputs_class, outputs_coord),
            'encoder_outputs': self.transformer.encoder_outputs  # Exposing encoder outputs for analysis
        }
        
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """
        This is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [{'pred_logits': a, 'pred_boxes': b} 
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """
    Simple multi-layer perceptron.
    
    Args:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        output_dim (int): output dimension
        num_layers (int): number of layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x