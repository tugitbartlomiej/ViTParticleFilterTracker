#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    """
    Implementacja sinusoidalnego kodowania pozycji.
    
    Args:
        num_pos_feats (int): wymiar kodowania pozycji
        temperature (float): temperatura dla kodowania
        normalize (bool): czy normalizować pozycje
        scale (float): skala dla kodowania pozycji
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is provided")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Forward pass dla kodowania pozycji.
        
        Args:
            x (Tensor): tensor wejściowy [batch_size, channels, height, width]
            
        Returns:
            Tensor: kodowanie pozycji [batch_size, channels, height, width]
        """
        # Nie potrzebujemy gradientów dla kodowania pozycji
        not_mask = torch.ones_like(x[:, 0])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Sprawdzamy, czy wymiar kanałów jest zgodny z oczekiwanym wymiarem modelu
        if pos.shape[1] != self.num_pos_feats * 2:
            # Jeśli wymiary nie są zgodne, dostosowujemy tensor przez padding lub obcięcie
            if pos.shape[1] < self.num_pos_feats * 2:
                # Padding, jeśli tensor jest za mały
                padding = torch.zeros(pos.shape[0], self.num_pos_feats * 2 - pos.shape[1], 
                                      pos.shape[2], pos.shape[3], device=pos.device)
                pos = torch.cat([pos, padding], dim=1)
            else:
                # Obcięcie, jeśli tensor jest za duży
                pos = pos[:, :self.num_pos_feats * 2, :, :]
                
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Implementacja wyuczalnego kodowania pozycji.
    
    Args:
        num_pos_feats (int): wymiar kodowania pozycji
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        """
        Forward pass dla wyuczalnego kodowania pozycji.
        
        Args:
            x (Tensor): tensor wejściowy [batch_size, channels, height, width]
            
        Returns:
            Tensor: kodowanie pozycji [batch_size, channels, height, width]
        """
        h, w = x.shape[-2:]
        
        # Ograniczamy wymiary, aby uniknąć wykraczania poza zakres embeddingów
        h = min(h, self.row_embed.num_embeddings)
        w = min(w, self.col_embed.num_embeddings)
        
        # Tworzymy indeksy dla embeddingów
        i = torch.arange(h, device=x.device)
        j = torch.arange(w, device=x.device)
        
        # Pobieramy embeddingu dla odpowiednich indeksów
        x_emb = self.col_embed(j)
        y_emb = self.row_embed(i)
        
        # Tworzymy siatkę 2D z embeddingów
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        return pos