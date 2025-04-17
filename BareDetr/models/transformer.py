#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional

class Transformer(nn.Module):
    """
    Transformer model for DETR.
    Bardzo elastyczna implementacja transformera, umożliwiająca łatwą modyfikację
    encodera i głów atencji.
    
    Args:
        d_model (int): hidden dimension
        nhead (int): number of attention heads
        num_encoder_layers (int): number of encoder layers
        num_decoder_layers (int): number of decoder layers
        dim_feedforward (int): dimension of feedforward network
        dropout (float): dropout rate
        activation (str): activation function name
        normalize_before (bool): whether to normalize before attention
        return_intermediate_dec (bool): whether to return intermediate decoder outputs
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # Enkoder - tutaj możesz modyfikować głowy atencji
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Dekoder - również możesz modyfikować jego architekturę
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # Dodajemy atrybut do przechowywania wyjść enkodera dla późniejszej analizy
        self.encoder_outputs = None
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, query_pos, pos_embed):
        """
        Forward pass for transformer.
        
        Args:
            src (Tensor): source features [batch_size, seq_len, hidden_dim]
            tgt (Tensor): target features [batch_size, num_queries, hidden_dim]
            query_pos (Tensor): query position encodings [batch_size, num_queries, hidden_dim]
            pos_embed (Tensor): position encodings [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tensor: output of decoder
        """
        # Sprawdzamy zgodność wymiarów
        if src.size(2) != self.d_model:
            # Jeśli wymiary się nie zgadzają, dostosowujemy src
            src = self._adjust_dimension(src, self.d_model)
            
        if pos_embed.size(2) != self.d_model:
            # Jeśli wymiary się nie zgadzają, dostosowujemy pos_embed
            pos_embed = self._adjust_dimension(pos_embed, self.d_model)
            
        if query_pos.size(2) != self.d_model:
            # Jeśli wymiary się nie zgadzają, dostosowujemy query_pos
            query_pos = self._adjust_dimension(query_pos, self.d_model)
            
        if tgt.size(2) != self.d_model:
            # Jeśli wymiary się nie zgadzają, dostosowujemy tgt
            tgt = self._adjust_dimension(tgt, self.d_model)
        
        bs, hw, c = src.shape
        src = src.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        tgt = tgt.transpose(0, 1)  # [num_queries, batch_size, hidden_dim]
        query_pos = query_pos.transpose(0, 1)  # [num_queries, batch_size, hidden_dim]
        pos_embed = pos_embed.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        
        # Encoder
        memory = self.encoder(src, pos=pos_embed)
        
        # Zapisanie wyjść enkodera dla późniejszej analizy
        self.encoder_outputs = memory.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Decoder
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_pos)
        
        # Zwrócenie wyjścia w formacie [batch_size, num_queries, hidden_dim]
        return hs.transpose(1, 2)
    
    def _adjust_dimension(self, tensor, target_dim):
        """
        Dostosowuje wymiar tensora do wymaganego wymiaru.
        
        Args:
            tensor (Tensor): tensor do dostosowania
            target_dim (int): wymagany wymiar
            
        Returns:
            Tensor: dostosowany tensor
        """
        curr_dim = tensor.size(2)
        
        if curr_dim < target_dim:
            # Dodajemy padding, jeśli tensor jest za mały
            padding = torch.zeros(*tensor.shape[:2], target_dim - curr_dim, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=2)
        elif curr_dim > target_dim:
            # Obcinamy, jeśli tensor jest za duży
            tensor = tensor[:, :, :target_dim]
            
        return tensor


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder składa się z N warstw encodera.
    
    Args:
        encoder_layer (nn.Module): warstwa encodera
        num_layers (int): liczba warstw
        norm (nn.Module, optional): normalizacja
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
        # Dodajemy atrybuty do analizy wag i działania głów atencji
        self.attention_weights = []
        self.attention_outputs = []

    def forward(self, src, mask=None, pos=None):
        """
        Forward pass dla encodera.
        
        Args:
            src (Tensor): input z shape [seq_len, batch_size, hidden_dim]
            mask (Tensor, optional): maska dla src
            pos (Tensor, optional): pozycje dla src
            
        Returns:
            Tensor: wyjście encodera
        """
        output = src
        
        # Czyszczenie poprzednich wyników
        self.attention_weights = []
        self.attention_outputs = []
        
        # Iteracja przez warstwy encodera
        for layer in self.layers:
            output, attn_weights, attn_output = layer(output, src_mask=mask, pos=pos)
            
            # Zapisywanie wyników dla analizy
            self.attention_weights.append(attn_weights)
            self.attention_outputs.append(attn_output)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output


class TransformerDecoder(nn.Module):
    """
    TransformerDecoder składa się z N warstw dekodera.
    
    Args:
        decoder_layer (nn.Module): warstwa dekodera
        num_layers (int): liczba warstw
        norm (nn.Module, optional): normalizacja
        return_intermediate (bool): czy zwracać wyjścia pośrednie
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                pos=None, query_pos=None):
        """
        Forward pass dla dekodera.
        
        Args:
            tgt (Tensor): input z shape [num_queries, batch_size, hidden_dim]
            memory (Tensor): wyjście encodera [seq_len, batch_size, hidden_dim]
            tgt_mask (Tensor, optional): maska dla tgt
            memory_mask (Tensor, optional): maska dla memory
            pos (Tensor, optional): pozycje dla memory
            query_pos (Tensor, optional): pozycje dla tgt
            
        Returns:
            Tensor: wyjście dekodera
        """
        output = tgt
        
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                
        if self.return_intermediate:
            return torch.stack(intermediate)
            
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    Pojedyncza warstwa TransformerEncoder z modyfikowalnymi głowami atencji.
    
    Args:
        d_model (int): wymiar modelu
        nhead (int): liczba głów atencji
        dim_feedforward (int): wymiar warstwy feed forward
        dropout (float): współczynnik dropout
        activation (str): funkcja aktywacji
        normalize_before (bool): czy normalizować przed czy po warstwie
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        # Uwaga! To jest element, który będziesz modyfikować
        # CustomMultiheadAttention pozwala na łatwą modyfikację mechanizmu atencji
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementacja FFN (Feed Forward Network)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        """Dodaje pozycje do tensora jeśli są dostępne."""
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, pos=None):
        """
        Forward pass dla warstwy encodera.
        
        Args:
            src (Tensor): input tensor
            src_mask (Tensor, optional): maska
            pos (Tensor, optional): pozycje
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (output, attention_weights, attention_output)
        """
        q = k = self.with_pos_embed(src, pos)
        
        # Czy normalizować przed czy po
        if self.normalize_before:
            src2, attn_weights = self.self_attn(self.norm1(q), self.norm1(k), self.norm1(src), attn_mask=src_mask)
        else:
            src2, attn_weights = self.self_attn(q, k, src, attn_mask=src_mask)
            
        # Zapisujemy output atencji przed residual connection i dropout
        attn_output = src2.clone()
        
        # Residual connection i dropout
        src = src + self.dropout1(src2)
        
        # Druga część warstwy encodera (Feed Forward)
        if self.normalize_before:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
            
        return src, attn_weights, attn_output


class TransformerDecoderLayer(nn.Module):
    """
    Pojedyncza warstwa TransformerDecoder.
    
    Args:
        d_model (int): wymiar modelu
        nhead (int): liczba głów atencji
        dim_feedforward (int): wymiar warstwy feed forward
        dropout (float): współczynnik dropout
        activation (str): funkcja aktywacji
        normalize_before (bool): czy normalizować przed czy po warstwie
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross-attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        """Dodaje pozycje do tensora jeśli są dostępne."""
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                pos=None, query_pos=None):
        """
        Forward pass dla warstwy dekodera.
        
        Args:
            tgt (Tensor): input tensor
            memory (Tensor): pamięć z encodera
            tgt_mask (Tensor, optional): maska dla tgt
            memory_mask (Tensor, optional): maska dla memory
            pos (Tensor, optional): pozycje dla memory
            query_pos (Tensor, optional): pozycje dla query
            
        Returns:
            Tensor: wyjście warstwy
        """
        q = self.with_pos_embed(tgt, query_pos)
        
        # Self-attention
        if self.normalize_before:
            tgt2 = self.self_attn(self.norm1(q), self.norm1(q), self.norm1(tgt), attn_mask=tgt_mask)[0]
        else:
            tgt2 = self.self_attn(q, q, tgt, attn_mask=tgt_mask)[0]
            
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)
            
        # Cross-attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        
        if self.normalize_before:
            tgt2 = self.multihead_attn(self.norm2(q), self.norm2(k), self.norm2(memory), attn_mask=memory_mask)[0]
        else:
            tgt2 = self.multihead_attn(q, k, memory, attn_mask=memory_mask)[0]
            
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)
            
        # Feed Forward
        if self.normalize_before:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        else:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
            
        return tgt


class CustomMultiheadAttention(nn.Module):
    """
    Zmodyfikowana implementacja wielogłowej atencji, którą można łatwo modyfikować.
    Ta klasa pozwala na łatwe eksperymentowanie z różnymi mechanizmami atencji.
    
    Args:
        embed_dim (int): wymiar embeddingu
        num_heads (int): liczba głów atencji
        dropout (float): współczynnik dropout
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Liniowe transformacje dla Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Projekcja wyjścia
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass dla CustomMultiheadAttention.
        
        Args:
            query (Tensor): tensor query [seq_len, batch, embed_dim]
            key (Tensor): tensor key [seq_len, batch, embed_dim]
            value (Tensor): tensor value [seq_len, batch, embed_dim]
            attn_mask (Tensor, optional): maska uwagi
            
        Returns:
            Tuple[Tensor, Tensor]: (output, attention_weights)
        """
        # Wymiary
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        # Transformacje projekcji
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape do formatu multihead
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Obliczenie scaled dot-product attention
        q = q / (self.head_dim ** 0.5)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        
        # Aplikacja maski jeśli została podana
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(0) == 0, float('-inf'))
            
        # Softmax po wymiarze src_len
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Agregacja wartości
        attn_output = torch.bmm(attn_weights, v)
        
        # Reshape z powrotem do oryginalnego formatu
        attn_output = attn_output.transpose(0, 1).reshape(tgt_len, bsz, embed_dim)
        
        # Projekcja wyjścia
        attn_output = self.out_proj(attn_output)
        
        # Zwracamy zarówno output jak i wagi atencji do analizy
        return attn_output, attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)
        

def _get_clones(module, N):
    """Tworzy N kopii modułu."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Zwraca funkcję aktywacji na podstawie nazwy."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation {activation} not supported")


def build_transformer(args):
    """
    Buduje model transformera na podstawie argumentów.
    
    Args:
        args: argumenty z parsera
        
    Returns:
        Transformer: model transformera
    """
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True
    )