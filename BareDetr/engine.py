#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import time
import datetime
import math
from typing import Dict, List, Optional
from collections import defaultdict
import torch.nn.functional as F

def train_one_epoch(model, optimizer, data_loader, device, epoch, clip_max_norm=0.1):
    """
    Trenuje model przez jedną epokę.
    
    Args:
        model (nn.Module): model do trenowania
        optimizer (torch.optim.Optimizer): optymalizator
        data_loader (DataLoader): loader danych treningowych
        device (torch.device): urządzenie
        epoch (int): numer epoki
        clip_max_norm (float): maksymalna norma gradientów
        
    Returns:
        dict: statystyki treningu
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)
        
        # Przenieś cele na urządzenie
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Oblicz straty
        loss_dict = criterion(outputs, targets)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Oblicz błąd klasyfikacji
        loss_dict_unscaled = {k: v for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        loss_value = sum(loss_dict_scaled.values()).item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_unscaled)
            return

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # Aktualizuj metryki
        metric_logger.update(loss=loss_value, **loss_dict_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Możesz tutaj dodać wizualizację wag atencji i innych metryk
        
    # Agregacja metryk z całej epoki
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Ewaluuje model na zbiorze walidacyjnym.
    
    Args:
        model (nn.Module): model do ewaluacji
        data_loader (DataLoader): loader danych walidacyjnych
        device (torch.device): urządzenie
        
    Returns:
        dict: statystyki ewaluacji
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device)
        
        # Przenieś cele na urządzenie
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Oblicz straty
        loss_dict = criterion(outputs, targets)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        
        # Oblicz ważoną sumę strat
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        loss_value = sum(loss_dict_scaled.values()).item()
        
        # Aktualizuj metryki
        metric_logger.update(loss=loss_value, **loss_dict_scaled)

    # Agregacja metryk z ewaluacji
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def criterion(outputs, targets):
    """
    Funkcja kosztu dla modelu DETR.
    
    Args:
        outputs (dict): wyjścia modelu
        targets (list[dict]): cele
        
    Returns:
        dict: wartości strat
    """
    # Wyjścia klasyfikatora i predykcje boxów
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']
    
    # Przygotowanie indeksów i etykiet dla dopasowania
    indices = []
    for i, target in enumerate(targets):
        if len(target['boxes']) == 0:
            indices.append(([], []))
            continue
            
        # Proste dopasowanie: pierwsze N zapytań do pierwszych N boxów
        n = min(out_logits.shape[1], len(target['boxes']))
        indices.append((torch.arange(n), torch.arange(n)))
    
    # Strata klasyfikacji (cross-entropy)
    loss_ce = 0
    for i, (idx_pred, idx_tgt) in enumerate(indices):
        if len(idx_pred) == 0:
            continue
            
        target_classes = torch.full(out_logits[i].shape[:1], 0,
                                   dtype=torch.int64, device=out_logits.device)
        target_classes[idx_pred] = targets[i]['labels'][idx_tgt]
        
        loss_ce += F.cross_entropy(out_logits[i], target_classes)
    
    loss_ce = loss_ce / len(targets)
    
    # Strata lokalizacji (L1)
    loss_bbox = 0
    for i, (idx_pred, idx_tgt) in enumerate(indices):
        if len(idx_pred) == 0:
            continue
            
        # L1 loss dla boxów
        loss_bbox += F.l1_loss(out_bbox[i][idx_pred], targets[i]['boxes'][idx_tgt], reduction='none').sum() / len(idx_pred)
    
    loss_bbox = loss_bbox / len(targets)
    
    # GIoU loss
    loss_giou = 0
    for i, (idx_pred, idx_tgt) in enumerate(indices):
        if len(idx_pred) == 0:
            continue
            
        # Konwertuj boxy z formatu [x_center, y_center, width, height] 
        # do formatu [x_min, y_min, x_max, y_max]
        pred_boxes = xywh_to_xyxy(out_bbox[i][idx_pred])
        target_boxes = xywh_to_xyxy(targets[i]['boxes'][idx_tgt])
        
        # Oblicz GIoU loss
        giou = box_iou(pred_boxes, target_boxes)
        loss_giou += (1 - torch.diag(giou)).sum() / len(idx_pred)
    
    loss_giou = loss_giou / len(targets)
    
    return {
        'loss_ce': loss_ce,
        'loss_bbox': loss_bbox,
        'loss_giou': loss_giou
    }


def xywh_to_xyxy(boxes):
    """
    Konwertuje boxy z formatu [x_center, y_center, width, height] 
    do formatu [x_min, y_min, x_max, y_max].
    
    Args:
        boxes (Tensor): boxy w formacie [x_center, y_center, width, height]
        
    Returns:
        Tensor: boxy w formacie [x_min, y_min, x_max, y_max]
    """
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    Oblicza IoU (Intersection over Union) między dwoma zestawami boxów.
    
    Args:
        boxes1 (Tensor): pierwszy zestaw boxów [N, 4]
        boxes2 (Tensor): drugi zestaw boxów [M, 4]
        
    Returns:
        Tensor: macierz IoU [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def box_area(boxes):
    """
    Oblicza pole powierzchni boxów.
    
    Args:
        boxes (Tensor): boxy w formacie [N, 4] (xyxy)
        
    Returns:
        Tensor: pola powierzchni [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class SmoothedValue:
    """
    Śledzi serię wartości i zapewnia dostęp do wygładzonych wartości
    przez okno ruchome i globalną średnią.
    """
    def __init__(self, window_size=20, fmt=None):
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(self.deque)
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(self.deque, dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def __str__(self):
        if self.fmt is None:
            return self.avg
        return self.fmt.format(avg=self.avg, value=self.avg)


class MetricLogger:
    """
    Logger metryki dla treningu i ewaluacji.
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Dodane jako placeholder, ale nie jest niezbędne dla pojedynczego procesu.
        W środowisku wieloprocesorowym można tu dodać synchronizację.
        """
        pass

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))