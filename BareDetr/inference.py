#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from data.dataset import CocoDetectionDataset
from data.transforms import get_transforms
from models.backbone import build_backbone
from models.detr import DETR
from models.transformer import build_transformer


def get_args_parser():
    parser = argparse.ArgumentParser('DETR inference script')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Annotation file')
    parser.add_argument('--img_folder', default='images', type=str,
                        help='Image folder')
    
    # Output parameters
    parser.add_argument('--output_dir', default='inference_results', type=str,
                        help='Path to save inference results')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--threshold', default=0.7, type=float,
                        help='Detection confidence threshold')
    
    return parser


def box_cxcywh_to_xyxy(x):
    """
    Konwertuje boxy z formatu [x_center, y_center, width, height] 
    do formatu [x_min, y_min, x_max, y_max].
    
    Args:
        x (Tensor): boxy w formacie [x_center, y_center, width, height]
        
    Returns:
        Tensor: boxy w formacie [x_min, y_min, x_max, y_max]
    """
    x_c, y_c, w, h = x.unbind(-1)
    
    # Opcjonalne ograniczenie wielkości bounding boxów jeśli są zbyt duże
    # Zakładamy, że końcówka narzędzia jest mała, więc limitujemy maksymalną wielkość
    max_size = 0.2  # maksymalnie 20% wymiaru obrazu
    w = torch.clamp(w, max=max_size)
    h = torch.clamp(h, max=max_size)
    
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def visualize_predictions(image, outputs, threshold=0.7, class_names=None):
    """
    Wizualizuje predykcje modelu na obrazie.
    
    Args:
        image (PIL.Image): obraz
        outputs (dict): wyjścia modelu
        threshold (float): próg pewności detekcji
        class_names (list): nazwy klas
        
    Returns:
        PIL.Image: obraz z narysowanymi bounding boxami
    """
    # Domyślne nazwy klas, jeśli nie podano
    if class_names is None:
        class_names = ['background', 'tooltip']
    
    # Konwertuj obraz do formatu PIL, jeśli jest w innym formacie
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)
    
    # Stwórz kopię obrazu do rysowania
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Pobierz wyniki z modelu
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Ignorujemy klasę "no object"
    keep = probas.max(-1).values > threshold
    
    # Konwertuj boxy do formatu xyxy z ograniczaniem wielkości
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][0, keep])
    
    # Skaluj boxy do wymiarów obrazu
    width, height = image.size
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    
    # Opcjonalne dodatkowe ograniczenie rozmiaru bezpośrednio w pikselach
    # Zakładamy, że końcówka narzędzia nie powinna być większa niż 150px w obu wymiarach
    max_pixel_size = 150
    for i in range(len(boxes)):
        box_width = boxes[i, 2] - boxes[i, 0]
        box_height = boxes[i, 3] - boxes[i, 1]
        
        if box_width > max_pixel_size:
            center_x = (boxes[i, 0] + boxes[i, 2]) / 2
            boxes[i, 0] = center_x - max_pixel_size / 2
            boxes[i, 2] = center_x + max_pixel_size / 2
            
        if box_height > max_pixel_size:
            center_y = (boxes[i, 1] + boxes[i, 3]) / 2
            boxes[i, 1] = center_y - max_pixel_size / 2
            boxes[i, 3] = center_y + max_pixel_size / 2
    
    # Pobierz klasy i pewność detekcji
    scores, labels = probas[keep].max(-1)
    
    # Rysuj bounding boxy
    for box, label, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = box.tolist()
        
        # Klasa detekowanego obiektu
        class_name = class_names[label.item() + 1]  # +1 bo ignorujemy tło
        
        # Rysuj bbox
        draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
        
        # Dodaj etykietę z nazwą klasy i pewnością
        label_text = f"{class_name}: {score.item():.2f}"
        draw.text((x0, y0), label_text, fill='red')
    
    return draw_image


def preprocess_image(image_path):
    """
    Preprocessuje obraz do formatu wymaganego przez model.
    
    Args:
        image_path (str): ścieżka do obrazu
        
    Returns:
        Tensor: preprocessowany obraz
    """
    # Wczytaj obraz jako PIL.Image
    image = Image.open(image_path).convert("RGB")
    
    # Transformacje dla inferencji
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Preprocessuj obraz
    processed_image = transform(image)
    
    # Zapisz oryginalny obraz do użycia w wizualizacji
    original_image = image
    
    return processed_image, original_image


def main(args):
    # Utwórz katalog wynikowy
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ustawienia urządzenia
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Wczytaj model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args'] if 'args' in checkpoint else None
    
    # Jeśli brak argumentów modelu w checkpoincie, użyj domyślnych
    if model_args is None:
        model_args = argparse.Namespace(
            backbone='resnet50',
            num_classes=2,
            num_queries=100,  # Można zmniejszyć do 10-20 dla pojedynczego obiektu
            hidden_dim=256,
            nheads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            pre_norm=False
        )
    
    # Zbuduj model
    backbone = build_backbone(model_args)
    transformer = build_transformer(model_args)
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=model_args.num_classes,
        num_queries=model_args.num_queries,
        hidden_dim=model_args.hidden_dim
    )
    
    # Wczytaj wagi modelu
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Dodajemy informację o debugowaniu bounding boxów
    print("UWAGA: Zastosowano ograniczenie wielkości bounding boxów do 20% obrazu")
    print("Maksymalny rozmiar w pikselach ustawiono na 150px")
    
    # Wczytaj dataset
    print("Loading dataset...")
    # Używamy fixed_size=True aby zapewnić spójność rozmiarów
    transform_val, _ = get_transforms(fixed_size=True)
    
    # Przygotuj ścieżki
    img_folder_path = os.path.join(args.data_path, args.img_folder)
    ann_file_path = os.path.join(args.data_path, args.ann_file)
    
    # Sprawdź, czy plik anotacji istnieje
    if not os.path.exists(ann_file_path):
        print(f"Warning: Annotation file {ann_file_path} does not exist!")
        json_files = [f for f in os.listdir(args.data_path) if f.endswith('.json')]
        if json_files:
            args.ann_file = json_files[0]
            ann_file_path = os.path.join(args.data_path, args.ann_file)
            print(f"Using first available JSON file: {args.ann_file}")
        else:
            print("No JSON files found in data directory.")
            return
    
    # Wczytaj anotacje COCO
    with open(ann_file_path, 'r') as f:
        coco_data = json.load(f)
    
    # Pobierz nazwy klas z anotacji
    categories = coco_data.get('categories', [])
    class_names = ['background'] + [cat['name'] for cat in categories]
    
    # Pobierz wszystkie zdjęcia z datasetu
    dataset = CocoDetectionDataset(
        img_folder=img_folder_path,
        ann_file=ann_file_path,
        transforms=transform_val
    )
    
    print(f"Found {len(dataset)} images in dataset.")
    
    # Iteruj po wszystkich obrazach w datasecie
    for idx in range(len(dataset)):
        image_tensor, target = dataset[idx]
        
        # Pobierz oryginalną ścieżkę do obrazu
        img_id = target.get('image_id', idx)
        img_info = next((img for img in coco_data['images'] if img['id'] == img_id), None)
        if img_info is None:
            print(f"Warning: No image info found for image_id {img_id}, skipping...")
            continue
        
        img_path = os.path.join(img_folder_path, img_info['file_name'])
        
        # Wczytaj oryginalny obraz
        original_img = Image.open(img_path).convert("RGB")
        
        # Przygotuj obraz do inferencji
        with torch.no_grad():
            # Dodaj wymiar batch
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Wykonaj inferencję
            outputs = model(image_tensor)
            
            # Wizualizuj wyniki
            result_img = visualize_predictions(
                original_img, 
                outputs, 
                threshold=args.threshold,
                class_names=class_names
            )
            
            # Zapisz wynik
            output_path = os.path.join(args.output_dir, f"result_{os.path.basename(img_path)}")
            result_img.save(output_path)
            print(f"Saved result to {output_path}")
    
    print(f"Inference completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)