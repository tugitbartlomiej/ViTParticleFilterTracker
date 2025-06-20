import argparse
import json
import math  # Dodany import
from pathlib import Path

import cv2  # Kluczowe do odczytu wideo i operacji na obrazach
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection

# --- Stałe dla kolorów (bez zmian) ---
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# --- Funkcje pomocnicze ---

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b

def plot_results(pil_img, scores, labels, boxes, id2label, font_path=None, font_size=15):
    """ Rysuje bounding boxy i etykiety tylko dla detekcji z wynikiem >= 0.8 """
    draw = ImageDraw.Draw(pil_img)
    try:
        if font_path and Path(font_path).exists(): font = ImageFont.truetype(font_path, font_size)
        else: font = ImageFont.load_default();
    except Exception as e: print(f"Warning: Error loading font {font_path}: {e}. Using default."); font = ImageFont.load_default()
    if font_path and not Path(font_path).exists() and not isinstance(font, ImageFont.FreeTypeFont): print(f"Warning: Font file not found at {font_path}. Using default font.")

    for score, label_id, box in zip(scores, labels, boxes):
        if score.item() >= 0.65: # Rysuj tylko jeśli pewność >= 80%
            box = [round(i, 2) for i in box.tolist()]
            label = id2label.get(label_id.item(), "N/A")
            color = COLORS[label_id.item() % len(COLORS)]; color_rgb = tuple(int(c * 255) for c in color)
            draw.rectangle(box, outline=color_rgb, width=3)
            text = f"{label}: {score:.2f}"
            if hasattr(font, 'getbbox'): text_bbox = font.getbbox(text); text_width=text_bbox[2]-text_bbox[0]; text_height=text_bbox[3]-text_bbox[1]
            elif hasattr(font, 'getsize'): text_width, text_height = font.getsize(text)
            else: text_width, text_height = 10*len(text), 12 # Fallback
            text_location = [box[0], box[1] - text_height - 2];
            if text_location[1] < 0: text_location[1] = box[1] + 1
            textbox_location = [text_location[0], text_location[1], text_location[0]+text_width+2, text_location[1]+text_height+2]
            draw.rectangle(textbox_location, fill=color_rgb)
            draw.text((text_location[0]+1, text_location[1]+1), text, fill="white", font=font)
    return pil_img

# Nowa funkcja pomocnicza do znajdowania dzielników
def find_closest_factors(n, target_ratio):
    """Znajduje parę dzielników (h, w) liczby n, taką że h*w=n
       i stosunek w/h jest najbliższy target_ratio."""
    if n <= 0: raise ValueError("n must be positive")
    if target_ratio <= 0: raise ValueError("target_ratio must be positive")

    best_h, best_w = 1, n
    min_diff = abs((n / 1.0) - target_ratio)

    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            # Sprawdź parę (h, w)
            ratio_hw = w / h
            diff_hw = abs(ratio_hw - target_ratio)
            if diff_hw < min_diff:
                min_diff = diff_hw
                best_h, best_w = h, w

            # Sprawdź parę (w, h) jeśli h*h != n
            if h * h != n:
                h2, w2 = w, h
                ratio_wh = w2 / h2
                diff_wh = abs(ratio_wh - target_ratio)
                if diff_wh < min_diff:
                    min_diff = diff_wh
                    best_h, best_w = h2, w2

    return best_h, best_w

# Zaktualizowana funkcja get_attention_heatmap
def get_attention_heatmap(model_outputs, pixel_values_shape, layer_idx=-1, head_aggregation="mean", query_aggregation="mean"):
    """Wyciąga i przetwarza mapy uwagi, dedukując wymiary mapy cech."""
    if not hasattr(model_outputs, 'cross_attentions') or model_outputs.cross_attentions is None:
        # print("Warning: 'cross_attentions' not found.") # Opcjonalnie można wyciszyć dla czystszego logu
        return None
    attentions = model_outputs.cross_attentions
    if layer_idx >= len(attentions): layer_idx = -1

    attn_layer = attentions[layer_idx]
    batch_size, num_heads, num_queries, seq_len = attn_layer.shape

    # --- ZMIENIONA LOGIKA DEDUKCJI WYMIARÓW ---
    _, _, h_proc, w_proc = pixel_values_shape
    if h_proc <= 0 or w_proc <= 0:
        print("Error: Invalid processed image dimensions in pixel_values_shape."); return None

    # Obsługa przypadku, gdy seq_len jest bardzo małe lub 0
    if seq_len <= 0:
        print(f"Warning: Invalid attention sequence length ({seq_len}). Skipping heatmap.")
        return None

    target_ratio = w_proc / h_proc

    try:
        h_feat, w_feat = find_closest_factors(seq_len, target_ratio)
        if h_feat * w_feat != seq_len:
             print(f"Error: Deduced factors {h_feat}x{w_feat} don't multiply to {seq_len}. Aborting heatmap.")
             return None
    except Exception as e:
        print(f"Error finding factors for seq_len {seq_len}: {e}"); return None
    # --- KONIEC ZMIENIONEJ LOGIKI ---

    if head_aggregation == "mean": attn_heads_agg = attn_layer.mean(dim=1)
    elif head_aggregation == "max": attn_heads_agg, _ = attn_layer.max(dim=1)
    else: attn_heads_agg = attn_layer.mean(dim=1)
    if query_aggregation == "mean": attn_queries_agg = attn_heads_agg.mean(dim=1)
    elif query_aggregation == "max": attn_queries_agg, _ = attn_heads_agg.max(dim=1)
    else: attn_queries_agg = attn_heads_agg.mean(dim=1)
    if batch_size == 0: return None

    heatmap_low_res = attn_queries_agg.reshape(batch_size, h_feat, w_feat)

    heatmap_low_res = heatmap_low_res.unsqueeze(1)
    target_height, target_width = pixel_values_shape[-2:]
    if target_height <= 0 or target_width <= 0: print("Warn: Invalid target dims for interpolation."); return None
    heatmap_high_res = torch.nn.functional.interpolate(heatmap_low_res, size=(target_height, target_width), mode='bilinear', align_corners=False)
    heatmap_high_res = heatmap_high_res.squeeze(1)
    for i in range(batch_size):
        min_val=torch.min(heatmap_high_res[i]); max_val=torch.max(heatmap_high_res[i])
        if max_val > min_val: heatmap_high_res[i] = (heatmap_high_res[i] - min_val) / (max_val - min_val)
        else: heatmap_high_res[i] = torch.zeros_like(heatmap_high_res[i])
    return heatmap_high_res

def overlay_heatmap_cv2(image_cv, heatmap_tensor, alpha=0.5, colormap=cv2.COLORMAP_JET):
    if heatmap_tensor is None: return image_cv
    heatmap_np = heatmap_tensor.cpu().numpy()
    heatmap_scaled = (heatmap_np * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, colormap)
    h, w, _ = image_cv.shape
    if h <= 0 or w <= 0: print("Warn: Invalid image dims for resizing heatmap."); return image_cv
    heatmap_resized = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)
    overlayed_image = cv2.addWeighted(image_cv, 1 - alpha, heatmap_resized, alpha, 0)
    return overlayed_image

# --- Funkcje przetwarzające różne typy wejść ---

def process_frame(frame_bgr, model, processor, id2label, device, args):
    """Przetwarza pojedynczą klatkę (NumPy BGR array). Zwraca obraz PIL z wynikami."""
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        original_size = image_pil.size

        inputs = processor(images=image_pil, return_tensors="pt")
        pixel_values_shape = inputs['pixel_values'].shape
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        target_sizes = torch.tensor([original_size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.confidence_threshold
        )[0]

        heatmap = get_attention_heatmap(
            outputs, pixel_values_shape=pixel_values_shape, layer_idx=args.attention_layer
        )

        if heatmap is not None:
            frame_with_heatmap_bgr = overlay_heatmap_cv2(
                frame_bgr, heatmap[0], alpha=args.heatmap_alpha,
                colormap=getattr(cv2, args.colormap, cv2.COLORMAP_JET)
            )
        else:
            frame_with_heatmap_bgr = frame_bgr

        final_image_pil = Image.fromarray(cv2.cvtColor(frame_with_heatmap_bgr, cv2.COLOR_BGR2RGB))
        scores = results["scores"].cpu(); labels = results["labels"].cpu(); boxes = results["boxes"].cpu()
        final_image_pil = plot_results(final_image_pil, scores, labels, boxes, id2label, args.font_path)

        return final_image_pil

    except Exception as e:
        print(f"\nError processing frame/image: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Główna logika ---

def run_prediction(args):
    """Główna funkcja uruchamiająca predykcję w zależności od typu wejścia."""
    print("Starting prediction script...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not model_path.exists(): print(f"Error: Model directory not found at {model_path}"); return
    if not input_path.exists(): print(f"Error: Input path not found at {input_path}"); return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model path: {model_path}")
    print(f"Input path: {input_path}")
    print(f"Output directory: {output_dir}")

    print(f"Loading model and processor from: {model_path}")
    try:
        processor = DetrImageProcessor.from_pretrained(str(model_path))
        model = DetrForObjectDetection.from_pretrained(str(model_path)).to(device)
        model.eval()
        print("Model and processor loaded successfully.")
    except Exception as e: print(f"Error loading model/processor: {e}"); return

    try:
        id2label = model.config.id2label
        if not id2label: raise ValueError("id2label empty in model config.")
        print(f"Loaded id2label mapping: {id2label}")
    except Exception as e:
        print(f"Error getting id2label from config: {e}")
        config_path = model_path / "config.json"
        if config_path.is_file():
            try:
                with open(config_path, 'r') as f: config_data = json.load(f)
                id2label = {int(k): v for k, v in config_data['id2label'].items()}
                print(f"Loaded id2label from config.json: {id2label}")
                if not id2label: raise ValueError("id2label empty in config.json.")
            except Exception as e_cfg: print(f"Error loading from config.json: {e_cfg}"); return
        else: print("Error: Cannot find id2label mapping."); return

    if input_path.is_file():
        video_suffixes = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        image_suffixes = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

        if input_path.suffix.lower() in video_suffixes:
            print("Input is a video file. Processing frames...")
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened(): print(f"Error: Could not open video file: {input_path}"); return
            try: total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            except Exception: total_frames = None
            pbar = tqdm(total=total_frames, desc="Processing Video") if total_frames else tqdm(desc="Processing Video")
            frame_count = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret: break
                frame_count += 1
                processed_pil_frame = process_frame(frame_bgr, model, processor, id2label, device, args)
                if processed_pil_frame:
                    output_filename = f"frame_{frame_count:06d}.png"
                    output_save_path = output_dir / output_filename
                    processed_pil_frame.save(output_save_path)
                pbar.update(1)
            pbar.close()
            cap.release()
            print(f"Finished processing video. Output frames saved in: {output_dir}")

        elif input_path.suffix.lower() in image_suffixes:
            print("Input is a single image file. Processing...")
            try:
                frame_bgr = cv2.imread(str(input_path))
                if frame_bgr is None: raise ValueError("Could not read image file using OpenCV.")
                processed_pil_frame = process_frame(frame_bgr, model, processor, id2label, device, args)
                if processed_pil_frame:
                    output_filename = f"{input_path.stem}_pred{input_path.suffix}"
                    output_save_path = output_dir / output_filename
                    processed_pil_frame.save(output_save_path)
                    print(f"Saved prediction to: {output_save_path}")
                else: print(f"Failed to process image: {input_path}")
            except Exception as e: print(f"Error processing image {input_path}: {e}")
        else: print(f"Error: Unsupported file type: {input_path.suffix}"); return

    elif input_path.is_dir():
        print("Input is a directory. Processing images...")
        image_files = []
        image_suffixes = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        for ext in image_suffixes: image_files.extend(list(input_path.glob(ext)))
        if not image_files: print(f"No supported image files found in directory: {input_path}"); return
        print(f"Found {len(image_files)} images.")
        for img_path in tqdm(image_files, desc="Processing Images"):
             try:
                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None: raise ValueError("Could not read image file.")
                processed_pil_frame = process_frame(frame_bgr, model, processor, id2label, device, args)
                if processed_pil_frame:
                    output_filename = f"{img_path.stem}_pred{img_path.suffix}"
                    output_save_path = output_dir / output_filename
                    processed_pil_frame.save(output_save_path)
                else: print(f"Failed to process image: {img_path}")
             except Exception as e: print(f"Error processing image {img_path}: {e}"); continue
        print(f"Finished processing directory. Output images saved in: {output_dir}")
    else: print(f"Error: Input path is not a valid file or directory: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DETR prediction on video/images and visualize attention heatmaps.")
    parser.add_argument("--model_path", type=str, default=r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/detr_training_output/final_model", help="Path to the trained DETR model directory.")
    parser.add_argument("--input_path", type=str, default=r'E:/Cataract/videos/micro/train02.mp4', help="Path to the input video file, image file, or directory containing images.")
    parser.add_argument("--output_dir", type=str, default="./detr_predictions_with_heatmaps", help="Directory to save the output frames/images.")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for DETR post-processing (initial filtering).")
    parser.add_argument("--heatmap_alpha", type=float, default=0.4, help="Opacity of the heatmap overlay (0.0-1.0).")
    parser.add_argument("--attention_layer", type=int, default=-1, help="Decoder layer for attention visualization (-1 for last).")
    parser.add_argument("--colormap", type=str, default="COLORMAP_VIRIDIS", help="OpenCV colormap name for heatmap.")
    parser.add_argument("--font_path", type=str, default=None, help="Path to .ttf font file for labels (optional).")
    args = parser.parse_args()

    if not 0.0 <= args.heatmap_alpha <= 1.0: print("Error: heatmap_alpha must be 0.0-1.0"); exit(1)
    if not hasattr(cv2, args.colormap): print(f"Error: Unknown OpenCV colormap '{args.colormap}'."); exit(1)
    if not 0.0 <= args.confidence_threshold <= 1.0: print("Error: confidence_threshold must be 0.0-1.0"); exit(1)

    print(f"--- Note: Bounding boxes will only be DRAWN if their score is >= 0.8 ---")
    print(f"--- Initial detection filtering uses threshold: {args.confidence_threshold} ---")
    run_prediction(args)