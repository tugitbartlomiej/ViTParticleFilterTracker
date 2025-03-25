import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor


class SurgicalToolDataset(Dataset):
    """Dataset dla obrazów narzędzi chirurgicznych z adnotacjami w formacie COCO."""

    def __init__(self, images_dir, annotations_file, processor=None, image_size=(640, 640)):
        """
        Inicjalizacja datasetu.

        Args:
            images_dir: Katalog z obrazami
            annotations_file: Ścieżka do pliku z adnotacjami w formacie COCO
            processor: Procesor obrazów DETR (opcjonalny, tylko do przetwarzania)
            image_size: Docelowy rozmiar obrazu - MUSI być zgodny z rozmiarem treningowym
        """
        print("Inicjalizacja datasetu...")
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.processor = processor

        # Wczytaj adnotacje COCO
        try:
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
            print(f"Pomyślnie wczytano adnotacje z {annotations_file}")
            print(f"Liczba obrazów w adnotacjach: {len(self.annotations['images'])}")
            print(f"Liczba adnotacji: {len(self.annotations['annotations'])}")
        except Exception as e:
            print(f"Błąd wczytywania adnotacji: {e}")
            raise

        # Ustaw kategorie na jedną klasę (narzędzie chirurgiczne)
        for ann in self.annotations['annotations']:
            ann['category_id'] = 0

        # Sprawdź istniejące obrazy i zbuduj mapowania
        self.valid_images = []
        self.id_to_filename = {}
        self.id_to_annotations = {}

        print(f"Wyszukiwanie obrazów w: {images_dir}")

        try:
            existing_files = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
            print(f"Znaleziono plików w katalogu: {len(existing_files)}")
        except Exception as e:
            print(f"Błąd dostępu do katalogu {images_dir}: {e}")
            existing_files = set()

        # Znajdź i zwaliduj obrazy
        valid_count = 0
        for img in self.annotations['images']:
            image_filename = img['file_name']
            if image_filename in existing_files:
                self.valid_images.append(img)
                self.id_to_filename[img['id']] = image_filename
                valid_count += 1

        # Posortuj obrazy według nazwy pliku
        self.valid_images.sort(key=lambda x: x['file_name'])

        # Powiąż adnotacje z obrazami
        valid_annotations = 0
        for ann in self.annotations['annotations']:
            if ann['image_id'] in self.id_to_filename:
                if ann['image_id'] not in self.id_to_annotations:
                    self.id_to_annotations[ann['image_id']] = []
                self.id_to_annotations[ann['image_id']].append(ann)
                valid_annotations += 1

        print(f"Wczytano {len(self.valid_images)} poprawnych obrazów z {len(self.annotations['images'])} w adnotacjach")
        print(
            f"Dataset zawiera {valid_annotations} poprawnych adnotacji z {len(self.annotations['annotations'])} wszystkich")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        """
        Pobierz element z datasetu.

        Zwraca słownik zawierający:
        - image: Oryginalny obraz PIL
        - image_id: ID obrazu
        - image_path: Ścieżka do obrazu
        - annotations: Lista adnotacji
        - width, height: Oryginalne wymiary obrazu
        """
        image_info = self.valid_images[idx]
        image_id = image_info['id']
        image_filename = self.id_to_filename[image_id]
        image_path = str(self.images_dir / image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Błąd otwierania obrazu {image_path}: {e}")
            raise e

        annotations = self.id_to_annotations.get(image_id, [])

        result = {
            "image": image,
            "image_id": image_id,
            "image_path": image_path,
            "annotations": annotations,
            "width": image_info.get('width', image.width),
            "height": image_info.get('height', image.height)
        }

        return result


def select_image_range(dataset, start_idx, end_idx):
    """
    Wybierz podzbiór obrazów z datasetu na podstawie zakresu indeksów.

    Args:
        dataset: Pełny dataset
        start_idx: Indeks początkowy (włącznie)
        end_idx: Indeks końcowy (wyłącznie)

    Returns:
        Podzbiór datasetu i rozmiar obrazu (subset, image_size)
    """
    # Zapamiętaj rozmiar obrazu z oryginalnego datasetu
    image_size = dataset.image_size

    # Określ prawidłowe indeksy
    start = max(0, min(start_idx, len(dataset) - 1))
    end = min(end_idx, len(dataset))

    if start >= end:
        raise ValueError(f"Niepoprawny zakres: start_idx ({start}) musi być mniejszy niż end_idx ({end})")

    # Utwórz zakres indeksów
    selected_indices = list(range(start, end))

    # Sprawdź, czy wybrane obrazy mają adnotacje
    valid_indices = []
    for idx in selected_indices:
        image_info = dataset.valid_images[idx]
        image_id = image_info['id']
        if image_id in dataset.id_to_annotations and len(dataset.id_to_annotations[image_id]) > 0:
            valid_indices.append(idx)

    print(f"Wybrano {len(valid_indices)} obrazów z adnotacjami z zakresu {start} do {end}")

    # Utwórz podzbiór z wybranymi indeksami
    subset = Subset(dataset, valid_indices)

    # Dodajemy atrybut image_size do obiektu Subset
    subset.image_size = image_size

    return subset


def load_model(model_path, device):
    """
    Load the DETR model and processor.

    Args:
        model_path: Path to the model directory
        device: Device to load the model on (cpu/cuda)

    Returns:
        Tuple of (model, processor)
    """
    print(f"Attempting to load model from {model_path}...")

    try:
        # Approach 1: Try fixing the model configuration directly
        from transformers import AutoConfig, DetrConfig

        # Load existing config and explicitly set num_queries to match checkpoint
        config = AutoConfig.from_pretrained(model_path)

        # Force the num_queries parameter to match the checkpoint's value
        if isinstance(config, DetrConfig):
            print(f"Original config num_queries: {config.num_queries}")
            config.num_queries = 1  # Explicitly set to match the checkpoint
            print(f"Updated config num_queries: {config.num_queries}")

        # Create model with the modified config
        model = DetrForObjectDetection.from_config(config)

        # Load the state dict manually
        import torch
        import os

        # Load state dict from the pytorch_model.bin file
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            print(f"Loading state dict from {state_dict_path}")
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"State dict not found at {state_dict_path}")

    except Exception as e:
        print(f"Approach 1 failed with error: {e}")
        print("Falling back to ignore_mismatched_sizes=True")

        # Approach 2: Use ignore_mismatched_sizes as a fallback
        model = DetrForObjectDetection.from_pretrained(model_path, ignore_mismatched_sizes=True)

    # Load the processor
    processor = DetrImageProcessor.from_pretrained(model_path)

    # Post-loading checks
    if hasattr(model.config, 'num_queries'):
        print(f"Loaded model has num_queries={model.config.num_queries}")

    # Move the model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    return model, processor
def run_inference(model, processor, dataset, device, confidence_threshold=0.3, debug_mode=False, image_size=(800, 800)):
    """
    Przeprowadź inferencję na wybranym zestawie danych.

    Args:
        model: Model DETR
        processor: Procesor obrazów
        dataset: Dataset z obrazami
        device: Urządzenie (cpu/cuda)
        confidence_threshold: Próg pewności detekcji
        debug_mode: Czy wyświetlać szczegółowe informacje diagnostyczne
        image_size: Rozmiar obrazu używany do przetwarzania (z parametru lub datasetu)

    Returns:
        Lista wyników inferencji
    """
    results = []

    # Pobierz rozmiar obrazu z datasetu (jeśli to możliwe) lub użyj przekazanego parametru
    try:
        if hasattr(dataset, 'image_size'):
            # Dataset ma bezpośrednio atrybut image_size
            actual_image_size = dataset.image_size
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'image_size'):
            # Dataset jest typu Subset
            actual_image_size = dataset.dataset.image_size
        else:
            # Użyj domyślnego rozmiaru
            actual_image_size = image_size
    except Exception:
        # W razie problemów, użyj domyślnego rozmiaru
        actual_image_size = image_size

    print(f"Uruchamianie inferencji z progiem pewności: {confidence_threshold}")
    print(f"Używany rozmiar obrazu: {actual_image_size}")

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Przeprowadzanie inferencji"):
            sample = dataset[idx]

            # Przygotuj obraz
            image = sample["image"]
            image_id = sample["image_id"]
            image_path = sample["image_path"]
            ground_truth = sample["annotations"]

            # ISTOTNE: Przetwórz obraz używając TYCH SAMYCH parametrów co podczas treningu
            inputs = processor(
                images=image,
                return_tensors="pt",
                size={'shortest_edge': dataset.image_size[0], 'longest_edge': dataset.image_size[1]}
            ).to(device)

            # Przeprowadź inferencję
            outputs = model(**inputs)

            # Wyświetl informacje diagnostyczne jeśli włączony tryb debug
            if debug_mode:
                probs = outputs.logits.softmax(-1)
                scores = probs[0, :, :-1].max(-1).values
                top_scores = torch.sort(scores, descending=True)[0][:5].tolist()
                print(f"\nObraz {idx}, ID {image_id}: {image_path}")
                print(f"Top 5 pewności: {[f'{s:.4f}' for s in top_scores]}")

            # Przetwórz wyniki
            target_sizes = torch.tensor([[sample["height"], sample["width"]]]).to(device)
            processed_outputs = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=confidence_threshold
            )[0]

            # Zbierz wyniki
            predictions = []
            for score, label, box in zip(processed_outputs["scores"],
                                         processed_outputs["labels"],
                                         processed_outputs["boxes"]):
                predictions.append({
                    "score": score.item(),
                    "label": label.item(),
                    "box": box.tolist()  # [x1, y1, x2, y2]
                })

            if debug_mode and predictions:
                print(f"Znaleziono {len(predictions)} detekcji:")
                for i, p in enumerate(predictions):
                    print(f"  {i + 1}: Pewność: {p['score']:.4f}, Box: {[round(x, 1) for x in p['box']]}")

            # KOD FILTRUJĄCY
            if len(predictions) > 1:
                # Wybierz tylko detekcję o najwyższej pewności
                predictions = [max(predictions, key=lambda x: x["score"])]

            # Zapisz wyniki
            results.append({
                "idx": idx,
                "image_id": image_id,
                "image_path": image_path,
                "predictions": predictions,
                "ground_truth": ground_truth
            })

    return results


def visualize_predictions(results, output_dir, dataset, draw_all_gt=True):
    """
    Wizualizuj wyniki inferencji i zapisz obrazy z oznaczonymi detekcjami.

    Args:
        results: Lista wyników inferencji
        output_dir: Katalog wyjściowy
        dataset: Dataset z oryginalnymi obrazami
        draw_all_gt: Czy rysować wszystkie ground truth boxes, nawet jeśli nie ma detekcji
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Zapisywanie wizualizacji do {output_dir}...")

    # Spróbuj załadować czcionkę
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        try:
            # Próba znalezienia czcionki systemowej
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except:
            font = ImageFont.load_default()

    for result in tqdm(results, desc="Generowanie wizualizacji"):
        # Pobierz obraz z datasetu
        if hasattr(dataset, 'indices'):  # Jeśli to podzbiór (Subset)
            actual_idx = dataset.indices[result["idx"]]
            sample = dataset.dataset[actual_idx]
        else:
            sample = dataset[result["idx"]]

        image = sample["image"].copy()
        draw = ImageDraw.Draw(image)

        # Narysuj prawdziwe bounding boxy (zielone)
        for ann in result["ground_truth"]:
            # Format COCO: [x, y, width, height]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Narysuj rzeczywisty bbox
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, max(0, y1 - 15)), "GT", fill="green", font=font)

        # Narysuj przewidziane bounding boxy (czerwone)
        for pred in result["predictions"]:
            x1, y1, x2, y2 = pred["box"]
            score = pred["score"]

            # Narysuj przewidziany bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"Score: {score:.2f}", fill="red", font=font)

        # Dodaj informacje o obrazie
        draw.text((10, 10), f"Image ID: {result['image_id']}", fill="blue", font=font)
        draw.text((10, 30), f"GT boxes: {len(result['ground_truth'])}, Pred boxes: {len(result['predictions'])}",
                  fill="blue", font=font)

        # Zapisz obraz z adnotacjami
        image_filename = os.path.basename(result["image_path"])
        output_path = os.path.join(output_dir, f"pred_{result['idx']}_{image_filename}")
        image.save(output_path)

    print(f"Zapisano {len(results)} wizualizacji w {output_dir}")


def calculate_metrics(results):
    """
    Oblicz metryki dla wyników inferencji.

    Args:
        results: Lista wyników inferencji

    Returns:
        Słownik z metrykami
    """
    total_images = len(results)
    images_with_detections = sum(1 for r in results if len(r["predictions"]) > 0)
    total_gt_boxes = sum(len(r["ground_truth"]) for r in results)
    total_pred_boxes = sum(len(r["predictions"]) for r in results)

    # Prosta analiza zgodności liczby detekcji
    correct_detection_count = sum(1 for r in results if len(r["ground_truth"]) == len(r["predictions"]))

    # Oblicz średni wynik pewności
    all_scores = [pred["score"] for result in results for pred in result["predictions"]]
    avg_confidence = np.mean(all_scores) if all_scores else 0

    # Dodatkowe analizy metryczne
    gt_match_stats = []
    for r in results:
        gt_count = len(r["ground_truth"])
        pred_count = len(r["predictions"])
        gt_match_stats.append({
            "gt_count": gt_count,
            "pred_count": pred_count,
            "match": gt_count == pred_count
        })

    # Oblicz rozkład liczby detekcji
    pred_counts = {}
    for stat in gt_match_stats:
        pred_count = stat["pred_count"]
        if pred_count not in pred_counts:
            pred_counts[pred_count] = 0
        pred_counts[pred_count] += 1

    metrics = {
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "detection_rate": images_with_detections / total_images if total_images > 0 else 0,
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "avg_gt_per_image": total_gt_boxes / total_images if total_images > 0 else 0,
        "avg_pred_per_image": total_pred_boxes / total_images if total_images > 0 else 0,
        "correct_detection_count": correct_detection_count,
        "correct_detection_rate": correct_detection_count / total_images if total_images > 0 else 0,
        "avg_confidence": avg_confidence,
        "prediction_counts": pred_counts
    }

    return metrics


def print_metrics(metrics):
    """Wyświetl metryki w czytelnej formie."""
    print("\n" + "=" * 50)
    print("WYNIKI INFERENCJI")
    print("=" * 50)

    print(f"Liczba obrazów: {metrics['total_images']}")
    print(f"Obrazy z detekcjami: {metrics['images_with_detections']} ({metrics['detection_rate']:.2%})")
    print(f"Całkowita liczba annotacji GT: {metrics['total_gt_boxes']}")
    print(f"Całkowita liczba detekcji: {metrics['total_pred_boxes']}")
    print(f"Średnia annotacji GT na obraz: {metrics['avg_gt_per_image']:.2f}")
    print(f"Średnia detekcji na obraz: {metrics['avg_pred_per_image']:.2f}")
    print(
        f"Obrazy z poprawną liczbą detekcji: {metrics['correct_detection_count']} ({metrics['correct_detection_rate']:.2%})")
    print(f"Średni wynik pewności detekcji: {metrics['avg_confidence']:.4f}")

    # Wyświetl rozkład detekcji
    print("\nRozkład liczby detekcji na obraz:")
    for count, num_images in sorted(metrics["prediction_counts"].items()):
        percent = num_images / metrics["total_images"] * 100
        print(f"  {count} detekcji: {num_images} obrazów ({percent:.1f}%)")

    print("=" * 50)


def save_results_json(results, output_path):
    """
    Zapisz wyniki inferencji do pliku JSON.

    Args:
        results: Lista wyników inferencji
        output_path: Ścieżka do zapisu pliku JSON
    """
    # Konwertuj ścieżki na ciągi znaków
    serializable_results = []
    for r in results:
        serializable_r = r.copy()
        serializable_r["image_path"] = str(serializable_r["image_path"])
        serializable_results.append(serializable_r)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)

    print(f"Zapisano wyniki do {output_path}")


def main():
    # ========== PARAMETRY KONFIGURACYJNE ==========
    # Ścieżki
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/images"
    annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_coco_200-300_20250325_022605.json"
    model_dir = "./training_ranged/best_model"
    output_dir = "./inference_ranged"

    # Zakres obrazów do inferencji
    start_idx = 500
    end_idx = 600

    # Parametry inferencji
    confidence_threshold = 0.1  # Obniżono z 0.2 dla lepszego wychwytywania detekcji
    debug_mode = True  # Pokaż szczegółowe informacje (pomocne przy diagnozowaniu)
    image_size = (800, 800)  # MUSI być zgodny z rozmiarem treningowym!
    # =============================================

    print("=" * 80)
    print("INFERENCJA MODELU DETR NA ZAKRESIE OBRAZÓW")
    print("=" * 80)
    print(f"Zakres obrazów: od {start_idx} do {end_idx}")
    print(f"Rozmiar obrazu: {image_size}")
    print(f"Próg pewności: {confidence_threshold}")

    # Utwórz katalogi wyjściowe
    os.makedirs(output_dir, exist_ok=True)
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    # Ustawienia urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")

    try:
        # Wczytaj model i procesor
        model, processor = load_model(model_dir, device)

        # Wczytaj dataset z POPRAWNYM rozmiarem obrazu
        dataset = SurgicalToolDataset(
            images_dir=images_dir,
            annotations_file=annotations_file,
            processor=processor,
            image_size=image_size  # Używaj tego samego rozmiaru co podczas treningu
        )

        if len(dataset) == 0:
            print("BŁĄD: Dataset nie zawiera żadnych obrazów!")
            return

        # Wybierz zakres obrazów
        selected_dataset = select_image_range(dataset, start_idx, end_idx)

        if len(selected_dataset) == 0:
            print("BŁĄD: Wybrany zakres nie zawiera żadnych obrazów z adnotacjami!")
            return

        # Przeprowadź inferencję
        results = run_inference(
            model=model,
            processor=processor,
            dataset=selected_dataset,
            device=device,
            confidence_threshold=confidence_threshold,
            debug_mode=debug_mode,
            image_size=image_size  # Przekaż jawnie rozmiar obrazu
        )

        # Wizualizuj wyniki
        visualize_predictions(
            results=results,
            output_dir=visualizations_dir,
            dataset=selected_dataset
        )

        # Oblicz metryki
        metrics = calculate_metrics(results)

        # Wyświetl metryki
        print_metrics(metrics)

        # Zapisz wyniki do pliku JSON
        save_results_json(
            results=results,
            output_path=os.path.join(output_dir, f"inference_results_{start_idx}_{end_idx}.json")
        )

        # Zapisz metryki do pliku
        with open(os.path.join(output_dir, f"metrics_{start_idx}_{end_idx}.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

        print("=" * 80)
        print("INFERENCJA ZAKOŃCZONA POMYŚLNIE")
        print("=" * 80)
        print(f"Wizualizacje zapisano w: {visualizations_dir}")
        print(f"Wyniki zapisano w: {output_dir}")

    except Exception as e:
        print(f"BŁĄD podczas inferencji: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
