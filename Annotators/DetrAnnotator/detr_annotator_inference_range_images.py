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
            image_size: Docelowy rozmiar obrazu
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
        print(f"Liczba obrazów w adnotacjach: {len(self.annotations['images'])}")

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
        - processed_image: Przetworzony obraz (jeśli procesor jest dostępny)
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
            "width": image_info['width'],
            "height": image_info['height']
        }

        # Jeśli dostępny jest procesor, przetwórz obraz dla modelu
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            result["processed_image"] = inputs

        return result


def select_image_range(dataset, start_idx, end_idx):
    """
    Wybierz podzbiór obrazów z datasetu na podstawie zakresu indeksów.

    Args:
        dataset: Pełny dataset
        start_idx: Indeks początkowy (włącznie)
        end_idx: Indeks końcowy (wyłącznie)

    Returns:
        Podzbiór datasetu
    """
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
    return Subset(dataset, valid_indices)


def load_model(model_path, device):
    print(f"Wczytywanie modelu z {model_path}...")

    try:
        model = DetrForObjectDetection.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True  # Add this parameter
        )
        processor = DetrImageProcessor.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("Model wczytany pomyślnie.")
        return model, processor
    except Exception as e:
        print(f"Błąd wczytywania modelu: {e}")
        raise


def run_inference(model, processor, dataset, device, confidence_threshold=0.5):
    """
    Przeprowadź inferencję na wybranym zestawie danych.

    Args:
        model: Model DETR
        processor: Procesor obrazów
        dataset: Dataset z obrazami
        device: Urządzenie (cpu/cuda)
        confidence_threshold: Próg pewności detekcji

    Returns:
        Lista wyników inferencji
    """
    results = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Przeprowadzanie inferencji"):
            sample = dataset[idx]

            # Przygotuj obraz
            image = sample["image"]
            image_id = sample["image_id"]
            image_path = sample["image_path"]
            ground_truth = sample["annotations"]

            # Przetwórz obraz dla modelu
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Przeprowadź inferencję
            outputs = model(**inputs)

            # Przetwórz wyniki
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
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

            # Zapisz wyniki
            results.append({
                "idx": idx,
                "image_id": image_id,
                "image_path": image_path,
                "predictions": predictions,
                "ground_truth": ground_truth
            })

    return results


def visualize_predictions(results, output_dir, dataset):
    """
    Wizualizuj wyniki inferencji i zapisz obrazy z oznaczonymi detekcjami.

    Args:
        results: Lista wyników inferencji
        output_dir: Katalog wyjściowy
        dataset: Dataset z oryginalnymi obrazami
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Zapisywanie wizualizacji do {output_dir}...")

    # Spróbuj załadować czcionkę
    try:
        font = ImageFont.truetype("arial.ttf", 15)
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
        "avg_confidence": avg_confidence
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

    # Zakres obrazów do inferencji - ZMODYFIKUJ TE WARTOŚCI, ABY WYBRAĆ INNY ZAKRES
    start_idx = 900  # Indeks początkowy (włącznie)
    end_idx = 1000  # Indeks końcowy (wyłącznie)

    # Parametry inferencji
    confidence_threshold = 0.2
    # =============================================

    print("=" * 80)
    print("INFERENCJA MODELU DETR NA ZAKRESIE OBRAZÓW")
    print("=" * 80)
    print(f"Zakres obrazów: od {start_idx} do {end_idx}")

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

        # Wczytaj dataset
        dataset = SurgicalToolDataset(
            images_dir=images_dir,
            annotations_file=annotations_file
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
            confidence_threshold=confidence_threshold
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