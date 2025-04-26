import os

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

# ========== HARDCODED PARAMETERS ==========
# Ścieżka do checkpointa
CHECKPOINT_PATH = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/checkpoints/checkpoint_epoch_44.pt"
# Katalog docelowy dla gotowego modelu
OUTPUT_DIR = "./detr_tool_tracking_model_final"
# Rozmiar obrazów używany podczas treningu
IMAGE_SIZE = (800, 800)
# Parametry modelu - MUSZĄ być takie same jak przy treningu!
NUM_QUERIES = 100  # Domyślna wartość, dopasuj do wartości użytej w treningu (prawdopodobnie 2 lub 5)
NUM_LABELS = 1  # Tylko jedna klasa (surgical_tool)


# ===========================================

def convert_checkpoint_to_model():
    """
    Konwertuje checkpoint z treningu DETR na gotowy model Hugging Face.
    """
    print(f"Wczytywanie checkpointa z: {CHECKPOINT_PATH}")

    # Sprawdź czy plik istnieje
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint nie istnieje: {CHECKPOINT_PATH}")

    # Wczytaj checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    # Utwórz katalog wyjściowy jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sprawdź rozmiar query_position_embeddings w checkpoincie, aby określić num_queries
    model_state_dict = checkpoint['model_state_dict']
    if 'model.query_position_embeddings.weight' in model_state_dict:
        actual_num_queries = model_state_dict['model.query_position_embeddings.weight'].size(0)
        print(f"Wykryto num_queries={actual_num_queries} w checkpoincie")
        NUM_QUERIES = actual_num_queries

    # Inicjalizuj model bazowy z odpowiednimi parametrami
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_LABELS,
        num_queries=NUM_QUERIES,
        ignore_mismatched_sizes=True
    )

    # Konfiguracja modelu dla jednej klasy
    model.config.id2label = {0: "surgical_tool"}
    model.config.label2id = {"surgical_tool": 0}
    model.config.num_labels = NUM_LABELS

    # Ustaw weights z checkpointa
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if missing_keys:
        print(f"Brakujące klucze: {missing_keys}")
    if unexpected_keys:
        print(f"Nieoczekiwane klucze: {unexpected_keys}")

    # Pobierz procesor
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={'shortest_edge': IMAGE_SIZE[0], 'longest_edge': IMAGE_SIZE[1]}
    )

    # Zapisz model i procesor
    print(f"Zapisywanie modelu do: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Zapisz również informacje o stanie treningu
    info = {
        "epoch": checkpoint['epoch'],
        "best_val_loss": checkpoint.get('best_val_loss', 'N/A'),
        "num_queries": NUM_QUERIES,
        "num_labels": NUM_LABELS
    }

    with open(os.path.join(OUTPUT_DIR, "checkpoint_info.txt"), "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    print(f"Konwersja zakończona pomyślnie. Model i procesor zapisane w: {OUTPUT_DIR}")
    print(f"Epoka: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"Najlepsza walidacyjna strata: {checkpoint['best_val_loss']:.4f}")


if __name__ == "__main__":
    print("=" * 80)
    print("KONWERSJA CHECKPOINTA DETR DO GOTOWEGO MODELU")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Katalog wyjściowy: {OUTPUT_DIR}")

    try:
        convert_checkpoint_to_model()
        print("=" * 80)
        print("KONWERSJA ZAKOŃCZONA POMYŚLNIE")
        print("=" * 80)
    except Exception as e:
        print(f"BŁĄD podczas konwersji: {str(e)}")
        import traceback

        traceback.print_exc()