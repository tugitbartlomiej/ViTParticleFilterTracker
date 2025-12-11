#!/usr/bin/env python3
"""
Stabilny skrypt do treningu YOLOv8 z obsługą błędów CUDA i lepszą konfiguracją.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO


def setup_cuda_environment():
    """Ustaw środowisko CUDA dla stabilności."""
    # Zmienne środowiskowe dla stabilności CUDA
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    # Wyłącz niektóre ostrzeżenia PyTorch
    warnings.filterwarnings("ignore", category=UserWarning)
    

def validate_dataset_yaml(yaml_path):
    """Sprawdź czy plik dataset.yaml jest poprawny."""
    import yaml
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset YAML nie istnieje: {yaml_path}")
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Brakujące klucze w dataset.yaml: {missing_keys}")
        
        print(f"✅ Dataset YAML poprawny: {yaml_path}")
        print(f"   - Dataset path: {config.get('path', 'N/A')}")
        print(f"   - Train: {config.get('train', 'N/A')}")
        print(f"   - Val: {config.get('val', 'N/A')}")
        print(f"   - Classes: {config.get('nc', 'N/A')}")
        print(f"   - Names: {config.get('names', 'N/A')}")
        
        return config
        
    except Exception as e:
        raise ValueError(f"Błąd w parsowaniu dataset.yaml: {e}")


def create_directories(project_dir, checkpoint_dir, best_model_dir):
    """Stwórz katalogi na wyniki."""
    dirs = [project_dir, checkpoint_dir, best_model_dir]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Katalog: {dir_path}")


def main():
    parser = argparse.ArgumentParser(description="Trening YOLOv8 ze stabilnymi ustawieniami")
    
    # Podstawowe argumenty
    parser.add_argument('--dataset_yaml_path', required=True, help='Ścieżka do dataset.yaml')
    parser.add_argument('--project_dir', default='./runs/train', help='Katalog projektu')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', help='Katalog checkpointów')
    parser.add_argument('--best_model_dir', default='./best_models', help='Katalog najlepszych modeli')
    
    # Parametry modelu
    parser.add_argument('--model_size', choices=['n', 's', 'm', 'l', 'x'], default='l', help='Rozmiar modelu YOLO')
    parser.add_argument('--epochs', type=int, default=200, help='Liczba epok')
    parser.add_argument('--batch_size', type=int, default=96, help='Rozmiar batch')
    parser.add_argument('--imgsz', type=int, default=640, help='Rozmiar obrazu')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--workers', type=int, default=1, help='Liczba workers (zalecane: 1)')
    parser.add_argument('--device', default='0,1,2,3,4,5,6,7', help='Urządzenia GPU')
    parser.add_argument('--save_period', type=int, default=10, help='Okres zapisu checkpointów')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    # Flagi
    parser.add_argument('--verbose', action='store_true', help='Tryb szczegółowy')
    parser.add_argument('--disable_ddp', action='store_true', help='Wyłącz DDP')
    parser.add_argument('--resume', help='Ścieżka do checkpointu do wznowienia')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLO TRAINING SCRIPT V4 - STABILNA WERSJA")
    print("=" * 80)
    
    # Ustaw środowisko CUDA
    setup_cuda_environment()
    
    # Sprawdź dostępność CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA dostępna: {torch.cuda.device_count()} GPU")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("⚠️ CUDA niedostępna - używa CPU")
    
    # Waliduj dataset.yaml
    try:
        dataset_config = validate_dataset_yaml(args.dataset_yaml_path)
    except Exception as e:
        print(f"❌ Błąd w konfiguracji datasetu: {e}")
        sys.exit(1)
    
    # Stwórz katalogi
    create_directories(args.project_dir, args.checkpoint_dir, args.best_model_dir)
    
    # Przygotuj model
    model_name = f"yolov8{args.model_size}.pt"
    print(f"🤖 Ładowanie modelu: {model_name}")
    
    try:
        model = YOLO(model_name)
        print(f"✅ Model załadowany: {model_name}")
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        sys.exit(1)
    
    # Przygotuj argumenty treningu
    train_args = {
        'data': args.dataset_yaml_path,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'lr0': args.learning_rate,
        'workers': args.workers,
        'device': args.device,
        'project': args.project_dir,
        'name': 'train',
        'save_period': args.save_period,
        'patience': args.patience,
        'verbose': args.verbose,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'seed': 42,
        'deterministic': True,
        'single_cls': True,  # Jedna klasa - tooltip
        'rect': False,  # Wyłącz rectangular training dla stabilności
        'cos_lr': False,  # Wyłącz cosine learning rate
        'close_mosaic': 10,  # Wyłącz mosaic w ostatnich epokach
        'amp': False,  # Wyłącz AMP dla stabilności na DGX-1
        'plots': True,
        'save': True,
        'save_txt': False,
        'save_conf': False,
        'cache': False,  # Wyłącz cache aby zaoszczędzić pamięć
    }
    
    # Dodatkowe ustawienia dla DDP
    if args.disable_ddp or torch.cuda.device_count() == 1:
        print("🔧 DDP wyłączone - trening na jednym procesie")
        train_args['ddp'] = False
    else:
        print("🔧 DDP włączone - trening rozproszony")
        train_args['ddp'] = True
    
    if args.resume:
        train_args['resume'] = args.resume
        print(f"🔄 Wznowienie treningu z: {args.resume}")
    
    print("\n📋 Parametry treningu:")
    for key, value in train_args.items():
        print(f"   {key}: {value}")
    
    print("\n🚀 Rozpoczynam trening...")
    print("=" * 80)
    
    try:
        # Uruchom trening
        results = model.train(**train_args)
        
        print("=" * 80)
        print("✅ TRENING ZAKOŃCZONY POMYŚLNIE!")
        
        # Znajdź najlepszy model
        train_dir = Path(args.project_dir) / 'train'
        best_model_path = train_dir / 'weights' / 'best.pt'
        last_model_path = train_dir / 'weights' / 'last.pt'
        
        # Skopiuj modele do docelowych katalogów
        if best_model_path.exists():
            import shutil
            best_dest = Path(args.best_model_dir) / 'best.pt'
            shutil.copy2(best_model_path, best_dest)
            print(f"💾 Najlepszy model skopiowany do: {best_dest}")
        
        if last_model_path.exists():
            import shutil
            last_dest = Path(args.checkpoint_dir) / 'last.pt'
            shutil.copy2(last_model_path, last_dest)
            print(f"💾 Ostatni checkpoint skopiowany do: {last_dest}")
        
        print(f"📊 Wyniki treningu: {train_dir}")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Trening przerwany przez użytkownika")
        return 1
        
    except Exception as e:
        print(f"\n❌ Błąd podczas treningu: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
