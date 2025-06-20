import argparse
import os
import signal
import sys
import warnings
from pathlib import Path

import yaml

# WAŻNE: Priorytet dla bibliotek z $TMPDIR/pylibs
pip_libs = os.environ.get('PYTHONPATH', '').split(':')[0]
if pip_libs and os.path.exists(pip_libs):
    sys.path.insert(0, pip_libs)

import torch
from ultralytics import YOLO

# Wyłączenie ostrzeżeń FutureWarning dotyczących biblioteki torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# Wyłączenie ostrzeżeń NumPy
warnings.filterwarnings("ignore", message=".*NumPy.*")
# Wyłączenie ostrzeżeń NVML
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")

# Diagnostyka środowiska
import numpy as np
print(f"Python executable: {sys.executable}")
print(f"Python path[0]: {sys.path[0]}")
print(f"PYTHONPATH env: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("-" * 80)


class YOLOTrainer:
    def __init__(
    self,
    dataset_yaml_path: str,
    model_size: str = 'l',
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    project_dir: str = 'surgical_tool_detection',
    checkpoint_dir: str = './checkpoints',
    best_model_dir: str = './best_model',
    learning_rate: float = 0.01,
    resume: bool = False,
    resume_path: str = None,
    workers: int = 0,
    device: str = None,
    save_period: int = 10,
    patience: int = 50,
    verbose: bool = True,
    disable_ddp: bool = False  # Nowa opcja
    ):
        """
        Inicjalizacja trenera YOLO z rozszerzonymi parametrami.
        """
        # Wyłączenie logowania przez wandb
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"

        self.dataset_yaml_path = Path(dataset_yaml_path)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.project_dir = Path(project_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_model_dir = Path(best_model_dir)
        self.learning_rate = learning_rate
        self.resume = resume
        self.resume_path = resume_path
        self.workers = workers
        self.save_period = save_period
        self.patience = patience
        self.verbose = verbose
        self.disable_ddp = disable_ddp
        
        # Sprawdzenie zmiennych środowiskowych dla DDP
        if os.environ.get('YOLO_DISABLE_DDP', '').lower() in ['1', 'true', 'yes']:
            self.disable_ddp = True
            print("DDP wyłączone przez zmienną środowiskową YOLO_DISABLE_DDP")
        
        # Ustawienie urządzenia - obsługa multi-GPU
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Sprawdzenie multi-GPU
        if ',' in str(self.device):
            # Multi-GPU training
            gpu_list = self.device.split(',')
            print(f"Multi-GPU training enabled on GPUs: {gpu_list}")
            
            # Jeśli włączono disable_ddp, użyj tylko pierwszego GPU
            if self.disable_ddp:
                print(f"DDP disabled - using only GPU {gpu_list[0]}")
                self.device = gpu_list[0]
            else:
                # Sprawdź czy możemy używać DDP
                if torch.cuda.device_count() < len(gpu_list):
                    print(f"WARNING: Requested {len(gpu_list)} GPUs but only {torch.cuda.device_count()} available")
                    print("Falling back to single GPU training")
                    self.device = '0'
                else:
                    self.device = self.device  # YOLO obsługuje format "0,1,2,3"
                    
        elif self.device == 'cuda' and torch.cuda.device_count() > 1:
            # Automatyczne wykrycie wszystkich GPU
            gpu_count = torch.cuda.device_count()
            if not self.disable_ddp:
                self.device = ','.join([str(i) for i in range(gpu_count)])
                print(f"Auto-detected {gpu_count} GPUs. Using: {self.device}")
            else:
                print(f"Auto-detected {gpu_count} GPUs but DDP disabled. Using GPU 0")
                self.device = '0'
                
        # Sprawdzenie ścieżek
        if not self.dataset_yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML file not found: {self.dataset_yaml_path}")
            
        # Weryfikacja i naprawa dataset.yaml
        self._verify_and_fix_dataset_yaml()
            
        # Tworzenie katalogów
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Inicjalizacja modelu YOLO
        if self.resume and self.resume_path:
            print(f"Resuming training from checkpoint: {self.resume_path}")
            self.model = YOLO(self.resume_path)
        else:
            print(f"Initializing new YOLOv8{self.model_size} model")
            self.model = YOLO(f'yolov8{self.model_size}.pt')

        # Ustawienie obsługi sygnałów przerwania
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        self._print_configuration()
    
    def _verify_and_fix_dataset_yaml(self):
        """Weryfikuj i napraw plik dataset.yaml jeśli to konieczne."""
        print(f"\nVerifying dataset configuration: {self.dataset_yaml_path}")
        
        try:
            with open(self.dataset_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            dataset_dir = self.dataset_yaml_path.parent
            modified = False
            
            # Sprawdź kluczowe pola
            for key in ['train', 'val', 'test']:
                if key in data and data[key]:
                    file_path = data[key]
                    # Jeśli to absolutna ścieżka, sprawdź czy istnieje
                    if os.path.isabs(file_path):
                        if not os.path.exists(file_path):
                            # Spróbuj znaleźć plik lokalnie
                            basename = os.path.basename(file_path)
                            local_path = dataset_dir / basename
                            if local_path.exists():
                                data[key] = str(local_path)
                                print(f"  ✓ Fixed {key} path: {local_path}")
                                modified = True
                            else:
                                print(f"  ⚠ Warning: {key} file not found: {file_path}")
                                # Sprawdź czy istnieje katalog z obrazami
                                images_dir = dataset_dir / "images" / key
                                if images_dir.exists():
                                    data[key] = str(images_dir)
                                    print(f"  ✓ Using images directory for {key}: {images_dir}")
                                    modified = True
                                else:
                                    print(f"  ⚠ Skipping {key} - file not found")
                                    data[key] = None
                                    modified = True
            
            # Sprawdź pole 'path'
            if 'path' in data and os.path.isabs(data['path']):
                data['path'] = str(dataset_dir)
                print(f"  ✓ Fixed dataset path: {dataset_dir}")
                modified = True
            
            # Zapisz poprawiony plik jeśli były zmiany
            if modified:
                backup_path = self.dataset_yaml_path.with_suffix('.yaml.backup')
                os.rename(self.dataset_yaml_path, backup_path)
                print(f"  ✓ Created backup: {backup_path}")
                
                with open(self.dataset_yaml_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
                print(f"  ✓ Saved corrected dataset.yaml")
            else:
                print("  ✓ Dataset configuration looks good")
                
        except Exception as e:
            print(f"  ⚠ Warning: Could not verify dataset.yaml: {e}")
            print("  Continuing with original file...")
    
    def _print_configuration(self):
        """Wyświetl konfigurację treningu."""
        print("=" * 80)
        print("YOLO TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Dataset: {self.dataset_yaml_path}")
        print(f"Model size: {self.model_size}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.imgsz}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Workers: {self.workers}")
        print(f"Project directory: {self.project_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Best model directory: {self.best_model_dir}")
        print(f"Save period: {self.save_period} epochs")
        print(f"Patience: {self.patience} epochs")
        print(f"Resume training: {self.resume}")
        print(f"DDP disabled: {self.disable_ddp}")
        if self.resume and self.resume_path:
            print(f"Resume from: {self.resume_path}")
        print("=" * 80)
        
        # Informacje o GPU
        if torch.cuda.is_available():
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"Available GPUs: {gpu_count}")
            for i in range(gpu_count):
                try:
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                    props = torch.cuda.get_device_properties(i)
                    print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
                except Exception as e:
                    print(f"  GPU {i}: Could not get info - {e}")
            
            # Sprawdź czy używamy multi-GPU
            if ',' in str(self.device):
                print(f"Multi-GPU training on devices: {self.device}")
                effective_batch_size = self.batch_size * len(self.device.split(','))
                print(f"Effective batch size (batch_size * num_gpus): {effective_batch_size}")
        else:
            print("WARNING: CUDA not available, training on CPU!")
        print("=" * 80)

    def _handle_interrupt(self, sig, frame):
        """Obsługa przerwania treningu."""
        print("\nInterrupt signal received. Saving checkpoint...")
        checkpoint_path = self.checkpoint_dir / "interrupted_checkpoint.pt"
        self.model.save(str(checkpoint_path))
        print(f"Checkpoint saved to: {checkpoint_path}")
        print("Exiting.")
        sys.exit(0)

    def train(self):
        """Uruchom trening modelu YOLO."""
        print("\nStarting YOLO training...")
        
        try:
            # Konfiguracja parametrów treningu
            train_args = {
                'data': str(self.dataset_yaml_path),
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.imgsz,
                'device': self.device,
                'project': str(self.project_dir),
                'name': "exp",
                'pretrained': True,
                'verbose': self.verbose,
                'workers': self.workers,
                'lr0': self.learning_rate,
                'save_period': self.save_period,
                'patience': self.patience,
                'exist_ok': True,  # Pozwala na nadpisanie istniejącego katalogu
                'save': True,
                'save_json': True,
                'plots': True,
                'val': True,  # Włącz walidację jeśli dostępna
            }
            
            # WAŻNE: Wyłącz DDP jeśli są problemy
            if self.disable_ddp and ',' in str(self.device):
                print("\n⚠️ WARNING: Multi-GPU requested but DDP disabled. Using DataParallel instead.")
                # Ultralytics automatycznie użyje DataParallel zamiast DDP
            
            # Dodaj resume jeśli włączone
            if self.resume:
                train_args['resume'] = True
                
            # Dodatkowe ustawienia dla stabilności na klastrze
            if os.environ.get('NCCL_DEBUG'):
                print(f"NCCL_DEBUG set to: {os.environ.get('NCCL_DEBUG')}")
            if os.environ.get('CUDA_DISABLE_NVML'):
                print("NVML disabled for compatibility")
                
            # Uruchom trening
            results = self.model.train(**train_args)
            
            print("\nTraining completed successfully.")
            
            # Zapisz najlepszy model w dedykowanym katalogu
            best_model_path = self.best_model_dir / "best.pt"
            
            # Szukaj najlepszego modelu w różnych możliwych lokalizacjach
            possible_best_paths = [
                Path(self.project_dir) / "exp" / "weights" / "best.pt",
                Path(self.project_dir) / "exp" / "best.pt",
                Path(self.project_dir) / "weights" / "best.pt",
            ]
            
            for path in possible_best_paths:
                if path.exists():
                    import shutil
                    shutil.copy(path, str(best_model_path))
                    print(f"Best model saved to: {best_model_path}")
                    break
            else:
                print("Warning: Could not find best.pt to copy")
            
            # Wyświetl podsumowanie wyników
            self._print_results_summary(results)
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            checkpoint_path = self.checkpoint_dir / "interrupted_checkpoint.pt"
            self.model.save(str(checkpoint_path))
            print(f"Checkpoint saved to: {checkpoint_path}")
            print("Exiting training loop.")
            sys.exit(0)
        except FileNotFoundError as e:
            print(f"\nERROR: File not found - {str(e)}")
            print("\nPossible solutions:")
            print("1. Check if the dataset was extracted correctly")
            print("2. Verify paths in dataset.yaml")
            print("3. Ensure all required files (train.txt, val.txt) exist")
            print("4. Try using absolute paths in dataset.yaml")
            sys.exit(1)
        except torch.distributed.DistBackendError as e:
            print(f"\nERROR: Distributed training error - {str(e)}")
            print("\nTrying fallback solutions:")
            print("1. Set environment variable: export YOLO_DISABLE_DDP=1")
            print("2. Use single GPU: --device 0")
            print("3. Check CUDA/NCCL compatibility")
            
            # Próba zapisania checkpointu awaryjnego
            try:
                emergency_checkpoint = self.checkpoint_dir / "emergency_checkpoint.pt"
                self.model.save(str(emergency_checkpoint))
                print(f"\nEmergency checkpoint saved to: {emergency_checkpoint}")
            except:
                print("Could not save emergency checkpoint")
            
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR during training: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Próba zapisania checkpointu awaryjnego
            try:
                emergency_checkpoint = self.checkpoint_dir / "emergency_checkpoint.pt"
                self.model.save(str(emergency_checkpoint))
                print(f"\nEmergency checkpoint saved to: {emergency_checkpoint}")
            except:
                print("Could not save emergency checkpoint")
            
            sys.exit(1)

    def _print_results_summary(self, results):
        """Wyświetl podsumowanie wyników treningu."""
        print("\n" + "=" * 80)
        print("TRAINING RESULTS SUMMARY")
        print("=" * 80)
        
        try:
            # Próba wyświetlenia różnych formatów wyników
            if hasattr(results, 'keys'):
                for key, value in results.items():
                    print(f"{key}: {value}")
            elif hasattr(results, '__dict__'):
                for key, value in results.__dict__.items():
                    if not key.startswith('_'):
                        print(f"{key}: {value}")
            else:
                print(f"Results: {results}")
        except Exception as e:
            print(f"Could not display detailed results: {e}")
            
        print("=" * 80)


def parse_arguments():
    """Parsuj argumenty linii poleceń."""
    parser = argparse.ArgumentParser(
        description='YOLO Surgical Tool Detection Training Script'
    )
    
    # Ścieżki
    parser.add_argument(
        '--dataset_yaml_path',
        type=str,
        required=False,
        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Datasets/Yolo/dataset.yaml",
        help='Path to dataset YAML configuration file'
    )
    
    parser.add_argument(
        '--project_dir',
        type=str,
        default='./yolo_training_output',
        help='Project directory for training outputs'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--best_model_dir',
        type=str,
        default='./best_model',
        help='Directory to save the best model'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_size',
        type=str,
        default='l',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO model size (n=nano, s=small, m=medium, l=large, x=extra-large)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use for training (cuda/cpu/0,1,2,3). Default: auto-detect'
    )
    
    # Checkpointing and resuming
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    
    parser.add_argument(
        '--resume_path',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--save_period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Patience for early stopping (epochs without improvement)'
    )
    
    # DDP control
    parser.add_argument(
        '--disable_ddp',
        action='store_true',
        help='Disable Distributed Data Parallel (DDP) for multi-GPU training'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable verbose output (overrides --verbose)'
    )
    
    args = parser.parse_args()
    
    # Handle quiet flag
    if args.quiet:
        args.verbose = False
        
    return args


def main():
    """Główna funkcja uruchamiająca trening."""
    # Parsuj argumenty
    args = parse_arguments()
    
    print("=" * 80)
    print("YOLO SURGICAL TOOL DETECTION TRAINING")
    print("=" * 80)
    
    try:
        # Utwórz i uruchom trenera
        trainer = YOLOTrainer(
            dataset_yaml_path=args.dataset_yaml_path,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            project_dir=args.project_dir,
            checkpoint_dir=args.checkpoint_dir,
            best_model_dir=args.best_model_dir,
            learning_rate=args.learning_rate,
            resume=args.resume,
            resume_path=args.resume_path,
            workers=args.workers,
            device=args.device,
            save_period=args.save_period,
            patience=args.patience,
            verbose=args.verbose,
            disable_ddp=args.disable_ddp
        )
        
        # Uruchom trening
        trainer.train()
        
        print("\nTraining completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())