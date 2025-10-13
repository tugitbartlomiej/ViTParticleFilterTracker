#!/usr/bin/env python3
"""
STRATEGY TESTING FRAMEWORK FOR DETR MIXED TRAINING
==================================================

Systematyczne testowanie różnych strategii uczenia na tym samym DINO dataset.
Celem jest znalezienie strategii która NIE powoduje catastrophic forgetting.

Usage: python strategy_testing_framework.py --strategy [strategy_name or "all"]
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from typing import Dict, Any, List

class StrategyTestingFramework:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.dino_dataset = self.base_dir / "test_dino_dataset"
        self.results_dir = self.base_dir / "strategy_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Base configuration shared by all strategies
        self.base_config = {
            'dino_dataset': str(self.dino_dataset),
            'tooltip_dataset_path': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Yolo/yolo_dataset_20250218/images/train",
            'tooltip_annotations_path': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Yolo/yolo_dataset_20250218/coco_annotations_from_yolo_dataset_20250218.json",
            'original_model_checkpoint': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/checkpoint_epoch_100.pth",
            'python_exe': 'py -3.11',
            'batch_size': 1,  # Force batch_size=1 to avoid tensor size issues
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Define all strategies
        self.strategies = self._define_strategies()
        
        # Results tracking
        self.results_summary = {
            'test_started': datetime.now().isoformat(),
            'strategies': {}
        }
    
    def _define_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Define all testing strategies"""
        return {
            # 1. BASELINE - Current failing approach
            'baseline': {
                'description': 'Current failing configuration for reference',
                'lr': 1e-6,
                'tooltip_ratio': 0.7,
                'background_ratio': 0.3,
                'epochs': 2,
                'freeze_backbone': False,
                'expected': 'catastrophic_forgetting'
            },
            
            # 2. ULTRA-GENTLE LEARNING
            'ultra_gentle': {
                'description': 'Ultra-low learning rate to minimize forgetting',
                'lr': 1e-8,  # 100x smaller than baseline
                'lr_backbone': 1e-9,
                'lr_classifier': 1e-7,
                'tooltip_ratio': 0.9,
                'background_ratio': 0.1,
                'epochs': 3,
                'freeze_backbone': False
            },
            
            # 3. FREEZE BACKBONE
            'freeze_backbone': {
                'description': 'Freeze feature extraction, train only classifier',
                'lr': 1e-5,  # Can be higher since only classifier
                'tooltip_ratio': 0.85,
                'background_ratio': 0.15,
                'epochs': 3,
                'freeze_backbone': True,
                'freeze_encoder': True,
                'trainable_layers': ['decoder', 'class_labels_classifier', 'bbox_predictor']
            },
            
            # 4. DYNAMIC FREEZING
            'dynamic_freezing': {
                'description': 'Alternating freeze/unfreeze schedule',
                'phases': [
                    {'epoch': 1, 'freeze_backbone': True, 'lr': 1e-6},
                    {'epoch': 2, 'freeze_backbone': False, 'lr': 1e-8},
                    {'epoch': 3, 'freeze_backbone': True, 'lr': 1e-7}
                ],
                'tooltip_ratio': 0.9,
                'background_ratio': 0.1
            },
            
            # 5. TOOLTIP-HEAVY 95/5
            'tooltip_heavy_95': {
                'description': 'Minimal background exposure',
                'lr': 1e-7,
                'tooltip_ratio': 0.95,
                'background_ratio': 0.05,
                'epochs': 2,
                'freeze_backbone': False
            },
            
            # 6. GRADUAL PHASE TRAINING
            'gradual_phases': {
                'description': 'Progressive background introduction',
                'phases': [
                    {'epoch': 1, 'tooltip_ratio': 1.0, 'background_ratio': 0.0, 'lr': 1e-7},
                    {'epoch': 2, 'tooltip_ratio': 0.9, 'background_ratio': 0.1, 'lr': 1e-8},
                    {'epoch': 3, 'tooltip_ratio': 0.8, 'background_ratio': 0.2, 'lr': 1e-9}
                ]
            },
            
            # 7. LOSS WEIGHTING
            'weighted_loss': {
                'description': 'Prioritize tooltip preservation via loss weighting',
                'lr': 1e-7,
                'tooltip_ratio': 0.85,
                'background_ratio': 0.15,
                'epochs': 2,
                'tooltip_loss_weight': 0.9,
                'background_loss_weight': 0.1
            },
            
            # 8. TWO-STAGE TRAINING
            'two_stage': {
                'description': 'Tooltip reinforcement then gentle mixing',
                'stages': [
                    {
                        'name': 'tooltip_reinforcement',
                        'epochs': 1,
                        'tooltip_ratio': 1.0,
                        'background_ratio': 0.0,
                        'lr': 1e-6
                    },
                    {
                        'name': 'gentle_mixing',
                        'epochs': 2,
                        'tooltip_ratio': 0.85,
                        'background_ratio': 0.15,
                        'lr': 1e-8
                    }
                ]
            },
            
            # 9. MEMORY BUFFER (Replay)
            'memory_buffer': {
                'description': 'Explicit tooltip memory replay',
                'lr': 1e-7,
                'tooltip_ratio': 0.8,
                'background_ratio': 0.2,
                'epochs': 2,
                'replay_buffer_size': 50,
                'replay_frequency': 'every_batch'
            },
            
            # 10. ELASTIC WEIGHT CONSOLIDATION
            'ewc_regularization': {
                'description': 'Preserve important weights via Fisher information',
                'lr': 1e-6,
                'tooltip_ratio': 0.85,
                'background_ratio': 0.15,
                'epochs': 2,
                'ewc_lambda': 0.1,
                'compute_fisher': True
            }
        }
    
    def log(self, message: str, strategy_name: str = None):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        # Strategy-specific log file
        if strategy_name:
            log_file = self.results_dir / f"{strategy_name}_training.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_message + "\n")
        
        # Master log file
        master_log = self.results_dir / "master_testing.log"
        with open(master_log, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    
    def run_training_script(self, strategy_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run training with specific strategy configuration"""
        self.log(f"\n{'='*60}", strategy_name)
        self.log(f"TESTING STRATEGY: {strategy_name.upper()}", strategy_name)
        self.log(f"Description: {config.get('description', 'N/A')}", strategy_name)
        self.log(f"{'='*60}", strategy_name)
        
        # Create strategy-specific output directory
        strategy_output = self.results_dir / f"{strategy_name}_output"
        strategy_output.mkdir(exist_ok=True)
        
        # Create training command for this strategy
        command = self._create_strategy_training_command(strategy_name, config, strategy_output)
        
        # Run training
        start_time = time.time()
        
        self.log(f"Running: {command}", strategy_name)
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.base_dir)
            )
            
            # Capture output in real-time
            training_metrics = {
                'losses': [],
                'val_losses': [],
                'epochs_completed': 0
            }
            
            while True:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    self.log(f"[TRAINING] {line}", strategy_name)
                    
                    # Extract metrics from output
                    if "Training Loss:" in line:
                        try:
                            loss = float(line.split(":")[-1].strip())
                            training_metrics['losses'].append(loss)
                        except:
                            pass
                    elif "Validation Loss:" in line:
                        try:
                            loss = float(line.split(":")[-1].strip())
                            training_metrics['val_losses'].append(loss)
                        except:
                            pass
                    elif "Epoch" in line and "/" in line:
                        try:
                            epoch_info = line.split("Epoch")[1].split("/")[0].strip()
                            training_metrics['epochs_completed'] = int(epoch_info)
                        except:
                            pass
                        
                elif process.poll() is not None:
                    break
            
            training_time = time.time() - start_time
            success = process.returncode == 0
            
        except Exception as e:
            self.log(f"ERROR during training: {e}", strategy_name)
            training_time = time.time() - start_time
            success = False
            training_metrics = {'error': str(e)}
        
        # Test the trained model
        self.log(f"\nTesting model performance...", strategy_name)
        test_results = self._test_model_performance(strategy_output / "mixed_gentle_model.pth", strategy_name)
        
        # Compile results
        results = {
            'strategy': strategy_name,
            'config': config,
            'success': success,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'training_metrics': training_metrics,
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save strategy results
        results_file = self.results_dir / f"{strategy_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"\nResults saved to: {results_file}", strategy_name)
        
        return results
    
    def _create_strategy_training_command(self, strategy_name: str, config: Dict[str, Any], output_dir: Path) -> str:
        """Create training command for strategy using strategy_trainer.py"""
        
        # Use strategy_trainer.py with appropriate arguments
        trainer_script = self.base_dir / "strategy_trainer.py"
        
        # Build command arguments - use proper tooltip dataset instead of DINO dataset for tooltip training
        args = [
            f'--strategy_name "{strategy_name}"',
            f'--tooltip_images_dir "{self.base_config["tooltip_dataset_path"]}"',
            f'--tooltip_annotations_path "{self.base_config["tooltip_annotations_path"]}"',
            f'--background_images_dir "{self.dino_dataset}/interesting_frames/train"',
            f'--background_annotations_path "{self.dino_dataset}/interesting_frames/annotations/train_annotations.json"',
            f'--checkpoint_path "{self.base_config["original_model_checkpoint"]}"',
            f'--output_dir "{output_dir}"',
            f'--lr {config.get("lr", 1e-7)}',
            f'--epochs {config.get("epochs", 2)}',
            f'--tooltip_ratio {config.get("tooltip_ratio", 0.9)}',
            f'--batch_size {self.base_config["batch_size"]}',
            f'--device {self.base_config["device"]}'
        ]
        
        # Add differential learning rates if specified
        if 'lr_backbone' in config:
            args.append(f'--lr_backbone {config["lr_backbone"]}')
        if 'lr_classifier' in config:
            args.append(f'--lr_classifier {config["lr_classifier"]}')
        
        # Add freezing options
        if config.get('freeze_backbone', False):
            args.append('--freeze_backbone')
        
        # Build full command
        command = f'{self.base_config["python_exe"]} "{trainer_script}" ' + ' '.join(args)
        
        return command
    
    
    def _test_model_performance(self, model_path: Path, strategy_name: str) -> Dict[str, Any]:
        """Test trained model on validation set with actual inference"""
        
        if not model_path.exists():
            return {'error': f'Model file not found: {model_path}'}
        
        try:
            import torch
            from transformers import DetrForObjectDetection, DetrImageProcessor
            from PIL import Image
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model
            self.log(f"Loading model for testing: {model_path}", strategy_name)
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=1,  # tooltip (+ no_object added automatically)
                ignore_mismatched_sizes=True
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            model.to(device)
            model.eval()
            
            # Load processor
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            
            # Test on validation images
            val_dir = self.dino_dataset / "interesting_frames" / "val"
            
            if not val_dir.exists():
                return {'error': 'Validation directory not found'}
            
            val_images = list(val_dir.glob("*.jpg"))[:20]  # Test on first 20 images
            
            total_detections = 0
            images_with_detections = 0
            confidences = []
            
            self.log(f"Testing on {len(val_images)} validation images...", strategy_name)
            
            for img_path in val_images:
                try:
                    # Load and process image
                    image = Image.open(img_path).convert('RGB')
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Post-process results
                    target_sizes = torch.tensor([image.size[::-1]]).to(device)
                    results = processor.post_process_object_detection(
                        outputs, 
                        target_sizes=target_sizes,
                        threshold=0.5  # Standard threshold
                    )[0]
                    
                    # Count detections
                    num_detections = len(results["scores"])
                    total_detections += num_detections
                    
                    if num_detections > 0:
                        images_with_detections += 1
                        confidences.extend([score.item() for score in results["scores"]])
                    
                except Exception as e:
                    self.log(f"Error processing {img_path.name}: {e}", strategy_name)
            
            # Calculate metrics
            detection_rate = (images_with_detections / len(val_images)) * 100 if val_images else 0
            avg_confidence = np.mean(confidences) if confidences else 0
            
            results = {
                'total_images_tested': len(val_images),
                'tooltip_frames_in_val': len([f for f in val_images if 'tooltip' in f.name]),
                'background_frames_in_val': len([f for f in val_images if 'background' in f.name]),
                'total_detections': total_detections,
                'images_with_detections': images_with_detections,
                'detection_rate_percent': detection_rate,
                'avg_confidence': avg_confidence,
                'model_size_mb': model_path.stat().st_size / (1024 * 1024),
                'test_successful': True
            }
            
            self.log(f"Model test results: {detection_rate:.1f}% detection rate, {avg_confidence:.3f} avg confidence", strategy_name)
            
            return results
            
        except Exception as e:
            self.log(f"Error testing model: {e}", strategy_name)
            return {'error': str(e), 'test_successful': False}
    
    def compare_strategies(self):
        """Compare all tested strategies and rank them"""
        print("\n" + "="*60)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*60)
        
        # Load all results
        all_results = []
        for strategy_name in self.strategies.keys():
            results_file = self.results_dir / f"{strategy_name}_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    all_results.append(json.load(f))
        
        if not all_results:
            print("No results found to compare")
            return
        
        # Sort by success and training time
        all_results.sort(key=lambda x: (x.get('success', False), -x.get('training_time_minutes', 999)))
        
        # Display comparison table
        print(f"\n{'Strategy':<20} {'Success':<10} {'Time (min)':<12} {'Final Loss':<12}")
        print("-" * 60)
        
        for result in all_results:
            strategy = result['strategy']
            success = "Yes" if result.get('success', False) else "No"
            time_min = f"{result.get('training_time_minutes', 0):.1f}"
            
            losses = result.get('training_metrics', {}).get('losses', [])
            final_loss = f"{losses[-1]:.4f}" if losses else "N/A"
            
            print(f"{strategy:<20} {success:<10} {time_min:<12} {final_loss:<12}")
        
        # Save comparison summary
        summary_file = self.results_dir / "strategy_comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'strategies_tested': len(all_results),
                'successful': sum(1 for r in all_results if r.get('success', False)),
                'results': all_results
            }, f, indent=2)
        
        print(f"\nComparison summary saved to: {summary_file}")
        
        # Identify best strategy
        successful_strategies = [r for r in all_results if r.get('success', False)]
        if successful_strategies:
            best = successful_strategies[0]  # Already sorted
            print(f"\nBEST STRATEGY: {best['strategy']}")
            print(f"  Training time: {best['training_time_minutes']:.1f} minutes")
            print(f"  Configuration: {json.dumps(best['config'], indent=4)}")
    
    def run_strategy(self, strategy_name: str):
        """Run a single strategy test"""
        if strategy_name not in self.strategies:
            print(f"ERROR: Unknown strategy '{strategy_name}'")
            print(f"Available strategies: {list(self.strategies.keys())}")
            return
        
        config = self.strategies[strategy_name]
        results = self.run_training_script(strategy_name, config)
        
        # Update summary
        self.results_summary['strategies'][strategy_name] = results
        
        # Save summary
        summary_file = self.results_dir / "testing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results_summary, f, indent=2)
        
        print(f"\nStrategy '{strategy_name}' testing completed!")
        print(f"Success: {results.get('success', False)}")
        print(f"Time: {results.get('training_time_minutes', 0):.1f} minutes")
    
    def run_all_strategies(self):
        """Run all strategies sequentially"""
        print("="*60)
        print("RUNNING ALL STRATEGIES SEQUENTIALLY")
        print(f"Total strategies to test: {len(self.strategies)}")
        print("="*60)
        
        for i, strategy_name in enumerate(self.strategies.keys(), 1):
            print(f"\n[{i}/{len(self.strategies)}] Testing strategy: {strategy_name}")
            self.run_strategy(strategy_name)
            
            # Small delay between strategies
            if i < len(self.strategies):
                print("\nWaiting 10 seconds before next strategy...")
                time.sleep(10)
        
        print("\n" + "="*60)
        print("ALL STRATEGIES TESTED!")
        print("="*60)
        
        # Run comparison
        self.compare_strategies()

def main():
    parser = argparse.ArgumentParser(description="Test DETR mixed training strategies")
    parser.add_argument('--strategy', type=str, default='all',
                       help='Strategy name or "all" to test all strategies')
    parser.add_argument('--compare', action='store_true',
                       help='Only run comparison of existing results')
    
    args = parser.parse_args()
    
    framework = StrategyTestingFramework()
    
    if args.compare:
        framework.compare_strategies()
    elif args.strategy == 'all':
        framework.run_all_strategies()
    else:
        framework.run_strategy(args.strategy)

if __name__ == "__main__":
    main()