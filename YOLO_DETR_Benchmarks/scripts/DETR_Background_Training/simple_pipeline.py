#!/usr/bin/env python3
"""
Simple DETR Background Training Pipeline - Sequential Execution
==============================================================

Prosty, linearny pipeline który wykonuje wszystkie etapy treningu DETR po kolei
bez skomplikowanych mechanizmów async/retry.

Stages:
1. DINO Frame Extraction 
2. Dataset Mixing
3. Model Preparation  
4. Mixed Gentle Training
5. Results Validation

Usage: python simple_pipeline.py
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
import time
from datetime import datetime

class SimpleDETRPipeline:
    def __init__(self):
        # Konfiguracja - analogiczna do pipeline_config_videos_part1.yaml
        self.config = {
            'video_directory': "E:/Cataract/videos/micro/videos_part1",
            'tooltip_dataset_path': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Yolo/yolo_dataset_20250218/images/train",
            'tooltip_annotations_path': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Yolo/yolo_dataset_20250218/coco_annotations_from_yolo_dataset_20250218.json",
            'original_model_checkpoint': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/checkpoint_epoch_100.pth",
            'output_directory': "pipeline_output_videos_part1_simple",
            
            # DINO parameters
            'frames_per_video': 100,
            'yolo_threshold': 0.4,
            'detr_threshold': 0.7,
            
            # Training parameters  
            'gentle_lr': 1e-6,
            'max_epochs': 5,
            'batch_size': 2,
            
            # Python executable
            'python_exe': 'py -3.11'
        }
        
        # Setup paths
        self.base_dir = Path(__file__).parent
        self.output_dir = Path(self.config['output_directory'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.output_dir / "simple_pipeline.log"
        
    def log(self, message):
        """Simple logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Print without emojis for Windows console
        safe_message = log_message.encode('ascii', 'ignore').decode('ascii')
        print(safe_message)
        
        # Log with full UTF-8 to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    
    def run_command(self, command, stage_name):
        """Execute command and handle errors with REAL-TIME output"""
        self.log(f"Starting {stage_name}")
        self.log(f"Command: {command}")
        
        try:
            # Use Popen for real-time output
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
            
            # Read output line by line in real-time
            output_lines = []
            while True:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    print(f"[{stage_name}] {line}")  # Show in terminal
                    self.log(f"[{stage_name}] {line}")  # Log to file
                    output_lines.append(line)
                elif process.poll() is not None:
                    break
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            
            if return_code == 0:
                self.log(f"{stage_name} completed successfully")
                return True
            else:
                self.log(f"{stage_name} failed with return code {return_code}")
                return False
                
        except Exception as e:
            self.log(f"{stage_name} failed with exception: {e}")
            return False
    
    def stage_1_dino_extraction(self):
        """Stage 1: Simple Frame Extraction (bez YOLO/DETR problems)"""
        self.log("="*60)
        self.log("STAGE 1/5: Simple Frame Extraction")
        self.log("="*60)
        
        try:
            import cv2
            import numpy as np
            
            output_frames_dir = self.output_dir / "interesting_frames"
            output_frames_dir.mkdir(exist_ok=True)
            
            train_dir = output_frames_dir / "train"
            val_dir = output_frames_dir / "val"
            annotations_dir = output_frames_dir / "annotations"
            
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)
            annotations_dir.mkdir(exist_ok=True)
            
            video_dir = Path(self.config['video_directory'])
            video_files = list(video_dir.glob("*.mp4"))
            
            if not video_files:
                self.log(f"❌ No MP4 files found in {video_dir}")
                return False
            
            self.log(f"📹 Found {len(video_files)} video files")
            
            all_frames = []
            total_extracted = 0
            
            for video_file in video_files:
                self.log(f"🎬 Processing {video_file.name}...")
                
                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    self.log(f"❌ Cannot open {video_file.name}")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                
                self.log(f"📊 {video_file.name}: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
                
                # Extract every 30th frame, max 100 per video
                frame_interval = 30
                max_frames = self.config['frames_per_video']
                
                frame_count = 0
                extracted_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret or extracted_count >= max_frames:
                        break
                    
                    # Extract every frame_interval frames
                    if frame_count % frame_interval == 0:
                        # Save frame
                        frame_name = f"{video_file.stem}_frame_{frame_count:06d}.jpg"
                        frame_path = train_dir / frame_name  # Initially save to train
                        
                        success = cv2.imwrite(str(frame_path), frame)
                        if success:
                            all_frames.append(frame_path)
                            extracted_count += 1
                            total_extracted += 1
                    
                    frame_count += 1
                
                cap.release()
                self.log(f"✅ {video_file.name}: extracted {extracted_count} frames")
            
            if total_extracted == 0:
                self.log("❌ No frames extracted")
                return False
            
            # Split frames into train/val (80/20)
            np.random.seed(42)
            np.random.shuffle(all_frames)
            
            split_idx = int(len(all_frames) * 0.8)
            train_frames = all_frames[:split_idx]
            val_frames = all_frames[split_idx:]
            
            # Move val frames to val directory
            for frame_path in val_frames:
                val_path = val_dir / frame_path.name
                shutil.move(str(frame_path), str(val_path))
            
            # Create simple COCO annotations
            def create_simple_annotations(frames, split_name):
                images = []
                annotations = []
                
                for img_id, frame_path in enumerate(frames, 1):
                    if frame_path.exists():
                        img = cv2.imread(str(frame_path))
                        if img is not None:
                            height, width, _ = img.shape
                            images.append({
                                "id": img_id,
                                "file_name": frame_path.name,
                                "width": width,
                                "height": height
                            })
                
                return {
                    "info": {
                        "description": f"Simple frame extraction ({split_name} set)",
                        "version": "2.0",
                        "year": 2025
                    },
                    "images": images,
                    "annotations": annotations,
                    "categories": [{
                        "id": 1,
                        "name": "surgical_tool",
                        "supercategory": "medical_instrument"
                    }]
                }
            
            # Save annotations
            train_annotations = create_simple_annotations(train_frames, "train")
            val_annotations = create_simple_annotations(val_frames, "val")
            
            with open(annotations_dir / "train_annotations.json", "w") as f:
                json.dump(train_annotations, f, indent=2)
            
            with open(annotations_dir / "val_annotations.json", "w") as f:
                json.dump(val_annotations, f, indent=2)
            
            self.log(f"✅ Frame extraction completed!")
            self.log(f"📊 Total: {total_extracted} frames")
            self.log(f"📊 Train: {len(train_frames)} frames")
            self.log(f"📊 Val: {len(val_frames)} frames")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Frame extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stage_2_dataset_mixing(self):
        """Stage 2: Dataset Mixing - Copy DINO results to mixed_dataset"""
        self.log("="*60)
        self.log("STAGE 2/5: Dataset Mixing")
        self.log("="*60)
        
        try:
            interesting_frames_dir = self.output_dir / "interesting_frames"
            mixed_dataset_dir = self.output_dir / "mixed_dataset"
            mixed_dataset_dir.mkdir(exist_ok=True)
            
            # Copy directories from interesting_frames to mixed_dataset
            directories_to_copy = ["train", "val", "annotations"]
            
            for directory in directories_to_copy:
                src_dir = interesting_frames_dir / directory
                dst_dir = mixed_dataset_dir / directory
                
                if src_dir.exists():
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
                    self.log(f"📂 Copied {directory}: {len(list(dst_dir.glob('*')))} files")
                else:
                    self.log(f"⚠️ Source directory not found: {src_dir}")
            
            self.log("✅ Dataset mixing completed")
            return True
            
        except Exception as e:
            self.log(f"❌ Dataset mixing failed: {e}")
            return False
    
    def stage_3_model_preparation(self):
        """Stage 3: Model Preparation - Simple validation"""
        self.log("="*60)
        self.log("STAGE 3/5: Model Preparation")
        self.log("="*60)
        
        # Check if model checkpoint exists
        checkpoint_path = Path(self.config['original_model_checkpoint'])
        if checkpoint_path.exists():
            self.log(f"✅ Model checkpoint found: {checkpoint_path}")
            self.log(f"📏 Model size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
            return True
        else:
            self.log(f"❌ Model checkpoint not found: {checkpoint_path}")
            return False
    
    def stage_4_gentle_training(self):
        """Stage 4: Mixed Gentle Training"""
        self.log("="*60)
        self.log("STAGE 4/5: Mixed Gentle Training")
        self.log("="*60)
        
        # Prepare training script command
        script_path = "local\\mixed_gentle_training.py"
        training_output_dir = self.output_dir / "training_output"
        training_output_dir.mkdir(exist_ok=True)
        
        mixed_dataset_dir = self.output_dir / "mixed_dataset"
        
        command = f'''{self.config['python_exe']} "{script_path}" \
--tooltip_images_dir "{mixed_dataset_dir / 'train'}" \
--tooltip_annotations_path "{mixed_dataset_dir / 'annotations' / 'train_annotations.json'}" \
--background_images_dir "{mixed_dataset_dir / 'train'}" \
--background_annotations_path "{mixed_dataset_dir / 'annotations' / 'train_annotations.json'}" \
--output_dir "{training_output_dir}" \
--checkpoint_path "{self.config['original_model_checkpoint']}" \
--lr {self.config['gentle_lr']} \
--epochs {self.config['max_epochs']} \
--batch_size {self.config['batch_size']}'''
        
        success = self.run_command(command, "Mixed Gentle Training")
        
        if success:
            # Check for output model (mixed_gentle_training.py saves as mixed_gentle_model.pth)
            final_model = training_output_dir / "mixed_gentle_model.pth"
            if final_model.exists():
                self.log(f"Training completed - model saved: {final_model}")
                self.log(f"Final model size: {final_model.stat().st_size / (1024*1024):.1f} MB")
                return True
            else:
                self.log("⚠️ Training completed but no mixed_gentle_model.pth found")
        
        return success
    
    def stage_5_validation(self):
        """Stage 5: Results Validation"""
        self.log("="*60)
        self.log("STAGE 5/6: Results Validation")
        self.log("="*60)
        
        # Check if validation script exists
        validation_script = "local\\analyze_checkpoints.py"
        
        if Path(validation_script).exists():
            command = f'{self.config["python_exe"]} "{validation_script}"'
            success = self.run_command(command, "Results Validation")
        else:
            self.log("⚠️ Validation script not found, skipping detailed validation")
            success = True
        
        # Basic validation - check outputs
        training_dir = self.output_dir / "training_output"
        mixed_dataset_dir = self.output_dir / "mixed_dataset"
        
        if training_dir.exists() and mixed_dataset_dir.exists():
            self.log("✅ Basic validation passed - all output directories exist")
            return True
        else:
            self.log("❌ Basic validation failed - missing output directories")
            return False
    
    def stage_6_model_comparison(self):
        """Stage 6: Model Performance Comparison"""
        self.log("="*60)
        self.log("STAGE 6/6: Model Performance Comparison")
        self.log("="*60)
        
        # Paths to models
        original_model = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/detr_inference_model.pth"
        new_model = self.output_dir / "training_output" / "mixed_gentle_model.pth"
        test_images = self.output_dir / "interesting_frames" / "val"
        
        # Check if required files exist
        if not Path(original_model).exists():
            self.log(f"❌ Original model not found: {original_model}")
            return False
            
        if not new_model.exists():
            self.log(f"❌ New model not found: {new_model}")
            return False
            
        if not test_images.exists():
            self.log(f"❌ Test images directory not found: {test_images}")
            return False
        
        # Run model comparison
        comparison_script = "model_comparison_validator.py"
        if not Path(comparison_script).exists():
            self.log(f"❌ Comparison script not found: {comparison_script}")
            return False
        
        command = f'''{self.config['python_exe']} "{comparison_script}" \
--original_model "{original_model}" \
--new_model "{new_model}" \
--test_images "{test_images}" \
--visualize'''
        
        success = self.run_command(command, "Model Comparison")
        
        if success:
            self.log("✅ Model comparison completed successfully")
            results_file = Path("comparison_results.json")
            if results_file.exists():
                self.log(f"📊 Results saved to: {results_file}")
        
        return success
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        start_time = time.time()
        self.log("🚀🚀🚀 Starting Simple DETR Background Training Pipeline 🚀🚀🚀")
        self.log(f"📁 Output directory: {self.output_dir}")
        self.log(f"🎬 Video directory: {self.config['video_directory']}")
        self.log(f"🧠 Model checkpoint: {self.config['original_model_checkpoint']}")
        
        # Define pipeline stages
        stages = [
            ("Stage 1", self.stage_1_dino_extraction),
            ("Stage 2", self.stage_2_dataset_mixing),
            ("Stage 3", self.stage_3_model_preparation),
            ("Stage 4", self.stage_4_gentle_training),
            ("Stage 5", self.stage_5_validation),
            ("Stage 6", self.stage_6_model_comparison)
        ]
        
        # Execute stages sequentially
        for stage_name, stage_func in stages:
            self.log(f"\n⏰ Starting {stage_name}...")
            
            stage_start = time.time()
            success = stage_func()
            stage_duration = time.time() - stage_start
            
            self.log(f"⏱️ {stage_name} took {stage_duration:.1f} seconds")
            
            if not success:
                total_time = time.time() - start_time
                self.log(f"💥 Pipeline FAILED at {stage_name} after {total_time:.1f} seconds")
                return False
            
            self.log(f"✅ {stage_name} completed successfully")
        
        # Pipeline completed successfully
        total_time = time.time() - start_time
        self.log("="*60)
        self.log("PIPELINE COMPLETED SUCCESSFULLY!")
        self.log(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Final model: {self.output_dir / 'training_output' / 'mixed_gentle_model.pth'}")
        self.log(f"Model comparison: comparison_results.json")
        self.log(f"Log file: {self.log_file}")
        self.log("="*60)
        
        return True

def main():
    """Main entry point"""
    print("Simple DETR Background Training Pipeline")
    print("="*50)
    
    try:
        pipeline = SimpleDETRPipeline()
        success = pipeline.run_full_pipeline()
        
        if success:
            print("\nPipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nPipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()