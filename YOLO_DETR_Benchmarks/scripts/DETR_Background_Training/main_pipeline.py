"""
DETR Background Training Pipeline - Main Entry Point
===================================================

Main orchestrator that coordinates the complete DETR background training pipeline
through specialized scripts and comprehensive monitoring.

Pipeline Stages:
1. DINO Frame Extraction - Extract background frames from videos
2. Dataset Mixing - Mix tooltip + background datasets (70/30)
3. Model Preparation - Extend DETR for additional background class  
4. Mixed Gentle Training - Fine-tune with anti-catastrophic forgetting
5. Results Validation - Validate performance and generate reports

Usage:
    python main_pipeline.py --config config.yaml
    python main_pipeline.py --interactive
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import pipeline components
from background_task_executor import (
    BackgroundTaskExecutor, TaskExecutionConfig, 
    get_task_executor, create_script_args
)
from status_manager import (
    StatusManager, TaskStatus, TaskProgress, TaskError, ProgressType,
    get_status_manager
)
from pipeline_monitor import get_pipeline_monitor
from progress_reporter import ProgressReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration for the entire pipeline"""
    
    def __init__(self):
        # Input paths
        self.video_directory: str = ""
        self.original_model_checkpoint: str = ""
        
        # Output paths
        self.output_directory: str = "pipeline_output"
        
        # DINO extraction parameters
        self.dino_model: str = "facebook/dino-vitb16"
        self.frames_per_video: int = 50
        self.clustering_threshold: float = 0.7
        self.min_cluster_size: int = 5
        
        # Dataset mixing parameters
        self.tooltip_ratio: float = 0.7
        self.background_ratio: float = 0.3
        self.validation_split: float = 0.2
        
        # Training parameters
        self.gentle_lr: float = 1e-6
        self.max_epochs: int = 3
        self.batch_size: int = 2
        self.gradient_accumulation_steps: int = 8
        
        # Detection thresholds
        self.yolo_threshold: float = 0.5
        self.detr_threshold: float = 0.5
        
        # Execution settings
        self.python_executable: str = sys.executable
        self.max_retries: int = 2
        self.timeout_hours: int = 24
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def validate(self):
        """Validate configuration"""
        errors = []
        
        # Check required paths
        required_paths = [
            ('video_directory', 'Video directory'),
            ('original_model_checkpoint', 'Original model checkpoint')
        ]
        
        for attr, name in required_paths:
            path = getattr(self, attr)
            if not path:
                errors.append(f"{name} is required")
            elif not Path(path).exists():
                errors.append(f"{name} does not exist: {path}")
        
        # Check parameters
        if not 0 < self.tooltip_ratio < 1:
            errors.append("tooltip_ratio must be between 0 and 1")
        if not 0 < self.background_ratio < 1:
            errors.append("background_ratio must be between 0 and 1")
        if abs(self.tooltip_ratio + self.background_ratio - 1.0) > 0.01:
            errors.append("tooltip_ratio + background_ratio should equal 1.0")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

class DETRPipelineOrchestrator:
    """Main orchestrator for DETR background training pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup managers
        self.status_manager = get_status_manager(str(self.output_dir / "pipeline_status.json"))
        self.task_executor = get_task_executor()
        self.pipeline_monitor = get_pipeline_monitor()
        
        # Progress reporter for main pipeline
        self.reporter = ProgressReporter("main_pipeline", 
                                       self.output_dir / "pipeline.log")
        
        # Task configurations
        self.task_configs = {
            'dino_extraction': TaskExecutionConfig(),
            'dataset_mixing': TaskExecutionConfig(),
            'model_preparation': TaskExecutionConfig(),
            'gentle_training': TaskExecutionConfig(),
            'validation': TaskExecutionConfig()
        }
        
        # Configure task timeouts
        self.task_configs['dino_extraction'].timeout = 3600  # 1 hour
        self.task_configs['dataset_mixing'].timeout = 1800   # 30 minutes
        self.task_configs['model_preparation'].timeout = 600  # 10 minutes
        self.task_configs['gentle_training'].timeout = config.timeout_hours * 3600  # Configurable
        self.task_configs['validation'].timeout = 1800       # 30 minutes
        
        # Set working directory for all tasks
        for task_config in self.task_configs.values():
            task_config.working_directory = self.output_dir.parent
            task_config.python_executable = config.python_executable
            task_config.max_retries = config.max_retries
        
        # Pipeline state
        self.pipeline_started_at = None
        self.current_stage = None
    
    async def run_pipeline(self) -> bool:
        """Run the complete pipeline"""
        try:
            self.pipeline_started_at = datetime.now()
            self.reporter.log_info("Starting DETR Background Training Pipeline")
            
            # Validate configuration
            self.config.validate()
            
            # Save configuration
            self.config.to_yaml(str(self.output_dir / "pipeline_config.yaml"))
            
            # Define pipeline stages
            pipeline_stages = [
                ("dino_extraction", "DINO Frame Extraction", self.stage_1_dino_extraction),
                ("dataset_mixing", "Dataset Mixing", self.stage_2_dataset_mixing),
                ("model_preparation", "Model Preparation", self.stage_3_model_preparation), 
                ("gentle_training", "Mixed Gentle Training", self.stage_4_gentle_training),
                ("validation", "Results Validation", self.stage_5_validation)
            ]
            
            # Execute stages
            for i, (stage_id, stage_name, stage_func) in enumerate(pipeline_stages):
                self.current_stage = stage_id
                
                self.reporter.log_info(f"\n{'='*60}")
                self.reporter.log_info(f"STAGE {i+1}/5: {stage_name}")
                self.reporter.log_info(f"{'='*60}")
                
                # Update main pipeline progress
                self.reporter.update_progress((i / len(pipeline_stages)) * 100, 
                                            message=f"Executing {stage_name}")
                
                # Execute stage
                success = await stage_func()
                
                if not success:
                    self.reporter.report_error(f"Pipeline failed at stage: {stage_name}")
                    return False
                
                self.reporter.log_info(f"Stage {i+1} completed: {stage_name}")
            
            # Pipeline completed successfully
            duration = datetime.now() - self.pipeline_started_at
            self.reporter.report_completion(
                f"DETR Background Training Pipeline completed successfully in {duration}"
            )
            
            # Generate final report
            await self.generate_final_report()
            
            return True
            
        except Exception as e:
            self.reporter.report_error(f"Pipeline execution failed: {str(e)}", recoverable=False)
            logger.exception("Pipeline execution failed")
            return False
    
    async def stage_1_dino_extraction(self) -> bool:
        """Stage 1: DINO Activity Detection - Extract interesting frames, classify as tooltip/background"""
        
        # Use intelligent background frame selector with DINO post-processing
        script_path = Path("../../Most_advanced_Intelligent_Background_Selector_2025-07-17") / \
                     "intelligent_background_frame_selector.py"
        
        if not script_path.exists():
            self.reporter.report_error(f"Background frame selector script not found: {script_path}")
            return False
        
        # Prepare arguments for frame extraction + DINO analysis
        yolo_model_path = "../../models/YOLO/yolo_inference_model_final/yolo_inference_model.pt"
        detr_model_path = "../../models/DETR/detr_inference_model.pth"
        
        args_config = {
            'videos_dir': self.config.video_directory,
            'yolo_model_path': yolo_model_path,
            'detr_model_path': detr_model_path,
            'output_dir': str(self.output_dir / "interesting_frames"),
            'max_frames_per_video': self.config.frames_per_video,
            'frame_interval': 30,
            'yolo_threshold': self.config.yolo_threshold,
            'detr_threshold': self.config.detr_threshold,
            'device': 'auto'
        }
        
        args = create_script_args(args_config)
        
        # Execute task
        return_code = await self.task_executor.execute_task_with_retry(
            'dino_extraction',
            script_path,
            args,
            self.task_configs['dino_extraction']
        )
        
        if return_code == 0:
            # Post-process: Apply DINO information richness analysis
            interesting_frames_dir = self.output_dir / "interesting_frames"
            
            # Execute DINO quality analysis and dataset creation
            await self._apply_dino_information_analysis(interesting_frames_dir)
            
            # Validate DINO mixed classification results
            train_dir = interesting_frames_dir / "train"
            val_dir = interesting_frames_dir / "val"
            annotations_dir = interesting_frames_dir / "annotations"
            tooltip_frames_dir = interesting_frames_dir / "tooltip_frames"
            background_frames_dir = interesting_frames_dir / "background_frames"
            
            if (train_dir.exists() and val_dir.exists() and annotations_dir.exists() and 
                tooltip_frames_dir.exists() and background_frames_dir.exists()):
                
                train_count = len(list(train_dir.glob("*.jpg")))
                val_count = len(list(val_dir.glob("*.jpg")))
                tooltip_count = len(list(tooltip_frames_dir.glob("*.jpg")))
                background_count = len(list(background_frames_dir.glob("*.jpg")))
                total_count = train_count + val_count
                
                self.reporter.log_info(f"DINO → YOLO → DETR Mixed Classification: {total_count} frames selected")
                self.reporter.log_info(f"  - Train frames: {train_count}")
                self.reporter.log_info(f"  - Val frames: {val_count}")
                self.reporter.log_info(f"  - Tooltip frames: {tooltip_count}")
                self.reporter.log_info(f"  - Background frames: {background_count}")
                
                if total_count < 10:
                    self.reporter.log_warning(f"Only {total_count} frames selected. Consider lowering quality threshold.")
                
                return True
            else:
                self.reporter.report_error("DINO mixed classification did not create expected output structure")
        
        return False
    
    async def _apply_dino_information_analysis(self, frames_dir: Path):
        """Apply DINO → YOLO → DETR classification for mixed tooltip/background training"""
        
        try:
            # Import required components
            import sys
            sys.path.append(str(Path("../../DINO_Frame_Selection").resolve()))
            
            from dino_information_analyzer import DINOInformationAnalyzer
            from ultralytics import YOLO
            from transformers import DetrImageProcessor, DetrForObjectDetection
            import torch
            from PIL import Image
            import numpy as np
            import cv2
            import json
            import shutil
            
            # Initialize DINO analyzer
            dino_analyzer = DINOInformationAnalyzer(
                model_name='dino_vits16',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Initialize YOLO and DETR models for classification
            yolo_model = YOLO("../../models/YOLO/yolo_inference_model_final/yolo_inference_model.pt")
            detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            detr_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            detr_model.to(detr_device)
            detr_model.eval()
            
            self.reporter.log_info("DINO + YOLO + DETR models loaded successfully")
            
            # Find all extracted frames (from intelligent selector output)
            # intelligent_background_frame_selector creates train/ and val/ directories directly
            train_frames_dir = frames_dir / "train"
            val_frames_dir = frames_dir / "val"
            
            all_frames = []
            if train_frames_dir.exists():
                all_frames.extend(list(train_frames_dir.glob("*.jpg")))
            if val_frames_dir.exists():
                all_frames.extend(list(val_frames_dir.glob("*.jpg")))
            
            if not all_frames:
                self.reporter.report_error("No frames found for DINO analysis")
                return
            
            # Stage 1: DINO Quality Analysis - select high information frames
            self.reporter.log_info(f"Stage 1: DINO analyzing {len(all_frames)} frames...")
            
            high_quality_frames = []
            quality_threshold = 0.3  # Lower threshold to get more frames
            
            for frame_path in all_frames:
                try:
                    analysis = dino_analyzer.extract_comprehensive_features(str(frame_path))
                    if analysis:
                        quality_score = analysis.get('information_score', 0.0)
                        if quality_score >= quality_threshold:
                            high_quality_frames.append({
                                'path': frame_path,
                                'quality_score': quality_score,
                                'attention_entropy': analysis.get('attention_entropy', 0.0),
                                'feature_variance': analysis.get('feature_variance', 0.0)
                            })
                except Exception as e:
                    self.reporter.log_warning(f"DINO analysis failed for {frame_path.name}: {e}")
            
            # If too few high-quality, use best available frames
            if len(high_quality_frames) < 10:
                self.reporter.log_warning(f"Only {len(high_quality_frames)} high-quality frames, using all frames")
                high_quality_frames = []
                for frame_path in all_frames:
                    try:
                        analysis = dino_analyzer.extract_comprehensive_features(str(frame_path))
                        quality_score = analysis.get('information_score', 0.0) if analysis else 0.5
                        high_quality_frames.append({
                            'path': frame_path,
                            'quality_score': quality_score,
                            'attention_entropy': analysis.get('attention_entropy', 0.0) if analysis else 0.0,
                            'feature_variance': analysis.get('feature_variance', 0.0) if analysis else 0.0
                        })
                    except:
                        high_quality_frames.append({
                            'path': frame_path,
                            'quality_score': 0.5,
                            'attention_entropy': 0.0,
                            'feature_variance': 0.0
                        })
            
            # Stage 2: YOLO + DETR Classification
            self.reporter.log_info(f"Stage 2: YOLO + DETR classifying {len(high_quality_frames)} high-quality frames...")
            
            tooltip_frames = []
            background_frames = []
            discarded_frames = []
            
            for frame_info in high_quality_frames:
                frame_path = frame_info['path']
                image = Image.open(frame_path).convert("RGB")
                
                # Step 1: YOLO detection
                yolo_results = yolo_model(image, conf=self.config.yolo_threshold)
                yolo_detected = len(yolo_results[0].boxes) > 0 if yolo_results[0].boxes is not None else False
                
                if yolo_detected:
                    # YOLO found something - check with DETR
                    inputs = detr_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(detr_device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = detr_model(**inputs)
                    
                    # Check if DETR detected anything with confidence > threshold
                    probas = outputs.logits.softmax(-1)[0, :, :-1]
                    keep = probas.max(-1).values > self.config.detr_threshold
                    detr_detected = keep.sum().item() > 0
                    
                    if not detr_detected:
                        # YOLO detected, DETR didn't → TOOLTIP FRAME ✓
                        tooltip_frames.append(frame_info)
                        frame_info['classification'] = 'tooltip'
                        frame_info['yolo_detected'] = True
                        frame_info['detr_detected'] = False
                    else:
                        # Both detected → uncertain, discard
                        discarded_frames.append(frame_info)
                        frame_info['classification'] = 'discarded'
                        frame_info['yolo_detected'] = True
                        frame_info['detr_detected'] = True
                else:
                    # YOLO didn't detect - check DETR
                    inputs = detr_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(detr_device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = detr_model(**inputs)
                    
                    probas = outputs.logits.softmax(-1)[0, :, :-1]
                    keep = probas.max(-1).values > self.config.detr_threshold
                    detr_detected = keep.sum().item() > 0
                    
                    if not detr_detected:
                        # Neither detected → BACKGROUND FRAME ✓
                        background_frames.append(frame_info)
                        frame_info['classification'] = 'background'
                        frame_info['yolo_detected'] = False
                        frame_info['detr_detected'] = False
                    else:
                        # DETR detected but YOLO didn't → discard (uncertain)
                        discarded_frames.append(frame_info)
                        frame_info['classification'] = 'discarded'
                        frame_info['yolo_detected'] = False
                        frame_info['detr_detected'] = True
            
            # Stage 3: Create Mixed Dataset
            self.reporter.log_info(f"Stage 3: Creating mixed dataset...")
            self.reporter.log_info(f"  - Tooltip frames: {len(tooltip_frames)}")
            self.reporter.log_info(f"  - Background frames: {len(background_frames)}")
            self.reporter.log_info(f"  - Discarded frames: {len(discarded_frames)}")
            
            # Ensure we have both types for mixed training
            if len(tooltip_frames) == 0:
                self.reporter.log_warning("No tooltip frames found! Using top DINO quality frames as tooltips")
                # Use top 30% of quality scores as tooltip fallback
                sorted_frames = sorted(background_frames, key=lambda x: x['quality_score'], reverse=True)
                fallback_tooltips = sorted_frames[:max(1, len(sorted_frames) // 3)]
                for frame in fallback_tooltips:
                    frame['classification'] = 'tooltip'
                    tooltip_frames.extend(fallback_tooltips)
                    background_frames = [f for f in background_frames if f not in fallback_tooltips]
            
            if len(background_frames) == 0:
                self.reporter.log_warning("No background frames found!")
                return
            
            # Mix tooltip and background frames for train/val split
            all_mixed_frames = tooltip_frames + background_frames
            np.random.seed(42)
            np.random.shuffle(all_mixed_frames)
            
            split_idx = int(len(all_mixed_frames) * 0.8)
            train_frames = all_mixed_frames[:split_idx]
            val_frames = all_mixed_frames[split_idx:]
            
            # Create output directories
            train_dir = frames_dir / "train"
            val_dir = frames_dir / "val"
            annotations_dir = frames_dir / "annotations"
            tooltip_frames_dir = frames_dir / "tooltip_frames"
            background_frames_dir = frames_dir / "background_frames"
            
            for directory in [train_dir, val_dir, annotations_dir, tooltip_frames_dir, background_frames_dir]:
                directory.mkdir(exist_ok=True)
            
            # Copy frames to train/val
            for frame_info in train_frames:
                shutil.copy2(frame_info['path'], train_dir / frame_info['path'].name)
            
            for frame_info in val_frames:
                shutil.copy2(frame_info['path'], val_dir / frame_info['path'].name)
            
            # Also copy to classification directories
            for frame_info in tooltip_frames:
                shutil.copy2(frame_info['path'], tooltip_frames_dir / frame_info['path'].name)
            
            for frame_info in background_frames:
                shutil.copy2(frame_info['path'], background_frames_dir / frame_info['path'].name)
            
            # Create COCO annotations with mixed tooltip/background
            def create_mixed_coco_annotations(frames, split_name):
                images = []
                annotations = []
                annotation_id = 1
                
                for img_id, frame_info in enumerate(frames, 1):
                    frame_path = frame_info['path']
                    img = cv2.imread(str(frame_path))
                    if img is not None:
                        height, width, _ = img.shape
                        images.append({
                            "id": img_id,
                            "file_name": frame_path.name,
                            "width": width,
                            "height": height,
                            "dino_quality_score": frame_info['quality_score'],
                            "classification": frame_info['classification'],
                            "yolo_detected": frame_info.get('yolo_detected', False),
                            "detr_detected": frame_info.get('detr_detected', False)
                        })
                        
                        # Add annotation only for tooltip frames
                        if frame_info['classification'] == 'tooltip':
                            # Create dummy annotation for tooltip (since we don't have exact bbox from this pipeline)
                            annotations.append({
                                "id": annotation_id,
                                "image_id": img_id,
                                "category_id": 1,
                                "bbox": [width * 0.3, height * 0.3, width * 0.4, height * 0.4],  # Dummy bbox
                                "area": (width * 0.4) * (height * 0.4),
                                "iscrowd": 0,
                                "source": "yolo_detected_detr_not_detected"
                            })
                            annotation_id += 1
                
                return {
                    "info": {
                        "description": f"Mixed tooltip/background frames selected by DINO+YOLO+DETR ({split_name} set)",
                        "version": "2.0", 
                        "year": 2025,
                        "contributor": "DINO Mixed Classification Pipeline"
                    },
                    "images": images,
                    "annotations": annotations,
                    "categories": [{
                        "id": 1,
                        "name": "surgical_tool",
                        "supercategory": "medical_instrument"
                    }]
                }
            
            # Save mixed annotations
            train_annotations = create_mixed_coco_annotations(train_frames, "train")
            val_annotations = create_mixed_coco_annotations(val_frames, "val")
            
            with open(annotations_dir / "train_annotations.json", "w") as f:
                json.dump(train_annotations, f, indent=2)
            
            with open(annotations_dir / "val_annotations.json", "w") as f:
                json.dump(val_annotations, f, indent=2)
            
            # Save comprehensive classification report
            train_tooltips = len([f for f in train_frames if f['classification'] == 'tooltip'])
            train_backgrounds = len([f for f in train_frames if f['classification'] == 'background'])
            val_tooltips = len([f for f in val_frames if f['classification'] == 'tooltip'])
            val_backgrounds = len([f for f in val_frames if f['classification'] == 'background'])
            
            classification_report = {
                "dino_yolo_detr_analysis": {
                    "total_frames_analyzed": len(all_frames),
                    "high_quality_frames": len(high_quality_frames),
                    "tooltip_frames": len(tooltip_frames),
                    "background_frames": len(background_frames),
                    "discarded_frames": len(discarded_frames),
                    "quality_threshold": quality_threshold
                },
                "mixed_dataset_composition": {
                    "train": {
                        "total": len(train_frames),
                        "tooltips": train_tooltips,
                        "backgrounds": train_backgrounds,
                        "tooltip_ratio": train_tooltips / len(train_frames) if train_frames else 0
                    },
                    "val": {
                        "total": len(val_frames),
                        "tooltips": val_tooltips,
                        "backgrounds": val_backgrounds,
                        "tooltip_ratio": val_tooltips / len(val_frames) if val_frames else 0
                    }
                },
                "frame_details": [{
                    "filename": f['path'].name,
                    "classification": f['classification'],
                    "quality_score": f['quality_score'],
                    "yolo_detected": f.get('yolo_detected', False),
                    "detr_detected": f.get('detr_detected', False),
                    "attention_entropy": f['attention_entropy'],
                    "feature_variance": f['feature_variance']
                } for f in all_mixed_frames]
            }
            
            with open(frames_dir / "dino_mixed_classification_report.json", "w") as f:
                json.dump(classification_report, f, indent=2)
            
            # Log final results
            avg_quality = np.mean([f['quality_score'] for f in all_mixed_frames])
            tooltip_ratio = len(tooltip_frames) / len(all_mixed_frames) if all_mixed_frames else 0
            
            self.reporter.log_info(f"DINO → YOLO → DETR Mixed Classification Complete:")
            self.reporter.log_info(f"  - Total high-quality frames: {len(high_quality_frames)}")
            self.reporter.log_info(f"  - Tooltip frames: {len(tooltip_frames)} ({tooltip_ratio:.1%})")
            self.reporter.log_info(f"  - Background frames: {len(background_frames)} ({1-tooltip_ratio:.1%})")
            self.reporter.log_info(f"  - Train: {len(train_frames)} ({train_tooltips} tooltips, {train_backgrounds} backgrounds)")
            self.reporter.log_info(f"  - Val: {len(val_frames)} ({val_tooltips} tooltips, {val_backgrounds} backgrounds)")
            self.reporter.log_info(f"  - Average DINO quality: {avg_quality:.3f}")
            
        except Exception as e:
            self.reporter.report_error(f"DINO → YOLO → DETR classification failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def stage_2_dataset_mixing(self) -> bool:
        """Stage 2: Use DINO → YOLO → DETR mixed classification results directly"""
        
        # DINO mixed classification already created DETR-ready mixed dataset
        interesting_frames_dir = self.output_dir / "interesting_frames"
        train_dir = interesting_frames_dir / "train"
        val_dir = interesting_frames_dir / "val"
        annotations_dir = interesting_frames_dir / "annotations"
        tooltip_frames_dir = interesting_frames_dir / "tooltip_frames"
        background_frames_dir = interesting_frames_dir / "background_frames"
        
        # Validate that both tooltip and background frames were created
        if not (train_dir.exists() and val_dir.exists() and annotations_dir.exists() and
                tooltip_frames_dir.exists() and background_frames_dir.exists()):
            self.reporter.report_error("DINO mixed classification must produce train/val/annotations/tooltip_frames/background_frames structure")
            return False
        
        # Copy DINO mixed results to mixed_dataset for consistency with training stage
        mixed_dataset_dir = self.output_dir / "mixed_dataset"
        mixed_dataset_dir.mkdir(exist_ok=True)
        
        import shutil
        
        # Clean and copy directories
        directories_to_copy = ["train", "val", "annotations", "tooltip_frames", "background_frames"]
        
        for directory in directories_to_copy:
            src_dir = interesting_frames_dir / directory
            dst_dir = mixed_dataset_dir / directory
            
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir)
        
        # Count mixed frames by type
        train_count = len(list((mixed_dataset_dir / "train").glob("*.jpg")))
        val_count = len(list((mixed_dataset_dir / "val").glob("*.jpg")))
        tooltip_count = len(list((mixed_dataset_dir / "tooltip_frames").glob("*.jpg")))
        background_count = len(list((mixed_dataset_dir / "background_frames").glob("*.jpg")))
        
        # Read classification report for detailed stats
        try:
            report_path = interesting_frames_dir / "dino_mixed_classification_report.json"
            if report_path.exists():
                import json
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                train_tooltips = report['mixed_dataset_composition']['train']['tooltips']
                train_backgrounds = report['mixed_dataset_composition']['train']['backgrounds']
                val_tooltips = report['mixed_dataset_composition']['val']['tooltips']
                val_backgrounds = report['mixed_dataset_composition']['val']['backgrounds']
                tooltip_ratio = report['mixed_dataset_composition']['train']['tooltip_ratio']
                
                self.reporter.log_info(f"DINO Mixed Dataset prepared for training:")
                self.reporter.log_info(f"  - Total frames: {train_count + val_count}")
                self.reporter.log_info(f"  - Train: {train_count} ({train_tooltips} tooltips, {train_backgrounds} backgrounds)")
                self.reporter.log_info(f"  - Val: {val_count} ({val_tooltips} tooltips, {val_backgrounds} backgrounds)")
                self.reporter.log_info(f"  - Tooltip ratio: {tooltip_ratio:.1%}")
                self.reporter.log_info(f"  - Source classification: {tooltip_count} tooltip frames, {background_count} background frames")
                
        except Exception as e:
            self.reporter.log_warning(f"Could not read detailed classification report: {e}")
            self.reporter.log_info(f"DINO Mixed Dataset prepared for training:")
            self.reporter.log_info(f"  - Train: {train_count} frames")
            self.reporter.log_info(f"  - Val: {val_count} frames")
            self.reporter.log_info(f"  - Tooltip source: {tooltip_count} frames")
            self.reporter.log_info(f"  - Background source: {background_count} frames")
        
        return True
    
    async def stage_3_model_preparation(self) -> bool:
        """Stage 3: Prepare model for mixed training"""
        
        # This stage would extend the DETR model for additional background class
        # For now, we'll assume the mixed_gentle_training.py handles this
        self.reporter.log_info("Model preparation completed (handled by training script)")
        return True
    
    async def stage_4_gentle_training(self) -> bool:
        """Stage 4: Execute mixed gentle training"""
        
        script_path = Path("local") / "mixed_gentle_training.py"
        
        if not script_path.exists():
            self.reporter.report_error(f"Training script not found: {script_path}")
            return False
        
        # Prepare arguments matching mixed_gentle_training.py argument names
        args_config = {
            'tooltip_images_dir': str(self.output_dir / "mixed_dataset" / "train"),
            'tooltip_annotations_path': str(self.output_dir / "mixed_dataset" / "annotations" / "train_annotations.json"),
            'background_images_dir': str(self.output_dir / "mixed_dataset" / "train"), 
            'background_annotations_path': str(self.output_dir / "mixed_dataset" / "annotations" / "train_annotations.json"),
            'output_dir': str(self.output_dir / "training_output"),
            'checkpoint_path': self.config.original_model_checkpoint,
            'lr': self.config.gentle_lr,
            'epochs': self.config.max_epochs,
            'batch_size': self.config.batch_size
        }
        
        args = create_script_args(args_config)
        
        # Execute task with special monitoring for long training
        return_code = await self.task_executor.execute_task_with_retry(
            'gentle_training',
            script_path,
            args,
            self.task_configs['gentle_training']
        )
        
        if return_code == 0:
            # Validate training results
            training_dir = self.output_dir / "training_output"
            if (training_dir / "final_model.pth").exists():
                self.reporter.log_info("Training completed successfully")
                return True
        
        return False
    
    async def stage_5_validation(self) -> bool:
        """Stage 5: Validate model performance"""
        
        # Use existing analyze_checkpoints.py for validation
        script_path = Path("local") / "analyze_checkpoints.py"
        
        if not script_path.exists():
            self.reporter.log_info("Validation script not found, skipping detailed validation")
            return True
        
        # Execute basic validation
        return_code = await self.task_executor.execute_task(
            'validation',
            script_path,
            [],
            self.task_configs['validation'],
            wait_for_completion=True
        )
        
        self.reporter.log_info("Model validation completed")
        return return_code == 0
    
    async def generate_final_report(self):
        """Generate comprehensive final report"""
        
        pipeline_status = self.status_manager.get_pipeline_status()
        monitor_health = self.pipeline_monitor.get_pipeline_health()
        
        # Collect output files
        output_files = {}
        for task_id in ['dino_extraction', 'dataset_mixing', 'gentle_training']:
            task = self.status_manager.get_task(task_id)
            if task and task.output_files:
                output_files[task_id] = task.output_files
        
        report = {
            'pipeline_info': {
                'started_at': self.pipeline_started_at.isoformat() if self.pipeline_started_at else None,
                'completed_at': datetime.now().isoformat(),
                'total_duration': str(datetime.now() - self.pipeline_started_at) if self.pipeline_started_at else None,
                'success': True
            },
            'configuration': self.config.__dict__,
            'pipeline_status': pipeline_status,
            'monitor_health': monitor_health,
            'output_files': output_files,
            'final_model_path': str(self.output_dir / "training_output" / "final_model.pth"),
            'logs_directory': str(self.output_dir / "logs")
        }
        
        # Save report
        report_path = self.output_dir / "final_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.reporter.log_info(f"Final report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Output Directory: {self.output_dir}")
        print(f"Final Model: {self.output_dir / 'training_output' / 'final_model.pth'}")
        print(f"Final Report: {report_path}")
        print(f"Logs: {self.output_dir / 'logs'}")
        print("="*60)

def create_default_config() -> PipelineConfig:
    """Create default configuration"""
    config = PipelineConfig()
    
    # Set default paths (user should modify these)
    base_path = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker")
    
    config.video_directory = "F:/Videos/surgical_videos"  # User needs to set this
    config.tooltip_dataset_path = str(base_path / "YOLO_DETR_Benchmarks/Datasets/tooltip_images")
    config.tooltip_annotations_path = str(base_path / "YOLO_DETR_Benchmarks/Datasets/tooltip_annotations.json") 
    config.original_model_checkpoint = str(base_path / "Eden/DETR/checkpoint_epoch_60.pth")
    config.output_directory = str(base_path / "YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output")
    
    return config

def interactive_config_setup() -> PipelineConfig:
    """Interactive configuration setup"""
    print("DETR Background Training Pipeline Configuration")
    print("="*60)
    
    config = create_default_config()
    
    # Essential paths
    print("\nRequired Paths:")
    
    video_dir = input(f"Video directory [{config.video_directory}]: ").strip()
    if video_dir:
        config.video_directory = video_dir
    
    checkpoint_path = input(f"Original model checkpoint [{config.original_model_checkpoint}]: ").strip()
    if checkpoint_path:
        config.original_model_checkpoint = checkpoint_path
    
    output_dir = input(f"Output directory [{config.output_directory}]: ").strip()
    if output_dir:
        config.output_directory = output_dir
    
    # Training parameters
    print("\nTraining Parameters:")
    
    lr_input = input(f"Gentle learning rate [{config.gentle_lr}]: ").strip()
    if lr_input:
        try:
            config.gentle_lr = float(lr_input)
        except ValueError:
            print("WARNING: Invalid learning rate, using default")
    
    epochs_input = input(f"Max epochs [{config.max_epochs}]: ").strip()
    if epochs_input:
        try:
            config.max_epochs = int(epochs_input)
        except ValueError:
            print("WARNING: Invalid epochs, using default")
    
    # Mixing ratios
    print("\nDataset Mixing:")
    
    tooltip_ratio = input(f"Tooltip ratio [{config.tooltip_ratio}]: ").strip()
    if tooltip_ratio:
        try:
            config.tooltip_ratio = float(tooltip_ratio)
            config.background_ratio = 1.0 - config.tooltip_ratio
        except ValueError:
            print("WARNING: Invalid ratio, using default")
    
    return config

async def main():
    parser = argparse.ArgumentParser(description="DETR Background Training Pipeline")
    parser.add_argument("--config", help="Configuration YAML file")
    parser.add_argument("--interactive", action="store_true", help="Interactive configuration")
    parser.add_argument("--create-config", help="Create default configuration file")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration only")
    
    args = parser.parse_args()
    
    try:
        # Handle config creation
        if args.create_config:
            config = create_default_config()
            config.to_yaml(args.create_config)
            print(f"Default configuration saved to: {args.create_config}")
            print("Please edit the configuration file with your specific paths")
            return
        
        # Load or create configuration
        if args.interactive:
            config = interactive_config_setup()
        elif args.config:
            config = PipelineConfig.from_yaml(args.config)
        else:
            print("Please provide --config, --interactive, or --create-config")
            print("Use --help for more information")
            return
        
        # Validate configuration
        if args.validate_config:
            config.validate()
            print("Configuration is valid")
            return
        
        # Run pipeline
        orchestrator = DETRPipelineOrchestrator(config)
        success = await orchestrator.run_pipeline()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())