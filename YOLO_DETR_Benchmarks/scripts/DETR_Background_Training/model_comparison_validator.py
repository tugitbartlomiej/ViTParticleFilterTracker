#!/usr/bin/env python3
"""
DETR Model Comparison & Validation Script
=========================================

Porównuje nowy mixed-trained model z oryginalnym detr_inference_model.pth
na tooltip detection performance.

HIPOTEZA: Mixed training (tooltips + background) → lepsze negative examples 
→ precyzyjniejsze wykrywanie tooltipów → mniej false positives
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class DETRModelComparator:
    def __init__(self, original_model_path, new_model_path, processor_name="facebook/detr-resnet-50"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = DetrImageProcessor.from_pretrained(processor_name)
        
        # Load models
        print(f"Loading models on {self.device}...")
        self.original_model = self._load_model(original_model_path, "Original DETR")
        self.new_model = self._load_model(new_model_path, "Mixed-trained DETR")
        
        # Detection threshold
        self.confidence_threshold = 0.5
        
    def _load_model(self, model_path, model_name):
        """Load DETR model from checkpoint with robust format handling"""
        try:
            print(f"Loading {model_name} from {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract state dict - handle various checkpoint formats
            state_dict = None
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Handle nested model_state_dict
                    nested = checkpoint['model_state_dict']
                    if isinstance(nested, dict) and 'model_state_dict' in nested:
                        state_dict = nested['model_state_dict']  # Double nesting
                    else:
                        state_dict = nested
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    # Assume checkpoint is the state_dict itself
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Determine model architecture from state_dict
            if 'class_labels_classifier.weight' in state_dict:
                num_classes = state_dict['class_labels_classifier.weight'].shape[0]
                print(f"Found class_labels_classifier with {num_classes} classes")
                
                # Create model with correct number of classes
                if num_classes <= 2:  # Custom tooltip model (1 class + no-object)
                    from transformers import DetrConfig
                    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
                    config.num_labels = 1  # 1 class (tooltip) + no-object handled automatically
                    model = DetrForObjectDetection(config)
                    print(f"Created custom DETR model with 1 class for {model_name}")
                else:  # Standard COCO model
                    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                    print(f"Created standard COCO DETR model for {model_name}")
            else:
                # Fallback - assume standard COCO model
                model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                print(f"Fallback to standard COCO model for {model_name}")
            
            # Load state dict with relaxed matching
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in {model_name}: {len(missing_keys)} keys")
                if len(missing_keys) < 10:  # Show if not too many
                    print(f"Missing: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in {model_name}: {len(unexpected_keys)} keys")
                if len(unexpected_keys) < 10:  # Show if not too many
                    print(f"Unexpected: {unexpected_keys}")
            
            model.to(self.device)
            model.eval()
            
            print(f"{model_name} loaded successfully")
            return model
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_objects(self, image_path, model, model_name):
        """Run object detection on image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=self.confidence_threshold
            )[0]
            
            # Extract detections
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections.append({
                    "confidence": score.item(),
                    "label": label.item(),
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                })
            
            return detections, len(detections)
            
        except Exception as e:
            print(f"Detection failed for {image_path} with {model_name}: {e}")
            return [], 0
    
    def compare_on_dataset(self, test_images_dir, results_file="comparison_results.json"):
        """Compare both models on test dataset"""
        print(f"\nCOMPARING MODELS ON TOOLTIP DETECTION")
        print("="*60)
        
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            print(f"Test directory not found: {test_dir}")
            return None
        
        # Get test images
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if not image_files:
            print(f"No images found in {test_dir}")
            return None
        
        print(f"Testing on {len(image_files)} images...")
        
        # Results storage
        results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "test_images_count": len(image_files),
                "confidence_threshold": self.confidence_threshold,
                "device": str(self.device)
            },
            "original_model": {
                "total_detections": 0,
                "images_with_detections": 0,
                "avg_confidence": 0,
                "detections_per_image": []
            },
            "new_model": {
                "total_detections": 0,
                "images_with_detections": 0,
                "avg_confidence": 0,
                "detections_per_image": []
            },
            "image_results": []
        }
        
        all_original_confidences = []
        all_new_confidences = []
        
        for i, image_path in enumerate(image_files):
            print(f"[{i+1}/{len(image_files)}] Processing {image_path.name}...")
            
            # Test original model
            orig_detections, orig_count = self.detect_objects(image_path, self.original_model, "Original")
            
            # Test new model  
            new_detections, new_count = self.detect_objects(image_path, self.new_model, "New")
            
            # Collect statistics
            orig_confidences = [d["confidence"] for d in orig_detections]
            new_confidences = [d["confidence"] for d in new_detections]
            
            all_original_confidences.extend(orig_confidences)
            all_new_confidences.extend(new_confidences)
            
            # Update results
            results["original_model"]["total_detections"] += orig_count
            results["new_model"]["total_detections"] += new_count
            
            if orig_count > 0:
                results["original_model"]["images_with_detections"] += 1
            if new_count > 0:
                results["new_model"]["images_with_detections"] += 1
                
            results["original_model"]["detections_per_image"].append(orig_count)
            results["new_model"]["detections_per_image"].append(new_count)
            
            # Store per-image results
            results["image_results"].append({
                "image": image_path.name,
                "original_detections": orig_count,
                "new_detections": new_count,
                "original_avg_confidence": np.mean(orig_confidences) if orig_confidences else 0,
                "new_avg_confidence": np.mean(new_confidences) if new_confidences else 0,
                "original_details": orig_detections,
                "new_details": new_detections
            })
        
        # Calculate final statistics
        if all_original_confidences:
            results["original_model"]["avg_confidence"] = np.mean(all_original_confidences)
        if all_new_confidences:
            results["new_model"]["avg_confidence"] = np.mean(all_new_confidences)
            
        # Save results
        results_path = Path(results_file)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_comparison_summary(results)
        
        print(f"\nDetailed results saved to: {results_path}")
        return results
    
    def _print_comparison_summary(self, results):
        """Print comparison summary"""
        print(f"\nCOMPARISON SUMMARY")
        print("="*50)
        
        orig = results["original_model"]
        new = results["new_model"] 
        test_info = results["test_info"]
        
        print(f"Test images: {test_info['test_images_count']}")
        print(f"Confidence threshold: {test_info['confidence_threshold']}")
        print()
        
        print("ORIGINAL MODEL (tooltip-only trained):")
        print(f"   Total detections: {orig['total_detections']}")
        print(f"   Images with detections: {orig['images_with_detections']}")
        print(f"   Avg confidence: {orig['avg_confidence']:.3f}")
        print(f"   Avg detections per image: {np.mean(orig['detections_per_image']):.2f}")
        
        print()
        print("NEW MODEL (mixed tooltip+background trained):")
        print(f"   Total detections: {new['total_detections']}")
        print(f"   Images with detections: {new['images_with_detections']}")
        print(f"   Avg confidence: {new['avg_confidence']:.3f}")
        print(f"   Avg detections per image: {np.mean(new['detections_per_image']):.2f}")
        
        print()
        print("PERFORMANCE COMPARISON:")
        
        # Detection rate comparison
        orig_detection_rate = orig['images_with_detections'] / test_info['test_images_count'] * 100
        new_detection_rate = new['images_with_detections'] / test_info['test_images_count'] * 100
        
        print(f"   Detection rate: Original {orig_detection_rate:.1f}% vs New {new_detection_rate:.1f}%")
        
        # Confidence comparison
        if orig['avg_confidence'] > 0 and new['avg_confidence'] > 0:
            conf_improvement = ((new['avg_confidence'] - orig['avg_confidence']) / orig['avg_confidence']) * 100
            print(f"   Confidence change: {conf_improvement:+.1f}%")
        
        # Total detections comparison  
        det_change = new['total_detections'] - orig['total_detections']
        print(f"   Detection count change: {det_change:+d}")
        
        print()
        if new['avg_confidence'] > orig['avg_confidence']:
            print("NEW MODEL shows HIGHER confidence!")
        elif abs(new['total_detections'] - orig['total_detections']) < 0.1 * orig['total_detections']:
            print("Models show SIMILAR performance")
        else:
            print("Results need analysis")
    
    def visualize_detections(self, image_path, output_dir="comparison_visualizations"):
        """Create side-by-side visualization of both models' detections"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Cannot load image: {image_path}")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get detections from both models
        orig_detections, _ = self.detect_objects(image_path, self.original_model, "Original")
        new_detections, _ = self.detect_objects(image_path, self.new_model, "New")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original model
        ax1.imshow(image_rgb)
        ax1.set_title(f"Original Model\n{len(orig_detections)} detections")
        self._draw_detections(ax1, orig_detections, color='red')
        ax1.axis('off')
        
        # New model
        ax2.imshow(image_rgb)
        ax2.set_title(f"Mixed-trained Model\n{len(new_detections)} detections") 
        self._draw_detections(ax2, new_detections, color='blue')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_file = output_dir / f"{Path(image_path).stem}_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {output_file}")
    
    def _draw_detections(self, ax, detections, color='red'):
        """Draw detection boxes on matplotlib axis"""
        from matplotlib.patches import Rectangle
        
        for detection in detections:
            bbox = detection["bbox"]  # [x1, y1, x2, y2]
            confidence = detection["confidence"]
            
            # Draw rectangle
            rect = Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                linewidth=2, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add confidence text
            ax.text(
                bbox[0], bbox[1] - 5,
                f"{confidence:.3f}",
                color=color,
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7)
            )

def main():
    parser = argparse.ArgumentParser(description="Compare DETR model performance")
    parser.add_argument("--original_model", 
                       default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/detr_inference_model.pth",
                       help="Path to original model")
    parser.add_argument("--new_model",
                       default="pipeline_output_videos_part1_simple/training_output/mixed_gentle_model.pth", 
                       help="Path to new mixed-trained model")
    parser.add_argument("--test_images",
                       default="pipeline_output_videos_part1_simple/interesting_frames/val",
                       help="Directory with test images")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization comparisons")
    
    args = parser.parse_args()
    
    # Check if models exist
    if not Path(args.original_model).exists():
        print(f"Original model not found: {args.original_model}")
        return
        
    if not Path(args.new_model).exists():
        print(f"New model not found: {args.new_model}")
        return
    
    print("DETR MODEL COMPARISON STARTING")
    print("="*50)
    print(f"Original model: {args.original_model}")
    print(f"New model: {args.new_model}")
    print(f"Test images: {args.test_images}")
    
    # Initialize comparator
    comparator = DETRModelComparator(args.original_model, args.new_model)
    
    if comparator.original_model is None or comparator.new_model is None:
        print("Failed to load models")
        return
    
    # Run comparison
    results = comparator.compare_on_dataset(args.test_images)
    
    if results is None:
        print("Comparison failed")
        return
    
    # Create visualizations if requested
    if args.visualize:
        test_dir = Path(args.test_images)
        sample_images = list(test_dir.glob("*.jpg"))[:5]  # First 5 images
        
        print(f"\nCreating visualizations for {len(sample_images)} sample images...")
        for img_path in sample_images:
            comparator.visualize_detections(img_path)
    
    print("\nModel comparison completed!")

if __name__ == "__main__":
    main()