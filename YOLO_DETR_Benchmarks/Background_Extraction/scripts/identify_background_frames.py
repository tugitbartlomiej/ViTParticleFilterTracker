import cv2
import os
import argparse
import numpy as np
from pathlib import Path
import shutil
from ultralytics import YOLO
import json

def load_yolo_model(model_path="yolov8n.pt"):
    """Load YOLO model for detection"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Trying to download default YOLOv8 model...")
        model = YOLO("yolov8n.pt")
        return model

def detect_objects_in_frame(model, frame_path, confidence_threshold=0.3):
    """
    Detect objects in a frame using YOLO
    
    Args:
        model: YOLO model
        frame_path: Path to frame image
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        List of detections with confidence scores
    """
    try:
        results = model(frame_path, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence >= confidence_threshold:
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        })
        
        return detections
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        return []

def is_background_frame(detections, target_classes=None):
    """
    Determine if frame is background based on detections
    
    Args:
        detections: List of detections from YOLO
        target_classes: List of classes to consider as non-background (e.g., ['person', 'knife', 'scissors'])
    
    Returns:
        Boolean indicating if frame is background
    """
    if target_classes is None:
        # Default classes that might appear in surgical videos
        target_classes = ['person', 'knife', 'scissors', 'spoon', 'fork', 'bottle', 'cup']
    
    # Check if any target objects are detected
    for detection in detections:
        if detection['class'] in target_classes:
            return False
    
    return True

def analyze_frame_quality(frame_path):
    """
    Analyze frame quality to filter out blurry or poor quality frames
    
    Args:
        frame_path: Path to frame image
    
    Returns:
        Dictionary with quality metrics
    """
    try:
        image = cv2.imread(frame_path)
        if image is None:
            return {'blur_score': 0, 'brightness': 0, 'contrast': 0, 'quality_good': False}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur score using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Quality thresholds
        quality_good = (blur_score > 100 and  # Not too blurry
                       brightness > 30 and brightness < 225 and  # Not too dark or bright
                       contrast > 20)  # Sufficient contrast
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'quality_good': quality_good
        }
    except Exception as e:
        print(f"Error analyzing frame quality for {frame_path}: {e}")
        return {'blur_score': 0, 'brightness': 0, 'contrast': 0, 'quality_good': False}

def process_frames_for_background(input_dir, output_dir, model_path="yolov8n.pt", 
                                confidence_threshold=0.3, max_background_frames=1000):
    """
    Process frames to identify and copy background frames
    
    Args:
        input_dir: Directory containing extracted frames
        output_dir: Directory to save background frames
        model_path: Path to YOLO model
        confidence_threshold: Minimum confidence for object detection
        max_background_frames: Maximum number of background frames to keep
    """
    # Load YOLO model
    print("Loading YOLO model...")
    model = load_yolo_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame files
    frame_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    frame_files = []
    for ext in frame_extensions:
        frame_files.extend(Path(input_dir).glob(f'*{ext}'))
        frame_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not frame_files:
        print(f"No frame files found in {input_dir}")
        return
    
    print(f"Processing {len(frame_files)} frames...")
    
    background_frames = []
    analysis_results = []
    
    for i, frame_path in enumerate(frame_files):
        if i % 50 == 0:
            print(f"Processed {i}/{len(frame_files)} frames...")
        
        # Detect objects in frame
        detections = detect_objects_in_frame(model, str(frame_path), confidence_threshold)
        
        # Check if frame is background
        is_background = is_background_frame(detections)
        
        # Analyze frame quality
        quality_metrics = analyze_frame_quality(str(frame_path))
        
        frame_info = {
            'frame_path': str(frame_path),
            'frame_name': frame_path.name,
            'is_background': is_background,
            'num_detections': len(detections),
            'detections': detections,
            'quality_metrics': quality_metrics
        }
        
        analysis_results.append(frame_info)
        
        # If it's a background frame with good quality, add to candidates
        if is_background and quality_metrics['quality_good']:
            background_frames.append(frame_info)
    
    # Sort background frames by quality (blur score descending)
    background_frames.sort(key=lambda x: x['quality_metrics']['blur_score'], reverse=True)
    
    # Limit number of background frames
    if len(background_frames) > max_background_frames:
        background_frames = background_frames[:max_background_frames]
    
    print(f"\nFound {len(background_frames)} high-quality background frames")
    
    # Copy background frames to output directory
    for frame_info in background_frames:
        src_path = frame_info['frame_path']
        dst_path = os.path.join(output_dir, frame_info['frame_name'])
        shutil.copy2(src_path, dst_path)
    
    # Save analysis results
    analysis_file = os.path.join(output_dir, 'analysis_results.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save background frames list
    background_file = os.path.join(output_dir, 'background_frames.json')
    with open(background_file, 'w') as f:
        json.dump(background_frames, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total frames processed: {len(frame_files)}")
    print(f"Background frames found: {len(background_frames)}")
    print(f"Background frames saved to: {output_dir}")
    print(f"Analysis results saved to: {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description='Identify background frames from extracted video frames')
    parser.add_argument('--input_dir', type=str, default='../extracted_frames',
                        help='Directory containing extracted frames')
    parser.add_argument('--output_dir', type=str, default='../background_frames',
                        help='Directory to save background frames')
    parser.add_argument('--model_path', type=str, default='yolov8n.pt',
                        help='Path to YOLO model')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold for object detection')
    parser.add_argument('--max_frames', type=int, default=1000,
                        help='Maximum number of background frames to keep')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    if not os.path.isabs(args.input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.input_dir = os.path.join(script_dir, args.input_dir)
    
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    process_frames_for_background(args.input_dir, args.output_dir, args.model_path, 
                                args.confidence, args.max_frames)

if __name__ == "__main__":
    main()