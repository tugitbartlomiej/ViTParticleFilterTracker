import cv2
import os
import random
import numpy as np
from pathlib import Path
import json

# Configuration
VIDEO_DIR = r"E:\Cataract\videos\micro"
OUTPUT_DIR = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\simple_test\test_frames"
NUM_FRAMES = 5  # Number of random frames to extract from each video

def extract_random_frames():
    """Extract random frames from videos and save them for both YOLO and DETR testing"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in {VIDEO_DIR}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Dictionary to store frame information
    frame_info = {}
    
    # Process each video
    for video_file in video_files[:3]:  # Process first 3 videos
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = video_file.replace('.mp4', '')
        
        print(f"\nProcessing video: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            continue
        
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in {video_name}: {total_frames}")
        
        # Generate random frame indices
        frame_indices = random.sample(range(0, total_frames), min(NUM_FRAMES, total_frames))
        frame_indices.sort()  # Sort for easier processing
        
        video_frames = []
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                frame_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(OUTPUT_DIR, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                video_frames.append({
                    'frame_index': frame_idx,
                    'filename': frame_filename,
                    'path': frame_path
                })
                
                print(f"Extracted frame {frame_idx} -> {frame_filename}")
        
        cap.release()
        frame_info[video_name] = video_frames
    
    # Save frame information to JSON file
    info_path = os.path.join(OUTPUT_DIR, 'frame_info.json')
    with open(info_path, 'w') as f:
        json.dump(frame_info, f, indent=2)
    
    print(f"\nFrame information saved to: {info_path}")
    print(f"Total frames extracted: {sum(len(frames) for frames in frame_info.values())}")

if __name__ == "__main__":
    print("Extracting random frames from videos...")
    extract_random_frames()
    print("Done!")