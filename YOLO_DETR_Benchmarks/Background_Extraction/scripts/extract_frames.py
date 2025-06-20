import cv2
import os
import argparse
from pathlib import Path
import numpy as np

def extract_frames_from_video(video_path, output_dir, frame_interval=30):
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_interval: Extract every N frames (default: 30 for ~1 frame per second at 30fps)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {Path(video_path).name}")
    print(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    print(f"Extracting every {frame_interval} frames...")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Generate filename with video name and frame number
            video_name = Path(video_path).stem
            frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            
            if extracted_count % 10 == 0:
                print(f"Extracted {extracted_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"Completed: Extracted {extracted_count} frames from {Path(video_path).name}")
    return extracted_count

def process_video_directory(video_dir, output_dir, frame_interval=30):
    """
    Process all videos in a directory
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save all extracted frames
        frame_interval: Extract every N frames
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_dir = Path(video_dir)
    
    if not video_dir.exists():
        print(f"Error: Video directory {video_dir} does not exist")
        return
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    total_extracted = 0
    for video_file in video_files:
        print(f"\nProcessing: {video_file.name}")
        extracted = extract_frames_from_video(str(video_file), output_dir, frame_interval)
        total_extracted += extracted
    
    print(f"\n=== SUMMARY ===")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Total frames extracted: {total_extracted}")
    print(f"Frames saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for background detection')
    parser.add_argument('--video_dir', type=str, default='E:/Cataract/videos/micro',
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='../extracted_frames',
                        help='Directory to save extracted frames')
    parser.add_argument('--frame_interval', type=int, default=30,
                        help='Extract every N frames (default: 30)')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    process_video_directory(args.video_dir, args.output_dir, args.frame_interval)

if __name__ == "__main__":
    main()