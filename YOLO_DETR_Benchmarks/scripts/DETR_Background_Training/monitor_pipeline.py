import json
import time
import os
import sys
from datetime import datetime

STATUS_FILE = r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\scripts\DETR_Background_Training\pipeline_output_single_test\pipeline_status.json'
LOG_DIR = r'pipeline_output_single_test\logs'

def load_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading status file: {e}")
        return None

def monitor_pipeline():
    while True:
        status = load_status()
        if not status:
            print("Could not read status file. Retrying...")
            time.sleep(10)
            continue
        
        pipeline_metadata = status.get('pipeline_metadata', {})
        tasks = status.get('tasks', {})
        
        current_stage = pipeline_metadata.get('current_stage', 'Unknown')
        started_at = pipeline_metadata.get('started_at')
        total_stages = pipeline_metadata.get('total_stages', 5)
        stage_names = pipeline_metadata.get('stage_names', [])
        
        runtime = (datetime.now() - datetime.fromisoformat(started_at)).total_seconds()
        
        print(f"\n--- DETR Background Training Pipeline Status ---")
        print(f"Pipeline ID: {pipeline_metadata.get('pipeline_id')}")
        print(f"Runtime: {runtime/60:.2f} minutes")
        print(f"Current Stage: {current_stage} ({stage_names.index(current_stage) + 1 if current_stage in stage_names else 'Pre-Stage'}/{total_stages})")
        
        # Detailed task status
        for task_id, task_info in tasks.items():
            print(f"\nTask: {task_info.get('name', task_id)}")
            print(f"Status: {task_info.get('status', 'Unknown')}")
            if task_info.get('error'):
                print(f"Error: {task_info['error'].get('message', 'Unknown error')}")
        
        time.sleep(10)

if __name__ == '__main__':
    monitor_pipeline()