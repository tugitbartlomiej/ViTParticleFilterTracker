#!/usr/bin/env python3
"""
Quick Status Check - Sprawdź obecny status pipelineu
===================================================
"""

import json
from pathlib import Path
from datetime import datetime

def check_pipeline_status():
    """Sprawdź aktualny status pipelineu"""
    
    status_file = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output_single_test/pipeline_status.json")
    
    print(f"=== PIPELINE STATUS CHECK ===")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Status file: {status_file}")
    
    if not status_file.exists():
        print("STATUS FILE NOT FOUND")
        return
    
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
        
        # Podstawowe info
        metadata = status_data.get("pipeline_metadata", {})
        current_stage = metadata.get("current_stage", "unknown")
        print(f"Current Stage: {current_stage.upper()}")
        
        # Aktywne taski
        tasks = status_data.get("tasks", {})
        running_tasks = []
        completed_tasks = []
        
        for task_id, task_info in tasks.items():
            status = task_info.get("status", "unknown")
            name = task_info.get("name", task_id)
            
            if status == "running":
                pid = task_info.get("process_id", "N/A")
                running_tasks.append(f"{name} (PID: {pid})")
            elif status == "completed":
                completed_tasks.append(name)
        
        if running_tasks:
            print("RUNNING TASKS:")
            for task in running_tasks:
                print(f"   -> {task}")
        else:
            print("NO RUNNING TASKS")
        
        if completed_tasks:
            print(f"COMPLETED: {len(completed_tasks)} tasks")
        
        # Ostatnia aktualizacja
        last_updated = status_data.get("last_updated", "")
        if last_updated:
            try:
                last_time = datetime.fromisoformat(last_updated)
                seconds_ago = int((datetime.now() - last_time).total_seconds())
                print(f"Last update: {seconds_ago}s ago")
            except:
                print(f"Last update: {last_updated}")
        
    except Exception as e:
        print(f"ERROR reading status: {e}")
    
    print("=" * 40)

if __name__ == "__main__":
    check_pipeline_status()