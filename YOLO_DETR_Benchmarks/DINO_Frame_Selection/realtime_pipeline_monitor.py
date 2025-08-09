#!/usr/bin/env python3
"""
Real-time Pipeline Monitor - Co 10 sekund status update
======================================================

Monitor działający w czasie rzeczywistym, który co 10 sekund 
sprawdza status pipelineu i wyświetla aktualne informacje.
"""

import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import threading

class RealTimePipelineMonitor:
    """
    Monitor pipelineu w czasie rzeczywistym z powiadomieniami co 10 sekund
    """
    
    def __init__(self, 
                 status_file="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output_single_test/pipeline_status.json",
                 update_interval=10):
        """
        Inicjalizacja monitora
        
        Args:
            status_file: Ścieżka do pliku statusu pipelineu
            update_interval: Interwał aktualizacji w sekundach (default: 10)
        """
        self.status_file = Path(status_file)
        self.update_interval = update_interval
        self.running = False
        self.last_status = {}
        self.start_time = None
        
        print(f"=== REAL-TIME PIPELINE MONITOR ===")
        print(f"Status file: {self.status_file}")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"Starting monitoring...")
        print("=" * 50)
    
    def read_pipeline_status(self):
        """Odczytaj aktualny status pipelineu"""
        try:
            if not self.status_file.exists():
                return {"error": f"Status file not found: {self.status_file}"}
            
            with open(self.status_file, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
            
            return status_data
        except Exception as e:
            return {"error": f"Failed to read status: {e}"}
    
    def format_time_duration(self, start_time_str):
        """Formatuj czas trwania"""
        try:
            start_time = datetime.fromisoformat(start_time_str)
            duration = datetime.now() - start_time
            
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            seconds = int(duration.total_seconds() % 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except:
            return "unknown"
    
    def get_active_processes(self):
        """Sprawdź aktywne procesy pipelineu"""
        try:
            # Sprawdź procesy Python związane z pipelinenem
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Więcej niż header
                    return len(lines) - 1  # Liczba procesów minus header
            return 0
        except:
            return "unknown"
    
    def display_status_update(self, status_data):
        """Wyświetl aktualny status"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n[{current_time}] === PIPELINE STATUS UPDATE ===")
        
        if "error" in status_data:
            print(f"ERROR: {status_data['error']}")
            return
        
        # Podstawowe informacje
        metadata = status_data.get("pipeline_metadata", {})
        pipeline_id = metadata.get("pipeline_id", "unknown")
        current_stage = metadata.get("current_stage", "unknown")
        started_at = metadata.get("started_at", "unknown")
        
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Current Stage: {current_stage.upper()}")
        print(f"Running Time: {self.format_time_duration(started_at)}")
        
        # Status tasków
        tasks = status_data.get("tasks", {})
        active_tasks = []
        completed_tasks = []
        failed_tasks = []
        
        for task_id, task_info in tasks.items():
            task_status = task_info.get("status", "unknown")
            task_name = task_info.get("name", task_id)
            
            if task_status == "running":
                active_tasks.append(f"{task_name} (PID: {task_info.get('process_id', 'N/A')})")
            elif task_status == "completed":
                completed_tasks.append(task_name)
            elif task_status == "failed":
                failed_tasks.append(task_name)
        
        # Wyświetl aktywne taski
        if active_tasks:
            print(f"ACTIVE TASKS ({len(active_tasks)}):")
            for task in active_tasks:
                print(f"  -> {task}")
        else:
            print("ACTIVE TASKS: None")
        
        # Wyświetl zakończone taski
        if completed_tasks:
            print(f"COMPLETED: {len(completed_tasks)} tasks")
        
        # Wyświetl nieudane taski
        if failed_tasks:
            print(f"FAILED: {len(failed_tasks)} tasks")
            for task in failed_tasks:
                print(f"  X {task}")
        
        # Sprawdź aktywne procesy systemowe
        active_processes = self.get_active_processes()
        print(f"System Python Processes: {active_processes}")
        
        # Ostatnia aktualizacja
        last_updated = status_data.get("last_updated", "unknown")
        if last_updated != "unknown":
            try:
                last_update_time = datetime.fromisoformat(last_updated)
                time_since_update = (datetime.now() - last_update_time).total_seconds()
                print(f"Last Update: {int(time_since_update)}s ago")
            except:
                print(f"Last Update: {last_updated}")
        
        print("-" * 50)
        
        # Sprawdź czy są zmiany od ostatniego razu
        if self.last_status != status_data:
            print("STATUS CHANGED - New activity detected!")
            self.last_status = status_data.copy()
        else:
            print("Status unchanged")
    
    def monitor_loop(self):
        """Główna pętla monitorowania"""
        self.start_time = datetime.now()
        
        while self.running:
            try:
                # Odczytaj aktualny status
                status_data = self.read_pipeline_status()
                
                # Wyświetl status
                self.display_status_update(status_data)
                
                # Sprawdź czy pipeline się zakończył
                if not status_data.get("error"):
                    tasks = status_data.get("tasks", {})
                    running_tasks = [t for t in tasks.values() if t.get("status") == "running"]
                    
                    if not running_tasks:
                        print("\nPIPELINE APPEARS TO BE FINISHED - No running tasks detected")
                        print("Monitor will continue checking for new activity...")
                
                # Czekaj określony czas
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                print(f"\nMonitoring interrupted by user")
                break
            except Exception as e:
                print(f"\nMonitoring error: {e}")
                time.sleep(self.update_interval)
    
    def start_monitoring(self):
        """Rozpocznij monitorowanie"""
        if self.running:
            print("Monitor is already running!")
            return
        
        self.running = True
        
        print(f"Starting real-time monitoring (every {self.update_interval}s)")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 50)
        
        try:
            self.monitor_loop()
        finally:
            self.running = False
            total_time = datetime.now() - self.start_time if self.start_time else None
            print(f"\nMonitoring stopped")
            if total_time:
                print(f"Total monitoring time: {self.format_time_duration(self.start_time.isoformat())}")
    
    def stop_monitoring(self):
        """Zatrzymaj monitorowanie"""
        self.running = False

def main():
    """Główna funkcja"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Pipeline Monitor')
    parser.add_argument('--status_file', type=str,
                        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_output_single_test/pipeline_status.json",
                        help='Path to pipeline status JSON file')
    parser.add_argument('--interval', type=int, default=10,
                        help='Update interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Stwórz i uruchom monitor
    monitor = RealTimePipelineMonitor(
        status_file=args.status_file,
        update_interval=args.interval
    )
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())