#!/usr/bin/env python3
"""
DETR Pipeline Real-time Watcher
===============================

Standalone script that monitors pipeline progress in real-time and displays
status updates in console. Run in separate terminal while pipeline is running.

Usage:
    python pipeline_watcher.py                    # Watch with 10s interval
    python pipeline_watcher.py --interval 5       # Watch with 5s interval
    python pipeline_watcher.py --output-dir path  # Watch specific pipeline output
    python pipeline_watcher.py --once             # Check status once and exit
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

class PipelineWatcher:
    """Real-time pipeline progress watcher"""
    
    def __init__(self, output_dir: str = "pipeline_output_single_test"):
        self.output_dir = Path(output_dir)
        self.status_file = self.output_dir / "pipeline_status.json"
        self.last_update = None
        self.last_status = None
        
        # Terminal colors for better visibility
        self.colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }
    
    def colorize(self, text: str, color: str) -> str:
        """Add color to text for terminal output"""
        return f"{self.colors.get(color, '')}{text}{self.colors['end']}"
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def read_pipeline_status(self) -> Optional[Dict]:
        """Read current pipeline status from JSON file"""
        try:
            if not self.status_file.exists():
                return None
            
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to read status: {e}"}
    
    def read_background_report(self) -> Optional[Dict]:
        """Read background frame extraction report"""
        try:
            report_file = self.output_dir / "background_frames" / "final_report.json"
            if not report_file.exists():
                return None
            
            with open(report_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def find_latest_task_for_stage(self, tasks: Dict, stage_id: str) -> Optional[Dict]:
        """Find the latest task attempt for a given stage"""
        matching_tasks = []
        
        for task_id, task_data in tasks.items():
            if task_id.startswith(stage_id + '_attempt_'):
                matching_tasks.append((task_id, task_data))
        
        if not matching_tasks:
            return None
        
        # Sort by task_id (attempt number) and return the latest
        matching_tasks.sort(key=lambda x: x[0])
        return matching_tasks[-1][1]
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def get_stage_emoji(self, stage_name: str, status: str) -> str:
        """Get emoji for pipeline stage"""
        stage_emojis = {
            'dino_extraction': '🔍',
            'dataset_mixing': '🔄', 
            'model_preparation': '🛠️',
            'gentle_training': '🎯',
            'validation': '✅'
        }
        
        if status == 'completed':
            return '✅'
        elif status == 'running':
            return '⚡'
        elif status == 'failed':
            return '❌'
        else:
            return stage_emojis.get(stage_name, '⏳')
    
    def display_pipeline_overview(self, status: Dict):
        """Display overall pipeline status"""
        print(self.colorize("═" * 80, 'cyan'))
        print(self.colorize("🚀 DETR BACKGROUND TRAINING PIPELINE MONITOR", 'bold'))
        print(self.colorize("═" * 80, 'cyan'))
        
        # Pipeline info from pipeline_metadata
        pipeline_metadata = status.get('pipeline_metadata', {})
        started_at = pipeline_metadata.get('started_at')
        if started_at:
            try:
                start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                elapsed = datetime.now() - start_time.replace(tzinfo=None)
                print(f"⏱️  Started: {start_time.strftime('%H:%M:%S')} | Elapsed: {self.format_duration(elapsed.total_seconds())}")
            except:
                print(f"⏱️  Started: {started_at}")
        
        # Overall status - determine from tasks
        tasks = status.get('tasks', {})
        if any(task.get('status') == 'running' for task in tasks.values()):
            pipeline_status = 'running'
        elif any(task.get('status') == 'failed' for task in tasks.values()):
            pipeline_status = 'failed'
        elif all(task.get('status') == 'completed' for task in tasks.values()) and tasks:
            pipeline_status = 'completed'
        else:
            pipeline_status = 'unknown'
            
        status_color = {
            'running': 'yellow',
            'completed': 'green', 
            'failed': 'red'
        }.get(pipeline_status, 'white')
        
        print(f"📊 Status: {self.colorize(pipeline_status.upper(), status_color)}")
        print()
    
    def display_stage_progress(self, status: Dict):
        """Display progress for each pipeline stage"""
        print(self.colorize("📋 PIPELINE STAGES", 'bold'))
        print("─" * 50)
        
        stages = [
            ('dino_extraction', 'DINO Frame Extraction'),
            ('dataset_mixing', 'Dataset Mixing'), 
            ('model_preparation', 'Model Preparation'),
            ('gentle_training', 'Mixed Gentle Training'),
            ('validation', 'Results Validation')
        ]
        
        tasks = status.get('tasks', {})
        if not tasks:
            print("No tasks found in status")
            return
        
        for stage_id, stage_name in stages:
            # Find the latest attempt for this stage
            task = self.find_latest_task_for_stage(tasks, stage_id)
            task_status = task.get('status', 'pending') if task else 'pending'
            
            emoji = self.get_stage_emoji(stage_id, task_status)
            
            # Format status with color
            status_text = task_status.upper()
            status_color = {
                'completed': 'green',
                'running': 'yellow',
                'failed': 'red',
                'pending': 'white'
            }.get(task_status, 'white')
            
            status_colored = self.colorize(status_text, status_color)
            
            # Progress info  
            progress_info = ""
            if task and 'progress' in task and task['progress']:
                progress = task['progress']
                current = progress.get('current', 0)
                total = progress.get('total', 100)
                if total and total > 0:
                    percentage = (current / total * 100) if isinstance(current, (int, float)) else 0
                    progress_info = f" ({percentage:.1f}%)"
                elif 'current' in progress:
                    progress_info = f" ({progress['current']})"
            
            # Duration info
            duration_info = ""
            if task and 'started_at' in task and task_status == 'running':
                try:
                    start_time = datetime.fromisoformat(task['started_at'].replace('Z', '+00:00'))
                    elapsed = datetime.now() - start_time.replace(tzinfo=None)
                    duration_info = f" - {self.format_duration(elapsed.total_seconds())}"
                except:
                    pass
            
            # PID info
            pid_info = ""
            if task and task_status == 'running' and 'process_id' in task:
                pid_info = f" (PID: {task['process_id']})"
            
            print(f"{emoji} {stage_name:<25} {status_colored}{progress_info}{duration_info}{pid_info}")
            
            # Show error if failed
            if task_status == 'failed' and 'error' in task:
                error_msg = task['error'].get('message', 'Unknown error')
                print(f"   {self.colorize('└─ Error:', 'red')} {error_msg[:60]}...")
        
        print()
    
    def display_background_extraction_details(self, background_report: Dict):
        """Display details from background frame extraction"""
        if not background_report:
            return
        
        print(self.colorize("🎬 BACKGROUND EXTRACTION DETAILS", 'bold'))
        print("─" * 50)
        
        stats = background_report.get('processing_stats', {})
        summary = background_report.get('summary', {})
        
        total_frames = stats.get('total_frames_extracted', 0)
        background_frames = stats.get('total_background_frames', 0)
        selected_frames = stats.get('total_selected_frames', 0)
        
        detection_rate = summary.get('background_detection_rate', 0) * 100
        selection_rate = summary.get('selection_rate', 0) * 100
        
        print(f"📹 Total frames extracted: {self.colorize(str(total_frames), 'cyan')}")
        print(f"🎯 Background frames found: {self.colorize(str(background_frames), 'green')} ({detection_rate:.1f}%)")
        print(f"✨ Final selected frames: {self.colorize(str(selected_frames), 'yellow')} ({selection_rate:.1f}%)")
        
        # Detection thresholds
        config = background_report.get('configuration', {})
        yolo_threshold = config.get('yolo_confidence_threshold', 0)
        detr_threshold = config.get('detr_confidence_threshold', 0)
        print(f"🔧 Thresholds: YOLO {yolo_threshold} | DETR {detr_threshold}")
        print()
    
    def display_training_progress(self, status: Dict):
        """Display training progress if available"""
        tasks = status.get('tasks', {})
        training_task = tasks.get('gentle_training', {})
        
        if training_task.get('status') == 'running':
            print(self.colorize("🎯 TRAINING PROGRESS", 'bold'))
            print("─" * 50)
            
            # Look for training logs or checkpoints
            training_dir = self.output_dir / "training_output"
            if training_dir.exists():
                checkpoints = list(training_dir.glob("checkpoint_*.pth"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    print(f"📁 Latest checkpoint: {self.colorize(latest_checkpoint.name, 'green')}")
                    
                    # File modification time
                    mod_time = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
                    time_ago = datetime.now() - mod_time
                    print(f"⏰ Updated: {self.format_duration(time_ago.total_seconds())} ago")
            print()
    
    def display_system_info(self):
        """Display system and file info"""
        print(self.colorize("💻 SYSTEM INFO", 'bold'))
        print("─" * 50)
        
        # Output directory status
        if self.output_dir.exists():
            dir_size = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())
            dir_size_mb = dir_size / (1024 * 1024)
            print(f"📂 Output dir: {self.colorize(str(self.output_dir), 'cyan')} ({dir_size_mb:.1f} MB)")
        else:
            print(f"📂 Output dir: {self.colorize('Not found', 'red')}")
        
        # Status file info
        if self.status_file.exists():
            mod_time = datetime.fromtimestamp(self.status_file.stat().st_mtime)
            time_ago = datetime.now() - mod_time
            print(f"📄 Status file: Updated {self.format_duration(time_ago.total_seconds())} ago")
        else:
            print(f"📄 Status file: {self.colorize('Not found', 'red')}")
        
        print(f"🕐 Current time: {datetime.now().strftime('%H:%M:%S')}")
        print()
    
    def watch_once(self) -> bool:
        """Check pipeline status once and display"""
        status = self.read_pipeline_status()
        background_report = self.read_background_report()
        
        if not status:
            print(self.colorize("❌ No pipeline status found", 'red'))
            print(f"Looking for: {self.status_file}")
            return False
        
        if 'error' in status:
            print(self.colorize(f"❌ Error reading status: {status['error']}", 'red'))
            return False
        
        # Display all sections
        try:
            self.display_pipeline_overview(status)
            self.display_stage_progress(status)
        except Exception as e:
            print(self.colorize(f"❌ Watcher error: {str(e)}", 'red'))
            return False
        
        if background_report:
            self.display_background_extraction_details(background_report)
        
        self.display_training_progress(status)
        self.display_system_info()
        
        return True
    
    def watch_continuous(self, interval: int = 10):
        """Watch pipeline continuously with specified interval"""
        print(self.colorize(f"🔄 Starting pipeline watcher (interval: {interval}s)", 'cyan'))
        print(self.colorize("Press Ctrl+C to stop", 'yellow'))
        print()
        
        try:
            while True:
                self.clear_screen()
                
                success = self.watch_once()
                
                if success:
                    status = self.read_pipeline_status()
                    pipeline_status = status.get('status', 'unknown')
                    
                    # Stop watching if pipeline completed or failed
                    if pipeline_status in ['completed', 'failed']:
                        print(self.colorize(f"Pipeline {pipeline_status}. Watcher stopping.", 'green' if pipeline_status == 'completed' else 'red'))
                        break
                
                # Wait for next check
                print(self.colorize(f"Next update in {interval}s... (Ctrl+C to stop)", 'cyan'))
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(self.colorize("\n🛑 Watcher stopped by user", 'yellow'))
        except Exception as e:
            print(self.colorize(f"\n❌ Watcher error: {e}", 'red'))

def main():
    parser = argparse.ArgumentParser(description="DETR Pipeline Real-time Watcher")
    parser.add_argument('--interval', '-i', type=int, default=10,
                       help='Update interval in seconds (default: 10)')
    parser.add_argument('--output-dir', '-o', default='pipeline_output_single_test',
                       help='Pipeline output directory to watch (default: pipeline_output_single_test)')
    parser.add_argument('--once', action='store_true',
                       help='Check status once and exit')
    
    args = parser.parse_args()
    
    # Validate output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        print("Make sure the pipeline is running or has been run in this directory")
        sys.exit(1)
    
    watcher = PipelineWatcher(args.output_dir)
    
    if args.once:
        success = watcher.watch_once()
        sys.exit(0 if success else 1)
    else:
        watcher.watch_continuous(args.interval)

if __name__ == "__main__":
    main()