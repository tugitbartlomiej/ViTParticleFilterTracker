"""
Pipeline Monitor - Progress Tracking Framework
==============================================

Monitors long-running pipeline tasks through log file analysis, 
process monitoring, and status file updates for Claude Code agents.

Features:
- Real-time log file monitoring
- Progress extraction from structured logs
- Process health monitoring  
- Automatic status updates
- Error detection and notification
- Resource usage monitoring
"""

import asyncio
import json
import logging
import os
import psutil
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Callable, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

from status_manager import StatusManager, TaskStatus, TaskProgress, TaskError, ProgressType, get_status_manager

logger = logging.getLogger(__name__)

class LogProgressPattern:
    """Pattern for extracting progress from log lines"""
    
    def __init__(self, pattern: str, progress_type: ProgressType, 
                 current_group: int = 1, total_group: Optional[int] = None,
                 message_group: Optional[int] = None):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.progress_type = progress_type
        self.current_group = current_group
        self.total_group = total_group
        self.message_group = message_group
    
    def extract_progress(self, line: str) -> Optional[TaskProgress]:
        """Extract progress from a log line"""
        match = self.pattern.search(line)
        if not match:
            return None
        
        try:
            current = match.group(self.current_group)
            total = match.group(self.total_group) if self.total_group else None
            message = match.group(self.message_group) if self.message_group else ""
            
            # Convert to appropriate types
            if self.progress_type == ProgressType.PERCENTAGE:
                current = float(current)
                total = 100.0
            elif self.progress_type == ProgressType.COUNT:
                current = int(current)
                total = int(total) if total else None
            elif self.progress_type == ProgressType.STAGE:
                # Keep as string
                pass
            elif self.progress_type == ProgressType.TIME_REMAINING:
                current = int(current)  # seconds
                
            return TaskProgress(
                progress_type=self.progress_type,
                current=current,
                total=total,
                message=message.strip()
            )
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse progress from line '{line}': {e}")
            return None

class LogFileMonitor(FileSystemEventHandler):
    """Monitors log files for progress updates"""
    
    def __init__(self, task_id: str, log_file: Path, 
                 progress_patterns: List[LogProgressPattern],
                 error_patterns: Optional[List[Pattern]] = None):
        self.task_id = task_id
        self.log_file = log_file
        self.progress_patterns = progress_patterns
        self.error_patterns = error_patterns or []
        self.status_manager = get_status_manager()
        self.last_position = 0
        self._lock = threading.Lock()
        
        # Track recent progress to avoid duplicate updates
        self.last_progress = None
        self.last_progress_time = None
        
        # Initial file position
        if self.log_file.exists():
            self.last_position = self.log_file.stat().st_size
    
    def on_modified(self, event):
        """Handle log file modification"""
        if event.src_path == str(self.log_file):
            self.process_new_content()
    
    def process_new_content(self):
        """Process new content in log file"""
        with self._lock:
            if not self.log_file.exists():
                return
                
            try:
                current_size = self.log_file.stat().st_size
                if current_size <= self.last_position:
                    return  # No new content
                
                # Read new content
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self.last_position)
                    new_lines = f.readlines()
                    self.last_position = f.tell()
                
                # Process each line
                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for progress patterns
                    progress = self.extract_progress_from_line(line)
                    if progress:
                        # Avoid duplicate progress updates
                        if not self.is_duplicate_progress(progress):
                            self.status_manager.update_progress(self.task_id, progress)
                            self.last_progress = progress
                            self.last_progress_time = datetime.now()
                    
                    # Check for error patterns
                    self.check_for_errors(line)
                        
            except Exception as e:
                logger.error(f"Error processing log file {self.log_file}: {e}")
    
    def extract_progress_from_line(self, line: str) -> Optional[TaskProgress]:
        """Extract progress information from a log line"""
        for pattern in self.progress_patterns:
            progress = pattern.extract_progress(line)
            if progress:
                return progress
        return None
    
    def is_duplicate_progress(self, progress: TaskProgress) -> bool:
        """Check if this progress update is a duplicate"""
        if not self.last_progress or not self.last_progress_time:
            return False
        
        # Don't update more than once per second for same progress
        time_diff = datetime.now() - self.last_progress_time
        if time_diff.total_seconds() < 1:
            return True
        
        # Check if progress values are the same
        if (self.last_progress.progress_type == progress.progress_type and
            self.last_progress.current == progress.current):
            return True
            
        return False
    
    def check_for_errors(self, line: str):
        """Check line for error patterns"""
        for error_pattern in self.error_patterns:
            if error_pattern.search(line):
                error = TaskError(
                    error_type="runtime_error",
                    message=f"Error detected in log: {line}",
                    recoverable=True
                )
                self.status_manager.fail_task(self.task_id, error)
                logger.error(f"Error detected in task {self.task_id}: {line}")
                break

class ProcessMonitor:
    """Monitors system processes for task health"""
    
    def __init__(self, task_id: str, process_id: int):
        self.task_id = task_id
        self.process_id = process_id
        self.status_manager = get_status_manager()
        self.process = None
        
        try:
            self.process = psutil.Process(process_id)
        except psutil.NoSuchProcess:
            logger.warning(f"Process {process_id} not found for task {task_id}")
    
    def is_running(self) -> bool:
        """Check if process is still running"""
        if not self.process:
            return False
            
        try:
            return self.process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        if not self.process or not self.is_running():
            return {}
        
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "num_threads": self.process.num_threads(),
                "status": self.process.status(),
                "create_time": self.process.create_time()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Could not get resource usage for process {self.process_id}: {e}")
            return {}
    
    def terminate_gracefully(self, timeout: int = 30) -> bool:
        """Terminate process gracefully"""
        if not self.process or not self.is_running():
            return True
        
        try:
            # Send SIGTERM
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=timeout)
                return True
            except psutil.TimeoutExpired:
                # Force kill if necessary
                logger.warning(f"Process {self.process_id} did not terminate gracefully, forcing kill")
                self.process.kill()
                self.process.wait(timeout=5)
                return True
                
        except psutil.NoSuchProcess:
            return True
        except Exception as e:
            logger.error(f"Error terminating process {self.process_id}: {e}")
            return False

class TaskMonitor:
    """Complete monitoring for a single task"""
    
    def __init__(self, task_id: str, log_file: Optional[Path] = None, 
                 process_id: Optional[int] = None):
        self.task_id = task_id
        self.log_file = log_file
        self.process_id = process_id
        self.status_manager = get_status_manager()
        
        # Monitoring components
        self.log_monitor = None
        self.process_monitor = None
        self.file_observer = None
        
        # Configuration
        self.progress_patterns = self.get_default_progress_patterns()
        self.error_patterns = self.get_default_error_patterns()
        
        self.setup_monitoring()
    
    def get_default_progress_patterns(self) -> List[LogProgressPattern]:
        """Get default progress extraction patterns"""
        return [
            # Percentage patterns
            LogProgressPattern(
                r"Progress:\s*(\d+(?:\.\d+)?)%",
                ProgressType.PERCENTAGE
            ),
            LogProgressPattern(
                r"(\d+(?:\.\d+)?)%\s*complete",
                ProgressType.PERCENTAGE  
            ),
            LogProgressPattern(
                r"Epoch\s+(\d+)/(\d+)",
                ProgressType.COUNT,
                current_group=1,
                total_group=2
            ),
            LogProgressPattern(
                r"(\d+)/(\d+)\s+(?:frames|images|samples)",
                ProgressType.COUNT,
                current_group=1,
                total_group=2
            ),
            LogProgressPattern(
                r"Stage:\s*(.+)",
                ProgressType.STAGE,
                current_group=1
            ),
            LogProgressPattern(
                r"Processing\s+(.+?)\.{3}",
                ProgressType.STAGE,
                current_group=1
            )
        ]
    
    def get_default_error_patterns(self) -> List[Pattern]:
        """Get default error detection patterns"""
        return [
            re.compile(r"ERROR", re.IGNORECASE),
            re.compile(r"FAILED", re.IGNORECASE),
            re.compile(r"Exception:", re.IGNORECASE),
            re.compile(r"Traceback", re.IGNORECASE),
            re.compile(r"CUDA.*out of memory", re.IGNORECASE),
            re.compile(r"Permission denied", re.IGNORECASE),
            re.compile(r"No such file or directory", re.IGNORECASE),
        ]
    
    def setup_monitoring(self):
        """Setup all monitoring components"""
        # Setup log file monitoring
        if self.log_file and self.log_file.exists():
            self.log_monitor = LogFileMonitor(
                self.task_id, 
                self.log_file,
                self.progress_patterns,
                self.error_patterns
            )
            
            # Setup file system observer
            self.file_observer = Observer()
            self.file_observer.schedule(
                self.log_monitor,
                str(self.log_file.parent),
                recursive=False
            )
            self.file_observer.start()
            
            # Process existing content
            self.log_monitor.process_new_content()
        
        # Setup process monitoring
        if self.process_id:
            self.process_monitor = ProcessMonitor(self.task_id, self.process_id)
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
        
        logger.info(f"Stopped monitoring for task {self.task_id}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the task"""
        status = {"task_id": self.task_id}
        
        # Process health
        if self.process_monitor:
            status["process_running"] = self.process_monitor.is_running()
            status["resource_usage"] = self.process_monitor.get_resource_usage()
        
        # Log file status
        if self.log_file:
            status["log_file_exists"] = self.log_file.exists()
            if self.log_file.exists():
                stat = self.log_file.stat()
                status["log_file_size"] = stat.st_size
                status["log_file_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        return status

class PipelineMonitor:
    """Main monitor for the entire pipeline"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.status_manager = get_status_manager()
        self.task_monitors: Dict[str, TaskMonitor] = {}
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start the main monitoring loop"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        
        # Stop all task monitors
        for task_monitor in self.task_monitors.values():
            task_monitor.stop_monitoring()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Pipeline monitoring stopped")
    
    def add_task_monitor(self, task_id: str, log_file: Optional[Path] = None,
                        process_id: Optional[int] = None):
        """Add monitoring for a specific task"""
        if task_id in self.task_monitors:
            logger.warning(f"Task monitor for {task_id} already exists")
            return
        
        task_monitor = TaskMonitor(task_id, log_file, process_id)
        self.task_monitors[task_id] = task_monitor
        logger.info(f"Added monitoring for task {task_id}")
    
    def remove_task_monitor(self, task_id: str):
        """Remove monitoring for a specific task"""
        if task_id in self.task_monitors:
            self.task_monitors[task_id].stop_monitoring()
            del self.task_monitors[task_id]
            logger.info(f"Removed monitoring for task {task_id}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_all_tasks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_all_tasks(self):
        """Check status of all monitored tasks"""
        all_tasks = self.status_manager.get_all_tasks()
        
        for task_id, task_info in all_tasks.items():
            try:
                self._check_single_task(task_id, task_info)
            except Exception as e:
                logger.error(f"Error checking task {task_id}: {e}")
    
    def _check_single_task(self, task_id: str, task_info):
        """Check status of a single task"""
        # Skip completed or failed tasks
        if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return
        
        # Check if process is still running
        if task_info.process_id:
            task_monitor = self.task_monitors.get(task_id)
            if task_monitor and task_monitor.process_monitor:
                if not task_monitor.process_monitor.is_running():
                    # Process stopped - check if it completed successfully
                    self._handle_process_stopped(task_id, task_info)
        
        # Check for stalled tasks (no progress updates for too long)
        if task_info.progress and task_info.progress.last_update:
            last_update = datetime.fromisoformat(task_info.progress.last_update)
            time_since_update = datetime.now() - last_update
            
            # If no update for over 30 minutes, consider it stalled
            if time_since_update > timedelta(minutes=30):
                self._handle_stalled_task(task_id, task_info)
    
    def _handle_process_stopped(self, task_id: str, task_info):
        """Handle when a process has stopped"""
        # Check log file for completion markers
        if task_info.log_file and Path(task_info.log_file).exists():
            with open(task_info.log_file, 'r') as f:
                content = f.read()
                
            # Look for completion markers
            if any(marker in content.lower() for marker in 
                   ['completed successfully', 'training finished', 'done', 'finished']):
                self.status_manager.complete_task(task_id)
                logger.info(f"Task {task_id} completed successfully")
            else:
                # Look for error markers
                error = TaskError(
                    error_type="process_terminated",
                    message="Process terminated unexpectedly",
                    recoverable=True
                )
                self.status_manager.fail_task(task_id, error)
                logger.error(f"Task {task_id} process terminated unexpectedly")
    
    def _handle_stalled_task(self, task_id: str, task_info):
        """Handle when a task appears to be stalled"""
        logger.warning(f"Task {task_id} appears to be stalled - no progress update for >30 minutes")
        
        # Could implement recovery logic here
        # For now, just log the warning
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health status"""
        health_status = {
            "monitor_running": self.running,
            "monitored_tasks": len(self.task_monitors),
            "last_check": datetime.now().isoformat(),
            "task_health": {}
        }
        
        for task_id, task_monitor in self.task_monitors.items():
            health_status["task_health"][task_id] = task_monitor.get_health_status()
        
        return health_status
    
    def generate_progress_report(self) -> str:
        """Generate a detailed progress report"""
        pipeline_status = self.status_manager.get_pipeline_status()
        health_status = self.get_pipeline_health()
        
        report = f"""
DETR Pipeline Progress Report
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Status:
- Progress: {pipeline_status['overall_progress']:.1f}%
- Running Tasks: {pipeline_status['running_tasks']}
- Completed: {pipeline_status['completed_tasks']}/{pipeline_status['total_tasks']}
- Failed: {pipeline_status['failed_tasks']}

"""
        
        if pipeline_status.get('estimated_completion'):
            est_time = datetime.fromisoformat(pipeline_status['estimated_completion'])
            report += f"Estimated Completion: {est_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if pipeline_status.get('current_stage_info'):
            stage_info = pipeline_status['current_stage_info']
            report += f"Current Stage: {stage_info.get('stage_name')}\n"
            if stage_info.get('elapsed_time'):
                report += f"Elapsed: {stage_info['elapsed_time']}\n"
        
        # Add resource usage for running tasks
        report += "\nResource Usage:\n"
        for task_id, health in health_status["task_health"].items():
            if health.get("process_running"):
                usage = health.get("resource_usage", {})
                if usage:
                    report += f"- {task_id}: CPU {usage.get('cpu_percent', 0):.1f}%, "
                    report += f"Memory {usage.get('memory_rss_mb', 0):.0f}MB\n"
        
        return report

# Global pipeline monitor instance
_global_pipeline_monitor: Optional[PipelineMonitor] = None

def get_pipeline_monitor() -> PipelineMonitor:
    """Get global pipeline monitor instance"""
    global _global_pipeline_monitor
    if _global_pipeline_monitor is None:
        _global_pipeline_monitor = PipelineMonitor()
    return _global_pipeline_monitor