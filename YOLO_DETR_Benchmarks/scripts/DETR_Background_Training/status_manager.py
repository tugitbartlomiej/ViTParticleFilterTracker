"""
Status Manager - JSON-based Communication System
===============================================

Handles communication between Claude Code agents and Python background processes
through structured JSON status files.

Features:
- Thread-safe status updates
- Progress tracking with timestamps
- Error logging and recovery
- Checkpoint management
- Multi-stage pipeline state
"""

import json
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status values for pipeline tasks"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"

class ProgressType(Enum):
    """Types of progress measurements"""
    PERCENTAGE = "percentage"      # 0-100
    COUNT = "count"               # current/total
    STAGE = "stage"              # current stage description
    TIME_REMAINING = "time_remaining"  # estimated seconds

@dataclass
class TaskProgress:
    """Progress information for a task"""
    progress_type: ProgressType
    current: Union[int, float, str]
    total: Optional[Union[int, float]] = None
    message: str = ""
    last_update: str = ""
    
    def __post_init__(self):
        if not self.last_update:
            self.last_update = datetime.now().isoformat()

@dataclass  
class TaskError:
    """Error information for failed tasks"""
    error_type: str
    message: str
    traceback: str = ""
    timestamp: str = ""
    recoverable: bool = True
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class TaskInfo:
    """Complete information about a pipeline task"""
    task_id: str
    name: str
    status: TaskStatus
    progress: Optional[TaskProgress] = None
    error: Optional[TaskError] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    process_id: Optional[int] = None
    output_files: List[str] = None
    log_file: Optional[str] = None
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []

class StatusManager:
    """Thread-safe manager for pipeline status and communication"""
    
    def __init__(self, status_file: str = "pipeline_status.json", auto_save_interval: int = 30):
        self.status_file = Path(status_file)
        self.auto_save_interval = auto_save_interval
        self._lock = threading.RLock()
        self._tasks: Dict[str, TaskInfo] = {}
        self._pipeline_metadata = {
            "pipeline_id": f"detr_pipeline_{int(time.time())}",
            "started_at": datetime.now().isoformat(),
            "current_stage": "initialization",
            "total_stages": 5,
            "stage_names": [
                "dino_extraction",
                "dataset_mixing", 
                "model_preparation",
                "gentle_training",
                "validation"
            ]
        }
        
        # Load existing status if available
        self._load_status()
        
        # Start auto-save thread
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()
    
    def _auto_save_loop(self):
        """Background thread for automatic status saving"""
        while True:
            time.sleep(self.auto_save_interval)
            try:
                self.save_status()
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
    
    def _load_status(self):
        """Load existing status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load pipeline metadata
                self._pipeline_metadata.update(data.get('pipeline_metadata', {}))
                
                # Load tasks
                tasks_data = data.get('tasks', {})
                for task_id, task_data in tasks_data.items():
                    
                    # Convert progress data
                    progress = None
                    if task_data.get('progress'):
                        progress_data = task_data['progress']
                        progress = TaskProgress(
                            progress_type=ProgressType(progress_data['progress_type']),
                            current=progress_data['current'],
                            total=progress_data.get('total'),
                            message=progress_data.get('message', ''),
                            last_update=progress_data.get('last_update', '')
                        )
                    
                    # Convert error data  
                    error = None
                    if task_data.get('error'):
                        error_data = task_data['error']
                        error = TaskError(
                            error_type=error_data['error_type'],
                            message=error_data['message'],
                            traceback=error_data.get('traceback', ''),
                            timestamp=error_data.get('timestamp', ''),
                            recoverable=error_data.get('recoverable', True),
                            retry_count=error_data.get('retry_count', 0)
                        )
                    
                    # Create TaskInfo
                    task_info = TaskInfo(
                        task_id=task_id,
                        name=task_data.get('name', ''),
                        status=TaskStatus(task_data.get('status', TaskStatus.PENDING.value)),
                        progress=progress,
                        error=error,
                        started_at=task_data.get('started_at'),
                        completed_at=task_data.get('completed_at'),
                        process_id=task_data.get('process_id'),
                        output_files=task_data.get('output_files', []),
                        log_file=task_data.get('log_file')
                    )
                    
                    self._tasks[task_id] = task_info
                    
                logger.info(f"Loaded status from {self.status_file}")
                
            except Exception as e:
                logger.error(f"Failed to load status: {e}")
    
    def save_status(self):
        """Save current status to file"""
        with self._lock:
            try:
                # Prepare data for JSON serialization
                tasks_data = {}
                for task_id, task_info in self._tasks.items():
                    task_dict = asdict(task_info)
                    
                    # Convert enums to strings
                    if task_dict.get('status'):
                        task_dict['status'] = task_dict['status'].value if hasattr(task_dict['status'], 'value') else task_dict['status']
                    
                    if task_dict.get('progress') and task_dict['progress'].get('progress_type'):
                        prog_type = task_dict['progress']['progress_type']
                        task_dict['progress']['progress_type'] = prog_type.value if hasattr(prog_type, 'value') else prog_type
                    
                    tasks_data[task_id] = task_dict
                
                status_data = {
                    "pipeline_metadata": self._pipeline_metadata,
                    "tasks": tasks_data,
                    "last_updated": datetime.now().isoformat()
                }
                
                # Atomic write
                temp_file = self.status_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(status_data, f, indent=2, ensure_ascii=False)
                
                temp_file.replace(self.status_file)
                logger.debug(f"Status saved to {self.status_file}")
                
            except Exception as e:
                logger.error(f"Failed to save status: {e}")
                raise
    
    def create_task(self, task_id: str, name: str, log_file: Optional[str] = None) -> TaskInfo:
        """Create a new task"""
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task {task_id} already exists")
            
            task_info = TaskInfo(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                log_file=log_file or f"{task_id}.log"
            )
            
            self._tasks[task_id] = task_info
            logger.info(f"Created task: {task_id} - {name}")
            return task_info
    
    def start_task(self, task_id: str, process_id: Optional[int] = None):
        """Mark task as started"""
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            task.process_id = process_id
            
            logger.info(f"Started task: {task_id} (PID: {process_id})")
    
    def complete_task(self, task_id: str, output_files: Optional[List[str]] = None):
        """Mark task as completed"""
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            
            if output_files:
                task.output_files.extend(output_files)
            
            logger.info(f"Completed task: {task_id}")
    
    def fail_task(self, task_id: str, error: TaskError):
        """Mark task as failed"""
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.now().isoformat()
            
            logger.error(f"Failed task: {task_id} - {error.message}")
    
    def update_progress(self, task_id: str, progress: TaskProgress):
        """Update task progress"""
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            progress.last_update = datetime.now().isoformat()
            self._tasks[task_id].progress = progress
            
            logger.debug(f"Updated progress for {task_id}: {progress.message}")
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """Get all tasks"""
        with self._lock:
            return self._tasks.copy()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        with self._lock:
            total_tasks = len(self._tasks)
            completed_tasks = sum(1 for task in self._tasks.values() if task.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for task in self._tasks.values() if task.status == TaskStatus.FAILED)
            running_tasks = sum(1 for task in self._tasks.values() if task.status == TaskStatus.RUNNING)
            
            # Calculate overall progress
            if total_tasks > 0:
                overall_progress = (completed_tasks / total_tasks) * 100
            else:
                overall_progress = 0
            
            return {
                "pipeline_metadata": self._pipeline_metadata.copy(),
                "overall_progress": overall_progress,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks, 
                "running_tasks": running_tasks,
                "pending_tasks": total_tasks - completed_tasks - failed_tasks - running_tasks,
                "estimated_completion": self._estimate_completion_time(),
                "current_stage_info": self._get_current_stage_info()
            }
    
    def _estimate_completion_time(self) -> Optional[str]:
        """Estimate pipeline completion time based on current progress"""
        completed_tasks = [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]
        running_tasks = [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]
        
        if not completed_tasks and not running_tasks:
            return None
            
        # Calculate average time per completed task
        total_duration = timedelta()
        for task in completed_tasks:
            if task.started_at and task.completed_at:
                start_time = datetime.fromisoformat(task.started_at)
                end_time = datetime.fromisoformat(task.completed_at)
                total_duration += end_time - start_time
        
        if len(completed_tasks) > 0:
            avg_duration = total_duration / len(completed_tasks)
            remaining_tasks = len([t for t in self._tasks.values() if t.status == TaskStatus.PENDING])
            estimated_remaining = avg_duration * remaining_tasks
            
            # Add time for currently running tasks (rough estimate)
            for task in running_tasks:
                if task.started_at:
                    start_time = datetime.fromisoformat(task.started_at)
                    elapsed = datetime.now() - start_time
                    # Assume running tasks are 50% complete
                    estimated_remaining += elapsed
            
            completion_time = datetime.now() + estimated_remaining
            return completion_time.isoformat()
        
        return None
    
    def _get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about current pipeline stage"""
        running_tasks = [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]
        
        if running_tasks:
            current_task = running_tasks[0]  # Assume single task per stage
            return {
                "stage_name": current_task.name,
                "task_id": current_task.task_id,
                "progress": asdict(current_task.progress) if current_task.progress else None,
                "elapsed_time": self._calculate_elapsed_time(current_task)
            }
        
        return {}
    
    def _calculate_elapsed_time(self, task: TaskInfo) -> Optional[str]:
        """Calculate elapsed time for a task"""
        if task.started_at:
            start_time = datetime.fromisoformat(task.started_at)
            elapsed = datetime.now() - start_time
            return str(elapsed).split('.')[0]  # Remove microseconds
        return None
    
    def cleanup_completed_tasks(self, keep_recent: int = 10):
        """Remove old completed tasks to keep status file manageable"""
        with self._lock:
            completed_tasks = [(task_id, task) for task_id, task in self._tasks.items() 
                             if task.status == TaskStatus.COMPLETED]
            
            # Sort by completion time
            completed_tasks.sort(key=lambda x: x[1].completed_at or "", reverse=True)
            
            # Keep only recent completed tasks
            if len(completed_tasks) > keep_recent:
                tasks_to_remove = completed_tasks[keep_recent:]
                for task_id, _ in tasks_to_remove:
                    del self._tasks[task_id]
                
                logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
    
    def reset_failed_tasks(self):
        """Reset failed tasks to pending for retry"""
        with self._lock:
            reset_count = 0
            for task in self._tasks.values():
                if task.status == TaskStatus.FAILED and task.error and task.error.recoverable:
                    task.status = TaskStatus.PENDING
                    task.completed_at = None
                    if task.error:
                        task.error.retry_count += 1
                    reset_count += 1
            
            logger.info(f"Reset {reset_count} failed tasks for retry")
    
    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        status = self.get_pipeline_status()
        
        report = f"""
DETR Background Training Pipeline Status
========================================

Pipeline ID: {status['pipeline_metadata']['pipeline_id']}
Started: {status['pipeline_metadata']['started_at']}
Overall Progress: {status['overall_progress']:.1f}%

Tasks Summary:
- Total: {status['total_tasks']}
- Completed: {status['completed_tasks']}
- Running: {status['running_tasks']}
- Pending: {status['pending_tasks']}
- Failed: {status['failed_tasks']}

"""
        
        if status.get('estimated_completion'):
            report += f"Estimated Completion: {status['estimated_completion']}\n"
        
        if status.get('current_stage_info'):
            stage_info = status['current_stage_info']
            report += f"\nCurrent Stage: {stage_info.get('stage_name', 'Unknown')}\n"
            if stage_info.get('elapsed_time'):
                report += f"Elapsed Time: {stage_info['elapsed_time']}\n"
        
        # Add task details
        report += "\nTask Details:\n"
        report += "-" * 40 + "\n"
        
        for task_id, task in self._tasks.items():
            status_symbol = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.RUNNING: "🔄", 
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.CANCELLED: "🚫"
            }.get(task.status, "❓")
            
            report += f"{status_symbol} {task.name} ({task_id})\n"
            
            if task.progress:
                if task.progress.progress_type == ProgressType.PERCENTAGE:
                    report += f"   Progress: {task.progress.current}%\n"
                elif task.progress.progress_type == ProgressType.COUNT:
                    report += f"   Progress: {task.progress.current}/{task.progress.total}\n"
                
                if task.progress.message:
                    report += f"   Status: {task.progress.message}\n"
            
            if task.error:
                report += f"   Error: {task.error.message}\n"
        
        return report

# Singleton instance for global access
_global_status_manager: Optional[StatusManager] = None

def get_status_manager(status_file: str = "pipeline_status.json") -> StatusManager:
    """Get global status manager instance"""
    global _global_status_manager
    if _global_status_manager is None:
        _global_status_manager = StatusManager(status_file)
    return _global_status_manager