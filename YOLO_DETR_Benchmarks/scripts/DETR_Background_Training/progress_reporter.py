"""
Progress Reporter - Helper for existing scripts
==============================================

Lightweight progress reporting module that existing scripts can import
to report their progress to the pipeline monitoring system.

Usage in existing scripts:
    from progress_reporter import ProgressReporter
    
    reporter = ProgressReporter()
    reporter.start_stage("Loading data")
    reporter.update_progress(50, "Processed 500/1000 images")
    reporter.complete_stage("Data loaded successfully")
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Import status manager if available, otherwise use fallback
try:
    from status_manager import get_status_manager, TaskProgress, ProgressType
    STATUS_MANAGER_AVAILABLE = True
except ImportError:
    STATUS_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProgressReporter:
    """Simple progress reporter that can be imported by existing scripts"""
    
    def __init__(self, task_id: Optional[str] = None, 
                 log_file: Optional[Union[str, Path]] = None):
        
        # Get task ID from environment or parameter
        self.task_id = task_id or os.environ.get('TASK_ID', 'unknown_task')
        
        # Setup log file
        if log_file:
            self.log_file = Path(log_file)
        elif os.environ.get('LOG_FILE'):
            self.log_file = Path(os.environ['LOG_FILE'])
        else:
            self.log_file = Path(f"{self.task_id}.log")
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file
        self.setup_logging()
        
        # Status manager (if available)
        self.status_manager = None
        if STATUS_MANAGER_AVAILABLE:
            try:
                self.status_manager = get_status_manager()
            except Exception as e:
                logger.debug(f"Could not get status manager: {e}")
        
        # Progress tracking
        self.current_stage = None
        self.stage_start_time = None
        self.last_progress_time = 0
        self.progress_update_interval = 5  # seconds
        
        logger.info(f"Progress reporter initialized for task {self.task_id}")
    
    def setup_logging(self):
        """Setup file logging for progress"""
        # Create a separate logger for progress
        self.progress_logger = logging.getLogger(f'progress.{self.task_id}')
        self.progress_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.progress_logger.handlers[:]:
            self.progress_logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.progress_logger.addHandler(file_handler)
        
        # Don't propagate to parent logger
        self.progress_logger.propagate = False
    
    def start_stage(self, stage_name: str):
        """Start a new processing stage"""
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
        message = f"Stage: {stage_name}"
        self.progress_logger.info(message)
        print(f"[STAGE] {message}")
        
        # Update status manager if available
        if self.status_manager:
            try:
                progress = TaskProgress(
                    progress_type=ProgressType.STAGE,
                    current=stage_name,
                    message=f"Started: {stage_name}"
                )
                self.status_manager.update_progress(self.task_id, progress)
            except Exception as e:
                logger.debug(f"Could not update status manager: {e}")
    
    def update_progress(self, current: Union[int, float], 
                       total: Optional[Union[int, float]] = None,
                       message: str = "", force: bool = False):
        """Update progress with current/total or percentage"""
        
        # Rate limiting - don't update too frequently
        current_time = time.time()
        if not force and (current_time - self.last_progress_time) < self.progress_update_interval:
            return
        
        self.last_progress_time = current_time
        
        # Determine progress type and format message
        if total is not None:
            # Count-based progress
            progress_msg = f"Progress: {current}/{total}"
            if message:
                progress_msg += f" - {message}"
            
            progress_type = ProgressType.COUNT
            progress_current = current
            progress_total = total
            
        elif isinstance(current, (int, float)) and current <= 100:
            # Percentage progress
            progress_msg = f"Progress: {current:.1f}%"
            if message:
                progress_msg += f" - {message}"
            
            progress_type = ProgressType.PERCENTAGE
            progress_current = current
            progress_total = 100
            
        else:
            # Stage-based progress
            progress_msg = f"Processing: {current}"
            if message:
                progress_msg += f" - {message}"
            
            progress_type = ProgressType.STAGE
            progress_current = str(current)
            progress_total = None
        
        # Log progress
        self.progress_logger.info(progress_msg)
        print(f"[PROGRESS] {progress_msg}")
        
        # Update status manager if available
        if self.status_manager:
            try:
                progress = TaskProgress(
                    progress_type=progress_type,
                    current=progress_current,
                    total=progress_total,
                    message=message
                )
                self.status_manager.update_progress(self.task_id, progress)
            except Exception as e:
                logger.debug(f"Could not update status manager: {e}")
    
    def complete_stage(self, message: str = ""):
        """Complete current stage"""
        if self.current_stage and self.stage_start_time:
            duration = time.time() - self.stage_start_time
            completion_msg = f"Completed: {self.current_stage}"
            if message:
                completion_msg += f" - {message}"
            completion_msg += f" (took {duration:.1f}s)"
            
            self.progress_logger.info(completion_msg)
            print(f"[COMPLETED] {completion_msg}")
        
        self.current_stage = None
        self.stage_start_time = None
    
    def report_error(self, error_message: str, recoverable: bool = True):
        """Report an error"""
        self.progress_logger.error(f"ERROR: {error_message}")
        print(f"[ERROR] {error_message}")
        
        # Update status manager if available
        if self.status_manager:
            try:
                from status_manager import TaskError
                error = TaskError(
                    error_type="script_error",
                    message=error_message,
                    recoverable=recoverable
                )
                self.status_manager.fail_task(self.task_id, error)
            except Exception as e:
                logger.debug(f"Could not update status manager: {e}")
    
    def report_completion(self, message: str = "Task completed successfully"):
        """Report successful completion"""
        self.progress_logger.info(f"COMPLETED: {message}")
        print(f"[SUCCESS] {message}")
        
        # Update status manager if available
        if self.status_manager:
            try:
                self.status_manager.complete_task(self.task_id)
            except Exception as e:
                logger.debug(f"Could not update status manager: {e}")
    
    def log_info(self, message: str):
        """Log an informational message"""
        self.progress_logger.info(message)
        print(f"[INFO] {message}")
    
    def log_warning(self, message: str):
        """Log a warning message"""
        self.progress_logger.warning(f"WARNING: {message}")
        print(f"[WARNING] {message}")
    
    def with_progress(self, iterable, total: Optional[int] = None, 
                     desc: str = "Processing", update_every: int = 1):
        """Wrapper for iterables with automatic progress reporting"""
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        
        self.start_stage(desc)
        
        for i, item in enumerate(iterable):
            if i % update_every == 0 or i == total - 1:
                if total:
                    self.update_progress(i + 1, total, f"{desc}: {i + 1}/{total}")
                else:
                    self.update_progress(f"Item {i + 1}", message=desc)
            
            yield item
        
        self.complete_stage(f"Processed {i + 1} items")
    
    def timed_stage(self, stage_name: str):
        """Context manager for timed stages"""
        return TimedStageContext(self, stage_name)

class TimedStageContext:
    """Context manager for timed progress stages"""
    
    def __init__(self, reporter: ProgressReporter, stage_name: str):
        self.reporter = reporter
        self.stage_name = stage_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.reporter.start_stage(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            duration = time.time() - self.start_time
            self.reporter.complete_stage(f"Completed in {duration:.1f}s")
        else:
            # Error occurred
            self.reporter.report_error(f"Stage '{self.stage_name}' failed: {exc_val}")
        
        return False  # Don't suppress exceptions

# Convenience functions for quick usage
_global_reporter: Optional[ProgressReporter] = None

def get_reporter(task_id: Optional[str] = None) -> ProgressReporter:
    """Get global progress reporter instance"""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = ProgressReporter(task_id)
    return _global_reporter

def start_stage(stage_name: str):
    """Quick function to start a stage"""
    get_reporter().start_stage(stage_name)

def update_progress(current: Union[int, float], 
                   total: Optional[Union[int, float]] = None,
                   message: str = ""):
    """Quick function to update progress"""
    get_reporter().update_progress(current, total, message)

def complete_stage(message: str = ""):
    """Quick function to complete a stage"""
    get_reporter().complete_stage(message)

def report_error(error_message: str, recoverable: bool = True):
    """Quick function to report an error"""
    get_reporter().report_error(error_message, recoverable)

def report_completion(message: str = "Task completed successfully"):
    """Quick function to report completion"""
    get_reporter().report_completion(message)

def log_info(message: str):
    """Quick function to log info"""
    get_reporter().log_info(message)

def log_warning(message: str):
    """Quick function to log warning"""
    get_reporter().log_warning(message)