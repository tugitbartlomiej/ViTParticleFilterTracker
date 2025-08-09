"""
Background Task Executor - Async Subprocess Management
=====================================================

Executes Python scripts as background processes with comprehensive monitoring,
error handling, and communication with Claude Code agents.

Features:
- Async subprocess execution with monitoring
- Environment isolation and dependency management  
- Resource limits and cleanup
- Graceful shutdown handling
- Progress reporting integration
- Error recovery and retry logic
"""

import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import psutil

from status_manager import StatusManager, TaskStatus, TaskError, get_status_manager
from pipeline_monitor import get_pipeline_monitor

logger = logging.getLogger(__name__)

class TaskExecutionConfig:
    """Configuration for task execution"""
    
    def __init__(self):
        # Execution settings
        self.python_executable: str = sys.executable
        self.working_directory: Optional[Path] = None
        self.environment_variables: Dict[str, str] = {}
        self.timeout: Optional[int] = None  # seconds
        
        # Resource limits
        self.max_memory_mb: Optional[int] = None
        self.max_cpu_percent: Optional[float] = None
        
        # Retry settings
        self.max_retries: int = 2
        self.retry_delay: int = 60  # seconds
        
        # Monitoring settings
        self.enable_monitoring: bool = True
        self.progress_check_interval: int = 30  # seconds
        
        # Cleanup settings
        self.cleanup_on_completion: bool = True
        self.keep_logs: bool = True

class BackgroundProcess:
    """Represents a running background process"""
    
    def __init__(self, task_id: str, script_path: Path, 
                 args: List[str], config: TaskExecutionConfig):
        self.task_id = task_id
        self.script_path = script_path
        self.args = args
        self.config = config
        
        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.psutil_process: Optional[psutil.Process] = None
        
        # Status tracking
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.return_code: Optional[int] = None
        self.stdout_file: Optional[Path] = None
        self.stderr_file: Optional[Path] = None
        
        # Managers
        self.status_manager = get_status_manager()
        self.pipeline_monitor = get_pipeline_monitor()
        
        # Setup logging
        self.log_file = Path(f"{task_id}.log")
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for this process"""
        # Create log files
        output_dir = Path("pipeline_output") / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stdout_file = output_dir / f"{self.task_id}_stdout.log"
        self.stderr_file = output_dir / f"{self.task_id}_stderr.log"
        self.log_file = output_dir / f"{self.task_id}.log"
    
    async def start(self) -> bool:
        """Start the background process"""
        try:
            # Prepare command
            cmd = self.build_command()
            
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.environment_variables)
            
            # Add status reporting to environment
            env['TASK_ID'] = self.task_id
            env['STATUS_FILE'] = str(self.status_manager.status_file)
            env['LOG_FILE'] = str(self.log_file)
            
            # Start process
            logger.info(f"Starting task {self.task_id}: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=open(self.stdout_file, 'w'),
                stderr=open(self.stderr_file, 'w'), 
                env=env,
                cwd=self.config.working_directory,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.pid = self.process.pid
            self.started_at = datetime.now()
            
            # Get psutil process for monitoring
            try:
                self.psutil_process = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                logger.warning(f"Could not create psutil process for PID {self.pid}")
            
            # Update status
            self.status_manager.start_task(self.task_id, self.pid)
            
            # Add to pipeline monitor
            if self.config.enable_monitoring:
                self.pipeline_monitor.add_task_monitor(
                    self.task_id, 
                    self.log_file,
                    self.pid
                )
            
            logger.info(f"Started task {self.task_id} with PID {self.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start task {self.task_id}: {e}")
            error = TaskError(
                error_type="start_error",
                message=str(e),
                recoverable=True
            )
            self.status_manager.fail_task(self.task_id, error)
            return False
    
    def build_command(self) -> List[str]:
        """Build the command to execute"""
        cmd = [self.config.python_executable, str(self.script_path)]
        cmd.extend(self.args)
        return cmd
    
    async def wait_for_completion(self, check_interval: int = 30) -> int:
        """Wait for process to complete with periodic checks"""
        if not self.process:
            return -1
        
        while True:
            # Check if process is still running
            if self.process.poll() is not None:
                # Process completed
                self.return_code = self.process.returncode
                self.completed_at = datetime.now()
                break
            
            # Check resource usage and limits
            if self.config.max_memory_mb or self.config.max_cpu_percent:
                await self.check_resource_limits()
            
            # Check timeout
            if self.config.timeout:
                elapsed = (datetime.now() - self.started_at).total_seconds()
                if elapsed > self.config.timeout:
                    logger.warning(f"Task {self.task_id} timed out after {elapsed} seconds")
                    await self.terminate()
                    self.return_code = -1
                    break
            
            # Wait before next check
            await asyncio.sleep(check_interval)
        
        # Handle completion
        await self.handle_completion()
        return self.return_code
    
    async def check_resource_limits(self):
        """Check if process exceeds resource limits"""
        if not self.psutil_process or not self.is_running():
            return
        
        try:
            # Check memory limit
            if self.config.max_memory_mb:
                memory_mb = self.psutil_process.memory_info().rss / 1024 / 1024
                if memory_mb > self.config.max_memory_mb:
                    logger.warning(f"Task {self.task_id} exceeded memory limit: {memory_mb:.1f}MB > {self.config.max_memory_mb}MB")
                    await self.terminate()
                    return
            
            # Check CPU limit (averaged over time)
            if self.config.max_cpu_percent:
                cpu_percent = self.psutil_process.cpu_percent()
                if cpu_percent > self.config.max_cpu_percent:
                    logger.warning(f"Task {self.task_id} exceeded CPU limit: {cpu_percent:.1f}% > {self.config.max_cpu_percent}%")
                    # Don't terminate immediately for CPU, just warn
            
        except psutil.NoSuchProcess:
            pass  # Process already gone
        except Exception as e:
            logger.debug(f"Error checking resource limits for {self.task_id}: {e}")
    
    def is_running(self) -> bool:
        """Check if process is still running"""
        if not self.process:
            return False
        return self.process.poll() is None
    
    async def terminate(self, timeout: int = 30):
        """Terminate the process gracefully"""
        if not self.is_running():
            return
        
        logger.info(f"Terminating task {self.task_id}")
        
        try:
            if os.name == 'nt':
                # Windows
                self.process.terminate()
            else:
                # Unix-like
                self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self.wait_for_process()),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                # Force kill
                logger.warning(f"Task {self.task_id} did not terminate gracefully, force killing")
                if os.name == 'nt':
                    self.process.kill()
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                
                await asyncio.sleep(2)
        
        except Exception as e:
            logger.error(f"Error terminating task {self.task_id}: {e}")
        
        self.return_code = -1
        self.completed_at = datetime.now()
    
    async def wait_for_process(self):
        """Wait for process to finish (used with timeout)"""
        while self.process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def handle_completion(self):
        """Handle process completion"""
        logger.info(f"Task {self.task_id} completed with return code {self.return_code}")
        
        # Close file handles
        if self.process:
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()
        
        # Update status based on return code
        if self.return_code == 0:
            # Success
            output_files = self.collect_output_files()
            self.status_manager.complete_task(self.task_id, output_files)
        else:
            # Failure
            error_message = await self.get_error_message()
            error = TaskError(
                error_type="execution_error",
                message=error_message,
                recoverable=True
            )
            self.status_manager.fail_task(self.task_id, error)
        
        # Cleanup
        if self.config.cleanup_on_completion:
            await self.cleanup()
        
        # Remove from monitor
        if self.config.enable_monitoring:
            self.pipeline_monitor.remove_task_monitor(self.task_id)
    
    def collect_output_files(self) -> List[str]:
        """Collect output files created by the task"""
        output_files = []
        
        # Add log files
        if self.log_file and self.log_file.exists():
            output_files.append(str(self.log_file))
        
        if self.stdout_file and self.stdout_file.exists():
            output_files.append(str(self.stdout_file))
        
        # Look for common output patterns
        output_dir = Path("pipeline_output") / self.task_id
        if output_dir.exists():
            output_files.extend([str(f) for f in output_dir.rglob("*") if f.is_file()])
        
        return output_files
    
    async def get_error_message(self) -> str:
        """Extract error message from stderr"""
        if not self.stderr_file or not self.stderr_file.exists():
            return f"Process exited with code {self.return_code}"
        
        try:
            with open(self.stderr_file, 'r') as f:
                stderr_content = f.read()
            
            # Extract last few lines for error message
            lines = stderr_content.strip().split('\n')
            error_lines = lines[-5:] if len(lines) > 5 else lines
            return '\n'.join(error_lines)
        
        except Exception:
            return f"Process exited with code {self.return_code}"
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Remove temporary files if configured
            if not self.config.keep_logs:
                for log_file in [self.stdout_file, self.stderr_file]:
                    if log_file and log_file.exists():
                        log_file.unlink()
            
            logger.debug(f"Cleaned up resources for task {self.task_id}")
        
        except Exception as e:
            logger.warning(f"Error during cleanup of task {self.task_id}: {e}")

class BackgroundTaskExecutor:
    """Main executor for background tasks"""
    
    def __init__(self):
        self.running_processes: Dict[str, BackgroundProcess] = {}
        self.status_manager = get_status_manager()
        self.pipeline_monitor = get_pipeline_monitor()
        
        # Start monitoring
        self.pipeline_monitor.start_monitoring()
    
    async def execute_task(self, task_id: str, script_path: Union[str, Path],
                          args: Optional[List[str]] = None, 
                          config: Optional[TaskExecutionConfig] = None,
                          wait_for_completion: bool = True) -> int:
        """Execute a task in background"""
        
        # Prepare parameters
        script_path = Path(script_path)
        args = args or []
        config = config or TaskExecutionConfig()
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Create task in status manager
        self.status_manager.create_task(
            task_id=task_id,
            name=script_path.stem,
            log_file=str(Path("pipeline_output") / "logs" / f"{task_id}.log")
        )
        
        # Create background process
        bg_process = BackgroundProcess(task_id, script_path, args, config)
        
        # Start the process
        if not await bg_process.start():
            return -1
        
        self.running_processes[task_id] = bg_process
        
        # Wait for completion or return immediately
        if wait_for_completion:
            return_code = await bg_process.wait_for_completion()
            self.running_processes.pop(task_id, None)
            return return_code
        else:
            # Return immediately, process runs in background
            return 0
    
    async def execute_task_with_retry(self, task_id: str, script_path: Union[str, Path],
                                    args: Optional[List[str]] = None,
                                    config: Optional[TaskExecutionConfig] = None) -> int:
        """Execute task with automatic retry on failure"""
        
        config = config or TaskExecutionConfig()
        last_error = None
        
        for attempt in range(config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for task {task_id}")
                    await asyncio.sleep(config.retry_delay)
                
                return_code = await self.execute_task(
                    f"{task_id}_attempt_{attempt}",
                    script_path, 
                    args,
                    config,
                    wait_for_completion=True
                )
                
                if return_code == 0:
                    return 0
                
                last_error = f"Process exited with code {return_code}"
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt} failed for task {task_id}: {e}")
        
        # All attempts failed
        logger.error(f"Task {task_id} failed after {config.max_retries + 1} attempts")
        error = TaskError(
            error_type="retry_exhausted",
            message=f"Failed after {config.max_retries + 1} attempts. Last error: {last_error}",
            recoverable=False,
            retry_count=config.max_retries
        )
        self.status_manager.fail_task(task_id, error)
        return -1
    
    def get_running_tasks(self) -> List[str]:
        """Get list of currently running task IDs"""
        return list(self.running_processes.keys())
    
    def is_task_running(self, task_id: str) -> bool:
        """Check if a specific task is running"""
        bg_process = self.running_processes.get(task_id)
        return bg_process is not None and bg_process.is_running()
    
    async def terminate_task(self, task_id: str, timeout: int = 30) -> bool:
        """Terminate a specific task"""
        bg_process = self.running_processes.get(task_id)
        if not bg_process:
            return False
        
        await bg_process.terminate(timeout)
        self.running_processes.pop(task_id, None)
        return True
    
    async def terminate_all_tasks(self, timeout: int = 30):
        """Terminate all running tasks"""
        if not self.running_processes:
            return
        
        logger.info("Terminating all running tasks")
        
        # Terminate all processes
        tasks = []
        for task_id, bg_process in self.running_processes.items():
            tasks.append(bg_process.terminate(timeout))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.running_processes.clear()
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific task"""
        bg_process = self.running_processes.get(task_id)
        if not bg_process:
            return {"error": "Task not found"}
        
        status = {
            "task_id": task_id,
            "pid": bg_process.pid,
            "is_running": bg_process.is_running(),
            "started_at": bg_process.started_at.isoformat() if bg_process.started_at else None,
            "return_code": bg_process.return_code,
            "log_file": str(bg_process.log_file)
        }
        
        # Add resource usage if available
        if bg_process.psutil_process and bg_process.is_running():
            try:
                status["cpu_percent"] = bg_process.psutil_process.cpu_percent()
                memory_info = bg_process.psutil_process.memory_info()
                status["memory_mb"] = memory_info.rss / 1024 / 1024
            except psutil.NoSuchProcess:
                pass
        
        return status
    
    async def shutdown(self):
        """Shutdown the executor gracefully"""
        logger.info("Shutting down background task executor")
        
        # Terminate all running tasks
        await self.terminate_all_tasks()
        
        # Stop monitoring
        self.pipeline_monitor.stop_monitoring()

# Global executor instance
_global_task_executor: Optional[BackgroundTaskExecutor] = None

def get_task_executor() -> BackgroundTaskExecutor:
    """Get global task executor instance"""
    global _global_task_executor
    if _global_task_executor is None:
        _global_task_executor = BackgroundTaskExecutor()
    return _global_task_executor

async def execute_script_async(task_id: str, script_path: Union[str, Path],
                              args: Optional[List[str]] = None,
                              config: Optional[TaskExecutionConfig] = None) -> int:
    """Convenience function to execute a script asynchronously"""
    executor = get_task_executor()
    return await executor.execute_task(task_id, script_path, args, config)

def create_script_args(script_config: Dict[str, Any]) -> List[str]:
    """Helper to create script arguments from configuration dictionary"""
    args = []
    for key, value in script_config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif value is not None:
            args.extend([f"--{key}", str(value)])
    return args