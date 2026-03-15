"""Base QThread worker with standard signals for pipeline operations."""

from PyQt6.QtCore import QThread, pyqtSignal


class PipelineWorker(QThread):
    """Base worker thread with progress/error/result signals.

    Subclass and override `run_task()` to implement specific work.
    """

    progress = pyqtSignal(int, str)  # percent, message
    stage_changed = pyqtSignal(str)  # stage name
    result_ready = pyqtSignal(dict)  # results
    error_occurred = pyqtSignal(str, str)  # error type, message
    log_message = pyqtSignal(str)  # log line

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def run(self):
        try:
            result = self.run_task()
            if not self._cancelled:
                self.result_ready.emit(result or {})
        except Exception as e:
            if not self._cancelled:
                self.error_occurred.emit(type(e).__name__, str(e))

    def run_task(self) -> dict:
        """Override in subclasses. Return result dict."""
        raise NotImplementedError

    def cancel(self):
        """Request cancellation. Subclasses should check self._cancelled."""
        self._cancelled = True

    def emit_progress(self, percent: int, message: str = ""):
        """Convenience method to emit progress."""
        self.progress.emit(percent, message)

    def emit_log(self, message: str):
        """Convenience method to emit log message."""
        self.log_message.emit(message)
