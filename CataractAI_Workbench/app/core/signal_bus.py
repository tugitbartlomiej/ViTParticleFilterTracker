"""Global signal bus for cross-component communication."""

from PyQt6.QtCore import QObject, pyqtSignal


class SignalBus(QObject):
    """Singleton signal bus for application-wide events."""

    # Pipeline events
    pipeline_started = pyqtSignal(str)  # pipeline name
    pipeline_finished = pyqtSignal(str, dict)  # pipeline name, results
    pipeline_error = pyqtSignal(str, str)  # pipeline name, error message

    # Training events
    epoch_completed = pyqtSignal(dict)  # {epoch, loss, lr, ...}
    training_started = pyqtSignal(str)  # model name
    training_finished = pyqtSignal(str, dict)  # model name, final metrics

    # Eden/SSH events
    ssh_connected = pyqtSignal(str)  # host
    ssh_disconnected = pyqtSignal(str)  # host
    job_status_changed = pyqtSignal(str, str)  # job_id, new_status
    remote_log_line = pyqtSignal(str, str)  # job_id, line

    # General
    status_message = pyqtSignal(str)  # message for status bar
    log_message = pyqtSignal(str, str)  # source, message


_instance = None


def get_signal_bus() -> SignalBus:
    """Get the global signal bus singleton."""
    global _instance
    if _instance is None:
        _instance = SignalBus()
    return _instance
