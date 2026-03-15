"""File transfer widget for upload / download between local and Eden."""

from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot

from ...core.ssh_manager import SSHManager
from ...widgets.file_picker import FilePicker


class _TransferWorker(QThread):
    """Background thread for a single SFTP transfer."""

    progress = pyqtSignal(int)           # percent 0-100
    finished = pyqtSignal(str)           # success message
    error = pyqtSignal(str)              # error message

    def __init__(
        self,
        ssh_manager: SSHManager,
        direction: str,                  # "upload" | "download"
        local_path: str,
        remote_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self._ssh = ssh_manager
        self._direction = direction
        self._local = local_path
        self._remote = remote_path
        self._total_bytes = 0

    def run(self):
        try:
            if self._direction == "upload":
                import os
                self._total_bytes = os.path.getsize(self._local)
                self._ssh.upload(
                    self._local, self._remote,
                    progress_callback=self._cb,
                )
                self.finished.emit(
                    f"Upload complete: {self._local} -> {self._remote}"
                )
            else:
                # For download we don't know total size upfront, but
                # paramiko's callback gives (transferred, total).
                self._ssh.download(
                    self._remote, self._local,
                    progress_callback=self._cb,
                )
                self.finished.emit(
                    f"Download complete: {self._remote} -> {self._local}"
                )
        except Exception as exc:
            self.error.emit(str(exc))

    def _cb(self, transferred: int, total: int):
        if total > 0:
            pct = int(transferred * 100 / total)
            self.progress.emit(min(pct, 100))


class FileTransfer(QWidget):
    """Upload / download panel with progress bar and transfer history."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ssh_manager: SSHManager | None = None
        self._worker: _TransferWorker | None = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- upload section --
        upload_box = QGroupBox("Local -> Eden (Upload)")
        ul = QVBoxLayout(upload_box)

        self._upload_local = FilePicker(label="Browse...", mode="file")
        ul.addWidget(QLabel("Local file:"))
        ul.addWidget(self._upload_local)

        row = QHBoxLayout()
        row.addWidget(QLabel("Remote path:"))
        self._upload_remote = QLineEdit()
        self._upload_remote.setPlaceholderText(
            "/mnt/evafs/faculty/home/bpiotrowski/models/my_model.pth"
        )
        row.addWidget(self._upload_remote)
        ul.addLayout(row)

        self._btn_upload = QPushButton("Upload")
        self._btn_upload.clicked.connect(self._on_upload)
        ul.addWidget(self._btn_upload)

        layout.addWidget(upload_box)

        # -- download section --
        download_box = QGroupBox("Eden -> Local (Download)")
        dl = QVBoxLayout(download_box)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Remote path:"))
        self._download_remote = QLineEdit()
        self._download_remote.setPlaceholderText(
            "/mnt/evafs/faculty/home/bpiotrowski/DETR/logs/slurm-123456.out"
        )
        row2.addWidget(self._download_remote)

        self._btn_browse_remote = QPushButton("Browse...")
        self._btn_browse_remote.setFixedWidth(80)
        self._btn_browse_remote.clicked.connect(self._browse_remote)
        row2.addWidget(self._btn_browse_remote)
        dl.addLayout(row2)

        self._download_local = FilePicker(label="Save As...", mode="save")
        dl.addWidget(QLabel("Local destination:"))
        dl.addWidget(self._download_local)

        self._btn_download = QPushButton("Download")
        self._btn_download.clicked.connect(self._on_download)
        dl.addWidget(self._btn_download)

        layout.addWidget(download_box)

        # -- progress bar --
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        # -- transfer history --
        layout.addWidget(QLabel("Recent Transfers:"))
        self._history = QListWidget()
        self._history.setMaximumHeight(140)
        layout.addWidget(self._history)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_ssh_manager(self, ssh_manager: SSHManager):
        self._ssh_manager = ssh_manager

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_upload(self):
        local = self._upload_local.path().strip()
        remote = self._upload_remote.text().strip()
        if not local or not remote:
            QMessageBox.warning(self, "Upload", "Both paths are required.")
            return
        self._start_transfer("upload", local, remote)

    def _on_download(self):
        remote = self._download_remote.text().strip()
        local = self._download_local.path().strip()
        if not remote or not local:
            QMessageBox.warning(self, "Download", "Both paths are required.")
            return
        self._start_transfer("download", local, remote)

    def _start_transfer(self, direction: str, local: str, remote: str):
        if self._ssh_manager is None or not self._ssh_manager.is_connected():
            QMessageBox.warning(
                self, "Transfer", "Not connected to Eden."
            )
            return

        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self, "Transfer", "A transfer is already in progress."
            )
            return

        self._progress.setValue(0)
        self._set_buttons_enabled(False)

        self._worker = _TransferWorker(
            self._ssh_manager, direction, local, remote
        )
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_transfer_finished)
        self._worker.error.connect(self._on_transfer_error)
        self._worker.start()

    @pyqtSlot(str)
    def _on_transfer_finished(self, msg: str):
        self._progress.setValue(100)
        self._set_buttons_enabled(True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._history.insertItem(0, QListWidgetItem(f"[{timestamp}] {msg}"))

    @pyqtSlot(str)
    def _on_transfer_error(self, msg: str):
        self._set_buttons_enabled(True)
        self._progress.setValue(0)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._history.insertItem(
            0, QListWidgetItem(f"[{timestamp}] ERROR: {msg}")
        )
        QMessageBox.critical(self, "Transfer Error", msg)

    def _set_buttons_enabled(self, enabled: bool):
        self._btn_upload.setEnabled(enabled)
        self._btn_download.setEnabled(enabled)

    def _browse_remote(self):
        """List remote home directory contents and fill the line edit."""
        if self._ssh_manager is None or not self._ssh_manager.is_connected():
            QMessageBox.warning(self, "Browse", "Not connected to Eden.")
            return

        try:
            entries = self._ssh_manager.list_dir(
                "/mnt/evafs/faculty/home/bpiotrowski"
            )
            names = [
                (e["name"] + "/" if e["is_dir"] else e["name"])
                for e in entries
            ]
            if names:
                from PyQt6.QtWidgets import QInputDialog

                chosen, ok = QInputDialog.getItem(
                    self,
                    "Remote Files",
                    "Select a file or directory:",
                    names,
                    editable=False,
                )
                if ok and chosen:
                    base = "/mnt/evafs/faculty/home/bpiotrowski/"
                    self._download_remote.setText(base + chosen.rstrip("/"))
        except Exception as exc:
            QMessageBox.critical(self, "Browse Error", str(exc))
