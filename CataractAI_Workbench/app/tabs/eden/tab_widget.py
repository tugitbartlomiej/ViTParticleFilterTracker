"""Eden HPC tab -- main composite widget for the Eden sub-plan.

Layout::

    +--[Connection]---+--[Tabs]--------------------------------------+
    | Host: [eden]    | [Jobs] [Cluster] [Logs] [Files] [Terminal]   |
    | User: [bpiotr.] |                                              |
    | [Connect]       | (content of selected sub-tab)                |
    | Status: *       |                                              |
    +--[Sessions]-----+                                              |
    | session_1       |                                              |
    | session_2       |                                              |
    +-----------------+----------------------------------------------+
"""

import logging

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

from ...core.signal_bus import get_signal_bus
from ...core.ssh_manager import SSHManager
from ...core.eden_client import EdenClient

from .job_manager import JobManager
from .cluster_dashboard import ClusterDashboard
from .log_viewer import LogViewer
from .file_transfer import FileTransfer
from .session_browser import SessionBrowser
from ...widgets.ssh_terminal import SSHTerminal

logger = logging.getLogger(__name__)


class EdenTab(QWidget):
    """Top-level Eden HPC tab with connection panel, session browser,
    and tabbed sub-views for jobs, cluster, logs, file transfer, and
    a simple SSH terminal.
    """

    _REFRESH_INTERVAL_MS = 30_000  # 30 seconds

    def __init__(self, parent=None):
        super().__init__(parent)

        self._signal_bus = get_signal_bus()
        self._ssh_manager: SSHManager | None = None
        self._eden_client: EdenClient | None = None
        self._connected = False

        self._init_ui()
        self._init_timers()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _init_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- left panel (connection + sessions) ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_connection_group())

        self._session_browser = SessionBrowser()
        self._session_browser.session_selected.connect(self._on_session_selected)
        left_layout.addWidget(self._session_browser)

        splitter.addWidget(left)

        # ---- right panel (sub-tabs) ----
        self._sub_tabs = QTabWidget()

        self._job_manager = JobManager()
        self._job_manager.refresh_requested.connect(self._refresh_jobs)
        self._job_manager.submit_requested.connect(self._submit_job)
        self._job_manager.cancel_requested.connect(self._cancel_job)
        self._job_manager.view_log_requested.connect(self._view_job_log)
        self._sub_tabs.addTab(self._job_manager, "Jobs")

        self._cluster_dashboard = ClusterDashboard()
        self._cluster_dashboard.refresh_requested.connect(self._refresh_cluster)
        self._sub_tabs.addTab(self._cluster_dashboard, "Cluster")

        self._log_viewer = LogViewer()
        self._sub_tabs.addTab(self._log_viewer, "Logs")

        self._file_transfer = FileTransfer()
        self._sub_tabs.addTab(self._file_transfer, "Files")

        self._terminal = SSHTerminal()
        self._sub_tabs.addTab(self._terminal, "Terminal")

        splitter.addWidget(self._sub_tabs)

        # Default proportions: left 250px, right stretches
        splitter.setSizes([250, 800])

        root_layout.addWidget(splitter)

    def _build_connection_group(self) -> QGroupBox:
        """Build the connection controls group-box."""
        group = QGroupBox("Connection")
        layout = QVBoxLayout(group)

        # Jump host
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Jump:"))
        self._edit_jump = QLineEdit("ssh.mini.pw.edu.pl")
        self._edit_jump.setReadOnly(True)
        row1.addWidget(self._edit_jump)
        layout.addLayout(row1)

        # Eden host
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Host:"))
        self._edit_host = QLineEdit("eden")
        self._edit_host.setReadOnly(True)
        row2.addWidget(self._edit_host)
        layout.addLayout(row2)

        # User
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("User:"))
        self._edit_user = QLineEdit("bpiotrowski")
        self._edit_user.setReadOnly(True)
        row3.addWidget(self._edit_user)
        layout.addLayout(row3)

        # Password
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Pass:"))
        self._edit_password = QLineEdit()
        self._edit_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._edit_password.setPlaceholderText("SSH password")
        self._edit_password.returnPressed.connect(self._on_connect)
        row4.addWidget(self._edit_password)
        layout.addLayout(row4)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        btn_row.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_disconnect.setEnabled(False)
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        btn_row.addWidget(self._btn_disconnect)
        layout.addLayout(btn_row)

        # Status indicator
        self._lbl_status = QLabel("Disconnected")
        self._lbl_status.setStyleSheet(
            "color: #FF6B6B; font-weight: bold;"
        )
        layout.addWidget(self._lbl_status)

        return group

    def _init_timers(self):
        """Periodic refresh timer for jobs and cluster info."""
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._periodic_refresh)

    # ==================================================================
    # Connection management
    # ==================================================================

    @pyqtSlot()
    def _on_connect(self):
        password = self._edit_password.text().strip()
        if not password:
            QMessageBox.warning(
                self, "Connect", "Please enter the SSH password."
            )
            return

        self._lbl_status.setText("Connecting...")
        self._lbl_status.setStyleSheet("color: #FFD93D; font-weight: bold;")
        self._btn_connect.setEnabled(False)

        # Force the UI to repaint before the blocking connect call
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        self._ssh_manager = SSHManager(
            jump_host=self._edit_jump.text(),
            jump_user="piotrowskib2",
            eden_host=self._edit_host.text(),
            eden_user=self._edit_user.text(),
        )

        ok = self._ssh_manager.connect(password=password)
        if ok:
            self._connected = True
            self._eden_client = EdenClient(self._ssh_manager)

            # Distribute SSH manager to sub-components
            self._log_viewer.set_ssh_manager(self._ssh_manager)
            self._file_transfer.set_ssh_manager(self._ssh_manager)
            self._session_browser.set_ssh_manager(self._ssh_manager)
            self._terminal.set_ssh_manager(self._ssh_manager)

            self._lbl_status.setText("Connected")
            self._lbl_status.setStyleSheet(
                "color: #6BCB77; font-weight: bold;"
            )
            self._btn_connect.setEnabled(False)
            self._btn_disconnect.setEnabled(True)

            # Notify the rest of the app
            self._signal_bus.ssh_connected.emit(self._edit_host.text())
            self._signal_bus.status_message.emit(
                "Connected to Eden HPC cluster."
            )

            # Initial data fetch
            self._refresh_jobs()
            self._refresh_cluster()

            # Start periodic refresh
            self._refresh_timer.start(self._REFRESH_INTERVAL_MS)

        else:
            self._connected = False
            self._lbl_status.setText("Connection failed")
            self._lbl_status.setStyleSheet(
                "color: #FF6B6B; font-weight: bold;"
            )
            self._btn_connect.setEnabled(True)
            QMessageBox.critical(
                self,
                "Connection Error",
                "Failed to connect to Eden.\n"
                "Check your password and network connectivity.",
            )

    @pyqtSlot()
    def _on_disconnect(self):
        self._refresh_timer.stop()

        if self._ssh_manager is not None:
            self._ssh_manager.disconnect()
            self._ssh_manager = None
            self._eden_client = None

        self._connected = False
        self._lbl_status.setText("Disconnected")
        self._lbl_status.setStyleSheet(
            "color: #FF6B6B; font-weight: bold;"
        )
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)

        self._signal_bus.ssh_disconnected.emit("eden")
        self._signal_bus.status_message.emit("Disconnected from Eden.")

    # ==================================================================
    # Data refresh
    # ==================================================================

    def _periodic_refresh(self):
        """Called by QTimer every 30s while connected."""
        if not self._connected:
            return
        self._refresh_jobs()
        self._refresh_cluster()

    @pyqtSlot()
    def _refresh_jobs(self):
        if self._eden_client is None:
            return
        try:
            jobs = self._eden_client.get_queue(user_only=True)
            self._job_manager.update_jobs(jobs)
        except Exception as exc:
            logger.error("Failed to refresh jobs: %s", exc)

    @pyqtSlot()
    def _refresh_cluster(self):
        if self._eden_client is None:
            return
        try:
            info = self._eden_client.get_cluster_info()
            self._cluster_dashboard.update_info(info)
        except Exception as exc:
            logger.error("Failed to refresh cluster info: %s", exc)

    # ==================================================================
    # Job actions
    # ==================================================================

    @pyqtSlot(str)
    def _submit_job(self, script_path: str):
        """Upload a local script and submit it via sbatch."""
        if self._eden_client is None:
            return
        try:
            # Upload to home directory
            remote_script = (
                f"/mnt/evafs/faculty/home/bpiotrowski/"
                f"{script_path.replace(chr(92), '/').split('/')[-1]}"
            )
            self._ssh_manager.upload(script_path, remote_script)
            job_id = self._eden_client.submit_job(remote_script)
            self._signal_bus.job_status_changed.emit(job_id, "SUBMITTED")
            self._signal_bus.status_message.emit(
                f"Submitted job {job_id}"
            )
            self._refresh_jobs()
        except Exception as exc:
            QMessageBox.critical(
                self, "Submit Error", str(exc)
            )

    @pyqtSlot(str)
    def _cancel_job(self, job_id: str):
        if self._eden_client is None:
            return
        try:
            self._eden_client.cancel_job(job_id)
            self._signal_bus.job_status_changed.emit(job_id, "CANCELLED")
            self._refresh_jobs()
        except Exception as exc:
            QMessageBox.critical(
                self, "Cancel Error", str(exc)
            )

    @pyqtSlot(str)
    def _view_job_log(self, job_id: str):
        """Switch to the Logs sub-tab and start streaming the job log."""
        log_path = f"slurm-{job_id}.out"
        self._sub_tabs.setCurrentWidget(self._log_viewer)
        self._log_viewer.start_streaming(log_path)

    # ==================================================================
    # Session browser callback
    # ==================================================================

    @pyqtSlot(str)
    def _on_session_selected(self, remote_path: str):
        """Open a session log file in the log viewer."""
        self._sub_tabs.setCurrentWidget(self._log_viewer)
        self._log_viewer.start_streaming(remote_path)
