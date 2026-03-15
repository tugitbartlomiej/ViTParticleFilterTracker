"""High-level Eden HPC adapter combining SSHManager + EdenClient."""

import logging
from typing import Generator, Optional

from ..app.core.ssh_manager import SSHManager
from ..app.core.eden_client import EdenClient

logger = logging.getLogger(__name__)


class EdenAdapter:
    """Facade that bundles SSH connectivity and SLURM operations.

    This is the single entry-point the GUI (or any other consumer)
    should use for Eden interactions.
    """

    def __init__(self):
        self._ssh = SSHManager()
        self._client = EdenClient(self._ssh)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ssh_manager(self) -> SSHManager:
        return self._ssh

    @property
    def eden_client(self) -> EdenClient:
        return self._client

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(
        self,
        password: Optional[str] = None,
        key_path: Optional[str] = None,
    ) -> bool:
        """Connect to Eden via the two-hop SSH tunnel.

        Returns ``True`` on success.
        """
        ok = self._ssh.connect(password=password, key_path=key_path)
        if ok:
            logger.info("EdenAdapter: connected to Eden.")
        else:
            logger.error("EdenAdapter: connection failed.")
        return ok

    def disconnect(self):
        """Disconnect from Eden."""
        self._ssh.disconnect()
        logger.info("EdenAdapter: disconnected.")

    def is_connected(self) -> bool:
        return self._ssh.is_connected()

    # ------------------------------------------------------------------
    # Dashboard data (jobs + cluster info in one call)
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> dict:
        """Return combined cluster information and job queue.

        Returns ``{"cluster": {...}, "jobs": [...]}``.
        """
        cluster = self._client.get_cluster_info()
        jobs = self._client.get_queue(user_only=True)
        return {"cluster": cluster, "jobs": jobs}

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def submit_and_monitor(self, script_path: str) -> str:
        """Submit a SLURM script and return the job ID.

        Raises ``RuntimeError`` on failure.
        """
        return self._client.submit_job(script_path)

    def cancel_job(self, job_id: str):
        self._client.cancel_job(job_id)

    # ------------------------------------------------------------------
    # Log streaming
    # ------------------------------------------------------------------

    def stream_job_log(self, job_id: str) -> Generator[str, None, None]:
        """Stream the log file for *job_id* via ``tail -f``.

        Yields lines as they appear.
        """
        # First, find the log file
        log_content = self._client.get_job_log(job_id)
        if not log_content:
            # Try default location
            log_path = f"slurm-{job_id}.out"
        else:
            log_path = f"slurm-{job_id}.out"

        yield from self._ssh.stream(f"tail -f {log_path}")

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        progress_cb=None,
    ):
        """Upload a file from local to Eden."""
        self._ssh.upload(local_path, remote_path, progress_callback=progress_cb)

    def download_file(
        self,
        remote_path: str,
        local_path: str,
        progress_cb=None,
    ):
        """Download a file from Eden to local."""
        self._ssh.download(remote_path, local_path, progress_callback=progress_cb)
