"""Thread-safe SSH manager with two-hop connection to Eden HPC cluster."""

import logging
import threading
import time
from pathlib import PurePosixPath
from typing import Generator, List, Optional

import paramiko

logger = logging.getLogger(__name__)


class SSHManager:
    """Two-hop Paramiko SSH connection: local -> jump host -> Eden.

    Thread-safe via an internal lock around execute / upload / download.
    Supports auto-reconnect on stale connections.
    """

    def __init__(
        self,
        jump_host: str = "ssh.mini.pw.edu.pl",
        jump_user: str = "piotrowskib2",
        eden_host: str = "eden",
        eden_user: str = "bpiotrowski",
    ):
        self.jump_host = jump_host
        self.jump_user = jump_user
        self.eden_host = eden_host
        self.eden_user = eden_user

        self._jump_client: Optional[paramiko.SSHClient] = None
        self._eden_client: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_running = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(
        self,
        password: Optional[str] = None,
        key_path: Optional[str] = None,
    ) -> bool:
        """Establish a two-hop SSH connection to Eden.

        Parameters
        ----------
        password : str | None
            Password used for both jump host and Eden.
        key_path : str | None
            Path to a private key file (optional).

        Returns
        -------
        bool
            ``True`` if the connection was established successfully.
        """
        with self._lock:
            try:
                # -- hop 1: connect to jump host --
                self._jump_client = paramiko.SSHClient()
                self._jump_client.set_missing_host_key_policy(
                    paramiko.AutoAddPolicy()
                )

                connect_kwargs: dict = {
                    "hostname": self.jump_host,
                    "username": self.jump_user,
                    "timeout": 30,
                }
                if key_path:
                    connect_kwargs["key_filename"] = key_path
                if password:
                    connect_kwargs["password"] = password

                logger.info("Connecting to jump host %s ...", self.jump_host)
                self._jump_client.connect(**connect_kwargs)

                # -- hop 2: tunnel through to eden --
                transport = self._jump_client.get_transport()
                if transport is None:
                    raise RuntimeError("Jump-host transport is None")

                channel = transport.open_channel(
                    "direct-tcpip",
                    (self.eden_host, 22),
                    ("127.0.0.1", 0),
                )

                self._eden_client = paramiko.SSHClient()
                self._eden_client.set_missing_host_key_policy(
                    paramiko.AutoAddPolicy()
                )

                eden_kwargs: dict = {
                    "hostname": self.eden_host,
                    "username": self.eden_user,
                    "sock": channel,
                    "timeout": 30,
                }
                if key_path:
                    eden_kwargs["key_filename"] = key_path
                if password:
                    eden_kwargs["password"] = password

                logger.info("Connecting to Eden via tunnel ...")
                self._eden_client.connect(**eden_kwargs)

                logger.info("SSH connection to Eden established.")
                self._start_heartbeat()
                return True

            except Exception:
                logger.exception("SSH connection failed")
                self._cleanup_clients()
                return False

    def disconnect(self):
        """Close all SSH connections gracefully."""
        self._stop_heartbeat()
        with self._lock:
            self._cleanup_clients()
            logger.info("SSH disconnected.")

    def is_connected(self) -> bool:
        """Return ``True`` if the Eden SSH session is still active."""
        if self._eden_client is None:
            return False
        transport = self._eden_client.get_transport()
        return transport is not None and transport.is_active()

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute(
        self,
        cmd: str,
        timeout: int = 30,
    ) -> tuple[str, str, int]:
        """Execute *cmd* on Eden and return ``(stdout, stderr, exit_code)``.

        Thread-safe.
        """
        with self._lock:
            self._ensure_connected()
            assert self._eden_client is not None

            logger.debug("SSH exec: %s", cmd)
            stdin, stdout, stderr = self._eden_client.exec_command(
                cmd, timeout=timeout
            )
            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            return out, err, exit_code

    def stream(self, cmd: str) -> Generator[str, None, None]:
        """Execute *cmd* and yield stdout lines as they arrive.

        Useful for ``tail -f`` style streaming.  Does **not** hold the
        lock while streaming so other operations can proceed.
        """
        self._ensure_connected()
        assert self._eden_client is not None

        transport = self._eden_client.get_transport()
        if transport is None:
            raise RuntimeError("SSH transport is None")

        channel = transport.open_session()
        channel.exec_command(cmd)

        buf = ""
        try:
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    data = channel.recv(4096).decode("utf-8", errors="replace")
                    buf += data
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        yield line
                else:
                    time.sleep(0.1)

            # Drain remaining data
            while channel.recv_ready():
                data = channel.recv(4096).decode("utf-8", errors="replace")
                buf += data
            if buf:
                for line in buf.split("\n"):
                    if line:
                        yield line
        finally:
            channel.close()

    # ------------------------------------------------------------------
    # File transfer (SFTP)
    # ------------------------------------------------------------------

    def _get_sftp(self) -> paramiko.SFTPClient:
        """Return (or create) a reusable SFTP client over the Eden channel."""
        if self._sftp is not None:
            try:
                self._sftp.stat(".")
                return self._sftp
            except Exception:
                self._sftp = None

        self._ensure_connected()
        assert self._eden_client is not None
        self._sftp = self._eden_client.open_sftp()
        return self._sftp

    def upload(
        self,
        local_path: str,
        remote_path: str,
        progress_callback=None,
    ):
        """Upload *local_path* to *remote_path* on Eden via SFTP.

        ``progress_callback(transferred_bytes, total_bytes)`` is called
        periodically during transfer.
        """
        with self._lock:
            sftp = self._get_sftp()
            logger.info("Uploading %s -> %s", local_path, remote_path)
            sftp.put(local_path, remote_path, callback=progress_callback)
            logger.info("Upload complete: %s", remote_path)

    def download(
        self,
        remote_path: str,
        local_path: str,
        progress_callback=None,
    ):
        """Download *remote_path* from Eden to *local_path*.

        ``progress_callback(transferred_bytes, total_bytes)`` is called
        periodically during transfer.
        """
        with self._lock:
            sftp = self._get_sftp()
            logger.info("Downloading %s -> %s", remote_path, local_path)
            sftp.get(remote_path, local_path, callback=progress_callback)
            logger.info("Download complete: %s", local_path)

    def list_dir(self, remote_path: str) -> List[dict]:
        """List contents of *remote_path* on Eden.

        Returns a list of dicts with keys:
        ``name``, ``size``, ``is_dir``, ``mtime``.
        """
        with self._lock:
            sftp = self._get_sftp()
            entries: List[dict] = []
            for attr in sftp.listdir_attr(remote_path):
                import stat as stat_mod

                entries.append(
                    {
                        "name": attr.filename,
                        "size": attr.st_size or 0,
                        "is_dir": stat_mod.S_ISDIR(attr.st_mode or 0),
                        "mtime": attr.st_mtime or 0,
                    }
                )
            return sorted(entries, key=lambda e: (not e["is_dir"], e["name"]))

    # ------------------------------------------------------------------
    # Heartbeat / auto-reconnect helpers
    # ------------------------------------------------------------------

    def _start_heartbeat(self):
        """Start a background heartbeat to keep the connection alive."""
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._heartbeat_running = False
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def _heartbeat_loop(self):
        """Send periodic keepalive pings."""
        while self._heartbeat_running:
            try:
                if self._eden_client is not None:
                    transport = self._eden_client.get_transport()
                    if transport is not None and transport.is_active():
                        transport.send_ignore()
            except Exception:
                logger.debug("Heartbeat failed — connection may be stale.")
            time.sleep(30)

    def _ensure_connected(self):
        """Raise if not connected."""
        if not self.is_connected():
            raise ConnectionError(
                "Not connected to Eden. Call connect() first."
            )

    def _cleanup_clients(self):
        """Close and nullify SSH clients and SFTP."""
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None

        if self._eden_client is not None:
            try:
                self._eden_client.close()
            except Exception:
                pass
            self._eden_client = None

        if self._jump_client is not None:
            try:
                self._jump_client.close()
            except Exception:
                pass
            self._jump_client = None
