"""High-level SLURM / Eden cluster operations built on SSHManager."""

import json
import logging
import re
from typing import List, Optional

from .ssh_manager import SSHManager

logger = logging.getLogger(__name__)


class EdenClient:
    """Convenience wrapper around :class:`SSHManager` for SLURM operations.

    Provides queue inspection, cluster info, job submission / cancellation,
    and log retrieval on the Eden HPC cluster.
    """

    def __init__(self, ssh_manager: SSHManager):
        self._ssh = ssh_manager

    # ------------------------------------------------------------------
    # Queue / Jobs
    # ------------------------------------------------------------------

    def get_queue(self, user_only: bool = True) -> List[dict]:
        """Return a list of SLURM job dicts for the current user.

        Tries ``squeue --json`` first (Slurm >= 21.08) and falls back to
        delimited text output.
        """
        user_flag = f"-u {self._ssh.eden_user}" if user_only else ""

        # Try JSON first
        out, err, rc = self._ssh.execute(
            f"squeue {user_flag} --json", timeout=15
        )
        if rc == 0 and out.strip():
            try:
                data = json.loads(out)
                jobs: List[dict] = []
                for j in data.get("jobs", []):
                    jobs.append(
                        {
                            "job_id": str(j.get("job_id", "")),
                            "name": j.get("name", ""),
                            "partition": j.get("partition", ""),
                            "state": j.get("job_state", "UNKNOWN"),
                            "time": j.get("time", "0:00"),
                            "time_limit": j.get("time_limit", {}).get("number", ""),
                            "node": j.get("nodes", ""),
                            "gpus": self._extract_gpu_field(j),
                        }
                    )
                return jobs
            except (json.JSONDecodeError, KeyError):
                logger.debug("JSON squeue failed, falling back to text.")

        # Fallback: delimited text
        out, err, rc = self._ssh.execute(
            f'squeue {user_flag} -o "%i|%j|%P|%T|%M|%l|%N|%b"', timeout=15
        )
        if rc != 0:
            logger.error("squeue failed: %s", err)
            return []

        jobs = []
        for line in out.strip().splitlines()[1:]:  # skip header
            parts = line.strip().split("|")
            if len(parts) < 8:
                continue
            jobs.append(
                {
                    "job_id": parts[0],
                    "name": parts[1],
                    "partition": parts[2],
                    "state": parts[3],
                    "time": parts[4],
                    "time_limit": parts[5],
                    "node": parts[6],
                    "gpus": parts[7],
                }
            )
        return jobs

    # ------------------------------------------------------------------
    # Cluster info
    # ------------------------------------------------------------------

    def get_cluster_info(self) -> dict:
        """Return structured information about cluster nodes.

        Returns ``{"nodes": [...]}``.  Each node dict has keys:
        ``name``, ``partition``, ``gpus``, ``cpus``, ``memory``, ``state``.
        """
        # Try JSON first
        out, err, rc = self._ssh.execute("sinfo --json", timeout=15)
        if rc == 0 and out.strip():
            try:
                data = json.loads(out)
                nodes: List[dict] = []
                for n in data.get("nodes", []):
                    nodes.append(
                        {
                            "name": n.get("name", ""),
                            "partition": ", ".join(n.get("partitions", [])),
                            "gpus": n.get("gres", ""),
                            "cpus": f'{n.get("alloc_cpus", 0)}/{n.get("cpus", 0)}',
                            "memory": n.get("real_memory", 0),
                            "state": n.get("state", "UNKNOWN"),
                        }
                    )
                return {"nodes": nodes}
            except (json.JSONDecodeError, KeyError):
                logger.debug("JSON sinfo failed, falling back to text.")

        # Fallback: delimited text
        out, err, rc = self._ssh.execute(
            'sinfo -N -o "%N|%P|%G|%C|%m|%T"', timeout=15
        )
        if rc != 0:
            logger.error("sinfo failed: %s", err)
            return {"nodes": []}

        nodes = []
        for line in out.strip().splitlines()[1:]:
            parts = line.strip().split("|")
            if len(parts) < 6:
                continue
            nodes.append(
                {
                    "name": parts[0],
                    "partition": parts[1].rstrip("*"),
                    "gpus": parts[2],
                    "cpus": parts[3],
                    "memory": parts[4],
                    "state": parts[5],
                }
            )
        return {"nodes": nodes}

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def submit_job(self, script_path: str) -> str:
        """Submit a SLURM batch script and return the job ID.

        Raises ``RuntimeError`` on failure.
        """
        out, err, rc = self._ssh.execute(f"sbatch {script_path}", timeout=30)
        if rc != 0:
            raise RuntimeError(f"sbatch failed: {err}")

        # Output is typically "Submitted batch job 123456"
        match = re.search(r"(\d+)", out)
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {out}")
        job_id = match.group(1)
        logger.info("Submitted job %s from %s", job_id, script_path)
        return job_id

    def cancel_job(self, job_id: str):
        """Cancel the SLURM job *job_id*."""
        out, err, rc = self._ssh.execute(f"scancel {job_id}", timeout=15)
        if rc != 0:
            logger.error("scancel %s failed: %s", job_id, err)
        else:
            logger.info("Cancelled job %s", job_id)

    # ------------------------------------------------------------------
    # Log retrieval
    # ------------------------------------------------------------------

    def get_job_log(self, job_id: str) -> str:
        """Locate and return the contents of the log file for *job_id*.

        Searches common SLURM output patterns:
        ``slurm-<job_id>.out``, then falls back to ``sacct``.
        """
        # Try common locations
        for pattern in [
            f"slurm-{job_id}.out",
            f"~/slurm-{job_id}.out",
            f"~/DETR/logs/slurm-{job_id}.out",
        ]:
            out, err, rc = self._ssh.execute(
                f"test -f {pattern} && cat {pattern}", timeout=30
            )
            if rc == 0 and out.strip():
                return out

        # Try sacct for the output file path
        out, err, rc = self._ssh.execute(
            f'sacct -j {job_id} --format=JobID,StdOut --noheader -P',
            timeout=15,
        )
        if rc == 0 and out.strip():
            for line in out.strip().splitlines():
                parts = line.split("|")
                if len(parts) >= 2 and parts[1].strip():
                    log_path = parts[1].strip()
                    content, _, rc2 = self._ssh.execute(
                        f"cat {log_path}", timeout=30
                    )
                    if rc2 == 0:
                        return content

        return ""

    def tail_log(self, log_path: str, lines: int = 50) -> str:
        """Return the last *lines* lines of *log_path* on Eden."""
        out, err, rc = self._ssh.execute(
            f"tail -n {lines} {log_path}", timeout=15
        )
        if rc != 0:
            logger.error("tail failed for %s: %s", log_path, err)
            return ""
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_gpu_field(job_dict: dict) -> str:
        """Extract a human-readable GPU string from a JSON job dict."""
        gres = job_dict.get("gres_detail", [])
        if gres:
            return ", ".join(gres)
        tres = job_dict.get("tres_alloc_str", "")
        match = re.search(r"gres/gpu[=:](\d+)", tres)
        if match:
            return f"gpu:{match.group(1)}"
        return ""
