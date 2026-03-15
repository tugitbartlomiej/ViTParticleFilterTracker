"""Adapter for discovering, running, and collecting results from benchmark scripts."""

import ast
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import QProcess

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from CataractAI_Workbench.app.core.project_paths import YOLO_DETR_BENCHMARKS


class BenchmarkAdapter:
    """High-level adapter for benchmark script discovery, execution, and results.

    Parameters
    ----------
    scripts_dir : str | None
        Directory containing ``benchmark_*.py`` scripts.
        Defaults to ``YOLO_DETR_BENCHMARKS / "scripts"``.
    """

    def __init__(self, scripts_dir: Optional[str] = None):
        if scripts_dir is not None:
            self._scripts_dir = Path(scripts_dir)
        else:
            self._scripts_dir = YOLO_DETR_BENCHMARKS / "scripts"
        self._benchmarks_root = YOLO_DETR_BENCHMARKS / "Benchmarks"

    # ------------------------------------------------------------------
    # Script discovery
    # ------------------------------------------------------------------

    def discover_scripts(self) -> List[dict]:
        """Scan *scripts_dir* for ``benchmark_*.py`` files.

        Returns
        -------
        list[dict]
            Each dict has keys: ``name``, ``path``, ``description``.
        """
        if not self._scripts_dir.exists():
            logger.warning("Scripts directory does not exist: %s", self._scripts_dir)
            return []

        results: List[dict] = []
        for script in sorted(self._scripts_dir.glob("benchmark_*.py")):
            description = self._extract_docstring(script)
            results.append(
                {
                    "name": script.stem,
                    "path": str(script),
                    "description": description,
                }
            )
        return results

    @staticmethod
    def _extract_docstring(script_path: Path) -> str:
        """Extract the first line of the module docstring from *script_path*."""
        try:
            source = script_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(script_path))
            docstring = ast.get_docstring(tree)
            if docstring:
                # Return the first non-empty line
                for line in docstring.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("="):
                        return stripped
            return ""
        except Exception as exc:
            logger.debug("Could not parse docstring from %s: %s", script_path, exc)
            return ""

    # ------------------------------------------------------------------
    # Argparse introspection
    # ------------------------------------------------------------------

    def get_script_args(self, script_path: str) -> List[dict]:
        """Run ``python <script> --help`` and parse the argparse output.

        Parameters
        ----------
        script_path : str
            Full path to a benchmark script.

        Returns
        -------
        list[dict]
            Each dict has keys: ``name``, ``type``, ``default``, ``help``,
            ``required``, ``choices``.
        """
        try:
            result = subprocess.run(
                [sys.executable, script_path, "--help"],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(self._scripts_dir),
            )
            help_text = result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Could not get --help for %s: %s", script_path, exc)
            return []

        return self._parse_help_text(help_text)

    @staticmethod
    def _parse_help_text(help_text: str) -> List[dict]:
        """Parse argparse ``--help`` output into structured argument list."""
        args: List[dict] = []

        # Patterns for different argument styles:
        # --name VALUE   description (default: X)
        # --name {a,b,c}  description
        # -n, --name VALUE  description
        arg_re = re.compile(
            r"^\s+"
            r"(?:-\w,\s+)?"            # optional short flag like -d,
            r"(--[\w-]+)"              # long flag (captured)
            r"(?:\s+(\S+))?"           # optional metavar / {choices}
            r"\s{2,}(.*)",             # description (at least 2 spaces gap)
            re.MULTILINE,
        )

        for match in arg_re.finditer(help_text):
            name = match.group(1).lstrip("-")
            metavar = match.group(2) or ""
            description = match.group(3).strip()

            # Skip the built-in help flag
            if name in ("help", "h"):
                continue

            # Detect type from metavar
            arg_type = "str"
            if metavar.upper() in ("INT", "N"):
                arg_type = "int"
            elif metavar.upper() in ("FLOAT",):
                arg_type = "float"

            # Detect choices (e.g. {cuda,cpu})
            choices: Optional[List[str]] = None
            choices_match = re.search(r"\{([^}]+)\}", metavar)
            if choices_match:
                choices = [c.strip() for c in choices_match.group(1).split(",")]

            # Detect default
            default = None
            default_match = re.search(r"\(default:\s*(.+?)\)", description)
            if default_match:
                default = default_match.group(1).strip()

            # Detect required
            required = "(required)" in description.lower()

            args.append(
                {
                    "name": name,
                    "type": arg_type,
                    "default": default,
                    "help": description,
                    "required": required,
                    "choices": choices,
                }
            )

        return args

    # ------------------------------------------------------------------
    # Benchmark execution
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        script_path: str,
        args: Dict[str, str],
        on_output: Optional[Callable[[str], None]] = None,
        parent=None,
    ) -> QProcess:
        """Start a benchmark script as a QProcess.

        Parameters
        ----------
        script_path : str
            Full path to the script.
        args : dict
            Mapping of argument name to value. Names should not include
            leading dashes.
        on_output : callable(str) | None
            Called for each stdout line.
        parent : QObject | None
            Parent QObject for the QProcess.

        Returns
        -------
        QProcess
            The running process. Caller is responsible for managing its
            lifecycle (e.g. connecting ``finished`` signal).
        """
        process = QProcess(parent)
        process.setWorkingDirectory(str(self._scripts_dir))

        # Build command arguments
        cmd_args = [script_path]
        for name, value in args.items():
            if value is None or value == "":
                continue
            # Boolean flags: only add the flag itself, no value
            if isinstance(value, bool):
                if value:
                    cmd_args.append(f"--{name}")
                continue
            cmd_args.append(f"--{name}")
            cmd_args.append(str(value))

        # Connect output reader
        if on_output is not None:
            def _read_stdout():
                data = process.readAllStandardOutput()
                if data:
                    text = bytes(data).decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        on_output(line)

            def _read_stderr():
                data = process.readAllStandardError()
                if data:
                    text = bytes(data).decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        on_output(f"[STDERR] {line}")

            process.readyReadStandardOutput.connect(_read_stdout)
            process.readyReadStandardError.connect(_read_stderr)

        logger.info("Starting benchmark: %s %s", sys.executable, " ".join(cmd_args))
        process.start(sys.executable, cmd_args)
        return process

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def find_results(self, output_dir: str) -> List[str]:
        """Find JSON result files in *output_dir*.

        Returns
        -------
        list[str]
            Paths to ``results_summary.json`` or ``results_*.json`` files.
        """
        d = Path(output_dir)
        if not d.exists():
            return []
        patterns = ["results_summary.json", "results_*.json"]
        found: List[str] = []
        for pat in patterns:
            for p in d.glob(pat):
                if str(p) not in found:
                    found.append(str(p))
        return sorted(found)

    def load_results(self, json_path: str) -> dict:
        """Load benchmark results from a JSON file.

        Returns
        -------
        dict
            Raw benchmark results.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_previous_runs(self) -> List[dict]:
        """Scan for ``BENCHMARK_*`` directories containing results.

        Returns
        -------
        list[dict]
            Each dict has keys: ``name``, ``path``, ``date``, ``result_files``.
        """
        runs: List[dict] = []
        root = self._benchmarks_root
        if not root.exists():
            return runs

        for d in sorted(root.iterdir(), reverse=True):
            if not d.is_dir() or not d.name.startswith("BENCHMARK_"):
                continue

            result_files = self.find_results(str(d))

            # Try to extract date from directory name
            # e.g. BENCHMARK_Q81_20251231_013537
            date_str = ""
            date_match = re.search(r"(\d{8}_\d{6})", d.name)
            if date_match:
                try:
                    dt = datetime.strptime(date_match.group(1), "%Y%m%d_%H%M%S")
                    date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

            runs.append(
                {
                    "name": d.name,
                    "path": str(d),
                    "date": date_str,
                    "result_files": result_files,
                }
            )

        return runs
