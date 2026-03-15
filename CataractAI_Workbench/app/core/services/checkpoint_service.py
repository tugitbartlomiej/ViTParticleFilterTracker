"""Checkpoint analysis service -- load .pth files, compute layer stats, compare."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .base import AnalysisService


class CheckpointAnalysisService(AnalysisService):
    """Load ``.pth`` checkpoints, compute layer stats, compare checkpoints."""

    def analyze(self, data: dict) -> dict:
        """Dispatch to sub-analysis based on ``data["action"]``.

        Accepted actions:
        * ``"load"``    -- load a checkpoint file
        * ``"stats"``   -- compute per-layer stats
        * ``"compare"`` -- compare two state dicts
        """
        action = data.get("action", "load")
        if action == "load":
            return self.load_checkpoint(data["path"])
        if action == "stats":
            return self.compute_layer_stats(data["state_dict"], data["layer_name"])
        if action == "compare":
            return self.compare_state_dicts(
                data["state_dict_a"],
                data["state_dict_b"],
                data.get("name_a", "A"),
                data.get("name_b", "B"),
            )
        raise ValueError(f"Unknown action: {action}")

    # -- loading -----------------------------------------------------

    @staticmethod
    def load_checkpoint(path: str) -> dict:
        """Load a ``.pth`` file and return ``{path, state_dict, metadata}``."""
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(raw, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in raw:
                    state_dict = raw[key]
                    metadata = {k: v for k, v in raw.items() if k != key}
                    return {"path": path, "state_dict": state_dict, "metadata": metadata}
            # Might be a raw state dict
            state_dict = raw
            metadata: dict = {}
        else:
            state_dict = raw.state_dict() if hasattr(raw, "state_dict") else {}
            metadata = {}

        return {"path": path, "state_dict": state_dict, "metadata": metadata}

    # -- per-layer stats ---------------------------------------------

    @staticmethod
    def compute_layer_stats(state_dict: dict, layer_name: str) -> dict:
        """Return descriptive statistics and raw values for *layer_name*."""
        tensor = state_dict.get(layer_name)
        if tensor is None or not hasattr(tensor, "shape"):
            return {"error": f"Layer '{layer_name}' has no tensor data."}

        t = tensor.float()
        return {
            "layer_name": layer_name,
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "dtype": str(tensor.dtype),
            "values_flat": t.detach().cpu().numpy().flatten(),
        }

    # -- comparison --------------------------------------------------

    @staticmethod
    def compare_state_dicts(
        sd_a: dict,
        sd_b: dict,
        name_a: str = "A",
        name_b: str = "B",
    ) -> dict:
        """Compare per-layer std between two state dicts.

        Returns ``{name_a, name_b, rows}`` where each row is
        ``(layer, std_a, std_b, delta)``, sorted by delta descending.
        """
        rows: List[Tuple[str, float, float, float]] = []
        common_keys = set(sd_a.keys()) & set(sd_b.keys())

        for name in sorted(common_keys):
            ta = sd_a[name]
            tb = sd_b[name]
            if not hasattr(ta, "std") or not hasattr(tb, "std"):
                continue
            std_a = ta.float().std().item()
            std_b = tb.float().std().item()
            delta = abs(std_b - std_a)
            rows.append((name, std_a, std_b, delta))

        rows.sort(key=lambda r: r[3], reverse=True)
        return {"name_a": name_a, "name_b": name_b, "rows": rows}

    # -- directory scanning ------------------------------------------

    @staticmethod
    def scan_directory(dir_path: str) -> List[dict]:
        """Return list of ``{name, path, size_mb}`` for every ``.pth`` in *dir_path*."""
        pth_files = sorted(
            Path(dir_path).glob("*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [
            {
                "name": p.name,
                "path": str(p),
                "size_mb": p.stat().st_size / (1024 * 1024),
            }
            for p in pth_files
        ]

    # -- parameter counting ------------------------------------------

    @staticmethod
    def count_params(state_dict: dict) -> Tuple[int, List[str]]:
        """Return ``(total_param_count, layer_names)``."""
        total = 0
        names: List[str] = []
        for name, tensor in state_dict.items():
            if hasattr(tensor, "numel"):
                total += tensor.numel()
                names.append(name)
        return total, names
