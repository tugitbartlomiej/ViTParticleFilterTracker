"""Registry for tracking model checkpoints and their metadata."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class ModelRegistry:
    """Track and manage model checkpoints across the project."""

    def __init__(self, registry_path: Optional[str] = None):
        self._registry_path = Path(registry_path) if registry_path else None
        self._models: dict[str, dict] = {}
        if self._registry_path and self._registry_path.exists():
            self._load()

    def _load(self):
        with open(self._registry_path, "r", encoding="utf-8") as f:
            self._models = json.load(f)

    def _save(self):
        if self._registry_path:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump(self._models, f, indent=2, default=str)

    def register(self, name: str, path: str, model_type: str = "detr", **metadata):
        """Register a model checkpoint."""
        p = Path(path)
        self._models[name] = {
            "path": str(p),
            "type": model_type,
            "size_mb": p.stat().st_size / (1024 * 1024) if p.exists() else 0,
            "registered_at": datetime.now().isoformat(),
            **metadata,
        }
        self._save()

    def get(self, name: str) -> Optional[dict]:
        return self._models.get(name)

    def list_models(self, model_type: Optional[str] = None) -> list[dict]:
        """List all registered models, optionally filtered by type."""
        models = []
        for name, info in self._models.items():
            if model_type and info.get("type") != model_type:
                continue
            models.append({"name": name, **info})
        return models

    def scan_directory(self, directory: str, model_type: str = "detr") -> list[dict]:
        """Scan a directory for .pth checkpoint files."""
        found = []
        for pth in sorted(Path(directory).glob("**/*.pth")):
            info = {
                "name": pth.stem,
                "path": str(pth),
                "type": model_type,
                "size_mb": pth.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(pth.stat().st_mtime).isoformat(),
            }
            found.append(info)
        return found

    def remove(self, name: str):
        """Remove a model from the registry (does not delete the file)."""
        self._models.pop(name, None)
        self._save()
