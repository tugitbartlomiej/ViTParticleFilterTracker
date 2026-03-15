"""Configuration manager for loading/saving YAML and JSON configs."""

import json
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigManager:
    """Load, merge, and save configuration from YAML and JSON files."""

    def __init__(self):
        self._configs: dict[str, dict] = {}

    def load(self, path: str | Path, name: Optional[str] = None) -> dict:
        """Load a config file (YAML or JSON) and cache it by name."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        if path.suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        key = name or path.stem
        self._configs[key] = data
        return data

    def get(self, name: str) -> dict:
        """Get a previously loaded config by name."""
        if name not in self._configs:
            raise KeyError(f"Config '{name}' not loaded")
        return self._configs[name]

    def save(self, data: dict, path: str | Path):
        """Save config to file (format determined by extension)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in (".yaml", ".yml"):
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def get_nested(self, name: str, *keys, default: Any = None) -> Any:
        """Get a nested value from a config. E.g., get_nested('config', 'selection', 'method')."""
        data = self.get(name)
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, default)
            else:
                return default
        return data
