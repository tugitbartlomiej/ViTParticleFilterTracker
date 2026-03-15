"""Tests for CataractAI_Workbench.app.core.config_manager.ConfigManager.

Covers load, get, save, get_nested, and round-trip scenarios.
"""

import json

import pytest
import yaml

from CataractAI_Workbench.app.core.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

class TestLoad:
    """Tests for ConfigManager.load()."""

    def test_load_yaml(self, sample_config_yaml):
        mgr = ConfigManager()
        data = mgr.load(sample_config_yaml)
        assert data["selection_method"] == "cluster"
        assert data["target_size"] == 20000
        assert data["features"]["dino_pca_dim"] == 32

    def test_load_json(self, sample_config_json):
        mgr = ConfigManager()
        data = mgr.load(sample_config_json)
        assert data["model"] == "detr"
        assert data["epochs"] == 5
        assert data["early_stopping"]["patience"] == 6

    def test_load_auto_names_from_stem(self, sample_config_yaml):
        mgr = ConfigManager()
        mgr.load(sample_config_yaml)
        # Default name is derived from the file stem ("config")
        retrieved = mgr.get("config")
        assert retrieved["selection_method"] == "cluster"

    def test_load_custom_name(self, sample_config_json):
        mgr = ConfigManager()
        mgr.load(sample_config_json, name="my_cfg")
        assert mgr.get("my_cfg")["model"] == "detr"

    def test_load_missing_file(self, tmp_path):
        mgr = ConfigManager()
        with pytest.raises(FileNotFoundError):
            mgr.load(str(tmp_path / "does_not_exist.yaml"))

    def test_load_unsupported_format(self, tmp_path):
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("key=value")
        mgr = ConfigManager()
        with pytest.raises(ValueError, match="Unsupported config format"):
            mgr.load(str(txt_file))

    def test_load_yml_extension(self, tmp_path):
        """The .yml extension is treated as YAML."""
        yml_file = tmp_path / "settings.yml"
        yml_file.write_text("debug: true\nlevel: 3\n")
        mgr = ConfigManager()
        data = mgr.load(str(yml_file))
        assert data["debug"] is True
        assert data["level"] == 3

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path):
        """An empty YAML file should yield an empty dict, not None."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        mgr = ConfigManager()
        data = mgr.load(str(empty))
        assert data == {}


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    """Tests for ConfigManager.get()."""

    def test_get_existing(self, sample_config_yaml):
        mgr = ConfigManager()
        mgr.load(sample_config_yaml, name="sel")
        data = mgr.get("sel")
        assert "weights" in data

    def test_get_missing_raises_key_error(self):
        mgr = ConfigManager()
        with pytest.raises(KeyError, match="not loaded"):
            mgr.get("nonexistent")


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

class TestSave:
    """Tests for ConfigManager.save()."""

    def test_save_yaml(self, tmp_path):
        mgr = ConfigManager()
        data = {"lr": 1e-4, "epochs": 10, "nested": {"a": 1}}
        out = tmp_path / "out.yaml"
        mgr.save(data, str(out))

        loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert loaded["lr"] == 1e-4
        assert loaded["nested"]["a"] == 1

    def test_save_json(self, tmp_path):
        mgr = ConfigManager()
        data = {"model": "yolo", "batch": 8}
        out = tmp_path / "out.json"
        mgr.save(data, str(out))

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["model"] == "yolo"
        assert loaded["batch"] == 8

    def test_save_creates_parent_dirs(self, tmp_path):
        mgr = ConfigManager()
        data = {"x": 42}
        out = tmp_path / "sub" / "dir" / "deep.yaml"
        mgr.save(data, str(out))
        assert out.exists()
        loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert loaded["x"] == 42

    def test_save_overwrites_existing(self, tmp_path):
        mgr = ConfigManager()
        out = tmp_path / "cfg.json"
        mgr.save({"v": 1}, str(out))
        mgr.save({"v": 2}, str(out))
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["v"] == 2


# ---------------------------------------------------------------------------
# get_nested
# ---------------------------------------------------------------------------

class TestGetNested:
    """Tests for ConfigManager.get_nested()."""

    def test_deep_access(self, sample_config_yaml):
        mgr = ConfigManager()
        mgr.load(sample_config_yaml, name="sel")
        assert mgr.get_nested("sel", "features", "dino_pca_dim") == 32
        assert mgr.get_nested("sel", "weights", "el2n") == 0.30

    def test_top_level_access(self, sample_config_yaml):
        mgr = ConfigManager()
        mgr.load(sample_config_yaml, name="sel")
        assert mgr.get_nested("sel", "target_size") == 20000

    def test_missing_key_returns_default(self, sample_config_yaml):
        mgr = ConfigManager()
        mgr.load(sample_config_yaml, name="sel")
        assert mgr.get_nested("sel", "nonexistent", default="fallback") == "fallback"
        assert mgr.get_nested("sel", "features", "missing_key", default=-1) == -1

    def test_missing_intermediate_key_returns_default(self, sample_config_json):
        mgr = ConfigManager()
        mgr.load(sample_config_json, name="train")
        # "early_stopping" exists but "early_stopping" -> "x" -> "y" does not
        result = mgr.get_nested("train", "early_stopping", "x", "y", default=None)
        assert result is None

    def test_nested_on_non_dict_returns_default(self, tmp_path):
        """If traversal hits a non-dict value, default is returned."""
        cfg_file = tmp_path / "flat.yaml"
        cfg_file.write_text("top: 42\n")
        mgr = ConfigManager()
        mgr.load(str(cfg_file), name="flat")
        assert mgr.get_nested("flat", "top", "sub", default="nope") == "nope"

    def test_get_nested_raises_for_unloaded_config(self):
        mgr = ConfigManager()
        with pytest.raises(KeyError):
            mgr.get_nested("missing_cfg", "key")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Load -> modify -> save -> reload and verify data integrity."""

    def test_yaml_round_trip(self, tmp_path):
        mgr = ConfigManager()
        original = {"model": "detr", "lr": 1e-5, "layers": [1, 2, 3]}
        src = tmp_path / "src.yaml"
        mgr.save(original, str(src))
        data = mgr.load(str(src), name="rt")
        assert data == original

        # Modify
        data["lr"] = 5e-6
        data["layers"].append(4)
        dst = tmp_path / "modified.yaml"
        mgr.save(data, str(dst))

        # Reload
        reloaded = mgr.load(str(dst), name="rt2")
        assert reloaded["lr"] == 5e-6
        assert reloaded["layers"] == [1, 2, 3, 4]
        assert reloaded["model"] == "detr"

    def test_json_round_trip(self, tmp_path):
        mgr = ConfigManager()
        original = {
            "experiment": "ablation",
            "params": {"alpha": 0.1, "beta": 0.9},
        }
        src = tmp_path / "exp.json"
        mgr.save(original, str(src))
        data = mgr.load(str(src), name="exp")
        assert data == original

        # Modify nested value
        data["params"]["alpha"] = 0.2
        dst = tmp_path / "exp_v2.json"
        mgr.save(data, str(dst))

        reloaded = mgr.load(str(dst), name="exp2")
        assert reloaded["params"]["alpha"] == 0.2
        assert reloaded["params"]["beta"] == 0.9

    def test_cross_format_round_trip(self, tmp_path):
        """Save as JSON, reload, save as YAML, reload, and verify."""
        mgr = ConfigManager()
        data = {"key": "value", "num": 123}

        json_path = tmp_path / "data.json"
        mgr.save(data, str(json_path))
        from_json = mgr.load(str(json_path), name="j")

        yaml_path = tmp_path / "data.yaml"
        mgr.save(from_json, str(yaml_path))
        from_yaml = mgr.load(str(yaml_path), name="y")

        assert from_yaml == data
