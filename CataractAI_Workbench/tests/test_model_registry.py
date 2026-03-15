"""Tests for CataractAI_Workbench.app.core.model_registry.ModelRegistry."""

import json

import pytest

from CataractAI_Workbench.app.core.model_registry import ModelRegistry


# ========================================================================
# register
# ========================================================================

class TestRegister:
    """Tests for ModelRegistry.register."""

    def test_register_stores_model(self, tmp_path):
        dummy = tmp_path / "model.pth"
        dummy.write_bytes(b"\x00" * 1024)
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        reg.register("my_model", str(dummy), model_type="detr")

        stored = reg.get("my_model")
        assert stored is not None
        assert stored["path"] == str(dummy)
        assert stored["type"] == "detr"
        assert stored["size_mb"] == pytest.approx(1024 / (1024 * 1024))
        assert "registered_at" in stored

    def test_register_with_extra_metadata(self, tmp_path):
        dummy = tmp_path / "model.pth"
        dummy.write_bytes(b"\x00" * 512)
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        reg.register("extra", str(dummy), model_type="yolo", epochs=100, mAP=0.85)

        stored = reg.get("extra")
        assert stored["epochs"] == 100
        assert stored["mAP"] == 0.85

    def test_register_nonexistent_file_gives_zero_size(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        reg.register("ghost", str(tmp_path / "does_not_exist.pth"))

        stored = reg.get("ghost")
        assert stored is not None
        assert stored["size_mb"] == 0

    def test_register_saves_to_disk(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        dummy = tmp_path / "model.pth"
        dummy.write_bytes(b"\x00" * 128)
        reg = ModelRegistry(str(registry_path))
        reg.register("disk_model", str(dummy))

        assert registry_path.exists()
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        assert "disk_model" in data

    def test_register_creates_parent_directories(self, tmp_path):
        nested_path = tmp_path / "a" / "b" / "c" / "registry.json"
        dummy = tmp_path / "model.pth"
        dummy.write_bytes(b"\x00")
        reg = ModelRegistry(str(nested_path))
        reg.register("nested", str(dummy))

        assert nested_path.exists()

    def test_duplicate_name_overwrites(self, tmp_path):
        dummy1 = tmp_path / "v1.pth"
        dummy1.write_bytes(b"\x00" * 100)
        dummy2 = tmp_path / "v2.pth"
        dummy2.write_bytes(b"\x00" * 200)

        reg = ModelRegistry(str(tmp_path / "registry.json"))
        reg.register("same_name", str(dummy1), model_type="detr")
        reg.register("same_name", str(dummy2), model_type="yolo")

        stored = reg.get("same_name")
        assert stored["path"] == str(dummy2)
        assert stored["type"] == "yolo"

    def test_default_model_type_is_detr(self, tmp_path):
        dummy = tmp_path / "model.pth"
        dummy.write_bytes(b"\x00")
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        reg.register("default_type", str(dummy))

        assert reg.get("default_type")["type"] == "detr"


# ========================================================================
# get
# ========================================================================

class TestGet:
    """Tests for ModelRegistry.get."""

    def test_get_existing(self, tmp_path):
        dummy = tmp_path / "model.pth"
        dummy.write_bytes(b"\x00")
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        reg.register("exists", str(dummy))

        assert reg.get("exists") is not None

    def test_get_nonexistent_returns_none(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        assert reg.get("nonexistent") is None


# ========================================================================
# list_models
# ========================================================================

class TestListModels:
    """Tests for ModelRegistry.list_models."""

    def test_list_all_models(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        d1 = tmp_path / "a.pth"
        d1.write_bytes(b"\x00")
        d2 = tmp_path / "b.pth"
        d2.write_bytes(b"\x00")
        reg.register("model_a", str(d1), model_type="detr")
        reg.register("model_b", str(d2), model_type="yolo")

        models = reg.list_models()
        assert len(models) == 2
        names = {m["name"] for m in models}
        assert names == {"model_a", "model_b"}

    def test_list_filtered_by_type(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        d1 = tmp_path / "a.pth"
        d1.write_bytes(b"\x00")
        d2 = tmp_path / "b.pth"
        d2.write_bytes(b"\x00")
        d3 = tmp_path / "c.pth"
        d3.write_bytes(b"\x00")
        reg.register("detr1", str(d1), model_type="detr")
        reg.register("yolo1", str(d2), model_type="yolo")
        reg.register("detr2", str(d3), model_type="detr")

        detr_models = reg.list_models(model_type="detr")
        assert len(detr_models) == 2
        assert all(m["type"] == "detr" for m in detr_models)

        yolo_models = reg.list_models(model_type="yolo")
        assert len(yolo_models) == 1
        assert yolo_models[0]["name"] == "yolo1"

    def test_list_filtered_no_matches(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        d1 = tmp_path / "a.pth"
        d1.write_bytes(b"\x00")
        reg.register("detr1", str(d1), model_type="detr")

        assert reg.list_models(model_type="yolo") == []

    def test_list_empty_registry(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        assert reg.list_models() == []

    def test_list_includes_name_field(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        d = tmp_path / "x.pth"
        d.write_bytes(b"\x00")
        reg.register("named_model", str(d))

        models = reg.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "named_model"
        assert "path" in models[0]
        assert "type" in models[0]


# ========================================================================
# scan_directory
# ========================================================================

class TestScanDirectory:
    """Tests for ModelRegistry.scan_directory."""

    def test_scan_finds_pth_files(self, tmp_path):
        (tmp_path / "a.pth").write_bytes(b"\x00" * 2048)
        (tmp_path / "b.pth").write_bytes(b"\x00" * 4096)
        (tmp_path / "c.txt").write_text("not a checkpoint")

        reg = ModelRegistry()
        found = reg.scan_directory(str(tmp_path))
        assert len(found) == 2
        names = {f["name"] for f in found}
        assert names == {"a", "b"}

    def test_scan_recursive(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.pth").write_bytes(b"\x00")
        (sub / "nested.pth").write_bytes(b"\x00")

        reg = ModelRegistry()
        found = reg.scan_directory(str(tmp_path))
        assert len(found) == 2
        names = {f["name"] for f in found}
        assert names == {"top", "nested"}

    def test_scan_empty_directory(self, tmp_path):
        reg = ModelRegistry()
        found = reg.scan_directory(str(tmp_path))
        assert found == []

    def test_scan_returns_correct_fields(self, tmp_path):
        f = tmp_path / "model.pth"
        f.write_bytes(b"\x00" * 1024)

        reg = ModelRegistry()
        found = reg.scan_directory(str(tmp_path))
        assert len(found) == 1
        entry = found[0]
        assert entry["name"] == "model"
        assert entry["path"] == str(f)
        assert entry["type"] == "detr"  # default
        assert entry["size_mb"] == pytest.approx(1024 / (1024 * 1024))
        assert "modified" in entry

    def test_scan_custom_model_type(self, tmp_path):
        (tmp_path / "yolo_best.pth").write_bytes(b"\x00")

        reg = ModelRegistry()
        found = reg.scan_directory(str(tmp_path), model_type="yolo")
        assert len(found) == 1
        assert found[0]["type"] == "yolo"

    def test_scan_ignores_non_pth_files(self, tmp_path):
        (tmp_path / "weights.pt").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.onnx").write_bytes(b"\x00")

        reg = ModelRegistry()
        found = reg.scan_directory(str(tmp_path))
        assert found == []


# ========================================================================
# remove
# ========================================================================

class TestRemove:
    """Tests for ModelRegistry.remove."""

    def test_remove_existing(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        d = tmp_path / "model.pth"
        d.write_bytes(b"\x00")
        reg.register("to_remove", str(d))
        assert reg.get("to_remove") is not None

        reg.remove("to_remove")
        assert reg.get("to_remove") is None

    def test_remove_nonexistent_does_not_raise(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        # Should not raise any exception
        reg.remove("never_existed")

    def test_remove_persists_to_disk(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        d = tmp_path / "model.pth"
        d.write_bytes(b"\x00")
        reg = ModelRegistry(str(registry_path))
        reg.register("ephemeral", str(d))
        reg.remove("ephemeral")

        data = json.loads(registry_path.read_text(encoding="utf-8"))
        assert "ephemeral" not in data

    def test_remove_only_target(self, tmp_path):
        reg = ModelRegistry(str(tmp_path / "registry.json"))
        d1 = tmp_path / "a.pth"
        d1.write_bytes(b"\x00")
        d2 = tmp_path / "b.pth"
        d2.write_bytes(b"\x00")
        reg.register("keep", str(d1))
        reg.register("delete", str(d2))

        reg.remove("delete")
        assert reg.get("keep") is not None
        assert reg.get("delete") is None


# ========================================================================
# Persistence  (register -> reload -> verify)
# ========================================================================

class TestPersistence:
    """Verify that registry data survives a reload from disk."""

    def test_reload_from_disk(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        d = tmp_path / "model.pth"
        d.write_bytes(b"\x00" * 512)

        # First instance: register
        reg1 = ModelRegistry(str(registry_path))
        reg1.register("persistent", str(d), model_type="yolo", note="hello")

        # Second instance: load from same file
        reg2 = ModelRegistry(str(registry_path))
        stored = reg2.get("persistent")
        assert stored is not None
        assert stored["type"] == "yolo"
        assert stored["note"] == "hello"

    def test_multiple_models_persist(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        d1 = tmp_path / "a.pth"
        d1.write_bytes(b"\x00")
        d2 = tmp_path / "b.pth"
        d2.write_bytes(b"\x00")

        reg1 = ModelRegistry(str(registry_path))
        reg1.register("alpha", str(d1), model_type="detr")
        reg1.register("beta", str(d2), model_type="yolo")

        reg2 = ModelRegistry(str(registry_path))
        assert len(reg2.list_models()) == 2
        assert reg2.get("alpha")["type"] == "detr"
        assert reg2.get("beta")["type"] == "yolo"

    def test_remove_persists_through_reload(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        d = tmp_path / "model.pth"
        d.write_bytes(b"\x00")

        reg1 = ModelRegistry(str(registry_path))
        reg1.register("temp", str(d))
        reg1.remove("temp")

        reg2 = ModelRegistry(str(registry_path))
        assert reg2.get("temp") is None
        assert reg2.list_models() == []


# ========================================================================
# Empty / no-path registry
# ========================================================================

class TestNoPathRegistry:
    """Tests for an in-memory registry with no file backing."""

    def test_in_memory_register_and_get(self, tmp_path):
        d = tmp_path / "model.pth"
        d.write_bytes(b"\x00")
        reg = ModelRegistry()  # No registry_path
        reg.register("mem_model", str(d))
        assert reg.get("mem_model") is not None

    def test_in_memory_remove(self, tmp_path):
        d = tmp_path / "model.pth"
        d.write_bytes(b"\x00")
        reg = ModelRegistry()
        reg.register("mem_model", str(d))
        reg.remove("mem_model")
        assert reg.get("mem_model") is None

    def test_empty_registry_list(self):
        reg = ModelRegistry()
        assert reg.list_models() == []
