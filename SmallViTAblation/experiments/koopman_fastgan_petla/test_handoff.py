"""Testy CPU handoffu: rzeczywiste TAR/SHA, procesy wyłącznie zamockowane.

Uruchomienie: python -B -m pytest -p no:cacheprovider test_handoff.py -q
Nie wywołujemy koordynatora, SSH, SCP, GPU ani prawdziwego os.kill.
Małe archiwa powstają w osobnym katalogu tempfile.mkdtemp. Celowo pozostają
po testach: ten zestaw nie usuwa plików i nie używa sprzątania tmp_path.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


HERE = Path(__file__).resolve().parent
REMOTE_RUN = PurePosixPath(
    "/home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla/results/run_1810448"
)
PARENT_PID = 2615518
CHILD_PID = 2616171
CHECKPOINT = b"Testowy checkpoint: tylko bajty, bez torch/pickle."
DIGEST = hashlib.sha256(CHECKPOINT).hexdigest()


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    # Jednoznaczny import lokalnego dane.py, bez zależności od cwd test runnera.
    dane = _load("handoff_test_dane", "dane.py")
    with patch.dict(sys.modules, {"dane": dane}):
        local = _load("handoff_test_local", "hybryda_lokalna.py")
        remote = _load("handoff_test_remote", "handoff_eden.py")
    return SimpleNamespace(local=local, remote=remote)


@pytest.fixture(autouse=True)
def block_external_actions(modules, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Ten test nie może uruchamiać procesów ani wysyłać sygnałów.")

    # Zastępujemy referencje modułów, a nie globalny os/subprocess pytesta.
    monkeypatch.setattr(modules.local, "ssh", forbidden)
    monkeypatch.setattr(modules.local, "subprocess", SimpleNamespace(run=forbidden))
    monkeypatch.setattr(modules.remote, "os", SimpleNamespace(kill=forbidden))


@pytest.fixture(scope="module")
def retained_root():
    return Path(tempfile.mkdtemp(prefix="koopman_handoff_cpu_"))


@pytest.fixture
def archive_factory(retained_root):
    counter = itertools.count()

    def build(*, mode="w", payloads=None, extra_members=()):
        # mkdtemp zapewnia izolację także przy równoległych uruchomieniach.
        directory = Path(tempfile.mkdtemp(prefix=f"case_{next(counter)}_", dir=retained_root))
        files = {
            "config.json": b'{"pretrain_epochs":4}',
            "pretrain/FINAL.ckpt": CHECKPOINT,
            "pretrain/FINAL.pth": b"Testowe wagi, bez deserializacji.",
            "pretrain/STATUS.json": b'{"state":"completed","global_step":400}',
            "probe/manifest.json": b"[]",
        } if payloads is None else dict(payloads)
        archive_path = directory / "handoff.tar"
        entries = []
        with tarfile.open(archive_path, mode) as archive:
            for name, payload in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                entries.append({"file": name, "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()})
            for member, payload in extra_members:
                archive.addfile(member, io.BytesIO(payload) if payload is not None else None)
        manifest = {"state": "ready", "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    "checkpoint_sha256": DIGEST, "files": entries}
        manifest_path = directory / "handoff.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace(archive=archive_path, manifest_path=manifest_path,
                               manifest=manifest, payloads=files, target=directory / "imported")

    return build


def _save_manifest(case):
    case.manifest_path.write_text(json.dumps(case.manifest), encoding="utf-8")


def _extract(modules, case):
    return modules.local.extract_checked(case.archive, case.manifest_path, case.target)


@pytest.mark.parametrize("mode", ["w", "w:gz"])
def test_valid_archive_restores_exact_nested_bytes_and_checkpoint(modules, archive_factory, mode):
    case = archive_factory(mode=mode)
    result = _extract(modules, case)
    assert result["archive_sha256"] == case.manifest["archive_sha256"]
    assert result["checkpoint_sha256"] == DIGEST
    restored = {p.relative_to(case.target).as_posix(): p.read_bytes()
                for p in case.target.rglob("*") if p.is_file()}
    assert restored == case.payloads


def test_corrupted_archive_is_rejected_before_target_creation(modules, archive_factory):
    case = archive_factory()
    with case.archive.open("ab") as stream:
        stream.write(b"Zmiana nawet poza ostatnim blokiem TAR.")
    with pytest.raises(ValueError):
        _extract(modules, case)
    assert not case.target.exists()


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", len(CHECKPOINT) + 1)])
def test_member_digest_and_size_are_checked_even_when_archive_hash_matches(
        modules, archive_factory, field, value):
    case = archive_factory()
    entry = next(row for row in case.manifest["files"] if row["file"] == "pretrain/FINAL.ckpt")
    entry[field] = value
    _save_manifest(case)
    with pytest.raises(ValueError):
        _extract(modules, case)


def test_downloaded_checkpoint_is_bound_to_release_digest(modules, archive_factory):
    case = archive_factory()
    # TAR i wszystkie hashe plików są poprawne, lecz potwierdzenie release
    # wskazuje inny checkpoint. Nie wolno raportować verified dla takiej paczki.
    case.manifest["checkpoint_sha256"] = hashlib.sha256(b"inny checkpoint").hexdigest()
    _save_manifest(case)
    with pytest.raises(ValueError):
        _extract(modules, case)


def test_manifest_cannot_acknowledge_a_checkpoint_absent_from_download(modules, archive_factory):
    case = archive_factory(payloads={"config.json": b"{}"})
    with pytest.raises(ValueError):
        _extract(modules, case)


@pytest.mark.parametrize("name", [
    "../escape.txt", "nested/../../escape.txt", "/absolute.txt",
    "C:/escape.txt", "C:relative.txt", "nested\\escape.txt", "\\\\host\\share\\escape.txt",
])
def test_traversal_and_platform_specific_paths_are_rejected_even_when_manifested(
        modules, archive_factory, name):
    case = archive_factory(payloads={"pretrain/FINAL.ckpt": CHECKPOINT, name: b"outside"})
    with pytest.raises(ValueError):
        _extract(modules, case)
    assert not (case.target.parent / "escape.txt").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE])
def test_links_directories_and_special_members_are_rejected(modules, archive_factory, kind):
    member = tarfile.TarInfo("unexpected_member")
    member.type = kind
    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        member.linkname = "../escape.txt"
    case = archive_factory(extra_members=[(member, None)])
    case.manifest["files"].append({"file": member.name, "bytes": 0,
                                   "sha256": hashlib.sha256(b"").hexdigest()})
    _save_manifest(case)
    with pytest.raises(ValueError):
        _extract(modules, case)
    assert not (case.target / member.name).exists()


def test_unlisted_member_is_rejected(modules, archive_factory):
    case = archive_factory()
    case.manifest["files"] = [r for r in case.manifest["files"] if r["file"] != "probe/manifest.json"]
    _save_manifest(case)
    with pytest.raises(ValueError):
        _extract(modules, case)


def test_manifest_member_missing_from_archive_is_rejected(modules, archive_factory):
    case = archive_factory()
    case.manifest["files"].append({"file": "missing.json", "bytes": 2,
                                   "sha256": hashlib.sha256(b"{}").hexdigest()})
    _save_manifest(case)
    with pytest.raises(ValueError):
        _extract(modules, case)


def test_duplicate_manifest_entry_is_rejected(modules, archive_factory):
    case = archive_factory()
    case.manifest["files"].append(dict(case.manifest["files"][0]))
    _save_manifest(case)
    with pytest.raises(ValueError):
        _extract(modules, case)


def test_duplicate_tar_member_cannot_replace_an_already_verified_file(modules, archive_factory):
    duplicate = tarfile.TarInfo("pretrain/FINAL.ckpt")
    duplicate.size = len(CHECKPOINT)
    case = archive_factory(extra_members=[(duplicate, CHECKPOINT)])
    with pytest.raises(ValueError):
        _extract(modules, case)


def test_existing_target_is_preserved(modules, archive_factory):
    case = archive_factory()
    case.target.mkdir()
    sentinel = case.target / "user_work.txt"
    sentinel.write_bytes(b"Nie nadpisuj cudzej pracy.")
    with pytest.raises(FileExistsError):
        _extract(modules, case)
    assert sentinel.read_bytes() == b"Nie nadpisuj cudzej pracy."
    assert not (case.target / "pretrain").exists()


class _ProcPath:
    """Minimalny odczyt /proc w pamięci; nigdy nie dotyka prawdziwych PID-ów."""

    def __init__(self, path, nodes):
        self.path, self.nodes = PurePosixPath(path), nodes

    def joinpath(self, name):
        return _ProcPath(self.path / name, self.nodes)

    def exists(self):
        return str(self.path) in self.nodes

    def read_bytes(self):
        return self.nodes[str(self.path)]

    def read_text(self):
        return self.read_bytes().decode("utf-8")


@pytest.fixture
def release_setup(modules, monkeypatch):
    def setup(*, done=True, digest=DIGEST, parent_state="T", child_state=None,
              present=True, argv=None, kill_error=None):
        if argv is None:
            argv = [b"python", b"/code/petla.py", b"--out", str(REMOTE_RUN).encode()]
        nodes = {}
        if present:
            nodes[f"/proc/{PARENT_PID}"] = b""
            nodes[f"/proc/{PARENT_PID}/cmdline"] = b"\0".join(argv) + b"\0"
            nodes[f"/proc/{PARENT_PID}/stat"] = f"{PARENT_PID} (python) {parent_state} 1 1 1".encode()
        if child_state is not None:
            nodes[f"/proc/{CHILD_PID}/stat"] = f"{CHILD_PID} (python) {child_state} 1 1 1".encode()
        writes, events = [], []

        def write(path, value):
            writes.append((path, copy.deepcopy(value)))
            events.append(("write", value.get("state")))

        def kill(pid, sig):
            events.append(("signal", pid, sig))
            if kill_error is not None:
                raise kill_error

        killed = Mock(side_effect=kill)
        monkeypatch.setattr(modules.remote, "RUN", REMOTE_RUN)
        monkeypatch.setattr(modules.remote, "Path", lambda p: _ProcPath(p, nodes))
        monkeypatch.setattr(modules.remote, "completed", Mock(return_value=done))
        monkeypatch.setattr(modules.remote, "sha256", Mock(return_value=digest))
        monkeypatch.setattr(modules.remote, "write_json", write)
        monkeypatch.setattr(modules.remote, "os", SimpleNamespace(kill=killed))
        # SIGKILL jest kontraktem hosta Linux; nie istnieje w signal na Windows.
        monkeypatch.setattr(modules.remote, "signal", SimpleNamespace(SIGKILL=9))
        return SimpleNamespace(kill=killed, writes=writes, events=events)

    return setup


@pytest.mark.parametrize("options", [
    {"done": False}, {"digest": "0" * 64}, {"present": False},
    {"parent_state": "R"}, {"parent_state": "S"}, {"parent_state": "Z"},
    {"argv": [b"python", b"/code/petla.py", b"--out", b"/other/run"]},
    {"argv": [b"python", b"/code/trening.py", b"--out", str(REMOTE_RUN).encode()]},
    {"argv": [b"python", b"/code/petla.py"]},
])
def test_release_refuses_wrong_completion_hash_pid_run_or_process_state(modules, release_setup, options):
    state = release_setup(**options)
    with pytest.raises(ValueError):
        modules.remote.release(DIGEST)
    state.kill.assert_not_called()
    assert not any(record.get("state") == "released_to_local" for _, record in state.writes)


@pytest.mark.parametrize("child_state", ["R", "S", "D", "T", "t", "I"])
def test_stopped_parent_does_not_authorize_release_while_training_child_is_alive(
        modules, release_setup, child_state):
    state = release_setup(parent_state="T", child_state=child_state)
    with pytest.raises(ValueError):
        modules.remote.release(DIGEST)
    state.kill.assert_not_called()
    assert not any(record.get("state") == "released_to_local" for _, record in state.writes)


@pytest.mark.parametrize("parent_state,child_state", [("T", None), ("T", "Z"), ("t", "Z")])
def test_release_signals_only_the_verified_parent_of_run_1810448(
        modules, release_setup, parent_state, child_state):
    state = release_setup(parent_state=parent_state, child_state=child_state)
    modules.remote.release(DIGEST)
    state.kill.assert_called_once_with(PARENT_PID, 9)
    receipts = [record for path, record in state.writes if path == REMOTE_RUN / "HANDOFF.json"]
    assert receipts[-1]["state"] == "released_to_local"
    assert receipts[-1]["checkpoint_sha256"] == DIGEST


def test_release_cannot_publish_success_when_signalling_fails(modules, release_setup):
    state = release_setup(kill_error=PermissionError("Odmowa sygnału: symulacja."))
    with pytest.raises(PermissionError):
        modules.remote.release(DIGEST)
    state.kill.assert_called_once_with(PARENT_PID, 9)
    assert not any(record.get("state") in ("released_to_local", "handed_off_to_local")
                   for _, record in state.writes), "Fałszywy receipt uruchomi lokalny fallback mimo błędu kill."
