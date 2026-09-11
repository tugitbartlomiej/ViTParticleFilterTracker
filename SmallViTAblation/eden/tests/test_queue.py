#!/usr/bin/env python3
"""
Testy kolejki i runnera (Agent 2) — bez GPU, na falszywym poleceniu treningu.

Pokrywa wymagane scenariusze:
  T2: 8 zadan, 2 workerow — kazde zadanie wykonane DOKLADNIE raz, stany done.
  T3: ubicie workera w polowie — po requeue-stale zadanie wraca do pending
      i sie dokonczy (marker: 2 uruchomienia, 1 done).
  T4: zadanie celowo failing — status failed z logiem, reszta idzie dalej.

Uruchomienie:  py -3.11 eden/tests/test_queue.py
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDEN = HERE.parent
REPOSITORY = EDEN.parents[1]
sys.path.insert(0, str(REPOSITORY))
from SmallViTAblation.eden import queue as qmod  # noqa: E402

RUNNER = EDEN / "runner.py"
FIXTURE = HERE / "fixture_task.py"
PROTOCOL = EDEN.parent / "configs" / "protocol.yaml"

FAST = ["--poll", "0.2", "--idle-exit-rounds", "3", "--heartbeat", "0.2",
        "--stale-timeout", "1.0", "--lock-ttl", "5"]


def make_queue_file(path: Path, n: int, prefix: str = "t") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            f.write(json.dumps({"run_id": f"{prefix}{i}", "seed": 42,
                                "budget": 10, "arch": "test"}) + "\n")


def start_worker(qdir: Path, data: Path, worker: str, marker_dir: Path,
                 env_extra: dict | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.Popen(
        [sys.executable, str(RUNNER),
         "--queue", str(qdir), "--data", str(data), "--worker", worker,
         "--protocol", str(PROTOCOL), "--out", str(data / "results"),
         "--cmd-template", "{python} {fixture} {run_id} {marker_dir}",
         "--template-var", f"fixture={FIXTURE}",
         "--template-var", f"marker_dir={marker_dir}",
         *FAST],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        **({"start_new_session": True} if os.name != "nt" else {}))


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def marker_lines(marker_dir: Path, run_id: str) -> int:
    p = marker_dir / run_id
    return len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0


def wait_for(cond, timeout_s: float, what: str) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if cond():
            return
        time.sleep(0.2)
    raise AssertionError(f"timeout ({timeout_s}s) czekajac na: {what}")


def wait_workers(procs: list[subprocess.Popen], timeout_s: float = 60) -> None:
    for p in procs:
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_tree(p)
            raise AssertionError(f"worker {p.pid} nie zakonczyl sie w {timeout_s}s")


def test_2_two_workers_eight_tasks(tmp_path: Path) -> None:
    print("T2: 8 zadan, 2 workerow, dokładnie raz...")
    root = tmp_path / "t2"
    root.mkdir(parents=True)
    qf = root / "queue.jsonl"
    make_queue_file(qf, 8)
    qdir = root / "qdir"
    marker = root / "markers"
    res = qmod.init_queue(qdir, qf)
    assert res == {"total": 8, "added": 8, "skipped_existing": 0}, res

    env = {"TASK_SLEEP": "0.05"}
    procs = [start_worker(qdir, root, str(w), marker, env) for w in (0, 1)]
    wait_workers(procs)

    st = qmod.read_status(qdir)
    assert st["counts"]["done"] == 8, st
    assert st["counts"]["pending"] == st["counts"]["running"] == st["counts"]["failed"] == 0
    for i in range(1, 9):
        n = marker_lines(marker, f"t{i}")
        assert n == 1, f"t{i} wykonane {n} razy (oczekiwano 1)"
    # wznowienie: ponowny init nic nie dubluje
    res2 = qmod.init_queue(qdir, qf)
    assert res2["added"] == 0 and res2["skipped_existing"] == 8, res2
    print("T2: PASS")


def test_3_kill_worker_requeue(tmp_path: Path) -> None:
    print("T3: ubicie workera w polowie + requeue-stale...")
    root = tmp_path / "t3"
    root.mkdir(parents=True)
    qf = root / "queue.jsonl"
    make_queue_file(qf, 3)
    qdir = root / "qdir"
    marker = root / "markers"
    qmod.init_queue(qdir, qf)

    # t2 spi 30 s — w tym czasie ubijamy workera
    proc = start_worker(qdir, root, "0", marker,
                        {"TASK_SLEEP": "0.05", "LONG_RUN_IDS": "t2",
                         "LONG_SLEEP": "30"})
    wait_for(lambda: any(i["run_id"] == "t2"
                         for i in qmod.read_status(qdir)["detail"]["running"]),
             30, "t2 w stanie running")
    wait_for(lambda: marker_lines(marker, "t2") >= 1, 30,
             "fixture t2 wystartowal")
    kill_tree(proc)
    proc.wait(timeout=10)

    # heartbeat umarl razem z workerem — po >1 s requeue-stale przywraca
    time.sleep(1.5)
    requeued = qmod.requeue_stale(qdir, timeout_s=1.0)
    assert "t2" in requeued, requeued
    st = qmod.read_status(qdir)
    assert st["counts"]["running"] == 0 and st["counts"]["pending"] >= 1, st

    # nowy worker dokonczy kolejke (t2 juz bez dlugiego snu)
    proc2 = start_worker(qdir, root, "0", marker, {"TASK_SLEEP": "0.05"})
    wait_workers([proc2])
    st = qmod.read_status(qdir)
    assert st["counts"]["done"] == 3, st
    assert marker_lines(marker, "t2") == 2, "t2 powinno wystartowac 2 razy"
    print("T3: PASS")


def test_4_failing_task(tmp_path: Path) -> None:
    print("T4: zadanie celowo failing, reszta idzie dalej...")
    root = tmp_path / "t4"
    root.mkdir(parents=True)
    qf = root / "queue.jsonl"
    make_queue_file(qf, 4)
    qdir = root / "qdir"
    marker = root / "markers"
    qmod.init_queue(qdir, qf)

    proc = start_worker(qdir, root, "0", marker,
                        {"TASK_SLEEP": "0.05", "FAIL_RUN_IDS": "t3"})
    wait_workers([proc])
    st = qmod.read_status(qdir)
    assert st["counts"]["done"] == 3, st
    assert st["counts"]["failed"] == 1, st
    failed = st["detail"]["failed"][0]
    assert failed["run_id"] == "t3", failed
    rec = json.loads((qdir / "state" / "failed" / "t3.json").read_text(encoding="utf-8"))
    assert rec.get("log") and Path(rec["log"]).is_file(), "brak logu zadan failed"
    assert "celowa awaria" in rec.get("error", "") or "exit" in rec.get("error", "").lower()
    # reszta poszla DALEJ: t4 wystartowalo po awarii t3
    assert marker_lines(marker, "t4") == 1
    print("T4: PASS")


def test_5_stale_worker_cannot_finalize_new_attempt(tmp_path: Path) -> None:
    print("T5: token dzierzawy blokuje starego workera...")
    root = tmp_path / "t5"
    root.mkdir(parents=True)
    qf = root / "queue.jsonl"
    make_queue_file(qf, 1)
    qdir = root / "qdir"
    qmod.init_queue(qdir, qf)
    old = qmod.claim_task(qdir, "old")
    assert old and old.get("_queue_lease_token")
    requeued = qmod.requeue_stale(qdir, timeout_s=-1, max_attempts=3)
    assert requeued == ["t1"]
    new = qmod.claim_task(qdir, "new")
    assert new and new["_queue_lease_token"] != old["_queue_lease_token"]
    assert not qmod.mark_done(qdir, "t1", "old", lease_token=old["_queue_lease_token"])
    assert qmod.mark_done(qdir, "t1", "new", lease_token=new["_queue_lease_token"])
    print("T5: PASS")


def test_6_queue_manifest_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "t6"
    root.mkdir(parents=True)
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    make_queue_file(first, 1, prefix="a")
    make_queue_file(second, 1, prefix="b")
    qdir = root / "qdir"
    qmod.init_queue(qdir, first)
    try:
        qmod.init_queue(qdir, second)
    except ValueError as exc:
        assert "different queue manifest" in str(exc)
    else:
        raise AssertionError("changed queue manifest was accepted in an existing qdir")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="qtest_"))
    failures = 0
    try:
        for fn in (test_2_two_workers_eight_tasks,
                   test_3_kill_worker_requeue,
                   test_4_failing_task,
                   test_5_stale_worker_cannot_finalize_new_attempt,
                   test_6_queue_manifest_is_immutable):
            try:
                fn(tmp)
            except Exception as e:
                failures += 1
                print(f"{fn.__name__}: FAIL — {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("WYNIK:", "WSZYSTKIE TESTY PASS" if failures == 0 else f"{failures} test(y) FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
