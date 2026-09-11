#!/usr/bin/env python3
"""
Kolejka biegow na plikach (Agent 2, eden/). Bez zaleznosci zewnetrznych -
blokada przez os.open(O_CREAT|O_EXCL) z retry i usuwaniem starej blokady.

Uklad katalogu kolejki (qdir):
    qdir/queue.jsonl                 kopia wejsciowej listy zadan (nietykalna)
    qdir/LOCK                        plik blokady (krotkotrwaly)
    qdir/state/pending/<run_id>.json
    qdir/state/running/<run_id>.json (mtime = ostatni heartbeat)
    qdir/state/done/<run_id>.json
    qdir/state/failed/<run_id>.json
    qdir/logs/<run_id>.log           log zadan failed (sciezke podaje worker)

Stany: pending -> running -> done | failed; running bez heartbeatu dluzej niz
timeout wraca do pending (requeue-stale). Wznawialnosc: init pomija zadania,
ktore juz maja stan; ponowne uruchomienie workerow podejmuje kolejke.

CLI:
    py -3.11 queue.py init <qdir> --queue-file queue.jsonl
    py -3.11 queue.py claim <qdir> --worker 0        # JSON zadania na stdout
    py -3.11 queue.py heartbeat <qdir> --run-id ID
    py -3.11 queue.py done <qdir> --run-id ID
    py -3.11 queue.py fail <qdir> --run-id ID [--log PLIK] [--message MSG]
    py -3.11 queue.py requeue-stale <qdir> --timeout 180
    py -3.11 queue.py status <qdir> [--json]

Kody wyjscia claim: 0 = zadanie pobrane, 3 = brak zadan.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

STATES = ("pending", "running", "done", "failed")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.=-]+$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# blokada pliku (stdlib, O_CREAT|O_EXCL + retry)
# --------------------------------------------------------------------------- #
class FileLock:
    def __init__(self, qdir: Path, ttl_s: float = 30.0, timeout_s: float = 120.0,
                 retry_s: float = 0.05):
        self.path = qdir / "LOCK"
        self.ttl_s = ttl_s
        self.timeout_s = timeout_s
        self.retry_s = retry_s
        self.fd: int | None = None

    def __enter__(self) -> "FileLock":
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{os.getpid()} {now_iso()}\n".encode())
                return self
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue  # ktos wlasnie zwolnil - probuj ponownie
                if age > self.ttl_s:
                    # stara blokada po martwym procesie
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Nie udalo sie zdobyc blokady {self.path}")
                time.sleep(self.retry_s)

    def __exit__(self, *exc) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# operacje na stanach (wszystkie pod blokada)
# --------------------------------------------------------------------------- #
def _state_file(qdir: Path, state: str, run_id: str) -> Path:
    return qdir / "state" / state / f"{run_id}.json"


def _find_state(qdir: Path, run_id: str) -> tuple[str, Path] | None:
    for st in STATES:
        p = _state_file(qdir, st, run_id)
        if p.exists():
            return st, p
    return None


def _write_state(path: Path, rec: dict) -> None:
    """Atomowy zapis: tmp + os.replace w tym samym katalogu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _move_state(qdir: Path, run_id: str, src: str, dst: str, updates: dict) -> dict:
    src_p = _state_file(qdir, src, run_id)
    rec = json.loads(src_p.read_text(encoding="utf-8"))
    rec.update(updates)
    _write_state(_state_file(qdir, dst, run_id), rec)
    src_p.unlink()
    return rec


def init_queue(qdir: Path, queue_file: Path, lock_ttl: float = 30.0) -> dict:
    """Inicjalizacja/wznowienie: dodaje do pending tylko zadania bez stanu."""
    qdir = Path(qdir)
    tasks = []
    with open(queue_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    ids = [t["run_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("queue.jsonl ma zduplikowane run_id")
    unsafe = [run_id for run_id in ids
              if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id)]
    if unsafe:
        raise ValueError(f"queue.jsonl ma niebezpieczne run_id: {unsafe[:3]}")

    for st in STATES:
        (qdir / "state" / st).mkdir(parents=True, exist_ok=True)
    (qdir / "logs").mkdir(parents=True, exist_ok=True)

    with FileLock(qdir, ttl_s=lock_ttl):
        frozen_queue = qdir / "queue.jsonl"
        if not frozen_queue.exists():
            shutil.copyfile(queue_file, frozen_queue)
        elif frozen_queue.read_bytes() != Path(queue_file).read_bytes():
            raise ValueError(
                "qdir was initialized from a different queue manifest; use a new qdir")
        added, skipped = 0, 0
        for t in tasks:
            if _find_state(qdir, t["run_id"]) is None:
                _write_state(_state_file(qdir, "pending", t["run_id"]),
                             {"state": "pending", "created_at": now_iso(), "task": t})
                added += 1
            else:
                skipped += 1
    return {"total": len(tasks), "added": added, "skipped_existing": skipped}


def claim_task(qdir: Path, worker: str, lock_ttl: float = 30.0) -> dict | None:
    """Atomowe pobranie zadania: pod blokada pending -> running.

    Zwraca rekord zadania (wiersz queue.jsonl) albo None, gdy pusto."""
    qdir = Path(qdir)
    with FileLock(qdir, ttl_s=lock_ttl):
        pending_dir = qdir / "state" / "pending"
        candidates = sorted(pending_dir.glob("*.json"))
        for cand in candidates:
            run_id = cand.stem
            if _state_file(qdir, "running", run_id).exists():
                # slad po killu w srodku poprzedniego claim: zadanie juz running
                cand.unlink()
                continue
            rec = json.loads(cand.read_text(encoding="utf-8"))
            attempt_id = uuid.uuid4().hex
            lease_token = uuid.uuid4().hex
            rec.update({
                "state": "running",
                "worker": str(worker),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "claimed_at": now_iso(),
                "runs": int(rec.get("runs", 0)) + 1,
                "attempt_id": attempt_id,
                "lease_token": lease_token,
            })
            dst = _state_file(qdir, "running", run_id)
            _write_state(dst, rec)
            try:
                cand.unlink()
            except FileNotFoundError:
                pass
            task = dict(rec["task"])
            task["_queue_attempt_id"] = attempt_id
            task["_queue_lease_token"] = lease_token
            return task
    return None


def heartbeat(qdir: Path, run_id: str, worker: str | None = None,
              lock_ttl: float = 30.0, lease_token: str | None = None) -> bool:
    """Dotkniecie pliku stanu running (mtime = zywy proces)."""
    qdir = Path(qdir)
    with FileLock(qdir, ttl_s=lock_ttl):
        p = _state_file(qdir, "running", run_id)
        if not p.exists():
            return False
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("lease_token") and lease_token != rec["lease_token"]:
            return False
        rec["heartbeat_at"] = now_iso()
        if worker is not None:
            rec["worker"] = str(worker)
        _write_state(p, rec)  # os.replace odswieza mtime
        return True


def mark_done(qdir: Path, run_id: str, worker: str | None = None,
              lock_ttl: float = 30.0, lease_token: str | None = None) -> bool:
    qdir = Path(qdir)
    with FileLock(qdir, ttl_s=lock_ttl):
        found = _find_state(qdir, run_id)
        if found is None or found[0] != "running":
            return False
        rec = json.loads(found[1].read_text(encoding="utf-8"))
        if rec.get("lease_token") and lease_token != rec["lease_token"]:
            return False
        updates = {"state": "done", "finished_at": now_iso()}
        if worker is not None:
            updates["worker"] = str(worker)
        _move_state(qdir, run_id, "running", "done", updates)
        return True


def mark_fail(qdir: Path, run_id: str, message: str = "", log_path: str | None = None,
              worker: str | None = None, lock_ttl: float = 30.0,
              lease_token: str | None = None) -> bool:
    qdir = Path(qdir)
    with FileLock(qdir, ttl_s=lock_ttl):
        found = _find_state(qdir, run_id)
        if found is None or found[0] != "running":
            return False
        rec = json.loads(found[1].read_text(encoding="utf-8"))
        if rec.get("lease_token") and lease_token != rec["lease_token"]:
            return False
        updates = {"state": "failed", "finished_at": now_iso(),
                   "error": message[:4000]}
        if log_path:
            updates["log"] = str(log_path)
        if worker is not None:
            updates["worker"] = str(worker)
        _move_state(qdir, run_id, "running", "failed", updates)
        return True


def requeue_stale(qdir: Path, timeout_s: float, lock_ttl: float = 30.0,
                  max_attempts: int = 3) -> list[str]:
    """Running bez heartbeatu > timeout_s -> pending (np. ubite limitem SLURM)."""
    qdir = Path(qdir)
    requeued = []
    with FileLock(qdir, ttl_s=lock_ttl):
        running_dir = qdir / "state" / "running"
        now = time.time()
        for p in sorted(running_dir.glob("*.json")):
            if now - p.stat().st_mtime > timeout_s:
                run_id = p.stem
                rec = json.loads(p.read_text(encoding="utf-8"))
                if int(rec.get("runs", 0)) >= max_attempts:
                    rec.update({"state": "failed", "finished_at": now_iso(),
                                "error": f"stale after {max_attempts} attempts"})
                    _write_state(_state_file(qdir, "failed", run_id), rec)
                    p.unlink()
                    continue
                rec.update({"state": "pending", "requeued_at": now_iso(),
                            "requeued_reason": f"stale > {timeout_s}s"})
                _write_state(_state_file(qdir, "pending", run_id), rec)
                p.unlink()
                requeued.append(run_id)
    return requeued


def read_status(qdir: Path) -> dict:
    qdir = Path(qdir)
    status: dict = {st: [] for st in STATES}
    for st in STATES:
        d = qdir / "state" / st
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                rec = json.loads(p.read_text(encoding="utf-8"))
                item = {"run_id": p.stem, "worker": rec.get("worker")}
                if st == "running":
                    item["heartbeat_age_s"] = round(time.time() - p.stat().st_mtime, 1)
                if st == "failed":
                    item["error"] = rec.get("error", "")[:200]
                status[st].append(item)
    return {"counts": {st: len(status[st]) for st in STATES}, "detail": status}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("qdir", type=Path)
    p.add_argument("--queue-file", required=True, type=Path)
    p.add_argument("--lock-ttl", type=float, default=30.0)

    p = sub.add_parser("claim")
    p.add_argument("qdir", type=Path)
    p.add_argument("--worker", required=True)
    p.add_argument("--lock-ttl", type=float, default=30.0)

    for name in ("heartbeat", "done", "fail"):
        p = sub.add_parser(name)
        p.add_argument("qdir", type=Path)
        p.add_argument("--run-id", required=True)
        p.add_argument("--worker", default=None)
        p.add_argument("--lock-ttl", type=float, default=30.0)
        p.add_argument("--lease-token", default=None)
        if name == "fail":
            p.add_argument("--log", default=None)
            p.add_argument("--message", default="")

    p = sub.add_parser("requeue-stale")
    p.add_argument("qdir", type=Path)
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--lock-ttl", type=float, default=30.0)
    p.add_argument("--max-attempts", type=int, default=3)

    p = sub.add_parser("status")
    p.add_argument("qdir", type=Path)
    p.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "init":
        res = init_queue(args.qdir, args.queue_file, args.lock_ttl)
        print(json.dumps(res))
        return 0
    if args.cmd == "claim":
        task = claim_task(args.qdir, args.worker, args.lock_ttl)
        if task is None:
            return 3
        print(json.dumps(task, ensure_ascii=False))
        return 0
    if args.cmd == "heartbeat":
        ok = heartbeat(args.qdir, args.run_id, args.worker, args.lock_ttl,
                       args.lease_token)
        return 0 if ok else 1
    if args.cmd == "done":
        ok = mark_done(args.qdir, args.run_id, args.worker, args.lock_ttl,
                       args.lease_token)
        return 0 if ok else 1
    if args.cmd == "fail":
        ok = mark_fail(args.qdir, args.run_id, args.message, args.log,
                       args.worker, args.lock_ttl, args.lease_token)
        return 0 if ok else 1
    if args.cmd == "requeue-stale":
        requeued = requeue_stale(args.qdir, args.timeout, args.lock_ttl,
                                 args.max_attempts)
        print(json.dumps({"requeued": requeued}))
        return 0
    if args.cmd == "status":
        st = read_status(args.qdir)
        if args.json:
            print(json.dumps(st, ensure_ascii=False, indent=2))
        else:
            print("  ".join(f"{k}={v}" for k, v in st["counts"].items()))
            for item in st["detail"]["running"]:
                print(f"  running: {item['run_id']} worker={item['worker']} "
                      f"hb_age={item['heartbeat_age_s']}s")
            for item in st["detail"]["failed"]:
                print(f"  FAILED: {item['run_id']}: {item['error']}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
