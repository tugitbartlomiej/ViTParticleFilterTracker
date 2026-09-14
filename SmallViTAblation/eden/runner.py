#!/usr/bin/env python3
"""
Worker wezla Eden (Agent 2). Jeden proces = jedna karta.

Petla: claim -> stage/model dispatcher (CUDA_VISIBLE_DEVICES=--worker)
-> heartbeat w osobnym watku -> done/fail -> nastepne zadanie.

Blad jednego biegu zapisuje failed z logiem i NIE zatrzymuje reszty.
Worker konczy, gdy kolejka jest pusta przez --idle-exit-rounds rund pollowania.
Przy kazdym idlu odpala requeue-stale (wznowienie po pracowniku ubitym
limitem czasu SLURM).

Uzycie:
    py -3.11 runner.py --queue QDIR --data /raid/ablation_data \
        --worker 0 --protocol ../configs/protocol.yaml [--out OUT]

Testy bez GPU: --cmd-template "{python} {fixture_task} {run_id}"
    (placeholdery: {run_id} {worker} {queue} {data} {out} {protocol}
     {variant_file} {python} + dowolne przez --template-var NAZWA=WARTOSC)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

EDEN_DIR = Path(__file__).resolve().parent
_QUEUE_SPEC = importlib.util.spec_from_file_location("small_vit_eden_queue", EDEN_DIR / "queue.py")
if _QUEUE_SPEC is None or _QUEUE_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load local Eden queue module")
qmod = importlib.util.module_from_spec(_QUEUE_SPEC)
_QUEUE_SPEC.loader.exec_module(qmod)

try:
    import yaml
except ImportError:
    yaml = None

TRAIN_SCRIPT = (Path(__file__).resolve().parents[1] / "pipeline" / "train"
                / "train_rfdetr.py")
TRANSFER_SCRIPT = (Path(__file__).resolve().parents[1] / "pipeline" / "transfer"
                   / "train_transfer.py")
MODULE_BY_STAGE = {
    "active_selection": "SmallViTAblation.pipeline.active_learning.simulate",
    "latent_diagnostics": "SmallViTAblation.pipeline.latent_dynamics.diagnose",
}


def log(qdir: Path, worker: str, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] worker={worker} {msg}"
    print(line, flush=True)
    try:
        with open(qdir / "logs" / f"worker_{worker}.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


class HeartbeatThread(threading.Thread):
    def __init__(self, qdir: Path, run_id: str, worker: str, interval_s: float,
                 lock_ttl: float, lease_token: str):
        super().__init__(daemon=True)
        self.qdir, self.run_id, self.worker = qdir, run_id, worker
        self.interval_s, self.lock_ttl = interval_s, lock_ttl
        self.lease_token = lease_token
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.wait(self.interval_s):
            try:
                ok = qmod.heartbeat(self.qdir, self.run_id, self.worker,
                                    self.lock_ttl, self.lease_token)
                if not ok:
                    print(f"[runner] lease lost for {self.run_id}", file=sys.stderr)
                    return
            except Exception as e:  # heartbeat nie moze zabic treningu
                print(f"[runner] heartbeat error: {e}", file=sys.stderr)

    def stop(self) -> None:
        self._stop_evt.set()


def protocol_queue_cfg(protocol_path: Path | None) -> dict:
    if protocol_path and yaml and Path(protocol_path).is_file():
        try:
            prot = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
            return prot.get("eden", {}).get("queue", {}) or {}
        except Exception:
            pass
    return {}


def split_cmdline(s: str) -> list[str]:
    """shlex.split, ktory na Windows nie zjada backslashy w sciezkach."""
    if os.name == "nt":
        parts = shlex.split(s, posix=False)
        return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p
                for p in parts]
    return shlex.split(s)


def build_command(args, task: dict, fmt: dict) -> list[str]:
    if args.cmd_template:
        return split_cmdline(args.cmd_template.format(**fmt))
    if task.get("argv") is not None:
        if not isinstance(task["argv"], list) or not all(isinstance(x, str) for x in task["argv"]):
            raise ValueError("task.argv must be a list of strings")
        return [part.format(**fmt) for part in task["argv"]]
    stage = task.get("stage", "train")
    if stage in MODULE_BY_STAGE:
        raise ValueError(f"stage {stage!r} requires an explicit argv list")
    if stage != "train":
        raise ValueError(f"unsupported task stage {stage!r}")
    arch = task.get("arch", "rf-detr-small")
    common = ["--run-id", task["run_id"],
              "--variant-file", fmt["variant_file"],
              "--protocol", str(args.protocol),
              "--data-root", str(args.data),
              "--out", str(args.out),
              "--train-seed", str(task.get("train_seed", task.get("seed", 0)))]
    if arch in {"rf-detr-small", "rf-detr-nano"}:
        role = "smoke" if arch == "rf-detr-nano" else (args.role or "main")
        cmd = [fmt["python"], str(TRAIN_SCRIPT), *common, "--role", role]
    elif arch in {"yolov8n", "yolo-current", "d-fine-s", "rtdetrv2-s"}:
        cmd = [fmt["python"], str(TRANSFER_SCRIPT), *common, "--arch", arch]
    else:
        raise ValueError(f"unsupported training architecture {arch!r}")
    profile = task.get("profile") or args.profile
    if profile:
        cmd += ["--profile", str(profile)]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_one(args, qdir: Path, task: dict, fmt: dict) -> bool:
    run_id = task["run_id"]
    attempt_id = task.get("_queue_attempt_id", "unknown")
    lease_token = task.get("_queue_lease_token")
    if not lease_token:
        raise ValueError("claimed task has no lease token")
    env = dict(os.environ)
    allocated_devices = env.get("CUDA_VISIBLE_DEVICES")
    if allocated_devices is not None:
        devices = [value.strip() for value in allocated_devices.split(",") if value.strip()]
        index = int(args.worker)
        if not 0 <= index < len(devices):
            raise ValueError(
                f"worker {args.worker} is outside the allocated CUDA devices: {allocated_devices}")
        env["CUDA_VISIBLE_DEVICES"] = devices[index]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(args.worker)
    env["PYTHONHASHSEED"] = str(task.get("seed", 0))

    cmd = build_command(args, task, fmt)
    log(qdir, args.worker, f"START {run_id}: {' '.join(cmd)}")

    log_path = qdir / "logs" / f"{run_id}__{attempt_id}.log"
    hb = HeartbeatThread(qdir, run_id, args.worker, args.heartbeat, args.lock_ttl,
                         lease_token)
    hb.start()
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"cmd: {cmd}\nworker: {args.worker}\n\n")
            lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    except Exception as e:
        rc, err = -1, f"subprocess exception: {e}"
    else:
        err = ""
    finally:
        hb.stop()
        hb.join(timeout=5)

    dt = time.time() - t0
    if rc == 0:
        finalized = qmod.mark_done(qdir, run_id, args.worker, args.lock_ttl,
                                   lease_token)
        if not finalized:
            log(qdir, args.worker, f"LEASE-LOST {run_id}; result not finalized")
            return False
        log(qdir, args.worker, f"DONE {run_id} w {dt:.0f}s")
        return True
    if not err:
        err = _tail(log_path, 800) or f"exit code {rc}"
    qmod.mark_fail(qdir, run_id, message=err, log_path=str(log_path),
                   worker=args.worker, lock_ttl=args.lock_ttl,
                   lease_token=lease_token)
    log(qdir, args.worker, f"FAIL {run_id} (rc={rc}) - log: {log_path}")
    return False


def _tail(path: Path, n_chars: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[-n_chars:]
    except OSError:
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True, type=Path, help="katalog kolejki (qdir)")
    ap.add_argument("--data", required=True, type=Path, help="data-root z images/ itd.")
    ap.add_argument("--worker", required=True, help="id workera = index karty GPU")
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="katalog wynikow (domyslnie: <data>/../results)")
    ap.add_argument("--variants-dir", type=Path, default=None,
                    help="domyslnie <data>/variants")
    ap.add_argument("--role", choices=["main", "smoke"], default=None)
    ap.add_argument("--profile", default=None,
                    help="default execution profile; task.profile takes precedence")
    ap.add_argument("--dry-run", action="store_true",
                    help="przekaz --dry-run do train_rfdetr.py")
    ap.add_argument("--cmd-template", default=None,
                    help="TESTY: szablon polecenia zamiast train_rfdetr.py")
    ap.add_argument("--template-var", action="append", default=[],
                    metavar="NAZWA=WARTOSC", help="dodatkowe placeholdery szablonu")
    ap.add_argument("--heartbeat", type=float, default=None)
    ap.add_argument("--stale-timeout", type=float, default=None)
    ap.add_argument("--poll", type=float, default=None)
    ap.add_argument("--idle-exit-rounds", type=int, default=None)
    ap.add_argument("--lock-ttl", type=float, default=30.0)
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="limit zadan na proces (0 = bez limitu; przydatne w testach)")
    ap.add_argument("--max-attempts", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = protocol_queue_cfg(args.protocol)
    args.heartbeat = args.heartbeat or float(cfg.get("heartbeat_interval_s", 30))
    stale_timeout = args.stale_timeout or float(cfg.get("stale_timeout_s", 180))
    poll = args.poll or float(cfg.get("poll_interval_s", 15))
    idle_rounds = args.idle_exit_rounds or int(cfg.get("idle_exit_rounds", 3))
    max_attempts = args.max_attempts or int(cfg.get("max_attempts", 3))

    qdir = args.queue.resolve()
    data = args.data.resolve()
    out = (args.out or (data.parent / "results")).resolve()
    variants_dir = (args.variants_dir or (data / "variants")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    extra_vars = {}
    for kv in args.template_var:
        k, _, v = kv.partition("=")
        extra_vars[k] = v

    n_done = n_fail = idle = 0
    while args.max_tasks == 0 or (n_done + n_fail) < args.max_tasks:
        task = qmod.claim_task(qdir, args.worker, args.lock_ttl)
        if task is None:
            idle += 1
            if idle >= idle_rounds:
                log(qdir, args.worker, "kolejka pusta - koncze")
                break
            # przy okazji: przywroc zadania po martwych workerach
            requeued = qmod.requeue_stale(qdir, stale_timeout, args.lock_ttl,
                                          max_attempts)
            if requeued:
                log(qdir, args.worker, f"requeue-stale: {requeued}")
                idle = 0
            time.sleep(poll)
            continue
        idle = 0
        run_id = task["run_id"]
        fmt = {
            "run_id": run_id,
            "worker": str(args.worker),
            "queue": str(qdir),
            "data": str(data),
            "out": str(out),
            "protocol": str(args.protocol),
            "variant_file": str(task.get("variant_file") or
                                (variants_dir / f"{run_id}.json")),
            "python": sys.executable,
            **extra_vars,
        }
        ok = run_one(args, qdir, task, fmt)
        n_done, n_fail = n_done + ok, n_fail + (not ok)

    log(qdir, args.worker, f"podsumowanie: done={n_done} failed={n_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
