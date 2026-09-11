"""Wspolne narzedzia Agent 1: konfiguracja, manifest, format cache cech.

Kontrakt cache (README.md sekcja 4 / AGENT_1 zadanie 1):
  plik .pkl = dict {
      "features":  np.ndarray float32, ksztalt (N, D),
      "frame_ids": list[str], dlugosci N, W TEJ SAMEJ KOLEJNOSCI co wiersze features,
      "meta":      dict (model, wersja, data, normalizacja, ...),
  }
Kolejnosc kanoniczna puli: frame_id posortowany rosnaco (leksykalnie).
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# ROOT = SmallViTAblation/  (pipeline/features/common.py -> parents[2])
ROOT = Path(__file__).resolve().parents[2]

GROUP_NAMES = ("dino", "sam", "fourier")


def load_protocol(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "protocol.yaml"
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def protocol_path_default() -> Path:
    return ROOT / "configs" / "protocol.yaml"


def resolve_under_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_manifest(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def train_pool(records: Iterable[dict]) -> list[dict]:
    """Klatki split=='train' w kolejnosci kanonicznej (sort po frame_id).

    Selekcja i cache widza WYLACZNIE ten podzial (README sekcja 5, regula 2).
    """
    pool = [r for r in records if r.get("split") == "train"]
    ids = [r.get("frame_id") for r in pool]
    if any(frame_id is None for frame_id in ids):
        raise ValueError("train pool contains a row without frame_id")
    if len(ids) != len(set(ids)):
        raise ValueError("train pool contains duplicate frame_id values")
    pool.sort(key=lambda r: r["frame_id"])
    return pool


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- cache -----

def save_feature_pkl(path: str | Path, features: np.ndarray,
                     frame_ids: list[str], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": np.ascontiguousarray(features, dtype=np.float32),
        "frame_ids": list(frame_ids),
        "meta": dict(meta),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_feature_pkl(path: str | Path) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    for key in ("features", "frame_ids"):
        if key not in payload:
            raise ValueError(f"{path}: brak pola '{key}' - stary format cache?")
    payload["features"] = np.asarray(payload["features"], dtype=np.float32)
    return payload


def gather_rows(cache: dict, pool_frame_ids: list[str], what: str) -> np.ndarray:
    """Przestawia wiersze cache do kolejnosci kanonicznej puli."""
    index = {fid: i for i, fid in enumerate(cache["frame_ids"])}
    missing = [fid for fid in pool_frame_ids if fid not in index]
    if missing:
        raise ValueError(
            f"{what}: w cache brakuje {len(missing)} klatek z puli train, "
            f"np. {missing[:3]}. Przelicz cache (build_cache.py).")
    rows = np.array([index[fid] for fid in pool_frame_ids], dtype=np.int64)
    return cache["features"][rows]


# ------------------------------------------------------------ checkpoint ----

class CheckpointedGroup:
    """Wznawialne liczenie jednej grupy cech.

    Stan czastkowy trzymany w <cache_dir>/<name>.partial.pkl; po zakonczeniu
    finalize() zapisuje <cache_dir>/<name>.pkl i usuwa plik czastkowy.
    """

    def __init__(self, cache_dir: Path, name: str, dim: int,
                 checkpoint_every: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.dim = dim
        self.checkpoint_every = checkpoint_every
        self.partial_path = self.cache_dir / f"{name}.partial.pkl"
        self.final_path = self.cache_dir / f"{name}.pkl"
        self.frame_ids: list[str] = []
        self.chunks: list[np.ndarray] = []
        self._since_save = 0
        self._load_partial()

    def _load_partial(self) -> None:
        if self.partial_path.exists():
            with open(self.partial_path, "rb") as f:
                state = pickle.load(f)
            if state.get("dim") == self.dim:
                self.frame_ids = list(state["frame_ids"])
                self.chunks = [np.asarray(state["features"], dtype=np.float32)]
                print(f"[checkpoint] {self.name}: wznowiono od "
                      f"{len(self.frame_ids)} klatek")
            else:
                print(f"[checkpoint] {self.name}: partial ma inny wymiar - "
                      f"zaczynam od zera")

    @property
    def done_ids(self) -> set[str]:
        return set(self.frame_ids)

    def append(self, frame_ids: list[str], features: np.ndarray) -> None:
        self.frame_ids.extend(frame_ids)
        self.chunks.append(np.asarray(features, dtype=np.float32))
        self._since_save += len(frame_ids)
        if self._since_save >= self.checkpoint_every:
            self.save_partial()

    def save_partial(self) -> None:
        feats = self._stack()
        state = {"frame_ids": self.frame_ids, "features": feats,
                 "dim": self.dim}
        tmp = self.partial_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.partial_path)
        self._since_save = 0

    def _stack(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.ascontiguousarray(np.vstack(self.chunks), dtype=np.float32)

    def finalize(self, pool_frame_ids: list[str], meta: dict) -> Path:
        """Zapisuje cache finalny w kolejnosci kanonicznej puli."""
        feats = self._stack()
        got = {fid: i for i, fid in enumerate(self.frame_ids)}
        missing = [fid for fid in pool_frame_ids if fid not in got]
        if missing:
            raise RuntimeError(
                f"{self.name}: nie policzono {len(missing)} klatek, "
                f"np. {missing[:3]}")
        order = np.array([got[fid] for fid in pool_frame_ids], dtype=np.int64)
        save_feature_pkl(self.final_path, feats[order], pool_frame_ids, meta)
        if self.partial_path.exists():
            self.partial_path.unlink()
        return self.final_path


def write_cache_manifest(cache_dir: Path, entries: dict[str, Any]) -> Path:
    """feature_cache/manifest.json - scala informacje o wszystkich grupach."""
    path = Path(cache_dir) / "manifest.json"
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.setdefault("groups", {})
    existing["groups"].update(entries)
    existing["updated_at"] = utc_now()
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path
