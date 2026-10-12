"""Niezależna wyszukiwarka DINOv2 + dokładny FAISS, Python 3.11.

API numeryczne wymaga tylko NumPy. Torch, Pillow, Transformers i FAISS są
importowane dopiero w operacjach, które ich potrzebują. Nie używamy RF-DETR.

Przykład na Edenie (polecenia do późniejszego, osobnego uruchomienia)::

    python wyszukiwanie.py build --manifest pool.jsonl --data-root /data \
        --out indeks --device cpu \
        --torch-hub-dir ~/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
        --weights ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
    python wyszukiwanie.py query --index indeks --image fastgan.png \
        --exclude-json '["uzyta-klatka"]' --out ranking.json --top-k 20

Alternatywa: build --model-dir /lokalny/snapshot/HuggingFace. Żaden tryb nie
pobiera modelu. Query domyślnie odczytuje źródło modelu z indeksu; opcjonalne
ścieżki zastępcze muszą wskazywać identyczne pliki (kontrola SHA-256).

Każdy bbox xywh daje osobny kwadratowy wycinek, z marginesem liczonym na
każdą stronę względem dłuższego boku. Czarne dopełnienie zachowuje proporcje
i orientację. Metadane odnoszą się do surowych pikseli pliku: brak obrotów
oraz automatycznego stosowania EXIF. Zapytanie jest gotowym wycinkiem FastGAN;
obraz prostokątny zostaje dopełniony do kwadratu. Tryb torch.hub stosuje PIL
bicubic resize 256, center crop 224 oraz mean/std ImageNet. Tryb HF używa
lokalnego procesora, z wyłączonym center crop, aby zachować cały wycinek.

Wyniki to kandydatury, nie automatyczna decyzja o dodaniu do treningu.
Różnorodność: najpierw najlepszy wynik z każdej grupy, potem drugi itd.,
w każdej rundzie malejąco według podobieństwa. Wynik cosinusowy nie jest
prawdopodobieństwem. Brak skalibrowanego progu nie daje gwarancji trafienia.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


API_VERSION = 1
ARTIFACTS = ("index.faiss", "features.npy", "metadata.jsonl", "config.json", "manifest.jsonl")
HUB_PREPROCESS = {"resize": 256, "center_crop": 224, "interpolation": "PIL_bicubic",
                  "mean": [.485, .456, .406], "std": [.229, .224, .225]}


def normalize_features(features: Any) -> np.ndarray:
    """Normalizacja L2 wektorów/wierszy, odporna na duże i małe skale."""
    if np.iscomplexobj(features):
        raise ValueError("Cechy muszą być rzeczywiste.")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] == 0:
        raise ValueError("Oczekiwano niepustego wektora albo macierzy cech.")
    if not np.isfinite(values).all():
        raise ValueError("Cechy zawierają NaN lub nieskończoność.")
    scale = np.max(np.abs(values), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Nie można normalizować zerowego wektora.")
    scaled = values / scale
    normalized = scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def normalized_dot(features: Any, query: Any) -> np.ndarray:
    """Dokładne podobieństwo kosinusowe; pomocnicze API CPU, bez FAISS."""
    matrix, vector = normalize_features(features), normalize_features(query)
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
        raise ValueError("Niezgodne wymiary cech i zapytania.")
    return matrix @ vector


def pool_tokens(tokens: Any, pooling: str = "meanpatch", num_register_tokens: int = 0) -> Any:
    """Łączenie tokenów NumPy/Torch; średnia pomija CLS i tokeny rejestrowe."""
    if len(tokens.shape) != 3 or tokens.shape[1] < 1 or num_register_tokens < 0:
        raise ValueError("Nieprawidłowy tensor tokenów.")
    if pooling == "cls":
        return tokens[:, 0, :]
    if pooling != "meanpatch" or tokens.shape[1] <= 1 + num_register_tokens:
        raise ValueError("Nieprawidłowe łączenie tokenów lub brak patchy.")
    return tokens[:, 1 + num_register_tokens:, :].mean(axis=1)


def _id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("Identyfikator musi być niepustym napisem albo liczbą całkowitą.")
    return str(value)


def load_exclusions(value: str | Path | list = "[]") -> set[str]:
    """Lista JSON w argumencie lub pliku; liczba 1 i napis '1' to to samo ID."""
    if isinstance(value, list):
        items = value
    elif str(value).lstrip().startswith(("[", "{")):
        items = json.loads(str(value))
    else:
        items = json.loads(Path(value).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(items, list):
        raise ValueError("Wykluczenia muszą być listą JSON identyfikatorów klatek.")
    return {_id(item) for item in items}


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} musi być dodatnią liczbą całkowitą.")


def rank_candidates(metadata: Sequence[dict], scores: Sequence[float], *,
                    exclude_ids: Iterable = (), top_k: int = 20,
                    diversify_groups: bool = True) -> list[dict]:
    """Filtr train/ID, najlepszy bbox każdej klatki i deterministyczny ranking."""
    _positive_int(top_k, "top_k")
    if len(metadata) != len(scores):
        raise ValueError("Liczba wyników i wierszy metadanych jest różna.")
    excluded = {_id(item) for item in exclude_ids}
    candidates = []
    for row, score in zip(metadata, scores):
        if row.get("split") != "train" or _id(row["frame_id"]) in excluded:
            continue
        if not math.isfinite(float(score)):
            raise ValueError("Niepoprawny wynik podobieństwa.")
        candidate = dict(row, score=float(score))
        candidate["frame_id"], candidate["group_id"] = _id(row["frame_id"]), _id(row["group_id"])
        candidates.append(candidate)
    candidates.sort(key=lambda r: (-r["score"], r["frame_id"],
                                   int(r.get("annotation_index", 0)), str(r["path"])))
    unique, seen, counts = [], set(), Counter()
    for row in candidates:
        if row["frame_id"] in seen:
            continue
        seen.add(row["frame_id"])
        counts[row["group_id"]] += 1
        unique.append(dict(row, similarity_rank=len(unique) + 1,
                           diversity_round=counts[row["group_id"]]))
    if diversify_groups:
        unique.sort(key=lambda r: (r["diversity_round"], -r["score"], r["frame_id"]))
    return [dict(row, rank=i + 1) for i, row in enumerate(unique[:top_k])]


def search_index(index: Any, query: Any, metadata: Sequence[dict], *,
                 exclude_ids: Iterable = (), top_k: int = 20,
                 diversify_groups: bool = True) -> dict:
    """FAISS IndexFlatIP lub zgodny obiekt; zwiększa pobranie aż ranking jest pełny.

    Remis na granicy wymaga dalszego pobrania. Różnorodność wymaga zobaczenia
    odpowiedniej liczby klatek z każdej dostępnej grupy, nawet z końca indeksu.
    Substytut NumPy znajduje się wyłącznie w pliku testowym.
    """
    _positive_int(top_k, "top_k")
    vector = normalize_features(query)
    if vector.ndim != 1 or vector.size != index.d or index.ntotal != len(metadata):
        raise ValueError("Niezgodność indeksu, metadanych lub wymiaru zapytania.")
    excluded = {_id(item) for item in exclude_ids}
    groups = defaultdict(set)
    for row in metadata:
        if row.get("split") == "train" and _id(row["frame_id"]) not in excluded:
            groups[_id(row["group_id"])].add(_id(row["frame_id"]))
    eligible = len(set().union(*groups.values())) if groups else 0
    target = min(top_k, eligible)
    selected, fetched, requests = [], 0, 0
    total = len(metadata)
    k = min(total, max(1, 2 * top_k))
    while target:
        distances, indices = index.search(vector.reshape(1, -1), k)
        requests += 1
        ids, values = indices[0], distances[0]
        if len(ids) != k or any(int(i) < 0 or int(i) >= total for i in ids):
            raise ValueError("Indeks nie zwrócił oczekiwanych, poprawnych wierszy.")
        fetched = k
        rows = [metadata[int(i)] for i in ids]
        ranked = rank_candidates(rows, values, exclude_ids=excluded,
                                 top_k=max(1, len(rows)), diversify_groups=diversify_groups)
        selected = ranked[:target]
        if k == total:
            break
        complete = len(selected) == target
        if complete and diversify_groups:
            rounds = max(row["diversity_round"] for row in selected)
            counts = Counter(row["group_id"] for row in ranked)
            complete = all(counts[group] >= min(rounds, len(frames)) for group, frames in groups.items())
        # Ścisła nierówność chroni także deterministyczny wybór przy remisach.
        if complete and float(np.min(values)) < min(row["score"] for row in selected):
            break
        k = min(total, max(k + 1, k * 2))
    return {"api_version": API_VERSION, "candidates": selected,
            "top_selection": selected[0] if selected else None,
            "selection_status": "requires_review" if selected else "no_candidates",
            "top_k": top_k, "eligible_frames": eligible, "returned_frames": len(selected),
            "retrieved_rows": fetched, "search_calls": requests,
            "exhausted": len(selected) == eligible,
            "ranking": "group_rounds_then_score" if diversify_groups else "score",
            "similarity_threshold": None, "excluded_ids": sorted(excluded),
            "note": "Ranking kandydatur bez gwarancji podobieństwa; wymagana osobna ocena."}


def square_crop_box(bbox: Sequence[float], margin: float = .1) -> tuple[int, int, int, int]:
    """Kwadrat xyxy, bez przycinania do obrazu; bbox obejmuje piksele w xywh."""
    values = np.asarray(bbox, dtype=np.float64)
    if (values.shape != (4,) or not np.isfinite(values).all() or
            values[2] <= 0 or values[3] <= 0 or not math.isfinite(margin) or margin < 0):
        raise ValueError("Bbox xywh musi mieć dodatni rozmiar, margines musi być nieujemny.")
    x, y, width, height = values
    half = max(width, height) * (1 + 2 * margin) / 2
    cx, cy = x + width / 2, y + height / 2
    left, top = math.floor(cx - half), math.floor(cy - half)
    side = max(math.ceil(cx + half) - left, math.ceil(cy + half) - top)
    return left, top, left + side, top + side


def prepare_query_image(image: Any) -> tuple[Any, list[int]]:
    """Gotowy wycinek zapytania: tylko RGB i dopełnienie, bez obrotu."""
    width, height = image.size
    box = square_crop_box([0, 0, width, height], 0)
    return image.convert("RGB").crop(box), list(box)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Błędny JSON w {path}, wiersz {number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Wiersz {number} w {path} nie jest obiektem JSON.")
        rows.append(row)
    return rows


def read_manifest(manifest: str | Path, data_root: str | Path) -> list[dict]:
    """Waliduje całą pulę i rozłączność grup; nie otwiera obrazów val/test."""
    root = Path(data_root).expanduser().resolve()
    rows, ids, paths, groups = [], set(), set(), {}
    for source in _read_jsonl(Path(manifest)):
        if not {"frame_id", "path", "split", "group_id", "annotations"} <= source.keys():
            raise ValueError("Manifest wymaga frame_id, path, split, group_id i annotations.")
        row = dict(source)
        fid, group, split = _id(row["frame_id"]), _id(row["group_id"]), row["split"]
        if split not in {"train", "val", "validation", "test"}:
            raise ValueError(f"Nieznany split: {split!r}.")
        split = "val" if split == "validation" else split
        if not isinstance(row["path"], str) or not row["path"].strip():
            raise ValueError("Brak ścieżki oryginalnej klatki.")
        path = (root / Path(row["path"]).expanduser()).resolve()
        if fid in ids or path in paths:
            raise ValueError(f"Powtórzony frame_id lub ścieżka: {fid}.")
        if group in groups and groups[group] != split:
            raise ValueError(f"Grupa {group} występuje w różnych podziałach.")
        if not isinstance(row["annotations"], list):
            raise ValueError("annotations musi być listą bbox xywh.")
        for annotation in row["annotations"]:
            if not isinstance(annotation, dict) or "bbox" not in annotation:
                raise ValueError("Adnotacja nie zawiera bbox xywh.")
            square_crop_box(annotation["bbox"], 0)
        ids.add(fid)
        paths.add(path)
        groups[group] = split
        rows.append(dict(row, frame_id=fid, group_id=group, split=split,
                         source_path=row["path"], path=str(path)))
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_crops(rows: Sequence[dict], margin: float = .1):
    """Generuje wycinki train i metadane z pozycją, skalą i skrótem zdjęcia."""
    from PIL import Image

    square_crop_box([0, 0, 1, 1], margin)
    for row in rows:
        if row["split"] != "train" or not row["annotations"]:
            continue
        image_hash = sha256_file(row["path"])
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        for i, annotation in enumerate(row["annotations"]):
            x, y, bw, bh = annotation["bbox"]
            if x >= width or y >= height or x + bw <= 0 or y + bh <= 0:
                raise ValueError(f"Bbox poza zdjęciem: {row['frame_id']}, adnotacja {i}.")
            box = square_crop_box(annotation["bbox"], margin)
            left, top, right, bottom = box
            meta = dict(row, annotation_index=i, bbox=annotation["bbox"], crop_xyxy=list(box),
                        image_size=[width, height], image_sha256=image_hash,
                        padding_ltrb=[max(0, -left), max(0, -top),
                                      max(0, right - width), max(0, bottom - height)],
                        bbox_center_fraction=[(x + bw / 2) / width, (y + bh / 2) / height],
                        bbox_size_fraction=[bw / width, bh / height])
            yield image.crop(box), meta


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Zapis i odczyt indeksu wymaga FAISS (np. faiss-cpu).") from exc
    return faiss


def _check_output_directory(out: str | Path) -> Path:
    path = Path(out).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Katalog wyjściowy nie jest pusty: {path}.")
    return path


def _validate_index_metadata(rows: Sequence[dict]) -> None:
    frames, paths, crops = {}, {}, set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Indeks może zawierać wyłącznie split train.")
        fid, group = _id(row["frame_id"]), _id(row["group_id"])
        path = str(row["path"])
        identity = (group, path)
        if fid in frames and frames[fid] != identity:
            raise ValueError("Sprzeczne metadane tej samej klatki.")
        if path in paths and paths[path] != fid:
            raise ValueError("Jedna ścieżka ma różne identyfikatory klatek.")
        key = (fid, row.get("annotation_index", 0))
        if key in crops:
            raise ValueError("Powtórzony identyfikator wycinka.")
        frames[fid], paths[path] = identity, fid
        crops.add(key)


def save_index(out: str | Path, features: Any, metadata: Sequence[dict],
               config: dict, manifest: str | Path) -> dict:
    """Zapisuje pięć artefaktów oraz checksums.json jako znacznik kompletności."""
    matrix = normalize_features(features)
    if matrix.ndim != 2 or not len(matrix) or len(matrix) != len(metadata):
        raise ValueError("Indeks wymaga niepustej macierzy i jednego wiersza metadanych na cechę.")
    _validate_index_metadata(metadata)
    output = _check_output_directory(out)
    manifest_bytes = Path(manifest).read_bytes()
    final_config = dict(config, api_version=API_VERSION, index_type="IndexFlatIP",
                        normalized=True, feature_count=len(matrix), feature_dim=matrix.shape[1],
                        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    config_text = _json(final_config)
    faiss = _faiss()
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    output.mkdir(parents=True, exist_ok=True)
    # Faiss serializuje do bajtów, co działa także dla ścieżek z polskimi znakami.
    with (output / "index.faiss").open("xb") as handle:
        handle.write(faiss.serialize_index(index).tobytes())
    with (output / "features.npy").open("xb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    with (output / "metadata.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for i, row in enumerate(metadata):
            handle.write(json.dumps(dict(row, row_index=i), ensure_ascii=False,
                                    sort_keys=True, allow_nan=False) + "\n")
    with (output / "manifest.jsonl").open("xb") as handle:
        handle.write(manifest_bytes)
    with (output / "config.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(config_text)
    checksums = {name: sha256_file(output / name) for name in ARTIFACTS}
    with (output / "checksums.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_json(checksums))
    return final_config


def load_index(path: str | Path) -> tuple[Any, np.ndarray, list[dict], dict]:
    """Sprawdza skróty przed odczytem FAISS i zgodność wierszy z macierzą."""
    root = Path(path).expanduser().resolve()
    if root.name == "index.faiss" and root.is_file():
        root = root.parent
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    if set(checksums) != set(ARTIFACTS):
        raise ValueError("Niekompletna lista skrótów indeksu.")
    for name in ARTIFACTS:
        if sha256_file(root / name) != checksums[name]:
            raise ValueError(f"Niezgodny SHA-256: {name}.")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("api_version") != API_VERSION or config.get("index_type") != "IndexFlatIP":
        raise ValueError("Nieobsługiwana wersja albo rodzaj indeksu.")
    if config["manifest_sha256"] != checksums["manifest.jsonl"]:
        raise ValueError("Niezgodny manifest indeksu.")
    features = np.load(root / "features.npy", allow_pickle=False)
    rows = _read_jsonl(root / "metadata.jsonl")
    _validate_index_metadata(rows)
    if (features.ndim != 2 or features.dtype != np.float32 or not len(rows) or
            features.shape != (config["feature_count"], config["feature_dim"]) or
            len(features) != len(rows) or
            any(row.get("row_index") != i for i, row in enumerate(rows)) or
            not np.isfinite(features).all() or
            not np.allclose(np.linalg.norm(features, axis=1), 1, rtol=1e-5, atol=1e-6)):
        raise ValueError("Niezgodne metadane albo nieznormalizowane cechy.")
    faiss = _faiss()
    index = faiss.deserialize_index(np.frombuffer((root / "index.faiss").read_bytes(), dtype=np.uint8))
    if (not isinstance(index, faiss.IndexFlatIP) or index.ntotal != len(rows) or
            index.d != features.shape[1] or index.metric_type != faiss.METRIC_INNER_PRODUCT or
            not np.array_equal(index.reconstruct_n(0, index.ntotal), features)):
        raise ValueError("Indeks FAISS jest niezgodny z zapisanymi cechami.")
    return index, features, rows, config


def resolve_model_source(*, model_dir: str | Path | None = None,
                         torch_hub_dir: str | Path | None = None,
                         weights: str | Path | None = None) -> dict:
    """Wybiera istniejące pliki i liczy ich skróty; nigdy nie pobiera modelu."""
    hub_mode = torch_hub_dir is not None or weights is not None
    if hub_mode:
        if model_dir is not None or torch_hub_dir is None or weights is None:
            raise ValueError("Podaj razem --torch-hub-dir i --weights, bez --model-dir.")
        repo, checkpoint = Path(torch_hub_dir).expanduser().resolve(), Path(weights).expanduser().resolve()
        if not (repo / "hubconf.py").is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Brak lokalnego hubconf.py lub pliku wag DINOv2.")
        source_hashes = {str(p.relative_to(repo)).replace(os.sep, "/"): sha256_file(p)
                         for p in sorted(repo.rglob("*.py")) if p.is_file()}
        weight_hash = sha256_file(checkpoint)
        return {"backend": "torch_hub", "architecture": "dinov2_vits14",
                "torch_hub_dir": str(repo), "weights": str(checkpoint),
                "weights_sha256": weight_hash, "repo_sha256": _digest_json(source_hashes),
                "model_sha256": _digest_json({"sources": source_hashes, "weights": weight_hash})}
    if model_dir is None:
        raise ValueError("Podaj --model-dir albo parę --torch-hub-dir i --weights.")
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise FileNotFoundError("--model-dir musi wskazywać istniejący lokalny snapshot HF.")
    model_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") not in {"dinov2", "dinov2_with_registers"}:
        raise ValueError("Dozwolony jest wyłącznie niezależny model DINOv2.")
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in {".json", ".safetensors"} or p.name.startswith("pytorch_model") and p.suffix == ".bin"))
    if not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise FileNotFoundError("Lokalny model nie zawiera wag.")
    return {"backend": "huggingface", "model_dir": str(root),
            "model_sha256": _digest_json({p.name: sha256_file(p) for p in files})}


def hub_preprocess(image: Any) -> np.ndarray:
    """PIL resize 256 -> center crop 224 -> CHW ImageNet, bez torchvision."""
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    if width != height:
        raise ValueError("Przed preprocessingiem wymagany jest kwadrat z dopełnieniem.")
    image = image.resize((256, 256), resample=Image.Resampling.BICUBIC).crop((16, 16, 240, 240))
    values = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    mean = np.asarray(HUB_PREPROCESS["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(HUB_PREPROCESS["std"], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((values - mean) / std)


class DinoEncoder:
    """Zamrożony niezależny DINOv2; ciężkie zależności są lokalne dla konstruktora."""

    def __init__(self, source: dict, *, device: str = "cpu", pooling: str = "meanpatch"):
        if device not in {"cpu", "cuda"} or pooling not in {"meanpatch", "cls"}:
            raise ValueError("Wybierz device cpu/cuda oraz pooling meanpatch/cls.")
        import torch

        self.torch, self.source, self.device, self.pooling = torch, source, device, pooling
        if source["backend"] == "torch_hub":
            # pretrained=False wyłącza gałąź pobierającą checkpoint z sieci.
            self.model = torch.hub.load(source["torch_hub_dir"], "dinov2_vits14",
                                        source="local", pretrained=False)
            state = torch.load(source["weights"], map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.preprocessing = dict(HUB_PREPROCESS)
        else:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(source["model_dir"],
                                local_files_only=True, trust_remote_code=False, use_fast=False)
            self.model = AutoModel.from_pretrained(source["model_dir"],
                                local_files_only=True, trust_remote_code=False)
            if self.model.config.model_type not in {"dinov2", "dinov2_with_registers"}:
                raise ValueError("Model nie jest niezależnym DINOv2.")
            self.preprocessing = {"processor": self.processor.to_dict(),
                                  "do_center_crop": False, "do_resize": True}
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        if not images or any(image.width != image.height for image in images):
            raise ValueError("Encoder wymaga niepustej listy kwadratowych obrazów.")
        with self.torch.inference_mode():
            if self.source["backend"] == "torch_hub":
                inputs = self.torch.from_numpy(np.stack([hub_preprocess(image) for image in images]))
                output = self.model.forward_features(inputs.to(self.device))
                pooled = (output["x_norm_clstoken"] if self.pooling == "cls" else
                          output["x_norm_patchtokens"].mean(dim=1))
            else:
                inputs = self.processor(images=list(images), return_tensors="pt",
                                        do_resize=True, do_center_crop=False)
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                output = self.model(**inputs, return_dict=True)
                pooled = pool_tokens(output.last_hidden_state, self.pooling,
                                     getattr(self.model.config, "num_register_tokens", 0))
            return normalize_features(pooled.detach().float().cpu().numpy())


def build_index(manifest: str | Path, data_root: str | Path, out: str | Path, *,
                model_dir: str | Path | None = None, torch_hub_dir: str | Path | None = None,
                weights: str | Path | None = None, device: str = "cpu",
                pooling: str = "meanpatch", margin: float = .1, batch_size: int = 32) -> dict:
    """Buduje indeks puli train. Zwraca konfigurację zapisanego indeksu, API v1."""
    _positive_int(batch_size, "batch_size")
    _check_output_directory(out)
    square_crop_box([0, 0, 1, 1], margin)
    manifest = Path(manifest).expanduser().resolve()
    manifest_hash = sha256_file(manifest)
    rows = read_manifest(manifest, data_root)
    if not any(row["split"] == "train" and row["annotations"] for row in rows):
        raise ValueError("Manifest nie zawiera oznaczonych klatek train.")
    _faiss()
    source = resolve_model_source(model_dir=model_dir, torch_hub_dir=torch_hub_dir, weights=weights)
    encoder = DinoEncoder(source, device=device, pooling=pooling)
    features, metadata, images = [], [], []
    for image, row in iter_crops(rows, margin):
        images.append(image)
        metadata.append(row)
        if len(images) == batch_size:
            features.append(encoder.encode(images))
            images.clear()
    if images:
        features.append(encoder.encode(images))
    if sha256_file(manifest) != manifest_hash:
        raise ValueError("Manifest zmienił się w czasie budowania indeksu.")
    config = {"model_source": source, "pooling": pooling, "device_at_build": device,
              "preprocessing": encoder.preprocessing, "margin": margin,
              "crop_policy": "square_bbox_black_padding_no_rotation_no_exif",
              "query_policy": "ready_crop_square_padding", "data_root": str(Path(data_root).resolve()),
              "manifest_path": str(manifest), "input_frames": len(rows),
              "indexed_frames": len({row["frame_id"] for row in metadata}),
              "excluded_nontrain_frames": sum(row["split"] != "train" for row in rows),
              "skipped_unannotated_train": sum(row["split"] == "train" and not row["annotations"] for row in rows)}
    return save_index(out, np.concatenate(features, axis=0), metadata, config, manifest)


def query_index(index: str | Path, image: str | Path, exclude_json: str | Path | list = "[]",
                out: str | Path | None = None, top_k: int = 20, *, device: str = "cpu",
                model_dir: str | Path | None = None, torch_hub_dir: str | Path | None = None,
                weights: str | Path | None = None, diversify_groups: bool = True) -> dict:
    """Zwraca/zapisuje kandydatury z oryginalnymi ścieżkami; API v1."""
    _positive_int(top_k, "top_k")
    if out is not None and Path(out).exists():
        raise FileExistsError(f"Plik wynikowy już istnieje: {out}.")
    faiss_index, _, rows, config = load_index(index)
    excluded = load_exclusions(exclude_json)
    image_path = Path(image).expanduser().resolve()
    index_root = Path(index).expanduser().resolve()
    if index_root.is_file():
        index_root = index_root.parent
    manifest_root = Path(config.get("data_root", index_root)).expanduser()
    for row in _read_jsonl(index_root / "manifest.jsonl"):
        if row["split"] != "train" and (manifest_root / row["path"]).resolve() == image_path:
            raise ValueError("Klatka val/test nie może sterować selekcją danych do treningu.")
    # Oryginalna klatka zapytania nie może wskazać samej siebie.
    excluded.update(_id(row["frame_id"]) for row in rows if Path(row["path"]).resolve() == image_path)
    stored_source = config["model_source"]
    if all(value is None for value in (model_dir, torch_hub_dir, weights)):
        model_dir, torch_hub_dir, weights = (stored_source.get(key)
                                           for key in ("model_dir", "torch_hub_dir", "weights"))
    source = resolve_model_source(model_dir=model_dir, torch_hub_dir=torch_hub_dir, weights=weights)
    if (source["backend"] != stored_source["backend"] or
            source["model_sha256"] != stored_source["model_sha256"]):
        raise ValueError("Model zapytania różni się od modelu indeksu; zbuduj indeks ponownie.")
    encoder = DinoEncoder(source, device=device, pooling=config["pooling"])
    if encoder.preprocessing != config["preprocessing"]:
        raise ValueError("Preprocessing zapytania różni się od preprocessingu indeksu.")
    from PIL import Image

    with Image.open(image_path) as original:
        crop, box = prepare_query_image(original)
        original_size = list(original.size)
    feature = encoder.encode([crop])[0]
    report = search_index(faiss_index, feature, rows, exclude_ids=excluded, top_k=top_k,
                          diversify_groups=diversify_groups)
    report.update(query_image=str(image_path), query_sha256=sha256_file(image_path),
                  query_crop_xyxy=box, query_image_size=original_size,
                  index=str(Path(index).expanduser().resolve()),
                  manifest_sha256=config["manifest_sha256"], config_sha256=_digest_json(config),
                  model_sha256=source["model_sha256"], pooling=config["pooling"])
    if out is not None:
        target = Path(out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_json(report))
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lokalny niezależny DINOv2 + FAISS. API v1.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Indeks oznaczonych klatek train.")
    build.add_argument("--manifest", required=True)
    build.add_argument("--data-root", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--pooling", choices=("meanpatch", "cls"), default="meanpatch")
    build.add_argument("--margin", type=float, default=.1, help="Margines na stronę względem dłuższego boku bbox.")
    build.add_argument("--batch-size", type=int, default=32)
    query = sub.add_parser("query", help="Ranking oryginalnych klatek; bez progu akceptacji.")
    query.add_argument("--index", required=True, help="Katalog indeksu lub jego index.faiss.")
    query.add_argument("--image", required=True, help="Gotowy wycinek zapytania FastGAN.")
    query.add_argument("--exclude-json", default="[]", help="Lista JSON ID lub ścieżka pliku z listą.")
    query.add_argument("--out", required=True)
    query.add_argument("--top-k", type=int, default=20)
    query.add_argument("--no-group-diversity", action="store_true", help="Ranking tylko według podobieństwa.")
    for command in (build, query):
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--model-dir", help="Istniejący snapshot HF; tylko lokalne pliki.")
        mode.add_argument("--torch-hub-dir", help="Istniejący katalog repozytorium torch.hub DINOv2.")
        command.add_argument("--weights", help="Lokalny dinov2_vits14_pretrain.pth, wymaga --torch-hub-dir.")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "build":
            result = build_index(**args)
        else:
            args["diversify_groups"] = not args.pop("no_group_diversity")
            result = query_index(**args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Błąd: {exc}\n")
    print(_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
