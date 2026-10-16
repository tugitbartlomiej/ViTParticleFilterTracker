"""Buduje dokładny manifest bazy 1500 i czyta istniejące obrazy bez ich kopiowania."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

WIDTH = 480
HEIGHT = 320
CAMPAIGN_LEGACY_SCALE = 2.0 / 3.0


def _basename(value: str) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def frame_id(value: str) -> str:
    stem = Path(_basename(value)).stem
    return f"cataract_lmm/tracking/{stem[:14]}/{stem[15:]}"


def build_manifest(coco_path: Path, read_bytes: Callable[[str], bytes]) -> list[dict]:
    """Zachowuje dokładną kolejność `images` z COCO i hashuje źródłowe bajty."""
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    images = coco["images"]
    raw_annotations = coco["annotations"]
    image_ids = [int(image["id"]) for image in images]
    annotation_ids = [int(annotation["id"]) for annotation in raw_annotations]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("COCO zawiera powtórzone identyfikatory images")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("COCO zawiera powtórzone identyfikatory annotations")
    unknown = sorted(
        {int(annotation["image_id"]) for annotation in raw_annotations} - set(image_ids)
    )
    if unknown:
        raise ValueError(f"COCO zawiera adnotacje spoza listy images: {unknown}")
    if len(raw_annotations) != len(images):
        raise ValueError(
            f"Liczba images ({len(images)}) i annotations ({len(raw_annotations)}) jest różna"
        )
    annotations: dict[int, list[dict]] = {}
    for annotation in raw_annotations:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    rows = []
    seen_names: set[str] = set()
    for index, image in enumerate(images):
        own = annotations.get(int(image["id"]), [])
        if len(own) != 1:
            raise ValueError(
                f"Obraz {image['id']} ma {len(own)} adnotacji; oczekiwano dokładnie jednej"
            )
        annotation = own[0]
        x, y, w, h = (float(v) for v in annotation["bbox"])
        if not (w > 0 and h > 0):
            raise ValueError(f"Adnotacja {annotation['id']} ma zdegenerowaną ramkę")
        source_width = int(image["width"])
        source_height = int(image["height"])
        if not (
            x >= 0
            and y >= 0
            and x + w <= source_width + 1e-6
            and y + h <= source_height + 1e-6
        ):
            raise ValueError(
                f"Adnotacja {annotation['id']} wychodzi poza obraz {source_width}x{source_height}"
            )
        source_name = str(image["file_name"])
        base_name = _basename(source_name)
        if base_name in seen_names:
            raise ValueError(f"Powtórzona nazwa pliku obrazu: {base_name}")
        seen_names.add(base_name)
        payload = read_bytes(source_name)
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"OpenCV nie zdekodował obrazu {base_name}")
        actual_height, actual_width = decoded.shape[:2]
        if (actual_width, actual_height) != (source_width, source_height):
            raise ValueError(
                f"COCO i bajty obrazu {base_name} mają różne rozmiary: "
                f"{source_width}x{source_height} vs {actual_width}x{actual_height}"
            )
        scale_x = WIDTH / source_width
        scale_y = HEIGHT / source_height
        rows.append(
            {
                "index": index,
                "image_id": int(image["id"]),
                "annotation_id": int(annotation["id"]),
                "image_name": base_name,
                "frame_id": frame_id(source_name),
                "source_width": source_width,
                "source_height": source_height,
                "source_dimensions_verified_from_bytes": True,
                "resize_scale_xy": [scale_x, scale_y],
                "bbox_xywh_source": [x, y, w, h],
                "bbox_xyxy_480": [
                    x * scale_x,
                    y * scale_y,
                    (x + w) * scale_x,
                    (y + h) * scale_y,
                ],
                "bbox_xyxy_480_campaign_legacy": [
                    x * CAMPAIGN_LEGACY_SCALE,
                    y * CAMPAIGN_LEGACY_SCALE,
                    (x + w) * CAMPAIGN_LEGACY_SCALE,
                    (y + h) * CAMPAIGN_LEGACY_SCALE,
                ],
                "image_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def canonical_jsonl(rows: list[dict]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def manifest_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(canonical_jsonl(rows)).hexdigest()


class LocalImageSource:
    """Czyta istniejące pliki wskazane przez COCO albo katalog `images`."""

    def __init__(self, data_root: Path | None = None):
        self.data_root = Path(data_root) if data_root else None

    def path(self, source_name: str) -> Path:
        direct = Path(source_name)
        if direct.exists():
            return direct
        if self.data_root is None:
            raise FileNotFoundError(source_name)
        candidate = self.data_root / "images" / _basename(source_name)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def read_bytes(self, source_name: str) -> bytes:
        return self.path(source_name).read_bytes()

    def close(self) -> None:
        return None


class TarImageSource:
    """Czyta obrazy wprost z istniejącego TAR; niczego nie wypakowuje."""

    def __init__(self, archive: Path):
        self.archive = Path(archive)
        self.tar = tarfile.open(self.archive, mode="r")
        self.members = {
            Path(member.name).name: member
            for member in self.tar.getmembers()
            if member.isfile() and member.name.replace("\\", "/").startswith("images/")
        }

    def read_bytes(self, source_name: str) -> bytes:
        name = _basename(source_name)
        member = self.members.get(name)
        if member is None:
            raise FileNotFoundError(f"{name} nie występuje w {self.archive}")
        stream = self.tar.extractfile(member)
        if stream is None:
            raise FileNotFoundError(member.name)
        return stream.read()

    def close(self) -> None:
        self.tar.close()


def decode_resize(payload: bytes) -> np.ndarray:
    """Dekoduje istniejący obraz i skaluje identycznie jak baza K1000."""
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV nie zdekodował obrazu")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise ValueError(f"Niespójny indeks manifestu przy pozycji {index}")
    return rows
