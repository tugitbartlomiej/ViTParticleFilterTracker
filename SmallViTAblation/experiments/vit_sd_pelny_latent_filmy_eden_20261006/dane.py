"""TRAIN video streaming in RAM; invalid alignment aborts instead of inventing frames.

Dependencies: torch, numpy, opencv-python, av (PyAV / FFmpeg).
MP4 bytes stay in BytesIO. No videos, images or latents are extracted to disk.
CSV frames are 1-based. All frames are emitted. Missing GT is has_tool=False
and bbox=zeros, an auxiliary-loss mask, never a detection negative. Output bbox
prefers valid provided cystotome_bbox xywh, with JSON polygon bounds fallback,
normalized xyxy in the original RGB frame; both resizes use the original RGB.
"""
import csv
import hashlib
import io
import json
import math
import random
import re
import zipfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class FrameCountMismatch(ValueError):
    """Reject an entire film before yielding any incorrectly paired samples."""


def load_manifest(manifest):
    """Accept a JSON-list path or a list of manifest rows."""
    rows = json.loads(Path(manifest).read_text(encoding='utf-8-sig')) if isinstance(manifest, (str, Path)) else list(manifest)
    if not isinstance(rows, list):
        raise ValueError('Manifest must be a JSON list')
    seen = set()
    result = []
    for row in rows:
        required = ('zip_path', 'member', 'video_id', 'split', 'annotation_zip')
        if not isinstance(row, dict) or any(not row.get(k) for k in required):
            raise ValueError(f'Invalid manifest row: {row!r}')
        row = dict(row)
        row['split'] = str(row['split']).lower()
        if row['video_id'] in seen:
            raise ValueError(f'Duplicate film: {row["video_id"]}')
        if Path(row['member']).stem != row['video_id']:
            raise ValueError(f'MP4 filename / film mismatch: {row}')
        seen.add(row['video_id'])
        result.append(row)
    return sorted(result, key=lambda r: r['video_id'])


def _polygon_bbox(text, where):
    """Return union xyxy for COCO flat JSON polygons; None means unknown label."""
    if not text or not text.strip():
        return None
    try:
        polygons = json.loads(text)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{where}: malformed cystotome_seg JSON') from e
    if polygons is None or polygons == []:
        return None
    if not isinstance(polygons, list):
        raise ValueError(f'{where}: cystotome_seg must be a list of polygons')
    points = []
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise ValueError(f'{where}: invalid flat polygon')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in polygon):
            raise ValueError(f'{where}: polygon coordinates must be finite numbers')
        points.extend(zip(polygon[::2], polygon[1::2]))
    a = np.asarray(points, dtype=np.float64)
    bbox = [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f'{where}: degenerate or negative polygon bbox')
    return bbox


def read_annotations(annotation_zip, video_id):
    """Read only annotation.csv from ZIP, never JPEG members; require dense 1..N."""
    with zipfile.ZipFile(annotation_zip) as archive:
        members = [n for n in archive.namelist() if Path(n).name == 'annotation.csv']
        if len(members) != 1:
            raise ValueError(f'{video_id}: expected exactly one annotation.csv in {annotation_zip}')
        with archive.open(members[0]) as binary, io.TextIOWrapper(binary, encoding='utf-8-sig', newline='') as text:
            reader = csv.DictReader(text)
            if not {'filename', 'frame_number', 'cystotome_seg', 'cystotome_bbox'}.issubset(reader.fieldnames or []):
                raise ValueError(f'{video_id}: missing CSV columns')
            annotations = {}
            for row in reader:
                try:
                    frame = int(row['frame_number'])
                except (TypeError, ValueError) as e:
                    raise ValueError(f'{video_id}: invalid CSV frame_number') from e
                if frame < 1 or frame in annotations:
                    raise ValueError(f'{video_id}: invalid/duplicate frame {frame}')
                stem = Path(row['filename']).stem
                match = re.fullmatch(re.escape(video_id) + r'_(\d+)', stem)
                if match is None or int(match.group(1)) != frame:
                    raise ValueError(f'{video_id}: filename/frame alignment mismatch at {frame}: {row["filename"]}')
                annotations[frame] = _tool_bbox(row, f'{video_id} frame {frame}')
    if not annotations or sorted(annotations) != list(range(1, len(annotations) + 1)):
        raise ValueError(f'{video_id}: CSV must cover every frame from 1 through N, including unknown labels')
    return annotations


def _provided_bbox(text, where):
    if not text or not text.strip():
        return None
    try:
        values = json.loads(text)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{where}: malformed cystotome_bbox JSON') from e
    if values is None or values == []:
        return None
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f'{where}: bbox must be [x,y,w,h]')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f'{where}: nonfinite/invalid bbox')
    x, y, w, h = map(float, values)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f'{where}: invalid provided bbox')
    return [x, y, x + w, y + h]


def _tool_bbox(row, where):
    """Prefer valid provided xywh; use polygon fallback, never create negative GT."""
    provided_error = None
    try:
        bbox = _provided_bbox(row.get('cystotome_bbox', ''), where)
    except ValueError as e:
        provided_error = e
        bbox = None
    if bbox is not None:
        return bbox
    bbox = _polygon_bbox(row.get('cystotome_seg', ''), where)
    if bbox is None and provided_error is not None:
        raise provided_error
    return bbox


def _decoded_frames(mp4_bytes):
    try:
        import av
    except ImportError as e:
        raise ImportError('VideoFrames requires PyAV: install av in the execution environment') from e
    with io.BytesIO(mp4_bytes) as buffer, av.open(buffer, mode='r') as container:
        if len(container.streams.video) != 1:
            raise ValueError('Expected exactly one MP4 video stream')
        stream = container.streams.video[0]
        # Bound CPU work per loader process; ranks/workers own independent containers.
        stream.codec_context.thread_count = 1
        yield from container.decode(stream)


def iter_video(row, validation_cache=None):
    """Preflight once per unchanged film in a worker, then single-pass repeated epochs.

    Cache holds only counts/dimensions and content hashes, not video/frame arrays.
    The first preflight catches mismatches before training receives that film.
    """
    film = row['video_id']
    annotations = read_annotations(row['annotation_zip'], film)
    if row.get('frame_count', row.get('annotation_count', len(annotations))) != len(annotations):
        raise ValueError(f'{film}: annotation_count changed since manifest build')
    with zipfile.ZipFile(row['zip_path']) as archive:
        infos = [i for i in archive.infolist() if i.filename == row['member']]
        if len(infos) != 1:
            raise ValueError(f'{film}: missing/duplicate MP4 member')
        mp4_bytes = archive.read(infos[0])  # ZIP CRC verified; bytes never written to disk.
    cache = validation_cache if validation_cache is not None else {}
    key = (str(row['zip_path']), row['member'], hashlib.sha256(mp4_bytes).hexdigest(),
           hashlib.sha256(json.dumps(annotations, sort_keys=True).encode()).hexdigest())
    count, dimensions = cache.get(key, (0, None))
    try:
        if key not in cache:
            for count, frame in enumerate(_decoded_frames(mp4_bytes), 1):
                if count not in annotations:
                    raise FrameCountMismatch(f'{film}: MP4 has more frames than CSV ({len(annotations)})')
                size = (frame.width, frame.height)
                if dimensions is None:
                    dimensions = size
                if size != dimensions or min(size) <= 0:
                    raise ValueError(f'{film}: varying/invalid video dimensions at {count}')
                bbox = annotations[count]
                if bbox is not None and (bbox[2] > size[0] or bbox[3] > size[1]):
                    raise ValueError(f'{film}: provided/fallback bbox outside decoded image at {count}')
            if count != len(annotations):
                raise FrameCountMismatch(f'{film}: CSV/MP4 frame count mismatch: CSV={len(annotations)}, decoded={count}')
            cache[key] = (count, dimensions)
        decoded = 0
        for decoded, frame in enumerate(_decoded_frames(mp4_bytes), 1):
            if decoded not in annotations:
                raise ValueError(f'{film}: second-pass alignment changed')
            bbox = annotations[decoded]
            rgb = frame.to_ndarray(format='rgb24')
            if (rgb.shape[1], rgb.shape[0]) != dimensions:
                raise ValueError(f'{film}: second-pass dimensions changed')
            image = cv2.resize(rgb, (576, 384), interpolation=cv2.INTER_AREA)
            vit_image = cv2.resize(rgb, (480, 320), interpolation=cv2.INTER_AREA)
            width, height = dimensions
            normalized = (np.asarray(bbox, dtype=np.float32) / np.asarray([width, height, width, height], dtype=np.float32)
                          if bbox is not None else np.zeros(4, dtype=np.float32))
            yield dict(image=torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255),
                       vit_image=torch.from_numpy(vit_image).permute(2, 0, 1).contiguous().float().div_(255),
                       bbox=torch.from_numpy(normalized), film=film, frame=decoded, has_tool=bbox is not None)
        if decoded != count:
            raise ValueError(f'{film}: second-pass frame count changed')
    except FrameCountMismatch:
        cache.pop(key, None)
        raise
    except Exception as e:
        cache.pop(key, None)
        raise RuntimeError(f'{film}: video/CSV validation or decoding failed: {e}') from e


class VideoFrames(IterableDataset):
    """Stable film shards by DDP rank and DataLoader worker, shuffled per local epoch.

    Output: image float32 [3,384,576], vit_image float32 [3,320,480],
    bbox float32 [4] normalized xyxy, film str, frame int (1-based), has_tool bool.
    All frames returned; missing GT has bbox=zeros and has_tool=False. Mask detection
    auxiliary loss by has_tool; reconstruction can use every frame. repeat=False
    gives one finite pass. repeat=True
    requires the training loop to set a fixed step count / DDP joining policy;
    unequal film lengths do not imply equal numbers of samples between ranks.
    Films stream sequentially within each shard. Manifest eligibility is not
    evidence that a short run (e.g. first 2000 updates) has already seen all films.
    """
    def __init__(self, manifest, split='train', rank=0, world_size=1, seed=20261006, repeat=True):
        super().__init__()
        if not isinstance(world_size, int) or world_size < 1 or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError('Require world_size >= 1 and 0 <= rank < world_size')
        self.rows = [r for r in load_manifest(manifest) if r['split'] == str(split).lower()]
        self.rank, self.world_size, self.seed, self.repeat = rank, world_size, int(seed), bool(repeat)
        self._validation_cache = {}

    def __iter__(self):
        # CSV can contain long polygon fields; field limit is process-local.
        csv.field_size_limit(32 * 1024 * 1024)
        cv2.setNumThreads(1)
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        assigned = self.rows[self.rank::self.world_size][worker_id::num_workers]
        if not assigned:
            if self.repeat:
                raise RuntimeError(f'No films assigned to rank {self.rank}, worker {worker_id}; reduce ranks/workers')
            return
        epoch = 0
        rejected = set()
        while True:
            order = list(assigned)
            random.Random(self.seed + epoch * 1000003 + self.rank * 1009 + worker_id).shuffle(order)
            emitted = False
            for row in order:
                if row['video_id'] in rejected:
                    continue
                try:
                    for sample in iter_video(row, self._validation_cache):
                        emitted = True
                        yield sample
                except FrameCountMismatch as error:
                    rejected.add(row['video_id'])
                    warnings.warn(json.dumps(dict(event='film_rejected_before_training',
                                                  film=row['video_id'], split=row['split'],
                                                  rank=self.rank, worker=worker_id,
                                                  reason=str(error))), RuntimeWarning)
            if not self.repeat:
                return
            if not emitted:
                raise RuntimeError('Assigned films produced no frames')
            epoch += 1
