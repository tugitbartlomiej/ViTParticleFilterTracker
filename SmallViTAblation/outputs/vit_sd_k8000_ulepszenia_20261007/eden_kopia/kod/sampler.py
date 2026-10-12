"""Online, bounded-concurrency film mixing for the isolated K8000 experiment.

No image, latent, frame or video files are written. Decoding and validation use
the unchanged dane.iter_video. Only its small count/dimension/hash metadata is
cached; decoded samples are never accumulated into a frame cache.
"""
from collections import deque
import csv
import hashlib
import json
import random
import warnings

import cv2
from torch.utils.data import IterableDataset, get_worker_info

if __package__:
    from .dane import FrameCountMismatch, iter_video, load_manifest, read_annotations
else:
    from dane import FrameCountMismatch, iter_video, load_manifest, read_annotations

__all__ = ['MixedVideoFrames']


def _require_integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _rng(*parts):
    """Stable across processes/PYTHONHASHSEED; never touch the global RNG."""
    key = json.dumps(parts, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big'))


def _select_frames(frame_count, frame_stride, seed, film, epoch):
    """Pure selection: one uniform 1-based frame in each nonempty stride window.

    The final partial window is included. Epochs are independent redraws, so an
    individual frame may recur (and must recur for singleton windows). Film
    selections do not depend on ranks, workers, buffer size or traversal order.
    """
    _require_integer('frame_count', frame_count, 1)
    _require_integer('frame_stride', frame_stride, 1)
    _require_integer('epoch', epoch, 0)
    rng = _rng('frames', int(seed), str(film), epoch)
    return frozenset(rng.randint(start, min(start + frame_stride - 1, frame_count))
                     for start in range(1, frame_count + 1, frame_stride))


def _shard_films(rows, rank, world_size, worker_id, num_workers):
    """Pure, stable rank-then-worker partition; each film has exactly one owner.

    Partition before shuffling. Different ranks may have different worker counts
    without overlaps, provided every rank uses the same manifest and split.
    """
    _require_integer('world_size', world_size, 1)
    _require_integer('rank', rank, 0)
    _require_integer('num_workers', num_workers, 1)
    _require_integer('worker_id', worker_id, 0)
    if rank >= world_size or worker_id >= num_workers:
        raise ValueError('Require rank < world_size and worker_id < num_workers')
    return list(rows[rank::world_size][worker_id::num_workers])


class MixedVideoFrames(IterableDataset):
    """Random-window frames, interleaved across at most film_buffer active films.

    API: MixedVideoFrames(manifest, split='train', rank=0, world_size=1,
        seed=20261007, frame_stride=600, film_buffer=8, repeat=True,
        epoch_offset=0).

    ``manifest`` is a JSON-list path or rows accepted by dane.load_manifest.
    Each iterator starts at epoch_offset. repeat=False yields one complete local
    epoch; repeat=True increments the local epoch after all eligible shard films
    finish. Recreating an iterator resets it to epoch_offset; resume a whole epoch
    by constructing a dataset with that offset (this is not a mid-epoch resume).

    Films are shuffled each epoch. In each round, every active slot yields at most
    one sample in a freshly shuffled order. Exhausted/rejected films are replaced
    immediately from the remaining films. This favors distinct films in batches,
    but uniqueness across round boundaries or at the end of a shard is not
    guaranteed. Each non-rejected film contributes ceil(CSV_count/frame_stride)
    samples per epoch, including one from a film shorter than frame_stride.

    Samples pass through unchanged: image float32 [3,384,576], vit_image float32
    [3,320,480], bbox float32 [4] normalized xyxy, has_tool bool, film str and
    frame int (1-based). has_tool=False denotes UNKNOWN annotation, never a
    negative detection label; the caller must mask auxiliary detection losses.
    Missing CSV rows remain fatal validation errors in the existing reader.

    RAM per iterator: B=min(film_buffer, eligible shard films) live decoders,
    their MP4 bytes/BytesIO, annotations, transient RGB/resized frames, and
    O(sum(ceil(N_f/frame_stride))) selected integers for those B films. The
    existing reader loads each compressed MP4 into RAM, so film_buffer is a
    concurrency bound, NOT a fixed byte budget. There is no decoded-frame queue.
    Manifest/shard/rejected IDs and validation metadata are O(shard/manifest
    size); each visited film retains at most one validation entry once closed
    (at most two while reopening changed content). DataLoader prefetch/batches
    and consumer-held samples are additional memory, multiplied across workers.

    FrameCountMismatch rejects a film for the lifetime of this iterator, with
    one JSON RuntimeWarning. rejection_records keeps only the last 128 bounded
    records; rejection_count counts all rejections. These are process-local
    diagnostics reset on iter(), not aggregated from DataLoader workers. Other
    errors propagate. Empty/all-invalid repeating shards raise RuntimeError;
    finite ones end normally. DDP callers still need fixed steps/join handling
    because shard lengths and valid-film counts can differ between ranks.
    """
    def __init__(self, manifest, split='train', rank=0, world_size=1,
                 seed=20261007, frame_stride=600, film_buffer=8, repeat=True,
                 epoch_offset=0):
        super().__init__()
        _shard_films((), rank, world_size, 0, 1)
        _require_integer('frame_stride', frame_stride, 1)
        _require_integer('film_buffer', film_buffer, 1)
        _require_integer('epoch_offset', epoch_offset, 0)
        self.rows = [r for r in load_manifest(manifest) if r['split'] == str(split).lower()]
        self.rank, self.world_size = rank, world_size
        self.seed = int(seed)
        self.frame_stride, self.film_buffer = frame_stride, film_buffer
        self.repeat, self.epoch_offset = bool(repeat), epoch_offset
        self.rejection_records = deque(maxlen=128)
        self.rejection_count = 0

    def _film_samples(self, row, epoch, validation_cache):
        # CSV is authoritative, even if optional manifest count fields are absent.
        # Release this temporary annotation map before iter_video loads its own.
        count = len(read_annotations(row['annotation_zip'], row['video_id']))
        selected = _select_frames(count, self.frame_stride, self.seed, row['video_id'], epoch)
        try:
            yield from iter_video(row, validation_cache, selected_frames=selected)
        finally:
            # Changed contents get new hash keys in dane.iter_video. Keep only the
            # latest metadata entry so repeated changes cannot grow the cache.
            if len(validation_cache) > 1:
                key = next(reversed(validation_cache))
                value = validation_cache[key]
                validation_cache.clear()
                validation_cache[key] = value

    def _record_rejection(self, row, error, worker_id, epoch):
        record = dict(event='film_rejected_before_training', film=str(row['video_id'])[:256],
                      split=str(row['split'])[:64], rank=self.rank, worker=worker_id,
                      epoch=epoch, reason=str(error)[:2048])
        self.rejection_count += 1
        self.rejection_records.append(record)
        warnings.warn(json.dumps(record, ensure_ascii=True), RuntimeWarning, stacklevel=2)

    def __iter__(self):
        csv.field_size_limit(32 * 1024 * 1024)
        cv2.setNumThreads(1)
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        assigned = _shard_films(self.rows, self.rank, self.world_size, worker_id, num_workers)
        self.rejection_records.clear()
        self.rejection_count = 0
        if not assigned:
            if self.repeat:
                raise RuntimeError(f'No films assigned to rank {self.rank}, worker {worker_id}; reduce ranks/workers')
            return

        epoch = self.epoch_offset
        rejected = set()
        caches = {}  # one tiny validation dict per visited film, never tensors
        while True:
            order = [row for row in assigned if row['video_id'] not in rejected]
            rng = _rng('order', self.seed, epoch, self.rank, worker_id)
            rng.shuffle(order)
            pending = iter(order)
            active = {}
            emitted = False

            def fill_slot(slot):
                row = next(pending, None)
                if row is None:
                    active.pop(slot, None)
                    return False
                cache = caches.setdefault(row['video_id'], {})
                active[slot] = (row, self._film_samples(row, epoch, cache))
                return True

            try:
                for slot in range(min(self.film_buffer, len(order))):
                    fill_slot(slot)
                while active:
                    slots = list(active)
                    rng.shuffle(slots)
                    for slot in slots:
                        while slot in active:
                            row, stream = active[slot]
                            try:
                                sample = next(stream)
                            except StopIteration:
                                pass
                            except FrameCountMismatch as error:
                                rejected.add(row['video_id'])
                                self._record_rejection(row, error, worker_id, epoch)
                            else:
                                emitted = True
                                yield sample
                                del sample
                                break
                            stream.close()
                            fill_slot(slot)
            finally:
                # Includes consumer cancellation and errors in any other stream.
                for _, stream in active.values():
                    stream.close()
                active.clear()

            if not self.repeat:
                return
            if not emitted:
                raise RuntimeError(f'Assigned films produced no frames: rank={self.rank}, '
                                   f'worker={worker_id}, epoch={epoch}, rejected={len(rejected)}')
            epoch += 1
