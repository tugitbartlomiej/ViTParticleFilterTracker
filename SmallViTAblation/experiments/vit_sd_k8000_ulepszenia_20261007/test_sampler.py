"""CPU-only sampler tests. All video/annotation fixtures stay in memory.

Run from SmallViTAblation with python -B -m unittest
experiments.vit_sd_k8000_ulepszenia_20261007.test_sampler -v.
"""
import builtins
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
import io
import itertools
import json
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import weakref
import zipfile

import numpy as np
import torch

if __package__:
    from . import dane, sampler
else:
    import dane
    import sampler


def manifest_row(film, count=None, split='train'):
    row = dict(video_id=film, split=split, zip_path='videos.zip',
               member=f'videos/{film}.mp4', annotation_zip=f'{film}.zip')
    if count is not None:
        row['frame_count'] = count
    return row


class MemoryVideos:
    """Replace slow ZIP/video IO, keeping the sampler and manifest parser real."""
    def __init__(self, counts, rejected=(), empty=()):
        self.counts = counts
        self.rejected = set(rejected)
        self.empty = set(empty)
        self.opened = Counter()
        self.active = set()
        self.peak = 0
        self.generated = 0
        self.last_sample = None

    def annotations(self, archive, film):
        return dict.fromkeys(range(1, self.counts[film] + 1))

    def video(self, row, validation_cache=None, selected_frames=None):
        film = row['video_id']
        self.opened[film] += 1
        self.active.add(film)
        self.peak = max(self.peak, len(self.active))
        try:
            if film in self.rejected:
                raise dane.FrameCountMismatch(f'{film}: CSV/MP4 count mismatch')
            if film in self.empty:
                return
            for frame in range(1, self.counts[film] + 1):
                if selected_frames is not None and frame not in selected_frames:
                    continue
                self.generated += 1
                # Small tensors make scheduling tests cheap; real output sizes
                # and transforms are checked by the lower-level integration test.
                sample = dict(image=torch.full((3, 2, 3), frame / 255),
                              vit_image=torch.full((3, 2, 3), frame / 255),
                              bbox=torch.zeros(4), has_tool=False,
                              film=film, frame=frame)
                self.last_sample = sample
                yield sample
        finally:
            self.active.remove(film)

    @contextmanager
    def installed(self):
        with patch.object(sampler, 'read_annotations', self.annotations), \
                patch.object(sampler, 'iter_video', self.video):
            yield self


def sample_ids(samples):
    return [(s['film'], s['frame']) for s in samples]


class FrameSelectionTests(unittest.TestCase):
    def test_one_one_based_frame_in_each_window_including_short_tail(self):
        for count, stride, windows in (
                (1, 600, [(1, 1)]),
                (17, 600, [(1, 17)]),
                (600, 600, [(1, 600)]),
                (601, 600, [(1, 600), (601, 601)]),
                (1205, 600, [(1, 600), (601, 1200), (1201, 1205)]),
                (3, 1, [(1, 1), (2, 2), (3, 3)])):
            for epoch in range(10):
                with self.subTest(count=count, stride=stride, epoch=epoch):
                    frames = sorted(sampler._select_frames(count, stride, 47, 'film', epoch))
                    self.assertEqual(len(frames), len(windows))
                    for frame, (lo, hi) in zip(frames, windows):
                        self.assertLessEqual(lo, frame)
                        self.assertLessEqual(frame, hi)

    def test_reproducible_redraws_per_epoch_and_independent_films(self):
        args = (6000, 600, 20261007, 'film-a')
        first = sampler._select_frames(*args, 0)
        self.assertEqual(first, sampler._select_frames(*args, 0))
        self.assertNotEqual(first, sampler._select_frames(*args, 1))
        self.assertNotEqual(first, sampler._select_frames(6000, 600, 20261007, 'film-b', 0))
        self.assertNotEqual(first, sampler._select_frames(6000, 600, 48, 'film-a', 0))
        # Each window draws independently, not one shared offset for a film.
        self.assertGreater(len({(f - 1) % 600 for f in first}), 1)

    def test_selection_does_not_change_global_rng(self):
        state = random.getstate()
        sampler._select_frames(1201, 600, 47, 'film', 0)
        self.assertEqual(state, random.getstate())

    def test_invalid_counts_and_strides_fail(self):
        for count, stride in ((0, 600), (-1, 1), (2.5, 1), (True, 1),
                              (5, 0), (5, -1), (5, 1.5), (5, True)):
            with self.subTest(count=count, stride=stride), self.assertRaises(ValueError):
                sampler._select_frames(count, stride, 47, 'film', 0)


class ShardTests(unittest.TestCase):
    def test_disjoint_full_coverage_even_with_unequal_workers_per_rank(self):
        rows = [manifest_row(f'f{i:02}') for i in range(23)]
        seen = []
        for rank, workers in enumerate((1, 3, 2)):
            for worker in range(workers):
                shard = sampler._shard_films(rows, rank, 3, worker, workers)
                seen.extend(r['video_id'] for r in shard)
        self.assertEqual(Counter(seen), Counter(r['video_id'] for r in rows))
        self.assertEqual(len(rows), 23)

    def test_dataset_uses_rank_then_worker_and_filters_split(self):
        rows = [manifest_row(f'f{i:02}', 1, 'TRAIN') for i in range(11)]
        rows.append(manifest_row('validation-only', 1, 'val'))
        videos = MemoryVideos({r['video_id']: 1 for r in rows})
        observed = []
        with videos.installed():
            for rank in range(2):
                for worker in range(3):
                    info = SimpleNamespace(id=worker, num_workers=3)
                    with patch.object(sampler, 'get_worker_info', return_value=info):
                        ds = sampler.MixedVideoFrames(rows, rank=rank, world_size=2, repeat=False)
                        films = [s['film'] for s in ds]
                        self.assertEqual(set(films), {f'f{i:02}' for i in range(rank + 2 * worker, 11, 6)})
                        observed.extend(films)
        self.assertEqual(Counter(observed), Counter(f'f{i:02}' for i in range(11)))

    def test_invalid_shards_fail(self):
        for rank, world, worker, workers in ((-1, 2, 0, 1), (2, 2, 0, 1),
                                             (0, 0, 0, 1), (0, 1, 1, 1),
                                             (0, 1, 0, 0), (True, 2, 0, 1)):
            with self.subTest(values=(rank, world, worker, workers)), self.assertRaises(ValueError):
                sampler._shard_films([], rank, world, worker, workers)


class MixedVideoTests(unittest.TestCase):
    def test_sampling_uses_csv_length_when_manifest_omits_counts(self):
        rows = [manifest_row('short'), manifest_row('long')]
        with MemoryVideos({'short': 17, 'long': 1205}).installed():
            ids = sample_ids(sampler.MixedVideoFrames(rows, repeat=False))
        by_film = {film: sorted(f for name, f in ids if name == film) for film in ('short', 'long')}
        self.assertEqual(len(by_film['short']), 1)
        self.assertTrue(1 <= by_film['short'][0] <= 17)
        self.assertEqual(len(by_film['long']), 3)
        self.assertTrue(1 <= by_film['long'][0] <= 600)
        self.assertTrue(601 <= by_film['long'][1] <= 1200)
        self.assertTrue(1201 <= by_film['long'][2] <= 1205)

    def test_default_stride_and_explicit_epoch_resume(self):
        rows = [manifest_row(f'f{i}', 1205) for i in range(3)]
        videos = MemoryVideos({r['video_id']: 1205 for r in rows})
        with videos.installed():
            ds = sampler.MixedVideoFrames(rows)
            iterator = iter(ds)
            try:
                epoch0 = sample_ids(itertools.islice(iterator, 9))
                epoch1 = sample_ids(itertools.islice(iterator, 9))
            finally:
                iterator.close()
            resumed = sample_ids(sampler.MixedVideoFrames(rows, epoch_offset=1, repeat=False))
            replay = sample_ids(sampler.MixedVideoFrames(list(reversed(rows)), repeat=False))
        self.assertEqual(epoch1, resumed)
        self.assertEqual(epoch0, replay)
        self.assertNotEqual(sorted(epoch0), sorted(epoch1))

    def test_film_selections_do_not_depend_on_buffer_or_shard(self):
        rows = [manifest_row(f'f{i}', 1210) for i in range(9)]
        with MemoryVideos({r['video_id']: 1210 for r in rows}).installed():
            baseline = sample_ids(sampler.MixedVideoFrames(rows, film_buffer=1, repeat=False))
            shards = []
            for rank in range(3):
                shards += sample_ids(sampler.MixedVideoFrames(rows, rank=rank, world_size=3,
                                                               film_buffer=4, repeat=False))
        self.assertEqual(sorted(shards), sorted(baseline))

    def test_round_robin_is_fair_and_randomizes_round_order(self):
        rows = [manifest_row(f'f{i}', 12) for i in range(4)]
        with MemoryVideos({r['video_id']: 12 for r in rows}).installed():
            samples = list(sampler.MixedVideoFrames(rows, film_buffer=4, frame_stride=1, repeat=False))
        films = [s['film'] for s in samples]
        rounds = [films[start:start + 4] for start in range(0, len(films), 4)]
        for round_films in rounds:
            self.assertEqual(set(round_films), {'f0', 'f1', 'f2', 'f3'})
        self.assertGreater(len({tuple(r) for r in rounds}), 1)
        self.assertEqual(Counter(films), Counter(dict.fromkeys(('f0', 'f1', 'f2', 'f3'), 12)))

    def test_replacement_and_unequal_films_have_full_coverage(self):
        counts = {'a': 1, 'b': 7, 'c': 2, 'd': 5, 'e': 1, 'f': 3}
        with MemoryVideos(counts).installed() as videos:
            ids = sample_ids(sampler.MixedVideoFrames([manifest_row(f) for f in counts],
                                                      film_buffer=2, frame_stride=1, repeat=False))
        self.assertEqual(set(ids), {(film, frame) for film, count in counts.items()
                                    for frame in range(1, count + 1)})
        self.assertEqual(len(ids), sum(counts.values()))
        self.assertLessEqual(videos.peak, 2)
        self.assertFalse(videos.active)

    def test_only_requested_samples_are_decoded_and_close_releases_all_films(self):
        counts = {f'f{i}': 1000 for i in range(20)}
        with MemoryVideos(counts).installed() as videos:
            ds = sampler.MixedVideoFrames([manifest_row(f) for f in counts], film_buffer=3,
                                          frame_stride=1, repeat=True)
            iterator = iter(ds)
            try:
                for _ in range(17):
                    next(iterator)
                self.assertEqual(videos.generated, 17)
                self.assertEqual(videos.peak, 3)
                self.assertLessEqual(len(videos.opened), 3)
            finally:
                iterator.close()
            self.assertFalse(videos.active)

    def test_unknown_annotation_preserves_mask_and_exact_output_fields(self):
        with MemoryVideos({'unknown': 1}).installed() as videos:
            samples = list(sampler.MixedVideoFrames([manifest_row('unknown')], repeat=False))
        self.assertEqual(len(samples), 1)
        self.assertEqual(set(samples[0]), {'image', 'vit_image', 'bbox', 'has_tool', 'film', 'frame'})
        self.assertIs(samples[0]['image'], videos.last_sample['image'])
        self.assertIs(samples[0]['vit_image'], videos.last_sample['vit_image'])
        self.assertFalse(samples[0]['has_tool'])
        self.assertEqual(samples[0]['bbox'].tolist(), [0, 0, 0, 0])

    def test_consumed_images_are_not_accumulated_in_sampler(self):
        counts = {f'f{i}': 100 for i in range(7)}
        videos = MemoryVideos(counts)
        references = []

        def tracked_video(row, validation_cache=None, selected_frames=None):
            for sample in videos.video(row, validation_cache, selected_frames):
                references.append(weakref.ref(sample['image']))
                yield sample

        with videos.installed(), patch.object(sampler, 'iter_video', tracked_video):
            iterator = iter(sampler.MixedVideoFrames([manifest_row(f) for f in counts],
                                                     frame_stride=1, film_buffer=3))
            try:
                for _ in range(350):
                    next(iterator)
                    self.assertLessEqual(sum(ref() is not None for ref in references), 3)
            finally:
                iterator.close()
        # The test fixture alone retains its last sample after all streams close.
        videos.last_sample = None
        self.assertFalse(any(ref() is not None for ref in references))

    def test_changed_content_validation_metadata_does_not_grow_each_epoch(self):
        videos = MemoryVideos({'film': 1})
        cache_sizes = []
        counter = itertools.count()

        def changing_video(row, validation_cache=None, selected_frames=None):
            # Mirror iter_video inserting a new metadata key when bytes change.
            validation_cache[('changed-hash', next(counter))] = (1, (8, 6))
            cache_sizes.append(len(validation_cache))
            yield from videos.video(row, validation_cache, selected_frames)

        with videos.installed(), patch.object(sampler, 'iter_video', changing_video):
            iterator = iter(sampler.MixedVideoFrames([manifest_row('film')]))
            try:
                for _ in range(50):
                    next(iterator)
            finally:
                iterator.close()
        self.assertEqual(len(cache_sizes), 50)
        self.assertLessEqual(max(cache_sizes), 2)

    def test_rejected_film_warns_once_and_is_not_retried_on_later_epochs(self):
        rows = [manifest_row('bad'), manifest_row('good')]
        with MemoryVideos({'bad': 1, 'good': 1}, rejected={'bad'}).installed() as videos, \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ds = sampler.MixedVideoFrames(rows)
            iterator = iter(ds)
            try:
                ids = sample_ids(itertools.islice(iterator, 4))
            finally:
                iterator.close()
        self.assertEqual(ids, [('good', 1)] * 4)
        self.assertEqual(videos.opened['bad'], 1)
        self.assertEqual(len(caught), 1)
        record = json.loads(str(caught[0].message))
        self.assertEqual(record['film'], 'bad')
        self.assertEqual(record['rank'], 0)
        self.assertEqual(record['worker'], 0)
        self.assertIn('count mismatch', record['reason'])
        self.assertEqual(ds.rejection_count, 1)
        self.assertEqual(list(ds.rejection_records), [record])

    def test_all_rejected_finishes_finite_pass_and_raises_in_repeat_mode(self):
        for repeat in (False, True):
            with self.subTest(repeat=repeat), \
                    MemoryVideos({'bad': 1}, rejected={'bad'}).installed() as videos, \
                    warnings.catch_warnings(record=True):
                ds = sampler.MixedVideoFrames([manifest_row('bad')], repeat=repeat)
                if repeat:
                    with self.assertRaisesRegex(RuntimeError, 'no frames'):
                        next(iter(ds))
                else:
                    self.assertEqual(list(ds), [])
                self.assertEqual(videos.opened['bad'], 1)
                self.assertFalse(videos.active)

    def test_empty_decoder_does_not_spin_forever(self):
        with MemoryVideos({'empty': 1}, empty={'empty'}).installed() as videos:
            ds = sampler.MixedVideoFrames([manifest_row('empty')])
            with self.assertRaisesRegex(RuntimeError, 'no frames'):
                next(iter(ds))
        self.assertEqual(videos.opened['empty'], 1)

    def test_empty_shard_has_clear_finite_and_repeating_semantics(self):
        self.assertEqual(list(sampler.MixedVideoFrames([], repeat=False)), [])
        with self.assertRaisesRegex(RuntimeError, 'No films assigned'):
            next(iter(sampler.MixedVideoFrames([])))

    def test_diagnostics_are_bounded_but_count_all_rejections(self):
        counts = {f'bad{i:03}': 1 for i in range(300)}
        with MemoryVideos(counts, rejected=counts).installed(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ds = sampler.MixedVideoFrames([manifest_row(f) for f in counts], repeat=False)
            self.assertEqual(list(ds), [])
        self.assertEqual(ds.rejection_count, 300)
        self.assertEqual(len(caught), 300)
        self.assertLessEqual(len(ds.rejection_records), 128)
        self.assertEqual(list(ds.rejection_records),
                         [json.loads(str(w.message)) for w in caught[-len(ds.rejection_records):]])

    def test_non_count_errors_propagate_and_close_other_streams(self):
        videos = MemoryVideos({'a': 100, 'b': 100, 'c': 100})
        original_video = videos.video

        def broken_video(row, validation_cache=None, selected_frames=None):
            if row['video_id'] == 'c':
                raise RuntimeError('corrupt video')
            yield from original_video(row, validation_cache, selected_frames)

        with videos.installed(), patch.object(sampler, 'iter_video', broken_video):
            ds = sampler.MixedVideoFrames([manifest_row(f) for f in videos.counts],
                                          frame_stride=1, film_buffer=3, repeat=False)
            with self.assertRaisesRegex(RuntimeError, 'corrupt video'):
                list(ds)
            self.assertFalse(videos.active)

    def test_very_long_rejection_reason_is_bounded(self):
        def bad_video(row, validation_cache=None, selected_frames=None):
            raise dane.FrameCountMismatch('x' * 100000)

        with MemoryVideos({'bad': 1}).installed(), patch.object(sampler, 'iter_video', bad_video), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ds = sampler.MixedVideoFrames([manifest_row('bad')], repeat=False)
            self.assertEqual(list(ds), [])
        self.assertEqual(len(caught), 1)
        self.assertLessEqual(len(ds.rejection_records[0]['reason']), 2048)
        self.assertTrue(ds.rejection_records[0]['reason'])

    def test_invalid_configuration_fails_before_io(self):
        for kwargs in ({'frame_stride': 0}, {'frame_stride': True}, {'film_buffer': 0},
                       {'film_buffer': 1.5}, {'film_buffer': True}, {'rank': -1},
                       {'rank': 2, 'world_size': 2}, {'world_size': 0},
                       {'epoch_offset': -1}, {'epoch_offset': 1.5}, {'epoch_offset': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sampler.MixedVideoFrames([], **kwargs)


@contextmanager
def forbid_disk_writes():
    """Real reads allowed; fail immediately if sampler/reader creates any file."""
    original_open, original_io_open, original_os_open = builtins.open, io.open, os.open

    def checked_open(original):
        def run(file, mode='r', *args, **kwargs):
            if any(c in mode for c in 'wax+'):
                raise AssertionError(f'Disk write attempted: {file!r}, {mode!r}')
            return original(file, mode, *args, **kwargs)
        return run

    def checked_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError(f'Disk write attempted: {path!r}')
        return original_os_open(path, flags, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch('builtins.open', checked_open(original_open)))
        stack.enter_context(patch('io.open', checked_open(original_io_open)))
        stack.enter_context(patch('os.open', checked_os_open))
        for name in ('mkdir', 'makedirs', 'remove', 'unlink', 'rename', 'replace'):
            stack.enter_context(patch.object(os, name, side_effect=AssertionError(f'Forbidden os.{name}')))
        yield


def zip_bytes(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class DecoderIntegrationTests(unittest.TestCase):
    @contextmanager
    def in_memory_archives(self, annotation_frames=(1, 2, 3), decoded_count=3):
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(['filename', 'frame_number', 'cystotome_seg', 'cystotome_bbox'])
        for frame in annotation_frames:
            writer.writerow([f'film_{frame:06}.jpg', frame, '', '[1,1,2,2]' if frame == 1 else ''])
        archives = {'film.zip': zip_bytes({'annotation.csv': text.getvalue()}),
                    'videos.zip': zip_bytes({'videos/film.mp4': b'in-memory-mp4-fixture'})}
        original_zip = zipfile.ZipFile
        decodes = []

        class Frame:
            width, height = 8, 6

            def to_ndarray(self, format):
                if format != 'rgb24':
                    raise AssertionError(format)
                return np.full((6, 8, 3), 127, dtype=np.uint8)

        def decode(data):
            self.assertEqual(data, b'in-memory-mp4-fixture')
            decodes.append(1)
            for _ in range(decoded_count):
                yield Frame()

        def open_zip(path, mode='r', *args, **kwargs):
            if mode != 'r':
                raise AssertionError(f'Archive write attempted: {path}')
            return original_zip(io.BytesIO(archives[str(path)]), mode, *args, **kwargs)

        with patch.object(dane.zipfile, 'ZipFile', open_zip), \
                patch.object(dane, '_decoded_frames', decode), forbid_disk_writes():
            yield decodes

    def test_existing_reader_shapes_labels_cache_and_no_disk_writes(self):
        with self.in_memory_archives() as decodes:
            ds = sampler.MixedVideoFrames([manifest_row('film')], frame_stride=1)
            iterator = iter(ds)
            try:
                samples = list(itertools.islice(iterator, 6))
            finally:
                iterator.close()
        self.assertEqual(sample_ids(samples), [('film', 1), ('film', 2), ('film', 3)] * 2)
        self.assertEqual(len(decodes), 3)  # one preflight + one decode per epoch
        first = samples[0]
        self.assertEqual(first['image'].shape, (3, 384, 576))
        self.assertEqual(first['vit_image'].shape, (3, 320, 480))
        self.assertEqual(first['image'].dtype, torch.float32)
        self.assertTrue(torch.allclose(first['image'], torch.full_like(first['image'], 127 / 255)))
        self.assertTrue(first['has_tool'])
        self.assertTrue(torch.allclose(first['bbox'], torch.tensor([1 / 8, 1 / 6, 3 / 8, 3 / 6])))
        self.assertFalse(samples[1]['has_tool'])
        self.assertEqual(samples[1]['bbox'].tolist(), [0, 0, 0, 0])

    def test_real_count_mismatch_rejects_before_any_sample(self):
        with self.in_memory_archives(decoded_count=2), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ds = sampler.MixedVideoFrames([manifest_row('film')], frame_stride=1, repeat=False)
            self.assertEqual(list(ds), [])
        self.assertEqual(len(caught), 1)
        self.assertIn('CSV=3, decoded=2', str(caught[0].message))

    def test_missing_csv_row_is_error_instead_of_negative_label(self):
        with self.in_memory_archives(annotation_frames=(1, 3)):
            ds = sampler.MixedVideoFrames([manifest_row('film')], frame_stride=1, repeat=False)
            with self.assertRaisesRegex(ValueError, 'CSV must cover every frame'):
                list(ds)


if __name__ == '__main__':
    unittest.main()
