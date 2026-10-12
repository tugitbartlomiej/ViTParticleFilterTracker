"""CPU-only checks; run python -B -m unittest discover -s <this directory> -p test_evaluation.py."""
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
torch.set_num_threads(1)


class TinyVAE(torch.nn.Module):
    def encode(self, image):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: image))

    def decode(self, latent):
        return SimpleNamespace(sample=latent)


class TinyTeacher(torch.nn.Module):
    def forward(self, image):
        latent = image.mean((1, 2, 3)).view(-1, 1, 1).expand(-1, 600, 768)
        return torch.zeros(image.shape[0], 20, 30, 5), latent


class TinyDenoiser(torch.nn.Module):
    def __init__(self, conditioned=False):
        super().__init__()
        self.conditioned = conditioned

    def forward(self, noise, timestep, condition):
        if condition.shape != (1, 600, 768) or torch.is_grad_enabled() or self.training:
            raise AssertionError('Evaluation changed full conditioning or train/no_grad contract')
        return torch.zeros_like(noise) + (condition.mean() * .02 if self.conditioned else 0)


def tiny_sample(film, level, has_tool=False):
    return dict(film=film, frame=33, image=torch.full((3, 12, 12), level),
                vit_image=torch.full((3, 320, 480), level),
                bbox=torch.tensor([0., 0., 1., 1.]), has_tool=has_tool)


def module(name):
    if not (Path(__file__).parent / (name + '.py')).exists():
        raise AssertionError(f'{name}.py API has not been implemented')
    return importlib.import_module(name)


class EvaluationTests(unittest.TestCase):
    def test_mask_maps_xy_to_row_major_600_tokens_and_rejects_invalid_gt(self):
        ev = module('evaluation')
        mask = ev.roi_mask([0, 0, 1 / 30, 1 / 20], True)
        self.assertEqual(mask.shape, (600,))
        self.assertEqual(mask.nonzero().flatten().tolist(), [0])
        self.assertEqual(ev.roi_mask([29 / 30, 19 / 20, 1, 1], True).nonzero().flatten().tolist(), [599])
        self.assertEqual(ev.roi_mask([0, 0, 1, 1], True).sum().item(), 600)
        for box in ([0, 0, 0, 0], [-.1, 0, 1, 1], [0, 0, math.nan, 1], [0, 0, 1.1, 1]):
            self.assertIsNone(ev.roi_mask(box, True))
        self.assertIsNone(ev.roi_mask([0, 0, 1, 1], False))

    def test_aggregation_weights_seeds_then_frames_then_films(self):
        ev = module('evaluation')
        records = [dict(film='A', frame=1, seed=1, full_relative_mse=0),
                   dict(film='A', frame=1, seed=2, full_relative_mse=2),
                   dict(film='A', frame=2, seed=1, full_relative_mse=3, roi_relative_mse=8),
                   dict(film='B', frame=1, seed=1, full_relative_mse=10, roi_relative_mse=4)]
        result = ev.aggregate_records(records)
        self.assertEqual(result['films'], 2)
        self.assertEqual(result['film_mean']['full_relative_mse'], 6)
        self.assertEqual(result['film_mean']['roi_relative_mse'], 6)
        self.assertIsNone(result['film_mean']['wrong_condition_mse'])
        self.assertNotIn('seed', result['film_mean'])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            ev.aggregate_records([records[0], records[0]])
        with self.assertRaises((ValueError, FloatingPointError)):
            ev.aggregate_records([dict(film='A', frame=1, seed=1, full_relative_mse=math.nan)])

    def test_wrong_controls_rotate_films_and_single_film_is_absent(self):
        ev = module('evaluation')
        samples = [dict(film=f, frame=n) for f in ('A', 'B', 'C') for n in (1, 2)]
        self.assertEqual(ev.wrong_condition_indices(samples), [2, 3, 4, 5, 0, 1])
        self.assertEqual(ev.wrong_condition_indices(samples[:2]), [None, None])

    def test_fixed_samples_even_spread_and_full_iterator_validation(self):
        ev = module('evaluation')
        rows = [dict(video_id=f'film{i}', split='val', frame_count=100,
                     zip_path='unused', member=f'film{i}.mp4', annotation_zip='unused') for i in range(7)]
        completed = []

        def video(row, selected_frames=None):
            for frame in sorted(selected_frames):
                yield dict(film=row['video_id'], frame=frame)
            completed.append(row['video_id'])
            if row['video_id'] == 'film3':
                raise ValueError('late second-pass failure')

        with patch.object(ev, 'iter_video', video):
            samples, rejected = ev.fixed_samples(rows[::-1], 'VAL', films=3)
        self.assertEqual(completed, ['film0', 'film3', 'film6'])
        self.assertEqual([(s['film'], s['frame']) for s in samples],
                         [('film0', 33), ('film0', 67), ('film6', 33), ('film6', 67)])
        self.assertEqual([r['film'] for r in rejected], ['film3'])
        for fractions in ((), (-.1,), (1.1,), (math.nan,)):
            with self.assertRaises(ValueError):
                ev.fixed_samples(rows, 'val', fractions=fractions)
        with self.assertRaises(ValueError):
            ev.fixed_samples(rows, 'val', films=0)

    def test_full_metrics_include_last_channel_and_roi(self):
        ev = module('evaluation')
        target = torch.ones(1, 600, 768)
        actual = target.clone()
        actual[:, 599, 767] = 3
        result = ev.latent_metrics(actual, target, ev.roi_mask([29 / 30, 19 / 20, 1, 1], True))
        self.assertAlmostEqual(result['full_relative_mse'], 4 / (600 * 768), places=10)
        self.assertAlmostEqual(result['roi_relative_mse'], 4 / 768, places=8)

    def test_eval_cancellation_restores_mixed_training_states_and_grad_flags(self):
        ev = module('evaluation')
        vae = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
        vae.train(); vae[1].eval()
        net = torch.nn.Linear(1, 1).train()
        vit = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        before = [m.training for root in (vae, net, vit) for m in root.modules()]
        with self.assertRaises(InterruptedError):
            ev.evaluate(vae, net, vit, [dict(film='A', frame=1)], 1, 'cpu', stop_check=lambda: True)
        self.assertEqual([m.training for root in (vae, net, vit) for m in root.modules()], before)
        self.assertTrue(net.weight.requires_grad)
        self.assertFalse(vit.weight.requires_grad)

    def test_real_ddim_pairing_reports_zero_change_for_ignored_condition(self):
        ev = module('evaluation')
        vae, net, vit = TinyVAE(), TinyDenoiser(), TinyTeacher()
        samples = [tiny_sample('A', .25, True), tiny_sample('B', .75)]
        result = ev.evaluate(vae, net, vit, samples, 1, 'cpu', steps=2, seeds=(11, 12))
        self.assertEqual(result['films'], 2)
        self.assertEqual(len(result['frames']), 4)
        self.assertEqual(result['film_mean']['condition_image_mse'], 0)
        self.assertEqual(result['film_mean']['relative_gain_vs_wrong_condition'], 0)
        self.assertEqual(result['controls']['status'], 'available')
        self.assertEqual(result['metric_film_counts']['roi_relative_mse'], 1)
        self.assertAlmostEqual(result['film_mean']['natural_detector_confidence'], .5)
        self.assertEqual(result['film_mean']['vae_reconstruction_full_mse'], 0)
        self.assertTrue(vae.training and net.training and vit.training)
        json.dumps(result, allow_nan=False)
        sensitive = ev.evaluate(vae, TinyDenoiser(True), vit, samples, 1, 'cpu', steps=2, seeds=(11,))
        self.assertGreater(sensitive['film_mean']['condition_image_mse'], 0)

    def test_one_film_has_missing_control_and_missing_gt_not_a_pass(self):
        ev = module('evaluation')
        result = ev.evaluate(TinyVAE(), TinyDenoiser(), TinyTeacher(),
                             [tiny_sample('A', .25)], 1, 'cpu', steps=2, seeds=(11,))
        self.assertEqual(result['controls']['status'], 'unavailable')
        for key in ('roi_relative_mse', 'wrong_condition_mse', 'condition_image_mse',
                    'reference_layout_iou_not_generated_ground_truth'):
            self.assertIsNone(result['film_mean'][key])
        self.assertFalse(result['frames'][0]['wrong_condition_available'])

    def test_invalid_sampler_and_scale_rejected_before_inference(self):
        ev = module('evaluation')
        for kwargs in (dict(steps=0), dict(steps=1001), dict(steps=1.5),
                       dict(spacing='other'), dict(seeds=()), dict(seeds=(1, 1)), dict(seeds=(-1,))):
            with self.assertRaises(ValueError):
                ev.evaluate(TinyVAE(), TinyDenoiser(), TinyTeacher(), [], 1, 'cpu', **kwargs)
        for scale in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                ev.evaluate(TinyVAE(), TinyDenoiser(), TinyTeacher(), [], scale, 'cpu')


def profile(steps, spacing, errors, elapsed):
    records = [dict(film=f, frame=33, seed=s, full_relative_mse=error)
               for f, error in zip(('A', 'B', 'C'), errors) for s in (20261007, 20261008)]
    return dict(steps=steps, spacing=spacing, elapsed_seconds=elapsed,
                val=module('evaluation').aggregate_records(records))


class SelectionTests(unittest.TestCase):
    def test_atomic_metrics_reject_nan_without_replacing_previous_result(self):
        diagnostic = module('diagnostyka')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'SAMPLER_SELECTION.json'
            diagnostic.atomic_json(path, dict(steps=50, spacing='leading'))
            with self.assertRaises(ValueError):
                diagnostic.atomic_json(path, dict(score=math.nan))
            self.assertEqual(json.loads(path.read_text())['steps'], 50)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['SAMPLER_SELECTION.json'])

    def test_fastest_profile_inside_two_percent_not_fastest_outside(self):
        select = module('diagnostyka').select_profile
        profiles = [profile(50, 'leading', [1, 1, 1], 1),
                    profile(100, 'leading', [.815, .815, .815], 3),
                    profile(200, 'trailing', [.8, .8, .8], 8),
                    profile(50, 'trailing', [.83, .83, .83], 2)]
        result = select(profiles)
        self.assertEqual((result['steps'], result['spacing']), (100, 'leading'))
        self.assertLessEqual(result['selected_val_full_relative_mse'], .8 * 1.02)

    def test_small_or_inconsistent_improvement_keeps_baseline(self):
        select = module('diagnostyka').select_profile
        for candidate in ([.99, .99, .99], [.5, .5, 1.1]):
            result = select([profile(50, 'leading', [1, 1, 1], 5),
                             profile(100, 'trailing', candidate, 1)])
            self.assertEqual((result['steps'], result['spacing']), (50, 'leading'))
        with self.assertRaises(ValueError):
            select([profile(50, 'leading', [1, 1, 1], 5), profile(1001, 'leading', [.5] * 3, 1)])

    def test_unmatched_or_single_film_never_claims_reliable_gain(self):
        select = module('diagnostyka').select_profile
        a = profile(50, 'leading', [1, 1, 1], 5)
        b = profile(100, 'trailing', [.5] * 3, 1)
        b['val']['frames'].pop()
        with self.assertRaises(ValueError):
            select([a, b])
        for p in (a, b):
            p['val'] = module('evaluation').aggregate_records(p['val']['frames'][:2])
        result = select([a, b])
        self.assertEqual(result['steps'], 50)
        self.assertFalse(result['reliable_gain'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
