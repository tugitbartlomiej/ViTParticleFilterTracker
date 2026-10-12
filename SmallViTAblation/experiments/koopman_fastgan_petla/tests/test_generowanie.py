"""Kontrole, zamrożenie modeli oraz rzeczywisty spadek straty w małym łańcuchu."""
from pathlib import Path
import hashlib
import importlib.util
import json
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generowanie import (DescriptorProjection, fixed_generator_output, make_control_targets,
                         optimize_latent, load_fastgan, shape_iou, generate)
from czastki import DifferentiableReader
if __package__:
    from .test_czastki import Nested, ToyNet
else:
    from test_czastki import Nested, ToyNet


class TinyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(.4))

    def forward(self, z):
        rgb = (z[:, :3] * self.gain).tanh()[..., None, None].expand(-1, -1, 24, 24)
        mask = torch.ones_like(rgb[:, :1])
        mask = mask.clone()
        mask[..., 12:, :] = -1
        # Stały, lecz niezerowy szum sprawdza powtarzalność i ochronę RNG.
        return [torch.cat([rgb + .001*torch.randn_like(rgb), mask], dim=1)]


class GenerationTests(unittest.TestCase):
    def projection(self):
        directions = np.zeros((1536, 1))
        directions[0, 0] = 1
        return DescriptorProjection(np.zeros(1536), np.ones(1536), directions, np.ones(1))

    def test_scaled_projection_and_random_control_norm(self):
        directions = np.zeros((1536, 2))
        directions[:2] = np.eye(2)
        p = DescriptorProjection(np.ones(1536), np.full(1536, 2), directions, [2, 3])
        np.testing.assert_allclose(p.project_numpy(np.full((1,1536), 5)), [[2, 2]])
        c = make_control_targets(np.array([1., 2.]), np.array([4., 6.]), [2., 3.], seed=42)
        np.testing.assert_array_equal(c['current'], [1, 2])
        np.testing.assert_array_equal(c['target'], [4, 6])
        self.assertAlmostEqual(np.linalg.norm((c['random_norm']-c['current'])/[2,3]),
                               np.linalg.norm(np.array([3,4])/[2,3]))
        zero = make_control_targets([1., 2.], [1., 2.], [2., 3.])
        np.testing.assert_array_equal(zero['random_norm'], [1., 2.])
        with self.assertRaises(ValueError):
            DescriptorProjection(np.zeros(1536), np.zeros(1536), directions, [1, 1])

    def test_fixed_noise_reproducible_without_rng_side_effect(self):
        gen = TinyGenerator()
        state = torch.random.get_rng_state().clone()
        z = torch.zeros(1, 3, requires_grad=True)
        a = fixed_generator_output(gen, z, seed=23)
        b = fixed_generator_output(gen, z, seed=23)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        a.sum().backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_tiny_chain_improves_loss_without_changing_weights_or_modes(self):
        net, generator = ToyNet().train(), TinyGenerator().train()
        net.dropout.eval()
        weights = [{k: v.detach().clone() for k,v in model.state_dict().items()}
                   for model in (net, generator)]
        anchor = torch.full((1,3,24,24), .75)
        mask = torch.zeros(1,1,24,24)
        mask[..., :12, :] = 1
        with DifferentiableReader(net, [0]*3, [1]*3, nested_tensor_factory=Nested) as reader:
            result = optimize_latent(generator, reader, anchor, mask, self.projection(),
                                     np.array([6.5]), torch.zeros(1,3), steps=24, lr=.15)
        self.assertLess(result['best']['total'], result['history'][0]['total'] * .2)
        self.assertLess(result['best']['feature'], result['history'][0]['feature'])
        self.assertEqual(len(result['history']), 25)
        self.assertEqual(result['gradient_steps'], 24)
        self.assertTrue(result['gradients_finite'])
        self.assertGreater(result['max_gradient_norm'], 0)
        self.assertTrue(result['feature_gradient_finite'])
        self.assertTrue(result['feature_gradient_required'])
        self.assertGreater(result['feature_gradient_max_norm'], 0)
        self.assertEqual(result['feature_gradient_max_norm'],
                         max(row['feature_gradient_norm'] for row in result['history']))
        self.assertEqual(shape_iou(result['rgba'][:,3:4], mask), 1)
        for model, before in zip((net, generator), weights):
            for k,v in model.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            self.assertTrue(model.training)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(net.dropout.training)

    def test_disconnected_features_rejected_despite_nonzero_prior_gradient(self):
        net, generator = ToyNet().train(), TinyGenerator().train()
        net.dropout.eval()
        mask = torch.zeros(1,1,24,24)
        mask[..., :12, :] = 1
        z = torch.ones(1,3, requires_grad=True)
        rgba = fixed_generator_output(generator, z)
        anchor = rgba[:, :3].detach()
        with DifferentiableReader(net, [0]*3, [1]*3, nested_tensor_factory=Nested) as reader:
            # Pozostałe składniki mają tu zerowe pochodne; tylko zprior
            # daje dodatni gradient, choć cel cech jest całkowicie odłączony.
            code = self.projection().project_torch(reader.descriptors(rgba[:, :3], mask)).detach()
            feature = (code-10).square().mean()
            self.assertGreater(float(feature), 1)
            total = (feature + .1*(rgba[:, :3]-anchor).abs().mean()
                     + (rgba[:, 3:4]-mask).abs().mean() + .001*z.square().mean())
            total_gradient, = torch.autograd.grad(total, z)
            torch.testing.assert_close(total_gradient, torch.full_like(z, .002/3))
            self.assertGreater(float(total_gradient.norm()), 0)
            projections = {
                'detached': lambda d: d[:, :1].detach(),
                'unused_z': lambda d: d[:, :1].detach().requires_grad_(True),
                'connected_zero': lambda d: d[:, :1]*0 + 6,
            }
            for kind, project in projections.items():
                with self.subTest(kind=kind):
                    projection = self.projection()
                    with patch.object(projection, 'project_torch', side_effect=project):
                        with self.assertRaisesRegex(RuntimeError, 'gradient.*straty cech'):
                            optimize_latent(generator, reader, anchor, mask, projection,
                                            [10.], z.detach(), steps=1)
                    self.assertTrue(net.training)
                    self.assertFalse(net.dropout.training)
                    self.assertTrue(generator.training)
                    self.assertTrue(all(p.requires_grad for p in net.parameters()))
                    self.assertTrue(all(p.requires_grad for p in generator.parameters()))

    def test_zero_feature_gradient_allowed_only_within_target_tolerance(self):
        mask = torch.zeros(1,1,24,24)
        mask[..., :12, :] = 1
        projection = self.projection()
        with DifferentiableReader(ToyNet(), [0]*3, [1]*3, nested_tensor_factory=Nested) as reader:
            for target in ([0.], [1e-7]):
                with self.subTest(target=target), \
                     patch.object(projection, 'project_torch', side_effect=lambda d: d[:, :1]*0):
                    result = optimize_latent(TinyGenerator(), reader, torch.ones(1,3,24,24)*.5,
                                             mask, projection, target, torch.ones(1,3), steps=2)
                self.assertEqual(result['feature_gradient_max_norm'], 0)
                self.assertTrue(result['feature_gradient_finite'])
                self.assertFalse(result['feature_gradient_required'])
                self.assertGreater(result['max_gradient_norm'], 0)
                self.assertTrue(all(row['feature_gradient_norm'] == 0 for row in result['history']))

    def test_nonfinite_feature_gradient_rejected_before_total_gradient(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return torch.full_like(gradient, float('nan'))

        mask = torch.zeros(1,1,24,24)
        mask[..., :12, :] = 1
        projection = self.projection()
        with DifferentiableReader(ToyNet(), [0]*3, [1]*3, nested_tensor_factory=Nested) as reader, \
             patch.object(projection, 'project_torch', side_effect=lambda d: BadGradient.apply(d[:, :1])):
            with self.assertRaisesRegex(FloatingPointError, 'gradient.*straty cech'):
                optimize_latent(TinyGenerator(), reader, torch.ones(1,3,24,24)*.5,
                                mask, projection, [10.], torch.ones(1,3), steps=1)

    def test_final_disconnection_is_not_hidden_by_earlier_nonzero_gradient(self):
        mask = torch.zeros(1,1,24,24)
        mask[..., :12, :] = 1
        projection = self.projection()
        project = projection.project_torch
        calls = 0

        def disconnect_later(descriptors):
            nonlocal calls
            calls += 1
            code = project(descriptors)
            return code if calls == 1 else code.detach()

        with DifferentiableReader(ToyNet(), [0]*3, [1]*3, nested_tensor_factory=Nested) as reader, \
             patch.object(projection, 'project_torch', side_effect=disconnect_later):
            with self.assertRaisesRegex(RuntimeError, 'gradient.*straty cech'):
                optimize_latent(TinyGenerator(), reader, torch.ones(1,3,24,24)*.5,
                                mask, projection, [10.], torch.ones(1,3), steps=1)

    def test_untrained_or_architecture_mismatch_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'models.py'
            source.write_text('raise AssertionError("Nie importuj przed weryfikacją")')
            ckpt = root/'bad.pth'
            torch.save({'ema': {}, 'step': 0, 'config': {}}, ckpt)
            with self.assertRaises(ValueError):
                load_fastgan(ckpt, root, 'cpu')
            torch.save({'ema': {'some_weight': torch.ones(1)}, 'step': 10, 'config': dict(channels=4, size=256, width=64,
                         architecture_sha256='wrong')}, ckpt)
            with self.assertRaisesRegex(ValueError, 'Skrót'):
                load_fastgan(ckpt, root, 'cpu')

    def test_verified_ema_is_loaded_strictly_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'models.py'
            source.write_text('''
import torch
from torch import nn
from torch.nn.utils import spectral_norm
class Generator(nn.Module):
    def __init__(self, ngf, nz, nc, im_size):
        super().__init__()
        assert (ngf,nz,nc,im_size)==(64,256,4,256)
        self.init=nn.Module()
        self.init.init=nn.Sequential(spectral_norm(nn.ConvTranspose2d(nz,1,1,bias=False)))
        self.to_big=spectral_norm(nn.Conv2d(1,nc,1,bias=False))
''')
            spec = importlib.util.spec_from_file_location('tiny_vendor', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ema = module.Generator(64,256,4,256).state_dict()
            config = dict(channels=4, size=256, width=64,
                          architecture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          data_manifest_sha256='a'*64)
            ckpt = root/'ema.pth'
            torch.save(dict(step=7, ema=ema, config=config), ckpt)
            rng = torch.random.get_rng_state().clone()
            net, meta = load_fastgan(ckpt, root, 'cpu')
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(meta['step'], 7)
            self.assertEqual(meta['checkpoint_sha256'], hashlib.sha256(ckpt.read_bytes()).hexdigest())
            self.assertFalse(net.training)
            self.assertFalse(any(p.requires_grad for p in net.parameters()))
            for name, value in net.state_dict().items():
                torch.testing.assert_close(value, ema[name], rtol=0, atol=0)
            ema['unexpected'] = torch.ones(1)
            torch.save(dict(step=7, ema=ema, config=config), ckpt)
            with self.assertRaises(RuntimeError):
                load_fastgan(ckpt, root, 'cpu')

    def test_three_arms_save_diagnostic_artifacts_and_equal_budgets(self):
        functional = torch.nn.functional
        original_grid_sample = functional.grid_sample
        def numerics_state():
            return (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
                    torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                    torch.backends.cuda.mem_efficient_sdp_enabled())

        def check_active():
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.math_sdp_enabled())
            self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
            self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            self.assertIsNot(functional.grid_sample, original_grid_sample)

        class ContextNet(ToyNet):
            def forward(inner, nested):
                check_active()
                # Rzeczywiste math SDPA i grid_sample w ścieżce cech do z.
                pooled = functional.adaptive_avg_pool2d(nested.tensors, (4,4))
                grid = functional.affine_grid(pooled.new_tensor([[[1,0,0],[0,1,0]]]),
                                              pooled.shape, align_corners=False)
                sampled = functional.grid_sample(pooled, grid, align_corners=False)
                tokens = sampled.flatten(2).transpose(1,2).unsqueeze(1)
                attended = functional.scaled_dot_product_attention(tokens, tokens, tokens, dropout_p=0.)
                image = attended[:,0].transpose(1,2).reshape_as(pooled)
                if image.requires_grad:
                    def backward_check(gradient):
                        check_active()
                        return gradient
                    image.register_hook(backward_check)
                return super().forward(Nested(image, nested.mask))

        class CropGenerator(TinyGenerator):
            def forward(self, z):
                return [torch.nn.functional.interpolate(super().forward(z)[0], (256,256), mode='nearest')]

        net = ContextNet()
        def wrapper(**kwargs):
            self.assertEqual(kwargs['num_classes'], 1)
            self.assertEqual(kwargs['resolution'], 384)
            self.assertIs(kwargs.get('amp'), False)
            self.assertTrue(Path(kwargs['pretrain_weights']).is_file())
            return SimpleNamespace(model=SimpleNamespace(model=net), means=[0]*3, stds=[1]*3)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.full((256,256,3), 255, np.uint8)).save(root/'anchor.png')
            mask = np.zeros((256,256), np.uint8)
            mask[:128] = 255
            Image.fromarray(mask).save(root/'support.png')
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps([dict(image='anchor.png', mask='support.png', role='fit',
                split='train', variant='original', labels=[1,0,1,0,1,.5])]))
            directions = root/'directions.npz'
            projection = self.projection()
            np.savez(directions, mean=projection.mean, scale=projection.scale, projection=projection.directions)
            target = root/'target.npy'
            np.save(target, [6.5])
            detector = root/'export.pth'
            detector.write_bytes(b'checkpoint testowy')
            factory = lambda net, means, stds: DifferentiableReader(net, means, stds, nested_tensor_factory=Nested)
            before_numerics = numerics_state()
            expected_current_path = root/'current.npy'
            # RGB=1 jest reprezentowane dokładnie również po średniej FP32.
            # Próbkowanie i attention zachowują stałą; backbone liczy 2*1+5.
            np.save(expected_current_path, [7.])
            with patch.dict(sys.modules, {'rfdetr': SimpleNamespace(RFDETRNano=wrapper)}), \
                 patch('generowanie.load_fastgan', return_value=(CropGenerator(), {'weights':'ema'})), \
                 patch('generowanie.DifferentiableReader', side_effect=factory), \
                 torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                report = generate(detector, root/'g.pth', root, manifest, directions, target,
                                  root/'out', 'cpu', maxsteps=3, restarts=2,
                                  expected_current_npy=expected_current_path)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
            self.assertEqual(numerics_state(), before_numerics)
            self.assertIs(functional.grid_sample, original_grid_sample)
            self.assertEqual(report['numerics']['matmul_precision'], 'highest')
            self.assertFalse(report['numerics']['amp'])
            self.assertFalse(report['numerics']['tf32'])
            self.assertEqual(report['numerics']['sdpa_backend'], 'math')
            saved_numerics = json.loads((root/'out'/'NUMERICS.json').read_text())
            self.assertEqual(saved_numerics, report['numerics'])
            self.assertEqual(json.loads((root/'out'/'report.json').read_text())['numerics'], saved_numerics)
            self.assertTrue(report['expected_current_check']['passed'])
            self.assertLessEqual(report['expected_current_check']['scaled_rmse'], 1e-5)
            self.assertEqual(report['expected_current_check']['anchor_index'], 0)
            self.assertEqual(report['classification'], 'diagnostic')
            self.assertFalse(report['labeled_generated_training'])
            self.assertFalse(report['semantic_metal_verified'])
            self.assertEqual(set(report['arms']), {'target','current','random_norm'})
            starts = []
            for name, arm in report['arms'].items():
                self.assertEqual(arm['gradient_budget'], 6)
                self.assertEqual(len(arm['attempts']), 2)
                self.assertTrue((root/'out'/arm['best_image']).is_file())
                self.assertEqual(len(arm['attempts'][0]['history']), 4)
                self.assertEqual(arm['attempts'][0]['gradient_steps'], 3)
                for attempt in arm['attempts']:
                    self.assertGreater(attempt['feature_gradient_max_norm'], 0)
                    self.assertTrue(attempt['feature_gradient_finite'])
                    self.assertTrue(attempt['gates']['feature_gradient_finite'])
                    self.assertTrue(attempt['gates']['feature_gradient_nonzero'])
                    saved_stats = json.loads((root/'out'/name/f"restart_{attempt['restart']:02d}.json").read_text())
                    self.assertEqual(saved_stats['feature_gradient_max_norm'], attempt['feature_gradient_max_norm'])
                    self.assertTrue(saved_stats['feature_gradient_finite'])
                    self.assertTrue(saved_stats['gates']['feature_gradient_finite'])
                with np.load(root/'out'/name/'restart_00.npz', allow_pickle=False) as saved:
                    starts.append(saved['z0'])
            for z in starts[1:]:
                np.testing.assert_array_equal(z, starts[0])
            self.assertEqual(json.loads((root/'out'/'status.json').read_text())['state'], 'complete')
            self.assertFalse(net.backbone._forward_hooks)
            # Cel już osiągnięty: pochodna straty cech jest poprawnie zerowa.
            # Bramka nie może uznać tego za zerwanie ścieżki przy dużym błędzie.
            np.savez(directions, mean=projection.mean, scale=projection.scale,
                     projection=np.zeros_like(projection.directions))
            np.save(target, [0.])
            with patch.dict(sys.modules, {'rfdetr': SimpleNamespace(RFDETRNano=wrapper)}), \
                 patch('generowanie.load_fastgan', return_value=(CropGenerator(), {'weights':'ema'})), \
                 patch('generowanie.DifferentiableReader', side_effect=factory):
                reached = generate(detector, root/'g.pth', root, manifest, directions, target,
                                   root/'at_target', 'cpu', maxsteps=1, restarts=1)
            for arm in reached['arms'].values():
                attempt = arm['attempts'][0]
                self.assertEqual(attempt['feature_gradient_max_norm'], 0)
                self.assertFalse(attempt['feature_gradient_required'])
                self.assertTrue(attempt['gates']['feature_gradient_finite'])
                self.assertTrue(attempt['gates']['feature_gradient_nonzero'])

            def fail_reader(*args, **kwargs):
                check_active()
                raise RuntimeError('przerwanie w kontekście diagnostycznym')

            with patch.dict(sys.modules, {'rfdetr': SimpleNamespace(RFDETRNano=wrapper)}), \
                 patch('generowanie.load_fastgan', return_value=(CropGenerator(), {'weights':'ema'})), \
                 patch('generowanie.DifferentiableReader', side_effect=fail_reader):
                with self.assertRaisesRegex(RuntimeError, 'przerwanie w kontekście'):
                    generate(detector, root/'g.pth', root, manifest, directions, target,
                             root/'failed', 'cpu', maxsteps=1, restarts=1)
            self.assertEqual(numerics_state(), before_numerics)
            self.assertIs(functional.grid_sample, original_grid_sample)

    def test_reference_mismatch_is_reported_before_any_optimization(self):
        net = ToyNet()
        wrapper = SimpleNamespace(model=SimpleNamespace(model=net), means=[0]*3, stds=[1]*3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.full((256,256,3), 128, np.uint8)).save(root/'anchor.png')
            mask = np.zeros((256,256), np.uint8)
            mask[:128] = 255
            Image.fromarray(mask).save(root/'mask.png')
            row = dict(image='anchor.png', mask='mask.png', role='fit', split='train',
                       variant='original', labels=[1,0,1,0,1,.5])
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps([row, row]))
            projection = self.projection()
            # Skala różna od 1 pozwala wykryć błędne użycie surowego RMSE.
            np.savez(root/'directions.npz', mean=projection.mean, scale=projection.scale,
                     projection=projection.directions, target_scale=[4.])
            np.save(root/'target.npy', [6.5])
            expected = root/'expected.npy'
            np.save(expected, [5+2*128/255+.1])
            detector = root/'export.pth'
            detector.write_bytes(b'checkpoint testowy')
            factory = lambda net, means, stds: DifferentiableReader(net, means, stds, nested_tensor_factory=Nested)
            with patch.dict(sys.modules, {'rfdetr': SimpleNamespace(RFDETRNano=lambda **kw: wrapper)}), \
                 patch('generowanie.load_fastgan', return_value=(TinyGenerator(), {'weights':'ema'})), \
                 patch('generowanie.DifferentiableReader', side_effect=factory), \
                 patch('generowanie.optimize_latent', side_effect=AssertionError('Nie optymalizuj przy rozbieżnej referencji')):
                with self.assertRaisesRegex(RuntimeError, 'expected_current'):
                    generate(detector, root/'g.pth', root, (manifest,1), root/'directions.npz',
                             root/'target.npy', root/'out', 'cpu', maxsteps=1, restarts=1,
                             expected_current_npy=expected)
            report = json.loads((root/'out'/'report.json').read_text())
            check = report['expected_current_check']
            self.assertFalse(check['passed'])
            self.assertAlmostEqual(check['scaled_rmse'], .025, places=5)
            self.assertEqual(check['anchor_index'], 1)
            self.assertEqual(check['expected_current_sha256'], hashlib.sha256(expected.read_bytes()).hexdigest())
            self.assertEqual(json.loads((root/'out'/'CURRENT_CHECK.json').read_text()), check)
            self.assertEqual(json.loads((root/'out'/'status.json').read_text())['state'], 'failed')
            self.assertEqual(report['arms'], {})
            self.assertFalse((root/'out'/'target').exists())
            self.assertFalse(net.backbone._forward_hooks)

    def test_expected_current_requires_finite_vector_in_projected_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector = root/'export.pth'
            detector.write_bytes(b'checkpoint testowy')
            p = self.projection()
            np.savez(root/'directions.npz', mean=p.mean, scale=p.scale, projection=p.directions)
            np.save(root/'target.npy', [6.5])
            for value in ([[6.]], [6.,7.], [float('nan')]):
                with self.subTest(value=value):
                    np.save(root/'expected.npy', value)
                    with self.assertRaisesRegex(ValueError, 'expected_current'):
                        generate(detector, root/'g.pth', root, root/'missing.json', root/'directions.npz',
                                 root/'target.npy', root/'out', 'cpu', expected_current_npy=root/'expected.npy')

    def test_cli_forwards_optional_expected_current_without_changing_budget(self):
        script = Path(__file__).resolve().parents[1]/'generator_cli.py'
        base = [str(script), '--detector', 'rf.pth', '--generator', 'g.pth', '--vendor', 'vendor',
                '--probe', 'manifest.json', '--anchor', '2', '--directions', 'directions.npz',
                '--target', 'target.npy', '--out', 'out', '--device', 'cpu', '--steps', '3', '--restarts', '1']
        for option, expected in [([], None), (['--expected-current', 'current_2.npy'], 'current_2.npy')]:
            with self.subTest(option=option), patch.object(sys, 'argv', base+option), \
                 patch('generowanie.generate') as called:
                runpy.run_path(str(script), run_name='__main__')
                self.assertEqual(called.call_args.kwargs['expected_current_npy'], expected)
                self.assertEqual(called.call_args.args[3], ('manifest.json', 2))
                self.assertEqual(called.call_args.args[-2:], (3, 1))

    def test_failure_restores_parameters_and_training_modes(self):
        net, generator = ToyNet().train(), TinyGenerator().train()
        net.dropout.eval()
        mask = torch.zeros(1,1,24,24)
        mask[..., :12, :] = 1
        with DifferentiableReader(net, [0]*3, [1]*3, nested_tensor_factory=Nested) as reader:
            with patch.object(reader, 'descriptors', side_effect=RuntimeError('przerwanie odczytu')):
                with self.assertRaisesRegex(RuntimeError, 'przerwanie'):
                    optimize_latent(generator, reader, torch.ones(1,3,24,24)*.5, mask,
                                    self.projection(), [6.5], torch.zeros(1,3), steps=1)
        self.assertTrue(net.training)
        self.assertFalse(net.dropout.training)
        self.assertTrue(generator.training)
        self.assertTrue(all(p.requires_grad for p in net.parameters()))
        self.assertTrue(all(p.requires_grad for p in generator.parameters()))


if __name__ == '__main__':
    unittest.main()
