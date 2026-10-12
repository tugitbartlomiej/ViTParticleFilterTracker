"""Ręcznie policzone momenty, surowy odczyt i niezależne grafy gradientu."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from czastki import DifferentiableReader, descriptor_numpy, descriptor_torch, extract_probe
from sonda import LABEL_NAMES, create_probe

torch.set_num_threads(1)


class Nested:
    def __init__(self, tensors, mask):
        self.tensors, self.mask = tensors, mask


class WindowedDinov2WithRegistersBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2.))
        self.bias = torch.nn.Parameter(torch.tensor(5.))

    def forward(self, x):
        f = torch.nn.functional.adaptive_avg_pool2d(x, (24, 24)).repeat(1, 128, 1, 1)
        return SimpleNamespace(feature_maps=[f * self.gain + self.bias])


class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = WindowedDinov2WithRegistersBackbone()
        self.dropout = torch.nn.Dropout(.1)

    def forward(self, nested):
        self.backbone(nested.tensors)
        return {}


class ParticleTests(unittest.TestCase):
    def test_modules_import_without_heavy_dependencies(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','rfdetr','torchvision','PIL'}:
            raise RuntimeError('ciężki import: ' + fullname)
sys.meta_path.insert(0, Block())
import czastki, sonda, generowanie
'''
        run = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_weighted_moments_have_hand_computed_values(self):
        feature = np.array([[[[1., 3.], [5., 7.]]]])
        mask = np.array([[[[1., 1.], [0., 0.]]]])
        expected = [[2., 6., 1., 0.]]
        np.testing.assert_allclose(descriptor_numpy(feature, mask), expected, atol=1e-12)
        result = descriptor_torch(torch.tensor(feature), torch.tensor(mask))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-12)
        # Stała cecha ma zerowe momenty, również dla nierównych wag maski.
        soft = np.array([[[[.1, .4], [.8, .2]]]])
        np.testing.assert_allclose(descriptor_numpy(feature * 0 + 7, soft), [[7, 7, 0, 0]], atol=1e-12)
        with self.assertRaises(ValueError):
            descriptor_numpy(feature, mask * 0)

    def test_raw_features_multiple_graphs_and_reader_does_not_change_mode(self):
        net = ToyNet().train()
        mask = torch.zeros(1, 1, 24, 24)
        mask[..., :12, :] = 1
        x = torch.full((1, 3, 24, 24), .25, requires_grad=True)
        y = torch.full((1, 3, 24, 24), .5, requires_grad=True)
        with DifferentiableReader(net, [0]*3, [1]*3, nested_tensor_factory=Nested) as reader:
            a = reader.descriptors(x, mask)
            b = reader.descriptors(y, mask)
            self.assertEqual(a.shape, (1, 1536))
            torch.testing.assert_close(a[:, :768], torch.full((1, 768), 5.5))
            (a.square().sum() + b.square().sum()).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum()), 0)
            self.assertGreater(float(y.grad.abs().sum()), 0)
            self.assertTrue(net.training)
        self.assertFalse(net.backbone._forward_hooks)

    def test_moment_gradient_matches_finite_differences(self):
        f = torch.tensor([[[[1., 3., -2.], [2., -1., 7.]]]], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([[[[.1, .4, .8], [.9, .2, .3]]]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda x: descriptor_torch(x, m), (f,)))

    def test_extraction_preserves_mixed_modes_and_saves_first_original(self):
        net = ToyNet().train()
        net.dropout.eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.full((24, 24, 3), 80, np.uint8)).save(root/'x.png')
            mask = np.zeros((24, 24), np.uint8)
            mask[:12] = 255
            Image.fromarray(mask).save(root/'m.png')
            records = [dict(image='x.png', mask='m.png', group_id='g', role='fit',
                            variant=v, labels=[1, 0, 1, 0, 1, .3, 1, 0, 0, 0],
                            label_names=list(LABEL_NAMES), split='train')
                       for v in ['black', 'original']]
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(records))
            data = extract_probe(net, [0]*3, [1]*3, manifest, out=root/'result.npz', batchsize=2,
                                 save_particles=True, nested_tensor_factory=Nested)
            self.assertEqual(data['descriptors'].shape, (2, 1536))
            self.assertEqual(data['labels'].shape, (2, len(LABEL_NAMES)))
            self.assertEqual(data['label_names'].tolist(), list(LABEL_NAMES))
            self.assertTrue(net.training)
            self.assertFalse(net.dropout.training)
            with np.load(root/'result.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['tokens'].shape, (576, 384))
                self.assertEqual(int(saved['particle_row']), 1)
                self.assertEqual(saved['label_names'].tolist(), list(LABEL_NAMES))
            self.assertFalse(net.backbone._forward_hooks)
            records[0]['image'] = 'missing.png'
            manifest.write_text(json.dumps(records))
            with self.assertRaises(FileNotFoundError):
                extract_probe(net, [0]*3, [1]*3, manifest, nested_tensor_factory=Nested)
            self.assertTrue(net.training)
            self.assertFalse(net.dropout.training)
            self.assertFalse(net.backbone._forward_hooks)

    def test_changed_probe_image_is_rejected_before_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.full((24,24,3), 80, np.uint8)
            Image.fromarray(rgb).save(root/'x.png')
            mask = np.zeros((24,24), np.uint8)
            mask[:12] = 255
            Image.fromarray(mask).save(root/'m.png')
            rows = [dict(image='x.png', mask='m.png', image_sha256=hashlib.sha256((root/'x.png').read_bytes()).hexdigest(),
                         group_id='g', role='fit', variant='original', labels=[1,0,1,0,1,.3,1,0,0,0],
                         label_names=list(LABEL_NAMES))]
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(rows))
            Image.fromarray(rgb+1).save(root/'x.png')
            net = ToyNet()
            with self.assertRaisesRegex(ValueError, 'Zmieniony'):
                extract_probe(net, [0]*3, [1]*3, manifest, nested_tensor_factory=Nested)
            self.assertFalse(net.backbone._forward_hooks)

    def test_label_names_are_verified_before_reading_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'manifest.json'
            canonical = ['presence', 'sinangle', 'cosangle', 'logscale', 'lengthfactor',
                         'backgroundbrightness', 'foreground_brightness', 'dx', 'dy', 'perspective_strength']
            wrong = canonical.copy()
            wrong[7], wrong[8] = wrong[8], wrong[7]
            net = ToyNet()
            row = dict(image='missing.png', mask='missing-mask.png', group_id='g', role='fit',
                       variant='original', labels=[1,0,1,0,1,.3,1,0,0,0], label_names=canonical)
            for names in (None, wrong, canonical[:6]):
                with self.subTest(names=names):
                    manifest.write_text(json.dumps([{**row, 'label_names': names}]))
                    with self.assertRaisesRegex(ValueError, 'label_names|LABEL_NAMES'):
                        extract_probe(net, [0]*3, [1]*3, manifest, nested_tensor_factory=Nested)
            manifest.write_text(json.dumps([row]))
            (root/'probe.json').write_text(json.dumps({'labels': wrong}))
            with self.assertRaisesRegex(ValueError, 'LABEL_NAMES'):
                extract_probe(net, [0]*3, [1]*3, manifest, nested_tensor_factory=Nested)
            self.assertFalse(net.backbone._forward_hooks)

    def test_new_probe_extracts_all_variants_with_ten_named_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgba = np.zeros((24,24,4), np.uint8)
            rgba[..., :3] = 80
            rgba[6:18, 9:13] = [200,120,30,255]
            records = []
            for group in range(3):
                filename = f'{group}.png'
                Image.fromarray(rgba).save(root/filename)
                records.append(dict(file=filename, frame_id=f'f{group}', group_id=f'g{group}', split='train'))
            source = root/'MANIFEST.json'
            source.write_text(json.dumps({'records': records}))
            manifest = create_probe(source, root, root/'probe', n_groups=3)
            result = extract_probe(ToyNet(), [0]*3, [1]*3, manifest, batchsize=8,
                                   nested_tensor_factory=Nested)
            self.assertEqual(result['descriptors'].shape, (48,1536))
            self.assertEqual(result['labels'].shape, (48,10))
            self.assertEqual(result['label_names'].tolist(), list(LABEL_NAMES))
            self.assertTrue(np.isfinite(result['descriptors']).all())
            self.assertEqual(result['tokens'].shape, (576,384))
            # Nazwy w probe.json mogą opisywać listę rekordów bez powtórzeń nazw.
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            for row in rows:
                row.pop('label_names')
            manifest.write_text(json.dumps(rows))
            from_sidecar = extract_probe(ToyNet(), [0]*3, [1]*3, manifest, batchsize=8,
                                        nested_tensor_factory=Nested)
            np.testing.assert_array_equal(from_sidecar['labels'], result['labels'])
            np.testing.assert_array_equal(from_sidecar['descriptors'], result['descriptors'])


if __name__ == '__main__':
    unittest.main()
