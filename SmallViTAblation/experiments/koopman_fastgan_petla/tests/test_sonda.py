"""Kontrakt sondy na małych, sztucznych wycinkach ręcznej maski."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonda import LABEL_NAMES, create_probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = []
        self.rgba = np.full((16, 16, 4), 80, dtype=np.uint8)
        self.rgba[..., 3] = 0
        self.rgba[3:13, 6:9, :3] = [200, 120, 30]
        self.rgba[3:13, 6:9, 3] = 255
        for g in range(12):
            name = f'{g:02d}.png'
            Image.fromarray(self.rgba).save(self.root / name)
            self.records.append(dict(file=name, frame_id=f'f{g}', group_id=f'g{g:02d}', split='train'))
        self.manifest = self.root / 'MANIFEST.json'
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({'records': self.records}), encoding='utf-8')

    def build(self, name='probe'):
        path = create_probe(self.manifest, self.root, self.root / name)
        return path, json.loads(path.read_text(encoding='utf-8'))

    def test_deterministic_group_split_and_order_independent_selection(self):
        path, rows = self.build()
        self.records.reverse()
        self.write_manifest()
        path2, rows2 = self.build('again')
        self.assertEqual(rows, rows2)
        self.assertEqual(len(rows), 192)
        metadata = json.loads((path.parent/'probe.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['labels'], list(LABEL_NAMES))
        self.assertEqual(metadata['schema_version'], 2)
        for role, count in [('fit', 8), ('holdout', 4)]:
            self.assertEqual(len({r['group_id'] for r in rows if r['role'] == role}), count)
        self.assertFalse({r['group_id'] for r in rows if r['role'] == 'fit'} &
                         {r['group_id'] for r in rows if r['role'] == 'holdout'})
        for row in rows:
            self.assertEqual(len(row['labels']), 10)
            self.assertEqual(row['label_names'], metadata['labels'])
            for key in ('image', 'mask'):
                self.assertFalse(Path(row[key]).is_absolute())
                self.assertEqual((path.parent / row[key]).read_bytes(),
                                 (path2.parent / row[key]).read_bytes())

    def test_composites_empty_control_and_geometry(self):
        path, rows = self.build()
        variants = {r['variant']: r for r in rows if r['group_id'] == rows[0]['group_id']}
        def arr(name, kind='image'):
            with Image.open(path.parent / variants[name][kind]) as image:
                return np.array(image)
        np.testing.assert_array_equal(arr('original'), self.rgba[..., :3])
        mask = self.rgba[..., 3] > 0
        for name, color in [('black', [0, 0, 0]), ('yellow', [255, 255, 0])]:
            image = arr(name)
            np.testing.assert_array_equal(image[mask], self.rgba[..., :3][mask])
            np.testing.assert_array_equal(image[~mask], np.tile(color, ((~mask).sum(), 1)))
        self.assertFalse(arr('empty_black').any())
        np.testing.assert_array_equal(arr('empty_black', 'mask'), arr('original', 'mask'))
        self.assertEqual(variants['empty_black']['labels'][0], 0)
        self.assertEqual(variants['empty_black']['semantic_presence'], 0)
        self.assertGreater(variants['empty_black']['mask_occupancy'], 0)
        np.testing.assert_array_equal(arr('rotation90', 'mask'), np.rot90(self.rgba[..., 3]))
        matrix = np.array(variants['rotation90']['transform_matrix'])
        np.testing.assert_allclose(matrix @ [6.5, 3.5, 1], [3.5, 9.5, 1])
        np.testing.assert_allclose(variants['rotation90']['labels'][1:4], [1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(variants['scale08']['labels'][3], np.log(.8))
        self.assertEqual(variants['length08']['labels'][4], .8)
        self.assertLess(arr('length08', 'mask').sum(), arr('original', 'mask').sum())
        np.testing.assert_allclose(variants['original']['labels'][6:], [1, 0, 0, 0])

    def test_shifts_match_normalized_labels_pixels_and_forward_matrix(self):
        # Dwadzieścia pikseli daje dokładne przesunięcie o dwa piksele.
        rgba = np.full((20,20,4), 80, np.uint8)
        rgba[..., 3] = 0
        rgba[5:15, 7:10, :3] = [200,120,30]
        rgba[5:15, 7:10, 3] = 255
        for row in self.records:
            Image.fromarray(rgba).save(self.root/row['file'])
        path, rows = self.build()
        variants = {r['variant']: r for r in rows if r['group_id'] == rows[0]['group_id']}
        for variant, dx, dy, x0, y0 in [('shift_left', -.1, 0, 5, 5),
                                        ('shift_right', .1, 0, 9, 5),
                                        ('shift_up', 0, -.1, 7, 3),
                                        ('shift_down', 0, .1, 7, 7)]:
            with self.subTest(variant=variant):
                row = variants[variant]
                np.testing.assert_allclose(row['labels'][6:], [1, dx, dy, 0])
                matrix, inverse = np.array(row['transform_matrix']), np.array(row['inverse_transform_matrix'])
                np.testing.assert_allclose(matrix @ [7.5, 5.5, 1], [x0+.5, y0+.5, 1])
                np.testing.assert_allclose(inverse @ matrix, np.eye(3))
                with Image.open(path.parent/row['mask']) as image:
                    mask = np.array(image)
                expected_mask = np.zeros((20,20), np.uint8)
                expected_mask[y0:y0+10, x0:x0+3] = 255
                np.testing.assert_array_equal(mask, expected_mask)
                with Image.open(path.parent/row['image']) as image:
                    rgb = np.array(image)
                np.testing.assert_array_equal(rgb[y0:y0+10, x0:x0+3],
                                              np.broadcast_to([200,120,30], (10,3,3)))
                self.assertEqual(row['support_retained_fraction'], 1.)

    def test_foreground_lighting_clips_metal_and_preserves_background_and_mask(self):
        # Piksel półprzezroczystej krawędzi sprawdza mieszanie po clippingu.
        self.rgba[3, 6, 3] = 128
        for row in self.records:
            Image.fromarray(self.rgba).save(self.root/row['file'])
        path, rows = self.build()
        variants = {r['variant']: r for r in rows if r['group_id'] == rows[0]['group_id']}
        for name, factor, expected in [('foreground_dim', .7, [140,84,21]),
                                        ('foreground_bright', 1.3, [255,156,39])]:
            with self.subTest(variant=name):
                row = variants[name]
                np.testing.assert_allclose(row['labels'][6:], [factor,0,0,0])
                np.testing.assert_array_equal(row['transform_matrix'], np.eye(3))
                with Image.open(path.parent/row['image']) as image:
                    rgb = np.array(image)
                with Image.open(path.parent/row['mask']) as image:
                    np.testing.assert_array_equal(np.array(image), self.rgba[..., 3])
                outside, inside = self.rgba[..., 3] == 0, self.rgba[..., 3] == 255
                np.testing.assert_array_equal(rgb[outside], self.rgba[..., :3][outside])
                np.testing.assert_array_equal(rgb[inside], np.tile(expected, (inside.sum(), 1)))
                self.assertEqual(row['photometric_scope'], 'hand_mask_only')
                if name == 'foreground_bright':
                    np.testing.assert_array_equal(rgb[3, 6], [228,138,35])

    def test_perspective_has_explicit_projective_mapping_and_shared_image_mask_warp(self):
        # Czerwony kanał kopiuje maskę, więc ten sam warp musi dać równe obrazy.
        rgba = self.rgba.copy()
        rgba[..., :3] = 0
        rgba[..., 0] = rgba[..., 3]
        for row in self.records:
            Image.fromarray(rgba).save(self.root/row['file'])
        path, rows = self.build()
        row = next(r for r in rows if r['variant'] == 'perspective_mild')
        matrix, inverse = np.array(row['transform_matrix']), np.array(row['inverse_transform_matrix'])
        np.testing.assert_allclose(row['labels'][6:], [1,0,0,.15])
        self.assertEqual(row['transform_type'], 'homography_2d')
        self.assertEqual(row['transform_scope'], 'crop_2d')
        self.assertFalse(row['new_3d_view'])
        self.assertNotEqual(matrix[2,0], 0)
        np.testing.assert_allclose(inverse @ matrix, np.eye(3), atol=1e-12)
        # u' = u/(1+.15*u), v' = v/(1+.15*u), u=(x-8)/16.
        for source, expected in [([8.,8.,1.], [8.,8.]),
                                 ([12.,4.,1.], [8+4/1.0375, 8-4/1.0375])]:
            mapped = matrix @ source
            np.testing.assert_allclose(mapped[:2]/mapped[2], expected, atol=1e-12)
        with Image.open(path.parent/row['image']) as image:
            rgb = np.array(image)
        with Image.open(path.parent/row['mask']) as image:
            mask = np.array(image)
        np.testing.assert_array_equal(rgb[..., 0], mask)
        self.assertFalse(np.array_equal(mask, rgba[..., 3]))
        self.assertGreater(mask.sum(), 0)
        self.assertTrue(np.isfinite(row['support_retained_fraction']))

    def test_shift_reports_clipping_of_support_at_image_border(self):
        rgba = np.zeros((20,20,4), np.uint8)
        rgba[6:14, :4, :3] = [200,120,30]
        rgba[6:14, :4, 3] = 255
        for row in self.records:
            Image.fromarray(rgba).save(self.root/row['file'])
        _, rows = self.build()
        row = next(r for r in rows if r['variant'] == 'shift_left')
        self.assertAlmostEqual(row['support_retained_fraction'], .5)

    def test_nontraining_groups_never_enter_probe(self):
        self.records.append(dict(file='never-open.png', frame_id='test', group_id='g00', split='test'))
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.build()
        path = create_probe(self.manifest, self.root, self.root / 'small', n_groups=3)
        rows = json.loads(path.read_text(encoding='utf-8'))
        self.assertNotIn('g00', {r['group_id'] for r in rows})

    def test_rgba_required_and_output_not_overwritten(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        for row in self.records:
            Image.fromarray(self.rgba[..., :3]).save(self.root / row['file'])
        with self.assertRaises(ValueError):
            self.build('rgb')


if __name__ == '__main__':
    unittest.main()
