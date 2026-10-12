import copy
import json
from pathlib import Path
import tempfile
import unittest

from przenosny_checkpoint import rebase_checkpoint, rebase_file, _sha


class PortableCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        for name in ('a.jpg', 'v.jpg'):
            (self.data / name).write_bytes(b'original-image-' + name.encode())
        self.ds = self.root / 'dataset'
        self.probe = self.root / 'probe'
        self.probe.mkdir()
        for name in ('x.png', 'mask.png'):
            (self.probe / name).write_bytes(name.encode())
        self.write(self.probe / 'manifest.json', [{'image': 'x.png', 'mask': 'mask.png'}])
        self.mapping = [{'frame_id': 'frame-a', 'group_id': 'op-a'}]
        self.write(self.ds / 'mapping.json', {'train': self.mapping})
        self.old_train = self.coco('/gone/data/a.jpg')
        self.old_val = self.coco('/gone/data/v.jpg')
        self.write(self.root / 'old-val.json', self.old_val)
        self.write(self.ds / 'train/_annotations.coco.json', self.coco(str(self.data / 'a.jpg')))
        self.write(self.ds / 'valid/_annotations.coco.json', self.coco(str(self.data / 'v.jpg')))
        contract = dict(recipe={'seed': 42}, model_config={'resolution': 384},
                        mapping=self.mapping, annotations=self.old_train,
                        validation_sha256=_sha(self.root / 'old-val.json'),
                        probe_sha256=_sha(self.probe / 'manifest.json'),
                        probe_files={name: _sha(self.probe / name) for name in ('x.png', 'mask.png')})
        self.ckpt = dict(state_dict={'weight': [1, 2]}, optimizer_states=[{'moments': [3]}],
                         lr_schedulers=[{'last_epoch': 4}], loops={'fit_loop': {'cursor': 9}},
                         experiment_rng={'python': [7]}, global_step=4,
                         experiment_boundary='before_next_epoch_v1', experiment_contract=contract,
                         experiment_recipe=contract['recipe'], experiment_mapping=self.mapping)
        self.kw = dict(dataset=self.ds, probe=self.probe / 'manifest.json',
                       old_validation_json=self.root / 'old-val.json',
                       old_data_root='/gone/data', new_data_root=self.data)

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')

    def coco(self, name):
        return dict(images=[dict(id=0, file_name=name, width=20, height=10)],
                    annotations=[dict(id=0, image_id=0, category_id=1, bbox=[1, 2, 3, 4], area=12, iscrowd=0)],
                    categories=[dict(id=1, name='instrument')], info={})

    def run_rebase(self):
        return rebase_checkpoint(self.ckpt, **self.kw, allow_unverified_image_bytes=True)

    def test_semantic_rebase_preserves_state_and_reports_missing_pixel_proof(self):
        before = copy.deepcopy(self.ckpt)
        result, audit = self.run_rebase()
        self.assertEqual(before, self.ckpt)
        for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'loops', 'experiment_rng'):
            self.assertIs(result[key], self.ckpt[key])
        for key in ('recipe', 'model_config', 'mapping', 'probe_sha256', 'probe_files'):
            self.assertEqual(result['experiment_contract'][key], self.ckpt['experiment_contract'][key])
        self.assertFalse(audit['image_bytes_verified'])
        self.assertFalse(audit['archive_sha_verified'])
        self.assertEqual(audit['unverified_images'], ['a.jpg', 'v.jpg'])
        self.assertEqual(result['experiment_contract']['validation_sha256'], _sha(self.ds / 'valid/_annotations.coco.json'))

    def test_missing_proof_rejected_without_explicit_external_proof_mode(self):
        with self.assertRaisesRegex(ValueError, 'historical image byte proof'):
            rebase_checkpoint(self.ckpt, **self.kw)

    def test_historical_hashes_verified_and_changed_bytes_rejected(self):
        self.ckpt['experiment_contract']['image_files'] = {n: _sha(self.data / n) for n in ('a.jpg', 'v.jpg')}
        _, audit = rebase_checkpoint(self.ckpt, **self.kw)
        self.assertTrue(audit['image_bytes_verified'])
        (self.data / 'a.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Changed image bytes'):
            self.run_rebase()

    def test_geometry_dimensions_categories_and_extra_frame_rejected(self):
        p = self.ds / 'train/_annotations.coco.json'
        original = json.loads(p.read_text())
        for part in ('geometry', 'dimensions', 'categories', 'extra'):
            with self.subTest(part=part):
                changed = copy.deepcopy(original)
                if part == 'geometry': changed['annotations'][0]['bbox'][0] = 9
                elif part == 'dimensions': changed['images'][0]['width'] = 99
                elif part == 'categories': changed['categories'][0]['name'] = 'other'
                else: changed['images'].append(dict(id=1, file_name=str(self.data / 'a.jpg'), width=20, height=10))
                self.write(p, changed)
                with self.assertRaises(ValueError): self.run_rebase()
        self.write(p, original)

    def test_mapping_and_probe_content_rejected(self):
        self.write(self.ds / 'mapping.json', {'train': [{'frame_id': 'different', 'group_id': 'op-a'}]})
        with self.assertRaisesRegex(ValueError, 'sample identities'): self.run_rebase()
        self.write(self.ds / 'mapping.json', {'train': self.mapping})
        (self.probe / 'mask.png').write_bytes(b'changed-mask')
        with self.assertRaisesRegex(ValueError, 'probe image/mask'): self.run_rebase()

    def test_original_validation_must_be_authenticated(self):
        self.write(self.root / 'old-val.json', self.coco('/gone/data/other.jpg'))
        with self.assertRaisesRegex(ValueError, 'Original validation'): self.run_rebase()

    def test_root_escape_and_relative_name_change_rejected(self):
        p = self.ds / 'train/_annotations.coco.json'
        for name in (str(self.root / 'a.jpg'), str(self.data / '..' / 'a.jpg'), str(self.data / 'v.jpg')):
            self.write(p, self.coco(name))
            with self.assertRaises(ValueError): self.run_rebase()

    def test_new_validation_geometry_rejected(self):
        val = self.coco(str(self.data / 'v.jpg'))
        val['annotations'][0]['area'] = 13
        self.write(self.ds / 'valid/_annotations.coco.json', val)
        with self.assertRaisesRegex(ValueError, 'COCO content'): self.run_rebase()

    def test_file_roundtrip_cpu_and_no_overwrite(self):
        import torch
        self.ckpt['state_dict']['weight'] = torch.tensor([1., 2.])
        self.ckpt['optimizer_states'][0]['moments'] = torch.tensor([.25])
        source, dest = self.root / 'parent.ckpt', self.root / 'moved.ckpt'
        torch.save(self.ckpt, source)
        source_hash = _sha(source)
        audit = rebase_file(checkpoint=source, newfile=dest, **self.kw, allow_unverified_image_bytes=True)
        moved = torch.load(dest, map_location='cpu', weights_only=False)
        self.assertTrue(torch.equal(moved['state_dict']['weight'], self.ckpt['state_dict']['weight']))
        self.assertTrue(torch.equal(moved['optimizer_states'][0]['moments'], self.ckpt['optimizer_states'][0]['moments']))
        self.assertEqual(source_hash, _sha(source))
        self.assertEqual(audit['source_checkpoint_sha256'], source_hash)
        with self.assertRaises(FileExistsError):
            rebase_file(checkpoint=source, newfile=dest, **self.kw, allow_unverified_image_bytes=True)

    def image_manifest(self, digest='a' * 64):
        path = self.root / 'IMAGE_BYTES.json'
        self.write(path, dict(checkpoint_sha256=digest, data_root='/gone/data',
                              files={n: _sha(self.data / n) for n in ('a.jpg', 'v.jpg')}))
        return path

    def test_bound_manifest_file_api_without_unverified_flag(self):
        import torch
        source = self.root / 'original.ckpt'
        torch.save(self.ckpt, source)
        manifest = self.image_manifest(_sha(source))
        target = self.root / 'portable.ckpt'
        audit = rebase_file(checkpoint=source, newfile=target, image_manifest=manifest, **self.kw)
        self.assertTrue(audit['image_bytes_verified'])
        self.assertTrue(audit['image_manifest_binding_verified'])
        moved = torch.load(target, map_location='cpu', weights_only=False)
        self.assertEqual(moved['experiment_contract']['image_files'], json.loads(manifest.read_text())['files'])
        kw = dict(self.kw, old_data_root=str(self.data),
                  old_validation_json=self.ds / 'valid/_annotations.coco.json')
        # A subsequent relocation uses inherited contract hashes, not the old manifest.
        _, second_audit = rebase_checkpoint(moved, **kw)
        self.assertTrue(second_audit['image_bytes_verified'])
        self.assertEqual(set(second_audit['image_evidence'].values()), {'checkpoint_hash'})

    def test_manifest_requires_matching_source_checkpoint(self):
        manifest = self.image_manifest()
        for digest in (None, 'b' * 64):
            with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
                rebase_checkpoint(self.ckpt, **self.kw, image_manifest=manifest,
                                  source_checkpoint_sha256=digest, allow_unverified_image_bytes=True)

    def test_manifest_wrong_root_missing_file_and_invalid_hash_rejected(self):
        path = self.image_manifest()
        original = json.loads(path.read_text())
        for variant in ('root', 'missing', 'extra', 'hash'):
            with self.subTest(variant=variant):
                bad = copy.deepcopy(original)
                if variant == 'root': bad['data_root'] = '/somewhere/else'
                elif variant == 'missing': del bad['files']['v.jpg']
                elif variant == 'extra': bad['files']['../escape'] = 'a' * 64
                else: bad['files']['a.jpg'] = 'invalid'
                self.write(path, bad)
                with self.assertRaises(ValueError):
                    rebase_checkpoint(self.ckpt, **self.kw, image_manifest=path,
                                      source_checkpoint_sha256='a' * 64, allow_unverified_image_bytes=True)

    def test_manifest_conflict_and_changed_image_rejected_even_with_optout(self):
        manifest = self.image_manifest()
        self.ckpt['experiment_contract']['image_files'] = {'a.jpg': 'b' * 64}
        with self.assertRaisesRegex(ValueError, 'Conflicting historical'):
            rebase_checkpoint(self.ckpt, **self.kw, image_manifest=manifest,
                              source_checkpoint_sha256='a' * 64)
        del self.ckpt['experiment_contract']['image_files']
        (self.data / 'v.jpg').write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'Changed image bytes'):
            rebase_checkpoint(self.ckpt, **self.kw, image_manifest=manifest,
                              source_checkpoint_sha256='a' * 64, allow_unverified_image_bytes=True)


if __name__ == '__main__':
    unittest.main()
