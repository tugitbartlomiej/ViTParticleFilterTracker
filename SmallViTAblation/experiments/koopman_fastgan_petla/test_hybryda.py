"""Transport hybrydowy: testy CPU, bez SSH, SLURM i modeli."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import hybryda_zdalne as transport
    import hybryda_worker as worker
except ModuleNotFoundError as exc:
    if exc.name not in {'hybryda_zdalne', 'hybryda_worker'}:
        raise
    transport = worker = None


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class HybridCase(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(transport, 'Brak transportu hybrydowego')
        self.assertIsNotNone(worker, 'Brak workera hybrydowego')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / 'parent'
        self.parent.mkdir()
        self.checkpoint = self.parent / 'FINAL.ckpt'
        self.checkpoint.write_bytes(b'parent checkpoint')
        (self.parent / 'FINAL.pth').write_bytes(b'export')
        self.config = self.root / 'config.json'
        dump(self.config, {'seed': 42, 'epoch_size': 1600})
        self.dataset = self.root / 'dataset'
        dump(self.dataset / 'mapping.json', {'train': [
            {'frame_id': 'old', 'group_id': 'g1'}, {'frame_id': 'new', 'group_id': 'g2'}]})
        self.remote = {key: transport.ROOT + '/original/' + value for key, value in {
            'checkpoint': 'pretrain/FINAL.ckpt', 'dataset': 'dataset',
            'probe': 'probe/manifest.json', 'config': 'config.json',
            'initial_weights': 'initial.pth', 'pool': 'pool_manifest.jsonl'}.items()}
        self.remote.update(data_root='/tmp/koopman-fastgan-123-abc/data',
                           pool_sha256='a' * 64, archive_sha256='b' * 64)
        dump(self.parent / 'REMOTE.json', self.remote)
        self.kwargs = dict(config=self.config, dataset=self.dataset, probe=self.root / 'probe.json',
                           out=self.root / 'result', initial_weights=self.root / 'initial.pth',
                           resume=self.checkpoint, added=True, stop_step=500, device='cuda')
        self.request_id = '1234abcd' * 4


class RequestTests(HybridCase):
    def test_request_preserves_order_budget_and_provenance(self):
        request = transport.make_request('trening.py', self.kwargs, self.request_id)
        self.assertEqual(request['train_ids'], ['old', 'new'])
        self.assertEqual(request['stop_step'], 500)
        self.assertTrue(request['added'])
        self.assertEqual(request['parent'], self.remote)
        self.assertEqual(request['pool_sha256'], 'a' * 64)
        self.assertEqual(request['parent_sha256'], hashlib.sha256(b'parent checkpoint').hexdigest())
        self.assertEqual(request['config'], {'seed': 42, 'epoch_size': 1600})

    def test_only_two_scripts_and_safe_request_ids_are_allowed(self):
        for script in ('petla.py', '../trening.py', 'trening.py; rm -rf /'):
            with self.subTest(script=script), self.assertRaises(ValueError):
                transport.make_request(script, self.kwargs, self.request_id)
        for identifier in ('../abc', 'a;whoami', '0' * 31, 'g' * 32):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                transport.make_request('trening.py', self.kwargs, identifier)

    def test_pool_hash_can_come_from_original_bundle_inputs(self):
        self.remote.pop('pool_sha256')
        dump(self.parent / 'REMOTE.json', self.remote)
        dump(self.root / 'INPUTS.json', {'pool_sha256': 'c' * 64})
        request = transport.make_request('trening.py', self.kwargs, self.request_id)
        self.assertEqual(request['pool_sha256'], 'c' * 64)
        self.assertIn('INPUTS.json', request['pool_sha256_source'])

    def test_missing_hash_duplicate_ids_and_remote_escape_are_rejected(self):
        self.remote.pop('pool_sha256')
        dump(self.parent / 'REMOTE.json', self.remote)
        with self.assertRaises(ValueError):
            transport.make_request('trening.py', self.kwargs, self.request_id)
        self.remote['pool_sha256'] = 'a' * 64
        self.remote['checkpoint'] = transport.ROOT + '/../foreign/FINAL.ckpt'
        dump(self.parent / 'REMOTE.json', self.remote)
        with self.assertRaises(ValueError):
            transport.make_request('trening.py', self.kwargs, self.request_id)

    def test_evaluation_uses_export_parent_and_never_requests_resume(self):
        request = transport.make_request('ocena.py', dict(weights=self.parent / 'FINAL.pth',
            dataset=self.dataset, out=self.root / 'eval', split='test', device='cuda'), self.request_id)
        self.assertEqual(request['split'], 'test')
        self.assertFalse(request['added'])
        self.assertNotIn('stop_step', request)
        self.assertEqual(request['parent_sha256'], hashlib.sha256(b'export').hexdigest())

    def test_handoff_hash_aliases_image_path_and_source_job(self):
        self.remote['data_archive_sha256'] = self.remote.pop('archive_sha256')
        self.remote.update(source_job='1810448', image_manifest=transport.ROOT + '/pretrain/IMAGE_BYTES.json',
            archive_hash_provenance='computed_at_handoff_from_original_archive; original startup did not record archive digest')
        dump(self.parent / 'REMOTE.json', self.remote)
        request = transport.make_request('trening.py', self.kwargs, self.request_id)
        self.assertEqual(request['archive_sha256'], 'b' * 64)
        self.assertEqual(request['parent']['source_job'], '1810448')
        self.assertEqual(request['archive_hash_provenance'], self.remote['archive_hash_provenance'])
        for updates in ({'archive_sha256': 'c' * 64}, {'image_manifest': '/tmp/foreign.json'},
                        {'image_manifest': transport.ROOT + '/../foreign.json'}):
            dump(self.parent / 'REMOTE.json', dict(self.remote, **updates))
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                transport.make_request('trening.py', self.kwargs, self.request_id)

    def test_duplicate_train_ids_are_rejected_before_transport(self):
        dump(self.dataset / 'mapping.json', {'train': [{'frame_id':'old'}, {'frame_id':'old'}]})
        with self.assertRaises(ValueError):
            transport.make_request('trening.py', self.kwargs, self.request_id)


class ArchiveTests(HybridCase):
    def archive(self, entries):
        path = self.root / 'input.tar'
        with tarfile.open(path, 'w') as archive:
            for name, content, kind in entries:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(content) if kind == tarfile.REGTYPE else 0
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = '../outside'
                archive.addfile(info, io.BytesIO(content) if info.isfile() else None)
        return path

    def test_unsafe_archives_never_publish_output(self):
        cases = [('.. /x', tarfile.REGTYPE), ('../escape', tarfile.REGTYPE),
                 ('/absolute', tarfile.REGTYPE), ('C:/escape', tarfile.REGTYPE),
                 ('x\\..\\escape', tarfile.REGTYPE), ('safe', tarfile.SYMTYPE),
                 ('safe', tarfile.LNKTYPE), ('NUL.txt', tarfile.REGTYPE), ('x:stream', tarfile.REGTYPE)]
        for name, kind in cases:
            with self.subTest(name=name, kind=kind), self.assertRaises(ValueError):
                transport.safe_extract(self.archive([(name, b'x', kind)]), self.root / 'unpacked')
            self.assertFalse((self.root / 'unpacked').exists())

    def test_exact_manifest_coverage_and_hashes_are_required(self):
        archive = self.archive([('a.json', b'{}', tarfile.REGTYPE)])
        entry = {'file': 'a.json', 'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}
        for entries in ([], [dict(entry, sha256='0' * 64)], [entry, dict(entry, file='missing')]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                transport.safe_extract(archive, self.root / 'unpacked', entries=entries)
            self.assertFalse((self.root / 'unpacked').exists())
        transport.safe_extract(archive, self.root / 'unpacked', entries=[entry])
        self.assertEqual((self.root / 'unpacked/a.json').read_bytes(), b'{}')

    def test_duplicate_case_collisions_and_existing_target_are_rejected(self):
        archive = self.archive([('x.json', b'{}', tarfile.REGTYPE), ('X.json', b'{}', tarfile.REGTYPE)])
        with self.assertRaises(ValueError):
            transport.safe_extract(archive, self.root / 'unpacked')
        target = self.root / 'existing'
        target.mkdir()
        (target / 'foreign').write_text('preserve')
        with self.assertRaises(FileExistsError):
            transport.safe_extract(archive, target)
        self.assertEqual((target / 'foreign').read_text(), 'preserve')

    def test_pack_includes_full_final_state_but_excludes_dense_checkpoints(self):
        source = self.root / 'training'
        source.mkdir()
        for name in ('FINAL.ckpt', 'FINAL.pth', 'LATENT_0000500.npz', 'CZASTKI_0000500.npz',
                     'REMOTE.json', 'STATUS.json', 'VALIDATION.json', 'FULL_0000500.ckpt', 'model_0000500.pth'):
            (source / name).write_bytes(b'fixture')
        archive, manifest = worker.pack_results(source, self.root / 'packed', self.request_id, 'trening.py')
        entries = json.loads(manifest.read_text())['files']
        names = {entry['file'] for entry in entries}
        self.assertIn('FINAL.ckpt', names)
        self.assertIn('CZASTKI_0000500.npz', names)
        self.assertNotIn('FULL_0000500.ckpt', names)
        self.assertNotIn('model_0000500.pth', names)
        transport.safe_extract(archive, self.root / 'roundtrip', entries=entries)
        self.assertEqual((self.root / 'roundtrip/FINAL.ckpt').read_bytes(), b'fixture')


class WorkerTests(HybridCase):
    def test_new_image_proof_binds_current_checkpoint_train_and_validation(self):
        data = self.root / 'pixels'
        data.mkdir()
        for name in ('old', 'new', 'val', 'test'):
            (data / name).write_bytes(name.encode())
        for split, names in [('train', ['old', 'new']), ('valid', ['val']), ('test', ['test'])]:
            dump(self.dataset / split / '_annotations.coco.json',
                 {'images': [{'file_name':str(data / name)} for name in names]})
        proof = worker.image_manifest(self.checkpoint, self.dataset, data)
        self.assertEqual(proof['checkpoint_sha256'], transport.sha256(self.checkpoint))
        self.assertEqual(proof['data_root'], str(data.resolve()))
        self.assertEqual(proof['files'], {name: transport.sha256(data / name) for name in ('old','new','val')})
        dump(self.dataset / 'valid/_annotations.coco.json', {'images':[{'file_name':str(self.config)}]})
        with self.assertRaises(ValueError):
            worker.image_manifest(self.checkpoint, self.dataset, data)

    def test_empty_test_stays_empty_and_unavailable_report_is_packable(self):
        from dane import make_dataset
        data = self.root / 'pixels'
        data.mkdir()
        rows = []
        for name, split in [('old','train'), ('val','val')]:
            (data / name).write_bytes(b'pixels')
            rows.append(dict(frame_id=name, group_id=name, split=split, path=name,
                             native_width=10, native_height=10, annotations=[]))
        dataset = make_dataset(rows, data, ['old'], self.root / 'remote-dataset')
        self.assertEqual(transport.read_json(dataset / 'test/_annotations.coco.json')['images'], [])
        self.assertEqual(len(transport.read_json(dataset / 'valid/_annotations.coco.json')['images']), 1)
        out = self.root / 'evaluation'
        dump(out / 'REPORT.json', {'split':'test', 'available':False, 'reason':'empty_split'})
        dump(out / 'REMOTE.json', self.remote)
        archive, manifest = worker.pack_results(out, self.root / 'eval-result', self.request_id, 'ocena.py')
        transport.safe_extract(archive, self.root / 'local-eval', entries=transport.read_json(manifest)['files'])
        self.assertFalse(transport.read_json(self.root / 'local-eval/REPORT.json')['available'])

    def test_slurm_exit_without_archive_is_failure_and_completed_status_is_preserved(self):
        result = self.root / 'remote-result'
        with patch.object(worker, 'result_root', return_value=result):
            for state in ('TIMEOUT', 'OUT_OF_MEMORY', 'COMPLETED', 'CANCELLED by 123'):
                response = subprocess.CompletedProcess([], 0, f'999|COMPLETED|0:0\n12345|{state}|1:0\n')
                with self.subTest(state=state), patch.object(worker.subprocess, 'run', return_value=response):
                    status = worker.status_report(self.request_id, '12345')
                self.assertEqual(status['state'], 'failed')
            completed = dict(request_id=self.request_id, job_id='12345', state='completed', archive_sha256='a'*64)
            dump(result / 'STATUS.json', completed)
            with patch.object(worker.subprocess, 'run', side_effect=AssertionError('Nie wolno wywołać sacct')):
                self.assertEqual(worker.status_report(self.request_id, '12345'), completed)
                worker.mark_failed(self.request_id, 'spóźniony trap')
                self.assertEqual(transport.read_json(result / 'STATUS.json'), completed)
            with self.assertRaises(ValueError):
                worker.status_report(self.request_id, '12346')

    def test_prepare_dataset_rebases_parent_order_then_adds_only_last_id(self):
        request = transport.make_request('trening.py', self.kwargs, self.request_id)
        calls = []
        rows = [{'frame_id':'old','split':'train','group_id':'g1'},
                {'frame_id':'new','split':'train','group_id':'g2'}]
        def make(rows_arg, data_root, ids, out, added=None):
            calls.append((list(ids), added, Path(out).name))
            return Path(out)
        with patch('dane.make_dataset', side_effect=make):
            base, branch = worker.rebuild_datasets(request, rows, self.root / 'data', self.root / 'remote-output',
                                                   parent_mapping=[{'frame_id':'old','group_id':'g1'}])
        self.assertEqual(calls, [(['old'], None, 'base_dataset'), (['old'], 'new', 'dataset')])
        self.assertEqual(base.name, 'base_dataset')
        self.assertEqual(branch.name, 'dataset')
        with self.assertRaises(ValueError):
            worker.rebuild_datasets(request, rows, self.root / 'data', self.root / 'remote-output',
                                    parent_mapping=[{'frame_id':'new','group_id':'g2'}])

    def test_worker_argv_keeps_stop_step_and_routes_eval_without_rebase(self):
        request = transport.make_request('trening.py', self.kwargs, self.request_id)
        argv = worker.stage_command(request, code=Path('/code'), dataset=Path('/new/dataset'),
                                    config=Path('/stable/config.json'), resume=Path('/scratch/rebased.ckpt'),
                                    out=Path('/result/training'))
        self.assertEqual(argv[argv.index('--stop-step') + 1], '500')
        self.assertEqual(argv[argv.index('--initial-weights') + 1], self.remote['initial_weights'])
        self.assertEqual(argv[argv.index('--resume') + 1], str(Path('/scratch/rebased.ckpt')))
        self.assertIn('--added', argv)
        request = transport.make_request('ocena.py', dict(weights=self.parent / 'FINAL.pth',
            dataset=self.dataset, out=self.root / 'eval', split='val'), self.request_id)
        argv = worker.stage_command(request, code=Path('/code'), dataset=Path('/new/dataset'),
                                    config=Path('/config.json'), resume=None, out=Path('/result/evaluation'))
        self.assertNotIn('--resume', argv)
        self.assertEqual(argv[argv.index('--weights') + 1], self.remote['checkpoint'].replace('.ckpt','.pth'))


class WorkerFlowTests(HybridCase):
    """Rzeczywisty tar/dataset/SHA/metadata; atrapy wyłącznie granic Eden i GPU."""

    def prepare(self, script):
        from dane import make_dataset
        data = self.root / 'original-data'
        data.mkdir()
        rows = []
        for name, split in [('old', 'train'), ('new', 'train'), ('val', 'val')]:
            (data / name).write_bytes(name.encode())
            rows.append(dict(frame_id=name, group_id=name, split=split, path=name,
                             native_width=10, native_height=10, annotations=[]))
        pool = data / 'pool_manifest.jsonl'
        pool.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        archive = self.root / 'data.tar'
        with tarfile.open(archive, 'w') as tar:
            for path in data.iterdir():
                tar.add(path, arcname=path.name)
        old_ids = ['old'] if script == 'trening.py' else ['old', 'new']
        old_dataset = make_dataset(rows, data, old_ids, self.root / 'old-dataset')
        dump(self.root / 'probe.json', {'immutable':True})
        (self.root / 'initial.pth').write_bytes(b'initial')
        dump(self.parent / 'IMAGE_BYTES.json', worker.image_manifest(self.checkpoint, old_dataset, data))
        self.remote.update(pool_sha256=transport.sha256(pool), archive_sha256=transport.sha256(archive),
                           source_job='1810448', image_manifest=transport.ROOT + '/original/IMAGE_BYTES.json',
                           archive_hash_provenance='computed_at_handoff_from_original_archive')
        dump(self.parent / 'REMOTE.json', self.remote)
        kwargs = self.kwargs if script == 'trening.py' else dict(weights=self.parent / 'FINAL.pth',
            dataset=self.dataset, out=self.root / 'local-evaluation', split='test')
        request = transport.make_request(script, kwargs, self.request_id)
        # Granice ścieżek Eden sprawdzają RequestTests; tutaj używamy własnego
        # katalogu tymczasowego również na Windows, bez tworzenia /tmp.
        parent = dict(request['parent'])
        for key, path in dict(checkpoint=self.checkpoint, dataset=old_dataset, data_root=data,
                probe=self.root / 'probe.json', config=self.config, initial_weights=self.root / 'initial.pth',
                pool=pool, image_manifest=self.parent / 'IMAGE_BYTES.json').items():
            parent[key] = path.as_posix()
        request['parent'] = parent
        dump(self.root / 'hybrid_requests' / (self.request_id + '.json'), request)
        scratch = self.root / f'koopman-hybrid-12345-{self.request_id}-fixture'
        (scratch / 'code').mkdir(parents=True)
        (scratch / 'code' / script).write_text('# atrapa, nigdy nie jest uruchamiana', encoding='utf-8')
        rebase_calls, commands = [], []

        def rebase(**kwargs):
            rebase_calls.append(kwargs)
            self.assertEqual(kwargs['image_manifest'], parent['image_manifest'])
            self.assertEqual(transport.read_json(Path(kwargs['dataset']) / 'mapping.json')['train'],
                             [{'frame_id':'old', 'group_id':'old'}])
            Path(kwargs['newfile']).write_bytes(b'rebased')
            return {'verified':True}

        def process(argv, **kwargs):
            commands.append(argv)
            out = Path(argv[argv.index('--out') + 1])
            out.mkdir()
            if script == 'trening.py':
                self.assertEqual(argv[argv.index('--stop-step') + 1], '500')
                for name in ('FINAL.ckpt', 'FINAL.pth', 'LATENT_0000500.npz', 'FULL_0000500.ckpt'):
                    (out / name).write_bytes(b'new ' + name.encode())
                dump(out / 'STATUS.json', {'state':'completed'})
                dump(out / 'VALIDATION.json', [])
            else:
                self.assertNotIn('--resume', argv)
                self.assertEqual(argv[argv.index('--weights') + 1], self.checkpoint.with_suffix('.pth').as_posix())
                dump(out / 'REPORT.json', {'split':'test', 'available':False, 'reason':'empty_split'})
            return subprocess.CompletedProcess(argv, 0)

        patches = [patch.object(worker, 'ROOT', str(self.root)),
            patch.object(worker, 'DATA_ARCHIVE', str(archive)),
            patch.object(worker, 'Path', side_effect=lambda p: self.root if p == '/tmp' else Path(p)),
            patch.object(worker, 'validate_request', side_effect=lambda value:value),
            patch.object(worker, 'remote_path', side_effect=lambda value:value),
            patch.dict(os.environ, {'SLURM_JOB_ID':'12345'}),
            patch.dict(sys.modules, {'przenosny_checkpoint':SimpleNamespace(rebase_file=rebase)}),
            patch.object(worker.subprocess, 'run', side_effect=process)]
        for replacement in patches:
            replacement.start()
            self.addCleanup(replacement.stop)
        return request, scratch, rebase_calls, commands

    def test_training_flow_publishes_new_bound_metadata_and_verifiable_archive(self):
        request, scratch, rebases, commands = self.prepare('trening.py')
        out = worker.run_worker(self.request_id, scratch)
        self.assertEqual((len(rebases), len(commands)), (1, 1))
        remote = transport.read_json(out / 'REMOTE.json')
        self.assertEqual(remote['source_job'], '12345')
        self.assertEqual(remote['parent_source_job'], '1810448')
        self.assertEqual(remote['pool_sha256'], request['pool_sha256'])
        self.assertEqual(remote['data_archive_sha256'], request['archive_sha256'])
        self.assertEqual(remote['archive_hash_provenance'], request['archive_hash_provenance'])
        self.assertNotEqual(remote['image_manifest'], request['parent']['image_manifest'])
        proof = transport.read_json(remote['image_manifest'])
        self.assertEqual(proof['checkpoint_sha256'], transport.sha256(out / 'FINAL.ckpt'))
        self.assertEqual(set(proof['files']), {'old','new','val'})
        self.assertFalse(transport.read_json(out / 'DATA_PROOF.json')['historical_archive_digest_claimed'])
        status = transport.read_json(out.parent / 'STATUS.json')
        self.assertEqual(status['state'], 'completed')
        self.assertEqual(status['archive_sha256'], transport.sha256(out.parent / 'result.tar'))
        entries = transport.read_json(out.parent / 'result.manifest.json')['files']
        transport.safe_extract(out.parent / 'result.tar', self.root / 'local-result', entries=entries)
        self.assertFalse((self.root / 'local-result/FULL_0000500.ckpt').exists())

    def test_evaluation_flow_keeps_parent_contract_and_never_rebases(self):
        request, scratch, rebases, commands = self.prepare('ocena.py')
        out = worker.run_worker(self.request_id, scratch)
        self.assertEqual((len(rebases), len(commands)), (0, 1))
        remote = transport.read_json(out / 'REMOTE.json')
        for key, value in request['parent'].items():
            self.assertEqual(remote[key], value)
        self.assertEqual(remote['evaluation_source_job'], '12345')
        self.assertFalse(transport.read_json(out / 'REPORT.json')['available'])
        self.assertEqual(transport.read_json(out.parent / 'STATUS.json')['state'], 'completed')

    def test_config_drift_fails_before_rebase_or_training(self):
        request, scratch, rebases, commands = self.prepare('trening.py')
        dump(self.config, {'seed':999, 'epoch_size':1600})
        with self.assertRaisesRegex(ValueError, 'receptura'):
            worker.run_worker(self.request_id, scratch)
        self.assertEqual((rebases, commands), ([], []))
        status = transport.read_json(self.root / 'hybrid_results' / self.request_id / 'STATUS.json')
        self.assertEqual(status['state'], 'failed')
        self.assertFalse((self.root / 'hybrid_results' / self.request_id / 'result.tar').exists())


class FakeCommands:
    """Atrapa procesu: nie uruchamia powłoki ani sieci, kopiuje tylko fixture."""
    def __init__(self, status, archive=None, manifest=None):
        self.calls = []
        self.options = []
        self.status = status
        self.archive, self.manifest = archive, manifest

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        self.options.append(kwargs)
        assert not kwargs.get('shell', False)
        output = ''
        if argv[0] == 'ssh':
            command = argv[-1]
            if 'sbatch' in command:
                output = '12345;eden\n'
            elif ' status ' in command:
                output = json.dumps(self.status)
        elif argv[0] == 'scp' and argv[-2].startswith('eden-cluster:'):
            source = self.manifest if argv[-2].endswith('.manifest.json') else self.archive
            Path(argv[-1]).write_bytes(source.read_bytes())
        return subprocess.CompletedProcess(argv, 0, output, '')


class DispatchTests(HybridCase):
    def test_success_submits_once_downloads_and_verifies(self):
        source = self.root / 'remote-training'
        source.mkdir()
        for name in ('FINAL.ckpt','FINAL.pth','LATENT_0000500.npz','STATUS.json','VALIDATION.json','REMOTE.json'):
            (source / name).write_bytes(b'fixture')
        archive, manifest = worker.pack_results(source, self.root / 'packed', self.request_id, 'trening.py')
        status = {'request_id':self.request_id,'state':'completed','archive_sha256':transport.sha256(archive),
                  'manifest_sha256':transport.sha256(manifest)}
        commands = FakeCommands(status, archive, manifest)
        client = transport.Transport(run=commands, sleep=lambda _: None)
        with patch.object(transport, 'new_request_id', return_value=self.request_id):
            result = client.dispatch('trening.py', self.root / 'train.log', self.kwargs)
        self.assertEqual(result, self.kwargs['out'].resolve())
        self.assertEqual((result / 'FINAL.ckpt').read_bytes(), b'fixture')
        submissions = [call for call in commands.calls if call[0]=='ssh' and 'sbatch' in call[-1]]
        self.assertEqual(len(submissions), 1)
        self.assertIn('--export=NIL', submissions[0][-1])
        self.assertIn(self.request_id, submissions[0][-1])
        for call, options in zip(commands.calls, commands.options):
            self.assertEqual(options['timeout'], 1800 if call[0] == 'scp' else 120)
        self.assertEqual(transport.read_json(self.root / 'train.log.manifest.json'), transport.read_json(manifest))

    def test_failure_never_resubmits_training(self):
        commands = FakeCommands({'request_id':self.request_id,'state':'failed','error':'out of memory'})
        client = transport.Transport(run=commands, sleep=lambda _: None)
        with patch.object(transport, 'new_request_id', return_value=self.request_id), self.assertRaises(RuntimeError):
            client.dispatch('trening.py', self.root / 'train.log', self.kwargs)
        self.assertEqual(sum(call[0]=='ssh' and 'sbatch' in call[-1] for call in commands.calls), 1)
        self.assertFalse(self.kwargs['out'].exists())

    def test_timeout_cancels_only_its_own_job(self):
        commands = FakeCommands({'request_id':self.request_id,'state':'pending'})
        ticks = iter([0., 0., 21.])
        waits = []
        client = transport.Transport(run=commands, monotonic=lambda:next(ticks), sleep=waits.append,
                                     timeout_seconds=20, poll_seconds=20)
        with patch.object(transport, 'new_request_id', return_value=self.request_id), self.assertRaises(TimeoutError):
            client.dispatch('trening.py', self.root / 'train.log', self.kwargs)
        cancels = [call[-1] for call in commands.calls if call[0]=='ssh' and 'scancel' in call[-1]]
        self.assertEqual(cancels, ['scancel 12345'])
        self.assertEqual(waits, [20])
        self.assertFalse(self.kwargs['out'].exists())


if __name__ == '__main__':
    unittest.main()
