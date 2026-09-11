"""Read-only selection audit. Runs small AST-isolated tests; never loads models.

Usage (from any directory):
    py -3.11 -B reproduce_selection_bugs.py
The output is created exclusively: an existing results file is never overwritten.
Use --out with a NEW filename to repeat. Dependencies: numpy, PyYAML, scikit-learn.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import re
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

DEFAULT_REPO = Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker')
S = 'SmallViTAblation/pipeline/'
A = 'AdvancedDatasetSelection/'
READ_HASHES = {}


def read(path):
    data = path.read_bytes()
    READ_HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')


def definitions(repo, relative, extra=None):
    """Compile definitions only, without module imports, main blocks or setup hooks."""
    path = repo / relative
    tree = ast.parse(read(path), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    ns = {'np': np, 'Path': Path, 're': re, 'logger': logging.getLogger('audit'),
          'StandardScaler': StandardScaler}
    ns.update(extra or {})
    exec(compile(module, str(path), 'exec'), ns)
    return ns


def run(repo, shared_audit):
    tests = []

    def check(name, function, kind='bug_reproduced'):
        try:
            detail = function()
            tests.append({'id': name, 'status': kind, 'evidence': detail})
        except Exception as exc:
            tests.append({'id': name, 'status': 'unexpected_failure',
                          'error': f'{type(exc).__name__}: {exc}'})

    bc = definitions(repo, S + 'features/build_cache.py')
    common = definitions(repo, S + 'features/common.py')
    sel = definitions(repo, S + 'selection/select_subset.py', {
        'GROUP_ORDER': ('dino', 'el2n', 'sam', 'fourier'),
        'GROUP_KEY': {'D': 'dino', 'E': 'el2n', 'S': 'sam', 'F': 'fourier'}})
    old = definitions(repo, A + 'selection_methods/cluster_selector.py')
    baselines = definitions(repo, S + 'selection/baselines.py', {
        'AUG_RE': re.compile(r'^(?P<src>.+)_aug_\d+$')})
    protocol = yaml.safe_load(read(repo / 'SmallViTAblation/configs/protocol.yaml'))

    def config():
        try:
            bc['build_extractors'](protocol, False)
        except KeyError as exc:
            assert exc.args == ('dino_model',)
            return {'exception': str(exc), 'features_keys': sorted(protocol['features']),
                    'dino_model_under_analysis': protocol.get('analysis', {}).get('dino_model')}
        raise AssertionError('Expected missing features.dino_model')
    check('production_config_wrong_nesting', config)

    def dimension():
        dino = bc['DinoExtractor']('dinov3_vitl16', 224, 32)
        try:
            getattr(dino, 'dim', None) or getattr(dino, 'DIM')
        except AttributeError as exc:
            return {'exception': str(exc)}
        raise AssertionError('Expected missing DINO dimension')
    check('production_dino_missing_dimension', dimension)

    def repository():
        tree = ast.parse(read(repo / (S + 'features/build_cache.py')))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DinoExtractor')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_init')
        assignments = [n for n in init.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id in ('repos', 'family', 'repo') for t in n.targets)]
        ns = {'self': types.SimpleNamespace(model_name='dinov3_vitl16')}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<repo-choice-only>', 'exec'), ns)
        assert ns['family'] == 'dinov' and ns['repo'] == 'facebookresearch/dinov2:main'
        return {'family': ns['family'], 'chosen_repo': ns['repo']}
    check('dinov3_routed_to_dinov2', repository)

    def weights():
        rng = np.random.RandomState(42)
        matrices = {g: rng.normal(size=(64, d)).astype(np.float32)
                    for g, d in [('dino', 32), ('el2n', 1), ('sam', 9), ('fourier', 1)]}
        captured = {}
        def clustering(x, k, seed, max_iter):
            captured['x'] = x.copy()
            return np.arange(len(x)) % k, np.zeros((k, x.shape[1]))
        sel['load_group_matrix'] = lambda name, *args: (matrices[name], {'fake': True})
        sel['cluster_kmeans'] = clustering
        ids = [f'f{i:03d}' for i in range(64)]
        w = dict(dino=.35, el2n=.30, sam=.20, fourier=.15)
        result = sel['select_one']({'run_id': 'audit', 'seed': 42, 'budget': 3,
            'groups': ['D', 'E', 'S', 'F'], 'weights': w}, [], ids, Path('not_read'),
            {'dino_pca_dim': 32, 'cluster_algorithm': 'kmeans', 'cluster_representative': 'centroid',
             'kmeans_iterations': 1})
        x = captured['x']
        energy = np.array([x[:, :32].var(0).sum(), x[:, 32].var(),
                           x[:, 33:42].var(0).sum(), x[:, 42].var()])
        expected = np.array([32*.35**2, .30**2, 9*.20**2, .15**2])
        assert np.allclose(energy, expected)
        original = old['ClusterBasedSelector'].__new__(old['ClusterBasedSelector'])
        original.dino_features = rng.normal(size=(64, 16))
        original.fourier_features = rng.normal(size=(64, 9))
        original.sam_scores = rng.normal(size=64)
        original.el2n_scores = rng.normal(size=64)
        original.weights = w
        ox = original.combine_features(dino_pca_dim=4)
        oe = [ox[:, :4].var(0).sum(), ox[:, 14].var(), ox[:, 13].var(), ox[:, 4:13].var(0).sum()]
        assert np.allclose(oe, [.35, .30, .20, .15])
        return {'group_order': ['D', 'E', 'S', 'F'], 'new_percent': (100*energy/energy.sum()).tolist(),
                'original_group_energy': oe, 'fake_metadata_accepted': len(result['frame_ids']) == 3,
                'note': 'Synthetic unit-variance example; not measured production feature contributions.'}
    check('group_weighting_and_fake_cache_acceptance', weights)

    def pca():
        rng = np.random.RandomState(42)
        x = rng.normal(size=(80, 16)) * np.arange(1, 17)
        y, dim = sel['pca_dino'](sel['standardize'](x), 4)
        variance = y.var(0)
        assert not np.allclose(variance, 1)
        return {'components': dim, 'new_component_variances': variance.tolist(),
                'note': 'New PCA components are not standardized after reduction.'}
    check('pca_order_and_unstandardized_components', pca)

    def kcenter():
        x = np.array([0, 10, 17, 27, 32, 38, 39, 40, 56], dtype=np.float32).reshape(-1, 1)
        labels = sel['cluster_kcenter_greedy'](x, 3)
        chosen = sel['pick_representatives'](x, labels, 3, 'centroid', 42)
        actual = x[chosen].ravel().tolist()
        greedy = [3, 8, 0]
        def radius(idx):
            return float(np.linalg.norm(x[:, None, :] - x[idx][None, :, :], axis=2).min(1).max())
        assert actual == [32., 56., 0.] and radius(greedy) == 13 and radius(chosen) == 15
        return {'points': x.ravel().tolist(), 'greedy_centers': x[greedy].ravel().tolist(),
                'returned': actual, 'greedy_radius': 13, 'returned_radius': 15}
    check('kcenter_centers_replaced_and_radius_worsened', kcenter)

    def coverage():
        sam = bc['FastSamExtractor']('unused', .4, .9)
        masks = np.zeros((2, 10, 10), dtype=bool)
        masks[:, :5, :] = True
        score = float(sam._complexity(masks, [1, 1], (10, 10))[3])
        union = float(masks.any(0).mean())
        scaled = float(sam._complexity(masks, [1, 1], (20, 20))[3])
        assert (score, union, scaled) == (1., .5, .25)
        return {'reported_overlap_coverage': score, 'true_union': union,
                'reported_with_different_original_size': scaled}
    check('sam_overlap_and_coordinate_grid', coverage)

    def uniform():
        pool = [{'frame_id': f'{pat}{i:02d}', 'patient_id': pat}
                for pat, size in [('a', 1), ('b', 10), ('c', 10)] for i in range(size)]
        idx = baselines['b_uniform'](pool, [r['frame_id'] for r in pool], 12, 42)
        counts = dict(Counter(pool[i]['patient_id'] for i in idx))
        assert counts == {'a': 1, 'b': 7, 'c': 4}
        return {'capacities': [1, 10, 10], 'budget': 12, 'observed': counts}
    check('uniform_leftover_bias', uniform)

    def duplicates():
        cache = {'frame_ids': ['a', 'a'], 'features': np.array([[1], [99]])}
        actual = common['gather_rows'](cache, ['a', 'a'], 'audit').ravel().tolist()
        assert actual == [99, 99]
        return {'rows_returned': actual}
    check('duplicate_cache_ids_last_wins', duplicates)

    records = [{'frame_id': 'a', 'split': 'train', 'path': 'images/a.jpg',
                'native_width': 20, 'native_height': 20, 'has_instrument': True},
               {'frame_id': 'b', 'split': 'test', 'path': 'images/b.jpg'}]
    mat = definitions(repo, S + 'selection/materialize.py', {
        'read_manifest': lambda _: records, 'utc_now': lambda: 'audit',
        'CATEGORY': {'id': 1, 'name': 'instrument', 'supercategory': 'tool'}})

    def coco():
        result = mat['build_coco']({'run_id': 'audit', 'frame_ids': ['a', 'a'], 'budget': 2}, Path('not_read'), False)
        assert len(result['images']) == 2 and not result['annotations']
        return {'images': len(result['images']), 'unique_paths': len({i['file_name'] for i in result['images']}),
                'annotations': len(result['annotations']), 'has_instrument': True, 'bbox_present': False}
    check('coco_duplicate_paths_and_silent_missing_bbox', coco)

    def split_guard():
        try:
            mat['build_coco']({'run_id': 'audit', 'frame_ids': ['b']}, Path('not_read'), False)
        except ValueError as exc:
            assert 'split=train' in str(exc)
            return {'rejection': str(exc)}
        raise AssertionError('Test split accepted')
    check('coco_train_only_guard', split_guard, 'guard_confirmed')

    def el2n():
        ns = definitions(repo, S + 'features/el2n_from_proxy.py')
        class Detector:
            def __init__(self, **kwargs): self.model = self
            def load_state_dict(self, *args, **kwargs): pass
            def eval(self): return self
            def to(self, *args): return self
            def predict(self, path, threshold):
                if path.endswith('negative_empty'): return []
                return [{'xyxy': [0, 0, 10, 10], 'confidence': .9, 'class_id': 99}]
        frames = [{'path': 'negative_empty'}, {'path': 'negative_false_positive'},
                  {'path': 'positive_wrong_class', 'bbox': [0, 0, 10, 10], 'has_instrument': True}]
        modules = {'rfdetr': types.SimpleNamespace(RFDETRSmall=Detector),
                   'torch': types.SimpleNamespace(load=lambda *a, **k: {},
                       cuda=types.SimpleNamespace(is_available=lambda: False))}
        with patch.dict(sys.modules, modules):
            values = ns['score_real']({'evaluation': {'map_conf_threshold': .001, 'iou_threshold': .5}},
                                     frames, Path('.'), Path('not_read'), 42)
        assert np.allclose(values, [1, 1, .1])
        equivalent = np.allclose(sel['standardize'](values.reshape(-1, 1)),
            sel['standardize'](ns['minmax01'](values)[0].reshape(-1, 1)))
        assert equivalent
        return {'input_cases': [r['path'] for r in frames], 'scores': values.tolist(),
                'minmax_cancels_under_standardization': bool(equivalent),
                'binary_positive_l2_at_p09': float(np.linalg.norm(np.array([.9, .1])-np.array([1., 0.]))),
                'limitation': 'Tests scoring logic only; does NOT validate RF-DETR package API.'}
    check('el2n_negatives_false_positives_class_matching', el2n)

    def stale_pool():
        obj = old['ClusterBasedSelector'].__new__(old['ClusterBasedSelector'])
        obj.fastsam = None
        obj.detr_el2n = None
        caches = {'fourier_features': (np.arange(9).reshape(1, 9), ['cached/test/image.jpg']),
                  'dino_features': np.ones((1, 2)), 'sam_scores': np.array([.5])}
        obj._load_cache = lambda name: caches.get(name)
        result = obj.extract_all_features(['requested/train/image.jpg'], use_cache=True, show_progress=False)
        assert result['valid_paths'] == ['cached/test/image.jpg']
        return {'requested': ['requested/train/image.jpg'], 'used': result['valid_paths']}
    check('original_cache_replaces_requested_pool', stale_pool)

    statistics = {}
    for folder in ('selected_dataset1', 'selected_dataset_weighted_features'):
        relative = A + f'output/{folder}/selection_reasons.json'
        data = json.loads(read(repo / relative))
        rows = data['images']
        sources = Counter(re.sub(r'_aug_\d+$', '', Path(r['filename']).stem) for r in rows)
        statistics[folder] = {'file': relative, 'summary': data['summary'], 'images': len(rows),
            'unique_filenames': len({r['filename'] for r in rows}), 'unique_sources': len(sources),
            'augmented_images': sum(bool(re.search(r'_aug_\d+$', Path(r['filename']).stem)) for r in rows),
            'multiple_variant_sources': sum(n > 1 for n in sources.values()),
            'singleton_clusters': sum(r['cluster_size'] == 1 for r in rows),
            'valid_directory_paths': sum('valid' in r['path'].replace('\\', '/').split('/') for r in rows),
            'q81_details_present': sum(bool(r.get('detr_q81')) for r in rows)}

    merged_path = repo / (A + 'output/selected_dataset1/merged_20k_annotations_fixed.json')
    merged = json.loads(read(merged_path))
    names = [r['file_name'] for r in merged['images']]
    reasons = json.loads(read(repo / (A + 'output/selected_dataset1/selection_reasons.json')))
    production = {'file': str(merged_path), 'images': len(names), 'annotations': len(merged['annotations']),
        'unique_sources': len({re.sub(r'_aug_\d+$', '', Path(n).stem) for n in names}),
        'augmented_images': sum(bool(re.search(r'_aug_\d+$', Path(n).stem)) for n in names),
        'augmented_percent': 100*sum(bool(re.search(r'_aug_\d+$', Path(n).stem)) for n in names)/len(names),
        'same_filename_multiset_as_selected_dataset1_reasons': Counter(names) == Counter(r['filename'] for r in reasons['images'])}
    assert (production['images'], production['unique_sources'], production['augmented_images']) == (20000, 15565, 14259)
    if shared_audit.exists():
        shared = json.loads(read(shared_audit))
        assert Path(shared['annotation_file']).resolve() == merged_path.resolve()
        leak_names = shared['matched_leaked_filenames']
        production['shared_evidence'] = {'file': str(shared_audit),
            'membership_current_video_split': shared['membership_current_video_split'],
            'leak_filenames_in_merged': len(set(names) & set(leak_names)),
            'provenance': 'User-provided counts, also read from production_pool_audit.json; split/leak-list derivation not independently rerun.'}
        assert sum(shared['membership_current_video_split'].values()) == len(names)
        assert len(set(names) & set(leak_names)) == 48

    changed = [p for p, sha in READ_HASHES.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha]
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'repo': str(repo),
        'scope': 'selection/features only; no eval/train inspection or execution; no model downloads',
        'method': 'AST-isolated original definitions, numpy/sklearn synthetic data, mocked detector and manifest reads',
        'tests': tests, 'unexpected_failures': sum(t['status'] == 'unexpected_failure' for t in tests),
        'selection_reasons': statistics, 'production_selected_dataset1': production,
        'python': sys.version, 'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'PyYAML', 'scikit-learn')},
        'rfdetr_available_in_this_interpreter': importlib.util.find_spec('rfdetr') is not None,
        'source_sha256': READ_HASHES, 'source_files_changed_during_execution': changed,
        'limitations': ['No real detector inference or training.', 'No image pixel duplicate test.',
                       'No independent reconstruction of split membership or historical checkpoint provenance.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('selection_repro_results.json'))
    parser.add_argument('--shared-audit', type=Path, default=Path(__file__).with_name('production_pool_audit.json'))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f'Will not overwrite an existing artifact: {args.out}')
    if args.out.resolve().is_relative_to(args.repo.resolve()):
        raise SystemExit('Output must be outside the repository')
    result = run(args.repo.resolve(), args.shared_audit)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({'out': str(args.out), 'tests': len(result['tests']),
        'unexpected_failures': result['unexpected_failures'],
        'source_files_changed': result['source_files_changed_during_execution'],
        'production': result['production_selected_dataset1']}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['unexpected_failures'] or result['source_files_changed_during_execution'] else 0)


if __name__ == '__main__':
    main()
