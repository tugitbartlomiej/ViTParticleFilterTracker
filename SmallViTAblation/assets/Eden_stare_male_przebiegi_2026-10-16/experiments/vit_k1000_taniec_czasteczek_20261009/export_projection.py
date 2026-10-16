"""CPU-only export of genuine frozen-model activation trajectories; no images.

Run export on the machine holding source NPZs, then verify the small copied
bundle locally. No model, optimizer, GPU, app or registry is imported/modified.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import numpy as np

SCHEMA_VERSION = 1
FAMILY = 'vit_k1000_taniec_czasteczek_20261009'
SOURCE_FAMILY = 'vit_k1000_1500_pelne_latenty_90_20261008'
MANIFEST_HASH = '33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7'
PANEL_HASH = 'f764ffb966e24e81e9e52257be1006c5365369f7ae382d181c8c23860930287f'
CHECKPOINT_HASH = '3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e'
N, TOKENS, DIMS, HORIZON = 1500, 600, 768, 500
STEPS = np.asarray([0, 100, 500], dtype=np.int32)
SEED = 20261009


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def tree_hash(records):
    return sha(''.join(f"{r['relative_path']}\0{r['sha256']}\n"
                       for r in sorted(records, key=lambda x: x['relative_path'])).encode())


def finite(a, label):
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite array: ' + label)


def states(z):
    before = z['before']
    trajectory = z['trajectory']
    if before.shape != (TOKENS, DIMS) or before.dtype != np.float32:
        raise ValueError('Unexpected before schema')
    if trajectory.shape != (2, TOKENS, DIMS) or trajectory.dtype != np.float32:
        raise ValueError('Unexpected trajectory schema')
    if not np.array_equal(z['trajectory_steps'], STEPS[1:]):
        raise ValueError('Missing or different genuine milestones; interpolation prohibited')
    x = np.concatenate((before[None], trajectory), axis=0)
    finite(x, 'before/milestones')
    return x


def read_frozen(path, expected):
    blob = path.read_bytes()
    if sha(blob) != expected:
        raise RuntimeError('Frozen source changed: ' + str(path))
    # The parsed arrays are from the very bytes whose SHA was checked.
    return np.load(io.BytesIO(blob), allow_pickle=False)


def stratified_sample(rows, sample_size=64):
    groups = {}
    for row in rows:
        film = row['frame_id'].rsplit('/', 1)[0]
        groups.setdefault(film, []).append(row['index'])
    films = sorted(groups)
    rng = np.random.default_rng(SEED)
    # Exactly 64 disjoint bands of ordered film IDs, one film and one frame each.
    # More films than sample slots: do not claim coverage of every film.
    bands = np.array_split(np.arange(len(films)), min(sample_size, len(films)))
    selected, strata = [], []
    for band_index, band in enumerate(bands):
        chosen_film = films[int(rng.choice(band))]
        index = int(rng.choice(sorted(groups[chosen_film])))
        selected.append(index)
        strata.append({'stratum': band_index, 'film_band': [films[int(v)] for v in band],
                       'chosen_film': chosen_film, 'image_index': index})
    if len(selected) != sample_size or len(set(selected)) != sample_size:
        raise ValueError('Expected 64 distinct stratified images')
    return selected, strata, len(films)


def project(x, mean, components):
    return ((x.astype(np.float64) - mean) @ components).astype(np.float32)


def export(source_root, out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'BUNDLE_MANIFEST.json').exists():
        raise FileExistsError('Completed export exists; verify it instead of overwriting')
    prod = source_root / 'results/production/latents'
    provenance = source_root / 'provenance'
    manifest_blob = (provenance / 'MANIFEST_1500.jsonl').read_bytes()
    panel_blob = (provenance / 'CONTROL_PANEL.json').read_bytes()
    if sha(manifest_blob) != MANIFEST_HASH or sha(panel_blob) != PANEL_HASH:
        raise RuntimeError('Remote provenance differs from verified local common data')
    rows = [json.loads(line) for line in manifest_blob.decode('utf-8-sig').splitlines()]
    rows.sort(key=lambda r: r['index'])
    if len(rows) != N or [r['index'] for r in rows] != list(range(N)):
        raise ValueError('Manifest indices must be exactly 0..1499')
    panel = {int(k): [int(v) for v in vs] for k, vs in json.loads(panel_blob).items()}
    control_images = np.asarray(sorted(panel), dtype=np.int32)
    control_tokens = np.asarray([panel[int(i)] for i in control_images], dtype=np.int32)
    if control_images.shape != (16,) or control_tokens.shape != (16, 8):
        raise ValueError('Control panel differs from 16 images x 8 actual tokens')
    if any(len(set(v)) != 8 for v in panel.values()):
        raise ValueError('Duplicate control token')
    expected_paths = [prod / f'{i:06d}.npz' for i in range(N)]
    actual_paths = sorted(prod.glob('*.npz'))
    if actual_paths != expected_paths:
        raise ValueError('Source filenames/count differ from exactly 1500 artifacts')

    source_inventory = []
    for i, path in enumerate(expected_paths):
        st = path.stat()
        source_inventory.append({'index': i,
            'relative_path': path.relative_to(source_root).as_posix(),
            'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': file_sha(path)})
        if (i+1) % 100 == 0:
            print(f'FREEZE {i+1}/{N} elapsed={time.monotonic()-started:.1f}s', flush=True)
    for filename, blob in [('MANIFEST_1500.jsonl', manifest_blob), ('CONTROL_PANEL.json', panel_blob)]:
        source_inventory.append({'relative_path': 'provenance/' + filename,
                                 'bytes': len(blob), 'sha256': sha(blob)})
    frozen_hash = tree_hash(source_inventory)
    save_json(out / 'SOURCE_FREEZE.json', {'source_root': str(source_root),
        'tree_hash_algorithm': 'sha256(relative_path + NUL + sha256 + LF), ordinal path sort',
        'tree_sha256': frozen_hash, 'files': source_inventory})
    selected, strata, film_count = stratified_sample(rows)

    # Streaming centered covariance (Chan merge), exclusively float64.
    count = 0
    mean = np.zeros(DIMS, dtype=np.float64)
    m2 = np.zeros((DIMS, DIMS), dtype=np.float64)
    for j, i in enumerate(selected):
        with read_frozen(expected_paths[i], source_inventory[i]['sha256']) as z:
            x = states(z).reshape(-1, DIMS).astype(np.float64)
        batch_count = len(x)
        batch_mean = x.mean(axis=0)
        centered = x - batch_mean
        difference = batch_mean - mean
        total = count + batch_count
        m2 += centered.T @ centered + np.outer(difference, difference) * (count * batch_count / total)
        mean += difference * batch_count / total
        count = total
        if (j+1) % 8 == 0:
            print(f'PCA {j+1}/64 points={count} elapsed={time.monotonic()-started:.1f}s', flush=True)
    covariance = m2 / (count-1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = vectors[:, order[:3]].copy()
    for column in range(3):
        pivot = int(np.abs(components[:, column]).argmax())
        if components[pivot, column] < 0:
            components[:, column] *= -1
    ratios = eigenvalues[:3] / np.trace(covariance)
    finite(mean, 'PCA mean'); finite(components, 'PCA components'); finite(ratios, 'PCA ratios')
    orthogonality_error = float(np.max(np.abs(components.T @ components - np.eye(3))))
    if orthogonality_error > 1e-10 or (eigenvalues[:3] <= 0).any():
        raise ValueError('Invalid PCA eigensystem')

    data = {
        'positions': np.empty((N, 3, TOKENS, 3), np.float32),
        'movement_norm': np.empty((N, 3, TOKENS), np.float32),
        'after_positions': np.empty((N, TOKENS, 3), np.float32),
        'after_norm': np.empty((N, TOKENS), np.float32),
        'control_positions': np.empty((16, 501, 8, 3), np.float32),
        'control_norm': np.empty((16, 501, 8), np.float32),
        'control_step_norm': np.empty((16, 501, 8), np.float32),
        'control_images': control_images, 'control_tokens': control_tokens,
        'confidence': np.empty((N, 501), np.float32),
        'iou': np.empty((N, 501), np.float32),
        'milestone_steps': STEPS, 'control_steps': np.arange(501, dtype=np.int32),
        'image_indices': np.arange(N, dtype=np.int32),
        'token_indices': np.arange(TOKENS, dtype=np.int32),
        'pca_mean': mean, 'pca_components': components,
        'pca_explained_variance': eigenvalues[:3],
        'pca_explained_variance_ratio': ratios,
        'pca_fit_image_indices': np.asarray(selected, dtype=np.int32),
    }
    controls_by_image = {int(v): k for k, v in enumerate(control_images)}
    metadata = []
    weights_hashes = set()
    for i, path in enumerate(expected_paths):
        with read_frozen(path, source_inventory[i]['sha256']) as z:
            x = states(z)
            meta = json.loads(z['metadata_json_utf8'].tobytes().decode('utf-8'))
            row = rows[i]
            for key in ('index','image_id','annotation_id','frame_id','image_name','image_sha256'):
                if meta[key] != row[key]:
                    raise ValueError(f'Metadata identity mismatch {i} {key}')
            if meta['manifest_sha256'] != MANIFEST_HASH or meta['checkpoint_sha256'] != CHECKPOINT_HASH:
                raise ValueError('Wrong checkpoint/manifest family')
            if meta['horizon'] != HORIZON or not np.array_equal(z['telemetry_step'], np.arange(501)):
                raise ValueError('Missing genuine telemetry steps')
            if not np.allclose(meta['bbox_xyxy_480'], row['bbox_xyxy_480'], rtol=0, atol=1e-9):
                raise ValueError('GT provenance mismatch')
            weights_hashes.add(meta['weights_buffers_hash'])
            data['positions'][i] = project(x, mean, components)
            dx = x.astype(np.float64) - x[0].astype(np.float64)
            data['movement_norm'][i] = np.sqrt(np.sum(dx*dx, axis=-1))
            after = z['after']; finite(after, 'after')
            data['after_positions'][i] = project(after, mean, components)
            da = after.astype(np.float64)-x[0].astype(np.float64)
            data['after_norm'][i] = np.sqrt(np.sum(da*da, axis=-1))
            for key in ('confidence', 'iou'):
                a = z['telemetry_'+key]
                if a.shape != (501,) or ((a < 0) | (a > 1)).any():
                    raise ValueError('Invalid telemetry '+key)
                finite(a, key); data[key][i] = a
            if i in controls_by_image:
                k = controls_by_image[i]
                tokens = control_tokens[k]
                if not np.array_equal(z['control_tokens'], tokens):
                    raise ValueError('Wrong true control token IDs')
                control = z['control_z']; finite(control, 'control_z')
                if control.shape != (501, 8, DIMS):
                    raise ValueError('Missing genuine control states')
                for step_index, step in enumerate(STEPS):
                    if not np.array_equal(control[int(step)], x[step_index, tokens]):
                        raise ValueError('Control/milestone raw mismatch')
                data['control_positions'][k] = project(control, mean, components)
                dc = control.astype(np.float64)-control[0].astype(np.float64)
                data['control_norm'][k] = np.sqrt(np.sum(dc*dc, axis=-1))
                dd = np.diff(control.astype(np.float64), axis=0)
                data['control_step_norm'][k,0] = 0
                data['control_step_norm'][k,1:] = np.sqrt(np.sum(dd*dd, axis=-1))
            meta['source_npz_sha256'] = source_inventory[i]['sha256']
            metadata.append(meta)
        if (i+1) % 100 == 0:
            elapsed = time.monotonic()-started
            print(f'PROJECT {i+1}/{N} elapsed={elapsed:.1f}s', flush=True)
    if len(weights_hashes) != 1:
        raise ValueError('Mixed frozen weights/buffers hashes')
    for key, a in data.items():
        finite(a, key)
        if a.dtype.hasobject:
            raise TypeError('Object arrays prohibited')

    # Fresh, disjoint-from-fit validation using an alternative double formula.
    fresh = [int(i) for i in np.rint(np.linspace(1, N-2, 32)).astype(int) if i not in selected][:16]
    errors = {'milestone_projection_max_abs':0., 'norm_max_abs':0.,
              'after_projection_max_abs':0., 'control_projection_max_abs':0.}
    for i in fresh:
        with read_frozen(expected_paths[i], source_inventory[i]['sha256']) as z:
            x = states(z).astype(np.float64)
            independent = np.einsum('...d,dc->...c', x, components)-mean @ components
            error = np.max(np.abs(data['positions'][i].astype(np.float64)-independent))
            errors['milestone_projection_max_abs'] = max(errors['milestone_projection_max_abs'], float(error))
            if not np.allclose(data['positions'][i], independent, rtol=2e-7, atol=3e-5):
                raise ValueError('Fresh projection verification failed')
            norm = np.linalg.norm(x-x[0], axis=-1)
            errors['norm_max_abs'] = max(errors['norm_max_abs'],float(np.max(np.abs(data['movement_norm'][i]-norm))))
            if not np.allclose(data['movement_norm'][i], norm, rtol=2e-7, atol=3e-5):
                raise ValueError('Full-768 norm verification failed')
            ap = z['after'].astype(np.float64) @ components-mean @ components
            errors['after_projection_max_abs'] = max(errors['after_projection_max_abs'],float(np.max(np.abs(data['after_positions'][i]-ap))))
    for k,i in enumerate(control_images):
        with read_frozen(expected_paths[int(i)], source_inventory[int(i)]['sha256']) as z:
            independent = np.einsum('...d,dc->...c', z['control_z'].astype(np.float64),components)-mean @ components
            error = float(np.max(np.abs(data['control_positions'][k]-independent)))
            errors['control_projection_max_abs'] = max(errors['control_projection_max_abs'],error)
            if not np.allclose(data['control_positions'][k],independent,rtol=2e-7,atol=3e-5):
                raise ValueError('Control projection verification failed')
    if not np.array_equal(data['movement_norm'][:,0],np.zeros((N,TOKENS),np.float32)):
        raise ValueError('Initial displacement must be exactly zero')

    schema = {key:{'shape':list(a.shape),'dtype':str(a.dtype)} for key,a in data.items()}
    basis_hash = sha(mean.astype('<f8').tobytes()+components.astype('<f8').tobytes())
    description = {
        'schema_version':SCHEMA_VERSION,'family':FAMILY,'source_family':SOURCE_FAMILY,
        'arrays':schema,'projection':'(z_float64 - pca_mean) @ pca_components; cast to float32',
        'basis_sha256_le_float64_mean_then_components':basis_hash,
        'axis_order':{'positions':['image_index','milestone_index','spatial_token_index','PCA_axis'],
            'control_positions':['control_image_slot','actual_step','control_token_slot','PCA_axis']},
        'movement_norm':'L2 in original 768D: z_step - z_before. Not a norm of PCA coordinates.',
        'control_norm':'L2 in original 768D relative to actual control_z[0].',
        'control_step_norm':'L2 in original 768D from previous actual iteration; step 0 is zero.',
        'after_positions':'best visited oracle target at metadata after_step; NOT generally step 500',
        'milestones':STEPS.tolist(),'exact_steps':list(range(501)),
        'all_particle_count':N*TOKENS,'exact_control_trajectory_count':128,
        'spatial_grid':{'rows':20,'columns':30,'index':'row * 30 + column'},
        'preview_contract':{'default_subsample':8000,'denominator':900000,
            'label':'8000 / 900000 cząsteczek (próbka)',
            'individual_image_tokens':600,'exact_mode':'Only the 128 listed control trajectories',
            'backend_sampling':'App owner; exporter does not remove any milestone particles'},
        'no_interpolation':True,'channel_standardization':False,
        'fit':{'seed':SEED,'sample_images':selected,'strata':strata,'source_films':film_count,
            'strategy':'64 equal bands of ordered film IDs; seeded film and frame choice in each band',
            'tokens_per_image':600,'states_per_image':[0,100,500],'total_fit_points':count,
            'covariance_dtype':'float64','covariance_algorithm':'centered batch covariance + Chan merge; divide N-1',
            'eigensolver':'numpy.linalg.eigh','sign_rule':'largest absolute loading positive',
            'explained_variance_ratio':ratios.tolist(),'explained_ratio_total':float(ratios.sum()),
            'no_refit_per_image_or_step':True,'orthogonality_max_abs_error':orthogonality_error},
        'checkpoint_sha256':CHECKPOINT_HASH,'manifest_sha256':MANIFEST_HASH,
        'control_panel_sha256':PANEL_HASH,'weights_buffers_hash':next(iter(weights_hashes)),
        'source_tree_sha256':frozen_hash,'images':metadata,
        'limitations':['3D projection loses information; ratios quantify fit-sample variance, not task utility.',
            'Trajectories are oracle activation edits with frozen weights, not weight-learning trajectories.',
            'No source pixels or image archives were accessed.'],
    }
    save_json(out/'metadata.json',description)
    data['schema_json_utf8'] = np.frombuffer(json.dumps({k:v for k,v in description.items() if k!='images'},
        ensure_ascii=False,allow_nan=False).encode('utf-8'),dtype=np.uint8).copy()
    temporary = out/'projection.npz.tmp'
    with temporary.open('wb') as f:np.savez_compressed(f,**data)
    temporary.replace(out/'projection.npz')

    # Recheck every source AFTER all calculations, before making a transferable bundle.
    for j,rec in enumerate(source_inventory):
        path=source_root/rec['relative_path']
        if file_sha(path)!=rec['sha256']:
            raise RuntimeError('Source changed during export; bundle must not be transferred')
        if (j+1)%100==0:print(f'RECHECK {j+1}/{len(source_inventory)}',flush=True)
    (out/'MANIFEST_1500.jsonl').write_bytes(manifest_blob)
    (out/'CONTROL_PANEL.json').write_bytes(panel_blob)
    validation={'finite_all_arrays':True,'object_arrays':False,'allow_pickle':False,
        'source_hashes_frozen_and_rechecked':True,'source_artifact_count':N,
        'source_compressed_bytes':sum(r['bytes'] for r in source_inventory[:N]),
        'fresh_nonfit_validation_images':fresh,'errors':errors,
        'true_control_milestone_bitwise_consistency':True,'schema':schema,
        'cpu_threads':2,'numpy_version':np.__version__,'python_version':platform.python_version(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'host':platform.node(),
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'elapsed_seconds':time.monotonic()-started,'script_sha256':file_sha(Path(__file__)),
        'basis_sha256':basis_hash,'source_tree_sha256':frozen_hash}
    save_json(out/'VALIDATION.json',validation)
    names=['projection.npz','metadata.json','SOURCE_FREEZE.json','VALIDATION.json',
           'MANIFEST_1500.jsonl','CONTROL_PANEL.json']
    bundle={'schema_version':1,'files':[{'filename':name,'bytes':(out/name).stat().st_size,
                                        'sha256':file_sha(out/name)} for name in names]}
    if (out/'projection.npz').stat().st_size>=70_000_000:
        raise ValueError('Projected compressed NPZ exceeds 70 MB contract')
    save_json(out/'BUNDLE_MANIFEST.json',bundle)
    verify(out)
    print('EXPORT_COMPLETE '+json.dumps({'npz_bytes':(out/'projection.npz').stat().st_size,
        'basis_ratios':ratios.tolist(),'seconds':time.monotonic()-started}),flush=True)


def verify(out):
    bundle=json.loads((out/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    npz_name=next(item['filename'] for item in bundle['files'] if item['filename'].endswith('.npz'))
    if npz_name not in ('projection.npz','particles.npz'):
        raise ValueError('Unexpected projected NPZ filename')
    for item in bundle['files']:
        p=out/item['filename']
        if p.stat().st_size!=item['bytes'] or file_sha(p)!=item['sha256']:
            raise ValueError('Bundle integrity failure: '+str(p))
    metadata=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    expected={'positions':(1500,3,600,3),'movement_norm':(1500,3,600),
        'control_positions':(16,501,8,3),'control_norm':(16,501,8),
        'control_images':(16,),'control_tokens':(16,8),'confidence':(1500,501),'iou':(1500,501)}
    with np.load(out/npz_name,allow_pickle=False) as z:
        for key in z.files:
            a=z[key];finite(a,key)
            if a.dtype.hasobject:raise ValueError('Unsafe object array')
        for key,shape in expected.items():
            if z[key].shape!=shape:raise ValueError((key,z[key].shape))
        if not np.array_equal(z['milestone_steps'],STEPS):raise ValueError('Wrong milestones')
        if not np.array_equal(z['control_steps'],np.arange(501)):raise ValueError('Wrong exact steps')
        for k,i in enumerate(z['control_images']):
            tokens=z['control_tokens'][k]
            for j,step in enumerate(STEPS):
                if not np.array_equal(z['control_positions'][k,int(step)],z['positions'][int(i),j,tokens]):
                    raise ValueError('Projected control/milestone alignment error')
        embedded=json.loads(z['schema_json_utf8'].tobytes().decode('utf-8'))
        if embedded['basis_sha256_le_float64_mean_then_components']!=metadata['basis_sha256_le_float64_mean_then_components']:
            raise ValueError('Embedded/external schema mismatch')
        if sha(z['pca_mean'].astype('<f8').tobytes()+z['pca_components'].astype('<f8').tobytes())!=metadata['basis_sha256_le_float64_mean_then_components']:
            raise ValueError('Basis hash mismatch')
    print('VERIFY_OK '+json.dumps({'files':len(bundle['files']),
        'bytes':sum(r['bytes'] for r in bundle['files']),
        'npz_sha256':next(r['sha256'] for r in bundle['files'] if r['filename']==npz_name)}),flush=True)


def finalize_name(out):
    """Rename an already verified old-name bundle without changing array bytes."""
    verify(out)
    executed=out/'export_projection_executed.py'
    validation=json.loads((out/'VALIDATION.json').read_text(encoding='utf-8'))
    if file_sha(executed)!=validation['script_sha256']:
        raise ValueError('Executed source snapshot differs from remote source hash')
    original=(out/'BUNDLE_MANIFEST.json').read_bytes()
    if (out/'particles.npz').exists():
        if (out/'projection.npz').exists():raise FileExistsError('Both NPZ names exist')
        (out/'particles.npz').rename(out/'projection.npz')
    else:raise FileNotFoundError('Expected original particles.npz for one-time normalization')
    (out/'BUNDLE_MANIFEST_EDEN.json').write_bytes(original)
    bundle=json.loads(original)
    for item in bundle['files']:
        if item['filename']=='particles.npz':item['filename']='projection.npz'
    for name in ('BUNDLE_MANIFEST_EDEN.json','export_projection_executed.py'):
        p=out/name
        bundle['files'].append({'filename':name,'bytes':p.stat().st_size,'sha256':file_sha(p)})
    save_json(out/'BUNDLE_MANIFEST.json',bundle)
    verify(out)
    save_json(out/'LOCAL_TRANSFER.json',{
        'verified_local_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'local_name':'projection.npz','remote_name':'particles.npz',
        'array_bytes_unchanged':True,'all_original_remote_bundle_hashes_match':True,
        'executed_script_hash_matches_remote_validation':True,
        'original_remote_bundle_manifest':'BUNDLE_MANIFEST_EDEN.json',
        'source_freeze_rechecked_remote':validation['source_hashes_frozen_and_rechecked'],
        'final_npz_sha256':file_sha(out/'projection.npz'),
        'local_python_version':platform.python_version(),'local_numpy_version':np.__version__})


def finish_existing(source_root, out):
    """Recover completed arrays after a validation-JSON serialization failure.

    Reuses the exact saved PCA/coordinates. Revalidates fresh raw sources and
    every frozen hash on a CPU allocation; does not refit or reproject all data.
    """
    started=time.monotonic()
    if (out/'BUNDLE_MANIFEST.json').exists():
        raise FileExistsError('Bundle is already complete')
    old_name='particles.npz' if (out/'particles.npz').exists() else 'projection.npz'
    frozen=json.loads((out/'SOURCE_FREEZE.json').read_text(encoding='utf-8'))
    description=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    records=frozen['files']
    by_index={int(r['index']):r for r in records if 'index' in r}
    if len(by_index)!=1500 or tree_hash(records)!=frozen['tree_sha256']:
        raise ValueError('Invalid source freeze')
    with np.load(out/old_name,allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    for k,a in arrays.items():
        finite(a,k)
        if a.dtype.hasobject:raise ValueError('Object arrays forbidden')
    mean=arrays['pca_mean'];components=arrays['pca_components']
    basis_hash=sha(mean.astype('<f8').tobytes()+components.astype('<f8').tobytes())
    if basis_hash!=description['basis_sha256_le_float64_mean_then_components']:
        raise ValueError('Saved basis hash mismatch')
    for key, schema in description['arrays'].items():
        if list(arrays[key].shape)!=schema['shape'] or str(arrays[key].dtype)!=schema['dtype']:
            raise ValueError('Saved array schema mismatch')
    manifest_path=source_root/'provenance/MANIFEST_1500.jsonl'
    panel_path=source_root/'provenance/CONTROL_PANEL.json'
    manifest_blob=manifest_path.read_bytes();panel_blob=panel_path.read_bytes()
    if sha(manifest_blob)!=MANIFEST_HASH or sha(panel_blob)!=PANEL_HASH:
        raise ValueError('Provenance mismatch during finalization')
    rows=[json.loads(line) for line in manifest_blob.decode('utf-8-sig').splitlines()]
    selected,_,_=stratified_sample(rows)
    if selected!=arrays['pca_fit_image_indices'].tolist():
        raise ValueError('PCA fit sample is not the fixed deterministic sample')
    fresh=[int(i) for i in np.rint(np.linspace(1,N-2,32)).astype(int) if i not in selected][:16]
    errors={'milestone_projection_max_abs':0.,'norm_max_abs':0.,
            'after_projection_max_abs':0.,'control_projection_max_abs':0.,
            'control_norm_max_abs':0.,'control_step_norm_max_abs':0.}
    for i in fresh:
        rec=by_index[i]
        with read_frozen(source_root/rec['relative_path'],rec['sha256']) as z:
            x=states(z).astype(np.float64)
            tests={
                'positions':(np.einsum('...d,dc->...c',x,components)-mean@components,'milestone_projection_max_abs'),
                'movement_norm':(np.linalg.norm(x-x[0],axis=-1),'norm_max_abs'),
                'after_positions':(z['after'].astype(np.float64)@components-mean@components,'after_projection_max_abs')}
            for key,(expected,error_key) in tests.items():
                errors[error_key]=max(errors[error_key],float(np.max(np.abs(arrays[key][i]-expected))))
                if not np.allclose(arrays[key][i],expected,rtol=2e-7,atol=3e-5):
                    raise ValueError('Fresh verification failed: '+key)
    for k,image in enumerate(arrays['control_images']):
        i=int(image);rec=by_index[i]
        with read_frozen(source_root/rec['relative_path'],rec['sha256']) as z:
            c=z['control_z'].astype(np.float64);x=states(z)
            if not np.array_equal(z['control_tokens'],arrays['control_tokens'][k]):
                raise ValueError('Actual control token mapping differs')
            for j,step in enumerate(STEPS):
                if not np.array_equal(c[int(step)],x[j,arrays['control_tokens'][k]]):
                    raise ValueError('Raw control/milestone mismatch')
            step_norm=np.zeros((501,8),np.float64)
            step_norm[1:]=np.linalg.norm(np.diff(c,axis=0),axis=-1)
            tests={
                'control_positions':(np.einsum('...d,dc->...c',c,components)-mean@components,'control_projection_max_abs'),
                'control_norm':(np.linalg.norm(c-c[0],axis=-1),'control_norm_max_abs'),
                'control_step_norm':(step_norm,'control_step_norm_max_abs')}
            for key,(expected,error_key) in tests.items():
                errors[error_key]=max(errors[error_key],float(np.max(np.abs(arrays[key][k]-expected))))
                if not np.allclose(arrays[key][k],expected,rtol=2e-7,atol=3e-5):
                    raise ValueError('Control verification failed: '+key)
    for j,rec in enumerate(records):
        if file_sha(source_root/rec['relative_path'])!=rec['sha256']:
            raise ValueError('Source changed during recovery')
        if (j+1)%100==0:print(f'FINAL_RECHECK {j+1}/{len(records)}',flush=True)
    (out/'MANIFEST_1500.jsonl').write_bytes(manifest_blob)
    (out/'CONTROL_PANEL.json').write_bytes(panel_blob)
    original=out/'export_projection_executed.py'
    validation={
        'finite_all_arrays':True,'object_arrays':False,'allow_pickle':False,
        'source_hashes_frozen_and_rechecked':True,'source_artifact_count':N,
        'source_compressed_bytes':sum(r['bytes'] for r in by_index.values()),
        'fresh_nonfit_validation_images':fresh,'errors':errors,
        'true_control_milestone_bitwise_consistency':True,'schema':description['arrays'],
        'cpu_threads':2,'numpy_version':np.__version__,'python_version':platform.python_version(),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'initial_export_slurm_job_id':'1825060',
        'host':platform.node(),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'finalization_elapsed_seconds':time.monotonic()-started,
        'script_sha256':file_sha(original),'recovery_script_sha256':file_sha(Path(__file__)),
        'basis_sha256':basis_hash,'source_tree_sha256':frozen['tree_sha256'],
        'recovered_after_numpy_int64_json_serialization_error':True,
        'pca_refitted_during_recovery':False,'array_bytes_changed_during_recovery':False}
    save_json(out/'VALIDATION.json',validation)
    names=[old_name,'metadata.json','SOURCE_FREEZE.json','VALIDATION.json','MANIFEST_1500.jsonl','CONTROL_PANEL.json']
    bundle={'schema_version':1,'files':[{'filename':name,'bytes':(out/name).stat().st_size,
            'sha256':file_sha(out/name)} for name in names]}
    if (out/old_name).stat().st_size>=70_000_000:raise ValueError('NPZ exceeds 70 MB')
    save_json(out/'BUNDLE_MANIFEST.json',bundle)
    verify(out)
    print('FINALIZATION_COMPLETE '+json.dumps({'npz_bytes':(out/old_name).stat().st_size,
        'seconds':time.monotonic()-started,'errors':errors}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['export','verify','finalize-name','finish-existing'])
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='export':
        if args.source_root is None:parser.error('--source-root required for export')
        export(args.source_root,args.output)
    elif args.mode=='finish-existing':
        if args.source_root is None:parser.error('--source-root required for finalization')
        finish_existing(args.source_root,args.output)
    elif args.mode=='finalize-name':finalize_name(args.output)
    else:verify(args.output)
