"""Przeniesienie własnego, zaufanego FINAL bez zmiany stanu uczenia.

CLI: --checkpoint --dataset --probe --old-validation-json --newfile
     --old-data-root --new-data-root [--image-manifest IMAGE_BYTES.json]

Domyślnie wymagany jest dowód bajtów obrazów: dostępny stary plik albo
experiment_contract.image_files {ścieżka względna POSIX lub stara absolutna: SHA256}.
Alternatywa: IMAGE_BYTES.json z checkpoint_sha256, data_root i files {relative:SHA256},
sporządzony przez zaufany handoff przed usunięciem starego scratch. Sprawdzamy hash
źródłowego pliku checkpointu, stary root, kompletność i wszystkie nowe bajty.
Stary kontrakt trening.py nie zawiera image_files. Gdy stare pliki zniknęły,
helper odmawia zapisu. --allow-unverified-image-bytes pozwala jawnie przenieść
sam kontrakt semantyczny, np. po OSOBNEJ kontroli archiwum źródłowego; raport
w checkpointcie nadal nie twierdzi, że zweryfikowano bajty ani SHA archiwum.
Nowe hashe są zapisywane jako obserwacja, nie jako dowód starej zawartości.
Manifest sondy i wszystkie jego obrazy/maski zawsze wymagają starych hashy.
Nie zmienia recipe, model_config, hparams, wag, optimizerów, RNG ani loops.
W szczególności nie naprawia innych ścieżek w model_config: muszą nadal pasować.
Używać datasetu rodzica, bez dopisywania nowej klatki podczas przenoszenia.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import tempfile


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _pure(value):
    s = str(value)
    return PureWindowsPath(s) if ('\\' in s or (len(s) > 1 and s[1] == ':')) else PurePosixPath(s)


def _relative(value, root):
    p, r = _pure(value), _pure(root)
    if type(p) is not type(r) or not p.is_absolute() or not r.is_absolute():
        raise ValueError('Paths must be absolute and use the same path flavour')
    if '..' in p.parts or '..' in r.parts:
        raise ValueError('Parent traversal is forbidden')
    try:
        rel = p.relative_to(r)
    except ValueError as exc:
        raise ValueError(f'Image outside allowed root: {value}') from exc
    if not rel.parts:
        raise ValueError('Image path equals data root')
    return rel.as_posix()


def _inside(root, rel):
    root = Path(root).resolve(strict=True)
    p = (root / rel).resolve(strict=True)
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError('Image resolves outside data root or is not a file')
    return p


def _canonical_coco(coco, root):
    result = deepcopy(coco)
    images = result['images']
    ids = [im['id'] for im in images]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate COCO image IDs')
    ann_ids = [a['id'] for a in result['annotations']]
    if len(ann_ids) != len(set(ann_ids)):
        raise ValueError('Duplicate annotation IDs')
    for a in result['annotations']:
        if a['image_id'] not in ids:
            raise ValueError('Annotation refers to missing image')
    for im in images:
        im['file_name'] = _relative(im['file_name'], root)
    # Deliberately preserve list order, IDs, geometry and all extra fields.
    return result


def rebase_checkpoint(checkpoint, *, dataset, probe, old_validation_json,
                      old_data_root, new_data_root, allow_unverified_image_bytes=False,
                      image_manifest=None, source_checkpoint_sha256=None):
    """Return (new_checkpoint, audit). No mutation of input; no CUDA operations.

    Heavy training state is shared unchanged in memory; only the contract is copied.
    The caller must keep the input immutable until saving the returned checkpoint.
    With image_manifest, the in-memory caller must supply source_checkpoint_sha256
    computed from the exact file it loaded. A dict cannot recover a file's digest.
    Prefer rebase_file, which computes and supplies that digest itself.
    """
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'loops', 'experiment_rng'):
        if not checkpoint.get(key):
            raise ValueError(f'Missing full checkpoint state: {key}')
    if checkpoint.get('experiment_boundary') != 'before_next_epoch_v1':
        raise ValueError('Expected canonical FINAL checkpoint')
    old = checkpoint['experiment_contract']
    if checkpoint.get('experiment_recipe') != old['recipe'] or checkpoint.get('experiment_mapping') != old['mapping']:
        raise ValueError('Inconsistent duplicated recipe/mapping')
    dataset, probe = Path(dataset), Path(probe)
    mapping = _json(dataset / 'mapping.json')['train']
    if mapping != old['mapping'] or len({r['frame_id'] for r in mapping}) != len(mapping):
        raise ValueError('Changed sample identities/order or duplicates')
    if _sha(old_validation_json) != old['validation_sha256']:
        raise ValueError('Original validation JSON does not match checkpoint provenance')
    if _sha(probe) != old['probe_sha256']:
        raise ValueError('Changed probe manifest')
    probe_hashes = {}
    for row in _json(probe):
        for key in ('image', 'mask'):
            rel = row[key]
            p = _pure(rel)
            if p.is_absolute() or '..' in p.parts or '\\' in rel:
                raise ValueError('Probe paths must be confined relative POSIX paths')
            probe_hashes[rel] = _sha(_inside(probe.parent, rel))
    if probe_hashes != old['probe_files']:
        raise ValueError('Changed probe image/mask bytes')
    new_train = _json(dataset / 'train' / '_annotations.coco.json')
    new_val_path = dataset / 'valid' / '_annotations.coco.json'
    pairs = [(old['annotations'], new_train), (_json(old_validation_json), _json(new_val_path))]
    manifest_files = {}
    manifest_sha = None
    if image_manifest is not None:
        payload = Path(image_manifest).read_bytes()
        manifest = json.loads(payload.decode('utf-8-sig'))
        manifest_sha = hashlib.sha256(payload).hexdigest()
        if not source_checkpoint_sha256 or manifest.get('checkpoint_sha256') != source_checkpoint_sha256:
            raise ValueError('Image manifest checkpoint SHA256 mismatch or missing source digest')
        if str(manifest.get('data_root')) != str(old_data_root):
            raise ValueError('Image manifest old data root mismatch')
        manifest_files = manifest.get('files')
        required = {_relative(im['file_name'], old_data_root) for before, _ in pairs for im in before['images']}
        if not isinstance(manifest_files, dict) or set(manifest_files) != required:
            raise ValueError('Image manifest must cover exactly old train and validation images')
        for rel, digest in manifest_files.items():
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError(f'Invalid image SHA256: {rel}')
    if len(new_train['images']) != len(mapping):
        raise ValueError('Mapping/image count mismatch')
    if [im['id'] for im in new_train['images']] != list(range(len(mapping))):
        raise ValueError('COCO IDs do not preserve mapping index order')
    observed, missing, evidence = {}, [], {}
    for before, after in pairs:
        if _canonical_coco(before, old_data_root) != _canonical_coco(after, new_data_root):
            raise ValueError('Changed COCO content beyond allowed root relocation')
        for source, dest in zip(before['images'], after['images']):
            rel = _relative(dest['file_name'], new_data_root)
            actual = _sha(_inside(new_data_root, rel))
            known = old.get('image_files', {})
            expected = known.get(rel, known.get(source['file_name']))
            origin = 'checkpoint_hash'
            if image_manifest is not None:
                supplied = manifest_files[rel]
                if expected is not None and expected != supplied:
                    raise ValueError(f'Conflicting historical image hashes: {rel}')
                expected = supplied
                origin = 'checkpoint_bound_image_manifest'
            # Never interpret a foreign-flavour absolute path as a local relative path.
            if expected is None and isinstance(_pure(source['file_name']), type(_pure(Path.cwd()))):
                old_path = Path(source['file_name'])
                if old_path.is_file():
                    expected = _sha(_inside(old_data_root, _relative(source['file_name'], old_data_root)))
                    origin = 'existing_old_file'
            if expected is None:
                missing.append(rel)
            elif actual != expected:
                raise ValueError(f'Changed image bytes: {rel}')
            else:
                evidence[rel] = origin
            observed[rel] = actual
    missing = sorted(set(missing))
    if missing and not allow_unverified_image_bytes:
        raise ValueError(f'No historical image byte proof for {len(missing)} files; archive proof must be verified separately')
    audit = dict(version=1, old_data_root=str(old_data_root), new_data_root=str(new_data_root),
                 train_count=len(mapping), validation_count=len(pairs[1][1]['images']),
                 image_bytes_verified=not missing, unverified_images=missing,
                 image_evidence=evidence, observed_image_sha256=observed,
                 probe_bytes_verified=True, archive_sha_verified=False,
                 previous_validation_sha256=old['validation_sha256'])
    if image_manifest is not None:
        audit.update(image_manifest_sha256=manifest_sha,
                     image_manifest_checkpoint_sha256=source_checkpoint_sha256,
                     image_manifest_binding_verified=True)
    result = dict(checkpoint)
    result['experiment_contract'] = deepcopy(old)
    result['experiment_contract']['annotations'] = new_train
    result['experiment_contract']['validation_sha256'] = _sha(new_val_path)
    # For proven files only, retain portable hash provenance across later relocations.
    if not missing:
        result['experiment_contract']['image_files'] = observed
    result['experiment_portability'] = audit
    return result, audit


def rebase_file(*, checkpoint, newfile, **kwargs):
    """Load a trusted own checkpoint on CPU; publish new file without overwrite."""
    import torch
    source, target = Path(checkpoint), Path(newfile)
    if target.exists() or source.resolve() == target.resolve():
        raise FileExistsError('Output must be a new file')
    source_sha = _sha(source)
    state = torch.load(source, map_location='cpu', weights_only=False)
    if _sha(source) != source_sha:
        raise ValueError('Source checkpoint changed while loading')
    if 'source_checkpoint_sha256' in kwargs:
        raise ValueError('rebase_file computes source_checkpoint_sha256 itself')
    result, audit = rebase_checkpoint(state, source_checkpoint_sha256=source_sha, **kwargs)
    audit['source_checkpoint_sha256'] = source_sha
    # Save fully before publishing. Hard link is atomic and refuses existing target.
    fd, tmp = tempfile.mkstemp(prefix='.checkpoint-rebase-', suffix='.tmp', dir=target.parent)
    os.close(fd)
    try:
        torch.save(result, tmp)
        os.link(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'dataset', 'probe', 'old-validation-json', 'newfile', 'old-data-root', 'new-data-root'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--allow-unverified-image-bytes', action='store_true')
    parser.add_argument('--image-manifest')
    args = vars(parser.parse_args())
    print(json.dumps(rebase_file(**args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
