"""Bind the Eden test-pass marker to the exact implementation and execution scripts."""
import argparse
import json
import os
from pathlib import Path
from .dane import sha256


def source_hashes():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and (p.suffix in ('.py', '.sh', '.slurm') or p.name in ('config.json', 'requirements-eden-tests.txt'))}


def verify(path):
    marker = json.loads(Path(path).read_text(encoding='utf-8'))
    if marker.get('status') != 'passed' or marker.get('code_sha256') != source_hashes():
        raise RuntimeError('Eden tests have not passed for this exact code/configuration; run tests first')
    if os.environ.get('INPUT_PROVENANCE'):
        inputs = json.loads(Path(os.environ['INPUT_PROVENANCE']).read_text(encoding='utf-8'))
        hashes = {name: row['sha256'] for name, row in inputs.items()}
        if marker.get('input_archives_sha256') != hashes:
            raise RuntimeError('input archives differ from the completed Eden integration test')
    return marker


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('test_pass_file', type=Path)
    verify(p.parse_args().test_pass_file)
