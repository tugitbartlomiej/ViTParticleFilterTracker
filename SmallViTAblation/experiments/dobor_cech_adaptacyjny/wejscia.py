"""Fingerprint actual input archives; manifests alone cannot establish pixel provenance."""
import argparse
from pathlib import Path
from .dane import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data-archive', 'context-archive', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    write_json(a.out, {name: dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path))
                      for name, path in [('data', a.data_archive), ('context', a.context_archive)]})


if __name__ == '__main__':
    main()
