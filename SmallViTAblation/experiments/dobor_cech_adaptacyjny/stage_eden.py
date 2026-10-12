"""Unpack trusted experiment archives into a new node-local directory, rejecting links/traversal."""
import argparse
from pathlib import Path
import tarfile


def extract(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    resolved = destination.resolve()
    with tarfile.open(archive) as source:
        for member in source:
            target = (destination/member.name).resolve()
            if not target.is_relative_to(resolved) or not (member.isfile() or member.isdir()):
                raise ValueError(f'unsafe archive member: {member.name}')
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as src, target.open('xb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
            else:
                target.mkdir(parents=True, exist_ok=True)
    matches = list(destination.rglob('pool_manifest.jsonl'))
    if len(matches) != 1:
        raise ValueError('archive must contain exactly one pool_manifest.jsonl')
    return matches[0].parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    print(extract(a.archive, a.destination))


if __name__ == '__main__':
    main()
