"""Bundle completed Eden run, final checkpoint and a verified file manifest."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    out = root / 'results_50k'
    report = json.loads((out / 'REPORT.json').read_text())
    if report['status'] != 'completed' or report['step'] != 50000:
        raise RuntimeError('Training did not complete the agreed budget')
    final = out / 'checkpoint_050000.pth'
    if not final.is_file():
        raise FileNotFoundError(final)
    paths = [p for p in sorted(out.iterdir()) if p.is_file() and p.suffix != '.pth' and not p.name.endswith('.tmp')]
    paths.append(final)
    paths += [root / x for x in ('GPU.csv', 'ENVIRONMENT.txt', 'INPUT_SHA256SUMS', 'eden_fastgan.slurm')]
    files = {p.relative_to(root).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}
    temp = root / 'wynik_koncowy.tar.tmp'
    with tarfile.open(temp, 'w') as archive:
        for p in paths:
            archive.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
    bundle = root / 'wynik_koncowy.tar'
    os.replace(temp, bundle)
    manifest = dict(status='completed', step=report['step'], archive=bundle.name,
                    archive_sha256=sha(bundle), archive_bytes=bundle.stat().st_size, files=files)
    temporary = root / 'DOWNLOAD.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(temporary, root / 'DOWNLOAD.json')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
