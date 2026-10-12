"""Eksport zakończonego pretreningu; świadome oddanie sterowania komputerowi.

Nie zatrzymuje treningu. Release wolno wywołać dopiero po sprawdzonym pobraniu
pakietu. Kończy wyłącznie uprzednio zamrożony orkiestrator tego jednego runu.
"""
import argparse
import os
from pathlib import Path
import signal
import shutil
import tarfile
from dane import read_json, write_json, sha256

ROOT=Path('/home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla')
RUN=ROOT/'results/run_1810448'
ARCHIVE=Path('/home2/faculty/bpiotrowski/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar')


def completed():
    status=read_json(RUN/'pretrain/STATUS.json')
    cfg=read_json(RUN/'config.json')
    expected=cfg['pretrain_epochs']*cfg['epoch_size']//(cfg['batch_size']*cfg['grad_accum_steps'])
    return status['state']=='completed' and status['global_step']==expected


def pack():
    if not completed():return {'state':'waiting_for_pretrain'}
    output=ROOT/'handoff_pretrain_1810448.tar'
    manifest=output.with_suffix('.manifest.json')
    if output.exists() and manifest.exists():return read_json(manifest)
    inputs=read_json(RUN/'INPUTS.json');args=inputs['arguments']
    pool=ROOT/'hybrid_inputs/pool_manifest.jsonl';pool.parent.mkdir(exist_ok=True)
    shutil.copyfile(args['pool'],pool)
    if sha256(pool)!=inputs['pool_sha256']:raise ValueError('Pool changed since training')
    old_root=Path(args['data_root']).resolve(strict=True)
    image_hashes={}
    for split in ('train','valid'):
        for image in read_json(RUN/'datasets/pretrain'/split/'_annotations.coco.json')['images']:
            path=Path(image['file_name']).resolve(strict=True)
            if not path.is_relative_to(old_root):raise ValueError('Image outside original data root')
            image_hashes[path.relative_to(old_root).as_posix()]=sha256(path)
    # These are the actual files used by the completed training, still present
    # on its node before cleanup. Bind the evidence to the exact checkpoint.
    write_json(RUN/'pretrain/IMAGE_BYTES.json',dict(checkpoint_sha256=sha256(RUN/'pretrain/FINAL.ckpt'),
        data_root=str(old_root),files=image_hashes))
    reference=dict(checkpoint=str(RUN/'pretrain/FINAL.ckpt'),dataset=str(RUN/'datasets/pretrain'),
        data_root=args['data_root'],probe=str(RUN/'probe/manifest.json'),config=str(RUN/'config.json'),
        initial_weights=args['initial_weights'],pool=str(pool),pool_sha256=sha256(pool),
        data_archive_sha256=sha256(ARCHIVE),source_job=1810448,image_manifest=str(RUN/'pretrain/IMAGE_BYTES.json'))
    reference['archive_sha256']=reference['data_archive_sha256']
    reference['archive_hash_provenance']='computed_at_handoff; not recorded before original pretraining'
    write_json(RUN/'pretrain/REMOTE.json',reference)
    files=[]
    allowed={'.json','.npz','.png','.npy'}
    paths=[RUN/'config.json',RUN/'INPUTS.json',*sorted((RUN/'probe').rglob('*')),
           *sorted((RUN/'pretrain').glob('*')),*sorted((RUN/'datasets/pretrain').rglob('*.json'))]
    temporary=output.with_suffix('.partial')
    with tarfile.open(temporary,'w') as archive:
        for path in paths:
            if not path.is_file() or path.is_symlink():continue
            if path.suffix not in allowed and path.name not in ('FINAL.ckpt','FINAL.pth'):continue
            name=path.relative_to(RUN).as_posix()
            archive.add(path,arcname=name,recursive=False)
            files.append(dict(file=name,bytes=path.stat().st_size,sha256=sha256(path)))
    temporary.replace(output)
    record=dict(state='ready',archive=str(output),archive_sha256=sha256(output),files=files,
                checkpoint_sha256=sha256(RUN/'pretrain/FINAL.ckpt'))
    write_json(manifest,record)
    write_json(RUN/'HANDOFF.json',dict(state='package_ready',manifest=str(manifest),parent_pid=2615518))
    return record


def release(checkpoint_sha):
    if not completed() or sha256(RUN/'pretrain/FINAL.ckpt')!=checkpoint_sha:
        raise ValueError('Pretrain completion / downloaded checkpoint mismatch')
    # Read /proc on the allocated node, not on the login node.
    process=Path('/proc/2615518')
    if not process.exists():raise ValueError('Expected orchestrator is absent')
    argv=process.joinpath('cmdline').read_bytes().split(b'\0')
    if b'--out' not in argv or argv[argv.index(b'--out')+1].decode()!=str(RUN):
        raise ValueError('PID no longer belongs to this run')
    if not any(v.endswith(b'/petla.py') for v in argv):raise ValueError('Wrong process type')
    state=process.joinpath('stat').read_text().split(') ',1)[1].split()[0]
    if state not in ('T','t'):raise ValueError('Orchestrator was not held at handoff')
    child=Path('/proc/2616171/stat')
    if child.exists() and child.read_text().split(') ',1)[1].split()[0]!='Z':
        raise ValueError('Training process has not exited')
    write_json(RUN/'HANDOFF.json',dict(state='release_requested',checkpoint_sha256=checkpoint_sha))
    os.kill(2615518,signal.SIGKILL)
    write_json(RUN/'HANDOFF.json',dict(state='released_to_local',checkpoint_sha256=checkpoint_sha,
        note='Intentional orchestration termination after completed pretrain; not a training failure.'))
    write_json(RUN/'STATUS.json',dict(state='handed_off_to_local',stage='local_analysis',source_job=1810448))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['pack','release'])
    ap.add_argument('--checkpoint-sha');a=ap.parse_args()
    if a.action=='pack':
        result=pack()
        print(__import__('json').dumps({k:v for k,v in result.items() if k!='files'}))
    else:release(a.checkpoint_sha)
