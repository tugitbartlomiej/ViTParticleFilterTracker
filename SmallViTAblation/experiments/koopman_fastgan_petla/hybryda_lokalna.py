"""Lokalny koordynator: pobierz zakończony pretrening, analizuj i generuj tutaj.

Kolejne treningi/oceny zleca przez hybryda_zdalne; nie wznawia dawnej pętli
obliczeń na Edenie. Nie uczy ponownie FastGAN-a. Stan zapisuje do HYBRID_STATE.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time
import traceback
from dane import read_json,write_json,sha256

HERE=Path(__file__).resolve().parent
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla'
RPY='/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'


def extract_checked(archive_path,manifest_path,target):
    manifest=read_json(manifest_path)
    if sha256(archive_path)!=manifest['archive_sha256']:raise ValueError('Archive SHA256 mismatch')
    target=Path(target).resolve();target.mkdir(parents=True,exist_ok=False)
    entries={row['file']:row for row in manifest['files']}
    if len(entries)!=len(manifest['files']):raise ValueError('Duplicate manifest entry')
    checkpoint=entries.get('pretrain/FINAL.ckpt')
    if checkpoint is None or checkpoint['sha256']!=manifest['checkpoint_sha256']:
        raise ValueError('Release digest does not identify the downloaded full checkpoint')
    seen=set()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            name=PurePosixPath(member.name)
            dest=(target/member.name).resolve()
            if (not member.isfile() or name.is_absolute() or '..' in name.parts or ':' in member.name
                or '\\' in member.name or not dest.is_relative_to(target) or member.name not in entries
                or member.name in seen):raise ValueError('Unsafe or unexpected archive member')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,dest.open('xb') as output:
                __import__('shutil').copyfileobj(source,output)
            entry=entries[member.name]
            if dest.stat().st_size!=entry['bytes'] or sha256(dest)!=entry['sha256']:
                raise ValueError('Downloaded file hash mismatch')
            seen.add(member.name)
    if seen!=entries.keys():raise ValueError('Missing archive member')
    return manifest


def ssh(command):
    result=subprocess.run(['ssh','-o','BatchMode=yes','eden-cluster',command],capture_output=True,text=True,check=True)
    return result.stdout


def run(a):
    envcfg=read_json(a.env)
    if not envcfg.get('ready'):raise ValueError('Local environment has not passed preflight')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(state='running',stage='waiting_for_pretrain',pid=os.getpid(),out=str(out),source_job=1810448,
               env=str(Path(a.env).resolve()),started_unix=time.time())
    def save(stage,**values):
        state.update(stage=stage,updated_unix=time.time(),**values)
        write_json(HERE/'HYBRID_STATE.json',state);write_json(out/'STATUS.json',state)
    save('waiting_for_pretrain')
    while True:
        response=json.loads(ssh('srun --jobid=1810448 --overlap --immediate=5 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 '
                               f'{RPY} {REMOTE}/code/handoff_eden.py pack').strip())
        if response['state']=='ready':break
        if response['state']!='waiting_for_pretrain':raise ValueError('Unexpected remote handoff state')
        time.sleep(20)
    save('downloading_pretrain')
    for suffix in ('.tar','.manifest.json'):
        subprocess.run(['scp',f'eden-cluster:{REMOTE}/handoff_pretrain_1810448{suffix}',str(out)],check=True)
    manifest=extract_checked(out/'handoff_pretrain_1810448.tar',out/'handoff_pretrain_1810448.manifest.json',out/'import')
    write_json(out/'DOWNLOAD_VERIFIED.json',dict(archive_sha256=manifest['archive_sha256'],
        checkpoint_sha256=manifest['checkpoint_sha256'],files=len(manifest['files'])))
    save('releasing_remote_orchestrator')
    digest=manifest['checkpoint_sha256']
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid digest')
    try:
        ssh(f'srun --jobid=1810448 --overlap --immediate=5 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 '
            f'{RPY} {REMOTE}/code/handoff_eden.py release --checkpoint-sha {digest}')
    except subprocess.CalledProcessError:
        # Ending the batch allocation may end this read-only job step's shell.
        handoff=json.loads(ssh(f'cat {REMOTE}/results/run_1810448/HANDOFF.json'))
        if handoff.get('state')!='released_to_local' or handoff.get('checkpoint_sha256')!=digest:raise
    save('local_pipeline')
    cmd=[envcfg['python'],'-u','-B',str(HERE/'petla.py'),'--config',str(out/'import/config.json'),
         '--out',str(out/'run'),'--device','cuda','--imported-pretrain',str(out/'import'),'--remote-training']
    for key in ('pool','data_root','initial_selection','initial_weights','crop_manifest','crop_root','generator','vendor','hub','dino_weights'):
        cmd.extend(['--'+key.replace('_','-'),envcfg[key]])
    childenv=os.environ.copy()
    childenv.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    pythonpath=envcfg.get('pythonpath',[])
    if pythonpath:childenv['PYTHONPATH']=os.pathsep.join(pythonpath)
    write_json(out/'COMMAND.json',dict(argv=cmd,local_python=envcfg['python']))
    with (out/'pipeline.log').open('x',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=childenv,check=True,cwd=HERE)
    report=read_json(out/'run/REPORT.json')
    save('completed',state='completed',report=str(out/'run/REPORT.json'),stop_reason=report['stop_reason'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--env',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    try:run(a)
    except Exception as error:
        failure=dict(state='failed',stage='hybrid_controller',error=str(error),traceback=traceback.format_exc(),updated_unix=time.time())
        write_json(HERE/'HYBRID_STATE.json',failure)
        raise
