"""Run diagnostic then controlled arms once; all media remain in memory."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from training_utils import ARMS
def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,path)
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser()
    for n in ('manifest','vit','checkpoint','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def request_stop(*_):
        (out/'STOP').touch()
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    if (out/'COMPARISON.json').exists() or (out/'STOP').exists():return 0
    here=Path(__file__).resolve().parent
    hashes=json.loads((here/'CODE_SHA256.json').read_text())
    for n,h in hashes.items():
        if digest(here/n)!=h:raise ValueError('Deployed source changed: '+n)
    common=['--manifest',a.manifest,'--vit',a.vit,'--checkpoint',a.checkpoint]
    launcher=[sys.executable,'-B',str(here/'deploy.py'),'launch-ddp','--standalone','--nproc_per_node=4']
    diag=out/'diagnostyka';diag.mkdir(exist_ok=True)
    selection=diag/'SAMPLER_SELECTION.json'
    def run(script,args):
        r=subprocess.run(launcher+[str(here/script)]+args,cwd=here)
        return r.returncode
    if not selection.exists():
        atomic(out/'STATUS.json',dict(state='sampler_diagnostic',time=time.time()))
        rc=run('diagnostyka.py',common+['--output',str(diag)]+(['--smoke'] if a.smoke else []))
        if rc:return rc
    if not selection.exists():raise RuntimeError('Diagnostic did not publish sampler selection')
    for arm in ARMS:
        if (out/'STOP').exists():return 0
        if (out/'STOP_TIME').exists():
            atomic(out/'RESUME_BOUNDARY.json',dict(state='between_arms',next_arm=arm,
                completed_arms=[x for x in ARMS if (out/x/'FINAL.json').exists()],
                code_sha256=hashes,time=time.time()))
            return 75
        if (out/arm/'FINAL.json').exists():continue
        args=common+['--output',str(out),'--arm',arm,'--sampler-selection',str(selection),
             '--steps','2' if a.smoke else '4000','--eval-every','2' if a.smoke else '1000',
             '--batch-size','1' if a.smoke else '4','--workers','2','--frame-stride','600','--film-buffer','8']
        if a.smoke:args+=['--smoke']
        rc=run('train.py',args)
        if rc:return rc
        if (out/'STOP').exists():return 0
        if not (out/arm/'FINAL.json').exists():raise RuntimeError('Arm lacks FINAL: '+arm)
    records=[]
    for arm in ARMS:
        b=json.loads((out/arm/'BEST.json').read_text())
        final=json.loads((out/arm/'FINAL.json').read_text())
        path=out/arm/'best_sd.pt'
        # Fresh CPU load proves checkpoint structure, not image quality.
        import torch
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['signature']['arm']==arm and checkpoint['new_steps']==b['new_steps']
        for group in ('net','vae'):
            assert all(torch.isfinite(t).all() for t in checkpoint[group].values() if t.is_floating_point())
        del checkpoint
        records.append(dict(arm=arm,new_steps=b['new_steps'],metrics=b['validation']['film_mean'],
            baseline=b['baseline_score'],final_reason=final['reason'],checkpoint=str(path),
            checkpoint_sha256=digest(path),fresh_checkpoint_load=True))
    winner=min(records,key=lambda x:x['metrics']['full_relative_mse'])
    summary=dict(state='smoke_completed' if a.smoke else 'completed',smoke=a.smoke,
        records=records,selected_by='minimum full_relative_mse, VAL only',winner=winner,
        sampler=json.loads(selection.read_text()),code_sha256=hashes,
        no_media_or_latent_disk_cache=True,visual_quality_not_verified_by_metrics_alone=True)
    atomic(out/'COMPARISON.json',summary)
    atomic(out/'STATUS.json',dict(state=summary['state'],winner=winner['arm']))
    if a.smoke:
        remote=here.parent
        atomic(remote/'SMOKE_OK.json',dict(state='passed',output=str(out),code_sha256=hashes,
              slurm_job=os.environ.get('SLURM_JOB_ID'),time=time.time()))
    return 0
if __name__=='__main__':sys.exit(main())
