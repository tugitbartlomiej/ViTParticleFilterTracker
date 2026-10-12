"""Mały rzeczywisty test GPU: pełny resume, parowanie, gradient i indeks.

Nie ocenia hipotezy naukowej. Cele generatora pochodzą z kontrolnej perturbacji,
nie są nazywane trafną prognozą Koopmana. Całość działa na wydzielonym GPU SLURM.
"""
from pathlib import Path
import numpy as np
from dane import read_json,write_json,load_pool,make_dataset
from petla import command,history,verify_schedules,parser

def run(a):
    out=Path(a.out)
    reused=bool(a.training_from)
    if reused:a.checks_only=True
    source=Path(a.training_from) if reused else out
    if not a.checks_only or reused:out.mkdir(parents=True,exist_ok=False)
    rows=load_pool(a.pool)
    train=[r for r in rows if r['split']=='train'][:20]
    rows=train+[r for r in rows if r['split']=='val'][:2]+[r for r in rows if r['split']=='test'][:2]
    if not a.checks_only or reused:(out/'pool.jsonl').write_text('\n'.join(__import__('json').dumps(r) for r in rows)+'\n')
    cfg=read_json(a.config);cfg.update(initial_budget=4,pretrain_epochs=1,rounds=1,snapshots_per_epoch=2,
                                    epoch_size=16,batch_size=1,grad_accum_steps=4,probe_groups=3)
    if not a.checks_only:write_json(out/'config.json',cfg)
    from sonda import create_probe
    probe=source/'probe'/'manifest.json' if a.checks_only else create_probe(a.crop_manifest,a.crop_root,out/'probe',3,42)
    ids=[r['frame_id'] for r in train[:4]];extra=train[4]['frame_id']
    base=make_dataset(rows,a.data_root,ids,out/'check_dataset' if a.checks_only else out/'datasets'/'base')
    prim=None if a.checks_only else make_dataset(rows,a.data_root,ids,out/'datasets'/'prim',extra)
    jobs=[] if a.checks_only else [('start',base,None,4,False),
            ('continuous',base,None,8,False),
            ('control',base,out/'start'/'FINAL.ckpt',8,False),
            ('repeat',base,out/'start'/'FINAL.ckpt',8,False),
            ('prim',prim,out/'start'/'FINAL.ckpt',8,True)]
    for name,ds,resume,stop,added in jobs:
        command('trening.py',out/f'{name}.log',config=out/'config.json',dataset=ds,probe=probe,out=out/name,
                initial_weights=a.initial_weights,resume=resume,stop_step=stop,added=added,device=a.device)
    from verify_states import verify
    checked=verify(source)
    error=checked['repeat']['weight_max_abs'];resume_error=checked['continuous']['weight_max_abs']
    pair=verify_schedules(source/'control',source/'prim',extra,1)
    # Boundary values must agree before updating the restored detector.
    steps,features=history([source/'start',source/'control'])
    if steps.tolist()!=[0,2,4,6,8]:raise AssertionError(f'Unexpected steps after resume: {steps}')
    from dynamika import fit_directions,project
    records=read_json(probe);fit=np.array([r['role']=='fit' for r in records]);labels=np.array([r['labels'] for r in records])
    basis=fit_directions(features,labels,fit,2,rank=3);code=project(features[-1],basis)
    np.savez_compressed(out/'directions.npz',mean=basis['mean'],scale=basis['scale'],projection=basis['projection'],target_scale=np.ones(code.shape[1]))
    target=code[0].copy();target[0]+=.01;np.save(out/'target.npy',target)
    np.save(out/'current.npy',code[0])
    command('generator_cli.py',out/'generator.log',detector=source/'control'/'FINAL.pth',generator=a.generator,vendor=a.vendor,
            probe=probe,anchor=0,directions=out/'directions.npz',target=out/'target.npy',out=out/'generation',device=a.device,steps=2,restarts=1,
            expected_current=out/'current.npy')
    generation=read_json(out/'generation'/'report.json')
    gradient_ok=all(attempt['feature_gradient_max_norm']>0 and attempt['feature_gradient_finite']
                    for arm in generation['arms'].values() for attempt in arm['attempts'])
    if not gradient_ok:raise AssertionError('No finite nonzero gradient through detector and generator')
    command('indeks_cli.py',out/'index.log',pool=out/'pool.jsonl',data_root=a.data_root,out=out/'index',hub=a.hub,dino_weights=a.dino_weights,device=a.device)
    write_json(out/'exclude.json',ids)
    command('indeks_cli.py',out/'query.log',index=out/'index',image=out/'generation'/'target'/'best.png',exclude=out/'exclude.json',out=out/'ranking.json',device=a.device,topk=3)
    ranking=read_json(out/'ranking.json')
    if not ranking['candidates']:raise AssertionError('Empty retrieval smoke')
    command('ocena.py',out/'test.log',weights=source/'control'/'FINAL.pth',dataset=base,out=out/'test',split='test',device=a.device)
    write_json(out/'REPORT.json',dict(state='completed',purpose='engineering_smoke_not_scientific_result',
        repeat_checkpoint_max_abs=error,resume_max_abs=resume_error,pairing=pair,steps=steps.tolist(),gradient_ok=gradient_ok,
        retrieval_candidates=len(ranking['candidates']),training_source=str(source)))

if __name__=='__main__':
    ap=parser();ap.add_argument('--checks-only',action='store_true');ap.add_argument('--training-from');run(ap.parse_args())
