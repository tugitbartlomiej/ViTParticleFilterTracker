"""Ograniczona pętla V1: prognoza -> generator -> realna klatka -> sparowany trening.

Każdy ciężki etap ma osobny proces, więc zwalnia VRAM po zakończeniu. Wyniki
testu nie są czytane przez regułę wyboru gałęzi. Raport negatywnej bramki to
wynik eksperymentu, nie wyjątek ani pretekst do zmiany progu po obejrzeniu danych.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import traceback
import importlib.metadata
import shutil
from pathlib import Path
import numpy as np
from dane import read_json,write_json,sha256,load_pool,make_dataset
from dynamika import fit_directions,project,fit_koopman,forecast,validate,compare_arms,to_jsonable,evaluate_directions

HERE=Path(__file__).resolve().parent
REMOTE_TRAINING=False

def command(script,log,**kwargs):
    if REMOTE_TRAINING and script in ('trening.py','ocena.py'):
        from hybryda_zdalne import dispatch
        return dispatch(script,log,kwargs)
    argv=[sys.executable,'-u','-B',str(HERE/script)]
    for key,value in kwargs.items():
        if value is None or value is False: continue
        argv.append('--'+key.replace('_','-'))
        if value is not True: argv.append(str(value))
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(argv,ensure_ascii=False)+'\n');stream.flush()
        subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True)

def history(dirs):
    found={}
    for directory in dirs:
        for path in sorted(Path(directory).glob('LATENT_*.npz')):
            with np.load(path) as data: step=int(data['step']); features=data['features'].copy()
            if step in found and not np.allclose(found[step],features,rtol=1e-5,atol=1e-5):
                raise ValueError('Latent at resume boundary changed before first update')
            found[step]=features
    steps=np.asarray(sorted(found))
    return steps,np.stack([found[s] for s in steps])

def map_score(path):
    report=read_json(path)
    if isinstance(report,list): report=report[0]
    for name in ('val/mAP_50_95','val/map','val/mAP'):
        if name in report:
            value=float(report[name])
            if not np.isfinite(value):raise ValueError('Nonfinite validation score')
            return value
    raise KeyError(f'No expected validation mAP in {path}: {list(report)}')

def verify_schedules(base,other,new_id,epoch):
    a=read_json(Path(base)/f'samples_epoch{epoch:03d}.json')
    b=read_json(Path(other)/f'samples_epoch{epoch:03d}.json')
    slots=b['replaced_slots']; aa=a['frame_ids'];bb=b['frame_ids']
    if len(aa)!=len(bb) or [i for i in range(len(aa)) if aa[i]!=bb[i]]!=slots:
        raise ValueError('Unpaired training streams')
    if any(bb[i]!=new_id for i in slots):raise ValueError('Wrong frame inserted')
    return dict(equal_presentations=True,replaced_slots=slots,new_frame=new_id)

def run(a):
    global REMOTE_TRAINING
    REMOTE_TRAINING=bool(getattr(a,'remote_training',False))
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    cfg=read_json(a.config);write_json(root/'config.json',cfg)
    argsvars={k:str(v) for k,v in vars(a).items()}
    write_json(root/'INPUTS.json',dict(arguments=argsvars,config_sha256=sha256(a.config),
        python=sys.version,packages={name:importlib.metadata.version(name) for name in ['torch','rfdetr','pytorch-lightning','numpy','faiss-cpu']},
        pool_sha256=sha256(a.pool),initial_selection_sha256=sha256(a.initial_selection),
        initial_weights_sha256=sha256(a.initial_weights),generator_sha256=sha256(a.generator),
        code_sha256={p.name:sha256(p) for p in HERE.glob('*.py')}))
    report=dict(state='running',started_unix=time.time(),rounds=[],test_used_for_selection=False,seed=cfg['seed'])
    def state(stage):
        report['stage']=stage;report['updated_unix']=time.time();write_json(root/'STATUS.json',report)
    rows=load_pool(a.pool);byid={r['frame_id']:r for r in rows}
    ids=[str(i) for i in read_json(a.initial_selection)['frame_ids']]
    if len(ids)!=cfg['initial_budget']:raise ValueError('Initial budget mismatch')
    from sonda import create_probe
    state('probe')
    imported=getattr(a,'imported_pretrain',None)
    if imported:
        imported=Path(imported)
        if read_json(imported/'config.json')!=cfg:raise ValueError('Imported pretrain recipe changed')
        done=read_json(imported/'pretrain'/'STATUS.json')
        expected=cfg['pretrain_epochs']*cfg['epoch_size']//(cfg['batch_size']*cfg['grad_accum_steps'])
        if done['state']!='completed' or done['global_step']!=expected:
            raise ValueError('Imported pretrain is incomplete')
        shutil.copytree(imported/'probe',root/'probe')
        probe=root/'probe'/'manifest.json'
        report['imported_pretrain']=str(imported)
        report['execution_split']={'training_and_evaluation':'eden','analysis_generation_retrieval':'local'}
    else:
        probe=create_probe(a.crop_manifest,a.crop_root,root/'probe',cfg['probe_groups'],cfg['seed'])
    records=read_json(probe)
    for record in records:
        original=byid.get(str(record['frame_id']))
        if original is None or original['split']!='train' or original['group_id']!=str(record['group_id']):
            raise ValueError('Probe provenance disagrees with main training pool')
    fit=np.asarray([r['role']=='fit' for r in records]);groups=[r['group_id'] for r in records]
    labels=np.asarray([r['labels'] for r in records])
    # Exclude probe source frames from acquisition; they remain training-domain diagnostics.
    excluded=set(ids)|{r['frame_id'] for r in records}
    state('visual_index')
    command('indeks_cli.py',root/'index.log',pool=a.pool,data_root=a.data_root,out=root/'index',
            hub=a.hub,dino_weights=a.dino_weights,device=a.device)
    dataset=make_dataset(rows,a.data_root,ids,root/'datasets'/'pretrain')
    steps_epoch=cfg['epoch_size']//(cfg['batch_size']*cfg['grad_accum_steps'])
    parent=root/'pretrain';state('pretrain')
    if imported:
        shutil.copytree(imported/'pretrain',parent)
    else:
        command('trening.py',root/'pretrain.log',config=root/'config.json',dataset=dataset,probe=probe,out=parent,
                initial_weights=a.initial_weights,stop_step=cfg['pretrain_epochs']*steps_epoch,device=a.device)
    lineage=[parent];evaluation=[(parent,dataset)];failures=0
    frozen_basis=None
    for r in range(cfg['rounds']):
        rd=root/f'round_{r+1:02d}';rd.mkdir()
        state(f'round_{r+1}:forecast')
        steps,features=history(lineage)
        horizon=cfg['snapshots_per_epoch'];fit_stop=len(steps)-horizon-1
        if fit_stop<3:raise ValueError('Too short history for forecast holdout')
        if frozen_basis is None:
            frozen_basis=fit_directions(features,labels,fit,min(2,fit_stop),rank=cfg['rank'])
        basis=frozen_basis; c=project(features,basis)
        quality=evaluate_directions(features,labels,basis,image_groups=groups,time_indices=range(fit_stop,len(steps)),
                                    label_names=records[0]['label_names'])
        write_json(rd/'DIRECTION_QUALITY.json',to_jsonable(quality))
        val=validate(c,steps,fit,fit_stop,image_groups=groups,horizons=(horizon,),seed=cfg['seed'],
                     n_permutations=cfg['permutations'])
        write_json(rd/'FORECAST.json',to_jsonable(val))
        rr=dict(round=r+1,parent=str(parent),forecast_accepted=val['accepted'],forecast_reason=val['reason'])
        report['rounds'].append(rr)
        if not val['accepted']:
            report['stop_reason']='forecast_not_better_than_controls';break
        # Refit only the operator on currently available past; fixed coordinate basis.
        model=fit_koopman(c,steps,fit,len(steps))
        target=forecast(model,c[-1],horizon);delta=target-c[-1]
        np.savez_compressed(rd/'directions.npz',mean=basis['mean'],scale=basis['scale'],
                            projection=basis['projection'],target_scale=model['scale'])
        write_json(rd/'OPERATOR.json',to_jsonable(model))
        eligible=[i for i,row in enumerate(records) if row['role']=='fit' and row['semantic_presence']==1]
        eligible.sort(key=lambda i:-float(np.linalg.norm(delta[i]/model['scale'])))
        selected=None
        state(f'round_{r+1}:generation')
        for q,i in enumerate(eligible[:cfg['max_queries_per_round']]):
            np.save(rd/f'target_{q}.npy',target[i])
            np.save(rd/f'current_{q}.npy',c[-1,i])
            command('generator_cli.py',rd/f'generation_{q}.log',detector=parent/'FINAL.pth',generator=a.generator,
                vendor=a.vendor,probe=probe,anchor=i,directions=rd/'directions.npz',target=rd/f'target_{q}.npy',
                out=rd/f'generation_{q}',device=a.device,steps=cfg['generation_steps'],restarts=cfg['generation_restarts'],
                expected_current=rd/f'current_{q}.npy')
            gen=read_json(rd/f'generation_{q}'/'report.json')
            arm=gen['arms']['target']
            if not arm['numerical_gates_pass']:continue
            exclude=rd/f'excluded_{q}.json';write_json(exclude,sorted(excluded))
            command('indeks_cli.py',rd/f'retrieval_{q}.log',index=root/'index',image=rd/f'generation_{q}'/arm['best_image'],
                    exclude=exclude,out=rd/f'retrieval_{q}.json',device=a.device,topk=cfg['retrieval_topk'])
            ranking=read_json(rd/f'retrieval_{q}.json');candidate=ranking['top_selection']
            if candidate is not None and candidate['score']>=cfg['min_visual_cosine']:
                selected=candidate;rr['query_index']=i;rr['selected']=candidate;break
        if selected is None:
            report['stop_reason']='generation_or_visual_similarity_gate_failed';break
        fid=selected['frame_id']
        rng=np.random.default_rng(cfg['seed']+r)
        candidates=sorted(k for k,row in byid.items() if row['split']=='train' and row['annotations'] and k not in excluded and k!=fid)
        if not candidates:
            report['stop_reason']='unused_pool_exhausted_for_random_control';break
        random_id=candidates[int(rng.integers(len(candidates)))]
        datasets={name:make_dataset(rows,a.data_root,ids,rd/'datasets'/name,added=extra)
                  for name,extra in [('control',None),('prim',fid),('random',random_id)]}
        outputs={name:rd/name for name in datasets};stop=int(steps[-1])+steps_epoch
        rr['parent_sha256']=sha256(parent/'FINAL.ckpt');rr['random_id']=random_id
        state(f'round_{r+1}:paired_training')
        for name in outputs:
            command('trening.py',rd/f'{name}.log',config=root/'config.json',dataset=datasets[name],probe=probe,out=outputs[name],
                initial_weights=a.initial_weights,resume=parent/'FINAL.ckpt',added=name!='control',stop_step=stop,device=a.device)
            evaluation.append((outputs[name],datasets[name]))
        epoch=int(steps[-1])//steps_epoch
        rr['pairing']={name:verify_schedules(outputs['control'],outputs[name],fid if name=='prim' else random_id,epoch)
                       for name in ['prim','random']}
        endpoint={name:project(history([directory])[1][-1],basis) for name,directory in outputs.items()}
        for name in ('prim','random'):
            comparison=compare_arms(c[-1],endpoint['control'],endpoint[name],delta,scale=model['scale'],image_groups=groups)
            write_json(rd/f'LATENT_EFFECT_{name}.json',to_jsonable(comparison))
        scores={name:map_score(path/'VALIDATION.json') for name,path in outputs.items()}
        rr['validation_map']=scores
        # Both real-frame policies must be compared; only requested Koopman policy can enter lineage.
        accept=scores['prim']>=max(scores['control'],scores['random'])+cfg['acceptance_delta_map']
        chosen='prim' if accept else 'control';rr['chosen_next']=chosen
        if accept:ids.append(fid);excluded.add(fid);failures=0
        else:failures+=1
        parent=outputs[chosen];dataset=datasets[chosen];lineage.append(parent)
        write_json(rd/'REPORT.json',rr)
        if failures>=cfg['early_stop_failed_rounds']:
            report['stop_reason']='no_validation_gain_in_consecutive_rounds';break
    else:report['stop_reason']='round_limit'
    # Only now, after all choices are frozen, evaluate the reserved test split.
    state('final_test')
    for path,ds in evaluation:
        command('ocena.py',path/'test.log',weights=path/'FINAL.pth',dataset=ds,out=path/'test',split='test',device=a.device)
    report.update(state='completed',stage='completed',final_checkpoint=str(parent/'FINAL.pth'),
                  selected_frame_ids=ids,finished_unix=time.time(),test_evaluations=[str(p/'test'/'REPORT.json') for p,_ in evaluation],
                  limitations=['Single training seed.','Generated mask is not ground truth.','Numeric gates do not certify semantic identity.',
                               'Test partition may have appeared in older project experiments; not claimed newly untouched.'])
    write_json(root/'REPORT.json',report);write_json(root/'STATUS.json',report)
    from raport_wizualny import make_report
    make_report(root)
    return report

def parser():
    ap=argparse.ArgumentParser()
    for key in ('config','pool','data-root','initial-selection','initial-weights','crop-manifest','crop-root','generator','vendor','hub','dino-weights','out'):
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--imported-pretrain')
    ap.add_argument('--remote-training',action='store_true')
    return ap

if __name__=='__main__':
    args=parser().parse_args()
    try:run(args)
    except Exception as exc:
        write_json(Path(args.out)/'FAILURE.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc(),unix=time.time()))
        raise
