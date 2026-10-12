"""Fit independent full-token operators and falsify predictions on later checkpoints.

Full-history matrices are refitted AFTER temporal evaluation; their fit errors
must never be presented as forecasts. Batch dimension never shares parameters.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import argparse,csv,json,time,sys,shutil
from pathlib import Path
import numpy as np
import torch
from dane import HERE,SOURCE,OUT,metadata,load_all,array,write,sha
from operator_lokalny import Operators,scalar_alpha,select_ridge,metrics

torch.set_num_threads(4)
RUN_OWNED=False

class RefusedRun(RuntimeError):
    """A refused launch must not relabel another or completed run as failed."""

def active_run(path):
    if not path.exists():return False
    import psutil
    saved=json.loads(path.read_text())
    try:
        proc=psutil.Process(saved.get('pid',-1))
        return proc.pid!=os.getpid() and proc.is_running() and Path(proc.cwd()).resolve()==HERE.resolve() and any('analiza.py' in a for a in proc.cmdline())
    except psutil.AccessDenied:
        raise RefusedRun('Cannot inspect recorded PID; refusing a possible duplicate')
    except (psutil.NoSuchProcess,ValueError):return False
def status(stage,**kw):
    obj=dict(stage=stage,pid=os.getpid(),time=time.time(),**kw)
    write(OUT/'STATUS.json',obj);print(json.dumps(obj),flush=True)

def cpu(a):return a.detach().cpu().numpy()

def run(args):
    global RUN_OWNED
    OUT.mkdir(exist_ok=True)
    if (OUT/'REPORT.json').exists():raise RefusedRun('Completed report exists; refusing to overwrite')
    if active_run(OUT/'STATUS.json'):raise RefusedRun('Recorded analysis process is still active')
    cfg,steps,mask=metadata()
    protocol=dict(source_config_sha256=sha(SOURCE/'CONFIG.json'),mask_sha256=sha(SOURCE/'MASK_WEIGHTS.npy'),
        state_shape=[72,576,73,384],same_image_same_token_only=True,calibration_fit_pairs=40,
        calibration_origins=list(range(40,48)),evaluation_fit_pairs=48,evaluation_origins=list(range(48,72)),
        rollout_origins=[48,50,52],rollout_horizon=20,final_fit_pairs=72,relative_ridges=[.001,.1],
        gate=dict(rmse_vs_last=.95,rmse_vs_persistence=.8,mean_cosine=.5),
        dtype='float64',pca=False,device=args.device,batch=args.batch,
        method_order=['koopman','last_increment','persistence','scalar_velocity'],
        script_hashes={p.name:sha(p) for p in HERE.glob('*.py')},plan_sha256=sha(HERE/'PLAN.md'),
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        prior_dataset_exposure=True,all_history_forecast_unverified=True)
    if (OUT/'PROTOCOL.json').exists():
        if not args.resume:raise RefusedRun('Partial run exists; use --resume after checking recorded PID')
        if json.loads((OUT/'PROTOCOL.json').read_text())!=protocol:raise RefusedRun('Frozen protocol changed')
    else:write(OUT/'PROTOCOL.json',protocol)
    RUN_OWNED=True
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in list(HERE.glob('*.py'))+[HERE/'PLAN.md']:
        if not (snap/p.name).exists():shutil.copyfile(p,snap/p.name)
    data=load_all(steps,status)
    coeff=array('C_FINAL',(72,576,72,384));mx=array('MEAN_X_FINAL',(72,576,384));md=array('MEAN_DELTA_FINAL',(72,576,384))
    rel=array('RELATIVE_RIDGE',(72,576));ridge=array('ABSOLUTE_RIDGE_FINAL',(72,576));rank=array('GRAM_RANK_FINAL',(72,576),'int16')
    validation=array('CALIBRATION_MSE',(72,576,2,8));mse=array('MSE_ONE_STEP',(72,576,24,4))
    cos=array('COSINE_ONE_STEP',(72,576,24,4));ratio=array('INCREMENT_RATIO',(72,576,24,4))
    long=array('MSE_ROLLOUT20',(72,576,3,4));fiterr=array('FIT_MSE',(72,576,2));solver=array('SOLVER_RESIDUAL',(72,576,2))
    all_arrays=[coeff,mx,md,rel,ridge,rank,validation,mse,cos,ratio,long,fiterr,solver]
    device=torch.device(args.device);start=time.perf_counter()
    for image in range(72):
        done=OUT/f'IMAGE_{image:03d}_DONE.json'
        if args.resume and done.exists():continue
        for first in range(0,576,args.batch):
            last=min(first+args.batch,576);sl=slice(first,last)
            states=torch.as_tensor(np.array(data[image,sl],dtype=np.float64),device=device)
            cal=[]
            for rr in [.001,.1]:
                m=Operators().fit(states[:,:40],states[:,1:41],rr)
                pred=m.predict(states[:,40:48]);cal.append(((pred-states[:,41:49])**2).mean(-1))
            errors=torch.stack(cal,dim=1);selected=select_ridge(errors.mean(-1))
            validation[image,sl]=cpu(errors);rel[image,sl]=cpu(selected)
            m=Operators().fit(states[:,:48],states[:,1:49],selected)
            fiterr[image,sl,0]=cpu(((m.predict(states[:,:48])-states[:,1:49])**2).mean((1,2)))
            solver[image,sl,0]=cpu(m.solver_residual)
            alpha=scalar_alpha(states,48)
            z=states[:,48:72];v=z-states[:,47:71];truth=states[:,49:73]
            forecasts=[m.predict(z),z+v,z,z+alpha[:,None,None]*v]
            for method,pred in enumerate(forecasts):
                met=metrics(pred,truth,z)
                mse[image,sl,:,method]=cpu(met['mse']);cos[image,sl,:,method]=cpu(met['cosine']);ratio[image,sl,:,method]=cpu(met['increment_ratio'])
            for index,origin in enumerate([48,50,52]):
                zz=states[:,origin:origin+1];vv=zz-states[:,origin-1:origin];truthh=states[:,origin+20:origin+21]
                # Closed scalar recurrence, without teacher forcing.
                acc=torch.zeros_like(alpha);power=torch.ones_like(alpha)
                for _ in range(20):power=power*alpha;acc=acc+power
                preds=[m.predict(zz,20),zz+20*vv,zz,zz+acc[:,None,None]*vv]
                for method,pred in enumerate(preds):long[image,sl,index,method]=cpu(((pred-truthh)**2).mean((1,2)))
            if image==66 and first<=299<last:
                k=299-first;K,b=m.explicit()
                np.savez(OUT/'EXAMPLE_EVALUATION_MATRIX.npz',K=cpu(K[k]),b=cpu(b[k]),
                    image_index=image,token_index=299,max_fit_step=245,convention='z_next=z@K+b')
                np.savez(OUT/'EXAMPLE_FORECASTS.npz',prediction=cpu(forecasts[0][k]),truth=cpu(truth[k]),
                    origins=np.array(steps[48:72]),z=cpu(z[k]),v=cpu(v[k]))
            # Full-history refit is isolated from all metrics above.
            mf=Operators().fit(states[:,:72],states[:,1:73],selected)
            coeff[image,sl]=cpu(mf.coeff);mx[image,sl]=cpu(mf.mx);md[image,sl]=cpu(mf.md);ridge[image,sl]=cpu(mf.ridge)
            fiterr[image,sl,1]=cpu(((mf.predict(states[:,:72])-states[:,1:73])**2).mean((1,2)))
            solver[image,sl,1]=cpu(mf.solver_residual)
            eigen=torch.linalg.eigvalsh(mf.gram)
            tol=72*torch.finfo(torch.float64).eps*torch.clamp(eigen[:,-1],min=1.)
            rank[image,sl]=cpu((eigen>tol[:,None]).sum(1))
            for a in [coeff[image,sl],mse[image,sl],long[image,sl]]:
                if not np.isfinite(a).all():raise RuntimeError(f'Nonfinite result image={image} token={first}')
            del states,m,mf
        for a in all_arrays:a.flush()
        write(done,dict(image=image,tokens=576,elapsed_seconds=time.perf_counter()-start))
        status('fitting_independent_operators',completed_images=image+1,total_images=72,elapsed_seconds=time.perf_counter()-start)
    del data
    report=summarize(cfg,mask,mse,cos,long,fiterr,rank,solver,rel)
    status('hashing_output',operators=41472)
    write(OUT/'OUTPUT_SHA256.json',{p.name:sha(p) for p in OUT.glob('*.npy')})
    write(OUT/'REPORT.json',report)
    write_report(report)
    plot(mse,cos,mask,report)
    status('completed',operators=41472,candidate_count=report['candidate_count'])
    print(json.dumps(report,ensure_ascii=False),flush=True)

def summarize(cfg,mask,mse,cos,long,fiterr,rank,solver,rel):
    per=mse.mean(2);cosmean=cos[:,:,:,0].mean(2)
    gate=(per[:,:,0]<=.95**2*per[:,:,1])&(per[:,:,0]<=.8**2*per[:,:,2])&(cosmean>=.5)
    foreground=mask>=.5
    names=['koopman','last_increment','persistence','scalar_velocity']
    def totals(values,selection):
        return {name:float(np.sqrt(np.mean(values[selection][:,:,i]))) for i,name in enumerate(names)}
    selection=np.ones((72,576),dtype=bool)
    one=totals(mse,selection);lr=totals(long,selection)
    bymask=totals(mse,foreground) if foreground.any() else None
    with (OUT/'PER_VECTOR.csv').open('w',encoding='utf-8',newline='') as f:
        columns=['image_index','token_index','group_id','variant','mask_coverage','relative_ridge','fit48_rmse','fit72_rmse','test_rmse','last_rmse','persistence_rmse','alpha_rmse','test_cosine','candidate','beats_alpha','gram_rank_final']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for i in range(72):
            for t in range(576):
                writer.writerow(dict(image_index=i,token_index=t,group_id=cfg['rows'][i]['group_id'],variant=cfg['rows'][i]['variant'],
                    mask_coverage=float(mask[i,t]),relative_ridge=float(rel[i,t]),fit48_rmse=float(np.sqrt(fiterr[i,t,0])),fit72_rmse=float(np.sqrt(fiterr[i,t,1])),
                    test_rmse=float(np.sqrt(per[i,t,0])),last_rmse=float(np.sqrt(per[i,t,1])),persistence_rmse=float(np.sqrt(per[i,t,2])),alpha_rmse=float(np.sqrt(per[i,t,3])),
                    test_cosine=float(cosmean[i,t]),candidate=bool(gate[i,t]),beats_alpha=bool(per[i,t,0]<per[i,t,3]),gram_rank_final=int(rank[i,t])))
    return dict(state='completed',images=72,tokens_per_image=576,operators=41472,dimension=384,checkpoint_count=73,
        final_pairs_per_operator=72,final_pairs_total=41472*72,test_transitions_per_operator=24,
        rmse_one_step=one,rmse_rollout100_updates=lr,rmse_foreground_one_step=bymask,
        candidate_count=int(gate.sum()),candidate_foreground_count=int((gate&foreground).sum()),foreground_units=int(foreground.sum()),
        candidates_also_beating_alpha=int((gate&(per[:,:,0]<per[:,:,3])).sum()),
        rank_min_max=[int(rank.min()),int(rank.max())],max_solver_residual=float(solver.max()),
        fit48_rmse=float(np.sqrt(fiterr[:,:,0].mean())),fit72_rmse=float(np.sqrt(fiterr[:,:,1].mean())),
        example=dict(image_index=66,token_index=299,rmse={name:float(np.sqrt(per[66,299,k])) for k,name in enumerate(names)},cosine=float(cosmean[66,299]),candidate=bool(gate[66,299])),
        pca=False,shared_operator_between_images=False,shared_operator_between_tokens=False,
        all_history_forecast_unverified=True,prior_dataset_exposure=True,
        limitations=['fixed local spatial token is not pure semantic identity','72 pairs do not uniquely identify arbitrary 384x384 map',
            'all final matrices refitted after test; only prefix48 models have held-out errors',
            'many dependent exploratory comparisons, not independent success certificates',
            'one-step supplies observed current state; rollout never supplies future intermediate states'])

def write_report(r):
    lines=['# Wyniki: osobna macierz dla każdego wektora każdego obrazu','',
        'Policzono 41472 niezależne modele: 72 obrazy × 576 wektorów po 384 współrzędne. Bez PCA i bez łączenia obrazów w regresji.',
        'Macierze końcowe korzystają z 72 par każdej własnej historii. Ocena dotyczy osobnych macierzy dopasowanych tylko do pierwszych 48 par.',
        'Pojedynczy krok to 5 aktualizacji RF-DETR. Końcowa przyszłość macierzy dopasowanych do wszystkich checkpointów nie jest zweryfikowana.','',
        '| Metoda | RMSE jednego checkpointu | RMSE 100 aktualizacji |','|---|---:|---:|']
    for name in r['rmse_one_step']:lines.append(f"| {name} | {r['rmse_one_step'][name]:.8f} | {r['rmse_rollout100_updates'][name]:.8f} |")
    lines+=['',f"Kandydaci spełniający ustalone kryteria: {r['candidate_count']}/41472; na maskach >=0,5: {r['candidate_foreground_count']}/{r['foreground_units']}.",
        f"Błąd dopasowania pierwszych 48 par: {r['fit48_rmse']:.8f}; wszystkich 72 par: {r['fit72_rmse']:.8f}.",
        'Te liczby są eksploracyjne. Nie wybierają automatycznie poprawnej tożsamości narzędzia ani nie stanowią korekty na wielokrotne testowanie.',
        'Błędy każdej jednostki: wyniki/PER_VECTOR.csv. Błędy każdego odcinka: MSE_ONE_STEP.npy (obraz,token,24 odcinki,4 metody).',
        'Macierze: C_FINAL.npy, MEAN_X_FINAL.npy, MEAN_DELTA_FINAL.npy, ABSOLUTE_RIDGE_FINAL.npy i jawna formuła w PLAN.md. Xc odtwarzane ze zweryfikowanych źródeł.',
        'Eksport jawnej macierzy: eksport.py --image 66 --token 299. Modele testowe odtwarza się z pierwszych 48 par i RELATIVE_RIDGE.npy.']
    (HERE/'WYNIKI_TEST.md').write_text('\n'.join(lines),encoding='utf-8')

def plot(mse,cos,mask,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    labels=['Osobne macierze','Ostatni przyrost','Brak zmian','Model alpha']
    origins=np.arange(245,365,5)
    for k,label in enumerate(labels):axes[0].plot(origins,np.sqrt(mse[:,:,:,k].mean((0,1))),label=label)
    axes[0].set(xlabel='Początek prognozy',ylabel='RMSE wszystkich wektorów',title='Późniejsze checkpointy, poza dopasowaniem')
    axes[0].legend();axes[0].grid(alpha=.2)
    ratio=np.sqrt(mse[:,:,:,0].mean(2)/np.maximum(mse[:,:,:,1].mean(2),1e-30))
    axes[1].hist(np.log10(np.maximum(ratio.ravel(),1e-12)),bins=70,label='Wszystkie wektory')
    axes[1].axvline(0,color='red',label='Tak samo jak ostatni przyrost')
    axes[1].set(xlabel='log10(RMSE macierzy / RMSE ostatniego przyrostu)',ylabel='Liczba wektorów',title='Rozkład 41472 osobnych wyników')
    axes[1].legend();fig.savefig(OUT/'WYNIKI.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--batch',type=int,default=32);p.add_argument('--resume',action='store_true');args=p.parse_args()
    try:
        with torch.no_grad():run(args)
    except Exception as e:
        if RUN_OWNED and OUT.exists():status('failed',error=repr(e))
        raise
