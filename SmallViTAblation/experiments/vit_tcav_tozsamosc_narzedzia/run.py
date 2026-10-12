"""Czy kierunek LDA/CAV wpływa na detekcję? Brak efektu bbox/AP obala poprawę detekcji."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys, time, argparse, traceback, platform
sys.dont_write_bytecode=True
import numpy as np, torch, cv2
from torch.nn import functional as F
from scipy.special import expit
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from dane import *
from model_snapshot import MalyViT,dekoduj
OLD=ROOT/'experiments/maly_vit_tozsamosc_czasteczki_pary_400'
REPS=50
FRACTIONS=np.array([.0001,.001,.005,.01])
START=time.time()
def progress(stage,**kw):
    js(OUT/'PROGRESS.json',dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kw))
    print(stage,kw,flush=True)
def register_research_family():
    decl=dict(rodzina=HERE.name,klasa='badawcza',pytanie='Czy oryginalny LDA i CAV768/J128 zmieniają detekcję?',role=['BIEG','CAV','LATENT','TCAV','FD'],wymagane=['tag','ziarno'],kontrole=['_permuted','_random'],uwagi='Rejestr rodzinny; wspólny rejestr poza autoryzowanymi dwoma katalogami.')
    if (HERE/'WYNIKI.json').exists():assert read(HERE/'WYNIKI.json')['rodzina']==HERE.name
    else:js(HERE/'WYNIKI.json',decl)
    js(OUT/'REGISTRATION.json',dict(**decl,registered_by='register_research_family',shared_registry_write=False,reason='Explicit user write boundary',plan_sha256=sha(HERE/'PLAN.md')))
def init_net():
    torch.set_num_threads(4);torch.manual_seed(SEED);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu'
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        if free>800*1024**2:
            torch.cuda.set_per_process_memory_fraction(.18);device='cuda'
    net=MalyViT().to(device).eval();ck=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert ck['krok']==1000;net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False)
    return net,device
def forward(net,device,pair):
    im=cv2.imread(str(PAIRS/pair['original']));assert im is not None
    im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(im).permute(2,0,1)[None].float().to(device)/255
    cache={};b=net.bloki[-1]
    hz=b.mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o.detach().clone()))
    hq=b.register_forward_hook(lambda m,a,o:cache.update(q=o.detach().clone()))
    with torch.no_grad():o,_=net(x)
    hz.remove();hq.remove()
    return x,o,cache['z'],cache['q']
def tail(net,q,delta):
    b=net.bloki[-1]
    change=b.ls2(F.linear(delta,b.mlp.fc2.weight,None))
    return net.glowa(net.norm(q+change)).reshape(1,20,30,5)
def gt(row):
    x,y,w,h=row['bbox_source_480'];return np.array([x,y,x+w,y+h],float)
def center(row):
    g=gt(row);cx=int(np.clip(round((g[0]+g[2])/32-.5),0,29));cy=int(np.clip(round((g[1]+g[3])/32-.5),0,19));return cy*30+cx
def overlap(boxes,g):
    lo=np.maximum(boxes[:,:2],g[:2]);hi=np.minimum(boxes[:,2:],g[2:]);a=np.maximum(hi-lo,0).prod(1)
    union=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)+(g[2:]-g[:2]).prod()-a
    return a/np.maximum(union,1e-12)
def detection(o,row):
    b,s=dekoduj(o.detach().cpu());b=b[0].numpy();s=s[0].numpy();iou=overlap(b,gt(row))
    return dict(boxes=b,scores=s,ious=iou,top1_iou=float(iou[0]),confidence=float(s[0]),fixed_logit=float(o.reshape(600,5)[center(row),0]),q=float(s[0]*iou[0]))
def film_groups(rows,ids):
    names=sorted({rows[i]['video_id'] for i in ids});return names,[np.array([j for j,i in enumerate(ids) if rows[i]['video_id']==f]) for f in names]
def film_mean(values,groups):return np.array([np.asarray(values)[g].mean(axis=0) for g in groups])
def ci(values,groups):
    fm=film_mean(values,groups);rng=np.random.default_rng(SEED+99);draw=rng.integers(0,len(groups),(2000,len(groups)))
    boots=fm[draw].mean(axis=1)
    return dict(mean=fm.mean(axis=0),CI95=np.quantile(boots,[.025,.975],axis=0),unit='whole film',resamples=2000)
def probe(pool,ids,v,b):
    score=np.r_[pool[ids,0]@v+b,pool[ids,1]@v+b];y=np.r_[np.ones(len(ids)),np.zeros(len(ids))]
    return dict(BA=float(balanced_accuracy_score(y,score>=0)),AP=float(average_precision_score(y,score)),AUROC=float(roc_auc_score(y,score)),mean_probability_original=float(expit(score[:len(ids)]).mean()),mean_probability_removed=float(expit(score[len(ids):]).mean()),scores=score)
def fit_separator(pool,ids,J,C,rng=None,rows=None,permute=False):
    flips={}
    if permute:flips={f:int(rng.integers(2)) for f in set(rows[i]['video_id'] for i in ids)}
    labels=np.array([1-flips.get(rows[i]['video_id'],0) if rows else 1 for i in ids])
    X=np.concatenate([pool[ids,0][:,J],pool[ids,1][:,J]]).astype(float);y=np.r_[labels,1-labels]
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1
    clf=LogisticRegression(C=C,solver='lbfgs',max_iter=2000,tol=1e-7).fit((X-mu)/sd,y)
    assert clf.n_iter_[0]<2000
    raw=clf.coef_[0]/sd; b=float(clf.intercept_[0]-raw@mu);v=np.zeros(768);v[J]=raw
    norm=np.linalg.norm(v);assert norm>1e-12
    return v/norm,v,b,mu,sd,flips
def learn(pool,rows,J,ids,lda):
    bank={};reports={};tr,va,te=[ids[k] for k in ['train','val','test']]
    filmnames,groups=film_groups(rows,tr);original_film_ids=[tr[g] for g in groups]
    for label,axes in [('full768',np.arange(768)),('J128',J)]:
        grid=[]
        for C in [.01,.1,1.]:
            fit=fit_separator(pool,tr,axes,C);m=probe(pool,va,fit[1],fit[2]);grid.append((m['BA'],C,fit))
        _,C,fit=max(grid,key=lambda a:(a[0],-a[1]));v,raw,b,mu,sd,_=fit
        bank[label]=v
        # Freeze fit and selection before accessing held-out predictions.
        js(OUT/f'FREEZE_{label}.json',dict(C=C,axes0=axes,selection='val BA; ties smaller C',validation_grid=[dict(C=c,BA=ba) for ba,c,_ in grid],raw_coef=raw,intercept=b,unit_CAV=v,train_mean=mu,train_sd=sd))
        reps=[];null=[];randoms=[];draws=[];flip_records=[]
        for rep in range(REPS):
            rng=np.random.default_rng(SEED+rep)
            selected=rng.integers(0,len(groups),len(groups));sample=np.concatenate([original_film_ids[k] for k in selected]);draws.append(sample)
            vf=fit_separator(pool,sample,axes,C,rows=rows);reps.append(vf[0])
            nf=fit_separator(pool,tr,axes,C,rng=rng,rows=rows,permute=True);null.append(nf[0]);flip_records.append(nf[-1])
            rv=np.zeros(768);rv[axes]=rng.normal(size=len(axes));rv/=np.linalg.norm(rv);randoms.append(rv)
            progress('CPU_separators',variant=label,repetition=rep+1,total=REPS)
        bank[label+'_reps']=np.array(reps);bank[label+'_permuted']=np.array(null);bank[label+'_random']=np.array(randoms)
        test=probe(pool,te,raw,b);scores=test['scores'];correct=np.mean(np.stack([scores[:len(te)]>=0,scores[len(te):]<0]),axis=0)
        reports[label]=dict(C=C,validation=probe(pool,va,raw,b),test=test,BA_film=ci(correct,film_groups(rows,te)[1]),orientation='positive logit predicts original')
        js(OUT/f'REPETITIONS_{label}.json',dict(train_bootstrap_indices=draws,permuted_film_flips=flip_records,seed=SEED,reps=REPS))
        np.savez_compressed(OUT/f'CAV_{label}.npz',unit=v,raw_coef=raw,intercept=b,mean=mu,sd=sd,axes0=axes,repetitions=reps,permuted=null,random=randoms)
    v=lda['lda_direction'].astype(float);assert abs(np.linalg.norm(v)-1)<1e-6
    assert (pool[tr,0]-pool[tr,1]).mean(0)@v>0
    bank['historical_LDA']=v
    # LDA was fit directly in raw 768 coordinates; no sd division or U rotation.
    b=-float(lda['lda_center']@v)-float(lda['lda_threshold'])
    m=probe(pool,te,v,b);m['mean_probability_original']=None;m['mean_probability_removed']=None
    m['score_semantics']='raw LDA projection minus historical threshold; not calibrated probability'
    reports['historical_LDA']=dict(test=m,historical=read(OLD/'IDENTYFIKACJA.json')['lda'],conversion='already unit raw-activation normal; center/threshold affect classifier only; no standardization or SVD U',source_sha256=sha(OLD/'PODPRZESTRZEN.npz'))
    np.savez_compressed(OUT/'CAV_historical_LDA.npz',unit=v,center=lda['lda_center'],threshold=lda['lda_threshold'])
    js(OUT/'PROBE_METRICS.json',reports)
    return bank,reports
def ap(dets,threshold):
    # One annotated GT per image, top10 local maxima, greedy matching. Standard precision envelope integral.
    ranked=sorted([(float(s),i,j) for i,d in enumerate(dets) for j,s in enumerate(d['scores'])],reverse=True)
    used=set();tp=[]
    for _,i,j in ranked:
        good=i not in used and dets[i]['ious'][j]>=threshold
        tp.append(int(good))
        if good:used.add(i)
    tp=np.cumsum(tp);fp=np.arange(1,len(tp)+1)-tp;rec=tp/len(dets);prec=tp/np.maximum(tp+fp,1)
    r=np.r_[0,rec,1];p=np.r_[0,prec,0];p=np.maximum.accumulate(p[::-1])[::-1];ix=np.flatnonzero(r[1:]!=r[:-1]);return float(((r[ix+1]-r[ix])*p[ix+1]).sum())
def detmetrics(dets):
    out=dict(AP50=ap(dets,.5),AP75=ap(dets,.75),AP50_95=float(np.mean([ap(dets,t) for t in np.arange(.5,.96,.05)])),mean_top1_IoU=float(np.mean([d['top1_iou'] for d in dets])),mean_confidence=float(np.mean([d['confidence'] for d in dets])))
    for t in [.5,.75]:
        hits=np.array([bool(np.any((d['scores']>=.5)&(d['ious']>=t))) for d in dets]);n=sum((d['scores']>=.5).sum() for d in dets);rec=float(hits.mean());precision=float(hits.sum()/max(n,1))
        out[f'recall_IoU{t}']=rec;out[f'precision_IoU{t}']=precision;out[f'F1_IoU{t}']=2*rec*precision/max(rec+precision,1e-12)
    return out
def run_detector(net,device,rows,pairs,w,ids,cache,bank,eps,smoke=False):
    selected=ids['test'][:2] if smoke else ids['test'];grad=[];base=[];maps=[];records=[];weights=[];verification=[];fdmaps=[];qstates=[];zstates=[]
    main={k:v for k,v in bank.items() if k in ['full768','J128','historical_LDA']}
    if not smoke:
        for label in ['full768','J128']:
            main[label+'_permuted']=bank[label+'_permuted'][0];main[label+'_random']=bank[label+'_random'][0]
    for count,i in enumerate(selected):
        x,o,z,q=forward(net,device,pairs[i]);a,b=cache['offsets'][i:i+2];tids=cache['token_ids'][a:b].astype(int)
        np.testing.assert_allclose(z[0,tids].cpu().numpy(),cache['states'][a:b,0],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(o.cpu().numpy()[0],cache['outputs'][i,0],rtol=2e-5,atol=3e-5)
        ww=w[i].astype(np.float32);lift=ww/np.linalg.norm(ww);weights.append(ww)
        D=torch.zeros_like(z,requires_grad=True);oz=tail(net,q,D)
        torch.testing.assert_close(oz,o,rtol=2e-5,atol=3e-5)
        target=center(rows[i]);derivative=torch.autograd.grad(oz.reshape(600,5)[target,0],D)[0]
        projected=(derivative[0]*torch.tensor(lift,device=device)[:,None]).sum(0).cpu().numpy();grad.append(projected)
        baseline=detection(o,rows[i]);base.append(baseline);maps.append(o.cpu().numpy()[0]);qstates.append(q.cpu().numpy()[0]);zstates.append(z.cpu().numpy()[0])
        norm=float(np.linalg.norm(cache['states'][a:b,0]));maxeps=min(float(eps[-1]),.01*norm)
        for name,v in main.items():
            direction=torch.tensor(lift,device=device)[None,:,None]*torch.tensor(v,dtype=torch.float32,device=device)[None,None,:]
            assert abs(float(direction.norm())-1)<1e-5
            for j,epsilon in enumerate(eps):
                e=min(float(epsilon),maxeps);changed=[]
                for sign in [-1,1]:
                    with torch.no_grad():oo=tail(net,q,sign*e*direction)
                    d=detection(oo,rows[i]);changed.append(d);fdmaps.append(oo.cpu().numpy()[0])
                    records.append(dict(source_index=int(i),film=rows[i]['video_id'],method=name,epsilon_index=j,epsilon=e,fraction=float(FRACTIONS[j]),sign=sign,activation_norm=norm,relative_norm=e/norm,analytic_sensitivity=float(projected@v),baseline=baseline,changed=d,score_response=d['fixed_logit']-baseline['fixed_logit']))
                    if count==0 and name in ['full768','historical_LDA'] and j==0:
                        hook=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,out:out+sign*e*direction)
                        with torch.no_grad():full,_=net(x)
                        hook.remove();err=float((full-oo).abs().max());assert err<5e-5
                        verification.append(dict(method=name,sign=sign,tail_vs_actual_GELU_hook_max_error=err))
                fd=(changed[1]['fixed_logit']-changed[0]['fixed_logit'])/(2*e)
                for r in records[-2:]:r['central_finite_difference']=fd;r['FD_absolute_error']=abs(fd-float(projected@v))
        # Check local linearity at the two smallest epsilons with an absolute cancellation tolerance.
        small=[r for r in records if r['source_index']==int(i) and r['epsilon_index']<2]
        assert all(r['FD_absolute_error']<.002+.05*abs(r['analytic_sensitivity']) for r in small)
        progress('detector_processing',processed=count+1,total=len(selected),source_index=int(i),device=device)
    if not smoke:
        np.savez_compressed(OUT/'FD_MAPS.npz',maps=np.array(fdmaps),source_indices=np.array([r['source_index'] for r in records]),method=np.array([r['method'] for r in records]),epsilon=np.array([r['epsilon'] for r in records]),epsilon_index=np.array([r['epsilon_index'] for r in records]),sign=np.array([r['sign'] for r in records]))
        np.savez_compressed(OUT/'HELDOUT_STATES.npz',test_indices=selected,original_GELU=np.array(zstates),last_block_output_q=np.array(qstates))
    return selected,np.array(grad),base,np.array(maps),records,np.array(weights),verification
def analyze(rows,test,grad,base,records,bank,probes):
    names,groups=film_groups(rows,test);stats={};pvalues={};rng=np.random.default_rng(SEED+700)
    signs=rng.choice([-1,1],(10000,len(groups)))
    for label in ['full768','J128','historical_LDA']:
        v=bank[label];s=grad@v;positive=s>0
        fm=film_mean(positive.astype(float),groups);tt=float(fm.mean());centered=fm-.5
        pf=float((1+np.count_nonzero(np.abs((signs*centered).mean(1))>=abs(centered.mean())))/(len(signs)+1))
        st=dict(frame_positive_fraction=float(positive.mean()),film_positive_fraction=tt,zero_sensitivity_images=int(np.count_nonzero(s==0)),positive_fraction_CI=ci(positive.astype(float),groups),signed_magnitude_CI=ci(s,groups),absolute_magnitude_CI=ci(abs(s),groups),film_signflip_p=pf,per_image_sensitivity=s)
        if label!='historical_LDA':
            sr=bank[label+'_reps']@grad.T;nr=bank[label+'_permuted']@grad.T;rr=bank[label+'_random']@grad.T
            rates=np.array([film_mean((z>0).astype(float),groups).mean() for z in sr]);null=np.array([film_mean((z>0).astype(float),groups).mean() for z in nr]);rand=np.array([film_mean((z>0).astype(float),groups).mean() for z in rr])
            p=float(ttest_1samp(rates,.5).pvalue) if np.std(rates)>0 else (0. if rates[0]!=.5 else 1.)
            empirical=lambda x:float((1+(np.abs(x-.5)>=abs(tt-.5)).sum())/(len(x)+1))
            st.update(repetition_TCAV=rates,permuted_TCAV=null,random_TCAV=rand,paper_ttest_vs_half_p=p,paper_ttest_Bonferroni2=min(1.,2*p),paper_ttest_caution='shared heldout films; CAV repetitions are not independent detector datasets',empirical_permuted_p=empirical(null),empirical_random_p=empirical(rand),empirical_Bonferroni4=dict(permuted=min(1,4*empirical(null)),random=min(1,4*empirical(rand))))
            np.savez_compressed(OUT/f'TCAV_{label}.npz',test_indices=test,sensitivity=s,repeat_sensitivities=sr,permuted_sensitivities=nr,random_sensitivities=rr,repeat_positive_fraction=rates)
        st['film_signflip_Bonferroni3']=min(1.,3*pf);stats[label]=st
    detstats={};basem=detmetrics(base)
    for name in sorted({r['method'] for r in records}):
        for j in range(4):
            for sign in [-1,1]:
                rs=[r for r in records if r['method']==name and r['epsilon_index']==j and r['sign']==sign]
                assert len(rs)==len(test);changed=[r['changed'] for r in rs];metrics=detmetrics(changed)
                diffs=np.array([r['changed']['top1_iou']-r['baseline']['top1_iou'] for r in rs]);resp=np.array([r['score_response'] for r in rs])
                hits=np.array([float(np.any((d['scores']>=.5)&(d['ious']>=.5))) for d in changed]);bhits=np.array([float(np.any((d['scores']>=.5)&(d['ious']>=.5))) for d in base])
                film_delta=film_mean(diffs,groups);p=float((1+(np.abs((signs*film_delta).mean(1))>=abs(film_delta.mean())).sum())/(len(signs)+1))
                key=f'{name}_eps{j}_{sign:+d}';pvalues[key]=p
                # Cluster bootstrap AP; same sampled whole films for before/after, 2000 resamples.
                draws=np.random.default_rng(SEED+99).integers(0,len(groups),(2000,len(groups)));apdelta=[]
                for draw in draws:
                    ix=np.concatenate([groups[k] for k in draw]);apdelta.append(ap([changed[k] for k in ix],.5)-ap([base[k] for k in ix],.5))
                detstats[key]=dict(metrics=metrics,delta_metrics={k:metrics[k]-basem[k] for k in metrics},film_IoU_delta=ci(diffs,groups),film_logit_delta=ci(resp,groups),film_recall_delta=ci(hits-bhits,groups),AP50_delta_film_bootstrap_CI95=np.quantile(apdelta,[.025,.975]),IoU_benefit_images=int((diffs>1e-6).sum()),IoU_degradation_images=int((diffs< -1e-6).sum()),FD_mean_absolute_error=float(np.mean([r['FD_absolute_error'] for r in rs])),IoU_film_signflip_p=p)
        progress('analysis_detection',method=name)
    for k,p in pvalues.items():detstats[k]['IoU_Bonferroni_all_arms']=min(1,len(pvalues)*p)
    result=dict(status='complete',experiment='ADAPTED ROI/token TCAV',seed=SEED,repetitions=REPS,test_pairs=len(test),test_films=len(groups),baseline_detection=basem,TCAV=stats,detection=detstats,probe=probes,limitations=['Known original ROI and single annotated tool bbox','Historical J128 and LDA selection reuse these films; not independent validation','LaMa artefact confounding','Local hidden perturbations need not correspond to images','Paper ttest across shared-film repetitions is descriptive; primary grouping by film','Empirical null resolution 1/51; Bonferroni4 prevents tiny p with only 50 nulls'])
    js(OUT/'RESULT.json',result)
    return result
def report(result):
    lines=['# Wynik adaptowanego TCAV ROI/token','', 'Eksperyment rozszerza wcześniejszą LDA o wpływ na detektor. SVD 8D nie jest kierunkiem LDA. Kierunki, sieć i podział są zamrożone; nie optymalizowano obrazów ani latentu.','', '## Aparat', 'Kim et al. 2018 §§3-4: separator, pochodna logitu, frakcja dodatnia i test względem 0.5. Tutaj uczymy wspólny 768D CAV ze średniej ROI; lift ma normę Frobeniusa 1. W historycznej LDA normalna jest już w raw 768D, więc nie dzielimy jej przez SD sondy SVD.','', '## Wynik','| Kierunek | TCAV po filmach | Średnia pochodna | Zero |','|---|---:|---:|---:|']
    for k,v in result['TCAV'].items():lines.append(f"|{k}|{v['film_positive_fraction']:.5f}|{float(v['signed_magnitude_CI']['mean']):.6g}|{v['zero_sensitivity_images']}|")
    lines+=['','TCAV i BA/AP sondy są odrębnymi pomiarami. Pełne probe scores, magnitudes i CI są w RESULT.json i PROBE_METRICS.json.','', '## Detekcja: wynik przed i po','| Ramię | Delta AP50 | Delta średniej IoU | Poprawa/pogorszenie IoU |','|---|---:|---:|---:|']
    for k,v in result['detection'].items():lines.append(f"|{k}|{v['delta_metrics']['AP50']:.6f}|{float(v['film_IoU_delta']['mean']):.6g}|{v['IoU_benefit_images']}/{v['IoU_degradation_images']}|")
    lines+=['','Nie utożsamiamy dodatniej zmiany logitu ani TCAV z poprawą detekcji. Raport obejmuje obie strony, wszystkie epsilon, także pogorszenie i zerowe zmiany. Dokładne bbox/IoU, score responses i błędy finite differences są w FINITE_DIFFERENCES.json.','', '## Podłoga', '50 film-bootstrap separatorów, 50 pair-label placebo z flipem całego filmu wyłącznie train i 50 losowych kierunków na dopasowanym wsparciu/normie. Film-bootstrap CI i testy sign-flip nie liczą klatek jako niezależnych filmów. Paper-style t-test jest opisowy; Bonferroni uwzględnia rodzinę porównań.','', '## Zastrzeżenia i czego wynik nie dowodzi', *result['limitations'], '', 'Nie wykazujemy nowego odkrycia konceptu, semantycznej czystości LDA, identyfikacji egzemplarza narzędzia ani generalizacji na niezależne filmy.']
    text='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI_TCAV.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    labels=list(result['TCAV']);values=[result['TCAV'][k]['film_positive_fraction'] for k in labels];axs[0].bar(labels,values);axs[0].axhline(.5,color='gray',ls='--');axs[0].set_ylim(0,1);axs[0].set_ylabel('TCAV positive fraction (film mean)')
    for k in labels:
        for sign in [-1,1]:axs[1].plot(FRACTIONS,[result['detection'][f'{k}_eps{j}_{sign:+d}']['film_IoU_delta']['mean'] for j in range(4)],marker='o',label=f'{k} {sign:+d}')
    axs[1].axhline(0,color='gray');axs[1].set_xlabel('activation norm fraction');axs[1].set_ylabel('mean delta top1 IoU (film mean)');axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'TCAV_DETECTION.png',dpi=150);plt.close(fig)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);register_research_family();threadpool_limits(4)
    if not args.smoke and (OUT/'RESULT.json').exists():raise RuntimeError('Existing complete result, refusing overwrite')
    js(OUT/('SMOKE_PID.json' if args.smoke else 'PID.json'),dict(pid=os.getpid(),started=time.time(),command=sys.argv,python=sys.executable))
    rows,pairs,w,J,ids=load();cache=dict(np.load(OLD/'TOKENY_PARY.npz'));lda=dict(np.load(OLD/'PODPRZESTRZEN.npz'));meta=read(OLD/'POMIAR.json')
    assert sha(CKPT)==meta['checkpoint_sha256'];assert sha(OLD/'TOKENY_PARY.npz')==meta['npz_sha256']
    assert sha(HERE/'model_snapshot.py')==meta['model_source_sha256'];assert sha(PAIRS/'TOKEN_MASKI.npz')==meta['token_masks_sha256']
    assert np.array_equal(cache['frame_ids'],[r['frame_id'] for r in rows])
    pool=cache['pooled'];norms=[]
    for i in range(400):
        a,b=cache['offsets'][i:i+2];tids=cache['token_ids'][a:b].astype(int);cw=cache['coverage'][a:b];cw=cw/cw.sum()
        np.testing.assert_allclose(w[i,tids],cw,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose((cache['states'][a:b]*cw[:,None,None]).sum(0),pool[i],rtol=2e-5,atol=2e-6)
        if i in ids['train']:norms.append(np.linalg.norm(cache['states'][a:b,0]))
    eps=np.median(norms)*FRACTIONS
    js(OUT/'SPLITS.json',dict(indices=ids,records=[dict(source_index=i,frame_id=r['frame_id'],film=r['video_id'],split=r['tcav_split']) for i,r in enumerate(rows)],source_split_sha256=sha(SPLIT)))
    np.savez_compressed(OUT/'POOLED_FEATURES.npz',pooled=pool,ROI_weights=w,J128=J,train=ids['train'],val=ids['val'],test=ids['test'])
    progress('cache_verified',pairs=400,checkpoint_sha256=sha(CKPT),feature_extraction='reused exact k1000 paired cache; heldout originals only for downstream gradient and verification')
    if args.smoke:bank={k:lda['lda_direction'] for k in ['full768','J128','historical_LDA']};probes={}
    else:bank,probes=learn(pool,rows,J,ids,lda)
    net,device=init_net();weights_sha=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
    test,grad,base,maps,records,ww,verification=run_detector(net,device,rows,pairs,w,ids,cache,bank,eps,args.smoke)
    assert weights_sha==hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
    js(OUT/('SMOKE.json' if args.smoke else 'FINITE_DIFFERENCES.json'),dict(status='passed' if args.smoke else 'complete',device=device,epsilons=eps,hook_verification=verification,records=records,weights_unchanged=True,gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
    if args.smoke:progress('smoke_passed',processed=len(test));return
    np.savez_compressed(OUT/'GRADIENTS_BASELINE.npz',test_indices=test,gradient_lift_covector=grad,baseline_maps=maps,ROI_weights=ww,ROI_lift=ww/np.linalg.norm(ww,axis=1,keepdims=True))
    # Packed map of every FD endpoint retained through decoded boxes; raw logits are reconstructible from q and directions.
    js(OUT/'PROVENANCE.json',dict(checkpoint_sha256=sha(CKPT),weights_sha256=weights_sha,source_before_sha256=sha(SOURCE/'LATENT_PELNY.npz'),cache_sha256=sha(OLD/'TOKENY_PARY.npz'),historical_LDA_sha256=sha(OLD/'PODPRZESTRZEN.npz'),paper_sha256=sha(PAPER),split_sha256=sha(SPLIT),model_source_sha256=sha(HERE/'model_snapshot.py'),code_hashes={p.name:sha(p) for p in HERE.glob('*.py')},python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,normalization='train per-channel SD; raw_coef=standard_coef/SD; unit CAV; ROI lift=w/||w||; pooled shift=epsilon*||w||*v',LDA_conversion='original saved raw unit direction, no U or probe_mu/probe_sd',epsilons=eps))
    progress('analysis_started',test_pairs=len(test));result=analyze(rows,test,grad,base,records,bank,probes);report(result)
    js(OUT/'BIEG_tcav.json',dict(tag='tcav',ziarno=SEED,status='complete',result_sha256=sha(OUT/'RESULT.json'),duration_seconds=time.time()-START))
    progress('complete',result=str(OUT/'RESULT.json'),duration_seconds=time.time()-START)
if __name__=='__main__':
    try:main()
    except Exception:
        progress('failed',traceback=traceback.format_exc());raise
