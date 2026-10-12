"""Czy pełny CAV odróżnia LaMa i zgadza się z pochodną detektora? Brak przewagi nad kontrolami lub błąd FD obala interpretację."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '4'
import sys, time, traceback, platform
sys.dont_write_bytecode = True
import numpy as np
import torch, cv2
from scipy.linalg import eigh, solve
from scipy.stats import ttest_1samp
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits
from dane import *
from model_snapshot import MalyViT
START = time.time()
REPS = 30
BLOCK = 4096
LAMBDAS = np.logspace(-5, 2, 8)
def progress(stage, **kw):
    state = dict(stage=stage, pid=os.getpid(), elapsed_seconds=time.time()-START, **kw)
    js(OUT / 'PROGRESS.json', state)
    print(json.dumps(state, default=str), flush=True)
def model_state(net):
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def image(path, device):
    a = cv2.imread(str(path)); assert a is not None
    a = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2RGB), (480,320), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(a).permute(2,0,1)[None].float().to(device)/255
def centre(row):
    x,y,w,h = row['bbox_source_480']
    return int(np.clip(round((y+h/2)/16-.5),0,19)), int(np.clip(round((x+w/2)/16-.5),0,29))
def metrics(y, score):
    return dict(BA=float(balanced_accuracy_score(y>0, score>=0)), AUROC=float(roc_auc_score(y>0, score)))
def film_ci(values, films, resamples=3000, frame_weighted=False):
    names = sorted(set(films)); groups = [np.flatnonzero(np.asarray(films)==f) for f in names]
    means = np.array([values[g].mean() for g in groups])
    sums = np.array([values[g].sum() for g in groups]); sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(SEED+222)
    draw = rng.integers(0,len(groups),(resamples,len(groups)))
    boot = sums[draw].sum(1)/sizes[draw].sum(1) if frame_weighted else means[draw].mean(1)
    point = float(np.mean(values)) if frame_weighted else float(means.mean())
    return dict(mean=point, CI95=np.quantile(boot,[.025,.975]), unit='whole film', films=len(groups), resamples=resamples, estimator='frame-weighted' if frame_weighted else 'equal-film')
def tcav_summary(sens, test_films):
    return dict(positive_fraction=float((sens>0).mean()), mean_signed=float(sens.mean()), mean_absolute=float(np.abs(sens).mean()), median_signed=float(np.median(sens)), min=float(sens.min()), max=float(sens.max()), zero_count=int((sens==0).sum()), positive_fraction_film_CI=film_ci((sens>0).astype(float),test_films), positive_fraction_frame_CI=film_ci((sens>0).astype(float),test_films,frame_weighted=True), signed_film_CI=film_ci(sens,test_films), absolute_film_CI=film_ci(np.abs(sens),test_films))
def main():
    OUT.mkdir(exist_ok=True)
    progress('initializing')
    rows,pairs,records,ids,films = load()
    y = np.array([r['label'] for r in records],float)
    tr,va,te = [ids[k] for k in ['train','val','test']]
    js(OUT/'SPLITS.json', dict(records=records, films=films, indices=ids, source=str(SPLIT), source_sha256=sha(SPLIT), member_order='pair-major original then synthetic'))
    js(OUT/'REGISTRATION.json', dict(rodzina=HERE.name,klasa='badawcza',plan_sha256=sha(HERE/'PLAN.md'), shared_registry_write=False,reason='Explicit user write scope',pid=os.getpid()))
    torch.set_num_threads(4); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device='cpu'
    gpu={}
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        gpu=dict(name=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total)
        if free > 700*1024**2:
            torch.cuda.set_per_process_memory_fraction(.12); device='cuda'
    ck_hash = sha(CKPT)
    ck = torch.load(CKPT,map_location='cpu',weights_only=True)
    assert ck['krok']==1000
    net=MalyViT().eval(); net.load_state_dict(ck['net'],strict=True)
    net.requires_grad_(False); net=net.to(device)
    before=model_state(net)
    js(OUT/'ENVIRONMENT.json',dict(pid=os.getpid(),python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,gpu=gpu,threads=4,checkpoint=str(CKPT),checkpoint_sha256=ck_hash,checkpoint_step=ck['krok'],state_sha256_before=before,seed=SEED,model_source_sha256=sha(HERE/'model_snapshot.py'),runner_sha256=sha(HERE/'run.py'),pdf_sha256=sha(PAPER)))
    # Complete all 800 genuine image forwards. Never import old pooled caches.
    features=np.lib.format.open_memmap(OUT/'LATENT_FULL.npy',mode='w+',dtype='float32',shape=(800,D))
    held={}
    handle=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o: held.update(z=o.detach()))
    norms=[]
    for i,r in enumerate(records):
        with torch.no_grad(): net(image(r['path'],device))
        z=held.pop('z'); assert tuple(z.shape)==(1,600,768)
        raw=z[0].float().cpu().numpy().reshape(-1)
        assert np.isfinite(raw).all()
        features[i]=raw
        if i in tr: norms.append(float(np.linalg.norm(raw)))
        if i==0 or (i+1)%20==0:
            features.flush(); progress('extracting_full_features',processed=i+1,total=800,device=device,shape=[600,768])
    handle.remove(); features.flush()
    cache_hash=sha(OUT/'LATENT_FULL.npy')
    js(OUT/'CACHE_MANIFEST.json',dict(shape=[800,600,768],storage_shape=[800,D],dtype='float32',bytes=(OUT/'LATENT_FULL.npy').stat().st_size,sha256=cache_hash,checkpoint_sha256=ck_hash,preprocessing='cv2 BGR->RGB resize480x320 INTER_AREA /255; model ImageNet normalization',hook='bloki[-1].mlp.act output GELU',source_images=records,no_pooling=True,no_PCA=True,no_ROI=True,all_channels=True,all_spatial_positions=True))
    # Raw full-feature Gram computed in bounded CPU blocks, centered on TRAIN only.
    G=np.zeros((800,800),dtype='float64')
    for j in range(0,D,BLOCK):
        X=np.asarray(features[:,j:j+BLOCK],dtype='float64')
        G += X@X.T
        if j%(BLOCK*12)==0: progress('full_linear_gram',coordinates=min(j+BLOCK,D),total=D)
    train_mu_dot=G[:,tr].mean(1); mu_norm2=float(G[np.ix_(tr,tr)].mean())
    Kraw=G-train_mu_dot[:,None]-train_mu_dot[None,:]+mu_norm2
    scale2=float(np.diag(Kraw)[tr].mean()); assert scale2>0
    K=Kraw/scale2
    np.save(OUT/'GRAM_CENTERED_SCALED.npy',K)
    mu=np.zeros(D,dtype='float64')
    for j in range(0,D,BLOCK): mu[j:j+BLOCK]=np.asarray(features[tr,j:j+BLOCK],dtype='float64').mean(0)
    np.save(OUT/'TRAIN_MEAN.npy',mu)
    eig,U=eigh(K[np.ix_(tr,tr)]); eig=np.maximum(eig,0)
    uy=U.T@y[tr]
    candidates=[]
    for lam in LAMBDAS:
        a=U@(uy/(eig+lam))
        m=metrics(y[va],K[np.ix_(va,tr)]@a)
        candidates.append(dict(lambda_=float(lam),**m))
    chosen=max(candidates,key=lambda z:(z['BA'],z['AUROC'],z['lambda_']))
    lam=chosen['lambda_']; alpha=U@(uy/(eig+lam))
    # Freeze selection before test evaluation.
    js(OUT/'FREEZE.json',dict(objective='sum_i (y_i - ((x_i-mu_train)/s) dot beta)^2 + lambda * ||beta||2^2; intercept fixed to train label mean 0 (balanced paired data)',regularization=lam,validation_grid=candidates,selection='val BA, then val AUROC, then largest lambda',squared_global_scalar_scale=scale2,featurewise_standardization=False,train_indices=tr,validation_indices=va,selection_frozen_before_test=True))
    alphas=np.zeros((len(tr),1+2*REPS)); alphas[:,0]=alpha
    draws=[]; flips=[]; train_film=np.array([records[i]['film'] for i in tr]); names=films['train']
    for r in range(REPS):
        rng=np.random.default_rng(SEED+r)
        sampled=rng.choice(names,len(names),replace=True)
        counts=np.array([np.count_nonzero(sampled==f) for f in train_film],float)
        active=np.flatnonzero(counts>0); root=np.sqrt(counts[active])
        A=K[np.ix_(tr[active],tr[active])]*root[:,None]*root[None,:]
        sol=solve(A+lam*np.eye(len(active)),root*y[tr[active]],assume_a='pos')
        alphas[active,1+r]=root*sol
        signs={f:int(rng.choice([-1,1])) for f in names}
        labels=y[tr]*np.array([signs[f] for f in train_film])
        alphas[:,1+REPS+r]=U@((U.T@labels)/(eig+lam))
        draws.append(sampled.tolist()); flips.append(signs)
    np.save(OUT/'DUAL_COEFFICIENTS.npy',alphas)
    norms2=np.sum(alphas*(Kraw[np.ix_(tr,tr)]@alphas),axis=0)/(scale2**2)
    normcoef=np.sqrt(np.maximum(norms2,0)); assert np.all(normcoef>1e-12)
    # Exactly reconstruct every coefficient. No pooling/restricted supports.
    count=1+3*REPS
    directions=np.lib.format.open_memmap(OUT/'CAV_BANK_FULL.npy',mode='w+',dtype='float32',shape=(count,D))
    coef=np.empty(D,dtype='float64')
    for j in range(0,D,BLOCK):
        centered=np.asarray(features[tr,j:j+BLOCK],dtype='float64')-mu[j:j+BLOCK]
        w=centered.T@alphas/scale2
        coef[j:j+BLOCK]=w[:,0]
        directions[:1+2*REPS,j:j+BLOCK]=(w/normcoef).T
    for r in range(REPS):
        rng=np.random.default_rng(SEED+10000+r)
        v=rng.normal(size=D); v/=np.linalg.norm(v)
        directions[1+2*REPS+r]=v.astype('float32')
    directions.flush()
    np.save(OUT/'CAV_FULL_600x768.npy',np.asarray(directions[0]).reshape(600,768))
    np.save(OUT/'COEF_FLAT.npy',coef)
    intercept=-float(mu@coef)
    js(OUT/'CAV_METADATA.json',dict(shape=[600,768],flatten_order='C token-major channel-minor',unit_norm=float(np.linalg.norm(directions[0])),raw_coef_norm=float(normcoef[0]),raw_intercept=intercept,positive_class='original',negative_class='LaMa-erased',main_index=0,bootstrap_indices=[1,1+REPS],film_permutation_indices=[1+REPS,1+2*REPS],random_indices=[1+2*REPS,count],repetitions=REPS,regularization_fixed_for_controls=True,bootstrap_draws=draws,permutation_film_signs=flips,adaptation='controlled matched LaMa negatives instead of random unrelated counterexamples; ridge linear separator'))
    # Verify exact reconstructed primal equals dual on all 800 rows.
    primal=np.zeros(800)
    for j in range(0,D,BLOCK): primal+=np.asarray(features[:,j:j+BLOCK],dtype='float64')@coef[j:j+BLOCK]
    primal+=intercept; dual=K[:,tr]@alpha
    primal_error=float(np.max(np.abs(primal-dual)))
    assert primal_error<1e-7,primal_error
    probe={s:metrics(y[ii],dual[ii]) for s,ii in ids.items()}
    test_film_all=[records[i]['film'] for i in te]
    probe['test']['BA_film_CI']=film_ci(((dual[te]>=0)==(y[te]>0)).astype(float),test_film_all)
    null_probe=[metrics(y[te],K[np.ix_(te,tr)]@alphas[:,1+REPS+r]) for r in range(REPS)]
    js(OUT/'PROBE_METRICS.json',dict(main=probe,permuted=null_probe,primal_dual_max_abs_error=primal_error))
    np.save(OUT/'PROBE_SCORES.npy',dual)
    progress('CAV_learned',test_BA=probe['test']['BA'],test_AUROC=probe['test']['AUROC'],regularization=lam)
    test_orig=np.array([i for i in te if records[i]['label']==1]); assert len(test_orig)==85
    gradients=np.lib.format.open_memmap(OUT/'GRADIENT_FULL.npy',mode='w+',dtype='float32',shape=(85,D))
    logits=[]; sparsity=[]
    for k,i in enumerate(test_orig):
        row=rows[records[i]['pair_index']]; cy,cx=centre(row)
        cache={}
        def gradient_hook(m,a,o):
            z=o.detach().requires_grad_(True); cache['z']=z; return z
        handle=net.bloki[-1].mlp.act.register_forward_hook(gradient_hook)
        out,_=net(image(records[i]['path'],device)); target=out[0,cy,cx,0]
        g=torch.autograd.grad(target,cache['z'])[0][0].detach().cpu().numpy()
        handle.remove(); assert g.shape==(600,768) and np.isfinite(g).all()
        gradients[k]=g.reshape(-1); logits.append(float(target.detach().cpu()))
        sparsity.append(int(np.count_nonzero(np.linalg.norm(g,axis=1))))
        if (k+1)%10==0 or k==0: gradients.flush(); progress('full_gradient_TCAV',processed=k+1,total=85)
    gradients.flush()
    sensitivity=np.zeros((85,count),dtype='float64')
    for j in range(0,D,BLOCK):
        sensitivity+=np.asarray(gradients[:,j:j+BLOCK],dtype='float64')@np.asarray(directions[:,j:j+BLOCK],dtype='float64').T
    np.save(OUT/'TCAV_SENSITIVITIES.npy',sensitivity)
    test_films=[records[i]['film'] for i in test_orig]
    main_s=tcav_summary(sensitivity[:,0],test_films)
    group_means=lambda ss: np.array([ss[np.asarray(test_films)==f].mean(axis=0) for f in sorted(set(test_films))])
    film_tcavs=group_means((sensitivity>0).astype(float)).mean(0)
    null_stats={}
    for name, a,b in [('bootstrap',1,1+REPS),('film_permutation',1+REPS,1+2*REPS),('full_random',1+2*REPS,count)]:
        fractions=(sensitivity[:,a:b]>0).mean(0)
        ns=film_tcavs[a:b]
        null_stats[name]=dict(frame_positive_fractions=fractions,film_positive_fractions=ns,mean_signed=sensitivity[:,a:b].mean(0),mean_absolute=np.abs(sensitivity[:,a:b]).mean(0),mean_TCAV=float(fractions.mean()),empirical_p_two_sided=(1+int(np.count_nonzero(np.abs(ns-.5)>=abs(film_tcavs[0]-.5))))/(REPS+1) if name!='bootstrap' else None)
    t=ttest_1samp((sensitivity[:,1:1+REPS]>0).mean(0),.5)
    js(OUT/'TCAV_METRICS.json',dict(main=main_s,controls=null_stats,paper_t_test_descriptive=dict(statistic=float(t.statistic) if np.isfinite(t.statistic) else None,p_value=float(t.pvalue) if np.isfinite(t.pvalue) else None,threshold=.05,valid_inference=False,reason='Shared train/heldout films; repetitions not independent datasets; original paper uses random negative sets, typically 500'),empirical_null_resolution=1/(REPS+1),test_original_indices=test_orig,films=test_films,GT_centre_logits=logits,nonzero_gradient_token_counts=sparsity,gradient_restriction=False,architecture_note='After final GELU fc2, residual, norm and head are token-local; zero elsewhere is architectural, not an imposed ROI mask'))
    # True complete forward-hook +/- epsilon interventions; compare to autograd.
    finite=[]; epsbase=float(np.median(norms))
    for k in [0,42,84]:
        i=test_orig[k]; cy,cx=centre(rows[records[i]['pair_index']]); x=image(records[i]['path'],device)
        for dindex in [0,1+REPS,1+2*REPS]:
            v=torch.from_numpy(np.array(directions[dindex]).reshape(1,600,768)).to(device)
            for fraction in [1e-5,1e-4,1e-3]:
                eps=epsbase*fraction; vals=[]
                for sign in [-1,1]:
                    h=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o,sgn=sign:o+sgn*eps*v)
                    with torch.no_grad(): out,_=net(x)
                    h.remove(); vals.append(float(out[0,cy,cx,0].cpu()))
                fd=(vals[1]-vals[0])/(2*eps); ad=float(sensitivity[k,dindex])
                finite.append(dict(test_index=k,image_index=int(i),direction=dindex,epsilon=eps,fraction=fraction,minus_logit=vals[0],plus_logit=vals[1],autograd=ad,central_difference=fd,absolute_error=abs(fd-ad),relative_error=abs(fd-ad)/max(abs(ad),1e-8),sign_agreement=bool(np.sign(fd)==np.sign(ad))))
    # Float32 cancellation for tiny derivatives is reported explicitly, not silently passed.
    main_fd=[f for f in finite if f['direction']==0 and f['fraction']==1e-3]
    fd_pass=all(f['absolute_error']<max(2e-5,.03*abs(f['autograd'])) for f in main_fd)
    js(OUT/'FINITE_DIFFERENCE.json',dict(records=finite,main_largest_epsilon_pass=fd_pass,tolerance='absolute error < max(2e-5, 3% |autograd|) for main CAV at train latent norm *1e-3 on 3 heldout images',epsilon_chosen_from_train_only=True,no_ROI_hook=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full=np.asarray(directions[0]).reshape(600,768); energy=(full.astype(float)**2)
    channel_energy=energy.sum(0); spatial=energy.sum(1).reshape(20,30)
    np.save(OUT/'CHANNEL_ENERGY.npy',channel_energy);np.save(OUT/'SPATIAL_ENERGY.npy',spatial)
    fig,ax=plt.subplots(figsize=(12,3)); ax.plot(channel_energy);ax.set(xlabel='Channel (all 768)',ylabel='Squared CAV energy',title='Full CAV: channel energy; no channel selection');fig.tight_layout();fig.savefig(OUT/'CHANNEL_ENERGY.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(spatial);fig.colorbar(im,ax=ax,label='Squared CAV energy');ax.set(title='Full CAV spatial energy: all 600 tokens',xlabel='Token column',ylabel='Token row');fig.tight_layout();fig.savefig(OUT/'SPATIAL_ENERGY.png',dpi=160);plt.close(fig)
    after=model_state(net); final_hash=sha(CKPT)
    assert before==after and ck_hash==final_hash
    state=dict(model_state_unchanged=before==after,checkpoint_unchanged=ck_hash==final_hash,BN_frozen=True,eval_mode=not net.training,parameters_require_grad=any(p.requires_grad for p in net.parameters()),state_sha256_after=after,checkpoint_sha256_after=final_hash,finite_difference_pass=fd_pass)
    js(OUT/'VERIFICATION.json',state)
    elapsed=time.time()-START
    result=dict(status='complete' if fd_pass else 'completed_with_FD_warning',pid=os.getpid(),runtime_seconds=elapsed,device=device,images=800,latent_shape=[600,768],dimensions=D,train_val_test_pairs=[240,75,85],train_val_test_films=[69,23,24],repetitions=REPS,probe=probe,tcav=main_s,verification=state,controls=null_stats)
    js(OUT/'METRICS.json',result)
    lines=['# Pełny CAV i adaptowany TCAV','', '## Pytanie i aparat','Czy pełny kierunek oryginał versus usunięcie LaMa jest liniowo rozdzielny i wpływa dodatnio na logit centrum narzędzia? Odczyt 800 rzeczywistych obrazów, pełne 600 × 768 GELU, checkpoint oryginalny k1000. Brak poolingu, PCA, selekcji kanałów lub maskowania gradientu.','', '## Wynik',f"Test sondy: BA {probe['test']['BA']:.6f}, AUROC {probe['test']['AUROC']:.6f}.",f"TCAV: {main_s['positive_fraction']:.6f} dodatnich pochodnych na 85 oryginałach; średnia signed {main_s['mean_signed']:.8g}, średnia |S| {main_s['mean_absolute']:.8g}.",f"Film-weighted TCAV {main_s['positive_fraction_film_CI']['mean']:.6f}, CI95 {np.asarray(main_s['positive_fraction_film_CI']['CI95']).tolist()}.",f"30 bootstrap CAV, 30 film-permutation CAV, 30 full-random normmatched. Empiryczne two-sided p dla film-permutation {null_stats['film_permutation']['empirical_p_two_sided']:.6f}, full-random {null_stats['full_random']['empirical_p_two_sided']:.6f}; rozdzielczość 1/31.",f"Finite differences: main largest epsilon pass = {fd_pass}; szczegóły i float32 cancellation w FINITE_DIFFERENCE.json.",f"Primal-dual error {primal_error:.3g}. Wagi i BN niezmienione: {before==after}. Checkpoint hash niezmieniony: {ck_hash==final_hash}.",f"Czas rzeczywisty {elapsed:.1f} sekund, device {device}, PID {os.getpid()}.",'','## Podłoga i adaptacje','Pełne permutacje znaków etykiet: jeden znak na film treningowy, oba człony każdej pary zachowane i zawsze przeciwne. 30 losowych kierunków w całych 460800 wymiarach, normy 1. Bootstrap całych filmów. Lambda dobrana tylko na val; kontrole i bootstrap używają zamrożonej lambdy. Gram solver jest dokładnym liniowym ridge z globalnym skalarem i średnią train, bez featurewise scaling. Detektor: pre-sigmoid logit w komórce ustalonej z GT, zamiast logitu klasy w oryginalnym artykule. LaMa matched counterexamples zamiast losowych obrazów. Test t względem .5 tylko opisowy, nie dowód istotności.','', '## Zastrzeżenia i czego wynik nie dowodzi','460800 wymiarów przy 480 próbkach train; kierunek może kodować pozycję, tło, artefakty LaMa i różnicę formatów JPEG/PNG. Heldout dotyczy dopasowania konceptu; obrazy mogły wcześniej uczyć ViT. To nie dowód wyłącznej tożsamości narzędzia ani przyczynowej poprawy treningu. Interwencja w latencie może wychodzić poza manifold obrazów. Gradient poza jednym tokenem naturalnie zeruje się przez token-local końcówkę sieci; nie narzucono ROI. Filmowe CI i empiryczne null pozostają warunkowe względem tego zbioru, checkpointu i kontroli.','', '## Źródła i reprodukcja',f'Checkpoint: `{CKPT}`; SHA256 `{ck_hash}`.',f'PDF przeczytany w całości, 10 stron: `{PAPER}`; §3.2-3.6 i przypis 1 str.4, §4.1-4.4.', 'Uruchomienie: `F:/Python/python.exe -B experiments/vit_cav_pelny_latent_pary400/run.py` z SmallViTAblation. Manifest zawiera 800 hashów obrazów. CAV_FULL_600x768.npy to jednostkowa macierz, COEF_FLAT.npy to nieznormalizowane współczynniki z intercept w CAV_METADATA.json.']
    (HERE/'WYNIKI_TCAV.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    js(OUT/'ARTIFACT_HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.npy','.png','.md']})
    progress(result['status'],runtime_seconds=elapsed,test_BA=probe['test']['BA'],test_AUROC=probe['test']['AUROC'],TCAV_positive_fraction=main_s['positive_fraction'],FD_pass=fd_pass)
if __name__=='__main__':
    try:
        with threadpool_limits(limits=4): main()
    except BaseException as e:
        traceback.print_exc()
        progress('failed',error=repr(e))
        raise
