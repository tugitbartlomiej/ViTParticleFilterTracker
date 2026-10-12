"""Czy zapisane pełne współczynniki, gradienty i źródła są zgodne? Niezgodność hashów, podziału lub iloczynów obala raport."""
import os, sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='4'
sys.dont_write_bytecode=True
import numpy as np
from sklearn.metrics import roc_auc_score
from dane import *
def main():
    env=read(OUT/'ENVIRONMENT.json'); cache=read(OUT/'CACHE_MANIFEST.json')
    split=read(OUT/'SPLITS.json'); met=read(OUT/'METRICS.json'); records=split['records']
    assert met['status']=='complete'
    assert sha(CKPT)==env['checkpoint_sha256']
    assert sha(PAPER)==env['pdf_sha256']
    assert sha(HERE/'run.py')==env['runner_sha256']
    assert sha(HERE/'model_snapshot.py')==env['model_source_sha256']
    assert sha(SPLIT)==split['source_sha256']
    assert sha(OUT/'LATENT_FULL.npy')==cache['sha256']
    for r in records: assert sha(r['path'])==r['sha256']
    X=np.load(OUT/'LATENT_FULL.npy',mmap_mode='r'); assert X.shape==(800,D) and X.dtype==np.float32
    v=np.load(OUT/'CAV_FULL_600x768.npy'); w=np.load(OUT/'COEF_FLAT.npy')
    assert v.shape==(600,768) and w.shape==(D,)
    assert np.allclose(v.reshape(-1),w/np.linalg.norm(w),rtol=1e-5,atol=1e-9)
    assert np.count_nonzero(np.linalg.norm(v,axis=0))==768
    assert np.count_nonzero(np.linalg.norm(v,axis=1))==600
    bank=np.load(OUT/'CAV_BANK_FULL.npy',mmap_mode='r'); assert bank.shape==(91,D)
    g=np.load(OUT/'GRADIENT_FULL.npy',mmap_mode='r'); assert g.shape==(85,D)
    s=np.load(OUT/'TCAV_SENSITIVITIES.npy'); assert s.shape==(85,91)
    # Independent direct main contraction, no proxy/ROI representation.
    direct=np.array([np.dot(gg.astype(float),v.reshape(-1).astype(float)) for gg in g])
    assert np.allclose(direct,s[:,0],rtol=1e-11,atol=1e-12)
    normbank=np.sqrt(np.einsum('ij,ij->i',bank,bank,dtype=np.float64))
    assert np.allclose(normbank,1,rtol=1e-6,atol=1e-7)
    for k in range(0,800,20): assert np.isfinite(X[k:k+20]).all()
    test=np.array(split['indices']['test']); scores=np.load(OUT/'PROBE_SCORES.npy')[test]
    labels=np.array([records[i]['label'] for i in test])>0
    fs=np.array([records[i]['film'] for i in test]); names=sorted(set(fs))
    groups=[np.flatnonzero(fs==f) for f in names]
    rng=np.random.default_rng(SEED+333); auc=[]
    for _ in range(3000):
        pick=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        auc.append(roc_auc_score(labels[pick],scores[pick]))
    auc_ci=dict(mean=float(roc_auc_score(labels,scores)),CI95=np.quantile(auc,[.025,.975]),unit='whole film',resamples=3000,estimator='AUROC on resampled frames from complete paired films')
    null=met['controls']; pperm=null['film_permutation']['empirical_p_two_sided']; prandom=null['full_random']['empirical_p_two_sided']
    result=dict(passed=True,source_hashes_unchanged=True,all_800_image_hashes_unchanged=True,runner_snapshot_matches=True,all_cache_finite=True,full_CAV_shape=list(v.shape),nonzero_channels=768,nonzero_positions=600,norm_bank_min=float(normbank.min()),norm_bank_max=float(normbank.max()),full_contraction_max_abs_error=float(np.max(np.abs(direct-s[:,0]))),positive_images=int((s[:,0]>0).sum()),negative_images=int((s[:,0]<0).sum()),test_AUROC_film_bootstrap=auc_ci,empirical_null=dict(film_permutation=pperm,full_random=prandom,Bonferroni_two_controls=dict(film_permutation=min(1,2*pperm),full_random=min(1,2*prandom)),reject_both_at_05=False,interpretation='Conditional empirical reference, not an unconditional significance test: lambda and preprocessing fixed from observed training data, label exchangeability under the chosen film null is an assumption'),source_manifest_hashes={str(p):sha(p) for p in [PAIRS/'PARY.json',PAIRS/'manifest.json',SPLIT,PAPER,HERE/'dane.py',HERE/'PLAN.md',HERE/'audit.py']})
    js(OUT/'AUDIT.json',result)
    extra='\n## Niezależny audyt artefaktów\n\nPotwierdzono ponownie wszystkie 800 hashów obrazów, cache, checkpoint, PDF, runner i snapshot modelu. CAV ma niezerowe współczynniki na wszystkich 600 pozycjach i wszystkich 768 kanałach. Pełny bezpośredni iloczyn gradientu i CAV odtwarza zapisane sensitivity.\n\nDodatnia pochodna: 7 obrazów, ujemna: 78. Empiryczne p film-permutation wynosi 0.064516, więc ta kontrola nie pozwala odrzucić null przy .05. Dla full-random p=0.032258; po Bonferronim dla dwóch kontroli 0.064516. Nie ogłaszamy statystycznie potwierdzonego semantycznego konceptu. Oba wyniki są warunkowymi odniesieniami do jawnych kontroli, przy stałej lambda i preprocessingu; hipoteza wymienności etykiet nie jest gwarantowana przez dane.\n\nAUROC film-bootstrap CI95: '+str(np.asarray(auc_ci['CI95']).tolist())+'. Wszystkie szczegóły w AUDIT.json.\n'
    for p in [HERE/'WYNIKI_TCAV.md',OUT/'REPORT.md']:
        t=p.read_text(encoding='utf-8')
        if '## Niezależny audyt artefaktów' not in t:p.write_text(t+extra,encoding='utf-8')
    js(OUT/'ARTIFACT_HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.npy','.png','.md']})
    print(json.dumps({k:v for k,v in result.items() if k!='source_manifest_hashes'},default=lambda x:x.tolist(),ensure_ascii=False))
if __name__=='__main__': main()
