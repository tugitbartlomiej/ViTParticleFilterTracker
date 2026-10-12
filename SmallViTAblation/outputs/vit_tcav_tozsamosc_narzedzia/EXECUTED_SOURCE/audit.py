"""Czy metryki i pochodzenie są spójne? Rozjazd cache/latent lub FD obala aparat."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
from sklearn.metrics import balanced_accuracy_score,average_precision_score,roc_auc_score
from dane import *
from run import OLD,film_groups,film_mean,ci,report
def main():
    result=read(OUT/'RESULT.json');rows,pairs,w,J,ids=load();test=ids['test'];films,groups=film_groups(rows,test)
    f=read(OUT/'FINITE_DIFFERENCES.json');rs=f['records'];assert len(rs)==85*7*4*2
    assert f['weights_unchanged'];maps=np.load(OUT/'FD_MAPS.npz');assert len(maps['maps'])==len(rs)
    cache=dict(np.load(OLD/'TOKENY_PARY.npz'));meta=read(OLD/'POMIAR.json')
    assert sha(PAIRS/'manifest.json')==meta['manifest_sha256']
    assert sha(PAIRS/'PARY.json')==meta['pair_manifest_sha256']
    source=read(SOURCE/'SELECTION.json');source_i=source['source_index'];i=source_i;assert rows[i]['frame_id']==source['record']['frame_id']
    before=np.load(SOURCE/'LATENT_PELNY.npz')['before'];a,b=cache['offsets'][i:i+2];tids=cache['token_ids'][a:b].astype(int)
    before_error=float(np.max(np.abs(before[tids]-cache['states'][a:b,0])))
    np.testing.assert_allclose(before[tids],cache['states'][a:b,0],rtol=2e-5,atol=3e-5)
    stats={};rng=np.random.default_rng(SEED+99);draws=rng.integers(0,len(groups),(2000,len(groups)))
    weights=np.zeros(len(test))
    for g in groups:weights[g]=1/len(groups)/len(g)
    y=np.r_[np.ones(len(test)),np.zeros(len(test))]
    for label,probe in result['probe'].items():
        scores=np.array(probe['test']['scores']);pred=scores>=0
        metric=dict(BA=float(balanced_accuracy_score(y,pred,sample_weight=np.tile(weights,2))),AP=float(average_precision_score(y,scores,sample_weight=np.tile(weights,2))),AUROC=float(roc_auc_score(y,scores,sample_weight=np.tile(weights,2))))
        boot=[]
        for draw in draws:
            # Each selected film has equal mass, paired classes stay together.
            ix=np.concatenate([groups[k] for k in draw]);bw=np.concatenate([np.full(len(groups[k]),1/len(draw)/len(groups[k])) for k in draw]);s=np.r_[scores[ix],scores[len(test)+ix]];yy=np.r_[np.ones(len(ix)),np.zeros(len(ix))]
            boot.append([balanced_accuracy_score(yy,s>=0,sample_weight=np.tile(bw,2)),average_precision_score(yy,s,sample_weight=np.tile(bw,2)),roc_auc_score(yy,s,sample_weight=np.tile(bw,2))])
        stats[label]=dict(film_weighted=metric,film_bootstrap_CI95=np.quantile(boot,[.025,.975],axis=0),CI_columns=['BA','AP','AUROC'],resamples=2000)
        result['probe'][label]['test_film_weighted']=stats[label]
    maxerr=max(r['FD_absolute_error'] for r in rs if r['epsilon_index']<2)
    # Every arm uses the identical ROI lift and bounded global epsilon for each image.
    for i in test:
        ir=[r for r in rs if r['source_index']==int(i)]
        assert all(r['relative_norm']<=.01000001 for r in ir)
        for j in range(4):assert len({r['epsilon'] for r in ir if r['epsilon_index']==j})==1
    audit=dict(status='passed',original_before_matches_historical_cache_max_error=before_error,source_index=source_i,source_frame_id=source['record']['frame_id'],manifest_and_pairs_hash_verified=True,finite_difference_endpoint_count=len(rs),FD_small_epsilon_max_error=maxerr,hook_max_error=max(v['tail_vs_actual_GELU_hook_max_error'] for v in f['hook_verification']),film_weighted_probe=stats,weights_unchanged=True,all_interventions_norm_bounded=True,model_training=False)
    js(OUT/'AUDIT.json',audit);js(OUT/'RESULT.json',result)
    os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
    report(result)
    appendix='\n## Audyt i wagi filmów\n\nHistoryczna LDA miała BA ważone po filmach; BA tabeli PROBE_METRICS jest nieważone po obrazach. AUDIT.json i RESULT.probe.test_film_weighted zawierają porównywalne metryki po filmach oraz CI dla BA, AP i AUROC. Zapisany oryginalny LATENT_PELNY.before zgadza się z cache tokenów ROI. Kierunek LDA nie został refitowany.\n'
    for p in [OUT/'RAPORT.md',HERE/'WYNIKI_TCAV.md']:
        with p.open('a',encoding='utf-8') as fp:fp.write(appendix)
    conclusion=('\nWykryto lokalny wpływ kierunków na logit, lecz nie wykazano poprawy jakości detekcji przy zadeklarowanych małych epsilon. '
                'Historyczna LDA ma TCAV po filmach 93.61 procent (CI 86.11–99.17 procent); full768 także 93.61, J128 90.35 procent. '
                'Pięć obrazów ma dokładnie zerową pochodną, ponieważ komórka środka bbox wypada poza wsparciem maski. '
                'Przy największym epsilon LDA plus zwiększa filmową średnią IoU o 0.00001692 (CI obejmuje zero), '
                'minus zmniejsza o 0.00001678 (CI obejmuje zero). AP50 zmienia się odpowiednio o -0.00022529 i -0.00001986; '
                'recall, precision i F1 przy progach .5/.75 nie zmieniają się. Znak konceptu nie daje gwarancji korzyści dla bbox.\n\n'
                'Paper-style test t względem .5 daje małe p, lecz empiryczne null nie potwierdzają swoistości: '
                'full768 placebo p=0.2941, random p=0.2549; J128 placebo p=0.4510, random p=0.3137 (po Bonferronim wszystkie 1). '
                'Są to dwustronne porównania odległości TCAV od .5. Film sign-flip względem .5 także nie jest testem specyficzności konceptu. '
                'Dobre rozpoznanie oryginał/LaMa przez sondę nie jest dowodem poprawy detektora.\n')
    for p in [OUT/'RAPORT.md',HERE/'WYNIKI_TCAV.md']:
        txt=p.read_text(encoding='utf-8');first,rest=txt.split('\n',1);p.write_text(first+'\n'+conclusion+rest,encoding='utf-8')
    p=read(OUT/'PROVENANCE.json');p['audit_source_sha256']=sha(HERE/'audit.py');p['audit_sha256']=sha(OUT/'AUDIT.json');p['original_before_cache_max_error']=before_error;js(OUT/'PROVENANCE.json',p)
    bie=read(OUT/'BIEG_tcav.json');bie['result_sha256']=sha(OUT/'RESULT.json');bie['audit_sha256']=sha(OUT/'AUDIT.json');js(OUT/'BIEG_tcav.json',bie)
    print('AUDIT PASSED',audit['hook_max_error'],maxerr,stats,flush=True)
if __name__=='__main__':main()
