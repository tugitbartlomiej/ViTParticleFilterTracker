"""Native channels only; rank/scaler/fit use train, selection uses val, test separate."""
from common import *
import argparse,csv,time,psutil,shutil,warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score,roc_auc_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
KS=[1,2,4,8,16,32,64,128,256,384,512,768]
CS=[.01,.1,1,10]

def sources():
 split=read(PREVIOUS_PAIRS/'PODZIAL.json');rows=read(PAIRS/'manifest.json');records=split['records'];assert len(rows)==len(records)==400
 for i,(r,s) in enumerate(zip(rows,records)):assert s['source_index']==i and r['frame_id']==s['frame_id'] and split['films'][r['video_id']]==s['split']
 return rows,records

def partition(rows,records,name):
 ids=np.array([r['source_index'] for r in records if r['split']==name]);films=sorted({rows[i]['video_id'] for i in ids});counts={f:sum(rows[i]['video_id']==f for i in ids) for f in films};w=np.array([1/len(films)/counts[rows[i]['video_id']] for i in ids]);assert abs(w.sum()-1)<1e-12
 return ids,films,w

def arrays(pool,ids,J):
 X=np.concatenate([pool[ids,0][:,J],pool[ids,1][:,J]]);y=np.r_[np.ones(len(ids),int),np.zeros(len(ids),int)];return X,y

def metrics(y,p,w):
 weights=np.tile(w,2)/2;labels=p>=.5;ba=float(balanced_accuracy_score(y,labels,sample_weight=weights));acc=float(accuracy_score(y,labels,sample_weight=weights));assert abs(ba-acc)<1e-12
 cm=confusion_matrix(y,labels,sample_weight=weights)
 return dict(film_weighted_sensitivity=float(cm[1,1]/cm[1].sum()),film_weighted_specificity=float(cm[0,0]/cm[0].sum()),unweighted_accuracy=float(accuracy_score(y,labels)),film_weighted_BA=ba,film_weighted_accuracy=acc,film_weighted_AUROC=float(roc_auc_score(y,p,sample_weight=weights)),confusion_weighted=confusion_matrix(y,labels,sample_weight=weights),threshold=.5)

def choose():
 assert (HERE/'PLAN.md').exists();threadpool_limits(4);start=time.time();rows,records=sources();pool=np.load(OLD/'IDENTYFIKACJA_K500.npz')['pooled400'].astype(float)
 ti,tf,tw=partition(rows,records,'train');vi,vf,vw=partition(rows,records,'val');assert (len(ti),len(vi),len(tf),len(vf))==(240,75,69,23)
 assert np.isfinite(pool[np.r_[ti,vi]]).all();delta=pool[ti,0]-pool[ti,1];mu_delta=tw@delta;sd_delta=np.sqrt(tw@((delta-mu_delta)**2));effect=mu_delta/np.maximum(sd_delta,1e-12);rank=np.lexsort((np.arange(768),-np.abs(effect)))
 both=np.concatenate([pool[ti,0],pool[ti,1]]);weights=np.tile(tw,2)/2;mu=weights@both;sd=np.sqrt(weights@((both-mu)**2));sd[sd<1e-12]=1.;grid=[];bank={}
 for K in KS:
  J=rank[:K];X,y=arrays(pool,ti,J);V,yv=arrays(pool,vi,J)
  for C in CS:
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);clf=LogisticRegression(C=C,penalty='l2',solver='lbfgs',max_iter=3000,tol=1e-8).fit((X-mu[J])/sd[J],y,sample_weight=np.tile(tw,2)*len(ti))
   converged=not any(issubclass(z.category,ConvergenceWarning) for z in caught);p=clf.predict_proba((V-mu[J])/sd[J])[:,1];m=metrics(yv,p,vw);key=f'K{K}_C{C}'
   grid.append(dict(K=K,C=C,key=key,converged=converged,iterations=int(clf.n_iter_[0]),validation=m));bank[key]=clf
  print('val K',K,'maxBA',max(r['validation']['film_weighted_BA'] for r in grid if r['K']==K),flush=True)
 eligible=[r for r in grid if r['validation']['film_weighted_BA']>.90 and r['converged']]
 if not eligible:
  js(HERE/'WYBOR_WALIDACJA.json',dict(status='no_val_candidate_over90',grid=grid,needs_new_train_val_path_before_test=True));print('NO VAL >90, test remains unread',flush=True);return
 selected=min(eligible,key=lambda r:(r['K'],-r['validation']['film_weighted_BA'],-r['validation']['film_weighted_AUROC'],r['C']));K=selected['K'];J=rank[:K];clf=bank[selected['key']]
 np.savez_compressed(HERE/'KANALY_MODEL.npz',channels0=J,channels1=J+1,rank0=rank,paired_mean_train=mu_delta,paired_sd_train=sd_delta,effect_train=effect,scaler_mean_train=mu,scaler_sd_train=sd,coef_standardized=clf.coef_[0],intercept_standardized=clf.intercept_,coef_physical=clf.coef_[0]/sd[J],intercept_physical=clf.intercept_-mu[J]@(clf.coef_[0]/sd[J]))
 with (HERE/'RANKING_KANALOW.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.writer(f);writer.writerow(['rank1','channel0','channel1','paired_mean_train','paired_sd_train','effect_train','selected'])
  for k,j in enumerate(rank):writer.writerow([k+1,j,j+1,mu_delta[j],sd_delta[j],effect[j],bool(k<K)])
 record=dict(status='frozen_before_native_classifier_test',selected=selected,channels0=J,channels1=J+1,grid=grid,train_pairs=len(ti),train_films=len(tf),val_pairs=len(vi),val_films=len(vf),ranking='train filmweighted abs(mean pairedDelta)/SDpairedDelta, ties channel0',
  selector='minimum K in predefined list with valBA>0.90, then greatestBA, greatestAUROC, smallestC',threshold=.5,class_weights_equal=True,film_weights_equal=True,scaler_film_weighted=True,
  full_pool_source_sha=sha(OLD/'IDENTYFIKACJA_K500.npz'),split_sha=sha(PREVIOUS_PAIRS/'PODZIAL.json'),manifest_sha=sha(PAIRS/'manifest.json'),checkpoint_sha=read(OLD/'IDENTYFIKACJA_K500.json')['checkpoint500_sha256'],model_sha=sha(HERE/'KANALY_MODEL.npz'),
  ignored_source_keys=['U500','eigenvalues','probe_mu','probe_sd','probe_coef','probe_intercept'],PCA_used=False,SVD_used=False,rotation_used=False,test_evaluated=False,source_provenance='pooled400: coverageweighted RAW lastGELU768 on originalROI same tokens in both members, before U projection',seconds=time.time()-start,free_disk=shutil.disk_usage(HERE).free,RAM_free=psutil.virtual_memory().available)
 js(HERE/'WYBOR_WALIDACJA.json',record);print('FREEZE K',K,'channels1',list(J+1),'C',selected['C'],'valBA',selected['validation']['film_weighted_BA'],flush=True)

def test():
 threadpool_limits(4);assert not (HERE/'OCENA_OBECNOSCI_TEST.json').exists(),'Native classifier test already exists; no automatic retest';choice=read(HERE/'WYBOR_WALIDACJA.json');assert choice['status']=='frozen_before_native_classifier_test' and not choice['test_evaluated'];assert choice['model_sha']==sha(HERE/'KANALY_MODEL.npz');assert choice['full_pool_source_sha']==sha(OLD/'IDENTYFIKACJA_K500.npz')
 assert choice['split_sha']==sha(PREVIOUS_PAIRS/'PODZIAL.json') and choice['manifest_sha']==sha(PAIRS/'manifest.json')
 rows,records=sources();pool=np.load(OLD/'IDENTYFIKACJA_K500.npz')['pooled400'].astype(float);ids,films,w=partition(rows,records,'test');assert len(ids)==85 and len(films)==24
 model=dict(np.load(HERE/'KANALY_MODEL.npz'));J=model['channels0'];X,y=arrays(pool,ids,J);raw=X@model['coef_physical']+model['intercept_physical'][0];standard=((X-model['scaler_mean_train'][J])/model['scaler_sd_train'][J])@model['coef_standardized']+model['intercept_standardized'][0];assert np.max(np.abs(raw-standard))<1e-10
 from scipy.special import expit
 p=expit(raw);m=metrics(y,p,w);byfilm={f:np.flatnonzero([rows[i]['video_id']==f for i in ids]) for f in films};perfilm=np.array([np.mean(np.r_[p[byfilm[f]]>=.5,p[len(ids)+byfilm[f]]<.5]) for f in films]);assert abs(perfilm.mean()-m['film_weighted_BA'])<1e-12
 rng=np.random.default_rng(20261002);bootstrap=np.mean(perfilm[rng.integers(0,len(films),(2000,len(films)))],axis=1);ci=np.quantile(bootstrap,[.025,.975]);np.savez_compressed(HERE/'OCENA_OBECNOSCI_TEST.npz',source_indices=ids,probability_original=p[:len(ids)],probability_removed=p[len(ids):],film_BA=perfilm)
 js(HERE/'OCENA_OBECNOSCI_TEST.json',dict(status='single_native_axis_test_complete',selected_K=len(J),selected_C=choice['selected']['C'],metrics=m,CI95_BA_film_bootstrap=ci,bootstrap_unit='whole film, all paired members kept',repeats=2000,test_pairs=len(ids),test_films=len(films),point_over90=m['film_weighted_BA']>.90,lower_CI_over90=float(ci[0])>.90,axes_model_sha=sha(HERE/'KANALY_MODEL.npz'),raw_standard_max_error=float(np.max(np.abs(raw-standard))),limitation='KnownROI pooledimage-level original versus LaMa, same testfilms examined by earlier classifiers; not untouched independent replication, not individual token/detector accuracy or identity versus other instrument.'))
 print('TEST NATIVE',m,'CI',ci,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);arg=p.parse_args();choose() if arg.stage=='validation' else test()
