"""Separate actual PyKoopman NNDMD temporal90/10 experiment. No prior outputs edited."""
from pathlib import Path
import os,sys
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'4')
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
sys.path.insert(0,str(DATA))
import gc,json,time,shutil,importlib.metadata as metadata
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from nndmd_fit import prepare_trajectories,window_weights,fit_package,save_nn,validate_module,learned_rollout,install_loss_and_rollout_adapter,SEED
from pykoopman_fit import build_pairs,fit_model,forecast,metrics_one,majority_sign,write_json,sha256,now,save_model
from report_pykoopman import sign_all
FIT=list(range(8))+list(range(12,34))
HOLD=[8,9,10,11]
HORIZONS=list(range(1,11))
GRID=[(5,1.),(10,1.),(20,1.),(10,0.)]
EPOCHS=20


def selection_rmse(values):
    rmses=values['identity_J128']['rmse_by_horizon_updates']
    return float(np.sqrt(np.mean([v*v for v in rmses.values()])))


def aggregate(rows):
    result=[]
    for h in HORIZONS:
        part=[r for r in rows if r['horizon_updates']==h*10]
        means={k:float(np.mean([r[k] for r in part if r[k] is not None])) if any(r[k] is not None for r in part) else None
               for k in part[0] if k not in ['image_index','horizon_updates']}
        means['mse_ratio_vs_persistence']=means['mse_state']/means['mse_persistence']
        means['mse_skill_vs_persistence']=1-means['mse_ratio_vs_persistence']
        for k in ['state','motion','persistence']:means['rmse_'+k]=float(np.sqrt(means['mse_'+k]))
        means['rmse_ratio_vs_persistence']=float(np.sqrt(means['mse_ratio_vs_persistence']))
        result.append({'horizon_updates':h*10,'images':len(part),**means})
    return result


def get_data():
    latent=np.load(DATA/'LATENT_768.npy',mmap_mode='r');masks=np.load(DATA/'MASKI.npz')
    assert latent.shape==(101,34,600,768)
    return latent,masks['coverage'],masks['tool'],np.load(DATA/'KANALY_J128.npz')['channels0']


class TemporalValidation(Callback):
    def __init__(self,label,latent,cov,tool,scale,channels):
        self.label=label;self.latent=latent;self.cov=cov;self.tool=tool;self.scale=scale;self.channels=channels
        self.history=[];self.best_score=float('inf');self.best_state=None;self.best_epoch=None;self.best_metric=None
    def on_train_epoch_end(self,trainer,module):
        epoch=trainer.current_epoch+1
        if epoch%2 and epoch!=EPOCHS:return
        was_training=module.training
        values=validate_module(module,self.latent,self.cov,self.tool,FIT,90,self.scale,horizons=HORIZONS,channels=self.channels)
        score=selection_rmse(values)
        logs={str(k)+'_last_batch':float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()
              if k in ['loss','rnn_loss','rec_loss','rnn_full','rec_full','rnn_J128','rec_J128']}
        row={'epoch':epoch,'selection_joint_RMSE_J128':score,**logs,**values};self.history.append(row)
        if score<self.best_score:
            self.best_score=score;self.best_epoch=epoch;self.best_metric=row
            self.best_state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
            torch.save({'state_dict':self.best_state,'epoch':epoch,'metrics':values,'score':score},ROOT/f'{self.label}_BEST_VALIDATION.pt')
        write_json(ROOT/f'{self.label}_HISTORIA.json',self.history)
        write_json(ROOT/'POSTEP.json',{'phase':'fit30_temporal_selection','candidate':self.label,'epoch':epoch,'max_epochs':EPOCHS,
                                     'RMSE_J128_joint':score,'best_epoch':self.best_epoch,'best_RMSE':self.best_score,'updated':now()})
        print(f'{self.label} epoch{epoch}/{EPOCHS}: temporalval jointRMSE_J128={score:.7f}, best={self.best_score:.7f}',flush=True)
        if was_training:module.train()


def train():
    assert not (ROOT/'ZAMROZENIE.json').exists(),'Already frozen; no retuning'
    assert torch.cuda.is_available()
    started=time.perf_counter();latent,cov,tool,channels=get_data()
    sources=['run.py','../nndmd_fit.py','../pykoopman_fit.py','../report_pykoopman.py']
    hashes={name:sha256(ROOT/name) for name in sources};protocol=sha256(ROOT/'PLAN_NNDMD.md')
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in sources:shutil.copyfile(ROOT/name,snapshot/Path(name).name)
    write_json(ROOT/'RUNTIME.json',{'started':now(),'source_hashes':hashes,'protocol_sha256':protocol,'seed':SEED,
       'versions':{k:metadata.version(k) for k in ['pykoopman','pydmd','torch','numpy','scipy','lightning']},
       'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'grid':GRID,'epochs_per_candidate':EPOCHS,
       'data_sha256':json.loads((DATA/'DANE_768.json').read_text())['latent_sha256']})
    trajectories,w,scale,meta=prepare_trajectories(latent,cov,tool,FIT,90)
    write_json(ROOT/'TRAJEKTORIE_TRAIN.json',meta)
    print(f'ACTUAL NNDMD GPU START:30FIT, {len(trajectories)} trajectories x91 states, 90 transitions, input/learned/output768',flush=True)
    candidates=[]
    for lf,beta in GRID:
        label=f'LF{lf}_B{int(beta)}'
        callback=TemporalValidation(label,latent,cov,tool,scale,channels)
        reg,config=fit_package(trajectories,w,scale,lf,EPOCHS,ROOT,label,callback,channels=channels,beta=beta)
        save_nn(ROOT,label,reg,{**config,'selected_epoch':callback.best_epoch},callback.best_state)
        candidates.append({'label':label,'look_forward':lf,'beta':beta,'best_epoch':callback.best_epoch,
                           'validation_joint_RMSE_J128':callback.best_score,'best_metrics':callback.best_metric,
                           'config':config})
        write_json(ROOT/'KANDYDACI.json',candidates)
        del reg,callback;gc.collect();torch.cuda.empty_cache()
    chosen=min(candidates,key=lambda c:(c['validation_joint_RMSE_J128'],c['look_forward'],c['best_epoch'],c['beta']))
    write_json(ROOT/'WYBOR.json',{'selected_at':now(),'selected':chosen,'candidates':candidates,'train_indices':FIT,
       'temporal_validation_indices':FIT,'train_end_updates':900,'validation_updates':[900,1000],
       'criterion':'sqrt mean MSE raw decodedJ128 over10horizons, coverage/equalimage','holdout_not_used':HOLD})
    for suffix in ['.pt','_A_LEARNED_768.npy','_GENERATOR_768.npy','_CONFIG.json']:
        shutil.copyfile(ROOT/f'NNDMD_{chosen["label"]}{suffix}',ROOT/f'NNDMD_PREFIX900{suffix}')
    # Fixed baseline selected in protocol, same training prefix, no final holdout reads.
    jlatent=np.load(DATA/'LATENT_J128.npy',mmap_mode='r')
    x,y,ww,pairmeta=build_pairs(jlatent,cov,tool,FIT,90)
    baseline=fit_model(x,y,ww,.01);save_model(ROOT,'EDMD_PREFIX900_J128.npz',baseline,channels)
    write_json(ROOT/'EDMD_PAIRS.json',pairmeta);del x,y,ww
    assert hashes=={name:sha256(ROOT/name) for name in sources},'Source changed during training'
    assert protocol==sha256(ROOT/'PLAN_NNDMD.md')
    write_json(ROOT/'ZAMROZENIE.json',{'frozen_at':now(),'selected':chosen,'model_file':'NNDMD_PREFIX900.pt',
       'model_sha256':sha256(ROOT/'NNDMD_PREFIX900.pt'),'baseline_sha256':sha256(ROOT/'EDMD_PREFIX900_J128.npz'),
       'source_hashes':hashes,'protocol_sha256':protocol,'fit_indices':FIT,'holdout_indices':HOLD,
       'fit_max_update':900,'forecast_origin_update':900,'forecast_end_update':1000,
       'train_transitions':90,'validation_transitions':10,'seconds_training_selection':time.perf_counter()-started,
       'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'dimensions':[768,768,768],
       'no_holdout_used_for_selection':True,'no_future_states_for_gradients_normalization':True})
    write_json(ROOT/'POSTEP.json',{'phase':'frozen','updated':now()});print('FROZEN PREFIX900; ready separate final test',flush=True)


def metric_row(image,h,start,truth,pred,w,majority,full=False):
    row={'image_index':image,'horizon_updates':h*10,**metrics_one(start,truth,pred,w,majority),**sign_all(truth-start,pred-start,w,majority)}
    if full:row={k.replace('all128','all768'):v for k,v in row.items()}
    return row


def evaluate():
    assert not (ROOT/'WYNIKI.json').exists(),'Completed; no repeated final test'
    frozen=json.loads((ROOT/'ZAMROZENIE.json').read_text())
    assert sha256(ROOT/'PLAN_NNDMD.md')==frozen['protocol_sha256']
    assert frozen['source_hashes']=={name:sha256(ROOT/name) for name in frozen['source_hashes']}
    assert sha256(ROOT/frozen['model_file'])==frozen['model_sha256']
    cp=torch.load(ROOT/frozen['model_file'],map_location='cpu');c=cp['config'];channels=np.array(c['channels0'])
    import pykoopman as pk
    reg=pk.regression.NNDMD(dt=1,look_forward=c['look_forward'],config_encoder=c['encoder'],config_decoder=c['decoder'])
    install_loss_and_rollout_adapter(reg._regressor,channels,c['identity_beta'])
    reg._regressor.load_state_dict(cp['state_dict']);module=reg._regressor.cuda().eval()
    module._fixed_inference_operator=torch.tensor(cp['A_learned'],device=module.device)
    latent,cov,tool,_=get_data();majority,_=majority_sign(latent,cov,tool,FIT,0,90)
    mean=np.zeros(768)
    for image in FIT:
        ids=np.flatnonzero(tool[image]&(cov[image]>0));w=cov[image,ids].astype(np.float64);w/=w.sum()
        mean+=np.sum(np.asarray(latent[90,image,ids],dtype=np.float64)*w[:,None],axis=0)/len(FIT)
    np.save(ROOT/'REVERSION_FIT900.npy',mean)
    edmd=np.load(ROOT/'EDMD_PREFIX900_J128.npz');baseline={'A':edmd['A'],'b':edmd['b']}
    rows={key:[] for key in ['full768','J128','reversion_full768','reversion_J128','EDMD_J128']};recon=[]
    full=np.lib.format.open_memmap(ROOT/'FORECAST_FULL768.npy',mode='w+',dtype=np.float32,shape=(10,4,600,768))
    jpred=np.lib.format.open_memmap(ROOT/'FORECAST_J128.npy',mode='w+',dtype=np.float32,shape=(10,4,600,128))
    for pos,image in enumerate(HOLD):
        start=np.array(latent[90,image],dtype=np.float32,copy=True);pred=learned_rollout(module,start,c['scale'],10)
        full[:,pos]=pred;jpred[:,pos]=pred[...,channels]
        epred=forecast(baseline,start[:,channels],10)
        ids=np.flatnonzero(tool[image]&(cov[image]>0));w=cov[image,ids].astype(np.float64);w/=w.sum()
        with torch.no_grad():
            rec=(module._decoder(module._encoder(torch.tensor(start[ids],device=module.device)/c['scale']))*c['scale']).cpu().numpy()
        recon.append({'image_index':image,'mse_full768':float(np.sum(w[:,None]*(rec.astype(float)-start[ids])**2)/768),
                      'mse_J128':float(np.sum(w[:,None]*(rec[:,channels].astype(float)-start[ids][:,channels])**2)/128)})
        for h in HORIZONS:
            truth=np.asarray(latent[90+h,image,ids],dtype=np.float64);s=start[ids].astype(float);p=pred[h-1,ids].astype(float)
            for key,ss,tt,pp,maj in [('full768',s,truth,p,majority),('J128',s[:,channels],truth[:,channels],p[:,channels],majority[channels]),
              ('reversion_full768',s,truth,np.broadcast_to(mean,s.shape),majority),
              ('reversion_J128',s[:,channels],truth[:,channels],np.broadcast_to(mean[channels],s[:,channels].shape),majority[channels]),
              ('EDMD_J128',s[:,channels],truth[:,channels],epred[h-1,ids],majority[channels])]:
                rows[key].append(metric_row(image,h,ss,tt,pp,w,maj,full='full768' in key))
        full.flush();jpred.flush();print(f'FINAL image{image} complete',flush=True)
    del full,jpred
    metrics={key:{'per_image_horizon':rr,'aggregate':aggregate(rr)} for key,rr in rows.items()}
    metrics['reconstruction_origin']={'per_image':recon,'rmse_full768':float(np.sqrt(np.mean([r['mse_full768'] for r in recon]))),
       'rmse_J128':float(np.sqrt(np.mean([r['mse_J128'] for r in recon])))}
    write_json(ROOT/'METRYKI_HOLDOUT.json',metrics)
    j=metrics['J128']['aggregate'];full=metrics['full768']['aggregate']
    result={'status':'complete','completed':now(),'method':'actual PyKoopmanNNDMD full768 with explicit loss/sampler adapter',
            'selected':frozen['selected'],'dimensions':[768,768,768],'train_transitions':90,'train_updates':[0,900],
            'validation_transitions':10,'validation_updates':[900,1000],'holdout_indices':HOLD,
            'exploratory_not_pristine':True,'no_future_measurements':True,'forecast_horizons_updates':list(range(10,101,10)),
            'J128_H10':j[0],'J128_H100':j[-1],'full768_H100':full[-1],'sign85_horizons':[r['horizon_updates'] for r in j if r['sign_accuracy_all128']>=.85],
            'model_sha256':frozen['model_sha256'],'reconstruction_origin':metrics['reconstruction_origin']}
    write_json(ROOT/'WYNIKI.json',result);write_json(ROOT/'POSTEP.json',{'phase':'complete','updated':now()})
    print(json.dumps({'H100J128RMSE':j[-1]['rmse_state'],'persistence':j[-1]['rmse_persistence'],
                      'H100signALL128':j[-1]['sign_accuracy_all128'],'sign85_horizons':result['sign85_horizons']},indent=2),flush=True)


def audit_report():
    result=json.loads((ROOT/'WYNIKI.json').read_text());metrics=json.loads((ROOT/'METRYKI_HOLDOUT.json').read_text())
    frozen=json.loads((ROOT/'ZAMROZENIE.json').read_text());cp=torch.load(ROOT/frozen['model_file'],map_location='cpu')
    a=cp['A_learned'].astype(float);sv=np.linalg.svd(a,compute_uv=False)
    full=np.load(ROOT/'FORECAST_FULL768.npy',mmap_mode='r');jpred=np.load(ROOT/'FORECAST_J128.npy',mmap_mode='r')
    channels=np.array(cp['config']['channels0']);same=all(np.array_equal(full[h][...,channels],jpred[h]) for h in range(10))
    checks={'A768square':a.shape==(768,768),'forecast_full_finite':bool(np.isfinite(full).all()),'J128_exact_full_slice':same,
       'model_sha_matches':sha256(ROOT/frozen['model_file'])==frozen['model_sha256'],
       'source_hashes_match':frozen['source_hashes']=={name:sha256(ROOT/name) for name in frozen['source_hashes']},
       'protocol_sha_matches':sha256(ROOT/'PLAN_NNDMD.md')==frozen['protocol_sha256'],
       'all_10_horizons':len(metrics['J128']['aggregate'])==10,'correct_sqrt_after_aggregation':all(abs(q['rmse_state']**2-q['mse_state'])<1e-14 for q in metrics['J128']['aggregate'])}
    assert all(checks.values())
    audit={'created':now(),'checks':checks,'all_passed':True,'spectral_radius_A':float(np.abs(np.linalg.eigvals(a)).max()),
      'norm2_A':float(sv[0]),'norm2_A_power10':float(np.linalg.norm(np.linalg.matrix_power(a,10),2)),
      'no_tuning':True,'reconstruction_origin':metrics['reconstruction_origin']}
    write_json(ROOT/'AUDYT.json',audit)
    selected=frozen['selected'];j=metrics['J128']['aggregate'];f=metrics['full768']['aggregate'];e=metrics['EDMD_J128']['aggregate'];r=metrics['reversion_J128']['aggregate']
    lines=['# NNDMD full768, temporal90/10: rzeczywisty trening i test','',
      f"Zakończono {result['completed']}. Wybrano {selected['label']}, epokę{selected['best_epoch']}; walidacyjny wspólny RMSE J128={selected['validation_joint_RMSE_J128']:.6f}. Wszystkie30FIT trenowano na91stanach0..900, czyli90przejściach co10aktualizacji. Walidacja900..1000 to10przejść i100aktualizacji. Nie jest to wcześniejsza prognoza500aktualizacji.",'',
      f"Główny wynik H100: signALLJ128={100*j[-1]['sign_accuracy_all128']:.3f}%, RMSE={j[-1]['rmse_state']:.6f}, persistence={j[-1]['rmse_persistence']:.6f}, ratioRMSE={j[-1]['rmse_ratio_vs_persistence']:.6f}, MSEskill={j[-1]['mse_skill_vs_persistence']:.6f}. Cel85%: {result['sign85_horizons'] or 'brak spełniających horyzontów'}.",'',
      'Actual pykoopman1.2.1 NNDMD.fit(listtrajectory), pełne wejście768, encoder768→768→768swish, learnedA768×768=exp(G), lineardecoder768→768. J128 odczytano dopiero z decodedfull768. Learned axes różnią się od native; brakPCA/redukcji. Jeden krokA oznacza10aktualizacji ViT. G nie dowodzi generatora pojedynczych aktualizacji. Decoder/encoder nie gwarantują dokładnej inwersji.','',
      'Strata: rnn_full+rec_full+beta*(rnn_J128+rec_J128). rnn to wielokrokowy maskedMSE decodedforecast, rec to rekonstrukcjaorigin i przyszłych celów w gradientowym TRAINoknie<=900. Adam1e-3 aktualizuje encoder, decoder i generator, z pierwszym i drugim momentem gradientów, beta1=.9,beta2=.999,eps1e-8,gradientclipnorm1. Zainstalowany pakiet niezmieniony; sampler/loss rozszerzone jawnym adapterem. Jeden wspólny scalar z ważonegoFITprefixu, mean0. Coverage/equalimage realizowane przez sampler w oczekiwaniu; końcowe metryki deterministycznie z tymi wagami. Logloss/składowe to ostatni minibatch, nie epochmeans.','',
      'Wybór wyłącznie30FIT temporalvalidation900..1000: sqrt(mean MSE po10horyzontach), rawdecodedJ128. MSE i ten RMSE mają ten sam argmin. Nie użyto tych przyszłych stanów w gradientach/normalizacji. Wybrany checkpoint bez refit na przyszłości, zamrożony przed końcowym4HOLDOUT; encoder origin900 użyty raz, potem learnedA i decoder bez futuremeasurements/re-encodingu.','',
      '| H aktualizacji | ALL128 znak | RMSE NN | RMSE persistence | RMSE EDMD J128 | RMSE reversion | MSEskill NN |','|---:|---:|---:|---:|---:|---:|---:|']
    for q,ee,rr in zip(j,e,r):lines.append(f"| {q['horizon_updates']} | {100*q['sign_accuracy_all128']:.3f}% | {q['rmse_state']:.6f} | {q['rmse_persistence']:.6f} | {ee['rmse_state']:.6f} | {rr['rmse_state']:.6f} | {q['mse_skill_vs_persistence']:.6f} |")
    lines += ['',f"Full768 przyH100: RMSE={f[-1]['rmse_state']:.6f}, persistence={f[-1]['rmse_persistence']:.6f}, skill={f[-1]['mse_skill_vs_persistence']:.6f}. Rekonstrukcja origin900: RMSEfull768={metrics['reconstruction_origin']['rmse_full768']:.6f}, J128={metrics['reconstruction_origin']['rmse_J128']:.6f}; osobny błąd, nie ścisła podłoga przyszłej prognozy.",'',
      f"Stabilność learnedA: rho={audit['spectral_radius_A']:.6f}, norm2A={audit['norm2_A']:.6f}, norm2A^10={audit['norm2_A_power10']:.6f}. Zapisany J128 bitowo zgodny z fullslice: {same}. Wszystkie kontrole audytu przeszły.",'',
      'BatchEDMD to nowy rzeczywisty weightedPyKoopman fit na tym samym30FITprefix0..900 w nativeJ128+constant, lambda=.01 ustalone wcześniej; nie kopiujemy starego modelu500. TylkoJ128, bez udawania pełnego768. Reversion to stała globalna ważona średnia tokenówtool30FIT na900. Persistence to zachowanie origin900. Wszystkie baselines identyczne dane/origin/horyzont/wagi, lecz cele i koszty wyszukiwania modeli różnią się.','',
      'Ograniczenia: czteryHOLDOUT już wcześniej oglądane, nowy wynik eksploracyjny. Wszystkie34obrazy są TRAINViT, holdout tylko identyfikacji dynamiki.25/30FIT w dokładnejbazie1500,5train poza nią. Jedna trajektoria uczenia, jeden seed,20epok/kandydat nie dowodzą globalnego optimum. Brak testu pomiędzy checkpointami co10. J128 wybrane wcześniej na500. Poprzednie prefix500 zostało zachowane; wyniku H100 z900 nie należy przedstawiać jako poprawy wcześniejszegoH500 z500.','',
      f"Koszt czterech kandydatów i wyboru {frozen['seconds_training_selection']:.2f}s, peak GPU tensorallocated {frozen['peak_gpu_allocated_bytes']/2**20:.2f}MiB (nie całkowita pamięć procesu).",'',
      'Artefakty: PLAN_NNDMD.md, source_snapshot/, RUNTIME.json, TRAJEKTORIE_TRAIN.json, KANDYDACI.json, WYBOR.json, ZAMROZENIE.json, NNDMD_PREFIX900.pt, NNDMD_PREFIX900_A_LEARNED_768.npy, FORECAST_FULL768.npy, FORECAST_J128.npy, METRYKI_HOLDOUT.json, WYNIKI.json, AUDYT.json.','',
      'Odtworzenie: izolowany nadrzędny .venv-pykoopman/python, run.py --stage train, następnie --stage evaluate i --stage audit-report, w świeżymkatalogu. Run odmawia retuningu po freeze. Źródła: https://github.com/dynamicslab/pykoopman/blob/v1.2.1/src/pykoopman/regression/_nndmd.py .']
    (ROOT/'RAPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','evaluate','audit-report'],required=True)
    stage=parser.parse_args().stage
    if stage=='train':train()
    elif stage=='evaluate':evaluate()
    else:audit_report()
