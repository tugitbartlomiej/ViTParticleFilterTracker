from pathlib import Path
import json,time,hashlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr,spearmanr
from sklearn.preprocessing import SplineTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wspoldzialanie_neuronow_20261015/przeciwne_reflektory'
NAMES=['eafb5a973be3d490','c584ef4647149c7c','2cca2743e1c74b14']

def corr(x,y):
    x=np.asarray(x);y=np.asarray(y);valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
    if len(x)<4 or np.std(x)<1e-12 or np.std(y)<1e-12:return dict(n=len(x),pearson=None,spearman=None)
    return dict(n=len(x),pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic))

def detrend(t,y,knots=6):
    B=SplineTransformer(n_knots=knots,degree=3,include_bias=True).fit_transform(np.asarray(t)[:,None])
    return y-B@np.linalg.lstsq(B,y,rcond=None)[0]

def broken(t,y,spacing):
    best=(float('inf'),None,None)
    for a in np.arange(3000,60001,spacing):
        for b in np.arange(a+3000,62641,spacing):
            X=np.c_[np.ones(len(t)),t/1094,np.maximum(t-a,0)/1094,np.maximum(t-b,0)/1094]
            beta=np.linalg.lstsq(X,y,rcond=None)[0];sse=float(np.sum((X@beta-y)**2))
            if sse<best[0]:best=(sse,[int(a),int(b)],beta)
    return dict(sse=best[0],break_steps=best[1],break_epochs=(np.array(best[1])/1094).tolist(),
                slopes_pp_per_epoch=(np.cumsum(best[2][1:])*100).tolist(),beta=best[2].tolist())

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    h=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'dane.npz') for n in NAMES]
    steps=h[0]['kroki'];P=h[0]['pary'].astype(float);ep=int(h[0]['na_epoke']);assert ep==1094
    for x in h[1:]:assert np.array_equal(x['kroki'],steps) and np.array_equal(x['pary'],P)
    prev=np.load(ROOT/'outputs/optyka_hipoteza_20261015/pomiar.npz');assert np.array_equal(prev['steps'],steps)
    featnames=list(prev['feature_names']);features=prev['features']
    geom=[];start=time.time();wd=Path(h[0]['wagi'].item());eye=torch.eye(768,dtype=torch.bool,device='cuda')
    for i,step in enumerate(steps):
        state=torch.load(wd/f'k{int(step):05d}.pt',map_location='cpu',weights_only=True)['net']
        V=state['bloki.5.mlp.fc2.weight'].to('cuda',dtype=torch.float64)
        norms=V.norm(dim=0);vn=V/norms;C=vn.T@vn
        mins=C.masked_fill(eye,1).min(1).values
        rec=float((mins<-.5).double().mean())
        assert abs(rec-P[i])<4e-8
        geom.append(dict(step=int(step),epoch=float(step/ep),opposed=rec,
            count=int((mins<-.5).sum()),opposed04=float((mins<-.4).double().mean()),
            opposed06=float((mins<-.6).double().mean()),mean_min_cos=float(mins.mean()),
            edge_fraction=float((C<-.5).sum())/(768*767),
            effective_rank=float(768**2/C.square().sum()),mean_weight_norm=float(norms.mean()),
            rotation_from_start=float(h[0]['obr_kat0'][i]),
            angular_drift=float(np.median(h[0]['obr_dryf'][i,:,3]))))
        if i%200==0:print(f'geometry {i+1}/{len(steps)} {time.time()-start:.1f}s',flush=True)
    frame=pd.DataFrame(geom)
    for b,x in enumerate(h):
        frame[f'iou_{b+1}']=x['iou'];frame[f'confidence_{b+1}']=x['pew']
        for key in ['ratio_power','ratio_box','net_norm','z_norm']:
            frame[f'{key}_{b+1}']=np.median(features[:,b,:8,featnames.index(key)],axis=1)
    frame.to_csv(OUT/'historia.csv',index=False)
    # Exact regular checkpoints only, no interpolation of correlation inputs.
    uniform=frame[(frame.step%100==0)].copy();assert np.array_equal(uniform.step,np.arange(0,65601,100))
    mainframe=uniform[uniform.step>=1000]
    metrics=[c for c in frame if c not in ['step','epoch','opposed','count']]
    records=[]
    def add(kind,metric,t,p,y):records.append(dict(analysis=kind,metric=metric,**corr(p,y)))
    for metric in metrics:
        add('all_regular_levels',metric,uniform.step,uniform.opposed,uniform[metric])
        add('from1000_levels',metric,mainframe.step,mainframe.opposed,mainframe[metric])
        for knots in [4,6,10]:
            add(f'time_spline_{knots}',metric,mainframe.step,detrend(mainframe.step.values,mainframe.opposed.values,knots),
                detrend(mainframe.step.values,mainframe[metric].values,knots))
        for stride in [100,500,1000]:
            sub=mainframe[mainframe.step%stride==0]
            add(f'increments_{stride}',metric,sub.step.values[1:],np.diff(sub.opposed),np.diff(sub[metric]))
        for a,b in zip([0,5,15,30,45],[5,15,30,45,60]):
            sub=uniform[(uniform.epoch>=a)&(uniform.epoch<b)]
            add(f'phase_{a}_{b}',metric,sub.step,sub.opposed,sub[metric])
    # Training losses: exact join at 100-step observations; minibatch values, not validation loss.
    log=pd.DataFrame(json.loads((wd.parent/'dane/LOG.json').read_text())).rename(columns={'krok':'step'})
    merged=mainframe[['step','opposed']].merge(log,on='step',validate='one_to_one')
    for metric in ['strata','heat','box']:
        add('from1000_levels','training_'+metric,merged.step,merged.opposed,merged[metric])
        add('increments_100','training_'+metric,merged.step.values[1:],np.diff(merged.opposed),np.diff(merged[metric]))
        add('time_spline_6','training_'+metric,merged.step,detrend(merged.step.values,merged.opposed.values),detrend(merged.step.values,merged[metric].values))
    ratings=pd.DataFrame(json.loads((wd.parent/'dane/OCENY.json').read_text())).rename(columns={'krok':'step'})
    ratings=ratings.merge(frame[['step','opposed']],on='step',validate='one_to_one')
    assert len(ratings)==126
    endepochs=ratings[ratings.epoka.notna()&(ratings.step>=1000)].sort_values('step')
    for metric in ['mAP50']:
        add('epoch_levels',metric,endepochs.step,endepochs.opposed,endepochs[metric])
        add('epoch_increments',metric,endepochs.step.values[1:],np.diff(endepochs.opposed),np.diff(endepochs[metric]))
        add('epoch_time_spline_6',metric,endepochs.step,detrend(endepochs.step.values,endepochs.opposed.values),detrend(endepochs.step.values,endepochs[metric].values))
    pd.DataFrame(records).to_csv(OUT/'korelacje.csv',index=False)
    milestones=[]
    for threshold in [0,.01,.10,.25,.50,.53]:
        selected=np.flatnonzero(P>0 if threshold==0 else P>=threshold)
        j=int(selected[0]);milestones.append(dict(threshold=threshold,step=int(steps[j]),epoch=float(steps[j]/ep),value=float(P[j])))
    phases=[]
    for a,b in zip([0,5,15,30,45],[5,15,30,45,60]):
        sub=uniform[(uniform.epoch>=a)&(uniform.epoch<b)]
        slope=np.polyfit(sub.epoch,sub.opposed,1)[0]*100
        phases.append(dict(from_epoch=a,to_epoch=b,slope_pp_epoch=float(slope)))
    fit=broken(uniform.step.values,uniform.opposed.values,500)
    fit2=broken(uniform.step.values,uniform.opposed.values,1000)
    final=frame.iloc[-1].to_dict();peak=frame.iloc[frame.opposed.argmax()].to_dict()
    report=dict(definition='mean_c [min_(d!=c) cos(W2[:,c],W2[:,d]) < -0.5]',
        cache_weight_agreement=True,history_length=len(steps),regular_count=len(uniform),main_count=len(mainframe),
        source_images=NAMES,epoch_steps=ep,final=final,peak=peak,milestones=milestones,phase_slopes=phases,
        piecewise_fit=fit,piecewise_grid1000=fit2,seconds=time.time()-start)
    (OUT/'wyniki.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Actual epoch x-axis throughout; app index axis is not retained.
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True)
    axes[0].plot(frame.epoch,frame.opposed*100,c='#d36b25',label='próg −0,5')
    for col,label in [('opposed04','próg −0,4'),('opposed06','próg −0,6')]:axes[0].plot(frame.epoch,frame[col]*100,alpha=.6,label=label)
    for br in fit['break_epochs']:axes[0].axvline(br,c='black',ls=':',lw=1)
    axes[0].set_ylabel('Reflektory z partnerem [%]');axes[0].legend(loc='lower right')
    colors=['#2379ba','#ca472b','#3b914c']
    for b,c in enumerate(colors):
        axes[1].plot(uniform.epoch,uniform[f'iou_{b+1}'],alpha=.75,lw=.8,c=c,label=f'obraz {b+1}')
        axes[2].plot(uniform.epoch,uniform[f'confidence_{b+1}'],alpha=.75,lw=.8,c=c)
        axes[3].plot(uniform.epoch,uniform[f'ratio_power_{b+1}'],alpha=.8,lw=1,c=c)
    axes[1].set_ylabel('IoU najpewniejszej ramki');axes[1].legend()
    axes[2].set_ylabel('Pewność najpewniejszej ramki')
    axes[3].set_ylabel('Aktywne sumowanie η\n8 tokenów środkowych');axes[3].axhline(1,c='black',ls=':',lw=.7)
    axes[3].set_xlabel('Epoka (1094 kroki)')
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,60)
    fig.suptitle('Przeciwne kierunki wag a detekcja i aktywne wkłady\nRzeczywista oś epok; P jest wspólne dla wszystkich obrazów')
    fig.tight_layout();fig.savefig(OUT/'przebieg_w_czasie.png',dpi=150);plt.close(fig)
    det=pd.DataFrame(records);chosen=['from1000_levels','time_spline_6','increments_100','increments_500','increments_1000']
    names=['Poziomy','Po usunięciu\ntrendu czasu','Przyrost 100\nkroków','Przyrost 500\nkroków','Przyrost 1000\nkroków']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for metric,ax in zip(['iou','confidence'],axes):
        for b,c in enumerate(colors):
            values=[float(det[(det.analysis==kind)&(det.metric==f'{metric}_{b+1}')].pearson.iloc[0]) for kind in chosen]
            ax.plot(np.arange(5),values,'o-',c=c,label=f'obraz {b+1}')
        ax.set_xticks(np.arange(5),names,fontsize=8);ax.set_title('IoU' if metric=='iou' else 'Pewność')
        ax.axhline(0,c='black',lw=.6);ax.set_ylim(-1,1);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Korelacja Pearsona z P');fig.suptitle('Co zostaje ze współzmienności po uwzględnieniu czasu uczenia?')
    fig.tight_layout();fig.savefig(OUT/'korelacje_po_trendzie.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ['final','peak']},indent=2))
    print('FINAL',json.dumps(final));print('PEAK',json.dumps(peak))
    print(det[(det.analysis.isin(chosen+['epoch_levels','epoch_increments','epoch_time_spline_6']))&(~det.metric.str.contains('opposed'))].to_string(index=False))

if __name__=='__main__':main()
