from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.preprocessing import SplineTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wspoldzialanie_neuronow_20261015/ruch_a_jakosc'
NAMES=['eafb5a973be3d490','c584ef4647149c7c','2cca2743e1c74b14']
PRIMARY=['speed_center','speed_ratio','coherence','vib_center','node_speed','eta_center']
LABELS=['Prędkość centrum w 192D','Prędkość centrum / tło','Zgodność kierunków ruchu','Odchylenie vib w aplikacji','Prędkość węzłów aplikacji','Aktywne wzmocnienie η']

def corr(x,y):
    if np.std(x)<1e-14 or np.std(y)<1e-14:return float('nan')
    return float(np.corrcoef(x,y)[0,1])

def residual(t,y,k):
    b=SplineTransformer(n_knots=k,degree=3,include_bias=True).fit_transform(t[:,None])
    return y-b@np.linalg.lstsq(b,y,rcond=None)[0]

def ci_block(x,y,block,seed):
    rng=np.random.default_rng(seed);n=len(x); vals=[]
    for _ in range(1000):
        start=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        idx=(start[:,None]+np.arange(block)).ravel()[:n]
        vals.append(corr(x[idx],y[idx]))
    return np.quantile(vals,[.025,.975]).tolist()

def match(t,high,radius):
    a=np.flatnonzero(~high);b=np.flatnonzero(high)
    if len(a)==0 or len(b)==0:return np.empty((0,2),int)
    distance=np.abs(t[a,None]-t[b][None,:])
    cost=np.full((len(a),len(b)+len(a)),1e9)
    cost[:,:len(b)]=np.where(distance<=radius,distance,1e9)
    cost[np.arange(len(a)),len(b)+np.arange(len(a))]=1e6
    rows,cols=linear_sum_assignment(cost)
    valid=cols<len(b);rows=rows[valid];cols=cols[valid]
    assert np.all(distance[rows,cols]<=radius)
    return np.c_[a[rows],b[cols]]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/optyka_hipoteza_20261015/pomiar.npz'
    p=np.load(source);steps=p['steps'];ts=np.arange(1000,65601,100)
    idx=np.searchsorted(steps,ts);before=np.searchsorted(steps,ts-100)
    assert np.array_equal(steps[idx],ts) and np.array_equal(steps[before],ts-100)
    xp=p['x_pre'].astype(np.float64);dx=(xp[idx]-xp[before])/100
    norms=np.linalg.norm(dx,axis=-1)
    features=p['features'];fn=list(p['feature_names']);ids=p['ids']
    assert xp.shape[1:]==(3,24,192)
    rows=[];counts=[];match_rows=[];pair_rows=[];correlations=[];checks=[];caches=[]
    for im,name in enumerate(NAMES):
        h=np.load(ROOT/'app/cache_uczenia/lasery'/name/'dane.npz');caches.append(h)
        assert np.array_equal(h['kroki'],steps)
        center=ids[im,:8];back=ids[im,16:24]
        assert len(set(ids[im]))==24 and not set(center)&set(back)
        speed=norms[:,im,:8].mean(1);bg=norms[:,im,16:24].mean(1)
        den=norms[:,im,:8].sum(1)
        assert np.all(bg>0) and np.all(den>0)
        nodes=h['wezly'].astype(np.float64)
        z=dict(image=np.full(len(ts),im+1),step=ts,epoch=ts/float(h['na_epoke']),
            iou=h['iou'][idx].astype(float),confidence=h['pew'][idx].astype(float),
            speed_center=speed,speed_background=bg,speed_ratio=speed/bg,
            coherence=np.linalg.norm(dx[:,im,:8].sum(1),axis=1)/den,
            vib_center=h['vib'][idx][:,center].mean(1).astype(float),
            node_speed=np.linalg.norm((nodes[idx][:,center]-nodes[before][:,center])/100,axis=-1).mean(1),
            eta_center=features[idx,im,:8,fn.index('ratio_power')].mean(1).astype(float),
            center_drift=np.linalg.norm(dx[:,im,:8].mean(1),axis=1),
            local_center=h['lok'][idx][:,center].mean(1).astype(float),
            global_app=h['glob'][idx].astype(float),
            box_speed=np.sqrt(np.mean((h['ramka'][idx]-h['ramka'][before])**2,axis=1))/100)
        high=(z['iou']>=.8)&(z['confidence']>=.8);z['high']=high
        frame=pd.DataFrame(z);rows.append(frame)
        checks.append(dict(image=im+1,finite=bool(np.isfinite(frame.select_dtypes('number')).all().all()),
            zero_bg=int((bg==0).sum()),zero_motion=int((den==0).sum()),
            coherence_range=[float(z['coherence'].min()),float(z['coherence'].max())],
            window=int(h['okno']),projection_retained_variance=float(h['udzial_D']),
            center_tokens=center.tolist(),rgb_hash=hashlib.sha256(h['rgb'].tobytes()).hexdigest()))
        assert checks[-1]['finite'] and np.all(z['coherence']<=1+1e-12)
        for th in [.75,.8,.85]:
            counts.append(dict(image=im+1,threshold=th,n_high=int(((z['iou']>=th)&(z['confidence']>=th)).sum()),n=len(ts)))
        for radius in [1000,2000]:
            pairs=match(ts,high,radius)
            assert len(np.unique(pairs))==pairs.size
            for low,good in pairs:
                pair_rows.append(dict(image=im+1,radius=radius,low_step=int(ts[low]),high_step=int(ts[good]),
                    gap=int(abs(ts[low]-ts[good])),iou_low=z['iou'][low],iou_high=z['iou'][good],
                    confidence_low=z['confidence'][low],confidence_high=z['confidence'][good]))
            for metric in PRIMARY:
                a=z[metric];low=a[pairs[:,0]];good=a[pairs[:,1]]
                rec=dict(image=im+1,radius=radius,metric=metric,n_pairs=len(pairs),
                    raw_high_median=float(np.median(a[high])) if high.any() else None,
                    raw_low_median=float(np.median(a[~high])) if (~high).any() else None)
                if len(pairs):
                    delta=good-low; rec.update(high_median=float(np.median(good)),low_median=float(np.median(low)),
                        median_paired_delta=float(np.median(delta)),mean_paired_delta=float(np.mean(delta)),
                        median_paired_pct=float(np.median(100*delta/low)),
                        fraction_high_larger=float(np.mean(delta>0)),
                        mean_gap=float(np.mean(abs(ts[pairs[:,0]]-ts[pairs[:,1]]))))
                match_rows.append(rec)
        for mi,metric in enumerate(PRIMARY):
            for ti,target in enumerate(['iou','confidence']):
                a=z[metric];b=z[target]
                for knots in [4,6,10]:
                    ra=residual(ts,a,knots);rb=residual(ts,b,knots)
                    rec=dict(image=im+1,metric=metric,target=target,knots=knots,
                        raw=corr(a,b),residual=corr(ra,rb),residual_spearman=float(spearmanr(ra,rb).statistic),
                        increments=corr(np.diff(a),np.diff(b)))
                    if knots==6:
                        for block in [20,50]:
                            lo,hi=ci_block(ra,rb,block,15000+im*100+mi*10+ti)
                            rec[f'ci{block}_low']=lo;rec[f'ci{block}_high']=hi
                    correlations.append(rec)
    history=pd.concat(rows,ignore_index=True);history.to_csv(OUT/'historia.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'progi.csv',index=False)
    matches=pd.DataFrame(match_rows);matches.to_csv(OUT/'porownania_dopasowane.csv',index=False)
    pairs=pd.DataFrame(pair_rows);pairs.to_csv(OUT/'pary_checkpointow.csv',index=False)
    cs=pd.DataFrame(correlations);cs.to_csv(OUT/'korelacje.csv',index=False)
    (OUT/'WERYFIKACJA.json').write_text(json.dumps(dict(states=647,step_delta=100,images=checks),indent=2),encoding='utf-8')
    fig,ax=plt.subplots(7,3,figsize=(15,19),sharex=True)
    for im,frame in enumerate(rows):
        t=frame.epoch.to_numpy();high=frame.high.to_numpy()
        ax[0,im].plot(t,frame.iou,label='IoU',lw=.8);ax[0,im].plot(t,frame.confidence,label='Pewność',lw=.8)
        ax[0,im].axhline(.8,color='black',ls=':',lw=.8);ax[0,im].set_title(f'Obraz {im+1}: high={high.sum()}/647')
        ax[0,im].set_ylim(0,1.05)
        for j,m in enumerate(PRIMARY):
            ax[j+1,im].plot(t,frame[m],color='#506680',lw=.7)
            ax[j+1,im].scatter(t[high],frame.loc[high,m],s=2,color='#15946b',alpha=.5)
            if im==0:ax[j+1,im].set_ylabel(LABELS[j])
        ax[-1,im].set_xlabel('Epoka (1094 kroki)')
        for a in ax[:,im]:a.grid(alpha=.2)
    ax[0,0].legend(fontsize=8);fig.suptitle('Ruch na tym samym zdjęciu; zielone punkty: IoU i pewność ≥ 0,8\nPrędkości na krok, z różnic co 100 kroków. Węzły/vib zawierają wygładzenie z przyszłości.',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'przebiegi.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,target in zip(axes,['iou','confidence']):
        mat=np.array([[cs[(cs.image==im)&(cs.metric==m)&(cs.target==target)&(cs.knots==6)].residual.iloc[0] for im in [1,2,3]] for m in PRIMARY])
        img=ax.imshow(mat,vmin=-.5,vmax=.5,cmap='RdBu_r',aspect='auto')
        for j in range(6):
            for k in range(3):ax.text(k,j,f'{mat[j,k]:+.2f}',ha='center',va='center')
        ax.set_xticks(range(3),['Obraz 1','Obraz 2','Obraz 3']);ax.set_yticks(range(6),LABELS);ax.set_title(target)
    axes[1].set_yticklabels([])
    fig.subplots_adjust(left=.25,right=.87,wspace=.18,bottom=.1,top=.85)
    cax=fig.add_axes([.90,.20,.018,.55])
    fig.colorbar(img,cax=cax,label='Korelacja reszt po spline czasu (6 węzłów)')
    fig.suptitle('Powiązanie z jakością po uwzględnieniu trendu uczenia')
    fig.savefig(OUT/'korelacje_po_czasie.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Ilustracja: minimum odstępu czasu, następnie największa różnica IoU.
    selected=pairs[(pairs.radius==1000)&(pairs.image==1)].copy()
    selected['delta_iou']=selected.iou_high-selected.iou_low
    selected=selected.sort_values(['gap','delta_iou'],ascending=[True,False]).iloc[0]
    h=caches[0];fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key,description in zip(axs,['low_step','high_step'],['Pozostały stan','Wysoka jakość']):
        k=int(selected[key]);ix=int(np.searchsorted(steps,k));ax.imshow(h['rgb'])
        for box,color,label in [(h['gt'],'lime','GT'),(h['ramka'][ix],'red','Predykcja')]:
            x1,y1,x2,y2=box;ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=color,lw=2,label=label))
        ax.set_title(f'{description}: krok {k}\nIoU={h["iou"][ix]:.3f}, pewność={h["pew"][ix]:.3f}');ax.axis('off')
    axs[0].legend();fig.tight_layout();fig.savefig(OUT/'przyklad_pary.png',dpi=160);plt.close(fig)
    summary=dict(counts=counts,matching=matches.astype(object).where(pd.notna(matches),None).to_dict('records'),correlations=correlations,example=selected.to_dict())
    (OUT/'wyniki.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    files=[source,Path(__file__),Path(__file__).with_name('ruch_a_jakosc_PROTOKOL.md')]+[ROOT/'app/cache_uczenia/lasery'/n/'dane.npz' for n in NAMES]
    (OUT/'MANIFEST.json').write_text(json.dumps([dict(path=str(f),size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],indent=2),encoding='utf-8')
    print(matches[matches.radius==1000][['image','metric','n_pairs','median_paired_pct','fraction_high_larger']].to_string(index=False))
    print(cs[cs.knots==6][['image','metric','target','raw','residual','increments']].to_string(index=False))

if __name__=='__main__':main()
