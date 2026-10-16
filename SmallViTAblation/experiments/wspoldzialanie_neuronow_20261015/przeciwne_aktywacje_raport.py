from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje'
NAMES=['eafb5a973be3d490','c584ef4647149c7c','2cca2743e1c74b14']

def table(df,dec=3):
    d=df.copy()
    for c in d.select_dtypes(include='number'):d[c]=d[c].map(lambda x:f'{x:.{dec}f}')
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in d.values)

def main():
    hist=pd.read_csv(OUT/'historia.csv');pair=pd.read_csv(OUT/'pary_pozny_trening.csv')
    meta=json.loads((OUT/'METADANE.json').read_text());ct=json.loads((OUT/'CZAS_QC.json').read_text())
    lag=pd.read_csv(OUT/'opoznienia.csv');null=pd.read_csv(OUT/'opoznienia_kontrola_przesuniec.csv')
    motion=pd.read_csv(OUT/'przeciwdzialanie_ruchowi.csv');select=pd.read_csv(OUT/'selektywnosc_neuronow.csv')
    late=hist[(hist.step>=55000)&hist.group.isin(['bbox','far'])&hist.pair_set.isin(['final','early','control'])]
    rows=[]
    for (name,im,g),x in late.groupby(['pair_set','image','group']):
        w=x.mean_strength*x.n_pairs*x.n_tokens
        r=dict(set=name,image=im,group=g)
        for col in ['cancellation','cancellation_shuffle','balance','destructive_weight','same_sign_weight','both_weight']:
            r[col]=float(np.average(x[col],weights=w))
        r['both_given_either']=float(x.both.sum()/x.either.sum());r['only_given_either']=float(x['only'].sum()/x.either.sum())
        r['both_all']=float(x.both.mean());r['neither_all']=float(x.neither.mean())
        rows.append(r)
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'podsumowanie_pozne.csv',index=False)
    pairrows=[]
    for (name,im,g),x in pair.groupby(['pair_set','image','group']):
        pairrows.append(dict(set=name,image=im,group=g,cancellation_median=x.cancellation.median(),
           cancellation_q25=x.cancellation.quantile(.25),cancellation_q75=x.cancellation.quantile(.75),
           coactive_above_half=float((x.both_given_either>.5).mean()),coactive_below_tenth=float((x.both_given_either<.1).mean())))
    ps=pd.DataFrame(pairrows);ps.to_csv(OUT/'rozklad_par.csv',index=False)
    fig,ax=plt.subplots(3,3,figsize=(15,11),sharex=True)
    for b in range(3):
        for g,col in [('bbox','tab:red'),('far','tab:blue')]:
            x=hist[(hist.image==b+1)&(hist.pair_set=='final')&(hist.group==g)]
            for j,key in enumerate(['both','cancellation','balance']):
                ax[b,j].plot(x.epoch,x[key],color=col,alpha=.25,lw=.6)
                ax[b,j].plot(x.epoch,x[key].rolling(11,center=True,min_periods=1).mean(),color=col,label=g)
        for j in range(3):ax[b,j].set_ylim(0,1);ax[b,j].set_ylabel(f'Obraz {b+1}');ax[b,j].grid(alpha=.2)
    for j,t in enumerate(['Oba powyżej progu (udział token–para)','Skrócenie sumy wkładów C','Równowaga długości wkładów']):ax[0,j].set_title(t)
    for a in ax[-1]:a.set_xlabel('Epoka (1094 kroki)')
    ax[0,0].legend();fig.suptitle('Stałe 844 pary wybrane na końcu: retrospekcja; bbox to obszar ramki, nie segmentacja')
    fig.tight_layout();fig.savefig(OUT/'wspolaktywacja_w_czasie.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for im in range(1,4):
        for r,kind in enumerate(['increments','detrended_levels']):
            for name,style in [('early','-'),('control','--')]:
                for g,col in [('bbox','tab:red'),('far','tab:blue')]:
                    x=lag[(lag.image==im)&(lag.series==kind)&(lag.pair_set==name)&(lag.group==g)]
                    ax[r,im-1].plot(x.lag_steps,x['mean'],style,color=col,label=f'{name}/{g}')
            ax[r,im-1].axhline(0,color='k',lw=.6);ax[r,im-1].grid(alpha=.2);ax[r,im-1].set_title(f'Obraz {im}, {kind}')
            ax[r,im-1].set_ylabel('Średnia korelacja po tokenach i parach')
    for a in ax[1]:a.set_xlabel('Opóźnienie drugiego neuronu (kroki treningu)')
    ax[0,0].legend(fontsize=8);fig.suptitle('Stałe pary z kroku 33000; ocena 33400–65600; opóźnienie nie oznacza kolejności forward')
    fig.tight_layout();fig.savefig(OUT/'opoznienia.png',dpi=160);plt.close(fig)
    # Spatial example chosen explicitly by late average joint strength times shortening.
    candidates=pair[(pair.pair_set=='final')&(pair.group=='bbox')&(pair.image==1)].copy()
    candidates['score']=candidates.strength*candidates.cancellation
    example=candidates.sort_values('score',ascending=False).iloc[0];c=int(example.c);d=int(example.d)
    h=np.load(ROOT/'app/cache_uczenia/lasery'/NAMES[0]/'dane.npz');oi=int(np.flatnonzero(h['kroki']==65600)[0])
    z=np.load(ROOT/'app/cache_uczenia/lasery'/NAMES[0]/'Z.npy',mmap_mode='r')[oi].astype(float)
    info=np.load(OUT/'indeksy.npz');vv=np.load(OUT/'V.npy',mmap_mode='r')[-1].astype(float);norm=np.linalg.norm(vv,axis=1)
    a=z[:,c]*norm[c];b=z[:,d]*norm[d];cos=vv[c]@vv[d]/norm[c]/norm[d]
    C=1-np.sqrt(np.maximum(a*a+b*b+2*a*b*cos,0))/np.maximum(abs(a)+abs(b),1e-20)
    strength=abs(a)+abs(b);C=np.ma.masked_where(strength<.05*np.quantile(strength,.95),C)
    e=vv[c]-vv[d];e/=np.linalg.norm(e);a=z[:,c]*(vv[c]@e);b=z[:,d]*(vv[d]@e)
    fig,ax=plt.subplots(1,4,figsize=(16,4));ax[0].imshow(h['rgb']);ax[0].set_title('Obraz 1 i etykieta bbox')
    for j,(v,t) in enumerate([(a,f'Wkład {c} na wspólnej osi'),(b,f'Wkład {d} na wspólnej osi'),(C,'Skrócenie C (ukryte słabe wkłady)')],1):
        lim=max(abs(a).max(),abs(b).max()) if j<3 else 1
        img=ax[j].imshow(v.reshape(20,30),extent=(0,480,320,0),cmap='coolwarm' if j<3 else 'viridis',vmin=-lim if j<3 else 0,vmax=lim)
        ax[j].set_title(t);fig.colorbar(img,ax=ax[j],shrink=.6)
    gt=h['gt']
    for aa in ax:aa.add_patch(Rectangle(gt[:2],gt[2]-gt[0],gt[3]-gt[1],fill=False,edgecolor='lime',lw=1.5));aa.set_xlim(0,480);aa.set_ylim(320,0)
    fig.suptitle(f'Przykład wybrany po wyniku: para {c}, {d}, krok 65600, cos={cos:.3f}; mapy nie są maską narzędzia')
    fig.tight_layout();fig.savefig(OUT/'przyklad_przestrzenny.png',dpi=160);plt.close(fig)
    # Save machine-readable summary first; prose is completed after checking numeric results.
    res=dict(example=dict(c=c,d=d,cos=float(cos),selection='maximum late strength*cancellation in image1 bbox'),
        late=summary.to_dict('records'),pair_distribution=ps.to_dict('records'),
        time_counts=ct['group_counts'],norm_matching=meta['control_norm_relative_max'])
    (OUT/'PODSUMOWANIE.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    qc=dict(history_rows=len(hist),late_rows=len(late),lag_rows=len(lag),
        finite_history=bool(np.isfinite(hist.select_dtypes('number')).all().all()),
        cancellation_range=[float(hist.cancellation.min()),float(hist.cancellation.max())],
        active_partition_max_error=float(abs(hist.both+hist['only']+hist.neither-1).max()),
        both_either_max_error=float(abs(hist.both+hist['only']-hist.either).max()),
        example_pair=[c,d],files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'experiments/wspoldzialanie_neuronow_20261015').glob('przeciwne_aktywacje*.py'))})
    assert qc['finite_history'] and qc['active_partition_max_error']<1e-6 and qc['both_either_max_error']<1e-6
    (OUT/'WERYFIKACJA.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(ps.to_string(index=False))

if __name__=='__main__':main()
