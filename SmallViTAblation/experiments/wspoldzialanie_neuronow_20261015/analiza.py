from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wspoldzialanie_neuronow_20261015'

def stats(df):
    metrics=['q_actual','joint_px','single_a_px','single_b_px','interaction_px','nonlinear_ratio',
             'linear_error_ratio','delta_iou','delta_err_px','cos','latent_cos','logit_change']
    return df.groupby('method')[[x for x in metrics if x in df]].median().round(6)

def main():
    d=pd.read_csv(OUT/'interwencje_lokalne.csv.gz')
    g=pd.read_csv(OUT/'interwencje_globalne.csv.gz')
    base=pd.read_csv(OUT/'ramki_bazowe.csv')
    final=d[(d.step>=55000)&(d.alpha==1)]
    held=d[(d.step>=35000)&(d.image==2)&(d.alpha==1)]
    gf=g[(g.step>=55000)&(g.alpha==1)]
    for name,df in [('lokalne_pozne',final),('transfer_obraz3',held),('globalne_pozne',gf)]:
        tab=stats(df);tab.to_csv(OUT/f'{name}.csv')
        print(name);print(tab.to_string())
    perimage=final.groupby(['image','method'])[['q_actual','joint_px','nonlinear_ratio','delta_iou']].median()
    perimage.to_csv(OUT/'lokalne_per_obraz.csv')
    globimage=gf.groupby(['image','method']).agg(joint_px=('joint_px','median'),delta_iou=('delta_iou','median'),
        switch_rate=('switched','mean'),iou_loss_005=('delta_iou',lambda x:np.mean(x<-.05)),
        iou_gain_005=('delta_iou',lambda x:np.mean(x>.05)))
    globimage.to_csv(OUT/'globalne_per_obraz.csv')
    methods=['functional_cancel','matched_cancel','functional_reinforce','matched_reinforce',
             'latent_cancel','latent_reinforce','top_sensitivity','top_latent','random']
    names=['Kompensacja\nramki','Kontrola\nkompensacji','Wzmocnienie\nramki','Kontrola\nwzmocnienia',
           'Kompensacja\nresidual','Wzmocnienie\nresidual','Największa\nczułość','Największy\nwkład','Losowe']
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for metric,ax,title in zip(['q_actual','joint_px','nonlinear_ratio','delta_iou'],axes.flat,
        ['Pozostała część osobnych efektów (q; mniej = silniejsze znoszenie)',
         'Przesunięcie ramki po usunięciu pary [RMS px]',
         'Interakcja nieliniowa / suma norm osobnych efektów',
         'Zmiana IoU po usunięciu pary (ujemna = pogorszenie)']):
        vals=[final[final.method==m][metric].dropna().values for m in methods]
        ax.boxplot(vals,showfliers=False,tick_labels=names)
        ax.set_title(title,fontsize=10);ax.tick_params(axis='x',labelsize=7)
        ax.grid(axis='y',alpha=.2)
        if metric=='delta_iou':ax.axhline(0,color='black',lw=.7)
    fig.suptitle('Ostatnie MLP: pełne ablacje par, kroki ≥ 55000, 3 obrazy × 8 tokenów\nPudełka: mediana i kwartyle obserwacji; nie są przedziałami ufności',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'wyniki_interwencji.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    cols={'functional_cancel':'#137c53','functional_reinforce':'#c34530','frozen_cancel':'#55b596',
          'frozen_reinforce':'#ee9e60','frozen_matched_cancel':'#777777'}
    for b,ax in enumerate(axes):
        for m,c in cols.items():
            part=d[(d.image==b)&(d.alpha==1)&(d.method==m)]
            ts=part.groupby('step').q_actual.median()
            ax.plot(ts.index,ts.values,label=m,color=c,ls='--' if m.startswith('frozen') else '-')
        ax.axvline(33000,color='black',lw=.7);ax.set_title(f'Obraz {b+1}');ax.set_xlabel('Krok uczenia')
        ax.grid(alpha=.2);ax.set_ylim(0,1.12)
    axes[0].set_ylabel('q: pozostała część osobnych efektów')
    axes[-1].legend(fontsize=6);fig.suptitle('Pary dobierane lokalnie i identyfikatory zamrożone na kroku 33000')
    fig.tight_layout();fig.savefig(OUT/'transfer_w_czasie.png',dpi=160);plt.close(fig)
    # Deterministic, representative example: final checkpoint, center token, image 3,
    # first pair from each functional selector. No picking by impressive outcome.
    examples=final[(final.step==65640)&(final.image==2)&(final.token_slot==0)&(final.pair_slot==0)&
                   final.method.isin(['functional_cancel','functional_reinforce'])]
    rgb=np.load(ROOT/'app/cache_uczenia/lasery/2cca2743e1c74b14/dane.npz')['rgb']
    gt=np.load(ROOT/'app/cache_uczenia/lasery/2cca2743e1c74b14/dane.npz')['gt']
    bb=base[(base.step==65640)&(base.image==2)&(base.token_slot==0)].iloc[0]
    box=bb[[f'b{k}' for k in range(4)]].to_numpy(float)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    example_records=[]
    for row,(_,e) in enumerate(examples.iterrows()):
        record=e.to_dict();record['base_box']=box.tolist();record['gt']=gt.tolist();example_records.append(record)
        for col,key in enumerate(['base','da','db','dj']):
            bx=box if key=='base' else box+e[[key+str(k) for k in range(4)]].to_numpy(float)
            ax=axes[row,col];ax.imshow(rgb)
            for xy,color in [(gt,'#58ee69'),(bx,'#ff7855')]:
                ax.add_patch(Rectangle(xy[:2],xy[2]-xy[0],xy[3]-xy[1],fill=False,ec=color,lw=1.5))
            label=['Bez ablacji',f'Tylko neuron {int(e.c)}',f'Tylko neuron {int(e.d)}','Oba neurony'][col]
            ax.set_title(f'{label}\nIoU={iou_np(bx,gt):.3f}',fontsize=10);ax.axis('off')
        axes[row,0].text(-.05,.5,e.method,transform=axes[row,0].transAxes,rotation=90,va='center',fontsize=8)
    fig.suptitle('Obraz 3, krok 65640, token przy środku GT: zielona = GT, pomarańczowa = odczyt tokenu\nIlustracja o z góry ustalonym wyborze; nie jest oceną całego detektora')
    fig.tight_layout();fig.savefig(OUT/'przyklad_ablacji.png',dpi=180);plt.close(fig)
    # Aggregate at image/checkpoint/token level before pairwise baseline contrasts.
    paired=final.groupby(['image','step','token_slot','method']).q_actual.mean().unstack('method')
    contrasts={}
    for target,control in [('functional_cancel','matched_cancel'),('functional_reinforce','matched_reinforce')]:
        v=paired[target]-paired[control]
        contrasts[target]=dict(mean_difference=float(v.mean()),fraction_cases_expected_direction=float((v<0).mean() if 'cancel' in target else (v>0).mean()))
    # Dependence: descriptive, not nominal statistical tests over correlated pairs.
    nonlinear=final.groupby('method').nonlinear_ratio.agg(['median',lambda x:x.quantile(.9),lambda x:x.quantile(.99)])
    nonlinear.columns=['median','p90','p99'];nonlinear.to_csv(OUT/'nieliniowosc_kwantyle.csv')
    alpha=d[(d.step>=55000)&d.method.isin(['functional_cancel','functional_reinforce'])].groupby(['method','alpha'])[['interaction_px','nonlinear_ratio','linear_error_ratio']].median()
    alpha.to_csv(OUT/'sila_ablacji.csv')
    extra=dict(invalid_ratios=int((~d.valid_ratio).sum()),invalid_by_method=d[~d.valid_ratio].groupby('method').size().to_dict(),
        contrasts=contrasts,examples=example_records,
        global_summary=gf.groupby('method').agg(median_shift=('joint_px','median'),median_iou_change=('delta_iou','median'),
            switch_rate=('switched','mean'),fraction_harm_005=('delta_iou',lambda x:float((x<-.05).mean())),
            fraction_help_005=('delta_iou',lambda x:float((x>.05).mean()))).to_dict('index'),
        baseline_iou_late=base[base.step>=55000].groupby('image').base_iou.median().to_dict(),
        baseline_best_iou_late=gf.drop_duplicates(['image','step']).groupby('image').base_iou.median().to_dict(),
        local_cos_correlation=float(spearmanr(final.cos,final.latent_cos).statistic))
    (OUT/'podsumowanie.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXTRA');print(json.dumps({k:v for k,v in extra.items() if k!='examples'},indent=2))
    print('EXAMPLES');print(examples[['method','c','d','single_a_px','single_b_px','joint_px','q_actual','nonlinear_ratio','base_iou','delta_iou']].to_string(index=False))
    print('ALPHA');print(alpha.to_string())

def iou_np(a,b):
    wh=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0);i=wh.prod()
    return i/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-i)

if __name__=='__main__':main()
