"""Plot actual saved pair counts on an epoch axis, without invented samples."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'Dane/opposite_pair_counts_by_epoch';OUT=ROOT/'Wykresy'
done=json.loads((DATA/'DONE.json').read_text())
for name,h in done['files'].items():assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h,name
rows=json.loads((DATA/'COUNTS_BY_EPOCH.json').read_text(encoding='utf-8'))
steps=sorted({r['step'] for r in rows});epochs=np.array(steps)*32/35000
lookup={(r['block'],r['step'],r['threshold_deg']):r for r in rows}
assert len(rows)==432 and len(steps)==12
def values(b,t):return np.array([lookup[b,s,t]['same_token_observed_unique_pairs'] for s in steps])
no170=all(not values(b,170).any() for b in range(1,7))
thresholds=[150,160,170] if no170 else [150,160,170,175,179]
colors={150:'#0072B2',160:'#D55E00',170:'#7A55A3',175:'#009E73',179:'#CC3377'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17,
    'svg.fonttype':'none','savefig.facecolor':'white'})
fig,axes=plt.subplots(2,3,figsize=(15,9.4),sharex=True,sharey=True)
fig.subplots_adjust(left=.075,right=.965,bottom=.165,top=.77,wspace=.22,hspace=.35)
fig.suptitle('Jak zmienia się liczba przeciwstawnych par w trakcie treningu?',
             x=.075,y=.967,ha='left',fontsize=20,fontweight='bold',color='#18314B')
fig.text(.075,.912,'6 bloków | 12 zapisanych checkpointów | 294 528 par na blok | pełne 192D ze znakiem GELU',
         fontsize=11,color='#44556A')
highest=max(int(values(b,t).max()) for b in range(1,7) for t in thresholds)
for b,ax in enumerate(axes.flat,1):
    ax.set_title(f'Blok {b}',fontweight='bold',color='#18314B')
    for t in thresholds:
        y=values(b,t)
        label=f'Kąt ≥{t}°'
        if t==170 and no170:label='≥170°, ≥175°, ≥179°: zawsze 0'
        ax.plot(epochs,y,marker='o' if t!=170 else '.',ms=4,lw=2 if t==150 else 1.6,
                color=colors[t],ls='--' if t==170 else '-',label=label,zorder=4 if t==150 else 3)
        if t==150 or y[-1]>0:
            offset=10 if t==150 else -15
            ax.annotate(str(int(y[-1])),(epochs[-1],float(y[-1])),xytext=(3,offset),
                        textcoords='offset points',ha='left',fontsize=10,fontweight='bold',color=colors[t])
    ax.axvline(55000*32/35000,color='#748597',ls=':',lw=1.2,zorder=1)
    ax.set(xlim=(-1,64),ylim=(-.6,highest+max(2,.17*highest)),xticks=np.arange(0,61,10),
           xlabel='Epoka = krok × 32 / 35 000',ylabel='Liczba unikalnych par')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=7));ax.tick_params(labelbottom=True,labelleft=True)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.068,.872),frameon=False,ncol=3,fontsize=10)
fig.text(.075,.105,'Kropki to rzeczywiste pomiary. Linie łączą zapisane checkpointy; nie oznaczają pomiaru w każdej epoce.',
         fontsize=10,color='#263D52')
fig.text(.075,.077,'Pionowa linia: najlepszy checkpoint 55k (50,29 epoki). Ostatni punkt: krok 65 640 (60,01 epoki).',
         fontsize=10,color='#263D52')
fig.text(.075,.043,'Para liczona raz w bloku, jeśli próg osiągnęła dla tego samego tokena na którejkolwiek z 50 klatek TEST. Seed3047.',
         fontsize=9,color='#68788A')
stem='PARY_PRZECIWNE_PRZEBIEG_EPOKI_BLOKI'
for ext in ['png','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=200)
plt.close(fig)
with (DATA/'PLOT_SERIES.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['block','step','epoch','pairs_ge150','pairs_ge160','pairs_ge170','pairs_ge175','pairs_ge179'])
    w.writeheader()
    for b in range(1,7):
        for s,e in zip(steps,epochs):w.writerow(dict(block=b,step=s,epoch=float(e),
            **{f'pairs_ge{t}':lookup[b,s,t]['same_token_observed_unique_pairs'] for t in [150,160,170,175,179]}))
meta=dict(status='PASS',steps=steps,epoch_axis=epochs.tolist(),count_source_sha256=hashlib.sha256((DATA/'COUNTS_BY_EPOCH.json').read_bytes()).hexdigest(),
          max_pairs_ge150_by_block={str(b):int(values(b,150).max()) for b in range(1,7)},no_pairs_ge170_at_any_saved_checkpoint=no170,
          interpolation='None: only stored checkpoints are measured; lines are visual connectors.',
          outputs={f'{stem}.{ext}':hashlib.sha256((OUT/f'{stem}.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
(DATA/'PLOT_QC.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False))
