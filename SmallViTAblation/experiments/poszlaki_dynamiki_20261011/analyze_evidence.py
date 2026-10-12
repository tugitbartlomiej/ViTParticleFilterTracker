"""Independent, read-only reanalysis of saved summaries; no training or raw trajectory recomputation.
Run with Python 3 + numpy/matplotlib. Outputs are kept separate from historical results.
"""
from pathlib import Path
import json, hashlib, csv
import numpy as np

ROOT = Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation')
SRC = ROOT/'outputs/porownanie_szklo_20261010/eden/wyniki/pokretla'
OUT = ROOT/'outputs/poszlaki_dynamiki_20261011'
OUT.mkdir(parents=True, exist_ok=True)

def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x): return None
    if isinstance(x, np.generic): return x.item()
    return x

def save(name, x):
    (OUT/name).write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

data={f.stem.removeprefix('ANALIZA_'):json.loads(f.read_text(encoding='utf-8')) for f in sorted(SRC.glob('ANALIZA_*.json'))}
# Legacy records store dimension/window only under S1. PCA32 is the legacy observer's format.
for a in data.values():
    a.setdefault('okno',a['S1']['okno'])
    a.setdefault('wymiar',a['S1']['wymiar'])
    a.setdefault('rzut','pca32' if a['wymiar']==32 else 'unknown')
inventory=[]
for name,a in data.items():
    m=a.get('metadata',{}); p=m.get('pokretla',{}); s=a['S1']; h=a['heterogenicznosc_CR']
    scores=[v for v in a.get('oceny',[]) if v['krok']<=a['okno'][1]]
    row=dict(analysis=name,run=a['bieg'],window=a['okno'],projection=a['rzut'],dimension=a['wymiar'],
             seed=p.get('ziarno'),lr=m.get('lr'),optimizer=p.get('opt'),model=p.get('model'),
             tokens_per_image=a.get('tokenow',600),images=a['sond'],layer=a.get('warstwa'),
             task='classification' if a['bieg'].startswith('klas_') else 'detection',
             d_shell=s['d_shell'],lindemann=s['lindemann_shell'],tau_1e=s['tau_1e'],
             msd_lag1=s['msd'][s['taus'].index(1)],msd_lag10=s['msd'][s['taus'].index(10)],
             C1_lag1=h['C'][h['taus'].index(1)][0],C1_lag10=h['C'][h['taus'].index(10)][0],
             cross_lag10=h['C_cross'][h['taus'].index(10)],
             relative_mobility_rank=a.get('trwalosc_ruchliwosci_CR',{}).get('pierwsze_ostatnie_wzgledne'),
             last_score_at_or_before_window_end=scores[-1] if scores else None)
    inventory.append(row)
save('inventory.json', inventory)

pairs={}
for run,late in [('dlugi_lr_5e-4_rp','12034_13128'),('klas_dlugi_lr_5e-4','10746_11840')]:
    aa=[data[run],data[run+'_'+late]]
    pair={'run':run,'windows':[a['okno'] for a in aa], 'same_run':True,
          'warning':'Different window lengths and starting kNN graphs; late/early ratios are descriptive.',
          'scores':[next(r for r in inventory if r['analysis']==n)['last_score_at_or_before_window_end'] for n in (run,run+'_'+late)]}
    pair['S1']={k:[a['S1'][k] for a in aa] for k in ['d_shell','l_klatka','lindemann_shell','tau_1e','alfa_plateau']}
    for tau in (1,10,105,640,1073):
        pair[f'S1_MSD_lag{tau}']=[a['S1']['msd'][a['S1']['taus'].index(tau)] for a in aa]
        pair[f'S0_MSD_lag{tau}']=[a['S0']['msd'][a['S0']['taus'].index(tau)] for a in aa]
    pair['full768_step_energy_subsample_mean']=[float(np.mean(a['energia_768']['E'])) for a in aa]
    pair['early_late_full768_step_energy_in_each_window']=[[a['energia_768']['E_300_400'],a['energia_768']['E_2000_2188']] for a in aa]
    pairs[run]=pair
save('same_run_comparisons.json',pairs)

decomp=[]
for name,a in data.items():
    rv=a.get('rozklad_wariancji')
    if rv is None: continue
    for i,tau in enumerate(rv['taus']):
        shares={k:v[i] for k,v in rv['udzial'].items()}
        decomp.append(dict(analysis=name,tau=tau,**shares,sum=sum(shares.values()),
                           missing_cross_term=1-sum(shares.values())))
save('nonorthogonal_energy_audit.json',decomp)

kernels={}
for name in ['lr_1.25e-4_rp_stan_e02','lr_5e-4_rp_stan_e02','dlugi_lr_5e-4_rp_stan_e12']:
    a=json.loads((SRC/f'JADRO_{name}.json').read_text(encoding='utf-8'))
    kernels[name]=dict(step=a['krok'],measured_window=a['okno_zmierzone'],
                      retrospective=True,measured=a['zmierzone'],families={
        k:{kk:vv for kk,vv in v.items() if kk!='E_768_na_token'} for k,v in a['rodziny'].items()},
        mobility_spearman=a['mapa_ruchliwosci_spearman'],
        BN_energy_ratio=a['rodziny']['bn']['E_768']/a['rodziny']['paczka']['E_768'])
save('kernel_audit.json',kernels)

# Sources are hashed so the report remains tied to the exact snapshot reviewed.
files=list(SRC.glob('ANALIZA_*.json'))+[SRC/f'JADRO_{k}.json' for k in kernels]
save('sources.json',[dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(run,pair) in zip(axes[0],pairs.items()):
    for n,label,color in [(run,'wczesne','#297db0'),(run+'_'+('_'.join(map(str,pair['windows'][1]))),'późne','#d56a28')]:
        a=data[n];s=a['S1'];mask=np.asarray(s['taus'])<=1073
        ax.loglog(np.asarray(s['taus'])[mask],np.asarray(s['msd'])[mask],label=f"{label}: {a['okno'][0]}–{a['okno'][1]}",color=color)
    ax.set(title='Detekcja' if 'klas' not in run else 'Klasyfikacja Imagenette',xlabel='Odstęp w krokach treningu',ylabel='MSD względem początkowych sąsiadów (RP128)')
    ax.legend();ax.grid(alpha=.15,which='both')
ax=axes[1,0]
for label,pred,col in [('ViT AdamW',lambda r:r['analysis'].startswith('lr_') and r['projection']=='rp128','#297db0'),
                       ('CNN AdamW',lambda r:r['analysis'].startswith('cnn_'),'#d56a28'),
                       ('ViT SGD',lambda r:r['analysis'].startswith('sgd_'),'#699346')]:
    rows=[r for r in inventory if pred(r)]
    ax.scatter([r['lr'] for r in rows],[r['C1_lag10'] for r in rows],label=label,c=col,s=38)
ax.set(xscale='log',xlabel='Learning rate (różne optymalizatory: różne skale)',ylabel='Korelacja amplitud lokalnego ruchu, lag 10',title='Efekt występuje w kilku warunkach');ax.legend();ax.grid(alpha=.15)
ax=axes[1,1]
x=np.arange(3);labels=['LR 1.25e-4 / ep. 2','LR 5e-4 / ep. 2','LR 5e-4 / ep. 12']
for j,(k,label,col) in enumerate([('szum','Reszta po odpowiedzi na średni gradient','#297db0'),('los_izo','Izotropowe zmiany wag','#d56a28')]):
    vals=[a['mobility_spearman'][k]['wzgledna'] for a in kernels.values()]
    ax.bar(x+(j-.5)*.32,vals,.32,label=label,color=col)
ax.set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel='Spearman map ruchliwości',title='Dopasowanie do poprzednich 200 kroków')
ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
fig.suptitle('Nowe poszlaki — niezależny przegląd zapisanych wyników, 11.10.2026',fontsize=15)
fig.savefig(OUT/'poszlaki.png',dpi=180)
fig.savefig(OUT/'poszlaki.pdf')
print(json.dumps(clean({'analyses':len(data),'unique_run_names':len(set(a['bieg'] for a in data.values())),
                       'pairs':pairs,'kernel_spearman':{k:v['mobility_spearman'] for k,v in kernels.items()},
                       'BN_energy_ratio':{k:v['BN_energy_ratio'] for k,v in kernels.items()},
                       'share_sum_range':[min(r['sum'] for r in decomp),max(r['sum'] for r in decomp)]}),ensure_ascii=False,indent=2))
