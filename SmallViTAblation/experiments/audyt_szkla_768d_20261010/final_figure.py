from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import OUT
def rd(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
mv=rd('DENSE_MULTISCALE_VACF.json');de=rd('dense_metrics.json');lo=rd('long_metrics.json');dc=rd('DIRECT_CONTROLS.json');ep=rd('EPOCH_SCALE_BN_CONTROLS.json')
plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
fig,ax=plt.subplots(2,2,figsize=(13,9))
for name,label,color in [('raw','Ruch surowy','#0072B2'),('cage','Względem sąsiadów','#D55E00')]:
    dt=np.array([int(k) for k in mv[name]]);v=np.array([a['mean'][1] for a in mv[name].values()])
    ax[0,0].semilogx(dt,v,'o-',label=label,color=color)
ax[0,0].axhline(0,color='gray',lw=1);ax[0,0].set(title='A. Podtrzymanie kierunku, potem zawracanie',xlabel='Długość sąsiednich przyrostów [kroki]',ylabel='Korelacja przyrostów')
w=de['cage']['windows']['early'];ax[0,1].loglog(w['lag'],w['mean'],color='#D55E00',label='Pomiar w 768D')
fit=w['fits']['colored_OU_plus_diffusion'];ax[0,1].loglog(w['lag'],fit['prediction'],'--',color='#333333',label='OU z pamięcią + dyfuzja')
ax[0,1].set(title='B. Podobną krzywą daje model bez klatek sąsiedzkich',xlabel='Opóźnienie [kroki]',ylabel='MSD względem sąsiadów')
for mode,label,color in [('dense','Gęsty bieg: 21 epok','#0072B2'),('long','Osobny bieg: 60 epok','#D55E00')]:
    a=[r for r in dc[mode]['dynamic_cage'] if 1000 in r['lag']]
    x=np.array([r['age'] for r in a]);y=np.array([r['mean'][r['lag'].index(1000)] for r in a])
    ax[1,0].semilogx(x,y,'o-',label=label,color=color)
ax[1,0].set(title='C. Stałe opóźnienie: 1000 kroków',xlabel='Wiek modelu [kroki]',ylabel='MSD względem odświeżanych sąsiadów')
for name,label,color in [('fp32_unit','FP32, standardowe BatchNorm','#0072B2'),('frozen_bn_unit','FP32, BatchNorm zamrożone w kroku 300','#009E73')]:
    a=[r for r in ep[name] if 5 in r['lag_epochs']]
    x=[r['age_epoch'] for r in a];y=[r['mean'][r['lag_epochs'].index(5)] for r in a]
    ax[1,1].plot(x,y,'o-',label=label,color=color)
ax[1,1].set(title='D. Kontrola skali: kierunki tokenów',xlabel='Wiek modelu [epoki]',ylabel='MSD tokenów długości 1 po 5 epokach')
for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=8)
fig.suptitle('Czy ruch tokenów przypomina szkło? Audyt pełnych 768 wymiarów',fontsize=16)
fig.text(.5,.015,'8 zdjęć, 4800 tokenów; rozgrzewka kończy się w kroku 300. Sąsiedzi to podobieństwo w latencie. Biegi analizowane oddzielnie.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(OUT/'wniosek_768d.png',dpi=180);fig.savefig(OUT/'wniosek_768d.svg');plt.close(fig)
print('Final figure written',flush=True)
