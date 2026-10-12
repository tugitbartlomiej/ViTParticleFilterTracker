"""Wykres wyłącznie wcześniej odczytanych dowodów; bez treningu."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/Path(__file__).parent.name
audit=json.loads((OUT/'evidence/CPU_AUDIT.json').read_text(encoding='utf-8'))
with np.load(OUT/'evidence/NATURAL_GELU_BOUNDS.npz') as b: upper=b['post_upper']
with np.load(ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz') as a:
    before=a['before'].max(axis=0);after=a['after'].max(axis=0)
fig,axes=plt.subplots(2,1,figsize=(11,7),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
ax=axes[0];channels=np.arange(768)
ax.plot(channels,after,lw=.8,color='#b45309',label='Maksimum historycznego after po 600 tokenach')
ax.plot(channels,upper,lw=1.3,color='#be123c',label='Konieczna górna granica zamrożonego modelu')
ax.plot(channels,before,lw=.8,color='#0369a1',label='Maksimum naturalnego before po 600 tokenach')
ax.set(xlabel='Kanał MLP (indeks od 0)',ylabel='Aktywacja po GELU',title='Historyczny cel K1000: naruszenie koniecznych granic naturalnego forwardu')
ax.legend(loc='upper left',frameon=False,fontsize=9);ax.grid(alpha=.15)
null=audit['historical_latent']['delta_null_energy_fraction']
ax=axes[1]
ax.barh([0],[1-null],color='#0369a1',label='Rowspace fc2')
ax.barh([0],[null],left=[1-null],color='#64748b',label='Kernel fc2')
ax.text((1-null)/2,0,f'{100*(1-null):.3f}%',ha='center',va='center',color='white')
ax.text(1-null+null/2,0,f'{100*null:.3f}%',ha='center',va='center',color='white')
ax.set(xlim=(0,1),yticks=[],xlabel='Udział energii historycznej delty',title='Składowa kernelu jest niewidoczna dla frozen tail przy stałym residual')
ax.legend(loc='lower right',frameon=False,fontsize=9)
fig.savefig(OUT/'AUDYT_OSIAGALNOSCI.png',dpi=190)
fig.savefig(OUT/'AUDYT_OSIAGALNOSCI.pdf')
print('Zapisano wykres PNG/PDF audytu.')
