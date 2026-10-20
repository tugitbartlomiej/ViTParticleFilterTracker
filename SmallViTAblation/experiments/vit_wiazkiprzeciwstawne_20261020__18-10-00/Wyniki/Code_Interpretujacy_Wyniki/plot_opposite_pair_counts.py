"""Plot the verified endpoint pair counts and write the requested short conclusion."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'Dane/opposite_pair_counts_20261020__18-52-50'
OUT = ROOT/'Wykresy'
SHORT = ROOT/'Krótki_Wniosek'
STEM = 'PARY_PRZECIWNE_0_55000_65640'
rows = json.loads((DATA/'COUNTS.json').read_text(encoding='utf-8'))
done = json.loads((DATA/'DONE.json').read_text(encoding='utf-8'))
for name,h in done['files'].items():
    assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h, name
assert len(rows)==108
steps=[0,55000,65640]
blocks=np.arange(1,7)
thresholds=[150,160,170,175,179]
colors=['#9AA8B5','#0072B2','#D55E00']
labels=['Krok 0','55 000: najlepszy','65 640: końcowy']
lookup={(r['step'],r['block'],r['threshold_deg']):r for r in rows}
def counts(step,threshold):
    return np.array([lookup[step,int(b),threshold]['same_token_observed_unique_pairs'] for b in blocks])
for s in steps:
    for b in blocks:
        for th in thresholds:
            r=lookup[s,int(b),th]
            assert r['all_unordered_pairs']==294528
            assert r['same_token_observed_unique_pairs']==r['possible_signed_unique_pairs']
totals={s:{t:int(counts(s,t).sum()) for t in thresholds} for s in steps}
assert totals[0][150]==0 and totals[55000][150]==25 and totals[65640][150]==24
assert all(totals[s][t]==0 for s in steps for t in [170,175,179])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(14.8,10.0))
fig.subplots_adjust(left=.075,right=.96,bottom=.155,top=.81,wspace=.28,hspace=.42)
fig.suptitle('Ile par wiązek ma niemal przeciwne kierunki?',x=.075,y=.967,ha='left',
             fontsize=21,fontweight='bold',color='#18314B')
fig.text(.075,.913,'6 bloków ViT | 294 528 par na blok | pełne 192D ze znakiem GELU | seed3047',
         fontsize=11,color='#44556A')
fig.text(.075,.875,'Dla progów 170°, 175° i 179°: ZERO par we wszystkich blokach i trzech checkpointach.',
         fontsize=12,color='#18314B',fontweight='bold')
for ax,threshold,title in [(axes[0,0],150,'A. Pary o kącie ≥150°'),(axes[0,1],160,'B. Pary o kącie ≥160°')]:
    for j,(step,color,label) in enumerate(zip(steps,colors,labels)):
        vals=counts(step,threshold)
        positions=blocks+(j-1)*.25
        ax.bar(positions,vals,width=.23,color=color,label=label,zorder=3)
        for x,y in zip(positions,vals):
            ax.annotate(str(int(y)),(x,float(y)),xytext=(0,5),textcoords='offset points',
                        ha='center',va='bottom',fontsize=9,color='#263D52')
    ax.set(title=title,xlabel='Blok',ylabel='Liczba unikalnych par',xticks=blocks)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0,13 if threshold==150 else 3.4)
    ax.set_axisbelow(True)
    if threshold==150:ax.legend(ncol=1,loc='upper left',fontsize=9)

ax=axes[1,0]
for step,color,label in zip(steps,colors,labels):
    mx=np.array([lookup[step,int(b),150]['maximum_observed_signed_angle'] for b in blocks])
    ax.plot(blocks,mx,'o-',color=color,label=label,lw=2,ms=5)
    for b,v in zip(blocks,mx):
        if step==65640:
            ax.annotate(f'{v:.1f}°'.replace('.',','),(b,v),xytext=(0,-16),
                        textcoords='offset points',ha='center',fontsize=8.5,color=color)
for y in [150,170,180]:ax.axhline(y,c='#8D9CAA',ls=':',lw=1)
ax.set(title='C. Największy kąt w każdym bloku',xlabel='Blok',ylabel='Kąt [°]',
       xticks=blocks,ylim=(98,184),yticks=[100,110,130,150,170,180])
ax.legend(loc='center right',bbox_to_anchor=(.98,.38),fontsize=8.5)
ax.text(.01,.03,'Oś Y powiększona: 98°-184°',transform=ax.transAxes,fontsize=8,color='#68788A')

ax=axes[1,1];ax.axis('off')
ax.set_title('D. Suma par z sześciu bloków',loc='left',pad=13)
table_rows=[[f'≥{t}°',str(totals[0][t]),str(totals[55000][t]),str(totals[65640][t])] for t in thresholds]
tab=ax.table(cellText=table_rows,colLabels=['Próg kąta','Krok 0','55 000','65 640'],
             cellLoc='center',colLoc='center',bbox=[0,.24,1,.70])
tab.auto_set_font_size(False);tab.set_fontsize(10.5)
for (ri,ci),cell in tab.get_celld().items():
    cell.set_edgecolor('#FFFFFF');cell.set_linewidth(2)
    if ri==0:cell.set_facecolor('#DFEAF3');cell.set_text_props(weight='bold',color='#18314B')
    else:cell.set_facecolor('#F0F4F8' if ri%2 else '#FFFFFF');cell.set_text_props(color='#263D52')
ax.text(0,.12,'Tylko blok 5 zawiera pary ≥160°: dwie w 55k i dwie na końcu.',
        transform=ax.transAxes,fontsize=10,color='#263D52',va='top',wrap=True)
ax.text(0,.035,'Największy kąt: 165,02° w 55k; 164,20° na końcu.',
        transform=ax.transAxes,fontsize=10,color='#263D52',va='top')

fig.text(.075,.092,'Para ID neuronów liczona raz, jeśli próg osiągnięto dla co najmniej jednego tego samego tokena w 50 klatkach TEST.',
         fontsize=9.3,color='#263D52')
fig.text(.075,.066,'Zera wkładów wykluczono (siła ≤1e-12). Zliczenia zgadzają się z globalną granicą wynikającą z wag i obu znaków aktywacji.',
         fontsize=9.3,color='#263D52')
fig.text(.075,.033,'55k = najlepszy checkpoint wybrany na VAL; 65 640 = rzeczywiście końcowy checkpoint. Sam kąt nie dowodzi znoszenia.',
         fontsize=9,color='#68788A')
OUT.mkdir(exist_ok=True);SHORT.mkdir(exist_ok=True)
for ext in ['png','svg']:
    fig.savefig(OUT/f'{STEM}.{ext}',dpi=200)
plt.close(fig)

conclusion='''LICZBA PAR WIĄZEK BLISKICH 180° - KRÓTKI WNIOSEK

Na początku treningu żaden z sześciu bloków nie ma pary o kącie co najmniej 150°.
W najlepszym checkpoincie 55 000 takich par jest łącznie 25, a w końcowym 65 640
jest ich 24. Rozkład na końcu: blok 1: 7, blok 2: 1, blok 3: 0, blok 4: 1,
blok 5: 6, blok 6: 9.

Próg 160° osiągają tylko dwie pary w bloku 5 - zarówno w najlepszym,
jak i końcowym checkpoincie. Największy kąt wynosi 165,02° w najlepszym
checkpoincie oraz 164,20° na końcu. Dla progów 170°, 175° i 179° liczba par
wynosi 0 w każdym bloku i każdym z trzech checkpointów.

Po treningu pojawiają się bardziej przeciwne wkłady, ale nie są one niemal
idealnie przeciwne. Brak par >=170° potwierdza także globalna granica kątów
obliczona z wag przy obu możliwych znakach aktywacji; nie jest to wyłącznie
brak ich zaobserwowania na 50 klatkach.

Kąty policzono w pełnych 192D ze znakiem GELU, bez projekcji 3D. Każdą parę
ID neuronów liczono raz w danym bloku i checkpoincie. Duży kąt sam w sobie
nie dowodzi znoszenia: potrzebne są także odpowiednie siły wkładów.

Dotyczy jednego treningu seed3047; geometria nie określa jeszcze wpływu
wyłączenia tych par na IoU ani score.
'''
conclusion_path=SHORT/'WNIOSEK_PARY_PRZECIWNE_0_55000_65640.txt'
conclusion_path.write_text(conclusion,encoding='utf-8')
with (DATA/'PLOT_COUNTS.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=['step','block','threshold_deg','unique_pairs'])
    writer.writeheader()
    for s in steps:
        for b in blocks:
            for th in thresholds:writer.writerow(dict(step=s,block=int(b),threshold_deg=th,unique_pairs=int(counts(s,th)[b-1])))
files=[OUT/f'{STEM}.png',OUT/f'{STEM}.svg',conclusion_path,DATA/'PLOT_COUNTS.csv']
(DATA/'PLOT_DELIVERY.json').write_text(json.dumps(dict(status='PASS',steps=steps,thresholds=thresholds,
    counts_source_sha256=hashlib.sha256((DATA/'COUNTS.json').read_bytes()).hexdigest(),
    totals=totals,files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),
    ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',figure=str(OUT/f'{STEM}.png'),conclusion=str(conclusion_path),totals=totals),ensure_ascii=False))
