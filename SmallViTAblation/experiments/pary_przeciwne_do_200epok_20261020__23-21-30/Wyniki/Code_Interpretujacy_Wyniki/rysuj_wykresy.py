"""Figures for opposite beam pairs (epochs 0-200), pair 370-391 and detection. PNG + SVG + PDF in Wyniki/Wykresy."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP=Path(__file__).resolve().parents[2]
DANE=EXP/'Wyniki/Dane';OUT=EXP/'Wyniki/Wykresy';OUT.mkdir(exist_ok=True)
# Reference categorical palette (validated, light mode), chart chrome and ink.
S1,S2,S3='#2a78d6','#eb6834','#1baf7a'
SURF,INK,INK2,MUTED,GRID,AXIS='#fcfcfb','#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
plt.rcParams.update({'font.family':['Segoe UI','DejaVu Sans'],'font.size':9,'axes.facecolor':SURF,'figure.facecolor':SURF,
    'axes.edgecolor':AXIS,'axes.labelcolor':INK2,'xtick.color':MUTED,'ytick.color':MUTED,'axes.titlecolor':INK,
    'axes.grid':True,'grid.color':GRID,'grid.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titlesize':10,'axes.titleweight':'bold','legend.frameon':False,'legend.fontsize':8,'lines.linewidth':1.6,
    'svg.fonttype':'none','pdf.fonttype':42})
EP_SRC=65640*32/35000;EP_BEST=55000*32/35000
made=[]
def rows(path):return list(csv.DictReader(open(path,encoding='utf-8')))
def f(x):return float(x) if x not in ('',None) else np.nan
def mark_epochs(ax,label=True):
    for e,t in [(EP_BEST,'najlepszy VAL (55 000)'),(EP_SRC,'koniec treningu źródłowego')]:
        ax.axvline(e,color=AXIS,lw=.8,zorder=0)
    if label:ax.text(EP_SRC+2,ax.get_ylim()[1],'start kontynuacji →',color=MUTED,fontsize=7,va='top')
def finish(fig,name,title,note):
    fig.suptitle(title,x=.01,ha='left',fontsize=11,fontweight='bold',color=INK)
    fig.text(.01,.005,note,fontsize=7,color=MUTED,ha='left',va='bottom')
    fig.tight_layout(rect=(0,.04,1,.94))
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=200)
    plt.close(fig);made.append(name)

def counts():
    s=rows(DANE/'polaczone_0_200/SUMA_BLOKOW_0_200.csv');c=rows(DANE/'polaczone_0_200/COUNTS_0_200.csv')
    ep=sorted({f(r['epoch_equivalent']) for r in s})
    def series(th,key):return [f(next(r for r in s if f(r['epoch_equivalent'])==e and f(r['threshold_deg'])==th)[key]) for e in ep]
    # 01: all blocks, three thresholds.
    fig,ax=plt.subplots(figsize=(8,4.2))
    ends={}
    for th,col in [(150,S1),(160,S2),(170,S3)]:
        y=series(th,'unique_pairs_all_blocks');ax.plot(ep,y,color=col,marker='o',ms=3.5,label=f'≥{th}°')
        ends.setdefault(int(y[-1]),[]).append(f'≥{th}°')
    for v,labels in ends.items():
        ax.annotate(f"{' i '.join(labels)}: {v}",(ep[-1],v),xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color=INK2)
    ax.set_ylim(0,None);mark_epochs(ax);ax.set_xlabel('epoka (krok × 32 / 35 000)');ax.set_ylabel('unikalne pary neuronów, 6 bloków')
    ax.legend(loc='upper right',title='kąt wkładów na tym samym tokenie',title_fontsize=8);ax.set_xlim(-3,228)
    finish(fig,'01_pary_przeciwne_0_200_progi','Pary przeciwnych wiązek MLP w treningu do epoki 200',
        'Seed 3047, 50 klatek TEST, wszystkie 294 528 par w każdym bloku. Progu ≥175° nie osiąga żadna para. Epoki 0–60: rodzic, 70–200: kontynuacja.')
    # 02: per block, >=150 and >=160.
    fig,axs=plt.subplots(2,3,figsize=(9,5),sharex=True,sharey=True)
    for b,ax in zip(range(1,7),axs.flat):
        for th,col in [(150,S1),(160,S2)]:
            y=[f(next(r for r in c if f(r['epoch_equivalent'])==e and int(r['block'])==b and f(r['threshold_deg'])==th)['same_token_observed_unique_pairs']) for e in ep]
            ax.plot(ep,y,color=col,marker='o',ms=2.5,label=f'≥{th}°')
        ax.set_title(f'blok {b}',loc='left');mark_epochs(ax,False)
    axs[0,0].legend(loc='upper left');[a.set_xlabel('epoka') for a in axs[1]];[a.set_ylabel('pary') for a in axs[:,0]]
    finish(fig,'02_pary_przeciwne_bloki','Pary przeciwnych wiązek w każdym bloku',
        'Pionowe linie: najlepszy checkpoint VAL (epoka 50) i koniec treningu źródłowego (epoka 60).')
    # 03: maximum observed signed angle per block.
    fig,axs=plt.subplots(2,3,figsize=(9,5),sharex=True,sharey=True)
    for b,ax in zip(range(1,7),axs.flat):
        y=[f(next(r for r in c if f(r['epoch_equivalent'])==e and int(r['block'])==b and f(r['threshold_deg'])==150)['maximum_observed_signed_angle']) for e in ep]
        ax.plot(ep,y,color=S1,marker='o',ms=2.5);ax.axhline(170,color=AXIS,lw=.8,zorder=0)
        ax.annotate(f'{y[-1]:.1f}°',(ep[-1],y[-1]),xytext=(-4,6),textcoords='offset points',ha='right',fontsize=8,color=INK2)
        ax.set_title(f'blok {b}',loc='left');mark_epochs(ax,False);ax.set_ylim(100,180)
    [a.set_xlabel('epoka') for a in axs[1]];[a.set_ylabel('największy kąt [°]') for a in axs[:,0]]
    finish(fig,'03_maksymalny_kat_bloki','Największy kąt pary wkładów na tym samym tokenie',
        'Linia pozioma: 170°. W bloku 6 kąt rośnie od epoki 150 do 173,0° w epoce 200 (para 370–391).')
    # 04: pair-token occurrences, all vs both active (separate panels, no shared scale).
    fig,axs=plt.subplots(1,2,figsize=(9,3.6))
    for ax,key,title in [(axs[0],'pair_token_count_all_blocks','wszystkie wystąpienia para–token ≥150°'),
                         (axs[1],None,'obie wiązki aktywne (≥5% maksimum tokena)')]:
        if key:y=series(150,key)
        else:y=[sum(f(r['same_token_active05_pair_token_count']) for r in c if f(r['epoch_equivalent'])==e and f(r['threshold_deg'])==150) for e in ep]
        ax.plot(ep,y,color=S1,marker='o',ms=3);ax.set_title(title,loc='left');ax.set_ylim(0,None);mark_epochs(ax,False);ax.set_xlabel('epoka')
    axs[0].set_ylabel('liczba par–tokenów (6 bloków)')
    finish(fig,'04_wystapienia_par_tokeny','Jak często przeciwne pary występują na tokenach',
        '30 000 tokenów z 50 klatek TEST. Para liczona osobno na każdym tokenie, na którym osiąga ≥150°.')

def pair():
    w=rows(DANE/'znoszenie_370_391/PARA_WAGI_0_200.csv');t=rows(DANE/'znoszenie_370_391/PARA_TOKENY_0_200.csv')
    ep=[f(r['epoch_equivalent']) for r in w]
    fig,axs=plt.subplots(2,2,figsize=(9,6.2),sharex=True)
    ax=axs[0,0];ax.plot(ep,[f(r['w2_angle_deg']) for r in w],color=S1,marker='o',ms=3,label='kąt wag wyjściowych W2')
    ax.plot(ep,[180-f(r['w2_angle_deg']) for r in w],color=S2,marker='o',ms=3,label='kąt wkładów przy przeciwnych znakach GELU')
    ax.set_title('kąty pary [°]',loc='left');ax.legend(loc='lower left',fontsize=7);ax.set_ylim(0,180)
    ax=axs[0,1];ax.plot(ep,[f(r['w1_cos_ln_folded']) for r in w],color=S1,marker='o',ms=3)
    ax.set_title('cos wag wejściowych W1 (po złożeniu z LN)',loc='left');ax.set_ylim(-.1,1.02)
    get=lambda cat,key:[f(next(r for r in t if f(r['epoch_equivalent'])==e and r['category']==cat)[key]) for e in ep]
    ax=axs[1,0]
    for cat,col,lab in [('same_sign',S1,'te same znaki (przy W2 bliskich: zgodne)'),('opposite_sign',S2,'przeciwne znaki (przy W2 bliskich: przeciwne)')]:
        ax.plot(ep,get(cat,'tokens'),color=col,marker='o',ms=3,label=lab)
    ax.set_yscale('log');ax.set_ylim(50,3e5);ax.set_title('tokeny, na których obie wiązki są niezerowe',loc='left');ax.legend(loc='upper left',fontsize=7);ax.set_xlabel('epoka')
    ax=axs[1,1]
    for cat,col,lab in [('both_nonzero',S1,'wszystkie tokeny'),('opposite_sign',S2,'tokeny przeciwne'),('opposite_sign_both_active05',S3,'przeciwne, obie aktywne ≥5%')]:
        ax.plot(ep,get(cat,'aggregate_kappa'),color=col,marker='o',ms=3,label=lab)
    ax.axhline(1,color=AXIS,lw=.8,zorder=0);ax.set_ylim(0,2.05);ax.set_xlabel('epoka')
    ax.set_title('κ = ‖u₁+u₂‖² / (‖u₁‖²+‖u₂‖²)',loc='left');ax.legend(loc='lower left',fontsize=7)
    for a in axs.flat:mark_epochs(a,False)
    finish(fig,'05_para_370_391_historia','Para 370–391 w bloku 6 od epoki 0 do 200',
        'κ: 2 = wkłady równe i zgodne, 1 = niezależne, 0 = pełne znoszenie. 26 checkpointów, 50 klatek TEST.')

def tokens200():
    w={int(r['step']):r for r in rows(DANE/'znoszenie_370_391/PARA_WAGI_0_200.csv')}[218750]
    d=np.load(DANE/'znoszenie_370_391/tokens_k218750.npz')
    za,zb,smax,yn=[d[k].astype(np.float64) for k in ('za','zb','smax','ynorm')]
    na,nb,ca=f(w['w2_norm_a']),f(w['w2_norm_b']),np.cos(np.radians(f(w['w2_angle_deg'])))
    sa,sb=np.abs(za)*na,np.abs(zb)*nb;nz=(sa>1e-12)&(sb>1e-12);opp=nz&(np.sign(za)!=np.sign(zb));same=nz&~opp
    with np.errstate(divide='ignore',invalid='ignore'):
        pn=np.sqrt(np.maximum(sa**2+sb**2+2*za*zb*na*nb*ca,0));ratio=np.minimum(sa,sb)/np.maximum(sa,sb);share=pn/yn;left=pn/(sa+sb)
    act=(sa>.05*smax)&(sb>.05*smax)
    fig,axs=plt.subplots(1,3,figsize=(10,3.6))
    bins=np.linspace(0,1,41)
    for ax,vals,title,xl in [(axs[0],ratio,'stosunek sił słabszej do silniejszej','min(s₁,s₂) / max(s₁,s₂)'),
                             (axs[1],left,'co zostaje z sumy sił','‖u₁+u₂‖ / (s₁+s₂)')]:
        ax.hist(vals[same],bins=bins,color=S1,alpha=.85,label=f'te same znaki ({same.sum()} tokenów)',density=True)
        ax.hist(vals[opp],bins=bins,color=S2,alpha=.85,label=f'przeciwne znaki ({opp.sum()})',density=True)
        ax.set_title(title,loc='left');ax.set_xlabel(xl);ax.set_ylabel('gęstość')
    axs[0].legend(loc='upper left',fontsize=7);axs[1].set_yscale('log')
    ax=axs[2];sb_=np.logspace(-6,0,49)
    ax.hist(share[same],bins=sb_,color=S1,alpha=.85,density=True);ax.hist(share[opp],bins=sb_,color=S2,alpha=.85,density=True)
    ax.set_xscale('log');ax.set_title('udział pary w wyjściu MLP',loc='left');ax.set_xlabel('‖u₁+u₂‖ / ‖wyjście MLP bloku 6‖')
    finish(fig,'06_para_370_391_tokeny_epoka200','Para 370–391 na tokenach w epoce 200',
        f'Przeciwne i obie aktywne ≥5%: {int((opp&act).sum())} tokenów. Wartość 0 w środkowym panelu oznacza pełne znoszenie.')

def ablation():
    k=rows(DANE/'znoszenie_370_391/KONTROLA_k218750.csv')
    groups=[('uniform','losowe'),('strength_matched','o podobnej sile'),('near_parallel_w2','prawie równoległe W2')]
    fig,axs=plt.subplots(1,3,figsize=(10,3.8))
    rng=np.random.default_rng(0)
    for ax,key,title,scale in [(axs[0],'interaction_ratio_logit','efekt pary / większy efekt pojedynczy',1),
                               (axs[1],'VAL_dIoU_ab','ΔIoU na VAL po wyłączeniu pary [×10⁻³]',1e3),
                               (axs[2],'TEST_dIoU_ab','ΔIoU na TEST po wyłączeniu pary [×10⁻³]',1e3)]:
        for i,(g,lab) in enumerate(groups):
            v=np.array([f(r[key]) for r in k if r['group']==g])*scale
            ax.scatter(i+rng.uniform(-.18,.18,len(v)),v,s=10,color=MUTED,alpha=.6,linewidths=0)
            ax.hlines(np.median(v),i-.28,i+.28,color=INK2,lw=1.2)
        tv=f(next(r for r in k if r['group']=='target')[key])*scale
        ax.scatter([3],[tv],s=60,color=S2,edgecolors=SURF,linewidths=1.5,zorder=3)
        ax.annotate(f'{tv:.3g}',(3,tv),xytext=(8,0),textcoords='offset points',va='center',fontsize=8,color=INK2)
        ax.set_xticks(range(4),[l for _,l in groups]+['370–391'],rotation=20,ha='right');ax.set_title(title,loc='left',fontsize=9)
        if key=='interaction_ratio_logit':ax.axhline(1,color=AXIS,lw=.8,zorder=0)
        else:ax.axhline(0,color=AXIS,lw=.8,zorder=0)
    finish(fig,'07_wylaczenie_pary_kontrola','Wyłączenie pary 370–391 na tle par kontrolnych (epoka 200)',
        'Po 100 par w każdej grupie kontrolnej, kreska = mediana. Iloraz <1 oznaczałby wzajemne znoszenie, >1 sumowanie się efektów. VAL i TEST po 500 klatek.')

def detection():
    d=rows(DANE/'detekcja_0_200/DETEKCJA_0_200.csv');bs=DANE/'detekcja_0_200/DETEKCJA_BOOTSTRAP.csv'
    boot=rows(bs) if bs.exists() else []
    panels=[('mean_top1_iou','średnie IoU zwycięskiego tokenu'),('mean_score_positive','średnia pewność (klatki z narzędziem)'),
            ('empty_mean_top1_score','średnia pewność na klatkach bez narzędzia')]
    for name,title,from_epoch in [('08_iou_i_pewnosc_0_200','IoU i pewność w treningu do epoki 200',0),
                                  ('09_iou_i_pewnosc_od_40_epoki','IoU i pewność od epoki 40 (przybliżenie)',40)]:
        fig,axs=plt.subplots(1,3,figsize=(10.5,3.8),sharex=True)
        for ax,(key,ptitle) in zip(axs,panels):
            last={}
            for split,col in [('VAL',S1),('TEST',S2)]:
                rr=[r for r in d if r['split']==split and f(r['epoch_equivalent'])>=from_epoch];ep=[f(r['epoch_equivalent']) for r in rr];y=[f(r[key]) for r in rr]
                ax.plot(ep,y,color=col,marker='o',ms=3,label=split);last[split]=(ep[-1],y[-1])
                b=[r for r in boot if r['split']==split and r['metric']==key and f(r['epoch_equivalent'])>=from_epoch]
                if b:ax.fill_between([f(r['epoch_equivalent']) for r in b],[f(r['lo']) for r in b],[f(r['hi']) for r in b],color=col,alpha=.15,linewidth=0)
            # Keep the two end labels apart: the higher one goes up, the lower one down.
            hi,lo=sorted(last,key=lambda s:-last[s][1])
            for s,dy in [(hi,6),(lo,-6)]:ax.annotate(f'{s} {last[s][1]:.3f}',last[s],xytext=(4,dy),textcoords='offset points',va='center',fontsize=7,color=INK2)
            ax.set_title(ptitle,loc='left',fontsize=9);ax.set_xlabel('epoka');mark_epochs(ax,False)
            if from_epoch==0:ax.set_ylim(0,1)
            ax.set_xlim(None,ep[-1]+30)
        axs[0].legend(loc='lower right' if from_epoch==0 else 'lower left')
        finish(fig,name,title,
            'VAL i TEST po 500 klatek (466/472 z narzędziem). Pasma: 95% bootstrap po filmach (2000 powtórzeń). Pewność = sigmoid logitu zwycięskiego tokenu.')

if __name__=='__main__':
    counts()
    if (DANE/'znoszenie_370_391/DONE.json').exists():pair();tokens200();ablation()
    if (DANE/'detekcja_0_200/DONE.json').exists():detection()
    index={n:{ext:hashlib.sha256((OUT/f'{n}.{ext}').read_bytes()).hexdigest() for ext in ('png','svg','pdf')} for n in made}
    (OUT/'INDEX_WYKRESOW.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=index),indent=2),encoding='utf-8')
    print('MADE',made)
