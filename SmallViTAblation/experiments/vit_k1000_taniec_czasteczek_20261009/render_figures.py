"""Czy wykresy statyczne zachowują wspólne osie i rzeczywisty timeline?
Brak danych pośrednich nie uprawnia do interpolowania ruchu.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'#f7f9fb',
                     'axes.facecolor':'#f7f9fb','axes.labelcolor':'#213649','text.color':'#213649',
                     'xtick.color':'#456075','ytick.color':'#456075','savefig.facecolor':'#f7f9fb'})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def main():
    with np.load(OUT/'data/projection.npz',allow_pickle=False) as f:
        a={k:f[k] for k in f.files}
    meta=json.loads((OUT/'data/metadata.json').read_text(encoding='utf-8'))
    rec=meta.get('images',meta.get('records',[]))
    variance=meta.get('explained_variance_ratio',meta.get('fit',meta.get('pca',{})).get('explained_variance_ratio',[]))
    flat=a['positions'].transpose(0,2,1,3).reshape(900000,3,3)
    ids=np.random.default_rng(20261009).choice(900000,12000,replace=False)
    p=flat[ids]
    lo=p[:,:,:2].min(axis=(0,1));hi=p[:,:,:2].max(axis=(0,1))
    pad=np.maximum(.04*(hi-lo),.03)
    fig,axs=plt.subplots(1,3,figsize=(15,5.6),sharex=True,sharey=True)
    for k,ax in enumerate(axs):
        ax.scatter(p[:,k,0],p[:,k,1],s=2,alpha=.35,color=['#456f8d','#178d9b','#e29438'][k],rasterized=True)
        ax.set_title(f'Krok {[0,100,500][k]}',fontweight='bold')
        ax.set_xlabel('PC1 · stała wspólna oś');ax.grid(alpha=.15)
        ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
    axs[0].set_ylabel('PC2 · stała wspólna oś')
    fig.suptitle('900 000 cząsteczek · trzy rzeczywiście zapisane stany',fontsize=18,fontweight='bold',y=.99)
    text='Te same 12 000 cząsteczek w każdym panelu; to stała próbka ze wszystkich 1500 obrazów. Brak interpolacji.'
    if len(variance)>=2:text+=f' PC1 + PC2: {100*sum(variance[:2]):.2f}% wariancji próbki dopasowania PCA.'
    fig.text(.05,.035,text,fontsize=9,wrap=True)
    fig.subplots_adjust(top=.82,bottom=.23,wspace=.15)
    save(fig,'01_cala_chmura_przed_100_500')

    panel=int(np.where(a['control_images']==100)[0][0]) if 100 in a['control_images'] else 0
    image=int(a['control_images'][panel]); tokens=a['control_tokens'][panel].astype(int)
    z=a['control_positions'][panel];dz=z-z[:1];norm=a['control_norm'][panel]
    colors=plt.cm.tab10(np.arange(8))
    fig=plt.figure(figsize=(15,9))
    ax=fig.add_subplot(2,2,1,projection='3d')
    for j,t in enumerate(tokens):
        ax.plot(dz[:,j,0],dz[:,j,1],dz[:,j,2],lw=1.4,color=colors[j],label=f'Patch {t}')
        ax.scatter(*dz[0,j],color=colors[j],s=18,marker='o')
        ax.scatter(*dz[-1,j],color=colors[j],s=28,marker='^')
    ax.set_title('Rzeczywiste trajektorie 8 cząsteczek')
    ax.set_xlabel('Δ PC1');ax.set_ylabel('Δ PC2');ax.set_zlabel('Δ PC3')
    ax.view_init(elev=23,azim=39)
    ax.legend(fontsize=7,ncol=2,loc='upper left',bbox_to_anchor=(-.08,1.05))
    ax=fig.add_subplot(2,2,2)
    steps=np.arange(501)
    for j,t in enumerate(tokens):ax.plot(steps,norm[:,j],color=colors[j],lw=1.4,label=f'Patch {t}')
    ax.set(title='Długość przesunięcia w pełnych 768D',xlabel='Krok optymalizacji',ylabel='Norma L2 względem startu');ax.grid(alpha=.2)
    ax=fig.add_subplot(2,2,3)
    ax.plot(steps,a['confidence'][image],color='#168694',label='Pewność')
    ax.plot(steps,a['iou'][image],color='#df9436',label='IoU')
    ax.axhline(.9,color='#718596',ls='--',lw=1,label='Próg 90%')
    ax.set(title='Detekcja przy podmianie aktywacji',xlabel='Krok optymalizacji',ylabel='Wartość',ylim=(0,1.02));ax.legend();ax.grid(alpha=.2)
    ax=fig.add_subplot(2,2,4)
    h=a['movement_norm'][image,2].reshape(20,30)
    im=ax.imshow(h,cmap='cividis',interpolation='nearest',aspect='auto')
    ax.set(title='Wszystkie 600 patchy · ruch w kroku 500',xlabel='Kolumna patcha',ylabel='Wiersz patcha')
    fig.colorbar(im,ax=ax,label='Norma L2 w 768D',shrink=.82)
    frame_id=rec[image].get('frame_id',str(image)) if rec else str(image)
    parts=frame_id.split('/')
    ident=f'{parts[-2]} · klatka {int(parts[-1])}' if len(parts)>1 and parts[-1].isdigit() else frame_id
    fig.suptitle(f'Taniec cząsteczek · obraz {image:04d} · {ident}',fontsize=18,fontweight='bold')
    fig.text(.06,.025,'Każdy punkt trajektorii pochodzi z zapisu. Wagi K1000 pozostawały zamrożone; optymalizowano aktywacje tego obrazu.\nKoło: krok 0; trójkąt: krok 500. Rzut 3D nie zachowuje całej geometrii 768D.',fontsize=9)
    fig.subplots_adjust(top=.89,bottom=.12,hspace=.34,wspace=.32)
    save(fig,'02_rzeczywiste_trajektorie_panel_kontrolny')

    fig,axs=plt.subplots(1,3,figsize=(15,5.5))
    steps=np.arange(501)
    axs[0].plot(steps,a['confidence'].mean(axis=0),color='#168694',label='Średnia pewność')
    axs[0].plot(steps,a['iou'].mean(axis=0),color='#df9436',label='Średnie IoU')
    axs[0].axhline(.9,color='#718596',ls='--',lw=1)
    axs[0].set(title='Średnia ze wszystkich 1500 obrazów',xlabel='Krok optymalizacji',ylabel='Wartość',ylim=(0,1.02));axs[0].legend(fontsize=8)
    success=((a['confidence']>=.9)&(a['iou']>=.9)).sum(axis=0)
    axs[1].plot(steps,success,color='#178d9b')
    axs[1].set(title='Oba progi spełnione w danym kroku',xlabel='Krok optymalizacji',ylabel='Liczba obrazów',ylim=(0,1500))
    heat=a['movement_norm'][:,2].mean(axis=0).reshape(20,30)
    im=axs[2].imshow(heat,cmap='cividis',interpolation='nearest',aspect='auto')
    axs[2].set(title='Średni ruch per patch · krok 500',xlabel='Kolumna patcha',ylabel='Wiersz patcha');fig.colorbar(im,ax=axs[2],label='L2 w 768D',shrink=.82)
    for ax in axs[:2]:ax.grid(alpha=.2)
    fig.suptitle('Ruch a wynik detekcji · cały zbiór',fontsize=18,fontweight='bold')
    fig.text(.055,.04,'Wspólny indeks kroków niezależnych optymalizacji. To wynik oracle na użytych etykietach, nie test generalizacji.\nSukces w danym kroku może różnić się od liczby obrazów, które osiągnęły sukces wcześniej.',fontsize=9)
    fig.subplots_adjust(top=.8,bottom=.23,wspace=.4)
    save(fig,'03_ruch_i_detekcja_caly_zbior')
    print(json.dumps({'figures':3,'selected_control_image':image,'saved':'PNG i PDF','pca_variance':variance},ensure_ascii=False))

if __name__=='__main__':main()
