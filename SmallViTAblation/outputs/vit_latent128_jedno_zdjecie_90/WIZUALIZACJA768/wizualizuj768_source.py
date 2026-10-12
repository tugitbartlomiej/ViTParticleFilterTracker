"""Pokazuje wszystkie768 osi i600 cząsteczek zapisanej interwencji, bez nowej optymalizacji."""
import csv,hashlib,json,math,shutil,time
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,SymLogNorm
from matplotlib.patches import Rectangle,Patch
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'outputs/vit_latent128_jedno_zdjecie_90'
OUT=SOURCE/'WIZUALIZACJA768'
ORANGE='#ec8b23';GREY='#89939c'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    plt.close(fig)

def macro_grid(ax,values,J,norm,title):
    data=values.reshape(24,32)
    image=ax.imshow(data,cmap='RdBu_r',norm=norm,interpolation='nearest')
    selected=set(map(int,J))
    for j in range(768):
        row,col=divmod(j,32)
        if j in selected:
            ax.add_patch(Rectangle((col-.47,row-.47),.94,.94,fill=False,edgecolor=ORANGE,linewidth=1.3))
        rgba=plt.get_cmap('RdBu_r')(norm(values[j]));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
        ax.text(col,row,str(j),ha='center',va='center',fontsize=6,color='black' if lum>.55 else 'white')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=12)
    return image

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    data=np.load(SOURCE/'LATENT_PELNY.npz');before=data['before'];after=data['after'];J=data['channels0']
    assert before.shape==after.shape==(600,768) and len(np.unique(J))==128
    identity=np.zeros(768,bool);identity[J]=True
    assert np.array_equal(before[:,~identity],after[:,~identity])
    result=json.loads((SOURCE/'RESULT.json').read_text(encoding='utf-8'))
    rgb=cv2.cvtColor(cv2.imread(str(SOURCE/'WEJSCIE_VIT.png')),cv2.COLOR_BGR2RGB)
    x,y,w,h=result['record']['bbox_xywh']
    rows=np.arange(600)//30;cols=np.arange(600)%30
    roi=((cols+.5)*16>x)&((cols+.5)*16<x+w)&((rows+.5)*16>y)&((rows+.5)*16<y+h)
    assert roi.any();means=[v[roi].mean(0) for v in (before,after)]
    rms=[np.sqrt((v[:,J]**2).mean(1)).reshape(20,30) for v in (before,after)]
    vlim=max(float(abs(m).max()) for m in means)
    color_norm=SymLogNorm(linthresh=.3,linscale=1,vmin=-vlim,vmax=vlim)
    rms_norm=Normalize(vmin=0,vmax=max(float(r.max()) for r in rms))
    # Every axis appears once in each board; the board layout has no latent geometry.
    fig=plt.figure(figsize=(18,12.4),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.55])
    photos=[];boards=[]
    for k,(phase,values,r,det) in enumerate(zip(('PRZED','PO'),means,rms,(result['baseline'],result['intervention']))):
        ax=fig.add_subplot(gs[0,k]);photos.append(ax)
        ax.imshow(rgb);ax.imshow(r,cmap='viridis',norm=rms_norm,alpha=.38,extent=(0,480,320,0),interpolation='nearest')
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='cyan',linewidth=1.8))
        b=det['bbox_xyxy'];ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='lime',linewidth=1.8))
        ax.set_title(f"{phase}: siła aktywacji128 kanałów na zdjęciu\nPewność {det['confidence']:.2%}; IoU {det['iou']:.2%}",fontsize=13)
        ax.set_xlim(0,480);ax.set_ylim(320,0);ax.axis('off')
        ax=fig.add_subplot(gs[1,k]);boards.append(ax)
        grid=macro_grid(ax,values,J,color_norm,f'{phase}: wszystkie768 makrowektorów (indeksy0–767)')
    fig.colorbar(plt.cm.ScalarMappable(norm=rms_norm,cmap='viridis'),ax=photos,shrink=.7,pad=.01,label='RMS aktywacji J128 w danej cząsteczce; wspólna skala')
    cb=fig.colorbar(grid,ax=boards,shrink=.75,pad=.01,label='Średnia współrzędna w obszarze ramki narzędzia; wspólna skala sym-log')
    fig.suptitle('To samo zdjęcie i stała baza768 osi. Pomarańczowa ramka:128 wybranych kierunków tożsamości.',fontsize=14)
    fig.supxlabel(f'Każdy kafelek to jedna oś; układ32×24 służy numeracji. Średnia z {roi.sum()} tokenów w ramce etykiety. Pozostałe640 osi: identyczne wartości.',fontsize=10)
    save(fig,'LATENT768_PRZED_PO')
    print('SAVED LATENT768_PRZED_PO',flush=True)
    # Fixed shared PCA directions from the joint, unstandardized token cloud.
    all_tokens=np.concatenate([before,after]).astype(np.float64);center=all_tokens.mean(0);centered=all_tokens-center
    covariance=centered.T@centered/(len(centered)-1)
    eig,vectors=np.linalg.eigh(covariance);order=np.argsort(eig)[::-1];P=vectors[:,order[:2]].T
    for k in range(2):
        if P[k,np.argmax(abs(P[k]))]<0:P[k]*=-1
    explained=eig[order[:2]]/max(float(eig.sum()),1e-20)
    projection=[(v-center)@P.T for v in (before,after)]
    components=[(m-center)[:,None]*P.T for m in means]
    np.savez_compressed(OUT/'RZUT2D.npz',projection_matrix=P,center=center,explained_variance_ratio=explained,
                        before=projection[0],after=projection[1],basis_vectors_2d=P.T,
                        mean_roi_before=means[0],mean_roi_after=means[1],components_roi_before=components[0],components_roi_after=components[1],channels0=J,roi_tokens=np.flatnonzero(roi))
    common=max(float(abs(v).max()) for v in components)*1.12
    fig,axs=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for ax,v,phase in zip(axs[:2],components,('PRZED','PO')):
        for chosen,color in ((~identity,GREY),(identity,ORANGE)):
            ends=v[chosen]
            ax.quiver(np.zeros(len(ends)),np.zeros(len(ends)),ends[:,0],ends[:,1],angles='xy',scale_units='xy',scale=1,color=color,alpha=.55 if not chosen is identity else .8,width=.0025,headwidth=3)
            ax.scatter(ends[:,0],ends[:,1],s=7 if color==GREY else 11,color=color,zorder=3)
        ax.set_xlim(-common,common);ax.set_ylim(-common,common);ax.set_aspect('equal');ax.grid(alpha=.15)
        ax.set_title(f'{phase}:768 składowych średniej cząsteczki\nw obszarze ramki narzędzia',fontsize=11)
        ax.set_xlabel('Wkład w rzut na kierunek PCA1');ax.set_ylabel('Wkład w rzut na kierunek PCA2')
        changed=(means[1]-means[0])*P
        important=J[np.argsort(np.linalg.norm(changed[:,J],axis=0))[-6:]]
        for j in important:ax.annotate(str(j),v[j],fontsize=8,xytext=(3,3),textcoords='offset points',color='#9c4c09')
    ax=axs[2]
    for selected,color in ((~identity,GREY),(identity,ORANGE)):
        ends=P[:,selected].T
        ax.quiver(np.zeros(len(ends)),np.zeros(len(ends)),ends[:,0],ends[:,1],angles='xy',scale_units='xy',scale=1,color=color,alpha=.45,width=.002)
        ax.scatter(ends[:,0],ends[:,1],color=color,s=8)
    bound=float(abs(P).max())*1.15;ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.set_aspect('equal');ax.grid(alpha=.15)
    ax.set_title('768 makrowektorów e_j: baza przed i po\npozostaje identyczna; zmieniają się współrzędne',fontsize=11)
    ax.set_xlabel('Rzut osi e_j na PCA1');ax.set_ylabel('Rzut osi e_j na PCA2')
    ax.legend(handles=[Patch(color=ORANGE,label='128 wybranych osi'),Patch(color=GREY,label='640 pozostałych osi')],loc='lower left',fontsize=8)
    fig.suptitle(f'Wspólny rzut2D przed i po: PCA1 {explained[0]:.1%}, PCA2 {explained[1]:.1%} wariancji. Strzałki to składowe, nie przemieszczenia w pikselach.',fontsize=12)
    fig.supxlabel('Lewo/środek: (średnia współrzędna ROI − środek wspólnej chmury) × rzut danej osi. Wszystkie768 składowych; wspólna skala przed/po.',fontsize=10)
    save(fig,'MAKROWEKTORY_RZUT2D')
    print('SAVED MAKROWEKTORY_RZUT2D',flush=True)
    # Full per-axis spatial atlas: both phases for every one of the768 dimensions.
    thumbnail=cv2.resize(rgb,(150,100),interpolation=cv2.INTER_AREA)
    cmap=plt.get_cmap('RdBu_r');all_channels=[]
    with PdfPages(OUT/'ATLAS768_MAPY_NA_ZDJECIU.pdf') as pdf:
        for page in range(32):
            fig=plt.figure(figsize=(16.5,11.7));gs=fig.add_gridspec(4,6,left=.025,right=.975,bottom=.045,top=.905,wspace=.10,hspace=.25)
            for slot in range(24):
                j=page*24+slot;all_channels.append(j);r,c=divmod(slot,6)
                sub=gs[r,c].subgridspec(1,2,wspace=.025)
                limit=max(float(abs(before[:,j]).max()),float(abs(after[:,j]).max()),1e-6)
                norm=SymLogNorm(linthresh=max(.01,limit/50),vmin=-limit,vmax=limit)
                for phase,v in enumerate((before,after)):
                    ax=fig.add_subplot(sub[0,phase]);colors=cmap(norm(v[:,j].reshape(20,30)))[:,:,:3]
                    color=cv2.resize(colors,(150,100),interpolation=cv2.INTER_NEAREST)
                    overlay=np.clip(thumbnail.astype(float)/255*.57+color*.43,0,1)
                    ax.imshow(overlay)
                    ax.add_patch(Rectangle((x/480*150,y/320*100),w/480*150,h/320*100,fill=False,edgecolor='#4be5e2',linewidth=.5))
                    ax.set_xticks([]);ax.set_yticks([])
                    ax.set_title(f'{j} '+('PRZED' if phase==0 else 'PO'),fontsize=6.8,color='#a3540d' if identity[j] else '#454d55',pad=2)
                    for spine in ax.spines.values():spine.set_color(ORANGE if identity[j] else '#abb1b5');spine.set_linewidth(1.4 if identity[j] else .4)
                    if phase==0:ax.set_xlabel(f"{'J128 · ' if identity[j] else ''}skala ±{limit:.3g}",fontsize=6,labelpad=1)
            fig.suptitle(f'Atlas768 osi — strona{page+1}/32 · kanały{page*24}–{page*24+23}\nPomarańczowa obwódka: wybrane J128. Para zdjęć: PRZED / PO.',fontsize=12,y=.975)
            fig.text(.5,.935,'Kolor: aktywacja danego kanału w siatce20×30 tokenów; nie mapa uwagi. Wspólna skala dla pary; skale różnych kanałów mogą się różnić.',ha='center',fontsize=8)
            fig.text(.5,.015,'Niebieski: ujemna współrzędna; czerwony: dodatnia. Ramka cyan: etykieta narzędzia. Indeksy kanałów0–767.',ha='center',fontsize=8)
            pdf.savefig(fig,dpi=130)
            if page==0:fig.savefig(OUT/'ATLAS768_STRONA1.png',dpi=160)
            plt.close(fig)
            if (page+1)%4==0:print('ATLAS',page+1,'/32',flush=True)
    assert all_channels==list(range(768))
    with (OUT/'OSIE768.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file);writer.writerow(['channel0','selected_J128','roi_mean_before','roi_mean_after','delta_roi_mean','PCA1_basis_projection','PCA2_basis_projection'])
        for j in range(768):writer.writerow([j,int(identity[j]),float(means[0][j]),float(means[1][j]),float(means[1][j]-means[0][j]),float(P[0,j]),float(P[1,j])])
    meta=dict(source_index=result['source_index'],full_shape=[600,768],selected_channels=J.tolist(),untouched640=True,
              roi_token_count=int(roi.sum()),roi_definition='token center inside annotated rectangle',all768_on_each_grid=True,
              atlas_channels=all_channels,atlas_pages=32,pca_fit='joint before/after token cloud, raw coordinates, centered, not standardized',
              explained_variance_ratio=explained.tolist(),projection_is_pixel_space=False,basis_unchanged=True,
              seconds=time.perf_counter()-start,source_latent_sha=hashlib.sha256((SOURCE/'LATENT_PELNY.npz').read_bytes()).hexdigest())
    (OUT/'METADATA.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    note=f'''# Jak czytać wizualizację768 osi

LATENT768_PRZED_PO: górne obrazy pokazują RMS współrzędnych w128 wybranych osiach każdego z600 tokenów. Nakładka przypisana siatce20×30 miejsc obrazu, nie kierunkowi wektora w pikselach. Zielona ramka to wynik detektora, cyan to etykieta. Dolne tablice mają po768 kafelków, po jednym na makrowektor e_j. Kolor pokazuje średnią współrzędną z_j w{roi.sum()} tokenach, których środki leżą w ramce narzędzia. To obszar ramki, nie maska samego narzędzia. Rozmieszczenie32×24 jest tylko indeksem kanałów, a nie rzutem geometrycznym latentu. Wybrane J128 mają pomarańczową obwódkę. Skala kolorów symetryczna logarytmiczna, wspólna przed i po.

MAKROWEKTORY_RZUT2D: ten sam rzut PCA dla obu stanów, dopasowany do wspólnej chmury1200 tokenów. Dwie osie zachowują{explained.sum():.2%} wariancji tej chmury. W dwóch pierwszych panelach pokazano wszystkie768 składowych średniej reprezentacji z obszaru ramki, po odjęciu środka wspólnej chmury: (mu_j-c_j)P e_j. Składowe w128 osiach są pomarańczowe, w640 pozostałych szare. Trzeci panel pokazuje samą bazę P e_j: identyczną przed i po. Nie obracaliśmy osi; zmieniły się współrzędne cząsteczek. Rzut do2D może nakładać strzałki i nie zachowuje pełnych odległości ani kątów w768D.

ATLAS768_MAPY_NA_ZDJECIU.pdf:32 strony, po24 osie, każdy kanał ma zdjęcie przed i po z aktywacją wszystkich600 tokenów. Wszystkie768 kanałów występują dokładnie raz. Każda para ma tę samą skalę, a różne osie własne skale opisane pod parą. To mapy aktywacji współrzędnych, nie mapy uwagi i nie dowód, że dany kanał koduje tylko narzędzie. J128 to zestaw128 osi wybrany we wcześniejszych badaniach; nie oznacza wyłącznej odpowiedzialności tych osi za detekcję.

Dane: jeden obraz z interwencją oracle, przy zamrożonych wagach. Piksele obrazu się nie zmieniły. Pozostałe640 współrzędnych każdego tokenu są bitowo identyczne. Pełne wartości i statystyki768 osi zapisano w OSIE768.csv; macierz rzutu i wszystkie rzuty tokenów w RZUT2D.npz. Obliczenia nie zmieniają zapisanego latentu ani checkpointu.
'''
    (OUT/'JAK_CZYTAC.md').write_text(note,encoding='utf-8')
    shutil.copy2(HERE/'wizualizuj768.py',OUT/'wizualizuj768_source.py')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    (OUT/'HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps({k:meta[k] for k in ('all768_on_each_grid','atlas_pages','roi_token_count','explained_variance_ratio','seconds')}),flush=True)

if __name__=='__main__':main()
