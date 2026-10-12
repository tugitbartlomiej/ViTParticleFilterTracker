"""Raportuje rzeczywisty wynik, kontroluje odtworzenie interwencji i rysuje metryki."""
from pathlib import Path
import json,hashlib,csv
import numpy as np,torch,cv2
from maximize import OUT,INPUT,CKPT,FLOOR,stats
from model_snapshot import MalyViT,dekoduj
from core import intervene

def main():
    result=json.loads((OUT/'RESULT.json').read_text());data=np.load(OUT/'TARGETS_FULL400.npz');channels=data['channels0'];target=data['target'];base=data['base'];delta=data['delta_raw'];assert target.shape==(400,600,128);np.testing.assert_allclose(target,base+delta,atol=0,rtol=0);assert target.min()>=FLOOR-1e-6
    maps=np.load(OUT/'DETECTIONS_FULL400.npz');rows=json.loads((INPUT/'manifest.json').read_text());net=MalyViT().cuda().eval();net.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True)['net']);net.requires_grad_(False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    J=torch.tensor(channels,device='cuda',dtype=torch.long);checks=[]
    for i in (0,199,399):
        im=cv2.cvtColor(cv2.imread(str(INPUT/rows[i]['image'])),cv2.COLOR_BGR2RGB);im=cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA);x=torch.from_numpy(im).permute(2,0,1)[None].float().cuda()/255
        d=torch.tensor(delta[i:i+1],device='cuda');h=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:intervene(o,d,J,1.))
        with torch.no_grad():actual,_=net(x)
        h.remove();expected=torch.tensor(maps['optimized'][i:i+1],device='cuda');torch.testing.assert_close(actual,expected,rtol=3e-5,atol=5e-5);checks.append(dict(source_index=i,max_output_error=float((actual-expected).abs().max())))
    verify=dict(actual_hook_checks=checks,target_equals_base_plus_delta=True,min_target=float(target.min()),shape=list(target.shape),range_GELU=True)
    (OUT/'VERIFICATION.json').write_text(json.dumps(verify,indent=2),encoding='utf-8')
    with (OUT/'KANALY128.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['channel0','channel1','mean_target','min_target','max_target','mean_delta','rms_delta'])
        for j,c in enumerate(channels):w.writerow([int(c),int(c+1),float(target[:,:,j].mean()),float(target[:,:,j].min()),float(target[:,:,j].max()),float(delta[:,:,j].mean()),float(np.sqrt((delta[:,:,j]**2).mean()))])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5));names=['baseline','optimized','random_control'];labels=['Bez zmiany','Optymalizacja J128','Losowe znaki'];x=np.arange(3)
    for offset,(key,label) in enumerate([('AP50','AP50'),('AP75','AP75'),('AP50_95','AP50:95')]):ax.bar(x+(offset-1)*.23,[result[m][key]*100 for m in names],width=.23,label=label)
    ax.set_xticks(x,labels);ax.set_ylim(0,105);ax.set_ylabel('AP, procent');ax.set_title('Zamrożony ViT k1000: 400 obrazów\nOptymalizacja latentu każdego obrazu z jego etykietą');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'POROWNANIE.png',dpi=160);plt.close(fig)
    # Deterministic examples, same images for before/after, selected without looking at gains.
    ids=[0,79,159,239,319,399];fig,axes=plt.subplots(3,2,figsize=(12,12))
    for ax,i in zip(axes.flat,ids):
        image=cv2.cvtColor(cv2.imread(str(INPUT/rows[i]['image'])),cv2.COLOR_BGR2RGB);image=cv2.resize(image,(480,320));ax.imshow(image)
        x,y,w,h=rows[i]['bbox_xywh'];ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,color='lime',lw=2,label='GT'))
        for name,color,label in [('baseline','red','Przed'),('optimized','cyan','Po')]:
            bx,sc=dekoduj(torch.tensor(maps[name][i:i+1]));k=int(sc[0].argmax());b=bx[0,k].numpy();ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,color=color,lw=1.5,label=f'{label}: {float(sc[0,k]):.3f}'))
        ax.set_xlim(0,480);ax.set_ylim(320,0);ax.set_title(f'Obraz {i}: najwyższa pewność, bez wyboru ramki przez GT');ax.legend(fontsize=8);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'PRZYKLADY.png',dpi=120);plt.close(fig)
    lines=['# Silna optymalizacja 128 kanałów: wynik','',f"Checkpoint1000; {result['images']} obrazów; {result['steps']} kroków Adam; lr {result['lr']}; {result['device']}; czas {result['seconds']:.1f}s.",'','| Metoda | AP50 | AP75 | AP50:95 | Strata |','|---|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        m=result[name];lines.append(f"| {label} | {m['AP50']:.6f} | {m['AP75']:.6f} | {m['AP50_95']:.6f} | {m['mean_loss']:.6f} |")
    lines+=['',f"Cel opisowy AP50>=0.95: {'osiągnięty' if result['optimized']['AP50']>=.95 else 'nieosiągnięty'}.",f"Wynik IoU>=0.5 i confidence>=0.9 dla najpewniejszej ramki: {100*result['optimized']['fraction_top_iou50_conf90']:.2f}% obrazów.",f"Największa bezwzględna zmiana współrzędnej: {result['max_abs_delta']:.6f}; minimalna wartość docelowa: {result['min_target']:.8f}.",'','Wagi i bufory pozostały identyczne. Sprawdzono zachowanie640 kanałów, zakres GELU, odtworzenie zmodyfikowanego forward na obrazach0/199/399.','',
        'To indywidualna optymalizacja aktywacji400 obrazów z ich etykietami. Nie są to nowe wyuczone wagi ani wynik generalizacji. Nie znaleziono obrazów realizujących te latenty. Zakres GELU jest zachowany, lecz duże aktywacje mogą leżeć daleko poza typowym rozkładem. Maksimum globalne nie zostało udowodnione.','',
        'Wcześniejsze AP0.226→0.277 dotyczyło k500,85 obrazów i małych ograniczeń. Nie porównywać bezpośrednio z k1000/400 jako efektu samego zwiększenia liczby kroków.','',
        'TARGETS_FULL400.npz zawiera wszystkie wartości per obraz/token/kanał. KANALY128.csv to statystyki, nie uniwersalne wartości docelowe. POROWNANIE.png i PRZYKLADY.png pokazują metryki oraz stałe przykłady.']
    (OUT/'RAPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT AND ACTUAL HOOK VERIFICATION PASS')
if __name__=='__main__':main()
