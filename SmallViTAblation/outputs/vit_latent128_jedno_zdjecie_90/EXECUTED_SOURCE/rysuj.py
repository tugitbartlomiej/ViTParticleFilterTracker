"""Pokazuje zapisane pomiary; nie zmienia obrazu wejściowego ani latentu."""
import json,hashlib,shutil
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/vit_latent128_jedno_zdjecie_90'

def main():
    result=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'))
    rgb=cv2.cvtColor(cv2.imread(str(OUT/'WEJSCIE_VIT.png')),cv2.COLOR_BGR2RGB)
    x,y,w,h=result['record']['bbox_xywh']
    fig,axs=plt.subplots(1,2,figsize=(12,4.7))
    for ax,d,title in zip(axs,(result['baseline'],result['intervention']),('Przed zmianą latentu','Po sztucznej zmianie 128 kanałów')):
        ax.imshow(rgb)
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='cyan',linewidth=2,label='Etykieta'))
        b=d['bbox_xyxy']
        ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='lime',linewidth=2,label='Detekcja ViT'))
        ax.set_title(f"{title}\nPewność: {d['confidence']:.2%} | IoU: {d['iou']:.2%}",fontsize=12)
        ax.axis('off');ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('To samo zdjęcie i zamrożony ViT; zmiana wyłącznie wewnętrznego latentu',y=.98,fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'DETEKCJA_PRZED_PO.png',dpi=160);plt.close(fig)
    shutil.copy2(HERE/'rysuj.py',OUT/'EXECUTED_SOURCE/rysuj.py')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='HASHES.json'}
    (OUT/'HASHES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')

if __name__=='__main__':main()
