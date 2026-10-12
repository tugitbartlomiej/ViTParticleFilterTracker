"""Render a reproducible comparison and audit saved PNG foreground pixels."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
OUT=HERE/'wyniki/trudne_pilot_20260923'
rows=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'))
errors=[]
for i,row in enumerate(rows):
    root=OUT/f'case_{i:02d}'
    mask=np.asarray(Image.open(root/'MASKA.png'))[:,:,0]>127
    original=np.asarray(Image.open(root/'original.png'))
    for arm in row['arms']:
        image=np.asarray(Image.open(root/f'{arm}.png'))
        error=int(np.abs(image.astype(int)-original.astype(int))[mask].max())
        if error: raise ValueError(f'Changed foreground pixels: case {i}, {arm}, {error}')
        errors.append(error)

# Show the FIRST case where the Koopman arm misses but the detector-only arm hits.
# This is an illustrative case, not a representative average; full gallery is retained.
idx=next((i for i,r in enumerate(rows) if r['arms']['detector']['hit'] and not r['arms']['detector_koopman']['hit']),0)
root=OUT/f'case_{idx:02d}'; row=rows[idx]
arms=['original','random','random_search','detector','detector_koopman']
names=['Oryginał','Losowy FastGAN','Najtrudniejszy z losowych','Sterowanie detektorem','Detektor + Koopman']
maps=[np.load(root/f'{a}_particles.npz')['identity'].reshape(24,24) for a in arms]
vmin,vmax=np.percentile(np.stack(maps),[2,98])
fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained')
for j,(arm,name,mp) in enumerate(zip(arms,names,maps)):
    info=row['arms'][arm]
    axes[0,j].imshow(Image.open(root/f'{arm}.png'))
    axes[0,j].set_title(name+'\n'+('TRAFIENIE' if info['hit'] else 'CHYBIENIE')+
                      f"; pewność {info['confidence']:.2f}\nIoU {info['iou']:.2f}",fontsize=11)
    if info['confidence']>=.5:
        x0,y0,x1,y1=np.array(info['box'])*256
        axes[0,j].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='yellow',lw=1.4))
    im=axes[1,j].imshow(mp,cmap='viridis',vmin=vmin,vmax=vmax,interpolation='nearest')
    weight=np.load(root/f'{arm}_particles.npz')['mask_weight'].reshape(24,24)
    axes[1,j].contour(weight,levels=[.025],colors=['red'],linewidths=.7)
    axes[1,j].set_title(f"Odczyt kierunku RDZEN\nkontrast {info['contrast']:.2f}; reszta {info['residual']:.2f}",fontsize=10)
    for ax in axes[:,j]: ax.axis('off')
fig.colorbar(im,ax=axes[1,:],shrink=.65,label='Aktywacja tokena; wspólna skala')
fig.suptitle('To samo prawdziwe narzędzie, zmienione otoczenie\n'+row['group_id']+
             ' — przykład wybrany według jawnego warunku; pełne wyniki obejmują wszystkie operacje',fontsize=14)
fig.savefig(OUT/'PRZYKLAD_CZASTKI.png',dpi=140)
plt.close(fig)
(OUT/'KONTROLA_PNG.json').write_text(json.dumps(dict(images_checked=len(errors),
       maximum_foreground_pixel_error=max(errors),illustrated_case=idx,
       illustration_rule='first detector hit and detector_koopman miss; not a typical example'),indent=2),encoding='utf-8')
print(json.dumps(dict(images_checked=len(errors),foreground_max_error=max(errors),illustrated_case=idx)))
