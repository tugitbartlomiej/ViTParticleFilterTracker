"""Czy jeden zmieniony latent daje top1 confidence i IoU>0.90? Brak obu obala sukces."""
import csv,hashlib,json,shutil,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model_snapshot import MalyViT,dekoduj,_giou
from metryki_source import iou

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
INPUT=ROOT/'outputs/vit_optymalizacja_latentu128/input'
OUT=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
PREVIOUS=ROOT/'outputs/vit_latent128_regulator_przestrzenny'
CKPT=Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k1000.pt')
FLOOR=-0.16997120
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik

def js(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights_hash(net):
    h=hashlib.sha256()
    for name,v in net.state_dict().items():h.update(name.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def detection(o,g):
    b,s=dekoduj(o.detach().cpu());boxes=b[0].numpy();scores=s[0].numpy();overlap=iou(boxes,g)
    return dict(confidence=float(scores[0]),iou=float(overlap[0]),bbox_xyxy=boxes[0].tolist(),
                all_boxes_xyxy=boxes.tolist(),all_scores=scores.tolist(),all_iou=overlap.tolist())

def fast(net,J,z,q,changed):
    b=net.bloki[-1];delta=b.ls2(F.linear(changed-z[:,:,J],b.mlp.fc2.weight[:,J],None))
    return net.glowa(net.norm(q+delta)).reshape(1,20,30,5)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Wynik już istnieje; nie nadpisuję go.')
    torch.set_num_threads(4);torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();rows=json.loads((INPUT/'manifest.json').read_text())
    J=torch.tensor(np.load(INPUT/'channels.npz')['channels0'],dtype=torch.long,device='cuda')
    old=np.load(PREVIOUS/'TEACHER_DETECTIONS.npz');ids=old['train_original'].tolist()
    teacher_maps=torch.from_numpy(old['maps']);base_maps=np.load(ROOT/'outputs/vit_latent128_maksymalizacja/DETECTIONS_FULL400.npz')['baseline']
    choices=[]
    for k,i in enumerate(ids):
        row=rows[i];x,y,w,h=row['bbox_xywh']
        if w<32 or h<24:continue
        g=np.array([x,y,x+w,y+h],np.float32);a=detection(teacher_maps[k:k+1],g);b=detection(torch.from_numpy(base_maps[i:i+1]),g)
        choices.append((min(a['confidence'],a['iou']),i,a,b))
    assert choices
    _,idx,old_teacher,old_base=max(choices,key=lambda c:(c[0],-c[1]))
    row=rows[idx];assert row['split']=='train' and sha(INPUT/row['image'])==row['sha256']
    x0,y0,w,h=row['bbox_xywh'];g=np.array([x0,y0,x0+w,y0+h],np.float32)
    js(OUT/'SELECTION.json',dict(source_index=idx,record=row,criterion='maximum minimum of historical teacher confidence and IoU on eligible train images',
                               candidates=len(choices),historical_teacher=old_teacher,historical_baseline=old_base))
    net=MalyViT().cuda().eval();ck=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert ck['krok']==1000;net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False);wh=weights_hash(net)
    cache={};block=net.bloki[-1]
    hz=block.mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o.detach().clone()))
    hq=block.register_forward_hook(lambda m,a,o:cache.update(q=o.detach().clone()))
    native=cv2.imread(str(INPUT/row['image']));assert native is not None
    rgb=cv2.resize(cv2.cvtColor(native,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    image=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255
    with torch.no_grad():baseline,_=net(image)
    hz.remove();hq.remove();z=cache['z'];q=cache['q']
    torch.testing.assert_close(fast(net,J,z,q,z[:,:,J]),baseline,rtol=1e-5,atol=2e-5)
    teacher=torch.load(PREVIOUS/'TEACHER_TRAIN.pt',map_location='cpu',weights_only=True)
    assert idx in teacher['train_ids'] and torch.equal(teacher['channels'],J.cpu())
    initial=teacher['targets'][idx:idx+1].cuda();del teacher
    cx=int(np.clip(round((g[0]+g[2])/32-.5),0,29));cy=int(np.clip(round((g[1]+g[3])/32-.5),0,19));center=cy*30+cx
    px,py=(cx+.5)*16,(cy+.5)*16
    distances=torch.tensor([[px-g[0],py-g[1],g[2]-px,g[3]-py]],device='cuda',dtype=torch.float32)/16
    assert (distances>0).all();target_log=distances.log()
    negmask=torch.ones(1,600,device='cuda');negmask[0,center]=0
    def objective(o):
        flat=o.reshape(1,600,5);logits=flat[:,:,0]
        positive=F.softplus(5.5-logits[0,center]);negative=(F.softplus(logits+2)*negmask).sum()/599
        rank=F.cross_entropy(logits,torch.tensor([center],device='cuda'))
        box=F.smooth_l1_loss(flat[:,center,1:],target_log)
        giou=_giou(flat[:,center,1:].clamp(-10,10).exp(),distances).mean()
        return positive+negative+.5*rank+20*box+10*(1-giou)
    best=initial.clamp(min=FLOOR).clone();best_loss=float(objective(fast(net,J,z,q,best)).detach());history=[];steps=0
    for lr in (.05,.15,.3):
        changed=best.detach().clone().requires_grad_();opt=torch.optim.Adam([changed],lr=lr)
        for step in range(1,2001):
            o=fast(net,J,z,q,changed);loss=objective(o)+1e-6*(changed-z[:,:,J]).square().mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert torch.isfinite(changed.grad).all();opt.step()
            with torch.no_grad():
                changed.clamp_(min=FLOOR);current_o=fast(net,J,z,q,changed);current_loss=float(objective(current_o))
                if current_loss<best_loss:best_loss=current_loss;best=changed.clone()
                if step%25==0:
                    d=detection(current_o,g);history.append(dict(lr=lr,step=step,objective=current_loss,confidence=d['confidence'],iou=d['iou']))
                    print(json.dumps(history[-1]),flush=True)
                    if d['confidence']>=.93 and d['iou']>=.95:best=changed.clone();break
        steps+=step;check=detection(fast(net,J,z,q,best),g)
        if check['confidence']>.9 and check['iou']>.9:break
    edited=z.clone();edited[:,:,J]=best
    other=torch.ones(768,dtype=torch.bool,device='cuda');other[J]=False
    assert torch.equal(edited[:,:,other],z[:,:,other]) and float(edited[:,:,J].min())>=FLOOR-1e-6
    arrays=dict(before=z[0].cpu().numpy(),after=edited[0].cpu().numpy(),delta=(edited-z)[0].cpu().numpy(),
                channels0=J.cpu().numpy(),channels1=J.cpu().numpy()+1,grid_shape=np.array([20,30]),
                token_row=np.arange(600)//30,token_col=np.arange(600)%30)
    np.savez_compressed(OUT/'LATENT_PELNY.npz',**arrays)
    np.savez_compressed(OUT/'LATENT128.npz',before=arrays['before'][:,arrays['channels0']],after=arrays['after'][:,arrays['channels0']],
                        delta=arrays['delta'][:,arrays['channels0']],channels0=arrays['channels0'],grid_shape=arrays['grid_shape'])
    loaded=np.load(OUT/'LATENT_PELNY.npz');restored=torch.from_numpy(loaded['after'])[None].cuda()
    assert torch.equal(restored,edited)
    hhook=block.mlp.act.register_forward_hook(lambda m,a,o:restored)
    with torch.no_grad():actual,_=net(image)
    hhook.remove();expected=fast(net,J,z,q,best)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5);assert weights_hash(net)==wh
    before=detection(baseline,g);after=detection(actual,g)
    assert after['confidence']>.9 and after['iou']>.9,f'Nie osiągnięto kryterium: {after}'
    selected=set(arrays['channels0'].tolist());header='token,row,col,'+','.join(('J128_' if j in selected else '')+f'ch{j:03d}' for j in range(768))
    coordinates=np.column_stack([np.arange(600),np.arange(600)//30,np.arange(600)%30])
    for label in ('before','after','delta'):
        np.savetxt(OUT/f'LATENT_PELNY_{label.upper()}.csv',np.column_stack([coordinates,arrays[label]]),delimiter=',',header=header,comments='',fmt=['%d']*3+['%.9g']*768)
    with (OUT/'LATENT128_WARTOSCI.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file);writer.writerow(['token','row','col','order128','channel0','channel1','before','after','delta'])
        for t in range(600):
            for k,j in enumerate(arrays['channels0']):writer.writerow([t,t//30,t%30,k,int(j),int(j)+1,*[float(arrays[a][t,j]) for a in ('before','after','delta')]])
    centers_x=(np.arange(600)%30+.5)*16;centers_y=(np.arange(600)//30+.5)*16
    roi=(centers_x>g[0])&(centers_x<g[2])&(centers_y>g[1])&(centers_y<g[3]);assert roi.any()
    summaries=[]
    for k,j in enumerate(arrays['channels0']):
        a=arrays['before'][:,j];b=arrays['after'][:,j];delta=b-a
        summaries.append(dict(order128=k,channel0=int(j),channel1=int(j)+1,before_mean=float(a.mean()),after_mean=float(b.mean()),
                              delta_mean=float(delta.mean()),delta_rms=float(np.sqrt((delta**2).mean())),delta_max_abs=float(abs(delta).max()),
                              before_roi_mean=float(a[roi].mean()),after_roi_mean=float(b[roi].mean()),delta_roi_mean=float(delta[roi].mean()),
                              after_min=float(b.min()),after_max=float(b.max())))
    with (OUT/'KANALY128_PODSUMOWANIE.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    shutil.copy2(INPUT/row['image'],OUT/'ZDJECIE_ZRODLOWE.jpg');cv2.imwrite(str(OUT/'WEJSCIE_VIT.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    shutil.copy2(CKPT,OUT/'MODEL_VIT_K1000.pt')
    # Visible annotations describe the actual frozen-network detections, not generated pixels.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,d,title in zip(axs,(before,after),('Before intervention','After J128 latent intervention')):
        ax.imshow(rgb);ax.add_patch(Rectangle((g[0],g[1]),g[2]-g[0],g[3]-g[1],fill=False,edgecolor='cyan',linewidth=2,label='Ground truth'))
        bx=d['bbox_xyxy'];ax.add_patch(Rectangle((bx[0],bx[1]),bx[2]-bx[0],bx[3]-bx[1],fill=False,edgecolor='lime',linewidth=2,label='Prediction'))
        ax.set_title(f"{title}\nConfidence {d['confidence']:.2%}; IoU {d['iou']:.2%}");ax.axis('off');ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Same image and frozen ViT; only 128 of 768 latent channels changed',fontsize=11);fig.tight_layout()
    fig.savefig(OUT/'DETEKCJA_PRZED_PO.png',dpi=160);plt.close(fig)
    change=arrays['delta'][:,arrays['channels0']]
    fig,axs=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.8,1]})
    vmax=float(np.quantile(abs(change),.99));im=axs[0].imshow(change.T,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    axs[0].set_xlabel('Token index (20 x 30 grid)');axs[0].set_ylabel('Order within selected J128');axs[0].set_title('Change of every token along J128')
    fig.colorbar(im,ax=axs[0],label='Delta activation (color clipped at 99th percentile)')
    rms=np.sqrt((change**2).mean(1)).reshape(20,30);im=axs[1].imshow(rms,cmap='magma');axs[1].set_title('RMS change across J128');axs[1].set_xlabel('Token column');axs[1].set_ylabel('Token row');fig.colorbar(im,ax=axs[1])
    fig.tight_layout();fig.savefig(OUT/'KANALY128_ZMIANA.png',dpi=160);plt.close(fig)
    result=dict(status='complete',ziarno=20261005,source_index=idx,record=row,baseline=before,intervention=after,
                confidence_over90=after['confidence']>.9,iou_over90=after['iou']>.9,uses_image_ground_truth=True,
                full_shape=[600,768],selected_shape=[600,128],channels0=arrays['channels0'].tolist(),
                weights_buffers_unchanged=True,unchanged640=True,latent_reload_bitwise_equal=True,
                full_forward_tail_max_error=float((actual-expected).abs().max().detach()),steps=steps,
                checkpoint_sha=sha(CKPT),manifest_sha=sha(INPUT/'manifest.json'),weights_hash=wh,
                device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,history=history,
                token_roi_definition='token grid center inside ground-truth rectangle; descriptive only')
    js(OUT/'RESULT.json',result);zapisz_wynik(__file__,'single_oracle',result)
    lines=['# Jedno zdjęcie z interwencją J128','',
           f"Obraz z puli400: indeks {idx}, film {row['film']}. Checkpoint1000 i wszystkie wagi/bufory zamrożone.",'',
           '| Pomiar najpewniejszej ramki | Przed | Po zmianie J128 |','|---|---:|---:|',
           f"| Confidence | {before['confidence']:.6f} | {after['confidence']:.6f} |",f"| IoU do etykiety | {before['iou']:.6f} | {after['iou']:.6f} |",'',
           'Pełny latent600×768 przed/po/różnica zapisany w LATENT_PELNY.npz oraz trzech CSV. Wybrane kolumny CSV oznaczone J128. LATENT128.npz i LATENT128_WARTOSCI.csv zawierają wszystkie600×128 wartości. KANALY128_PODSUMOWANIE.csv ma statystyki każdej osi, w tym tokenów z obszaru GT. Nie istnieje jeden skalar danej osi dla całego obrazu; każda z600 cząsteczek ma własną współrzędną.','',
           'To pojedynczy korzystnie wybrany przykład z optymalizacją na jego etykiecie. Confidence nie jest dokładnością na nowych obrazach. Te same piksele bez wstawienia latentu dają wynik przed. Zakres GELU zachowany; nie dowodzi to osiągalności tego latentu z naturalnego zdjęcia.','',
           'Kontrole: odczytany z dysku pełny latent jest bitowo identyczny; pełny forward odtwarza szybki tail; pozostałe640 kanałów identyczne; hash wag/buforów zachowany.','',
           '| Kanał (indeks0) | Średnia przed | Średnia po | RMS zmiany | Średnia zmiana w ROI |','|---|---:|---:|---:|---:|']
    for s in sorted(summaries,key=lambda s:-s['delta_rms'])[:16]:lines.append(f"| {s['channel0']} | {s['before_mean']:.6f} | {s['after_mean']:.6f} | {s['delta_rms']:.6f} | {s['delta_roi_mean']:.6f} |")
    text='\n'.join(lines)+'\n';(OUT/'RAPORT.md').write_text(text,encoding='utf-8');(HERE/'WYNIKI_SINGLE.md').write_text(text,encoding='utf-8')
    provenance={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='HASHES.json'};js(OUT/'HASHES.json',provenance)
    source=OUT/'EXECUTED_SOURCE';source.mkdir(exist_ok=True)
    for pattern in ('*.py','*.md','WYNIKI.json'):
        for p in HERE.glob(pattern):shutil.copy2(p,source/p.name)
    print('COMPLETE',json.dumps(dict(source_index=idx,before=before,after=after,steps=steps,seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
