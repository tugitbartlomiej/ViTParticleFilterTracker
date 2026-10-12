"""Które rzeczywiste zdjęcie najbardziej zbliża się do sztucznego celu600x768?
Duży dystans i brak przewagi nad zdjęciem źródłowym obalają znalezienie nowej
naturalnej realizacji celu w tej puli. Wybór wyłącznie według latentu.
"""
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from model_snapshot import MalyViT, dekoduj

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'outputs/vit_optymalizacja_latentu128/input'
SOURCE = ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
OUT = ROOT/'outputs/vit_latent768_najblizsze_zdjecie'
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik


def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def js(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rows = json.loads((INPUT/'manifest.json').read_text(encoding='utf-8'))
    with np.load(SOURCE/'LATENT_PELNY.npz') as f:
        target_np = f['after'].copy()
        channels_np = f['channels0'].copy()
    assert target_np.shape == (600,768) and len(rows)==400
    target = torch.from_numpy(target_np).cuda()[None]
    channels = torch.as_tensor(channels_np,device='cuda',dtype=torch.long)
    other = torch.ones(768,dtype=torch.bool,device='cuda');other[channels]=False
    scale = target.square().mean(1,keepdim=True).sqrt().clamp_min(.25)
    net = MalyViT().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net'])
    net.zapisuj_latent(True)
    def weight_hash():
        h=hashlib.sha256()
        for name,x in net.state_dict().items():
            h.update(name.encode());h.update(x.cpu().numpy().tobytes())
        return h.hexdigest()
    weights_before=weight_hash()
    def tensor(p):
        rgb=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float().cuda()/255
    def distances(z):
        d=(z-target).square()
        normalized=d/scale.square()
        cosine=torch.nn.functional.cosine_similarity(z.flatten(1),target.expand(z.shape[0],-1,-1).flatten(1),dim=1)
        return dict(mse768=d.mean((1,2)),mse128=d[:,:,channels].mean((1,2)),mse640=d[:,:,other].mean((1,2)),weighted_energy=.5*(normalized[:,:,channels].mean((1,2))+normalized[:,:,other].mean((1,2))),cosine768=cosine)
    ranking=[]
    with torch.inference_mode():
        for lo in range(0,len(rows),4):
            subset=rows[lo:lo+4]
            for r in subset:
                assert sha(INPUT/r['image'])==r['sha256']
            output,z=net(torch.stack([tensor(INPUT/r['image']) for r in subset]))
            values=distances(z)
            boxes,scores=dekoduj(output)
            for k,r in enumerate(subset):
                item={name:float(value[k]) for name,value in values.items()}
                item.update(index=r['index'],image=r['image'],film=r['film'],split=r['split'],confidence=float(scores[k,0]),bbox_xyxy=boxes[k,0].cpu().tolist())
                ranking.append(item)
            if (lo+4)%40==0:
                print('SCANNED',min(lo+4,len(rows)),len(rows),flush=True)
                js(OUT/'PROGRESS.json',dict(status='running',scanned=min(lo+4,len(rows)),total=len(rows)))
        ranking.sort(key=lambda x:(x['mse768'],x['index']))
        best=ranking[0]
        row=next(r for r in rows if r['index']==best['index'])
        output,z=net(tensor(INPUT/row['image'])[None])
        measured={k:float(v[0]) for k,v in distances(z).items()}
        for name,value in measured.items():
            assert np.isclose(value,best[name],rtol=1e-4,atol=2e-6),(name,value,best[name])
        boxes,scores=dekoduj(output)
        best.update(measured)
        best.update(confidence=float(scores[0,0]),bbox_xyxy=boxes[0,0].cpu().tolist())
        natural=z[0].cpu().numpy().copy()
        detector=output.cpu().numpy()
    assert weight_hash()==weights_before
    photo=OUT/'NAJBLIZSZE_ZDJECIE.jpg'
    shutil.copy2(INPUT/row['image'],photo)
    assert sha(photo)==row['sha256']
    np.savez_compressed(OUT/'LATENT_NAJBLIZSZEGO_ZDJECIA.npz',natural=natural,target=target_np,delta=natural-target_np,channels0=channels_np,grid_shape=np.array([20,30]),detector_output=detector)
    np.savez_compressed(OUT/'LATENT128_NAJBLIZSZEGO_ZDJECIA.npz',natural=natural[:,channels_np],target=target_np[:,channels_np],delta=natural[:,channels_np]-target_np[:,channels_np],channels0=channels_np)
    with open(OUT/'RANKING400.csv','w',newline='',encoding='utf-8') as f:
        fields=['rank','index','image','film','split','mse768','mse128','mse640','weighted_energy','cosine768','confidence']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for i,item in enumerate(ranking,1):writer.writerow(dict(item,rank=i))
    original=next(r for r in ranking if r['index']==29)
    best128=min(ranking,key=lambda r:(r['mse128'],r['index']))
    best_weighted=min(ranking,key=lambda r:(r['weighted_energy'],r['index']))
    sdpath=ROOT/'outputs/vit_latent768_sd_od_szumu/RESULT.json'
    sd=json.loads(sdpath.read_text(encoding='utf-8'))['generated'] if sdpath.exists() else None
    details=dict(status='complete',ziarno=20261005,seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),torch=torch.__version__,images_scanned=len(rows),selection='minimum raw MSE768 across all600 tokens; no activation intervention',best=best,best128=best128,best_weighted=best_weighted,original_source=original,generated_sd=sd,source_image_selected=best['index']==29,target_sha256=sha(SOURCE/'LATENT_PELNY.npz'),checkpoint_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt'),manifest_sha256=sha(INPUT/'manifest.json'),weights_unchanged=True,image=str(photo),image_sha256=sha(photo),relative_improvement_vs_original=1-best['mse768']/original['mse768'],relative_improvement_vs_sd=1-best['mse768']/sd['mse768'] if sd else None,limitation='Nearest within400 existing photographs only; not evidence of exact target realizability. Positionwise comparison assumes matching token grid. Confidence is not AP.')
    js(OUT/'RESULT.json',details)
    zapisz_wynik(__file__,'nearest400',details)
    js(OUT/'PROGRESS.json',dict(status='complete',scanned=len(rows)))
    report=f"# Zdjęcie najbliższe sztucznemu latentowi\n\nPrzeszukano{len(rows)} zdjęć. Kryterium: surowy MSE600×768, bez podmiany aktywacji i bez etykiet w rankingu.\n\nWybrane zdjęcie: indeks{best['index']}, film{best['film']}. MSE768={best['mse768']:.6f}, MSE128={best['mse128']:.6f}, MSE640={best['mse640']:.6f}. Cosinus pełnego wektora={best['cosine768']:.6f}. Pewność naturalnego detektora={100*best['confidence']:.2f}%.\n\nMSE768 oryginału29={original['mse768']:.6f}. Poprawa względem oryginału={100*details['relative_improvement_vs_original']:.3f}%. Wybór tylko128 kanałów wskazuje indeks{best128['index']}. Wybór ważony wskazuje indeks{best_weighted['index']}.\n\nNajbliższe nie oznacza dokładnego odtworzenia celu. Wynik dotyczy jednej ustalonej puli400 obrazów; pewność nie jest trafnością ani mAP.\n"
    (OUT/'RAPORT.md').write_text(report,encoding='utf-8')
    (Path(__file__).parent/'WYNIKI_SZUKANIE.md').write_text(report,encoding='utf-8')
    shutil.copy2(__file__,OUT/'szukaj_source.py')
    js(OUT/'HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='HASHES.json'})
    print('COMPLETE',json.dumps(dict(best=best,best128=best128['index'],source_selected=details['source_image_selected'],seconds=details['seconds'])),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        js(OUT/'FAILURE.json',dict(status='failed',type=type(e).__name__,error=str(e)))
        raise
