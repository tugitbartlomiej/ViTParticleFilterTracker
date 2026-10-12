"""Czy cały lokalny zbiór ma naturalną realizację celu600x768? Utrzymujący się
duży błąd obala znalezienie bliskiej realizacji. Bez subsamplingu i interwencji.
"""
print('IMPORTING',flush=True)
import csv
import hashlib
import json
import queue
import shutil
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from model_snapshot import MalyViT, dekoduj

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_latent768_pelny_zbior'
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
ARCHIVES=ROOT.parent/'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
SPLITS=ROOT/'feature_cache/eden_static36/data/tracking_static36/split_manifest.json'
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik


def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def js(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(p)


def main():
    start=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Zakonczony wynik istnieje; uzyj osobnego biegu')
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    groups=json.loads(SPLITS.read_text())['groups']
    inventory=[];counts=Counter();fingerprint=hashlib.sha256()
    for path in sorted(ARCHIVES.glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            members=[x for x in z.infolist() if x.filename.lower().endswith(('.jpg','.jpeg','.png'))]
            for x in members:
                fingerprint.update(f'{path.name}|{x.filename}|{x.CRC}|{x.file_size}\n'.encode())
            split=groups.get(path.stem,'unknown')
            counts[split]+=len(members)
            inventory.append(dict(path=str(path),film=path.stem,split=split,count=len(members),bytes=path.stat().st_size))
    total=sum(counts.values())
    assert len(inventory)==170 and total==461547,(len(inventory),total)
    js(OUT/'INVENTORY.json',dict(archives=inventory,total=total,split_counts=dict(counts),zip_directory_sha256=fingerprint.hexdigest(),split_sha256=sha(SPLITS)))
    with np.load(SOURCE/'LATENT_PELNY.npz') as f:
        target_np=f['after'].copy();channels_np=f['channels0'].copy()
    target=torch.from_numpy(target_np).cuda()[None]
    channels=torch.as_tensor(channels_np,dtype=torch.long,device='cuda')
    other=torch.ones(768,dtype=torch.bool,device='cuda');other[channels]=False
    net=MalyViT().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(SOURCE/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net'])
    net.zapisuj_latent(True)
    def weight_hash():
        h=hashlib.sha256()
        for name,x in net.state_dict().items():h.update(name.encode());h.update(x.cpu().numpy().tobytes())
        return h.hexdigest()
    weights_before=weight_hash()
    def decode(raw):
        image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise RuntimeError('Nieudany odczyt obrazu')
        return cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    def measure(z):
        diff=(z-target).square()
        values=torch.stack([diff.mean((1,2)),diff[:,:,channels].mean((1,2)),diff[:,:,other].mean((1,2)),torch.nn.functional.cosine_similarity(z.flatten(1),target.expand(z.shape[0],-1,-1).flatten(1),dim=1)],1)
        return values.cpu().numpy()
    def save_best(record,raw,z,output,label):
        (OUT/f'{label}.jpg').write_bytes(raw)
        np.savez_compressed(OUT/f'LATENT_{label}.npz',natural=z.cpu().numpy(),target=target_np,delta=z.cpu().numpy()-target_np,channels0=channels_np,grid_shape=np.array([20,30]),detector_output=output.cpu().numpy())
        js(OUT/f'{label}.json',record)

    pending=queue.Queue(maxsize=3)
    stop=threading.Event()
    def put(item):
        while not stop.is_set():
            try:pending.put(item,timeout=.5);return
            except queue.Full:pass
    def producer():
        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                for entry in inventory:
                    with zipfile.ZipFile(entry['path']) as z:
                        names=sorted(x.filename for x in z.infolist() if x.filename.lower().endswith(('.jpg','.jpeg','.png')))
                        for lo in range(0,len(names),16):
                            if stop.is_set():return
                            selected=names[lo:lo+16]
                            raw=[z.read(n) for n in selected] # ZIP verifies CRC before returning bytes.
                            images=np.stack(list(executor.map(decode,raw)))
                            records=[dict(archive=entry['path'],member=n,film=entry['film'],split=entry['split']) for n in selected]
                            put((records,images,raw))
            put(None)
        except BaseException as e:put(e)
    worker=threading.Thread(target=producer,daemon=True);worker.start()
    print('READY',total,dict(counts),flush=True)
    best=None;best_train=None;processed=0;seen=Counter();top=[];top_train=[]
    infer_start=time.perf_counter();last_report=infer_start
    fields=['ordinal','film','split','archive','member','mse768','mse128','mse640','cosine768','confidence']
    try:
        with open(OUT/'RANKING_CALY_ZBIOR.csv','w',newline='',encoding='utf-8') as stream,torch.inference_mode():
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            while True:
                item=pending.get()
                if item is None:break
                if isinstance(item,BaseException):raise item
                records,images,raw=item
                inputs=torch.from_numpy(images).permute(0,3,1,2).float().cuda()/255
                output,z=net(inputs)
                metrics=measure(z)
                boxes,scores=dekoduj(output)
                boxes=boxes.cpu().numpy();scores=scores.cpu().numpy()
                assert np.isfinite(metrics).all()
                for k,record in enumerate(records):
                    processed+=1;seen[record['split']]+=1
                    record.update(ordinal=processed,mse768=float(metrics[k,0]),mse128=float(metrics[k,1]),mse640=float(metrics[k,2]),cosine768=float(metrics[k,3]),confidence=float(scores[k,0]),bbox_xyxy=boxes[k,0].tolist())
                    writer.writerow(record)
                    top.append(record)
                    if record['split']=='train':top_train.append(record)
                    if best is None or record['mse768']<best['mse768']:
                        best=record.copy();save_best(best,raw[k],z[k],output[k],'NAJBLIZSZE_ZDJECIE')
                    if record['split']=='train' and (best_train is None or record['mse768']<best_train['mse768']):
                        best_train=record.copy();save_best(best_train,raw[k],z[k],output[k],'NAJBLIZSZE_TRAIN')
                top=sorted(top,key=lambda r:(r['mse768'],r['ordinal']))[:64]
                top_train=sorted(top_train,key=lambda r:(r['mse768'],r['ordinal']))[:64]
                now=time.perf_counter()
                if now-last_report>=10 or processed==total:
                    speed=processed/max(now-infer_start,.001)
                    progress=dict(status='running',processed=processed,total=total,split_processed=dict(seen),best=best,best_train=best_train,images_per_second=speed,eta_seconds=(total-processed)/speed,seconds=now-start)
                    js(OUT/'PROGRESS.json',progress);stream.flush();last_report=now
                    print('PROGRESS',processed,total,'mse',best['mse768'],'fps',round(speed,2),'eta_min',round(progress['eta_seconds']/60,1),flush=True)
            assert processed==total and seen==counts,(processed,total,seen,counts)
            # Recompute final candidates one by one, without changing any activations.
            candidates={ (r['archive'],r['member']):r for r in top+top_train }
            best_final=None;best_train_final=None
            refined=[]
            for record in candidates.values():
                with zipfile.ZipFile(record['archive']) as archive:raw=archive.read(record['member'])
                inputs=torch.from_numpy(decode(raw)).permute(2,0,1)[None].float().cuda()/255
                output,z=net(inputs);v=measure(z)[0]
                assert np.isclose(v[0],record['mse768'],rtol=1e-4,atol=2e-5)
                boxes,scores=dekoduj(output)
                r=dict(record,mse768=float(v[0]),mse128=float(v[1]),mse640=float(v[2]),cosine768=float(v[3]),confidence=float(scores[0,0]),bbox_xyxy=boxes[0,0].cpu().tolist())
                refined.append(r)
                if best_final is None or r['mse768']<best_final['mse768']:
                    best_final=r;save_best(r,raw,z[0],output[0],'NAJBLIZSZE_ZDJECIE')
                if r['split']=='train' and (best_train_final is None or r['mse768']<best_train_final['mse768']):
                    best_train_final=r;save_best(r,raw,z[0],output[0],'NAJBLIZSZE_TRAIN')
            js(OUT/'FINAL_TOP_CANDIDATES.json',sorted(refined,key=lambda r:r['mse768']))
    finally:stop.set()
    assert weight_hash()==weights_before
    old=json.loads((ROOT/'outputs/vit_latent768_najblizsze_zdjecie/RESULT.json').read_text())
    result=dict(status='complete',ziarno=20261005,images_scanned=processed,archives=len(inventory),split_counts=dict(seen),best=best_final,best_train=best_train_final,previous400=old['best'],original_source=old['original_source'],sd=old['generated_sd'],improvement_vs400=1-best_final['mse768']/old['best']['mse768'],selection='raw MSE of all600x768; no intervention; exhaustive local tracking image ZIP members',target_sha256=sha(SOURCE/'LATENT_PELNY.npz'),checkpoint_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt'),zip_directory_sha256=fingerprint.hexdigest(),weights_unchanged=True,seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),torch=torch.__version__,image=str(OUT/'NAJBLIZSZE_ZDJECIE.jpg'),image_sha256=sha(OUT/'NAJBLIZSZE_ZDJECIE.jpg'),limitation='Exact local archive coverage only. Images across splits used for retrieval, not evaluation. Artificial target need not have a natural preimage. Confidence is not accuracy or AP.')
    js(OUT/'RESULT.json',result)
    js(OUT/'PROGRESS.json',dict(status='complete',processed=processed,total=total,best=best_final,best_train=best_train_final,seconds=result['seconds']))
    zapisz_wynik(__file__,'full_tracking',result)
    report=f"# Najbliższe zdjęcie w całym lokalnym zbiorzeTracking\n\nPrzeszukano{processed} zdjęć ze170ZIP, bez subsamplingu. Podziały: {dict(seen)}.\n\nNajlepszy obraz: {best_final['film']}/{best_final['member']} ({best_final['split']}). MSE768={best_final['mse768']:.6f}; MSE128={best_final['mse128']:.6f}; MSE640={best_final['mse640']:.6f}; cosinus={best_final['cosine768']:.6f}. Pewność={100*best_final['confidence']:.2f}%.\n\nPoprawa MSE768 względem minimum400={100*result['improvement_vs400']:.3f}%. Najlepszytrain: {best_train_final['film']}/{best_train_final['member']}, MSE768={best_train_final['mse768']:.6f}.\n\nWynik jest najbliższy według MSE; nie gwarantuje bliskiej rekonstrukcji sztucznego celu. Pełny rankingCSV zachowuje wynik każdego obrazu. Nie jest to pomiar mAP ani jakości na niewidzianych danych.\n"
    (OUT/'RAPORT.md').write_text(report,encoding='utf-8')
    (Path(__file__).parent/'WYNIKI_PELNY_ZBIOR.md').write_text(report,encoding='utf-8')
    shutil.copy2(__file__,OUT/'szukaj_source.py')
    js(OUT/'HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='HASHES.json'})
    print('COMPLETE',json.dumps(dict(best=best_final,best_train=best_train_final,images=processed,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        js(OUT/'FAILURE.json',dict(status='failed',type=type(e).__name__,error=str(e)))
        raise
