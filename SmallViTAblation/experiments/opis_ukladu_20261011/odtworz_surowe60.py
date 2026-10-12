"""Inferencja istniejących 806 checkpointów, pełny GELU768, FP32; bez treningu."""
from pathlib import Path
import hashlib, importlib.util, json, time, zipfile
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009'
OUT = ROOT / 'outputs/surowe_trajektorie_60epok_768d_20261011'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()

def save(name, data):
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(OUT / name)

def main():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    probes = json.loads((SRC/'wyniki/gesty/gesty/SONDY.json').read_text(encoding='utf-8'))['pelne']
    allrows = [json.loads(s) for s in (SRC/'wejscia/MANIFEST_35000.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [allrows[i] for i in probes]
    fs = sorted((ROOT/'assets/MalyViT_35000_od_zera_s47/wagi').glob('k*.pt'), key=lambda p:int(p.stem[1:]))
    steps = np.array([int(p.stem[1:]) for p in fs], dtype=np.int64)
    assert len(fs)==806 and steps[0]==0 and steps[-1]==65640
    if (OUT/'LATENT_768_FP32.npy').exists():
        raise RuntimeError('Istniejący eksport: nie nadpisuję danych')
    images=[]
    for i,r in zip(probes, rows):
        with zipfile.ZipFile(r['archive']) as z: raw=z.read(r['member'])
        assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
        (OUT/f'sonda_pool_{i}.jpg').write_bytes(raw)
        im=cv2.cvtColor(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA))
    spec=importlib.util.spec_from_file_location('raw60_model',SRC/'kod/model.py')
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    net=M.MalyViT().cuda().eval();net.zapisuj_latent(True)
    X=torch.from_numpy(np.stack(images)).cuda().permute(0,3,1,2).float()/255
    shape=(len(fs),len(rows),600,768)
    mm=np.lib.format.open_memmap(OUT/'LATENT_768_FP32.partial.npy', mode='w+', dtype=np.float32, shape=shape)
    np.save(OUT/'KROKI.npy',steps)
    save('SONDY.json',dict(pool_rows=probes,images=rows,tokens_order='row-major: token=30*y+x',grid=[20,30]))
    meta=dict(status='running',shape=list(shape),axes=['checkpoint','image','token','channel'],dtype='float32',
              precision='FP32 forward; autocast off; TF32 off',training_images=35000,epochs=60,
              source_run='pelny (1826056 + 1826076)',resume_step=22974,device=torch.cuda.get_device_name(),
              torch_version=torch.__version__,model_source=str(SRC/'kod/model.py'),model_sha256=sha(SRC/'kod/model.py'),
              exporter_sha256=sha(Path(__file__)),inference_only=True,pca=False,completed_states=0,
              output_file='LATENT_768_FP32.npy',bytes=806*8*600*768*4,checkpoints=[])
    save('MANIFEST.json',meta)
    start=time.time();qa=[]
    with torch.inference_mode():
        for j,(k,p) in enumerate(zip(steps,fs)):
            ck=torch.load(p,map_location='cpu',weights_only=True)
            assert ck['krok']==int(k)
            net.load_state_dict(ck['net'],strict=True)
            _,z=net(X);a=z.cpu().numpy()
            assert a.shape==(8,600,768) and np.isfinite(a).all()
            mm[j]=a
            meta['checkpoints'].append(dict(step=int(k),path=str(p),sha256=sha(p)))
            cp=ROOT/'outputs/audyt_szkla_768d_20261010/precision_controls'/f'k{k:05d}.npz'
            if cp.exists() and int(k) in [300,1094,2188,22974,65640]:
                with np.load(cp) as q:
                    ref=q['fp32'];d=a-ref
                    qa.append(dict(step=int(k),max_abs=float(np.abs(d).max()),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(ref),1e-30))))
            if j==0:
                _,again=net(X)
                assert np.array_equal(a,again.cpu().numpy())
            if j%25==0 or j==len(fs)-1:
                mm.flush();meta['completed_states']=j+1;meta['elapsed_seconds']=time.time()-start
                save('MANIFEST.json',meta)
                print(f'{j+1}/{len(fs)} krok {k}, {time.time()-start:.1f}s',flush=True)
    mm.flush();del mm
    (OUT/'LATENT_768_FP32.partial.npy').replace(OUT/'LATENT_768_FP32.npy')
    meta.update(status='complete',completed_states=len(fs),control_comparisons=qa,repeat_step0_bitwise_equal=True)
    save('MANIFEST.json',meta)
    save('KONTROLA.json',dict(finite_all=True,shape=list(shape),steps_strictly_increasing=bool(np.all(np.diff(steps)>0)),
                            epoch_endpoints_present=all(e*1094 in steps for e in range(61)),comparisons=qa,
                            note='Różnice względem wcześniejszego FP32 na innych ustawieniach GPU mogą być niezerowe.'))
    print('COMPLETE',OUT,flush=True)

if __name__=='__main__':main()
