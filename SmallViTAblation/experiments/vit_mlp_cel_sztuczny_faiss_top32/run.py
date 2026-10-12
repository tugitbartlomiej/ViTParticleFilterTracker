"""Artificial full latent -> best10k MLP -> DINO -> exact FAISS35k retrieval."""
import os,sys,json,hashlib,subprocess,zipfile,io,html,time
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'outputs'/HERE.name
LIB=ROOT/'outputs/vit_dino_faiss_35000';MODEL=ROOT/'outputs/vit_translator_mlp_random10000/MODEL_best.pt'
TARGET=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
sys.path.insert(0,str(ROOT/'experiments/vit_translator_mlp_pelny_latent'))
from model import load
sys.path.insert(0,str(ROOT/'experiments/vit_kierunek_nadzorowany_detekcja128'))
from model_snapshot import MalyViT
torch.set_num_threads(4);cv2.setNumThreads(1)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,o):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(t,p)
def latent_stats(real,target,j):
    a=real[:,j].flatten().astype(np.float64);b=target[:,j].flatten().astype(np.float64)
    mse=float(np.mean((a-b)**2));energy=float(np.mean(b*b));cos=float(a@b/(max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)))
    other=np.setdiff1d(np.arange(768),j)
    return dict(MSE128=mse,cosine128=cos,value_agreement128=100*max(0,1-np.sqrt(mse/energy)),MSE640=float(np.mean((real[:,other]-target[:,other])**2)))
def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True);(OUT/'zdjecia').mkdir(exist_ok=True)
    cfg=json.loads((LIB/'CONFIG.json').read_text(encoding='utf-8'));spec=cfg['spec']
    assert sha(LIB/'manifest.jsonl')==cfg['manifest_sha256']
    rows=[json.loads(s) for s in (LIB/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==35000 and len({r['id'] for r in rows})==35000 and all(r['split']=='TRAIN' and r['row']==i for i,r in enumerate(rows))
    features=np.load(LIB/'features.npy',mmap_mode='r');assert features.shape==(35000,384)
    for lo in range(0,35000,1000):
        p=LIB/'chunks'/f'{lo:06d}.npy';meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['config_hash']==cfg['config_hash'] and sha(p)==meta['sha256']
        assert np.array_equal(features[lo:lo+1000],np.load(p))
    model,checkpoint=load(MODEL,'cpu');config=checkpoint['config']
    assert config['dino_model']==spec['model']=='dinov2_vits14'
    assert config['dino_weights_sha256']==spec['weights_sha256']==sha(spec['weights'])
    assert spec['preprocessing']==dict(decode='PIL RGB',resize='shorter side 256, torchvision Resize(256), PIL BICUBIC, antialias=True',crop='torchvision CenterCrop(224)',tensor='ToTensor(), float32 [0,1], CHW',mean=[.485,.456,.406],std=[.229,.224,.225],augmentation=False)
    assert config['preprocess']=='RGB Resize(shorter edge 256, bicubic), CenterCrop224, ImageNet normalize'
    assert cfg['precision']=='fp32' and spec['dimension']==384
    target=np.load(TARGET);before=target['before'];after=target['after'];j=target['channels0'].astype(int)
    assert before.shape==after.shape==(600,768) and j.shape==(128,)
    with torch.inference_mode():
        queries=model(torch.tensor(np.stack([after,before])))[0].numpy()
    second,_=load(MODEL,'cpu')
    with torch.inference_mode():again=second(torch.tensor(np.stack([after,before])))[0].numpy()
    np.testing.assert_array_equal(queries,again);del model,second,checkpoint
    child=subprocess.run([sys.executable,'-X','utf8',str(HERE/'faiss_search.py')],input=json.dumps(dict(queries=queries.tolist())),text=True,capture_output=True,check=True)
    result=json.loads(child.stdout)
    vit=MalyViT().eval();assert sha(config['vit_checkpoint'])==config['vit_checkpoint_sha256']
    vit.load_state_dict(torch.load(config['vit_checkpoint'],map_location='cpu',weights_only=True)['net']);vit.zapisuj_latent(True)
    records=[];images=[]
    for rank,(rid,score) in enumerate(zip(result['ids'][0],result['scores'][0]),1):
        row=rows[rid]
        with zipfile.ZipFile(row['archive']) as z:raw=z.read(row['member'])
        assert hashlib.sha256(raw).hexdigest()==row['image_sha256']
        with Image.open(io.BytesIO(raw)) as im:im.load();im=im.convert('RGB')
        images.append(im.copy());native_name=f"{rank:02d}_{row['film']}_klatka_{row['frame']}{Path(row['member']).suffix.lower()}"
        (OUT/'zdjecia'/native_name).write_bytes(raw)
        arr=cv2.resize(np.asarray(im),(480,320),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(arr).permute(2,0,1).float()[None]/255
        with torch.inference_mode():_,real=vit(x)
        stats=latent_stats(real[0].numpy(),after,j)
        # Only scalar measurements survive; do not save any per-image latent arrays.
        records.append(dict(rank=rank,library_row=rid,DINO_cosine=score,id=row['id'],film=row['film'],frame=row['frame'],source_archive=row['archive'],source_member=row['member'],image_sha256=row['image_sha256'],image='zdjecia/'+native_name,gt=row['gt'],natural_vit_against_artificial_target=stats))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    sheet=Image.new('RGB',(1280,8*255),(246,246,240));draw=ImageDraw.Draw(sheet)
    for k,(im,r) in enumerate(zip(images,records)):
        x=(k%4)*320;y=(k//4)*255;im.thumbnail((304,196));sheet.paste(im,(x+8,y+8))
        draw.text((x+8,y+206),f"{r['rank']:02d}. DINO cosine {r['DINO_cosine']:.4f}",font=font,fill=(25,40,45))
        draw.text((x+8,y+228),r['id'],font=small,fill=(65,65,65))
    sheet.save(OUT/'TOP32.jpg',quality=95)
    images[0].save(OUT/'NAJBLIZSZE_ZDJECIE.png')
    controls=[dict(rank=i+1,library_row=rid,id=rows[rid]['id'],DINO_cosine=s) for i,(rid,s) in enumerate(zip(result['ids'][1],result['scores'][1]))]
    answer=dict(query='full original artificial600x768 target translated by best MLP trained on random10000; no SD',translator=str(MODEL),translator_sha256=sha(MODEL),translator_selected_step=3500,target_sha256=sha(TARGET),original_vit_checkpoint_sha256=config['vit_checkpoint_sha256'],library_images=35000,library_manifest_sha256=cfg['manifest_sha256'],library_features_sha256=sha(LIB/'features.npy'),same_DINO_representation_verified=True,fresh_mlp_reload_identical=True,no_saved_latent_arrays=True,translated_before_after_cosine=float(queries[0]@queries[1]),top32=records,natural_before_control_top32=controls,top32_overlap_with_before=len(set(result['ids'][0])&set(result['ids'][1])),FAISS_QA=result['faiss_bruteforce_max_errors'],index_self_query_passed=result['self_query_passed'],seconds=time.time()-start)
    write('RESULT.json',answer)
    cards=''.join(f'<a href="{html.escape(r["image"])}"><img src="{html.escape(r["image"])}"><strong>{r["rank"]}. cosine DINO {r["DINO_cosine"]:.4f}</strong><br>{html.escape(r["id"])}<br>Naturalny ViT: cosine128 {r["natural_vit_against_artificial_target"]["cosine128"]:.4f}</a>' for r in records)
    (OUT/'GALERIA.html').write_text('<!doctype html><meta charset="utf-8"><title>ViT MLP DINO FAISS top32</title><style>body{font:16px Arial;background:#f7f7ee;color:#263238;margin:30px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}a{color:inherit;text-decoration:none}img{width:100%;height:210px;object-fit:contain}</style><h1>Sztuczny latent ViT → MLP → DINO → 32 zdjęcia</h1><p>Najbliższe w bazie35000 TRAIN. Cosine DINO nie oznacza zgodności sztucznego latentu ani poprawy detekcji.</p><div class="grid">'+cards+'</div>',encoding='utf-8')
    b=records[0];st=b['natural_vit_against_artificial_target']
    (OUT/'REPORT.md').write_text(f"# Sztuczny latent → MLP → DINO → FAISS\n\nZweryfikowano wagi, preprocessing,35000 wierszy, każdy z35 chunków, świeże wczytanie MLP i zgodność FAISS z brute force. Indeks IndexFlatIP zbudowany w RAM w osobnym procesie, ponieważ finalizacja istniejącego procesu indeksowania zatrzymała się na konflikcie OpenMP. Nie liczymy ponownie biblioteki DINO i nie zapisujemy tablic latentów.\n\nNajbliższy obraz: {b['id']}. Cosine DINO: {b['DINO_cosine']:.6f}. Naturalny forward oryginalnego ViT względem sztucznego celu: MSE128 {st['MSE128']:.6f}, cosine128 {st['cosine128']:.6f}, umowna zgodność wartości128 {st['value_agreement128']:.4f} procent. To kandydat według przybliżonego translatora, nie dowód realizacji celu przez prawdziwy obraz.\n\nCosine między przetłumaczonym celem a niezmienionym latentem: {answer['translated_before_after_cosine']:.6f}; wspólne zdjęcia w top32: {answer['top32_overlap_with_before']}. Wyłącznie skalary i metadane zapisane w RESULT.json; kopie32 zdjęć w zdjecia/, galeria i przegląd TOP32.jpg.\n",encoding='utf-8')
    print(json.dumps(dict(best=b,translated_before_after_cosine=answer['translated_before_after_cosine'],top32_overlap=answer['top32_overlap_with_before'],seconds=answer['seconds']),ensure_ascii=False))
if __name__=='__main__':main()
