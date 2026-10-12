"""Czy lokalny podgląd pokazuje rzeczywiste, wspólnie rzutowane stany?
Brak zgodności indeksów/czasu lub interpolacja obala poprawność wizualizacji.
"""
import argparse, atexit, functools, hashlib, io, json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from dense_timeline import DenseTimeline
from trajectory_data import selected_payload
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009'
DATA=OUT/'data'
IMAGE_ROOT=ROOT/'experiments/koopman_fastgan_petla/local_inputs/data/images'
DENSE=DenseTimeline(DATA/'dense')
LOCK=threading.Lock()
STATE={}

def load():
    with LOCK:
        if STATE:return STATE
        path=DATA/'projection.npz'
        if not path.exists() or not (DATA/'BUNDLE_MANIFEST.json').exists() or not (DATA/'LOCAL_TRANSFER.json').exists():return None
        transfer=json.loads((DATA/'LOCAL_TRANSFER.json').read_text(encoding='utf-8'))
        if not transfer.get('all_original_remote_bundle_hashes_match'):return None
        with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        metadata=json.loads((DATA/'metadata.json').read_text(encoding='utf-8'))
        assert a['positions'].shape==(1500,3,600,3)
        assert a['movement_norm'].shape==(1500,3,600)
        assert a['control_positions'].shape==(16,501,8,3)
        assert a['confidence'].shape==a['iou'].shape==(1500,501)
        assert all(np.isfinite(v).all() for v in a.values() if np.issubdtype(v.dtype,np.floating))
        STATE.update(arrays=a,metadata=metadata)
        return STATE

def records(meta):
    return meta.get('images',meta.get('records',[]))

def image_index(image):
    """Selection is an identity, never a clamped or negative array index."""
    if type(image) is not int or not 0<=image<1500:
        raise ValueError('Indeks obrazu musi być liczbą całkowitą 0..1499')
    return image

def image_binary(image):
    """Original verified JPEG, no resize, overlays, copies on disk or image cache."""
    image_index(image)
    state=load()
    if not state:raise ValueError('Dane jeszcze niegotowe')
    images=state['metadata'].get('images')
    if not isinstance(images,list) or len(images)!=1500:
        raise ValueError('Brak pełnej listy tożsamości 1500 obrazów')
    row=images[image]
    if not isinstance(row,dict) or type(row.get('index')) is not int or row['index']!=image:
        raise ValueError('Niezgodna kolejność obrazu w metadata')
    name=row.get('image_name')
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:jpg|jpeg)',name,re.IGNORECASE):
        raise ValueError('Niewłaściwa nazwa zdjęcia JPEG')
    expected=row.get('image_sha256')
    if not isinstance(expected,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',expected):
        raise ValueError('Brak poprawnego image_sha256')
    root=IMAGE_ROOT.resolve()
    path=(root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Ścieżka zdjęcia wychodzi poza katalog obrazów')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected.lower():
        raise ValueError('Niezgodna suma image_sha256 zdjęcia')
    from PIL import Image, ImageFile
    # Image.open is lazy. Check the complete original JPEG without depending
    # on a process-wide permissive truncated-image decoder flag.
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ValueError('Dekoder JPEG nie może dopuszczać uciętych zdjęć')
    try:
        with Image.open(io.BytesIO(raw)) as decoded:
            if decoded.format!='JPEG':raise ValueError('Plik nie jest zdjęciem JPEG')
            decoded.load()
    except OSError as exc:
        raise ValueError('Nie można zdekodować zdjęcia JPEG') from exc
    return raw

def dense_image_frame(image,step,movement,geometry=True):
    """Image-major 600-token slice from the same verified binary as the cloud."""
    image_index(image)
    dense=DENSE.load()
    if dense is None or type(step) is not int or step not in dense['steps']:
        raise ValueError('Brak zweryfikowanego rzeczywistego kroku obrazu')
    state=load()
    if not state:raise ValueError('Dane jeszcze niegotowe')
    a=state['arrays']
    if a['confidence'].shape!=a['iou'].shape or a['confidence'].shape!=(1500,501):
        raise ValueError('Niezgodna rzeczywista oś metryk obrazów 0..500')
    start=image*600;stop=start+600
    blob=DENSE.binary(step)
    if len(blob)!=dense['point_count']*16 or stop>dense['point_count']:
        raise ValueError('Niezgodna tożsamość obrazu/cząsteczek w dense')
    packed=np.frombuffer(blob,dtype='<f4').reshape(dense['point_count'],4)[start:stop]
    xyz=packed[:,:3].copy()
    if movement:
        initial=DENSE.binary(0)
        if len(initial)!=len(blob):raise ValueError('Niezgodny bazowy zapis dense')
        xyz-=np.frombuffer(initial,dtype='<f4').reshape(dense['point_count'],4)[start:stop,:3]
    norm=packed[:,3]
    if not np.isfinite(packed).all() or (norm<0).any():raise ValueError('Niepoprawne cząsteczki obrazu')
    confidence=float(a['confidence'][image,step]);iou=float(a['iou'][image,step])
    if not all(np.isfinite(v) and 0<=v<=1 for v in (confidence,iou)):
        raise ValueError('Niepoprawne rzeczywiste metryki obrazu')
    both90=int(np.sum((a['confidence'][:,step]>=.9)&(a['iou'][:,step]>=.9)))
    # Selected patch heat is the same full-768D norm as particle color, so use
    # the shared particle norm scale (global heat_scale is an image average).
    return {'xyz':xyz.astype(float).round(5).tolist() if geometry else [],
            'norm':norm.astype(float).round(5).tolist() if geometry else [],
            'tokens':list(range(600)) if geometry else [],
            'images':[image]*600 if geometry else [],
            'indices':list(range(start,stop)) if geometry else [],
            'bounds':dense['bounds']['movement' if movement else 'natural'],
            'lines':[],'heatmap':norm.reshape(20,30).astype(float).round(5).tolist(),
            'count':600,'total':600,'step':step,'mode':'image','image':image,
            'confidence':confidence,'iou':iou,'both90':both90,
            'selected_both90':int(confidence>=.9 and iou>=.9),'selected_image_count':1,
            'bar_tokens':[],'bar_norm':[],'norm_median':float(np.median(norm)),
            'norm_max':float(norm.max()),'norm_scale':dense['norm_scale'],'heat_scale':dense['norm_scale']}

def limits(pos):
    flat=pos.reshape(-1,3)
    lo=flat.min(axis=0);hi=flat.max(axis=0);pad=np.maximum((hi-lo)*.04,.03)
    return np.column_stack((lo-pad,hi+pad)).round(5).tolist()

def meta_payload():
    s=load()
    if not s:return {'ready':False}
    a=s['arrays'];meta=s['metadata']
    ids=a['control_images'].astype(int).tolist()
    rec=records(meta)
    return {'ready':True,'metadata':meta,'control_images':ids,'images':rec,
            'pca_variance':meta.get('explained_variance_ratio',meta.get('fit',meta.get('pca',{})).get('explained_variance_ratio',[])),
            'milestones':[0,100,500],'global_steps':DENSE.load()['steps'] if DENSE.load() else [0,100,500], 'dense_ready':bool(DENSE.load())}

def curves(mode,image):
    if mode=='image':image_index(image)
    if mode=='global' and DENSE.load():return DENSE.load()['curves']
    return sparse_curves(mode,image)

@functools.lru_cache(maxsize=80)
def sparse_curves(mode,image):
    a=load()['arrays']
    if mode=='exact' and image<0:
        ix=a['control_images'].astype(int)
        return {'confidence':a['confidence'][ix].mean(axis=0).tolist(),
                'iou':a['iou'][ix].mean(axis=0).tolist(),'label':'Średnia panelu 16 obrazów'}
    if mode=='global':
        c=a['confidence'].mean(axis=0);q=a['iou'].mean(axis=0)
        success=np.mean((a['confidence']>=.9)&(a['iou']>=.9),axis=0)
        return {'confidence':c.tolist(),'iou':q.tolist(),'success_fraction':success.tolist(),'label':'Średnia z1500 obrazów'}
    return {'steps':list(range(501)),'confidence':a['confidence'][image].tolist(),'iou':a['iou'][image].tolist(),'label':f'Obraz{image}'}

def cloud_binary(step):
    if DENSE.load():return DENSE.binary(step)
    return sparse_cloud_binary(step)

@functools.lru_cache(maxsize=4)
def trajectory_binary(spec):
    return selected_payload(DENSE, spec)

@functools.lru_cache(maxsize=3)
def sparse_cloud_binary(step):
    """Little-endian Float32: image-major, token-major, x/y/z/full-768D norm."""
    k={0:0,100:1,500:2}.get(step)
    if k is None:raise ValueError('Zapisane stany: 0,100,500')
    a=load()['arrays']
    packed=np.empty((1500,600,4),dtype='<f4')
    packed[:,:,:3]=a['positions'][:,k]
    packed[:,:,3]=a['movement_norm'][:,k]
    return packed.tobytes(order='C')

def frame(mode,image,step,movement,trails,geometry=True):
    if mode=='global' and DENSE.load():return DENSE.frame(step,movement,image,geometry)
    if mode=='image':
        image_index(image)
        if DENSE.load():return dense_image_frame(image,step,movement,geometry)
    a=load()['arrays'];meta=load()['metadata']; rec=records(meta)
    all_control=mode=='exact' and image<0
    image=-1 if all_control else max(0,min(1499,image))
    step=max(0,min(500,step)); fullstep=step
    if mode=='exact':
        if all_control:
            xy=a['control_positions'].transpose(1,0,2,3).reshape(501,128,3)
            norms=a['control_norm'].transpose(1,0,2).reshape(501,128)
            tokens=a['control_tokens'].astype(int).reshape(128)
            im=np.repeat(a['control_images'].astype(int),8)
        else:
            panel=np.where(a['control_images']==image)[0]
            if not len(panel):raise ValueError('Ten obraz nie ma zapisu co krok')
            panel=int(panel[0]);xy=a['control_positions'][panel]
            norms=a['control_norm'][panel];tokens=a['control_tokens'][panel].astype(int)
            im=np.full(8,image)
        if movement:xy=xy-xy[:1]
        xyz=xy[step];norm=norms[step];bounds=limits(xy)
        lines=xy[max(0,step-40):step+1].transpose(1,0,2).tolist() if trails else []
        heat=None;count=total=len(tokens)
        norm_scale=float(norms.max());heat_scale=norm_scale
    else:
        k={0:0,100:1,500:2}.get(step)
        if k is None:raise ValueError('Cała chmura ma tylko stany0,100,500')
        if mode=='global':
            flat=a['positions'].transpose(0,2,1,3).reshape(900000,3,3)
            idx=np.random.default_rng(20261009).choice(900000,8000,replace=False) if geometry else np.arange(900000)
            xy=flat[idx].transpose(1,0,2)
            im=idx//600;tokens=idx%600
            norm=a['movement_norm'][im,k,tokens];heat=a['movement_norm'][:,k].mean(axis=0).reshape(20,30)
            count=len(idx);total=900000
            norm_scale=float(a['movement_norm'][im,:,tokens].max())
            heat_scale=float(a['movement_norm'].mean(axis=0).max())
        else:
            xy=a['positions'][image];im=np.full(600,image);tokens=np.arange(600)
            norm=a['movement_norm'][image,k];heat=norm.reshape(20,30);count=total=600
            norm_scale=float(a['movement_norm'][image].max());heat_scale=norm_scale
        if movement:xy=xy-xy[:1]
        xyz=xy[k];bounds=limits(xy);lines=[]
    metric_images=a['control_images'].astype(int) if all_control else (np.arange(1500) if mode=='global' else np.array([image]))
    bar_tokens=np.unique(tokens) if mode=='exact' else np.array([],dtype=int)
    bar_norm=np.array([norm[tokens==t].mean() for t in bar_tokens])
    return {'xyz':xyz.astype(float).round(5).tolist() if geometry else [],'norm':norm.astype(float).round(5).tolist() if geometry else [],'tokens':tokens.tolist() if geometry else [],'images':im.tolist() if geometry else [],
            'bounds':bounds,'lines':lines,'heatmap':None if heat is None else heat.astype(float).round(5).tolist(),
            'count':count,'total':total,'step':fullstep,'mode':mode,'image':image,
            'confidence':float(a['confidence'][metric_images,step].mean()),
            'iou':float(a['iou'][metric_images,step].mean()),
            'both90':int(np.sum((a['confidence'][:,step]>=.9)&(a['iou'][:,step]>=.9))),
            'selected_both90':int(np.sum((a['confidence'][metric_images,step]>=.9)&(a['iou'][metric_images,step]>=.9))),
            'selected_image_count':len(metric_images),'bar_tokens':bar_tokens.tolist(),'bar_norm':bar_norm.astype(float).round(5).tolist(),
            'norm_median':float(np.median(norm)),'norm_max':float(np.max(norm)),
            'norm_scale':max(norm_scale,1e-6),'heat_scale':max(heat_scale,1e-6)}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            u=urlparse(self.path);q=parse_qs(u.query,keep_blank_values=True)
            def value(key,default):return q.get(key,[default])[0]
            if u.path.startswith('/api/'):
                if u.path=='/api/image':
                    if set(q)!={'image'} or len(q['image'])!=1 or not re.fullmatch(r'0|[1-9][0-9]{0,3}',q['image'][0]):
                        raise ValueError('Wymagany pojedynczy parametr image=0..1499')
                    body=image_binary(int(q['image'][0]))
                    self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',len(body));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body);return
                if u.path=='/api/cloud.bin':
                    if not load():raise ValueError('Dane jeszcze niegotowe')
                    body=cloud_binary(int(value('step','0')))
                    self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',len(body));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body);return
                if u.path=='/api/trajectories.bin':
                    if set(q)!={'images'} or len(q['images'])!=1:
                        raise ValueError('Wymagany pojedynczy parametr images')
                    body,meta=trajectory_binary(q['images'][0])
                    self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',len(body));self.send_header('X-Trajectory-Meta',json.dumps(meta,separators=(',',':')));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body);return
                if u.path=='/api/meta':d=meta_payload()
                elif u.path=='/api/tool-mask':
                    from tool_masks import payload
                    ix=image_index(int(value('image','0')))
                    image_binary(ix) # Verify the original source image identity before overlay.
                    d=payload(load()['metadata']['images'][ix])
                elif u.path=='/api/timeline':
                    dense=DENSE.load()
                    d={'ready':bool(dense),'steps':dense['steps'] if dense else [0,100,500],'stride':dense['stride'] if dense else None,'point_count':900000}
                elif u.path=='/api/raw':
                    from raw_space import request as raw_request
                    d=raw_request(load(),q)
                elif not load():d={'ready':False}
                elif u.path=='/api/frame':d=frame(value('mode','global'),int(value('image','0')),int(value('step','0')),value('movement','0')=='1',value('trails','0')=='1',value('geometry','1')=='1')
                elif u.path=='/api/curves':d=curves(value('mode','global'),int(value('image','0')))
                else:raise ValueError('Nieznany widok')
                body=json.dumps(d,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
                mime='application/json; charset=utf-8'
            else:
                allowed={'/':OUT/'WIZUALIZACJA.html','/plotly.min.js':OUT/'plotly.min.js','/cloud-webgl.js':Path(__file__).with_name('cloud-webgl.js'),'/trajectory-webgl.js':Path(__file__).with_name('trajectory-webgl.js')}
                path=allowed.get(u.path)
                if path is None:self.send_error(404);return
                body=path.read_bytes();mime='text/html; charset=utf-8' if u.path=='/' else 'text/javascript'
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',len(body));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
        # HTTP reason phrases are Latin-1; Polish diagnostics belong in the
        # UTF-8 error body rather than the status line.
        except ValueError as e:self.send_error(400,'Invalid request',str(e))
        except FileNotFoundError as e:self.send_error(404,'Not found',str(e))
        except Exception as e:
            self.send_error(500,'Backend error',str(e))
    def log_message(self,*_):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8774);args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'SERVER.json').write_text(json.dumps({'pid':os.getpid(),'url':f'http://127.0.0.1:{args.port}/','source':str(Path(__file__).resolve())}),encoding='utf-8')
    print(f'Podgląd lokalny http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
