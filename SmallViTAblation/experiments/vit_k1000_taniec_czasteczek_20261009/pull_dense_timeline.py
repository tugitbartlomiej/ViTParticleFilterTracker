"""Resumable transfer; only publish the viewer gate after all SHA256 checks."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from dense_timeline import DenseTimeline

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(p,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--remote',required=True);p.add_argument('--host',default='eden-cluster');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/vit_k1000_taniec_czasteczek_20261009/data/dense');args=p.parse_args()
    if not args.remote.startswith('/home2/faculty/bpiotrowski/smallvit_runs/') or any(c in args.remote for c in '\n\r\t $;`"\''):raise ValueError('Wymagana jawna ścieżka własnego eksperymentu')
    out=args.output;out.mkdir(parents=True,exist_ok=True);lock=out/'TRANSFER.lock'
    # OS advisory lock survives an abandoned file without duplicate workers.
    import msvcrt
    with lock.open('a+b') as locked:
        locked.seek(0);locked.write(b'0');locked.flush();locked.seek(0)
        try:msvcrt.locking(locked.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Transfer tej osi czasu już działa')
        started=time.perf_counter();incoming=out/'manifest.incoming.json'
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=20',f'{args.host}:{args.remote}/manifest.json',str(incoming)],check=True)
        manifest=json.loads(incoming.read_text(encoding='utf-8'));helper=DenseTimeline(out)
        if manifest.get('status')!='complete' or manifest.get('interpolation') is not False:raise ValueError('Producent niegotowy')
        for ordinal,step in enumerate(manifest['steps'],1):
            record=manifest['files'][str(step)];target=helper.path(record['name'])
            if target.exists() and target.stat().st_size==record['bytes'] and sha(target)==record['sha256']:continue
            part=target.with_suffix('.part')
            subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=20',f'{args.host}:{args.remote}/{record["name"]}',str(part)],check=True)
            if part.stat().st_size!=record['bytes'] or sha(part)!=record['sha256']:raise ValueError(f'Uszkodzony transfer kroku {step}')
            os.replace(part,target)
            atomic_json(out/'TRANSFER.json',{'status':'running','verified_frames':ordinal,'total_frames':len(manifest['steps']),'last_step':step,'elapsed_seconds':time.perf_counter()-started})
        # Only now can the server see the complete dense timeline.
        os.replace(incoming,out/'manifest.json');validated=helper.load()
        if validated is None:raise ValueError('Nie udało się włączyć zweryfikowanej osi')
        atomic_json(out/'TRANSFER.json',{'status':'complete','sha256_all_match':True,'frames':len(manifest['steps']),'point_count':manifest['point_count'],'remote':args.remote,'elapsed_seconds':time.perf_counter()-started})
        print('COMPLETE: rzeczywisty ruch dostępny w podglądzie po odświeżeniu.',flush=True)

if __name__=='__main__':main()
