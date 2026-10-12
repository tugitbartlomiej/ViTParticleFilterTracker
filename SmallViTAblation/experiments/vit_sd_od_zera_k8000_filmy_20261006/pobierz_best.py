"""Resumable, bounded parallel download of the owned checkpoint; verify before publish."""
import concurrent.futures as futures
import hashlib, json, os, subprocess, time, socket
from contextlib import contextmanager
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_sd_od_zera_k8000_filmy_20261006'
DEST=OUT/'eden_kopia/trening/best_sd.pt'
PART=DEST.with_suffix('.pt.part')
STATE=OUT/'TRANSFER_BEST.json'
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy/wyniki/trening/best_sd.pt'
SHA='6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7'
SIZE=546367845
BLOCK=4*1024*1024
PORT=None

@contextmanager
def ssh_tunnel():
    # ProxyJump via Windows stdin/stdout stalled large transfers. A loopback-only
    # SSH forward uses the same gateway and credentials and keeps host checks.
    global PORT
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); PORT=sock.getsockname()[1]
    process=subprocess.Popen(['ssh','-n','-N','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes',
        '-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
        '-o','ServerAliveCountMax=3','-L',f'127.0.0.1:{PORT}:eden:22','eden-jump'],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if process.poll() is not None: raise RuntimeError('SSH forward exited')
            try:
                with socket.create_connection(('127.0.0.1',PORT),timeout=.2): pass
                break
            except OSError: time.sleep(.25)
        else: raise RuntimeError('SSH forward failed to start')
        yield
    finally:
        if process.poll() is None: process.terminate()
        process.wait(timeout=10)

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save(obj):
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf8')
    os.replace(tmp,STATE)

def fetch(i):
    n=min(BLOCK,SIZE-i*BLOCK)
    cmd=['ssh','-n','-p',str(PORT),'-o','HostKeyAlias=eden','-o','StrictHostKeyChecking=yes',
         '-o','BatchMode=yes','-o','ConnectTimeout=15',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','bpiotrowski@127.0.0.1',
         f'dd if={REMOTE} bs={BLOCK} skip={i} count=1 status=none']
    for attempt in range(3):
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        try:
            data,_=p.communicate(timeout=120)
            if p.returncode==0 and len(data)==n: return i,data
        except subprocess.TimeoutExpired:
            # Own SSH process and its ProxyJump child only. No other SSH sessions.
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
            p.kill()
        time.sleep(1)
    raise RuntimeError(f'Cannot retrieve chunk {i}')

def main():
    if DEST.exists() and DEST.stat().st_size==SIZE and digest(DEST)==SHA:
        print('Already verified',flush=True); return
    state=json.loads(STATE.read_text()) if STATE.exists() and PART.exists() else {}
    done=set(state.get('done',[])) if state.get('sha256')==SHA else set()
    if not done:
        with PART.open('wb') as f: f.truncate(SIZE)
    state=dict(phase='downloading',pid=os.getpid(),sha256=SHA,size=SIZE,done=sorted(done),started=time.time())
    save(state)
    count=(SIZE+BLOCK-1)//BLOCK
    errors=[]
    with ssh_tunnel(), futures.ThreadPoolExecutor(max_workers=1) as pool, PART.open('r+b') as f:
        jobs=[pool.submit(fetch,i) for i in range(count) if i not in done]
        for job in futures.as_completed(jobs):
            try:
                i,data=job.result()
            except Exception as exc:
                errors.append(str(exc)); state['errors']=errors; save(state)
                continue
            f.seek(i*BLOCK); f.write(data); f.flush()
            done.add(i); state.update(done=sorted(done),completed_bytes=sum(min(BLOCK,SIZE-j*BLOCK) for j in done),updated=time.time())
            save(state)
            if len(done)%8==0 or len(done)==count: print(f'{len(done)}/{count} chunks',flush=True)
    if errors:
        state.update(phase='incomplete',finished=time.time()); save(state)
        raise RuntimeError('; '.join(errors))
    actual=digest(PART)
    assert actual==SHA, f'Hash mismatch: {actual}'
    os.replace(PART,DEST)
    state.update(phase='verified',finished=time.time(),local_sha256=actual)
    save(state); print('SHA256 verified',flush=True)

if __name__=='__main__': main()
