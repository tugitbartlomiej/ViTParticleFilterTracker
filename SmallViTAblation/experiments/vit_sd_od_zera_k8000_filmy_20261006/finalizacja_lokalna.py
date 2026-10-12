"""One local transfer/verification process. Never submits or resumes any training."""
import os,sys,subprocess,json,time,traceback,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/vit_sd_od_zera_k8000_filmy_20261006'
STATE=OUT/'FINALIZACJA_LOKALNA.json'

def save(phase,**kw):
    p=STATE.with_suffix('.tmp')
    p.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw),indent=2),encoding='utf8')
    os.replace(p,STATE)

def main():
    import msvcrt
    lock=(OUT/'.finalizacja.lock').open('a+b')
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        print('Another finalization process already owns the lock',flush=True)
        return
    env=os.environ.copy()
    env.update(PYTHONPATH='F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/deps',
        USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    save('transfer')
    subprocess.run([sys.executable,'-B',str(HERE/'pobierz_best.py')],env=env,check=True)
    save('natural_local_test')
    # Test keeps preview bytes in RAM. The parent waits and owns no GPU resources.
    subprocess.run([sys.executable,'-B',str(HERE/'test_lokalny_ram.py')],env=env,check=True)
    save('test_process_exited')

if __name__=='__main__':
    try: main()
    except Exception:
        save('failed',error=traceback.format_exc()); raise
