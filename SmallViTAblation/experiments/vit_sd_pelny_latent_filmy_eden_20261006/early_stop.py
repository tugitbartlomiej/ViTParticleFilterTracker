"""Preserve best held-out checkpoint and stop persistent validation deterioration.

Only reads scalar validation reports; no images or activations are saved.
The watcher waits for the exact evaluated checkpoint to be published atomically.
"""
import argparse,json,os,time,shutil,traceback
from pathlib import Path

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,indent=2,allow_nan=False))
    os.replace(t,p)

def update(state,step,score,patience=4,relative=.03,min_step=2000):
    if not isinstance(score,(int,float)) or not 0 <= score < float('inf'):
        raise ValueError('Validation score must be finite and nonnegative')
    improved=state.get('best_score') is None or score < state['best_score']
    if improved:
        state.update(best_score=score,best_step=step,bad_evaluations=0)
    elif step >= min_step and score > state['best_score']*(1+relative):
        state['bad_evaluations']=state.get('bad_evaluations',0)+1
    else:state['bad_evaluations']=0
    state.update(last_eval_step=step,last_score=score)
    return improved,state['bad_evaluations']>=patience

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,default=15)
    a=p.parse_args();out=a.output
    cfg=dict(patience=4,relative=.03,min_step=2000,metric='val_film_mean_mse')
    state=read(out/'EARLY_STOP.json') or dict(best_score=None,last_eval_step=-1,bad_evaluations=0)
    state.update(pid=os.getpid(),state='watching',config=cfg)
    write(out/'EARLY_STOP.json',state)
    try:
        while True:
            status=read(out/'STATUS.json') or {}
            report=read(out/'EVAL.json')
            if report and report['step']>state['last_eval_step']:
                step=report['step'];checkpoint=out/f'checkpoint_{step:08d}.pt'
                if checkpoint.is_file():
                    improved,stop=update(state,step,report[cfg['metric']],cfg['patience'],cfg['relative'],cfg['min_step'])
                    if improved:
                        temp=out/'best_checkpoint.tmp';temp.unlink(missing_ok=True)
                        try:os.link(checkpoint,temp)
                        except OSError:shutil.copyfile(checkpoint,temp)
                        os.replace(temp,out/'best_checkpoint.pt')
                        write(out/'BEST_VALIDATION.json',dict(step=step,metric=cfg['metric'],score=state['best_score'],checkpoint='best_checkpoint.pt',evaluated_films=report['val_films'],samples=report['val_samples']))
                    with (out/'EARLY_STOP_HISTORY.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=step,score=state['last_score'],best_step=state['best_step'],best_score=state['best_score'],bad_evaluations=state['bad_evaluations'],improved=improved))+'\n')
                    if stop:
                        (out/'STOP').touch()
                        state.update(state='stop_requested',reason='Four consecutive evaluated scores >3% worse than best, after step 2000; finish current update and save')
                        write(out/'EARLY_STOP.json',state)
                        print(json.dumps(state),flush=True)
                        return
                    write(out/'EARLY_STOP.json',state)
            if status.get('state') in ('complete','stopped','failed'):
                state.update(state='finished',training_state=status['state']);write(out/'EARLY_STOP.json',state)
                return
            if (out/'STOP').exists():
                state.update(state='external_stop');write(out/'EARLY_STOP.json',state);return
            time.sleep(a.interval)
    except Exception:
        state.update(state='failed',error=traceback.format_exc());write(out/'EARLY_STOP.json',state);raise

if __name__=='__main__':main()
