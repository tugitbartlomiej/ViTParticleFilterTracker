"""Read-only verification of completed output; writes checksums and QA summary."""
import argparse,hashlib,json,os
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args();o=a.folder
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
runtime=json.loads((o/'RUNTIME.json').read_text());metrics=json.loads((o/'METRICS.json').read_text())
checks={}
checks['executed_snapshot_hash_matches']=sha(o/'code/run.py')==runtime['code_sha256']
checks['runtime_complete']=runtime['status']=='complete'
checks['one_result_per_model_split_group_horizon']=len({(x['split'],x['group'],x['horizon'],x['model']) for x in metrics})==len(metrics)
checks['nonnegative_rmse']=all(x['rmse']>=0 for x in metrics)
checks['skill_recomputes']=all(abs(x['skill']-(1-x['sse']/x['persistence_sse']))<1e-12 for x in metrics)
checks['test_origin_counts']=all(x['n_origins']==201-x['horizon'] for x in metrics if x['split']=='test')
assert all(checks.values()),checks
files=[o/'RUNTIME.json',o/'MODEL_SELECTION_BEFORE_TEST.json',o/'METRICS.csv',o/'REDUCED_TRAJECTORIES.npz',o/'POD_MODES.npy',o/'code/run.py',o/'code/report.py']
files += [o.parent/'source_eden'/n for n in ['Z.npy','S_nowa.npy','metadata.json']]
manifest=[]
for p in files:
    digest=sha(p);manifest.append({'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':digest});print(p.name,digest,flush=True)
(o/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(o/'FINAL_QA.json').write_text(json.dumps({'status':'passed','checks':checks,'n_metric_rows':len(metrics),
 'source_remote':'eden-cluster:/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/wyniki/pelny/dane',
 'batch_source':'ACTUAL_BATCH_SEQUENCE.npz reconstructed from continuous RNG47, annotations order1500',
 'process_status':'Numerical run completed; source training was not repeated.'},indent=2)+'\n')
