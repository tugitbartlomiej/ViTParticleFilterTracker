"""Read-only numeric audit of downloaded smoke, no fitting and no test retuning."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    prov=json.loads((root/'PROVENANCE.json').read_text())
    checks={}
    for name,expected in prov['source_hashes'].items():
        file=root/'code_snapshot'/name
        checks[name]=file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==expected
    if not all(checks.values()):raise RuntimeError(('Snapshot mismatch',checks))
    rows=list(csv.DictReader((root/'METRICS_PER_TRAJECTORY.csv').open(encoding='utf-8')))
    selected=[r for r in rows if r['split']=='test' and r['phase']=='fit' and r['horizon']=='1' and r['method']=='pydmd_learnedB']
    energy=defaultdict(float)
    for r in selected:energy[r['image']]+=float(r['rmse'])**2*int(r['n'])
    angles=json.loads((root/'DIRECTION_CHANGES.json').read_text());turning={}
    for ph in [0,1]:
        part=[r for r in angles if r['phase']==ph];n=sum(r['n_active_pairs'] for r in part)
        turning[ph]={'active_pairs':n,'mean_cos':sum((r['cos_mean'] or 0)*r['n_active_pairs'] for r in part)/n,'reversal_fraction':sum((r['reversal_fraction'] or 0)*r['n_active_pairs'] for r in part)/n}
    inputs=json.loads((root/'INPUTS.json').read_text());counts=defaultdict(int)
    for r in inputs:counts[r['analysis_split']+'|'+r['metadata']['status']]+=1
    out={'source_snapshots_match_executed_hashes':checks,'test_fit_learnedB_sse_share_by_image':{k:v/sum(energy.values()) for k,v in energy.items()},'full768_step_direction_changes':turning,'split_outcome_counts':dict(counts),'evaluation_role_masks':sorted(set(r['role_mask'] for r in rows)),'interpretation':'No targets in old panel; no failure in validation/test; model hyperparameters not changed based on test.'}
    (root/'AUDIT_SMOKE.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
