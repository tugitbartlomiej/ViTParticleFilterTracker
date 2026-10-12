"""Czy finalne artefakty odtwarzają pomiar? Rozjazd hashów obala pochodzenie."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import ast, shutil
from datetime import datetime,timezone
import scipy,sklearn,timm
from scipy.stats import ttest_1samp
from dane import *
def main():
    r=read(OUT/'RESULT.json');a=read(OUT/'AUDIT.json');assert r['status']=='complete' and a['status']=='passed'
    rows,pairs,w,J,ids=load();g=np.load(OUT/'GRADIENTS_BASELINE.npz')['gradient_lift_covector']
    zero=(np.linalg.norm(g,axis=1)==0);centers=[]
    for i in ids['test']:
        x,y,bw,bh=rows[i]['bbox_source_480'];cx=int(np.clip(round((2*x+bw)/32-.5),0,29));cy=int(np.clip(round((2*y+bh)/32-.5),0,19));centers.append(cy*30+cx)
    assert np.array_equal(zero,np.array([w[i,t]==0 for i,t in zip(ids['test'],centers)]))
    for label in ['full768','J128']:
        f=np.load(OUT/f'TCAV_{label}.npz');scores=(f['repeat_sensitivities']>0).mean(1)
        p=float(ttest_1samp(scores,.5).pvalue)
        r['TCAV'][label]['paper_frame_unweighted_repeat_TCAV']=scores
        r['TCAV'][label]['paper_frame_unweighted_ttest_p']=p
        r['TCAV'][label]['paper_frame_unweighted_ttest_Bonferroni2']=min(1,2*p)
    js(OUT/'RESULT.json',r)
    p=read(OUT/'PROVENANCE.json');p['versions_additional']=dict(scipy=scipy.__version__,sklearn=sklearn.__version__,timm=timm.__version__);p['finalizer_source_sha256']=sha(HERE/'finalize.py');js(OUT/'PROVENANCE.json',p)
    for code in HERE.glob('*.py'):
        ast.parse(code.read_text(encoding='utf-8'));shutil.copy2(code,OUT/'EXECUTED_SOURCE'/code.name)
    for name in ['PLAN.md','README.md','WYNIKI.json']:shutil.copy2(HERE/name,OUT/'EXECUTED_SOURCE'/name)
    # Check initial executed model/dane/run source hashes still match archived files.
    for name,h in p['code_hashes'].items():assert sha(OUT/'EXECUTED_SOURCE'/name)==h
    files={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ['HASHES.json','PROGRESS.json','BIEG_tcav.json']}
    js(OUT/'HASHES.json',files)
    bie=read(OUT/'BIEG_tcav.json');bie['result_sha256']=sha(OUT/'RESULT.json');bie['audit_sha256']=sha(OUT/'AUDIT.json');bie['hashes_sha256']=sha(OUT/'HASHES.json');js(OUT/'BIEG_tcav.json',bie)
    progress=read(OUT/'PROGRESS.json');progress['audited']=True;progress['finalized_utc']=datetime.now(timezone.utc).isoformat();progress['hashes_sha256']=sha(OUT/'HASHES.json');js(OUT/'PROGRESS.json',progress)
    print('FINALIZED',len(files),'artifacts; zero sensitivity matches mask support; source snapshots verified',flush=True)
if __name__=='__main__':main()
