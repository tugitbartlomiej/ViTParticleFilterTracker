"""Submit exactly two owned CPU jobs after a verified producer merge submission.

Unknown/partial submission outcomes are recorded and never blindly retried.
"""
import argparse,datetime,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path

ROOT=Path('/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009')
PRODUCER=Path('/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009')

def main():
    p=argparse.ArgumentParser();p.add_argument('--merge-job');a=p.parse_args()
    producer_file=PRODUCER/'submission/PRODUCTION_SUBMITTED.json'
    producer=json.loads(producer_file.read_text())
    if producer['status']!='submitted':raise RuntimeError('Producer submission not complete')
    actual=str(producer['jobs']['merge']).split(';')[0]
    if a.merge_job and a.merge_job!=actual:raise RuntimeError('Merge ID differs from producer marker')
    if not re.fullmatch(r'[0-9]+',actual):raise RuntimeError('Invalid merge job ID')
    marker=ROOT/'results/PRODUCTION_SUBMITTED.json'
    if marker.exists():
        old=json.loads(marker.read_text());print(json.dumps(old,indent=2))
        if old.get('status')=='submitted':return
        raise RuntimeError('Partial/unknown submission; inspect saved jobs, never auto-resubmit')
    # Complete local tests before making any new scheduler mutation.
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'.mpl-cache'))
    result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','test_pipeline.py'],cwd=ROOT/'code',env=env,text=True,capture_output=True)
    (ROOT/'results/TESTS_PRODUCTION_GATE.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Production gate tests failed')
    frozen=ROOT/'code_frozen_production';frozen.mkdir()
    hashes={}
    for file in (ROOT/'code').iterdir():
        if file.is_file() and file.suffix in ['.py','.slurm','.md']:
            destination=frozen/file.name;shutil.copy2(file,destination);hashes[file.name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    value={'status':'submitting','merge_job':actual,'producer_marker_sha256':hashlib.sha256(producer_file.read_bytes()).hexdigest(),'jobs':{},'commands':{},'source_hashes':hashes,'frozen_code':str(frozen),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with marker.open('x') as f:json.dump(value,f,indent=2)
    def update():
        temporary=marker.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(marker)
    for key,script in [('panel1024','panel.slurm'),('population900k','population.slurm')]:
        cmd=['sbatch','--parsable',f'--dependency=afterok:{actual}','--kill-on-invalid-dep=yes']
        cmd.append(str(frozen/script))
        if key=='panel1024':cmd.append(str(PRODUCER/'results/merged/TEMPORAL_PANEL_MANIFEST.json'))
        value['commands'][key]=cmd;update()
        out=subprocess.run(cmd,text=True,capture_output=True)
        value[key+'_submission_stdout']=out.stdout;value[key+'_submission_stderr']=out.stderr;update()
        if out.returncode:raise RuntimeError(f'Submission failed for {key}; marker preserved')
        job=out.stdout.strip().split(';')[0]
        if not re.fullmatch(r'[0-9]+',job):raise RuntimeError('Unknown submission response, marker preserved')
        value['jobs'][key]=job;update()
    value['status']='submitted';value['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();update()
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
