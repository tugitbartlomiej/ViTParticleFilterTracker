"""One supplementary submission; does not modify existing code or job markers."""
import hashlib,json,os,re,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009')
PRODUCER=Path('/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    marker=ROOT/'results/SUPPLEMENT_DMDC_SUBMITTED.json'
    if marker.exists():
        record=json.loads(marker.read_text());print(json.dumps(record,indent=2))
        if record.get('status')=='submitted':return
        raise RuntimeError('Partial/unknown outcome; never auto-resubmit')
    producer=json.loads((PRODUCER/'submission/PRODUCTION_SUBMITTED.json').read_text())
    if producer.get('status')!='submitted' or str(producer['jobs']['merge']).split(';')[0]!='1825166':raise RuntimeError('Unexpected producer merge identity')
    original=ROOT/'results/PRODUCTION_SUBMITTED.json';original_hash=sha(original);existing=json.loads(original.read_text())
    def old_unchanged():
        return sha(original)==original_hash and all(sha(ROOT/'code_frozen_production'/name)==expected for name,expected in existing['source_hashes'].items())
    if not old_unchanged():raise RuntimeError('Existing frozen source checksum mismatch before supplement')
    stage=ROOT/'supplement_stage'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'.mpl-cache'))
    tests=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','test_supplement.py'],cwd=stage,env=env,capture_output=True,text=True)
    testlog=ROOT/'results/TESTS_READY_POPULATION_DMDC_SUPPLEMENT.txt';testlog.write_text(tests.stdout+tests.stderr)
    if tests.returncode:raise RuntimeError('Supplement gate failed; no job submitted')
    subprocess.run(['bash','-n',str(stage/'population_dmdc.slurm')],check=True)
    frozen=ROOT/'code_frozen_population_dmdc_supplement';frozen.mkdir()
    hashes={}
    for f in stage.iterdir():
        if f.is_file() and f.suffix in ['.py','.slurm','.md']:
            shutil.copy2(f,frozen/f.name);hashes[f.name]=sha(frozen/f.name)
    cmd=['sbatch','--parsable','--dependency=afterok:1825166','--kill-on-invalid-dep=yes',str(frozen/'population_dmdc.slurm')]
    record={'status':'submitting','dependency':'afterok:1825166','commands':[cmd],'snapshot':str(frozen),'source_hashes':hashes,'existing_frozen_production_unchanged_before':old_unchanged(),
      'existing_job_ids':existing['jobs'],'tests_log':str(testlog),'tests_log_sha256':sha(testlog),'output_dir':str(ROOT/'results/population900k_dmdc_supplement'),'started_utc':datetime.now(timezone.utc).isoformat()}
    with marker.open('x') as f:json.dump(record,f,indent=2)
    def update():
        temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(marker)
    result=subprocess.run(cmd,capture_output=True,text=True)
    record.update(sbatch_stdout=result.stdout,sbatch_stderr=result.stderr,sbatch_returncode=result.returncode);update()
    if result.returncode:raise RuntimeError('Submission failed; marker preserved')
    job=result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'[0-9]+',job):raise RuntimeError('Unknown scheduler response; no retry')
    record.update(status='submitted',job_id=job,existing_frozen_production_unchanged_after=old_unchanged(),finished_utc=datetime.now(timezone.utc).isoformat());update()
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
