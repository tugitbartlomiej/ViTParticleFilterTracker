"""Upload code and the exact frozen checkpoint; launch one SLURM job only on request."""
import subprocess,json,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs'/HERE.name
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy'
OLD='/home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy'
PY='/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'
CK=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
def call(args):
    p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',timeout=180)
    if p.returncode: raise RuntimeError(p.stderr+'\n'+p.stdout)
    return p.stdout.strip()
def ssh(command): return call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','eden-cluster',command])
OUT.mkdir(parents=True,exist_ok=True)
files=['run.py','model.py','model_snapshot.py','dane.py','build_manifest.py','trening.slurm']
for n in files:
    if not (HERE/n).is_file():raise FileNotFoundError(HERE/n)
call(['scp',*[str(HERE/n) for n in files],f'eden-cluster:{REMOTE}/kod/'])
call(['scp',str(CK),f'eden-cluster:{REMOTE}/wejscia/MODEL_VIT_K1000.pt'])
manifest=ssh(f'{PY} -B {REMOTE}/kod/build_manifest.py --video-dir /home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/videos --annotation-dir /home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations --split-manifest {OLD}/wejscia/pool_manifest.jsonl --output {REMOTE}/wejscia/FILMY.json --include-val')
hash_remote=ssh(f'sha256sum {REMOTE}/wejscia/MODEL_VIT_K1000.pt').split()[0]
assert hash_remote==hashlib.file_digest(CK.open('rb'),'sha256').hexdigest()
call(['scp',f'eden-cluster:{REMOTE}/wejscia/FILMY.json',str(OUT/'FILMY.json')])
result=dict(remote_root=REMOTE,manifest_build=manifest,checkpoint_sha256=hash_remote,code_sha256={n:hashlib.file_digest((HERE/n).open('rb'),'sha256').hexdigest() for n in files})
if '--submit' in sys.argv:
    if (OUT/'SUBMISSION.json').exists():raise RuntimeError('Submission record exists: inspect it instead of launching a duplicate')
    result['job_id']=ssh(f'sbatch --parsable {REMOTE}/kod/trening.slurm')
    (OUT/'SUBMISSION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
else:(OUT/'DEPLOYMENT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
