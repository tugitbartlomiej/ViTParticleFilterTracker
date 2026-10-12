import os,json,runpy
from pathlib import Path
p=Path(os.environ['INFRA_OUTPUT'])/'UUID_MAP.json'
m=json.loads(p.read_text(encoding='utf-8')); local=int(os.environ['LOCAL_RANK'])
os.environ['CUDA_VISIBLE_DEVICES']=m['devices'][local]['uuid']
runpy.run_path(str(Path(__file__).with_name('gpu_smoke.py')),run_name='__main__')
