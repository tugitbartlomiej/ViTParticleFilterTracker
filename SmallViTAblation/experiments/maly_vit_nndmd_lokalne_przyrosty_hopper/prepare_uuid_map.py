"""Use allocation-visible NVML MIG UUIDs; reject wrong count/profile instead of using hidden devices."""
from pathlib import Path
import json,os,re,subprocess
def mapping(text):
 parents=[];current=None
 for line in text.splitlines():
  gpu=re.match(r'GPU (\d+):.*\(UUID: (GPU-[^)]+)\)',line)
  mig=re.match(r'\s+MIG (\S+)\s+Device\s+\d+:\s+\(UUID: (MIG-[^)]+)\)',line)
  if gpu:current=dict(physical_index=int(gpu[1]),uuid=gpu[2],children=[]);parents.append(current)
  elif mig:
   assert current is not None;current['children'].append(dict(uuid=mig[2],profile=mig[1],physical_index=current['physical_index']))
 full=[dict(uuid=p['uuid'],profile='full_h200',physical_index=p['physical_index']) for p in parents if not p['children']]
 children=[c for p in parents for c in p['children']]
 assert len(full)==4 and sum(c['profile']=='4g.71gb' for c in children)==4 and sum(c['profile']=='3g.71gb' for c in children)==3,'NVML allocation-visible profile counts must equal Slurm request4full+4MIG4g+3MIG3g'
 rows=full+children;assert len(rows)==len(set(r['uuid'] for r in rows))==11
 return rows
if __name__=='__main__':
 text=subprocess.check_output(['nvidia-smi','-L'],text=True);rows=mapping(text)
 path=Path(os.environ['INFRA_OUTPUT'])/'UUID_MAP.json'
 path.write_text(json.dumps(dict(devices=rows,nvidia_smi_L=text,slurm_job_id=os.environ['SLURM_JOB_ID'],slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),original_cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),scope='allocation-visible NVML enumeration; inaccessible MIG instances excluded'),indent=2),encoding='utf-8')
 print(path.read_text(),flush=True)
