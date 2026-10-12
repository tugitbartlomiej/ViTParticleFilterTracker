"""Local submission evidence only; no cluster submission or NN execution."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,json,hashlib,shutil
HERE=Path(__file__).resolve().parent
def main(job,state):
 now=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()
 run='/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_multigpu'
 for name in ['submit_status.json','EDEN_PROVENANCE.json']:
  old=HERE/name;archive=HERE/'failure_1819551'/('LOCAL_'+name)
  if old.exists() and not archive.exists():shutil.copyfile(old,archive)
 row=dict(job_id=str(job),status=state,submitted=True,updated=now,remote_run=run,
  stdout=run+'/slurm-'+str(job)+'.out',stderr=run+'/slurm-'+str(job)+'.err',
  GPUs=2,single_node=True,GPU_type='full A100, DGX only',global_batch_images=32,tail_images=28,
  actual_training_steps=0,phase='submitted; scratch/source extraction and training steps not yet confirmed',
  previous_failed_jobs=['1819498','1819551'],test_jobs={'infrastructure':'1819567 COMPLETED','model_equivalence':'1819570 COMPLETED PASS'},
  guards=['final core synthetic GPU600','parallel source0..10','actual source-data 1vs2 global32/tail28'],
  final_hashes_sha256=hashlib.sha256((HERE/'run_multigpu/EXECUTION_FREEZE.sha256').read_bytes()).hexdigest())
 for p in [HERE/'submit_status.json',HERE/'run_multigpu/SUBMISSION.json']:
  p.write_text(json.dumps(row,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 old=json.loads((HERE/'EDEN_PROVENANCE.json').read_text(encoding='utf-8-sig'))
 old.update(job_id=str(job),remote_run=run,multigpu_submission=row,remote_tests='15 prior PASS + corrected fixture targeted PASS',
  runtime_compatibility_note='Actual2A100 NCCL model loss/gradient/Adam/telemetry/checkpoint proof passed; full source-data guard before fitting',
  remote_final_hash_check='18/18 PASS',updated=now)
 (HERE/'EDEN_PROVENANCE.json').write_text(json.dumps(old,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 status=json.loads((HERE/'status.json').read_text(encoding='utf-8-sig'))
 status.update(job_id=str(job),phase='pending_resources' if state=='PENDING' else 'scheduled_preflight',
  logpath=row['stdout'],world_size=2,actual_training_steps=0,actual_preflight_optimizer_steps=0,
  previous_failed_jobs=['1819498','1819551'],updated=now)
 (HERE/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(row))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--state',default='PENDING');a=p.parse_args();main(a.job,a.state)
