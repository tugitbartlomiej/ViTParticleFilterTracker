from pathlib import Path
import json,hashlib,datetime
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names=['model.py','state.py','train.py','gpu_smoke.py','CONFIG.json','PLAN_MODELU.md','test_expanded.py','test_driver.py','test_distributed.py']
log=(HERE/'TEST_CPU_FINAL.log').read_text(encoding='utf-8')
assert 'Ran 19 tests' in log and log.rstrip().endswith('OK')
proofs={}
for world in [4,11]:
    data=json.loads((HERE/f'PROOF_CPU_GLOO_{world}.json').read_text())
    assert data['status']=='PASS'
    cases=[case for rank in data['cases'] for case in rank]
    proofs[str(world)]={'max_gradient_error':max(x['gradient_max_error'] for x in cases),'max_weights_error':max(x['weights_max_error'] for x in cases),'global_batches':[32,28]}
row={'phase':'code_CPU_validated_pending_remote_GPU_guard','updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'code_ready':True,'cpu_validated':True,'gpu_model_validated':False,'submitted_full_fit':False,'actual_training_steps':0,
    'cpu_tests':19,'CPU_log_sha256':sha(HERE/'TEST_CPU_FINAL.log'),'CPU_proofs':proofs,
    'hashes':{name:sha(HERE/name) for name in names},'scope':'new late-window anchored residual; no localGPU fitting',
    'runtime_cpu':'PyKoopman1.2.1, Torch2.1.2+cu121 on CPU; remoteGPU guard still required'}
(HERE/'CODE_READINESS.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
print(json.dumps(row,indent=2))
