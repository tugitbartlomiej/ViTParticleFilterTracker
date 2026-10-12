import subprocess,json,time
from pathlib import Path
O=Path(__file__).resolve().parent
args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=1','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','eden-cluster','timeout 55s python3 -B -']
start=time.monotonic()
try:
    r=subprocess.run(args,input=(O/'remote_inventory.py').read_bytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=58)
    (O/'SSH_STATUS.json').write_text(json.dumps({'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'timeout_seconds':58,'stderr':r.stderr.decode(errors='replace')},indent=2),encoding='utf-8')
    if r.returncode: raise RuntimeError('Remote inventory failed; see SSH_STATUS.json')
    d=json.loads(r.stdout)
    (O/'REMOTE_INVENTORY.json').write_text(json.dumps(d,separators=(',',':')),encoding='utf-8')
    print('REMOTE',d['image_file_count'],'tar images;',len(d['coco']['train']['images']),'checkpoint train;',len(d['index']['rows']),'index rows',flush=True)
except subprocess.TimeoutExpired:
    (O/'SSH_STATUS.json').write_text(json.dumps({'status':'bounded_timeout','timeout_seconds':58}),encoding='utf-8')
    raise
