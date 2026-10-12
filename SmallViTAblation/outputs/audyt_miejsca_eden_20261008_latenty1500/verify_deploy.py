import subprocess,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
code=r'''
import tarfile,hashlib,os,json,pathlib
base='/home2/faculty/bpiotrowski/smallvit_runs'
targets=[(base+'/2026-10-08_vit_k8000_podmiana_1z256/deploy_bundle.tar.gz',[base+'/2026-10-08_vit_k8000_podmiana_1z256']), (base+'/code_bundle_vit8z256.tar.gz',[base+'/2026-10-08_vit_k8000_podmiana_8z256',base+'/2026-10-08_vit_k8000_podmiana_8z256/code'])]
out=[]
def sha(f):
 h=hashlib.sha256()
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for archive,roots in targets:
 before=os.stat(archive);rows=[]
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar.getmembers():
   if member.isdir():continue
   item=dict(name=member.name,size=member.size,verified_paths=[])
   if not member.isfile() or member.name.startswith('/') or '..' in pathlib.PurePosixPath(member.name).parts:
    item['error']='nonregular or unsafe member';rows.append(item);continue
   with tar.extractfile(member) as f:item['sha256']=sha(f)
   for root in roots:
    p=os.path.join(root,member.name)
    if not os.path.isfile(p) or os.path.islink(p):continue
    if os.stat(p).st_uid!=os.getuid() or os.stat(p).st_size!=member.size:continue
    if not os.path.realpath(p).startswith(os.path.realpath(root)+'/'):continue
    s0=os.stat(p)
    with open(p,'rb') as f:h=sha(f)
    s1=os.stat(p)
    if h==item['sha256'] and (s0.st_size,s0.st_mtime_ns)==(s1.st_size,s1.st_mtime_ns):item['verified_paths'].append(p)
   rows.append(item)
 after=os.stat(archive)
 out.append(dict(archive=archive,stable=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),all_regular_members_verified=all(x['verified_paths'] for x in rows),members=rows))
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','eden-cluster','python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
(OUT/'deploy.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stderr)
d=json.loads(r.stdout);(OUT/'deploy_verified.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print([(x['archive'],x['all_regular_members_verified'],len(x['members'])) for x in d]);print('unmatched',[(x['archive'],y['name']) for x in d for y in x['members'] if not y['verified_paths']][:20])
