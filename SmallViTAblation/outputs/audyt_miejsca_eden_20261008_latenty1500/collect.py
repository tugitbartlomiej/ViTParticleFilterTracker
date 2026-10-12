import subprocess,json,os,stat,datetime
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
def remote(cmd):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','eden-cluster',cmd],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    return dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
cmd='id; hostname; date -Is; lfs quota -u bpiotrowski /mnt/evafs; df -B1 /home2/faculty/bpiotrowski/smallvit_runs; squeue -u bpiotrowski -o "%.18i %.12T %.50j %.25R"; sacct -u bpiotrowski -S 2026-09-14 -X --parsable2 --format=JobID,JobName%80,State,Start,End,WorkDir%200 --noheader'
(OUT/'system_jobs.json').write_text(json.dumps(remote(cmd),ensure_ascii=False,indent=2),encoding='utf-8')
code=r'''
import os,stat,json,time
home='/home2/faculty/bpiotrowski'
roots=[home+'/smallvit_runs',home+'/.cache/pip',home+'/.cache/huggingface',home+'/.cache/torch']
uid=os.getuid(); files=[]; links=[]; skipped=[]; errors=[]; dirs=[]
protected={'datasets','dataset','wejscia','images','filmy','filmy3','obrysy','train','val','test','data'}
for root in roots:
 if not os.path.exists(root): continue
 dev=os.stat(root).st_dev
 for current,ds,fs in os.walk(root,followlinks=False):
  try: st=os.lstat(current)
  except OSError as e: errors.append([current,str(e)]); ds[:]=[]; continue
  if st.st_uid!=uid or st.st_dev!=dev: skipped.append([current,'owner/device']); ds[:]=[];continue
  dirs.append(dict(path=current,dev=st.st_dev,ino=st.st_ino,blocks=st.st_blocks*512))
  keep=[]
  for d in ds:
   p=os.path.join(current,d)
   try:
    s=os.lstat(p)
    if stat.S_ISLNK(s.st_mode):links.append(dict(path=p,target=os.readlink(p),realpath=os.path.realpath(p),blocks=s.st_blocks*512));continue
    if d.lower() in protected: skipped.append([p,'protected_source_no_scan']);continue
    if s.st_uid!=uid or s.st_dev!=dev:skipped.append([p,'owner/device']);continue
    keep.append(d)
   except OSError as e:errors.append([p,str(e)])
  ds[:]=keep
  for f in fs:
   p=os.path.join(current,f)
   try:
    s=os.lstat(p)
    if s.st_uid!=uid:skipped.append([p,'owner']);continue
    if stat.S_ISLNK(s.st_mode):links.append(dict(path=p,target=os.readlink(p),realpath=os.path.realpath(p),blocks=s.st_blocks*512));continue
    if not stat.S_ISREG(s.st_mode):continue
    files.append(dict(path=p,realpath=os.path.realpath(p),size=s.st_size,blocks=s.st_blocks*512,dev=s.st_dev,ino=s.st_ino,nlink=s.st_nlink,mtime=s.st_mtime))
   except OSError as e:errors.append([p,str(e)])
print(json.dumps(dict(timestamp=time.time(),roots=[dict(path=r,realpath=os.path.realpath(r)) for r in roots],files=files,dirs=dirs,symlinks=links,skipped=skipped,errors=errors)))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','eden-cluster','python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
(OUT/'inventory.stderr.txt').write_text(p.stderr,encoding='utf-8')
if p.returncode:raise RuntimeError(p.stderr)
data=json.loads(p.stdout)
(OUT/'inventory.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
files=data['files']; byrun={}
for f in files:
 rel=f['path'].split('/smallvit_runs/')
 if len(rel)<2:continue
 run=rel[1].split('/')[0]; byrun.setdefault(run,[]).append(f)
for run,items in sorted(byrun.items(),key=lambda t:sum(f['blocks'] for f in t[1]),reverse=True):
 print(run,round(sum(f['blocks'] for f in items)/1e9,3),len(items))
 print('\n'.join('  '+str(round(f['size']/1e9,4))+' '+f['path'].split('/smallvit_runs/')[1] for f in sorted(items,key=lambda x:x['size'],reverse=True)[:8]))
print('files',len(files),'symlinks',len(data['symlinks']),'skipped',data['skipped'],'errors',data['errors'])
