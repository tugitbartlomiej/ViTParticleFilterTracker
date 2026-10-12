"""Reproduce exact SymPy checks and a kernel-checked Lean proof; no network."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
LEAN=Path('C:/Users/bartl/.elan/toolchains/leanprover--lean4---v4.28.0/bin/lean.exe')
commands=[('sympy',[sys.executable,str(ROOT/'verify_memory.py')]),
          ('lean_version',[str(LEAN),'--version']),
          ('lean',[str(LEAN),str(ROOT/'DiscreteMemory.lean')])]
results={}
for name,command in commands:
    run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    results[name]={'command':command,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
results['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in ['DiscreteMemory.lean','verify_memory.py','run_verification.py']}
results['all_checks_passed']=all(results[name]['exit_code']==0 for name,_ in commands)
(ROOT/'verification_result.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'all_checks_passed':results['all_checks_passed'],'output':str(ROOT/'verification_result.json')}))
if not results['all_checks_passed']: sys.exit(1)
