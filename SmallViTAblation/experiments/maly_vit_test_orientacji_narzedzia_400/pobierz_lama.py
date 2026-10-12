"""Czy dostępny ONNX LaMa ma rzeczywiste wejścia zgodne z inpaintingiem?
Brak pliku, niezgodne IO lub niesprawdzony hash obala gotowość runtime.
"""
import json,urllib.request,hashlib,time,os,shutil
from pathlib import Path
import onnxruntime as ort

HERE=Path(__file__).resolve().parent
os.environ['HF_HOME']='F:/HuggingFace'
from huggingface_hub import hf_hub_download
M=HERE/'model_lama';M.mkdir(exist_ok=True)
repo='edgetools/lama';info=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}'))
rev=info['sha'];record=dict(repo=repo,revision=rev,license='apache-2.0',files={},hf_home=os.environ['HF_HOME'])
for name in ('README.md','LICENSE','lama_fp32.onnx'):
    url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
    old=M/name
    if name=='lama_fp32.onnx' and old.exists():
        # Recover the completed initial download without transferring weights twice.
        digest=hashlib.sha256(old.read_bytes()).hexdigest()
        cached=Path(os.environ.get('HF_HUB_CACHE','F:/HuggingFace/hub'))/'models--edgetools--lama'
        blob=cached/'blobs'/digest;pointer=cached/'snapshots'/rev/name
        blob.parent.mkdir(parents=True,exist_ok=True);pointer.parent.mkdir(parents=True,exist_ok=True)
        if blob.exists():raise RuntimeError('Existing blob: review before moving owned download')
        shutil.move(str(old),str(blob));os.link(blob,pointer)
        print('moved completed download into HF cache',pointer,flush=True)
    out=Path(hf_hub_download(repo_id=repo,filename=name,revision=rev))
    record['files'][name]=dict(sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,url=url,cache_path=str(out))
opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
session=ort.InferenceSession(record['files']['lama_fp32.onnx']['cache_path'],sess_options=opt,providers=['CPUExecutionProvider'])
record.update(ort_version=ort.__version__,available_providers=ort.get_available_providers(),
    inputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_inputs()],
    outputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_outputs()])
(M/'MODEL.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
