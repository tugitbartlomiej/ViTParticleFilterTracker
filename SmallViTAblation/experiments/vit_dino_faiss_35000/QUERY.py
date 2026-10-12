"""Wyszukuje top32 dla prawdziwego obrazu query; nie używa celu translatora."""
import argparse,json
from pathlib import Path
import worker as w

def main():
    p=argparse.ArgumentParser(); p.add_argument('image',type=Path); p.add_argument('--topk',type=int,default=32); a=p.parse_args()
    if not 1<=a.topk<=32: p.error('topk must be 1..32')
    config=json.loads((w.OUT/'CONFIG.json').read_text()); w.CONFIG_HASH=config['config_hash']; w.AMP=config['precision']=='amp_fp16'
    assert w.sha(w.OUT/'manifest.jsonl')==config['manifest_sha256']
    assert config['spec']==w.SPEC
    w.torch.set_num_threads(4); w.faiss.omp_set_num_threads(4)
    w.torch.backends.cuda.matmul.allow_tf32=False; w.torch.backends.cudnn.allow_tf32=False
    w.torch.cuda.set_per_process_memory_fraction(2*1024**3/w.torch.cuda.get_device_properties(0).total_memory,0)
    w.gpu_wait(); model=w.load_model('cuda')
    rows=[json.loads(s) for s in (w.OUT/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    index=w.faiss.read_index(str(w.OUT/'index.faiss')); assert len(rows)==index.ntotal==35000
    result,_=w.export_query(model,index,rows,a.image,a.topk)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
