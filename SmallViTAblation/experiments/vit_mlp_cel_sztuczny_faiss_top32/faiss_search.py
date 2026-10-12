"""Isolated FAISS process: never import torch (Windows OpenMP runtimes)."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json
from pathlib import Path
import numpy as np,faiss
faiss.omp_set_num_threads(4)
ROOT=Path(__file__).resolve().parents[2]
request=json.loads(sys.stdin.read())
features=np.load(ROOT/'outputs/vit_dino_faiss_35000/features.npy',mmap_mode='r')
assert features.shape==(35000,384) and np.isfinite(features).all()
norm=np.linalg.norm(features,axis=1);assert np.max(abs(norm-1))<2e-5
index=faiss.IndexFlatIP(384);index.add(np.ascontiguousarray(features))
queries=np.array(request['queries'],dtype=np.float32)
assert queries.shape==(2,384) and np.max(abs(np.linalg.norm(queries,axis=1)-1))<2e-5
scores,ids=index.search(queries,32)
qa=[]
for q,s in zip(queries,scores):
    brute=np.asarray(features)@q;expected=np.sort(brute)[-32:][::-1]
    error=float(np.max(abs(expected-s)));assert error<3e-6;qa.append(error)
self_scores,self_ids=index.search(np.asarray(features[:1]),1);assert abs(float(self_scores[0,0])-1)<2e-5
print(json.dumps(dict(scores=scores.tolist(),ids=ids.tolist(),faiss_bruteforce_max_errors=qa,self_query_passed=True,index_total=index.ntotal,faiss_version=faiss.__version__)))
