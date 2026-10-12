import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import json,hashlib,numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'maly_vit_langevin_kierunki_tozsamosci'
SOURCE=HERE.parent/'maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def js(p,x):
 def conv(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,Path):return str(v)
  raise TypeError(type(v))
 Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=conv,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
