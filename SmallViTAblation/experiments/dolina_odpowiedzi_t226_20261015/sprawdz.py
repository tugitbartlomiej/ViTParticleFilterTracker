"""Niezależna kontrola geometrii na rzeczywistym module modelu i dekodowanie MP4."""
from pathlib import Path
import importlib.util
import json
import subprocess

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/dolina_odpowiedzi_t226_20261015'
torch.set_num_threads(4)
dev='cuda' if torch.cuda.is_available() else 'cpu'
d=np.load(OUT/'geometria.npz')
spec=importlib.util.spec_from_file_location('maly_model_kontrola',ROOT/'experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
net=m.MalyViT().to(dev).eval()
st=torch.load(ROOT/'assets/MalyViT_35000_od_zera_s47/wagi/k65640.pt',map_location=dev,weights_only=True)
net.load_state_dict(st.get('net',st),strict=True)
actual_state=[]
hook=net.norm.register_forward_pre_hook(lambda mod,args:actual_state.append(args[0].detach()))
with torch.no_grad():
    o,_=net(torch.tensor(d['rgb'].copy(),device=dev).permute(2,0,1)[None].float()/255)
hook.remove()
raw=actual_state[0][0,226].cpu().numpy()
cached_rel=float(np.linalg.norm(raw-d['x0'])/np.linalg.norm(raw))
assert cached_rel<.001
# Rachunek w podwójnej precyzji, lecz na oryginalnych modułach i ich epsilonach.
net=net.double()
for p in net.parameters():p.requires_grad_(False)
head=lambda x:net.glowa(net.norm(x))
xs=torch.tensor(d['xstar'],device=dev,dtype=torch.float64)
j=torch.autograd.functional.jacobian(head,xs).detach().cpu().numpy()
assert np.max(abs(j-d['J']))<1e-12
N=d['N']
assert N.shape==(192,187)
orth=float(np.max(abs(N.T@N-np.eye(187))))
null=float(np.linalg.norm(j@N,axis=0).max())
assert orth<1e-12 and null<1e-12
sheet=d['sheet_x'].reshape(-1,192)
path=d['path_x']
with torch.no_grad():
    sheet_o=head(torch.tensor(sheet,device=dev)).cpu().numpy()
    path_o=head(torch.tensor(path,device=dev)).cpu().numpy()
sheet_error=float(np.max(abs(sheet_o-d['target'])))
path_error=float(np.max(abs(path_o-d['target'])))
assert sheet_error<1e-11 and path_error<1e-11
rank_samples=[]
for idx in (0,16,32,528,544,560,1056,1072,1088):
    x=torch.tensor(sheet[idx],device=dev)
    jj=torch.autograd.functional.jacobian(head,x)
    s=torch.linalg.svdvals(jj)
    rank_samples.append(int((s>s[0]*1e-6).sum()))
assert rank_samples==[5]*9
# Weryfikacja IoU niezależną formułą xyxy.
uv=d['uv'];N0=N[:,:2]
uu,vv=np.meshgrid(uv,uv)
dot=(sheet-d['xstar'])@N0
assert np.max(abs(dot[:,0]-uu.ravel()))<1e-10
assert np.max(abs(dot[:,1]-vv.ravel()))<1e-10
vol=d['volume_metrics'];max_vol_error=0.
for delta_idx in (0,8,16,24,32):
    x=sheet[::17]+d['volume_delta'][delta_idx]*d['normal']
    with torch.no_grad():oo=head(torch.tensor(x,device=dev)).cpu().numpy()
    p=1/(1+np.exp(-oo[:,0]));ltrb=np.exp(oo[:,1:])*16
    boxes=np.stack([264-ltrb[:,0],120-ltrb[:,1],264+ltrb[:,2],120+ltrb[:,3]],axis=1)
    gt=d['gt']
    inter=np.maximum(0,np.minimum(boxes[:,2:],gt[2:])-np.maximum(boxes[:,:2],gt[:2])).prod(1)
    areas=(boxes[:,2:]-boxes[:,:2]).prod(1);agt=(gt[2:]-gt[:2]).prod()
    iou=inter/(areas+agt-inter)
    max_vol_error=max(max_vol_error,float(abs(iou-vol[delta_idx,::17,1]).max()),float(abs(p-vol[delta_idx,::17,0]).max()))
assert max_vol_error<1e-10
probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_frames','-show_entries','format=duration,size','-of','json',str(OUT/'dolina_odpowiedzi_t226.mp4')],capture_output=True,text=True,check=True)
video=json.loads(probe.stdout)
stream=video['streams'][0]
assert stream['width']==1920 and stream['height']==1080 and int(stream['nb_frames'])==1500
assert float(video['format']['duration'])==50.
decode=subprocess.run(['ffmpeg','-v','error','-i',str(OUT/'dolina_odpowiedzi_t226.mp4'),'-f','null','-'],capture_output=True,text=True,check=True)
assert not decode.stderr.strip()
report=dict(original_module_load='strict=True',cached_state_relative_difference=cached_rel,
            baseline_probability_actual=float(o[0,7,16,0].sigmoid()),orthonormal_basis_error=orth,
            max_JN_norm=null,sheet_max_output_error=sheet_error,path_max_output_error=path_error,
            rank_on_nine_surface_points=rank_samples,independent_volume_metrics_error=max_vol_error,
            video=video,full_video_decode='passed, no errors')
(OUT/'WERYFIKACJA.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
