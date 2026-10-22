import os, sys, time
from pathlib import Path
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import torch, lasery_dane as LD
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
p = Path(h.wagi) / f'k{int(h.kroki[300]):05d}.pt'; print(p, p.stat().st_size / 1e6, 'MB')
ks = [int(k) for k in h.kroki[300:340]]
t0 = time.perf_counter(); sts = [torch.load(Path(h.wagi) / f'k{k:05d}.pt', map_location='cpu', weights_only=True) for k in ks]; print('load', (time.perf_counter()-t0)/40*1000, 'ms')
print(type(sts[0]), list(sts[0].keys())[:5])
M = LD.modul_modelu(); net = M.MalyViT().cuda().eval().requires_grad_(False)
rgb = torch.from_numpy(h.rgb.copy()).permute(2,0,1)[None].float().div(255).cuda()
t0 = time.perf_counter()
for st in sts: net.load_state_dict(st.get('net', st), strict=True)
torch.cuda.synchronize(); print('load_state_dict', (time.perf_counter()-t0)/40*1000, 'ms')
t0 = time.perf_counter()
for st in sts:
    with torch.no_grad(): net(rgb)
torch.cuda.synchronize(); print('forward', (time.perf_counter()-t0)/40*1000, 'ms')
sd = [{k: v.cuda() for k, v in st.get('net', st).items()} for st in sts[:5]]
t0 = time.perf_counter()
for s in sd: net.load_state_dict(s, strict=True)
torch.cuda.synchronize(); print('load_state_dict z GPU', (time.perf_counter()-t0)/5*1000, 'ms')
