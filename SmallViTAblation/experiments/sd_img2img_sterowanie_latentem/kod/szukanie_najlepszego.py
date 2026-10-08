"""Najbliższe klatki puli TRAIN (DINOv2 CLS, jak etap 11) dla najlepszego obrazu optymalizacji warunku (g148) i dla
zdjęcia celu; dla każdej znalezionej klatki latent ViT K8000 i jej postęp wzdłuż δ = z* − z0 (z0 = zdjęcie celu).
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/szukanie_najlepszego.py [g148]
"""
import io, json, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
KAT = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_warunku'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
VIT = ROOT / 'assets' / 'MalyViT_K8000_s47_losowe' / 'MALY_VIT_K8000_s47_losowe.pth'
CEL_J, N_TOP = 10389, 5
OBRAZ = sys.argv[1] if len(sys.argv) > 1 else 'g148'
sys.path.insert(0, str(ROOT / 'app'))
import cv2, numpy as np, torch
import torchvision.transforms as T
from PIL import Image
import korekta_latentu as KL

MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def klatka(j):
    with zipfile.ZipFile(MAN[j]['archive']) as z:
        im = np.asarray(Image.open(io.BytesIO(z.read(MAN[j]['member']))).convert('RGB'))
    return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)

def main():
    spec = json.loads((DINO / 'CONFIG.json').read_text(encoding='utf-8'))['spec']
    model = torch.hub.load(spec['hub_repo'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(spec['weights'], map_location='cpu')); model = model.cuda().eval()
    PRE = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485, .456, .406), (.229, .224, .225))])
    Fp = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32); Fp /= np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12
    @torch.inference_mode()
    def cls(rgb):
        f = model.forward_features(PRE(Image.fromarray(np.ascontiguousarray(rgb))).unsqueeze(0).cuda())['x_norm_clstoken'][0].float().cpu().numpy()
        return f / (np.linalg.norm(f) + 1e-12)
    st = KL.Stanowisko(VIT)
    W2 = st.blok.mlp.fc2.weight.detach().float()
    L = np.load(KAT / 'LATENTY.npz')
    z0 = torch.from_numpy(L['z0'].astype(np.float32)).cuda(); zs = torch.from_numpy(L['z_sztuczny'].astype(np.float32)).cuda()
    d = zs - z0; d_w = d @ W2.T
    def postep(rgb):
        z = st.przebieg(rgb)['z']; dz = z - z0
        return dict(ulamek=float((dz * d).sum() / (d * d).sum()), ulamek_widoczny=float(((dz @ W2.T) * d_w).sum() / (d_w * d_w).sum()),
                    cos_przyrost=float((dz * d).sum() / (dz.norm() * d.norm() + 1e-12)))
    rgb0 = klatka(CEL_J)
    gen = cv2.cvtColor(cv2.imread(str(KAT / 'obrazy' / f'{OBRAZ}.png')), cv2.COLOR_BGR2RGB)
    wyn = {}
    def kafel(im, t1, t2='', kol=(255, 255, 255)):
        y = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2BGR).copy(); cv2.rectangle(y, (0, 0), (480, 48), (0, 0, 0), -1)
        cv2.putText(y, t1, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .55, kol, 1, cv2.LINE_AA); cv2.putText(y, t2, (6, 41), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
        return y
    wiersze = []
    for nazwa, q_img in [(f'obraz SD {OBRAZ}', gen), ('zdjecie celu 0055/4019', rgb0)]:
        s = Fp @ cls(q_img); s[CEL_J] = -9
        top = np.argsort(-s)[:N_TOP]
        p_q = postep(q_img)
        rek = dict(zapytanie=nazwa, postep_zapytania=p_q, top=[])
        k = [kafel(q_img, nazwa, f'postep {100 * p_q["ulamek"]:.1f}%  widoczny {100 * p_q["ulamek_widoczny"]:.1f}%', (255, 255, 0))]
        for j in top:
            im = klatka(int(j)); p = postep(im)
            rek['top'].append(dict(j=int(j), film=MAN[j]['film'], klatka=int(MAN[j]['frame']), cos_dino=float(s[j]), **p))
            k.append(kafel(im, f'{MAN[j]["film"]} / {MAN[j]["frame"]}  DINO cos {s[j]:.2f}', f'postep {100 * p["ulamek"]:.1f}%  widoczny {100 * p["ulamek_widoczny"]:.1f}%',
                           (0, 255, 0) if MAN[j]['film'] == MAN[CEL_J]['film'] else (0, 200, 255)))
        wyn[nazwa] = rek; wiersze.append(np.concatenate([cv2.resize(x, (384, 256)) for x in k], 1))
        print(nazwa, [(t['film'][3:7], t['klatka'], round(t['cos_dino'], 2), round(100 * t['ulamek'], 1), round(100 * t['ulamek_widoczny'], 1)) for t in rek['top']])
    cv2.imwrite(str(KAT / f'SZUKANIE_TRAIN_{OBRAZ}.jpg'), np.concatenate(wiersze, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (KAT / f'SZUKANIE_TRAIN_{OBRAZ}.json').write_text(json.dumps(dict(wyniki=wyn, opis=__doc__), indent=1, ensure_ascii=False), encoding='utf-8')

if __name__ == '__main__':
    main()
