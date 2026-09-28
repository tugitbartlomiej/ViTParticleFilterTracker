"""Kontrola po wyniku (ziarno 42 lokalnie): czy obraz FastGAN „target” (przewidziany latent) prowadzi do innych
prawdziwych klatek niż obraz „current” (obecny latent, bez prognozy). Wspólna część top-90 i top-22 (≈134/6)."""
import json, os, sys
from pathlib import Path
import numpy as np
T = Path(__file__).resolve().parent; L = T.parent
os.environ.update(MALY_VIT_WEJSCIA=str(T / 'wejscia'), MALY_VIT_DANE=str(L.parent / 'koopman_fastgan_petla/local_inputs/data'),
                  MALY_VIT_CACHE=str(T / 'cache_test'), MALY_VIT_WYNIKI=str(T / 'wyniki_test'))
sys.path.insert(0, str(T / 'wejscia/wyszukiwanie')); import wyszukiwanie as WY
from PIL import Image
W = T / 'wejscia'
cfg = json.loads((W / 'index/config.json').read_text(encoding='utf-8'))
enc = WY.DinoEncoder(WY.resolve_model_source(torch_hub_dir=W / 'dino', weights=W / 'dinov2_vits14_pretrain.pth'), device='cuda', pooling=cfg['pooling'])
cechy = WY.normalize_features(np.load(W / 'index/features.npy')); meta = WY._read_jsonl(W / 'index/metadata.jsonl')
wyn = []
for kd in sorted((L / 'wyniki/petla_s42/koopman').glob('k*/kotwica_*')):
    ref = json.loads((kd / 'zapytanie_target.json').read_text(encoding='utf-8'))
    wykl = ref.get('excluded_ids', [])
    a = [c['frame_id'] for c in ref['candidates']]
    with Image.open(kd / 'current.png') as im:
        crop, _ = WY.prepare_query_image(im)
    q = enc.encode([crop])[0]
    b = [c['frame_id'] for c in WY.rank_candidates(meta, cechy @ q, exclude_ids=wykl, top_k=90, diversify_groups=True)]
    with Image.open(kd / 'target.png') as im:
        ct, _ = WY.prepare_query_image(im)
    with Image.open(kd / 'current.png') as im:
        cc, _ = WY.prepare_query_image(im)
    cos = float(enc.encode([ct])[0] @ enc.encode([cc])[0])
    wyn.append(dict(kotwica=f'{kd.parent.name}/{kd.name}', top90=len(set(a) & set(b)), top22=len(set(a[:22]) & set(b[:22])), cos_dino_target_current=cos))
    print(wyn[-1], flush=True)
print('średnio: top90 %.1f/90, top22 %.1f/22, cos %.3f' % (np.mean([w['top90'] for w in wyn]), np.mean([w['top22'] for w in wyn]), np.mean([w['cos_dino_target_current'] for w in wyn])))
(L / 'wyniki/KONTROLA_TARGET_CURRENT_s42.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
