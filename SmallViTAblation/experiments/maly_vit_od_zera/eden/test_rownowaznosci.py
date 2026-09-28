"""Sprawdzenie wersji Eden na CPU przed zgłoszeniem: (1) ścieżki obrazów bazy po przemapowaniu dają te same piksele
co wersja lokalna, (2) wyszukiwanie numpy (rank_candidates na wszystkich cechach) = query_index z FAISS dla
zapisanych zapytań kampanii lokalnej (ta sama kolejność top-90)."""
import json, os, sys
from pathlib import Path
import numpy as np
T = Path(__file__).resolve().parent
L = T.parent
os.environ.update(MALY_VIT_WEJSCIA=str(T / 'wejscia'), MALY_VIT_DANE=str(L.parent / 'koopman_fastgan_petla/local_inputs/data'),
                  MALY_VIT_CACHE=str(T / 'cache_test'), MALY_VIT_WYNIKI=str(T / 'wyniki_test'))
sys.path.insert(0, str(T / 'kod'))
import dane as DE
b = DE._coco(DE.DS / 'train/_annotations.coco.json')
braki = [r['sciezka'] for r in b if not Path(r['sciezka']).exists()]
print('baza:', len(b), 'brakujących plików:', len(braki))
X_lok = np.load(L / 'cache/baza1500.npy', mmap_mode='r')
idx = [0, 700, 1499]
same = all(np.array_equal(DE.wczytaj(b[i]['sciezka']), X_lok[i]) for i in idx)
print('piksele jak lokalnie (3 próbki):', same)
p = DE.pula(DE.wykluczone_bazowe(b)); print('pula:', len(p), 'brak plików:', sum(not Path(r['sciezka']).exists() for r in p[:200]), '/200')
sys.path.insert(0, str(T / 'wejscia/wyszukiwanie')); import wyszukiwanie as WY
cfg = json.loads((DE.INDEKS / 'config.json').read_text(encoding='utf-8'))
zr = WY.resolve_model_source(torch_hub_dir=DE.WEJ / 'dino', weights=DE.WEJ / 'dinov2_vits14_pretrain.pth')
print('DINOv2 zgodny z indeksem:', zr['model_sha256'] == cfg['model_source']['model_sha256'])
enc = WY.DinoEncoder(zr, device='cpu', pooling=cfg['pooling'])
cechy = WY.normalize_features(np.load(DE.INDEKS / 'features.npy')); meta = WY._read_jsonl(DE.INDEKS / 'metadata.jsonl')
wykl = DE.wykluczone_bazowe(b)
from PIL import Image
for kd in sorted((L / 'wyniki/petla_s42/koopman/k04').glob('kotwica_*'))[:3]:
    ref = json.loads((kd / 'zapytanie_target.json').read_text(encoding='utf-8'))['candidates']
    with Image.open(kd / 'target.png') as im:
        crop, _ = WY.prepare_query_image(im)
    q = enc.encode([crop])[0]
    wyn = WY.rank_candidates(meta, cechy @ q, exclude_ids=sorted(wykl), top_k=90, diversify_groups=True)
    print(kd.name, 'top-90 identyczne:', [c['frame_id'] for c in wyn] == [c['frame_id'] for c in ref],
          'max różnica wyniku:', max(abs(a['score'] - c['score']) for a, c in zip(wyn, ref)))
