import json, math
from pathlib import Path
from PIL import Image, ImageDraw

root=Path(__file__).resolve().parents[2]
out=root/'outputs/vit_sd_32_stary_sd_nowy_latent_20261006'
rows=json.loads((out/'METRICS.json').read_text(encoding='utf-8'))
files=sorted(out.glob('*_seed_*.png'))
assert {p.name for p in files}=={r['file'] for r in rows},'Unmeasured image: finish its evaluation first'
manifest=json.loads((out/'MANIFEST.json').read_text(encoding='utf-8'))
sheet=Image.new('RGB',(1152,math.ceil(len(rows)/4)*212),'white');draw=ImageDraw.Draw(sheet);cards=[]
for i,row in enumerate(rows):
    x=i%4*288;y=i//4*212
    with Image.open(out/row['file']) as im: sheet.paste(im.resize((288,192)),(x,y))
    draw.text((x+4,y+193),f"{row['index']:02d} seed {row['seed']}",fill='black')
    cards.append(f'<figure><img src="{row["file"]}"><figcaption>{row["index"]:02d}; J128={row["metrics"]["value_agreement128"]:.2f}%; cosine={row["metrics"]["cosine128"]:.4f}</figcaption></figure>')
sheet.save(out/'PRZEGLAD.png')
(out/'GALERIA.html').write_text(f'<meta charset="utf-8"><title>Stary SD — {len(rows)} prób</title><style>main{{display:grid;grid-template-columns:repeat(4,1fr)}}img{{width:100%}}figure{{margin:8px}}body{{font:16px sans-serif}}</style><h1>Stary SD: {len(rows)} obrazów sterowanych pełnym sztucznym latentem</h1><p>Generowanie zatrzymane na polecenie użytkownika. Start od niezależnych szumów. Zgodność wartości J128 i cosine są osobnymi miarami.</p><main>'+''.join(cards)+'</main>',encoding='utf-8')
result=dict(status='stopped_by_user',manifest=manifest,samples=rows,completed=len(rows),planned=32,note='Natural forward evaluation; no generated-image ground truth or IoU claimed.')
(out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
(out/'TEST_LOKALNY_STATUS.json').write_text(json.dumps(dict(phase='stopped_by_user',completed=len(rows),total=32)),encoding='utf-8')
best=max(rows,key=lambda r:r['metrics']['value_agreement128'])
print(json.dumps(dict(completed=len(rows),best=best)))
