from pathlib import Path
import json, hashlib, shutil, html, csv
import fitz
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT=Path(__file__).resolve().parent
PRES=OUT.parent
ROOT=PRES.parents[1]
PDF=PRES/'PIPELINE_LATENT_VIT_SD_LANGEVIN.pdf'
metadata=json.loads((PRES/'MANIFEST.json').read_text(encoding='utf-8'))
pairs=json.loads((ROOT/'experiments/maly_vit_test_orientacji_narzedzia_400/PARY.json').read_text(encoding='utf-8-sig'))
pairroot=ROOT/'experiments/maly_vit_test_orientacji_narzedzia_400'
def rel(p):return ROOT/p
sonda=rel('outputs/vit_latent128_jedno_zdjecie_90/ZDJECIE_ZRODLOWE.jpg')
sd=rel('outputs/vit_sd_obraz_najblizszy/OBRAZ_ZAPYTANIA.png')
found=rel('outputs/vit_sd_obraz_najblizszy/NAJBLIZSZE_ZDJECIE.png')
assets={
1:[('sonda_zrodlowa',sonda),('obraz_sd',sd),('znalezione_zdjecie_train',found)],
3:[('siatka_600_tokenow',PRES/'materialy/siatka600.png')],
5:[],
7:[('detekcja_przed_i_po',rel('outputs/vit_latent128_jedno_zdjecie_90/DETEKCJA_PRZED_PO.png'))],
8:[('768_kanalow_przed_i_po',rel('outputs/vit_latent128_jedno_zdjecie_90/WIZUALIZACJA768/LATENT768_PRZED_PO.png'))],
9:[('makrowektory_przed_po_roznica',PRES/'makrowektory_wartosci/MAKROWEKTORY_SLAJD9.png')],
12:[('szum_startowy',rel('outputs/vit_latent768_sd_od_szumu/SZUM_STARTOWY.png')),
    ('obraz_po_ddim',rel('outputs/vit_latent768_sd_od_szumu/OBRAZ_SD_PO_DDIM.png')),
    ('obraz_po_dalszej_optymalizacji',rel('outputs/vit_latent768_sd_od_szumu/OBRAZ_SD.png'))],
13:[('wynik_sd_z_szumu',sd)],
14:[('zdjecie_startowe_img2img',rel('outputs/vit_latent768_sd_od_najblizszego/ZDJECIE_STARTOWE.jpg')),
    ('podglad_zaszumienia_vae',rel('outputs/vit_latent768_sd_od_najblizszego/PODGLAD_ZASZUMIENIA_s35.png')),
    ('wynik_sd_img2img',rel('outputs/vit_latent768_sd_od_najblizszego/OBRAZ_SD.png'))],
15:[('zapytanie_sd',sd),('najblizsze_zdjecie_train',found)],
21:[]}
for row,idx in enumerate([0,2],1):
    for key,label in [('original','oryginal'),('mask_edit','maska_edycji'),('synthetic','po_lama')]:
        assets[5].append((f'para_{row}_{label}',pairroot/Path(pairs[idx][key].replace('\\','/'))))
for i,seed in enumerate([2026100511,2026100523,2026100537,2026100549],1):
    assets[21].append((f'wariant_sd_{i}_seed_{seed}',rel(f'outputs/vit_sd_cztery_szumy/OBRAZ_{seed}_prowadzony.png')))
assets={((n-1) if n>14 else n):v for n,v in assets.items() if n!=14}
doc=fitz.open(str(PDF));rows=[]
for page_no,page in enumerate(doc,1):
    images=page.get_image_info(xrefs=True)
    if not images:continue
    images.sort(key=lambda im:(round((im['bbox'][1]+im['bbox'][3])/2,1),im['bbox'][0]))
    assert len(images)==len(assets[page_no]),(page_no,len(images))
    for slot,(im,(label,source)) in enumerate(zip(images,assets[page_no]),1):
        assert source.is_file(),source
        with Image.open(source) as source_image:
            size=source_image.size
            rgb=source_image.convert('RGB')
            source_pixels=rgb.tobytes()
        assert size==(im['width'],im['height']),(page_no,label,size,im['width'],im['height'])
        embedded=doc.extract_image(im['xref'])
        exact_embedded_bytes=source.read_bytes()==embedded['image']
        pix=fitz.Pixmap(doc,im['xref'])
        if pix.colorspace.n!=3 or pix.alpha:pix=fitz.Pixmap(fitz.csRGB,pix)
        exact_pixels=(not pix.alpha and pix.samples==source_pixels)
        assert exact_embedded_bytes or exact_pixels,(page_no,label,'source does not match PDF image')
        filename=f'slajd_{page_no:02d}_{slot:02d}_{label}{source.suffix.lower()}'
        target=OUT/filename;shutil.copy2(source,target)
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
        rows.append(dict(order=len(rows)+1,slide=page_no,position=slot,
                         slide_title=metadata['pages'][page_no-1],description=label.replace('_',' '),
                         filename=filename,source=str(source),sha256=sha,width=size[0],height=size[1],
                         matched_to_pdf_by='embedded bytes' if exact_embedded_bytes else 'decoded RGB pixels'))
doc.close();assert len(rows)==23
(OUT/'MANIFEST.json').write_text(json.dumps({'pdf':str(PDF),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
    'count':len(rows),'ordering':'slide ascending, then visual reading order (row, left to right)',
    'all_copies_byte_identical_to_source':True,'all_sources_matched_to_displayed_pdf_images':True,'images':rows},ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'SPIS.csv').open('w',newline='',encoding='utf-8-sig') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
cards=[]
for r in rows:
    esc=lambda s:html.escape(str(s),quote=True)
    cards.append(f'<figure><a href="{esc(r["filename"])}"><img loading="lazy" src="{esc(r["filename"])}"></a><figcaption><b>Slajd {r["slide"]:02d} / obraz {r["position"]}</b><br>{esc(r["description"])}<small>{esc(r["slide_title"])}</small></figcaption></figure>')
(OUT/'GALERIA.html').write_text('<!doctype html><html lang="pl"><meta charset="utf-8"><title>Obrazy prezentacji w kolejności</title><style>body{font:17px Arial;background:#edf3f5;color:#142d42;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}figure{margin:0;background:white;border-radius:10px;padding:16px}img{width:100%;height:240px;object-fit:contain}figcaption{margin-top:12px;line-height:1.5}small{display:block;color:#506779}</style><h1>Obrazy prezentacji w kolejności</h1><p>23 kopie oryginalnych obrazów i wykresów. Kliknij miniaturę, aby otworzyć pełną rozdzielczość.</p><main>'+''.join(cards)+'</main></html>',encoding='utf-8')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',19)
cw,ch=420,300;sheet=Image.new('RGB',(4*cw,7*ch),'#edf3f5');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
    x,y=(i%4)*cw,(i//4)*ch
    draw.rounded_rectangle((x+8,y+8,x+cw-8,y+ch-8),radius=10,fill='white')
    with Image.open(OUT/r['filename']) as im:
        thumb=ImageOps.contain(im.convert('RGB'),(cw-36,210))
        sheet.paste(thumb,(x+(cw-thumb.width)//2,y+18+(210-thumb.height)//2))
    draw.text((x+18,y+232),f'Slajd {r["slide"]:02d} / obraz {r["position"]}',font=bold,fill='#142d42')
    caption=r['description']
    if len(caption)>43:caption=caption[:40]+'...'
    draw.text((x+18,y+260),caption,font=font,fill='#506779')
sheet.save(OUT/'PODGLAD_W_KOLEJNOSCI.jpg',quality=92)
(OUT/'README.md').write_text('# Obrazy użyte w prezentacji\n\n23 obrazy i wykresów skopiowano bez zmiany plików źródłowych. Kolejność: numer slajdu, następnie rząd i pozycja od lewej. Powtórzenia obrazu na kolejnych slajdach zachowano, aby odtworzyć przebieg prezentacji.\n\n- GALERIA.html: podpisane miniatury w kolejności, otwierające obrazy w pełnej rozdzielczości.\n- PODGLAD_W_KOLEJNOSCI.jpg: wspólny podgląd.\n- SPIS.csv i MANIFEST.json: nazwy, opisy, źródła, numery slajdów i sumy SHA256.\n\nKażdy plik dopasowano do rzeczywistego obrazu osadzonego w bieżącym PDF-ie (bajty JPEG albo piksele RGB). Uwzględniono poprawiony wykres makrowektorów na slajdzie9. Slajdy bez osadzonych zdjęć nie mają osobnych obrazów; rysunki wektorowe nie są fotografiami. PDF pozostawiono bez zmian.\n',encoding='utf-8')
print(json.dumps({'count':len(rows),'folder':str(OUT),'gallery':str(OUT/'GALERIA.html'),'copies_verified':True,'pdf_image_matching_verified':True},ensure_ascii=False))
