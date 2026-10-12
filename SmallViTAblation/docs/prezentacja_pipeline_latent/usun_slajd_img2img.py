from pathlib import Path
import json,hashlib,shutil
import fitz

PRES=Path(__file__).resolve().parent
ROOT=PRES.parents[1]
PDF=PRES/'PIPELINE_LATENT_VIT_SD_LANGEVIN.pdf'
title='Img2img: zdjęcie jako punkt startowy'
doc=fitz.open(str(PDF));matches=[i for i,p in enumerate(doc) if title in p.get_text()]
assert len(matches)==1,matches
removed=matches[0];assert removed==13
old_hash=hashlib.sha256(PDF.read_bytes()).hexdigest()
backup=PRES/'archiwum_pdf';backup.mkdir(exist_ok=True)
shutil.copy2(PDF,backup/f'przed_usunieciem_img2img_{old_hash[:12]}.pdf')
old_titles=[]
for p in doc:
    old_titles.append(p.get_text())
doc.delete_page(removed)
fontfile='C:/Windows/Fonts/arial.ttf';font=fitz.Font(fontfile=fontfile)
for i in range(removed,len(doc)):
    page=doc[i]
    page.add_redact_annot(fitz.Rect(1180,675,1240,706),fill=(1,1,1))
    page.apply_redactions(images=0,graphics=0)
    number=f'{i+1:02d}'
    page.insert_font(fontname='SlideArial',fontfile=fontfile)
    page.insert_text((1228-font.text_length(number,fontsize=12),692),number,
        fontname='SlideArial',fontsize=12,color=(80/255,103/255,121/255))
temp=PRES/'PDF_BEZ_IMG2IMG.tmp';doc.save(str(temp),garbage=4,deflate=True);doc.close()
check=fitz.open(str(temp));assert len(check)==24
assert all(title not in p.get_text() for p in check)
for i,page in enumerate(check):
    previous=i if i<removed else i+1
    actual=page.get_text().splitlines()
    prior=old_titles[previous].splitlines()
    if i>=removed:
        prior=[s for s in prior if s!=f'{previous+1:02d}']
        actual=[s for s in actual if s!=f'{i+1:02d}']
    assert actual==prior,(i,actual[-4:],prior[-4:])
check.close();temp.replace(PDF)
manifest=json.loads((PRES/'MANIFEST.json').read_text(encoding='utf-8'))
assert manifest['pages'].pop(removed)==title
manifest['removed_slide']={'title':title,'previous_page':14,'remaining_pages':24,
    'previous_pdf_sha256':old_hash,'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
    'only_page_removed_and_following_footer_numbers_updated':True}
(PRES/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
doc=fitz.open(str(PDF))
(PRES/'TEKST_PREZENTACJI.txt').write_text('\n\n'.join(p.get_text() for p in doc),encoding='utf-8')
preview=PRES/'podglad'
archive=preview/'archiwum_przed_usunieciem_img2img';archive.mkdir(exist_ok=True)
for p in list(preview.glob('strona-*.png'))+list(preview.glob('kontakt_*.png')):
    shutil.move(str(p),str(archive/p.name))
for i,page in enumerate(doc):
    page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(str(preview/f'strona-{i+1:02d}.png'))
doc.close()
for path in [ROOT/'tools/prezentacja_pipeline_latent.py',PRES/'prezentacja_source.py']:
    source=path.read_text(encoding='utf-8')
    start=source.index("# 14\nslide('Img2img:")
    end=source.index('# 15\n',start)
    path.write_text(source[:start]+source[end:],encoding='utf-8')
gallery=PRES/'zdjecia_w_kolejnosci'
archive=gallery/'archiwum_przed_usunieciem_img2img';archive.mkdir(exist_ok=True)
for p in list(gallery.glob('slajd_*'))+[gallery/n for n in ['GALERIA.html','MANIFEST.json','SPIS.csv','README.md','PODGLAD_W_KOLEJNOSCI.jpg']]:
    if p.is_file():shutil.move(str(p),str(archive/p.name))
builder=gallery/'zbierz.py'
source=builder.read_text(encoding='utf-8')
source=source.replace("doc=fitz.open(str(PDF));rows=[]", "assets={((n-1) if n>14 else n):v for n,v in assets.items() if n!=14}\ndoc=fitz.open(str(PDF));rows=[]")
source=source.replace('assert len(rows)==26','assert len(rows)==23').replace('26 kopii','23 kopie').replace('26 obrazów','23 obrazy')
builder.write_text(source,encoding='utf-8')
(PRES/'README.md').write_text('# Prezentacja pipeline latentu\n\nPDF ma24 slajdy. Usunięto slajd Img2img: zdjęcie jako punkt startowy; numerację kolejnych slajdów uaktualniono. Galeria zdjecia_w_kolejnosci przedstawia obrazy bieżącej wersji. Wcześniejsze kopie zachowano w archiwum.\n',encoding='utf-8')
(PRES/'QA_USUNIECIE_IMG2IMG.json').write_text(json.dumps(manifest['removed_slide'],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest['removed_slide'],ensure_ascii=False))
