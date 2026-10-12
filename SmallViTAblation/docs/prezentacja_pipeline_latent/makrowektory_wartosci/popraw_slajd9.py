from pathlib import Path
import json, hashlib, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from pypdf import PdfReader, PdfWriter
import fitz

OUT=Path(__file__).resolve().parent
PRES=OUT.parent
ROOT=PRES.parents[1]
PDF=PRES/'PIPELINE_LATENT_VIT_SD_LANGEVIN.pdf'
with np.load(ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz') as f:
    a=f['before'].astype(float); b=f['after'].astype(float); J=f['channels0'].astype(int)
sel=np.zeros(768,dtype=bool);sel[J]=True
assert np.array_equal(a[:,~sel],b[:,~sel])
v=[a.mean(0),b.mean(0),(b-a).mean(0)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13})
fig,axs=plt.subplots(1,3,figsize=(18,5),layout='constrained')
limit=max(abs(v[0]).max(),abs(v[1]).max())*1.1
for i,(ax,z,title) in enumerate(zip(axs,v,['PRZED','PO','RÓŻNICA: po - przed'])):
    ax.bar(np.arange(1,769),z,color=np.where(sel,'#DF7920','#83939D'),width=1,linewidth=0)
    ax.axhline(0,lw=.8,color='#142D42');ax.set_xlim(0,769)
    ax.set_xticks([1,128,256,384,512,640,768]);ax.tick_params(axis='x',labelsize=11)
    ax.set_xlabel('Numer makrowektora (1-768)',fontsize=14);ax.set_title(title,fontsize=18,fontweight='bold')
    ax.set_ylabel('Średnia wartość kanału' if i<2 else 'Zmiana wartości kanału',fontsize=13)
    ax.grid(axis='y',alpha=.18)
    if i<2:ax.set_ylim(-limit,limit)
fig.savefig(OUT/'MAKROWEKTORY_SLAJD9.png',dpi=180)
plt.close(fig)

for name,file in [('Arial','arial.ttf'),('ArialB','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
replacement=OUT/'SLAJD9.pdf'
c=canvas.Canvas(str(replacement),pagesize=(1280,720),pageCompression=1)
def text(x,y,s,size=20,color='#142D42',bold=False):
    c.setFillColor(HexColor(color));c.setFont('ArialB' if bold else 'Arial',size);c.drawString(x,y,s)
c.setFillColor(HexColor('#FFFFFF'));c.rect(0,0,1280,720,fill=1,stroke=0)
c.setFillColor(HexColor('#087E8B'));c.rect(0,708,1280,12,fill=1,stroke=0)
text(52,672,'WIZUALIZACJA',15,'#087E8B',True)
text(52,622,'768 makrowektorów: przed zmianą, po zmianie i różnica',31,bold=True)
text(52,579,'Każdy makrowektor odpowiada jednej osi latentu. Wykres pokazuje średnią wartość jej kanału.',19,'#506779')
text(52,550,'Pomarańczowy: wybrane 128 kanałów. Szary: pozostałe 640, zachowane dokładnie.',18,'#506779')
c.drawImage(str(OUT/'MAKROWEKTORY_SLAJD9.png'),52,177,width=1176,height=327,mask='auto')
text(52,145,'Kierunki makrowektorów pozostają stałe. Porównujemy wartości przypisane ich kanałom.',21,bold=True)
text(52,111,'PRZED i PO: ta sama skala. RÓŻNICA: osobna skala. Numery 1-768 odpowiadają indeksom 0-767.',16,'#506779')
text(52,84,'Bez PCA. Wartości są średnimi aktywacjami kanałów; nie oznaczają zmiany długości osi bazowych.',16,'#506779')
c.setStrokeColor(HexColor('#DFE7EB'));c.line(52,51,1228,51)
text(52,28,'SmallViTAblation | Stan materiału: 5 października 2026',12,'#506779')
text(1208,28,'09',12,'#506779');c.save()

original=PdfReader(str(PDF));old_hash=hashlib.sha256(PDF.read_bytes()).hexdigest()
backup=OUT/f'PDF_PRZED_POPRAWKA_{old_hash[:12]}.pdf'
if not backup.exists():shutil.copy2(PDF,backup)
writer=PdfWriter()
for i,page in enumerate(original.pages):
    writer.add_page(PdfReader(str(replacement)).pages[0] if i==8 else page)
if original.metadata:writer.add_metadata({str(k):str(v) for k,v in original.metadata.items() if v is not None})
temp=OUT/'PDF_POPRAWIONY.tmp'
with temp.open('wb') as f:writer.write(f)
new=PdfReader(str(temp));assert len(new.pages)==len(original.pages)==25
assert 'cząsteczk' not in new.pages[8].extract_text().lower()
unchanged=[]
for i in range(25):
    if i==8:continue
    assert new.pages[i].extract_text()==original.pages[i].extract_text(),i
    assert new.pages[i].get_contents().get_data()==original.pages[i].get_contents().get_data(),i
    unchanged.append(i+1)
temp.replace(PDF)
doc=fitz.open(str(PDF));pix=doc[8].get_pixmap(matrix=fitz.Matrix(1.8,1.8))
pix.save(str(OUT/'SLAJD9_PODGLAD.png'));doc.close()
shutil.copy2(OUT/'SLAJD9_PODGLAD.png',PRES/'podglad/strona-09.png')
(PRES/'TEKST_PREZENTACJI.txt').write_text('\n\n'.join(p.extract_text() for p in new.pages),encoding='utf-8')
manifest_path=PRES/'MANIFEST.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['pages'][8]='768 makrowektorów: przed zmianą, po zmianie i różnica'
manifest['slide9_revision']={'method':'signed mean over all 600 tokens, native 768 coordinates, no PCA',
    'script':str(Path(__file__).resolve()),'figure':str(OUT/'MAKROWEKTORY_SLAJD9.png'),
    'original_pdf_sha256':old_hash,'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
    'other_pages_verified_unchanged':unchanged}
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'QA_SLAJD9.json').write_text(json.dumps(manifest['slide9_revision'],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':25,'changed_pages':[9],'other_pages_verified':24,'preview':str(OUT/'SLAJD9_PODGLAD.png')},ensure_ascii=False))
