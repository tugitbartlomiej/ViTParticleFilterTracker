from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
with np.load(SRC) as f:
    a, b = f['before'].astype(np.float64), f['after'].astype(np.float64)
    selected = np.asarray(f['channels0'], dtype=int)
assert a.shape == b.shape == (600,768)
mask = np.zeros(768,dtype=bool); mask[selected] = True
assert np.array_equal(a[:,~mask], b[:,~mask])
means = [a.mean(0), b.mean(0)]
rms = [np.sqrt((a*a).mean(0)), np.sqrt((b*b).mean(0))]
means.append(means[1]-means[0]); rms.append(rms[1]-rms[0])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
orange, grey = '#df7920', '#71828e'
colors = np.where(mask, orange, grey)
rows = ['PRZED zmianą', 'PO sztucznej zmianie', 'RÓŻNICA: po - przed']
x = np.arange(1,769)
fig, axes = plt.subplots(3,2,figsize=(22,12),layout='constrained')
for col, data in enumerate([means,rms]):
    lim = max(np.max(np.abs(data[0])), np.max(np.abs(data[1]))) * 1.08
    for row in range(3):
        ax=axes[row,col]
        ax.bar(x,data[row],color=colors,width=1,linewidth=0)
        ax.axhline(0,color='#233747',lw=.8)
        ax.set_xlim(0,769); ax.set_xticks([1,128,256,384,512,640,768])
        ax.set_xlabel('Numer osi w prezentacji (1-768; indeks danych = numer - 1)')
        ax.set_ylabel('Średnia współrzędna' if col==0 else ('Zmiana RMS' if row==2 else 'RMS współrzędnej'))
        ax.set_title(rows[row]+(' | wartości ze znakiem' if col==0 else ' | wielkość aktywacji'))
        if row<2: ax.set_ylim((-lim,lim) if col==0 else (0,lim))
        ax.grid(axis='y',alpha=.18)
fig.suptitle('768 makrowektorów: wartości przypisane stałym osiom przed i po korekcie\n'
             'Średnia i RMS ze wszystkich 600 cząsteczek jednego obrazu; bez PCA',fontsize=20)
fig.supxlabel('Pomarańczowy: 128 wybranych kanałów. Szary: 640 pozostałych, zachowanych dokładnie.\n'
              'Osie bazowe nie zmieniły kierunku ani normy. Pokazano statystyki współrzędnych, nie siły gradientu.',fontsize=12)
fig.savefig(OUT/'MAKROWEKTORY_PRZED_PO_ROZNICA.png',dpi=150)
plt.close(fig)

lim=max(np.max(np.abs(means[0])),np.max(np.abs(means[1])))
for row,name in enumerate(['PRZED','PO','ROZNICA']):
    values=means[row].reshape(24,32)
    scale=lim if row<2 else max(np.max(np.abs(values)),1e-12)
    fig,ax=plt.subplots(figsize=(24,19),layout='constrained')
    im=ax.imshow(values,cmap='RdBu_r',vmin=-scale,vmax=scale,interpolation='nearest')
    for j,v in enumerate(means[row]):
        y,xx=divmod(j,32)
        ax.text(xx,y,f'{j+1}\n{v:.2f}',ha='center',va='center',fontsize=7,
                color='white' if abs(v)>scale*.58 else '#152535')
        if mask[j]: ax.add_patch(Rectangle((xx-.5,y-.5),1,1,fill=False,edgecolor=orange,lw=1.8))
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(f'{rows[row]}: wszystkie 768 osi\nWartość w kafelku = średnia współrzędna ze wszystkich 600 cząsteczek',fontsize=23,pad=20)
    cb=fig.colorbar(im,ax=ax,shrink=.7,pad=.015)
    cb.set_label('Po - przed' if row==2 else 'Średnia współrzędna')
    fig.supxlabel('Każdy kafelek: numer osi (1-768) i wartość; pomarańczowa obwódka: wybrane J128.\n'
                  'Układ 32 × 24 to spis osi, nie geometria latentu ani siatka obrazu.\n'
                  'PRZED i PO mają wspólną skalę; RÓŻNICA ma osobną skalę wokół zera. Baza jest stała.',fontsize=14)
    fig.savefig(OUT/f'MAKROWEKTORY_{name}.png',dpi=150)
    plt.close(fig)

with (OUT/'WARTOSCI_768.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f)
    writer.writerow(['axis_display_1based','channel_native_0based','selected_J128','mean_before','mean_after','mean_delta','rms_before','rms_after','rms_delta','max_abs_token_delta'])
    for j in range(768):
        writer.writerow([j+1,j,int(mask[j]),*(v[j] for v in means),*(v[j] for v in rms),float(np.max(np.abs(b[:,j]-a[:,j])))])
record={'source':str(SRC),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
        'shape':[600,768],'aggregation':'all 600 tokens, separate signed mean and RMS per channel',
        'base_axes_unchanged':True,'unselected640_bitwise_unchanged':True,
        'selected_channel_count':int(mask.sum()),'not_PCA':True,
        'max_abs_unselected_delta':float(np.max(np.abs(b[:,~mask]-a[:,~mask]))),
        'images':[]}
for path in sorted(OUT.glob('*.png')):
    with Image.open(path) as im: size=list(im.size);im.verify()
    record['images'].append({'name':path.name,'size':size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'WERYFIKACJA.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'JAK_CZYTAC.md').write_text('''# Wartości przypisane 768 osiom latentu

Zapisano podpisane wykresy PRZED, PO i RÓŻNICA oraz wspólne zestawienie. Źródłem jest oryginalny latent jednego zdjęcia, przed i po sztucznej korekcie J128 przy zamrożonych wagach. Nie jest to latent po treningu Lorentza ani latent obrazu z SD.

Makrowektory e_j są stałymi osiami bazowymi: ich kierunki i jednostkowe normy nie zmieniają się. Zmieniają się współrzędne cząsteczek z_j. Aby pokazać jedną wartość na oś, wykres agreguje wszystkie 600 cząsteczek: średnia zachowuje znak, RMS opisuje wielkość aktywacji i ogranicza efekt znoszenia dodatnich i ujemnych wartości. Są to statystyki współrzędnych, nie siły paczek/gradientu. Każda cząsteczka nadal ma własną wartość.

Pomarańczowy wyróżnia wybrane 128 kanałów. Pozostałe 640 współrzędnych nie zmieniło się dla żadnej cząsteczki. Numery na obrazach są od1 do768; CSV podaje również natywne indeksy od0 do767. Pełne liczby w CSV, kafelki zaokrąglają do2 miejsc.

Atlas32x24 jest indeksem kanałów, nie mapą przestrzenną. Stany PRZED i PO mają tę samą skalę kolorów. RÓŻNICA ma osobną skalę symetryczną wokół0. Nie zastosowano PCA ani odejmowania środka chmury. Różnica oznacza po-przed i nie jest dowodem poprawy na innych obrazach.

Istniejący PDF prezentacji pozostawiono bez zmian; nowe materiały są w tym folderze.
''',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
