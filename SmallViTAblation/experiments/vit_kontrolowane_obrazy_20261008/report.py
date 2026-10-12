from pathlib import Path
import csv,json,shutil,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/vit_kontrolowane_obrazy_20261008/pilot_v1'
def read(name):
    with (OUT/name).open(encoding='utf8') as f:return list(csv.DictReader(f))
result=json.loads((OUT/'RESULT.json').read_text(encoding='utf8'));probes=read('PROBES.csv');motion=read('MOTION.csv');controls=read('CONTROLS.csv');transfer=read('NATURAL_TRANSFER.csv')
factors=['pozycja','kolor','proporcje'];summary={}
def mean(rows,column):return float(np.mean([float(r[column]) for r in rows]))
for method in ['svd','contrast']:
    summary[method]={}
    for domain in ['real_cutout','fastgan']:
        matrix=np.array([[mean([r for r in probes if r['method']==method and r['domain']==domain and r['basis']==f and r['perturbation']==g],'capture') for g in factors] for f in factors])
        summary[method][domain]=matrix.tolist()
    natural=np.array([[mean([r for r in transfer if r['method']==method and r['basis']==f and r['perturbation']==g],'capture') for g in factors] for f in factors]);summary[method]['natural_final']=natural.tolist()
    for f in factors:
        data=[r for r in motion if r['method']==method and r['factor']==f]
        x=np.array([float(r['motion_fraction']) for r in data]);y=np.array([float(r['delta_iou']) for r in data])
        from scipy.stats import spearmanr
        summary[method][f]=dict(motion_fraction=float(x.mean()),spearman=float(spearmanr(x,y).statistic),rollback_effect=mean(data,'rollback_effect'),random_norm_effect=mean(data,'random_norm_effect'))
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
# Recheck orthogonality and archive completeness independently.
states=np.load(OUT/'BASES.npz');orth=max(float(np.max(np.abs(q.T@q-np.eye(8)))) for checkpoint in states['bases'] for q in checkpoint)
assert orth<1e-4;assert len(probes)==5*6*3*8;assert len(motion)==4*6*4;assert len(controls)==5*11;assert len(transfer)==4*6*3
qa=json.loads((OUT/'QA.json').read_text(encoding='utf8'));qa.update(orthonormal_max_error=orth,probe_rows=len(probes),motion_rows=len(motion),natural_transfer_rows=len(transfer),status='PASS');(OUT/'QA.json').write_text(json.dumps(qa,indent=2),encoding='utf8')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(13,8))
for row,method in enumerate(['svd','contrast']):
    for col,domain in enumerate(['real_cutout','fastgan','natural_final']):
        matrix=np.array(summary[method][domain]);ax=axes[row,col];ax.imshow(matrix,vmin=0,vmax=1)
        ax.set_xticks(range(3),factors,rotation=20);ax.set_yticks(range(3),factors);ax.set_title(f'{method}: {domain}')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<.5 else 'black')
fig.suptitle('Udział odpowiedzi przechwycony przez sondę; wiersz = sonda, kolumna = zmiana')
fig.tight_layout();fig.savefig(OUT/'SELECTIVITY.png',dpi=160);plt.close(fig)
# Contact sheet of one real anchor plus one generated anchor; includes labels.
tiles=[]
for anchor in [8,12]:
    strip=[]
    for variant,label in [(0,'baseline'),(1,'position +32 px'),(5,'color Lab a+12'),(9,'aspect x1.25')]:
        p=OUT/'images'/f'anchor{anchor:02d}_variant{variant:02d}_{["baseline","pozycja","kolor","proporcje"][[0,1,5,9].index(variant)]}.png'
        im=cv2.imread(str(p));im=cv2.resize(im,(360,240));cv2.putText(im,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1,cv2.LINE_AA);strip.append(im)
    tiles.append(np.concatenate(strip,axis=1))
cv2.imwrite(str(OUT/'EXAMPLES.png'),np.concatenate(tiles,axis=0))
lines=['# Kontrolowane obrazy: wynik eksperymentu','',f'Wygenerowano {result["controlled_images"]} kontrolowanych wariantów i {result["controls"]} jednolitych/gradientowych kontroli. Dwanaście kotwic z adnotowanych narzędzi, cztery próbki FastGAN EMA po 50 tysiącach kroków. Pięć checkpointów: 0, 250, 500, 750, 1000. Obliczenia CPU, {result["elapsed_seconds"]:.1f} s (bez dodatkowego testu przeniesienia na naturalne klatki).','', 'Osiem wycinków dopasowało sondy. Cztery inne wycinki i cztery obrazy FastGAN sprawdziły odpowiedź. Wszystkie rzeczywiste obrazy pochodzą z treningu detektora: nie jest to jego niezależna walidacja. FastGAN RGB i maski są diagnostyczne, nie prawdziwe etykiety. SD nie został załadowany z powodu zajętych zasobów; nie jest wymagany do dokładnej zmiany czynników.','', 'SVD szuka dominujących kierunków odpowiedzi. Wariant contrast maksymalizuje znormalizowaną odpowiedź na jeden czynnik względem pozostałych, z ustaloną regularyzacją 0.01/768 i ośmioma kierunkami. Używa tylko obrazów fit. Większa selektywność może odbywać się kosztem bardzo małego udziału całej odpowiedzi; sam kontrast nie dowodzi przyczynowego modułu.','']
for method in ['svd','contrast']:
    lines+=['',f'## Sondy {method}','']
    for domain in ['real_cutout','fastgan','natural_final']:
        lines.append(f'{domain}: wiersze to pozycja/kolor/proporcje; kolumny w tej samej kolejności. Liczby to przechwycona energia odpowiedzi, nie accuracy klasyfikacji.')
        lines.extend(['```',np.array2string(np.array(summary[method][domain]),precision=3),'```',''])
    for f in factors:
        s=summary[method][f];lines.append(f'- {f}: udział ruchu na naturalnych obrazach {s["motion_fraction"]:.3f}; korelacja z ΔIoU {s["spearman"]:.3f}; efekt cofnięcia {s["rollback_effect"]:.5f}, losowa kontrola tej samej normy {s["random_norm_effect"]:.5f}.')
lines+=['','## Granice interpretacji','', 'Kształt oznacza deformację proporcji narzędzia w 2D, nie całą semantykę kształtu. W kontrolowanych obrazach tło nie jest deformowane. W NATURAL_TRANSFER.csv pozycja i proporcje dotyczą całego kadru, a kolor tylko maski narzędzia; nie są to identyczne warunki. Wyniki transferu dotyczą końcowego checkpointu, pozostałe macierze są średnią z pięciu checkpointów.','', 'Ruch mierzony na niezmienionych naturalnych klatkach. Baza ingerencji pochodzi z wcześniejszego checkpointu. Interwencja dotyczy gałęzi MLP po GELU, nie całej sieci ani rezyduum. Latent po ingerencji może wyjść poza rozkład aktywacji. IoU dotyczy ramki o najwyższej pewności; zmiana rankingu może powodować skok. Korelacje obejmują tylko 16 zależnych pomiarów na metodę/czynnik: opisowe, bez wnioskowania o istotności czy nowości naukowej.','', 'Jednolite obrazy służą odpowiedzi na kolor/jasność i diagnostyce fałszywych wskazań. Próba obejmuje najwyżej 10 lokalnych szczytów; liczniki nie są pełnym mAP ani liczbą wszystkich możliwych ramek. Szczegóły: CONTROLS.csv.','', 'QA.json: dokładne przesunięcie pikseli, zachowanie maski i tła przy zmianie koloru, identyczność zerowej interwencji, ortonormalność kierunków i kompletność tabel. Nie oznacza to potwierdzenia semantycznej poprawności wszystkich masek FastGAN.','', 'Kod i plany w experiments/vit_kontrolowane_obrazy_20261008. Obrazy, manifesty i wyniki w bieżącym katalogu.']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');shutil.copyfile(Path(__file__).with_name('PLAN.md'),OUT/'PLAN.md')
print(json.dumps(summary,indent=2,ensure_ascii=False));print('QA PASS')
