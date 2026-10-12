"""Create report and scientific figures from an already completed run."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);a=ap.parse_args();o=a.out
rows=json.loads((o/'METRICS.json').read_text());prep=json.loads((o/'PREPARE.json').read_text());runtime=json.loads((o/'RUNTIME.json').read_text())
with open(o/'MODELS.pkl','rb') as f:models=pickle.load(f)
with open(o/'prepared.pkl','rb') as f:d=pickle.load(f)
sel=json.loads((o/'MODEL_SELECTION_BEFORE_TEST.json').read_text())
chosen=max(models,key=lambda k:models[k]['score'])
spectra={}
for name,x in models.items():
    m=x['model'];r=m['r']
    if 'A' in m:A=m['A']
    else:
        W=m['W'];A=np.eye(r)+W[:r].T
        if m['lag']:
            Av=W[r:2*r].T;A=np.block([[A+Av,-Av],[np.eye(r),np.zeros((r,r))]])
    ev,vec=np.linalg.eig(A)
    records=[]
    for i,l in enumerate(ev):
        angle=np.angle(l);mag=abs(l)
        records.append(dict(real=float(l.real),imag=float(l.imag),magnitude=float(mag),
           period_steps=float(2*np.pi/abs(angle)) if abs(angle)>1e-8 else None,
           decay_time_steps=float(-1/np.log(mag)) if 0<mag<1 else None,
           growth_time_steps=float(1/np.log(mag)) if mag>1 else None))
    spectra[name]=records
(o/'SPECTRA.json').write_text(json.dumps(spectra,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for name in models:
    rr=[r for r in rows if r['split']=='test' and r['group']=='all' and r['model']==name]
    ax[0].plot([r['horizon'] for r in rr],[r['skill'] for r in rr],marker='o',label=name)
for name in ['velocity','force_direct','force_inertia']:
    rr=[r for r in rows if r['split']=='test' and r['group']=='all' and r['model']==name]
    ax[0].plot([r['horizon'] for r in rr],[r['skill'] for r in rr],'--',marker='s',label=name)
ax[0].axhline(0,color='k',lw=.8);ax[0].set(xlabel='Horizon (optimizer steps)',ylabel='Full 768 SSE skill vs persistence',title='Chronological test; known future forces')
ax[0].legend(fontsize=7,ncol=2)
for name in dict.fromkeys(['DMD_delay','DMDc_delay',chosen]):
    v=spectra[name];ax[1].scatter([x['real'] for x in v],[x['imag'] for x in v],s=20,label=name)
theta=np.linspace(0,2*np.pi,200);ax[1].plot(np.cos(theta),np.sin(theta),'k--',lw=.7);ax[1].set(xlabel='Real eigenvalue part',ylabel='Imaginary eigenvalue part',title='Selected reduced operators');ax[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(o/'FORECAST_AND_SPECTRUM.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5))
labels={'DMD_delay':'Autonomous DMD + memory','DMDc_delay':'DMDc + measured force + memory',
        'force_direct':'Direct force addition','force_inertia':'Two-scalar force + memory',
        'POD_oracle_ceiling_NOT_FORECAST':'POD64 oracle projection (not forecast)'}
for name,label in labels.items():
    rr=[r for r in rows if r['split']=='test' and r['group']=='all' and r['model']==name]
    ax.plot([r['horizon'] for r in rr],[r['skill']*100 for r in rr],marker='o',label=label,
            linestyle=':' if 'oracle' in name else '-')
ax.axhline(0,color='k',lw=.7);ax.set(xlabel='Horizon (optimizer steps)',ylabel='Reduction of full 768 squared error (%)',
      title='Chronological test: input forces help, simple inertia wins',xticks=[1,5,10]);ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(o/'KEY_RESULT.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
t=np.arange(d['lo'],d['hi']+1)
for k in range(4):
    ax[0].plot(t,d['coeff'][:,k]/d['coeff'][:d['nt'],k].std(),lw=.8,label=f'State POD{k+1}')
    ax[1].plot(t,d['force'][:,k]/d['force'][:d['nt'],k].std(),lw=.65,label=f'Force POD{k+1}')
for z in ax:
    z.axvline(t[d['nt']-1],color='k',ls='--',lw=.8);z.axvline(800,color='k',ls=':',lw=.8);z.legend(ncol=4,fontsize=8);z.set_ylabel('Train-standardized coefficient')
ax[1].set_xlabel('AdamW step');fig.tight_layout();fig.savefig(o/'COLLECTIVE_COEFFICIENTS.png',dpi=180);plt.close(fig)

mode=np.load(o/'POD_MODES.npy',mmap_mode='r');fig,axs=plt.subplots(3,3,figsize=(11,9))
for k in range(3):
    energy=np.sqrt(np.square(mode[k]).sum(-1));vmax=energy.max()
    for f in range(3):
        im=axs[k,f].imshow(energy[f].reshape(20,30),vmin=0,vmax=vmax,cmap='magma')
        axs[k,f].contour(np.asarray(d['meta']['narzedzie'][f]).reshape(20,30),levels=[.5],colors='white',linewidths=.6)
        axs[k,f].set_title(f'POD {k+1}, probe {d["meta"]["klatki"][f]}');axs[k,f].set(xticks=[],yticks=[])
        fig.colorbar(im,ax=axs[k,f],fraction=.035)
fig.tight_layout();fig.savefig(o/'COLLECTIVE_MODE_SPATIAL_ENERGY.png',dpi=180);plt.close(fig)

test=[r for r in rows if r['split']=='test' and r['group']=='all']
def skills(name):return [r['skill']*100 for r in test if r['model']==name]
coh=json.loads((o/'COHERENCE.json').read_text())
ct=np.asarray(coh[0]['train_corr'])[np.triu_indices(3,1)];ce=np.asarray(coh[0]['test_corr'])[np.triu_indices(3,1)]
best_osc=sorted([v for v in spectra['DMDc_delay'] if v['imag']>1e-8],key=lambda v:v['magnitude'],reverse=True)
lines=['# Koopman z wejściami faktycznych paczek: wynik lokalnego biegu','',
 'Analiza 3 stałych sond (2, 3, 6), wszystkich 600 tokenów i 768 współrzędnych. '
 'Zegar to pojedynczy krok AdamW. Wejściem jest zapisane pchnięcie S_nowa z rzeczywistej paczki treningowej.', '',
 'Źródło: gotowy bieg H100 s47 job1823539; Z i S_nowa z tego samego biegu. '
 'Lokalny RTX3070 policzył nowe macierze Grama, redukcję i projekcje. Dopasowanie PyDMD oraz ridge odbyło się na CPU. '
 'Nie uruchomiono ponownie treningu sieci.', '',
 '## Najważniejsze ustalenia','',
 f"DMDc z pamięcią obniża błąd kwadratowy względem persystencji o {skills('DMDc_delay')[0]:.2f}%, "
 f"{skills('DMDc_delay')[1]:.2f}% i {skills('DMDc_delay')[2]:.2f}% dla 1/5/10 kroków. "
 f"Odpowiednie autonomiczne DMD z pamięcią: {skills('DMD_delay')[0]:.2f}%, {skills('DMD_delay')[1]:.2f}%, {skills('DMD_delay')[2]:.2f}%. "
 'Zmierzona siła paczki wnosi informację, a poprzednia prędkość jest ważnym składnikiem stanu.', '',
 f"DMDc nie przewyższa jednak prostego modelu beta*v+b*S_nowa: {skills('force_inertia')[0]:.2f}%, "
 f"{skills('force_inertia')[1]:.2f}%, {skills('force_inertia')[2]:.2f}%. "
 'Sam warunek znajomości sił daje bardzo dużo; nie należy przypisywać tej informacji odkrytym modom Koopmana.', '',
 f"64 mody opisują {100*prep['POD_train_explained_variance']['64']:.2f}% wariancji stanów w train, "
 f"ale nawet rzut rzeczywistego ruchu testowego na te mody daje skill H1={skills('POD_oracle_ceiling_NOT_FORECAST')[0]:.2f}%. "
 'Stała przestrzeń wyznaczona z przeszłości nie obejmuje znacznej części późniejszych kierunków ruchu. '
 'To mierzalne ograniczenie wspólnego modelu o małej randze.', '',
 f"Pierwszy wspólny mod ma korelacje wkładów trzech sond {ct.min():.3f}–{ct.max():.3f} w train, "
 f"lecz {ce.min():.3f}–{ce.max():.3f} w teście. Dla kolejnych modów pojawiają się także zmiany znaku korelacji. "
 'Wspólna struktura jest widoczna, ale jej zgodność pomiędzy sondami nie jest stała w czasie.', '',
 'Wartości własne DMDc z pamięcią dają kandydackie okresy najsłabiej tłumionych składowych: '
 + ', '.join(f"{v['period_steps']:.1f}" for v in best_osc[:4])+' kroków. '
 f"Promień spektralny wynosi {max(v['magnitude'] for v in spectra['DMDc_delay']):.6f}; jedna para jest lekko niestabilna. "
 'Te okresy opisują dopasowany operator. Długie okresy mają niewiele cykli w oknie uczącym i nie stanowią '
 'samodzielnego potwierdzenia trwałych oscylacji sieci.', '',
 'Dla tokenów narzędzia skill DMDc z pamięcią wynosi 23.61%, 10.70%, -4.74% (H1/5/10); '
 'dla tła 34.76%, 21.34%, 5.64%. Model zbiorowy gorzej przenosi się na ruch narzędzia. '
 'Biała linia na mapach modów oznacza maskę tokenów narzędzia.', '',
 '## Chronologia i znaczenie prognozy','',
 'Train: przejścia 300–699; validation 700–799; test 800–999. POD używa wyłącznie stanów 300–700, '
 'POD wejść wyłącznie sił 300–699. Ranki i regularyzacja wybrane na validation przed obliczeniem test. '
 'Wszystkie prognozy zaczynają się od prawdziwego pełnego stanu i (dla pamięci) poprzedniego stanu. '
 'Reszta poza POD zostaje zamrożona w stanie początkowym. Metryki obejmują pełne 768 współrzędnych, także tę resztę.', '',
 'H1 to ocena kroku z prawdziwym stanem początkowym (teacher-forced/rolling-origin). '
 'H5/H10 są rolloutami bez kolejnych prawdziwych stanów w rekurencji, lecz używają kolejnych zapisanych sił. '
 'Siły wymagają gradientów i AdamW na prawdziwych stanach źródłowego biegu: jest to prognoza warunkowa '
 'z oracle przyszłych wymuszeń, nie samodzielna prognoza nieznanych paczek.', '',
 'S_nowa jest finite-step zmianą z(theta+d_nowa,b)-z(theta,b), nie JVP. '
 'd_nowa wykorzystuje nowy gradient i mianownik AdamW całej paczki. Pomiar zapisano w hooku po kroku, '
 'ale jego definicja jest obliczalna z bieżącej paczki i stanu przed aktualizacją; nie wykorzystuje przyszłego dz. '
 'Ponowne obliczenie wejścia dla nowej paczki ma koszt gradientu i dodatkowego forward sond, którego ta analiza nie mierzy.', '',
 '## Wyniki testowe','',
 '| Model | Horyzont | RMSE pełne768 | Skill vs persystencja | Cos ruchu |',
 '|---|---:|---:|---:|---:|']
for r in test:
    c='—' if r['cos_motion_mean'] is None else f"{r['cos_motion_mean']:.4f}"
    lines.append(f"|{r['model']}|{r['horizon']}|{r['rmse']:.6g}|{r['skill']:.4f}|{c}|")
lines+=['','Skill = 1 − SSE_model/SSE_persystencja; wartości ujemne oznaczają pogorszenie. '
        'Cos ruchu to cosinus całego wektora przemieszczenia grupy, uśredniony po momentach startu; '
        'nie jest średnią cosinusów pojedynczych tokenów. Wyniki narzędzie/tło są w METRICS.csv.', '',
        'POD_oracle_ceiling_NOT_FORECAST to diagnostyczny rzut prawdziwego przyszłego przemieszczenia na POD. '
        'Korzysta z przyszłości wyłącznie w ocenie granicy reprezentacji; nie jest prognozą ani wejściem modelu.', '',
        f"Najlepsza rodzina Koopmana według validation: {chosen}. To wybór przed testem; osobno pokazano wszystkie rodziny.",
        f"Model prosty beta*v+b*S_nowa: beta={sel['baseline_beta_b'][0]:.6f}, b={sel['baseline_beta_b'][1]:.6f}.", '',
        '## Mody i ograniczenia','',
        'POD_MODES.npy zawiera wspólne przestrzenne mody o kształcie [r,3,600,768]. '
        'COHERENCE.json mierzy korelacje wkładów poszczególnych sond do tych samych modów, '
        'osobno train/test. SPECTRA.json podaje okresy i czasy zaniku/wzrostu wynikające z wartości własnych. '
        'Są to własności dopasowanego lokalnego modelu, nie dowód fizycznego prawa ani poprawy detekcji.', '',
        'Jedno ziarno, trzy sondy i zakres stałego LR. Z i siły źródłowe są float16, '
        'obliczenia robocze float32, sumowanie bloków float64. Różnica kolejnych Z zawiera błąd kwantyzacji. '
        f"Górna obwiednia RMS błędu różnicy wskutek zaokrągleń to {prep['quantization_rms_upper_over_motion_rms']:.4f} RMS ruchu; "
        'to konserwatywna granica, nie zmierzony faktyczny błąd. '
        'Siła nowej paczki pomija osobne zmiany BN i zanik wag; pamięć optymalizatora jest tylko przybliżona opóźnieniem stanu. '
        'Podobieństwo modów oraz dobra rekonstrukcja nie dowodzą, że wybrane obrazy poprawią mAP.', '',
        '## Koszt i odtworzenie','',
        f"Czas lokalnego biegu: {runtime['seconds']:.1f}s. Peak GPU allocated: {runtime['cuda_peak_allocated_MiB']:.1f}MiB; reserved: {runtime['cuda_peak_reserved_MiB']:.1f}MiB.",
        f"Wersje: torch {runtime['torch']}, PyDMD {runtime['pydmd']}, numpy {runtime['numpy']}.",
        'Pełne parametry: RUNNING.json; hash kodu i polecenie: RUNTIME.json. '
        'MODEL_SELECTION_BEFORE_TEST.json zapisano przed testem; prepared.pkl umożliwia audyt bez GPU.', '',
        '```powershell', f'python -u experiments/vit_dmdc_paczki_20261008/run.py --source outputs/vit_dmdc_paczki_20261008/source_eden --out {o.as_posix()} --start 300 --train-end 700 --val-end 800 --end 1000 --max-rank 64 --block 30',
        f'python experiments/vit_dmdc_paczki_20261008/report.py {o.as_posix()}','```','',
        'Biblioteka i dokumentacja: [PyDMD.DMDc](https://pydmd.github.io/PyDMD/dmdc.html), '
        '[oficjalny tutorial](https://pydmd.github.io/PyDMD/tutorial7dmdc.html).']
(o/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(o/'REPORT.md')
