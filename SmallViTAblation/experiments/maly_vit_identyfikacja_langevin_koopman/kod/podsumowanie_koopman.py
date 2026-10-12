"""Eksportuje zamrożony wybór i diagnozuje przesunięcie cech, bez nowego wyboru.
Nie uczy encoderów ani nie zmienia prognoz. Porównuje odtwarzane parametry z NPZ.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from koopman_obserwator import (OUT,HERE,load,raw_images,pca_fit,AE,drift_fit,complete_fit,
                               noise_fit,forecast,filtered_forecast)


def main():
    z=json.loads((OUT/'METRYKI.json').read_text(encoding='utf-8'))
    old=json.loads((HERE/'wyniki/krotki_horyzont/METRYKI.json').read_text(encoding='utf-8'))
    saved=np.load(OUT/'PROGNOZY.npz')
    d={s:load(s) for s in (47,48)};q={s:raw_images(s) for s in d}
    pc=pca_fit(q[47],2000);dump={};diagnosis={}
    for s,ts in ((47,np.arange(1000,2001,5)),(48,np.arange(2100,2951,5))):
        scores=(q[s][ts]-pc['mu'])@pc['V']/pc['sd']
        norm=np.linalg.norm(scores,axis=1)
        diagnosis[str(s)]=dict(pca_normalized_norm_median=float(np.median(norm)),
                               pca_normalized_norm_p90=float(np.quantile(norm,.9)))
    for h,rr in z['results'].items():
        H=int(h);n=rr['selected'];cfg=rr['configs'][n]
        kind=cfg['kind'];mu=z['ae']['2000']['mu'];sd=z['ae']['2000']['sd']
        lift=dict(kind=kind,mu=mu,sd=sd)
        if kind=='learned':
            net=AE();net.load_state_dict(torch.load(OUT/'AE_cut2000.pt',map_location='cpu',weights_only=True))
            lift['net']=net
        op=drift_fit(d[47],H,lift,2000)
        model=complete_fit(d[47],q[47],H,op,pc,cfg['variant'],cfg['ridge'],2000,2500)
        t=np.arange(2100,2951,5)
        if n.endswith('_kf'):
            noise=noise_fit(model,d[47],q[47],2000,2500,cfg['noise']['fraction'])
            p=filtered_forecast(model,noise,d[48],q[48],t,2020)
            dump[f'H{H}_noise']=np.array([noise[k] for k in ('rho','Q','R','variance')])
        else:p=forecast(model,d[48],q[48],t)
        discrepancy=float(np.max(np.abs(p-saved[f'H{H}_s48_{n}'])))
        assert discrepancy<1e-5, (h,discrepancy)
        diagnosis[h]=dict(prediction_replay_max_abs=discrepancy,matched=True)
        dump[f'H{H}_K']=op['K'];dump[f'H{H}_lift_mean_sd']=np.array([mu,sd])
        for k in ('W','mu','sd','offset'):
            if k in model:dump[f'H{H}_history_{k}']=np.asarray(model[k])
    for k,v in pc.items():dump['image_pca_'+k]=v
    np.savez_compressed(OUT/'MODELE_WYBRANE.npz',**dump)
    (OUT/'DIAGNOSTYKA.json').write_text(json.dumps(diagnosis,indent=2),encoding='utf-8')
    Hs=[int(h) for h in z['results']]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    variants=[('new','Validation-selected EDMD / AE / KF'),('old_langevin20','Old direct Langevin'),
              ('koopman_lift_gle20_input','Earlier direct lifting + GLE + effects')]
    for n,label in variants:
        rows=[z['results'][str(h)]['test']['48'][z['results'][str(h)]['selected']] if n=='new' else old['results'][str(h)]['test']['48'][n] for h in Hs]
        axes[0].plot(Hs,[r['skill_percent'] for r in rows],'o-',label=label)
        axes[1].plot(Hs,[r['sign_percent'] for r in rows],'o-',label=label)
        axes[2].plot(Hs,[r['balanced_sign_percent'] for r in rows],'o-',label=label)
    for ax,title in zip(axes,['MSE reduction (%)','Sign accuracy: all probes (%)','Balanced sign accuracy (%)']):
        ax.set_xscale('log');ax.set_xlabel('Horizon (updates)');ax.set_title(title)
        ax.axhline(80,color='black',linestyle='--',linewidth=1);ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle('s48 exploratory transfer; choices frozen on s47 validation')
    fig.tight_layout();fig.savefig(OUT/'KRZYWE.png',dpi=160);plt.close(fig)
    p=OUT/'WYNIKI.md'
    text=p.read_text(encoding='utf-8')
    marker='\n## Pełne porównanie poza dopasowaniem'
    if marker in text:text=text.split(marker)[0]
    lines=[marker,'', '| H | Ramię | Redukcja MSE [%] | Znak [%] | Balanced [%] |','|---:|---|---:|---:|---:|']
    for h,rr in z['results'].items():
        for n,a in rr['test']['48'].items():
            lines.append(f"| {h} | {n} | {a['skill_percent']:.2f} | {a['sign_percent']:.2f} | {a['balanced_sign_percent']:.2f} |")
    train=diagnosis['47']['pca_normalized_norm_median'];test=diagnosis['48']['pca_normalized_norm_median']
    lines += ['', '![Krzywe porównania](KRZYWE.png)', '',
        f'PCA cech obrazowych: mediana normy w standaryzowanych8 współrzędnych dla treningu s47 wynosi{train:.2f}, a dla s48 {test:.2f}. '
        'To zmierzone przesunięcie rozkładu cech. Cechy pochodzą z oddzielnie uczonych sieci i zmieniają się także podczas ich uczenia; '
        'nie udowodniono wspólnej semantycznej zgodności tych współrzędnych między biegami. '
        'Nie przypisujemy samej korelacji diagnostycznej roli jedynej przyczyny pogorszenia. '
        'Przyszły test powinien porównać wspólny zamrożony encoder obrazów albo kalibrację dopasowania reprezentacji z dostępnej przeszłości, '
        'wraz z kontrolą mniej zmiennych deskryptorów obrazu.', '',
        'KF jest filtrem pojedynczej ukrytej reszty dołączonej do deterministycznego predyktora pamięci. '
        'Nie jest pełnym filtrem wszystkich wag, momentów AdamW,600 cząsteczek ani wspólnej niepewności wszystkich opóźnionych stanów. '
        'Dla przyjętego AR1 pomiar eta jest bezpośredni; ta obserwowalność skalarnego modelu nie przechodzi na pełną sieć. '
        'EKF/UKF i pełna rozszerzona realizacja pozostają przyszłym etapem.', '',
        'Wybrany H1 jest gorszy od wcześniejszego direct modelu; w tym EDMD/AE pilotażu żaden walidacyjnie wybrany H nie osiągnął80procent. '
        'Nie zastępujemy go wariantem wybranym po obejrzeniu s48. '
        'Wcześniejszy wynik80,98procent znaków naH1 pozostaje rezultatem direct lifting+GLE+historycznych wpływów, nie EDMD.', '',
        'MODELE_WYBRANE.npz przechowuje odtworzone zamrożone operatory, PCA, współczynniki i parametry KF. '
        'Rekonstrukcja prognoz zgadza się z pierwotnym PROGNOZY.npz do1e-5; DIAGNOSTYKA.json zapisuje rzeczywiste błędy odtworzenia. '
        'AE_cut1400.pt i AE_cut2000.pt są małymi wagami encoder/decoder (CPU), nie checkpointami ViT.']
    p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(diagnosis))


if __name__=='__main__':main()
