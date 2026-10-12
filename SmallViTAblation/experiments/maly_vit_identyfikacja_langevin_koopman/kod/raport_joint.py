"""Report saved joint pilot; does not fit or select on test."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dane import HERE

P=HERE/'wyniki/joint_controlled'
r=json.loads((P/'METRYKI.json').read_text());hs=r['hs']
pred=np.load(P/'PROGNOZY.npz');models=np.load(P/'MODELE.npz');input_diag={}
for h in hs:
    chosen=r['results'][str(h)]['selected'];t=pred[f'H{h}_s48_times']
    E=pred[f'H{h}_s48_truth']-pred[f'H{h}_s48_{chosen}']
    raw=np.load(HERE/'wyniki/koopman_obserwator/OBRAZY_s48.npy')
    u=(raw[t]-models['pre2500_qmu'])@models['pre2500_V']/models['pre2500_qsd']
    cor=[float(np.corrcoef(E.mean(1),u[:,j])[0,1]) for j in range(4)]
    prediction=pred[f'H{h}_s48_{chosen}'];cuts=np.quantile(np.abs(prediction),[0,.25,.5,.75,1])
    bucket=[float(np.mean(E[(np.abs(prediction)>=cuts[j])&(np.abs(prediction)<=cuts[j+1])]**2)) for j in range(4)]
    input_diag[str(h)]=dict(residual_mean_vs_last_completed_image_PCA_corr=cor,
                          residual_mse_by_prediction_abs_quartile=bucket,
                          scope='post hoc exploratory diagnostics; no future inputs to forecast')
(P/'DIAGNOSTYKA_INPUT.json').write_text(json.dumps(input_diag,indent=2),encoding='utf-8')
rows=[];cross=[];arms=[];threshold={};edmd_rows=[]
fig,ax=plt.subplots(1,3,figsize=(14,4))
for metric,title in [('skill_percent','Redukcja MSE [%]'),('sign_percent','Znak wszystkich sond [%]'),('balanced_sign_percent','Balanced accuracy [%]')]:
    a=ax[['skill_percent','sign_percent','balanced_sign_percent'].index(metric)]
    for seed in ('48','47'):
        y=[r['results'][str(h)]['test'][seed][r['results'][str(h)]['selected']][metric] for h in hs]
        a.plot(hs,y,'o-',label='joint s'+seed+(' eksploracyjne' if seed=='48' else ' późne'))
    for name,label in [('legacy','dawny Langevin'),('chosen','wcześniejszy direct')]:
        y=[]
        for h in hs:
            old=r['old_direct'][str(h)];d=old['test']['48']
            key=old['selected_overall'] if name=='chosen' else next(k for k in d if 'langevin' in k)
            y.append(d[key][metric])
        a.plot(hs,y,'--',label=label+' s48')
    a.axhline(80,color='gray',ls=':');a.set(xlabel='H aktualizacji',ylabel=title,xscale='log');a.grid(alpha=.2)
    a.legend(fontsize=7)
fig.tight_layout();fig.savefig(P/'KRZYWE.png',dpi=160);plt.close(fig)
for h in hs:
    v=r['results'][str(h)];name=v['selected'];a=v['test']['48'][name];b=v['test']['47'][name]
    c=r['configs'][f'H{h}_{name}']
    rows.append(f"| {h} | {name} | {a['skill_percent']:.2f} | {a['sign_percent']:.2f} | {a['balanced_sign_percent']:.2f} | {a['mae']:.4f} | {a['rmse']:.4f} | {a['coverage90_percent']:.2f} |")
    cross.append(f"| {h} | {b['skill_percent']:.2f} | {b['sign_percent']:.2f} | {b['balanced_sign_percent']:.2f} | {c['ridge']} | {c['rank']}/{c['columns']} | {c['condition']:.2e} |")
    for arm,a0 in v['test']['48'].items():
        arms.append(f"| {h} | {arm} | {a0['skill_percent']:.2f} | {a0['sign_percent']:.2f} | {a0['balanced_sign_percent']:.2f} |")
    for arm in ('fixed_only','fixed_memory','learned_only'):
        a0=r['old_edmd_without_kf'][str(h)]['48'][arm]
        edmd_rows.append(f"| {h} | {arm} | {a0['skill_percent']:.2f} | {a0['sign_percent']:.2f} |")
for metric in ('skill_percent','sign_percent','balanced_sign_percent'):
    valid=[h for h in hs if r['results'][str(h)]['test']['48'][r['results'][str(h)]['selected']][metric]>=80]
    threshold[metric]=max(valid) if valid else None
(P/'PROG80.json').write_text(json.dumps(dict(exploratory=True,max_measured_H=threshold),indent=2))
text='''# Wspólna identyfikacja Koopman-GLE z historią obrazowych wejść

Wykonano odrębny pilot CPU bez KF. Wszystkie bloki operatora, pamięci stanu oraz pamięci wejść dopasowano w jednym problemie ridge do całego przyrostu liftingu. To naprawia konstrukcję estymacji; nie zapewnia przewagi empirycznej. s48 było wcześniej oglądane i pozostaje eksploracyjne. Zapisana trajektoria s47 późno także nie jest nowym niezależnym biegiem.

## Wyprowadzenie i estymacja

Rozważmy wybrany lifting x oraz ukryty stan h, z historycznym sterowaniem u:

```
x[k+1] = A x[k] + B h[k] + D u[k] + noise_x[k]
h[k+1] = E h[k] + C x[k] + F u[k] + noise_h[k]
h[k] = E^k h[0] + sum(j<k) E^(k-1-j)(C x[j]+F u[j]+noise_h[j])
x[k+1] = A x[k] + D u[k]
           + sum(j<k) B E^(k-1-j) C x[j]
           + sum(j<k) B E^(k-1-j) F u[j]
           + B E^k h[0] + colored_residual[k]
```

To algebraiczna eliminacja przy przyjętej liniowej realizacji w liftingu. Jądra stanu i wejścia mają wspólne ukryte E, ale różne C/F. Nie dowodzi istnienia dokładnego skończonego liftingu dla ViT. Nieznane u[k] w chwili stanu x[k] pozostaje nieznane również po eliminacji. Model pełnego AdamW obejmowałby wagi, momenty, bufory i politykę paczek; obecna obserwabla jest tylko ustalonym skalarnym rzutem LDA.

Lean dowodzi siedmiu tożsamości skalarnych nad Int: rozdzielenia sum pamięci, eliminacji, zależności pamięci tylko od j<k oraz rozwinięcia przykładu. SymPy sprawdza macierzowy nieprzemienny przykład2x2 k0..3 i joint blockridge. [Zakres i założenia](../../matematyka/CONTROLLED_MEMORY.md). Dowód algebraiczny nie gwarantuje prognozy ani semantycznej tożsamości.

Pilot marginalny: ψ=(z,z²), gdzie z jest s minus historyczna średnia500, standaryzowane na train. Referencja okna zamrożona. Target to Δψ_H, regresory: bieżący ψ,20 opóźnień Δψ, obrazy wykonanych paczek w rozłącznych grupach ostatnia1, poprzednie4 i poprzednie15. Operator i oba jądra są estymowane JOINT. Obrazowy q[t] pochodzi z paczki źródłowej t−1; bez q[t+1:]. Dla opt dodatkowo osobne wykonane N/P. Intercept i normalizacja wyłącznie train. To projekcja marginalna K_H z historią, nie semigrupa ani eksperymentalnie zidentyfikowany mechanizm przyczynowy. Pamięć przyrostów jest jawną bazą temporalną; nie nadano jej statusu dokładnego kernelu MZ.

Osobny controlled_images fit używa historycznego u[k] wywołującego x[k+1], ale przy rollout zastępuje przyszłe wejścia średnią20 ostatnich wykonanych paczek. Wszystkie bloki one-step nadal estymowane wspólnie. Modeluje przyszłą politykę w bardzo uproszczony sposób; nie odczytuje przyszłych paczek. Addytywne wejście nie pokrywa wszystkich możliwych zależności stanu i obrazu; biliniowe/interakcyjne wejście pozostaje uzasadnioną przyszłą ablacją.

Protokół zapisany [przed estymacją](PLAN.md). Train do1800, walidacja1800..2000, refit do2500; końce wszystkich etykiet train/refit dostępne przed cut. Ridge0.001/0.1/10 i ramiona wybierane po MSE walidacji osobnoH; brak kary wiążącejH. PCA4 train-only opisuje rzeczywiste wykonane obrazy przez zapisane embeddingi paczki; nie sondy. To embeddingi osobno uczonych sieci, nie wspólny zamrożony encoder ani surowy piksel. Proste deskryptory pochodzą z tej samej sieci/LDA. Zgodność przestrzeni między biegami nie jest udowodniona.

## Wyniki wyboru na walidacji, s48

| H | Model | Redukcja MSE % | Znak % | Balanced % | MAE | RMSE | Pokrycie90 % |
|---|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

80 procent osiągnięto wyłącznie dla znaku/balanced na H1:80,92. Nie osiągnięto80 redukcjiMSE. Maksymalny badany H spełniający próg znaku to1; zapis sond co10 oznacza inny H niż jedna aktualizacja. Nie stwierdzono monotoniczności ani potwierdzenia na świeżym biegu. Poprzedni direct H1 miał64,96 redukcjiMSE i80,98 znaku; nowyjoint H1 ma64,12 i80,92, zatem nie wykazano poprawy względem poprzedniego najlepszego wariantu.

Samo obrazowe sterowanie H1: pamięć62,31 redukcjiMSE i79,85 znaku; obrazy62,43 i79,77. Kontrolowany one-step z modelowanymi przyszłymi obrazami62,50 i79,93. Największy wzrost do64,12/80,92 pochodzi z ramienia z dodatkowymi historycznymi N/P, które nie są samymi cechami obrazów. Nie przypisujemy całego wyniku obrazowemu sterowaniu. Stary sequentialEDMD+images miał silne negatywy; w tym etapie poprawiono również lifting, podział i czas fit, więc różnica nie izoluje przyczynowo samej metodyJOINT.

## Późny s47 i identyfikowalność

| H | Redukcja MSE % | Znak % | Balanced % | Ridge | Ranga/kolumny | Uwarunkowanie |
|---|---:|---:|---:|---:|---|---:|
'''+ '\n'.join(cross)+'''

Ranga macierzy dotyczy zdefiniowanych regresorów przy danej normalizacji, nie obserwowalności pełnego ViT. Wielokrotne sondy współdzielą wejście i bieg; liczba wierszy nie jest liczbą niezależnych replik. Pełna ranga numeryczna nie dowodzi rozdziału mechanizmów, identyfikowalności semantycznej ani reakcji na interwencję. Ridge stabilizuje estymację, nie tworzy brakującego wzbudzenia. [PrzeglądAstry](../../PRZEGLAD_ASTRA_STEROWANIE.md) wyjaśnia także poprawną residualizacjęFWL/ridge.

## Diagnostyka reszt i granice

H1s48 wybranego opt: skośność−0,019, nadmiarowa kurtoza5,94, korelacja błędu² z |prognozą|0,224. ACF reszt dla początków oddalonych o5/10/25 aktualizacji:−0,036/−0,002/+0,006. Niska ACF kilku lagów nie dowodzi białości; duża kurtoza przeczy prostemu założeniu dokładnie gaussowskiego szumu. H>1 etykiety zachodzą na siebie, co samo może wprowadzać korelacje. Globalny przedział90 z walidacji pokrywa87,70 procent H1s48: kalibracja nie jest idealna. Reszty służą diagnozie niedomknięcia, bezFDT dla AdamW.

Wszystkie36 kontroli prefiksowych przechodzą, w tym rollout control bez źródłowych przyszłych wejść. Sprawdzenie wejścia przyszłego jest kodowe, nie domniemane z nazwy wariantu. Zachowano wszystkie ramiona i negatywy. Rozkład/ACF nie ustanawia skutecznego odzyskania ukrytego sygnału; świeży odłożony bieg i wspólny encoder obrazów pozostają potrzebne.

## Wszystkie ramiona s48

| H | Model | Redukcja MSE % | Znak % | Balanced % |
|---|---|---:|---:|---:|
'''+ '\n'.join(arms)+'''

## Artefakty

METRYKI.json: wybór,val,test,diagnostyka,konfiguracje i stare porównania bezKF. PROGNOZY.npz: wszystkie przewidywania i cele. MODELE.npz: bloki operatorów i preprocessing. PROG80.json: maksymalne zmierzoneH. KRZYWE.png: porównanie horyzontów. [Kod](../../kod/joint_controlled.py). CPU'''+f"{r['seconds']:.2f}"+'''s, bez treninguViT, GPU, Edenu i commitów.

![Krzywe](KRZYWE.png)
'''
text+='\n## Zapisany EDMD bezKF na identycznych celach s48\n\n| H | Model | Redukcja MSE % | Znak % |\n|---|---|---:|---:|\n'+'\n'.join(edmd_rows)+'\n\nTo zachowane ramiona, nie nowy wybór po teście. Szczegóły wcześniejszego podziału train/drift są inne, dlatego nie izolujemy skutku jednej zmiany algorytmu.\n'
text+='\nDiagnostyka korelacji średniej reszty z ostatnimi wykonanymi obrazami (PCA4) i błędu po kwartylach amplitudy prognozy: DIAGNOSTYKA_INPUT.json. Korelacja nie identyfikuje przyczyny ani niezależności. [Pełne4 źródła i119 wyników DeepSearch](../../zrodla/LITERATURA_JOINT.md).\n'
(P/'WYNIKI.md').write_text(text,encoding='utf-8')

if __name__=='__main__':print(P/'WYNIKI.md')
